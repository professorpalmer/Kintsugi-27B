"""Check a running ./run.sh serve: chat 6/6 and the tool-call tests t1-t4, all through HTTP. Standard library only.

    python kit/serve_check.py [--url http://127.0.0.1:8080/v1]

chat: the 6 prompts of kit/chat_sanity.py, thinking off, temperature 0, streamed.
t1: a streamed tool call (read_file) with a system message and 3 tools; the call parses, finish_reason tool_calls.
t2: turn 2 of t1 with the tool result: an answer from the result, finish_reason stop; the prompt reuses turn 1
    (cached_tokens > 0) and reads few new tokens.
t3: a multi-line shell command that must hold a blank line, after text with a blank line in it (the case behind
    the "empty grammar stack" fault on Windows): the argument comes back with the blank line, no tags in the text.
t4: not streamed, thinking off, two developer messages first and a system message after the user turn (as Codex
    and Claude Code send): HTTP 200, a tool call or an answer, no tags in the text.
Every response: no <think>, </think>, <tool_call> or <function= in the visible text. Exit code 1 on any failure.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from chat_sanity import PROMPTS  # noqa: E402

TAGS = ("<think>", "</think>", "<tool_call>", "</tool_call>", "<function=", "<parameter=")
TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file and return its content.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                                     "required": ["path"]}}},
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command and return its output.",
                                      "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                                                     "required": ["command"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write text to a file.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                                                                                      "content": {"type": "string"}},
                                                     "required": ["path", "content"]}}},
]
SYSTEM = "You are a coding agent in a small Python repository. Use the tools to look at files before you answer."


def post(url, body, stream):
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer any-key"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        if not stream:
            out = json.loads(r.read())
            m = out["choices"][0]["message"]
            return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "",
                    "tool_calls": m.get("tool_calls") or [], "finish": out["choices"][0]["finish_reason"],
                    "usage": out.get("usage", {}), "seconds": time.time() - t0, "status": r.status}
        res = {"content": "", "reasoning": "", "tool_calls": [], "finish": None, "usage": {}, "status": r.status,
               "first_byte_s": None, "done": False}
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            if res["first_byte_s"] is None:
                res["first_byte_s"] = time.time() - t0
            data = line[5:].strip()
            if data == "[DONE]":
                res["done"] = True
                break
            ev = json.loads(data)
            if ev.get("usage"):
                res["usage"] = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                res["content"] += d.get("content") or ""
                res["reasoning"] += d.get("reasoning_content") or ""
                for tc in d.get("tool_calls") or []:
                    res["tool_calls"].append(tc)
                if ch.get("finish_reason"):
                    res["finish"] = ch["finish_reason"]
        res["seconds"] = time.time() - t0
        return res


def clean(r):
    return not any(t in r["content"] for t in TAGS)


def args_of(call):
    return json.loads(call["function"]["arguments"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows, ok_all = [], True

    def record(name, ok, r, note=""):
        nonlocal ok_all
        ok_all &= ok
        rows.append({"test": name, "ok": ok, "note": note, "finish": r.get("finish"),
                     "tool_calls": [c["function"] for c in r.get("tool_calls", [])], "content": r["content"][:300],
                     "usage": r.get("usage"), "seconds": round(r.get("seconds", 0), 2)})
        print(f"{'ok ' if ok else 'BAD'} {name}: {note} [{r.get('finish')}, {r.get('seconds', 0):.1f} s]", flush=True)

    # chat 6/6
    n_ok = 0
    for q, want in PROMPTS:
        r = post(a.url, {"model": "kintsugi", "messages": [{"role": "user", "content": q}], "stream": True,
                         "temperature": 0, "max_tokens": 96, "chat_template_kwargs": {"enable_thinking": False}}, True)
        ok = any(w.lower() in r["content"].lower() for w in want) and clean(r) and r["done"] and not r["reasoning"]
        n_ok += ok
        record("chat", ok, r, repr(r["content"].strip()[:60]))

    # t1
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "What is the value of DEFAULT_TIMEOUT in src/config.py?"}]
    r1 = post(a.url, {"model": "kintsugi", "messages": msgs, "tools": TOOLS, "stream": True, "temperature": 0,
                      "stream_options": {"include_usage": True}}, True)
    c1 = r1["tool_calls"]
    ok = (r1["finish"] == "tool_calls" and len(c1) >= 1 and c1[0]["function"]["name"] == "read_file"
          and "config.py" in args_of(c1[0]).get("path", "") and clean(r1) and r1["done"])
    record("t1 streamed tool call", ok, r1, f"{[c['function'] for c in c1]}")

    # t2
    msgs2 = msgs + [{"role": "assistant", "content": r1["content"], "reasoning_content": r1["reasoning"],
                     "tool_calls": [{"id": c.get("id", "call_1"), "type": "function", "function": c["function"]}
                                    for c in c1]}]
    for c in c1:
        msgs2.append({"role": "tool", "tool_call_id": c.get("id", "call_1"),
                      "content": "import os\n\nDEFAULT_TIMEOUT = 37\nRETRIES = 3\n"})
    r2 = post(a.url, {"model": "kintsugi", "messages": msgs2, "tools": TOOLS, "stream": True, "temperature": 0,
                      "stream_options": {"include_usage": True}}, True)
    u = r2.get("usage") or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    ok = (r2["finish"] == "stop" and "37" in r2["content"] and clean(r2) and cached > 0
          and u.get("prompt_tokens", 0) - cached < 400)
    record("t2 tool result -> answer", ok, r2, f"answer {r2['content'].strip()[:60]!r}, prompt "
                                               f"{u.get('prompt_tokens')} tok, {cached} reused")

    # t3
    q3 = ("Run this exact two-line shell script with the bash tool, keeping the empty line between the two "
          "commands:\n\necho first\n\necho second\n\nSay one short sentence first, then call the tool.")
    r3 = post(a.url, {"model": "kintsugi", "messages": [{"role": "system", "content": SYSTEM},
                                                         {"role": "user", "content": q3}],
                      "tools": TOOLS, "stream": True, "temperature": 0}, True)
    c3 = r3["tool_calls"]
    cmd = args_of(c3[0]).get("command", "") if c3 else ""
    ok = (r3["finish"] == "tool_calls" and bool(c3) and c3[0]["function"]["name"] == "bash"
          and "echo first" in cmd and "echo second" in cmd and "\n\n" in cmd and clean(r3))
    record("t3 blank line in text and argument", ok, r3, f"command {cmd!r}, text {r3['content'].strip()[:50]!r}")

    # t4
    msgs4 = [{"role": "developer", "content": "You are Codex, a coding agent."},
             {"role": "developer", "content": "Answer briefly."},
             {"role": "user", "content": "Create the file notes.txt with the single line: hello mac"},
             {"role": "system", "content": "<environment>cwd=/repo, shell=zsh</environment>"}]
    r4 = post(a.url, {"model": "kintsugi", "messages": msgs4, "tools": TOOLS, "stream": False, "temperature": 0,
                      "chat_template_kwargs": {"enable_thinking": False}}, False)
    c4 = r4["tool_calls"]
    ok = (r4["status"] == 200 and clean(r4) and not r4["reasoning"]
          and (r4["finish"] == "stop" or (c4 and c4[0]["function"]["name"] in ("write_file", "bash"))))
    record("t4 not streamed, developer + later system", ok, r4, f"{[c['function'] for c in c4]}")

    res = {"chat_correct": n_ok, "chat_of": len(PROMPTS), "all_ok": ok_all, "rows": rows}
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
