"""Needles through a running ./run.sh serve at the server's largest context, as one chat.

    python kit/serve_needles.py [--url http://127.0.0.1:8080/v1] [--n 24] [--margin N]

Turn 1: the needle document of kit/common.py (the server's context minus the margin) and the question for vault 1.
The default margin is 48 tokens per turn + 640: the 24 turns grew the chat by 843 tokens (at 257k and at 6k), and the
chat template and the first question add about 385 tokens. At 6k (16 GB profile) turn 24 then has about 5.6k tokens.
Turns 2..n: the question for the next vault. Thinking off, temperature 0. Each turn after the first must read only
its new tokens (the earlier turns stay in the server's cache). Exit code 1 if a needle is wrong or a turn reads the
whole chat again.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODEL_DIR, first_code, log, needle_doc  # noqa: E402


def call(url, messages):
    body = {"model": "kintsugi", "messages": messages, "stream": False, "temperature": 0, "max_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=7200) as r:
        out = json.loads(r.read())
    return out["choices"][0]["message"].get("content") or "", out.get("usage", {}), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--margin", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    ctx = json.loads(urllib.request.urlopen(a.url + "/models", timeout=30).read())["data"][0]["context_length"]
    doc, codes = needle_doc(tok, ctx - (a.margin or 48 * a.n + 640), a.n)
    q = "What is the access code for vault {}? Answer with the 4 digits only."
    messages, rows, ok_all = [], [], True
    for v in range(1, a.n + 1):
        messages.append({"role": "user", "content": (doc if v == 1 else "") + q.format(v)})
        ans, u, s = call(a.url, messages)
        messages.append({"role": "assistant", "content": ans})
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        new = u.get("prompt_tokens", 0) - cached
        hit = first_code(ans) == codes[v]
        reuse_ok = v == 1 or new < 200
        ok_all &= hit and reuse_ok
        rows.append({"vault": v, "want": codes[v], "answer": ans.strip()[:40], "ok": hit,
                     "prompt_tokens": u.get("prompt_tokens"), "cached_tokens": cached, "seconds": round(s, 1)})
        log(f"needle {v:2d}/{a.n}: {'ok ' if hit else 'BAD'} want {codes[v]} got {ans.strip()[:20]!r}; prompt "
            f"{u.get('prompt_tokens')} tok, {cached} reused{'' if reuse_ok else ' (REUSE MISSED)'}, {s:.1f} s")
    res = {"context_length": ctx, "turn1_prompt_tokens": rows[0]["prompt_tokens"], "turn1_seconds": rows[0]["seconds"],
           "correct": sum(r["ok"] for r in rows), "of": a.n, "all_ok": ok_all, "rows": rows}
    log(f"server needles: {res['correct']}/{a.n} at {res['turn1_prompt_tokens']} tokens, turn 1 {res['turn1_seconds']} s")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
