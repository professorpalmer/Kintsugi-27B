"""Offline test of the output parser of ./run.sh serve and chat (kit/userapi.py OutputParser). No model, no GPU.

Each case goes through the parser in chunks of 1, 2, 3, 5 and 1000 characters; all chunk sizes must give the same
reasoning, text, tool calls and count of malformed calls. Cases: blank lines in text and in a call argument, a call
inside the reasoning (before </think>), an open <tool_call> with no body (a malformed call, counted), a literal "<".
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import userapi as u  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": n, "parameters": {"type": "object", "properties": {k: {"type": "string"}}}}}
         for n, k in (("bash", "command"), ("read_file", "path"))]
CALL_B = "<tool_call>\n<function=bash>\n<parameter=command>\necho a\n\necho b\n</parameter>\n</function>\n</tool_call>"
CALL_R = "<tool_call>\n<function=read_file>\n<parameter=path>\nx.py\n</parameter>\n</function>\n</tool_call>"
cases = {
 "a normal": ("reason</think>\n\nHello\n\nworld\n\n" + CALL_B, dict(reasoning="reason", text="Hello\n\nworld", calls=[("bash", {"command": "echo a\n\necho b"})], bad=0)),
 "b call in reasoning": ("reason\n" + CALL_R + "\n</think>\n\n", dict(reasoning="reason", text="", calls=[("read_file", {"path": "x.py"})], bad=0)),
 "b2 call in reasoning, then text": ("reason\n" + CALL_R + "\nDone.", dict(reasoning="reason", text="Done.", calls=[("read_file", {"path": "x.py"})], bad=0)),
 "c bare open tag": ("</think>\n\nI will call.\n<tool_call>", dict(reasoning="", text="I will call.", calls=[], bad=1)),
 "c2 open tag, partial body": ("</think>\n\n<tool_call>\n<function=bash>\n<parameter=comm", dict(reasoning="", text="", calls=None, bad=None)),
 "d plain": ("think\n</think>\n\nplain answer", dict(reasoning="think", text="plain answer", calls=[], bad=0)),
 "e literal lt": ("x</think>\n\na < b and <tool is fine", dict(reasoning="x", text="a < b and <tool is fine", calls=[], bad=0)),
}
fail = 0
for name, (src, want) in cases.items():
    outs = set()
    for n in (1, 2, 3, 5, 1000):
        p = u.OutputParser(in_reasoning=True, tools=TOOLS)
        ev = []
        for i in range(0, len(src), n):
            ev += list(p.feed(src[i:i + n]))
        ev += list(p.close())
        r = "".join(x for k, x in ev if k == "reasoning"); t = "".join(x for k, x in ev if k == "text")
        calls = [(x["function"]["name"], json.loads(x["function"]["arguments"])) for k, x in ev if k == "tool_call"]
        bad = sum(k == "bad_tool_call" for k, _ in ev)
        outs.add(json.dumps([r, t, calls, bad]))
    got = json.loads(next(iter(outs)))
    ok = len(outs) == 1 and all(want[k] is None or v == want[k] for k, v in zip(("reasoning", "text", "calls", "bad"), [got[0], got[1], [tuple(c) for c in got[2]], got[3]]))
    fail += not ok
    print("ok " if ok else "BAD", name, "consistent" if len(outs) == 1 else f"INCONSISTENT {outs}", got)
sys.exit(1 if fail else 0)
