"""Step 4: short chat sanity test, greedy, thinking off. Once without MTP and once with MTP (the outputs should match;
MTP changes speed, not text, except at rare near-ties)."""
from __future__ import annotations

import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, log, write_json  # noqa: E402

PROMPTS = [
    ("What is 17 * 23? Answer with the number only.", ["391"]),
    ("What is the capital of Japan? Answer with one word.", ["Tokyo"]),
    ("At sea level, water boils at how many degrees Celsius? Answer with the number only.", ["100"]),
    ("Write a Python function `square(x)` that returns x squared. Code only.", ["def square", "x * x", "x**2", "x ** 2"]),
    ("Translate to French: 'The cat sleeps on the chair.'", ["chat", "chaise"]),
    ("List the first five prime numbers, separated by commas.", ["2, 3, 5, 7, 11", "2,3,5,7,11"]),
]


def run(mtp: bool):
    from mlx_lm.sample_utils import make_sampler
    model, tok, t_load = load(mtp=mtp)
    if mtp:
        from escha_mlx.mtp import stream_generate
    else:
        from mlx_lm.generate import stream_generate
    rows = []
    for q, want in PROMPTS:
        prompt = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                         tokenize=False, enable_thinking=False)
        text, last, t0 = "", None, time.time()
        for r in stream_generate(model, tok, prompt, max_tokens=96, sampler=make_sampler(temp=0.0)):
            text += r.text
            last = r
        ok = any(w.lower() in text.lower() for w in want)
        rows.append({"prompt": q, "answer": text, "ok": ok, "prompt_tps": last.prompt_tps,
                     "generation_tps": last.generation_tps, "generation_tokens": last.generation_tokens,
                     "peak_memory_gb": last.peak_memory, "wall_s": round(time.time() - t0, 1)})
        log(f"chat {'MTP' if mtp else 'AR '}: {'ok ' if ok else 'BAD'} {text.strip()[:60]!r} "
            f"({last.generation_tps:.1f} tok/s)")
    del model
    gc.collect()
    return {"load_s": round(t_load, 1), "rows": rows, "correct": sum(r["ok"] for r in rows), "of": len(rows)}


def main():
    res = {"ar": run(False)}
    try:
        res["mtp"] = run(True)
        res["mtp_same_text"] = [a["answer"] == b["answer"] for a, b in zip(res["ar"]["rows"], res["mtp"]["rows"])]
    except Exception as e:  # noqa: BLE001
        res["mtp"] = {"error": repr(e)}
        log(f"chat MTP failed: {e!r}")
    write_json("chat.json", res)


if __name__ == "__main__":
    main()
