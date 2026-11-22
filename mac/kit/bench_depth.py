"""Output speed with MTP at 4k, 32k and the largest context, through the user-mode engine (kit/userapi.py, the code
that ./run.sh chat and ./run.sh serve run). The speed step (kit/speed_needles.py) gives the same depths without MTP.

For each depth: a new conversation (the session caches are cleared), one user message of filler text that ends with a
coding question, thinking off, greedy, 256 output tokens. Prefill time is the time to the first token. Output speed is
the tokens after the first one divided by their time. MTP speed depends on the text: this is code after a long
context, not a measure for every task.

    python kit/bench_depth.py [--depths 4096,32768,max] [--tokens 256]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PROFILE, footprint, log, sentence, write_json  # noqa: E402

QUESTION = ("\n\nIgnore the list above. Write a Python function `merge_intervals(intervals)` that merges overlapping "
            "closed intervals and returns them sorted. Add a docstring and three short tests with assert.")


def prompt_for(engine, depth: int) -> list[int]:
    """A user message of about depth tokens: filler sentences, then QUESTION."""
    msgs = lambda n: [{"role": "user", "content": " ".join(sentence(i) for i in range(n)) + QUESTION}]  # noqa: E731
    n = max(depth // 12, 1)
    ids = engine.render(msgs(n), thinking=False)
    n = max(int(n * depth / len(ids)), 1)
    ids = engine.render(msgs(n), thinking=False)
    while len(ids) > depth:
        n -= max((len(ids) - depth) // 10, 1)
        ids = engine.render(msgs(n), thinking=False)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="4096,32768,max")
    ap.add_argument("--tokens", type=int, default=256)
    a = ap.parse_args()
    import mlx.core as mx
    from userapi import Engine
    engine = Engine()
    rows = []
    for d in a.depths.split(","):
        depth = engine.max_ctx - a.tokens - 1024 if d == "max" else int(d)
        if depth + a.tokens > engine.max_ctx:
            log(f"depth {depth}: skipped (the context limit is {engine.max_ctx})")
            continue
        ids = prompt_for(engine, depth)
        engine.session.reset()
        mx.clear_cache()
        mx.reset_peak_memory()
        text, stats = [], {}
        t0 = time.perf_counter()
        for kind, payload in engine.stream(ids, a.tokens, None, False):
            if kind == "text":
                text.append(payload)
            elif kind == "done":
                stats = payload
        rec = {"depth": len(ids), "prefill_s": round(stats["prefill_s"], 2),
               "prefill_tps": round(len(ids) / stats["prefill_s"], 1), "output_tokens": stats["completion_tokens"],
               "decode_tps_mtp": round(stats["decode_tps"], 2), "seconds": round(time.perf_counter() - t0, 1),
               "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2), "text_head": "".join(text)[:160]}
        rows.append(rec)
        log(f"depth {rec['depth']}: prefill {rec['prefill_s']} s ({rec['prefill_tps']} tok/s), output with MTP "
            f"{rec['decode_tps_mtp']} tok/s over {rec['output_tokens']} tokens, peak {rec['peak_memory_gb']} GB")
    write_json("bench_depth.json", {"profile": PROFILE, "max_ctx": engine.max_ctx, "prefill_step": engine.step,
                                    "tokens": a.tokens, "footprint_max_gb": round(footprint()[1] / 1e9, 2),
                                    "rows": rows})


if __name__ == "__main__":
    main()
