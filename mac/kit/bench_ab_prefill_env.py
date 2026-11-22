"""Paired in-process A/B of one per-call escha knob on PREFILL: prefill the same CTX tokens with A and B, alternating
order, and compare tok/s and the last logits (bit for bit).

    python kit/bench_ab_prefill_env.py --env NAME --a VAL --b VAL [--ctx 4096] [--rounds 4]
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, log, prefill  # noqa: E402
from bench_longprefill import ids_for  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--rounds", type=int, default=4)
    a = ap.parse_args()
    import mlx.core as mx
    import numpy as np
    model, tok, _ = load()
    ids = ids_for(tok, a.ctx)

    def run(val):
        os.environ[a.env] = val
        cache = model.make_cache()
        mx.synchronize()
        t0 = time.perf_counter()
        _, last = prefill(model, cache, ids)
        mx.eval(last)
        mx.synchronize()
        dt = time.perf_counter() - t0
        out = np.array(last)
        del cache
        mx.clear_cache()
        return len(ids) / dt, out

    run(a.b)
    ratios, same = [], True
    for r in range(a.rounds):
        order = (a.a, a.b) if r % 2 == 0 else (a.b, a.a)
        res = {v: run(v) for v in order}
        ratios.append(res[a.a][0] / res[a.b][0])
        same &= bool(np.array_equal(res[a.a][1], res[a.b][1]))
        log(f"round {r}: A {res[a.a][0]:.1f} tok/s, B {res[a.b][0]:.1f} tok/s, ratio {ratios[-1]:.4f}")
    log(f"ctx {a.ctx} prefill: median A/B {statistics.median(ratios):.4f}, same last logits {same}")


if __name__ == "__main__":
    main()
