"""Paired in-process A/B of ESCHA_MLX_DENSE_FUSED_M1: the latched flag is flipped on every coded linear between
rounds (A B B A A B ...), so slow drift of the machine cancels in the paired ratio. AR decode from a fresh prefill.

    python kit/bench_ab_fused.py [--ctx 4096] [--tokens 128] [--rounds 8] [--out file.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import decode, load, log, prefill  # noqa: E402
from bench_longprefill import ids_for  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import mlx.core as mx
    from escha_mlx import dense
    model, tok, _ = load()
    lins = [m for _, m in model.named_modules() if isinstance(m, dense.EschaLinear)]
    ids = ids_for(tok, a.ctx)

    def run(fused):
        for lin in lins:
            lin._fused_m1 = fused
        cache = model.make_cache()
        _, last = prefill(model, cache, ids)
        out, dt = decode(model, cache, last, a.tokens)
        del cache
        mx.clear_cache()
        return (len(out) - 1) / dt, out

    run(True)
    run(False)                                           # warm both paths
    rows, pairs, same = [], [], True
    for i in range(a.rounds):
        order = (True, False) if i % 2 == 0 else (False, True)
        got = {}
        for f in order:
            got[f] = run(f)
        same = same and got[True][1] == got[False][1]
        ratio = got[True][0] / got[False][0]
        pairs.append(ratio)
        rows.append({"round": i, "fused_tps": round(got[True][0], 2), "unfused_tps": round(got[False][0], 2),
                     "ratio": round(ratio, 4)})
        log(f"round {i}: fused {got[True][0]:.2f}, unfused {got[False][0]:.2f}, ratio {ratio:.4f}")
    pairs.sort()
    res = {"ctx": a.ctx, "tokens": a.tokens, "rounds": rows, "median_ratio": round(statistics.median(pairs), 4),
           "min_ratio": round(pairs[0], 4), "max_ratio": round(pairs[-1], 4), "rounds_faster": sum(p > 1 for p in pairs),
           "same_tokens": same}
    log(f"ctx {a.ctx}: median fused/unfused {res['median_ratio']} (min {res['min_ratio']}, max {res['max_ratio']}), "
        f"fused faster in {res['rounds_faster']}/{a.rounds} rounds, same tokens {same}")
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
