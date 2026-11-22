"""Paired in-process A/B of one environment knob that escha reads per call (for example ESCHA_MLX_DENSE_PF): the value is flipped between
rounds (A B B A A B ...), so slow drift of the machine cancels in the paired ratio. AR decode from a fresh prefill.

    python kit/bench_ab_env.py --env NAME --a VAL --b VAL [--ctx 4096] [--tokens 128] [--rounds 8] [--out file.json]
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
    ap.add_argument("--env", required=True)
    ap.add_argument("--a", required=True, help="value of arm A (the candidate)")
    ap.add_argument("--b", required=True, help="value of arm B (the baseline)")
    a = ap.parse_args()
    import mlx.core as mx
    model, tok, _ = load()
    ids = ids_for(tok, a.ctx)

    def run(fused):
        os.environ[a.env] = a.a if fused else a.b
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
        rows.append({"round": i, "a_tps": round(got[True][0], 2), "b_tps": round(got[False][0], 2),
                     "ratio": round(ratio, 4)})
        log(f"round {i}: A {got[True][0]:.2f}, B {got[False][0]:.2f}, ratio {ratio:.4f}")
    pairs.sort()
    res = {"env": a.env, "a": a.a, "b": a.b, "ctx": a.ctx, "tokens": a.tokens, "rounds": rows, "median_ratio": round(statistics.median(pairs), 4),
           "min_ratio": round(pairs[0], 4), "max_ratio": round(pairs[-1], 4), "rounds_faster": sum(p > 1 for p in pairs),
           "same_tokens": same}
    log(f"{a.env} {a.a} vs {a.b}, ctx {a.ctx}: median A/B {res['median_ratio']} (min {res['min_ratio']}, max {res['max_ratio']}), "
        f"A faster in {res['rounds_faster']}/{a.rounds} rounds, same tokens {same}")
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
