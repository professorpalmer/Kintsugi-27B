"""Paired in-process A/B of one environment knob that escha reads per call (for example ESCHA_MLX_DENSE_PF) on MTP
decode: the value is flipped between rounds, alternating order. MTP tok/s over the bench_mtp prompts, and whether both arms
give the same tokens.

    python kit/bench_ab_env_mtp.py --env NAME --a VAL --b VAL [--rounds 6] [--tokens 192] [--out file.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import chat_ids, load, log  # noqa: E402
from bench_mtp import PROMPTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=192)
    ap.add_argument("--out", default="")
    ap.add_argument("--env", required=True)
    ap.add_argument("--a", required=True, help="value of arm A (the candidate)")
    ap.add_argument("--b", required=True, help="value of arm B (the baseline)")
    a = ap.parse_args()
    import mlx.core as mx
    from escha_mlx import mtp
    model, tok, _ = load(mtp=True)

    def set_pair(on):
        os.environ[a.env] = a.a if on else a.b

    def gen_all():
        n_all, t_all, outs = 0, 0.0, []
        for p in PROMPTS:
            ids = mx.array(chat_ids(tok, p), dtype=mx.uint32)
            out, t0 = [], None
            for t, _, _ in mtp.mtp_generate_step(ids, model, max_tokens=a.tokens):
                if t0 is None:
                    t0 = time.perf_counter()
                out.append(t)
            n_all += len(out) - 1
            t_all += time.perf_counter() - t0
            outs.append(out)
        return n_all / t_all, outs

    for on in (True, False):                             # warm both paths
        set_pair(on)
        gen_all()
    res, pairs, same = {"env": a.env, "a": a.a, "b": a.b, "rounds": []}, [], True
    for i in range(a.rounds):
        got = {}
        for on in ((True, False) if i % 2 == 0 else (False, True)):
            set_pair(on)
            got[on] = gen_all()
        same = same and got[True][1] == got[False][1]
        pairs.append(got[True][0] / got[False][0])
        res["rounds"].append({"a": round(got[True][0], 2), "b": round(got[False][0], 2),
                              "ratio": round(pairs[-1], 4)})
        log(f"round {i}: A {got[True][0]:.2f}, B {got[False][0]:.2f} tok/s, ratio {pairs[-1]:.4f}")
    res["median_ratio"] = round(statistics.median(pairs), 4)
    res["rounds_faster"] = sum(p > 1 for p in pairs)
    res["same_tokens"] = same
    log(f"median MTP ratio {res['median_ratio']}, faster {res['rounds_faster']}/{a.rounds}, same tokens {same}")
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
