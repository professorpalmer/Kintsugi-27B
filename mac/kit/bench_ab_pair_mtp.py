"""Paired in-process A/B of ESCHA_MLX_DENSE_PAIR_M1 on MTP decode: the latched flag is flipped on every coded linear
(target and MTP head) between rounds, alternating order. MTP tok/s over the bench_mtp prompts, and whether both arms
give the same tokens.

    ESCHA_MLX_DENSE_PAIR_M1=1 python kit/bench_ab_pair_mtp.py [--rounds 6] [--tokens 192] [--out file.json]
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
    a = ap.parse_args()
    import mlx.core as mx
    from escha_mlx import dense, mtp
    model, tok, _ = load(mtp=True)
    lins = [m for _, m in model.named_modules() if isinstance(m, dense.EschaLinear)]
    log(f"{len(lins)} coded linears")

    def set_pair(on):
        for lin in lins:
            lin._pair_m1 = on

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
    res, pairs, same = {"rounds": []}, [], True
    for i in range(a.rounds):
        got = {}
        for on in ((True, False) if i % 2 == 0 else (False, True)):
            set_pair(on)
            got[on] = gen_all()
        same = same and got[True][1] == got[False][1]
        pairs.append(got[True][0] / got[False][0])
        res["rounds"].append({"pair": round(got[True][0], 2), "single": round(got[False][0], 2),
                              "ratio": round(pairs[-1], 4)})
        log(f"round {i}: pair {got[True][0]:.2f}, single {got[False][0]:.2f} tok/s, ratio {pairs[-1]:.4f}")
    res["median_ratio"] = round(statistics.median(pairs), 4)
    res["rounds_faster"] = sum(p > 1 for p in pairs)
    res["same_tokens"] = same
    log(f"median MTP ratio {res['median_ratio']}, faster {res['rounds_faster']}/{a.rounds}, same tokens {same}")
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
