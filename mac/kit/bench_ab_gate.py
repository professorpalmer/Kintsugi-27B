"""Paired in-process A/B of ESCHA_MLX_GDN_GATE_ROWS (8 vs 0, flipped on the installed wrappers between rounds):
MTP tok/s over the bench_mtp prompts, plain forward time at M=1/4/8, and per arm how many MTP outputs equal AR.

    python kit/bench_ab_gate.py [--rounds 6] [--tokens 192] [--out file.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import chat_ids, decode, load, log, prefill  # noqa: E402
from bench_mtp import PROMPTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=192)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import mlx.core as mx
    from escha_mlx import mtp
    from escha_mlx.models.qwen3_5 import _GateRowsLinear
    model, tok, _ = load(mtp=True)
    wraps = [m for _, m in model.named_modules() if isinstance(m, _GateRowsLinear)]
    log(f"{len(wraps)} gate row wrappers")

    def set_rows(n):
        for w in wraps:
            w._max_rows = n

    def fwd_table():
        ids = chat_ids(tok, PROMPTS[0])
        row = {}
        for m in (1, 4, 8):
            cache = model.make_cache()
            prefill(model, cache, ids)
            x = mx.array([[ids[-1]] * m], dtype=mx.uint32)
            for _ in range(2):
                mx.eval(model(x, cache=cache), [c.state for c in cache])
            best = 1e9
            for _ in range(8):
                mx.synchronize()
                t0 = time.perf_counter()
                mx.eval(model(x, cache=cache), [c.state for c in cache])
                mx.synchronize()
                best = min(best, time.perf_counter() - t0)
            row[m] = round(best * 1e3, 2)
            del cache
        return row

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

    set_rows(0)
    ar = []
    for p in PROMPTS:
        cache = model.make_cache()
        _, last = prefill(model, cache, chat_ids(tok, p))
        ar.append(decode(model, cache, last, a.tokens)[0])
        del cache
    res = {"forward_ms": {}, "rounds": []}
    for n in (8, 0):
        set_rows(n)
        res["forward_ms"][n] = fwd_table()
        log(f"forward rows={n}: {res['forward_ms'][n]}")
        gen_all()
    same = {8: None, 0: None}
    pairs = []
    for i in range(a.rounds):
        got = {}
        for n in ((8, 0) if i % 2 == 0 else (0, 8)):
            set_rows(n)
            got[n] = gen_all()
            if same[n] is None:
                same[n] = sum(o == r for o, r in zip(got[n][1], ar))
        pairs.append(got[8][0] / got[0][0])
        res["rounds"].append({"rows8": round(got[8][0], 2), "rows0": round(got[0][0], 2), "ratio": round(pairs[-1], 4)})
        log(f"round {i}: rows8 {got[8][0]:.2f}, rows0 {got[0][0]:.2f} tok/s, ratio {pairs[-1]:.4f}")
    res["median_ratio"] = round(statistics.median(pairs), 4)
    res["rounds_faster"] = sum(p > 1 for p in pairs)
    res["mtp_equals_ar"] = {"rows8": same[8], "rows0": same[0], "prompts": len(PROMPTS)}
    log(f"median MTP ratio {res['median_ratio']}, faster {res['rounds_faster']}/{a.rounds}; MTP==AR rows8 {same[8]}/{len(PROMPTS)}, rows0 {same[0]}/{len(PROMPTS)}")
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
