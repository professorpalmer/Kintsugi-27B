"""A/B/A prefill speed in one process, one model load: the prefill GEMM paths of the patched escha-mlx.

  rows   shipped row-blocked kernel (decodes the code once per 16 rows; ESCHA_MLX_DENSE_RECON=0)
  f16    decode once per chunk to f16 + mx.matmul                (ESCHA_MLX_DENSE_RECON=64, ESCHA_MLX_DENSE_INT8=off)
  int8   decode once per chunk to int8 + MPP int8 GEMM           (ESCHA_MLX_DENSE_RECON=64, ESCHA_MLX_DENSE_INT8=auto)

The first arm runs again at the end (drift control).

    python kit/bench_prefill.py [--tokens 4096] [--chunk 2048] [--arms int8,f16,rows,int8+flash]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, log, sentence  # noqa: E402


def ids_for(tok, n):
    ids = tok.encode(" ".join(sentence(i) for i in range(n // 10 + 64)))
    while len(ids) < n:
        ids += ids
    return ids[:n]


def run(model, ids, chunk):
    import mlx.core as mx
    from escha_mlx.kv_cache import reserve
    cache = model.make_cache()
    reserve(cache, len(ids) + 1024)
    mx.synchronize()
    t0 = time.perf_counter()
    for a in range(0, len(ids), chunk):
        out = model(mx.array(ids[a:a + chunk])[None], cache=cache)
        mx.eval(out, [c.state for c in cache])
    dt = time.perf_counter() - t0
    peak = mx.get_peak_memory() / 1e9
    del cache
    mx.clear_cache()
    mx.reset_peak_memory()
    return len(ids) / dt, peak


def set_arm(model, arm):
    """arm = rows | f16 | int8, with an optional +flash (head-dim-256 flash attention on; else off)."""
    from escha_mlx import dense
    gemm, _, flash = arm.partition("+")
    recon = 0 if gemm == "rows" else 64
    for _, m in model.named_modules():
        if isinstance(m, dense.EschaLinear):
            m._recon_min = recon
    os.environ["ESCHA_MLX_DENSE_INT8"] = "off" if gemm == "f16" else "auto"
    os.environ["ESCHA_MLX_ATTN_FLASH"] = "auto" if flash == "flash" else "off"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--arms", default="int8,f16,rows")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    model, tok, _ = load()
    ids = ids_for(tok, a.tokens)
    arms = a.arms.split(",")
    for arm in arms:                                     # warm every arm at this shape
        set_arm(model, arm)
        run(model, ids[:a.chunk], a.chunk)
    res = {"tokens": a.tokens, "chunk": a.chunk}
    for i, arm in enumerate(arms + [arms[0]]):
        set_arm(model, arm)
        tps, peak = run(model, ids, a.chunk)
        name = arm if i < len(arms) else arm + "_again"
        res[name] = round(tps, 1)
        res[name + "_peak_gb"] = round(peak, 2)
        log(f"{name}: {tps:.1f} tok/s, peak {peak:.2f} GB")
    print(json.dumps(res))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
