"""GPU facts for the speed plan: read bandwidth, f16/bf16/f32 GEMM rate at model shapes, int8 matmul support,
and the rate of msl.decode_tiles (trellis code -> f16) at one real projection. Run with the GPU otherwise idle.

    python kit/probe_gpu.py
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def timeit(fn, n=10, warm=3):
    import mlx.core as mx
    for _ in range(warm):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / n


def main():
    import mlx.core as mx
    res = {"device": mx.device_info()}
    # read bandwidth: sum of a 2 GB buffer
    a = mx.random.uniform(shape=(256 * 1024 * 1024,)).astype(mx.float32)   # 1 GiB
    mx.eval(a)
    t = timeit(lambda: a.sum(), n=10)
    res["read_gbps"] = a.nbytes / t / 1e9
    del a
    # GEMM at model shapes: rows x 5120 x 17408 (MLP up), rows x 17408 x 5120 (down)
    g = {}
    for dt in (mx.float16, mx.bfloat16, mx.float32):
        for rows in (256, 1024, 2048):
            for ic, oc in ((5120, 17408), (17408, 5120)):
                x = mx.random.normal((rows, ic)).astype(dt)
                w = mx.random.normal((ic, oc)).astype(dt)
                mx.eval(x, w)
                t = timeit(lambda: x @ w, n=5)
                g[f"{dt}:{rows}x{ic}x{oc}"] = round(2 * rows * ic * oc / t / 1e12, 2)
    res["gemm_tflops"] = g
    try:
        xi = mx.ones((1024, 5120), dtype=mx.int8)
        wi = mx.ones((5120, 17408), dtype=mx.int8)
        t = timeit(lambda: xi @ wi, n=5)
        res["int8_matmul_tops"] = round(2 * 1024 * 5120 * 17408 / t / 1e12, 2)
    except Exception as e:  # noqa: BLE001
        res["int8_matmul_tops"] = f"unsupported: {e!r}"[:200]
    # decode_tiles rate on a real projection
    try:
        from common import load
        from escha_mlx import dense, msl
        model, _, _ = load()
        lins = [m for _, m in model.named_modules() if isinstance(m, dense.EschaLinear)]
        big = max(lins, key=lambda m: m._w.IC * m._w.OC)
        w = big._w
        t = timeit(lambda: msl.decode_tiles(w.code, w.K, w.IC, w.OC), n=10)
        nw = sum(m._w.IC * m._w.OC for m in lins)
        res["decode_tiles"] = {"IC": w.IC, "OC": w.OC, "K": w.K, "s": t, "gweights_per_s": w.IC * w.OC / t / 1e9,
                               "all_coded_weights": nw, "est_full_model_decode_s": nw / (w.IC * w.OC / t)}
    except Exception as e:  # noqa: BLE001
        res["decode_tiles"] = f"error: {e!r}"[:300]
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
