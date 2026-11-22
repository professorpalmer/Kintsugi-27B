"""Small-row coded GEMM (direct_out kernel) in isolation: time per call at M rows, and variants of its row loop.

  ship     msl._dense_gemm_rows_direct_out_source as shipped (x rows as half, converted per FMA group)
  xf32     same kernel, x rows given as f32 (no half->float converts; (float)half is exact, so bit-identical)
  fake     ship with a 2-op cba_decode stub (timing only: shows the decode share at each M)

For each of the largest coded linears (one K2, one K3) and M in --ms: min of 30 synced calls. The gemv row is the
shipped M=1 path (input Hadamard + dense_gemv + output Hadamard).

    python kit/bench_rows_kernel.py [--ms 1,2,4,8] [--out file.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, log  # noqa: E402

FAKE = """
static inline half cba_decode(uint x) {
    return as_type<half>((ushort)((x & 0x8FFFu) ^ 0x3B60u));
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="1,2,4,8,16")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import mlx.core as mx
    import numpy as np
    from escha_mlx import dense, msl
    model, _, _ = load()
    lins = [m for _, m in model.named_modules() if isinstance(m, dense.EschaLinear)]
    pick = {}
    for lin in lins:
        k = lin.K
        if k not in pick or lin._w.IC * lin._w.OC > pick[k]._w.IC * pick[k]._w.OC:
            pick[k] = lin
    lut = msl.use_lut()
    RS = float(dense.RS)
    kerns = {}

    def kernel(name, K, R):
        key = (name, K, R)
        if key not in kerns:
            src = msl._dense_gemm_rows_direct_out_source(K, lut, R, RS)
            header = msl._HEADER
            if name == "xf32":
                src2 = (src.replace("device const half2* xp = (device const half2*)(",
                                    "device const float2* xp = (device const float2*)(")
                           .replace("half2 xv0 = xp[0], xv1 = xp[4];", "float2 xv0 = xp[0], xv1 = xp[4];")
                           .replace("(float)xv0", "xv0").replace("(float)xv1", "xv1"))
                assert src2 != src
                src = src2
            if name == "fake":
                header = msl._HEADER.split("static inline half cba_decode")[0] + FAKE
            kerns[key] = mx.fast.metal_kernel(
                name=f"bench_direct_out_{name}_k{K}_r{R}", input_names=["xh", "code", "rout"] + (["lut"] if lut else []),
                output_names=["out"], header=header, source=src)
        return kerns[key]

    def call(name, lin, x):
        w = lin._w
        m = x.shape[0]
        if name == "gemv":
            return lin._output_rows(lin._gemv(x))
        if name == "rows_sep":                           # shipped multi-row path, output transform separate
            r = msl.dense_block_r(m)
            mid = (msl.dense_gemm_rows_direct(x, w.code, w.K, w.IC, w.OC, r) if r <= 8
                   else msl.dense_gemm_rows(x, w.code, w.K, w.IC, w.OC, r))
            return lin._output_rows(mid)
        if name == "mat":                                # simdgroup-matrix R=16 (rows padded by the kernel guard)
            return lin._output_rows(msl.dense_gemm_mat(x, w.code, w.K, w.IC, w.OC))
        inputs = [x, w.code.reshape(-1), w.rout] + ([msl._lut_array()] if lut else [])
        return kernel(name, lin.K, m)(
            inputs=inputs, template=[("TK", w.IC // 16), ("TN", w.OC // 16), ("IC", w.IC), ("OC", w.OC), ("M", m)],
            grid=(256 * (w.OC // 128), 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(m, w.OC)], output_dtypes=[mx.float16])[0]

    N = 50

    def chain_time(step, x0):
        """Seconds per link of a chain of N dependent links (min of 5 synced runs)."""
        best = 1e9
        for rep in range(6):
            mx.synchronize()
            t0 = time.perf_counter()
            x = x0
            for _ in range(N):
                x = step(x)
            mx.eval(x)
            mx.synchronize()
            if rep:
                best = min(best, time.perf_counter() - t0)
        return best / N

    res = {"links": N, "rows": []}
    for K, lin in sorted(pick.items()):
        w = lin._w
        log(f"K{K}: IC {w.IC} OC {w.OC}, code {w.code.nbytes / 1e6:.1f} MB")
        for m in [int(x) for x in a.ms.split(",")]:
            x = mx.random.normal((m, w.IC)).astype(mx.float16)
            xh = lin._input_rows(x)
            mx.eval(xh)
            row = {"K": K, "IC": w.IC, "OC": w.OC, "M": m}
            ref = np.array(call("ship", lin, xh))
            same = np.array_equal(ref, np.array(call("rows_sep", lin, xh))) if m > 1 else True
            for dt in (mx.float16, mx.float32):
                xd = xh.astype(dt)
                mx.eval(xd)
                glue = chain_time(lambda v, xd=xd: xd + v * 0, xd)
                row["glue_us_" + ("f32" if dt == mx.float32 else "f16")] = round(glue * 1e6, 1)
            arms = ["ship", "fake"] + (["gemv"] if m == 1 else ["rows_sep", "mat"])
            if m > 1:
                ref_sep = np.array(call("rows_sep", lin, xh)).astype(np.float32)
                got = np.array(call("mat", lin, xh)).astype(np.float32)
                row["mat_rel_diff"] = float(np.abs(got - ref_sep).mean() / max(1e-12, np.abs(ref_sep).mean()))
                row["mat_max_diff"] = float(np.abs(got - ref_sep).max())
            for name in arms:
                xd = xh
                glue = row["glue_us_f16"]
                t = chain_time(lambda v, name=name, xd=xd: xd + call(name, lin, v)[:, :w.IC].astype(xd.dtype) * 0, xd)
                row[name + "_us"] = round(t * 1e6 - glue, 1)
            row["ship_equals_rows_sep"] = bool(same) if m > 1 else None      # no rows_sep arm at one row
            row["ship_gbps"] = round(w.code.nbytes / (row["ship_us"] * 1e-6) / 1e9, 1)
            res["rows"].append(row)
            log(f"  M={m}: " + ", ".join(f"{k} {v}" for k, v in row.items() if k.endswith("_us") or k.endswith("gbps")
                                        or k.startswith("glue")) + f", ship==rows_sep {same if m > 1 else 'n/a'}, mat rel {row.get('mat_rel_diff', 0):.1e}")
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
