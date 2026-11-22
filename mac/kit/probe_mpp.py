"""Metal 4 MPP matmul2d rate on this GPU: half x half -> float, int8 x int8 -> int32, half x int8 -> float.
Same kernel shape for each type, so the ratio is the answer. Also mx.matmul f16 as the reference.

    python kit/probe_mpp.py [--m 2048] [--k 5120] [--n 17408]
"""
from __future__ import annotations

import argparse
import json
import time

HEADER = """
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""


def source(ta: str, tb: str, tc: str, bm: int, bn: int, sg: int) -> str:
    return f"""
    constexpr auto desc = matmul2d_descriptor({bm}, {bn}, static_cast<int>(dynamic_extent), false, false, true);
    matmul2d<desc, execution_simdgroups<{sg}>> op;
    tensor<device {ta}, dextents<int32_t, 2>, tensor_inline> A((device {ta}*)a, dextents<int32_t, 2>(K, M));
    tensor<device {tb}, dextents<int32_t, 2>, tensor_inline> B((device {tb}*)b, dextents<int32_t, 2>(N, K));
    tensor<device {tc}, dextents<int32_t, 2>, tensor_inline> C((device {tc}*)c, dextents<int32_t, 2>(N, M));
    uint m0 = threadgroup_position_in_grid.y * {bm};
    uint n0 = threadgroup_position_in_grid.x * {bn};
    auto mA = A.slice(0, m0);
    auto mB = B.slice(n0, 0);
    auto mC = C.slice(n0, m0);
    op.run(mA, mB, mC);
"""


def main():
    import mlx.core as mx
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=2048)
    ap.add_argument("--k", type=int, default=5120)
    ap.add_argument("--n", type=int, default=17408)
    a = ap.parse_args()
    M, K, N = a.m, a.k, a.n
    res = {"shape": [M, K, N]}
    mt = {"half": mx.float16, "int8_t": mx.int8, "float": mx.float32, "int32_t": mx.int32}

    def timeit(fn, n=10):
        for _ in range(3):
            mx.eval(fn())
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            mx.eval(fn())
        mx.synchronize()
        return (time.perf_counter() - t0) / n

    x16 = (mx.random.normal((M, K)) * 0.1).astype(mx.float16)
    w16 = (mx.random.normal((K, N)) * 0.1).astype(mx.float16)
    mx.eval(x16, w16)
    res["mx.matmul f16 TFLOPS"] = round(2 * M * K * N / timeit(lambda: x16 @ w16) / 1e12, 2)

    x8 = mx.random.randint(-100, 100, (M, K)).astype(mx.int8)
    w8 = mx.random.randint(-100, 100, (K, N)).astype(mx.int8)
    mx.eval(x8, w8)
    inputs = {"half": (x16, w16), "int8_t": (x8, w8)}
    for ta, tb, tc in (("half", "half", "float"), ("int8_t", "int8_t", "int32_t"), ("half", "int8_t", "float")):
        for bm, bn, sg in ((64, 32, 4), (128, 64, 4), (64, 64, 4), (128, 128, 8)):
            key = f"{ta}x{tb}->{tc} tile {bm}x{bn} sg{sg}"
            try:
                kern = mx.fast.metal_kernel(
                    name=f"mpp_{ta}_{tb}_{tc}_{bm}_{bn}_{sg}".replace("_t", ""),
                    input_names=["a", "b"], output_names=["c"], header=HEADER,
                    source=source(ta, tb, tc, bm, bn, sg))
                A = inputs[ta][0]
                B = inputs[tb][1]

                def run():
                    return kern(inputs=[A, B], template=[("M", M), ("N", N), ("K", K)],
                                grid=((N // bn) * 32 * sg, M // bm, 1), threadgroup=(32 * sg, 1, 1),
                                output_shapes=[(M, N)], output_dtypes=[mt[tc]])[0]
                c = run()
                mx.eval(c)
                if ta == "half" and tb == "half":
                    ref = (x16.astype(mx.float32) @ w16.astype(mx.float32))
                    err = float(mx.abs(c - ref).max() / mx.abs(ref).max())
                elif ta == "int8_t" and tb == "int8_t":
                    ref = (x8.astype(mx.float32) @ w8.astype(mx.float32))
                    err = float(mx.abs(c.astype(mx.float32) - ref).max() / mx.abs(ref).max())
                else:
                    ref = (x16.astype(mx.float32) @ w8.astype(mx.float32))
                    err = float(mx.abs(c - ref).max() / mx.abs(ref).max())
                t = timeit(run)
                res[key] = {"tops": round(2 * M * K * N / t / 1e12, 2), "max_rel_err": err}
            except Exception as e:  # noqa: BLE001
                res[key] = f"error: {e!r}"[:400]
            print(key, res[key], flush=True)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
