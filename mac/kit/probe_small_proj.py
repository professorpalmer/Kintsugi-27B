"""Type and cost of the small GDN projections (in_proj_a / in_proj_b, hidden -> 48) at M=1..8, 48 chained."""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load  # noqa: E402


def main():
    import mlx.core as mx
    model, _, _ = load()
    lays = [l.linear_attn for l in model.language_model.model.layers if getattr(l, "is_linear", False)]
    la = lays[0]
    for n in ("in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj"):
        m = getattr(la, n)
        info = {k: (v.shape, v.dtype) for k, v in m.items() if hasattr(v, "shape")} if hasattr(m, "items") else {}
        print(f"{n}: {type(m).__module__}.{type(m).__name__} {info} bits={getattr(m, 'bits', None)} "
              f"gs={getattr(m, 'group_size', None)}", flush=True)

    def chain(M, name):
        x0 = (mx.random.normal((1, M, 5120)) * 0.5).astype(mx.float16)
        mx.eval(x0)
        best = 1e9
        for rep in range(6):
            mx.synchronize()
            t0 = time.perf_counter()
            x = x0
            outs = []
            for l in lays:
                y = getattr(l, name)(x)
                outs.append(y)
                x = x0 + (y.sum() * 0).astype(x0.dtype)
            mx.eval(outs, x)
            mx.synchronize()
            if rep:
                best = min(best, time.perf_counter() - t0)
        return best * 1e3

    for n in ("in_proj_a", "in_proj_b"):
        print(n + ": " + ", ".join(f"M={M} {chain(M, n):.2f}" for M in (1, 2, 4, 8, 1)) + f" ms/{len(lays)}", flush=True)


if __name__ == "__main__":
    main()
