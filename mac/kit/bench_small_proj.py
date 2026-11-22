"""in_proj_a + in_proj_b (two f16 48x5120 linears per GDN layer) at M rows: variants, 48 layers chained, and whether
each equals the shipped output bit for bit.

  ship     two nn.Linear calls (x @ W.T each)
  cat      one [96, 5120] weight, x @ Wcat.T, then split
  catT     one pre-transposed contiguous [5120, 96] weight, x @ WcatT
  wx       Wcat @ x.T (output [96, M]), then transpose
  rows     M separate one-row products with Wcat (the GEMV kernel), then stack
  wrap     the modules as installed: with ESCHA_MLX_GDN_GATE_ROWS on, 2 x M one-row calls (48 wide each)
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load  # noqa: E402


def main():
    import mlx.core as mx
    import numpy as np
    model, _, _ = load()
    lays = [l.linear_attn for l in model.language_model.model.layers if getattr(l, "is_linear", False)]
    def plain(m):                                        # unwrap the escha stage-9 row wrapper if installed
        return getattr(m, "inner", m)
    Wa = [plain(l.in_proj_a).weight for l in lays]
    Wb = [plain(l.in_proj_b).weight for l in lays]
    Wc = [mx.concatenate([a, b], axis=0) for a, b in zip(Wa, Wb)]
    WcT = [mx.contiguous(w.T) for w in Wc]
    mx.eval(Wc, WcT)
    n = Wa[0].shape[0]

    def ship(i, x):                                      # the plain nn.Linear calls (one GEMM at 2+ rows)
        return plain(lays[i].in_proj_a)(x), plain(lays[i].in_proj_b)(x)

    def cat(i, x):
        y = x @ Wc[i].T
        return y[..., :n], y[..., n:]

    def catT(i, x):
        y = x @ WcT[i]
        return y[..., :n], y[..., n:]

    def wx(i, x):
        y = (Wc[i] @ x.reshape(-1, x.shape[-1]).T).T.reshape(*x.shape[:-1], 2 * n)
        return y[..., :n], y[..., n:]

    def rows(i, x):
        r = x.reshape(-1, x.shape[-1])
        y = mx.concatenate([r[j:j + 1] @ Wc[i].T for j in range(r.shape[0])], axis=0).reshape(*x.shape[:-1], 2 * n)
        return y[..., :n], y[..., n:]

    def wrap(i, x):                                      # the installed module (stage-9 row wrapper when on)
        return lays[i].in_proj_a(x), lays[i].in_proj_b(x)

    arms = {"ship": ship, "cat": cat, "catT": catT, "wx": wx, "rows": rows, "wrap": wrap}

    def chain(M, fn):
        x0 = (mx.random.normal((1, M, 5120)) * 0.5).astype(mx.float16)
        mx.eval(x0)
        best = 1e9
        for rep in range(6):
            mx.synchronize()
            t0 = time.perf_counter()
            x = x0
            outs = []
            for i in range(len(lays)):
                a, b = fn(i, x)
                outs += [a, b]
                x = x0 + ((a.sum() + b.sum()) * 0).astype(x0.dtype)
            mx.eval(outs, x)
            mx.synchronize()
            if rep:
                best = min(best, time.perf_counter() - t0)
        return best * 1e3

    for M in (1, 2, 4, 8, 16):
        x = (mx.random.normal((1, M, 5120)) * 0.5).astype(mx.float16)
        ra, rb = ship(0, x)
        line = []
        for name, fn in arms.items():
            a, b = fn(0, x)
            same = np.array_equal(np.array(a), np.array(ra)) and np.array_equal(np.array(b), np.array(rb))
            line.append(f"{name} {chain(M, fn):.2f}{'' if same else '*'}")
        print(f"M={M}: " + ", ".join(line) + f" ms/{len(lays)}  (* = not bit-identical to ship)", flush=True)


if __name__ == "__main__":
    main()
