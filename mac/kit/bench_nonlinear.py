"""Where does the non-linear part of a forward go, and why does it jump from M=1 to M=2? All arms stub the coded
linears to zeros (timing only, wrong values). Synced forward of M tokens at 1k context, min of 8, cache state
evaluated (so MLX cannot skip the body).

  zl            coded linears -> zeros (the rest of the forward)
  zl_attn0      + full-attention core (_attention) -> zeros
  zl_gdn0       + gated_delta_update -> zeros (state unchanged)
  zl_both0      + both

    python kit/bench_nonlinear.py [--ms 1,2,4,8] [--out file.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, log, prefill  # noqa: E402
from bench_longprefill import ids_for  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="1,2,4,8")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import mlx.core as mx
    import mlx_lm.models.qwen3_5 as mq
    from escha_mlx import dense
    from escha_mlx.models import qwen3_5 as eq
    model, tok, _ = load()
    ids = ids_for(tok, 1024)
    orig = {"attn": eq._attention, "gdn": mq.gated_delta_update}

    def zero_call(self, x):
        return mx.zeros(x.shape[:-1] + (self._w.OC,), x.dtype)

    dense.EschaLinear.__call__ = zero_call
    dense.project_pair = lambda f, s, x: (zero_call(f, x), zero_call(s, x))

    def attn0(q, k, v, cache, scale, mask):
        return mx.zeros(q.shape[:-1] + (v.shape[-1],), q.dtype)

    def gdn0(q, k, v, a_, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):
        B, T = q.shape[:2]
        Hv, Dv = v.shape[-2:]
        if state is None:
            state = mx.zeros((B, Hv, Dv, q.shape[-1]), dtype=mx.float32)
        return mx.zeros((B, T, Hv, Dv), v.dtype), state

    from mlx_lm.models import gated_delta as gd

    def gdn_kstub(q, k, v, a_, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):
        g = gd.compute_g(A_log, a_, dt_bias)
        beta = mx.sigmoid(b)
        parts = {"q": q, "k": k, "v": v, "g": g, "beta": beta}
        sel = (os.environ.get("NL_LIVE") or "q,k,v,g,beta").split(",")
        live = sum(parts[n].astype(mx.float32).sum() for n in sel)
        B, T = q.shape[:2]
        Hv, Dv = v.shape[-2:]
        y = mx.zeros((B, T, Hv, Dv), v.dtype) + (live * 0).astype(v.dtype)
        if state is None:
            state = mx.zeros((B, Hv, Dv, q.shape[-1]), dtype=mx.float16)
        return y, state + (live * 0).astype(state.dtype)

    arms = {"zl": (orig["attn"], orig["gdn"]), "zl_attn0": (attn0, orig["gdn"]), "zl_kstub": (orig["attn"], gdn_kstub),
            "zl_gdn0": (orig["attn"], gdn0), "zl_both0": (attn0, gdn0)}
    res = {}
    for name in (os.environ.get("NL_ARMS") or "zl,zl_attn0,zl_gdn0,zl_both0,zl").split(","):
        eq._attention, mq.gated_delta_update = arms[name]
        row = {}
        for m in [int(x) for x in a.ms.split(",")]:
            cache = model.make_cache()
            prefill(model, cache, ids)
            x = mx.array([[ids[0]] * m], dtype=mx.uint32)
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
            mx.clear_cache()
        res[name if name not in res else name + "_again"] = row
        log(f"{name}: " + ", ".join(f"M={m} {v} ms" for m, v in row.items()))
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
