"""Flash attention (escha_mlx.flash_attn) against the blocked SDPA path: correctness, then time per call.

Shape of one Kintsugi full-attention layer in prefill: 24 query heads, 4 K/V heads, head dim 256, one chunk of L
queries at the end of S keys (causal).

    python kit/bench_flash.py [--L 2048] [--S 8192,32768,131072] [--cfg 32:4:0,16:4:0,32:8:0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    import mlx.core as mx
    from escha_mlx.flash_attn import flash_attention
    from escha_mlx.models.qwen3_5 import _attention

    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=2048)
    ap.add_argument("--S", default="8192,32768,131072")
    ap.add_argument("--cfg", default="32:6:1:0,16:6:1:1,32:3:2:0,32:2:2:0,32:1:2:1")  # BK:GH:RB:DB
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    HQ, HKV, D = 24, 4, 256
    scale = D ** -0.5
    cfgs = [tuple(int(x) for x in c.split(":")) for c in a.cfg.split(",")]
    kw = lambda c: dict(BK=c[0], GH=c[1], RB=c[2], DB=bool(c[3]))  # noqa: E731

    def ref32(q, k, v, causal):
        L, S = q.shape[2], k.shape[2]
        kk = mx.repeat(k, HQ // HKV, axis=1).astype(mx.float32)
        vv = mx.repeat(v, HQ // HKV, axis=1).astype(mx.float32)
        s = (q.astype(mx.float32) @ kk.swapaxes(-1, -2)) * scale
        if causal:
            qi = mx.arange(L)[:, None] + (S - L)
            s = mx.where(mx.arange(S)[None] <= qi, s, -mx.inf)
        return mx.softmax(s, axis=-1) @ vv

    res = {"correct": [], "speed": []}
    for (L, S, causal) in ((64, 64, True), (100, 300, True), (2048, 2048, True), (77, 1000, False), (512, 4133, True)):
        q = mx.random.normal((1, HQ, L, D)).astype(mx.float16)
        k = mx.random.normal((1, HKV, S, D)).astype(mx.float16)
        v = mx.random.normal((1, HKV, S, D)).astype(mx.float16)
        r = ref32(q, k, v, causal)
        if S > 64:                                       # K/V as slices of a larger buffer, as a reserved cache gives
            kb = mx.zeros((1, HKV, S + 37, D), mx.float16)
            vb = mx.zeros((1, HKV, S + 37, D), mx.float16)
            kb[:, :, :S] = k
            vb[:, :, :S] = v
            k, v = kb[:, :, :S], vb[:, :, :S]
            q = (q.transpose(0, 2, 1, 3) + 0).transpose(0, 2, 1, 3)   # same values, [B, L, H, D] memory: strided q view
        for c in cfgs:
            o = flash_attention(q, k, v, scale, causal, **kw(c))
            err = float(mx.abs(o.astype(mx.float32) - r).max())
            res["correct"].append({"L": L, "S": S, "causal": causal, "cfg": list(c), "max_abs_err": err})
            print(f"L={L} S={S} causal={causal} cfg={c}: max err {err:.4f}", flush=True)

    def timeit(fn, n):                                  # minimum of single synced calls: robust to other GPU load
        for _ in range(2):
            mx.eval(fn())
        mx.synchronize()
        best = float("inf")
        for _ in range(n):
            t0 = time.perf_counter()
            mx.eval(fn())
            mx.synchronize()
            best = min(best, time.perf_counter() - t0)
        return best

    L = a.L
    for S in [int(s) for s in a.S.split(",")]:
        q = mx.random.normal((1, HQ, L, D)).astype(mx.float16)
        k = mx.random.normal((1, HKV, S, D)).astype(mx.float16)
        v = mx.random.normal((1, HKV, S, D)).astype(mx.float16)
        mx.eval(q, k, v)
        fl = 4 * HQ * D * L * (S - L / 2)               # QK + PV over the causal triangle at the end of S
        n = max(2, int(20 * 8192 / S))
        tb = timeit(lambda: _attention(q, k, v, None, scale, "causal"), n)
        row = {"S": S, "blocked_ms": round(tb * 1e3, 1), "blocked_tf": round(fl / tb / 1e12, 1)}
        line = f"S={S}: blocked {tb * 1e3:.0f} ms ({fl / tb / 1e12:.1f} TF)"
        for c in cfgs:
            try:
                tf = timeit(lambda: flash_attention(q, k, v, scale, True, **kw(c)), n)
                row["flash_" + "_".join(map(str, c)) + "_ms"] = round(tf * 1e3, 1)
                line += f" | {c}: {tf * 1e3:.0f} ms ({fl / tf / 1e12:.1f} TF) x{tb / tf:.2f}"
            except Exception as e:  # noqa: BLE001
                row["flash_" + "_".join(map(str, c))] = f"error {e!r}"[:200]
                line += f" | {c}: ERR"
        res["speed"].append(row)
        print(line, flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
