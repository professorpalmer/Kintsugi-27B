"""Prefill transient memory per prefill step: MLX peak during the prefill minus MLX active memory after it.

Active after the prefill holds the model, the reserved K/V cache and the GDN state, so the difference is what one
prefill step needs on top: activations, decoded weights (decode-once path), attention scores. memory_plan() must keep
at least this as its margin, or a Mac plans a context that does not fit.

    python kit/probe_transient.py [--depths 4096,32768] [--steps 256,512,1024,2048]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import GB, load, log, prefill, sentence  # noqa: E402


def ids_for(tok, n):
    ids = tok.encode(" ".join(sentence(i) for i in range(n // 10 + 64)))
    while len(ids) < n:
        ids += ids
    return ids[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="4096,32768")
    ap.add_argument("--steps", default="256,512,1024,2048")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import mlx.core as mx
    from escha_mlx.kv_cache import reserve
    model, tok, _ = load()
    model_active = mx.get_active_memory()
    rows = []
    for depth in [int(d) for d in a.depths.split(",")]:
        ids = ids_for(tok, depth)
        for step in [int(s) for s in a.steps.split(",")]:
            cache = model.make_cache()
            reserve(cache, depth + 1024)
            mx.synchronize()
            mx.clear_cache()
            mx.reset_peak_memory()
            t, _ = prefill(model, cache, ids, step=step)
            mx.synchronize()
            peak, active = mx.get_peak_memory(), mx.get_active_memory()
            r = {"depth": depth, "step": step, "prefill_tps": round(depth / t, 1), "peak_gb": round(peak / GB, 3),
                 "active_after_gb": round(active / GB, 3), "model_gb": round(model_active / GB, 3),
                 "transient_gb": round((peak - active) / GB, 3)}
            rows.append(r)
            log(f"depth {depth} step {step}: transient {r['transient_gb']} GB (peak {r['peak_gb']}, "
                f"active after {r['active_after_gb']}), {r['prefill_tps']} tok/s")
            del cache
            mx.clear_cache()
    print(json.dumps(rows, indent=1))
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
