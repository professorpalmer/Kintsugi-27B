"""Long prefill, chunk by chunk: time per chunk, MLX active/peak memory and swap, with or without a pre-sized
K/V cache (mlx-lm KVCache grows by concatenate: each growth copies the whole cache and holds two copies).

    python kit/bench_longprefill.py --tokens 131072 [--chunk 2048] [--prealloc]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, log, sentence  # noqa: E402


def ids_for(tok, n):
    ids = tok.encode(" ".join(sentence(i) for i in range(n // 10 + 64)))
    while len(ids) < n:
        ids += ids
    return ids[:n]


def swap_mb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return out.split("used = ")[1].split("M")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=131072)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--prealloc", action="store_true")
    a = ap.parse_args()
    import mlx.core as mx
    model, tok, _ = load()
    ids = ids_for(tok, a.tokens)
    cache = model.make_cache()
    if a.prealloc:
        from escha_mlx.kv_cache import reserve
        reserve(cache, a.tokens + 1024)
    t_all = time.perf_counter()
    t0 = t_all
    for i, s in enumerate(range(0, len(ids), a.chunk)):
        out = model(mx.array(ids[s:s + a.chunk])[None], cache=cache)
        mx.eval(out, [c.state for c in cache])
        if (i + 1) % 8 == 0 or s + a.chunk >= len(ids):
            t = time.perf_counter()
            n = min(8, i + 1) * a.chunk
            log(f"offset {s + a.chunk:7d}: {n / (t - t0):7.1f} tok/s (last {n}), active {mx.get_active_memory() / 1e9:5.1f} GB, "
                f"peak {mx.get_peak_memory() / 1e9:5.1f} GB, swap {swap_mb()} MB")
            t0 = t
    log(f"total: {len(ids) / (time.perf_counter() - t_all):.1f} tok/s over {len(ids)} tokens")


if __name__ == "__main__":
    main()
