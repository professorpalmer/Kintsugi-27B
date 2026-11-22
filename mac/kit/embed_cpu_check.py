"""16 GB profile check: the CPU-side embedding (ESCHA_MLX_EMBED_CPU=1) gives the same bits as the GPU embedding.

It builds both forms of the Q8 embedding from the converted checkpoint (no other weights) and compares every row of
the vocabulary, in chunks. GPU memory: the GPU form only (about 1.6 GB). Writes results/embed_cpu_check.json.

    python embed_cpu_check.py [--chunk 4096]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODEL_DIR, log, write_json  # noqa: E402

NAME = "model.language_model.embed_tokens."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=4096)
    a = ap.parse_args()
    os.environ["ESCHA_MLX_EMBED_CPU"] = "1"          # the scales come back as memory maps too
    import mlx.core as mx
    from pathlib import Path

    from escha_mlx import quant
    from escha_mlx.loader import _iter_tensors
    t = {}
    for name, w in _iter_tensors(Path(MODEL_DIR)):
        if name.startswith(NAME):
            t[name[len(NAME):]] = w
        if len(t) == 2:
            break
    w8, gs = t["weight_int8"], t["weight_group_scale"]
    if not isinstance(w8, np.memmap) or not isinstance(gs, np.memmap):
        raise SystemExit("embedding tensors are not memory maps")
    cpu = quant.make_embedding_cpu(w8, gs)
    gpu = quant.make_embedding_grouped(w8, gs)
    mx.eval(gpu.parameters())
    gpu_bytes = sum(v.nbytes for v in (gpu.weight, gpu.scales, gpu.biases))
    n = w8.shape[0]
    t0, diff, first_bad = time.time(), 0, None
    for a0 in range(0, n, a.chunk):
        ids = np.arange(a0, min(n, a0 + a.chunk), dtype=np.int32)
        g = np.array(gpu(mx.array(ids)))
        c = np.array(cpu(mx.array(ids)))
        bad = int((g.view(np.uint16) != c.view(np.uint16)).sum())
        if bad and first_bad is None:
            first_bad = int(a0)
        diff += bad
    # negative control: the same compare between neighbor rows must see differences
    ids = np.arange(0, a.chunk, dtype=np.int32)
    control = int((np.array(gpu(mx.array(ids + 1))).view(np.uint16)
                   != np.array(cpu(mx.array(ids))).view(np.uint16)).sum())
    # the module also takes a [B, S] batch and gives [B, S, K]
    ids2 = mx.array(np.random.default_rng(3).integers(0, n, size=(2, 37)).astype(np.int32))
    shape_ok = tuple(cpu(ids2).shape) == (2, 37, w8.shape[1])
    batch_same = bool(np.array_equal(np.array(cpu(ids2)).view(np.uint16), np.array(gpu(ids2)).view(np.uint16)))
    res = {"rows": int(n), "width": int(w8.shape[1]), "values": int(n * w8.shape[1]), "values_not_equal": diff,
           "first_bad_row_chunk": first_bad, "negative_control_values_not_equal": control, "batch_shape_ok": shape_ok, "batch_bit_equal": batch_same,
           "gpu_form_bytes": int(gpu_bytes), "file_bytes": int(w8.nbytes + gs.nbytes),
           "seconds": round(time.time() - t0, 1),
           "pass": diff == 0 and control > 0 and shape_ok and batch_same}
    write_json("embed_cpu_check.json", res)
    log(f"embed CPU check: {res['values_not_equal']} of {res['values']} values differ; batch ok {batch_same}; "
        f"GPU form {gpu_bytes / 1e9:.3f} GB moved off the GPU; {'PASS' if res['pass'] else 'FAIL'}")
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
