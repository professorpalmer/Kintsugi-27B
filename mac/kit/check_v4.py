"""Bit check of ESCHA_MLX_DENSE_V4 on the real model: every coded linear at 1, 2 and 4 rows, and every pair site
(project_pair) at 2 and 4 rows, staged loads (V4=1) against scalar loads (V4=0). Exit 1 on any mismatch.

    python kit/check_v4.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, log  # noqa: E402


def main():
    import mlx.core as mx
    import numpy as np
    from escha_mlx import dense
    model, _, _ = load()
    lins = [m for _, m in model.named_modules() if isinstance(m, dense.EschaLinear)]
    lins = list({id(m): m for m in lins}.values())
    rng = np.random.default_rng(0)
    bad = n = 0
    for lin in lins:
        for m in (1, 2, 4):
            x = mx.array((rng.standard_normal((1, m, lin._w.IC)) * 0.5).astype(np.float16))
            os.environ["ESCHA_MLX_DENSE_V4"] = "0"
            ref = np.array(lin(x))
            os.environ["ESCHA_MLX_DENSE_V4"] = "1"
            bad += not np.array_equal(np.array(lin(x)), ref)
            n += 1
    pairs, seen = [], set()
    for _, mod in model.named_modules():
        for pa, pb in (("gate_proj", "up_proj"), ("in_proj_qkv", "in_proj_z"), ("k_proj", "v_proj")):
            f, s = getattr(mod, pa, None), getattr(mod, pb, None)
            if isinstance(f, dense.EschaLinear) and isinstance(s, dense.EschaLinear) and (id(f), id(s)) not in seen:
                seen.add((id(f), id(s)))
                pairs.append((f, s))
    pbad = pn = 0
    for f, s in pairs:
        for m in (2, 4):
            x = mx.array((rng.standard_normal((1, m, f._w.IC)) * 0.5).astype(np.float16))
            os.environ["ESCHA_MLX_DENSE_V4"] = "0"
            r = [np.array(t) for t in dense.project_pair(f, s, x)]
            os.environ["ESCHA_MLX_DENSE_V4"] = "1"
            g = [np.array(t) for t in dense.project_pair(f, s, x)]
            pbad += not all(np.array_equal(a, b) for a, b in zip(r, g))
            pn += 1
    os.environ.pop("ESCHA_MLX_DENSE_V4", None)
    log(f"v4 bit check: {len(lins)} linears x 3 row counts = {n} calls, {bad} mismatch; "
        f"{len(pairs)} pair sites x 2 row counts = {pn} calls, {pbad} mismatch")
    if bad or pbad:
        sys.exit(1)


if __name__ == "__main__":
    main()
