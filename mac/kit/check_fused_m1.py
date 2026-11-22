"""Every coded linear of the loaded model (target and MTP head): one-row output with ESCHA_MLX_DENSE_FUSED_M1 on
must equal the unfused chain bit for bit.

    python kit/check_fused_m1.py [--rows 4]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, log  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import mlx.core as mx
    import numpy as np
    import escha_mlx
    from escha_mlx import dense
    log(f"escha_mlx from {os.path.dirname(escha_mlx.__file__).replace(os.path.expanduser('~'), '~')}")
    model, _, _ = load(mtp=True)
    lins = []
    seen = set()
    for name, m in model.named_modules():
        if isinstance(m, dense.EschaLinear) and id(m) not in seen:
            seen.add(id(m))
            lins.append((name, m))
    if hasattr(model, "mtp"):
        for name, m in model.mtp.named_modules():
            if isinstance(m, dense.EschaLinear) and id(m) not in seen:
                seen.add(id(m))
                lins.append(("mtp." + name, m))
    rng = np.random.default_rng(5)
    bad, fused_on = [], 0
    for name, lin in lins:
        fused_on += bool(lin._fused_m1)
        for _ in range(a.rows):
            x = mx.array((rng.standard_normal((1, 1, lin._w.IC)) * 3).astype(np.float16))
            lin._fused_m1 = True
            y1 = np.array(lin(x))
            lin._fused_m1 = False
            y0 = np.array(lin(x))
            lin._fused_m1 = True
            if not np.array_equal(y1, y0):
                bad.append({"linear": name, "max_diff": float(np.abs(y1.astype(np.float32) - y0).max())})
                break
    res = {"linears": len(lins), "fused_on_at_load": fused_on, "rows_each": a.rows, "mismatch": bad}
    log(f"{len(lins)} coded linears ({fused_on} fused at load), {a.rows} rows each: {len(bad)} mismatch")
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
