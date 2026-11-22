"""Convert Kintsugi-27B-12G.gguf into the escha-mlx form, then check the result against the GGUF.

    python convert.py [--gguf models/Kintsugi-27B-12G.gguf] [--out models/Kintsugi-27B-MLX]

Runs on any OS with Python 3.10+ and numpy (no MLX needed). The checks (G1, G3, G4, G5 of the gate) compare the
converted files with the GGUF byte by byte or value by value; results go to results/convert_checks.json.
Skips the conversion when the output already passed the checks (delete the output folder to force it).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("KINTSUGI_RESULTS", os.path.join(ROOT, "results"))


def verdict(r: dict) -> tuple[bool, str]:
    """The verdict is the checker's own pass flags; nothing is decided here (issue #1)."""
    parts = {"G1 coded bytes": r.get("G1", {}).get("pass"), "G3 small tensors": r.get("G3", {}).get("pass"),
             "G4a head+embedding": r.get("G4a", {}).get("pass"), "G4b MTP": r.get("G4b", {}).get("pass"),
             "G5 metadata": r.get("G5", {}).get("pass")}
    ok = all(v is True for v in parts.values())
    note = "; ".join(f"{k} {'pass' if v else 'FAIL'}" for k, v in parts.items())
    g3 = r.get("G3", {}).get("classes", {})
    if "ssm_a = -exp(A_log)" in g3:
        a = g3["ssm_a = -exp(A_log)"]
        note += f" (G3 ssm_a: max {a.get('max_ulp')} ulp, histogram {a.get('ulp_histogram')}"
    if "conv1d" in g3:
        c = g3["conv1d"]
        note += (f"; G3 conv1d: {c['values_not_equal']} values unequal, all in the f16 subnormal range: "
                 f"{c.get('unequal_outside_f16_subnormal') == 0}, max abs error {c['max_abs_err']:.1e})")
    return ok, ("PASS: " if ok else "FAIL: ") + note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=os.path.join(ROOT, "models", "Kintsugi-27B-12G.gguf"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "Kintsugi-27B-MLX"))
    ap.add_argument("--template", default=os.path.join(ROOT, "template"))
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    rpath = os.path.join(RESULTS, "convert_checks.json")
    done = os.path.join(a.out, "kintsugi_conversion.json")
    if os.path.exists(done) and os.path.exists(rpath) and json.load(open(rpath)).get("verdict", "").startswith("PASS"):
        print(f"already converted and checked: {a.out}")
        return
    py = sys.executable
    subprocess.run([py, os.path.join(ROOT, "tools", "convert.py"), "--gguf", a.gguf, "--template", a.template,
                    "--out", a.out, "--no-hash"], check=True)
    if os.path.exists(rpath):
        os.remove(rpath)
    subprocess.run([py, os.path.join(ROOT, "tools", "check.py"), "--gguf", a.gguf, "--conv", a.out,
                    "--escha-mlx", os.path.join(ROOT, "escha-mlx"), "--only", "G1,G3,G4,G5", "--out", rpath],
                   check=True)
    r = json.load(open(rpath, encoding="utf-8"))
    ok, text = verdict(r)
    r["verdict"] = text
    json.dump(r, open(rpath, "w", encoding="utf-8"), indent=2)
    print(text)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
