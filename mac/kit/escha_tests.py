"""Step 2: escha-mlx's own test suite (Metal kernel gates against the goldens) plus our added tests
(tests/test_kintsugi_grouped.py: per-group Q8 repack, 4-bit MTP linears, the per-group head kernel)."""
from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, log, out_path, write_json  # noqa: E402


def pytest(args, name):
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", *args], cwd=os.path.join(ROOT, "escha-mlx"),
                       capture_output=True, text=True)
    open(out_path(f"pytest_{name}.txt"), "w", encoding="utf-8").write(p.stdout + "\n" + p.stderr)
    lines = [ln for ln in p.stdout.strip().splitlines() if ln.strip()]
    return p.returncode, (lines[-1] if lines else "no output")


def main():
    rc_k, s_k = pytest(["tests/test_kintsugi_grouped.py"], "kintsugi")
    log(f"tests (ours): {s_k}")
    rc_a, s_a = pytest(["tests"], "all")
    log(f"tests (escha-mlx suite + ours): {s_a}")
    write_json("escha_tests.json", {"kintsugi_tests": s_k, "kintsugi_rc": rc_k, "suite": s_a, "suite_rc": rc_a,
                                    "summary": f"ours: {s_k} | full suite: {s_a}"})
    sys.exit(0 if rc_k == 0 and rc_a == 0 else 1)


if __name__ == "__main__":
    main()
