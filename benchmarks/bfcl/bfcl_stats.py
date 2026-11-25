"""BFCL v3 multi-turn: Kintsugi-27B against Qwen3.8-27B BF16, from the per-entry results in this folder.
    python bfcl_stats.py
Needs Python 3 and numpy. Reads kintsugi-27b.csv, kintsugi-27b-repeat.csv and qwen3.8-27b-bf16.csv (columns id, pass;
1 = the official `bfcl evaluate` passed the entry).

Method: the 4 entries of one scenario (base, missing function, missing parameter and long context with the same index)
are paired. For each scenario s, d_s = the sum over its 4 entries of (pass_A - pass_B); D = the sum of d_s.
p value: a two-sided sign-flip test over the scenarios (exact if 20 or fewer scenarios have d_s != 0, else 100,000
random sign vectors). 95 % interval: a scenario bootstrap (10,000 resamples). Random generator: numpy
default_rng(20261124) for each of the two, so the output is the same on every run."""
import csv
import itertools
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CATS = ["multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"]


def load(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return {r["id"]: int(r["pass"]) for r in csv.DictReader(f)}


def signflip(d, rng):
    d = np.asarray([x for x in d if x != 0], dtype=float)
    obs = abs(d.sum())
    if len(d) == 0:
        return 1.0
    if len(d) <= 20:
        tot = [abs(sum(s * x for s, x in zip(signs, d))) for signs in itertools.product((1, -1), repeat=len(d))]
        return float(np.mean(np.asarray(tot) >= obs - 1e-9))
    s = rng.choice((-1.0, 1.0), size=(100000, len(d)))
    return float((np.abs(s @ d) >= obs - 1e-9).mean())


def compare(name_a, A, name_b, B):
    ids = sorted(set(A) & set(B))
    d = {}
    for i in ids:
        s = int(i.rsplit("_", 1)[1])
        d[s] = d.get(s, 0) + A[i] - B[i]
    dv = np.asarray([d[s] for s in sorted(d)], dtype=float)
    p = signflip(dv, np.random.default_rng(20261124))
    rng = np.random.default_rng(20261124)
    bs = np.array([dv[rng.integers(0, len(dv), len(dv))].sum() for _ in range(10000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"{name_a} vs {name_b}: {len(ids)} paired entries, {len(dv)} scenarios")
    for c in CATS:
        a = sum(A[i] for i in ids if i.rsplit("_", 1)[0] == c)
        b = sum(B[i] for i in ids if i.rsplit("_", 1)[0] == c)
        print(f"  {c[len('multi_turn_'):]:<14} {a:>4} vs {b:>4} of 200  ({a - b:+d})")
    print(f"  total          {sum(A[i] for i in ids):>4} vs {sum(B[i] for i in ids):>4} of {len(ids)}  (D = {int(dv.sum()):+d})")
    print(f"  scenarios: {name_a} higher in {int((dv > 0).sum())}, {name_b} higher in {int((dv < 0).sum())}, "
          f"level in {int((dv == 0).sum())}")
    print(f"  entries that differ: {sum(1 for i in ids if A[i] != B[i])}")
    print(f"  sign-flip p (scenarios) = {p:.4g}; scenario bootstrap 95 % interval [{lo:+.0f}, {hi:+.0f}]")
    print()


if __name__ == "__main__":
    K, R, B = load("kintsugi-27b.csv"), load("kintsugi-27b-repeat.csv"), load("qwen3.8-27b-bf16.csv")
    compare("Kintsugi-27B", K, "Qwen3.8-27B BF16", B)
    compare("Kintsugi-27B repeat", R, "Kintsugi-27B", K)
