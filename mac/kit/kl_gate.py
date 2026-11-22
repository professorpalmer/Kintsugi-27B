"""Paired KL gate for one per-call escha knob: score each document of a golden set with A and with B in ONE
process (same model, same prefill step), then bound the relative KL change with a paired bootstrap over documents.

    python kit/kl_gate.py --golden DIR --env ESCHA_MLX_ATTN_FLASH --a on --b off [--step 2048] [--docs N]
                          [--line 0.005] [--out kl_gate.json]

DIR has the golden-mini format (kld.json, top parts, rows.f32, gen.u8, tokens.i32). The statistic is
R = sum(KL_A) / sum(KL_B) - 1 over the "gen" rows of all documents. The gate passes when the upper end of the
95 % bootstrap interval (97.5th percentile, 10,000 resamples of documents) is at most --line.
The arm order alternates per document (A B, then B A).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PROFILE, log, write_json  # noqa: E402
from kl_mini import Ref  # noqa: E402


def score_doc(model, ref, i, step):
    import mlx.core as mx
    doc = ref.docs[i]
    toks = ref.doc_tokens(i).tolist()
    sf = doc["score_from"]
    cache = model.make_cache()
    acc = {"kl": 0.0, "top1": 0, "nll_q": 0.0, "nll_p": 0.0, "gen": 0}
    for a0 in range(0, len(toks) - 1, step):
        b0 = min(len(toks) - 1, a0 + step)
        logits = model(mx.array(toks[a0:b0])[None], cache=cache)[0]
        mx.eval(logits)
        lo, hi = max(a0, sf), b0
        if hi > lo:
            cand = np.array(logits[lo - a0:hi - a0].astype(mx.float32))
            for s in range(0, hi - lo, 64):
                kl, t1, nq, npp, gen = ref.score_rows(i, lo - sf + s, cand[s:s + 64])
                acc["kl"] += float(kl[gen].sum())
                acc["top1"] += int(t1[gen].sum())
                acc["nll_q"] += float(nq[gen].sum())
                acc["nll_p"] += float(npp[gen].sum())
                acc["gen"] += int(gen.sum())
        del logits
    del cache
    mx.clear_cache()
    return acc


def bootstrap(ka, kb, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ka), size=(n, len(ka)))
    r = ka[idx].sum(1) / kb[idx].sum(1) - 1.0
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 95)), float(np.percentile(r, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--step", type=int, default=2048)
    ap.add_argument("--docs", type=int, default=0)
    ap.add_argument("--line", type=float, default=0.005)
    ap.add_argument("--out", default="kl_gate.json")
    ap.add_argument("--exclude", default="", help="regex over module names: these projections keep the f16 GEMM")
    ap.add_argument("--b-from", default="", help="kl_gate.json of an earlier run: reuse its arm B per document")
    a = ap.parse_args()

    import mlx.core as mx
    from common import load
    ref = Ref(a.golden)
    model, _, t_load = load(last_logit=False)
    log(f"gate: {a.env} {a.a} vs {a.b}, step {a.step}, model loaded in {t_load:.0f} s")
    excluded = 0
    if a.exclude:
        import re
        for name, m in model.named_modules():
            if hasattr(m, "int8_ok") and re.search(a.exclude, name):
                m.int8_ok = False
                excluded += 1
        log(f"exclude {a.exclude!r}: {excluded} projections keep the f16 GEMM")
        if not excluded:
            raise SystemExit(f"--exclude {a.exclude!r} matches no projection with int8_ok: the run would test nothing")
    b_prev = None
    if a.b_from:
        prev = json.load(open(a.b_from))
        if prev["step"] != a.step or prev["env"] != a.env or prev["b"] != a.b:
            raise SystemExit("--b-from: step, env or arm B differ")
        b_prev = {d["doc"]: d["b"] for d in prev["per_doc"]}
    n_docs = len(ref.docs) if not a.docs else min(a.docs, len(ref.docs))
    per_doc, t0 = [], time.time()
    for i in range(n_docs):
        order = (a.a, a.b) if i % 2 == 0 else (a.b, a.a)
        if b_prev is not None:
            order = (a.a,)
        res = {}
        for val in order:
            os.environ[a.env] = val
            td = time.time()
            res[val] = score_doc(model, ref, i, a.step)
            res[val]["seconds"] = round(time.time() - td, 1)
        ra, rb = res[a.a], (res[a.b] if b_prev is None else b_prev[i])
        doc = ref.docs[i]
        per_doc.append({"doc": i, "public_doc": doc.get("public_doc"), "source": doc["source"],
                        "tokens": doc["tokens"], "gen_rows": ra["gen"], "a": ra, "b": rb})
        log(f"doc {i + 1}/{n_docs} ({doc['source']}, {doc['tokens']} tok): KL A {ra['kl'] / max(1, ra['gen']):.5f} "
            f"B {rb['kl'] / max(1, rb['gen']):.5f}, top-1 A {ra['top1']} B {rb['top1']} of {ra['gen']}")
        write_json("kl_gate_partial.json", per_doc)
    ka = np.array([d["a"]["kl"] for d in per_doc])
    kb = np.array([d["b"]["kl"] for d in per_doc])
    g = sum(d["gen_rows"] for d in per_doc)
    lo, up95, up975 = bootstrap(ka, kb)
    rel = float(ka.sum() / kb.sum() - 1.0)

    def summ(arm):
        return {"kld_mean": sum(d[arm]["kl"] for d in per_doc) / g,
                "top1": sum(d[arm]["top1"] for d in per_doc) / g,
                "ppl_ratio": float(np.exp((sum(d[arm]["nll_q"] for d in per_doc)
                                           - sum(d[arm]["nll_p"] for d in per_doc)) / g))}

    out = {"env": a.env, "a": a.a, "b": a.b, "step": a.step, "profile": PROFILE, "golden": os.path.basename(
        os.path.normpath(a.golden)), "documents": len(per_doc), "gen_rows": g, "arm_a": summ("a"), "arm_b": summ("b"),
        "rel_kl_change": rel, "boot_ci95": [lo, up975], "boot_upper_one_sided_95": up95, "line": a.line,
        "pass": up975 <= a.line, "exclude": a.exclude, "excluded": excluded,
        "b_from": os.path.basename(os.path.dirname(a.b_from)) if a.b_from else "",
        "docs_a_lower_or_equal": int((ka <= kb).sum()),
        "seconds": round(time.time() - t0), "peak_memory_gb": mx.get_peak_memory() / 1e9, "per_doc": per_doc}
    write_json(a.out, out)
    log(f"gate {'PASS' if out['pass'] else 'FAIL'}: KL A {out['arm_a']['kld_mean']:.5f} B {out['arm_b']['kld_mean']:.5f}, "
        f"change {rel * 100:+.3f} %, 95 % CI [{lo * 100:+.3f}, {up975 * 100:+.3f}] %, line {a.line * 100:+.2f} %")


if __name__ == "__main__":
    main()
