"""Step 3: KL against the BF16 reference on golden-mini (24 documents, about 18k scored rows).

The arithmetic is that of the quant challenge's score.py (StillDeadcode/qwen3.8-27b-quant-challenge, CC BY 4.0) and
of our CUDA scorer (escha_kld): the reference's top 512 tokens plus one tail cell, the reference renormalized, the
candidate's tail = everything outside the reference's 512; summaries over the "gen" rows. The CUDA engine's
per-document results on the same documents are in golden-mini/cuda_reference.json; this step reports both.

    python kl_mini.py [--docs N]       (--self-test: check the scorer on synthetic data, no model; any OS)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PROFILE, ROOT, footprint, log, write_json  # noqa: E402

GOLDEN = os.path.join(ROOT, "golden-mini")
TINY = np.finfo(np.float64).tiny


class Ref:
    def __init__(self, d=GOLDEN):
        self.meta = json.load(open(os.path.join(d, "kld.json"), encoding="utf-8"))
        self.top, self.nv, rows = self.meta["top"], self.meta["n_vocab"], self.meta["rows"]
        blob = b"".join(open(os.path.join(d, p), "rb").read() for p in self.meta["top_parts"])
        self.head = np.frombuffer(blob, np.dtype([("id", "<i4"), ("lp", "<f4")])).reshape(rows, self.top)
        self.rows = np.fromfile(os.path.join(d, "rows.f32"), np.float32).reshape(rows, 3)
        self.gen = np.fromfile(os.path.join(d, "gen.u8"), np.uint8).astype(bool)
        self.tokens = np.fromfile(os.path.join(d, "tokens.i32"), np.int32)
        self.docs = self.meta["docs"]
        self.off, o = [], 0
        for doc in self.docs:
            self.off.append(o)
            o += doc["tokens"]

    def doc_tokens(self, i):
        return self.tokens[self.off[i]:self.off[i] + self.docs[i]["tokens"]]

    def n_rows(self, i):
        return self.docs[i]["tokens"] - 1 - self.docs[i]["score_from"]

    def score_rows(self, i, j0, cand):
        """(kl, top1, nll_q, nll_p, gen) for scored rows j0 .. j0+len(cand) of document i."""
        doc = self.docs[i]
        cand = np.asarray(cand, np.float64)
        n = cand.shape[0]
        r0 = doc["row0"] + j0
        lq = cand - cand.max(1, keepdims=True)
        lq -= np.log(np.exp(lq).sum(1, keepdims=True))
        head = self.head[r0:r0 + n]
        ids, lp = head["id"].astype(np.int64), head["lp"].astype(np.float64)
        p = np.exp(lp)
        lq_head = np.take_along_axis(lq, ids, 1)
        e = np.exp(lq)
        np.put_along_axis(e, ids, 0.0, 1)
        q_tail = e.sum(1)
        p_tail = self.rows[r0:r0 + n, 2].astype(np.float64)
        S = p.sum(1) + np.where(p_tail > 0, p_tail, 0)
        A = (np.where(p > 0, p * (lp - lq_head), 0)).sum(1)
        A += np.where(p_tail > 0, p_tail * (np.log(np.where(p_tail > 0, p_tail, 1)) - np.log(np.maximum(q_tail, TINY))), 0)
        kl = A / S - np.log(S)
        nxt = self.doc_tokens(i)[doc["score_from"] + 1 + j0:doc["score_from"] + 1 + j0 + n]
        top1 = lq.argmax(1) == ids[:, 0]
        nll_q = -lq[np.arange(n), nxt]
        return kl, top1, nll_q, self.rows[r0:r0 + n, 0].astype(np.float64), self.gen[r0:r0 + n]


def self_test():
    """Candidate = the reference itself (head log-probs, tail spread evenly): KL must be ~0, top-1 100 %."""
    ref = Ref()
    i = 0
    n = min(64, ref.n_rows(i))
    r0 = ref.docs[i]["row0"]
    head = ref.head[r0:r0 + n]
    cand = np.full((n, ref.nv), -np.inf)
    for k in range(n):
        ids, lp = head["id"][k].astype(np.int64), head["lp"][k].astype(np.float64)
        p_tail = max(float(ref.rows[r0 + k, 2]), 1e-30)
        rest = np.ones(ref.nv, bool)
        rest[ids] = False
        cand[k, rest] = np.log(p_tail / rest.sum())
        cand[k, ids] = lp
    kl, top1, *_ = ref.score_rows(i, 0, cand)
    ok = float(np.abs(kl).max()) < 1e-6 and bool(top1.all())
    print(json.dumps({"self_test": "pass" if ok else "FAIL", "max_abs_kl": float(np.abs(kl).max()),
                      "top1": float(top1.mean())}))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--docs", type=int, default=0, help="first N documents only (0 = all 24)")
    ap.add_argument("--step", type=int, default=256)   # all-position logits: 256 x 248,320 per step (memory)
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)

    import mlx.core as mx
    from common import load
    ref = Ref()
    cuda = {d["public_doc"]: d for d in json.load(open(os.path.join(GOLDEN, "cuda_reference.json")))["per_doc"]}
    model, _, t_load = load(last_logit=False)
    log(f"KL: model loaded in {t_load:.0f} s (all-position logits)")
    per_doc, t0 = [], time.time()
    docs = range(len(ref.docs)) if not a.docs else range(min(a.docs, len(ref.docs)))
    for i in docs:
        doc = ref.docs[i]
        toks = ref.doc_tokens(i).tolist()
        n_rows, sf = ref.n_rows(i), doc["score_from"]
        cache = model.make_cache()
        acc = {"kl": 0.0, "top1": 0, "nll_q": 0.0, "nll_p": 0.0, "gen": 0}
        td = time.time()
        for a0 in range(0, len(toks) - 1, a.step):
            b0 = min(len(toks) - 1, a0 + a.step)       # positions a0..b0-1 predict a0+1..b0
            logits = model(mx.array(toks[a0:b0])[None], cache=cache)[0]
            mx.eval(logits)
            lo, hi = max(a0, sf), b0
            if hi > lo:
                cand = np.array(logits[lo - a0:hi - a0].astype(mx.float32))
                for s in range(0, hi - lo, 64):
                    kl, t1, nq, npp, gen = ref.score_rows(i, lo - sf + s, cand[s:s + 64])
                    acc["kl"] += float(kl[gen].sum()); acc["top1"] += int(t1[gen].sum())
                    acc["nll_q"] += float(nq[gen].sum()); acc["nll_p"] += float(npp[gen].sum())
                    acc["gen"] += int(gen.sum())
            del logits
        mx.clear_cache()
        c = cuda.get(doc["public_doc"], {})
        rec = {"doc": i, "public_doc": doc["public_doc"], "source": doc["source"], "tokens": doc["tokens"],
               "gen_rows": acc["gen"], "kl_mean": acc["kl"] / max(1, acc["gen"]),
               "top1": acc["top1"] / max(1, acc["gen"]), "kl_sum": acc["kl"], "nll_q": acc["nll_q"],
               "nll_p": acc["nll_p"], "seconds": round(time.time() - td, 1),
               "cuda_kl_mean": c.get("kl_sum", 0) / max(1, c.get("gen_rows", 1)),
               "cuda_gen_rows": c.get("gen_rows")}
        per_doc.append(rec)
        log(f"KL doc {i + 1}/{len(docs)} ({doc['source']}, {doc['tokens']} tok): mac {rec['kl_mean']:.4f} "
            f"cuda {rec['cuda_kl_mean']:.4f}, {rec['seconds']} s")
        write_json("kl_partial.json", per_doc)
    g = sum(r["gen_rows"] for r in per_doc)
    cg = sum(cuda[r["public_doc"]]["gen_rows"] for r in per_doc)
    res = {"documents": len(per_doc), "gen_rows": g,
           "mac": {"kld_mean": sum(r["kl_sum"] for r in per_doc) / g,
                   "top1": sum(r["top1"] * r["gen_rows"] for r in per_doc) / g,
                   "ppl_ratio": float(np.exp((sum(r["nll_q"] for r in per_doc) - sum(r["nll_p"] for r in per_doc)) / g))},
           "cuda_same_docs": {"kld_mean": sum(cuda[r["public_doc"]]["kl_sum"] for r in per_doc) / cg,
                              "top1": sum(cuda[r["public_doc"]]["top1"] for r in per_doc) / cg,
                              "gen_rows": cg},
           "mac_lower_or_equal_docs": sum(1 for r in per_doc if r["kl_mean"] <= r["cuda_kl_mean"]),
           "seconds": round(time.time() - t0), "profile": PROFILE, "peak_memory_gb": mx.get_peak_memory() / 1e9,
           "footprint_max_gb": footprint()[1] / 1e9, "per_doc": per_doc}
    res["ratio_mac_over_cuda"] = res["mac"]["kld_mean"] / res["cuda_same_docs"]["kld_mean"]
    write_json("kl.json", res)
    log(f"KL done: mac {res['mac']['kld_mean']:.5f} vs cuda {res['cuda_same_docs']['kld_mean']:.5f} "
        f"(ratio {res['ratio_mac_over_cuda']:.3f}), top-1 {res['mac']['top1']:.4f} vs "
        f"{res['cuda_same_docs']['top1']:.4f}")


if __name__ == "__main__":
    main()
