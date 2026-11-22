"""Make golden-mini/: a small slice of the Qwen3.8-27B quant-challenge BF16 reference (CC BY 4.0, StillDeadcode)
plus our CUDA engine's per-document KL on the same documents, for the Mac KL check.

    python -I tools/make_golden_mini.py --golden <quant-challenge/golden/public> --cuda <kld report.json> --out golden-mini

Document choice (fixed, no randomness): the documents of the CUDA report with 256..2048 tokens, taken round-robin
over sources in report order, until the scored rows reach --rows (default 20,000). top.bin is split into parts of
at most 45 MB so that no file in the repo is near GitHub's 50 MB warning.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict

import numpy as np

PART = 45 * 1024 * 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--cuda", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=20000)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.golden, "kld.json"), encoding="utf-8"))
    rep = json.load(open(a.cuda, encoding="utf-8"))
    top, nv = meta["top"], meta["n_vocab"]
    docs = meta["docs"]
    off, o = [], 0
    for d in docs:
        off.append(o)
        o += d["tokens"]
    by_src = OrderedDict()
    for r in rep["per_doc"]:
        d = docs[r["doc"]]
        if 256 <= d["tokens"] <= 2048:
            by_src.setdefault(d["source"], []).append(r)
    chosen, rows = [], 0
    while rows < a.rows and any(by_src.values()):
        for src in list(by_src):
            if not by_src[src]:
                continue
            r = by_src[src].pop(0)
            d = docs[r["doc"]]
            chosen.append(r)
            rows += d["tokens"] - 1 - d["score_from"]
            if rows >= a.rows:
                break
    chosen.sort(key=lambda r: r["doc"])
    os.makedirs(a.out, exist_ok=True)
    tok_all = np.memmap(os.path.join(a.golden, "tokens.i32"), np.int32, "r")
    head = np.memmap(os.path.join(a.golden, "top.bin"), np.uint8, "r", shape=(meta["rows"], top * 8))
    rws = np.memmap(os.path.join(a.golden, "rows.f32"), np.float32, "r", shape=(meta["rows"], 3))
    gen = np.fromfile(os.path.join(a.golden, "gen.u8"), np.uint8)
    toks, tops, rr, gg, new_docs, cuda = [], [], [], [], [], []
    row0 = 0
    for r in chosen:
        i = r["doc"]
        d = docs[i]
        n = d["tokens"] - 1 - d["score_from"]
        toks.append(np.asarray(tok_all[off[i]:off[i] + d["tokens"]]))
        tops.append(np.asarray(head[d["row0"]:d["row0"] + n]))
        rr.append(np.asarray(rws[d["row0"]:d["row0"] + n]))
        gg.append(gen[d["row0"]:d["row0"] + n])
        new_docs.append({"source": d["source"], "tokens": d["tokens"], "score_from": d["score_from"], "row0": row0,
                         "public_doc": i})
        cuda.append({"public_doc": i, "source": d["source"], "gen_rows": r["gen_rows"], "kl_sum": r["kl_sum"],
                     "top1": r["top1"], "nll_q": r["nll_q"], "nll_p": r["nll_p"]})
        row0 += n
    np.concatenate(toks).astype(np.int32).tofile(os.path.join(a.out, "tokens.i32"))
    np.concatenate(rr).astype(np.float32).tofile(os.path.join(a.out, "rows.f32"))
    np.concatenate(gg).astype(np.uint8).tofile(os.path.join(a.out, "gen.u8"))
    blob = np.concatenate(tops).tobytes()
    parts = []
    for k in range(0, len(blob), PART):
        name = f"top.bin.part{len(parts):02d}"
        with open(os.path.join(a.out, name), "wb") as f:
            f.write(blob[k:k + PART])
        parts.append(name)
    new_meta = {"format": meta["format"], "model": meta["model"], "n_vocab": nv, "top": top, "rows": row0,
                "top_parts": parts, "docs": new_docs,
                "source": "StillDeadcode/qwen3.8-27b-quant-challenge, golden/public (CC BY 4.0); a subset of "
                          f"{len(new_docs)} documents, rows re-based; content unchanged"}
    json.dump(new_meta, open(os.path.join(a.out, "kld.json"), "w", encoding="utf-8"), indent=1)
    g = sum(c["gen_rows"] for c in cuda)
    ref = {"engine": rep["engine"], "model": os.path.basename(rep["model"]), "kv": rep["kv"], "batch": rep["batch"],
           "created": rep["created"], "note": "per-document results of the CUDA engine (RTX 4070) on the same documents; "
           "the same tensor data as Kintsugi-27B-12G.gguf",
           "gen_rows": g, "kld_mean": sum(c["kl_sum"] for c in cuda) / g,
           "top1_agreement": sum(c["top1"] for c in cuda) / g, "per_doc": cuda}
    json.dump(ref, open(os.path.join(a.out, "cuda_reference.json"), "w", encoding="utf-8"), indent=1)
    print(f"{len(new_docs)} documents, {row0} rows, {g} gen rows, CUDA mean KL {ref['kld_mean']:.6f}, "
          f"top-1 {ref['top1_agreement']:.4f}, top.bin {len(blob):,} B in {len(parts)} parts")


if __name__ == "__main__":
    main()
