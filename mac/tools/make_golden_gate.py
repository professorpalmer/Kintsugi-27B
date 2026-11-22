"""Build a ~100-document KL gate set from the public quant-challenge golden set (StillDeadcode, CC BY 4.0) by HTTP
byte ranges of top.bin, in the golden-mini format that kit/kl_mini.py reads.

Choice (fixed, no randomness): all golden-mini documents, then documents with 256..2048 tokens round-robin over
sources in file order, until N documents.
    python -I make_golden_gate.py --src <dir with src_kld.json, src_tokens.i32, src_rows.f32, src_gen.u8> \
        --mini <golden-mini> --out <dir> [--n 100]
The --src files are golden/public/{kld.json, tokens.i32, rows.f32, gen.u8} of the same dataset, with "src_" added to
each name. The set has 100 documents, 87,352 rows and about 358 MB of top data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from collections import OrderedDict

import numpy as np

# Pinned dataset revision: the --src files must come from the same revision (lastModified 2026-10-27).
REVISION = "2dcf0061cb683152742b904e2965ce7cd92b2da6"
URL = ("https://huggingface.co/datasets/StillDeadcode/qwen3.8-27b-quant-challenge/resolve/"
       f"{REVISION}/golden/public/top.bin")
PART = 45 * 1024 * 1024


def fetch(lo: int, hi: int) -> bytes:
    req = urllib.request.Request(URL, headers={"Range": f"bytes={lo}-{hi - 1}", "User-Agent": "kl-gate-builder"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    if len(data) != hi - lo:
        raise RuntimeError(f"range {lo}-{hi}: got {len(data)} bytes")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--mini", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.src, "src_kld.json"), encoding="utf-8"))
    docs, top = meta["docs"], meta["top"]
    rowb = top * 8
    off, o = [], 0
    for d in docs:
        off.append(o)
        o += d["tokens"]
    mini = [d["public_doc"] for d in json.load(open(os.path.join(a.mini, "kld.json")))["docs"]]
    chosen = list(mini)
    by_src = OrderedDict()
    for i, d in enumerate(docs):
        if i not in chosen and 256 <= d["tokens"] <= 2048:
            by_src.setdefault(d["source"], []).append(i)
    while len(chosen) < a.n and any(by_src.values()):
        for src in list(by_src):
            if by_src[src] and len(chosen) < a.n:
                chosen.append(by_src[src].pop(0))
    chosen.sort()
    tok_all = np.fromfile(os.path.join(a.src, "src_tokens.i32"), np.int32)
    rws = np.fromfile(os.path.join(a.src, "src_rows.f32"), np.float32).reshape(meta["rows"], 3)
    gen = np.fromfile(os.path.join(a.src, "src_gen.u8"), np.uint8)
    os.makedirs(a.out, exist_ok=True)
    toks, rr, gg, new_docs, blobs = [], [], [], [], []
    row0 = 0
    for k, i in enumerate(chosen):
        d = docs[i]
        n = d["tokens"] - 1 - d["score_from"]
        toks.append(tok_all[off[i]:off[i] + d["tokens"]])
        rr.append(rws[d["row0"]:d["row0"] + n])
        gg.append(gen[d["row0"]:d["row0"] + n])
        blobs.append(fetch(d["row0"] * rowb, (d["row0"] + n) * rowb))
        new_docs.append({"source": d["source"], "tokens": d["tokens"], "score_from": d["score_from"], "row0": row0,
                         "public_doc": i})
        row0 += n
        print(f"{k + 1}/{len(chosen)} doc {i} {d['source']} rows {n}", flush=True)
    np.concatenate(toks).astype(np.int32).tofile(os.path.join(a.out, "tokens.i32"))
    np.concatenate(rr).astype(np.float32).tofile(os.path.join(a.out, "rows.f32"))
    np.concatenate(gg).astype(np.uint8).tofile(os.path.join(a.out, "gen.u8"))
    blob = b"".join(blobs)
    parts = []
    for p in range(0, len(blob), PART):
        name = f"top.bin.part{len(parts):02d}"
        open(os.path.join(a.out, name), "wb").write(blob[p:p + PART])
        parts.append(name)
    out_meta = {"format": meta["format"], "model": meta["model"], "n_vocab": meta["n_vocab"], "top": top,
                "rows": row0, "top_parts": parts, "docs": new_docs, "revision": REVISION,
                "sha256": {f: hashlib.sha256(open(os.path.join(a.out, f), "rb").read()).hexdigest()
                           for f in parts + ["tokens.i32", "rows.f32", "gen.u8"]},
                "source": "StillDeadcode/qwen3.8-27b-quant-challenge, golden/public (CC BY 4.0); a subset of "
                          f"{len(chosen)} documents, rows re-based; content unchanged"}
    json.dump(out_meta, open(os.path.join(a.out, "kld.json"), "w"), indent=1)
    print(f"done: {len(chosen)} docs, {row0} rows, {len(blob) / 1e6:.0f} MB top")


if __name__ == "__main__":
    main()
