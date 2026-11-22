"""Draft-only vocabulary prefix for the MTP proposer (FR-Spec shape). The draft head scores only token ids < N (a row
slice of the Q8 head, no requantization). Verify keeps the full Q8 head, so output is the target's choice. Arms
alternate per round; reports tok/s, tokens/verify, same tokens."""
import json, os, sys, time
os.environ["ESCHA_MLX_MTP_DRAFT_VOCAB"] = "0"   # the full-head arm must really be full (stage 14 default is 98304)
sys.path.insert(0, "kit")
import mlx.core as mx
from common import chat_ids, load, log
from bench_mtp import PROMPTS
model, tok, _ = load(mtp=True)
from escha_mlx import mtp
head = mtp._unwrapped_head(model)
import numpy as np
from escha_mlx.quant import EschaQ8Head
log(f"head {type(head).__name__} weight {head.weight.shape} scales {head.scales.shape} group {head._group_size}")
orig_logits = mtp.MTPHead.logits
SL = {}
def make(n):
    if n == 0:
        return orig_logits
    if n not in SL:
        SL[n] = EschaQ8Head(np.array(head.weight[:n]), np.array(head.scales[:n]), np.array(head.biases[:n]),
                            head._group_size)
        mx.eval(SL[n].parameters())
    sub = SL[n]
    def lg(self, hidden):
        return sub(hidden)
    return lg
calls = {"n": 0}
orig_tf = mtp._target_tree_forward
def counted(*a, **k):
    calls["n"] += 1
    return orig_tf(*a, **k)
mtp._target_tree_forward = counted
def gen(prompt, n):
    ids = mx.array(chat_ids(tok, prompt), dtype=mx.uint32)
    calls["n"] = 0
    out, t0 = [], None
    for t, _, _ in mtp.mtp_generate_step(ids, model, max_tokens=n):
        if t0 is None:
            t0 = time.perf_counter()
        out.append(t)
    return out, (len(out) - 1) / (time.perf_counter() - t0), (len(out) - 1) / max(1, calls["n"])
for n in (0, 49152, 65536, 98304, 131072):
    mtp.MTPHead.logits = make(n); gen(PROMPTS[0], 32)
ARMS = [0, 49152, 65536, 98304, 131072]
R = 6
res = {b: {"tps": [], "acc": [], "out": {}} for b in ARMS}
for rnd in range(R):
    order = ARMS if rnd % 2 == 0 else ARMS[::-1]
    for b in order:
        mtp.MTPHead.logits = make(b)
        tps, acc = [], []
        for i, p in enumerate(PROMPTS):
            out, t, a = gen(p, 192)
            tps.append(t); acc.append(a)
            res[b]["out"].setdefault(i, out)
        res[b]["tps"].append(sum(tps) / len(tps)); res[b]["acc"].append(sum(acc) / len(acc))
    log(f"round {rnd}: " + ", ".join(f"v{b} {res[b]['tps'][-1]:.2f} tok/s {res[b]['acc'][-1]:.3f} tok/verify" for b in ARMS))
mtp.MTPHead.logits = orig_logits
import statistics as st
summary = {}
for b in ARMS:
    same = [res[b]["out"][i] == res[0]["out"][i] for i in range(len(PROMPTS))]
    ratios = [res[b]["tps"][r] / res[0]["tps"][r] for r in range(R)]
    summary[f"v{b}"] = {"tps_median": st.median(res[b]["tps"]), "acc_median": st.median(res[b]["acc"]),
                        "ratio_vs_full_per_round": [round(x, 4) for x in ratios], "same_tokens_as_full": same}
    log(f"v{b}: median {st.median(res[b]['tps']):.2f} tok/s, {st.median(res[b]['acc']):.3f} tok/verify, "
        f"ratio vs full {[round(x,3) for x in ratios]}, same tokens {same}")
import os; json.dump(summary, open(os.path.join(os.environ["KINTSUGI_RESULTS"], "draft_vocab.json"), "w"), indent=1)
