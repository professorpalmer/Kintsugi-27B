"""Paired MTP A/B of the draft vocabulary prefix per language. Arms: full (ESCHA_MLX_MTP_DRAFT_VOCAB=0), full_aa (the
same settings again: an A/A arm that shows the noise band), adaptive
(the shipped rule: prefix only while the last 64 context tokens are below N), and static (prefix always, forced).
Arms alternate per round. Reports tok/s ratio vs full, tokens per verify, and same tokens, for each prompt."""
import json, os, sys, time
import statistics as st
sys.path.insert(0, "kit")
import mlx.core as mx
from common import chat_ids, load, log
N = int(os.environ.get("DRAFT_N", "98304"))
PROMPTS = {
    "en_code": "Write a Python function that merges two sorted lists into one sorted list, with a short docstring.",
    "en_story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "fr": "Écris un court texte en français sur une journée de pluie à Paris.",
    "de": "Schreibe einen kurzen Text auf Deutsch über einen Spaziergang im Wald im Herbst.",
    "ru": "Напиши короткий рассказ на русском языке о зимнем вечере в деревне.",
    "zh": "请用中文写一段关于春天在公园里散步的短文。",
    "ja": "日本語で、秋の京都を旅する短い文章を書いてください。",
}
model, tok, _ = load(mtp=True)
from escha_mlx import mtp
log(f"draft N {N}, control_from {getattr(model.mtp, 'control_from', None)}")
real_vocab = mtp._DraftVocab.vocab
calls = {"n": 0}
orig_tf = mtp._target_tree_forward
def counted(*a, **k):
    calls["n"] += 1
    return orig_tf(*a, **k)
mtp._target_tree_forward = counted
def arm(name):
    os.environ["ESCHA_MLX_MTP_DRAFT_VOCAB"] = "0" if name.startswith("full") else str(N)
    mtp._DraftVocab.vocab = property(lambda self: self.n) if name == "static" else real_vocab
def gen(prompt, n):
    ids = mx.array(chat_ids(tok, prompt), dtype=mx.uint32)
    calls["n"] = 0
    out, t0 = [], None
    for t, _, _ in mtp.mtp_generate_step(ids, model, max_tokens=n):
        if t0 is None:
            t0 = time.perf_counter()
        out.append(t)
    return out, (len(out) - 1) / (time.perf_counter() - t0), (len(out) - 1) / max(1, calls["n"])
ARMS, R = ["full", "full_aa", "adaptive", "static"], int(os.environ.get("ROUNDS", "4"))
for a in ARMS:
    arm(a); gen(PROMPTS["en_code"], 32)
res = {a: {p: {"tps": [], "acc": [], "out": None} for p in PROMPTS} for a in ARMS}
for rnd in range(R):
    for a in (ARMS if rnd % 2 == 0 else ARMS[::-1]):
        arm(a)
        for p, text in PROMPTS.items():
            out, t, acc = gen(text, 128)
            r = res[a][p]; r["tps"].append(t); r["acc"].append(acc)
            if r["out"] is None:
                r["out"] = out
    log(f"round {rnd} done")
summary = {}
for p in PROMPTS:
    row = {}
    for a in ARMS:
        ratios = [res[a][p]["tps"][i] / res["full"][p]["tps"][i] for i in range(R)]
        row[a] = {"median_ratio": round(st.median(ratios), 4), "ratios": [round(x, 3) for x in ratios],
                  "tok_per_verify": round(st.median(res[a][p]["acc"]), 3), "same_tokens": res[a][p]["out"] == res["full"][p]["out"]}
    summary[p] = row
    log(f"{p:9s} " + "  ".join(f"{a} {row[a]['median_ratio']:.3f} ({row[a]['tok_per_verify']:.2f}/v, same {row[a]['same_tokens']})" for a in ARMS))
json.dump(summary, open(os.path.join(os.environ["KINTSUGI_RESULTS"], "draft_lang.json"), "w"), indent=1)
