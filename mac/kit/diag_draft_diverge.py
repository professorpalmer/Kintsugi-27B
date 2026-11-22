"""Find where MTP with a draft vocabulary prefix diverges from MTP with the full draft head, and compare both to plain
AR decode. At the first divergent position, report the target's top-2 logit gap from a plain AR forward."""
import json, os, sys
os.environ["ESCHA_MLX_MTP_DRAFT_VOCAB"] = "0"   # the full-head arm must really be full (stage 14 default is 98304)
sys.path.insert(0, "kit")
import numpy as np
import mlx.core as mx
from common import chat_ids, load, log
from bench_mtp import PROMPTS
model, tok, _ = load(mtp=True)
from escha_mlx import mtp
from escha_mlx.quant import EschaQ8Head
head = mtp._unwrapped_head(model)
orig = mtp.MTPHead.logits
def prefix(n):
    sub = EschaQ8Head(np.array(head.weight[:n]), np.array(head.scales[:n]), np.array(head.biases[:n]), head._group_size)
    return lambda self, h: sub(h)
def run_mtp(ids, n):
    return [t for t, _, _ in mtp.mtp_generate_step(ids, model, max_tokens=n)]
def run_ar(ids, n):
    from mlx_lm.generate import generate_step
    return [int(t) for t, _ in zip((t for t, _ in generate_step(ids, model, max_tokens=n)), range(n))]
def gap_at(ids, toks, pos):
    full = mx.concatenate([ids, mx.array(toks[:pos], dtype=mx.uint32)])
    cache = model.make_cache()
    logits = model(full[None], cache=cache)[0, -1].astype(mx.float32)
    top = mx.argsort(-logits)[:3]
    v = logits[top]
    mx.eval(top, v)
    return [int(x) for x in top.tolist()], [float(x) for x in v.tolist()]
out = {}
for n_sub, pi in ((32768, 3), (16384, 2)):
    ids = mx.array(chat_ids(tok, PROMPTS[pi]), dtype=mx.uint32)
    mtp.MTPHead.logits = orig
    a = run_mtp(ids, 192)
    mtp.MTPHead.logits = prefix(n_sub)
    b = run_mtp(ids, 192)
    mtp.MTPHead.logits = orig
    ar = run_ar(ids, 192)
    d = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
    rec = {"prompt": pi, "vocab": n_sub, "first_diverge": d,
           "full_eq_ar_len": next((i for i in range(min(len(a), len(ar))) if a[i] != ar[i]), len(a)),
           "sub_eq_ar_len": next((i for i in range(min(len(b), len(ar))) if b[i] != ar[i]), len(b))}
    if d is not None:
        rec["full_tok"], rec["sub_tok"], rec["ar_tok"] = a[d], b[d], ar[d] if d < len(ar) else None
        rec["ar_top3_ids"], rec["ar_top3_logits"] = gap_at(ids, a, d)
    log(json.dumps(rec)); out[f"{n_sub}_{pi}"] = rec
json.dump(out, open(os.path.join(os.environ["KINTSUGI_RESULTS"], "diverge.json"), "w"), indent=1)
