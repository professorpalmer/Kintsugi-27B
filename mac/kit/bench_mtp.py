"""MTP tree sweep: output tok/s and accepted tokens per target forward, per tree shape. One model load.

One row-blocked forward shares the weight decode across all verify rows, so a wider draft (more verify rows) costs
less than the same number of AR steps, and it can raise the accepted length. (The one-row AR path is load-latency
bound; see ESCHA_MLX_DENSE_PF. The extra cost per verify row is an inference about FMA issue, not a measurement.)

    python kit/bench_mtp.py [--nodes 3,5,7,9,13] [--depths 3] [--topk 3] [--tokens 256]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import chat_ids, load, log  # noqa: E402

PROMPTS = [
    "Write a Python function that parses an ISO 8601 date string without any library, with tests.",
    "Explain how a hash map handles collisions. Give two methods and their costs.",
    "Write a short story of about 200 words about a lighthouse keeper.",
    "List the steps to make bread from flour, water, salt and yeast, with times.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", default="3,5,7,9,13")
    ap.add_argument("--depths", default="3")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import mlx.core as mx
    model, tok, _ = load(mtp=True)
    from escha_mlx import mtp

    calls = {"n": 0}
    orig_tf = mtp._target_tree_forward

    def counted(*args, **kw):
        calls["n"] += 1
        return orig_tf(*args, **kw)
    mtp._target_tree_forward = counted

    def gen(prompt, n):
        ids = mx.array(chat_ids(tok, prompt), dtype=mx.uint32)
        calls["n"] = 0
        out, t0, t_first = [], None, None
        for t, _, _ in mtp.mtp_generate_step(ids, model, max_tokens=n):
            if t0 is None:
                t0 = time.perf_counter()
            out.append(t)
        dt = time.perf_counter() - t0
        return out, (len(out) - 1) / dt, (len(out) - 1) / max(1, calls["n"])

    def ar(prompt, n):
        from common import decode, prefill
        ids = chat_ids(tok, prompt)
        cache = model.make_cache()
        _, last = prefill(model, cache, ids)
        out, dt = decode(model, cache, last, n)
        return out, (len(out) - 1) / dt

    gen(PROMPTS[0], 32)                                  # warm
    rows = []
    ar_tps = [ar(p, a.tokens)[1] for p in PROMPTS]
    ar_ref = [ar(p, a.tokens)[0] for p in PROMPTS]
    log(f"AR: {sum(ar_tps) / len(ar_tps):.2f} tok/s")
    rows.append({"shape": "AR", "tps": sum(ar_tps) / len(ar_tps)})
    for depth in [int(d) for d in a.depths.split(",")]:
        for nodes in [int(n) for n in a.nodes.split(",")]:
            cap = a.topk + max(0, depth - 1) * a.topk ** 2
            if nodes > cap:
                continue
            spec = mtp._TreeSpec(topk=a.topk, depth=depth, draft_nodes=nodes)
            mtp._propose_tree.__defaults__ = (spec,)
            tps, acc, same = [], [], []
            for p, ref_out in zip(PROMPTS, ar_ref):
                out, t, ac = gen(p, a.tokens)
                tps.append(t)
                acc.append(ac)
                same.append(out[:len(ref_out)] == ref_out[:len(out)])
            r = {"shape": f"topk{a.topk} depth{depth} nodes{nodes}", "tps": sum(tps) / len(tps),
                 "tokens_per_verify": sum(acc) / len(acc), "same_text_as_ar": same}
            rows.append(r)
            log(f"{r['shape']}: {r['tps']:.2f} tok/s, {r['tokens_per_verify']:.2f} tok/verify, same {same}")
    print(json.dumps(rows, indent=1))
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
