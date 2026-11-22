"""Check escha_mlx.mtp_session (the cache reuse of ./run.sh chat and serve): a second turn that reuses the first
turn's caches must give the same tokens as the same prompt read from empty caches (greedy).

Case A: the first prompt is a whole number of prefill steps (the engine's step and reserve), so both runs cut the
prompt at the same places: the tokens must be equal, all of them. Case B: any length: the share of equal tokens is reported (the prefill GEMM is
not bit-identical across cuts, so a near-tie can differ). Also: the second turn reads only its new tokens.

Prefix snapshots (8 at the chunk ends): case C: the second prompt leaves the first one after 2 chunks (an edited
turn): it must reuse 2 chunks from a live snapshot. Case D: an unrelated short prompt comes between, then a prompt that
shares the first 3 chunks: it must reuse 3 chunks from the pinned prefix. C and D cut at chunk ends, so the tokens must
be equal to a fresh read, all of them.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PREFILL_STEP, load, log, memory_plan, needle_doc, write_json  # noqa: E402


def main():
    from escha_mlx.kv_cache import reserve
    from escha_mlx.mtp_session import MTPSession
    model, tok, _ = load(mtp=True)
    # The engine's path (kit/userapi.py): the plan's prefill step (128 in the 16 GB profile) and a reserve of n + 1024.
    step = int(memory_plan().get("prefill_step") or PREFILL_STEP)
    s = MTPSession(model, prefill_step_size=step, reserve=lambda cache, n: reserve(cache, n + 1024), snapshots=8,
                   snapshot_max=8 * step)
    doc, _ = needle_doc(tok, max(3000, 4 * step))
    base = list(tok.encode(doc, add_special_tokens=False))
    res = {}
    for case, n1 in (("A_aligned", 2 * step), ("B_any", 1000)):
        p1 = base[:n1]
        p2 = p1 + list(tok.encode(" Question: what is the access code for vault 3?", add_special_tokens=False))
        s.reset()
        first = list(s.generate(p1, 40))                  # turn 1, then turn 2 reuses its prompt
        t0 = time.time()
        reuse = list(s.generate(p2, 64))
        reused, t_reuse = s.reused, time.time() - t0
        s.reset()
        t0 = time.time()
        fresh = list(s.generate(p2, 64))
        t_fresh = time.time() - t0
        same = sum(a == b for a, b in zip(reuse, fresh))
        prefix = next((i for i, (a, b) in enumerate(zip(reuse, fresh)) if a != b), len(fresh))
        res[case] = {"turn1_tokens": len(first), "prompt2": len(p2), "reused": reused, "equal_tokens": same,
                     "of": len(fresh), "equal_prefix": prefix, "seconds_reuse": round(t_reuse, 2),
                     "seconds_fresh": round(t_fresh, 2)}
        log(f"{case}: reused {reused} of {len(p2)}, equal {same}/{len(fresh)} (prefix {prefix}), "
            f"{t_reuse:.2f} s with reuse vs {t_fresh:.2f} s fresh")
    question = list(tok.encode(" Question: what is the access code for vault 3?", add_special_tokens=False))
    side = list(tok.encode("Write a short title for a chat about rain.", add_special_tokens=False))
    p1 = base[:3 * step + 100]
    for case, keep, between, source in (("C_live_snapshot", 2 * step, False, "live"),
                                         ("D_pinned_prefix", 3 * step, True, "pinned")):
        p2 = base[:keep] + question
        s.reset()
        s.drop_extras()
        list(s.generate(p1, 40))
        if between:
            list(s.generate(side, 16))
        t0 = time.time()
        reuse = list(s.generate(p2, 64))
        reused, src, t_reuse = s.reused, s.reused_from, time.time() - t0
        s.reset()
        s.drop_extras()
        t0 = time.time()
        fresh = list(s.generate(p2, 64))
        t_fresh = time.time() - t0
        same = sum(a == b for a, b in zip(reuse, fresh))
        res[case] = {"prompt2": len(p2), "reused": reused, "reused_from": src, "want_reused": keep,
                     "want_from": source, "equal_tokens": same, "of": len(fresh),
                     "seconds_reuse": round(t_reuse, 2), "seconds_fresh": round(t_fresh, 2)}
        log(f"{case}: reused {reused} of {len(p2)} from {src}, equal {same}/{len(fresh)}, "
            f"{t_reuse:.2f} s with reuse vs {t_fresh:.2f} s fresh")
    res["pass"] = (res["A_aligned"]["equal_tokens"] == res["A_aligned"]["of"]
                   and res["A_aligned"]["reused"] == 2 * step and res["B_any"]["reused"] == 1000
                   and all(res[c]["reused"] == res[c]["want_reused"] and res[c]["reused_from"] == res[c]["want_from"]
                           and res[c]["equal_tokens"] == res[c]["of"] for c in ("C_live_snapshot", "D_pinned_prefix")))
    log(f"session check {'PASS' if res['pass'] else 'FAIL'}")
    write_json("check_session.json", res)
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
