"""Step 5: decode and prefill speed at 4k, 32k and the largest context that fits, then 24 needles at that context.

One model load. For each depth: a fresh cache, a filler prompt of that many tokens, prefill (chunks of 2048), then 128
greedy tokens. The largest depth is the needle document itself: its prefill gives the prefill speed at that depth,
then 128 decode tokens give the decode speed, then each of the 24 questions runs on top of the same prefix (the cache
rolls back after each answer). Every depth writes its result at once, so a later failure keeps the earlier numbers.

    python speed_needles.py [--max-ctx N] [--skip-needles]
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (P16, PROFILE, append_jsonl, chat_ids, decode, first_code, footprint,  # noqa: E402
                    load, log, memory_plan, needle_doc, prefill, restore, sentence, snapshot, swap_used_mb,
                    wired_for, write_json)

DECODE_N = 128


def filler_ids(tok, n: int) -> list[int]:
    text = " ".join(sentence(i) for i in range(n // 10 + 64))
    ids = tok.encode(text)
    while len(ids) < n:
        ids = ids + ids
    return ids[:n]


def one_depth(model, tok, depth: int, step: int) -> dict:
    import mlx.core as mx
    cache = model.make_cache()
    ids = filler_ids(tok, depth)
    t_pf, last = prefill(model, cache, ids, step=step)
    out, t_dec = decode(model, cache, last, DECODE_N)
    rec = {"depth": depth, "prefill_tokens": len(ids), "prefill_s": round(t_pf, 2),
           "prefill_tps": len(ids) / t_pf, "decode_tokens": len(out) - 1, "decode_s": round(t_dec, 2),
           "decode_tps": (len(out) - 1) / t_dec if t_dec > 0 else None,
           "peak_memory_gb": mx.get_peak_memory() / 1e9, "active_memory_gb": mx.get_active_memory() / 1e9,
           "cache_memory_gb": mx.get_cache_memory() / 1e9, "footprint_max_gb": footprint()[1] / 1e9,
           "prefill_step": step, "profile": PROFILE}
    del cache
    gc.collect()
    mx.clear_cache()
    return rec


def needles(model, tok, depth: int, step: int, n: int = 24) -> dict:
    import mlx.core as mx
    doc, codes = needle_doc(tok, depth, n)
    q = {v: f"What is the access code for vault {v}? Answer with the 4 digits only." for v in range(1, n + 1)}
    full = {v: chat_ids(tok, doc + q[v]) for v in range(1, n + 1)}
    lcp = full[1]
    for v in range(2, n + 1):
        k = 0
        while k < min(len(lcp), len(full[v])) and lcp[k] == full[v][k]:
            k += 1
        lcp = lcp[:k]
    prefix = lcp[:-1]                              # keep at least one token for every question
    stop = set(getattr(tok, "eos_token_ids", None) or [])
    if getattr(tok, "eos_token_id", None) is not None:
        stop.add(tok.eos_token_id)
    cache = model.make_cache()
    log(f"needles: prefill of {len(prefix)} tokens (this is the long step)")
    t_pf, last = prefill(model, cache, prefix, step=step)
    snap = snapshot(cache)
    speed = {"depth": len(prefix), "prefill_tokens": len(prefix), "prefill_s": round(t_pf, 2),
             "prefill_tps": len(prefix) / t_pf}
    out, t_dec = decode(model, cache, last, DECODE_N)
    restore(cache, snap)
    speed.update({"decode_tokens": len(out) - 1, "decode_s": round(t_dec, 2),
                  "decode_tps": (len(out) - 1) / t_dec if t_dec > 0 else None,
                  "peak_memory_gb": mx.get_peak_memory() / 1e9, "cache_memory_gb": mx.get_cache_memory() / 1e9,
                  "footprint_max_gb": footprint()[1] / 1e9, "prefill_step": step, "profile": PROFILE,
                  "source": "needle document"})
    append_jsonl("speed.jsonl", speed)
    log(f"depth {len(prefix)}: prefill {speed['prefill_tps']:.1f} tok/s, decode {speed['decode_tps']:.2f} tok/s")
    ok = 0
    for v in range(1, n + 1):
        t0 = time.time()
        suffix = full[v][len(prefix):]
        _, last = prefill(model, cache, suffix, step=step)
        toks, _ = decode(model, cache, last, 12, stop_ids=stop)
        restore(cache, snap)
        ans = tok.decode([t for t in toks if t not in stop])
        hit = first_code(ans) == codes[v]
        ok += hit
        rec = {"depth": len(prefix), "vault": v, "position_frac": round((v - 0.5) / n, 3), "want": codes[v],
               "answer": ans.strip()[:40], "ok": hit, "wall_s": round(time.time() - t0, 1)}
        append_jsonl("needles.jsonl", rec)
        log(f"needle {v:2d}/{n}: {'ok ' if hit else 'BAD'} want {codes[v]} got {ans.strip()[:20]!r}")
    res = {"depth": len(prefix), "correct": ok, "of": n, "peak_memory_gb": mx.get_peak_memory() / 1e9}
    append_jsonl("needles.jsonl", res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-ctx", type=int, default=0, help="override the planned largest context")
    ap.add_argument("--skip-needles", action="store_true")
    a = ap.parse_args()
    import mlx.core as mx
    swap0 = swap_used_mb()
    # plan with a first load (no wiring), then reload wired if the largest run is near the cap
    model, tok, t_load = load()
    plan = memory_plan()
    plan["load_peak_gb"] = mx.get_peak_memory() / 1e9
    peaks = [plan["load_peak_gb"]]
    plan["load_footprint_max_gb"] = footprint()[1] / 1e9
    if P16 and not plan["fits"]:
        write_json("memory_plan.json", plan)
        log(plan["message"])
        sys.exit(1)
    max_ctx = a.max_ctx or int(os.environ.get("KINTSUGI_MAX_CTX", "0")) or plan["max_ctx_under_cap"]
    max_ctx = min(max_ctx, plan["max_ctx_under_cap"])
    plan["planned_max_ctx"] = max_ctx
    wired = wired_for(max_ctx, plan)
    plan["escha_mlx_wired_gb"] = wired
    write_json("memory_plan.json", plan)
    log(f"memory: cap {plan['metal_cap_bytes'] / 1e9:.1f} GB, model {plan['model_active_bytes'] / 1e9:.1f} GB, "
        f"largest context {max_ctx}, prefill step {plan['prefill_step']}")
    if "optional_raise" in plan:
        r = plan["optional_raise"]
        log(f"OPTIONAL (not run by this kit): `{r['command']}` would allow about {r['would_allow_ctx']} tokens. "
            f"It needs your password and resets at reboot. Run it yourself before ./run.sh if you want it.")
    if wired:
        del model
        gc.collect()
        mx.clear_cache()
        mx.reset_peak_memory()
        model, tok, t_load = load(wired_gb=wired)
        peaks.append(mx.get_peak_memory() / 1e9)
        log(f"reloaded with ESCHA_MLX_WIRED_GB={wired:.1f} (process only; not a system setting)")
    for depth in (4096, 32768):
        if depth + 2048 > max_ctx:
            log(f"depth {depth}: skipped (does not fit)")
            continue
        mx.reset_peak_memory()
        rec = one_depth(model, tok, depth, plan["prefill_step"])
        append_jsonl("speed.jsonl", rec)
        peaks.append(rec["peak_memory_gb"])
        log(f"depth {depth}: prefill {rec['prefill_tps']:.1f} tok/s, decode {rec['decode_tps']:.2f} tok/s, "
            f"peak {rec['peak_memory_gb']:.1f} GB")
    if not a.skip_needles and max_ctx >= 8192:
        mx.reset_peak_memory()
        res = needles(model, tok, max_ctx - 1024, plan["prefill_step"])
        peaks.append(res["peak_memory_gb"])
        log(f"needles: {res['correct']}/{res['of']} at {res['depth']} tokens")
    run = {"profile": PROFILE, "metal_cap_bytes": plan["metal_cap_bytes"], "planned_max_ctx": max_ctx,
           "peak_memory_gb_max": max(peaks), "peaks_gb": peaks,
           "footprint_max_gb": footprint()[1] / 1e9, "swap_used_mb_start": swap0, "swap_used_mb_end": swap_used_mb()}
    write_json("memory_run.json", run)
    log(f"memory: footprint max {run['footprint_max_gb']:.2f} GB, swap {swap0} MB -> {run['swap_used_mb_end']} MB")


if __name__ == "__main__":
    main()
