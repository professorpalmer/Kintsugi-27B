"""Shared helpers for the Mac steps: paths, JSON results, model load, memory plan, needle document, timing loops."""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.environ.get("KINTSUGI_MLX_DIR", os.path.join(ROOT, "models", "Kintsugi-27B-MLX"))
RESULTS = os.environ.get("KINTSUGI_RESULTS", os.path.join(ROOT, "results"))
KV_BYTES_PER_TOKEN = 16 * 4 * 256 * 2 * 2      # 16 attention layers x 4 KV heads x 256 x (K, V) x f16 = 64 KiB
# MLX peak minus active memory during one prefill step, at depth (kit/probe_transient.py), max of both decode-once
# paths, rounded up. At 32k: f16 (the default) 1.88 / 2.37 / 3.42 / 4.68 GB, int8 1.34 / 1.92 / 3.09 / 4.56 GB;
# int8 at 131k with step 2048: 4.5 GB. memory_plan() keeps this free.
PREFILL_TRANSIENT_GB = {256: 2.0, 512: 2.5, 1024: 3.6, 2048: 4.9}
GB = 1e9


RAM_16GB = 16 * 2**30      # 17,179,869,184 bytes


def _ram_bytes() -> int:
    try:
        return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
                                  timeout=10).stdout.strip() or 0)
    except Exception:
        return 0


RAM_BYTES = _ram_bytes()


def _profile() -> tuple[str, bool]:
    """(profile, selected by the kit itself). KINTSUGI_PROFILE=16gb|full sets it. Else '16gb' on a Mac with 16 GiB
    of RAM or less (experimental, opt-in above that: an 18 GB Mac keeps 'full'), and 'full' on all others."""
    want = os.environ.get("KINTSUGI_PROFILE", "").strip().lower()
    if want in ("16gb", "full"):
        return want, False
    if want:
        raise SystemExit(f"KINTSUGI_PROFILE={want!r}: use 16gb or full")
    return ("16gb", True) if 0 < RAM_BYTES <= RAM_16GB else ("full", False)


PROFILE, PROFILE_AUTO = _profile()
P16 = PROFILE == "16gb"
if P16 and os.environ.get("MallocLargeCache") != "0":
    # The load frees about 1.8 GB of numpy buffers (coded weights, head repack). libmalloc's large cache keeps
    # them dirty in this process, so they count in its footprint and a 16 GB Mac swaps them. libmalloc reads
    # MallocLargeCache only at process start: start this script again with it set (probed on macOS 26).
    os.environ["MallocLargeCache"] = "0"
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, [sys.executable] + sys.orig_argv[1:])
if PROFILE_AUTO:
    print("16 GB profile selected (experimental; not yet tested on a real 16 GB Mac)", file=sys.stderr, flush=True)
# 16 GB profile: the embedding stays on the CPU side (ESCHA_MLX_EMBED_CPU), the context is 16k if it fits under
# the GPU working set, else 8k, and the MLX buffer cache is kept small (cached buffers count in the working set).
MAX_CTX = 16384 if P16 else 262144
CTX_STEPS_16 = (16384, 8192)
# One prefill step's transient GPU buffers grow with the step (probed on M5 Max, CPU embedding, depth 0 / 8k / 15k):
# 128 tokens 0.84 / 0.86 / 0.83 GB, 256 tokens 1.15 / 1.00 / 1.13 GB, 512 tokens 1.54 / 1.62 / 1.37 GB.
# Under a 16 GB Mac's working set (10.67 GiB) only 128 leaves a margin at 8k, so the 16 GB profile uses 128.
PREFILL_STEP = 128 if P16 else 2048
PREFILL_TRANSIENT_16 = 0.9 * GB
# MLX buffer cache limit (cached buffers count in the footprint and the GPU working set). KINTSUGI_CACHE_LIMIT_GB
# overrides it. Below 32 GiB of RAM the full profile keeps 0.5 GB: 1.5 GB more for K/V (about 24k tokens on a
# 24 GB Mac) for 1-6 % prefill speed (probed on the M5 Max, steps 256 / 512, depth 4k / 32k, A/B/A).
CACHE_LIMIT = float(os.environ.get("KINTSUGI_CACHE_LIMIT_GB")
                    or (0.25 if P16 else 0.5 if 0 < RAM_BYTES < 32 * 2**30 else 2)) * GB
# 16 GB profile, escha-mlx settings (probed on the M5 with the M1/M2 paths, depth 7k, step 128):
# - ESCHA_MLX_DENSE_RECON=0: the decode-once prefill GEMM adds 0.62 GB of transient (1.48 vs 0.86 GB), over the cap.
# - ESCHA_MLX_KV_GROW_MAX=4096: one K/V growth step without a reservation stays at 256 MB.
ESCHA_ENV_16 = {"ESCHA_MLX_EMBED_CPU": "1", "ESCHA_MLX_DENSE_RECON": "0", "ESCHA_MLX_KV_GROW_MAX": "4096"}
# K/V growth rounding, two GDN states (a needle run keeps one rollback copy; 75.5 MB each) and small buffers.
# Probed: active memory after an 8k prefill is 0.82 GB above the loaded model, the K/V formula gives 0.54 GB.
STATE_OVERHEAD_16 = 0.3 * GB


def out_path(name: str) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    return os.path.join(RESULTS, name)


def write_json(name: str, obj) -> None:
    with open(out_path(name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def append_jsonl(name: str, obj) -> None:
    with open(out_path(name), "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(out_path("progress.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sysctl(name: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def footprint() -> tuple[float, float]:
    """(now, lifetime max) physical footprint of this process in bytes: CPU heap + GPU buffers + touched maps that
    are dirty. This is what the 16 GB Mac must hold besides macOS. proc_pid_rusage, rusage_info_v4."""
    import ctypes

    class _RU(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"f{i}", ctypes.c_uint64) for i in range(40)]
    try:
        r = _RU()
        if ctypes.CDLL("/usr/lib/libSystem.B.dylib").proc_pid_rusage(os.getpid(), 4, ctypes.byref(r)) != 0:
            return 0.0, 0.0
        return float(r.f7), float(r.f28)          # ri_phys_footprint, ri_lifetime_max_phys_footprint
    except Exception:
        return 0.0, 0.0


def swap_used_mb() -> float | None:
    m = re.search(r"used = ([\d.]+)M", sysctl("vm.swapusage"))
    return float(m.group(1)) if m else None


def gpu_cap() -> float:
    """The GPU working-set cap the kit plans for: Metal's recommended size, or KINTSUGI_CAP_GB if smaller (to
    simulate a smaller Mac on a larger one)."""
    import mlx.core as mx
    cap = float(mx.device_info().get("max_recommended_working_set_size", 0))
    sim = os.environ.get("KINTSUGI_CAP_GB", "").strip()
    if sim:
        cap = min(cap, float(sim) * GB) if cap else float(sim) * GB
    return cap


def apply_limits() -> dict:
    """Limit the MLX buffer cache. MLX keeps freed buffers for reuse, and its default limit is above the free RAM:
    at 131k the cache held 18.6 GB and macOS swapped. 2 GB keeps the footprint flat at no speed cost (372 vs 357
    tok/s prefill at 131k, speedlab/longmem_blocked_*.log); the 16 GB profile uses 0.25 GB. In the 16 GB profile
    (or with a simulated cap), also tell MLX the cap. Process only."""
    import mlx.core as mx
    mx.set_cache_limit(int(CACHE_LIMIT))
    if not (P16 or os.environ.get("KINTSUGI_CAP_GB")):
        return {"cache_limit_bytes": CACHE_LIMIT}
    cap = gpu_cap()
    mx.set_memory_limit(int(cap))
    return {"memory_limit_bytes": cap, "cache_limit_bytes": CACHE_LIMIT}


def load(mtp: bool = False, last_logit: bool = True, wired_gb: float | None = None):
    """Load the converted checkpoint with the patched escha-mlx. Env switches must be set before the import."""
    os.environ["ESCHA_MLX_LAST_LOGIT"] = "1" if last_logit else "0"
    if P16:
        os.environ.update(ESCHA_ENV_16)
    if wired_gb:
        os.environ["ESCHA_MLX_WIRED_GB"] = f"{wired_gb:.1f}"
    else:
        os.environ.pop("ESCHA_MLX_WIRED_GB", None)
    import mlx.core as mx
    from escha_mlx.loader import load as escha_load
    apply_limits()
    t0 = time.time()
    model, tok = escha_load(MODEL_DIR, load_mtp=mtp)
    mx.eval(model.parameters())
    return model, tok, time.time() - t0


def memory_plan_16(active: float, cap: float, steps=CTX_STEPS_16) -> dict:
    """16 GB profile: the largest of CTX_STEPS_16 whose peak fits under the cap. The peak counts the model, the
    K/V cache, the state overhead, one prefill step's transient buffers and the MLX buffer cache limit."""
    margin = STATE_OVERHEAD_16 + PREFILL_TRANSIENT_16 + CACHE_LIMIT
    rows = [{"ctx": c, "peak_bytes": active + margin + c * KV_BYTES_PER_TOKEN} for c in steps]
    fit = [r["ctx"] for r in rows if r["peak_bytes"] <= cap]
    plan = {"profile": PROFILE, "metal_cap_bytes": cap, "model_active_bytes": active, "margin_bytes": margin,
            "kv_bytes_per_token": KV_BYTES_PER_TOKEN, "candidates": rows,
            "max_ctx_under_cap": max(fit) if fit else 0, "fits": bool(fit)}
    if not fit:
        short = rows[-1]["peak_bytes"] - cap
        plan["message"] = (f"The model does not fit on this Mac: it needs {rows[-1]['peak_bytes'] / GB:.2f} GB of GPU "
                           f"memory for {rows[-1]['ctx']} tokens, the GPU working set is {cap / GB:.2f} GB "
                           f"({short / GB:.2f} GB short). Close other apps and run again. A Mac with 16 GB or more "
                           f"of RAM is necessary.")
    return plan


def memory_plan(ctx_steps_16=CTX_STEPS_16) -> dict:
    """What the current Metal working-set cap allows. Read only: changes nothing on the system. ctx_steps_16: the
    contexts the 16 GB profile may take (the user mode adds a smaller one, because the MTP head needs 0.32 GB)."""
    import mlx.core as mx
    info = mx.device_info()
    cap = gpu_cap()
    ram = float(info.get("memory_size", 0)) or float(sysctl("hw.memsize") or 0)
    active = float(mx.get_active_memory())
    if P16:
        plan = memory_plan_16(active, cap, ctx_steps_16)
        plan.update({"ram_bytes": ram, "wired_limit_mb_now": sysctl("iogpu.wired_limit_mb"),
                     "prefill_step": PREFILL_STEP, "cache_limit_bytes": CACHE_LIMIT,
                     "simulated_cap_gb": os.environ.get("KINTSUGI_CAP_GB") or None})
        return plan
    cache_limit = CACHE_LIMIT

    def ctx_for(step):                 # margin: one prefill step's transient + the MLX buffer cache + 0.5 GB slack
        margin = PREFILL_TRANSIENT_GB[step] * GB + cache_limit + 0.5 * GB
        ctx = int(max(0.0, cap - active - margin) // KV_BYTES_PER_TOKEN) // 1024 * 1024
        return min(ctx, MAX_CTX - 1024), margin

    # The largest step is up to 1.6x faster (kit/probe_transient.py) but needs more margin. Take the largest step
    # that keeps at least 7/8 of the context the smallest step allows; a 16 GB Mac then keeps its context.
    smallest = min(PREFILL_TRANSIENT_GB)
    base = ctx_for(smallest)[0]
    step = max(s for s in PREFILL_TRANSIENT_GB if s == smallest or (base and ctx_for(s)[0] * 8 >= base * 7))
    ctx, margin = ctx_for(step)
    plan = {"ram_bytes": ram, "metal_cap_bytes": cap, "model_active_bytes": active, "margin_bytes": margin,
            "prefill_step": step, "cache_limit_bytes": cache_limit,
            "kv_bytes_per_token": KV_BYTES_PER_TOKEN, "max_ctx_under_cap": ctx,
            "wired_limit_mb_now": sysctl("iogpu.wired_limit_mb")}
    if ctx < MAX_CTX - 1024 and ram:
        # what a raised cap would give: leave 6 GB of the RAM to macOS and other apps
        want = min(ram - 6 * GB, active + margin + (MAX_CTX - 1024) * KV_BYTES_PER_TOKEN)
        if want > cap + GB:
            plan["optional_raise"] = {
                "command": f"sudo sysctl iogpu.wired_limit_mb={int(want / 1e6)}",
                "would_allow_ctx": int((want - active - margin) // KV_BYTES_PER_TOKEN) // 1024 * 1024,
                "note": "optional; resets at reboot; the kit never runs it"}
    return plan


def wired_for(ctx: int, plan: dict) -> float | None:
    """ESCHA_MLX_WIRED_GB for a run of ctx tokens: only when the peak is near the cap (escha-mlx's advice)."""
    peak = plan["model_active_bytes"] + ctx * KV_BYTES_PER_TOKEN + plan["margin_bytes"]
    cap = plan["metal_cap_bytes"]
    if peak < cap - 2 * GB:
        return None
    return min(cap - 0.5 * GB, peak) / GB


# ---- the needle document (the same construction as our CUDA needle harness) -------------

SUBJ = ["the archive", "a survey", "the committee", "one report", "the ledger", "a memo", "the register", "field notes"]
VERB = ["records", "lists", "mentions", "summarizes", "notes", "confirms", "tracks", "describes"]
OBJ = ["shipments", "rainfall totals", "meeting dates", "inventory counts", "repair logs", "visitor numbers",
       "budget lines", "soil samples", "route changes", "maintenance windows"]


def sentence(i: int) -> str:
    return f"{SUBJ[i % 8].capitalize()} {VERB[(i // 8) % 8]} {OBJ[(i // 64) % 10]} for entry {i}."


def needle_doc(tok, depth: int, n: int = 24):
    rng = random.Random(7)
    codes = {v: f"{rng.randrange(1000, 10000)}" for v in range(1, n + 1)}
    n_sent = max(1, depth // 12)
    probe = " ".join(sentence(i) for i in range(n_sent))
    n_sent = max(1, int(n_sent * depth / max(1, len(tok.encode(probe)))))
    slots = {int(n_sent * (k + 0.5) / n): v for k, v in enumerate(range(1, n + 1))}
    parts = []
    for i in range(n_sent):
        if i in slots:
            v = slots[i]
            parts.append(f"The access code for vault {v} is {codes[v]}.")
        parts.append(sentence(i))
    return "Facility notes:\n" + " ".join(parts) + "\n\n", codes


def chat_ids(tok, text: str) -> list[int]:
    # render to text first: the return type of tokenize=True differs between transformers versions
    s = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False,
                                enable_thinking=False)
    return list(tok.encode(s, add_special_tokens=False))


def first_code(text: str) -> str | None:
    m = re.findall(r"\d{4}", text)
    return m[0] if m else None


# ---- timing loops --------------------------------------------------------------------------------------------------

def prefill(model, cache, ids, step: int = PREFILL_STEP):
    """Feed ids through the model in chunks; returns (seconds, last logits row)."""
    import mlx.core as mx
    t0 = time.perf_counter()
    logits = None
    try:                                         # size the K/V cache once for the whole prompt (+ decode room)
        from escha_mlx.kv_cache import reserve
        base = max((getattr(c, "offset", 0) or 0) for c in cache)
        reserve(cache, base + len(ids) + 1024)
    except ImportError:
        pass
    for a in range(0, len(ids), step):
        logits = model(mx.array(ids[a:a + step])[None], cache=cache)
        mx.eval(logits, [c.state for c in cache])
    return time.perf_counter() - t0, logits[:, -1, :]


def decode(model, cache, last_logits, n: int, stop_ids=()):
    """Greedy decode, pipelined like mlx-lm. Returns (tokens, seconds for the tokens after the first)."""
    import mlx.core as mx

    def step(y):
        return mx.argmax(model(y.reshape(1, 1), cache=cache)[:, -1, :], axis=-1)
    y = mx.argmax(last_logits, axis=-1)
    mx.eval(y)
    out, t0, nxt = [], time.perf_counter(), None
    while True:
        if len(out) + 1 < n:
            nxt = step(y)
            mx.async_eval(nxt)
        t = int(y.item())
        out.append(t)
        if len(out) == 1:
            t0 = time.perf_counter()        # the clock starts when the first token is known
        if t in stop_ids or len(out) >= n:
            break
        y = nxt
    dt = time.perf_counter() - t0
    if nxt is not None:
        mx.eval(nxt)                        # do not leave a pending step behind (the caller may roll back)
    return out, dt


def snapshot(cache):
    """Cheap rollback point: KV caches keep only their offset (trim later), GDN caches keep their array refs."""
    snap = []
    for c in cache:
        if hasattr(c, "is_trimmable") and c.is_trimmable() and hasattr(c, "offset"):
            snap.append(("kv", c.offset))
        else:
            snap.append(("arrays", list(c.state)))
    return snap


def restore(cache, snap) -> None:
    for c, (kind, v) in zip(cache, snap):
        if kind == "kv":
            c.trim(c.offset - v)
        else:
            c.state = list(v)


def env_versions() -> dict:
    import importlib.metadata as md
    out = {"python": sys.version.split()[0]}
    for p in ("mlx", "mlx-metal", "mlx-lm", "numpy", "escha-mlx"):
        try:
            out[p] = md.version(p)
        except Exception:
            out[p] = None
    return out
