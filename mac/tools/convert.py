"""Convert Kintsugi-27B (GGUF for the Escha llama.cpp fork) into the escha-mlx checkpoint form.

    python -I tools/convert.py --gguf <Kintsugi-27B-12G.gguf> --template <Escha W2 export dir> --out <dir>

What it writes (see README.md for the format):
  * the 400 coded projections: escha_code (bytes copied; the V-head permutation of the three linear-attention
    projections undone), escha_rin / escha_rout (our F16 vectors, already folded; no s_in / s_out), escha_config.
    The per-linear bias of the GGUF is all zeros and is not written.
  * embedding and head: weight_int8 [N, K] + weight_group_scale f16 [N, K/32]: the Q8_0 blocks split, exact.
    Needs the escha-mlx patch in patches/ (the stock loader reads one scale per row only).
  * norms and the GDN small tensors: the engine transforms undone (norm + 1, -exp(A_log), V-head order), f16.
  * mtp/model.safetensors: norms f16; the 8 matrices as MLX affine 4-bit, group 32, uint32 weight + f32 scales and
    biases (scale = d, bias = -8 d): exact for Q4_0. Needs the same patch.
  * config.json, tokenizer and template files from the template export; K per projection set from our code;
    the chat template from the GGUF.

The template export is used for config and tokenizer files only. No weight comes from it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kio  # noqa: E402
from kio import ATT, LIN, MLP, P, GGUF, blocks, coded_perm, unpermute  # noqa: E402

F16, F32 = np.float16, np.float32
N_LAYERS = 64
TEMPLATE_FILES = ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
MTP_NORMS = {"nextn.shared_head_norm": "norm", "nextn.enorm": "pre_fc_norm_embedding",
             "nextn.hnorm": "pre_fc_norm_hidden", "attn_norm": "layers.0.input_layernorm",
             "post_attention_norm": "layers.0.post_attention_layernorm",
             "attn_q_norm": "layers.0.self_attn.q_norm", "attn_k_norm": "layers.0.self_attn.k_norm"}
MTP_MATS = {"nextn.eh_proj": "fc", **{g: "layers.0." + h for h, g in list(ATT.items()) + list(MLP.items())}}

# ---------------------------------------------------------------------------------------------------------------
# small tensors: undo the engine transforms
# ---------------------------------------------------------------------------------------------------------------

NOTES: dict[str, int] = {}
#: Which exp the A_log search used (issue #1: numpy float32 exp gives other bits on ARM than on x86, where the GGUF's
#: ssm_a was made). The checker accepts up to 2 ulp for ssm_a.
EXP_USED = {"function": "numpy.exp on float32 arrays", "platform": platform.platform(),
            "machine": platform.machine(), "numpy": np.__version__}


def note(key: str, n: int = 1) -> None:
    NOTES[key] = NOTES.get(key, 0) + n


def norm_back(g: np.ndarray) -> np.ndarray:
    """GGUF norm = f32(w + 1) -> w as f16. g - 1 is exact in f32 for g in [0.5, 2] (Sterbenz) and for g >= 1."""
    g = np.asarray(g, dtype=F32)
    w = (g - F32(1.0)).astype(F32)
    w16 = w.astype(F16)
    ok = (w16.astype(F32) + F32(1.0)).astype(F32).view(np.uint32) == g.view(np.uint32)
    note("norm values", g.size)
    note("norm values not recovered exactly", int((~ok).sum()))
    return w16


def plain_back(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=F32)
    x16 = x.astype(F16)
    note("plain f32 values", x.size)
    note("plain f32 values not exact in f16", int((x16.astype(F32).view(np.uint32) != x.view(np.uint32)).sum()))
    return x16


def _bf16_round(x32: np.ndarray) -> np.ndarray:
    u = x32.astype(F32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(F32)


def alog_back(a: np.ndarray) -> np.ndarray:
    """GGUF ssm_a = -exp(f32(A_log)) with A_log a BF16 value -> that A_log (searched, then checked), as f16."""
    a = np.asarray(a, dtype=F32)
    c0 = _bf16_round(np.log(-a.astype(np.float64)).astype(F32))
    out = np.empty_like(a)
    for i in range(a.size):
        base = int(c0[i].view(np.uint32)) >> 16
        cands = np.array([(base + d) << 16 for d in range(-8, 9)], dtype=np.uint32).view(F32)
        hit = np.nonzero((-np.exp(cands)).astype(F32).view(np.uint32) == a[i].view(np.uint32))[0]
        if hit.size:
            out[i] = cands[hit[0]]
        else:
            out[i] = c0[i]
            note("A_log values with no exact BF16 source (nearest written)")
    note("A_log values", a.size)
    o16 = out.astype(F16)
    note("A_log values not exact in f16", int((o16.astype(F32).view(np.uint32) != out.view(np.uint32)).sum()))
    return o16


# ---------------------------------------------------------------------------------------------------------------
# tensor plan
# ---------------------------------------------------------------------------------------------------------------

def coded_items(g: GGUF, il: int, hf: str, name: str, vax: str | None, kmap: dict):
    code_g = g.get(name + ".escha_code")
    tk, tn, wpt = code_g.shape
    K, ic, oc = wpt // 16, tk * 16, tn * 16
    assert K in (2, 3), (name, code_g.shape)
    perm = coded_perm(vax, ic, oc)
    base = f"{P}layers.{il}.{hf}"
    kmap[base] = (K, ic, oc)

    def code():
        c = np.asarray(code_g)
        if perm is None:
            return c
        return unpermute(c, perm[1], axis=1 if perm[0] == "out" else 0)

    def vec(which):
        def f():
            v = np.asarray(g.get(name + ".escha_" + which))
            if perm is not None and ((which == "rout") == (perm[0] == "out")):
                v = unpermute(v, perm[2], axis=0)
            return v
        return f

    bias = np.asarray(g.get(name + ".bias"))
    if np.any(bias != 0):
        raise ValueError(f"{name}.bias is not all zeros: this converter drops it")
    note("coded projections")
    return [(base + ".escha_code", np.int16, (tk, tn, wpt), code),
            (base + ".escha_rin", F16, (ic,), vec("rin")),
            (base + ".escha_rout", F16, (oc,), vec("rout")),
            (base + ".escha_config", np.int32, (6,), lambda: np.array([16, K, 2, 1, ic, oc], dtype=np.int32))]


def layer_items(g: GGUF, il: int, kmap: dict):
    B, L = f"blk.{il}.", f"{P}layers.{il}."
    items = [(L + "input_layernorm.weight", F16, (5120,), lambda: norm_back(g.get(B + "attn_norm.weight"))),
             (L + "post_attention_layernorm.weight", F16, (5120,),
              lambda: norm_back(g.get(B + "post_attention_norm.weight")))]
    if B + "ssm_a" in g.tensors:
        h1 = blocks(48, 1)
        cidx = blocks(10240, 128, 4096)
        items += [
            (L + "linear_attn.A_log", F16, (48,), lambda: unpermute(alog_back(g.get(B + "ssm_a")), h1, 0)),
            (L + "linear_attn.dt_bias", F16, (48,), lambda: unpermute(plain_back(g.get(B + "ssm_dt.bias")), h1, 0)),
            (L + "linear_attn.conv1d.weight", F16, (10240, 1, 4),
             lambda: unpermute(plain_back(g.get(B + "ssm_conv1d.weight")), cidx, 0).reshape(10240, 1, 4)),
            (L + "linear_attn.norm.weight", F16, (128,), lambda: plain_back(g.get(B + "ssm_norm.weight"))),
            (L + "linear_attn.in_proj_a.weight", F16, (48, 5120),
             lambda: unpermute(np.asarray(g.get(B + "ssm_alpha.weight")), h1, 0)),
            (L + "linear_attn.in_proj_b.weight", F16, (48, 5120),
             lambda: unpermute(np.asarray(g.get(B + "ssm_beta.weight")), h1, 0)),
        ]
        for hf, (name, vax) in LIN.items():
            items += coded_items(g, il, hf, B + name, vax, kmap)
    else:
        items += [(L + "self_attn.q_norm.weight", F16, (256,), lambda: norm_back(g.get(B + "attn_q_norm.weight"))),
                  (L + "self_attn.k_norm.weight", F16, (256,), lambda: norm_back(g.get(B + "attn_k_norm.weight")))]
        for hf, name in ATT.items():
            items += coded_items(g, il, hf, B + name, None, kmap)
    for hf, name in MLP.items():
        items += coded_items(g, il, hf, B + name, None, kmap)
    return items


def q8_items(g: GGUF, gname: str, base: str):
    raw = g.blocks(gname)
    rows, nb, _ = raw.shape
    step = 16384

    def q():
        out = np.empty((rows, nb * 32), dtype=np.int8)
        for a in range(0, rows, step):
            out[a:a + step] = kio.q8_0_split(raw[a:a + step])[0]
        return out

    def d():
        out = np.empty((rows, nb), dtype=F16)
        for a in range(0, rows, step):
            out[a:a + step] = kio.q8_0_split(raw[a:a + step])[1]
        return out
    note("Q8_0 tensors split")
    return [(base + ".weight_int8", np.int8, (rows, nb * 32), q),
            (base + ".weight_group_scale", F16, (rows, nb), d)]


def q4_items(g: GGUF, gname: str, base: str):
    raw = g.blocks(gname)
    rows, nb, _ = raw.shape
    cache = {}

    def split():
        if not cache:
            q, d = kio.q4_0_split(raw)
            cache["w"] = kio.mlx_pack(q, 4)
            d32 = d.astype(F32)
            cache["s"] = d32
            cache["b"] = (F32(-8.0) * d32).astype(F32)
        return cache
    note("Q4_0 tensors repacked")
    return [(base + ".weight", np.uint32, (rows, nb * 32 // 8), lambda: split()["w"]),
            (base + ".scales", F32, (rows, nb), lambda: split()["s"]),
            (base + ".biases", F32, (rows, nb), lambda: split()["b"])]


# ---------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------

def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--template", required=True, help="Escha W2 HF export dir (config and tokenizer files only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-hash", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    g = GGUF(a.gguf)
    if g.kv.get("general.architecture") != "qwen35" or int(g.kv.get("qwen35.block_count", 0)) != N_LAYERS + 1:
        raise ValueError("expected a qwen35 GGUF with 64 layers + 1 MTP block")
    for t in ("token_embd.weight", "output.weight"):
        if g.type_of(t) != kio.Q8_0:
            raise ValueError(f"{t}: expected Q8_0")
    os.makedirs(os.path.join(a.out, "mtp"), exist_ok=True)

    kmap: dict = {}
    shard1 = q8_items(g, "token_embd.weight", P + "embed_tokens")
    for il in range(N_LAYERS // 2):
        shard1 += layer_items(g, il, kmap)
    shard2 = []
    for il in range(N_LAYERS // 2, N_LAYERS):
        shard2 += layer_items(g, il, kmap)
    shard2 += [(P + "norm.weight", F16, (5120,), lambda: norm_back(g.get("output_norm.weight")))]
    shard2 += q8_items(g, "output.weight", "lm_head")

    names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    weight_map, total = {}, 0
    for fname, items in zip(names, (shard1, shard2)):
        n = kio.write_safetensors(os.path.join(a.out, fname), items, {"format": "pt"})
        total += n
        weight_map.update({it[0]: fname for it in items})
        print(f"{fname}: {len(items)} tensors, {n:,} B ({time.time() - t0:.0f} s)", flush=True)
    with open(os.path.join(a.out, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)

    # MTP block
    mtp = []
    for gn, hn in MTP_NORMS.items():
        shape = (256,) if hn.endswith(("q_norm", "k_norm")) else (5120,)
        mtp.append(("mtp." + hn + ".weight", F16, shape,
                    lambda gn=gn: norm_back(g.get("blk.64." + gn + ".weight"))))
    for gn, hn in MTP_MATS.items():
        t = "blk.64." + gn + ".weight"
        if g.type_of(t) != kio.Q4_0:
            raise ValueError(f"{t}: expected Q4_0")
        mtp += q4_items(g, t, "mtp." + hn)
    kio.write_safetensors(os.path.join(a.out, "mtp", "model.safetensors"), mtp, {"format": "pt"})
    print(f"mtp/model.safetensors: {len(mtp)} tensors ({time.time() - t0:.0f} s)", flush=True)

    # config, tokenizer, template
    cfg = json.load(open(os.path.join(a.template, "config.json"), encoding="utf-8"))
    meta = cfg["quantization_config"]["layer_meta"]
    if set(meta) != set(kmap):
        raise ValueError("template layer_meta does not list the same 400 projections")
    changed = 0
    for base, (K, ic, oc) in kmap.items():
        m = meta[base]
        if (m["in_features"], m["out_features"]) != (ic, oc):
            raise ValueError(f"{base}: shape differs from the template")
        changed += int(m["K"] != K)
        m["K"] = K
    cfg["_name_or_path"] = "Kintsugi-27B"
    with open(os.path.join(a.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    shutil.copy(os.path.join(a.template, "quantize_config.json"), a.out)
    for fn in TEMPLATE_FILES:
        shutil.copy(os.path.join(a.template, fn), a.out)
    tmpl = g.kv["tokenizer.chat_template"]
    with open(os.path.join(a.out, "chat_template.jinja"), "w", encoding="utf-8", newline="") as f:
        f.write(tmpl)
    tc = json.load(open(os.path.join(a.out, "tokenizer_config.json"), encoding="utf-8"))
    if "chat_template" in tc:
        tc["chat_template"] = tmpl
        with open(os.path.join(a.out, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(tc, f, indent=2, ensure_ascii=False)
    mcfg = json.load(open(os.path.join(a.template, "mtp", "config.json"), encoding="utf-8"))
    mcfg["quantization"] = {"group_size": 32, "bits": 4, "mode": "affine"}
    with open(os.path.join(a.out, "mtp", "config.json"), "w", encoding="utf-8") as f:
        json.dump(mcfg, f, indent=2)

    prov = {"source": os.path.basename(a.gguf), "source_bytes": os.path.getsize(a.gguf),
            "source_sha256": None if a.no_hash else sha256(a.gguf), "general.name": g.kv.get("general.name"),
            "converter": "kintsugi-mlx tools/convert.py", "K_changed_vs_template": changed,
            "K_counts": {f"K{k}": sum(1 for v in kmap.values() if v[0] == k) for k in (2, 3)},
            "notes": NOTES, "A_log_exp_used": EXP_USED, "seconds": round(time.time() - t0)}
    with open(os.path.join(a.out, "kintsugi_conversion.json"), "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
    print(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
