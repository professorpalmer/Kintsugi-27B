"""Mac-free checks of a converted Kintsugi checkpoint (gate G1-G5 + diagnostic D1 of MAC-FEASIBILITY.md).

    python -I tools/check.py --gguf <Kintsugi GGUF> --conv <converted dir> --escha-mlx <escha-mlx clone>
                             [--template <Escha W2 export, for D1>] [--only G1,G2,...]

Reference decode: escha-mlx `escha_mlx/ref.py` (NumPy only; imported by path, the package __init__ is not run).
Engine decode: work/engine_escha.exe (tools/engine_escha.c, the engine's CPU op copied without change).
Results: logs/check.json and stdout.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kio  # noqa: E402
from kio import ATT, LIN, MLP, P, GGUF, Safe, blocks, coded_perm  # noqa: E402

F16, F32 = np.float16, np.float32
ROOT = os.path.dirname(HERE)
WORK = os.path.join(ROOT, "work")
EXE = os.path.join(WORK, "engine_escha.exe")
G2_LAYERS = (0, 3, 34, 63)
G2_TOL = 1e-3


def bits_equal(a, b) -> bool:
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def projections(g: GGUF):
    """(layer, HF name, GGUF name, V axis) for all 400 coded projections."""
    out = []
    for il in range(64):
        B = f"blk.{il}."
        if B + "ssm_a" in g.tensors:
            out += [(il, hf, B + n, v) for hf, (n, v) in LIN.items()]
        else:
            out += [(il, hf, B + n, None) for hf, n in ATT.items()]
        out += [(il, hf, B + n, None) for hf, n in MLP.items()]
    return out


def forward_perm(a, perm, which):
    """Apply the GGUF converter's permutation to an HF-order array. which: code / rin / rout."""
    if perm is None:
        return a
    axis_name, tidx, cidx = perm
    if which == "code":
        return np.take(a, tidx, axis=1 if axis_name == "out" else 0)
    if (which == "rout") == (axis_name == "out"):
        return a[cidx]
    return a


# ---------------------------------------------------------------------------------------------------------------

def g1(g, s, res):
    bad, n = [], 0
    for il, hf, gn, vax in projections(g):
        base = f"{P}layers.{il}.{hf}"
        code_c = np.asarray(s.get(base + ".escha_code"))
        tk, tn, wpt = code_c.shape
        perm = coded_perm(vax, tk * 16, tn * 16)
        for which, conv in (("code", code_c), ("rin", np.asarray(s.get(base + ".escha_rin"))),
                            ("rout", np.asarray(s.get(base + ".escha_rout")))):
            if not bits_equal(forward_perm(conv, perm, which), np.asarray(g.get(gn + ".escha_" + which))):
                bad.append(f"{base}.{which}")
        cfg = [int(v) for v in np.asarray(s.get(base + ".escha_config"))]
        if cfg != [16, wpt // 16, 2, 1, tk * 16, tn * 16]:
            bad.append(f"{base}.escha_config {cfg}")
        n += 1
    res["G1"] = {"projections": n, "failures": bad, "pass": n == 400 and not bad}


def g2(g, s, ref, res, seed=1234):
    rng = np.random.default_rng(seed)
    rows = []
    os.makedirs(WORK, exist_ok=True)
    for il, hf, gn, vax in projections(g):
        if il not in G2_LAYERS:
            continue
        base = f"{P}layers.{il}.{hf}"
        code_c = np.asarray(s.get(base + ".escha_code"))
        tk, tn, wpt = code_c.shape
        K, ic, oc = wpt // 16, tk * 16, tn * 16
        perm = coded_perm(vax, ic, oc)
        t0 = time.time()
        # reference: escha-mlx NumPy decode of the converted (HF-order) stream
        w_ref = ref.reconstruct_fast(code_c, ic, oc, K)                       # f16 [IC, OC]
        rin_c, rout_c = np.asarray(s.get(base + ".escha_rin")), np.asarray(s.get(base + ".escha_rout"))
        ri, ro = ref.fold_scales(rin_c, rout_c, None, None)
        x = rng.standard_normal((4, ic)).astype(F32)
        y_ref = ref.expert_linear(x, None, ri, ro, K, w_bare=w_ref)          # f16 [4, OC]
        # engine: the GGUF tensors as they are, inputs in GGUF order
        x_g = x[:, perm[2]] if perm is not None and perm[0] == "in" else x
        files = {k: os.path.join(WORK, k + ".bin") for k in ("code", "dep", "w", "x", "rin", "rout", "y")}
        np.ascontiguousarray(g.get(gn + ".escha_code")).tofile(files["code"])
        np.ascontiguousarray(g.get(f"escha_dep_k{K}")).tofile(files["dep"])
        np.ascontiguousarray(x_g).tofile(files["x"])
        np.ascontiguousarray(g.get(gn + ".escha_rin")).tofile(files["rin"])
        np.ascontiguousarray(g.get(gn + ".escha_rout")).tofile(files["rout"])
        subprocess.run([EXE, files["code"], files["dep"], str(ic), str(oc), str(K), files["w"],
                        files["x"], "4", files["rin"], files["rout"], files["y"]], check=True)
        w_eng = np.fromfile(files["w"], dtype=F32).reshape(ic, oc)
        y_eng = np.fromfile(files["y"], dtype=F32).reshape(4, oc)
        # map the reference into GGUF order
        w_r = w_ref.astype(F32)
        y_r = y_ref.astype(F32)
        if perm is not None and perm[0] == "out":
            w_r, y_r = w_r[:, perm[2]], y_r[:, perm[2]]
        elif perm is not None:
            w_r = w_r[perm[2], :]
        exact = bits_equal(w_r, w_eng)
        n_diff = int((w_r.view(np.uint32) != w_eng.view(np.uint32)).sum())
        rel = float(np.linalg.norm(y_r - y_eng) / np.linalg.norm(y_eng))
        rows.append({"projection": base, "K": K, "shape": [ic, oc], "weights_exact": exact,
                     "weights_bits_diff": n_diff, "linear_rel_l2": rel,
                     "linear_max_abs": float(np.abs(y_r - y_eng).max()), "seconds": round(time.time() - t0, 1)})
        print(f"  G2 {base}: K{K} weights {'exact' if exact else f'DIFF {n_diff}'}, linear rel L2 {rel:.2e}",
              flush=True)
        for f in files.values():
            if os.path.exists(f):
                os.remove(f)
    res["G2"] = {"layers": list(G2_LAYERS), "rows": rows, "n": len(rows),
                 "G2a_pass": bool(rows) and all(r["weights_exact"] for r in rows),
                 "G2b_pass": bool(rows) and all(r["linear_rel_l2"] <= G2_TOL for r in rows),
                 "G2b_max_rel_l2": max(r["linear_rel_l2"] for r in rows), "G2b_tolerance": G2_TOL}


#: G3 rule, per class (fixed 2026-11-20 after the first Mac run, issue #1; written before the rerun):
#:   ssm_a = -exp(A_log): max 2 ulp (f32). The GGUF value came from numpy float32 exp on x86; ARM numpy gives other
#:     bits for the same input (up to 1 ulp per exp), and the runtime computes exp on Metal anyway. The ulp
#:     histogram is recorded.
#:   conv1d: every unequal value must be in the f16 subnormal range (|GGUF value| < 2**-14, where escha-mlx's f16
#:     storage cannot hold a BF16 value exactly) with abs error <= 3e-8; the count is recorded.
#:   every other class: bit-exact.
G3_ULP_MAX = 2
G3_SUBNORMAL = 2.0 ** -14
G3_CONV1D_MAX_ABS = 3e-8


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance in f32 units in the last place, over the ordered integer line (sign handled, -0 == +0)."""
    def ordered(x):
        i = np.ascontiguousarray(x, dtype=F32).view(np.int32).astype(np.int64)
        return np.where(i < 0, -(i & 0x7FFFFFFF), i)
    return np.abs(ordered(a) - ordered(b))


def g3(g, s, mt, res):
    h1, cidx = blocks(48, 1), blocks(10240, 128, 4096)
    stats = {}

    def rec(cls, conv, gg):
        gg = np.ascontiguousarray(gg)
        conv = np.ascontiguousarray(conv).astype(gg.dtype)
        d = stats.setdefault(cls, {"tensors": 0, "values": 0, "values_not_equal": 0, "max_abs_err": 0.0})
        d["tensors"] += 1
        d["values"] += gg.size
        ne = conv.view(np.uint8).reshape(gg.size, -1) != gg.view(np.uint8).reshape(gg.size, -1)
        ne = ne.any(axis=1)
        d["values_not_equal"] += int(ne.sum())
        if ne.any():
            d["max_abs_err"] = max(d["max_abs_err"],
                                   float(np.abs(conv.astype(np.float64) - gg.astype(np.float64)).max()))
        if cls.startswith("ssm_a"):
            u = ulp_distance(conv.ravel(), gg.ravel())
            hist = d.setdefault("ulp_histogram", {})
            for k, n in zip(*np.unique(u, return_counts=True)):
                hist[str(int(k))] = hist.get(str(int(k)), 0) + int(n)
            d["max_ulp"] = max(d.get("max_ulp", 0), int(u.max()))
        if cls == "conv1d":
            flat_c, flat_g = conv.ravel()[ne], gg.ravel()[ne]
            d["unequal_outside_f16_subnormal"] = d.get("unequal_outside_f16_subnormal", 0) + int(
                (np.abs(flat_g.astype(np.float64)) >= G3_SUBNORMAL).sum())

    one = F32(1.0)
    norm = lambda w16: (np.asarray(w16).astype(F32) + one).astype(F32)  # noqa: E731
    for il in range(64):
        B, L = f"blk.{il}.", f"{P}layers.{il}."
        rec("norm", norm(s.get(L + "input_layernorm.weight")), g.get(B + "attn_norm.weight"))
        rec("norm", norm(s.get(L + "post_attention_layernorm.weight")), g.get(B + "post_attention_norm.weight"))
        if B + "ssm_a" in g.tensors:
            alog = np.asarray(s.get(L + "linear_attn.A_log")).astype(F32)
            rec("ssm_a = -exp(A_log)", (-np.exp(alog)).astype(F32)[h1], g.get(B + "ssm_a"))
            rec("dt_bias", np.asarray(s.get(L + "linear_attn.dt_bias")).astype(F32)[h1], g.get(B + "ssm_dt.bias"))
            conv = np.asarray(s.get(L + "linear_attn.conv1d.weight")).astype(F32).reshape(-1, 4)[cidx]
            rec("conv1d", conv, g.get(B + "ssm_conv1d.weight"))
            rec("linear_attn.norm", np.asarray(s.get(L + "linear_attn.norm.weight")).astype(F32),
                g.get(B + "ssm_norm.weight"))
            rec("in_proj_a/b", np.asarray(s.get(L + "linear_attn.in_proj_a.weight"))[h1], g.get(B + "ssm_alpha.weight"))
            rec("in_proj_a/b", np.asarray(s.get(L + "linear_attn.in_proj_b.weight"))[h1], g.get(B + "ssm_beta.weight"))
        else:
            rec("norm", norm(s.get(L + "self_attn.q_norm.weight")), g.get(B + "attn_q_norm.weight"))
            rec("norm", norm(s.get(L + "self_attn.k_norm.weight")), g.get(B + "attn_k_norm.weight"))
    rec("norm", norm(s.get(P + "norm.weight")), g.get("output_norm.weight"))
    mtp_norms = {"nextn.shared_head_norm": "norm", "nextn.enorm": "pre_fc_norm_embedding",
                 "nextn.hnorm": "pre_fc_norm_hidden", "attn_norm": "layers.0.input_layernorm",
                 "post_attention_norm": "layers.0.post_attention_layernorm",
                 "attn_q_norm": "layers.0.self_attn.q_norm", "attn_k_norm": "layers.0.self_attn.k_norm"}
    for gn, hn in mtp_norms.items():
        rec("mtp norm", norm(mt.get("mtp." + hn + ".weight")), g.get("blk.64." + gn + ".weight"))
    for cls, d in stats.items():
        if cls.startswith("ssm_a"):
            d["rule"] = f"max ulp <= {G3_ULP_MAX}"
            d["pass"] = d.get("max_ulp", 0) <= G3_ULP_MAX
        elif cls == "conv1d":
            d["rule"] = (f"unequal values only in the f16 subnormal range (|x| < 2^-14), abs error <= "
                         f"{G3_CONV1D_MAX_ABS:g}; count recorded")
            d["pass"] = (d.get("unequal_outside_f16_subnormal", 0) == 0
                         and d["max_abs_err"] <= G3_CONV1D_MAX_ABS)
        else:
            d["rule"] = "bit-exact"
            d["pass"] = d["values_not_equal"] == 0
    res["G3"] = {"classes": stats, "pass": all(d["pass"] for d in stats.values()),
                 "failed_classes": [c for c, d in stats.items() if not d["pass"]],
                 "exp_used": {"platform": platform.platform(), "machine": platform.machine(),
                              "numpy": np.__version__}}


def g4(g, s, mt, res):
    out = {}
    for gn, base in (("token_embd.weight", P + "embed_tokens"), ("output.weight", "lm_head")):
        raw = g.blocks(gn)
        w8 = s.get(base + ".weight_int8")
        gs = s.get(base + ".weight_group_scale")
        rows, diff = raw.shape[0], 0
        for a in range(0, rows, 8192):
            b = min(rows, a + 8192)
            # the patched escha-mlx loader (quant.pack_q8_grouped): q = w8 XOR 0x80 as little-endian uint32,
            # scales = f32(d), biases = -128 * f32(d)
            q8 = np.ascontiguousarray(w8[a:b])
            packed = (q8.view(np.uint8) ^ np.uint8(0x80)).view(np.uint32)
            d32 = np.asarray(gs[a:b]).astype(F32)
            mlx = kio.mlx_affine_dequant(packed, d32, (F32(-128.0) * d32).astype(F32), 8)
            eng = kio.ggml_dequant_q8_0(raw[a:b])
            diff += int((mlx.view(np.uint32) != eng.view(np.uint32)).sum())
        out[gn] = {"rows": rows, "values_not_equal": diff}
    res["G4a"] = {"tensors": out, "pass": all(v["values_not_equal"] == 0 for v in out.values())}
    out = {}
    mats = {"nextn.eh_proj": "fc", **{g_: "layers.0." + h for h, g_ in list(ATT.items()) + list(MLP.items())}}
    for gn, hn in mats.items():
        raw = g.blocks("blk.64." + gn + ".weight")
        base = "mtp." + hn
        mlx = kio.mlx_affine_dequant(np.asarray(mt.get(base + ".weight")), np.asarray(mt.get(base + ".scales")),
                                     np.asarray(mt.get(base + ".biases")), 4)
        eng = kio.ggml_dequant_q4_0(raw)
        bits_ne = mlx.view(np.uint32) != eng.view(np.uint32)
        # ggml gives (q - 8) * d = -0.0 for q = 8 when d < 0; the affine form gives 8d + (-8d) = +0.0.
        # Equal as numbers; a dot product cannot tell them apart.
        signed_zero = bits_ne & (mlx == 0) & (eng == 0)
        out[base] = {"shape": list(eng.shape), "values_not_equal": int((mlx != eng).sum()),
                     "bits_not_equal": int(bits_ne.sum()), "bits_not_equal_signed_zero_only": int(signed_zero.sum())}
    res["G4b"] = {"tensors": out, "n": len(out),
                  "pass": len(out) == 8 and all(v["values_not_equal"] == 0 for v in out.values())}


def g5(g, s, conv_dir, res):
    cfg = json.load(open(os.path.join(conv_dir, "config.json"), encoding="utf-8"))
    meta = cfg["quantization_config"]["layer_meta"]
    bad = []
    for il, hf, gn, vax in projections(g):
        base = f"{P}layers.{il}.{hf}"
        K = int(s.idx[base + ".escha_code"][2]["shape"][2]) // 16
        if meta[base]["K"] != K or int(np.asarray(s.get(base + ".escha_config"))[1]) != K:
            bad.append(base)
    tok = json.load(open(os.path.join(conv_dir, "tokenizer.json"), encoding="utf-8"))
    vocab = dict(tok["model"]["vocab"])
    for t in tok.get("added_tokens", []):
        vocab[t["content"]] = t["id"]
    gt = g.kv["tokenizer.ggml.tokens"]
    tok_diff = sum(1 for t, i in vocab.items() if i >= len(gt) or gt[i] != t)
    tmpl = open(os.path.join(conv_dir, "chat_template.jinja"), encoding="utf-8", newline="").read()
    tc = json.load(open(os.path.join(conv_dir, "tokenizer_config.json"), encoding="utf-8"))
    res["G5"] = {"K_mismatch": bad, "gguf_tokens": len(gt), "hf_vocab_entries": len(vocab),
                 "token_text_mismatch": tok_diff, "chat_template_equal": tmpl == g.kv["tokenizer.chat_template"],
                 "tokenizer_config_template_equal": tc.get("chat_template", tmpl) == g.kv["tokenizer.chat_template"],
                 "K_counts": {f"K{k}": sum(1 for m in meta.values() if m["K"] == k) for k in (2, 3)}}
    r = res["G5"]
    r["pass"] = (not bad and tok_diff == 0 and r["chat_template_equal"] and r["tokenizer_config_template_equal"])


def d1(g, s, ref, template, res):
    """Effective dense weight of layer 0 in_proj_qkv, V part: Kintsugi (HF order) vs Escha W2."""
    t = Safe(sorted(glob.glob(os.path.join(template, "model-0000*.safetensors"))))
    base = P + "layers.0.linear_attn.in_proj_qkv"

    def eff(code, rin, rout):
        tk, tn, wpt = code.shape
        w = ref.reconstruct_fast(np.asarray(code), tk * 16, tn * 16, wpt // 16).astype(F32)
        w = ref.h128(w.T).T * F32(ref.RS)
        w = ref.h128(w) * F32(ref.RS)
        return w * rin.astype(F32)[:, None] * rout.astype(F32)[None, :]

    kin = eff(s.get(base + ".escha_code"), np.asarray(s.get(base + ".escha_rin")),
              np.asarray(s.get(base + ".escha_rout")))
    esc = eff(t.get(base + ".escha_code"),
              np.asarray(t.get(base + ".escha_rin")).astype(F32) * np.asarray(t.get(base + ".escha_s_in")),
              np.asarray(t.get(base + ".escha_rout")).astype(F32) * np.asarray(t.get(base + ".escha_s_out")))
    cidx = blocks(10240, 128, 4096)
    kin_gguf_order = kin[:, cidx]
    v = slice(4096, 10240)
    cos = lambda a, b: float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))  # noqa: E731
    res["D1"] = {"cos_right_order": cos(kin[:, v], esc[:, v]), "cos_gguf_order": cos(kin_gguf_order[:, v], esc[:, v]),
                 "cos_qk_part": cos(kin[:, :4096], esc[:, :4096])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--conv", required=True)
    ap.add_argument("--escha-mlx", required=True)
    ap.add_argument("--template", default="")
    ap.add_argument("--only", default="G1,G2,G3,G4,G5,D1")
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "check.json"))
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.escha_mlx, "escha_mlx"))
    import ref  # noqa: E402  (escha-mlx NumPy reference; numpy only)
    commit = subprocess.run(["git", "-C", a.escha_mlx, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    g = GGUF(a.gguf)
    s = Safe(sorted(glob.glob(os.path.join(a.conv, "model-0000*.safetensors"))))
    mt = Safe([os.path.join(a.conv, "mtp", "model.safetensors")])
    res = {"escha_mlx_commit": commit, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    if os.path.exists(a.out):
        old = json.load(open(a.out, encoding="utf-8"))
        res = {**old, **res}
    only = set(a.only.split(","))
    for key, fn in (("G1", lambda: g1(g, s, res)), ("G3", lambda: g3(g, s, mt, res)),
                    ("G4", lambda: g4(g, s, mt, res)), ("G5", lambda: g5(g, s, a.conv, res)),
                    ("D1", lambda: d1(g, s, ref, a.template, res) if a.template else None),
                    ("G2", lambda: g2(g, s, ref, res))):
        if key in only:
            t0 = time.time()
            print(f"{key} ...", flush=True)
            fn()
            print(f"{key} done in {time.time() - t0:.0f} s: "
                  f"{json.dumps({k: v for k, v in res.get(key, res.get(key + 'a', {})).items() if k != 'rows'})[:600]}",
                  flush=True)
    res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
