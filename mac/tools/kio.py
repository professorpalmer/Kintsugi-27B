"""Shared I/O for the Kintsugi -> escha-mlx converter and its checks.

- GGUF: a minimal reader (metadata + tensors as numpy memmaps). Quantized types come back as raw bytes.
- Safetensors: a minimal streaming writer (one tensor in memory at a time) and a reader (numpy memmaps).
- The V-head permutation that the GGUF converter applies to the linear-attention tensors, and its inverse.

No third-party package except numpy.
"""
from __future__ import annotations

import json
import os
import struct

import numpy as np

# ---------------------------------------------------------------------------------------------------------------
# GGUF
# ---------------------------------------------------------------------------------------------------------------

_T = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_NP = {0: np.float32, 1: np.float16, 24: np.int8, 25: np.int16, 26: np.int32, 27: np.int64, 28: np.float64}
#: ggml block types: type id -> (weights per block, bytes per block)
_BLOCK = {2: (32, 18), 8: (32, 34)}          # Q4_0, Q8_0
Q4_0, Q8_0 = 2, 8


class GGUF:
    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                raise ValueError(f"{path}: not a GGUF file")
            rd = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]  # noqa: E731
            rstr = lambda: f.read(rd("Q")).decode("utf-8", "surrogateescape")              # noqa: E731

            def rval(t):
                if t == 8:
                    return rstr()
                if t == 9:
                    et, n = rd("I"), rd("Q")
                    if et in _T:
                        sz = struct.calcsize("<" + _T[et])
                        return np.frombuffer(f.read(sz * n), dtype="<" + _T[et].replace("?", "b"))
                    return [rval(et) for _ in range(n)]
                return rd(_T[t])

            self.version, nt, nkv = rd("I"), rd("Q"), rd("Q")
            self.kv = {}
            for _ in range(nkv):
                k = rstr()
                self.kv[k] = rval(rd("I"))
            self.tensors = {}
            for _ in range(nt):
                name = rstr()
                nd = rd("I")
                dims = [rd("Q") for _ in range(nd)]
                ty, off = rd("I"), rd("Q")
                self.tensors[name] = (dims, ty, off)
            align = int(self.kv.get("general.alignment", 32))
            self.data_start = (f.tell() + align - 1) // align * align

    def type_of(self, name: str) -> int:
        return self.tensors[name][1]

    def get(self, name: str) -> np.ndarray:
        """Plain tensor in numpy order (ggml dims reversed)."""
        dims, ty, off = self.tensors[name]
        n = int(np.prod(dims))
        a = np.memmap(self.path, dtype=_NP[ty], mode="r", offset=self.data_start + off, shape=(n,))
        return a.reshape(dims[::-1])

    def blocks(self, name: str) -> np.ndarray:
        """Block-quantized tensor as raw bytes [rows, blocks_per_row, bytes_per_block]."""
        dims, ty, off = self.tensors[name]
        per, nbytes = _BLOCK[ty]
        cols, rows = dims[0], int(np.prod(dims[1:]))
        a = np.memmap(self.path, dtype=np.uint8, mode="r", offset=self.data_start + off,
                      shape=(rows, cols // per, nbytes))
        return a


def q8_0_split(raw: np.ndarray):
    """Q8_0 bytes [rows, nb, 34] -> (int8 q [rows, nb*32], f16 d [rows, nb])."""
    raw = np.ascontiguousarray(raw)
    d = raw[:, :, 0:2].copy().view(np.float16).reshape(raw.shape[0], raw.shape[1])
    q = raw[:, :, 2:].copy().view(np.int8).reshape(raw.shape[0], raw.shape[1] * 32)
    return q, d


def q4_0_split(raw: np.ndarray):
    """Q4_0 bytes [rows, nb, 18] -> (uint8 q in 0..15 [rows, nb*32] in weight order, f16 d [rows, nb]).

    ggml layout (dequantize_row_q4_0): byte j of qs holds weight j (low nibble) and weight j+16 (high nibble).
    """
    raw = np.ascontiguousarray(raw)
    rows, nb = raw.shape[0], raw.shape[1]
    d = raw[:, :, 0:2].copy().view(np.float16).reshape(rows, nb)
    qs = raw[:, :, 2:]
    q = np.empty((rows, nb, 32), dtype=np.uint8)
    q[:, :, :16] = qs & 0x0F
    q[:, :, 16:] = qs >> 4
    return q.reshape(rows, nb * 32), d


def ggml_dequant_q8_0(raw: np.ndarray) -> np.ndarray:
    """ggml dequantize_row_q8_0: y = f32(d) * q, in f32."""
    q, d = q8_0_split(raw)
    return (q.reshape(q.shape[0], -1, 32).astype(np.float32)
            * d.astype(np.float32)[:, :, None]).reshape(q.shape)


def ggml_dequant_q4_0(raw: np.ndarray) -> np.ndarray:
    """ggml dequantize_row_q4_0: y = (q - 8) * f32(d), in f32."""
    q, d = q4_0_split(raw)
    x = q.reshape(q.shape[0], -1, 32).astype(np.int32) - 8
    return (x.astype(np.float32) * d.astype(np.float32)[:, :, None]).reshape(q.shape)


# ---------------------------------------------------------------------------------------------------------------
# MLX affine packing (exact for the ggml blocks above)
# ---------------------------------------------------------------------------------------------------------------

def mlx_pack(q_unsigned: np.ndarray, bits: int) -> np.ndarray:
    """Unsigned codes [rows, cols] -> MLX packed uint32 [rows, cols*bits/32], element k of a word at bit k*bits."""
    rows, cols = q_unsigned.shape
    per = 32 // bits
    q = q_unsigned.astype(np.uint32).reshape(rows, cols // per, per)
    out = np.zeros((rows, cols // per), dtype=np.uint32)
    for k in range(per):
        out |= q[:, :, k] << np.uint32(k * bits)
    return out


def mlx_unpack(packed: np.ndarray, bits: int) -> np.ndarray:
    rows, words = packed.shape
    per = 32 // bits
    mask = np.uint32((1 << bits) - 1)
    out = np.empty((rows, words, per), dtype=np.uint32)
    for k in range(per):
        out[:, :, k] = (packed >> np.uint32(k * bits)) & mask
    return out.reshape(rows, words * per)


def mlx_affine_dequant(packed: np.ndarray, scales: np.ndarray, biases: np.ndarray, bits: int) -> np.ndarray:
    """NumPy model of mx.dequantize (affine): w = scale * q + bias, in the scales' dtype (f32 here)."""
    q = mlx_unpack(packed, bits).astype(np.float32)
    rows, cols = q.shape
    g = cols // scales.shape[1]
    q = q.reshape(rows, -1, g)
    w = q * scales.astype(np.float32)[:, :, None]
    w = w + biases.astype(np.float32)[:, :, None]
    return w.reshape(rows, cols)


# ---------------------------------------------------------------------------------------------------------------
# V-head permutation (tools/convert_escha_to_gguf.py of the port: blocks(), HEADS)
# ---------------------------------------------------------------------------------------------------------------

NK, VPK, HD = 16, 3, 128          # linear attention: 16 K heads, 3 V heads per K head, head dim 128
HEADS = np.array([g * VPK + r for r in range(VPK) for g in range(NK)])   # GGUF head j <- HF head HEADS[j]


def blocks(n_units: int, unit: int, first: int = 0) -> np.ndarray:
    """Index vector of the GGUF converter: gguf = hf[idx] along one axis."""
    idx = np.arange(first + len(HEADS) * unit)
    v = (HEADS[:, None] * unit + np.arange(unit)[None, :]).reshape(-1) + first
    idx[first:] = v
    assert len(idx) == n_units, (len(idx), n_units)
    return idx


def unpermute(a: np.ndarray, idx: np.ndarray, axis: int) -> np.ndarray:
    """Inverse of `a_gguf = take(a_hf, idx, axis)`: returns a_hf."""
    out = np.empty_like(a)
    sl = [slice(None)] * a.ndim
    sl[axis] = idx
    out[tuple(sl)] = a
    return out


def coded_perm(vax: str | None, ic: int, oc: int):
    """(axis name, index on 16-tiles, index on channels) for one coded projection, or None."""
    if vax == "out_qkv":
        return "out", blocks(oc // 16, HD // 16, 2 * NK * HD // 16), blocks(oc, HD, 2 * NK * HD)
    if vax == "out":
        return "out", blocks(oc // 16, HD // 16), blocks(oc, HD)
    if vax == "in":
        return "in", blocks(ic // 16, HD // 16), blocks(ic, HD)
    return None


# HF projection -> (GGUF base, V axis)
LIN = {"linear_attn.in_proj_qkv": ("attn_qkv", "out_qkv"), "linear_attn.in_proj_z": ("attn_gate", "out"),
       "linear_attn.out_proj": ("ssm_out", "in")}
ATT = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
       "self_attn.o_proj": "attn_output"}
MLP = {"mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}
P = "model.language_model."


# ---------------------------------------------------------------------------------------------------------------
# safetensors
# ---------------------------------------------------------------------------------------------------------------

_ST_DT = {np.dtype(np.int8): "I8", np.dtype(np.int16): "I16", np.dtype(np.int32): "I32", np.dtype(np.uint8): "U8",
          np.dtype(np.uint32): "U32", np.dtype(np.float16): "F16", np.dtype(np.float32): "F32"}
_ST_NP = {v: k for k, v in _ST_DT.items()}


def write_safetensors(path: str, items, metadata: dict | None = None) -> int:
    """items: list of (name, dtype, shape, producer). producer() returns the array; it runs once, at write time."""
    header, off = {}, 0
    for name, dt, shape, _ in items:
        n = int(np.prod(shape)) * np.dtype(dt).itemsize
        header[name] = {"dtype": _ST_DT[np.dtype(dt)], "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    if metadata:
        header["__metadata__"] = metadata
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for name, dt, shape, produce in items:
            a = np.ascontiguousarray(produce())
            if a.dtype != np.dtype(dt) or tuple(a.shape) != tuple(shape):
                raise ValueError(f"{name}: produced {a.dtype} {a.shape}, declared {np.dtype(dt)} {tuple(shape)}")
            f.write(memoryview(a).cast("B"))
    os.replace(tmp, path)
    return off


class Safe:
    """Read-only safetensors view over one or more files (numpy memmaps)."""

    def __init__(self, paths):
        self.idx = {}
        for p in paths:
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                for k, v in json.loads(f.read(n)).items():
                    if k != "__metadata__":
                        self.idx[k] = (p, 8 + n, v)

    def __contains__(self, k):
        return k in self.idx

    def keys(self):
        return self.idx.keys()

    def get(self, name: str) -> np.ndarray:
        p, base, v = self.idx[name]
        a, b = v["data_offsets"]
        return np.memmap(p, dtype=_ST_NP[v["dtype"]], mode="r", offset=base + a, shape=tuple(v["shape"]))
