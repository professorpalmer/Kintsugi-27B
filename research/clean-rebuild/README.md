# Clean rebuild: the release file from public inputs only

**Status:** finished (2026-11-19 to 2026-11-20). **Finding:** every tensor and every metadata key of
`Kintsugi-27B-12G.gguf` comes from our own fits or straight from Qwen's BF16 release. The rebuild changed the model
output by +0.0 % KL. A Q8_0 head with one scale per 32 weights then gave a small free gain (-0.1 % KL).

## Question

Our first model files were built on top of a third-party export of the same format (the Escha W2 export by Escha
Labs, Apache-2.0): the coded projections were ours, but the other tensors and the metadata came from that export. Can we build the file from public inputs only, and is
it the same model?

## What we did

1. **Inventory.** Every tensor of the candidate file with its type and source, every metadata key with the source of
   its value.

   | Class | Tensors | Share of the file | Source before the rebuild |
   |---|---:|---:|---|
   | Coded projections (`escha_code`, `escha_rin`, `escha_rout` of 400 projections) | 1,200 | 71.4 % | our fits |
   | `.bias` of the 400 coded projections | 400 | 0.15 % | bias bytes of the third-party export (not applied by the engine) |
   | Codec tables (`escha_lut`, `escha_dep_k2`, `escha_dep_k3`) | 3 | < 0.01 % | computed by our converter |
   | Embedding and head (Q8_0), norms, GDN small tensors | 451 | 26.2 % | the third-party export |
   | MTP block (Q4_0 matrices, F32 norms) | 15 | 2.3 % | the third-party export |

2. **Fetch Qwen's tensors.** From `Qwen/Qwen3.8-27B` at a pinned revision, by HTTP byte range: only the 466 tensors
   that are not coded projections (5.58 GiB). Each range was checked against its shard header (size, dtype) and logged
   with its sha256.
3. **Compare before we replace.** Is the export's data a plain function of Qwen's BF16, or tuned?
   - Norms, GDN small tensors and the MTP block: every value equals the f16 cast of Qwen's BF16 value. The largest
     difference is 2.98e-08, in values too small for f16.
   - Embedding and head: plain per-row int8 (scale = row max / 127) of those values: 100 % of rows equal for one
     rounding method. No tune.
   - The 400 bias vectors were not a function of Qwen's weights (they came from the export's own tuning). Our fits have
     no bias, and the engine does not apply coded-projection bias. The clean file has zeros there.
4. **Build.** The coded tensors byte for byte from our file; every other tensor from Qwen's BF16 with the steps of our
   converter; the metadata from Qwen's Hugging Face files through llama.cpp's `convert_hf_to_gguf.py --vocab-only`,
   plus our own keys (name, description, base model, format version). The chat template is Qwen's current file. Then
   read every tensor and key back against the build manifest.
5. **GPU check.** Load, 16 greedy tokens on a raw and a chat prompt, wikitext perplexity, and KL on pass p1 of the
   quant challenge's golden set (295 documents), with the same engine and arguments as the old file.

Gate (written before the comparison, build and GPU runs): mean KL not more than 0.5 % above the old file, top-1
agreement within 0.1 point, and the load check passes.

Where measured: our rig (one RTX 4070 12 GB, Windows 11), 2026-11-19.

## Results

**Rebuilt tensors:**

| Class | Made from | Bytes equal to the old file |
|---|---|---|
| Coded projections | our fits, copied | 1,200 / 1,200 |
| Bias of coded projections | zeros | 0 / 400 (was the export's bias) |
| Codec tables | computed with the escha-mlx reference functions | 3 / 3 |
| Head | Qwen BF16, per-row int8 | equal |
| Embedding | Qwen BF16, per-row int8 | equal, except 28 codes in 2 rows (values in the f16 subnormal range; the export rounded the f16 cast, we round the exact BF16 value) |
| Norms, `ssm_a`, `ssm_dt.bias`, `ssm_alpha`, `ssm_beta` | Qwen BF16 | equal, except 3 norm tensors (subnormal values now exact) |
| `ssm_conv1d` | Qwen BF16 | 0 / 48 (a few subnormal values now exact) |
| MTP block | Qwen BF16, Q4_0 by gguf-py | 15 / 15 |

Read-back: 2,069 of 2,069 tensors and 44 of 44 metadata keys pass.

**Same model (first rebuild, on an earlier map of ours):**

| File | Mean KL p1 | Top-1 | Wikitext PPL | 16 greedy tokens |
|---|---:|---:|---:|---|
| Old file | 0.051187 | 91.362 % | 6.3193 | (reference) |
| Clean rebuild | 0.051189 | 91.358 % | 6.3197 | same token ids, same chat text |

KL change +0.0 % [-0.0, +0.0]; every category within 0.00005. **Gate: pass.**

**A free gain: the Q8_0 head with one scale per 32 weights ("h32").** The per-row int8 embedding and head have one
scale per row. gguf-py's Q8_0 (one f16 scale per 32 weights) has the same size and less error against BF16:

| Tensor | Per-row int8 (relative RMS error) | Q8_0 per 32 |
|---|---:|---:|
| Embedding | 0.009009 | 0.005374 |
| Head | 0.011972 | 0.005503 |

| File | Mean KL p1 | Change | Documents lower | Top-1 |
|---|---:|---|---:|---:|
| Clean rebuild, per-row head | 0.051189 | | | 91.358 % |
| Clean rebuild, h32 head | 0.051118 | **-0.14 % [-0.2, -0.1]** | 194 / 295 | 91.381 % |

Line (adopt if KL falls with the interval under 0 and top-1 does not fall): **pass.** With the two-stage head on, the
h32 file gave the same 16 tokens and the same chat text as with the head off.

**The release file (V3ac-h32).** The same clean build on V3a (our fits on the L0xRE-V2 bit layout; see
[bit-allocation](../bit-allocation/README.md)) with the h32 head:

| Release gate | Line | Result |
|---|---|---|
| R0 read-back | fits byte-equal to V3a, bias zero, every other tensor equal to the manifest | pass |
| R1 load | the same 16 greedy tokens with the two-stage head off and on; 0 servers left | pass |
| R2 KL p1 against V3a | not more than +0.5 % (upper end of the interval), top-1 not down more than 0.1 point | **0.05640 against 0.05647: -0.1 % [-0.2, -0.1]**, top-1 90.93 % against 90.91 % |
| R3a retrieval at 117k | 24/24 + 20/20 | pass |
| R3b retrieval at 245k, bundle defaults | 24/24 + 20/20 | pass |
| R4 HumanEval 164, medium thinking | >= 158 | 161 / 164 |

`Kintsugi-27B-12G.gguf` is this build with the name "Kintsugi-27B". Its tensor directory and tensor data are equal to
V3ac-h32's (a hash of the data region), so every gate result above holds for it. Only the name key differs.

## The engine build: no local paths

The first release engine build carried local source paths in its DLLs (from assert messages): a Windows user name and
a local folder name. We rebuilt the same commit (c07efbec5) from a neutral drive path. Result: 0 user or folder strings
in its 12 files (the first build had 956), and the same output as the first build in every bundle smoke test. The
public bundles ship this neutral build. The web UI assets are given pre-built, because the UI's npm build fails under
a mapped drive.

## Decision

- Release files are clean builds: our coded projections, Qwen's other tensors, Qwen's metadata, zero bias, Q8_0 head
  and embedding with one scale per 32 weights.
- No byte in the release file comes from a third-party file. Where a rebuilt tensor is byte-equal to the export, both
  are the same function of Qwen's weights.
- The bit layout (which projections get 3 bits) is still the L0xRE-V2 layout by YourHighnessLA, credited in
  [CREDITS.md](../../CREDITS.md).

## What we learned

1. **Compare before you replace.** The comparison showed that the export's plain tensors were a plain function of
   Qwen's weights; only the bias was not. That told us exactly what the rebuild must change.
2. **A provenance check can find a free gain.** Looking at how the head was made showed a better quantizer at the same
   size.
3. **Scan the binaries too.** Compilers put source paths into the DLLs (assert messages). Only a string scan of the
   binaries shows them.

## Limits

- The h32 gain is small (0.1 %). The coded body is the larger error.
- The first clean rebuild and the h32 test used an earlier map of ours (V3s2); the release gates R0-R4 ran on the
  release map.
