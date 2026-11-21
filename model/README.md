# The Kintsugi-27B model file

| | |
|---|---|
| File | `Kintsugi-27B-12G.gguf` |
| Size | 10,546,687,168 bytes (10.5 GB) |
| SHA-256 | `0510bae82445dc53dbc7d0d842d8b7e146b8eafce0837e39b157d411e48c8a74` |
| Download | [huggingface.co/CaryPalmer/Kintsugi-27B](https://huggingface.co/CaryPalmer/Kintsugi-27B) |
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (27 billion weights) |
| Licence | Apache-2.0 |

The same file works with every bundle version (v1 to v1.0.3).

## Format

- **Escha W2**: EXL3 trellis code (exllamav3's MCG codebook, 16 x 16 tiles) with a 128-point Hadamard transform and
  per-channel input and output scales on each coded projection.
- **400 coded projections**: in each of the 64 layers, the MLP (gate, up, down), plus the attention projections (16
  full-attention layers: q, k, v, output) or the linear-attention projections (48 GDN layers: qkv, gate, output).
  227 of them use 2 bits per weight and 173 use 3 bits: **2.469 bits per weight**, 7.51 GB of code.
- **Output head and token embedding**: Q8_0.
- **MTP block** (the model's own next-token-prediction head, used for drafting): Q4_0, with F32 norms.

## Tensors that keep full precision

- All norms (F32).
- The small linear-attention (GDN) tensors: conv1d, A, dt and norm in F32; alpha and beta in F16.
- The Hadamard input and output scales of each coded projection (F16), and the codec tables (F16 / I16).

Every tensor that is not a coded projection comes from Qwen's BF16 release.

## What can load it

Only the Kintsugi engine (see [engine/](../engine/README.md)), which is in the bundle. Stock llama.cpp, Ollama and LM
Studio cannot load it: they do not have the Escha op.

## The parameter count on Hugging Face

Hugging Face counts the elements of the tensors in the file and shows about 7B. The coded tensors store packed 2-bit
and 3-bit codes, so their element count is much smaller than the number of weights they hold. The model has 27 billion
weights.

## Chat template

The template inside the .gguf is Qwen's original. It refuses a system message after the first turn, which Claude Code
and Codex send. Bundle v1.0.2 and later carry a fixed template (`bundle/templates/kintsugi.jinja`) and use it by
default. If you run the file another way, use that template.

## How it was made

- **Bit layout:** the L0xRE-V2 bit allocation by YourHighnessLA (L0xRE), used unchanged: 2 or 3 bits for each of the
  400 projections. We built our own maps with lower KL. They failed the recall, long-answer or agent gates, so the
  release keeps this layout: see [research/bit-allocation/](../research/bit-allocation/README.md).
- **Head:** Q8_0 with one scale for each 32 weights, read in two stages by the engine (see
  [engine/](../engine/README.md)).
- **Fits:** our own, made with exllamav3's `quantize_exl3` (turboderp) on Qwen's BF16 weights, with the public
  Hessians of the Qwen3.8-27B quant challenge (StillDeadcode, CC BY 4.0).
- **Codec tables:** computed with the reference functions of escha-mlx (Escha Labs, Apache-2.0).
- **Measured** with the golden BF16 set and the KL scorer of the same quant challenge: see
  [benchmarks/](../benchmarks/README.md).

Full credits: [CREDITS.md](../CREDITS.md).
