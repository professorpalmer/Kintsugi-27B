# Credits

Kintsugi-27B stands on a lot of other people's work. If we missed you, tell us and we will add you.

## The model

- **Qwen team (Alibaba)**: Qwen3.8-27B, the base weights, tokenizer, chat template and the pretrained MTP head
  (Apache-2.0). Every tensor that is not a coded projection comes from their BF16 release.
- **turboderp (exllamav3)**: the EXL3 trellis codec and MCG codebook, the `quantize_exl3` quantizer that made our fits,
  and the CUDA GEMV kernels in our engine (MIT).
- **Albert Tseng, Qingyao Sun, David Hou, Christopher De Sa (QTIP, Cornell RelaxML)**: the trellis quantization method
  that EXL3 builds on.
- **Escha Labs (Zihao Tu, Yangzihao Wang, Jiaming Pan and the escha-mlx authors)**: the Escha W2 format (EXL3 code with
  Hadamard input / output scales) and the `escha-mlx` reference, which our converter uses to compute the codec tables
  and our tests use as ground truth (Apache-2.0). Escha W2 was where this project started.
- **YourHighnessLA (L0xRE)**: the L0xRE / Escha W2 release, whose format work and original bit layout (the bit
  allocation of L0xRE-V2) were the starting point of ours, and the L0xRE runtime, whose int8 prefill design we
  re-implemented as our fast read.
- **StillDeadcode (Qwen3.8-27B quant challenge)**: the public Hessians we fitted with, and the golden BF16 set and KL
  scorer we measured with (CC BY 4.0). The Hessians come from openly licensed corpora; credit to their publishers
  (Hugging Face FineWeb-Edu, EPFL FineWeb2-HQ, Wikimedia, Common Pile, BigCode The Stack, NVIDIA When2Call and the other
  sources listed on the dataset card).

## The engine

- **The ggml authors and every llama.cpp contributor** (ggml-org, MIT): the engine is a llama.cpp fork.
- **Ajay9o9 (AjHomelab), llama.cpp-escha**: the Escha op for ggml (CPU and CUDA), native dense Qwen3.8-27B W2 support,
  the tensor-core prefill, the MTP verify speed-up (MIT). Our port starts from these commits, with his authorship kept.
- **spiritbuun, buun-llama-cpp**: the ggml port of the exllamav3 kernels that showed the GEMV split without a
  cooperative launch (MIT).
- **PrismML (Pasha Khosravi, Brian / bri-prism, Pooya Khosravi)**: the llama.cpp fork our engine descends from (MIT),
  including the DFlash and speculative-decoding fixes and the GB10 kernel work.
- **sudoingX**: the batch-invariant mode (`GGML_CUDA_BATCH_INVARIANT`) and the planar activation layout; our identity
  checks depend on it.
- **Chirag Patnaik**: DFlash2 support in llama.cpp (local convolution + candidate selector).
- **alesha-pro**: the Mirai S port to ggml in our engine base, and the control-vector projection mode.
- **Mirai Labs**: the Mirai S trellis codec whose port is in the engine base (its kernels are left out of this build).
- **usmaneth**: DFlash fix for borrowed embeddings and head with Hadamard transforms; review fixes.
- **zhaoyilun**: Hadamard inverse on the MTP token-embedding lookup for qwen35.
- **cklxx**: made the CUDA ADD + RMS_NORM + MUL fusion reachable.
- **Empressia**: `input_image` in `function_call_output` for the server.
- **lenny76**: AVX-512 single-token flash attention, PQ2_0 VNNI GEMM, Hadamard tensors kept out of CPU repack.
- **abdulaziz**: PQ2_0 x86 AVX2 / AVX-VNNI kernels and fixes.
- **Scott Guyton**: SSE2 / SSSE3 and SYCL kernels for PTQ1_0 and PQ2_0.
- **Strike3D**: AVX2 / AVX-VNNI kernels for the Q1_0 repack path.
- **Ankit Khandelwal**: vectorized HIP PTQ1_0 dot product.
- **Mark Schatza**: native PQ2_0 unpacking in the HIP MMQ loader.
- **Jason Titus**: Metal PTQ1_0 mat-vec for two to four columns.
- **Tomer Weller**: Metal language 4.0 when the tensor API is there.

## Drafting

- **syv-ai / HyperQwen**: the idea to draft out of the prompt (lookup drafting), on in this bundle.
- **Jian Chen, Yesheng Liang, Zhijian Liu (z-lab, DFlash)**: block-diffusion speculative decoding.
- **Inco AI**: the DFlash2 drafter for Qwen3.8-27B (Apache-2.0).
- **ggml-org**: the GGUF conversion of that drafter. (The DFlash profiles are in our development builds, not in this
  release bundle.)
- **Magnitude (magnitudedev)**: mixed K / V cache types on the tensor-core path. (Credited for the idea; that option is
  under test and not in this build.)

## Runtime pieces

- **NVIDIA**: the CUDA runtime and cuBLAS DLLs in `bin\` (NVIDIA CUDA EULA; not covered by our licences).
- **Bytecode Alliance (wasmtime)** and **VMware Labs (WebAssembly Language Runtimes)**, with **CPython (PSF)**: the
  sandbox of the tool layer (the code interpreter is off in this build).

## Benchmarks we report

HumanEval (OpenAI), AppWorld (Trivedi et al., Stony Brook NLP), WikiText-2 (Merity et al.).

## Kintsugi-27B itself

Cary Palmer ([professorpalmer](https://github.com/professorpalmer)): the fits, the bit map search, the clean rebuild,
the engine port and kernels, the tiered K/V cache, the two-stage head, the launcher, the tool layer, and the
measurements.

No person or company above endorses or sponsors this project.
