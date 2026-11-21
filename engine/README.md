# The Kintsugi-27B engine

The engine is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp). Only this engine can load the
Kintsugi-27B model file. It adds:

- **Escha W2 / EXL3 trellis weights**: the Escha op for ggml (CPU and CUDA), the exllamav3 CUDA GEMV kernels for
  decode, a decode-once-then-GEMM prefill on the tensor cores, and a fast int8 read of long prompts
  (`GGML_ESCHA_PREFILL=i8`, on in the bundle; `recon` = the exact read).
- **A tiered K/V cache** (`--kv-vram-cells N`): the first N cells of each K/V tensor are in VRAM and the other cells
  are in pinned system RAM, in one CUDA address range. This is how the full 262,144-token context fits a 12 GB card.
- **A two-stage output head** (`LLAMA_HEAD2=N`): a small q4_0 copy of the head in VRAM reads all rows; the exact Q8_0
  rows in system RAM score the N best candidates again.
- **Drafting**: the model's own MTP head at every depth, lookup drafting from the prompt, and DFlash2 block drafting
  with a separate drafter file (`--spec-type draft-dflash -md <file>`; the bundle uses it when the file is there).
- **Batch-invariant kernels** (`GGML_CUDA_BATCH_INVARIANT=1`): the output for a token does not change with the batch
  it runs in.
- **Saved chats**: slot saves in the background, and delta saves (each save holds only the new part).

## Source

The source of each release build is a release asset. Licence: MIT, the same as llama.cpp (`LICENSE` in the tar).

| Engine | Bundles | Source asset | sha256 |
|---|---|---|---|
| `5f393adb5` | v1.1 (Windows and Linux) | `Kintsugi-27B-v1.1-engine-src-5f393adb5.tar.gz` (37,360,262 bytes) | `ded8fd4ee8c67c1088b173bd49a21ab201a1407d1ee64b2bcd01f9d3ebcaf4a6` |
| `af70fc544` | v1.0.4 (Linux) | `Kintsugi-27B-v1.0.4-engine-src-af70fc544.tar.gz` (37,359,975 bytes) | `553c2222a156f5734be39206b19cb0bbd084d881bc473c9298d340ad807edd00` |
| `c07efbec5` | v1 to v1.0.3 (Windows) | `Kintsugi-27B-v1-engine-src-c07efbec5.tar.gz` (37,358,085 bytes) | `0e8f2a1c3c293ce83ca9892e2ba61a3b3beaaa8870ecc907f2a047092d6391b7` |

`5f393adb5` = `af70fc544` + one fix: the spill check at load writes its warning only when the GPU memory is at its
limit (before, pinned system memory of the tiered cache could give a false "spill at load").

## Build

The release build leaves out the Mirai S kernels that are also in this source (`GGML_CUDA_MIRAI=OFF`; the loader then
refuses Mirai S files with a clear message). Release flags:

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75-real;86-real;89-real" -DGGML_CUDA_MIRAI=OFF -DLLAMA_BUILD_UI=OFF -DGGML_CUDA_GRAPHS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF -DGGML_RPC=OFF -DGGML_BLAS=OFF -DGGML_NATIVE=OFF -DGGML_CCACHE=OFF
cmake --build build --target llama-server llama-perplexity test-backend-ops
```

- **Linux:** also give `-DLLAMA_USE_PREBUILT_UI=OFF`. Without it, the `llama-ui-assets` step tries to download the
  prebuilt web UI and the build stops there.
- **One GPU family only:** set `CMAKE_CUDA_ARCHITECTURES` to it, for example `75-real` (RTX 20), `86-real` (RTX 30)
  or `89-real` (RTX 40).
- **CUDA toolkit:** 12.2 or newer. The Windows bundle uses CUDA 13.

## NVIDIA driver

- **Windows bundle:** it carries the CUDA 13 runtime, which needs driver 580 or newer. From v1.0.3, the launcher stops
  with a message on an older driver.
- **Linux bundle:** it carries the CUDA 12.8 runtime and works with driver 570 or newer (see the fallback below).
- **Your own build with CUDA 12.x:** CUDA 12.x itself runs on older drivers. But drivers older than the 580 family
  (for example 570) refuse the default type of pinned system memory that the tiered K/V cache uses. Thus the c07efbec5
  engine does not load on them.
- **From engine `af70fc544` (Linux v1.0.4, all v1.1 bundles):** a fallback. If the driver refuses the default host
  memory type, the engine uses the host NUMA type and writes one WARN line. If the driver refuses both, the load stops with "update the NVIDIA driver
  to 580 or newer". We checked it on driver 570.133 (Linux, RTX 2080 Ti): the model loads, and tool calls work. Test
  switch: `GGML_CUDA_HOSTMEM_FORCE=host|numa|pinned`.

## Where it comes from

- **llama.cpp** by the ggml authors and all its contributors (MIT).
- **PrismML's fork** ([PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp), MIT) is the direct base.
- **The Mirai S port** by alesha-pro of Mirai Labs' trellis codec is in this source; the Kintsugi build leaves its
  kernels out.
- **Ajay9o9 (llama.cpp-escha)**: his commits are the base of the Escha op, with his authorship kept.
- **turboderp (exllamav3)**: the EXL3 codec and the CUDA GEMV kernels (MIT; the licence is in
  `ggml/src/ggml-cuda/exl3/`).

The full list is in [CREDITS.md](../CREDITS.md).
