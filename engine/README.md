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
| `7d9bb6296` | v1.1.7, v1.2 and v1.2.1 (Windows and Linux) | `Kintsugi-27B-v1.1.7-engine-src-7d9bb6296.tar.gz` (43,086,685 bytes) | `7e0e727c66095394a38600d32f6dc737e93c0ec98272bb5c649be763e5602039` |
| `8f12baacf` | v1.1.6 (Windows and Linux) | `Kintsugi-27B-v1.1.6-engine-src-8f12baacf.tar.gz` (43,075,805 bytes) | `c1c7812e1193f5604ee243cead1b269d49d160d72e1accb2b5564bc0d4081d1d` |
| `d512c2d77` | v1.1.5 (Windows and Linux) | `Kintsugi-27B-v1.1.5-engine-src-d512c2d77.tar.gz` (43,085,588 bytes) | `488b500da95dbffb0a9845d2661027fcb374c153a6675debb8765a78942c63fa` |
| `b9c5b427d` | v1.1.2 to v1.1.4 (Windows and Linux) | `Kintsugi-27B-v1.1.2-engine-src-b9c5b427d.tar.gz` (43,067,633 bytes) | `ebc9780d37977b077b190b172a6123e60d25cd6b439222e03a910c3d280427b0` |
| `5f393adb5` | v1.1 (Windows and Linux), v1.1.1 (Linux) | `Kintsugi-27B-v1.1-engine-src-5f393adb5.tar.gz` (37,360,262 bytes) | `ded8fd4ee8c67c1088b173bd49a21ab201a1407d1ee64b2bcd01f9d3ebcaf4a6` |
| `af70fc544` | v1.0.4 (Linux) | `Kintsugi-27B-v1.0.4-engine-src-af70fc544.tar.gz` (37,359,975 bytes) | `553c2222a156f5734be39206b19cb0bbd084d881bc473c9298d340ad807edd00` |
| `c07efbec5` | v1 to v1.0.3 (Windows) | `Kintsugi-27B-v1-engine-src-c07efbec5.tar.gz` (37,358,085 bytes) | `0e8f2a1c3c293ce83ca9892e2ba61a3b3beaaa8870ecc907f2a047092d6391b7` |

`7d9bb6296` = `8f12baacf` + (issue #17): the layer split of a model load no longer divides by 0 when every card
reports 0 bytes free (an equal split; before: `invalid vector subscript` at the drafter load); host-memory errors
name system memory with the numbers (tier host pages, `cudaMalloc` on Windows, `ggml_init`, the server after a
failed load), and the server warns after the load when little commit is left. Test hook:
`GGML_CUDA_TEST_HOST_LIMIT_MIB=N` makes pinned host pages past N MiB fail as out of memory.

`8f12baacf` = `d512c2d77` + one server option, `--reasoning-max-tokens-honor N` (the launchers pass 4096): with thinking
on, a client `max_tokens` of N or more (and below `--reasoning-max-tokens-floor`) is kept, and the reasoning budget of
that request becomes `max_tokens` minus 1,024. Smaller caps are still raised to the floor. New `test-chat` case. The
GPU code is unchanged. In the Linux tarball of v1.1.6 the binaries have `RUNPATH $ORIGIN` (rewritten with patchelf; a
build of your own gets the same with `-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH='$ORIGIN'`).

`d512c2d77` = `b9c5b427d` + tool-call fixes in `common/chat.cpp` (Qwen3-Coder tool format, issue #15): with
`tool_choice` "required" the text before the call is at most 1,024 characters, then only the call is allowed; with
"none" the tool definitions are not shown to the model (as llama.cpp PR #30434 by frozenblade1224), the grammar
blocks the start of a call, and the answer can not be empty. New `test-chat` cases on the Kintsugi-27B template. The
GPU code is unchanged.

`b9c5b427d` = `5f393adb5` + two changes: the grammar guard of llama.cpp #32381 (Aldehir Rojas: an anchored tool-call
trigger pattern must not fall back to a search; Kintsugi's own trigger is a fixed tag, so its tool calls do not
change), and a check at start for cards that run the portable code (below). The source asset also holds the built web
chat page (`tools/ui/dist`), so the build needs no npm and no download.

`5f393adb5` = `af70fc544` + one fix: the spill check at load writes its warning only when the GPU memory is at its
limit (before, pinned system memory of the tiered cache could give a false "spill at load").

## Build

The release build leaves out the Mirai S kernels that are also in this source (`GGML_CUDA_MIRAI=OFF`; the loader then
refuses Mirai S files with a clear message). Release flags:

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75-real;86-real;89-real;90;120-real" -DGGML_CUDA_MIRAI=OFF -DLLAMA_BUILD_UI=OFF -DGGML_CUDA_GRAPHS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF -DGGML_RPC=OFF -DGGML_BLAS=OFF -DGGML_NATIVE=OFF -DGGML_CCACHE=OFF
cmake --build build --target llama-server llama-perplexity test-backend-ops
```

- **Linux:** also give `-DLLAMA_USE_PREBUILT_UI=OFF`. Without it, the `llama-ui-assets` step tries to download the
  prebuilt web UI and the build stops there. Also give `-DGGML_CUDA_NCCL=OFF` (from v1.1.1): with NCCL on, the engine
  needs `libnccl.so.2` at run time, which a machine with only the NVIDIA driver does not have.
- **Architectures (v1.1.2):** native code for `75` (RTX 20), `86` (RTX 30), `89` (RTX 40), `90` (H100 / H200) and `120`
  (RTX 50; ggml's CMake turns `120` into `120a`, which runs on compute capability 12.0 only), plus PTX for `compute_90`
  (the `90` entry without a suffix gives both). A card with no native code (for example B200, or 12.1) runs that PTX:
  the driver compiles it at the first start and keeps it in its cache (`CUDA_CACHE_MAXSIZE`; the launchers set 4 GiB
  for such cards). The engine then picks its kernels for compute capability 9.0, the arch of the code it runs, and
  writes one INFO line. The PTX is `compute_90`, not `compute_120`, because a `120-virtual` entry would also become
  arch-specific. CUDA loads only the code of your card: the other architectures cost no VRAM or speed, only file size.
- **One GPU family only:** set `CMAKE_CUDA_ARCHITECTURES` to it, for example `75-real` (RTX 20), `86-real` (RTX 30),
  `89-real` (RTX 40), `90-real` (H100) or `120-real` (RTX 50).
- **CUDA toolkit:** 12.2 or newer; 12.8 or newer for RTX 50. The Windows bundle uses CUDA 13, the Linux bundle 12.8.
- **AVX CPU library (v1.2.1, #21):** with `GGML_NATIVE=OFF` the CPU library is built for AVX2 + FMA + BMI2 + F16C.
  The bundles also carry a second build of the same source (target `ggml-cpu` only) for CPUs with AVX + F16C and no
  AVX2. Linux (GCC): add `-DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_BMI2=OFF` (AVX, F16C and SSE4.2 stay on); the file is
  `bin/cpu-avx/libggml-cpu.so.0`. Windows (MSVC): add `-DGGML_AVX=ON -DGGML_AVX2=OFF -DGGML_BMI2=OFF -DGGML_AVX512=OFF
  -DGGML_AVX_VNNI=OFF` and set `CFLAGS` and `CXXFLAGS` to `/D__F16C__ /DGGML_F16C /d2Qvec-sse2only` at the first
  configure (MSVC has no F16C option; without `/d2Qvec-sse2only` the MSVC 14.44 auto-vectorizer put 4 AVX2
  instructions into the `/arch:AVX` build); the file is `bin\ggml-cpu-avx.dll`.

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
