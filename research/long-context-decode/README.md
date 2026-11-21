# Long-context decode on a 12 GB card: the tiered K/V cache, and what did not help

**Status:** finished (2026-11-18 to 2026-11-20). **Finding:** past about 144k tokens, decode speed is set by the PCIe
stream of the K/V cells in system RAM. Five attempts to make it faster or more exact did not ship: a sparse V read,
two forms of a better copy pipeline, a 3-bit K cache, and a 2-bit first stage for the head. One kernel finding (fp16
V sums in attention) was real and fixed, but stays off by default.

Where measured: our rig (one RTX 4070 12 GB, Windows 11, memory overclock profile as in
[benchmarks](../../benchmarks/README.md)), development engine builds. Decode tables use the deep profile (the bundle
defaults: window 262,144, K/V q4_0, MTP draft 3 + lookup, two-stage head), a fresh server for each depth, and the mean
of 3 prompts (code, prose, bash).

## Question

How does the full 262,144-token window fit a 12 GB card, why is decode slower past about 144k tokens, and what can
make it faster without a loss of quality?

## The tiered K/V cache

- Each K/V tensor is one CUDA address range. The first N cells are VRAM pages; the other cells are pinned system-RAM
  pages (2 MiB pages). The launcher computes N from the free VRAM. On the rig with the bundle defaults, N = 143,872
  cells.
- One cell of K + V costs 18,432 bytes over the 16 full-attention layers (q4_0). The 48 linear-attention (GDN) layers
  keep a fixed-size state, not a K/V cache.
- Up to the VRAM line, all K/V reads are in VRAM. Past it, each decode step streams the cold cells of all 16 attention
  layers over PCIe. A prefetch pipeline copies the next layer's cold cells while the GPU computes.

### A fault we found on the way (fixed before the release)

On 11-18, the KL scorer gave wrong numbers past the VRAM line (one 64,321-token document: KL 0.72 with a 64,512-token
context, 0.018 with all cells in VRAM). Cause: when the K tensor is not a whole number of 2 MiB pages, the last K page
also holds the first V cells, so V has two host runs. On the prompt-read path, the write-back to host pages used only
the first run, so the V rows of the real tail kept old bytes. It happens only when the context is not a multiple of
32,768 cells (q4_0 / q8_0), so a server with `-c 262144` did not have it; the scorer did. Fix: one write-back for each
host run. A new engine test (`test-kv-tier`) showed 669 of 10,240 V rows wrong before the fix and all bytes equal after
it. A server probe at context 61,440 gave 0/11 equal top-10 log-probs before the fix and 11/11 after.

## Why decode slows down past the VRAM line

The step model (from measured numbers, deep profile):

- Up to the line a step is compute-bound: about 42 ms + 0.085 ms per 1,000 cells, for about 3.1 tokens per step.
- Past the line, each step streams (depth - 143,872) x 18,432 bytes at the measured 24 GB/s. The step takes
  max(compute, stream + 26 ms); the 26 ms are the part the pipeline did not hide.
- At 245k: 101k cold cells = 1.87 GB = 78 ms per step; step = max(63, 104) = 104 ms, so 30 tok/s. Measured: 30.1.
  Three quarters of the step is the cold stream.

Measured decode (deep profile, an earlier fit of ours in the same format, 11-19):

| Depth | 0 | 64k | 117k | 200k | 245k |
|---|---:|---:|---:|---:|---:|
| Decode, tok/s | 74.6 | 64.3 | 60.3 | 44.1 | 30.5 |

A profile of one 245k step (11-20) found where the 26 ms go: one staging buffer per K and V is shared by all layers.
The copy of layer i+1 waits until attention op i has read that buffer, so the copy engine is idle during every
attention op. Per graph of 16 attention ops: 76.2 ms of copies (4.8 ms each at 24.4 GB/s), 44.4 ms of stalls on the
main stream, 17.5 ms of copy-engine idle time; step 97.4 ms.

## What we tried

| # | Attempt | Idea | Line (written before) | Result | Status |
|---|---|---|---|---|---|
| 1 | **F1: fp32 V sum** | The attention kernels sum V in fp16. Sum in fp32. | Adopt only if it turns a retrieval failure into a pass | Real kernel error, removed; no effect on retrieval or NLL; slower | **Off by default** (switch kept) |
| 2 | **S1 / S1v2: sparse V read** | Read the cold V rows only where the attention weight is large; a tile mean for the rest | Decode >= 1.25x dense at 200k and 245k | 0.66x (200k), 0.88x (245k); S1v2 kernel 2.28 ms against a 2.0 ms line | **Closed** |
| 3 | **Option 1: hide the cold stream** | Small rotating head buffers, so the copy engine works during each attention op | >= 1.20x at 245k, token-identical | 1.188x; fix round 1.151x | **Closed** |
| 4 | **Option 2: K q3_0 + V q4_0** | Fewer bytes per cell, so a longer VRAM line and a shorter cold stream | Default: KL <= +2.0 % overall, chat <= +3.0 % | +3.8 % [+3.3, +4.3] overall, chat +3.7 % | **Not a default** (opt-in at most; parked) |
| 5 | **E4: 2-bit first stage of the head** | Free about 600-900 MiB of VRAM for more K/V cells | KL within +0.2 %; decode not more than 3 % slower | own layout: KL +11.7 % (N 512); stock Q2_K: KL +0.12 %, decode -6.7 % / -7.4 % | **Closed** (Q4_0 first stage stays) |

### 1. F1: fp16 V sums lose part of wide heads at depth

**The finding (11-19).** A study of the attention outputs (one 245k agent trace, the real cache rows, against an
exact fp64 softmax) found that the prompt-read attention kernel keeps its V sum in fp16. In the first (layer 3) and the
last (layer 63) attention layers, wide heads lose part of their output, and the loss grows with depth:

| Layer 3 at depth | 30k | 100k | 200k | 245k |
|---|---:|---:|---:|---:|
| Error energy of the layer output | 0.0016 | 0.0049 | 0.0298 | **0.0505** |
| q4_0's own error on the same output | 0.0017 | 0.0020 | 0.0006 | 0.0008 |
| Output length against exact | 0.97 | 0.96 | 0.88 | **0.83** |

At 245k the layer-3 output is **17 % too short** (single heads that spread over 5,000-26,000 cells lose 40 %): 60
times the error that the q4_0 cache adds there. An offline fp16 sum reproduced the engine rows to 5e-7, so the cause
is the kernel's fp16 accumulation, not the cache. No K/V precision test had seen it, because every arm shared it.

**The fix.** fp32 V accumulation behind the switch `GGML_CUDA_FA_VKQ_F32=1`. Its first build failed the op test
against the CPU (infinite outputs, then an illegal memory access: a row-index fault in the new tile path); the op test
caught a fault that would have given wrong attention. After the fix: op tests pass, batch identity passes, and the
layer-3 error at 245k falls from 5.1e-2 to 2.3e-6 (output length 0.9988-0.9999).

**What it changes for the model:**

| Test | fp16 V sum (default) | fp32 V sum (F1) |
|---|---|---|
| KL against the golden reference, 7 long documents, all bands | (reference) | +0.82 % [+0.43, +1.13] (worse) |
| The model's NLL of the true next token, 128-256k band | (reference) | -0.00024 [-0.00066, +0.00007] (no change) |
| Retrieval at 245k, release-candidate weights, deep profile | 24/24 + 20/20 | 24/24 + 20/20 |
| Retrieval at 117k on the weights that failed it (V3s2, see [bit-allocation](../bit-allocation/README.md)) | 23/24 + 19/20 | 23/24 + 19/20 (same two items wrong) |
| Cold read speed at 117k / 200k (a profile with fewer VRAM cells) | (reference) | -13.1 % / -17.0 % |
| Decode at 117k (the same profile) | (reference) | -7.5 % |

- KL gets *worse* with the exact sum, even at 0-32k where the kernel error is tiny. The likely reason: the golden
  reference was made with an engine that also sums V in fp16 (the dataset's own scorer runs llama.cpp with flash
  attention on). So KL against this reference cannot judge this fix. Comparisons between weight sets on one engine
  stay valid.
- In the model's own terms the effect is small: the true-token NLL moves by at most 0.1 % in every band.
- **Decision (11-19):** F1 stays off by default. The switch stays in the engine for other models with wide heads.
- To be clear: the 17 % is an output-length loss of one layer, not a recall loss. Needles and copies at 245k pass
  with fp16 sums and with fp32 sums.

### 2. S1 / S1v2: sparse V read past the VRAM line

- **Study (offline, one 245k trace):** with one weight threshold per layer, a tile-mean tail and the newest cells hot,
  15 of 16 layers need at most 6.8 % of the cold V rows at an output error of at most 25 % of q4_0's own. Layer 3 needed
  10.4-12.5 % (kept whole in VRAM). Verdict: near GO, as an opt-in deep profile.
- **Memory:** to score every cold cell exactly, all K must be in VRAM (2,304 MiB at 262,144). That left room for only
  about 12,000 hot V cells, not the study's 53,248. S1 did not fit the profile with the DFlash drafter at all.
- **S1 (11-20):** the sparse path ran (16 layers, 0 errors, 0 spill), but decode was 31.7 against 48.2 tok/s at 200k
  (0.66x) and 26.4 against 30.1 at 245k (0.88x). Line 1.25x: **fail**. The sparse step grows with its own passes over
  all of K (up to 18 passes per layer per step for 33-row verify batches).
- **S1v2:** one K pass, a hot region, a size cutoff. The per-op kernel time at 245k (5 queries) was 2.28 ms against a
  2.0 ms line, in three separate runs (within 1.5 % of each other): **fail**, and S1 closed (as written before the
  run). The stock dense kernel with all K and V in VRAM takes 1.07 ms on the same shape. To go on would need the score
  pass on tensor cores and a select without per-tile barriers.

### 3. Option 1: hide the cold stream

- Plan: in the staging buffer, map the first chunk of each layer's cold K and V to one of two small rotating buffers,
  so the copy of layer i+2 can run while attention op i+1 computes. Predicted 1.23x at 245k.
- Gate: identity 50/50; prose decode at 245k >= 1.20x (2 runs per arm, fresh servers); no regression over 3 % at 64k
  and 117k; 0 spill.
- Result: identity 50/50; 245k prose 30.10 -> 35.75 tok/s = **1.188x** (runs within 0.3 %): fail by 0.012. 64k and
  117k: no regression. A copy-heavy case gained only 1.012x: that case is compute-bound (lookup drafts), so the line
  for it was a design error of ours; the fix round used "no regression" for it.
- One fix round (heads + a prefetch across graphs): **1.151x**, worse than the first form. **Closed.** The +19 % of
  the first form is recorded for a later design.

### 4. Option 2: a 3-bit K cache

- The step model predicts +25 % at 245k for K q3_0 + V q4_0 (16,384 bytes per cell; VRAM line about 161,856).
- KL on the golden set, release weights: q4_0 + q4_0 (shipped) 0.05742, top-1 90.88 %; K q3_0 + V q4_0 0.05959,
  top-1 90.63 %: **+3.8 % [+3.3, +4.3]**, lower on only 42 of 295 documents. Chat +3.7 %; every category worse.
- Default lines (+2.0 % overall, +3.0 % chat) not met: K q3_0 can only be an opt-in long-context mode. Its depth and
  retrieval checks did not run; the work is parked. (Idea credit: Magnitude, mixed K/V cache types.)
- For scale, from an earlier K/V study on long documents: q4_0 + q4_0 costs +2.8 % KL against q8_0, and q3_0 + q3_0
  costs +13.0 %.

### 5. E4: a 2-bit first stage for the two-stage head

The two-stage head reads all vocabulary rows from a small copy in VRAM, then scores the best candidates again with
the exact Q8_0 rows in system RAM. A 2-bit first stage would free more VRAM for K/V cells.

- Own layout ("h2", one scale per 32 weights, 379 MiB): batch-exact, but KL +11.7 % at N = 512 candidates (line
  +0.2 %), and decode -6.5 %.
- The stock Q2_K layout, one kernel geometry for every batch size: KL +0.12 % at N = 1,024 (pass), but decode -6.7 %
  (version 1) and -7.4 % (version 2) against the exact head (line: at most -3 %). The Q2_K dot product costs more than
  the bytes it saves on this card.
- **Closed:** the first stage stays Q4_0 (-606 MiB of VRAM, decode about -1 %).

## Decision

- Shipped: the tiered K/V cache with q4_0 K/V and the VRAM line from the launcher; the write-back fix; the two-stage
  head with a Q4_0 first stage.
- Off or closed: F1 (switch kept, off), S1 / S1v2, option 1, E4. Option 2 is not a default.

## What we learned

1. **Past the VRAM line, bytes over PCIe set the speed.** The step model predicted 30 tok/s at 245k and we measured
   30.1. A lever helps only if it sends fewer bytes or hides more of them.
2. **Fewer bytes is not enough if the kernel work grows.** The sparse read saved bytes and lost time in its own
   passes over K.
3. **A shared reference can hide a shared fault.** Every K/V arm and the golden reference had the fp16 V sum. Only a
   reference-free check (an exact softmax on the real cache rows) found it.
4. **A large error in one layer can be small for the model.** The 17 % output loss of layer 3 moved the true-token
   NLL by less than 0.1 % and did not change retrieval.
5. **Op tests pay.** The first F1 build would have given wrong attention; the op test against the CPU caught it.

## Limits

- One card (RTX 4070, host reads at the measured 24 GB/s). A card with another bus or more VRAM moves the line.
- The F1 and sparse-V studies used one long agent trace for the kernel errors and 7 long public documents for KL and
  NLL.
- Each speed number is one run per arm unless the table says otherwise; our run-to-run noise is up to about 5 %.
