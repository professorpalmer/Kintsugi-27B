# Cross-architecture checks: Turing, Ampere, Blackwell and Hopper against the Ada rig

**Status:** finished (2026-11-20; Blackwell and Hopper added 2026-11-22). **Finding:** on Turing (sm_75) and Ampere (sm_86) the engine computes the same
model as on our Ada rig: mean KL within +0.02 % and +0.92 %, and full retrieval at 117k. The output is not
token-identical, and that is rounding, not wrong math.

## Question

The release engine has native code for sm_75 (RTX 20), sm_86 (RTX 30) and sm_89 (RTX 40). We develop on one RTX 4070
(sm_89). Do the sm_75 and sm_86 kernels compute the same model, inside a 12 GB memory line?

## What we did

Rented one box for each architecture (Linux). We make no speed claims from these boxes: drivers, clocks and hosts
differ from the rig.

| | Ampere (sm_86) | Turing (sm_75) |
|---|---|---|
| Card | RTX 3060 12 GB | Quadro RTX 6000 (22 GB), held to a 12,288 MiB line: the launcher's memory arithmetic, ceiling and spill guard as for a 12 GB card. Same kernels and tiered path as an RTX 2060 12 GB. |
| Engine | the release engine source one commit before the public source tar (the same kernels), built on the box for `86-real` | public release source `Kintsugi-27B-v1-engine-src-c07efbec5.tar.gz`, built on the box for `75-real` with `GGML_CUDA_MIRAI=OFF` |
| Build | no compile error | no compile error (537/537 steps) |
| Model file | an earlier fit of ours in the same format (V5bc-h32, see [bit-allocation](../bit-allocation/README.md)), sha256 checked on the box | the release file `Kintsugi-27B-12G.gguf`, downloaded from Hugging Face, sha256 checked on the box |
| Date | 2026-11-20 | 2026-11-20 |

**Gate (written before the runs):**

- **KL, the deciding check.** The golden set of the Qwen3.8-27B quant challenge, pass p1 (295 documents, 376,086
  scored tokens), the same scorer and arguments as the rig's value for the same file (K/V q8_0, all K/V in VRAM, batch
  256). Line: mean KL within **+/-2 %** of the rig and top-1 agreement within **0.3 points**.
- **Retrieval** at 117,000 tokens, thinking off: 24/24 needles and 20/20 exact code copies.
- **Memory:** 0 spill lines and 0 out-of-memory in every server log; on Turing, VRAM at or under 12,005 MiB.
- **Identity** against a rig record (50 prompts, greedy, 96 tokens). For Ampere the line was >= 45/50; for Turing,
  after the Ampere result, information only.

## Results

| Check | Line | Ampere (RTX 3060) | Turing (Quadro RTX 6000 at 12 GB) |
|---|---|---|---|
| Mean KL, box against rig | +/-2 % | 0.052932 against 0.052451: **+0.92 %** | 0.056409 against 0.056398: **+0.02 %** |
| Top-1 agreement, box against rig | 0.3 points | 91.215 % against 91.276 %: -0.06 points | 90.933 % against 90.934 %: -0.001 points |
| Retrieval at 117k | 24/24 + 20/20 | 24/24 + 20/20 | 24/24 + 20/20 |
| Spill / out-of-memory | 0 | 0 / 0 (8 server logs) | 0 / 0 (4 server logs); VRAM 11,661 MiB at load, 11,671-11,681 MiB at the end |
| K/V cells in VRAM (262,144 window) | record | 143,104 | 144,128 |
| Identity against the rig | Ampere >= 45/50; Turing info | **31/50: fail as written** | 27/50 (info) |
| **Verdict** | | **pass** (KL decides; see below) | **pass** |

- On Ampere, the running mean of KL stayed +0.85 % to +1.1 % over the rig's at every checkpoint (26 to 201
  documents), so the small rise is spread evenly, not from a few documents. The rig value came from an earlier
  development engine and the box ran the release engine, so the +0.92 % also holds any difference between those two
  engines.
- On Turing, the engine logs that the chunked GDN prefill has no BF16 tensor-core schedule for compute capability 7.5,
  so Turing takes the recurrent GDN kernel. The KL result shows that this path computes the same model.
- The first Turing box (driver 570) ran the KL check but could not load the server: see
  [old-driver](../old-driver/README.md). Retrieval, memory and identity ran on a second Turing box with driver 615.

## Why identity differs but KL matches

Identity asks for the same 96 greedy tokens. One near tie between two tokens, decided by a last-bit difference, sends
the rest of the answer down another path. Other kernel tile choices and, on Turing, another GDN kernel, round
differently.

On Ampere we probed every first difference: the box's log-probability of its own token and of the rig's token.

- In 2 of the 19 differing prompts, the first answer token differed: the prompt read itself rounds differently on
  sm_86, not only the decode tail.
- **16 of 19 first differences were near ties** (the rig's token in the box's top 2, gap under 0.5 nat). The other 3
  had the rig's token at rank 3, with gaps of 0.17, 0.69 and 0.72 nat.
- At 7 of the 16 near ties, a fresh read of the prompt plus the rig's tokens put the rig's token first on the box too.
  The order of two near-equal tokens then depends on how the tokens were batched: also rounding.

A real fault (wrong math) would move the whole distribution, and KL over 376,086 positions would show it. KL stayed
within 1 %. So KL is the test that decides across architectures; identity stays a test within one architecture.

## Decision

- Ampere and Turing cards with 12 GB or more are supported by the release engine (with a 580+ driver for the Windows
  bundle).
- The identity line of 45/50 for another architecture was a guess, and it could not separate rounding from wrong math.
  We replaced it with the KL line for cross-architecture checks.

## What we learned

1. **Identity is the right test on one architecture and the wrong test across architectures.**
2. **KL against a fixed reference separates rounding from faults**, if the line is set before the run.
3. **A capped big card is a fair stand-in for a 12 GB card** for correctness, if the launcher's memory arithmetic and
   spill guard use the 12 GB figures.

## Limits

- One box and one run for each check. No speed comparison.
- The Ampere check used an earlier fit of ours in the same format, not the release file. The kernels and the format are
  the same.
- We found no rentable RTX 2060 12 GB box with enough system RAM; the Turing result comes from a Quadro RTX 6000 held
  to 12 GB.
- No RTX 50 (Blackwell) build in v1 to v1.1.1; see the next section for v1.1.2.

## v1.1.2: Blackwell (RTX 50) and Hopper (H100), 2026-11-22

**Finding:** the v1.1.2 engine computes the same model on an RTX 5070 (sm_120) and an H100 (sm_90) as on our RTX 4070:
mean KL within +0.03 % and +0.02 %, and full retrieval at 117k and 245k.

v1.1.2 adds native code for sm_90 and sm_120 (ggml's build turns `120` into the arch-specific `120a`) and portable
`compute_90` PTX for newer cards. We rented one box for each, with the v1.1.2 Linux release engine built on the RTX 50
box (CUDA 12.8, the same binaries on both boxes), the release model file from Hugging Face (sha256 checked) and the
bundle launcher. Lines were written before the runs; they are the lines of the checks above.

| Check | Line | Blackwell: RTX 5070 12 GB | Hopper: H100 SXM 80 GB |
|---|---|---|---|
| Mean KL, box against rig (0.056398) | +/-2 % | 0.056414: **+0.03 %** | 0.056407: **+0.02 %** |
| Top-1 agreement (90.934 %) | 0.3 points | 90.928 %: -0.006 points | 90.921 %: -0.013 points |
| Retrieval at 117k | 24/24 + 20/20 | 24/24 + 20/20 | 24/24 + 20/20 |
| Retrieval at 245k | 24/24 + 20/20 | 24/24 + 20/20 | 24/24 + 20/20 |
| Tool calls t1-t4 | pass | pass | pass |
| Spill / out-of-memory / CUDA errors | 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| K/V cells in VRAM (262,144 window) | record | 129,536 (the driver keeps 453 MiB) | 262,144 (all) |
| Same tokens with PDL on and off | 50/50 | 50/50 | 50/50 |
| Identity against the rig | info | 22/50 | 20/50 |
| **Verdict** | | **pass** | **pass** |

- **PDL.** Code built for sm_90 and newer launches kernels with programmatic dependent launch (a kernel may start
  before the previous one ends and waits inside for its inputs). Our build had never used it. We checked that every
  such kernel our fork changes waits before it reads, and on both boxes the output is the same with PDL off
  (`GGML_CUDA_PDL=0`). PDL is also faster: the identity set decodes at 103 vs 76 tok/s (RTX 5070) and 128 vs 87 tok/s
  (H100) with PDL on vs off.
- **Portable code (PTX).** With `CUDA_FORCE_PTX_JIT=1` the RTX 5070 runs the `compute_90` PTX instead of its own code,
  as a B200 or a compute capability 12.1 card would. KL on a quarter of the set: 0.063993 vs 0.064009 with its own
  code (-0.03 %), top-1 89.643 vs 89.632 %. The engine then chooses its kernels for 9.0, the arch of the code it runs
  (one INFO line); before v1.1.2 the host side would have chosen for 12.0 while the device ran 9.0 code. On the H100,
  forced JIT also forces cuBLAS to compile from PTX, which its CUDA 12.8 build cannot do for sm_90, so that run stops
  at cuBLAS start; it says nothing about our code. With an engine built with the PTX only (no native code at all), the RTX 5070 compiles our code at the first
  start (89 s to the first answer, 8 s with native code), and later starts take 10 s: the driver keeps the compiled code
  in its cache (90 MiB). The output is the same in both starts (50/50), and decode speed matches the native code.
- **The Windows build.** The Windows bundle's code comes from CUDA 13.4, the probed Linux build from 12.8. We built the
  sm_120 code with the exact Windows compiler (CUDA 13.4.92) on the RTX 5070 box: KL on the quarter set 0.064003 vs
  0.064009 (-0.01 %), top-1 89.643 vs 89.632 %.
- **Hopper prompt reads** use the recurrent GDN kernel: the chunked GDN prefill (from llama.cpp #29353) is enabled only
  on the architectures it was tuned for (RTX 30 / 40 / 50 and GB10). The result is the same model; reads are slower
  than they could be.
- No speed claims from rented boxes. For the record (one run, launcher defaults, MTP): RTX 5070 decode 78 tok/s at 4k and 64 tok/s at 117k,
  prompt read 1,673 / 1,020 tok/s;
  H100 decode 91 tok/s at 4k and 79 tok/s at 117k, prompt read 3,162 / 2,246 tok/s.
