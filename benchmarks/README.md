# Benchmarks: the numbers on the card

All results below are for Kintsugi-27B on our own engine. The card's fidelity, decode and restore rows were measured
again on 2026-11-21 on the exact v1.0.3 bundle (zip sha256 `ac7b3b54...`) with the release file, at the shipped
settings. The model file of the release is `Kintsugi-27B-12G.gguf` (sha256
`0510bae82445dc53dbc7d0d842d8b7e146b8eafce0837e39b157d411e48c8a74`).

**Rig:** one RTX 4070 12 GB, Windows 11, display on the integrated GPU, NVIDIA driver 580+ family, MSI Afterburner
memory profile (`nvidia-smi clocks.mem` 11,751 MHz under load).

**Launcher settings ("deep" profile = the bundle's defaults):** window 262,144 tokens, K/V cache q4_0, 143,872 K/V
cells in VRAM, two-stage head, MTP draft 3 + lookup drafting, int8 fast read, micro-batch 1,024.

| Card number | Test and set | Settings | Card | Date | Result |
|---|---|---|---|---|---|
| HumanEval 161/164 | HumanEval (OpenAI), all 164 problems; pass@1, one answer for each problem | Plain server (no layer), thinking on with effort "medium", up to 24,576 output tokens, the server's default sampling from the model file (temperature 1.0, top-k 20, top-p 0.95) | RTX 4070 (rig) | 2026-11-20 | **161 / 164 (98.2 %)** |
| Recall at 245k: 24/24 + 20/20 | 24 needle questions (a fact planted in a long text) and 20 exact copies of a code function, each at a depth of 245,000 tokens | Thinking off, launcher defaults | RTX 4070 (rig) | 2026-11-20 | **24 / 24 needles, 20 / 20 copies** |
| "also perfect at 117k" | The same 24 + 20 at 117,000 tokens | Another launcher profile (131,072-token window, a DFlash drafter; not in the bundle), thinking off | RTX 4070 (rig) | 2026-11-20 | **24 / 24, 20 / 20** |
| Fidelity to BF16: 90.8 %, KL 0.058, PPL within 3.5 % | The golden BF16 set of the Qwen3.8-27B quant challenge (StillDeadcode, CC BY 4.0): 295 public documents, 376,086 scored tokens; its KL scorer, ported to our engine | **Shipped settings** (the v1.0.3 launcher's engine settings): K/V q4_0, two-stage head, int8 fast read, micro-batch 1,024. The same result with the tiered cache forced to system RAM past 4,096 tokens. For reference, with q8_0 K/V all in VRAM and the exact read: top-1 90.934 %, KL 0.056398, ratio 1.0332. | RTX 4070 (rig) | 2026-11-21 | **same top token 90.787 %, mean KL 0.057864, perplexity ratio 1.0345** |
| AppWorld 151/168 | AppWorld test_normal (168 tasks), simplified ReAct code agent | Temperature 1.0, effort "medium", 50 steps, seeds 100 and 101; plain server, launcher defaults | Rented RTX 4070 boxes (Linux; quality only, no speed) | 2026-11-20 | **151 / 168 on seed 100 and 151 / 168 on seed 101** (89.9 %). A rerun of seed 101 on another box gave the same 168 outcomes. Note: seeds 100 and 101 were also used to choose the bit map, so this score is biased upward; use the fresh-seed row below. |
| AppWorld vs full-precision BF16 (fresh seeds) | AppWorld test_normal (168 tasks), same agent, on seeds 102 and 103, which were never used to choose anything; 336 task-runs per arm; paired sign test on the tasks where two arms differ | All arms in our engine (commit `af70fc544`), temperature 1.0, effort "medium", 50 steps. **BF16:** Qwen3.8-27B BF16 on an A100, f16 K/V, no drafting. **Kintsugi weights only:** the same engine and settings as BF16. **Kintsugi as shipped:** RTX 4070, release binaries, q4_0 tiered K/V, two-stage head, MTP drafting | A100 (BF16 arm, Kintsugi weights-only arm) and RTX 4070 (shipped arm), rented, Linux | 2026-11-21 | **BF16 293 / 336** (142 + 151), **Kintsugi weights only 295 / 336** (146 + 149), **Kintsugi as shipped 294 / 336** (143 + 151). No significant difference: weights only vs BF16 23 / 21 discordant (p = 0.88); shipped vs BF16 25 / 24 (p = 1.0); shipped vs weights only 19 / 20 (p = 1.0). The same BF16 arm scored 281 on seeds 100 and 101, so seed-to-seed noise is about +/- 10 task-runs. |
| BFCL multi-turn vs full-precision BF16: 505 vs 461 of 800 | Berkeley Function Calling Leaderboard v3 multi-turn: base, missing function, missing parameter, long context (200 entries each); native function calling, official scorer (see [bfcl/](bfcl/README.md), with the per-entry results and the script) | **Kintsugi:** v1.1.7 release launcher, launcher defaults. **BF16:** Qwen3.8-27B BF16 in our engine (`af70fc544`), f16 K/V, no drafting. Both: effort "medium", the release template and layer, temperature 0.001 | Rented RTX 3090 (Kintsugi) and A100 SXM4 80 GB (BF16) | 2026-11-25 | **Kintsugi 505 / 800, BF16 461 / 800**: +44, higher in all 4 categories; by scenario p = 0.0011, 95 % interval +19 to +69. A second Kintsugi run: 504. Kintsugi was not run in the BF16 setup, so a setup effect is not ruled out. |
| Decode 67 / 59 / 49 / 31 tok/s | Decode speed at 64k / 117k / 200k / 245k tokens of context: 400 tokens on each of 3 prompts (code, prose, bash), mean of the 3; a fresh server for each depth; 2 runs | The v1.0.3 launcher's defaults (MTP draft 3 + lookup), plain server (layer off) | RTX 4070 (rig) | 2026-11-21 | **Mean of 2 runs: 66.6 / 59.1 / 48.5 / 31.4 tok/s** (runs: 66.3 + 66.8, 55.3 + 62.8, 48.6 + 48.4, 31.5 + 31.3). At 117k both runs accepted the same draft tokens; run 1's steps were slower for an unknown reason, so its mean is a low estimate. Past 143,872 tokens, the K/V cells are in system RAM and stream over PCIe at each step: that is why 200k and 245k are slower. Cold read of a 245k prompt: 589 tok/s (414 s). |
| Decode with the DFlash2 drafter (v1.1): 93 / 89 / 80 / 30 tok/s | The same decode test at 4k / 64k / 117k / 245k tokens; 2 runs | The v1.1 launcher with the optional drafter file (DFlash2 draft 7, drafter micro-batch 64), plain server; 97,280 K/V cells in VRAM in these runs | RTX 4070 (rig) | 2026-11-21 | **Mean of 2 runs: 92.7 / 88.9 / 79.6 / 30.2 tok/s**; MTP + lookup on the same runs 72.6 / 66.6 / 59.1 / 31.4 (+28 / +34 / +35 / -4 %). The released launcher keeps more room for the GPU driver and so 89,600 cells in VRAM on this card (80,640 on a Linux RTX 4070): one run at 117k gave 78.6 tok/s against 63.8 for MTP (+23 %); 4k and 64k are not affected (all in VRAM). Identity: the same 50 / 50 greedy outputs as without drafting; needles and copies 24 / 24 + 20 / 20 at 117k and 245k; AppWorld 150 and 151 / 168 (seeds 100, 101). |
| Long conversations: 2.8 s | Restore a saved 245,935-token chat after a restart of the server and the layer, then the next turn | The v1.0.3 bundle, layer on, launcher defaults | RTX 4070 (rig) | 2026-11-21 | **Restore 2.81 s; first token of the next answer 9.8 s** (a cold read of 245k takes about 414 s); the same answer as before the restart |
| Works with: Claude Code, Codex, Pi, OpenCode | 6 small coding tasks in a small Python repository (see [harnesses/](../harnesses/README.md)) | Bundle v1.0.2 template; through the saved-chat layer and with the plain server | RTX 4070 (rig) | 2026-11-20 | **5 / 6 in each harness, on both paths**; no malformed tool calls, no leaked tags, no loops |
| Voxel pagoda (Killy's test): 3 / 6 first try, 6 / 6 after repair | One HTML page with a five-tier voxel pagoda in three.js, checked in a headless browser; up to 3 repair turns with the probe's facts (see [pagoda/](pagoda/README.md), with all final pages and screenshots) | v1.1 Windows bundle, launcher defaults, effort medium; 6 samples | RTX 4070 (rig) | 2026-11-21 | **Clean on the first try 3 / 6, after one repair 5 / 6, after three 6 / 6**; 0 console errors; by eye 4 of 6 are clear five-tier pagodas |
| Standing rules at depth: no decay to 237k | 5 house rules in the system prompt (2 shown in the trace, 3 only in the system prompt); 24 tasks that tempt a rule, after 0 to 237,229 tokens of the model's own coding-agent trace; the first response scored by script; 2 seeds, 240 responses per arm; Fisher lines written before the run (see [persistence-at-depth](../research/persistence-at-depth/README.md)) | **Kintsugi:** v1.1 Windows bundle, launcher defaults, layer off, effort medium. **BF16:** Qwen3.8-27B BF16 in our engine (`af70fc544`), f16 K/V, no drafting; the same sampling defaults and template | RTX 4070 (rig) for Kintsugi; rented A100 SXM4 80 GB for BF16 | 2026-11-22 | **Kintsugi vs BF16:** R1 81/81 vs 81/81, R3 120/120 vs 120/120, R2 56/60 vs 59/60, R4 44/44 vs 47/47, R5 59/60 vs 59/60. No decay with depth in either arm; Kintsugi not worse on any rule or depth (smallest p = 0.48). Read by hand: 2 real R2 breaks for Kintsugi (at 32k and 200k), 0 for BF16; the other fails are scorer-strict. |

## Checks on each GPU architecture

These checks show that the kernels compute the same model on each architecture. Speeds from rented boxes are not
comparable with the rig (other drivers, clocks and hosts). We make no speed claims from them.

| Architecture | Card | What we checked | Result |
|---|---|---|---|
| Ada (sm_89) | RTX 4070 12 GB (rig) | Everything in the table above | See above |
| Ampere (sm_86) | RTX 3060 12 GB (rented box) | KL on the same golden set against the rig, on the same model file (an earlier fit of ours in the same format); needles at 117k | Mean KL +0.92 % against the rig (line: within 2 %), same top token 91.215 % against 91.276 %; needles 24 / 24 |
| Ampere (sm_86) | RTX 3060 12 GB, Debian 13, driver 610.43 (a user's own machine: Doğukan Ürker, [#15](https://github.com/professorpalmer/Kintsugi-27B/issues/15)) | The full 262,144-token window at the launcher defaults (v1.0.4; v1.1.1 and v1.1.4 rechecked); decode by depth; HumanEval; his own recall set; tool calls | 11.6 GB of VRAM at every depth, no spill or out-of-memory; decode (thinking off) 41.5 tok/s short, 33.2 at 64k, 31.6 at 117k, 25.6 at 200k, 16.8 at 245k; HumanEval 161 / 160 / 161 of 164 in three runs; all needles and code copies correct at 117k, 245k and 258k; the two tool_choice problems he found are fixed in v1.1.5 |
| Turing (sm_75) | Quadro RTX 6000, held to a 12 GB limit (the same kernels and memory path as an RTX 2060 12 GB; rented boxes) | KL of the release file; needles and code copies at 117k; no memory spill at the 12 GB limit | Mean KL 0.056409 against the rig's 0.056398 (+0.02 %), same top token 90.933 % against 90.934 %; 24 / 24 needles, 20 / 20 copies; 0 spill, 0 out-of-memory |

On Turing and Ampere, the output is not token-identical to the RTX 4070 in every case, because rounding differs between
architectures. KL is the test that decides.

## Head-to-head: Inkling 2 Mini on the same 12 GB card

![Kintsugi-27B vs Inkling 2 Mini on one 12 GB card](kintsugi-vs-inkling.png)

We ran Kintsugi-27B and Inkling 2 Mini on the same type of card, with the same harness, on 2026-11-20. All runs used
rented RTX 4070-class 12 GB boxes (Linux).

**The two setups**

| | Kintsugi-27B | Inkling 2 Mini |
|---|---|---|
| Model | Qwen3.8-27B at 2.47 bits per weight (this release) | Thinking Machines, 26.4B looped MoE, Apache-2.0 |
| File | `Kintsugi-27B-12G.gguf` (the release file) | Unsloth's UD-Q2_K_XL GGUF, 9,138,715,328 bytes |
| Engine | The Kintsugi engine of the release | Unsloth's llama.cpp source release b12588-mix-c0c3de8, built from source. Stock llama.cpp cannot load this model yet. |
| Sampling | As for the card numbers above (temperature 1.0) | The vendor's: temperature 1.0, top-p 1.0, top-k 0, min-p 0 |
| Window and K/V | Up to 262,144 tokens on 12 GB (tiered K/V cache) | 65,536 tokens, f16 K/V |

**Results**

| Test | Kintsugi-27B | Inkling 2 Mini | Verdict |
|---|---|---|---|
| AppWorld test_normal, fresh seeds 102 and 103 (336 task-runs; corrected 2026-11-21, see the note below) | **294 / 336** (87.5 %) | 283 / 336 (84.2 %) | No significant difference. Kintsugi solved 11 more task-runs; paired sign test on the discordant tasks: 27 vs 16, p = 0.13. |
| HumanEval 164, medium thinking, pass@1 | 159 / 164 | 160 / 164 | Even (discordant 4 vs 3, p = 1.00). |
| Retrieval at depth 60,000 (24 needles + 20 exact code copies) | 24 / 24 and 20 / 20 | 24 / 24 and 20 / 20 | Even. |
| Retrieval at depth 64,000 | 24 / 24 and 20 / 20 | Stopped with CUDA out of memory at the first decode (67,584-token window) | Only Kintsugi. |
| Retrieval at depth 117,000 and 245,000 | 24 / 24 and 20 / 20 at each depth | Cannot load a window this long on 12 GB | Only Kintsugi. |
| Context on 12 GB | 262,144 tokens | Tests ran at 65,536. It loads up to about 70k, with short answers only. q8_0 K/V does not fit, even at 32k. | Kintsugi has 4x the context. |

The HumanEval number on the card (161) is from the home RTX 4070. Here, 159 is the same model on the rented box. This
difference is normal run-to-run spread.

**Correction (2026-11-21).** This table first reported AppWorld on seeds 100 and 101: Kintsugi-27B 302 / 336, Inkling 2
Mini 283 / 336, "Kintsugi is better, p = 0.0079". Those two seeds were also the seeds on which we chose Kintsugi-27B's
bit map, so its score on them was biased upward. We reran both models on two fresh seeds (102 and 103) with the same
settings. Inkling 2 Mini scored 283 again; Kintsugi-27B scored 294. The difference is not significant (p = 0.13), so we
withdrew the claim that Kintsugi-27B is better on AppWorld.

**Fairness notes**

- Inkling 2 Mini has no switch to turn thinking off. Thus its retrieval ran at its lowest effort ("none"), with 1,024
  more answer tokens than Kintsugi had. This helps Inkling a little.
- One Inkling AppWorld run went past its 65,536-token window (1 of 336).
- We logged each change to a setting before we saw its result.

**What this does NOT claim**

- Thinking Machines published results at full precision. We do not compare with those results.
- This compares the two models on one 12 GB card, which is where most people run them.
- It compares the best 12 GB build of each that existed on 2026-11-20: our own tuned build of Qwen3.8-27B against
  Unsloth's public build of Inkling 2 Mini. A build of Inkling made with our methods could score higher. So this is not
  a verdict on the two base models.
- We give no speed numbers, because rented boxes are not comparable for speed.
