# Bit allocation: which projections get the third bit

**Status:** finished (2026-11-18 to 2026-11-20). **Finding:** the maps with the lowest KL did not win the task gates.
KL rank is not task rank.

The shipped bit layout is the **L0xRE-V2 bit layout by YourHighnessLA (L0xRE)**. We refitted every projection with
our own fitter, but each of the 400 choices of 2 or 3 bits is his. This page shows how we tried to make a better
layout of our own, and why none of our layouts shipped.

## Question

Kintsugi-27B codes 400 projections. Each projection uses 2 bits per weight (K2) or 3 bits per weight (K3). At the
release size (2.469 bits per weight: 227 x K2 + 173 x K3), which projections must get K3?

## Terms

| Term | Meaning |
|---|---|
| K2, K3 | 2 or 3 bits per weight for one coded projection. One K step costs 0.625 MiB per 5,242,880 weights: attn_k / attn_v 0.625 MiB, ssm_out 3.75 MiB, attn_qkv 6.25 MiB, each MLP projection 10.6 MiB. |
| Roles | MLP: ffn_gate, ffn_up, ffn_down (64 layers). Full attention (16 layers): attn_q, attn_k, attn_v, attn_output. Linear attention (GDN, 48 layers): attn_qkv, attn_gate, ssm_out. |
| Map | The list of K for all 400 projections. |
| Fit library | Our fits of every projection at K2 and at K3: exllamav3 `quantize_exl3` on Qwen's BF16 weights, with the public Hessians of the Qwen3.8-27B quant challenge (StillDeadcode, CC BY 4.0). A new map takes its projections from this library. Only the map changes between the files on this page. |
| **V3a** | Our fits on the L0xRE-V2 bit layout (all 400 choices as in L0xRE-V2). 2.4690 bits per weight. |
| **V3ac-h32** | V3a, rebuilt from public inputs only, with a Q8_0 embedding and head (one scale per 32 weights). This is the release file `Kintsugi-27B-12G.gguf` (same tensor data). See [clean-rebuild](../clean-rebuild/README.md). |
| KL | Mean KL divergence against BF16 on the golden set of the quant challenge, with its scorer (ported to our engine). K/V q8_0, all K/V in VRAM. Lower is better. "-7 %" means 7 % lower mean KL than the reference. Intervals are 95 % paired bootstrap intervals over documents. |
| p1, p2 | Two halves of the public golden set (documents up to 32,768 tokens). p1 = 295 documents, 376,086 scored tokens. p2 = the other 294 documents. We built every map on p1 only, so p2 is an untouched check. |
| T, C | Subsets of p1 for fast scores: T = every 8th document (74 documents), C = 149 other documents (no document shared with T). The sensitivity table uses T; C checks a map on documents the table did not see. |
| G2 | Retrieval at 117,000 tokens: 24 needle questions (a planted number) + 20 exact copies of a code function. Line: 24/24 + 20/20. Thinking off. 131,072-token window profile (a development profile, not in the bundle). |
| R3a, R3b, R4 | Release gates: R3a = G2 on the release engine; R3b = the same 24 + 20 at 245,000 tokens with the bundle defaults; R4 = HumanEval 164, medium thinking, line >= 158. |
| Margin | For one answer token: log-probability of the right token minus that of the best wrong token (nats), with the right answer forced. Below 0 = greedy decoding gives a wrong token. The min margin of an item is the smallest margin over its answer tokens. |
| W1, W2 | Wide held-out recall probes at 117,000 tokens: 12 new needles + 12 copies of functions from our own private code (so the model cannot know them from training). W2 uses a new seed and new items. A map passes W2 if it loses no item that V3a gets right, its margin sum is higher than V3a's (interval over items above 0), and no item with a V3a margin >= 1.0 falls under 0.5. |
| AppWorld line | AppWorld test_normal (168 tasks), seeds 100 and 101, pooled (336 task-runs). A map can replace the release only with >= 296 of 336 AND not significantly worse than V3ac-h32 (exact paired sign test p >= 0.05). |

Where measured: KL, retrieval, margins and HumanEval on our rig (one RTX 4070 12 GB, Windows 11). AppWorld on rented
RTX 4070 boxes (Linux). Every pass line on this page was written before the run that it judges.

## What we did

1. **Refit of the L0xRE-V2 layout (V3a), then the release file (V3ac-h32).** Same map, our fits.
2. **Maps from a proxy loss (V3b, V3c).** A greedy search that minimizes the sum of the per-projection proxy losses
   (the Hessian-weighted fit error of each projection).
3. **Two hybrids to find the cause (V3h1, V3h2).**
4. **A measured sensitivity table (V3s).** Change one group of projections in V3a (one role, one band of layers),
   score KL on T, and get the KL change per MiB. Then build maps greedily from the table, and measure every map.
5. **Same-size maps from the table (V3s1, V3s2), and a larger one (H1).** Then retrieval at 117k, a bisection of the
   retrieval loss, and a KL / NLL check by depth band.
6. **Our own map from scratch (V5a, V5b, V5c, V5d).** Candidates from the table, judged by retrieval first, then the
   release gates. V5b went through all of them as V5bc-h32 and then met AppWorld.
7. **A tools-aimed map (V5t).** V5b with the projections that cost the tools class kept at K3.

## Results

### 1. The release map and the proxy-loss maps

| File | Map | Size | KL p1 | Against V3a | Documents lower |
|---|---|---|---:|---|---:|
| V3a | L0xRE-V2 layout | 2.4690 bpw | 0.05647 | (reference) | |
| V3ac-h32 (release) | L0xRE-V2 layout, Q8_0 head per 32 | same | 0.05640 | -0.1 % [-0.2, -0.1] | 191 / 295 |
| V3b | greedy by proxy loss | same bits as V3a | 0.07720 | **+36.7 % [+33.7, +39.7]** | 17 / 295 |
| V3c | greedy by proxy loss | 2.6179 bpw (+431 MiB) | 0.06491 | **+15.0 % [+12.8, +17.2]** | 49 / 295 |
| V3h1 | V3a with the MLP projections of layers 49-63 at K2 (27 projections, -286 MiB) | 2.370 bpw | 0.08256 | +46.2 % [+43.1, +49.2] | |
| V3h2 | V3a with 58 upgrades of V3b, no downgrade | 2.585 bpw (+336 MiB) | 0.04873 | -13.7 % [-14.4, -13.0] | 275 / 295 |

- V3b cut the proxy sum by 17.3 % and made the model 36.7 % worse. With 431 MiB more (V3c), it was still 15 % worse.
- An upgrade cannot cost KL, so the loss is in V3b's downgrades. 27 of its 33 downgrades put the MLP projections of
  the last layers (49-63) at K2. V3h1 shows that this one family costs 46 % of KL. The proxy loss of these
  projections is small, so the proxy sum does not see their effect on the output.
- **Lesson 1: a local proxy loss is a wrong guide for the bit map.** The L0xRE-V2 layout holds this knowledge (the
  last MLP layers at K3), and our fits inherit it in V3a.

### 2. The sensitivity table (subset T, KL against V3a)

One row = one group of projections moved one K step. "Per MiB" = KL change for each MiB of code added (upgrade) or
freed (downgrade).

| Group | Projections | MiB | KL change | Per MiB |
|---|---:|---:|---|---:|
| ssm_out to K3 (the 30 that are K2 in V3a) | 30 | +112.5 | -8.28 % [-9.05, -7.36] | 0.074 % |
| ffn_gate 48-54, 56 to K3 | 8 | +85.0 | -5.48 % [-6.37, -4.57] | 0.064 % |
| attn_qkv 0-15 to K3 | 12 | +75.0 | -4.50 % [-5.84, -3.23] | 0.060 % |
| attn_output to K3 (the 12 at K2) | 12 | +45.0 | -2.40 % [-2.96, -1.96] | 0.053 % |
| attn_qkv, all 48, to K3 | 48 | +300.0 | -14.86 % [-15.76, -13.91] | 0.050 % |
| ffn_gate 32-63 to K3 (24 at K2) | 24 | +255.0 | -12.10 % [-13.79, -10.59] | 0.048 % |
| attn_gate to K3 (47) | 47 | +176.2 | -6.32 % [-7.45, -5.61] | 0.036 % |
| ffn_gate 0-31 to K3 | 32 | +340.0 | -9.48 % [-10.11, -8.82] | 0.028 % |
| ffn_up 0-20 to K3 (16 at K2) | 16 | +170.0 | -3.23 % [-3.73, -2.62] | 0.019 % |
| attn_q to K3 (16) | 16 | +120.0 | -1.70 % [-2.35, -1.17] | 0.014 % |
| ffn_up 32-47 to K2 | 16 | -170.0 | +4.10 % [+3.42, +4.78] | 0.024 % |
| ffn_up 21-31 to K2 | 11 | -116.9 | +3.05 % [+2.13, +3.96] | 0.026 % |
| ffn_up 4, 5, 10, 11, 15 to K2 | 5 | -53.1 | +1.74 % [+1.27, +2.22] | 0.033 % |
| ffn_down 16-31 to K2 | 16 | -170.0 | +6.10 % [+5.18, +7.32] | 0.036 % |
| ffn_down 32-47 to K2 | 16 | -170.0 | +6.95 % [+5.79, +8.29] | 0.041 % |
| ffn_down 0-15 to K2 (14 at K3) | 14 | -148.8 | +6.24 % [+4.74, +7.35] | 0.042 % |
| ffn_up 48-55 to K2 | 8 | -85.0 | +4.64 % [+3.73, +5.73] | 0.055 % |
| attn_k to K2 (16) | 16 | -10.0 | +0.80 % [+0.48, +1.12] | 0.080 % |
| ffn_down 48-63 to K2 | 16 | -170.0 | +21.68 % [+17.85, +25.30] | 0.128 % |
| ffn_up 56-63 to K2 | 8 | -85.0 | +12.54 % [+10.55, +14.58] | 0.148 % |
| attn_v to K2 (16) | 16 | -10.0 | +2.13 % [+1.53, +2.78] | 0.213 % |

How the table was checked:

- The scorer repeats exactly (one model file gives the same rows, row by row). So the only noise of a subset score is
  the choice of documents. Subset T ranked six known maps in the same order as the full p1.
- T overstates gains by 15-20 % against C, but the order on T was the order on C every time.
- Effects add only in part. Two groups in different sub-blocks added to 98 % of the sum of their parts; a map of six
  groups got 82 %. We planned with "sum x 0.8" and measured every map.
- By KL, the best sinks for a third bit are ssm_out, the late ffn_gate, attn_qkv and attn_output. The cheapest
  sources are the ffn_up of the middle layers. The last MLP layers, attn_k and attn_v must stay at K3.

### 3. Same-size maps from the table, and retrieval

| Map | Changes against V3a | MiB | KL p1 | KL p2 (untouched) | G2 at 117k |
|---|---|---:|---|---|---|
| V3s1 | ssm_out (30), attn_output (12), ffn_gate 32-47 to K3; ffn_up 4, 5, 10, 11, 15, 21-47 to K2 (90 projections) | -12.5 | -6.7 % [-7.8, -5.7] | | probe: both items wrong |
| V3s2 | ssm_out (30), attn_output (12), ffn_gate 39-54 + 56 to K3; the same 32 ffn_up to K2 (91 projections) | -1.9 | **-9.3 % [-10.5, -8.3]** | **-8.2 % [-11.1, -5.8]** | **23/24 + 19/20: FAIL** |
| H1 | V3s2 + all 48 attn_qkv + ffn_gate 35-38 to K3 (143 projections) | +340.6 | -23.4 % [-24.5, -22.3] | -25.8 % [-28.6, -22.1] | 24/24 + 19/20: FAIL |

- V3s2 passed every KL line: p1, the untouched p2, and every category of both. Then it failed G2 on two items: needle
  "vault 8" (want 6991, answer 2542) and the copy of `_check_tzname` (first line right, body wrong). V3a passed the
  same run 24/24 + 20/20. Runs with the exact prompt read and with the fp32 attention sum (see
  [long-context-decode](../long-context-decode/README.md)) failed the same two items, so the cause is the V3s2 weights.
- **KL by depth band does not see it.** On the 7 public documents longer than 32,768 tokens (context 262,144), V3s2 is
  better than V3a in every band, in KL (-8 % to -13 %) and in the model's own NLL of the true next token (-0.20 % at
  0-32k to -1.24 % at 128-256k, intervals under 0 past 32k). Average next-token quality at depth went up while recall
  of one exact fact far back went down.

**Bisection with margins.** A fast probe runs G2's settings with only the two failed items (about 5 minutes). The
control (V3s2) failed both items again, so the probe reaches the fault. Margins use the bench's own prompts (117,319
and 110,992 tokens).

| Map | Has the 32 ffn_up downgrades | Vault 8 min margin | `_check_tzname` min margin |
|---|---|---:|---:|
| V3a | no | +1.62 | +0.43 |
| V3a + only the 32 ffn_up downgrades | yes | wrong answer (2542) | right |
| V3s1 | yes | -2.09 | -1.44 |
| V3s2 + ffn_gate 35-38 | yes | -1.56 | -1.21 |
| V3s2 + all attn_qkv | yes | wrong answer (2542) | -0.35 |

- Two server starts of V3a gave the same margins to 0.000 nats: the engine path repeats exactly.
- Every map with the ffn_up downgrades moves the first digit of vault 8 from +1.62 to about -1.6 / -2.1.
- `_check_tzname` sits near the edge on V3a itself (+0.43). We wrote the rule before these numbers: an item with a
  V3a margin under 1.0 is fragile, and its flips make no rule.

### 4. Our own map (V5)

| Map | Changes against V3a | MiB | KL on T | Vault 8 / `_check_tzname` margin | G2 | W2 | Verdict |
|---|---|---:|---|---|---|---|---|
| V5a | ssm_out, attn_output, ffn_gate 48-63, attn_qkv 0-15 to K3; ffn_down 0-31 to K2 (92) | -1.2 | -8.52 % [-9.84, -7.35] | +1.29 / -0.20 | 24/24 + **19/20** | pass | **FAIL** (G2; W1 margin line missed by 0.43) |
| V5b | ssm_out, attn_output, ffn_gate 48-63, all attn_qkv to K3; ffn_up 4-15, ffn_down 0-47 to K2 (149) | +0.6 | -8.94 % [-10.67, -7.54] | +2.37 / +0.44 | **24/24 + 20/20** | pass | to the release gates as V5bc-h32 |
| V5c | ssm_out, ffn_gate 48-63, all attn_qkv to K3; ffn_up 4-47, ffn_down 16-31 to K2 (134) | -12.5 | -10.60 % [-12.27, -8.73] | **-1.94** / +0.72 | (not run) | | **FAIL** (a negative first-digit margin on G2's own prompt) |
| V5d | ffn_gate 48-54 + 56, attn_qkv 0-15 + 32-63 to K3; ffn_down 0-31 to K2 (74) | -8.8 | -4.95 % [-6.13, -4.01] | +3.18 / +1.03 | 24/24 + 20/20 | fail (no margin gain: -0.08 [-7.43, +7.86]) | not a candidate |

- V5c passed the W1 wide probe and still lost vault 8. So the wide probes lack power for this kind of loss: their
  items are easy for V3a (min margins 2.4-7.2 nats on W1, >= 5.04 on W2). **G2 is required for every map**; the wide
  probes only add to it.
- V5b has the ffn_up downgrade of layers 4-15 and keeps vault 8 (+2.37). So the vault-8 loss sits in the ffn_up
  downgrades of layers **21-47** (a sign from one map, not a proof). Every map with them lost vault 8; every map
  without them kept it.

**V5bc-h32 (V5b with the release head) against the release V3ac-h32:**

| Gate | Line | V5bc-h32 | V3ac-h32 |
|---|---|---|---|
| R2: KL p1 | lower than the release, interval under 0 | **-7.0 % [-8.3, -5.6]**, lower on 242 / 295 | (reference) |
| KL by class (p1) | | tools -4.2 %, code -10.5 %, agentic -7.9 %, reasoning -5.7 %, general -6.7 % | |
| R3a: 117k | 24/24 + 20/20 | pass | pass |
| R3b: 245k | 24/24 + 20/20 | pass | pass |
| R4: HumanEval | >= 158 | 162 / 164 | 161 / 164 |
| AppWorld, seed 100 | | 141 / 168 | 151 / 168 |
| AppWorld, seed 101 (both arms on one box) | | 144 / 168 | 151 / 168 |
| **AppWorld, pooled** | >= 296 and not significantly worse | **285 / 336** | **302 / 336** |

Pooled discordant tasks 15 / 32, sign test **p = 0.0186**. V5bc-h32 was better on every KL class, passed every
retrieval and HumanEval gate, and lost the agent test. The wins did not repeat on the same tasks (1 task went to
V3ac-h32 on both seeds, 2 to V5bc-h32), so the gap is spread over many tasks. The two seeds share their tasks, so the
pooled p is a little optimistic; that was the rule as written.

V5bc-h32 also showed a chat flag: on two long WildChat documents of p2, mean KL 1.120 against 0.698 (document 93, a
long reasoning answer) and 1.051 against 0.337 (document 529, not English). On all 46 chat documents the change was
+44.6 % [-6.6, +55.9] with the median equal, so by its own line it stayed. From then on, documents 93 and 529 were a
named check for every new map: line <= 0.873 and <= 0.422 (V3ac-h32 + 25 %).

**V5t (tools-aimed):** on subset C, ffn_down 16-31 at K2 cost the tools class +16.6 %, mostly in layers 20-27
(+5.0 % and +9.6 %). V5t = V5b with ffn_down 20-27 kept at K3, paid by attn_qkv 16-31 and the ssm_out of layer 1 left
at K2 (+6.88 MiB against V3a, +0.096 % bits).

| Gate | Line | V5t | V3ac-h32 |
|---|---|---|---|
| KL on T / on C (against V3a) | interval under 0 | -9.63 % / -7.51 % | |
| KL p1 | | **-7.9 %** (tools -9.8 %, code -12.4 %, agentic -6.4 %, reasoning -7.0 %, general -7.5 %) | (reference) |
| G2: 117k | 24/24 + 20/20 | pass (vault 8 +4.45, `_check_tzname` +0.00) | pass |
| W2 | margin gain, nothing lost | pass (+23.20 [+10.07, +35.88]) | |
| R3b: 245k | 24/24 + 20/20 | pass | pass |
| R4: HumanEval | >= 158 | 161 / 164 | 161 / 164 |
| AppWorld, pooled | >= 296, not significantly worse | 301 / 336 (p = 1.00, discordant 17 / 18) | 302 / 336 |
| Named chat check | <= 0.873 / <= 0.422 | **1.398 / 0.582 (2.0x / 1.7x the release): flagged** | 0.698 / 0.337 |

A rerun of V3ac-h32 seed 101 on the same box type gave the same 151, with 0 discordant tasks: the AppWorld plate
repeats exactly for one seed and one host type.

## Decision log

| Date and time | Decision | Why |
|---|---|---|
| 11-18 15:49 | V3a (our fits on the L0xRE-V2 layout) becomes our best weight set. | Our fits, same layout. |
| 11-18 16:05 / 17:08 | V3b and V3c fail; greedy-by-proxy is closed. | +36.7 % and +15.0 % KL against V3a. |
| 11-19 10:01 | V3s2 becomes the default candidate. | -9.3 % KL at the same size; p2 agrees. |
| 11-19 13:45 | Start our own map from scratch (V5). | The table shows the layout under-uses the best KL sinks. |
| 11-19 14:23 | Release files get the Q8_0 head with one scale per 32 weights. | -0.14 % KL, free (same size). |
| 11-19 15:25 | Back to V3a. Retrieval at 117k becomes a gate for every map. | V3s2 fails G2; KL did not see it. |
| 11-20 00:25 | No "must stay K3" rule from the two probe items. | Both are near the edge on V3a (+1.62, +0.43). |
| 11-20 04:30 | V5a fails. | G2 19/20, and the W1 margin line as written. |
| 11-20 06:46 | V3ac-h32 passes R0-R4 and becomes the launcher default. | Every release line passes; HumanEval 161. |
| 11-20 07:06 | G2 is required for every map; V5c fails by its exact-prompt margin. | The wide probe passed a map that loses vault 8. |
| 11-20 10:17 | V5bc-h32 passes R0-R4 and becomes the launcher default in our development tree. | KL -7.0 % against V3ac-h32; HumanEval 162. |
| 11-20 15:24 | **Back to V3ac-h32**; a new final line: a map must also hold AppWorld (>= 296 of 336, not significantly worse). | AppWorld 285 against 302, p = 0.0186. |
| 11-20 16:47 | Public release v1 = V3ac-h32. | |
| 11-20 19:22 | **V5t does not ship.** | It passed every gate but gives no task gain (HumanEval, AppWorld and retrieval equal), its KL gain predicts neither agent work nor retrieval on our own evidence, it is 2.0x / 1.7x worse on the two long chat documents, and a swap costs every user a new 10.5 GB download. |

## What we learned

1. **KL rank is not task rank.** Lower KL gave a worse agent score (V5bc-h32: -7.0 % KL, -17 AppWorld task-runs) and
   lost recall (V3s2: -9.3 % KL, G2 fail). Better KL classes for tools and agentic did not predict AppWorld either.
2. **A proxy loss is a worse guide than KL.** The proxy-loss map (V3b) was 36.7 % worse in KL at the same size.
3. **KL or NLL by depth band does not stand in for retrieval.** Retrieval needs its own gate.
4. **Recall probes need hard items.** Easy items (high margins on the reference) cannot see a recall loss. Margins on
   the exact gate prompt are a fast and exact stand-in for a greedy answer.
5. **The ffn_up of layers 21-47 carries recall.** Every map that put them at K2 lost the same 117k needle.
6. **HumanEval is at its ceiling.** All our files scored 158-162 of 164, with a spread of up to 3 problems between
   runs of one file. We use it as a health gate (>= 158), not to rank files.
7. **Write the line before the run, and keep it.** Two lines were wrong on first use (a capped margin score, a class
   line against a map that had failed). We changed them only for later maps, and we say so.

## Limits

- One task plate for agents (AppWorld test_normal, 2 seeds). It sees differences of about 10 task-runs or more. A
  smaller real difference reads as "not worse".
- The sensitivity table measures single groups against V3a on 74 documents. Interactions between groups are
  measured only for the maps we built.
- Retrieval is one needle set and one copy set at 117k and 245k. The vault-8 and `_check_tzname` items are single
  items; we treat them as signs, not rules.
- The chat flag rests on a few long WildChat documents. We have no long free-chat quality test.
- All maps use our fit library. Other fits can have other sensitivities.

## Next

A map guided by task damage instead of KL (in progress, private).
