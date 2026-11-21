# Research: finished studies

These are the finished studies behind Kintsugi-27B, with their numbers, their failures and the decisions they led to.
Work that is still in progress is not here. Each study says where and when each number was measured. We wrote every
pass line before the run that it judges.

| Study | Status | Finding in one sentence |
|---|---|---|
| [bit-allocation/](bit-allocation/README.md) | finished, nothing shipped | At the same size, our own bit maps had up to 9 % lower KL than the shipped layout (the L0xRE-V2 layout, refitted by us) but lost recall at 117k, lost AppWorld, or were worse on long chat: KL rank is not task rank. |
| [harness-gate/](harness-gate/README.md) | finished, fixes shipped in v1.0.1 and v1.0.2 | Claude Code and Codex failed every request on the model's own chat template; with a fixed template, all four harnesses pass 5 of 6 tasks with no malformed calls, leaks or loops. |
| [old-driver/](old-driver/README.md) | finished, launcher checks shipped in v1.0.3, engine fix in the next release | Drivers of the 570 family refuse the pinned memory type of the tiered K/V cache; a HOST_NUMA fallback loads and passes on two 570 boxes, and the launcher now stops a silent CPU fallback on Windows. |
| [long-context-decode/](long-context-decode/README.md) | finished | Past about 144k tokens, the PCIe stream of the cold K/V cells sets the decode speed; a sparse V read, two copy-pipeline designs, a 3-bit K cache and a 2-bit head stage did not pass their gates, and an fp32 attention sum fixed a real kernel error but did not change recall or the true-token NLL. |
| [cross-architecture/](cross-architecture/README.md) | finished | On Turing and Ampere the engine computes the same model as on Ada (KL +0.02 % and +0.92 %, retrieval 24/24 + 20/20 at 117k); identity differs by rounding only. |
| [clean-rebuild/](clean-rebuild/README.md) | finished, shipped in v1 | The release file is built from our fits and Qwen's BF16 release only (+0.0 % KL against the old build), and a per-32 Q8_0 head gives -0.1 % KL for free. |

Common terms (each study defines its own once):

- **KL**: mean KL divergence against BF16 on the golden set of the Qwen3.8-27B quant challenge (StillDeadcode,
  CC BY 4.0), with its scorer ported to our engine.
- **Retrieval at 117k / 245k**: 24 needle questions and 20 exact copies of a code function, at that depth.
- **Rig**: one RTX 4070 12 GB, Windows 11. Speed numbers come from the rig only.
