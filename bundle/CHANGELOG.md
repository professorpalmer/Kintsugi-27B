# Kintsugi-27B bundle: changelog

The model file (`Kintsugi-27B-12G.gguf`, sha256 `0510bae82445dc53dbc7d0d842d8b7e146b8eafce0837e39b157d411e48c8a74`) is
the same for every version below: when a new bundle comes out, download only the new zip. Each release on the
[Releases page](https://github.com/professorpalmer/Kintsugi-27B/releases) also has the engine source
(`Kintsugi-27B-v1-engine-src-c07efbec5.tar.gz`, sha256 `0e8f2a1c3c293ce83ca9892e2ba61a3b3beaaa8870ecc907f2a047092d6391b7`).

## v1.1 (2026-11-21)

- **Faster decode with an optional drafter.** Download `dflash-Qwen3.8-27B-Q4_0.gguf` (1.1 GB) from [Hugging
  Face](https://huggingface.co/CaryPalmer/Kintsugi-27B) into `models`. The launcher then uses it (the `spec` line says
  "DFlash2 draft 7"): on an RTX 4070 (the shipped settings), about 25-35 % faster decode at 4k-64k tokens of context,
  about 23 % at 117k, and the same speed at 245k. Without the file, the launcher uses MTP + lookup as before and prints
  one note. `KINTSUGI_DRAFT=mtp|dflash|off` chooses by hand. The drafter is Inco AI's DFlash2 in ggml-org's Q4_0 GGUF,
  unchanged (Apache-2.0).
- **The launcher reads the memory that the NVIDIA driver keeps for itself** ("reserved" in the `kv` line: 272 MiB on
  our Windows RTX 4070, about 410 MiB on Linux RTX 4070s) and keeps the server under the rest. Before, on a card with a
  large reservation, the server could stop with "out of memory" a few requests after the start (seen on Linux with the
  drafter). With the drafter, the launcher also keeps 128 MiB more room for the CUDA graphs of drafting.
- The engine (`5f393adb5`) works with NVIDIA drivers that refuse the default pinned-memory type (Linux 570): it uses
  another type and writes one WARN line. No false "spill at load" message when the GPU memory has room left.
- The `kv` line says why it chose its memory margin. A model file reached through a symbolic link is sized correctly.
- On RTX 20 (Turing) cards, drafting (MTP or DFlash2) is not bit-identical to one-token decoding (near ties can round
  the other way); the fidelity is the same. On RTX 30 / 40 it is bit-identical.
- Windows: `Kintsugi-27B-v1.1-win-x64.zip`, sha256 `20660dcdb2f28d5a58a527ca0d1a4e9a82568f53e6877232c292dfb1c631d8e9`.
- Linux: `Kintsugi-27B-v1.1-linux-x86_64.tar.gz`, sha256
  `e03537ecfc0e53d7ff13e3c6bcec1bea74547a301d13c4d497d9d3b1542b4079`.
- Engine source: `Kintsugi-27B-v1.1-engine-src-5f393adb5.tar.gz`, sha256
  `ded8fd4ee8c67c1088b173bd49a21ab201a1407d1ee64b2bcd01f9d3ebcaf4a6`.

## v1.0.4 (2026-11-21)

- A Linux bundle (x86_64): `Kintsugi-27B-v1.0.4-linux-x86_64.tar.gz`, sha256
  `19e43162561de515713e52158ade72e8164aa42ba40704ef8f5462033c92b28a`. Same engine features, defaults and checks as
  the Windows bundle; launcher `start-server.sh`, stop `stop.sh`. Tested on Ubuntu 22.04 and 24.04 (RTX 4070).
- The Linux engine (`Kintsugi-27B-v1.0.4-engine-src-af70fc544.tar.gz`, sha256
  `553c2222a156f5734be39206b19cb0bbd084d881bc473c9298d340ad807edd00`) works with NVIDIA drivers 570 and newer: on a
  driver older than the 580 family it uses another type of pinned system memory and writes one WARN line (tested on
  driver 570.133, RTX 2080 Ti).
- Windows: no change. The Windows zip stays the v1.0.3 file.

## v1.0.3 (2026-11-20)

- The launcher checks the NVIDIA driver before it starts. With a driver older than 580, it stops and tells you to
  update the driver. Before, an old driver let the engine fall back to the CPU: the server said "ready" but answered
  very slowly, with no message.
- After the model loads, the launcher also checks that the engine uses the GPU. If not, it stops with a clear message.
- The engine files are the same as in v1.0.2.
- Zip: `Kintsugi-27B-v1.0.3-win-x64.zip`, sha256 `ac7b3b541fdff682e596d883bb841bb5fb5e84db4a9c10ad6c1dc51637b1b61b`.

## v1.0.2 (2026-11-20)

- Claude Code and OpenAI Codex work. Before, both stopped on the first request with "System message must be at the
  beginning": Claude Code sends a system message after the first turn, and Codex sends two developer messages. The
  bundle now has its own chat template (`templates\kintsugi.jinja`) that accepts them. For all other requests the
  template gives the same prompt as before: 52 test chats gave byte-identical prompts and answers.
- New harness page (`HARNESSES.md` in the zip; [harnesses/README.md](../harnesses/README.md) here): the settings for
  Claude Code, Codex, Pi and OpenCode. Each harness passed 5 of 6 coding tasks, with no malformed tool calls, no leaked
  tags and no loops.
- `$env:KINTSUGI_CHAT_TEMPLATE = "model"` uses the template inside the model file; a path uses your own template.
- The engine files are the same as in v1.
- Zip: `Kintsugi-27B-v1.0.2-win-x64.zip`, sha256 `15e1fcfb17260ab3daea6edd747f55b332322f175d99d1d5f460e7e35853428a`.

## v1.0.1 (2026-11-20)

- The layer streams `/v1/messages` (Claude Code) and `/v1/responses` (Codex) live. Before, those answers arrived in one
  piece at the end.
- API-key headers (`x-api-key`, `anthropic-*`, `openai-*`) go through the layer.
- `stop.ps1` waits (up to 30 s) for a chat save in progress, so a stop right after a long answer keeps that chat.
- The engine files are the same as in v1.
- Zip: `Kintsugi-27B-v1.0.1-win-x64.zip`, sha256 `e80fa76c23ed51600a0f4972c84dbb2e2a3d8fe45a7f2aa36e4ebfa1b2b4295e`.

## v1 (2026-11-20)

- First public release: the engine (a llama.cpp fork with EXL3 trellis kernels, built for RTX 20 / 30 / 40), the
  launcher, the saved-chat layer, licences and credits.
- Zip: `Kintsugi-27B-v1-win-x64.zip`, sha256 `e69f66d3e18bfc6b3b6d9ebf8aae345c511571162daf6435962398dc68d6ce02`.
