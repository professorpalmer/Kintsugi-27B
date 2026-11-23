# Kintsugi-27B bundle: changelog

The model file (`Kintsugi-27B-12G.gguf`, sha256 `0510bae82445dc53dbc7d0d842d8b7e146b8eafce0837e39b157d411e48c8a74`) is
the same for every version below: when a new bundle comes out, download only the new zip. Each release on the
[Releases page](https://github.com/professorpalmer/Kintsugi-27B/releases) also has the engine source
(`Kintsugi-27B-v1-engine-src-c07efbec5.tar.gz`, sha256 `0e8f2a1c3c293ce83ca9892e2ba61a3b3beaaa8870ecc907f2a047092d6391b7`).

## v1.1.4 (2026-11-23)

- **Launcher only: which setting to lower when the VRAM is short.** When you set `KINTSUGI_CELLS` by hand and the
  value is higher than this card can hold with the drafting mode, the launcher prints the safe maximum (its own count
  for this card, this mode and the VRAM in use) and what to do: "lower KINTSUGI_CELLS to N, or set KINTSUGI_DRAFT=mtp".
  It keeps your value. Amel (@donamelo) reported an RTX 3080 Ti 12 GB with a display on the card (329 MiB in use before the
  start) and 83,712 cells set by hand: with DFlash2 the first prompt read failed ("failed to allocate compute pp
  buffers"). On a rented RTX 3080 Ti with 328 MiB held we got the same failure; the safe maximum there is 64,000 cells
  with DFlash2 (42,752 with the display margin) and 118,016 with MTP.
- **`KINTSUGI_DRAFT=dflash` and a failed warm-up:** the launcher now prints the error lines (and, for out of memory,
  the setting to lower), keeps the log as `logs/server-<port>.dflash-failed.log`, stops the server and the layer, and
  exits with code 1. It no longer says "ready" for a server that cannot read a prompt. With `KINTSUGI_DRAFT=auto` the
  v1.1.3 fallback to MTP stays.
- **Warm-up:** with `KINTSUGI_UB` above 1024, the warm-up prompt is longer, so that it fills one full micro-batch and a
  shortage shows at the start. At the default (1024) it does not change (1,337 tokens).
- Checks: the launchers print the same settings and server command as v1.1.3 in every drafting mode (Windows and
  Linux); the start cases (defaults, too many cells with dflash and with auto, MTP, cells under the maximum, and the
  v1.1.3 fallback cases) pass on an RTX 4070 with Windows and on an RTX 3080 Ti with Linux, and again from the
  release zip and tarball, with no server or layer left running after each. On Windows, too many cells did not fail:
  the driver moved 9 MiB to system RAM and the launcher printed its SPILL line. The engine and the model file do not
  change: download only the new zip or tarball.

## v1.1.3 (2026-11-22)

- **Launcher only: DFlash2 falls back to MTP once.** When the launcher picks the DFlash2 drafter by itself (the drafter
  file is in `models`, `KINTSUGI_DRAFT=auto`) and the server stops during the load or the warm-up request fails with
  it, the launcher keeps that log as `logs/server-<port>.dflash-failed.log`, prints its error lines, and starts once more
  with MTP + lookup drafting. `KINTSUGI_DRAFT=dflash` still stops with the error. A user reported a DFlash2 warm-up
  failure on an RTX 3080 Ti 12 GB with Windows; on a rented RTX 3080 Ti with Linux we could not make it fail (VRAM peak
  10,988 MiB of 11,901 at our test settings), so please send us the kept log if you see the fallback note.
- The launcher no longer says "ready" when the server stopped during the warm-up request.
- Checks: the Windows launcher prints the same settings and server command as v1.1.2 in every drafting mode; five start
  cases (DFlash2, a broken drafter, a refused warm-up, a forced broken DFlash2, MTP) pass on an RTX 4070 with Windows
  and on an RTX 3080 Ti with Linux, with no server or layer left running after each. The engine and the model file do
  not change: download only the new zip or tarball.

## v1.1.2 (2026-11-22)

- **Optional images.** Download Qwen's vision projector (`mmproj-Qwen3.8-27B-Q8_0.gguf`, 629 MB, from ggml-org's
  Qwen3.8-27B-GGUF page) into `models`, and the launcher turns images on: in chat requests (OpenAI and Anthropic
  formats) and in tool results. On a card under 16 GB the projector runs on the CPU (no VRAM cost, 5-25 s per image);
  from 16 GB on the GPU (1-3 s). `KINTSUGI_VISION=auto|cpu|gpu|off`. With images on, drafting is MTP + lookup (the
  DFlash2 drafter does not work with images yet). Our checks: 19/20 on our image set (the same as stock llama.cpp with
  the same files), images after a 117k-token conversation 4/4, the same text output with the projector loaded (50 / 50
  prompts), Claude Code 5/6 plus an image task. The built-in web page is not tested with images.
- **RTX 50 (Blackwell) and H100 / H200 (Hopper).** The engine now has native code for RTX 50 (sm_120) and Hopper
  (sm_90), as well as RTX 20 / 30 / 40, and portable code (compute_90 PTX) for newer cards. Tested on rented Linux machines
  (RTX 5070 12 GB, H100 80 GB): the same model fidelity as our RTX 4070 (mean KL on the quant-challenge golden set
  0.056414 and 0.056407 vs 0.056398; top-1 within 0.02 points), 24 / 24 needles and 20 / 20 exact code copies at 117k
  and 245k tokens, tool calls t1-t4, and no memory spill. CUDA loads only your card's code, so the other cards' code
  costs no VRAM or speed, only download size (Windows zip 679 MB, was 581 MB; Linux tarball 786 MB, was 684 MB).
- **Newer cards** (for example B200, or compute capability 12.1) run the portable code. The driver compiles it at the
  first start and keeps it in its cache: in our test (an RTX 5070 run on that code) the first start took about 1.5
  minutes and later starts about 10 s. The launcher sets a 4 GiB cache for these cards. On Windows this needs a driver
  that supports CUDA 13.4. Not tested by us on such cards.
- **The launcher checks the card** (compute capability) and stops with a clear message on a card the bundle has no code
  for (GTX 10 and older, V100, A100). On a card with less than 12 GB it prints a clearer warning.
- **Linux: the web chat page works** (http://127.0.0.1:8080/); v1.0.4-v1.1.1 Linux engines were built without it.
- A grammar fix from llama.cpp (#32381, Aldehir Rojas): an anchored tool-call trigger pattern no longer falls back to
  a search. Kintsugi's own tool calls use a fixed tag, so they do not change.
- RTX 20 / 30 / 40: no change in output. On our RTX 4070 the v1.1.2 engine gives the same tokens as v1.1 (50 / 50
  prompts) at the same speed.
- Engine source `b9c5b427d`. Windows: `Kintsugi-27B-v1.1.2-win-x64.zip`, sha256 `64c53e48886f95e083f41223296aeb4dc30f8f6630bdca91db6d03d0d190e4e3`. Linux:
  `Kintsugi-27B-v1.1.2-linux-x86_64.tar.gz`, sha256 `46f19175b63e6a425f2f2ff9face014aa1f36f2bba6185bc26203689e756d86a`.

## v1.1.1 (2026-11-21, Linux only)

- **Linux: the engine no longer needs NCCL (libnccl.so.2).** The v1.0.4 and v1.1 Linux engines did not start on a
  machine with only the NVIDIA driver installed (the start stopped with "error while loading shared libraries:
  libnccl.so.2: cannot open shared object file"). The engine is now built without NCCL, and `bin/` also carries the
  OpenMP runtime (`libgomp.so.1`). No other libraries are necessary.
- The model file, the engine source (`5f393adb5`) and the numerics do not change.
- The README in the tarball names the v1.1.1 tarball in its quick start (the v1.1 tarball still said v1.0.4 there).
- Tested on a machine with only the NVIDIA driver (Ubuntu 22.04.5, driver 575.51.03, RTX 4060 Ti 16 GB): it starts and
  passes the release smokes; the output is token-identical to v1.1 on the same machine (50 / 50 prompts).
- Windows: no change. The v1.1.1 release also carries the unchanged v1.1 Windows zip and engine source.
- Linux: `Kintsugi-27B-v1.1.1-linux-x86_64.tar.gz`, sha256 `0d55dd1bc0e943276208ea8343e95d93a42b3f5a025c11c7000c9d95e092d701`.

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
