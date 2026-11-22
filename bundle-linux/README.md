# Kintsugi-27B v1.1.2 for Linux (x86_64, RTX 20 / 30 / 40 / 50 and H100, 12 GB)

These are the text files of the Linux tarball; download the tarball from
[Releases](https://github.com/professorpalmer/Kintsugi-27B/releases/latest). The tarball also holds the engine
(`bin/`), the layer, the chat template and the licences, which are in [bundle/](../bundle/README.md) here.

Kintsugi-27B is Qwen3.8-27B at 2.47 bits per weight, built to run on one 12 GB NVIDIA card with the full
262,144-token context. Part of the long context lives in system RAM, so a long chat works on 12 GB. Free to use for
anything, under Apache-2.0. This is the Linux bundle: the same model file, engine features and defaults as the
Windows bundle. Problems or numbers from your card: please open an issue at
https://github.com/professorpalmer/Kintsugi-27B/issues.

## What you need

- An NVIDIA RTX 20, 30, 40 or 50 series card with 12 GB of VRAM or more, or an H100 / H200. CUDA loads only your
  card's code, so the code for the other cards costs no VRAM or speed. Older cards (GTX 10 and before) and V100 / A100
  have no code in the bundle: the launcher says so and stops.
- NVIDIA driver 570 or newer. The engine uses the CUDA 12.8 runtime (in `bin/`). On drivers older than the 580
  family, the engine uses another type of pinned system memory for the long context, and writes one WARN line about it.
- Linux x86_64 with glibc 2.34 or newer and the libstdc++ of GCC 12 or newer (the engine needs GLIBC_2.34 and
  GLIBCXX_3.4.30). Ubuntu 22.04 and 24.04 have both.
- `curl` and `nvidia-smi` (it comes with the driver). 32 GB of system RAM or more is best for very long chats.
- No other libraries are necessary: the bundle carries the CUDA runtime, cuBLAS and the OpenMP runtime (libgomp) in
  `bin/`. From v1.1.1 the engine does not need NCCL (libnccl.so.2); v1.0.4 and v1.1 did not start without it.
- Optional: Python 3.10 or newer, for saved chats. No extra Python packages are needed.

## Download and start

1. Download `Kintsugi-27B-v1.1.2-linux-x86_64.tar.gz` from the Releases page and unpack it:

   ```
   tar -xzf Kintsugi-27B-v1.1.2-linux-x86_64.tar.gz
   cd Kintsugi-27B
   ```

2. Download `Kintsugi-27B-12G.gguf` from https://huggingface.co/CaryPalmer/Kintsugi-27B into `models/`, and check it:

   ```
   sha256sum models/Kintsugi-27B-12G.gguf
   # 0510bae82445dc53dbc7d0d842d8b7e146b8eafce0837e39b157d411e48c8a74
   ```

3. Start the server and wait for the line `ready  http://127.0.0.1:8080/v1` (the first start takes about a minute):

   ```
   ./start-server.sh --wait
   ```

4. Chat (below). The server listens only on this computer.
5. Stop it: `./stop.sh`.

## Faster decode (optional download)

Download `dflash-Qwen3.8-27B-Q4_0.gguf` (1.1 GB) from https://huggingface.co/CaryPalmer/Kintsugi-27B into the `models/`
folder. At the next start the launcher uses it (the `spec` line says "DFlash2 draft 7"). On an RTX 4070 (the shipped
settings) decode is about 25-35 % faster at 4k-64k tokens of context, about 23 % faster at 117k, and the same at 245k.
Without the file the launcher uses the model's own MTP drafter, as before, and prints one note. The answers are the same
with either drafter. To choose by hand: `KINTSUGI_DRAFT=mtp ./start-server.sh --wait` (or `dflash`, `off`).

The drafter is Inco AI's DFlash2 in ggml-org's Q4_0 GGUF, unchanged (Apache-2.0; see NOTICE and CREDITS.md).

## Images (optional download)

Kintsugi-27B can read images with Qwen's vision projector. It is an optional download; the model file does not
change. Download `mmproj-Qwen3.8-27B-Q8_0.gguf` (629,247,008 bytes, SHA-256
`2e968a6af97ce35d8971890b257b9b7edabf20ad91450501fa53162a19ee33eb`) from
https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF and put it in `models/`. The launcher then turns images on (the
`vision` line). Two modes (`KINTSUGI_VISION`; the default `auto` picks by the card):

- **cpu** (auto on cards under 16 GB): the projector runs on the processor. No VRAM cost: every K/V cell stays in
  VRAM. An image takes 5-25 s to read (about 5 s for a small screenshot, 25 s for a 2,048-token photo).
- **gpu** (auto on cards with 16 GB or more): about 1.3 GB of VRAM. An image takes 1-3 s. On a 12 GB card this
  leaves about 61,000 K/V cells in VRAM instead of 144,000 (the window stays 262,144 tokens; long conversations
  decode more slowly); a 16 GB card keeps the whole window in VRAM.
- `KINTSUGI_VISION=off`: images off with the file present; `gpu` / `cpu` force a mode.
- With images on, drafting is MTP + lookup: the DFlash2 drafter does not work with images yet (the launcher
  switches to MTP; `KINTSUGI_DRAFT=dflash` with images on stops with a message).
- Saved conversations: a request with an image is not saved; with images on, saves are full saves (no small delta
  saves).
- `KINTSUGI_IMAGE_MAX_TOKENS` (default 2048): a larger image is scaled down to this many tokens.

Images work in chat requests (OpenAI and Anthropic formats) and in tool results (for example a screenshot). The
built-in web page is not tested with images.

## How to chat

- **In the browser:** open http://127.0.0.1:8080/.
- **From an app:** point any OpenAI-compatible app at `http://127.0.0.1:8080/v1` (model name: anything; no key).
- **Coding harnesses** (Claude Code, Codex, Pi, OpenCode): see [harnesses/README.md](../harnesses/README.md) for the settings we tested (in the tarball: `HARNESSES.md`). Pi needs
  Node.js 20 or newer; Ubuntu 24.04's own Node 18 fails at Pi's start with a SyntaxError.
- **A quick test:**

  ```
  curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Hello! Who are you?"}]}'
  ```

Saved chats: with Python 3.10 or newer, the start also runs a small layer in front of the server. It saves a long chat
to disk after each answer, so after a restart the next turn of that chat starts in seconds, not minutes.

Settings are environment variables, for example `KINTSUGI_CTX=131072 ./start-server.sh --wait`. The list is at the
top of `start-server.sh`. By default only this computer can connect; `KINTSUGI_HOST=0.0.0.0` opens it to your network:
then also set `KINTSUGI_KEY_FILE` to a file holding an API key. Logs are in `logs/`.

## Known limits

- RTX 50 and H100 were tested on rented machines (RTX 5070 12 GB; H100 80 GB): the same model fidelity (KL) as our
  RTX 4070, full recall at 117k and 245k tokens. Since v1.1.2 the browser chat page also works on Linux.
- Newer cards (for example B200, or compute capability 12.1) run the engine's portable code: the driver compiles it at
  the first start (about 1.5 minutes in our test on an RTX 5070 run on that code; later starts about 10 s, from the driver's cache). We have not tested these cards.
- Decode at a very long context is slower: about 31 tok/s at 245k tokens on an RTX 4070. Short chats are much faster.
- The first read of a very long prompt takes about 7 minutes for 245k tokens on an RTX 4070; later turns read only the
  new text.
- One chat at a time (one slot).
- On RTX 20 (Turing) cards, drafting (MTP or DFlash2) is not bit-identical to one-token decoding: near ties can round
  the other way in the verify path. The fidelity is the same (our Turing KL check). On RTX 30 / 40 it is bit-identical.
- Tested with an RTX 4070 on Ubuntu 22.04 (drivers 580.178 and 595.91) and 24.04 (driver 615.71). The fallback for
  570 drivers was tested on an RTX 2080 Ti (driver 570.133). v1.1.1 was also started on a machine with only the NVIDIA
  driver installed (Ubuntu 22.04.5, driver 575.51.03, RTX 4060 Ti 16 GB, no CUDA toolkit, no NCCL: it starts and passes the release
  smokes; the output is token-identical to v1.1 on the same machine, 50 / 50 prompts). Built on Ubuntu 22.04 with GCC 11, without NCCL.

Licences and credits: [LICENSE](../LICENSE), [NOTICE](NOTICE) (the Linux notice), [CREDITS.md](../CREDITS.md).
