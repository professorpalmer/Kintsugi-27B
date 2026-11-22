# Kintsugi-27B v1.1.1 for Linux (x86_64, RTX 20 / 30 / 40, 12 GB)

These are the text files of the Linux tarball; download the tarball from
[Releases](https://github.com/professorpalmer/Kintsugi-27B/releases/latest). The tarball also holds the engine
(`bin/`), the layer, the chat template and the licences, which are in [bundle/](../bundle/README.md) here.

Kintsugi-27B is Qwen3.8-27B at 2.47 bits per weight, built to run on one 12 GB NVIDIA card with the full
262,144-token context. Part of the long context lives in system RAM, so a long chat works on 12 GB. Free to use for
anything, under Apache-2.0. This is the Linux bundle: the same model file, engine features and defaults as the
Windows bundle. Problems or numbers from your card: please open an issue at
https://github.com/professorpalmer/Kintsugi-27B/issues.

## What you need

- An NVIDIA RTX 20, 30 or 40 series card with 12 GB of VRAM or more. RTX 50 (Blackwell) and H100 (Hopper) are not built in this release yet: coming soon.
- NVIDIA driver 570 or newer. The engine uses the CUDA 12.8 runtime (in `bin/`). On drivers older than the 580
  family, the engine uses another type of pinned system memory for the long context, and writes one WARN line about it.
- Linux x86_64 with glibc 2.34 or newer and the libstdc++ of GCC 12 or newer (the engine needs GLIBC_2.34 and
  GLIBCXX_3.4.30). Ubuntu 22.04 and 24.04 have both.
- `curl` and `nvidia-smi` (it comes with the driver). 32 GB of system RAM or more is best for very long chats.
- No other libraries are necessary: the bundle carries the CUDA runtime, cuBLAS and the OpenMP runtime (libgomp) in
  `bin/`. From v1.1.1 the engine does not need NCCL (libnccl.so.2); v1.0.4 and v1.1 did not start without it.
- Optional: Python 3.10 or newer, for saved chats. No extra Python packages are needed.

## Download and start

1. Download `Kintsugi-27B-v1.1.1-linux-x86_64.tar.gz` from the Releases page and unpack it:

   ```
   tar -xzf Kintsugi-27B-v1.1.1-linux-x86_64.tar.gz
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
