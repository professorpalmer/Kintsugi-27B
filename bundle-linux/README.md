# Kintsugi-27B v1.0.4 for Linux (x86_64, RTX 20 / 30 / 40, 12 GB)

These are the text files of the Linux tarball; download the tarball from
[Releases](https://github.com/professorpalmer/Kintsugi-27B/releases/latest). The tarball also holds the engine
(`bin/`), the layer, the chat template and the licences, which are in [bundle/](../bundle/README.md) here.

Kintsugi-27B is Qwen3.8-27B at 2.47 bits per weight, built to run on one 12 GB NVIDIA card with the full
262,144-token context. Part of the long context lives in system RAM, so a long chat works on 12 GB. Free to use for
anything, under Apache-2.0. This is the Linux bundle: the same model file, engine features and defaults as the
Windows bundle. Problems or numbers from your card: please open an issue at
https://github.com/professorpalmer/Kintsugi-27B/issues.

## What you need

- An NVIDIA RTX 20, 30 or 40 series card with 12 GB of VRAM or more. RTX 50 (Blackwell) is not built in this release.
- NVIDIA driver 570 or newer. The engine uses the CUDA 12.8 runtime (in `bin/`). On drivers older than the 580
  family, the engine uses another type of pinned system memory for the long context, and writes one WARN line about it.
- Linux x86_64 with glibc 2.34 or newer and the libstdc++ of GCC 12 or newer (the engine needs GLIBC_2.34 and
  GLIBCXX_3.4.30). Ubuntu 22.04 and 24.04 have both.
- `curl` and `nvidia-smi` (it comes with the driver). 32 GB of system RAM or more is best for very long chats.
- Optional: Python 3.10 or newer, for saved chats. No extra Python packages are needed.

## Download and start

1. Download `Kintsugi-27B-v1.0.4-linux-x86_64.tar.gz` from the Releases page and unpack it:

   ```
   tar -xzf Kintsugi-27B-v1.0.4-linux-x86_64.tar.gz
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
- Tested on Ubuntu 22.04 and 24.04 with an RTX 4070 (drivers of the 580 and 610 families). The fallback for 570
  drivers was tested on an RTX 2080 Ti (driver 570.133). Built on Ubuntu 22.04 with GCC 11.

Licences and credits: [LICENSE](../LICENSE), [NOTICE](NOTICE) (the Linux notice), [CREDITS.md](../CREDITS.md).
