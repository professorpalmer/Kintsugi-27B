# Kintsugi-27B v1.0.2 (Windows, RTX 20 / 30 / 40, 12 GB)

![Kintsugi-27B](kintsugi-card.png)

Kintsugi-27B is Qwen3.8-27B at 2.47 bits per weight, built to run on one 12 GB NVIDIA card with the full
262,144-token context. Part of the long context lives in system RAM, so a long chat works on 12 GB. Free to use for
anything, under Apache-2.0. Found a problem or have numbers from your card? Please open a GitHub issue (template at
the end); pull requests are welcome.

**Download:** the Windows bundle (engine, launcher, saved-chat layer) is on the
[Releases page](https://github.com/professorpalmer/Kintsugi-27B/releases); the model file is on
[Hugging Face](https://huggingface.co/CaryPalmer/Kintsugi-27B).

## What you need

- An NVIDIA RTX 20, 30 or 40 series card with 12 GB of VRAM or more: for example RTX 2060 12 GB (being verified),
  3060 12 GB, 3080 12 GB, 3080 Ti, 3090, 4060 Ti 16 GB, 4070 and up, 4090. RTX 50 series (Blackwell) is NOT supported
  in this build, by choice: the project targets the cards most people already own. Cards with less than 12 GB can not
  hold the model. On 16-24 GB cards the launcher keeps the whole context in VRAM, so long-context speed is much higher
  there.
- A recent NVIDIA driver (the bundle carries its own CUDA 13 runtime; the driver must support CUDA 13).
- Windows 10 or 11, 64-bit. 32 GB of system RAM or more is best for very long chats.
- About 11 GB of disk for the model, plus room for saved chats (up to 20 GB, in `slots\`).
- Optional: Python 3.10 or newer, for saved chats (see below). No extra Python packages are needed.

## Download and start

1. Download the bundle zip from the Releases page and unzip it to a folder, for example `C:\Kintsugi`.
2. Download `Kintsugi-27B-12G.gguf` from https://huggingface.co/CaryPalmer/Kintsugi-27B and put
   it in the `models` folder. Its SHA-256 is
   `0510bae82445dc53dbc7d0d842d8b7e146b8eafce0837e39b157d411e48c8a74`; check it in PowerShell with
   `Get-FileHash models\Kintsugi-27B-12G.gguf`.
3. Open PowerShell in the folder and start the server:

   ```
   powershell -ExecutionPolicy Bypass -File start-server.ps1 -Wait
   ```

   The first start takes about a minute. Wait for the line `ready  http://127.0.0.1:8080/v1`.
4. Chat (see "How to chat" below). The server listens only on this computer.
5. To stop it: `powershell -ExecutionPolicy Bypass -File stop.ps1`.

## How to chat

- **In the browser:** open http://127.0.0.1:8080/ (the chat page that comes with the server).
- **From an app:** point any OpenAI-compatible app at `http://127.0.0.1:8080/v1` (model name: anything; no key).
- **Agent harnesses** (Claude Code, Codex, Pi, OpenCode): see `HARNESSES.md` for the settings we tested.
- **From PowerShell** (a quick test):

  ```
  curl.exe http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello! Who are you?\"}]}'
  ```

Saved chats: with Python 3.10 or newer in PATH (no extra packages), the start also runs a small layer in front of
the server. It saves a long chat to disk after each answer, so after a restart the next turn of that chat starts in
seconds, not minutes. Without Python, the server runs plain and the start prints a note; everything else works.

Settings are environment variables, for example `$env:KINTSUGI_CTX = 131072` before the start. The list is at the top
of `start-server.ps1`. By default only this computer can connect. `$env:KINTSUGI_HOST = '0.0.0.0'` opens it to your
network: if you do that, also set `$env:KINTSUGI_KEY_FILE` to a file holding an API key, or anyone on your network can
use your GPU.

## What to try

1. **A long document.** Paste a long text (a book chapter, a long report, a code base: 50k to 200k tokens) and ask
   questions whose answers are in the middle or near the start.
2. **A multi-turn chat with a restart.** Have a long chat (a few turns), stop the server, start it again, and continue
   the same chat. With saved chats on, the next answer should start quickly.
3. **A coding task.** Ask for a function or a small program, run it, and ask for fixes.
4. **A tool or agent task**, if you have a harness that uses tools (function calls): any task you would give a strong
   local model.

## What to send back

Copy this into a GitHub issue and fill it in:

```
Card and VRAM:
Driver version (nvidia-smi, top line):
Windows version:
System RAM:
Python for saved chats (yes / no):

From the start output:
  the "kv" line (kv q4_0, tiered: cells ...):
  the "warm" line (VRAM ... MiB):
  any "note" or "SPILL" line:

Speed (logs\server-8080.log has lines like "eval time = ... tokens per second"):
  a short chat (under 4k tokens):        tok/s
  around 32k tokens of context:          tok/s
  around 128k tokens (if you tried):     tok/s
  the longest you tried (how many tokens, how fast):

Errors (the exact text, and the last 30 lines of logs\server-8080.log):

How the answers felt (long-document answers, chat after a restart, code, tools):
```

## Known limits

- Decode at a very long context is slow: about 30 tok/s at 245k tokens on an RTX 4070, and slower on an RTX 3060
  (it has less memory bandwidth). Short chats are much faster.
- The first read of a very long prompt takes a while: about 7 minutes for 245k tokens on an RTX 4070. Later turns of
  the same chat read only the new text.
- Very long reasoning answers can be less faithful than short ones. If an answer goes off track, ask again or ask for a
  shorter answer.
- One chat at a time (one slot). The server listens on 127.0.0.1 only.
- RTX 50 series (Blackwell) is not built in this release, by choice.

## Changes

- v1.0.2: works with Claude Code and OpenAI Codex (a chat template fix for system or developer messages after the
  first one). Tested with Claude Code, Codex, Pi and OpenCode: see HARNESSES.md
- v1.0.1: the layer streams /v1/messages and /v1/responses live (Claude Code, Codex); API-key headers forwarded;
  stop.ps1 waits for an in-flight chat save.

Model updates ship as a new .gguf file; the bundle stays the same.

Licences and credits: `LICENSE`, `NOTICE`, `CREDITS.md`.
