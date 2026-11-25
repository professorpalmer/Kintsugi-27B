# Kintsugi-27B bundle: user guide (Windows, RTX 20 / 30 / 40 / 50 and H100, 12 GB)

This folder holds the text files of the Windows zip. Download the zip from
[Releases](https://github.com/professorpalmer/Kintsugi-27B/releases/latest). The zip also holds the engine (`bin\`),
which is not in this folder. The changes in each version are in [CHANGELOG.md](CHANGELOG.md).

Kintsugi-27B is Qwen3.8-27B at 2.47 bits per weight. It runs on one 12 GB NVIDIA card with the full 262,144-token
context. Part of the long context is in system RAM, so a long chat works on 12 GB. You can use it for anything, under
Apache-2.0. If you find a problem or have numbers from your card, please open a GitHub issue (the template is at the
end). Pull requests are welcome.

## What you need

- An NVIDIA RTX 20, 30, 40 or 50 series card with 12 GB of VRAM or more, or an H100 / H200. Examples: RTX 2060 12 GB,
  3060 12 GB, 3080 12 GB, 3080 Ti, 3090, 4060 Ti 16 GB, 4070 and up, 4090, 5060 Ti 16 GB, 5070 and up, 5090.
  - The bundle carries code for all of them (RTX 50 and H100 from v1.1.2). CUDA loads only your card's code, so this
    costs no VRAM or speed, only download size.
  - Older cards (GTX 10 and before) and V100 / A100 have no code in the bundle: the launcher says so and stops.
  - Cards with less than 12 GB cannot hold the model.
  - On 16-24 GB cards, the launcher keeps the whole context in VRAM, so long-context speed is much higher.
  - We checked Turing (RTX 20 class, on a card limited to 12 GB), Ampere, Ada, Blackwell (RTX 5070) and Hopper (H100):
    the results are the same within 1 % ([cross-architecture](../research/cross-architecture/README.md)).
- NVIDIA driver 580 or newer (the bundle carries its own CUDA 13 runtime). The launcher stops with a message if the
  driver is too old.
- Windows 10 or 11, 64-bit. 32 GB of system RAM or more is best for very long chats.
- About 11 GB of disk for the model, plus room for saved chats (up to 20 GB, in `slots\`).
- Optional: Python 3.10 or newer, for saved chats (see below). No extra Python packages are needed.

## Download and start

1. Download the bundle zip from the Releases page and unzip it to a folder, for example `C:\Kintsugi`.
2. Download `Kintsugi-27B-12G.gguf` from [Hugging Face](https://huggingface.co/CaryPalmer/Kintsugi-27B). Put it in
   the `models` folder. Its SHA-256 is `0510bae82445dc53dbc7d0d842d8b7e146b8eafce0837e39b157d411e48c8a74`. To check
   it, run `Get-FileHash models\Kintsugi-27B-12G.gguf` in PowerShell.
3. Open PowerShell in the folder and start the server:

   ```
   powershell -ExecutionPolicy Bypass -File start-server.ps1 -Wait
   ```

   The first start takes about a minute. Wait for the line `ready  http://127.0.0.1:8080/v1`.
4. Chat (see "How to chat" below). The server listens only on this computer.
5. To stop it: `powershell -ExecutionPolicy Bypass -File stop.ps1`.

## Faster decode (optional download)

Download `dflash-Qwen3.8-27B-Q4_0.gguf` (1.1 GB) from https://huggingface.co/CaryPalmer/Kintsugi-27B into the `models`
folder. At the next start the launcher uses it (the `spec` line says "DFlash2 draft 7"). On an RTX 4070 (the shipped
settings) decode is about 25-35 % faster at 4k-64k tokens of context, about 23 % faster at 117k, and the same at 245k.
Without the file the launcher uses the model's own MTP drafter, as before, and prints one note. The answers are the same
with either drafter. To choose by hand: `$env:KINTSUGI_DRAFT = 'mtp'` (or `dflash`, `off`).

The drafter is Inco AI's DFlash2 in ggml-org's Q4_0 GGUF, unchanged (Apache-2.0; see NOTICE and CREDITS.md).

## Images (optional download)

Kintsugi-27B can read images with Qwen's vision projector. It is an optional download; the model file does not
change. Download `mmproj-Qwen3.8-27B-Q8_0.gguf` (629,247,008 bytes, SHA-256
`2e968a6af97ce35d8971890b257b9b7edabf20ad91450501fa53162a19ee33eb`) from
https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF and put it in `models\`. The launcher then turns images on (the
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

## Reasoning effort (optional)

The model thinks before it answers. The reasoning effort sets how much it thinks. The default is `medium`. The
model accepts three values: `xhigh`, `medium` and `low`.

- **For all requests:** set `$env:KINTSUGI_EFFORT = 'xhigh'` before the start, or start with
  `start-server.ps1 -Effort xhigh`. The `spec` line shows the effort.
- **For one request:** send `"reasoning_effort": "xhigh"` (or `"chat_template_kwargs": {"reasoning_effort": "xhigh"}`).
  `"reasoning_effort": "none"` turns the thinking off for that request. Any other word (for example `"high"`, which
  some apps send) gets the default of the launcher. The request does not fail; the server log shows a line for it.
- **What we measured:** on 70 tasks of an agent benchmark (AppWorld), `xhigh` gave 3 answers that the task did not
  ask for, `medium` gave 13 (p = 0.013).
- **What it costs:** with `xhigh` the thinking is longer, and each task took about 1.7 times as long. A few more tasks
  stopped early (the model gave up, or the window was full). For this reason the default stays `medium`. We did not
  measure `low`.
- **Saved chats:** the effort changes the start of the prompt. A saved chat is used again only with the same effort.

## How to chat

- **In the browser:** open http://127.0.0.1:8080/ (the chat page that comes with the server).
- **From an app:** point any OpenAI-compatible app at `http://127.0.0.1:8080/v1` (model name: anything; no key).
- **Answer length (`max_tokens`):** with thinking on (the default), a `max_tokens` of 4,096 or more is kept, and the
  thinking of that request ends early enough to leave room for the answer. A smaller value is raised to 24,576,
  because the thinking alone can need more. For short capped answers turn the thinking off (`KINTSUGI_THINK=off`,
  or `"chat_template_kwargs": {"enable_thinking": false}` in the request): then every `max_tokens` is kept.
- **System memory:** the K/V cells that do not fit in VRAM are pinned system RAM (they can not go to the page file),
  and other programs (the desktop, a browser) can hold VRAM that nvidia-smi does not show. Before the start the
  launcher counts both; when the window does not fit, it starts with the largest window that does and prints a
  `memory` line with the numbers. `KINTSUGI_MEMCHECK=0` keeps the full window; `KINTSUGI_CACHE_RAM` sets the
  server's prompt cache in RAM (default 4096 MiB).
- **Agent harnesses** (Claude Code, Codex, Pi, OpenCode): see [harnesses/README.md](../harnesses/README.md) for the
  settings we tested (in the zip, the same text is `HARNESSES.md`).
- **From PowerShell** (a quick test):

  ```
  curl.exe http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello! Who are you?\"}]}'
  ```

**Saved chats:** with Python 3.10 or newer in PATH (no extra packages), the start also runs a small layer in front of
the server. The layer saves a long chat to disk after each answer. After a restart, the next turn of that chat starts
in seconds, not minutes. Without Python, the server runs plain and the start prints a note. Everything else works.

Disk use of saved chats: they go to `slots\` (up to 20 GB by default; past that, the files used longest ago are
removed). A long chat keeps one full save (about 4.7 GB at 245k tokens) and smaller additions for the later turns. Change
the limit with `KINTSUGI_SLOT_CACHE_GB`, move the folder with `KINTSUGI_SLOT_SAVE_PATH`, or turn saving off with
`KINTSUGI_LAYER=0`.

**Settings** are environment variables. For example, set `$env:KINTSUGI_CTX = 131072` before the start. The list is at
the top of [start-server.ps1](start-server.ps1). By default, only this computer can connect.
`$env:KINTSUGI_HOST = '0.0.0.0'` opens the server to your network. If you do that, also set `$env:KINTSUGI_KEY_FILE`
to a file that holds an API key. If you do not, anyone on your network can use your GPU.

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
- One chat at a time (one slot). By default, the server listens on 127.0.0.1 only.
- RTX 50 and H100 were tested on rented Linux machines; the Windows build of that code (CUDA 13.4) passed the same
  fidelity check on the RTX 5070 under Linux. We have not run it on an RTX 50 under Windows ourselves.
- Newer cards (for example B200, or compute capability 12.1) run the engine's portable code: the driver compiles it at
  the first start (about 1.5 minutes in our test on an RTX 5070 run on that code; later starts about 10 s, from the driver's cache). We have not tested these cards.
  On Windows this needs a driver that supports CUDA 13.4.
- On RTX 20 (Turing) cards, drafting (MTP or DFlash2) is not bit-identical to one-token decoding: near ties can round
  the other way in the verify path. The fidelity is the same (our Turing KL check). On RTX 30 / 40 it is bit-identical.

## Files in this folder

| File | What it does |
|---|---|
| [start-server.ps1](start-server.ps1) | The launcher: checks the driver, sizes the K/V cache for your card, starts the server and the layer. |
| [stop.ps1](stop.ps1) | Stops the server and the layer of this folder (it waits for a chat save in progress). |
| [layer/](layer/README.txt) | The saved-chat layer (Python, standard library only). |
| [templates/kintsugi.jinja](templates/kintsugi.jinja) | The chat template the launcher uses (accepts a system or developer message after the first turn). |
| [models/README.txt](models/README.txt) | Where the model file goes. |
| [licenses/](licenses/) | The licences of the engine and its parts. |

Model updates ship as a new .gguf file; the bundle stays the same.

Licences and credits: [LICENSE](../LICENSE), [NOTICE](../NOTICE), [CREDITS.md](../CREDITS.md).
