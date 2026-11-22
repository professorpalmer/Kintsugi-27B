# Kintsugi-27B on Apple Silicon

Kintsugi-27B runs on a Mac with Apple Silicon: chat in the terminal, or an OpenAI-compatible server for coding agents
(OpenCode, Pi and other clients). It uses escha-mlx (the MLX runtime of Escha Labs) with a small patch.

Tested on one Mac only: M5 Max, 40 GPU cores, 48 GB. 16 GB Macs and M1 to M4: untested on real hardware.

## Install

```bash
git clone https://github.com/professorpalmer/Kintsugi-27B.git
cd Kintsugi-27B/mac
./run.sh install
```

`./run.sh install` checks the Mac, makes a local Python environment, gets escha-mlx and applies the patch, downloads
`Kintsugi-27B-12G.gguf` (10.5 GB, sha256 checked) and converts it (byte checks against the GGUF). It takes 15-40
minutes, mostly the download. You can stop it and run it again: it skips the complete steps.

It needs Apple Silicon, macOS 14 or later, Python 3.10 to 3.13, about 25 GB of free disk and 24 GB of RAM or more
(16 GB: the experimental 16 GB profile, see below). The table under "Measure your Mac" has the details.

On a 16 GB Mac, `./run.sh chat` and `./run.sh serve` have a context of about 6,000 tokens (the 16 GB profile with
MTP; simulated on a larger Mac, untested on real hardware). That is enough for a chat, not for a coding agent: the
system prompt of OpenCode alone is about 18,000 tokens. Use a Mac with 24 GB or more for coding agents.

## Use

### Chat in the terminal

```bash
./run.sh chat
```

Type a message. Commands: `/think on|off` (reasoning, default on), `/effort low|medium|high`, `/show` (show or hide
the reasoning), `/reset` (new chat), `/quit`. Each turn reads only the new tokens: the earlier turns stay in the cache.

### Server for coding agents

```bash
./run.sh serve
```

The server listens on `http://127.0.0.1:8080/v1` (this Mac only). The model name can be any text, for example
`kintsugi`. Any API key works unless you set one.

| endpoint | what |
|---|---|
| `POST /v1/chat/completions` | streamed or not; `tools` and `tool_calls`; reasoning in `reasoning_content` |
| `GET /v1/models` | the model and its context length |
| `GET /health` | status, requests, malformed tool calls, errors |

Request fields: `messages`, `tools`, `stream`, `stream_options.include_usage`, `max_tokens` /
`max_completion_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stop`, `reasoning_effort` (low / medium /
high, or `none`), `chat_template_kwargs` (`enable_thinking`, `reasoning_effort`).

Defaults (the same as the Windows bundle): reasoning effort medium, 20,480 reasoning tokens (then the model must
answer), 24,576 output tokens, temperature 1.0, top_p 0.95, top_k 20. MTP (multi-token prediction) is on. If the
context has less room, the output limit is the room, and the reasoning budget leaves 2,048 tokens for the answer.

| setting | default | what |
|---|---|---|
| `KINTSUGI_PORT` | `8080` | port |
| `KINTSUGI_HOST` | `127.0.0.1` | address. Another address lets other machines use this Mac's GPU: set a key too |
| `KINTSUGI_KEY_FILE` | none | file that holds an API key; clients must send `Authorization: Bearer <key>` |
| `KINTSUGI_EFFORT` | `medium` | default reasoning effort |
| `KINTSUGI_MAX_CTX` | the largest that fits | a lower context limit in tokens (it cannot raise the limit) |

One request runs at a time. Others wait. The server keeps the caches of the last conversation: when a request starts
with the previous prompt (the next turn of a chat or of a coding agent), only the new tokens are read. It also keeps
snapshots of the first 16k tokens (8 snapshots on 40 GB or more, 4 below, none in the 16 GB profile). A new
conversation that starts with the same text, for example a coding agent's next session with the same system prompt
and tools, reads only the part after the shared start. `KINTSUGI_SNAPSHOTS=0` turns the snapshots off.

### OpenCode

Put this in `~/.config/opencode/opencode.json` (or in `opencode.json` in your project):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "kintsugi/kintsugi",
  "small_model": "kintsugi/kintsugi",
  "provider": {
    "kintsugi": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Kintsugi-27B (local)",
      "options": { "baseURL": "http://127.0.0.1:8080/v1", "apiKey": "any-key" },
      "models": { "kintsugi": { "name": "Kintsugi-27B", "limit": { "context": 131072, "output": 24576 } } }
    }
  }
}
```

Set `context` to the context that `./run.sh serve` prints at start (or less).

### Other clients

Any client for the OpenAI Chat Completions API works: base URL `http://127.0.0.1:8080/v1`, model `kintsugi`.

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "kintsugi", "messages": [{"role": "user", "content": "Say hello."}]}'
```

The client must send `reasoning_content` of earlier turns back (OpenCode does). Else each turn reads the whole chat
again.

## Speed

Release table, measured on a clean clone of this kit on M5 Max, 40 GPU cores, 48 GB, macOS 26.4 (int8 prefill and
flash attention on, 2026-11-22). Other Macs: not measured.

| context | prompt read | output, MTP off | output, MTP on | peak MLX memory |
|---|---|---|---|---|
| 4k | 1,127 tok/s (3.6 s) | 38.0 tok/s | 75.8 tok/s | 14.9 GB |
| 32k | 821 tok/s (39.9 s) | 32.3 tok/s | 48.0 tok/s | 16.8 GB |
| 259k | 299 tok/s (866 s) | 13.5 tok/s | 16.4 tok/s | 31.7 GB |

- Fidelity: KL on 24 documents, Mac 0.07782 against the CUDA engine's 0.07785; same top token 89.0 % on both.
- Recall: 24 of 24 needles at 259,265 tokens.
- MTP speed depends on the text (42-90 tok/s in the short chat checks). The MTP-on column is a coding question after
  the filler text, thinking off, greedy, 256 tokens (`kit/bench_depth.py`).

Earlier measurements with `./run.sh serve`, before the int8 prefill:

| case (M5 Max, 40 GPU cores, 48 GB) | value |
|---|---|
| output speed, chat prompt of 400-600 tokens | 55-77 tok/s |
| output speed, OpenCode session at 18k tokens of context | 38-63 tok/s |
| first prompt of the first OpenCode session (18k tokens) | 25-31 s |
| first prompt of the next OpenCode sessions (16,384 tokens shared) | 2.9-3.1 s |
| next turn of an OpenCode session (90-734 new tokens) | 0.8-1.8 s to read the new tokens |
| first prompt of 257,241 tokens (the largest context on 48 GB) | 1,043 s (247 tok/s) |
| next turn of that chat (36 new tokens) | 1.5-2.2 s; 24 of 24 needles found |

To measure your own Mac, see "Measure your Mac" below.

## Measure your Mac

The test kit measures Kintsugi-27B on your Mac and puts the results in one zip file. Run it if you want to help,
then send the zip back.

The kit does these steps:

1. It checks the Mac (macOS, Apple Silicon, RAM, free disk).
2. It installs MLX in a local Python environment. It does not change your system Python.
3. It gets escha-mlx (the MLX runtime of Escha Labs) and applies a small patch.
4. It downloads `Kintsugi-27B-12G.gguf` (10.5 GB) and checks its sha256.
5. It converts the GGUF into the escha-mlx form. Then it compares the converted files with the GGUF, byte by byte.
6. It runs the checks: escha-mlx's own kernel tests, KL against the BF16 reference, a short chat test, the cache reuse of
   `./run.sh chat`, the server checks (chat and tool calls through HTTP), decode and prefill speed at 4k / 32k / the largest context that fits, and 24 needles at that largest context.
7. It puts all results in one zip file: `results-<chip>-<RAM>GB.zip`. Send that file back.

### One command

```bash
./run.sh
```

You can stop the kit at any time. Run `./run.sh` again: it skips the steps that are complete (the download resumes).

### What you need

| item | value |
|---|---|
| Mac | Apple Silicon (M1 to M5), macOS 14 or later |
| RAM | 24 GB or more for the full tests. With 16 GB, the kit uses the experimental 16 GB profile (see below). Between 16 and 24 GB, the kit stops (`KINTSUGI_PROFILE=16gb ./run.sh` uses the 16 GB profile). Below 16 GB, the kit stops (`KINTSUGI_FORCE=1 ./run.sh` tries anyway). |
| free disk | about 25 GB (GGUF 10.5 GB + converted model 10.6 GB + Python packages) |
| Python | 3.10 to 3.13. macOS's own `python3` is 3.9: if the kit stops here, do `brew install python@3.12` or use the python.org installer. |
| network | about 11 GB of downloads (Hugging Face, GitHub, PyPI) |
| power | Connect the charger. Close large apps (browsers with many tabs, games, virtual machines). |

### Time

| step | estimate for a 24 GB Mac (M4 / M5 class), not measured |
|---|---|
| download + sha256 | 10-30 min (it depends on your network) |
| convert + byte checks | 5-10 min |
| escha-mlx tests | about 5 min |
| KL (24 documents, about 20,000 tokens) | 10-20 min |
| chat test (without and with MTP) | about 5 min |
| speed 4k + 32k | 10-20 min |
| largest context + 24 needles | 30-60 min (about 100k tokens on 24 GB) |
| **total** | **about 1.5 to 3 hours** |

On a Mac with 48 GB or more, the largest context is 262k tokens. Its prefill alone can take 1 to 2 hours. To use a
smaller limit, set it before you start: `KINTSUGI_MAX_CTX=131072 ./run.sh`.

## The 16 GB profile

This profile is experimental. Not yet tested on a real 16 GB Mac. The memory cap (11.45 GB) is recalled from
llama.cpp logs, not measured.

On a Mac with 16 GB of RAM or less, the kit uses the 16 GB profile by itself and prints a note. On other Macs, set
`KINTSUGI_PROFILE=16gb` to use it. The weights are the same: nothing is quantized again.

- The token embedding stays in the model file, memory-mapped on the CPU side (`ESCHA_MLX_EMBED_CPU=1`). Only the
  rows of the current tokens are read. This removes 1.59 GB from the GPU. The output is the same, bit for bit
  (step `embed` checks all 248,320 rows).
- The kit's context is 16k tokens if it fits under the GPU working set, else 8k (chat and serve: also 6k, because the
  MTP head needs 0.32 GB more; see above). The kit does not test 32k or more.
- Prefill runs in steps of 128 tokens (a step of 512 needs 0.7 GB more GPU memory, which a 16 GB Mac does not
  have at 8k). The MLX buffer cache is limited to 0.25 GB.
- The fast prefill paths stay off: `ESCHA_MLX_DENSE_RECON=0` (decode-once GEMM: 0.62 GB more transient at step
  128). So the int8 prefill GEMM (on by default on M5 and later) does not run either. Flash attention
  (`ESCHA_MLX_ATTN_FLASH=auto`) is on for M5 and later and off by itself on M1 to M4. One K/V growth step is at
  most 4,096 positions (`ESCHA_MLX_KV_GROW_MAX=4096`, 256 MB).
- The kit starts Python with `MallocLargeCache=0`. Without it, 1.8 GB of freed load buffers stay in the process
  and a 16 GB Mac would swap them.
- If the model does not fit, the speed step stops with a message that gives the numbers.
- To simulate a smaller Mac on a larger one: `KINTSUGI_PROFILE=16gb KINTSUGI_CAP_GB=11.45 ./run_checks.sh`.

On a real 16 GB Mac, the KL step is the step most likely to swap: in the simulation its process footprint was
11.96 GB (MLX peak 10.89 GB), above the 11.45 GB cap.

Before the run on a 16 GB Mac, close other apps. In the 16 GB profile the model needs about 9.2 GB of GPU memory
(9.5 GB with the MTP head of chat and serve), and macOS gives the GPU about 10.7 GB on a 16 GB Mac (simulated).

## Safety

- The kit does not use `sudo`. It does not change system settings.
- All files stay in this folder: `.venv/`, `escha-mlx/`, `models/`, `results/`.
- The Metal memory limit: macOS gives the GPU about 75 % of the RAM. If more memory would let the kit test a longer
  context, the kit prints a command (`sudo sysctl iogpu.wired_limit_mb=...`). The kit never runs it. It is optional.
  If you run it yourself before `./run.sh`, it stays in effect until the next restart.
- To remove everything: delete this folder.

## What runs where

| step | Mac only? | tested before release |
|---|---|---|
| `convert.py` and its byte checks (`tools/check.py` G1, G3, G4, G5) | no: any OS with Python and numpy | yes, on Windows (see `evidence/`) |
| G2: decode of 26 projections by escha-mlx's NumPy reference against our engine's own CPU code (`tools/engine_escha.c`) | no: Windows only (MSVC) | yes, on Windows: all weights exact |
| KL scorer arithmetic (`kit/kl_mini.py --self-test`) | no | yes |
| needle document and chat template (`kit/common.py`) | no | yes (the codes equal those of our CUDA runs) |
| `setup.sh`, escha-mlx tests, KL with the model, chat, speed, needles | **yes, Mac only** | **no: first run is on your Mac** |
| the patched escha-mlx (per-group head kernel, 4-bit MTP loader) | **yes, Mac only** | syntax only; its tests run in step "tests" |

## The files

| path | what it is |
|---|---|
| `run.sh` | `install`, `chat`, `serve`; with no argument the kit: setup, convert, checks, zip |
| `kit/serve.py`, `kit/chat.py`, `kit/userapi.py` | the server, the terminal chat, and their engine and output parser |
| `templates/kintsugi.jinja` | the chat template (the same as the Windows bundle) |
| `kit/serve_check.py`, `kit/check_session.py`, `kit/serve_needles.py` | checks of the server and of the cache reuse |
| `setup.sh` | machine check, venv, pinned MLX (`mlx==0.32.0`, `mlx-lm==0.31.3`), escha-mlx at `69f7218` + patch, download, sha256 |
| `convert.py` | GGUF to escha-mlx form (`tools/convert.py`), then the byte checks (`tools/check.py`) |
| `run_checks.sh` | the Mac checks (`kit/*.py`), then the results zip |
| `patches/escha-mlx-69f7218-kintsugi.patch` | our change to escha-mlx: per-group Q8 scales for the head and embedding, a per-group head kernel, a 4-bit MTP loader, the session cache (`mtp_session.py`), tests |
| `template/` | config and tokenizer files of the Escha W2 export (no weights) |
| `golden-mini/` | 24 documents of the BF16 reference (top 512 + tail per token) and our CUDA engine's KL on the same documents |
| `evidence/` | the Windows results of the byte checks |
| `evidence/flash-gate-*.json` | the paired KL gate for flash attention (100 public golden documents, `kit/kl_gate.py`; build the set with `tools/make_golden_gate.py`) |
| `evidence/int8-gate-*.json` | the paired KL gate for the int8 prefill GEMM (100 public golden documents): first and last 4 layers on f16 at step 2048 and step 128 (both pass), all layers in int8 (fails) |

## What the conversion does

Kintsugi-27B keeps its weights in the Escha W2 trellis code (EXL3 codebook, Hadamard transforms). escha-mlx runs that
code on the GPU with its own Metal kernels. The conversion copies the coded weights. It does not quantize again.

- 400 coded projections: the bytes are copied. For the 3 linear-attention projections, the conversion undoes a
  block order that our llama.cpp files use. A byte check proves the round trip (400 of 400 equal).
- Embedding and head: Q8_0 (one scale per 32 weights). They become MLX 8-bit with group 32. Each value is equal.
- MTP head: Q4_0. It becomes MLX 4-bit with group 32. Each value is equal (some zeros change sign: -0 becomes +0).
- Norms and small linear-attention tensors: the engine transforms are undone and stored as f16. The check (G3) has
  one rule per class, in `tools/check.py`:
  - `ssm_a = -exp(A_log)`: at most 2 ulp. numpy's float32 `exp` gives other bits on ARM than on x86 (where the GGUF
    was made). On Windows the result is 0 ulp; on an M5 Max it is at most 2 ulp.
  - conv1d: 1,508 values are not equal. All are in the f16 subnormal range, with an error of at most 3e-8. Escha's
    own export has the same f16 form.
  - all other classes: bit-exact.

## Credits and licences

- **Escha Labs**: escha-mlx, the MLX runtime and Metal kernels this kit uses (Apache-2.0,
  https://github.com/EschaLabs/escha-mlx). Our patch keeps its licence and adds a change notice to each changed file.
  The files in `template/` come from the Escha W2 export (Apache-2.0).
- **Qwen team (Alibaba)**: Qwen3.8-27B, the base model and tokenizer (Apache-2.0).
- **turboderp (exllamav3)**: the EXL3 trellis codec (MIT).
- **StillDeadcode**: the Qwen3.8-27B quant challenge. `golden-mini/` is a subset of its BF16 reference
  (CC BY 4.0, https://huggingface.co/datasets/StillDeadcode/qwen3.8-27b-quant-challenge). We selected 24 documents
  and changed nothing else. The KL arithmetic follows its `score.py`.
- **The ggml authors and Ajay9o9 (llama.cpp-escha)**: `tools/engine_escha.c` copies the Escha op's CPU code from our
  llama.cpp fork (MIT, see `licenses/llama.cpp-MIT.txt`).
- **Apple**: MLX and mlx-lm (MIT).

Our own code in this repository: Apache-2.0 (`LICENSE`).
