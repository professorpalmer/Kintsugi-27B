# Use Kintsugi-27B with a coding harness

Kintsugi-27B works with four common coding harnesses: Claude Code, Codex CLI, Pi and OpenCode. This page gives the
version that we tested, the result, and the minimum settings for each harness.

## Before you start

1. Start the server (see the [bundle guide](../bundle/README.md); in the zip it is `README.md`). Wait for the line
   `ready  http://127.0.0.1:8080/v1`.
2. Use these values in the harness:
   - Address: `http://127.0.0.1:8080` (some harnesses need `/v1` at the end; see each harness below).
   - API key: any text, for example `any-key`. The server does not check the key, unless you set
     `KINTSUGI_KEY_FILE`. If you set a key file, use the key from that file.
   - Model name: any text, for example `kintsugi`. The server has one model.
3. Claude Code and Codex need bundle v1.0.2 or later. Older bundles stop them on the first request with
   "System message must be at the beginning". Pi and OpenCode work with v1.0.1 and later.

## What we tested

We gave each harness the same 6 small tasks in a small Python repository. The tasks were: read a file, fix a function
and run its test, make a new file, search a word in many files, a task of more than 5 tool calls, and a question about
a file of about 42,000 tokens. We ran each harness two times: through the saved-chat layer (the default), and with
the plain server (`$env:KINTSUGI_LAYER = 0`). Windows 11, RTX 4070 12 GB.

| Harness | Version tested | Tasks passed (layer / plain server) | Errors |
|---|---|---|---|
| Claude Code | 2.1.330 | 5 of 6 / 5 of 6 | none |
| Codex CLI | 0.171.0 | 5 of 6 / 5 of 6 | none |
| Pi | 0.73.1 | 5 of 6 / 5 of 6 | none |
| OpenCode | 1.19.6 | 5 of 6 / 5 of 6 | none |

"Errors" counts tool calls that the harness could not read, think tags or tool tags in the visible text, loops, and
turns that read the full chat again. We found none of them.

The same task failed in each harness: "Which files call legacy_connect( ?". The model also gave the file that defines
the function. This is an answer of the model, not a fault of the harness or the server.

## Claude Code

Set these variables before you start Claude Code (PowerShell):

```
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:8080"
$env:ANTHROPIC_API_KEY = "any-key"
$env:ANTHROPIC_DEFAULT_OPUS_MODEL = "kintsugi"
$env:ANTHROPIC_DEFAULT_SONNET_MODEL = "kintsugi"
$env:ANTHROPIC_DEFAULT_HAIKU_MODEL = "kintsugi"
$env:CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = "1"
claude --model kintsugi
```

Claude Code uses the Anthropic API (`/v1/messages`). No login is necessary when `ANTHROPIC_API_KEY` is set. Claude
Code can show the note "unrecognized model": you can ignore it.

## Codex CLI

Codex reads `config.toml` in its home folder (`%USERPROFILE%\.codex\config.toml`). Add these lines:

```
model = "kintsugi"
model_provider = "kintsugi"

[model_providers.kintsugi]
name = "Kintsugi"
base_url = "http://127.0.0.1:8080/v1"
env_key = "KINTSUGI_API_KEY"
wire_api = "responses"
```

Then, in PowerShell:

```
$env:KINTSUGI_API_KEY = "any-key"
codex
```

Codex uses the Responses API (`/v1/responses`). Codex 0.171.0 does not accept `wire_api = "chat"` any more. Codex can
show the note "Model metadata for kintsugi not found": you can ignore it.

## Pi

Pi reads custom models from `%USERPROFILE%\.pi\agent\models.json`. Put this text in that file:

```
{
  "providers": {
    "kintsugi": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "any-key",
      "models": [
        { "id": "kintsugi", "name": "Kintsugi-27B (local)", "reasoning": true, "contextWindow": 262144, "maxTokens": 24576 }
      ]
    }
  }
}
```

Then start Pi with:

```
pi --provider kintsugi --model kintsugi
```

## OpenCode

OpenCode reads `%USERPROFILE%\.config\opencode\opencode.json`. Put this text in that file:

```
{
  "$schema": "https://opencode.ai/config.json",
  "model": "kintsugi/kintsugi",
  "small_model": "kintsugi/kintsugi",
  "provider": {
    "kintsugi": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Kintsugi-27B (local)",
      "options": { "baseURL": "http://127.0.0.1:8080/v1", "apiKey": "any-key" },
      "models": { "kintsugi": { "name": "Kintsugi-27B", "limit": { "context": 262144, "output": 24576 } } }
    }
  }
}
```

Then start OpenCode with `opencode` (the `model` line in the file selects Kintsugi). To run one task without the
interactive screen, use `opencode run -m kintsugi/kintsugi "your task"`.

At its first start, OpenCode downloads the package `@ai-sdk/openai-compatible` from npm.

## If a harness does not work

- Make sure that the server shows `ready` and that the harness uses port 8080.
- Make sure that the bundle is v1.0.2 or later for Claude Code and Codex.
- Look in `logs\` in the bundle folder: `server-8080.log` and `layer-8080.log`.
- Please open a GitHub issue with the harness name, its version, and the last lines of those two logs.
