# Kintsugi-27B bundle: changelog

The model file (`Kintsugi-27B-12G.gguf`, sha256 `0510bae82445dc53dbc7d0d842d8b7e146b8eafce0837e39b157d411e48c8a74`) is
the same for every version below: when a new bundle comes out, download only the new zip. Each release on the
[Releases page](https://github.com/professorpalmer/Kintsugi-27B/releases) also has the engine source
(`Kintsugi-27B-v1-engine-src-c07efbec5.tar.gz`, sha256 `0e8f2a1c3c293ce83ca9892e2ba61a3b3beaaa8870ecc907f2a047092d6391b7`).

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
