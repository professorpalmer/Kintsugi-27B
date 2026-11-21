# Kintsugi-27B

Kintsugi-27B is Qwen3.8-27B at 2.47 bits per weight, made to run on one 12 GB NVIDIA card with the full
262,144-token context. It comes as a model file and a Windows bundle (engine, launcher, saved-chat layer).

Built on: the **L0xRE-V2 bit allocation** by YourHighnessLA (used unchanged), the **Escha W2 format** by Escha Labs,
and the **EXL3 trellis codec** by turboderp (exllamav3, after QTIP). Full list: [CREDITS.md](CREDITS.md).

**What is special**

- **2.47 bits per weight**: 27 billion weights in a 10.5 GB file (EXL3 trellis code).
- **The full 262,144-token context on 12 GB**: the tiered K/V cache keeps about the first 144k tokens in
  VRAM (about 90k with the optional drafter) and the rest in system RAM.
- **Optional faster decode (v1.1)**: with the DFlash2 drafter file (1.1 GB, Hugging Face), about 25-35 % faster
  decode at 4k-64k tokens and about 23 % at 117k (RTX 4070, the shipped settings); the same greedy outputs on RTX
  30 / 40.
- **Saved chats**: a saved 245k-token chat restores in 2.8 s, so a long chat continues after a restart.
- **Gated on tasks, not only on KL**: every candidate bit map had to pass long-context recall, long-answer and agent
  tests. Our maps with lower KL failed them and did not ship ([the study](research/bit-allocation/README.md)).
- **Works with coding harnesses**: Claude Code, Codex, Pi and OpenCode.
- **Runs on RTX 20, 30 and 40 cards** with 12 GB or more.
- **Fully open**: Apache-2.0 for the model and our files, MIT for the engine. The engine source is in every release.
  The NVIDIA CUDA DLLs in the zip are NVIDIA's (see [NOTICE](NOTICE)).
- **Head-to-head on one 12 GB card**: against Inkling 2 Mini, AppWorld 302 vs 283 of 336 and 4x the context
  ([details](benchmarks/README.md#head-to-head-inkling-2-mini-on-the-same-12-gb-card)).

![Kintsugi-27B](kintsugi-card.png)

## Quick start (Windows)

1. Download the zip from [Releases (latest)](https://github.com/professorpalmer/Kintsugi-27B/releases/latest) and
   unzip it, for example to `C:\Kintsugi`.
2. Download `Kintsugi-27B-12G.gguf` from [Hugging Face](https://huggingface.co/CaryPalmer/Kintsugi-27B).
3. Put the file in the `models` folder of the bundle. Optional: also download `dflash-Qwen3.8-27B-Q4_0.gguf` from the
   same page into `models` for faster decode (the launcher uses it when it is there).
4. In PowerShell, in that folder: `powershell -ExecutionPolicy Bypass -File start-server.ps1 -Wait`. Wait for the
   line `ready  http://127.0.0.1:8080/v1`.
5. Open http://127.0.0.1:8080/ in a browser, or point an OpenAI-compatible app or a coding harness at
   `http://127.0.0.1:8080/v1`.

You need an NVIDIA driver 580 or newer. The full guide is in [bundle/README.md](bundle/README.md).

**Linux (x86_64):** download `Kintsugi-27B-v1.1-linux-x86_64.tar.gz` from the same Releases page, then
`tar -xzf` it, put the model in `models/`, and run `./start-server.sh --wait`. It needs NVIDIA driver 570 or newer and
glibc 2.34+ (Ubuntu 22.04 and 24.04). Full guide: [bundle-linux/README.md](bundle-linux/README.md).

## Index

| Folder | What is in it |
|---|---|
| [bundle/](bundle/README.md) | The user guide, the text files of the Windows zip (launcher, layer, chat template) and the changelog. |
| [bundle-linux/](bundle-linux/README.md) | The Linux guide and launcher scripts of the Linux tarball (the layer and template are the same as in bundle/). |
| [harnesses/](harnesses/README.md) | Settings for Claude Code, Codex, Pi and OpenCode, and what we tested. |
| [model/](model/README.md) | The model file: format, bits, which tensors keep full precision, and what can load it. |
| [engine/](engine/README.md) | The engine (a llama.cpp fork): where the source is, how to build it, the driver needs. |
| [benchmarks/](benchmarks/README.md) | Each number on the card: the test, the settings, the card and the date. |
| [research/](research/README.md) | Finished studies: bit allocation, harness gate, old drivers, long-context decode, cross-architecture, clean rebuild. |

Licences: [LICENSE](LICENSE) (Apache-2.0) and [NOTICE](NOTICE). Credits: [CREDITS.md](CREDITS.md). Problems,
numbers from your card and fixes: please open an issue or a pull request.
