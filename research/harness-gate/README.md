# Harness gate: four coding harnesses against the release bundle

**Status:** finished (2026-11-20). **Finding:** two harnesses failed every request because the model's own chat
template refused a system message after the first turn. With a fixed template, all four harnesses pass 5 of 6 tasks
on both paths.

The settings for each harness are in [harnesses/README.md](../../harnesses/README.md). This page is the study behind
it.

## Question

Most users drive the model from their own tool-calling harness. Does Kintsugi-27B work in the common ones, through
the saved-chat layer and with the plain server? If not, what is the cause: the model, the server, the chat template,
the layer or the harness?

## What we did

**Endpoint.** The release bundle v1 (then v1.0.1), unzipped fresh, the release model file (sha256 checked). Two arms
for each harness:

- **L**: through the layer (the launcher defaults).
- **D**: direct to the server (`KINTSUGI_LAYER=0`), same arguments.

A small capture proxy sat between the harness and the endpoint. It wrote every request and response (streams byte for
byte) and changed nothing on the wire. All counts come from these captures and from the repository state after each
run, not from the harness's own summary.

**Harnesses.** Official packages, dummy keys, no logins, telemetry off, 127.0.0.1 only.

| Harness | Version | Wire API to our server |
|---|---|---|
| Claude Code | 2.1.330 | `/v1/messages` (Anthropic) |
| OpenAI Codex CLI | 0.171.0 | `/v1/responses` (this version does not accept `wire_api = "chat"` any more) |
| Pi coding agent | 0.73.1 | `/v1/chat/completions` |
| OpenCode | 1.19.6 | `/v1/chat/completions` |

**Phase 0: wire probes (no harness).** On each wire API and arm: a tools array (read, edit, write, bash, grep), stream
on and off, a tool-call turn, the tool result, the answer; system prompt as `system`, `developer` and Anthropic
`system`; thinking on and off; one long tool result (42,132 tokens). Checks: the tool call parses, no `<think>` or
tool tags in the visible text, the right stop reason, a clean stream end, and prompt-cache reuse on turn 2.

**The six tasks.** A small Python repository (standard library and `unittest`), reset to the same commit before every
run. The same words for every harness:

| # | Task | Pass |
|---|---|---|
| 1 | read: "In src/config.py, what is the value of DEFAULT_TIMEOUT?" | the right number |
| 2 | edit + test: fix `slugify` and run its tests | the tests pass, and the run called the test command |
| 3 | create a file with one given line | the file holds exactly that line |
| 4 | grep: "Which files under src/ call legacy_connect( ?" | exactly the 4 right files |
| 5 | multi-step: add `count_words`, add a test, run the tests, count the words of a file | tests pass, the number is right |
| 6 | long context: read a handbook (42,132 tokens as the server read it), answer from section 37 | the right period |

**Recorded for each run:** pass or fail; tool calls; malformed or rejected tool calls; leaked think or tool tags;
loops (the same call 3 or more times); stalls (no token for 120 s); refusals to use tools; full re-reads (a turn that
reads more than 90 % of a prompt over 8k tokens that only extends the last one).

**Line (written before the runs):** a harness works on an arm if it passes 5 or more of 6 tasks, with 0 malformed-call
crashes and no full re-read on every turn. Every failure gets a root cause.

Where measured: our rig (one RTX 4070 12 GB, Windows 11), 2026-11-20.

## Results

### Phase 0 (v1 bundle)

Every exchange passed on both arms: tool calls parsed, no leaked tags, right stop reasons, clean stream ends. Turn 2
reused the prefix on every API; the 42k-token turn read only the new tool result. But:

| API, stream on | First byte of turn 1: layer (v1) | Direct | Layer (patched) |
|---|---:|---:|---:|
| `/v1/messages` | 0.51 s (= total: buffered) | 0.11 s | 0.12 s |
| `/v1/responses` | 0.51 s (= total: buffered) | 0.10 s | 0.11 s |
| `/v1/chat/completions` | 0.18 s | 0.11 s | 0.16 s |

### Faults found, and the fixes

| # | Fault | How found | Root cause | Fix | Shipped in |
|---|---|---|---|---|---|
| 1 | Streamed `/v1/messages` and `/v1/responses` answers arrived in one piece at the end. On a long prompt read (minutes at 200k), a client with an idle timeout gets nothing until the end. | Code reading, then phase 0 | Layer: those paths read the whole upstream answer, then replied | Relay the stream chunk by chunk, with the server's own keep-alive bytes | v1.0.1 |
| 2 | `x-api-key` and `anthropic-*` / `openai-*` headers did not reach the server. With a key file set, a client that sends only `x-api-key` got 401. | Code reading, stub test | Layer: only Authorization and Content-Type were forwarded | Forward Accept, x-api-key and every anthropic-* / openai-* header (no Host, no Cookie) | v1.0.1 |
| 3 | Those paths could run while the layer saved or restored a chat (one server slot). | Code reading | Layer: no lock on those paths | They run under the saved-chat lock | v1.0.1 |
| 4 | A stop right after a long answer lost that answer's save and left a 371 MB `.part` file. | Phase 0 log | `stop.ps1` did not ask the layer about a background save, and the layer's "busy" flag is false during that save | New "saving" field; `stop.ps1` waits up to 30 s while busy or saving | v1.0.1 |
| 5 | **Claude Code: HTTP 500 on every request.** | Matrix, first run | Template: Claude Code sends the user turn, then a `system` message (its environment block). The model's template raises "System message must be at the beginning" for any system message that is not first. | A template that renders a later system message in place as a system turn. Only the raising branch changes. | v1.0.2 |
| 6 | **Codex: HTTP 500 on every request.** | Matrix | Template: Codex puts two `developer` messages first; the server maps developer to system, so the second one is a later system message | The same template fix | v1.0.2 |

Checks of the fixes:

- **Stop check (fault 4), on the GPU:** a 42k-token tool turn, then `stop.ps1` at once. It waited 2.2 s, the save was
  indexed (953 MB, full), no `.part` file was left. After a new start, the chat was restored in 0.51 s and the next
  turn read only 24 new tokens (answer in 3.6 s; without the save, a 42k re-read takes about 30 s).
- **Template render check (faults 5 and 6):** 9 normal request shapes (user only; system first; system first + tools;
  tools without a system message; a tool turn; a tool turn + answer + next user turn; two tool calls in one turn;
  thinking off; multi-part content) render **byte-identical** with the old and the fixed template. The later-system shape is refused by the old template and rendered by the fix. In the
  v1.0.2 bundle smoke, 52 test chats gave byte-identical prompts and answers.

### The matrix

| Harness | v1.0.1 as shipped (L / D) | With the v1.0.2 template (L / D) |
|---|---|---|
| Claude Code 2.1.330 | 0/6 / 0/6 (HTTP 500 on the first request) | **5/6 / 5/6** |
| Codex 0.171.0 | 0/6 / 0/6 (HTTP 500 on the first request) | **5/6 / 5/6** |
| Pi 0.73.1 | **5/6 / 5/6** | (not needed) |
| OpenCode 1.19.6 | **5/6 / 5/6** | (not needed) |

Over 4 harnesses x 2 arms x 6 tasks: 0 malformed or rejected tool calls, 0 leaked think or tool tags, 0 loops,
0 stalls, 0 refusals to use tools, 0 full re-reads. Later turns of a task read 19-1,976 new tokens.

**The one failed task** is the same in every harness: task 4. The model listed the 4 files that call
`legacy_connect(` and also the file that defines it. That is the model's answer, not a fault of the harness, the
server or the layer.

**One rescore, said plainly:** Codex (with the fix, arm L) wrote task 3's file with a UTF-8 byte-order mark (Windows
PowerShell 5.1 adds one). The first check compared bytes and failed it. After that result we clarified the rule (a
byte-order mark is not part of the line) and rescored every task-3 result with it. Only this one changed (fail to
pass). The raw first score stays on record.

### Layer-feature A/B: skipped

The plan had a phase 2: the layer's harness-proofing features (stray-tag filter, think fixes, tool-call cleanup) ON
against OFF. Line: ON becomes the default only if it removes failure classes that we can count. The matrix had 0
malformed calls, 0 leaked tags and 0 think leaks, so there was nothing for the features to remove. Decision: phase 2
skipped; the features stay OFF; the zero counts are the reason. We revisit this if a tester reports a leak.

## Decision

- v1.0.1 (layer stream relay, headers, lock; `stop.ps1` waits for a save) and v1.0.2 (fixed chat template, on by
  default) shipped on 2026-11-20.
- Saved chats for `/v1/messages` and `/v1/responses`: later. A save helps only if the next request's rendered prompt
  starts with the saved tokens. On these two APIs the server renders the prompt from another body format, and the
  layer's save logic reads chat bodies only. That needs its own render check and save test. Until then, those paths
  rely on the server's own prompt cache (one slot, kept between turns).

## What we learned

1. **Test the real harness, not only the wire.** The wire probes passed, and two of four harnesses still failed every
   request. The trigger was a message order that our probes did not send.
2. **The template is part of the model's interface.** Both hard failures came from one branch of the chat template.
3. **Code reading finds what a short test hides.** The buffered streams cost nothing on short answers; on a long
   read they hide every byte until the end.
4. **Run harnesses in a clean environment.** A harness started from our own shell inherited our environment and read
   instruction files from parent folders. We reran every harness with a clean environment, in a folder with no parent
   instruction files, and the capture tool now redacts personal data.

## Limits

- One run per harness, arm and task (temperature as each harness sends it). One small repository and six tasks.
- One version of each harness. Harnesses change their message order and APIs often (Codex dropped `wire_api =
  "chat"`).
- Windows 11 only.
