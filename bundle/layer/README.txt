The layer of Kintsugi-27B: an OpenAI-compatible proxy in front of llama-server (start-server.ps1 starts it).

In this build it does one job: saved conversations. After each answer it saves a long conversation to disk (slots\;
a full save, then small deltas). When a request continues a saved conversation that the server does not hold (after a
restart, or when two chats take turns), it restores the saved state first, so the server reads only the new text.

Its other tools (the run_python code interpreter, API cards, the API check, example checks) are off in this build
(start-server.ps1 passes --no-interpreter and the other --no-* flags).

Needs: Python 3.10 or newer, standard library only (tested with 3.12 in a bare venv). The sandbox imports wasmtime only
when a sandbox tool runs, and they are all off here. Without Python, start-server.ps1 runs the plain server and prints
a note.

Files: bonsai_layer.py (the proxy), slot_cache.py (saved conversations), sandbox_path.py and wasi-python\sandbox.py
(the sandbox, unused here), apicards*.py, apilint.py, example_check.py, stray_tags.py (tools, off here).
Licence: Apache-2.0 (see LICENSE in the bundle's folder).
