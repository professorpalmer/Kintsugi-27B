"""Chat with Kintsugi-27B in the terminal: ./run.sh chat

Commands: /think on|off (default on), /effort low|medium|high, /show (show or hide the reasoning), /reset (new
chat), /quit. Each turn reads only the new tokens: the previous turns stay in the cache.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import userapi  # noqa: E402

DIM, END = ("\033[2m", "\033[0m") if sys.stdout.isatty() else ("", "")


def main():
    print("Loading Kintsugi-27B ...", flush=True)
    eng = userapi.Engine()
    print(f"Ready (context {eng.max_ctx} tokens, MTP on). Type a message. /think off, /effort low|medium|high, "
          f"/show, /reset, /quit.")
    messages, thinking, effort, show = [], True, userapi.DEFAULT_EFFORT, True
    sampler = eng.sampler({})
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, arg = line.partition(" ")
            if cmd == "/quit":
                return
            if cmd == "/reset":
                messages = []
                eng.session.reset()
                print("new chat")
            elif cmd == "/think":
                thinking = arg.strip() != "off"
                print(f"thinking {'on' if thinking else 'off'}")
            elif cmd == "/effort" and arg.strip() in userapi.EFFORTS:
                effort = arg.strip()
                print(f"reasoning effort {effort}")
            elif cmd == "/show":
                show = not show
                print(f"reasoning {'shown' if show else 'hidden'}")
            else:
                print("commands: /think on|off, /effort low|medium|high, /show, /reset, /quit")
            continue
        messages.append({"role": "user", "content": line})
        ids = eng.render(messages, effort=effort, thinking=thinking)
        if len(ids) >= eng.max_ctx - 16:
            print(f"the chat has {len(ids)} tokens, the context is {eng.max_ctx}: /reset to start a new chat")
            messages.pop()
            continue
        reasoning, text, stats, said = [], [], {}, False
        try:
            for kind, x in eng.stream(ids, min(userapi.DEFAULT_MAX_TOKENS, eng.max_ctx - len(ids)), sampler,
                                      thinking):
                if kind == "reasoning":
                    reasoning.append(x)
                    if show:
                        print(DIM + x + END, end="", flush=True)
                elif kind == "text":
                    if not said and reasoning and show:
                        print("\n")
                    said = True
                    text.append(x)
                    print(x, end="", flush=True)
                elif kind == "done":
                    stats = x
        except KeyboardInterrupt:
            print("\n(stopped)")
        messages.append({"role": "assistant", "content": "".join(text), "reasoning_content": "".join(reasoning)})
        if stats:
            print(f"\n{DIM}[{stats['completion_tokens']} tokens, {stats.get('decode_tps', 0):.1f} tok/s; prompt "
                  f"{stats['prompt_tokens']} tokens, {stats['reused_tokens']} from the cache]{END}")


if __name__ == "__main__":
    main()
