"""Shared parts of the user mode (kit/serve.py and kit/chat.py): the engine, prompt rendering with
templates/kintsugi.jinja, and the parser that splits the output into reasoning, text and tool calls.

The engine keeps one conversation's caches between requests (escha_mlx.mtp_session.MTPSession): when a new prompt
starts with the previous prompt, only the new tokens are read. Generation uses the MTP tree with the adaptive draft
prefix. One request runs at a time.
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

TEMPLATE = os.environ.get("KINTSUGI_CHAT_TEMPLATE") or os.path.join(common.ROOT, "templates", "kintsugi.jinja")
EFFORTS = {"low": "low", "minimal": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh"}
# The Windows bundle's defaults: reasoning effort medium, 20,480 reasoning tokens, 24,576 output tokens.
DEFAULT_EFFORT = os.environ.get("KINTSUGI_EFFORT", "medium")
REASONING_BUDGET = int(os.environ.get("KINTSUGI_REASONING_BUDGET", "20480"))
BUDGET_MESSAGE = "\n\nNow produce the complete answer.\n</think>\n\n"
DEFAULT_MAX_TOKENS = int(os.environ.get("KINTSUGI_MAX_TOKENS", "24576"))
ANSWER_ROOM = 2048              # tokens kept for the answer when the reasoning budget is cut to fit max_tokens
# The model's generation_config.json: temperature 1.0, top_k 20, top_p 0.95.
DEFAULT_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}


class Engine:
    def __init__(self):
        import mlx.core as mx
        from escha_mlx.mtp_session import MTPSession
        self.mx = mx
        self.model, self.tok, self.load_s = common.load(mtp=True, last_logit=True)
        # 16 GB profile: the MTP head adds 0.32 GB, so 8k does not fit under the kit's margin; 6k does (probed by the
        # 16 GB review of #21). No fit: stop with the plan's message (never guess a context).
        plan = common.memory_plan(ctx_steps_16=common.CTX_STEPS_16 + (6144,))
        self.max_ctx = int(plan.get("max_ctx_under_cap") or 0)
        if self.max_ctx <= 0:
            raise SystemExit(plan.get("message") or "The model does not fit in this Mac's GPU working set. Close "
                                                    "other apps and run again.")
        if int(os.environ.get("KINTSUGI_MAX_CTX", "0") or 0) > 0:      # a lower limit only: the plan is the cap
            self.max_ctx = min(self.max_ctx, int(os.environ["KINTSUGI_MAX_CTX"]))
        self.step = int(plan.get("prefill_step") or common.PREFILL_STEP)
        # Prefix snapshots (escha_mlx.mtp_session): 8 GDN snapshots at the chunk ends up to 16k tokens and one pinned
        # prefix, so a new conversation with the same system prompt and tools (a coding agent's next session) reads
        # only its new part. About 0.6 GB + 1.2 GB on a 48 GB Mac (derived); off in the 16 GB profile. A request whose
        # prompt + output room comes within 40,960 tokens of the context limit drops them (2.5 GB = 40,960 tokens of
        # K/V), so the memory plan at the limit holds.
        snaps = int(os.environ.get("KINTSUGI_SNAPSHOTS", "0" if common.P16 else ("4" if common.RAM_BYTES < 40 * 2**30
                                                                                 else "8")))
        self.session = MTPSession(self.model, prefill_step_size=self.step,
                                  reserve=self._size_kv, snapshots=snaps, snapshot_max=snaps * self.step,
                                  extras_limit=self.max_ctx - 40960)
        # The K/V is sized once for the prompt and this request's output room, at most the context (+ 64 for the MTP
        # tree's extra positions), and never grows past that limit: it stays inside the memory plan (16 GB review of
        # #21: a full 6k chat grew it to 10,240 positions).
        self._room = 1024
        self.template = open(TEMPLATE, encoding="utf-8").read()
        self.eos = set(self.tok.eos_token_ids) | {self.tok.convert_tokens_to_ids("<|im_end|>")}
        self.lock = threading.Lock()
        # MLX arrays belong to the GPU stream of the thread that made them, and the session keeps arrays between
        # requests: so all generation runs on this one thread (an HTTP server answers each request on a new one).
        self._jobs: queue.Queue = queue.Queue()
        threading.Thread(target=self._worker, name="kintsugi-gpu", daemon=True).start()

    def _size_kv(self, cache, n: int) -> None:
        from escha_mlx.kv_cache import limit, reserve
        reserve(cache, min(n + self._room, self.max_ctx) + 64)
        limit(cache, self.max_ctx + 64)

    def _worker(self):
        while True:
            fn, out = self._jobs.get()
            try:
                for ev in fn():
                    out.put(ev)
            except BaseException as e:  # noqa: BLE001 -- handed to the caller's thread
                out.put(("_error", e))
            out.put(None)

    # ---- prompt ------------------------------------------------------------------------------------------------
    def render(self, messages, tools=None, effort=None, thinking=True) -> list[int]:
        kw = {"enable_thinking": bool(thinking)}
        if thinking:
            kw["reasoning_effort"] = EFFORTS.get(str(effort or DEFAULT_EFFORT).lower(), "medium")
        s = self.tok.apply_chat_template(normalize(messages), tools=tools or None, chat_template=self.template,
                                         add_generation_prompt=True, tokenize=False, **kw)
        return list(self.tok.encode(s, add_special_tokens=False))

    # ---- generation --------------------------------------------------------------------------------------------
    def sampler(self, opts):
        from mlx_lm.sample_utils import make_sampler
        o = dict(DEFAULT_SAMPLING)
        o.update({k: v for k, v in opts.items() if v is not None})
        if o["temperature"] <= 0:
            return None
        return make_sampler(temp=float(o["temperature"]), top_p=float(o["top_p"]), top_k=int(o["top_k"]),
                            min_p=float(o["min_p"]))

    def stream(self, *args, **kw):
        """_stream on the generation thread; events come back through a queue."""
        out: queue.Queue = queue.Queue()
        self._jobs.put((lambda: self._stream(*args, **kw), out))
        while (ev := out.get()) is not None:
            if ev[0] == "_error":
                raise ev[1]
            yield ev

    def _stream(self, prompt_ids, max_tokens, sampler, thinking, tools=None, stop=None, cancel=None):
        """Yield (kind, payload) events: ("reasoning"|"text", str), ("tool_call", dict), ("bad_tool_call", str),
        ("done", stats). Applies the reasoning budget: past it, the budget message and </think> go in and the answer
        follows."""
        parser = OutputParser(in_reasoning=thinking, stop=stop, tools=tools)
        # The reasoning budget leaves room for the answer when max_tokens is small (a short context, a client limit).
        budget = REASONING_BUDGET
        if budget and max_tokens < budget + ANSWER_ROOM:
            budget = max(max_tokens - ANSWER_ROOM, max_tokens // 2)
        stats = {"prompt_tokens": len(prompt_ids), "completion_tokens": 0, "reused_tokens": 0, "finish_reason": "stop"}
        t0 = time.perf_counter()
        ids = list(prompt_ids)
        generated: list[int] = []
        budget_used = False
        reasoning_tokens = 0
        first = True
        while True:
            detok = self.tok.detokenizer
            detok.reset()
            self._room = max_tokens - len(generated)
            gen = self.session.generate(ids, max_tokens - len(generated), sampler)
            budget_hit = False
            for tid in gen:
                if first:
                    stats["reused_tokens"] = self.session.reused
                    stats["prefill_s"] = time.perf_counter() - t0
                    t_dec = time.perf_counter()
                    first = False
                if tid in self.eos:
                    break
                generated.append(tid)
                reasoning_tokens += parser.in_reasoning
                detok.add_token(tid)
                for ev in parser.feed(detok.last_segment):
                    yield ev
                if parser.stopped or (cancel is not None and cancel()):
                    break
                if parser.in_reasoning and budget and reasoning_tokens >= budget and not budget_used:
                    budget_hit = True
                    break
            else:
                stats["finish_reason"] = "length"
            gen.close()
            detok.finalize()
            for ev in parser.feed(detok.last_segment):
                yield ev
            if budget_hit:
                budget_used = True
                ids = list(prompt_ids) + generated + list(self.tok.encode(BUDGET_MESSAGE, add_special_tokens=False))
                yield from parser.feed(BUDGET_MESSAGE)
                continue
            break
        for ev in parser.close():
            yield ev
        if parser.tool_calls:
            stats["finish_reason"] = "tool_calls"
        stats["completion_tokens"] = len(generated)
        dec = time.perf_counter() - t_dec if not first else 0.0
        stats["decode_tps"] = len(generated) / dec if dec > 0 else 0.0
        stats["seconds"] = time.perf_counter() - t0
        yield "done", stats


def normalize(messages):
    """OpenAI messages -> what the template reads: developer -> system, tool call arguments as objects, content parts
    as text."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "developer":
            m["role"] = "system"
        c = m.get("content")
        if isinstance(c, list):
            m["content"] = "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type", "text") == "text")
        if c is None:
            m["content"] = ""
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                fn = dict(tc.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        fn["arguments"] = {"arguments": args}
                tc["function"] = fn
                calls.append(tc)
            m["tool_calls"] = calls
        if m.get("reasoning") and not m.get("reasoning_content"):
            m["reasoning_content"] = m["reasoning"]
        out.append(m)
    return out


class OutputParser:
    """Streams text into reasoning / text / tool calls. Tags: </think> ends the reasoning, <tool_call>...</tool_call>
    holds one call in the qwen3_coder XML form. Text that could be the start of a tag is held back until it is
    known."""

    THINK_END, CALL_START, CALL_END = "</think>", "<tool_call>", "</tool_call>"

    def __init__(self, in_reasoning: bool, stop=None, tools=None):
        self.in_reasoning = in_reasoning
        self.in_call = False
        self.buf = ""
        self.reasoning = ""
        self.text = ""
        self.tool_calls: list[dict] = []
        self.bad_calls: list[str] = []
        self.stop = [s for s in ([stop] if isinstance(stop, str) else (stop or [])) if s]
        self.stopped = False
        self._lead = True          # strip the blank lines after </think>
        self._late_think_end = False
        self.tools = tools

    def _hold(self, s: str, tags) -> int:
        """Length of the tail of s that could be the start of one of the tags."""
        n = 0
        for t in tags:
            for k in range(min(len(t) - 1, len(s)), 0, -1):
                if s.endswith(t[:k]):
                    n = max(n, k)
                    break
        return n

    def _emit_text(self, s: str):
        if self._lead:
            s = s.lstrip("\n")
            if not s:
                return
            self._lead = False
        for st in self.stop:               # held-back tails keep a stop string from going out in part
            i = s.find(st)
            if i >= 0:
                s, self.stopped = s[:i], True
                break
        if s:
            self.text += s
            yield "text", s

    def feed(self, chunk: str):
        self.buf += chunk
        while self.buf and not self.stopped:
            if self.in_reasoning:
                i = self.buf.find(self.THINK_END)
                j = self.buf.find(self.CALL_START)          # a call before </think> ends the reasoning
                if i < 0 and j < 0:
                    keep = self._hold(self.buf, [self.THINK_END, self.CALL_START])
                    head = self.buf[:len(self.buf) - keep]
                    keep += len(head) - len(head.rstrip("\n"))
                    out, self.buf = self.buf[:len(self.buf) - keep], self.buf[len(self.buf) - keep:]
                    if out:
                        self.reasoning += out
                        yield "reasoning", out
                    return
                call_first = j >= 0 and (i < 0 or j < i)
                k = j if call_first else i
                out = self.buf[:k].rstrip("\n")
                if out:
                    self.reasoning += out
                    yield "reasoning", out
                self.in_reasoning = False
                self._lead = True
                if call_first:
                    self.buf = self.buf[k + len(self.CALL_START):]
                    self.in_call = True
                    self._late_think_end = True                 # a </think> after the call is not text
                else:
                    self.buf = self.buf[k + len(self.THINK_END):]
            elif self.in_call:
                i = self.buf.find(self.CALL_END)
                if i < 0:
                    return
                body, self.buf = self.buf[:i], self.buf[i + len(self.CALL_END):]
                self.in_call = False
                yield from self._call(body)
            else:
                if self._late_think_end:                   # drop one </think> right after a call in the reasoning
                    lead = self.buf.lstrip()
                    if lead.startswith(self.THINK_END):
                        self.buf = lead[len(self.THINK_END):]
                        self._late_think_end = False
                        continue
                    if self.THINK_END.startswith(lead):
                        return                              # not known yet
                    self._late_think_end = False
                i = self.buf.find(self.CALL_START)
                if i < 0:
                    keep = self._hold(self.buf, [self.CALL_START] + self.stop)
                    head = self.buf[:len(self.buf) - keep]                   # newlines before a call go
                    keep += len(head) - len(head.rstrip("\n"))
                    out, self.buf = self.buf[:len(self.buf) - keep], self.buf[len(self.buf) - keep:]
                    if out:
                        yield from self._emit_text(out)
                    return
                if i:
                    yield from self._emit_text(self.buf[:i].rstrip("\n"))
                self.buf = self.buf[i + len(self.CALL_START):]
                self.in_call = True

    def _call(self, body: str):
        from mlx_lm.tool_parsers.qwen3_coder import parse_tool_call
        try:
            if "<function=" not in body:
                raise ValueError("no <function=...> in the call")
            call = parse_tool_call(body.strip(), self.tools)
        except Exception:  # noqa: BLE001 -- an unreadable call goes back as text and is counted
            self.bad_calls.append(body)
            yield "bad_tool_call", body
            return
        call = {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                "function": {"name": call["name"], "arguments": json.dumps(call["arguments"], ensure_ascii=False)}}
        self.tool_calls.append(call)
        yield "tool_call", call

    def close(self):
        if self.in_call:                   # also an open <tool_call> with no body: a malformed call, counted
            yield from self._call(self.buf)
        elif self.buf and not self.stopped:
            if self.in_reasoning:
                self.reasoning += self.buf
                yield "reasoning", self.buf
            else:
                yield from self._emit_text(self.buf.rstrip("\n"))
        self.buf = ""
