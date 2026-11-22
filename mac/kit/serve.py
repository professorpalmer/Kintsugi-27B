"""OpenAI-compatible server for Kintsugi-27B on this Mac: ./run.sh serve

    POST /v1/chat/completions   streaming or not; tools (tool_calls); reasoning in "reasoning_content"
    GET  /v1/models, /health

It listens on 127.0.0.1:8080 (this Mac only) unless KINTSUGI_HOST / KINTSUGI_PORT say otherwise. With
KINTSUGI_KEY_FILE set, clients must send "Authorization: Bearer <key>". One request runs at a time; others wait.
The prompt of each request is rendered with templates/kintsugi.jinja (the template of the Windows bundle). When a
prompt starts with the previous prompt (a chat or a coding harness), only the new tokens are read.

Request fields read: messages, tools, stream, stream_options.include_usage, max_tokens / max_completion_tokens,
temperature, top_p, top_k, min_p, seed, stop, reasoning_effort (low / medium / high), chat_template_kwargs
(enable_thinking, reasoning_effort). Defaults: reasoning effort medium, the model's sampling (temperature 1.0,
top_k 20, top_p 0.95), 24,576 output tokens, 20,480 reasoning tokens.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import userapi  # noqa: E402

MODEL_ID = "kintsugi"
ENGINE: userapi.Engine | None = None
API_KEY: str | None = None
STATS = {"requests": 0, "bad_tool_calls": 0, "errors": 0}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "kintsugi-mlx"

    def log_message(self, fmt, *args):      # one line per request is printed by the request itself
        pass

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, msg: str, kind: str = "invalid_request_error") -> None:
        self._json(code, {"error": {"message": msg, "type": kind, "code": kind}})

    def _auth(self) -> bool:
        if not API_KEY:
            return True
        if self.headers.get("Authorization", "") == f"Bearer {API_KEY}":
            return True
        self._error(401, "missing or wrong API key", "authentication_error")
        return False

    def do_GET(self):
        if not self._auth():
            return
        path = self.path.split("?")[0].rstrip("/")
        if path in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "kintsugi",
                                                         "context_length": ENGINE.max_ctx}]})
        elif path in ("/health", "/v1/health", ""):
            self._json(200, {"status": "ok", "model": MODEL_ID, "context_length": ENGINE.max_ctx, **STATS})
        else:
            self._error(404, f"no route {self.path}", "not_found")

    def do_POST(self):
        if not self._auth():
            return
        path = self.path.split("?")[0].rstrip("/")
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError) as e:
            return self._error(400, f"body is not JSON: {e}")
        if path in ("/v1/chat/completions", "/chat/completions"):
            return self._chat(body)
        self._error(404, f"no route {self.path}", "not_found")

    # ---- chat completions --------------------------------------------------------------------------------------
    def _chat(self, req: dict) -> None:
        msgs = req.get("messages")
        if not isinstance(msgs, list) or not msgs:
            return self._error(400, "messages must be a non-empty list")
        tools = req.get("tools") or None
        kw = req.get("chat_template_kwargs") or {}
        thinking = kw.get("enable_thinking", req.get("enable_thinking", True)) is not False
        effort = kw.get("reasoning_effort") or req.get("reasoning_effort")
        if isinstance(req.get("reasoning"), dict):          # {"reasoning": {"effort": ...}}
            effort = effort or req["reasoning"].get("effort")
        if effort in ("none", "off"):
            thinking = False
        try:
            ids = ENGINE.render(msgs, tools, effort, thinking)
        except Exception as e:  # noqa: BLE001 -- template errors go back to the client
            STATS["errors"] += 1
            return self._error(400, f"chat template: {e}")
        room = ENGINE.max_ctx - len(ids)
        if room < 16:
            return self._error(400, f"the prompt has {len(ids)} tokens; this Mac's context is {ENGINE.max_ctx} "
                                    f"tokens", "context_length_exceeded")
        want = req.get("max_completion_tokens") or req.get("max_tokens") or userapi.DEFAULT_MAX_TOKENS
        max_tokens = max(1, min(int(want), room))
        opts = {"temperature": req.get("temperature"), "top_p": req.get("top_p"), "top_k": req.get("top_k"),
                "min_p": req.get("min_p")}
        stream = bool(req.get("stream"))
        usage_in_stream = bool((req.get("stream_options") or {}).get("include_usage"))
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        with ENGINE.lock:
            STATS["requests"] += 1
            if req.get("seed") is not None:
                ENGINE.mx.random.seed(int(req["seed"]))
            sampler = ENGINE.sampler(opts)
            if stream:
                self._stream(cid, created, ids, max_tokens, sampler, thinking, tools, req.get("stop"), usage_in_stream)
                return
            try:
                self._whole(cid, created, ids, max_tokens, sampler, thinking, tools, req.get("stop"))
            except Exception as e:  # noqa: BLE001 -- a failed request answers 500, the server stays up
                STATS["errors"] += 1
                log(f"error: {e!r}")
                self._error(500, f"generation failed: {e}", "server_error")

    def _chunk(self, cid, created, delta, finish=None, usage=None):
        obj = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
               "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if usage is not None:
            obj = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                   "choices": [], "usage": usage}
        data = b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def _stream(self, cid, created, ids, max_tokens, sampler, thinking, tools, stop, usage_in_stream):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        gone = [False]
        stats = {}
        n_calls = 0
        try:
            self._chunk(cid, created, {"role": "assistant", "content": ""})
            for kind, x in ENGINE.stream(ids, max_tokens, sampler, thinking, tools, stop, cancel=lambda: gone[0]):
                try:
                    if kind == "reasoning":
                        self._chunk(cid, created, {"reasoning_content": x})
                    elif kind == "text":
                        self._chunk(cid, created, {"content": x})
                    elif kind == "tool_call":
                        self._chunk(cid, created, {"tool_calls": [{"index": n_calls, **x}]})
                        n_calls += 1
                    elif kind == "bad_tool_call":
                        STATS["bad_tool_calls"] += 1
                        self._chunk(cid, created, {"content": "<tool_call>" + x + "</tool_call>"})
                    elif kind == "done":
                        stats = x
                except (BrokenPipeError, ConnectionResetError):
                    gone[0] = True
            if not gone[0]:
                self._chunk(cid, created, {}, finish=stats.get("finish_reason", "stop"))
                if usage_in_stream:
                    self._chunk(cid, created, None, usage=_usage(stats))
        except (BrokenPipeError, ConnectionResetError):
            gone[0] = True
        except Exception as e:  # noqa: BLE001 -- report the failure in the stream, keep the server up
            STATS["errors"] += 1
            log(f"error: {e!r}")
            try:
                self._chunk(cid, created, {"content": f"[server error: {e}]"}, finish="error")
            except (BrokenPipeError, ConnectionResetError):
                gone[0] = True
        if not gone[0]:
            try:
                done = b"data: [DONE]\n\n"
                self.wfile.write(f"{len(done):x}\r\n".encode() + done + b"\r\n0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
        _log_request(ids, stats, n_calls, gone[0])

    def _whole(self, cid, created, ids, max_tokens, sampler, thinking, tools, stop):
        reasoning, text, calls, stats = [], [], [], {}
        for kind, x in ENGINE.stream(ids, max_tokens, sampler, thinking, tools, stop):
            if kind == "reasoning":
                reasoning.append(x)
            elif kind == "text":
                text.append(x)
            elif kind == "tool_call":
                calls.append(x)
            elif kind == "bad_tool_call":
                STATS["bad_tool_calls"] += 1
                text.append("<tool_call>" + x + "</tool_call>")
            elif kind == "done":
                stats = x
        msg = {"role": "assistant", "content": "".join(text) or (None if calls else "")}
        if reasoning:
            msg["reasoning_content"] = "".join(reasoning)
        if calls:
            msg["tool_calls"] = calls
        self._json(200, {"id": cid, "object": "chat.completion", "created": created, "model": MODEL_ID,
                         "choices": [{"index": 0, "message": msg, "finish_reason": stats.get("finish_reason", "stop")}],
                         "usage": _usage(stats)})
        _log_request(ids, stats, len(calls), False)


def _usage(stats):
    return {"prompt_tokens": stats.get("prompt_tokens", 0), "completion_tokens": stats.get("completion_tokens", 0),
            "total_tokens": stats.get("prompt_tokens", 0) + stats.get("completion_tokens", 0),
            "prompt_tokens_details": {"cached_tokens": stats.get("reused_tokens", 0)}}


def _log_request(ids, stats, n_calls, gone):
    if not stats:
        return
    new = stats.get("prompt_tokens", 0) - stats.get("reused_tokens", 0)
    pre = stats.get("prefill_s", 0.0)
    log(f"prompt {stats.get('prompt_tokens')} tok ({stats.get('reused_tokens')} reused, {new} read in {pre:.1f} s), "
        f"output {stats.get('completion_tokens')} tok at {stats.get('decode_tps', 0):.1f} tok/s, "
        f"{n_calls} tool call(s), {stats.get('finish_reason')}{', client gone' if gone else ''}")


def main():
    global ENGINE, API_KEY
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("KINTSUGI_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("KINTSUGI_PORT", "8080")))
    a = ap.parse_args()
    key_file = os.environ.get("KINTSUGI_KEY_FILE")
    if key_file:
        API_KEY = open(key_file, encoding="utf-8").read().strip()
    if a.host not in ("127.0.0.1", "localhost", "::1") and not API_KEY:
        log(f"warning: listening on {a.host} with no API key: anyone on your network can use this Mac's GPU. "
            f"Set KINTSUGI_KEY_FILE.")
    log("loading Kintsugi-27B ...")
    ENGINE = userapi.Engine()
    log(f"model loaded in {ENGINE.load_s:.0f} s; context {ENGINE.max_ctx} tokens, prefill step {ENGINE.step}; "
        f"MTP on; template {os.path.relpath(userapi.TEMPLATE, userapi.common.ROOT)}")
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    log(f"ready  http://{a.host}:{a.port}/v1   (model name: any, for example {MODEL_ID}; stop with Ctrl+C)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
