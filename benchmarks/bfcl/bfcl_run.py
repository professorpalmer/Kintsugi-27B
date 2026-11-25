"""BFCL (gorilla berkeley-function-call-leaderboard, Apache-2.0) with our OpenAI-compatible server, FC mode.

Registers ONE model entry, "kintsugi-local-FC", and then runs the official `bfcl` command line with the arguments given:
    python bfcl_run.py generate --model kintsugi-local-FC --test-category multi_turn --skip-server-setup ...
    python bfcl_run.py evaluate --model kintsugi-local-FC --test-category multi_turn [--partial-eval]

Mode: native function calling (FC). The official OpenAICompletionsHandler sends the BFCL functions as OpenAI `tools`
to /v1/chat/completions; the server applies its own chat template (--jinja, templates/kintsugi.jinja: Qwen3.8 XML
tool calls) and its own tool-call parser, and returns `tool_calls`. This is how the release serves tools to every
harness. The prompting mode (BFCL writes the tool list into the system prompt and parses text) and the local OSS
handlers (raw /v1/completions with a Python copy of an older Qwen template) are not used.

Changes against the official handler (everything else is the official code, pinned commit 6ea57973c7a6097fd7c5915698c54c17c5b1b6c8):
- each LM call sends a `seed` = crc32("<base>:<entry id>:<call index>") & 0x7fffffff (base: env BFCL_SEED_BASE, default
  20261124). So no two calls share a seed. BFCL itself sends no seed.
- the reply's `reasoning_content` is copied into the result file (metadata "reasoning_content") for reading. The chat
  history object is the official one (the full reply message; the SDK sends it back as it came).
- the model name sent is env BFCL_SERVED_MODEL (default "kintsugi", the release alias).
- env BFCL_CLIENT_TIMEOUT (seconds): the request timeout of the OpenAI client (unset: the SDK default, 600 s).
Endpoint: env OPENAI_BASE_URL (for example http://127.0.0.1:18090/v1), OPENAI_API_KEY (any text; local server).
"""
import os
import sys
import zlib

from bfcl_eval.constants import model_config as mc
from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler

SEED_BASE = int(os.getenv("BFCL_SEED_BASE", "20261124"))


class KintsugiFCHandler(OpenAICompletionsHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # the SDK default is 600 s per request (x 3 tries); a BF16 call that runs to the thinking
        # budget needs 760-980 s on an A100. BFCL_CLIENT_TIMEOUT (seconds) raises it; unset = the SDK default.
        if os.getenv("BFCL_CLIENT_TIMEOUT"):
            self.client = self.client.with_options(timeout=float(os.getenv("BFCL_CLIENT_TIMEOUT")))

    def _pre_query_processing_FC(self, inference_data, test_entry):
        inference_data = super()._pre_query_processing_FC(inference_data, test_entry)
        inference_data["_k_id"] = test_entry["id"]
        inference_data["_k_call"] = 0
        return inference_data

    def _query_FC(self, inference_data):
        message = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}
        call = inference_data.get("_k_call", 0)
        inference_data["_k_call"] = call + 1
        seed = zlib.crc32(f"{SEED_BASE}:{inference_data.get('_k_id', '')}:{call}".encode()) & 0x7FFFFFFF
        kwargs = {"messages": message, "model": self.model_name, "temperature": self.temperature, "store": False,
                  "seed": seed}
        if len(tools) > 0:
            kwargs["tools"] = tools
        return self.generate_with_backoff(**kwargs)

    def _parse_query_response_FC(self, api_response):
        r = super()._parse_query_response_FC(api_response)
        msg = api_response.choices[0].message
        rc = getattr(msg, "reasoning_content", None)
        if rc is None and getattr(msg, "model_extra", None):
            rc = msg.model_extra.get("reasoning_content")
        if rc:
            r["reasoning_content"] = rc
        return r


mc.MODEL_CONFIG_MAPPING["kintsugi-local-FC"] = mc.ModelConfig(
    model_name=os.getenv("BFCL_SERVED_MODEL", "kintsugi"),
    display_name="Kintsugi local (FC)",
    url="https://github.com/professorpalmer/Kintsugi-27B",
    org="local",
    license="Apache-2.0",
    model_handler=KintsugiFCHandler,
    input_price=None,
    output_price=None,
    is_fc_model=True,
    underscore_to_dot=True,
)

if __name__ == "__main__":
    from bfcl_eval.__main__ import cli
    sys.argv = ["bfcl"] + sys.argv[1:]
    cli()
