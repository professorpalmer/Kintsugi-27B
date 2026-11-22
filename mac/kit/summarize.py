"""Last step: one summary (summary.md + summary.json) and the zip to send back: results-<chip>-<ram>.zip."""
from __future__ import annotations

import json
import os
import re
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, ROOT, out_path  # noqa: E402


def rd(name, default=None):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return default
    if name.endswith(".jsonl"):
        return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]
    return json.load(open(p, encoding="utf-8"))


def main():
    env = rd("env.json", {})
    conv = rd("convert_checks.json", {})
    kl = rd("kl.json")
    chat = rd("chat.json")
    speed = rd("speed.jsonl", [])
    ndl = [r for r in rd("needles.jsonl", []) if "correct" in r]
    plan = rd("memory_plan.json", {})
    tests = rd("escha_tests.json", {})
    emb = rd("embed_cpu_check.json")
    mem = rd("memory_run.json")
    sess = rd("check_session.json")
    srv = rd("serve_check.json")
    ram_gb = round(env.get("ram_bytes", 0) / 2**30)
    lines = [f"# Kintsugi-27B on {env.get('chip', '?')} ({ram_gb} GB, {env.get('gpu_cores', '?')} GPU cores), "
             f"macOS {env.get('macos', '?')}", ""]
    lines.append(f"- convert checks: {conv.get('verdict', 'not run')}")
    lines.append(f"- profile: {plan.get('profile', 'full')}")
    if emb:
        lines.append(f"- CPU embedding vs GPU embedding: {emb['values_not_equal']} of {emb['values']} values differ "
                     f"({'pass' if emb['pass'] else 'FAIL'})")
    lines.append(f"- escha-mlx tests: {tests.get('summary', 'not run')}")
    if kl:
        lines.append(f"- KL (24 docs, {kl['gen_rows']} rows): Mac {kl['mac']['kld_mean']:.5f}, CUDA "
                     f"{kl['cuda_same_docs']['kld_mean']:.5f}, ratio {kl['ratio_mac_over_cuda']:.3f}; top-1 Mac "
                     f"{kl['mac']['top1']:.4f}, CUDA {kl['cuda_same_docs']['top1']:.4f}")
    else:
        lines.append("- KL: not run")
    if chat:
        lines.append(f"- chat: AR {chat['ar']['correct']}/{chat['ar']['of']}"
                     + (f", MTP {chat['mtp']['correct']}/{chat['mtp']['of']}" if "correct" in chat.get("mtp", {})
                        else f", MTP {chat.get('mtp', {}).get('error', 'not run')}"))
    if sess:
        lines.append(f"- chat cache reuse: {'pass' if sess['pass'] else 'FAIL'} (aligned turn "
                     f"{sess['A_aligned']['equal_tokens']}/{sess['A_aligned']['of']} tokens equal to a fresh read)")
    if srv:
        tools = [r for r in srv["rows"] if r["test"].startswith("t")]
        lines.append(f"- server (./run.sh serve): chat {srv['chat_correct']}/{srv['chat_of']}, tool calls "
                     f"{sum(r['ok'] for r in tools)}/{len(tools)}")
    lines.append(f"- largest context under the Metal cap: {plan.get('planned_max_ctx', '?')}")
    lines += ["", "| depth | prefill tok/s | decode tok/s | peak GB |", "|---:|---:|---:|---:|"]
    for r in speed:
        lines.append(f"| {r['depth']} | {r['prefill_tps']:.1f} | {r['decode_tps'] or 0:.2f} | "
                     f"{r.get('peak_memory_gb', 0):.1f} |")
    for r in ndl:
        lines.append(f"\n- needles: {r['correct']}/{r['of']} at {r['depth']} tokens")
    if mem:
        lines.append(f"- memory: GPU cap {mem['metal_cap_bytes'] / 1e9:.2f} GB, MLX peak {mem['peak_memory_gb_max']:.2f} GB, "
                     f"process footprint max {mem['footprint_max_gb']:.2f} GB, swap used "
                     f"{mem['swap_used_mb_start']} MB -> {mem['swap_used_mb_end']} MB")
    md = "\n".join(lines) + "\n"
    open(out_path("summary.md"), "w", encoding="utf-8").write(md)
    json.dump({"env": env, "convert": conv.get("verdict"), "kl": kl and {k: v for k, v in kl.items() if k != "per_doc"},
               "chat": chat and {k: chat[k]["correct"] for k in ("ar", "mtp") if "correct" in chat.get(k, {})},
               "speed": speed, "needles": ndl, "plan": plan, "embed_cpu_check": emb, "memory_run": mem,
               "session": sess and sess.get("pass"), "serve": srv and {k: srv[k] for k in ("chat_correct", "all_ok")}},
              open(out_path("summary.json"), "w", encoding="utf-8"), indent=2)
    print(md)
    chip = re.sub(r"[^A-Za-z0-9]+", "-", env.get("chip", "mac")).strip("-")
    zpath = os.path.join(ROOT, f"results-{chip}-{ram_gb}GB.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for base, _, files in os.walk(RESULTS):
            for f in files:
                p = os.path.join(base, f)
                z.write(p, os.path.relpath(p, os.path.dirname(RESULTS)))
    print(f"\nSEND THIS FILE BACK: {zpath}")


if __name__ == "__main__":
    main()
