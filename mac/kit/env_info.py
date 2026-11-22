"""Step 0: record the machine. Read only."""
from __future__ import annotations

import json
import platform
import re
import subprocess

from common import env_versions, sysctl, write_json


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return ""


def main():
    sp = run(["system_profiler", "SPHardwareDataType", "SPDisplaysDataType"])
    gpu_cores = re.search(r"Total Number of Cores:\s*(\d+)\s*$", sp.split("Graphics/Displays")[-1], re.M)
    info = {
        "chip": sysctl("machdep.cpu.brand_string"),
        "model": sysctl("hw.model"),
        "ram_bytes": int(sysctl("hw.memsize") or 0),
        "cpu_perf_cores": sysctl("hw.perflevel0.physicalcpu"),
        "cpu_eff_cores": sysctl("hw.perflevel1.physicalcpu"),
        "gpu_cores": int(gpu_cores.group(1)) if gpu_cores else None,
        "macos": run(["sw_vers", "-productVersion"]).strip(),
        "macos_build": run(["sw_vers", "-buildVersion"]).strip(),
        "iogpu_wired_limit_mb": sysctl("iogpu.wired_limit_mb"),
        "power": run(["pmset", "-g", "batt"]).strip().splitlines()[:2],
        "low_power_mode": [ln.strip() for ln in run(["pmset", "-g"]).splitlines() if "lowpowermode" in ln],
        "platform": platform.platform(),
        "versions": env_versions(),
    }
    try:
        import mlx.core as mx
        info["mlx_device_info"] = {k: (v if isinstance(v, (int, float, str)) else str(v))
                                   for k, v in mx.device_info().items()}
    except Exception as e:  # noqa: BLE001
        info["mlx_device_info"] = f"error: {e}"
    write_json("env.json", info)
    print(json.dumps({k: info[k] for k in ("chip", "ram_bytes", "gpu_cores", "macos")}))


if __name__ == "__main__":
    main()
