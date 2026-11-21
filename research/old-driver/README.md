# Old NVIDIA drivers: the tiered K/V cache on driver 570, and a silent CPU fallback on Windows

**Status:** finished (2026-11-20). **Finding:** drivers of the 570 family refuse the type of pinned system memory
that the tiered K/V cache asks for, so the release engine does not load on them. An engine fix falls back to the host
NUMA type; it loads and passes on two 570 boxes, and goes into the next engine release. On Windows, an old driver made
the bundle run on the CPU and still print "ready"; launcher checks in v1.0.3 stop that.

## Question

Why did the release engine not load on a rented Turing box, and what does a user with an old driver see?

## Background: how the tiered K/V cache uses system RAM

The tiered K/V cache keeps the first N cells of each K/V tensor in VRAM and the other cells in system RAM, inside
**one CUDA device address range**. The engine makes the system-RAM pages with `cuMemCreate` at the location type
`CU_MEM_LOCATION_TYPE_HOST`, and maps them into the same range as the VRAM pages (`cuMemMap`). The two-stage output
head keeps its exact rows in system RAM in the same way.

## What we found

1. **The load fails on driver 570.** On a rented Quadro RTX 6000 (Linux, driver 570.86.10, CUDA driver API 12.8), the
   release engine (commit c07efbec5, built for sm_75) stopped at load:
   `cuMemCreate(host, 1290.0 MiB) failed: invalid argument`, then "unable to allocate ... head2 ... buffer". The KL
   check on the same box passed (it uses no tiered memory; see [cross-architecture](../cross-architecture/README.md)).
2. **The cause is the driver, not Turing.** A small probe program on the box asked for each host location type:

   | Driver | Kernel module | HOST | HOST_NUMA (id 0) | HOST_NUMA_CURRENT |
   |---|---|---|---|---|
   | 570.86.10 | proprietary | refused ("invalid argument") | accepted | |
   | 570.133.07 | open | refused ("invalid argument") | accepted | refused |
   | 595, 610, 615 (other rented boxes) and the rig's Windows driver | | accepted | | |

   Both 570 boxes refuse HOST, with both kernel module types. So the cause is the driver family. Every box with a 580+
   driver loaded. (The CUDA 12.8 header documents the refusal.)
3. **Pinned memory from `cuMemHostAlloc` is not a way out.** A tiered buffer is one address range that mixes VRAM pages
   and host pages, and only `cuMemCreate` handles can be mapped into it. A `cuMemHostAlloc` pointer is a separate
   range: the whole K/V would have to live in system RAM, and every read would cross PCIe (10-20x slower).

## The engine fix (commit 30d6a711e, in the next engine release)

- The one host `cuMemCreate` site (the tier allocator, used by the tiered K/V and the two-stage head) tries HOST
  first. On "invalid value" or "not supported" it changes to HOST_NUMA with the device's host NUMA id (0 if the driver
  gives none) and keeps that path for the process.
- One log line names the path: INFO for HOST, **WARN for the HOST_NUMA fallback** (so it shows at the default log
  level), INFO when forced.
- If the driver refuses both types, the load stops with: "this NVIDIA driver can not map system RAM into the tiered
  K/V cache (cuMemCreate refuses the host location types): update the NVIDIA driver to 580 or newer".
- Test switch: `GGML_CUDA_HOSTMEM_FORCE=host|numa|pinned` (`pinned` = no host path, so it must give the stop message).
- The change is in one file (`ggml/src/ggml-cuda/ggml-cuda.cu`).

## Results

**On a real 570 box** (rented, 2x RTX 2080 Ti, driver 570.133.07, open module; one GPU used, held to a 12,288 MiB
line; engine built from source with CUDA 12.8):

| Check | Line | Release engine c07efbec5 | Fix 30d6a711e |
|---|---|---|---|
| Load | loads | **fails** (`cuMemCreate(host) ... invalid argument`) | **loads** (11,653 MiB, 144,128 K/V cells in VRAM) |
| Path line | exactly one WARN line | | one WARN: "cuMemCreate on CU_MEM_LOCATION_TYPE_HOST_NUMA, node 0: this driver refuses CU_MEM_LOCATION_TYPE_HOST (older than the 580 family)" |
| Tool calls t1-t4 | pass | | pass |
| Retrieval at 117,000 tokens, thinking off | 24/24 needles + 20/20 copies | | 24/24 + 20/20 |
| Spill and out-of-memory | 0, VRAM <= 12,005 MiB | | 0, 11,667 MiB at the end |
| Forced `pinned` | stop message | | the stop message, no crash |

**On our rig** (RTX 4070, Windows, driver 580+ family; the fix forced to each path):

| Check | Line | Result |
|---|---|---|
| Forced HOST_NUMA: identity | 50/50 token-identical against the release engine | 50/50 |
| 117k with only 65,536 cells in VRAM (so every read reaches host pages): decode | NUMA within 5 % of HOST | HOST 53.7 tok/s, HOST_NUMA 54.1 tok/s (+0.7 %) |
| The same run: retrieval | 24/24 + 20/20 on both paths | 24/24 + 20/20 on both |
| Forced `pinned` | stop message, no crash, 0 servers left | pass |
| The bundle path with the fix (layer on, defaults): identity, 52 chats, t1-t4 | 50/50, 52/52, pass | pass (rerun; see Limits) |

Speeds in this table are from the rig only. We make no speed claim from rented boxes.

## Windows: a silent CPU fallback, found by code reading

The Windows bundle carries the CUDA 13 runtime, which needs driver 580 or newer. So on Windows a 570 driver stops
before the tiered cache code. We read the code to see what a user then gets (no run):

1. The CUDA runtime starts, and the engine's CUDA set-up fails with "CUDA driver version is insufficient for CUDA
   runtime version". The CUDA backend then has 0 devices; nothing stops.
2. The server log gets "no usable GPU found, --gpu-layers option will be ignored" and a misleading hint that the build
   has no GPU support.
3. The model loads on the CPU: the Escha op has a CPU path.
4. The launcher's own checks passed (nvidia-smi works on 570), it did not show the server log, and it printed
   `ready`. The user got a very slow server and no message about the driver.

**v1.0.3 (launcher only; the engine files are byte-equal to v1.0.2):**

- Before anything starts: read the driver version from nvidia-smi. Major version under 580: stop with "NVIDIA driver
  <v> is too old: Kintsugi needs driver 580 or newer (CUDA 13). Update the driver, then start again." If the version
  cannot be read, print a WARN and go on.
- After the load: if the server log has "failed to initialize CUDA" or "no usable GPU found", stop the server and the
  layer and print "the engine could not use the GPU (see <log>): update the NVIDIA driver to 580 or newer".

Gate (written before the runs), all passed on 2026-11-20:

| Check | Result |
|---|---|
| A test driver value 570.86 (`KINTSUGI_TEST_DRIVER`) | the exact stop message, exit code 1, no server and no layer started |
| The GPU-use check, taken from the shipped launcher by the PowerShell parser | the stop message for a log with the CUDA init line and for "no usable GPU found"; nothing for a normal server log or a missing file |
| A normal start on the rig (layer on, defaults) | no warning, ready, t1-t4 pass, 0 spill, `stop.ps1` leaves 0 processes |
| Zip difference against v1.0.2 | only the launcher, README and NOTICE |

## Decision

- v1.0.3 shipped the launcher checks on 2026-11-20.
- The engine fix 30d6a711e passed its gate and goes into the next bundle and its engine source. It does not change
  anything for the Windows bundle today (CUDA 13 needs 580+ anyway). It matters for engines built with a CUDA 12.x
  toolkit: Linux boxes and users who build from source. Until then, [engine/README.md](../../engine/README.md) says
  that drivers older than the 580 family do not load the c07efbec5 engine.
- Linux build note found on the way: add `-DLLAMA_USE_PREBUILT_UI=OFF`, else the web UI asset step stops the build.

## What we learned

1. **Test on old drivers, not only on old cards.** The fault looked like a Turing fault and was a driver fault.
2. **A fallback must not hide a slow path.** The engine's CPU fallback is correct code, but a "ready" line on top of
   it hides a server that is far too slow to use. The launcher must check that the GPU is in use.
3. **Say the path in the log.** One WARN line at the default log level tells a user which memory path runs.

## Limits

- Two 570 boxes (one proprietary, one open module), both Linux. We did not test a 575 driver; its CUDA driver API is
  also 12.x, so it probably refuses HOST too.
- The first rig run of the fix through the bundle path, at debug log level, had one mid-decode CUDA error. Three
  reruns of the same path (two with the fix, one with the published engine; two at debug log level) had none. The
  error text was lost. A soak test of the published bundle is planned to look for it.
