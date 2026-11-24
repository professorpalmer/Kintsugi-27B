#!/usr/bin/env bash
# Kintsugi-27B on one 12 GB NVIDIA card (RTX 20 / 30 / 40 / 50, H100), Linux x86_64: 262,144-token window, tiered K/V cache,
# MTP + lookup drafting. The same defaults and checks as start-server.ps1 of the Windows bundle.
#
#   ./start-server.sh [--port 8080] [--wait] [--dry-run]
#
#   --wait     wait until the server answers, send one warm-up request, print the VRAM in use, then return
#   --dry-run  print the settings and the server command line, start nothing
#
# The model file goes in models/Kintsugi-27B-12G.gguf (see README.md). It listens on 127.0.0.1 only (this computer),
# unless you set KINTSUGI_HOST. Stop it with: ./stop.sh
#
# Settings (environment variables; each one also works with the prefix ESCHA_ in place of KINTSUGI_):
#   KINTSUGI_MODEL         model file (default models/Kintsugi-27B-12G.gguf)
#   KINTSUGI_CTX           window in tokens (default 262144; with the tiered cache it is rounded up to a multiple of 32,768)
#   KINTSUGI_KV            K/V cache type: q4_0 (default) | q8_0 | q3_0
#   KINTSUGI_MARGIN        MiB kept free under the card's limit (default 250 with no display on the card, 600 with one)
#   KINTSUGI_CELLS         K/V cells in VRAM (skips the arithmetic below)
#   KINTSUGI_DRAFT         auto (default: DFlash2 draft 7 when the drafter file is there, else MTP 3 + lookup) | dflash |
#                          mtp | off; a number = the MTP draft size (0 = no drafting)
#   KINTSUGI_DRAFTER       DFlash2 drafter file (default models/dflash-Qwen3.8-27B-Q4_0.gguf; an optional download)
#   KINTSUGI_VISION        auto (default: images on when the projector file is there; the projector on the CPU on a card
#                          under 16 GB, on the GPU from 16 GB) | gpu | cpu | off. cpu: no VRAM cost, an image reads in
#                          5-25 s; gpu: ~1.3 GB of VRAM (fewer K/V cells in VRAM), an image reads in 1-3 s
#   KINTSUGI_MMPROJ        vision projector file (default models/mmproj-Qwen3.8-27B-Q8_0.gguf; an optional download)
#   KINTSUGI_IMAGE_MAX_TOKENS  tokens for one image at most (default 2048; a larger image is scaled down to fit)
#   KINTSUGI_THINK         medium (default) | off
#   KINTSUGI_READ          fast (default: int8 read of long prompts) | exact
#   KINTSUGI_HEAD2         two-stage output head: exact candidates for each row (default 512; 0 = off)
#   KINTSUGI_HOST          address clients connect to (default 127.0.0.1 = this computer only; 0.0.0.0 = your network:
#                          then set KINTSUGI_KEY_FILE too, or anyone on the network can use your GPU)
#   KINTSUGI_KEY_FILE      API key file; with it, clients must send "Authorization: Bearer <key>"
#   KINTSUGI_LAYER         1 (default) = saved conversations through the layer (needs Python 3.10+); 0 = plain server
#   KINTSUGI_SLOT_SAVE_PATH  folder of the saved conversations (default slots/ in this folder)
#   KINTSUGI_SLOT_CACHE_GB   size limit of that folder in GB (default 20; the oldest files are removed past it)
#   KINTSUGI_UB            micro-batch for prompt reads (default 1024; 512 saves about 146 MiB and reads 8-13 % slower)
#   KINTSUGI_ALIAS         the model name that /v1/models lists and replies carry (default kintsugi; an --alias in
#                          KINTSUGI_EXTRA wins)
#   KINTSUGI_EXTRA         more llama-server arguments
#   KINTSUGI_CHAT_TEMPLATE chat template file (default templates/kintsugi.jinja); "model" = the template inside the
#                          model file; any other value = the path of your own template
#   KINTSUGI_TEST_DRIVER   replaces the driver version that nvidia-smi gives (tests)
set -u
# the bundle folder, resolved (readlink -f): the same path whether the script, the folder or a parent is reached
# through a symlink, so start-server.sh and stop.sh always agree on the layer and engine paths
ROOT="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
pick() {   # pick NAME DEFAULT: KINTSUGI_NAME, else ESCHA_NAME, else DEFAULT
  local a="KINTSUGI_$1" b="ESCHA_$1"
  if [ -n "${!a:-}" ]; then echo "${!a}"; elif [ -n "${!b:-}" ]; then echo "${!b}"; else echo "$2"; fi
}
PORT=8080; WAIT=0; DRY=0
ORIG_ARGS=("$@")   # for the one DFlash2 -> MTP retry (retry_mtp below)
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --wait) WAIT=1; shift ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown argument: $1 (use --port N, --wait, --dry-run)"; exit 2 ;;
  esac
done

MODEL="$(pick MODEL "$ROOT/models/Kintsugi-27B-12G.gguf")"
BIN="$ROOT/bin"
CTX="$(pick CTX 262144)"
KV="$(pick KV q4_0)"
THINK="$(pick THINK medium)"
# The CUDA runtime in bin/ sets the driver floor. This build uses CUDA 12.8, which needs driver 570 or newer.
MIN_DRIVER=570
if [ ! -f "$MODEL" ]; then
  echo "Kintsugi-27B: model file not found: $MODEL"
  echo "  Download Kintsugi-27B-12G.gguf from https://huggingface.co/CaryPalmer/Kintsugi-27B"
  echo "  and put it in $ROOT/models. SHA-256 0510bae82445dc53dbc7d0d842d8b7e146b8eafce0837e39b157d411e48c8a74"
  echo "  (check it with: sha256sum models/Kintsugi-27B-12G.gguf). Or set KINTSUGI_MODEL to the file's path."
  exit 1
fi
[ -x "$BIN/llama-server" ] || { echo "llama-server not found in $BIN"; exit 1; }
command -v nvidia-smi > /dev/null || { echo "nvidia-smi not found: install the NVIDIA driver"; exit 1; }
# Every shared library of the engine must be found (bin/ first). A missing one stops the server at once, and the loader
# error is only in the log.
MISSING="$(LD_LIBRARY_PATH="$BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ldd "$BIN/llama-server" 2> /dev/null | awk '/not found/ { print $1 }' | sort -u | tr '\n' ' ')"
if [ -n "$MISSING" ]; then
  echo "Kintsugi-27B: the engine needs shared libraries that this computer does not have: $MISSING"
  exit 1
fi
command -v curl > /dev/null || { echo "curl not found: install curl (the start uses it to check the server)"; exit 1; }

# ---- NVIDIA driver ------------------------------------------------------------------------------------------------------
DRIVER="$(pick TEST_DRIVER '')"
[ -z "$DRIVER" ] && DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2> /dev/null | head -1 | tr -d ' ')"
if [[ "$DRIVER" =~ ^([0-9]+)\. ]]; then
  if [ "${BASH_REMATCH[1]}" -lt "$MIN_DRIVER" ]; then
    echo "NVIDIA driver $DRIVER is too old: Kintsugi needs driver $MIN_DRIVER or newer (CUDA 12.8). Update the driver, then start again."
    exit 1
  fi
else
  echo "WARN   could not read the NVIDIA driver version from nvidia-smi: the driver check is skipped"
fi

# ---- Card generation: bin/ has native code for compute capability 7.5 / 8.6 / 8.9 (RTX 20 / 30 / 40), 9.0 (H100 / H200)
# and 12.0 (RTX 50), and portable code (compute_90 PTX) that newer cards compile at the first start. CUDA loads only the
# code of your card, so the code for the other cards costs no VRAM or speed. KINTSUGI_TEST_CC=<x.y> replaces the value
# that nvidia-smi gives (tests).
CC="$(pick TEST_CC '')"
[ -z "$CC" ] && CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 2> /dev/null | head -1 | tr -d ' ')"
SUPPORTED="RTX 20 / 30 / 40 / 50 cards with 12 GB or more, and H100 / H200"
if [[ "$CC" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
  CCMAJ=${BASH_REMATCH[1]}; CCMIN=${BASH_REMATCH[2]}; CCN=$((10 * CCMAJ + CCMIN))
  if [ "$CCN" -lt 75 ]; then
    echo "This NVIDIA card (compute capability $CC) is older than RTX 20 (Turing, 7.5): Kintsugi-27B has no code for it. Supported: $SUPPORTED."
    exit 1
  elif { [ "$CCMAJ" = 7 ] && [ "$CCN" -ne 75 ]; } || { [ "$CCMAJ" = 8 ] && [ "$CCMIN" -lt 6 ]; }; then
    echo "This NVIDIA card (compute capability $CC, for example a V100 / A100 / A30) has no code in this bundle. Supported: $SUPPORTED."
    exit 1
  elif [ "$CCN" -ge 90 ] && [ "$CCN" -ne 90 ] && [ "$CCN" -ne 120 ]; then
    echo "note   this card (compute capability $CC) runs the engine's portable code: the driver compiles it at the first start (several minutes, once, then cached). We have not tested this card."
    # the driver's compile cache must hold all of it, or every start compiles again: 4 GiB (the largest value)
    export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-4294967296}"
  fi
else
  echo "WARN   could not read the card's compute capability from nvidia-smi: the card check is skipped"
fi

# A driver the runtime cannot use (or no CUDA device) does not stop the engine: it runs on the CPU, very slowly, and still
# answers. no_gpu_message LOG prints the stop message when the server log shows that.
no_gpu_message() {
  [ -f "$1" ] && grep -q -F -e 'failed to initialize CUDA' -e 'no usable GPU found' "$1" \
    && echo "the engine could not use the GPU (see $1): update the NVIDIA driver to $MIN_DRIVER or newer"
}
SP=""; LP=""
stop_no_gpu() {
  [ -n "$SP" ] && kill -9 "$SP" 2> /dev/null
  [ -n "$LP" ] && kill -9 "$LP" 2> /dev/null
  echo "$1"
  exit 1
}

# ---- VRAM line: how many K/V cells fit in VRAM; the rest of the window lives in system RAM -----------------------------
case "$KV" in
  q8_0) HALF=17408 ;; q4_0) HALF=9216 ;; q3_0) HALF=7168 ;;
  *) echo "KINTSUGI_KV must be one of: q8_0, q4_0, q3_0"; exit 1 ;;
esac
CELL=$((2 * HALF))
IFS=',' read -r TOTAL USED DISP < <(nvidia-smi --query-gpu=memory.total,memory.used,display_active --format=csv,noheader,nounits | head -1)
TOTAL=$(echo "$TOTAL" | tr -dc '0-9'); USED=$(echo "$USED" | tr -dc '0-9'); DISP=$(echo "$DISP" | tr -d ' ')
# A display on the card = a graphics process (type G or C+G) in nvidia-smi's process table. display_active is not used:
# a headless server GPU can report "Enabled" (seen on a rented RTX 4070, driver 580). KINTSUGI_MARGIN overrides.
GFX=$(nvidia-smi 2> /dev/null | grep -c -E '^\|[[:space:]]+[0-9]+[[:space:]]+[^[:space:]]+[[:space:]]+[^[:space:]]+[[:space:]]+[0-9]+[[:space:]]+(G|C\+G)[[:space:]]')
HEADLESS=1; [ "${GFX:-0}" -gt 0 ] && HEADLESS=0
[ "$TOTAL" -lt 11500 ] && echo "WARN   this card has $TOTAL MiB of VRAM. Kintsugi-27B needs a 12 GB card (the model alone takes about 9.5 GB): it may not load, or most of the context will be in system RAM and slow."
MARGIN="$(pick MARGIN $([ $HEADLESS = 1 ] && echo 250 || echo 600))"
if [ -n "$(pick MARGIN '')" ]; then MARGIN_WHY="KINTSUGI_MARGIN"; elif [ $HEADLESS = 1 ]; then MARGIN_WHY="no display on this card"; else MARGIN_WHY="display on this card: a graphics process in nvidia-smi; KINTSUGI_MARGIN=250 if that is wrong"; fi
# Fixed footprint at load without the K/V cells (measured on an RTX 4070 with the 10,546,686,688-byte file); a model file
# of another size moves it by the difference.
SIZE=$(stat -L -c %s "$MODEL")   # -L: the size of the model file itself, also when $MODEL is a symlink
HEAD_MIB=$(( (SIZE - 10546686688) / 1048576 ))
# ---- Drafting: DFlash2 (draft 7) when its drafter file is there, else the model's own MTP block (draft 3) + lookup ------
# The DFlash2 drafter is an optional download (models/dflash-Qwen3.8-27B-Q4_0.gguf, see README.md). It takes about 0.8 GB
# of VRAM, so fewer K/V cells fit in VRAM; measured on an RTX 4070 with these settings: about +25 to +35 % decode at
# 4k-64k tokens, about +23 % at 117k, the same speed at 245k.
DRAFTER="$(pick DRAFTER "$ROOT/models/dflash-Qwen3.8-27B-Q4_0.gguf")"
DRAFT_SET="$(pick DRAFT auto | tr '[:upper:]' '[:lower:]')"
DRAFT_NOTE=""; MTP_DRAFT=3
case "$DRAFT_SET" in
  auto)   if [ -f "$DRAFTER" ]; then MODE=dflash; else MODE=mtp
            DRAFT_NOTE="note   DFlash2 drafter not found ($DRAFTER): MTP + lookup drafting. For about 25-35 % faster decode up to ~64k tokens (about 23 % at 117k), download dflash-Qwen3.8-27B-Q4_0.gguf from https://huggingface.co/CaryPalmer/Kintsugi-27B into models/"; fi ;;
  dflash) [ -f "$DRAFTER" ] || { echo "DFlash2 drafter not found: $DRAFTER (set KINTSUGI_DRAFTER, or KINTSUGI_DRAFT=auto)"; exit 1; }; MODE=dflash ;;
  mtp)    MODE=mtp ;;
  off)    MODE=off ;;
  *[!0-9]*|"") echo "KINTSUGI_DRAFT must be auto, dflash, mtp, off or a number"; exit 1 ;;
  *)      MTP_DRAFT=$DRAFT_SET; if [ "$MTP_DRAFT" -gt 0 ]; then MODE=mtp; else MODE=off; fi ;;
esac
HEAD2="$(pick HEAD2 512)"
HEAD2_MIB=0; [ "$HEAD2" -gt 0 ] && HEAD2_MIB=606
# With DFlash2 (draft micro-batch 64): 10,238 MiB at load for the 1,094,346,080-byte drafter (RTX 4070); another drafter
# file moves it by the difference.
DRAFTER_MIB=0
[ "$MODE" = dflash ] && DRAFTER_MIB=$(( 765 + ($(stat -L -c %s "$DRAFTER") - 1094346080) / 1048576 ))
# ---- Vision: the Qwen3.8 projector (mmproj) reads images; an optional download -------------------------------------
# auto: on a card under 16 GB the projector runs on the CPU (it keeps every K/V cell in VRAM; on a 12 GB card the GPU
# projector would leave about 61,000 of 144,000), on 16 GB or more on the GPU (faster image reads).
# On the GPU it takes VRAM for its weights and for the work of one image at the cap (measured on an RTX 4070), so fewer
# K/V cells fit in VRAM; the window stays the same. On the CPU (KINTSUGI_VISION=cpu) it takes almost no VRAM and reads an
# image more slowly.
MMPROJ="$(pick MMPROJ "$ROOT/models/mmproj-Qwen3.8-27B-Q8_0.gguf")"
case "$(pick VISION auto | tr '[:upper:]' '[:lower:]')" in
  auto) if [ ! -f "$MMPROJ" ]; then VISION=off; elif [ "$TOTAL" -ge 15500 ]; then VISION=gpu; else VISION=cpu; fi ;;
  gpu)  [ -f "$MMPROJ" ] || { echo "vision projector not found: $MMPROJ (set KINTSUGI_MMPROJ, or KINTSUGI_VISION=auto)"; exit 1; }; VISION=gpu ;;
  cpu)  [ -f "$MMPROJ" ] || { echo "vision projector not found: $MMPROJ (set KINTSUGI_MMPROJ, or KINTSUGI_VISION=auto)"; exit 1; }; VISION=cpu ;;
  off)  VISION=off ;;
  *)    echo "KINTSUGI_VISION must be auto, gpu, cpu or off"; exit 1 ;;
esac
# DFlash2 does not work with images yet (an image late in a long conversation fails in the drafter): with vision on,
# drafting is MTP + lookup.
if [ "$VISION" != off ] && [ "$MODE" = dflash ]; then
  [ "$DRAFT_SET" = dflash ] && { echo "KINTSUGI_DRAFT=dflash does not work with vision on: set KINTSUGI_VISION=off for DFlash2, or KINTSUGI_DRAFT=auto"; exit 1; }
  MODE=mtp; DRAFTER_MIB=0
  DRAFT_NOTE="note   vision on: MTP + lookup drafting (the DFlash2 drafter does not work with images yet; KINTSUGI_VISION=off uses it)"
fi
IMG_MAX="$(pick IMAGE_MAX_TOKENS 2048)"
# Measured for the 629,247,008-byte Q8_0 projector at a 2048-token cap; another file moves it by the difference.
VISION_MIB=0; [ "$VISION" = gpu ] && VISION_MIB=$(( 852 + ($(stat -L -c %s "$MMPROJ") - 629247008) / 1048576 ))
FIXED=$((9473 + HEAD_MIB - HEAD2_MIB + DRAFTER_MIB + VISION_MIB))
# The first request adds about 222 MiB once (cuBLAS, kernel code at first use); the warm-up request pays it at start.
# Requests of many new lengths then add up to about 210 MiB (CUDA graphs); the margin covers that. With DFlash2, 128 MiB
# more: Linux needs about 50 MiB more for it than Windows, and the rest keeps the peak 64 MiB or more under the limit.
GROWTH=222; [ "$MODE" = dflash ] && GROWTH=$((222 + 128))
[ "$VISION" = gpu ] && GROWTH=$((GROWTH + 512))
# The driver keeps part of the card for itself ("Reserved" in nvidia-smi -q; 416 MiB on an RTX 4070 with Linux driver 595,
# 272 MiB with Windows). No program can use it: past the rest, an allocation fails (out of memory). Ceiling = the smaller
# of 97.7 % of the card and the card less the reserved part less 8 MiB. If nvidia-smi does not show the reserved part,
# 512 MiB is assumed. KINTSUGI_CEILING_MIB overrides.
RESERVED=$(nvidia-smi -q -d MEMORY -i 0 2> /dev/null | awk '/FB Memory Usage/ { fb = 1; next } fb && /Memory Usage/ { exit } fb && $1 == "Reserved" { print $3; exit }' | tr -dc '0-9')
if [ -z "$RESERVED" ]; then
  RESERVED=512
  echo "note   nvidia-smi does not show the card's reserved memory: 512 MiB assumed (KINTSUGI_CEILING_MIB overrides)"
fi
CEIL_A=$((TOTAL * 977 / 1000)); CEIL_B=$((TOTAL - RESERVED - 8)); [ "$CEIL_B" -lt "$CEIL_A" ] && CEIL_A=$CEIL_B
CEILING="$(pick CEILING_MIB $CEIL_A)"
cells_for() {   # cells_for FIXED: K/V cells that fit in VRAM (staging = one layer's K and V of the cells in system RAM)
  local lim=$TOTAL; [ "$CEILING" -lt "$lim" ] && lim=$CEILING
  local budget=$(( (lim - USED - MARGIN - $1 - GROWTH) * 1048576 - CTX * CELL / 16 ))
  local c=$(( budget * 16 / (CELL * 15) / 256 * 256 ))
  [ "$c" -lt 3328 ] && c=3328
  [ "$c" -gt "$CTX" ] && c=$CTX
  echo "$c"
}
CELLS="$(pick CELLS "$(cells_for $FIXED)")"
[ "$CELLS" -gt "$CTX" ] && CELLS=$CTX
if [ "$CELLS" -lt "$CTX" ] && [ $((CTX % 32768)) -ne 0 ]; then
  OLD=$CTX
  CTX=$(( (CTX + 32767) / 32768 * 32768 ))
  [ "$OLD" -le 262144 ] && [ "$CTX" -gt 262144 ] && CTX=262144
  echo "note   window $OLD rounded up to $CTX (a multiple of 32,768 with the tiered K/V cache)"
  CELLS="$(pick CELLS "$(cells_for $FIXED)")"; [ "$CELLS" -gt "$CTX" ] && CELLS=$CTX
fi
[ "$CELLS" -le 3328 ] && echo "note   only $CELLS K/V cells fit in VRAM: decode reads most of the K/V cache from system RAM (close other programs that use the card)"
# ---- Hand-set cells: the launcher's own count for this card and drafting mode is the safe maximum. Past it, the first
# ---- prompt read can fail with "out of memory" (RTX 3080 Ti with a display, DFlash2, 83,712 cells set by hand: about
# ---- 700 MiB too many). The launcher keeps the value that you set and tells you which setting to lower.
OWN_CELLS=$(cells_for $FIXED)
OWN_MTP=$OWN_CELLS
if [ "$MODE" = dflash ]; then
  G0=$GROWTH; GROWTH=$((GROWTH - 128)); OWN_MTP=$(cells_for $((FIXED - DRAFTER_MIB))); GROWTH=$G0
fi
lower_line() {   # lower_line: which setting to lower, for this card and drafting mode
  if [ "$MODE" = dflash ]; then
    echo "lower KINTSUGI_CELLS to $OWN_CELLS, or set KINTSUGI_DRAFT=mtp (MTP + lookup: at most $OWN_MTP cells)"
  else
    echo "lower KINTSUGI_CELLS to $OWN_CELLS"
  fi
}
if [ -n "$(pick CELLS '')" ] && [ "$CELLS" -gt "$OWN_CELLS" ]; then
  case "$MODE" in dflash) M_TXT="DFlash2 drafting" ;; mtp) M_TXT="MTP + lookup drafting" ;; *) M_TXT="no drafting" ;; esac
  echo "WARN   KINTSUGI_CELLS=$CELLS is more than this card can hold with $M_TXT: the safe maximum is $OWN_CELLS cells (free $((TOTAL - USED)) MiB, ceiling $CEILING, margin $MARGIN). A prompt read can fail with \"out of memory\"."
  echo "       To fix it: $(lower_line). The launcher keeps your value."
fi

# ---- Drafting arguments ----------------------------------------------------------------------------------------------
DRAFT=0; [ "$MODE" = mtp ] && DRAFT=$MTP_DRAFT; [ "$MODE" = dflash ] && DRAFT=7
SPEC=()
if [ "$MODE" = dflash ]; then
  SPEC=(--spec-type draft-dflash -md "$DRAFTER" --spec-draft-n-max 7 -ctkd q4_0 -ctvd q4_0)
elif [ "$DRAFT" -gt 0 ]; then
  TAIL=$DRAFT; [ "$TAIL" -lt 4 ] && TAIL=4
  SPEC=(--spec-type ngram-mod,draft-mtp --spec-draft-n-max "$DRAFT" --spec-draft-n-max-tail "$TAIL"
        -ctkd q8_0 -ctvd q8_0 --spec-lookup-n-max 32 --spec-draft-window 16384)
fi

# ---- Engine switches -------------------------------------------------------------------------------------------------
export GGML_CUDA_BATCH_INVARIANT=1
SET_DFLASH_UB=0; [ "$MODE" = dflash ] && [ -z "${LLAMA_DFLASH_DRAFT_UBATCH:-}" ] && SET_DFLASH_UB=1
[ "$MODE" = dflash ] && export LLAMA_DFLASH_DRAFT_UBATCH="${LLAMA_DFLASH_DRAFT_UBATCH:-64}"
if [ "$DRAFT" -gt 0 ]; then export LLAMA_RS_REPLAY=1 GGML_CUDA_SHARED_POOL=1; export LLAMA_MTP_DRAFT_UBATCH="${LLAMA_MTP_DRAFT_UBATCH:-256}"; fi
export LLAMA_ARG_CHECKPOINT_EVERY_NT=8192
export GGML_CUDA_POOL_RESERVE_MIB="${GGML_CUDA_POOL_RESERVE_MIB:-96}"
export GGML_CUDA_DEDICATED_CEILING_MIB="${GGML_CUDA_DEDICATED_CEILING_MIB:-$CEILING}"
if [ "$HEAD2" -gt 0 ]; then export LLAMA_HEAD2="$HEAD2" LLAMA_HEAD2_TYPE="${LLAMA_HEAD2_TYPE:-q4_0}"; fi
READ="$(pick READ fast)"
if [ -z "${GGML_ESCHA_PREFILL:-}" ]; then
  if [ "$READ" = exact ]; then export GGML_ESCHA_PREFILL=recon; else export GGML_ESCHA_PREFILL=i8; fi
fi
export GGML_ESCHA_I8_MIN_ROWS="${GGML_ESCHA_I8_MIN_ROWS:-256}"
if [ "$THINK" = off ]; then
  export LLAMA_ARG_CHAT_TEMPLATE_KWARGS='{"reasoning_effort":"medium","enable_thinking":false}'
else
  export LLAMA_ARG_CHAT_TEMPLATE_KWARGS='{"reasoning_effort":"medium"}'
fi
export LD_LIBRARY_PATH="$BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"   # the engine and the CUDA runtime libraries are in bin/

KEY_FILE="$(pick KEY_FILE '')"
LISTEN="$(pick HOST 127.0.0.1)"
if [ "$LISTEN" != 127.0.0.1 ] && [ "$LISTEN" != localhost ] && [ -z "$KEY_FILE" ]; then
  echo "WARNING listening on $LISTEN with NO API key: anyone who can reach this computer can use the server (and your GPU). Set KINTSUGI_KEY_FILE to a file with a key."
fi

# ---- Saved conversations: the layer (layer/bonsai_layer.py) runs in front of the server on the client port; the
# ---- server moves to 127.0.0.1:<port + 10000>. It needs Python 3.10 or newer (standard library only).
USE_LAYER=0; [ "$(pick LAYER 1)" = 1 ] && USE_LAYER=1
PY=""
if [ $USE_LAYER = 1 ]; then
  PY="$(command -v python3 || command -v python || true)"
  if [ -z "$PY" ]; then
    echo "note   no python3 in PATH: saved conversations are off (the plain server runs). Install Python 3.10+ to use them, or set KINTSUGI_LAYER=0 to hide this note."
    USE_LAYER=0
  elif ! "$PY" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2> /dev/null; then
    echo "note   saved conversations are off: they need Python 3.10 or newer (the python in PATH is older or does not run). The plain server runs."
    USE_LAYER=0
  fi
fi
SERVER_PORT=$PORT
SLOT_DIR=""
if [ $USE_LAYER = 1 ]; then
  SERVER_PORT=$((PORT + 10000))
  SLOT_DIR="$(pick SLOT_SAVE_PATH "$ROOT/slots")"
  [ $DRY = 0 ] && mkdir -p "$SLOT_DIR" && SLOT_DIR="$(cd "$SLOT_DIR" && pwd -P)"
fi
UB="$(pick UB 1024)"
TEMPLATE="$(pick CHAT_TEMPLATE "$ROOT/templates/kintsugi.jinja")"
if [ "$TEMPLATE" != model ] && [ ! -f "$TEMPLATE" ]; then echo "chat template not found: $TEMPLATE (set KINTSUGI_CHAT_TEMPLATE)"; exit 1; fi
SRV_HOST=$LISTEN; [ $USE_LAYER = 1 ] && SRV_HOST=127.0.0.1
ARGS=(-m "$MODEL" -ngl 99 -c "$CTX" -fa on -np 1 -ub "$UB" --fit off --jinja
      -ctk "$KV" -ctv "$KV" --kv-vram-cells "$CELLS" --kq-mask-packed --cache-ram 4096 --backend-sampling
      --reasoning-budget 20480 -n 24576 --reasoning-budget-message "Now produce the complete answer."
      --reasoning-effort-allow medium --reasoning-effort-fallback medium --reasoning-max-tokens-floor 24576
      --host "$SRV_HOST" --port "$SERVER_PORT" ${SPEC[@]+"${SPEC[@]}"})
[ "$TEMPLATE" != model ] && ARGS+=(--chat-template-file "$TEMPLATE")
[ -n "$KEY_FILE" ] && ARGS+=(--api-key-file "$KEY_FILE")
[ -n "$SLOT_DIR" ] && ARGS+=(--slot-save-path "$SLOT_DIR")
[ "$VISION" != off ] && ARGS+=(--mmproj "$MMPROJ" --image-max-tokens "$IMG_MAX")
[ "$VISION" = cpu ] && ARGS+=(--no-mmproj-offload)
EXTRA="$(pick EXTRA '')"
# The model name clients see (/v1/models, the "model" of each reply): "kintsugi", not the model file's path.
ALIAS="$(pick ALIAS kintsugi)"
[[ " $EXTRA " =~ [[:space:]](--alias|-a)[[:space:]] ]] || ARGS+=(--alias "$ALIAS")
[ -n "$EXTRA" ] && read -r -a EXTRA_ARGS <<< "$EXTRA" && ARGS+=("${EXTRA_ARGS[@]}")

echo "Kintsugi-27B"
echo "model  $(basename "$MODEL")"
echo "listen $LISTEN:$PORT$([ $USE_LAYER = 1 ] && echo " (layer; llama-server on 127.0.0.1:$SERVER_PORT)")$([ -n "$KEY_FILE" ] && echo ", API key on")"
echo "kv     $KV, tiered: cells 0..$CELLS in VRAM, $CELLS..$CTX in system RAM (free $((TOTAL - USED)) MiB, reserved $RESERVED, ceiling $CEILING, fixed $FIXED, growth $GROWTH, margin $MARGIN ($MARGIN_WHY))"
echo "template $([ "$TEMPLATE" = model ] && echo "the model file's own" || echo "$TEMPLATE")"
[ "$HEAD2" -gt 0 ] && echo "head   two-stage: first stage q4_0 in VRAM, $HEAD2 exact candidates for each row from system RAM"
case "$MODE" in off) SPEC_TXT=off ;; dflash) SPEC_TXT="DFlash2 draft 7 ($(basename "$DRAFTER"))" ;; *) SPEC_TXT="MTP draft $DRAFT + lookup" ;; esac
echo "spec   $SPEC_TXT; thinking $THINK; read $GGML_ESCHA_PREFILL"
[ -n "$DRAFT_NOTE" ] && echo "$DRAFT_NOTE"
if [ "$VISION" = off ]; then echo "vision off"; else echo "vision $VISION ($(basename "$MMPROJ")), up to $IMG_MAX tokens per image"; fi
if [ $DRY = 1 ]; then echo "args   ${ARGS[*]}"; exit 0; fi

LOGS="$ROOT/logs"; mkdir -p "$LOGS"
LOG="$LOGS/server-$PORT.log"
setsid nohup "$BIN/llama-server" "${ARGS[@]}" > "$LOG.out" 2> "$LOG" < /dev/null &
SP=$!
echo "$SP" > "$LOGS/server-$PORT.pid"
echo "pid    $SP   log $LOG"
LAYER_LOG="$LOGS/layer-$PORT.log"
if [ $USE_LAYER = 1 ]; then
  LAYER_KEY=""; [ -n "$KEY_FILE" ] && LAYER_KEY="$(tr -d '\r\n' < "$KEY_FILE")"
  BONSAI_SLOT_SAVE_PATH="$SLOT_DIR" BONSAI_SLOT_CACHE_GB="$(pick SLOT_CACHE_GB 20)" BONSAI_SLOT_CACHE_AFTER=1 BONSAI_LAYER_KEY="$LAYER_KEY" \
    setsid nohup "$PY" "$ROOT/layer/bonsai_layer.py" --host "$LISTEN" --port "$PORT" --upstream "http://127.0.0.1:$SERVER_PORT" \
      --no-interpreter --no-cards --no-lint --no-example-check --no-finish-note --no-input-file --no-strip-stray-tags \
      > "$LAYER_LOG.out" 2> "$LAYER_LOG" < /dev/null &
  LP=$!
  echo "$LP" > "$LOGS/layer-$PORT.pid"
  echo "layer  pid $LP   log $LAYER_LOG   clients use :$PORT; saved conversations in $SLOT_DIR"
fi
echo "stop   $ROOT/stop.sh"
alive() { kill -0 "$1" 2> /dev/null; }
stop_both() {   # stop the server and the layer, then wait until they are gone
  local x i
  for x in ${LP:-} $SP; do alive "$x" && kill "$x" 2> /dev/null; done
  for i in $(seq 1 30); do alive "$SP" || { [ -n "${LP:-}" ] && alive "$LP"; } || break; sleep 1; done
  for x in ${LP:-} $SP; do alive "$x" && kill -9 "$x" 2> /dev/null; done
}
# DFlash2 -> MTP, once: when DFlash2 was chosen automatically (KINTSUGI_DRAFT=auto) and the server stops during the load
# or the warm-up request fails, keep that log, stop the server (and the layer) and start again with MTP + lookup
# drafting. KINTSUGI_DRAFT=dflash fails instead. Only with --wait (the warm-up needs it).
retry_mtp() {   # retry_mtp <why>: returns only when no retry applies
  [ "$MODE" = dflash ] && [ "$DRAFT_SET" = auto ] && [ -z "${KINTSUGI_DFLASH_RETRIED:-}" ] || return 0
  local fail="${LOG%.log}.dflash-failed.log" i u
  cp -f "$LOG" "$fail" 2> /dev/null
  echo "note   DFlash2 start failed ($1): starting again once with MTP + lookup drafting (KINTSUGI_DRAFT=mtp skips DFlash2)."
  echo "       Please report it with your card, driver and the log $fail at https://github.com/professorpalmer/Kintsugi-27B/issues"
  grep -a -i -E "out of memory|cudaMalloc|failed to allocate|CUDA error|error|failed|abort" "$LOG" 2> /dev/null | tail -6 | cut -c1-240 | sed "s/^/  log  /"
  stop_both
  # the driver frees the memory of a stopped process a moment later; the cell count of the retry reads the free VRAM
  for i in $(seq 1 20); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -dc '0-9')
    [ -n "$u" ] && [ "$u" -le $((USED + 64)) ] && break; sleep 1
  done
  [ "$SET_DFLASH_UB" = 1 ] && unset LLAMA_DFLASH_DRAFT_UBATCH
  exec env KINTSUGI_DRAFT=mtp KINTSUGI_DFLASH_RETRIED=1 bash "$ROOT/start-server.sh" ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"}
}
if [ $WAIT = 0 ]; then
  # No wait for the load: the engine writes its GPU set-up to the log before the model load starts.
  for i in $(seq 1 60); do
    { ! alive "$SP" || grep -q -F 'load_model: loading model' "$LOG" 2> /dev/null; } && break
    sleep 1
  done
  M="$(no_gpu_message "$LOG")"; [ -n "$M" ] && stop_no_gpu "$M"
  if ! alive "$SP"; then
    [ -n "$LP" ] && kill "$LP" 2> /dev/null   # do not leave the layer running with no server behind it
    echo "server stopped during load: see $LOG"; tail -n 3 "$LOG" | sed 's/^/  /'; exit 1
  fi
  exit 0
fi
AUTH=(); [ -n "$KEY_FILE" ] && AUTH=(-H "Authorization: Bearer $(tr -d '\r\n' < "$KEY_FILE")")
for i in $(seq 1 150); do
  if curl -sf -m 3 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
    M="$(no_gpu_message "$LOG")"; [ -n "$M" ] && stop_no_gpu "$M"
    # Warm-up request (about 1,340 tokens in, 8 out; more with KINTSUGI_UB above 1024, so that the prompt fills one full
    # micro-batch): the engine takes its first-use memory and its prompt-read buffers now, so the VRAM printed below is
    # the peak, and a shortage shows here and not at the first request.
    NWARM=$((UB / 11 + 16)); [ "$NWARM" -lt 110 ] && NWARM=110
    TEXT="$(for n in $(seq 1 $NWARM); do printf 'Entry %d of the log has nothing to report. ' "$n"; done)"
    BODY="{\"model\":\"warmup\",\"max_tokens\":8,\"temperature\":0,\"cache_prompt\":false,\"chat_template_kwargs\":{\"enable_thinking\":false},\"messages\":[{\"role\":\"user\",\"content\":\"${TEXT}Say OK.\"}]}"
    T0=$(date +%s.%N)
    WARM_OK=1
    curl -sf -m 600 ${AUTH[@]+"${AUTH[@]}"} -H 'Content-Type: application/json' -d "$BODY" "http://127.0.0.1:$PORT/v1/chat/completions" > /dev/null || WARM_OK=0
    sleep 0.5
    if [ $WARM_OK = 0 ] || ! alive "$SP"; then
      if alive "$SP"; then retry_mtp "the warm-up request failed"; else retry_mtp "the server stopped during the warm-up request"; fi
      OOM=0; grep -a -q -i -E "out of memory|failed to allocate" "$LOG" 2> /dev/null && OOM=1
      if [ "$MODE" = dflash ] && [ "$DRAFT_SET" = dflash ]; then
        # KINTSUGI_DRAFT=dflash: no fallback to MTP and no "ready"; keep the log, stop the server, exit 1
        FAIL="${LOG%.log}.dflash-failed.log"; cp -f "$LOG" "$FAIL" 2> /dev/null
        echo "note   the warm-up request failed with KINTSUGI_DRAFT=dflash (no fallback to MTP when DFlash2 is set). Log kept: $FAIL"
        grep -a -i -E "out of memory|cudaMalloc|failed to allocate|CUDA error|error|failed|abort" "$LOG" 2> /dev/null | tail -6 | cut -c1-240 | sed "s/^/  log  /"
        [ $OOM = 1 ] && echo "VRAM   the server is short of VRAM with DFlash2 (out of memory). To fix it: $(lower_line)."
        stop_both
        echo "stopped: the server is not ready (the server and the layer are stopped)"
        exit 1
      fi
      echo "note   the warm-up request failed"
      [ $OOM = 1 ] && echo "VRAM   the server is short of VRAM (out of memory in $LOG): requests will fail. Stop it ($ROOT/stop.sh), then $(lower_line)."
      alive "$SP" || { echo "server stopped after the warm-up request: see $LOG"; exit 1; }
    fi
    T1=$(date +%s.%N)
    VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -dc '0-9')
    PROC=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2> /dev/null | awk -F', *' -v p="$SP" '$1 == p { print $2 }')
    echo "warm   $(awk -v a="$T0" -v b="$T1" 'BEGIN { printf "%.1f", b - a }') s; VRAM $VRAM MiB (ceiling $CEILING), server process ${PROC:-?} MiB"
    # Linux has no silent move of GPU memory to system RAM (an allocation that does not fit fails); the engine's own
    # spill lines are still checked.
    if grep -q 'memory_guard: spill' "$LOG" 2> /dev/null; then
      echo "SPILL  the engine reports GPU memory in system RAM: start again with a larger KINTSUGI_MARGIN (now $MARGIN) or fewer KINTSUGI_CELLS (now $CELLS)."
    fi
    echo "ready  http://127.0.0.1:$PORT/v1"
    exit 0
  fi
  if ! alive "$SP"; then
    M="$(no_gpu_message "$LOG")"; [ -n "$M" ] && stop_no_gpu "$M"
    retry_mtp "the server stopped during the load"
    [ -n "$LP" ] && kill "$LP" 2> /dev/null   # do not leave the layer running with no server behind it
    echo "server stopped during load: see $LOG"; tail -n 3 "$LOG" | sed 's/^/  /'; exit 1
  fi
  if [ -n "$LP" ] && ! alive "$LP"; then echo "layer stopped: see $LAYER_LOG"; exit 1; fi
  sleep 2
done
echo "not ready after 300 s: see $LOG"
exit 1
