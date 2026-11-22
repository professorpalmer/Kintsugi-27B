#!/usr/bin/env bash
# Kintsugi-27B on Apple Silicon: setup (Mac only).
#   - checks macOS, Apple Silicon, RAM and free disk
#   - makes a Python venv (.venv) and installs the pinned MLX packages into it
#   - clones escha-mlx at 69f7218 and applies patches/escha-mlx-69f7218-kintsugi.patch
#   - downloads Kintsugi-27B-12G.gguf (10.5 GB, resumable) and checks its sha256
# It needs no sudo and changes no system setting. Safe to run again: done steps are skipped.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd)
ESCHA_COMMIT=69f72187
GGUF_URL="https://huggingface.co/CaryPalmer/Kintsugi-27B/resolve/main/Kintsugi-27B-12G.gguf"
GGUF_SHA=0510bae82445dc53dbc7d0d842d8b7e146b8eafce0837e39b157d411e48c8a74
GGUF_BYTES=10546687168
NEED_DISK_GB=25
MIN_RAM_GB=24       # the full profile
MIN_RAM_GB_16=16    # the 16 GB profile (experimental): embedding on the CPU side, context 8k or 16k (kit/common.py)
PINS="mlx==0.32.0 mlx-lm==0.31.3 numpy>=2.0 safetensors>=0.4 pytest>=7"

say()  { printf '\n== %s\n' "$*"; }
fail() { printf '\nSTOP: %s\n' "$*" >&2; exit 1; }

say "1/5 machine check"
[ "$(uname -s)" = "Darwin" ] || fail "this kit runs on macOS only."
[ "$(uname -m)" = "arm64" ]  || fail "this kit needs Apple Silicon (arm64). This shell reports $(uname -m). If this is a Mac with Apple Silicon, do not run the terminal under Rosetta."
MACOS=$(sw_vers -productVersion); MAJOR=${MACOS%%.*}
[ "$MAJOR" -ge 14 ] || fail "macOS 14 or later is necessary (this Mac has $MACOS)."
RAM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
CHIP=$(sysctl -n machdep.cpu.brand_string)
echo "chip: $CHIP, RAM: ${RAM_GB} GB, macOS $MACOS"
NOTE_16="16 GB profile (experimental; not yet tested on a real 16 GB Mac): the token embedding stays on the CPU side,
the context is 8k (16k if it fits), and 32k / 262k are not tested. Close other apps (browsers, chat apps) before the run:
the model needs about 9.5 GB of GPU memory."
if [ "$RAM_GB" -lt "$MIN_RAM_GB" ]; then
  if [ "$RAM_GB" -eq "$MIN_RAM_GB_16" ]; then
    echo "$NOTE_16"
  elif [ "$RAM_GB" -gt "$MIN_RAM_GB_16" ] && [ "${KINTSUGI_PROFILE:-}" = "16gb" ]; then
    echo "${RAM_GB} GB RAM, KINTSUGI_PROFILE=16gb."
    echo "$NOTE_16"
  elif [ "${KINTSUGI_FORCE:-0}" = "1" ]; then
    WANT_GB=$MIN_RAM_GB; [ "$RAM_GB" -lt "$MIN_RAM_GB_16" ] && WANT_GB=$MIN_RAM_GB_16
    echo "WARNING: ${RAM_GB} GB RAM is less than ${WANT_GB} GB. Continuing because KINTSUGI_FORCE=1."
  elif [ "$RAM_GB" -gt "$MIN_RAM_GB_16" ]; then
    fail "${RAM_GB} GB RAM is less than ${MIN_RAM_GB} GB. The full profile needs ${MIN_RAM_GB} GB. To use the experimental
   16 GB profile: KINTSUGI_PROFILE=16gb ./run.sh. To try the full profile anyway: KINTSUGI_FORCE=1 ./run.sh"
  else
    fail "${RAM_GB} GB RAM is less than ${MIN_RAM_GB_16} GB. The model needs about 9.5 GB of GPU memory and does not fit.
   To try anyway: KINTSUGI_FORCE=1 ./run.sh"
  fi
fi
HAVE_GB=$(df -g "$ROOT" | awk 'NR==2 {print $4}')
NEED=$NEED_DISK_GB
[ -f models/Kintsugi-27B-12G.gguf ] && NEED=$((NEED - 10))
[ -d models/Kintsugi-27B-MLX ] && NEED=$((NEED - 10))
[ "$HAVE_GB" -ge "$NEED" ] || fail "free disk is ${HAVE_GB} GB; this kit needs about ${NEED} GB more in $ROOT."
echo "free disk: ${HAVE_GB} GB (need about ${NEED} GB)"

say "2/5 Python"
PY=""
for c in python3.12 python3.13 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then
    if "$c" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)' 2>/dev/null; then PY=$(command -v "$c"); break; fi
  fi
done
if [ -z "$PY" ]; then
  fail "Python 3.10 to 3.13 is necessary (macOS's own python3 is 3.9). Install one of these, then run ./run.sh again:
   - Homebrew:   brew install python@3.12
   - python.org: https://www.python.org/downloads/macos/  (the 3.12 installer)"
fi
echo "python: $PY ($("$PY" --version))"
if [ ! -x .venv/bin/python ]; then
  "$PY" -m venv .venv
fi
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet $PINS
echo "installed: $(.venv/bin/python -c 'import mlx.core as mx, importlib.metadata as m; print("mlx", m.version("mlx"), "mlx-lm", m.version("mlx-lm"), "metal", mx.metal.is_available())')"

say "3/5 escha-mlx at $ESCHA_COMMIT + our patch"
if [ ! -d escha-mlx/.git ]; then
  git clone --quiet https://github.com/EschaLabs/escha-mlx.git escha-mlx
fi
( cd escha-mlx
  git fetch --quiet origin || true
  if git apply --reverse --check ../patches/escha-mlx-69f7218-kintsugi.patch 2>/dev/null; then
    echo "patch already applied"
  else
    # A new patch version (or local work) is in escha-mlx/. Delete nothing: stash all changes (tracked and
    # untracked), leave any branch where it is, and check out the pinned commit detached, then apply our patch.
    if [ -n "$(git status --porcelain)" ]; then
      git stash push --quiet --include-untracked -m "kintsugi-kit backup $(date +%F-%H%M)"
      echo "local changes in escha-mlx/ are saved: git -C escha-mlx stash list"
    fi
    git checkout --quiet --detach "$ESCHA_COMMIT"
    git apply ../patches/escha-mlx-69f7218-kintsugi.patch
    echo "escha-mlx $(git rev-parse --short HEAD) + patch applied"
  fi )
.venv/bin/python -m pip install --quiet --no-deps -e ./escha-mlx

say "4/5 model download (10.5 GB; it resumes if it stops)"
mkdir -p models
G=models/Kintsugi-27B-12G.gguf
if [ -f "$G.ok" ] && [ "$(stat -Lf %z "$G" 2>/dev/null || echo 0)" = "$GGUF_BYTES" ]; then
  echo "already downloaded and checked"
else
  if [ "$(stat -Lf %z "$G" 2>/dev/null || echo 0)" -gt "$GGUF_BYTES" ]; then rm -f "$G"; fi
  until [ "$(stat -Lf %z "$G" 2>/dev/null || echo 0)" = "$GGUF_BYTES" ]; do
    curl -L --fail --retry 5 --retry-delay 5 -C - -o "$G" "$GGUF_URL" || { echo "download stopped; retrying in 10 s"; sleep 10; }
  done
  say "5/5 sha256 check (about 1 minute)"
  GOT=$(shasum -a 256 "$G" | awk '{print $1}')
  [ "$GOT" = "$GGUF_SHA" ] || { rm -f "$G.ok"; fail "sha256 mismatch: got $GOT, want $GGUF_SHA. Delete $G and run again."; }
  touch "$G.ok"
  echo "sha256 OK"
fi
echo
echo "setup done."
