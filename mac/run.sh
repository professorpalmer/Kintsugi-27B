#!/usr/bin/env bash
# Kintsugi-27B on a Mac.
#   ./run.sh install   setup + convert (+ byte checks): the model is then ready to use
#   ./run.sh chat      chat in this terminal
#   ./run.sh serve     OpenAI-compatible server on http://127.0.0.1:8080/v1 (KINTSUGI_PORT, KINTSUGI_HOST)
#   ./run.sh           the full kit: setup -> convert (+ byte checks) -> checks -> results zip (measure your Mac)
set -uo pipefail
cd "$(dirname "$0")"
case "${1:-}" in
  install)
    ./setup.sh || { echo "setup stopped (see the message above)."; exit 1; }
    .venv/bin/python convert.py || { echo "convert or its byte checks FAILED (see the message above)."; exit 1; }
    echo "Installed. Next: ./run.sh chat   or   ./run.sh serve"
    exit 0 ;;
  chat|serve)
    MODE=$1; shift
    if [ ! -x .venv/bin/python ] || [ ! -f models/Kintsugi-27B-MLX/config.json ]; then
      echo "The model is not installed yet. Run ./run.sh install first (download and convert, once)."
      exit 1
    fi
    exec .venv/bin/python "kit/$MODE.py" "$@" ;;
  "") ;;
  *) echo "usage: ./run.sh [install | chat | serve]"; exit 2 ;;
esac
export KINTSUGI_RESULTS="$(pwd)/results"
mkdir -p results
START=$(date +%s)
echo "Kintsugi-27B Mac kit. Started $(date). Log: results/run.log"
(
  ./setup.sh || { echo "setup stopped (see the message above)."; exit 1; }
  printf '\n== convert (%s)\n' "$(date +%H:%M:%S)"
  if ! .venv/bin/python convert.py; then
    echo "convert or its byte checks FAILED: the model checks are not run. Making the results zip."
    .venv/bin/python kit/env_info.py || true
    .venv/bin/python kit/summarize.py
    exit 1
  fi
  ./run_checks.sh
) 2>&1 | tee -a results/run.log
RC=${PIPESTATUS[0]}
echo "total time: $(( ($(date +%s) - START) / 60 )) min" | tee -a results/run.log
Z=$(ls -1t results-*.zip 2>/dev/null | head -1)
[ -n "$Z" ] && echo "SEND THIS FILE BACK: $(pwd)/$Z"
exit "$RC"
