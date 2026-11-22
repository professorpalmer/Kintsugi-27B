#!/usr/bin/env bash
# Kintsugi-27B on Apple Silicon: the checks (Mac only). ./run.sh runs setup, convert and this script in order.
# Every step writes into results/. A failed step does not stop the next one. At the end: results-<chip>-<ram>.zip.
# No sudo, no system setting changes. If a larger Metal memory limit would help, the kit prints the command; you
# decide whether to run it.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
export KINTSUGI_RESULTS="$(pwd)/results"
mkdir -p results
STATUS=results/steps.txt

step() {  # step <name> <command...>
  local name=$1; shift
  printf '\n== %s (%s)\n' "$name" "$(date +%H:%M:%S)"
  local t0 r
  t0=$(date +%s)
  if "$@" 2>&1 | tee "results/$name.log"; then r=ok; else r=FAILED; fi
  echo "$name: $r ($(( $(date +%s) - t0 )) s)" | tee -a "$STATUS"
}

PROFILE=$($PY -c 'import sys; sys.path.insert(0, "kit"); import common; print(common.PROFILE)')
echo "profile: $PROFILE"
step env      $PY kit/env_info.py
[ "$PROFILE" = "16gb" ] && step embed $PY kit/embed_cpu_check.py
step tests    $PY kit/escha_tests.py
step kl       $PY kit/kl_mini.py
step chat     $PY kit/chat_sanity.py
step parser   $PY kit/test_parser.py
step session  $PY kit/check_session.py
serve_step() {   # the user-mode server on a spare port, then its checks (chat 6/6, tool calls t1-t4)
  KINTSUGI_PORT=8093 $PY kit/serve.py > results/serve_server.log 2>&1 &
  local srv=$! i
  for i in $(seq 1 120); do grep -q "ready" results/serve_server.log && break; kill -0 $srv 2>/dev/null || break; sleep 2; done
  $PY kit/serve_check.py --url http://127.0.0.1:8093/v1 --out results/serve_check.json
  local rc=$?
  kill $srv 2>/dev/null; wait $srv 2>/dev/null
  return $rc
}
step serve    serve_step
step speed    $PY kit/speed_needles.py
step summary  $PY kit/summarize.py
