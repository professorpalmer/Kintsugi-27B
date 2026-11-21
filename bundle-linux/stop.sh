#!/usr/bin/env bash
# Stop the Kintsugi-27B server and the layer started from THIS folder. Servers started from other folders are left alone.
#   ./stop.sh
# If the layer is saving a chat right now (it saves a long chat after each answer), the stop waits for that save, up to
# 30 s, so the chat can be restored after the next start. With no layer (layer off, or no Python) it stops at once.
# the bundle folder, resolved (readlink -f): the same path whether the script, the folder or a parent is reached
# through a symlink, so start-server.sh and stop.sh always agree on the layer and engine paths
ROOT="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
BIN="$ROOT/bin"   # ROOT is resolved, as /proc/<pid>/exe shows the program file
LAYER="$ROOT/layer/bonsai_layer.py"

# the layer processes of this folder: python with this folder's bonsai_layer.py on its command line
layers=()
for d in /proc/[0-9]*; do
  [ -r "$d/cmdline" ] || continue
  cmd="$(tr '\0' ' ' < "$d/cmdline" 2> /dev/null)"
  case "$cmd" in *"$LAYER"*) layers+=("${d#/proc/}") ;; esac
done

# ---- wait for a chat save in flight: the layer answers GET /layer/slot-cache with "busy" (a chat request with its save
#      is running) and "saving" (a background save after an answer is still being written)
for lp in ${layers[@]+"${layers[@]}"}; do
  cmd="$(tr '\0' ' ' < "/proc/$lp/cmdline" 2> /dev/null)"
  port="$(echo "$cmd" | sed -n 's/.*--port \([0-9]*\).*/\1/p')"; port="${port:-8080}"
  host="$(echo "$cmd" | sed -n 's/.*--host \([^ ]*\).*/\1/p')"; host="${host:-127.0.0.1}"
  [ "$host" = 0.0.0.0 ] || [ "$host" = "::" ] && host=127.0.0.1
  told=0
  for i in $(seq 1 60); do
    r="$(curl -sf -m 2 "http://$host:$port/layer/slot-cache" 2> /dev/null)" || break   # no answer: stop at once
    echo "$r" | grep -q -E '"(busy|saving)": *true' || break
    [ $told = 0 ] && { echo "waiting for a chat save to finish..."; told=1; }
    sleep 0.5
  done
done

n=0
# the servers of this folder: a llama-server process whose program file is in this folder's bin/
for d in /proc/[0-9]*; do
  exe="$(readlink "$d/exe" 2> /dev/null)" || continue
  [ "$exe" = "$BIN/llama-server" ] || continue
  p="${d#/proc/}"
  kill "$p" 2> /dev/null
  for i in $(seq 1 30); do kill -0 "$p" 2> /dev/null || break; sleep 0.5; done
  kill -9 "$p" 2> /dev/null
  n=$((n + 1))
done
for lp in ${layers[@]+"${layers[@]}"}; do kill -9 "$lp" 2> /dev/null && n=$((n + 1)); done
rm -f "$ROOT"/logs/server-*.pid "$ROOT"/logs/layer-*.pid
echo "stopped $n"
