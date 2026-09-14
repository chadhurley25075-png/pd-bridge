#!/usr/bin/env bash
# pd_memguard.sh — stop the prefill engine before the Spark runs out of unified memory.
#
# The limit is expressed on the box's nominal scale: stop when used reaches PD_MEM_STOP_GB of PD_MEM_TOTAL_NOMINAL_GB
# (default 125 of 128). It is applied as a FRACTION of the MemTotal the kernel actually reports (121 GiB on a GB10), so
# the GB/GiB/firmware-reserve question cannot move the threshold: 125/128 trips with ~2.8 GiB still available.
#   used = MemTotal - MemAvailable (what the kernel cannot reclaim)
#
# At the limit the vLLM engine is terminated (TERM, then KILL after a grace period) and a marker is written, so the
# benchmark driver ends the run instead of recording a native fallback. Upstream lost a box to swap thrash at 1M tokens
# and needed the power button (README, "memory floor"); this trips first.
#
# Headroom warning: one R0 prefill step copies ~2 GB device->host (8192 tokens x 256 KiB), so a thin margin can be
# crossed between two polls. The poll is 0.2 s for that reason.
#
# Exits 3 when it tripped, 0 when the engine exited on its own (after having been seen).
set -uo pipefail

: "${PD_MEM_STOP_GB:=125}"
: "${PD_MEM_TOTAL_NOMINAL_GB:=128}"
: "${PD_MEMGUARD_INTERVAL:=0.2}"
: "${PD_MEMGUARD_MARKER:=$HOME/pd_logs/MEMGUARD_TRIPPED}"
: "${PD_MEMGUARD_PATTERN:=bin/vllm serve|VLLM::EngineCore}"
: "${PD_MEMGUARD_GRACE_S:=5}"

meminfo() { awk '/^MemTotal:/ { t = $2 } /^MemAvailable:/ { a = $2 } END { print t, a }' /proc/meminfo; }
gib() { awk -v k="$1" 'BEGIN { printf "%.1f", k / 1048576 }'; }
log() { echo "$(date '+%F %T') memguard: $*"; }

# Matching processes, never this guard or any shell that launched it (their command lines contain "pd_memguard";
# a bare `pkill -f PATTERN` also kills the ssh shell whose command line happens to spell the pattern).
targets() {
  local p
  for p in $(pgrep -f "$PD_MEMGUARD_PATTERN"); do
    [ "$p" = "$$" ] && continue
    tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q pd_memguard && continue
    echo "$p"
  done
}

read -r total_kb _ < <(meminfo)
limit_kb=$(awk -v t="$total_kb" -v s="$PD_MEM_STOP_GB" -v n="$PD_MEM_TOTAL_NOMINAL_GB" 'BEGIN { printf "%d", t * s / n }')

mkdir -p "$(dirname "$PD_MEMGUARD_MARKER")"
rm -f "$PD_MEMGUARD_MARKER"
log "armed: stop at ${PD_MEM_STOP_GB}/${PD_MEM_TOTAL_NOMINAL_GB} of MemTotal $(gib "$total_kb") GiB = used >= $(gib "$limit_kb") GiB (available <= $(gib $((total_kb - limit_kb))) GiB), every ${PD_MEMGUARD_INTERVAL}s, pattern '${PD_MEMGUARD_PATTERN}'"

peak_kb=0; seen=0; tick=0
while true; do
  read -r total_kb avail_kb < <(meminfo)
  used_kb=$((total_kb - avail_kb))
  [ "$used_kb" -gt "$peak_kb" ] && peak_kb=$used_kb

  if [ "$used_kb" -ge "$limit_kb" ]; then
    msg="used $(gib "$used_kb") GiB >= $(gib "$limit_kb") GiB (${PD_MEM_STOP_GB}/${PD_MEM_TOTAL_NOMINAL_GB}), available $(gib "$avail_kb") GiB"
    log "TRIPPED: $msg — terminating the engine"
    echo "$(date +%s) $msg" > "$PD_MEMGUARD_MARKER"
    pids=$(targets)
    [ -n "$pids" ] && kill -TERM $pids 2>/dev/null
    for _ in $(seq 1 $((PD_MEMGUARD_GRACE_S * 5))); do
      [ -z "$(targets)" ] && break
      sleep 0.2
    done
    pids=$(targets)
    if [ -n "$pids" ]; then
      log "still alive after ${PD_MEMGUARD_GRACE_S}s — SIGKILL $pids"
      kill -KILL $pids 2>/dev/null
    fi
    read -r total_kb avail_kb < <(meminfo)
    log "engine stopped; available now $(gib "$avail_kb") GiB"
    exit 3
  fi

  if [ -n "$(targets)" ]; then
    seen=1
  elif [ "$seen" = 1 ]; then
    log "engine exited on its own; peak used $(gib "$peak_kb") GiB"
    exit 0
  fi

  tick=$((tick + 1))
  [ $((tick % 300)) -eq 0 ] && log "used $(gib "$used_kb") GiB, peak $(gib "$peak_kb") GiB, limit $(gib "$limit_kb") GiB"   # once a minute
  sleep "$PD_MEMGUARD_INTERVAL"
done
