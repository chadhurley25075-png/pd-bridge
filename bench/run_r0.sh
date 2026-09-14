#!/usr/bin/env bash
# run_r0.sh — R0 matrix for the plain-attention bridge (docs/RDMA.md): native oMLX vs bridged, same sitting,
# fresh seed per run (the decoder's block cache is keyed by tokens only, so a reused seed is a warm run).
#   NATIVE_URL=http://127.0.0.1:8011 BRIDGE_URL=http://127.0.0.1:8012 ./bench/run_r0.sh
# Qwen3 tokenizer on bench_cold's stdlib document: ~4.4 chars/token (measured 2026-09-14).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${NATIVE_URL:=http://127.0.0.1:8011}"
: "${BRIDGE_URL:=http://127.0.0.1:8012}"
: "${MODEL:=Qwen3-32B}"
: "${SIZES:=36000 72000 142000}"      # ~8K, ~16K, ~32K tokens
: "${SEED0:=5000}"
: "${LEGS:=native bridge}"
: "${PY:=python3}"
: "${PD_SPARK_SSH:=}"                    # user@prefill-box: when set, a tripped memory guard ends the run
: "${PD_MEMGUARD_MARKER:=~/pd_logs/MEMGUARD_TRIPPED}"
OUT="$HERE/results/r0_$(date +%Y-%m-%d).jsonl"
mkdir -p "$HERE/results"

# The Spark memory watchdog (spark/pd_memguard.sh) kills the engine at its limit and leaves a marker. A bridged row
# after that would be a native fallback, so the run ends here instead of recording it.
memguard_check() {
  local m
  m=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$PD_SPARK_SSH" "cat $PD_MEMGUARD_MARKER 2>/dev/null") || true
  if [ -n "$m" ]; then
    echo "run_r0: STOPPED — Spark memory guard tripped: $m" >&2
    echo "{\"leg\":\"abort\",\"ts\":$(date +%s),\"memguard\":\"$m\"}" >> "$OUT"
    exit 3
  fi
}

seed=$SEED0
for chars in $SIZES; do
  for leg in $LEGS; do
    seed=$((seed + 1))
    url=$NATIVE_URL; [ "$leg" = "bridge" ] && url=$BRIDGE_URL
    memguard_check
    # 64 tokens ends inside Qwen3's reasoning and scores a MISS on both legs (smoke run 2026-09-14); upstream uses 300
    line=$("$PY" "$HERE/bench_cold.py" --chars "$chars" --seed "$seed" --url "$url" --model "$MODEL" --max-tokens "${MAX_TOKENS:-300}")
    echo "{\"leg\":\"$leg\",\"ts\":$(date +%s),\"result\":$line}" | tee -a "$OUT"
    memguard_check
  done
done
echo "appended to $OUT" >&2
