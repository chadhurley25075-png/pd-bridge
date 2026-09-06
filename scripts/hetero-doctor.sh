#!/usr/bin/env bash
# doctor — walk the whole chain and say exactly which link is broken. Read config from hetero.env / environment.
set -uo pipefail
ok=0; bad=0
chk(){ if eval "$2" >/dev/null 2>&1; then echo "  ok   $1"; ok=$((ok+1)); else echo "  FAIL $1  ($2)"; bad=$((bad+1)); fi }
echo "hetero-doctor:"
chk "PD_SPARK engine reachable"        'curl -s -m 5 "${PD_SPARK:?set PD_SPARK}/v1/models" | grep -q deepseek'
chk "capture share reachable"          'curl -s -m 5 "${PD_SPARK%:*}:8010/_ls/"'
chk "oMLX reachable"                   'curl -s -m 5 "${PD_OMLX:-http://127.0.0.1:8011}/v1/models"'
chk "front door healthy"               'curl -s -m 5 "${FRONT:-http://127.0.0.1:8012}/health" | grep -q "\"ok\""'
chk "decoder cache dir exists"         'test -d "${PD_CACHE_DIR:-$HOME/.omlx/cache}"'
if [ -n "${PD_SPARK_SSH:-}" ]; then chk "projection weights on prefiller" 'ssh -o BatchMode=yes -o ConnectTimeout=5 "$PD_SPARK_SSH" "ls /home/*/pd_v3/dv4_proj_weights.safetensors"'; fi
echo "doctor: $ok ok, $bad failed"
[ $bad -eq 0 ]
