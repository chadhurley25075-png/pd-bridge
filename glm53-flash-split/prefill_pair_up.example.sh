#!/usr/bin/env bash
# Example: bring the GLM-5.3-Flash prefill pair up WITH Ash Hart's Glm53HandoffConnector, on top of MiaAI-Lab's
# two-Spark EXL3 recipe (github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks, AGPL-3.0 — cloned, not vendored here).
# Stock (unmodified) weights on the Sparks, so the decoder's MLX build reads the same model.
# Run on the head node; the recipe's start.sh handles both ranks. Values are the ones we ran on 2026-10-03/04.
set -uo pipefail
cd "${MIA_RECIPE_DIR:?clone MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks and point MIA_RECIPE_DIR at it}" || exit 1
export SKIP_PULL=1 SKIP_DOWNLOAD=1 SKIP_SYNC=1 SKIP_BUILD=1
# 14,336-token chunks left too little KV for a 196K window (needs 14.5 GiB, had 8.2) -> the recipe's 7,168 stands
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-7168}"
# 262,144 failed by ~20 MB of KV once the connector's arena is allocated; 196,608 is what served
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-196608}"
export KV_CACHE_DTYPE=auto
export PYTORCH_CUDA_ALLOC_CONF=
# the connector module (glm53_handoff_connector.py) installed into this user base inside the container
export GLM53_EXTRA_ENV="PYTHONUSERBASE=/root/.cache/huggingface/glm53-handoff"
export EXTRA_ARGS='--cudagraph-capture-sizes 1 2 4 8 16 24 32 --kv-transfer-config {"kv_connector":"Glm53HandoffConnector","kv_role":"kv_producer","kv_connector_module_path":"glm53_handoff_connector","engine_id":"glm53-prefill","kv_connector_extra_config":{"handoff_path":"/dev/shm/glm53-handoff","boundary_tokens":4096,"capture_window":2048}}'
./start.sh restart; rc=$?
# the first two requests after a restart prefilled at ~450-700 tok/s (vs ~1,350-1,600 warm): pay that here
if [ "$rc" = 0 ]; then
  python3 - <<'PY'
import json, urllib.request
for n in (4000, 12000):
    body = {"model": "glm-5.3-flash", "prompt": [1000 + (i * 37) % 50000 for i in range(n)], "max_tokens": 1}
    try:
        urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8888/v1/completions", data=json.dumps(body).encode(),
                               headers={"content-type": "application/json"}), timeout=300).read()
        print("warm-up", n, "ok", flush=True)
    except Exception as e:
        print("warm-up", n, "failed", e, flush=True)
PY
fi
exit $rc
