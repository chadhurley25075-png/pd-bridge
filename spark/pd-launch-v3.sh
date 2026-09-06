#!/usr/bin/env bash
# pd-launch-v3.sh — launch the vLLM prefill engine with the v3 pooled-capture hook.
#
# Run this on EACH prefill node.  Rank 0 = TP head, rank 1 = worker (--headless).
# Only rank 0 captures, but the worker imports the same sitecustomize, so ship
# the pd_v3 payload to both nodes.
#
# Prereqs on each node, in $PD_V3:
#   pd_pool_torch.py
#   dv4_proj_weights.safetensors   exported from the DECODER's MLX model —
#   dv4_proj_weights.json          see studio/pd_export_proj_weights.py
#   capture_sitecustomize_v3.py
#
# Two settings are load-bearing and were learned the hard way:
#   * --no-enable-prefix-caching.  With prefix caching ON, vLLM skips a repeated
#     document prefix, the hook observes 0 tokens for those layers, and the
#     decoder waits forever for rows that will never arrive.  The decoder owns
#     the caches; the prefill engine must compute every token it is asked to pool.
#   * --enforce-eager.  This is a prefill-only engine, so CUDA graphs buy nothing
#     and cost ~13 minutes of startup.  The v3 hook also skips stream capture.
#
# PD_HOOK=off  ->  control run: no sitecustomize, no capture env.  Use this to
# measure pure engine prefill time for the benchmark protocol.
#
# usage: source ../config.env && ./pd-launch-v3.sh <0|1>

set -uo pipefail
NODE_RANK="${1:?usage: pd-launch-v3.sh <0|1>   (0 = TP head, 1 = worker)}"
HEADLESS_FLAG=""; [ "$NODE_RANK" = "1" ] && HEADLESS_FLAG="--headless"

: "${PD_HEAD_IP:?set PD_HEAD_IP (see config.example.env)}"
: "${HF_DIR:?set HF_DIR}"
: "${PD_V3:?set PD_V3}"
: "${PD_CAPTURE_DIR:?set PD_CAPTURE_DIR}"
PD_VLLM_IMAGE="${PD_VLLM_IMAGE:-aidendle94/sparkrun-vllm-ds4-gb10:production-ready}"
PD_NCCL_IB_HCA="${PD_NCCL_IB_HCA:-rocep1s0f0}"
PD_NCCL_IB_GID_INDEX="${PD_NCCL_IB_GID_INDEX:-3}"
PD_NCCL_IFNAME="${PD_NCCL_IFNAME:-eth0}"
PD_TP_SIZE="${PD_TP_SIZE:-2}"
PD_MAX_MODEL_LEN="${PD_MAX_MODEL_LEN:-262144}"

for f in pd_pool_torch.py dv4_proj_weights.safetensors dv4_proj_weights.json; do
  [ -f "$PD_V3/$f" ] || { echo "MISSING $PD_V3/$f - see README, 'Exporting projection weights'"; exit 1; }
done
HOOK="$PD_V3/capture_sitecustomize_v3.py"
[ -f "$HOOK" ] || HOOK="$(dirname "$0")/capture_sitecustomize_v3.py"
[ -f "$HOOK" ] || { echo "MISSING capture_sitecustomize_v3.py"; exit 1; }

HOOK_MOUNT=(-v "$HOOK:/opt/env/lib/python3.12/site-packages/sitecustomize.py:ro"
            -e PD_CAPTURE_DIR=/pd_capture -e PD_CAPTURE_IDLE_S=2.0
            -e PD_POOL_PATH=/pd_v3 -e PD_PROJ_WEIGHTS=/pd_v3/dv4_proj_weights.safetensors)
[ "${PD_HOOK:-on}" = "off" ] && HOOK_MOUNT=() && HOOK="(none - control run)"
echo "hook=$HOOK  pd_v3=$PD_V3  rank=$NODE_RANK"

mkdir -p "$PD_CAPTURE_DIR" "$HF_DIR/vllm-cache"
docker rm -f vllm_pd 2>/dev/null || true
docker run --gpus all -d --privileged --network host --ipc host --shm-size 10g \
  --ulimit memlock=-1 --restart no \
  --device /dev/infiniband:/dev/infiniband \
  -v "$HF_DIR:/cache/huggingface" \
  "${HOOK_MOUNT[@]}" \
  -v "$PD_V3:/pd_v3:ro" \
  -v "$PD_CAPTURE_DIR:/pd_capture" \
  --name vllm_pd \
  -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e VLLM_CACHE_ROOT=/cache/huggingface/vllm-cache \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_USE_B12X_MOE=1 -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA="$PD_NCCL_IB_HCA" -e NCCL_IB_GID_INDEX="$PD_NCCL_IB_GID_INDEX" \
  -e NCCL_SOCKET_IFNAME="$PD_NCCL_IFNAME" -e GLOO_SOCKET_IFNAME="$PD_NCCL_IFNAME" -e TP_SOCKET_IFNAME="$PD_NCCL_IFNAME" \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  --entrypoint bash \
  "$PD_VLLM_IMAGE" \
  -lc "exec /usr/local/bin/dsv4-vllm-entrypoint serve deepseek-ai/DeepSeek-V4-Flash \
       --served-model-name deepseek-v4-flash --host 0.0.0.0 --port 8000 --trust-remote-code \
       --tensor-parallel-size $PD_TP_SIZE --pipeline-parallel-size 1 --kv-cache-dtype fp8 \
       --block-size 256 --max-model-len $PD_MAX_MODEL_LEN --max-num-seqs 1 \
       --max-num-batched-tokens 8192 --gpu-memory-utilization 0.75 \
       --no-enable-prefix-caching --tokenizer-mode deepseek_v4 \
       --distributed-executor-backend mp --enforce-eager \
       --nnodes 2 --node-rank $NODE_RANK --master-addr $PD_HEAD_IP --master-port 29501 $HEADLESS_FLAG"
echo "launched vllm_pd node-rank=$NODE_RANK headless='$HEADLESS_FLAG' rc=$?"
sleep 2; docker ps --format '{{.Names}} | {{.Status}}' | grep vllm_pd || echo "WARN: container not in ps (docker logs vllm_pd)"
