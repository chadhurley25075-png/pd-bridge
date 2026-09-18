#!/usr/bin/env bash
# pd-launch-v3.sh — PROPOSED launcher for hook v3 (pooled capture with the Mac's projection weights).
# (SISTER B, 2026-09-06 — NOT run by the sister; Compass integrates + restarts.)
# Diff vs pd-launch.sh:
#   * sitecustomize = capture_sitecustomize_v3.py
#   * /home/chad-hurley/pd_v3 (Sister A's drop: pd_pool_torch.py, dv4_proj_weights.safetensors + .json,
#     and capture_sitecustomize_v3.py) is mounted read-only at /pd_v3; the hook imports pd_pool_torch and
#     loads the weights from there (PD_POOL_PATH / PD_PROJ_WEIGHTS)
#   * PD_CAPTURE_IDLE_S default 15.0 (9/6): 2.0 mid-flushed BETWEEN prefill chunks (inter-chunk GPU idle
#     exceeds 2 s; bench4/bench5 autopsies). The front door now signals /_flush when its engine call
#     returns (the only party that knows prefill is over); idle is the backstop, and 15 s lands inside
#     the front's 45 s grace even if the signal fails. Override: PD_CAPTURE_IDLE_S=<s> before launching.
#   * --enforce-eager kept (v3 skips stream capture anyway; prefill-only engine)
#   * PREFIX CACHING OFF (9/6): with it on, a repeated document prefix is skipped by vLLM, the hook sees 0 tokens and the
#     Mac side waits forever. The pooled capture needs every token computed; the Mac holds the caches, not the Spark.
# Prereqs on EACH Spark: /home/chad-hurley/pd_v3/{pd_pool_torch.py,dv4_proj_weights.safetensors,
#   dv4_proj_weights.json} + capture_sitecustomize_v3.py (in pd_v3, else ~/pd_lab/spark). Only rank 0
#   (spark-06) captures, but the worker imports the same sitecustomize so ship the files to both.
# usage: pd-launch-v3.sh <0|1>   0 = head (spark-06 .220.16), 1 = worker (spark-07 .220.15, --headless)
set -uo pipefail
NODE_RANK="${1:?usage: pd-launch-v3.sh <0|1>}"
HEADLESS_FLAG=""; [ "$NODE_RANK" = "1" ] && HEADLESS_FLAG="--headless"
HEAD_IP=192.168.220.16
case "$(hostname)" in
  spark-06) HF_DIR=/home/chad-hurley/.cache/huggingface ;;
  spark-07) HF_DIR=/mnt/compass_keep/hf_stage ;;
  *) echo "unknown node $(hostname)"; exit 1 ;;
esac
PD_V3=/home/chad-hurley/pd_v3
for f in pd_pool_torch.py dv4_proj_weights.safetensors dv4_proj_weights.json; do
  [ -f "$PD_V3/$f" ] || { echo "MISSING $PD_V3/$f — copy it from Cerebro first"; exit 1; }
done
HOOK="$PD_V3/capture_sitecustomize_v3.py"; [ -f "$HOOK" ] || HOOK=/home/chad-hurley/pd_lab/spark/capture_sitecustomize_v3.py
[ -f "$HOOK" ] || { echo "MISSING capture_sitecustomize_v3.py (looked in $PD_V3 and ~/pd_lab/spark)"; exit 1; }
# PD_HOOK=off → CONTROL RUN: no sitecustomize, no capture env (pure engine prefill timing for the benchmark protocol)
HOOK_MOUNT=(-v "$HOOK:/opt/env/lib/python3.12/site-packages/sitecustomize.py:ro" -e PD_CAPTURE_DIR=/pd_capture -e PD_CAPTURE_IDLE_S=${PD_CAPTURE_IDLE_S:-15.0} -e PD_POOL_PATH=/pd_v3 -e PD_PROJ_WEIGHTS=/pd_v3/dv4_proj_weights.safetensors)
[ "${PD_HOOK:-on}" = "off" ] && HOOK_MOUNT=() && HOOK="(none — control run)"
echo "hook=$HOOK  pd_v3=$PD_V3"
mkdir -p /home/chad-hurley/pd_capture "$HF_DIR/vllm-cache"
# 9/17: resolve the RoCE v2 GID index for this node's .220.x IPv4 at launch time. Hardcoding 5 broke when the
# GID table shifted (MCDMA static neighbours add link-local entries); spark-06 had v2/IPv4 at 6, spark-07 at 5.
GID_IDX=""
for i in $(seq 0 15); do
  g=$(cat /sys/class/infiniband/rocep1s0f0/ports/1/gids/$i 2>/dev/null); t=$(cat /sys/class/infiniband/rocep1s0f0/ports/1/gid_attrs/types/$i 2>/dev/null)
  case "$g" in 0000:0000:0000:0000:0000:ffff:*) [ "$t" = "RoCE v2" ] && { GID_IDX=$i; break; } ;; esac
done
[ -n "$GID_IDX" ] || { echo "no RoCE v2 IPv4 GID on rocep1s0f0 — is .220.x up?"; exit 1; }
echo "NCCL_IB_GID_INDEX=$GID_IDX ($(cat /sys/class/infiniband/rocep1s0f0/ports/1/gids/$GID_IDX))"
docker rm -f vllm_pd 2>/dev/null || true
docker run --gpus all -d --privileged --network host --ipc host --shm-size 10g \
  --ulimit memlock=-1 --restart no \
  --device /dev/infiniband:/dev/infiniband \
  -v "$HF_DIR:/cache/huggingface" \
  "${HOOK_MOUNT[@]}" \
  -v "$PD_V3:/pd_v3:ro" \
  -v /home/chad-hurley/pd_capture:/pd_capture \
  --name vllm_pd \
  -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e VLLM_CACHE_ROOT=/cache/huggingface/vllm-cache \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_USE_B12X_MOE=1 -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0 -e NCCL_IB_GID_INDEX=$GID_IDX \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 -e TP_SOCKET_IFNAME=enp1s0f0np0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  --entrypoint bash \
  aidendle94/sparkrun-vllm-ds4-gb10:production-ready \
  -lc "exec /usr/local/bin/dsv4-vllm-entrypoint serve deepseek-ai/DeepSeek-V4-Flash --served-model-name deepseek-v4-flash --host 0.0.0.0 --port 8000 --trust-remote-code --tensor-parallel-size 2 --pipeline-parallel-size 1 --kv-cache-dtype fp8 --block-size 256 --max-model-len ${PD_MAX_MODEL_LEN:-262144} --max-num-seqs 1 --max-num-batched-tokens ${PD_CHUNK:-8192} --gpu-memory-utilization ${PD_GPU_UTIL:-0.75} --no-enable-prefix-caching --tokenizer-mode deepseek_v4 --distributed-executor-backend mp --enforce-eager --nnodes 2 --node-rank $NODE_RANK --master-addr $HEAD_IP --master-port 29501 $HEADLESS_FLAG"
echo "launched vllm_pd node-rank=$NODE_RANK headless='$HEADLESS_FLAG' rc=$?"
sleep 2; docker ps --format '{{.Names}} | {{.Status}}' | grep vllm_pd || echo "WARN: container not in ps (docker logs vllm_pd)"
