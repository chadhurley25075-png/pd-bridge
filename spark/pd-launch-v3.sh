#!/usr/bin/env bash
# pd-launch-v3.sh — launch the vLLM prefill pair with hook v3 (pooled capture with the decoder's projection weights).
#   * sitecustomize = capture_sitecustomize_v3.py (or the KV-return side-car, see PD_KV_RETURN below)
#   * $PD_V3 (pd_pool_torch.py, dv4_proj_weights.safetensors + .json, and the hook) is mounted read-only at /pd_v3;
#     the hook imports pd_pool_torch and loads the weights from there (PD_POOL_PATH / PD_PROJ_WEIGHTS)
#   * PD_CAPTURE_IDLE_S default 15.0: 2.0 mid-flushed BETWEEN prefill chunks (inter-chunk GPU idle exceeds 2 s;
#     docs/FINDING-bench4-cold-fallback.md). The front door signals /_flush when its engine call returns; idle is
#     the backstop, and 15 s lands inside the front's 45 s grace even if the signal fails.
#   * --enforce-eager (prefill-only engine; CUDA graphs buy nothing)
#   * PREFIX CACHING OFF by default: with it on, a repeated document prefix is skipped by vLLM, the hook sees 0 tokens
#     and the decoder waits forever. The pooled capture needs every token computed. (PD_KV_RETURN=1 is the one
#     exception — it turns prefix caching on deliberately and teaches the hook to resume; see studio/kv_return/.)
# Config (config.example.env): PD_HEAD_IP (interconnect address of rank 0, vLLM --master-addr), HF_DIR, PD_V3,
#   PD_CAPTURE_DIR (host staging dir for captures), PD_NCCL_IB_HCA, PD_NCCL_IFNAME, PD_VLLM_IMAGE, optional
#   PD_NCCL_IB_GID_INDEX (otherwise resolved at launch, see below).
# Only rank 0 captures, but the worker imports the same sitecustomize, so ship $PD_V3 to both nodes.
# usage: pd-launch-v3.sh <0|1>   0 = head (rank 0), 1 = worker (--headless)
set -uo pipefail
NODE_RANK="${1:?usage: pd-launch-v3.sh <0|1>}"
HEADLESS_FLAG=""; [ "$NODE_RANK" = "1" ] && HEADLESS_FLAG="--headless"
HEAD_IP="${PD_HEAD_IP:?set PD_HEAD_IP (rank 0 interconnect address) — see config.example.env}"
HF_DIR="${HF_DIR:-$HOME/.cache/huggingface}"
PD_V3="${PD_V3:-$HOME/pd_v3}"
CAP_HOST_DIR="${PD_CAPTURE_DIR:-$HOME/pd_capture}"
IB_HCA="${PD_NCCL_IB_HCA:-rocep1s0f0}"
NET_IF="${PD_NCCL_IFNAME:-enp1s0f0np0}"
IMAGE="${PD_VLLM_IMAGE:-aidendle94/sparkrun-vllm-ds4-gb10:production-ready}"
for f in pd_pool_torch.py dv4_proj_weights.safetensors dv4_proj_weights.json; do
  [ -f "$PD_V3/$f" ] || { echo "MISSING $PD_V3/$f — run 'make weights' on the decoder and copy it over first"; exit 1; }
done
HOOK_NAME=capture_sitecustomize_v3.py
[ "${PD_KV_RETURN:-0}" = "1" ] && HOOK_NAME=capture_sitecustomize_v3.kvreturn.py   # the resume path lives in the side-car
HOOK="$PD_V3/$HOOK_NAME"; [ -f "$HOOK" ] || HOOK="$(cd "$(dirname "$0")" && pwd)/$HOOK_NAME"
[ -f "$HOOK" ] || { echo "MISSING $HOOK_NAME (looked in $PD_V3 and next to this script)"; exit 1; }
# PD_HOOK=off → CONTROL RUN: no sitecustomize, no capture env (pure engine prefill timing for the benchmark protocol)
HOOK_MOUNT=(-v "$HOOK:/opt/env/lib/python3.12/site-packages/sitecustomize.py:ro" -e PD_CAPTURE_DIR=/pd_capture -e PD_CAPTURE_IDLE_S=${PD_CAPTURE_IDLE_S:-15.0} -e PD_POOL_PATH=/pd_v3 -e PD_PROJ_WEIGHTS=/pd_v3/dv4_proj_weights.safetensors)
# 2026-09-18 KV RETURN (default OFF): PD_KV_RETURN=1 turns vLLM prefix caching ON (the engine keeps the previous
# turn's KV blocks resident and computes only the new tail -> the hook's first chunk starts at S>0) and arms the hook's
# resume path (it seeds itself from <stamp>/kvstate.safetensors and captures [S,T) + the end state; the front merges).
# Everything else — connector-free pooled capture, TP2, chunk size — is unchanged. Requires a container relaunch.
PREFIX_FLAG="--no-enable-prefix-caching"
if [ "${PD_KV_RETURN:-0}" = "1" ]; then
  PREFIX_FLAG="--enable-prefix-caching"
  HOOK_MOUNT+=(-e PD_KV_RETURN=1 -e PD_KV_RETURN_KEEP=${PD_KV_RETURN_KEEP:-256})
  echo "KV return armed: prefix caching ON, hook resume ON ($HOOK_NAME)"
fi
[ "${PD_HOOK:-on}" = "off" ] && HOOK_MOUNT=() && HOOK="(none — control run)"
echo "hook=$HOOK  pd_v3=$PD_V3"
mkdir -p "$CAP_HOST_DIR" "$HF_DIR/vllm-cache"
# 2026-09-17: resolve the RoCE v2 GID index for this node's interconnect IPv4 at launch time. Hardcoding it broke when
# the GID table shifted (MCDMA static neighbours add link-local entries); our two nodes ended up at 6 and 5.
GID_IDX="${PD_NCCL_IB_GID_INDEX_FORCE:-}"
if [ -z "$GID_IDX" ]; then
  for i in $(seq 0 15); do
    g=$(cat /sys/class/infiniband/$IB_HCA/ports/1/gids/$i 2>/dev/null); t=$(cat /sys/class/infiniband/$IB_HCA/ports/1/gid_attrs/types/$i 2>/dev/null)
    case "$g" in 0000:0000:0000:0000:0000:ffff:*) [ "$t" = "RoCE v2" ] && { GID_IDX=$i; break; } ;; esac
  done
fi
[ -n "$GID_IDX" ] || { echo "no RoCE v2 IPv4 GID on $IB_HCA — is the interconnect up? (or set PD_NCCL_IB_GID_INDEX_FORCE)"; exit 1; }
echo "NCCL_IB_GID_INDEX=$GID_IDX ($(cat /sys/class/infiniband/$IB_HCA/ports/1/gids/$GID_IDX 2>/dev/null))"
docker rm -f vllm_pd 2>/dev/null || true
docker run --gpus all -d --privileged --network host --ipc host --shm-size 10g \
  --ulimit memlock=-1 --restart no \
  --device /dev/infiniband:/dev/infiniband \
  -v "$HF_DIR:/cache/huggingface" \
  "${HOOK_MOUNT[@]}" \
  -v "$PD_V3:/pd_v3:ro" \
  -v "$CAP_HOST_DIR:/pd_capture" \
  --name vllm_pd \
  -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e VLLM_CACHE_ROOT=/cache/huggingface/vllm-cache \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_USE_B12X_MOE=1 -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=$IB_HCA -e NCCL_IB_GID_INDEX=$GID_IDX \
  -e NCCL_SOCKET_IFNAME=$NET_IF -e GLOO_SOCKET_IFNAME=$NET_IF -e TP_SOCKET_IFNAME=$NET_IF \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  --entrypoint bash \
  "$IMAGE" \
  -lc "exec /usr/local/bin/dsv4-vllm-entrypoint serve deepseek-ai/DeepSeek-V4-Flash --served-model-name deepseek-v4-flash --host 0.0.0.0 --port 8000 --trust-remote-code --tensor-parallel-size 2 --pipeline-parallel-size 1 --kv-cache-dtype fp8 --block-size 256 --max-model-len ${PD_MAX_MODEL_LEN:-262144} --max-num-seqs 1 --max-num-batched-tokens ${PD_CHUNK:-8192} --gpu-memory-utilization ${PD_GPU_UTIL:-0.75} $PREFIX_FLAG --tokenizer-mode deepseek_v4 --distributed-executor-backend mp --enforce-eager --nnodes 2 --node-rank $NODE_RANK --master-addr $HEAD_IP --master-port 29501 $HEADLESS_FLAG"
echo "launched vllm_pd node-rank=$NODE_RANK headless='$HEADLESS_FLAG' rc=$?"
sleep 2; docker ps --format '{{.Names}} | {{.Status}}' | grep vllm_pd || echo "WARN: container not in ps (docker logs vllm_pd)"
