#!/usr/bin/env bash
# pd-launch-kv.sh — vLLM (native venv, TP1) prefill engine with the pd_kv_connector capture, for a
# plain-attention model (Qwen3 dense). R0 of docs/RDMA.md.
#   PD_HOOK=off ./spark/pd-launch-kv.sh    same engine, no connector: the control run
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

: "${PD_VLLM_BIN:=$HOME/vllm-env/bin/vllm}"
: "${PD_KV_MODEL:=$HOME/models/Qwen3-32B}"
: "${PD_KV_SERVED:=qwen3-32b}"
: "${PD_CAPTURE_DIR:=$HOME/pd_capture}"
: "${PD_KV_HOST:=0.0.0.0}"
: "${PD_KV_PORT:=8000}"
: "${PD_GPU_UTIL:=0.75}"          # GB10 is one unified pool: weights (61 GiB) + KV arena + everything else
: "${PD_MAX_MODEL_LEN:=40960}"    # Qwen3-32B native window; raising it means YaRN — read FINDING-stale-limits first
: "${PD_MAX_BATCHED:=8192}"
: "${PD_OMLX_BLOCK:=256}"         # must equal the decoder's paged_cache_block_size

mkdir -p "$PD_CAPTURE_DIR"
# The venv is not activated: flashinfer JIT-compiles its sampler at warmup and looks for `ninja` on PATH,
# which lives next to the vllm binary. Without this the engine loads 61 GiB of weights and then dies.
export CUDA_HOME="${CUDA_HOME:-${PD_CUDA_HOME:-/usr/local/cuda}}"   # the same JIT needs nvcc; it is not on a non-login PATH
export PATH="$(dirname "$PD_VLLM_BIN"):$CUDA_HOME/bin:$PATH"
args=(serve "$PD_KV_MODEL"
  --served-model-name "$PD_KV_SERVED"
  --host "$PD_KV_HOST" --port "$PD_KV_PORT"
  --dtype bfloat16
  --max-model-len "$PD_MAX_MODEL_LEN"
  --gpu-memory-utilization "$PD_GPU_UTIL"
  --max-num-batched-tokens "$PD_MAX_BATCHED"
  --no-enable-prefix-caching)     # a skipped prefix is rows this engine never computes

# Prefill-kernel variants (docs/RDMA.md, O1). Defaults are the measured baseline.
#   PD_EAGER=off         torch.compile fusions (+ CUDA graphs, which prefill does not use); longer boot
#   PD_ATTN_BACKEND=X    FLASH_ATTN (default pick on GB10) | FLASHINFER | TRITON_ATTN
#   PD_QUANT=fp8         online FP8 weights: changes the K/V the decoder receives, so retrieval must be re-checked
[ "${PD_EAGER:-on}" != "off" ] && args+=(--enforce-eager)
[ -n "${PD_ATTN_BACKEND:-}" ] && args+=(--attention-backend "$PD_ATTN_BACKEND")
[ -n "${PD_QUANT:-}" ] && args+=(--quantization "$PD_QUANT")

if [ "${PD_HOOK:-on}" != "off" ]; then
  export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
  extra="\"capture_dir\":\"$PD_CAPTURE_DIR\",\"omlx_block\":$PD_OMLX_BLOCK"
  if [ -n "${PD_RDMA_STREAM_HOST:-}" ]; then
    # R2: stream every block over RDMA to `pd_rdma recvd` on the Mac while prefill runs, instead of NVMe staging.
    # The registered send buffer needs RLIMIT_MEMLOCK >= PD_RDMA_BUF_MIB in this shell.
    echo "pd-launch-kv: rdma stream -> $PD_RDMA_STREAM_HOST:${PD_RDMA_STREAM_PORT:-18516}, memlock $(ulimit -l)" >&2
    extra="$extra,\"rdma_host\":\"$PD_RDMA_STREAM_HOST\",\"rdma_port\":${PD_RDMA_STREAM_PORT:-18516},\"rdma_lib\":\"$HERE/../rdma/libpd_rdma_tx.so\",\"rdma_dev\":\"${PD_RDMA_DEV:-rocep1s0f1}\",\"rdma_gid_index\":${PD_RDMA_GID_INDEX:-3},\"rdma_buf_mib\":${PD_RDMA_BUF_MIB:-128}"
  fi
  args+=(--kv-transfer-config "{\"kv_connector\":\"PdKvConnector\",\"kv_connector_module_path\":\"pd_kv_connector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{$extra}}")
fi

if [ "${PD_MEMGUARD:-on}" != "off" ]; then
  # memory watchdog: terminates this engine when used memory reaches PD_MEM_LIMIT_GB (default 125 GB); see pd_memguard.sh
  mkdir -p "$HOME/pd_logs"
  nohup "$HERE/pd_memguard.sh" >> "$HOME/pd_logs/memguard.log" 2>&1 < /dev/null &
fi

echo "pd-launch-kv: hook=${PD_HOOK:-on} memguard=${PD_MEMGUARD:-on} limit=${PD_MEM_LIMIT_GB:-125}GB model=$PD_KV_MODEL port=$PD_KV_PORT capture=$PD_CAPTURE_DIR" >&2
exec "$PD_VLLM_BIN" "${args[@]}"
