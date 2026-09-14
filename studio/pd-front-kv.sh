#!/usr/bin/env bash
# pd-front-kv.sh — front door for the plain-attention bridge (Qwen3-32B) on the Mac. R0 of docs/RDMA.md.
# Every default below is overridable from the environment; nothing here guesses at a peer.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

: "${OMLX_PYTHON:=$HOME/omlx/bin/python}"
export PD_MODE=kv
export PD_MODEL="${PD_MODEL:-$HOME/models/Qwen3-32B}"
export PD_MODEL_NAME="${PD_MODEL_NAME:-Qwen3-32B}"         # model id as oMLX serves it (its directory name)
export PD_OMLX="${PD_OMLX:-http://127.0.0.1:8011}"
export PD_CACHE_DIR="${PD_CACHE_DIR:-$HOME/.omlx/cache}"
export PD_PORT="${PD_PORT:-8012}"
export PD_SPARK="${PD_SPARK:?set PD_SPARK to the prefill engine, e.g. http://10.0.0.2:8000}"
export PD_SPARK_MODEL="${PD_SPARK_MODEL:-qwen3-32b}"
export PD_SHARE_PORT="${PD_SHARE_PORT:-8110}"
export PD_PULL_DIR="${PD_PULL_DIR:-$HOME/pd_pull}"
export PD_TRANSPORT="${PD_TRANSPORT:-tcp10}"
# R1: PD_TRANSPORT=rdma pulls captures with rdma/pd_rdma. The MelonDMA RoCE profile (MELONDMA_LOCAL_IP/LOCAL_MAC/
# REMOTE_MAC) is machine-specific and stays out of the repo: put the exports in ~/.config/pd-bridge/rdma.env.
[ -f "$HOME/.config/pd-bridge/rdma.env" ] && . "$HOME/.config/pd-bridge/rdma.env"
export PD_RDMA_DEV="${PD_RDMA_DEV:-mlx5_0}"
export PD_MIN_TOKENS="${PD_MIN_TOKENS:-2048}"
export PD_MIN_TAIL="${PD_MIN_TAIL:-2048}"

exec "$OMLX_PYTHON" "$HERE/pd_front.py"
