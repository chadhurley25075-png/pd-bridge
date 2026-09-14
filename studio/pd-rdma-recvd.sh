#!/usr/bin/env bash
# pd-rdma-recvd.sh — R2 receiver on the Mac (docs/RDMA.md): the Spark connector RDMA-writes every capture block
# here while the prefill runs, and the front door (PD_TRANSPORT=rdma2) waits for <PD_PULL_DIR>/<tag>/DONE locally.
# Control TCP binds to the Mac's 10GbE address; the connector dials it. Start this before the vLLM engine.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# MelonDMA static RoCE profile (MELONDMA_LOCAL_IP/LOCAL_MAC/REMOTE_MAC), kept out of the repo
[ -f "$HOME/.config/pd-bridge/rdma.env" ] && . "$HOME/.config/pd-bridge/rdma.env"

: "${PD_RDMA_BIN:=$HERE/../rdma/pd_rdma}"
: "${PD_PULL_DIR:=$HOME/pd_pull}"               # must equal the front door's PD_PULL_DIR
: "${PD_RDMA_RECV_BIND:?set PD_RDMA_RECV_BIND to the address of this Mac on the control link (e.g. 10.0.0.1)}"
: "${PD_RDMA_STREAM_PORT:=18516}"
: "${PD_RDMA_DEV:=mlx5_0}"
: "${PD_RDMA_ARENA_MIB:=128}"                   # one 64 MiB Qwen3-32B block + headroom; the DEXT allows 512 MiB pinned
: "${PD_CACHE_DIR:=$HOME/.omlx/cache}"             # R4: oMLX-native blocks land here directly (must equal oMLX's --paged-ssd-cache-dir)

mkdir -p "$PD_PULL_DIR" "$PD_CACHE_DIR"
exec "$PD_RDMA_BIN" recvd --root "$PD_PULL_DIR" --omlx-cache "$PD_CACHE_DIR" --bind "$PD_RDMA_RECV_BIND" \
  --port "$PD_RDMA_STREAM_PORT" --dev "$PD_RDMA_DEV" --arena-mib "$PD_RDMA_ARENA_MIB"
