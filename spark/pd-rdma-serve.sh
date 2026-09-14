#!/usr/bin/env bash
# pd-rdma-serve.sh — RDMA file server for pd-bridge captures on the Spark. R1 of docs/RDMA.md.
# Control/handshake TCP binds to the 10GbE address (the Mac has no netif on the RDMA link); data goes over RoCEv2.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

: "${PD_RDMA_BIN:=$HERE/../rdma/pd_rdma}"
: "${PD_CAPTURE_DIR:=$HOME/pd_capture}"
: "${PD_RDMA_BIND:?set PD_RDMA_BIND to the address of this box on the control link (e.g. 10.0.0.2)}"
: "${PD_RDMA_PORT:=18515}"
: "${PD_RDMA_DEV:=rocep1s0f1}"      # the RoCE port cabled to the Mac (ibv_devices)
: "${PD_RDMA_GID_INDEX:=3}"         # show_gids: the RoCEv2 GID of the RDMA link address
: "${PD_RDMA_BUF_MIB:=128}"

exec "$PD_RDMA_BIN" serve --root "$PD_CAPTURE_DIR" --bind "$PD_RDMA_BIND" --port "$PD_RDMA_PORT" \
  --dev "$PD_RDMA_DEV" --gid-index "$PD_RDMA_GID_INDEX" --buf-mib "$PD_RDMA_BUF_MIB"
