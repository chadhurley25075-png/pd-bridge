# SPDX-License-Identifier: Apache-2.0
"""pd_omlx_block.py — oMLX 0.6.4 paged-SSD prefix-cache block files for a plain-KVCache model, built without MLX or
oMLX. R4 of docs/RDMA.md.

The prefill side knows everything such a block contains: the token ids (so the chain hash), the model geometry and
the K/V rows. Building the file there and landing it directly in the decoder's cache directory removes the decoder's
mx.load, KVCache assembly and store_cache pass.

Format — read off server-written Qwen3-32B blocks on 2026-09-14 and from omlx/cache/paged_ssd_cache.py:
  file        [u64 little-endian header length][JSON header, space-padded to a multiple of 8][tensor bytes]
  tensors     layer_{i}_state_0 (keys) then layer_{i}_state_1 (values), per layer in order, BF16 [1, n_kv, block, head_dim]
  JSON        tensors in insertion order, then "__metadata__"; separators (",", ":")
  signature   json.dumps(payload, sort_keys=True, separators=(",", ":")), payload_layout split_recurrent_v1
  chain hash  sha256(model_name || parent_hash or b"omlx-root" || str(tuple(block_token_ids)))

Pure stdlib, so it imports in the vLLM venv and is checked byte for byte against real oMLX files on the Mac
(studio/test_omlx_block.py).
"""
import hashlib
import json
import struct
import time

FORMAT_VERSION = "5"            # oMLX 0.6.4 stamps 5 when the split-GDN layout is enabled (its default)
PAYLOAD_LAYOUT = "split_recurrent_v1"


def chain_hashes(token_ids, model_name: str, block: int) -> list[bytes]:
    """oMLX compute_block_hash, chained over full blocks. Token ids must be Python ints: str(tuple(...)) is hashed."""
    out, parent = [], None
    for i in range(len(token_ids) // block):
        h = hashlib.sha256()
        if model_name:
            h.update(model_name.encode("utf-8"))
        h.update(parent if parent else b"omlx-root")
        h.update(bytes(str(tuple(int(t) for t in token_ids[i * block:(i + 1) * block])), "utf-8"))
        parent = h.digest()
        out.append(parent)
    return out


def header(block_hash: bytes, model_name: str, n_layers: int, n_kv: int, block: int, head_dim: int,
           created_at: float | str | None = None) -> tuple[bytes, int]:
    """(length prefix + padded JSON header, tensor byte count). Tensor data follows immediately: per layer, keys
    then values, each n_kv * block * head_dim bf16 values."""
    per = n_kv * block * head_dim * 2
    tensors, off = {}, 0
    for i in range(n_layers):
        for s in (0, 1):
            tensors[f"layer_{i}_state_{s}"] = {"dtype": "BF16", "shape": [1, n_kv, block, head_dim],
                                               "data_offsets": [off, off + per]}
            off += per
    types = ["KVCache"] * n_layers
    signature = json.dumps({"model_name": model_name, "num_layers": n_layers, "block_size": block,
                            "layer_cache_types": types, "payload_layout": PAYLOAD_LAYOUT},
                           sort_keys=True, separators=(",", ":"))
    meta = {
        "omlx_cache_format_version": FORMAT_VERSION,
        "block_hash": block_hash.hex(),
        "token_count": str(block),
        "num_layers": str(n_layers),
        "model_name": model_name,
        "block_size": str(block),
        "cache_signature": signature,
        "payload_layout": PAYLOAD_LAYOUT,
        "created_at": created_at if isinstance(created_at, str) else str(time.time() if created_at is None else created_at),
        "layer_cache_types": json.dumps(types),
        "layer_meta_states": json.dumps([[] for _ in range(n_layers)]),
    }
    for i in range(n_layers):
        meta[f"layer_{i}_state_count"] = "2"
    doc = dict(tensors)
    doc["__metadata__"] = meta
    js = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    js += b" " * ((8 - len(js) % 8) % 8)
    return struct.pack("<Q", len(js)) + js, off
