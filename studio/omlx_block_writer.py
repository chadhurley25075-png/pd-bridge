#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
omlx_block_writer.py — write oMLX 0.6.4 paged-SSD prefix-cache blocks for
DeepSeek-V4-Flash that are byte/numerically identical to what the live
server writes, by driving oMLX's OWN store pipeline
(BlockAwarePrefixCache.store_cache -> PagedSSDCacheManager.save_block ->
_write_safetensors_no_mx) with the same extracted/compacted snapshot
states the scheduler hands it.

Run inside the oMLX venv ($OMLX_PYTHON on the Mac decode nodes).

HOW oMLX STORES A DV4 REQUEST (verified against omlx 0.6.4 source 2026-09-05)
-----------------------------------------------------------------------------
* block_size for DeepSeek-V4 = 2048 (RotatingKVCache window 128 + PoolingCache
  -> Scheduler._POOLING_ROTATING_BLOCK_SIZE). Prefill chunks are clamped to
  2048-token boundaries; at every boundary tc = k*2048 the scheduler snapshots
  the LIVE per-layer caches (Scheduler._emit_prefill_boundary_snapshot ->
  _extract_prefill_snapshot_states) and compacts PoolingCache `pooled` to the
  block delta (omlx.cache.pooling_delta.compact_pooling_cache_snapshot).
* At request end, tokens = prompt_token_ids + output_token_ids truncated to
  the latest boundary snapshot; store_cache() walks full blocks, chain-hashes
  them and, for these non-sliceable layers, writes the BOUNDARY SNAPSHOT of
  every layer into that block's file (no per-token KV slices).
* Per block file (43 layers):
    layer 0,1  RotatingKVCache : keys [1,1,128,512] bf16 (last 128 tokens),
               values [1,1,128,0] (zero-dim marker), meta (0,128,tc,128)
    layer 2..42 CacheList(RotatingKVCache, PoolingCache[, PoolingCache]):
               sub_0 rotating as above; sub_1/sub_2 stored as
               'PoolingCacheDelta' with 6 elements: buf_kv(None at a boundary),
               buf_gate(None), pooled DELTA rows [1, 2048/ratio, D],
               prev_win_kv/gate ([1,1,4,2D] for ratio-4 layers, None for
               ratio-128), int64 [start,end] absolute pooled-row range.
* Hash chain: sha256(model_name || parent_hash|b"omlx-root" ||
  str(tuple(block_token_ids)) [|| str(extra_keys)]).
* File: <ssd_cache_dir>/<hash_hex[0]>/<hash_hex>.safetensors.
* The server only sees blocks its in-memory PagedSSDCacheIndex knows; the
  index is built by _scan_existing_files() at model load. There is no rescan
  API: after dropping files into ~/.omlx/cache the oMLX server must be
  restarted (or the model unloaded/reloaded) for lookups to hit them.

USAGE
-----
    from omlx_block_writer import BlockWriter
    w = BlockWriter(model_name="DV4-Flash-MXFP4-MLX", out_dir="/tmp/pd_blocks",
                    cache_list_factory=model.make_cache)
    # during a chunked prefill clamped to 2048-token boundaries:
    for chunk in chunks:
        model(chunk, cache=cache)
        mx.eval([c.state for c in cache])
        if tokens_processed % 2048 == 0:
            w.snapshot(cache, tokens_processed)        # extract+compact NOW
    paths = w.finalize(token_ids)                      # writes the block files

    # convenience one-shot when only boundary snapshots are at hand:
    write_blocks(snapshots={2048: cache_at_2048, ...}, token_ids=ids,
                 model_name=..., out_dir=..., cache_list_factory=...)

The caller MUST pass the exact token ids oMLX would hash: the request's
prompt_token_ids (chat-template applied), in order. Token ids are NOT stored
in block files, so they cannot be recovered from an existing block.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

# --- make the oMLX DV4 cache classes importable exactly as the server has them
try:
    import omlx.patches.deepseek_v4 as _dv4_patch

    _dv4_patch.apply_pooling_cache_support()
except Exception as _e:  # pragma: no cover - the server venv always has it
    logging.getLogger(__name__).warning("deepseek_v4 patch unavailable: %s", _e)

from omlx.cache.hybrid_cache import ModelCacheConfig
from omlx.cache.paged_cache import PagedCacheManager, compute_block_hash
from omlx.cache.paged_ssd_cache import (
    PagedSSDCacheManager,
    cachelist_subtypes_from_cache_list,
)
from omlx.cache.pooling_delta import compact_pooling_cache_snapshot
from omlx.cache.prefix_cache import BlockAwarePrefixCache
from omlx.cache.type_registry import CacheTypeRegistry

logger = logging.getLogger("omlx_block_writer")

BLOCK_SIZE = 2048  # DeepSeek-V4 family (Scheduler._POOLING_ROTATING_BLOCK_SIZE)

_KB_PER_TOKEN_EST = 10.0   # ~KB of snapshot state per token per boundary (measured ~20 MB per 2048-token boundary)
_ROTATING_NAMES = (
    "RotatingKVCache",
    "BatchRotatingKVCache",
    "PrefillReadyRotatingKVCache",
    "BufferedRotatingKVCache",
)


# --------------------------------------------------------------------------
# State extraction — a faithful copy of Scheduler._extract_cache_states /
# _normalize_rotating_snapshot_state / _extract_prefill_snapshot_states for
# the cache classes DeepSeek-V4 uses (RotatingKVCache, CacheList(Rotating,
# Pooling[, Pooling])).
# --------------------------------------------------------------------------
def _normalize_rotating_snapshot_state(layer_cache, state, meta_state):
    """Canonicalize a RotatingKVCache state to the latest max_size tokens in
    temporal order with _idx == keys.shape[2] (scheduler.py:7410)."""
    if not isinstance(state, (list, tuple)) or len(state) < 2:
        return state, (tuple(meta_state) if isinstance(meta_state, (list, tuple)) else ())
    keys, values = state[0], state[1]
    if keys is None or values is None or not hasattr(keys, "shape"):
        return state, (tuple(meta_state) if isinstance(meta_state, (list, tuple)) else ())
    keep = int(meta_state[0]) if meta_state and len(meta_state) >= 1 else int(getattr(layer_cache, "keep", 0))
    max_size = int(meta_state[1]) if meta_state and len(meta_state) >= 2 else int(getattr(layer_cache, "max_size", keys.shape[2]))
    offset = int(meta_state[2]) if meta_state and len(meta_state) >= 3 else int(getattr(layer_cache, "offset", keys.shape[2]))
    ordered_keys, ordered_values = keys, values
    temporal_order = getattr(layer_cache, "_temporal_order", None)
    if callable(temporal_order):
        try:
            ordered_keys = temporal_order(keys)
            ordered_values = temporal_order(values)
        except Exception:
            ordered_keys, ordered_values = keys, values
    original_len = int(ordered_keys.shape[2]) if len(ordered_keys.shape) >= 3 else 0
    nk, nv = ordered_keys, ordered_values
    if max_size > 0 and original_len > max_size:
        if keep > 0 and keep < max_size:
            tail = max_size - keep
            nk = mx.concatenate([ordered_keys[..., :keep, :], ordered_keys[..., -tail:, :]], axis=2)
            nv = mx.concatenate([ordered_values[..., :keep, :], ordered_values[..., -tail:, :]], axis=2)
        else:
            nk = ordered_keys[..., -max_size:, :]
            nv = ordered_values[..., -max_size:, :]
        nk, nv = mx.contiguous(nk), mx.contiguous(nv)
    normalized_len = int(nk.shape[2]) if len(nk.shape) >= 3 else 0
    return (nk, nv), (str(keep), str(max_size), str(offset), str(normalized_len))


def extract_cache_states(raw_cache: list[Any], model_name: str = "") -> tuple[list[dict[str, Any]], ModelCacheConfig | None]:
    """Extract per-layer state dicts exactly like Scheduler._extract_cache_states."""
    model_cache_config = None
    if not any(c is None for c in raw_cache):
        try:
            model_cache_config = ModelCacheConfig.from_cache_list(raw_cache, model_name=model_name)
        except Exception as e:
            logger.debug("ModelCacheConfig failed: %s", e)
    extracted: list[dict[str, Any]] = []
    for layer_idx, layer_cache in enumerate(raw_cache):
        if layer_cache is None:
            extracted.append({"state": (), "meta_state": (), "class_name": "KVCache", "cache_type": "KVCache"})
            continue
        class_name = type(layer_cache).__name__
        if class_name == "CacheList":
            handler = CacheTypeRegistry.get_handler_by_class_name("CacheList")
            sd = handler.extract_state(layer_cache)
            sub_states = list(sd.get("sub_states", []))
            sub_class_names = list(sd.get("sub_class_names", []))
            sub_meta_states = list(sd.get("sub_meta_states", []))
            for sub_idx, sub_cache in enumerate(getattr(layer_cache, "caches", ())):
                if sub_idx >= len(sub_states):
                    break
                if type(sub_cache).__name__ in _ROTATING_NAMES:
                    st, meta = _normalize_rotating_snapshot_state(
                        sub_cache, sub_states[sub_idx],
                        sub_meta_states[sub_idx] if sub_idx < len(sub_meta_states) else getattr(sub_cache, "meta_state", ()),
                    )
                    sub_states[sub_idx] = st
                    if sub_idx < len(sub_meta_states):
                        sub_meta_states[sub_idx] = meta
            extracted.append({
                "state": sub_states,
                "meta_state": (sub_class_names, sub_meta_states),
                "sub_class_names": sub_class_names,
                "class_name": "CacheList",
                "cache_type": "CacheList",
            })
            continue
        state = layer_cache.state
        meta = getattr(layer_cache, "meta_state", ())
        if class_name in _ROTATING_NAMES or CacheTypeRegistry.is_rotating_family(class_name):
            state, meta = _normalize_rotating_snapshot_state(layer_cache, state, meta)
        try:
            cache_type_name = CacheTypeRegistry.detect_cache_type(layer_cache).value
        except Exception:
            cache_type_name = class_name
        extracted.append({
            "state": tuple(state) if isinstance(state, (list, tuple)) else state,
            "meta_state": tuple(meta) if isinstance(meta, (list, tuple)) else meta,
            "class_name": class_name,
            "cache_type": cache_type_name,
        })
    return extracted, model_cache_config


def _copy_containers(value):
    if isinstance(value, list):
        return [_copy_containers(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_copy_containers(v) for v in value)
    return value


def _eval_leaves(extracted):
    leaves, pending = [], [ls.get("state") for ls in extracted]
    while pending:
        v = pending.pop()
        if isinstance(v, mx.array):
            leaves.append(v)
        elif isinstance(v, (list, tuple)):
            pending.extend(v)
        elif isinstance(v, dict):
            pending.extend(v.values())
    if leaves:
        mx.eval(leaves)


def snapshot_boundary_states(cache_list: list[Any], token_count: int, model_name: str = "", block_size: int = BLOCK_SIZE) -> list[dict[str, Any]]:
    """Extract + compact + materialize the cache state at a 2048 boundary,
    exactly as the scheduler's prefill-time boundary snapshot does."""
    if token_count % block_size != 0:
        raise ValueError(f"token_count {token_count} is not a multiple of block_size {block_size}")
    extracted, _ = extract_cache_states(cache_list, model_name)
    for ls in extracted:
        ls["state"] = _copy_containers(ls.get("state"))
        ls["meta_state"] = _copy_containers(ls.get("meta_state"))
    _eval_leaves(extracted)
    compact_pooling_cache_snapshot(extracted, token_count, block_size)
    # the compacted deltas are lazy views -> materialize like _compact_boundary_snapshot_value
    delta_arrays = []
    for layer in extracted:
        for sub_idx in layer.get("pooling_delta_ranges", {}):
            pooled = layer["state"][int(sub_idx)][2]
            if isinstance(pooled, mx.array):
                delta_arrays.append(pooled)
    if delta_arrays:
        mx.eval(*delta_arrays)
    # detach from the live cache buffers (clone) so later chunks cannot alias
    for layer in extracted:
        layer["state"] = _clone_state(layer["state"])
    return extracted


def _clone_state(v):
    if isinstance(v, mx.array):
        out = mx.contiguous(v) if hasattr(mx, "contiguous") else v + 0
        mx.eval(out)
        return out
    if isinstance(v, list):
        return [_clone_state(x) for x in v]
    if isinstance(v, tuple):
        return tuple(_clone_state(x) for x in v)
    return v


# --------------------------------------------------------------------------
# The writer
# --------------------------------------------------------------------------
class _ModelStub:
    """Minimal stand-in for BlockAwarePrefixCache(model=...): it only calls
    make_cache() to learn the cache-layer count."""

    def __init__(self, cache_list_factory):
        self._f = cache_list_factory

    def make_cache(self):
        return self._f()


class BlockWriter:
    def __init__(
        self,
        model_name: str,
        out_dir: str | os.PathLike,
        cache_list_factory,
        block_size: int = BLOCK_SIZE,
        gdn_ssd_split_enabled: bool = True,  # S1 stamps version 5 / split_recurrent_v1
        max_size_bytes: int = 200 * 1024**3,
    ):
        self.model_name = model_name
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.block_size = block_size
        self.cache_list_factory = cache_list_factory
        fresh = list(cache_list_factory())
        self.num_layers = len(fresh)
        cfg = ModelCacheConfig.from_cache_list(fresh, model_name=model_name)
        self.layer_cache_types = cfg.get_type_names()
        self.cachelist_subtypes = cachelist_subtypes_from_cache_list(fresh)
        self.paged = PagedCacheManager(block_size=block_size, max_blocks=100000, enable_caching=True,
                                       model_name=model_name, initial_blocks=256)
        self.ssd = PagedSSDCacheManager(
            cache_dir=self.out_dir, max_size_bytes=max_size_bytes, hot_cache_max_bytes=0,
            hot_cache_only=False, hot_cache_write_through=False,
            expected_model_name=model_name, expected_num_layers=self.num_layers,
            expected_block_size=block_size, expected_block_size_tokens=block_size,
            gdn_ssd_split_enabled=gdn_ssd_split_enabled,
        )
        self.ssd.set_expected_layer_signature(self.layer_cache_types, cachelist_subtypes=self.cachelist_subtypes)
        self.paged.set_paged_ssd_cache_manager(self.ssd)
        self.prefix = BlockAwarePrefixCache(_ModelStub(cache_list_factory), self.paged, self.ssd,
                                            gdn_ssd_split_enabled=gdn_ssd_split_enabled)
        self.snapshots: dict[int, list[dict[str, Any]]] = {}
        self.stream_boundaries = os.environ.get("PD_STREAM_BOUNDARIES", "1") != "0"

    # -- capture -----------------------------------------------------------
    def snapshot(self, cache_list: list[Any], token_count: int) -> None:
        """Call at every 2048-token boundary during prefill (cache state must
        already be evaluated for exactly `token_count` tokens)."""
        self.snapshots[token_count] = snapshot_boundary_states(cache_list, token_count, self.model_name, self.block_size)

    def add_extracted_snapshot(self, token_count: int, extracted: list[dict[str, Any]]) -> None:
        self.snapshots[token_count] = extracted

    # -- incremental streaming (9/6, qwenmax seat): the assemble path can store each boundary THE MOMENT it
    # is snapshotted, so peak memory is ONE boundary snapshot (~20 MB * boundary index / N) instead of the
    # whole N(N+1)/2 accumulation. This is what lets a 100K-token capture (51 boundaries) fit beside a
    # 156 GB resident model. finalize() transparently handles a mix of already-stored and pending snapshots.
    def begin_stream(self, token_ids, request_id: str = "pd-bridge") -> None:
        self._stream_ids = list(token_ids)
        self._stream_request_id = request_id
        self._stream_cfg = ModelCacheConfig.from_cache_list(list(self.cache_list_factory()), model_name=self.model_name)
        self._stored: list[int] = []

    def store_boundary(self, token_count: int) -> None:
        """Store one already-snapshotted boundary immediately and release it. Requires begin_stream()."""
        ids = self._stream_ids
        if not (0 < token_count <= len(ids) and token_count % self.block_size == 0):
            raise ValueError(f"store_boundary: {token_count} not a block-aligned boundary within {len(ids)} tokens")
        snap = self.snapshots.pop(token_count)
        table = self.prefix.store_cache(
            self._stream_request_id, ids[:token_count], snap,
            model_cache_config=self._stream_cfg,
            boundary_snapshots=None,
            extra_keys=None, extra_key_token_start=None, extra_key_ranges=None,
            hot_cache_write_back=True,
        )
        del snap
        if table is None:
            raise RuntimeError(f"store_cache returned None at boundary {token_count}")
        self._stored.append(token_count)

    # -- write -------------------------------------------------------------
    def finalize(self, token_ids: list[int], request_id: str = "pd-bridge") -> list[Path]:
        if not self.snapshots and not getattr(self, "_stored", None):
            raise ValueError("no boundary snapshots captured")
        stored = sorted(getattr(self, "_stored", []))
        valid = sorted(set(stored) | {tc for tc in self.snapshots if 0 < tc <= len(token_ids) and tc % self.block_size == 0})
        if not valid:
            raise ValueError("no boundary snapshot within token_ids")
        latest = valid[-1]
        tokens = list(token_ids[:latest])
        model_cache_config = ModelCacheConfig.from_cache_list(list(self.cache_list_factory()), model_name=self.model_name)
        hashes = self.chain_hashes(tokens)
        paths = [self.ssd._get_file_path(h) for h in hashes]
        self._expected_paths = paths
        # STREAMING STORE (9/6): batching every boundary snapshot into one store_cache made peak memory grow
        # QUADRATICALLY with prompt length (each snapshot materialises the cumulative cache: boundary k ~ k*20 MB,
        # so N boundaries ~ N(N+1)/2 * 20 MB — 39 boundaries ok, 47 exhausted a 256 GB Mac holding a 156 GB model).
        # Store one boundary at a time, newest last, releasing each snapshot as we go: peak is now ONE snapshot.
        # `stream_boundaries=False` restores the old batched behaviour for comparison.
        # Peak-memory preflight (diagnosability): with the batched path every snapshot is held at once
        # (~N*(N+1)/2 blocks worth); with the streaming path only what is still pending. Log before storing
        # so an out-of-headroom failure is diagnosable instead of surfacing as "block files not written".
        _n = len(valid)
        _est_gb = (_n * (_n + 1) / 2) * self.block_size * _KB_PER_TOKEN_EST / 1e6
        logger.info("finalize: %d boundaries (%d already streamed), %d tokens, batched-path peak ~%.1f GB",
                    _n, len(getattr(self, "_stored", [])), latest, _est_gb)
        table = None
        if self.stream_boundaries:
            for tc in valid:
                if tc not in self.snapshots:
                    continue  # already stored incrementally via store_boundary()
                snap = self.snapshots.pop(tc)
                table = self.prefix.store_cache(
                    request_id, list(token_ids[:tc]), snap,
                    model_cache_config=model_cache_config,
                    boundary_snapshots=None,
                    extra_keys=None, extra_key_token_start=None, extra_key_ranges=None,
                    hot_cache_write_back=True,
                )
                del snap
                if table is None:
                    raise RuntimeError(f"store_cache returned None at boundary {tc}")
        else:
            cache_to_store = self.snapshots[latest]
            intermediate = {tc: self.snapshots[tc] for tc in valid if tc != latest}
            table = self.prefix.store_cache(
                request_id, tokens, cache_to_store,
                model_cache_config=model_cache_config,
                boundary_snapshots=intermediate,
                extra_keys=None, extra_key_token_start=None, extra_key_ranges=None,
                hot_cache_write_back=True,
            )
            if table is None:
                raise RuntimeError("store_cache returned None")
        self._drain()
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise RuntimeError(f"{len(missing)} block files not written: {missing[:2]}")
        logger.info("wrote %d blocks for %d tokens -> %s", len(paths), latest, self.out_dir)
        return paths

    def chain_hashes(self, tokens: list[int]) -> list[bytes]:
        out, parent = [], None
        for i in range(len(tokens) // self.block_size):
            blk = tokens[i * self.block_size:(i + 1) * self.block_size]
            h = compute_block_hash(parent, blk, extra_keys=None, model_name=self.model_name)
            out.append(h)
            parent = h
        return out

    def _drain(self, timeout: float = 45.0) -> None:
        """Wait until the background writer has committed every expected file.

        Watches the files themselves (exists + size stable across two polls)
        rather than the manager's pending-write bookkeeping, which can keep an
        entry after the rename and stall a naive wait forever."""
        expected = getattr(self, "_expected_paths", None) or []
        t0 = time.time()
        last_sizes: dict[Path, int] = {}
        stall_t0 = time.time()
        n_done = -1
        while time.time() - t0 < timeout:
            sizes = {p: (p.stat().st_size if p.exists() else -1) for p in expected}
            if expected and all(s > 0 for s in sizes.values()) and sizes == last_sizes:
                break
            done = sum(1 for s in sizes.values() if s > 0)
            if done != n_done:
                n_done, stall_t0 = done, time.time()
            elif time.time() - stall_t0 > 30.0:
                logger.warning("drain stalled: %d/%d blocks after 30s with no progress", done, len(expected))
                break
            last_sizes = sizes
            time.sleep(0.2)

    def close(self):
        """Stop the SSD writer thread without blocking the caller."""
        import threading

        def _c():
            try:
                self.ssd.close()
            except Exception:
                pass

        t = threading.Thread(target=_c, daemon=True)
        t.start()
        t.join(timeout=10.0)


def write_blocks(snapshots: dict[int, list[Any]], token_ids: list[int], model_name: str,
                 out_dir: str | os.PathLike, cache_list_factory, block_size: int = BLOCK_SIZE) -> list[Path]:
    """One-shot: snapshots = {token_count: live cache_list at that boundary}
    (raw cache objects, NOT yet extracted) -> block files."""
    w = BlockWriter(model_name, out_dir, cache_list_factory, block_size=block_size)
    for tc, cl in sorted(snapshots.items()):
        w.snapshot(cl, tc)
    try:
        return w.finalize(list(token_ids))
    finally:
        w.close()


def chain_hashes_for(token_ids: list[int], model_name: str, block_size: int = BLOCK_SIZE) -> list[str]:
    out, parent = [], None
    for i in range(len(token_ids) // block_size):
        blk = list(token_ids[i * block_size:(i + 1) * block_size])
        h = compute_block_hash(parent, blk, extra_keys=None, model_name=model_name)
        out.append(h.hex())
        parent = h
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="print oMLX chain hashes for a token-id JSON list")
    ap.add_argument("--tokens", required=True, help="json file with a list of ints")
    ap.add_argument("--model-name", required=True)
    a = ap.parse_args()
    ids = json.load(open(a.tokens))
    for i, h in enumerate(chain_hashes_for(ids, a.model_name)):
        print(f"block {i:3d} tokens[{i*BLOCK_SIZE}:{(i+1)*BLOCK_SIZE}] -> {h[0]}/{h}.safetensors")
