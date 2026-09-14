"""pd_omlx_hooks.py — measurement hooks inside a running oMLX server (docs/RDMA.md, O2/O4).

Loaded only when the server is started with PD_OMLX_HOOKS=1 (a one-line .pth in the oMLX venv imports it), so the
front door and the bench scripts that share the venv never see it. Patches are applied right after the oMLX modules
they touch finish importing.

PD_OMLX_TIMING=1   log the restore split. Both patched calls return lazy MLX graphs, so the hook evaluates the result
                   before returning: the work is the same work the first forward pass would do, now attributed.
                     [pd-timing] reconstruct  blocks, tokens, call ms (graph), eval ms (read + concat)
                     [pd-timing] merge        tokens, eval ms (BatchKVCache.merge: zeros + copy)
"""
import logging
import os
import sys
import time
import importlib.abc
import importlib.machinery

log = logging.getLogger("omlx.pd_hooks")
TIMING = os.environ.get("PD_OMLX_TIMING") == "1"


def _eval_cache_list(caches):
    import mlx.core as mx
    arrs = []
    for c in caches or []:
        for name in ("keys", "values"):
            a = getattr(c, name, None)
            if a is not None:
                arrs.append(a)
    if arrs:
        mx.eval(arrs)
    return arrs


STAGE = os.environ.get("PD_OMLX_STAGE") == "1"
_STAGED = {}
_STAGED_LOCK = __import__("threading").Lock()


class _Stage:
    """O2c: the bridge announces a request's block hashes before prefill starts (POST /pd/stage). This thread allocates
    the whole restored cache at once — per layer K and V, [1, kv_heads, blocks*block + reserve, head_dim], as uint16 so
    numpy can write into it — and copies every block file in as `pd_rdma recvd` lands it (atomic rename), one copy from
    the page cache, while the Spark is still prefilling. The restore then hands these arrays to the request as bf16 views:
    no block read, no concat, no allocation on the critical path."""

    def __init__(self, hashes, layers, kv_heads, head_dim, block, reserve, cache_dir, timeout_s):
        import threading
        self.hashes = [bytes.fromhex(h) for h in hashes]
        self.n, self.layers, self.kv, self.hd, self.block = len(self.hashes), layers, kv_heads, head_dim, block
        self.cap = ((self.n * block + max(reserve, 0) + block - 1) // block) * block
        self.cache_dir, self.timeout_s = cache_dir, timeout_s
        self.key = self.hashes[-1].hex() if self.hashes else ""
        self.ready, self.error, self.arrs, self.created = 0, None, None, time.time()
        self.t_alloc_ms = self.t_copy_ms = self.t_touch_ms = self.t_keep_ms = 0.0
        self.per_block = []                    # (landed at ms since copy start, copy ms, touch ms)
        self.thread = threading.Thread(target=self._run, name="pd_stage", daemon=True)
        self.thread.start()

    def _run(self):
        import json, mmap, struct
        import mlx.core as mx
        import numpy as np
        try:
            t0 = time.perf_counter()
            # GPU-stream allocation: buffers allocated on the CPU stream and written through numpy cost the first GPU
            # write ~1 s at 16–18 GB (micro-test: 100 ms vs 2.7 ms at 2.4 GB), which would land on the critical path.
            arrs = [mx.zeros((1, self.kv, self.cap, self.hd), dtype=mx.uint16, stream=mx.gpu) for _ in range(2 * self.layers)]
            mx.eval(arrs)
            views = [np.array(a, copy=False) for a in arrs]
            self.arrs = arrs
            self.t_alloc_ms = (time.perf_counter() - t0) * 1e3
            t1 = time.perf_counter()
            B = self.block
            for j, h in enumerate(self.hashes):
                hx = h.hex()
                path = os.path.join(self.cache_dir, hx[0], hx + ".safetensors")
                last_keep = time.perf_counter()
                while not os.path.isfile(path):
                    if time.time() - self.created > self.timeout_s:
                        self.error = f"block {j} never landed"
                        return
                    # Keep the staged buffers warm across the prefill step's gap. After a 9 s gap the first GPU touch
                    # took 1,546 ms in the server (4–5 ms mid-step), holding back the final burst. Micro-test, 8 GB, 9 s
                    # gap: no keepalive 336 ms, one element per buffer 132 ms, an unrelated GPU op 129 ms, a sum over a
                    # 256-row slice of every buffer every 0.5 s 5.5 ms (9 ms per keepalive). A real read per buffer it is.
                    if time.perf_counter() - last_keep > 0.5:
                        tk = time.perf_counter()
                        mx.eval([mx.sum(a[:, :, :B, :], stream=mx.gpu) for a in arrs])
                        self.t_keep_ms += (time.perf_counter() - tk) * 1e3
                        last_keep = time.perf_counter()
                    time.sleep(0.01)
                t_seen = time.perf_counter()
                with open(path, "rb") as f:
                    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                    try:
                        hl = struct.unpack("<Q", mm[:8])[0]
                        hdr = json.loads(mm[8:8 + hl])
                        base = 8 + hl
                        for i in range(self.layers):
                            for s in (0, 1):
                                t = hdr[f"layer_{i}_state_{s}"]
                                if t["dtype"] != "BF16" or t["shape"] != [1, self.kv, B, self.hd]:
                                    self.error = f"block {j} layer {i}: {t['dtype']} {t['shape']}"
                                    return
                                lo, hi = t["data_offsets"]
                                src = np.frombuffer(mm, dtype=np.uint16, count=(hi - lo) // 2, offset=base + lo)
                                views[2 * i + s][:, :, j * B:(j + 1) * B, :] = src.reshape(1, self.kv, B, self.hd)
                                del src
                    finally:
                        mm.close()
                # Read-only GPU touch of this block's rows. In the running server the first GPU write into pages numpy
                # dirtied costs ~1.1 s at 32K even for GPU-stream buffers; without a pass it lands in the tail append.
                # Measured variants (decoder after handoff): one pass after the last block 1.45–1.55 s, per block
                # 1.38–1.68 s, touch only while ahead of the wire 2.08 s. Per block is kept: simplest, same band.
                tt = time.perf_counter()
                mx.eval([mx.sum(a[:, :, j * B:(j + 1) * B, :], stream=mx.gpu) for a in arrs])
                t_touch = (time.perf_counter() - tt) * 1e3
                self.t_touch_ms += t_touch
                self.per_block.append((round((t_seen - t1) * 1e3), round((tt - t_seen) * 1e3, 1), round(t_touch, 1)))
                self.ready = j + 1
            self.t_copy_ms = (time.perf_counter() - t1) * 1e3
            log.info("[pd-stage] %s staged %d blocks (%d tokens, cap %d): alloc %.0f ms, span %.0f ms while prefilling "
                     "(of which GPU touch %.0f ms, keepalive %.0f ms)", self.key[:16], self.n, self.n * B, self.cap,
                     self.t_alloc_ms, self.t_copy_ms, self.t_touch_ms, self.t_keep_ms)
            log.info("[pd-stage] %s last 34 blocks (landed ms, copy ms, touch ms): %s", self.key[:16], self.per_block[-34:])
        except Exception as e:
            self.error = repr(e)
            log.warning("[pd-stage] %s failed: %r", self.key[:16], e)


def _stage_request(body):
    cache_dir = os.path.expanduser(os.environ.get("PD_CACHE_DIR", "~/.omlx/cache"))
    now = time.time()
    with _STAGED_LOCK:
        for k in [k for k, e in _STAGED.items() if now - e.created > 900]:   # never collected by a request: drop
            _STAGED.pop(k, None)
    e = _Stage(body["hashes"], int(body.get("layers", 64)), int(body.get("kv_heads", 8)), int(body.get("head_dim", 128)),
               int(body.get("block", 256)), int(body.get("reserve", RESERVE or 1024)), cache_dir, float(body.get("timeout_s", 600)))
    with _STAGED_LOCK:
        _STAGED[e.key] = e
    return {"ok": True, "key": e.key, "blocks": e.n, "cap_tokens": e.cap}


def _staged_restore(prefix_cache, block_table):
    """Return KVCache objects backed by a finished stage whose hashes start with this request's matched blocks."""
    hashes = []
    for bid in block_table.block_ids:
        b = prefix_cache.paged_cache.allocated_blocks.get(bid)
        if b is None or b.block_hash is None:
            return None
        hashes.append(bytes(b.block_hash))
    with _STAGED_LOCK:
        entries = list(_STAGED.values())
    for e in entries:
        if e.error or not hashes or len(hashes) > e.n or e.hashes[:len(hashes)] != hashes:
            continue
        deadline = time.time() + 15            # blocks land before DONE; the last copies may still be running
        while e.ready < len(hashes) and not e.error and time.time() < deadline:
            time.sleep(0.005)
        if e.ready < len(hashes) or e.arrs is None:
            log.warning("[pd-stage] %s not ready (%d/%d, %s); falling back to disk restore", e.key[:16], e.ready, len(hashes), e.error)
            return None
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache
        n_tok = len(hashes) * e.block
        caches = []
        for i in range(e.layers):
            c = KVCache()
            c.keys = e.arrs[2 * i].view(mx.bfloat16)
            c.values = e.arrs[2 * i + 1].view(mx.bfloat16)
            c.offset = n_tok
            caches.append(c)
        # The bf16 views must be the only owners of the buffers: KVCache appends by slice assignment, and MLX writes in
        # place only when it can donate the buffer. A live uint16 base in the stage would force a full copy per layer.
        e.arrs = None
        with _STAGED_LOCK:
            _STAGED.pop(e.key, None)           # the arrays now belong to the request's cache
        return caches
    return None


def _patch_server(mod):
    if not STAGE:
        return
    from fastapi import Request as _Req

    @mod.app.post("/pd/stage")
    async def pd_stage(req: _Req):
        return _stage_request(await req.json())

    log.info("[pd-hooks] POST /pd/stage registered")


def _patch_prefix_cache(mod):
    cls = mod.BlockAwarePrefixCache
    orig = cls.reconstruct_cache

    def reconstruct_cache(self, block_table, *a, **kw):
        t0 = time.perf_counter()
        if STAGE and block_table is not None and block_table.block_ids:
            staged = _staged_restore(self, block_table)
            if staged is not None:
                import mlx.core as mx
                mx.eval([t for c in staged for t in (c.keys, c.values)])
                log.info("[pd-stage] restore from staged arrays: blocks=%d tokens=%d in %.1f ms",
                         len(block_table.block_ids), staged[0].offset, (time.perf_counter() - t0) * 1e3)
                return staged
        res = orig(self, block_table, *a, **kw)
        t1 = time.perf_counter()
        if TIMING and res:
            arrs = _eval_cache_list(res)
            t2 = time.perf_counter()
            tokens = arrs[0].shape[2] if arrs else 0
            log.info("[pd-timing] reconstruct blocks=%d tokens=%d call=%.1fms eval=%.1fms",
                     len(block_table.block_ids), tokens, (t1 - t0) * 1e3, (t2 - t1) * 1e3)
        return res

    cls.reconstruct_cache = reconstruct_cache
    log.info("[pd-hooks] BlockAwarePrefixCache.reconstruct_cache wrapped (timing=%s)", TIMING)

    if TIMING:
        # the prefix walk that runs before the restore (has_block per block, disk-index fallback for bridged files)
        orig_fetch = cls.fetch_cache

        def fetch_cache(self, request_id, tokens, *a, **kw):
            t0 = time.perf_counter()
            res = orig_fetch(self, request_id, tokens, *a, **kw)
            bt = res[0] if isinstance(res, tuple) and res else None
            if tokens is not None and len(tokens) >= 1024:
                log.info("[pd-timing] fetch_cache tokens=%d matched=%s in %.1f ms", len(tokens),
                         getattr(bt, "num_tokens", None), (time.perf_counter() - t0) * 1e3)
            return res

        cls.fetch_cache = fetch_cache

    if TIMING:
        # A restored cache is exactly prompt-sized, so the first append of the uncached tail takes the growth branch of
        # KVCache.update_and_fetch: zeros + concatenate, i.e. a full copy of every layer. Time only large growths.
        from mlx_lm.models.cache import KVCache
        import mlx.core as mx
        orig_uaf = KVCache.update_and_fetch
        acc = {"ms": 0.0, "layers": 0, "prev": 0}

        def update_and_fetch(self, keys, values):
            prev = self.offset
            grows = self.keys is not None and prev >= 16384 and prev + keys.shape[2] > self.keys.shape[2]
            if not grows:
                return orig_uaf(self, keys, values)
            t0 = time.perf_counter()
            res = orig_uaf(self, keys, values)
            mx.eval(self.keys, self.values)
            acc["ms"] += (time.perf_counter() - t0) * 1e3
            acc["layers"] += 1
            acc["prev"] = prev
            if acc["layers"] % 64 == 0:
                log.info("[pd-timing] kvcache-grow prev=%d layers=%d eval=%.1fms", acc["prev"], acc["layers"], acc["ms"])
                acc.update(ms=0.0, layers=0)
            return res

        KVCache.update_and_fetch = update_and_fetch
        log.info("[pd-hooks] KVCache.update_and_fetch growth timing on")


TAIL = os.environ.get("PD_OMLX_TAIL") == "1"
TAIL_DIR = os.path.join(os.path.expanduser(os.environ.get("PD_CACHE_DIR", "~/.omlx/cache")), "pd_tail")


def _prompt_key(ids):
    import hashlib, struct
    return hashlib.sha256(struct.pack(f"<{len(ids)}i", *ids)).hexdigest()


def _install_tail(sched, request):
    """O2b: the Spark computed the rows after the last full block too. The front door lands them (all but the prompt's
    last token) at pd_tail/<sha256(prompt ids)>.safetensors. When a restored cache stops exactly at the last full block,
    append those rows in place (O2a reserve) and leave one token for the decoder instead of the whole tail."""
    ids = list(request.prompt_token_ids or [])
    cache = getattr(request, "prompt_cache", None)
    if not ids or not cache:
        return
    path = os.path.join(TAIL_DIR, _prompt_key(ids) + ".safetensors")
    if not os.path.isfile(path):
        return
    import mlx.core as mx
    t0 = time.perf_counter()
    cached = int(request.cached_tokens or 0)
    arrays = mx.load(path)
    n = arrays["layer_0_state_0"].shape[2]
    if cached + n != len(ids) - 1 or len(request.remaining_tokens or []) != n + 1 or len(cache) * 2 != len(arrays):
        log.warning("[pd-tail] %s does not fit: cached=%d tail=%d prompt=%d remaining=%d layers=%d/%d",
                    os.path.basename(path)[:16], cached, n, len(ids), len(request.remaining_tokens or []),
                    len(cache), len(arrays) // 2)
        return
    for i, layer in enumerate(cache):
        if getattr(layer, "offset", None) != cached:
            log.warning("[pd-tail] layer %d offset %s != cached %d; tail not installed", i, getattr(layer, "offset", None), cached)
            return
    for i, layer in enumerate(cache):
        layer.update_and_fetch(arrays[f"layer_{i}_state_0"], arrays[f"layer_{i}_state_1"])
    mx.eval([t for layer in cache for t in (layer.keys, layer.values)])
    request.cached_tokens = cached + n
    request.remaining_tokens = ids[-1:]
    log.info("[pd-tail] installed %d tail rows for %s: cached %d -> %d, remaining 1 (%.1f ms)",
             n, request.request_id, cached, cached + n, (time.perf_counter() - t0) * 1e3)


def _patch_scheduler(mod):
    if TAIL:
        orig_prepare = mod.Scheduler._prepare_prefix_cache_for_request

        def _prepare_prefix_cache_for_request(self, request):
            already = request.request_id in self._prefix_cache_prepared
            t0 = time.perf_counter()
            if TIMING and not already:
                arrived = getattr(request, "arrival_time", None)
                log.info("[pd-timing] prepare start %s (queued %.1f ms)", request.request_id,
                         (time.time() - arrived) * 1e3 if isinstance(arrived, (int, float)) and arrived > 1e9 else -1.0)
            orig_prepare(self, request)
            if TIMING and not already:
                log.info("[pd-timing] prepare upstream done in %.1f ms", (time.perf_counter() - t0) * 1e3)
            if not already:
                try:
                    _install_tail(self, request)
                except Exception as e:   # a failed install leaves the request exactly as upstream prepared it
                    log.warning("[pd-tail] install failed for %s: %r", request.request_id, e)

        mod.Scheduler._prepare_prefix_cache_for_request = _prepare_prefix_cache_for_request
        log.info("[pd-hooks] tail install on (%s)", TAIL_DIR)

    # `import mlx_lm.generate` yields the generate() function (mlx_lm re-exports it), not the module
    gen = mod._mlx_lm_generate_module
    orig_merge = gen._merge_caches           # oMLX's _patched_merge_caches, installed by this import

    def merge_caches(caches):
        t0 = time.perf_counter()
        res = orig_merge(caches)
        if TIMING and res:
            arrs = _eval_cache_list(res)
            tokens = arrs[0].shape[2] if arrs else 0
            if tokens >= 1024:               # decode-step merges of empty caches are noise
                log.info("[pd-timing] merge rows=%d tokens=%d eval=%.1fms", len(caches), tokens,
                         (time.perf_counter() - t0) * 1e3)
        return res

    gen._merge_caches = merge_caches
    log.info("[pd-hooks] mlx_lm.generate._merge_caches wrapped (timing=%s)", TIMING)


RESERVE = int(os.environ.get("PD_OMLX_RESERVE_TOKENS", "0") or 0)


def _patch_type_handlers(mod):
    """O2a: restore a KVCache into a buffer with RESERVE spare positions. Upstream concatenates the blocks into an
    exactly prompt-sized array, so the first append of the uncached tail takes KVCache's growth branch and copies the
    whole cache again (2.0 s at 62K). Concatenating the blocks together with a zero tail is still one copy, and the tail
    then lands in place. offset stays the real token count; KVCache.state slices by offset."""
    if RESERVE <= 0:
        return
    import mlx.core as mx
    cls = mod.KVCacheHandler
    orig_concat, orig_rebuild = cls.concatenate_states, cls.reconstruct_cache

    def concatenate_states(self, states):
        keys = [s["keys"] for s in states if s.get("keys") is not None]
        values = [s["values"] for s in states if s.get("values") is not None]
        if not keys or not values or keys[0].ndim != 4:
            return orig_concat(self, states)
        n = sum(k.shape[2] for k in keys)
        step = 256
        spare = ((n + RESERVE + step - 1) // step) * step - n
        kz = mx.zeros((keys[0].shape[0], keys[0].shape[1], spare, keys[0].shape[3]), dtype=keys[0].dtype)
        vz = mx.zeros((values[0].shape[0], values[0].shape[1], spare, values[0].shape[3]), dtype=values[0].dtype)
        return {"keys": mx.concatenate(keys + [kz], axis=2), "values": mx.concatenate(values + [vz], axis=2),
                "offset": n, "pd_offset": n, "cache_type": self.cache_type.value}

    def reconstruct_cache(self, state, meta_state=None):
        cache = orig_rebuild(self, state, meta_state)
        if cache is not None and "pd_offset" in state:
            cache.offset = state["pd_offset"]           # upstream sets offset = keys.shape[2], which now includes spare
        return cache

    cls.concatenate_states, cls.reconstruct_cache = concatenate_states, reconstruct_cache
    log.info("[pd-hooks] KVCacheHandler restore reserve = %d tokens", RESERVE)


_PATCHES = {
    "omlx.cache.prefix_cache": _patch_prefix_cache,
    "omlx.cache.type_handlers": _patch_type_handlers,
    "omlx.scheduler": _patch_scheduler,
    "omlx.server": _patch_server,
}


class _Loader(importlib.abc.Loader):
    def __init__(self, inner, name):
        self.inner, self.name = inner, name

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        try:
            _PATCHES[self.name](module)
        except Exception as e:  # never break the server over a measurement hook
            log.warning("[pd-hooks] patch of %s failed: %r", self.name, e)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _PATCHES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader, name)
        return spec


sys.meta_path.insert(0, _Finder())
