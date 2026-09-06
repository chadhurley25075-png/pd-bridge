"""
capture_sitecustomize_v3.py — (operator) DV4-Flash heterogeneous P/D, Spark-side POOLED capture hook.
(AGENT B, 2026-09-06 — DESIGN-v3-pooled.md "v3 CAPTURE FILE FORMAT" (09:40) + "DECISION 09:55")

v2 shipped 43 x [T,4096] bf16 hidden states (36 GB @ 102K tokens). v3 computes the finished
DV4 caches ON THE SPARK, with the MAC'S OWN (dequantized MXFP4) projection weights, and ships ~1 GB:
  * SWA window   : PRE-RoPE kv = rmsnorm(hidden @ wkv.T, kv_norm) rows [b-128, b) at every 2048
                   boundary b and at T                                  -> kvwin_{b} / kvwin_end
  * compressor   : kv_score = [hidden @ comp.wkv.T | hidden @ comp.wgate.T] pooled by
                   pd_pool_torch.pool_layer (AGENT A's torch port of the MLX math)
                                                                          -> pooled, buf_*, prev_*_{b}
  * indexer      : same with idx.* weights on the ratio-4 layers (head_dim 128)
                                                                          -> idx_pooled, idx_buf_*, idx_prev_*_{b}

HOOK POINT (unchanged from v2; in-image plugin /opt/env/lib/python3.12/site-packages/vllm/models/deepseek_v4/):
  attention.py:603  DeepseekV4MultiHeadLatentAttentionWrapper.attention_impl(self, hidden_states,
                    positions, out) — `hidden_states` [num_tokens,4096] bf16 is the attention INPUT
                    (= MLX attn_input); `self.layer_name` (attention.py:297) carries `.layers.N.`.
                    Reached via the custom op body deepseek_v4_attention (attention.py:740), so the
                    class-level wrap runs on every call even under torch.compile.
vLLM's own kv_score / fp8 caches are NOT used (DECISION 09:55). Profile / dummy runs are skipped exactly
like the plugin does (forward_context.attn_metadata not a dict); CUDA-graph capture is skipped
(torch.cuda.is_current_stream_capturing()); TP rank 0 only (hidden_states is replicated across TP).

DATA PATH (v3.1, launch-lean): the forward thread does the MINIMUM — at layer 0 of each chunk one
device clone of `positions` (shared by all 43 layers of that chunk), then per layer: one CUDA event
recorded on the compute stream, hidden_states.record_stream(side), queue.put. No projections, no
copies, no host waits, no cur.wait_event. A single worker thread consumes the queue in order and,
on the side stream, does side.wait_event(ev) (DEVICE-side), the fused projection (one call per
layer per chunk, no piece loop), pool_layer, and the window/prev bookkeeping. The only host sync per
CHUNK is one event wait + one 8-byte-per-token D2H to learn the chunk's start position; the final
flush synchronizes the side stream once. On idle (PD_CAPTURE_IDLE_S, default 2 s) or a new request
at position 0 it writes PD_CAPTURE_DIR/<stamp>/layer_XX.safetensors + manifest.json + DONE (DONE
last; manifest.json is also written at request start with "T": null).

WEIGHTS: PD_PROJ_WEIGHTS (default /pd_v3/dv4_proj_weights.safetensors; sidecar .json with
eps/ratios) — keys layer_{i}.wkv.weight [512,4096] · layer_{i}.kv_norm.weight [512] ·
layer_{i}.comp.{wkv,wgate}.weight [out_dim,4096] · layer_{i}.comp.ape [ratio,out_dim] f32 ·
layer_{i}.comp.norm.weight [512] · layer_{i}.idx.{wkv,wgate}.weight [256,4096] · layer_{i}.idx.ape
[4,256] f32 · layer_{i}.idx.norm.weight [128]. Preloaded to pinned host RAM in a background thread
when the attention module is imported (PD_PROJ_PRELOAD=0 disables), moved to the GPU on the side
stream at the first captured chunk. Missing file => the hook logs LOUDLY and captures nothing.

Env: PD_CAPTURE_DIR (arms the hook) · PD_CAPTURE_IDLE_S=2.0 · PD_CAPTURE_BLOCK=2048 ·
     PD_CAPTURE_MIN_T=64 (discard shorter captures) · PD_CAPTURE_PIECE=0 (0 = never split a chunk;
     >0 = projection piece rows) · PD_CAPTURE_OPCOUNT=1 (sample the aten-op count of ONE worker item) ·
     PD_PROJ_WEIGHTS · PD_POOL_PATH (extra import dir for pd_pool_torch) · PD_CAPTURE_V3_NOARM=1 (import
     without installing the import hook; the selftest uses it).
"""
import importlib.abc
import importlib.machinery
import json
import os
import queue
import re
import shutil
import sys
import threading
import time

_DIR = os.environ.get("PD_CAPTURE_DIR")
_IDLE_S = float(os.environ.get("PD_CAPTURE_IDLE_S", "2.0"))
_BLOCK = int(os.environ.get("PD_CAPTURE_BLOCK", "2048"))
_MIN_T = int(os.environ.get("PD_CAPTURE_MIN_T", "64"))
_PIECE = int(os.environ.get("PD_CAPTURE_PIECE", "0"))
_OPCOUNT = os.environ.get("PD_CAPTURE_OPCOUNT", "1") != "0"
_WEIGHTS = os.environ.get("PD_PROJ_WEIGHTS", "/pd_v3/dv4_proj_weights.safetensors")
_PRELOAD = os.environ.get("PD_PROJ_PRELOAD", "1") != "0"
_WIN = 128                       # DV4 sliding window (config sliding_window=128)
_TAIL_RAW = 128 + 8              # raw compressor rows kept: remainder (<=127) + prev window (4)
_TARGET_ATTN = "vllm.models.deepseek_v4.attention"
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
_VERSION = 3
# DV4-Flash compress_ratios for the 43 real layers (config.json lists 44: index 43 = the MTP draft layer,
# never captured; layer 42 IS ratio 4 — config.json + the MLX export agree). Used only if the weights
# sidecar .json carries no ratios.
_RATIOS_FALLBACK = [0, 0] + [4, 128] * 20 + [4]
# safetensors key suffix -> pd_pool_torch.project_layer / load_proj_weights dict key
_WKEYS = {"wkv.weight": "wkv", "kv_norm.weight": "kv_norm", "comp.wkv.weight": "comp_wkv",
          "comp.wgate.weight": "comp_wgate", "comp.ape": "comp_ape", "comp.norm.weight": "comp_norm",
          "idx.wkv.weight": "idx_wkv", "idx.wgate.weight": "idx_wgate", "idx.ape": "idx_ape",
          "idx.norm.weight": "idx_norm"}


def _log(msg):
    sys.stderr.write(f"[pd_capture_v3 pid={os.getpid()}] {msg}\n")
    sys.stderr.flush()


# ----------------------------------------------------------------------------- pool module
_POOL_MOD = None


def set_pool_module(mod):
    """Inject a pool implementation (tests)."""
    global _POOL_MOD
    _POOL_MOD = mod


def _pool():
    global _POOL_MOD
    if _POOL_MOD is None:
        extra = os.environ.get("PD_POOL_PATH")
        for p in ([extra] if extra else []) + [os.path.dirname(os.path.abspath(__file__)),
                                               os.path.dirname(os.path.abspath(_WEIGHTS))]:
            if p and p not in sys.path:
                sys.path.append(p)
        try:
            import pd_pool_torch  # AGENT A
        except Exception as e:  # fail LOUDLY: a silent fallback would ship wrong caches
            raise RuntimeError(
                "pd_pool_torch (AGENT A's torch pooling port) is not importable: "
                f"{e!r}. Mount it next to sitecustomize.py / the weights, or set PD_POOL_PATH.") from e
        for name in ("pool_layer", "DSv4Rope", "rmsnorm"):
            if not hasattr(pd_pool_torch, name):
                raise RuntimeError(f"pd_pool_torch lacks {name}() — signature drift vs DESIGN-v3")
        _POOL_MOD = pd_pool_torch
    return _POOL_MOD


def _boundaries(lo, hi, block=_BLOCK):
    """Multiples of `block` b with lo < b <= hi."""
    first = (lo // block + 1) * block
    return list(range(first, hi + 1, block))


# ----------------------------------------------------------------------------- weights
class _Weights:
    """dv4_proj_weights.safetensors (+ .json) -> per-layer dict {suffix: tensor}; CPU pinned then GPU."""

    def __init__(self, path=_WEIGHTS):
        self.path = path
        self.json_path = os.path.splitext(path)[0] + ".json"
        self.cpu = None          # li -> {suffix: pinned cpu tensor}
        self.gpu = {}            # li -> {suffix: device tensor}
        self.ratios = None
        self.eps = None
        self.meta = {}
        self.err = None
        self.lock = threading.Lock()

    def load_cpu(self):
        with self.lock:
            if self.cpu is not None or self.err is not None:
                return self.cpu is not None
            import torch
            from safetensors.torch import load_file
            t0 = time.time()
            try:
                raw = load_file(self.path)
            except Exception as e:
                self.err = f"cannot load {self.path}: {e!r}"
                _log("ERROR " + self.err)
                return False
            if os.path.isfile(self.json_path):
                try:
                    self.meta = json.load(open(self.json_path))
                except Exception as e:
                    _log(f"WARNING {self.json_path} unreadable ({e!r}); using fallbacks")
            self.ratios = self.meta.get("ratios") or self.meta.get("compress_ratios")
            self.eps = self.meta.get("eps") or self.meta.get("rms_norm_eps")
            per = {}
            nbytes = 0
            for k, v in raw.items():
                m = re.match(r"layer_(\d+)\.(.+)$", k)
                if not m:
                    continue
                li, suffix = int(m.group(1)), m.group(2)
                if suffix not in _WKEYS:
                    continue
                try:
                    v = v.pin_memory()
                except Exception:
                    pass
                per.setdefault(li, {})[_WKEYS[suffix]] = v
                nbytes += v.numel() * v.element_size()
            if self.ratios is None:
                self.ratios = [(4 if "idx_wkv" in per[i] else (128 if "comp_wkv" in per[i] else 0)) if i in per else None
                               for i in range(max(per) + 1)] if per else None
                _log("weights sidecar carries no ratios; derived them from the key layout")
            if self.eps is None:
                self.eps = getattr(_pool(), "RMS_NORM_EPS", 1e-6)
            for li, d in per.items():   # scalar fields project_layer expects (pd_pool_torch.load_proj_weights layout)
                d["ratio"] = self.ratio_from(li)
                d["eps"] = self.eps
                d["head_dim"] = int(self.meta.get("head_dim", 512))
                if "comp_wkv" in d:
                    d["comp_out_dim"] = int(d["comp_wkv"].shape[0])
            self.cpu = per
            _log(f"projection weights loaded to host: {len(per)} layers, {nbytes/1e9:.2f} GB in {time.time()-t0:.1f}s "
                 f"(eps={self.eps}, ratios={'sidecar' if 'ratios' in self.meta or 'compress_ratios' in self.meta else 'derived'})")
            return True

    def layer(self, li, device):
        """Per-layer dict on `device` (H2D on the CURRENT stream, async from pinned memory)."""
        w = self.gpu.get(li)
        if w is None:
            if not self.load_cpu():
                raise RuntimeError(self.err)
            src = self.cpu.get(li)
            if src is None:
                raise RuntimeError(f"no projection weights for layer {li} in {self.path}")
            w = {k: (v.to(device, non_blocking=True) if hasattr(v, "to") else v) for k, v in src.items()}
            self.gpu[li] = w
        return w

    def ratio_from(self, li):
        if self.ratios is not None and li < len(self.ratios) and self.ratios[li] is not None:
            return int(self.ratios[li])
        return _RATIOS_FALLBACK[li] if li < len(_RATIOS_FALLBACK) else 0

    def ratio(self, li):
        if self.cpu is None:
            self.load_cpu()
        return self.ratio_from(li)


def project_layer_torch(hidden, W):
    """hidden [L,4096] bf16, W = per-layer dict (pd_pool_torch.load_proj_weights layout) ->
    (kv_pre [L,512], kv_score [L,2*out_dim] | None, idx_kv_score [L,512] | None).
    Same contract as pd_pool_torch.project_layer, which is preferred when present."""
    import torch
    P = _pool()
    kv_pre = P.rmsnorm(hidden @ W["wkv"].T, W["kv_norm"], W["eps"])
    kv_score = idx = None
    if W["ratio"] > 0:
        kv_score = torch.cat([hidden @ W["comp_wkv"].T, hidden @ W["comp_wgate"].T], -1)
    if W["ratio"] == 4:
        idx = torch.cat([hidden @ W["idx_wkv"].T, hidden @ W["idx_wgate"].T], -1)
    return kv_pre, kv_score, idx


# ----------------------------------------------------------------------------- per-layer state
class _KVState:
    """Rolling last-128 PRE-RoPE kv rows + snapshots at every block boundary."""

    def __init__(self):
        self.tail = None      # [<=128, 512] device bf16
        self.snaps = {}       # b -> [128,512]
        self.next_pos = 0
        self.rows = 0
        self.gaps = []

    def feed(self, kv, start):
        import torch
        n = kv.shape[0]
        if start != self.next_pos:
            self.gaps.append((self.next_pos, start))
        full = base = None   # lazily built cat(tail, kv) — only when a window straddles the chunk start
        for b in _boundaries(start, start + n):
            if b - _WIN >= start:
                self.snaps[b] = kv[b - _WIN - start:b - start].clone()
            else:
                if full is None:
                    full = kv if self.tail is None else torch.cat([self.tail, kv], 0)
                    base = start - (0 if self.tail is None else self.tail.shape[0])
                lo = max(b - _WIN, base)
                self.snaps[b] = full[lo - base:b - base].clone()
        if n >= _WIN or self.tail is None:
            self.tail = kv[-_WIN:].clone()
        else:
            if full is None:
                full = torch.cat([self.tail, kv], 0)
            self.tail = full[-_WIN:].clone()
        self.next_pos = start + n
        self.rows += n

    def export(self, out, T):
        for b, t in self.snaps.items():
            out[f"kvwin_{b}"] = t
        if self.tail is not None:
            out["kvwin_end"] = self.tail


class _PoolState:
    """Pooled accumulator + pool_layer carry + raw-row tail for the prev/buf exports."""

    def __init__(self, kind, ratio, head_dim, ape, norm_w, eps, rope):
        self.kind = kind                      # "main" | "idx"
        self.ratio = ratio
        self.head_dim = head_dim
        self.coff = 2 if ratio == 4 else 1
        self.out_dim = self.coff * head_dim
        self.ape, self.norm_w, self.eps, self.rope = ape, norm_w, eps, rope
        self.pooled = []                      # list of [p_i, head_dim]
        self.carry = None
        self.tail = None                      # [<=_TAIL_RAW, 2*out_dim] raw kv_score rows
        self.prev = {}                        # b -> raw rows [4, 2*out_dim]  (ratio 4 only)
        self.next_pos = 0
        self.rows = 0
        self.gaps = []

    def feed(self, x, start):
        import torch
        n = x.shape[0]
        if start != self.next_pos:
            self.gaps.append((self.next_pos, start))
        full = base = None   # lazily built cat(tail, x)
        if self.ratio == 4:
            for b in _boundaries(start, start + n):
                if b - 4 >= start:
                    self.prev[b] = x[b - 4 - start:b - start].clone()
                else:
                    if full is None:
                        full = x if self.tail is None else torch.cat([self.tail, x], 0)
                        base = start - (0 if self.tail is None else self.tail.shape[0])
                    if b - 4 >= base:
                        self.prev[b] = full[b - 4 - base:b - base].clone()
        pooled_new, self.carry = _pool().pool_layer(
            x, self.ratio, self.head_dim, self.ape, self.norm_w, self.eps, self.rope, start, self.carry)
        if pooled_new is not None and pooled_new.numel():
            self.pooled.append(pooled_new)
        if n >= _TAIL_RAW or self.tail is None:
            self.tail = x[-_TAIL_RAW:].clone()
        else:
            if full is None:
                full = torch.cat([self.tail, x], 0)
            self.tail = full[-_TAIL_RAW:].clone()
        self.next_pos = start + n
        self.rows += n

    def export(self, out, T):
        import torch
        pre = "idx_" if self.kind == "idx" else ""
        if self.pooled:
            out[f"{pre}pooled"] = torch.cat(self.pooled, 0)
        rem = T % self.ratio
        if self.tail is not None:
            tl = self.tail.shape[0]
            if rem:
                buf = self.tail[tl - rem:]
                out[f"{pre}buf_kv"] = buf[:, :self.out_dim]
                out[f"{pre}buf_gate"] = buf[:, self.out_dim:]
            if self.ratio == 4:
                # prev window at the end = the last COMPLETE window [T-rem-4, T-rem)  (== carry prev_*)
                lo, hi = tl - rem - 4, tl - rem
                if lo >= 0:
                    self.prev["end"] = self.tail[lo:hi]
        if self.ratio == 4:
            for b, rows in self.prev.items():
                out[f"{pre}prev_kv_{b}"] = rows[:, :self.out_dim]
                out[f"{pre}prev_gate_{b}"] = rows[:, self.out_dim:]


class _Req:
    def __init__(self, stamp, root):
        self.stamp = stamp
        self.dir = os.path.join(root, stamp)
        os.makedirs(self.dir, exist_ok=True)
        self.kv = {}        # li -> _KVState
        self.main = {}      # li -> _PoolState
        self.idx = {}       # li -> _PoolState
        self.first_t = time.time()
        self.last_t = self.first_t
        self.calls = 0
        self.partial_start = None
        with open(os.path.join(self.dir, "manifest.json"), "w") as f:
            json.dump({"version": _VERSION, "stamp": stamp, "T": None, "started_at": self.first_t}, f)


# ----------------------------------------------------------------------------- the capture
class _Capture:
    _FLUSH = object()

    def __init__(self, root=_DIR, idle_s=_IDLE_S, weights=None, start_threads=True):
        self.root = root
        self.idle_s = idle_s
        self.weights = weights or _Weights()
        self.q = queue.Queue()
        self.rank = None
        self.stream = None
        self.req = None
        self.lock = threading.Lock()
        self.rope_cache = {}
        self.done_stamps = []
        self.errors = 0
        self.projector = None       # "pd_pool_torch.project_layer" | "hook"
        self.rope_source = None
        self.enqueue_s = 0.0        # forward-thread Python time inside record()
        self.pool_s = 0.0           # worker Python time per item (projection + pooling + bookkeeping enqueue)
        self.project_s = 0.0
        self.host_syncs = 0
        self.items = 0
        self.chunks = 0
        self.chunk = None           # current chunk descriptor (positions clone shared by all layers)
        self.last_ev = None         # last CUDA event recorded on the compute stream (9/6 mid-flush guard)
        self.opcount = None         # sampled aten-op histogram of one worker item
        self.dead = None
        if start_threads:
            threading.Thread(target=self._worker, daemon=True, name="pd_capture_v3_worker").start()
            threading.Thread(target=self._watch, daemon=True, name="pd_capture_v3_watch").start()

    # -- rank / stream / rope -----------------------------------------------------------------
    def is_rank0(self):
        if self.rank is None:
            try:
                from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
                self.rank = get_tensor_model_parallel_rank()
            except Exception as e:
                _log(f"rank query failed ({e!r}); treating as rank 0")
                return True
        return self.rank == 0

    def _side(self, device):
        import torch
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=device)
        return self.stream

    def _rope(self, ratio):
        """Compressor RoPE (freq_scale = ratio). Preferred: AGENT A's pd_pool_torch.make_ropes(weights json
        meta) -> "comp4" / "comp128" (the indexer's "idx" == comp4: same theta/yarn/freq_scale 4). Fallback
        (no make_ropes or no rope fields in the sidecar): DSv4Rope from rope_config(). The SWA kv rows are
        shipped PRE-RoPE, so the layer-0/1 (theta 10000, no yarn) vs >=2 (160000 + yarn) SWA distinction is
        the Mac assembler's job, not this hook's."""
        if ratio not in self.rope_cache:
            P = _pool()
            m = self.weights.meta if self.weights.cpu is not None else {}
            if hasattr(P, "make_ropes") and "compress_rope_theta" in m and "rope_scaling" in m:
                ropes = P.make_ropes(m)
                self.rope_cache[4] = ropes["comp4"]
                self.rope_cache[128] = ropes["comp128"]
                self.rope_source = "pd_pool_torch.make_ropes(weights.json)"
            else:
                cfg = self.rope_config()
                self.rope_cache[ratio] = P.DSv4Rope(cfg["dims"], cfg["base"], cfg["yarn"], cfg["max_pos"], ratio)
                self.rope_source = "DSv4Rope(" + cfg["source"] + ")"
        return self.rope_cache[ratio]

    def rope_config(self):
        P = _pool()
        m = self.weights.meta if self.weights.cpu is not None else {}
        src = "weights.json" if "compress_rope_theta" in m else "pd_pool_torch constants"
        return {"dims": int(m.get("qk_rope_head_dim", P.QK_ROPE_HEAD_DIM)),
                "base": float(m.get("compress_rope_theta", P.COMPRESS_ROPE_THETA)),
                "yarn": m.get("rope_scaling", P.ROPE_SCALING),
                "max_pos": int(m.get("max_position_embeddings", P.MAX_POSITION_EMBEDDINGS)),
                "freq_scale": "ratio", "source": src}

    def _project(self, hidden, W):
        P = _pool()
        if self.projector is None:
            self.projector = "pd_pool_torch.project_layer" if hasattr(P, "project_layer") else "hook"
            _log(f"projection code path: {self.projector}")
        if self.projector == "hook":
            return project_layer_torch(hidden, W)
        return P.project_layer(hidden, W)

    # -- forward-thread entry point ------------------------------------------------------------
    class _Chunk:
        __slots__ = ("posd", "start", "n", "items")

        def __init__(self, posd):
            self.posd, self.start, self.n, self.items = posd, None, None, 0

    def record(self, layer_name, hidden_states, positions):
        """Forward thread: the absolute minimum. One positions clone per CHUNK (at layer 0), then per
        layer: event on the compute stream + record_stream + queue.put. No projections, no host waits."""
        import torch
        if self.dead:
            return
        if not layer_name or ".mtp" in layer_name:
            return
        m = _LAYER_RE.search(layer_name)
        if m is None:
            return
        li = int(m.group(1))
        if torch.cuda.is_current_stream_capturing() or positions.numel() == 0:
            return
        t0 = time.time()
        dev = hidden_states.device
        s = self._side(dev)
        cur = torch.cuda.current_stream(dev)
        if li == 0 or self.chunk is None or self.chunk.n != positions.shape[0]:
            # positions is a view of vLLM's persistent input buffer (overwritten next step): clone it once
            # per chunk on the compute stream; every layer of this forward step shares the same tensor.
            posd = positions.detach().clone()
            posd.record_stream(s)
            self.chunk = _Capture._Chunk(posd)
            self.chunk.n = int(positions.shape[0])
            self.chunks += 1
        ch = self.chunk
        ch.items += 1
        ev = torch.cuda.Event()
        ev.record(cur)                         # "hidden_states (and the positions clone) are complete"
        self.last_ev = ev                      # 9/6: watcher consults this — see _watch
        hidden_states.record_stream(s)         # allocator must not recycle it before the side stream is done
        self.q.put((li, hidden_states.detach(), ch, ev))
        self.enqueue_s += time.time() - t0

    # -- worker ------------------------------------------------------------------------------
    def ingest(self, li, ratio, kv_pre, kv_score, idx, start):
        """Bookkeeping for one layer-chunk (worker thread, side stream current). Public for the selftest."""
        with self.lock:
            if self.req is not None and li == 0 and self.req.calls > 0:
                st = self.req.kv.get(0)
                if start == 0 or (st is not None and start != st.next_pos):
                    self._finish("new-request")
            if self.req is None:
                t0 = time.time()
                self.req = _Req(time.strftime("%Y%m%d-%H%M%S") + f"-{int((t0 % 1) * 1000):03d}", self.root)
                if start != 0:
                    self.req.partial_start = start
                    _log(f"WARNING request starts at position {start} (prefix hit?) — capture will be partial")
            r = self.req
            r.kv.setdefault(li, _KVState()).feed(kv_pre, start)
            if kv_score is not None:
                if li not in r.main:
                    W = self.weights.layer(li, kv_score.device)
                    r.main[li] = _PoolState("main", ratio, W["comp_norm"].shape[0], W["comp_ape"],
                                            W["comp_norm"], W["eps"], self._rope(ratio))
                r.main[li].feed(kv_score, start)
            if idx is not None:
                if li not in r.idx:
                    W = self.weights.layer(li, idx.device)
                    r.idx[li] = _PoolState("idx", 4, W["idx_norm"].shape[0], W["idx_ape"], W["idx_norm"],
                                           W["eps"], self._rope(4))
                r.idx[li].feed(idx, start)
            r.calls += 1
            r.last_t = time.time()

    def _process(self, li, hidden, ch, ev):
        """Worker thread. Side stream current. Device-side wait on the compute-stream event, then the
        fused projection (ONE call per layer per chunk) + bookkeeping — no host sync here."""
        import torch
        s = self.stream
        s.wait_event(ev)
        dev = hidden.device
        W = self.weights.layer(li, dev)            # H2D once per layer (async from pinned)
        h = hidden if hidden.dtype == torch.bfloat16 else hidden.to(torch.bfloat16)
        t0 = time.time()
        if _PIECE > 0 and h.shape[0] > _PIECE:
            kv_parts, ks_parts, ix_parts = [], [], []
            for a in range(0, h.shape[0], _PIECE):
                kv_pre, kv_score, idx = self._project(h[a:a + _PIECE], W)
                kv_parts.append(kv_pre)
                if kv_score is not None:
                    ks_parts.append(kv_score)
                if idx is not None:
                    ix_parts.append(idx)
            kv_pre = torch.cat(kv_parts, 0)
            kv_score = torch.cat(ks_parts, 0) if ks_parts else None
            idx = torch.cat(ix_parts, 0) if ix_parts else None
        else:
            kv_pre, kv_score, idx = self._project(h, W)
        self.project_s += time.time() - t0
        self.ingest(li, W["ratio"], kv_pre, kv_score, idx, ch.start)

    def _resolve_chunk(self, ch, ev):
        """The one host sync per CHUNK: learn start/n from the positions clone (first item of the chunk)."""
        if ch.start is None:
            ev.synchronize()
            self.host_syncs += 1
            posh = ch.posd.to("cpu")
            n = int(posh.numel())
            start = int(posh[0])
            if n and int(posh[-1]) - start + 1 != n:
                _log(f"WARNING non-contiguous positions {start}..{int(posh[-1])} n={n}")
            ch.start, ch.n = start, n
            ch.posd = None
        return ch.start

    def _worker(self):
        import torch
        while True:
            item = self.q.get()
            try:
                if item is _Capture._FLUSH:
                    with self.lock:
                        if self.req is not None:
                            self._finish("idle")
                    continue
                li, hidden, ch, ev = item
                self._resolve_chunk(ch, ev)
                t0 = time.time()
                self.items += 1
                with torch.cuda.stream(self.stream):
                    if _OPCOUNT and self.opcount is None and self.items > 43 and self.weights.ratio(li) == 4:
                        self._process_counted(li, hidden, ch, ev)
                    else:
                        self._process(li, hidden, ch, ev)
                self.pool_s += time.time() - t0
            except Exception as e:
                self.errors += 1
                _log(f"worker error (ignored): {e!r}")
            finally:
                self.q.task_done()

    def _process_counted(self, li, hidden, ch, ev):
        """Run ONE item under a TorchDispatchMode that counts aten ops = a cheap upper-bound proxy for
        kernel launches (each aten op launches >=0 kernels). Sampled once per process (ratio-4 layer,
        after the first chunk, so it includes projection + main pool + indexer pool + bookkeeping)."""
        try:
            from torch.utils._python_dispatch import TorchDispatchMode
        except Exception:
            self.opcount = {"error": "TorchDispatchMode unavailable"}
            return self._process(li, hidden, ch, ev)
        counts = {}

        class _Count(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                name = str(func.overloadpacket if hasattr(func, "overloadpacket") else func)
                counts[name] = counts.get(name, 0) + 1
                return func(*args, **(kwargs or {}))

        with _Count():
            self._process(li, hidden, ch, ev)
        total = sum(counts.values())
        top = dict(sorted(counts.items(), key=lambda kv: -kv[1])[:12])
        self.opcount = {"layer": li, "rows": int(hidden.shape[0]), "aten_ops_total": total, "top": top}
        _log(f"aten-op sample (layer {li}, {hidden.shape[0]} rows): {total} ops; top {top}")

    def _watch(self):
        while True:
            time.sleep(0.1)
            try:
                with self.lock:
                    r = self.req
                    if r is None:
                        continue
                    # 9/6 MID-FLUSH GUARD (FINDING-bench4-cold-fallback.md root cause A): with chunked prefill
                    # (--max-num-batched-tokens 8192) the forward thread enqueues each chunk in a burst, then the GPU
                    # grinds ~4 s per chunk. q.empty() + 2 s idle could fire BETWEEN chunks, flushing a request that
                    # is still running (manifest T = 16384 of 18424 etc.). The last compute-stream event is pending
                    # exactly while the GPU is still inside the request — so an unfinished event means: do not flush.
                    gpu_busy = self.last_ev is not None and not self.last_ev.query()
                    # chunk-alignment guard: a complete forward step ingests one item per layer, so r.calls is a
                    # multiple of the layer count except MID-BURST. seed 705 (9/6): the sentinel landed inside the
                    # final chunk's enqueue burst and split it across two captures (41 orphaned items ->
                    # AssertionError('start_pos 106496 != processed 0'), tail boundary never built).
                    # NOTE: r must be None-checked BEFORE any r.* access — the first cut of this guard read
                    # r.kv unconditionally and killed the watcher thread on every post-flush tick (no idle
                    # flushes at all; every bridge declined "no captures at all").
                    nl = (max(r.kv) + 1) if r.kv else 0
                    aligned = (nl == 0 or r.calls % nl == 0)
                    fire = (self.q.empty() and not gpu_busy and aligned
                            and time.time() - r.last_t >= self.idle_s)
                if fire:
                    self.q.put(_Capture._FLUSH)
                    time.sleep(self.idle_s)  # one sentinel per idle period
            except Exception as e:
                # the watcher must NEVER die silently: no idle flushes = no captures = every bridge declines
                _log(f"watcher error (ignored): {e!r}")

    def flush(self, reason="manual"):
        """Synchronous flush (tests). Waits until every queued item has been ingested."""
        self.q.join()
        with self.lock:
            if self.req is not None:
                return self._finish(reason)
        return None

    # -- writer (lock held) ------------------------------------------------------------------
    def _finish(self, reason):
        import torch
        from safetensors.torch import save_file
        r, self.req = self.req, None
        t0 = time.time()
        T = max([s.next_pos for s in r.kv.values()] or [0])
        if T < _MIN_T:
            shutil.rmtree(r.dir, ignore_errors=True)
            _log(f"discarded {r.stamp}: T={T} < PD_CAPTURE_MIN_T={_MIN_T} ({reason})")
            return None
        layers = sorted(set(r.kv) | set(r.main) | set(r.idx))
        num_layers = (max(layers) + 1) if layers else 0
        bounds = _boundaries(0, T)
        ratios = [None] * num_layers
        counts = {}
        nbytes = 0
        if self.stream is not None:
            self.stream.synchronize()
        for li in layers:
            out = {}
            if li in r.kv:
                r.kv[li].export(out, T)
            if li in r.main:
                r.main[li].export(out, T)
                ratios[li] = r.main[li].ratio
            elif li in r.kv:
                ratios[li] = 0
            if li in r.idx:
                r.idx[li].export(out, T)
            host = {}
            for k, v in out.items():
                v = v.detach()
                if v.dtype != torch.bfloat16:
                    v = v.to(torch.bfloat16)
                host[k] = v.contiguous().cpu()
                nbytes += host[k].numel() * 2
            path = os.path.join(r.dir, f"layer_{li:02d}.safetensors")
            save_file(host, path + ".tmp")
            os.replace(path + ".tmp", path)
            counts[f"layer_{li:02d}"] = {"kv": r.kv[li].rows if li in r.kv else 0,
                                        "main": r.main[li].rows if li in r.main else 0,
                                        "idx": r.idx[li].rows if li in r.idx else 0}
        gaps = {f"layer_{li:02d}": st.gaps for li, st in r.kv.items() if st.gaps}
        meta = {"version": _VERSION, "T": T, "num_layers": num_layers, "ratios": ratios, "boundaries": bounds,
                "end": T, "stamp": r.stamp, "block": _BLOCK, "window": _WIN, "layers_written": len(layers),
                "tokens_per_layer": counts, "calls": r.calls, "capture_span_s": round(r.last_t - r.first_t, 3),
                "write_s": round(time.time() - t0, 3), "bytes": nbytes, "flush_reason": reason,
                "projector": self.projector, "weights": self.weights.path, "eps": self.weights.eps,
                "weights_model": (self.weights.meta.get("model") or self.weights.meta.get("note")),
                "rope": self.rope_config(), "rope_source": self.rope_source, "enqueue_s_total": round(self.enqueue_s, 3),
                "pool_s_total": round(self.pool_s, 3), "project_s_total": round(self.project_s, 3),
                "worker_items": self.items, "chunks": self.chunks, "host_syncs": self.host_syncs,
                "launches_per_call": (self.opcount or {}).get("aten_ops_total"),
                "launches_note": "sampled aten-op count of ONE ratio-4 layer-chunk (upper-bound proxy for kernel launches; not a hardware count)",
                "aten_ops_sample": self.opcount, "piece": _PIECE, "partial_start": r.partial_start,
                "position_gaps": gaps, "worker_errors": self.errors}
        with open(os.path.join(r.dir, "manifest.json"), "w") as f:
            json.dump(meta, f, indent=1, default=str)
        with open(os.path.join(r.dir, "DONE"), "w") as f:
            f.write(reason)
        self.done_stamps.append(r.stamp)
        _log(f"finished {r.stamp}: {len(layers)} layers, T={T}, boundaries={len(bounds)}, calls={r.calls}, "
             f"span={meta['capture_span_s']}s, write={meta['write_s']}s, {nbytes/1e6:.1f} MB ({reason}); "
             f"enqueue={self.enqueue_s:.2f}s worker={self.pool_s:.2f}s (project {self.project_s:.2f}s) "
             f"items={self.items} chunks={self.chunks} host_syncs={self.host_syncs}"
             + (f" WARNING gaps={gaps}" if gaps else "") + (f" PARTIAL start={r.partial_start}" if r.partial_start else ""))
        return r.dir


_CAP = None


def _cap():
    global _CAP
    if _CAP is None:
        _CAP = _Capture()
    return _CAP


# ----------------------------------------------------------------------------- patch
def _patch_attention(mod):
    cls = getattr(mod, "DeepseekV4MultiHeadLatentAttentionWrapper", None)
    if cls is None:
        _log("attention wrapper class not found; no patch applied")
        return
    if getattr(cls, "_pd_capture_v3", False):
        return
    orig = cls.attention_impl

    def attention_impl(self, hidden_states, positions, out):
        try:
            from vllm.forward_context import get_forward_context
            if isinstance(get_forward_context().attn_metadata, dict):
                cap = _cap()
                if cap.is_rank0():
                    cap.record(getattr(self, "layer_name", getattr(self, "prefix", "")), hidden_states, positions)
        except Exception as e:
            cap = _cap()
            cap.errors += 1
            if cap.errors <= 5 or cap.errors % 100 == 0:
                _log(f"capture error #{cap.errors} (ignored): {e!r}")
            fatal = ("pd_pool_torch" in repr(e)) or ("cannot load" in repr(e)) or ("no projection weights" in repr(e))
            if cap.errors >= 5 and cap.dead is None and fatal:
                cap.dead = repr(e)
                _log(f"DISARMED for this process after {cap.errors} errors: {cap.dead}")
        return orig(self, hidden_states, positions, out)

    attention_impl._pd_capture_orig = orig
    cls.attention_impl = attention_impl
    cls._pd_capture_v3 = True
    _log(f"v3 patched {cls.__module__}.{cls.__name__}.attention_impl (hidden_states -> Mac-weight projections + pooling)")
    if _PRELOAD:
        threading.Thread(target=_cap().weights.load_cpu, daemon=True, name="pd_capture_v3_preload").start()


class _PostImportFinder(importlib.abc.MetaPathFinder):
    _busy = False

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET_ATTN or _PostImportFinder._busy:
            return None
        _PostImportFinder._busy = True
        try:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            _PostImportFinder._busy = False
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            _patch_attention(module)

        loader.exec_module = exec_module  # type: ignore[attr-defined]
        return spec


if _DIR and os.environ.get("PD_CAPTURE_V3_NOARM") != "1":
    os.makedirs(_DIR, exist_ok=True)
    if _TARGET_ATTN in sys.modules:
        _patch_attention(sys.modules[_TARGET_ATTN])
    else:
        sys.meta_path.insert(0, _PostImportFinder())
    _log(f"v3.1 armed (pooled capture with Mac weights, launch-lean; dir={_DIR}, idle {_IDLE_S}s, block {_BLOCK}, "
         f"weights={_WEIGHTS}, piece={_PIECE or 'none'})")
