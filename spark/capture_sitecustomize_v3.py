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

DATA PATH: the forward thread enqueues ONLY async GPU work on one side stream `s` (s.wait_stream(cur),
hidden.record_stream(s); NO cur.wait_event — the compute stream never waits on us): the three
projections in <=4096-row pieces, a pinned copy of `positions`, an event. A single worker thread
consumes the queue in order, waits on the event (host wait on the worker thread only), and runs
pool_layer + window/prev bookkeeping on the same side stream. On idle (PD_CAPTURE_IDLE_S, default
2 s) or a new request at position 0 it writes PD_CAPTURE_DIR/<stamp>/layer_XX.safetensors +
manifest.json + DONE (DONE last; manifest.json is also written at request start with "T": null).

WEIGHTS: PD_PROJ_WEIGHTS (default /pd_v3/dv4_proj_weights.safetensors; sidecar .json with
eps/ratios) — keys layer_{i}.wkv.weight [512,4096] · layer_{i}.kv_norm.weight [512] ·
layer_{i}.comp.{wkv,wgate}.weight [out_dim,4096] · layer_{i}.comp.ape [ratio,out_dim] f32 ·
layer_{i}.comp.norm.weight [512] · layer_{i}.idx.{wkv,wgate}.weight [256,4096] · layer_{i}.idx.ape
[4,256] f32 · layer_{i}.idx.norm.weight [128]. Preloaded to pinned host RAM in a background thread
when the attention module is imported (PD_PROJ_PRELOAD=0 disables), moved to the GPU on the side
stream at the first captured chunk. Missing file => the hook logs LOUDLY and captures nothing.

Env: PD_CAPTURE_DIR (arms the hook) · PD_CAPTURE_IDLE_S=2.0 · PD_CAPTURE_BLOCK=2048 ·
     PD_CAPTURE_MIN_T=64 (discard shorter captures) · PD_CAPTURE_PIECE=4096 (projection piece rows) ·
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
_PIECE = int(os.environ.get("PD_CAPTURE_PIECE", "4096"))
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
        full = kv if self.tail is None else torch.cat([self.tail, kv], 0)
        base = start - (0 if self.tail is None else self.tail.shape[0])
        for b in _boundaries(start, start + n):
            lo = max(b - _WIN, base)
            self.snaps[b] = full[lo - base:b - base].clone()
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
        full = x if self.tail is None else torch.cat([self.tail, x], 0)
        base = start - (0 if self.tail is None else self.tail.shape[0])
        if self.ratio == 4:
            for b in _boundaries(start, start + n):
                if b - 4 >= base:
                    self.prev[b] = full[b - 4 - base:b - base].clone()
        pooled_new, self.carry = _pool().pool_layer(
            x, self.ratio, self.head_dim, self.ape, self.norm_w, self.eps, self.rope, start, self.carry)
        if pooled_new is not None and pooled_new.numel():
            self.pooled.append(pooled_new)
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
        self.enqueue_s = 0.0
        self.pool_s = 0.0
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
    def record(self, layer_name, hidden_states, positions):
        """Forward thread: enqueue the projections on the side stream; never a host sync."""
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
        # positions is a view of vLLM's persistent input buffer (overwritten next step): copy it on the
        # compute stream NOW (async 8-byte-per-token D2H, no host wait) rather than later on the side stream.
        # 9/6 (operator): a fresh pinned host allocation per call (cudaHostAlloc) is synchronous and slow — it was ~9 s of forward-thread
        # time at 19.8K tokens. Clone positions on the DEVICE instead (async, tiny); the worker moves it to the host after the event.
        posh = positions.detach().clone()
        s.wait_stream(cur)                     # side stream sees hidden_states + the positions clone complete
        hidden_states.record_stream(s)         # allocator must not recycle it before the side stream is done
        ratio = self.weights.ratio(li)
        with torch.cuda.stream(s):
            W = self.weights.layer(li, dev)           # H2D once per layer (async from pinned)
            h = hidden_states.detach()
            if h.dtype != torch.bfloat16:
                h = h.to(torch.bfloat16)
            kv_parts, ks_parts, ix_parts = [], [], []
            for a in range(0, h.shape[0], _PIECE):
                kv_pre, kv_score, idx = self._project(h[a:a + _PIECE], W)
                kv_parts.append(kv_pre)
                if kv_score is not None:
                    ks_parts.append(kv_score)
                if idx is not None:
                    ix_parts.append(idx)
            kv_pre = torch.cat(kv_parts, 0) if len(kv_parts) > 1 else kv_parts[0]
            kv_score = (torch.cat(ks_parts, 0) if len(ks_parts) > 1 else ks_parts[0]) if ks_parts else None
            idx = (torch.cat(ix_parts, 0) if len(ix_parts) > 1 else ix_parts[0]) if ix_parts else None
            ev = torch.cuda.Event()
            ev.record(s)
        self.enqueue_s += time.time() - t0
        self.q.put((li, ratio, kv_pre, kv_score, idx, posh, ev))

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
                li, ratio, kv_pre, kv_score, idx, posh, ev = item
                ev.synchronize()   # worker thread only; needed to read positions on the host
                posh = posh.to("cpu")
                n = int(posh.numel())
                start = int(posh[0])
                if n and int(posh[-1]) - start + 1 != n:
                    _log(f"WARNING layer {li}: non-contiguous positions {start}..{int(posh[-1])} n={n}")
                t0 = time.time()
                with torch.cuda.stream(self.stream):
                    self.ingest(li, ratio, kv_pre, kv_score, idx, start)
                self.pool_s += time.time() - t0
            except Exception as e:
                self.errors += 1
                _log(f"worker error (ignored): {e!r}")
            finally:
                self.q.task_done()

    def _watch(self):
        while True:
            time.sleep(0.1)
            with self.lock:
                r = self.req
                fire = r is not None and self.q.empty() and time.time() - r.last_t >= self.idle_s
            if fire:
                self.q.put(_Capture._FLUSH)
                time.sleep(self.idle_s)  # one sentinel per idle period

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
                "pool_s_total": round(self.pool_s, 3), "partial_start": r.partial_start,
                "position_gaps": gaps, "worker_errors": self.errors}
        with open(os.path.join(r.dir, "manifest.json"), "w") as f:
            json.dump(meta, f, indent=1, default=str)
        with open(os.path.join(r.dir, "DONE"), "w") as f:
            f.write(reason)
        self.done_stamps.append(r.stamp)
        _log(f"finished {r.stamp}: {len(layers)} layers, T={T}, boundaries={len(bounds)}, calls={r.calls}, "
             f"span={meta['capture_span_s']}s, write={meta['write_s']}s, {nbytes/1e6:.1f} MB ({reason})"
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
    _log(f"v3 armed (pooled capture with Mac weights; dir={_DIR}, idle {_IDLE_S}s, block {_BLOCK}, "
         f"weights={_WEIGHTS}, piece={_PIECE})")
