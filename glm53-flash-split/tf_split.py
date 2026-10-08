"""TensorFold split prefill for GLM-5.3-Flash (pd-bridge, 2026-10-03).

A DGX Spark pair (vLLM TP2, MiaAI-Lab's EXL3 recipe + Ash Hart's Glm53HandoffConnector) prefills a long prompt's stable
prefix; the state crosses the MCDMA door into the Studio's handoff daemon; this module turns it into TensorFold's own
per-layer caches (KDACache / MLACache) and stores it in TensorFold's prompt store as a prefix. The prompt then resumes
there and TensorFold prefills only the tail and decodes everything (MTP drafting, its exact decode kernels).

Transport, blob parsing and layout assembly are Ash Hart's (``glm53_split``, from his MCDMA GLM-5.3-Flash handoff kit,
imported unchanged — not vendored here); only the target cache and the hook are ours.

Hook: ``PromptFill._start_fill`` (tensorfold.server.prompt_fill). Before a prompt's first chunk, when its stable prefix
(the history boundary) is long and not already stored, the split runs, the prefix is inserted into ``self.checkpoints``
and its length is added to the prompt's chunk starts (``LaneEngine.prompt_chunks``), so the store's own lookup hits it.

Config ``~/.glm53/tf_split.json`` (hot-reloaded): mode off|on|compare, url, model, peers (TP rank order), socket,
remote_dir, min_tokens, min_delta, max_handoff_tokens, big_handoff_tokens, timeout_s, conv_columns, frozen_marker_ids,
peer_capture_marker, log. Any failure falls back to TensorFold's own prefill (logged), never fails the request.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

CONFIG_PATH = os.path.expanduser(os.environ.get("TF_SPLIT_CONFIG", "~/.glm53/tf_split.json"))
SPLIT_SRC = os.path.expanduser(os.environ.get("TF_SPLIT_ASH_SRC", "~/glm53_handoff/studio"))   # dir holding glm53_split.py
_DEFAULTS = {"mode": "off", "url": "http://PREFILL_HEAD:8888/v1", "model": "glm-5.3-flash",
             "peers": ["PREFILL_RANK0", "PREFILL_RANK1"], "socket": "/tmp/handoffd.sock", "remote_dir": "/dev/shm/glm53-handoff",
             "min_tokens": 4096, "min_delta": 4096, "max_handoff_tokens": 120000, "timeout_s": 900.0,
             "conv_columns": "first", "recurrent_transpose": False, "fp8_scale": 1.0, "stream": False,
             "big_handoff_tokens": 40000, "frozen_marker_ids": [], "peer_capture_marker": "/tmp/drift_capture_active",
             "log": "~/.glm53/tf_split.log"}
import tf_split_state as _state             # survives hot reloads of this module
for _k, _v in (("PENDING", {}), ("BUDGET", [0]), ("BUDGET_COND", threading.Condition())):
    if not hasattr(_state, _k):                  # an older state module still loaded in a running server
        setattr(_state, _k, _v)
STATS: dict[str, Any] = _state.STATS
_LOCK = _state.LOCK
_EXTRA_STARTS: dict[int, str] = _state.EXTRA_STARTS   # boundary -> sha1 of prompt[:boundary]


def _say(text: str) -> None:
    sys.stderr.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [tf-split] {text}\n")
    sys.stderr.flush()


class _Config:
    def __init__(self) -> None:
        self.cfg, self._stamp = dict(_DEFAULTS), ()

    def get(self) -> dict:
        try:
            st = os.stat(CONFIG_PATH)
            stamp: Any = (st.st_mtime_ns, st.st_size)
        except OSError:
            stamp = None
        if stamp != self._stamp:
            self._stamp = stamp
            cfg = dict(_DEFAULTS)
            if stamp is not None:
                try:
                    cfg.update(json.loads(Path(CONFIG_PATH).read_text()))
                except Exception as exc:  # noqa: BLE001
                    _say(f"config ignored: {exc!r}")
                    cfg = dict(self.cfg)
            self.cfg = cfg
            _say(f"config {cfg}")
        return self.cfg


CONFIG = _Config()


_E4M3_LUT = None


def _e4m3_lut(bits: np.ndarray) -> np.ndarray:
    """fp8 e4m3 -> fp32 through a 256-entry table: the same values as Ash's elementwise decode, ~10x faster."""
    global _E4M3_LUT
    if _E4M3_LUT is None:
        import glm53_split as _g  # noqa: PLC0415
        _E4M3_LUT = _g._e4m3_to_float.__wrapped__(np.arange(256, dtype=np.uint8)) if hasattr(_g._e4m3_to_float, "__wrapped__") \
            else _g._ORIG_E4M3(np.arange(256, dtype=np.uint8))
    return _E4M3_LUT[np.asarray(bits, dtype=np.uint8)]


def ash():
    """Ash Hart's glm53_split module (transport, blobs, assembly) — imported from the Studio's checkout of his kit."""

    if SPLIT_SRC not in sys.path:
        sys.path.insert(0, SPLIT_SRC)
    import glm53_split  # noqa: PLC0415
    if not hasattr(glm53_split, "_ORIG_E4M3"):
        glm53_split._ORIG_E4M3 = glm53_split._e4m3_to_float
        glm53_split._e4m3_to_float = _e4m3_lut
    return glm53_split


def _sha(tokens) -> str:
    return hashlib.sha1(np.asarray(tokens, dtype=np.int64).tobytes()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# assembled vLLM state -> TensorFold's per-layer caches
# ---------------------------------------------------------------------------------------------------------------
def build_cache(assembled, T: int, runtime) -> list[Any]:
    from tensorfold.families.glm5_next import config as C
    from tensorfold.families.glm5_next.caches import KDACache, MLACache

    act = C.act()
    model = runtime.model
    caches = model.make_cache()
    step = MLACache.step
    cap = -(-T // step) * step
    for i, layer in enumerate(model.layers):
        c = caches[i]
        if layer.is_linear:
            st = assembled.gdn.get(i)
            if st is None or len(st) != 2:
                raise ValueError(f"layer {i}: KDA state missing")
            conv, rec = st
            c.conv = mx.contiguous(conv.reshape(conv.shape[-2], conv.shape[-1]).astype(act))      # [taps-1, 3 width]
            c.ssm = mx.contiguous(rec.astype(mx.float32))                                          # [1, H, d, d]
            c.offset = T
            assert isinstance(c, KDACache)
        else:
            fa, st = assembled.fa.get(i), assembled.gdn.get(i)
            if fa is None or st is None or st[2] is None:
                raise ValueError(f"layer {i}: sparse-layer state missing")
            lat = fa[0].reshape(T, -1)                                     # [T, 512]
            pooled = st[2].reshape(-1, st[2].shape[-1])                   # [T//4, 128]
            if int(pooled.shape[0]) != T // 4:
                raise ValueError(f"layer {i}: pooled rows {pooled.shape[0]} != {T // 4}")
            keys = mx.zeros((cap, int(lat.shape[1])), dtype=act)
            keys[:T] = lat.astype(act)
            pool = mx.zeros((cap // 4, int(pooled.shape[1])), dtype=act)
            pool[:T // 4] = pooled.astype(act)
            c.keys, c.pool = keys, pool
            # raw indexer keys / gates only feed the pooling of blocks completed later (all rows >= T, T % 4 == 0)
            c.ik = mx.zeros((cap, int(pooled.shape[1])), dtype=act)
            c.ig = mx.zeros((cap, int(pooled.shape[1])), dtype=act)
            c.offset = T
            assert isinstance(c, MLACache)
    if getattr(runtime, "mtp", None) is not None:
        caches.append(runtime.new_mtp_cache())          # the draft head sees the tail it prefills, not the Sparks' rows
    mx.eval(*[a for c in caches for a in getattr(c, "state", [])])
    return caches


def _rel(a: mx.array, b: mx.array) -> float:
    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    return float((mx.sqrt(mx.sum(mx.square(a32 - b32))) / (mx.sqrt(mx.sum(mx.square(b32))) + 1e-12)).item())


def compare(cache: list[Any], tokens: list[int], runtime, next_token: int | None) -> dict:
    """TensorFold's own prefill of the same tokens versus the Sparks' state: per component, and the next token."""

    from tensorfold.engine.lane_engine import LaneEngine

    model = runtime.model
    T = len(tokens)
    local = model.make_cache()
    mx.eval(model.hidden(mx.array([tokens], dtype=mx.uint32), local))
    rep: dict[str, Any] = {"T": T}
    lat, pool, conv, ssm = [], [], [], []
    for i, layer in enumerate(model.layers):
        a, b = cache[i], local[i]
        if layer.is_linear:
            conv.append(_rel(a.conv, b.conv)); ssm.append(_rel(a.ssm, b.ssm))
        else:
            lat.append(_rel(a.keys[:T], b.keys[:T])); pool.append(_rel(a.pool[:T // 4], b.pool[:T // 4]))
    med = lambda xs: None if not xs else {"median": round(float(np.median(xs)), 4), "max": round(float(np.max(xs)), 4)}
    rep.update({"latent": med(lat), "pool": med(pool), "conv": med(conv), "ssm": med(ssm)})
    alt = getattr(compare, "_alt", None)
    if alt is not None:     # Ash's alternative layouts, against TensorFold's own state
        lin = [i for i, l in enumerate(model.layers) if l.is_linear]
        sp = [i for i, l in enumerate(model.layers) if not l.is_linear]
        def m(xs):
            return med([x for x in xs if x is not None])
        rep["alt"] = {
            "conv_last": m([_rel(alt["conv_last"][i].reshape(local[i].conv.shape), local[i].conv) for i in lin if i in alt.get("conv_last", {})]),
            "conv_first": m([_rel(alt["conv_first"][i].reshape(local[i].conv.shape), local[i].conv) for i in lin if i in alt.get("conv_first", {})]),
            "rec_T": m([_rel(alt["rec_T"][i], local[i].ssm) for i in lin if i in alt.get("rec_T", {})]),
            "pool_norot": m([_rel(alt["pooled_norot"][i].reshape(-1, 128), local[i].pool[:T // 4]) for i in sp if i in alt.get("pooled_norot", {})]),
            "pool_inter": m([_rel(alt["pooled_inter"][i].reshape(-1, 128), local[i].pool[:T // 4]) for i in sp if i in alt.get("pooled_inter", {})]),
        }
        i0 = sp[0]
        a, b = cache[i0].keys, local[i0].keys
        rep["latent_detail"] = {"layer": i0, "first128": round(_rel(a[:128], b[:128]), 4),
                                "last128": round(_rel(a[T - 128:T], b[T - 128:T]), 4),
                                "shift+1": round(_rel(a[1:T], b[:T - 1]), 4), "shift-1": round(_rel(a[:T - 1], b[1:T]), 4),
                                "norm_ratio": round(float((mx.linalg.norm(a[:T].astype(mx.float32)) / mx.linalg.norm(b[:T].astype(mx.float32))).item()), 4)}
        lat_by_layer = {i: round(_rel(cache[i].keys[:T], local[i].keys[:T]), 3) for i in sp}
        rep["latent_by_layer"] = lat_by_layer
        ssm_by_layer = {i: round(_rel(cache[i].ssm, local[i].ssm), 3) for i in lin[:6]}
        rep["ssm_first_layers"] = ssm_by_layer
    if next_token is not None:
        copy = LaneEngine.copy_single_cache
        theirs = copy(cache[:len(model.layers)]); ours = copy(local)
        lt = model.head(model.hidden(mx.array([[int(next_token)]], dtype=mx.uint32), theirs))[0, -1].astype(mx.float32)
        lo = model.head(model.hidden(mx.array([[int(next_token)]], dtype=mx.uint32), ours))[0, -1].astype(mx.float32)
        pt, po = lt - mx.logsumexp(lt), lo - mx.logsumexp(lo)
        kl = float(mx.sum(mx.exp(po) * (po - pt)).item())
        tt = sorted(mx.argpartition(-lt, kth=4)[:5].tolist(), key=lambda t: -float(lt[t].item()))
        to = sorted(mx.argpartition(-lo, kth=4)[:5].tolist(), key=lambda t: -float(lo[t].item()))
        rep["next_token"] = {"kl_local_vs_spark": round(kl, 5), "top1_same": tt[0] == to[0],
                             "top5_overlap": len(set(tt) & set(to))}
        # the third reference: the Sparks' OWN next-token distribution (vLLM, same prompt + next token)
        try:
            cfg = CONFIG.get()
            native = ash().vllm_top_logprobs(cfg["url"], cfg["model"], list(tokens) + [int(next_token)], top=20)
            nat = {int(t): float(v) for t, v in native if str(t).lstrip("-").isdigit()}
            def topk_kl(p_log: mx.array, q: dict) -> float:     # KL(p || q) over q's top-20, renormalised both
                ids = list(q)
                pl = np.array([float(p_log[i].item()) for i in ids]); ql = np.array([q[i] for i in ids])
                pl -= np.log(np.exp(pl).sum()); ql -= np.log(np.exp(ql).sum())
                return float((np.exp(pl) * (pl - ql)).sum())
            nt = sorted(nat, key=lambda t: -nat[t])
            rep["native"] = {"spark_native_top5": nt[:5], "split_top5": tt, "local_top5": to,
                             "top1_split_eq_native": tt[0] == nt[0], "top1_local_eq_native": to[0] == nt[0],
                             "kl20_split_vs_native": round(topk_kl(pt, nat), 5),
                             "kl20_local_vs_native": round(topk_kl(po, nat), 5)}
        except Exception as exc:  # noqa: BLE001
            rep["native"] = {"error": repr(exc)}
    return rep


# ---------------------------------------------------------------------------------------------------------------
# the split, in two phases
#   network (the request's own thread, in Scheduler.submit): door lock -> Sparks prefill -> MCDMA pull into the
#     Studio daemon's shared memory. TensorFold's scheduler keeps decoding every other stream meanwhile.
#   build (the scheduler thread, in PromptFill._start_fill): Ash's assembly -> TensorFold caches -> stored prefix,
#     then the door lock is released (the blobs live in the daemon's shm until the next pull).
# ---------------------------------------------------------------------------------------------------------------
DOOR_LOCK = "/tmp/mcdma_door.lock"
PENDING: dict[str, dict] = _state.PENDING if hasattr(_state, "PENDING") else {}


def _door_lock(timeout: float):
    import fcntl
    fd = os.open(DOOR_LOCK, os.O_CREAT | os.O_RDWR, 0o600)
    end = time.time() + timeout
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            if time.time() > end:
                os.close(fd)
                raise RuntimeError("MCDMA door busy (lock held) past the timeout")
            time.sleep(0.05)


def _door_unlock(fd) -> None:
    import fcntl
    if fd is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


_ENGINE: list = [None]


def sched_engine(_store):
    return _ENGINE[0]


def _last_message_start(engine, prompt, stable: int) -> int:
    """The largest message-opener position strictly inside the prompt's history (0: unknown)."""
    try:
        plan_ = getattr(engine, "prefill_plan", None)
        if plan_ is None:
            return 0
        ids = np.fromiter((int(t) for t in prompt), dtype=np.int64, count=len(prompt))
        openers = list(getattr(plan_, "openers", ()) or ())
        pts = [int(p) for p in np.flatnonzero(np.isin(ids, openers))] if openers else []
        pts += [int(p) for p in plan_.points(ids)]
        inner = [p for p in pts if 0 < p < stable - 8]
        return max(inner) if inner else 0
    except Exception as exc:  # noqa: BLE001
        _say(f"message points unavailable: {exc!r}")
        return 0


def plan(store, job, cfg) -> int:
    """The handoff boundary for this job (0: no split)."""

    if str(cfg.get("mode", "off")) not in ("on", "compare") or store is None:
        return 0
    if getattr(job, "vision", None) is not None or getattr(job, "label_ids", None) or getattr(job, "background", False):
        return 0
    prompt = job.prompt_ids
    n = len(prompt)
    hist = int(getattr(job, "history_len", 0) or 0)
    stable = min(hist if hist > 0 else n - 1, n - 1)
    # 2026-10-04: history_len covers the LAST message too, so a stored prefix never matched
    # a NEW question on the same text. Prefer the start of the last message (TensorFold's own message points), when
    # that still hands off enough; the last message itself is prefilled by TensorFold.
    last_msg = _last_message_start(sched_engine(store), prompt, stable)
    have = store.longest(list(prompt))
    if last_msg and last_msg - (last_msg % 4) - have >= int(cfg["min_delta"]):
        stable = last_msg
    boundary = stable - (stable % 4)
    if boundary < int(cfg["min_tokens"]):
        return 0
    cap = int(cfg.get("max_handoff_tokens") or 0)
    if cap and boundary > cap:
        # the pair's KV holds ~1.9x a handoff: hand over the first `cap` tokens, TensorFold prefills the rest
        _say(f"split capped: {boundary} stable tokens > max_handoff_tokens {cap}; handing off the first {cap - cap % 4}")
        boundary = cap - cap % 4
    if boundary - store.longest(list(prompt)) < int(cfg["min_delta"]):
        return 0
    return boundary


def _sweep() -> None:
    now = time.time()
    for key, p in list(PENDING.items()):
        if now - p["at"] > 600:                     # its job never reached the scheduler (cancelled): free the door
            PENDING.pop(key, None)
            _door_unlock(p.get("fd"))
            _say(f"pending {key} expired; door released")


def _budget_take(tokens: int, cap: int, timeout: float) -> bool:
    """Reserve `tokens` of the pair's handoff capacity (the sum of in-flight handoffs stays <= cap)."""
    cond = _state.BUDGET_COND
    end = time.time() + timeout
    with cond:
        while _state.BUDGET[0] + tokens > cap and _state.BUDGET[0] > 0:
            left = end - time.time()
            if left <= 0:
                return False
            cond.wait(min(left, 1.0))
        _state.BUDGET[0] += tokens
        return True


def _budget_give(tokens: int) -> None:
    cond = _state.BUDGET_COND
    with cond:
        _state.BUDGET[0] = max(0, _state.BUDGET[0] - tokens)
        cond.notify_all()


def mark_frozen_end(job, cfg) -> None:
    """Optional: if the prompt contains the token run ``frozen_marker_ids`` (a header your client puts right after a
    long, rarely-changing prefix), ask TensorFold to keep a checkpoint just before it, so the NEXT request on the same
    frozen prefix is warm — not only the third (2026-10-04: without it, the 2nd request on a 700K prefix re-read ~600K
    tokens). Off when ``frozen_marker_ids`` is empty."""
    try:
        mark = np.asarray(cfg.get("frozen_marker_ids") or [], dtype=np.int64)
        k = len(mark)
        ids = np.asarray(job.prompt_ids, dtype=np.int64)
        if k == 0 or len(ids) < 100000:
            return
        cand = np.flatnonzero(ids[: len(ids) - k] == mark[0])
        for c in cand[::-1]:
            if np.array_equal(ids[c:c + k], mark):
                at = int(c) - int(cfg.get("frozen_marker_back", 0))   # back over any separator tokens before the marker
                if at > 0:
                    job.shared_prefix_lens = tuple(sorted(set(tuple(job.shared_prefix_lens or ()) + (at,))))
                    _say(f"frozen prefix ends at {at}: TensorFold keeps a checkpoint there")
                return
    except Exception as exc:  # noqa: BLE001
        _say(f"frozen-end mark failed: {exc!r}")


def network_phase(sched, job) -> None:
    """Sparks prefill (concurrent requests share the pair, within its KV budget) -> door lock -> pull."""
    cfg = CONFIG.get()
    mark_frozen_end(job, cfg)
    _ENGINE[0] = getattr(sched, "engine", None)
    _sweep()
    boundary = plan(getattr(sched, "checkpoints", None), job, cfg)
    if not boundary:
        return
    A = ash()
    tokens = list(job.prompt_ids[:boundary])
    handoff_id = f"tfs-{_sha(tokens)[:10]}-{int(time.time() * 1000) % 10000000}"
    peers = list(cfg["peers"])
    cap = int(cfg.get("max_handoff_tokens") or 120000)
    t0 = time.perf_counter()
    if not _budget_take(boundary, cap, float(cfg["timeout_s"])):
        _say(f"split skipped: the pair's handoff budget stayed full ({_state.BUDGET[0]} in flight)")
        return
    t_budget = time.perf_counter() - t0
    fd = None
    big = boundary >= int(cfg.get("big_handoff_tokens", 40000))
    if big:
        # another consumer capturing on the same pair (we ran Ash Hart's Drift) borrows the pair's KV; a near-capacity
        # handoff beside it can livelock (2026-10-04 00:50) -> take turns through two marker files
        marker = str(cfg.get("peer_capture_marker") or "/tmp/drift_capture_active")
        for _ in range(60):
            try:
                if time.time() - os.stat(marker).st_mtime > 90:
                    break
            except OSError:
                break
            time.sleep(1.0)
        try:
            Path("/tmp/tf_split_inflight").write_text(f"{boundary} {time.time()}")
        except OSError:
            pass
    _say(f"split: {boundary} of {len(job.prompt_ids)} tokens -> Sparks (job {job.job_id}, budget wait {t_budget:.2f}s)")
    try:
        # a livelocked pair (KV past ~1.9x the handoff) never finishes: give up after ~2x the slow-path time, which
        # closes the connection, which makes vLLM abort the request (2026-10-04 00:24: a 120K handoff spun 10+ min)
        budget_s = min(float(cfg["timeout_s"]), 60.0 + boundary / float(cfg.get("min_spark_tok_s", 600)))
        A.vllm_prefill(cfg["url"], cfg["model"], tokens, handoff_id, bool(cfg.get("stream", False)), budget_s)
        t1 = time.perf_counter()
        fd = _door_lock(float(cfg["timeout_s"]))     # one puller on the MCDMA device; held until the blobs are built
        t_lock = time.perf_counter() - t1
        follower = A.Follower(cfg["socket"], handoff_id, peers, cfg["remote_dir"], float(cfg["timeout_s"]))
        follower.run()
        t2 = time.perf_counter()
    except Exception as exc:  # noqa: BLE001 - TensorFold prefills it itself
        _door_unlock(fd)
        _budget_give(boundary)
        if big:
            try:
                os.unlink("/tmp/tf_split_inflight")
            except OSError:
                pass
        STATS["failures"] += 1
        STATS["last_error"] = repr(exc)
        _say(f"split failed (network), TensorFold prefills locally: {exc!r}")
        _log({"error": repr(exc), "tokens": boundary, "phase": "network"}, cfg)
        return
    _budget_give(boundary)                           # the pair's KV for it is free once the export is pulled
    if big:
        try:
            os.unlink("/tmp/tf_split_inflight")
        except OSError:
            pass
    PENDING[str(job.job_id)] = {"at": time.time(), "fd": fd, "follower": follower, "peers": peers, "handoff_id": handoff_id,
                                "boundary": boundary, "sha": _sha(tokens), "cfg": cfg,
                                "t": {"budget_wait_s": round(t_budget, 3), "spark_prefill_s": round(t1 - t0 - t_budget, 3),
                                      "door_wait_s": round(t_lock, 3), "door_s": round(t2 - t1 - t_lock, 3),
                                      "spark_tok_s": round(boundary / max(t1 - t0 - t_budget, 1e-6), 1)}}


def build_phase(fill, filling) -> None:
    job = filling.job
    p = PENDING.pop(str(job.job_id), None)
    if p is None:
        return
    cfg = p["cfg"]
    store = getattr(fill, "checkpoints", None)
    prompt = list(job.prompt_ids)
    boundary = p["boundary"]
    try:
        if _sha(prompt[:boundary]) != p["sha"]:
            raise RuntimeError("prompt changed between submit and fill")
        A = ash()
        t2 = time.perf_counter()
        ranks = A.blobs_from_follow(p["follower"].result, p["peers"], p["handoff_id"])
        assembled = A.assemble(ranks, boundary, cfg)
        t3 = time.perf_counter()
        runtime = fill.engine.model
        cache = build_cache(assembled, boundary, runtime)
        t4 = time.perf_counter()
        stats = {"tokens": boundary, "handoff_id": p["handoff_id"], **p["t"], "assemble_s": round(t3 - t2, 3),
                 "build_s": round(t4 - t3, 3), "MiB": round(sum(r.header["data_bytes"] for r in ranks) / 2**20, 1),
                 "notes": assembled.notes[:4],
                 "arrivals": {q: p["follower"].result[q]["arrivals"] for q in p["peers"]}}
        if str(cfg.get("mode")) == "compare":
            nxt = int(prompt[boundary]) if len(prompt) > boundary else None
            compare._alt = assembled.alt
            try:
                stats["compare"] = compare(cache, prompt[:boundary], runtime, nxt)
            finally:
                compare._alt = None
            try:
                Path(os.path.expanduser("~/.glm53/tf_split_compare_last.json")).write_text(json.dumps(stats["compare"], indent=1))
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001
        STATS["failures"] += 1
        STATS["last_error"] = repr(exc)
        _say(f"split failed (build), TensorFold prefills locally: {exc!r}\n{traceback.format_exc()}")
        _log({"error": repr(exc), "tokens": boundary, "phase": "build"}, cfg)
        return
    finally:
        _door_unlock(p.get("fd"))
    _EXTRA_STARTS[boundary] = p["sha"]
    store.insert(prompt[:boundary], cache, last_prompt=prompt)
    filling.starts = fill.engine.prompt_chunks(prompt)
    STATS["splits"] += 1
    STATS["last"] = stats
    stats["stored"] = store.longest(prompt) == boundary
    _say(f"split done {json.dumps(stats)}")
    _log(stats, cfg)


def _log(stats: dict, cfg: dict) -> None:
    path = os.path.expanduser(str(cfg.get("log") or ""))
    if not path:
        return
    try:
        with open(path, "a") as fh:
            fh.write(json.dumps({"ts": time.time(), "at": time.strftime("%Y-%m-%d %H:%M:%S"), **stats}) + "\n")
    except Exception as exc:  # noqa: BLE001
        _say(f"log failed: {exc!r}")


# ---------------------------------------------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------------------------------------------
def _live():
    """This module, reloaded when its file changed (fixes land without reloading the model)."""
    import importlib
    mod = sys.modules[__name__]
    try:
        stamp = os.stat(mod.__file__).st_mtime_ns
    except OSError:
        return mod
    if _state.STAMP[0] is None:
        _state.STAMP[0] = stamp
    elif stamp != _state.STAMP[0]:
        _state.STAMP[0] = stamp
        try:
            mod = importlib.reload(mod)
            _say("module reloaded")
        except Exception as exc:  # noqa: BLE001
            _say(f"reload failed, keeping loaded code: {exc!r}")
    return sys.modules[__name__]


def install() -> None:
    from tensorfold.engine.lane_engine import LaneEngine
    from tensorfold.engine.prefill_plan import PromptChunks
    from tensorfold.server.prompt_fill import PromptFill

    if getattr(PromptFill._start_fill, "_tf_split", False):
        return
    original_fill = PromptFill._start_fill

    def _start_fill(self, filling):
        try:
            _live().build_phase(self, filling)
        except Exception as exc:  # noqa: BLE001 - never the request's failure
            _say(f"hook error: {exc!r}")
        return original_fill(self, filling)

    _start_fill._tf_split = True
    PromptFill._start_fill = _start_fill

    from tensorfold.server.scheduler import Scheduler
    original_submit = Scheduler.submit

    def submit(self, job):
        try:
            _live().network_phase(self, job)
        except Exception as exc:  # noqa: BLE001
            _say(f"submit hook error: {exc!r}")
        return original_submit(self, job)

    Scheduler.submit = submit

    original_chunks = LaneEngine.prompt_chunks

    def prompt_chunks(self, prompt_ids):
        chunks = original_chunks(self, prompt_ids)
        if not _EXTRA_STARTS or chunks.starts is None:
            return chunks
        n = len(prompt_ids)
        if len(_EXTRA_STARTS) > 64:
            for b in sorted(_EXTRA_STARTS)[:-64]:
                _EXTRA_STARTS.pop(b, None)
        extra = [b for b, h in list(_EXTRA_STARTS.items()) if b < n and b not in chunks._set and _sha(prompt_ids[:b]) == h]
        if not extra:
            return chunks
        return PromptChunks(sorted(set(chunks.starts) | set(extra)), chunks.length, step=chunks.step)

    PromptChunks_ok = hasattr(PromptChunks, "floor")
    LaneEngine.prompt_chunks = prompt_chunks
    CONFIG.get()
    _say(f"installed on PromptFill._start_fill + LaneEngine.prompt_chunks (config {CONFIG_PATH}, ash {SPLIT_SRC}, "
         f"chunks ok {PromptChunks_ok})")
