# SPDX-License-Identifier: Apache-2.0
"""pd_kv_connector.py — vLLM v1 KV connector (kv_producer) that streams a request's post-RoPE K/V rows out
of the paged cache as decoder-sized blocks. For plain-attention models (Qwen3 dense), whose decoder cache IS
the prefill engine's K/V rows: no projection, no pooling. See docs/RDMA.md.

Official hook instead of a sitecustomize patch: vLLM calls save_kv_layer() after every attention layer and
wait_for_save() when the forward pass exits, so a chunked prefill is captured step by step as it runs.

Only requests that carry kv_transfer_params={"pd_tag": "<id>"} are captured; everything else is untouched.
Prefix caching MUST be off on this engine (a skipped prefix means rows that are never computed here).

Three transports, chosen per capture:

  file (R0/R1)  rows go device->host every step, blocks are cut on the host and staged on NVMe:
                <capture_dir>/<pd_tag>/blk_000000.safetensors ... manifest.json, DONE (written last)
  rdma2 (R2)    when kv_connector_extra_config names rdma_host/rdma_lib: rows stay on the GPU until a block is
                complete, the block is copied device->host straight into the registered buffer of
                rdma/libpd_rdma_tx.so and RDMA-written to `pd_rdma recvd` on the Mac while prefill continues.
  omlx (R4)     as rdma2, and the request also names "omlx_model" (+ "omlx_block"): the block goes out as a complete
                oMLX-native file (chain hash, header, layer tensors — spark/pd_omlx_block.py) and the receiver lands
                it directly in the decoder's cache directory. Nothing is assembled or re-stored on the Mac.

RDMA sends run on a sender thread by default (rdma_async, default true): the forward pass only stacks each block
on the GPU in wire order and queues it with a CUDA event; the thread copies it in ONE contiguous copy into the
registered buffer on its own CUDA stream, RDMA-writes it and waits for the ack. The manifest and DONE are queued
behind the last block, so they still arrive last. The queue is bounded, so a stalled link applies backpressure
instead of growing GPU memory.

A local manifest is kept on the Spark for the record. For omlx captures the rows after the last full block (all but the
prompt's last token) follow the last block as tail.safetensors, so the decoder installs them instead of recomputing
up to 255 tokens over the whole context (manifest: tail_rows_shipped). If the RDMA sender cannot be opened the capture
falls back to files and the manifest says so; a front door waiting for RDMA then declines loudly instead of reporting
a bridged number.
"""
import ctypes
import json
import os
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import save_file

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.v1.core.sched.output import SchedulerOutput

try:
    import pd_omlx_block
except Exception:  # pragma: no cover - same directory as this module on the Spark
    pd_omlx_block = None

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger("vllm.pd_kv_connector")   # vLLM only configures handlers under "vllm."; a bare module name logs nowhere

_VERSION = "kv4"
_TAG_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MODEL_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
# Bounded queue of stacked blocks waiting on the GPU for the sender. 32 = one full 8,192-token step, so a step never
# waits on the wire; 32 x 64 MiB = 2 GiB on top of the R4b peak (107.5 of the guard's 118.8 GiB). 16 made each step
# send half its blocks while the forward pass waited.
_SEND_QUEUE_BLOCKS = 32


def _params_of(sampling_params) -> dict:
    extra = getattr(sampling_params, "extra_args", None) or {}
    return extra.get("kv_transfer_params") or {}


def _tag_of(params: dict) -> str | None:
    tag = params.get("pd_tag")
    if tag is None:
        return None
    if not isinstance(tag, str) or not _TAG_RE.match(tag):
        logger.warning("pd_kv: ignoring malformed pd_tag %r", tag)
        return None
    return tag


class _RdmaTx:
    """Sender: rdma/libpd_rdma_tx.so loaded into this worker. One QP to `pd_rdma recvd` on the Mac and one
    registered buffer. Every call raises on failure; the library itself never exits the process. Timings split into
    copy (device->host), wire (the RDMA WRITE, from the library's own counter) and ack (control round trip + the
    receiver's file write). Used from one thread at a time."""

    def __init__(self, lib_path: str, host: str, port: int, dev: str | None, gid_index: int, buf_mib: int, mtu: int = 4096):
        lib = ctypes.CDLL(lib_path)
        lib.pd_tx_open.restype = ctypes.c_void_p
        lib.pd_tx_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.pd_tx_buf.restype = ctypes.c_void_p
        lib.pd_tx_buf.argtypes = [ctypes.c_void_p]
        lib.pd_tx_capacity.restype = ctypes.c_size_t
        lib.pd_tx_capacity.argtypes = [ctypes.c_void_p]
        lib.pd_tx_block.argtypes = [ctypes.c_void_p, ctypes.c_char_p] + [ctypes.c_int] * 5
        lib.pd_tx_oblk.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        lib.pd_tx_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        lib.pd_tx_done.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        lib.pd_tx_dead.argtypes = [ctypes.c_void_p]
        lib.pd_tx_wire_s.restype = ctypes.c_double
        lib.pd_tx_wire_s.argtypes = [ctypes.c_void_p]
        lib.pd_tx_close.argtypes = [ctypes.c_void_p]
        h = lib.pd_tx_open(host.encode(), int(port), dev.encode() if dev else None, int(gid_index), int(buf_mib), int(mtu))
        if not h:
            raise RuntimeError(f"pd_tx_open({host}:{port}) failed — is `pd_rdma recvd` running on the Mac?")
        self.lib, self.h = lib, ctypes.c_void_p(h)
        self.capacity = int(lib.pd_tx_capacity(self.h))
        self.ptr = int(lib.pd_tx_buf(self.h))
        self.host = torch.frombuffer((ctypes.c_uint8 * self.capacity).from_address(self.ptr), dtype=torch.uint8)
        # Page-lock the buffer for CUDA too, so device->host lands in it by DMA. This is a separate 128 MiB staging
        # buffer, not vLLM's KV (pinning the KV itself halved prefill in the llama.cpp work).
        self.cuda_registered = False
        try:
            rc = torch.cuda.cudart().cudaHostRegister(self.ptr, self.capacity, 0)
            self.cuda_registered = int(getattr(rc, "value", rc)) == 0
        except Exception as e:  # pragma: no cover - platform dependent
            logger.warning("pd_kv: cudaHostRegister unavailable (%s); device->host copies stay pageable", e)

    def _copy_in(self, offset: int, kv: torch.Tensor, stream) -> float:
        """ONE copy of a contiguous GPU tensor into a contiguous view of the registered buffer. A non-contiguous
        destination makes torch stage through pageable memory and loses the DMA path (R4's 10 s at 32K)."""
        n = kv.numel() * kv.element_size()
        dst = self.host[offset: offset + n].view(kv.dtype).view(kv.shape)
        t0 = time.time()
        if stream is not None:
            with torch.cuda.stream(stream):
                dst.copy_(kv, non_blocking=self.cuda_registered)
            stream.synchronize()
        else:
            dst.copy_(kv)
        return time.time() - t0

    def _send(self, call) -> tuple[float, float]:
        t0 = time.time()
        w0 = self.lib.pd_tx_wire_s(self.h)
        if call():
            raise RuntimeError("rdma send failed")
        wire = self.lib.pd_tx_wire_s(self.h) - w0
        return wire, (time.time() - t0) - wire

    def block(self, tag: str, index: int, kv: torch.Tensor, stream=None) -> tuple[float, float, float]:
        """R2: kv [2, L, H, B, D] contiguous (k then v); the receiver writes blk_<index>.safetensors."""
        _, L, H, B, D = kv.shape
        if kv.numel() * 2 > self.capacity:
            raise RuntimeError(f"block of {kv.numel() * 2} bytes exceeds the {self.capacity}-byte registered buffer")
        copy = self._copy_in(0, kv, stream)
        wire, ack = self._send(lambda: self.lib.pd_tx_block(self.h, tag.encode(), int(index), int(L), int(H), int(B), int(D)))
        return copy, wire, ack

    def oblk(self, hash_hex: str, model_name: str, kv: torch.Tensor, stream=None) -> tuple[float, float, float]:
        """R4: kv [L, 2, H, B, D] contiguous, i.e. layer_i_state_0 then layer_i_state_1 per layer — the file's own tensor
        order. Header + body are built in place in the registered buffer; the receiver lands the file in the oMLX cache."""
        L, _, H, B, D = kv.shape
        hdr, data_len = pd_omlx_block.header(bytes.fromhex(hash_hex), model_name, L, H, B, D)
        total = len(hdr) + data_len
        if total > self.capacity or data_len != kv.numel() * 2:
            raise RuntimeError(f"oMLX block of {total} bytes does not fit / match the {self.capacity}-byte registered buffer")
        ctypes.memmove(self.ptr, hdr, len(hdr))
        copy = self._copy_in(len(hdr), kv, stream)
        wire, ack = self._send(lambda: self.lib.pd_tx_oblk(self.h, hash_hex.encode(), total))
        return copy, wire, ack

    def file(self, tag: str, name: str, data: bytes) -> None:
        if len(data) > self.capacity:
            raise RuntimeError(f"{name} ({len(data)} bytes) exceeds the registered buffer")
        ctypes.memmove(self.ptr, data, len(data))
        if self.lib.pd_tx_file(self.h, tag.encode(), name.encode(), len(data)):
            raise RuntimeError(f"pd_tx_file {tag}/{name} failed")

    def done(self, tag: str, status: str) -> None:
        if self.lib.pd_tx_done(self.h, tag.encode(), status.encode()):
            raise RuntimeError(f"pd_tx_done {tag} failed")

    def dead(self) -> bool:
        return bool(self.lib.pd_tx_dead(self.h))

    def close(self) -> None:
        if self.cuda_registered:
            try:
                torch.cuda.cudart().cudaHostUnregister(self.ptr)
            except Exception:
                pass
        self.lib.pd_tx_close(self.h)


@dataclass
class _Chunk:
    """Rows [start, start + len(slots)) of one tagged request, computed in this step."""
    tag: str
    prompt_len: int
    start: int
    slots: torch.Tensor                  # int64, cpu; paged-cache slot per token position
    last: bool                           # this step reaches prompt_len
    omlx_model: str | None = None        # R4: decoder model id the chain hashes are keyed on
    hashes: list | None = None           # R4: hex chain hash per full block; only on a chunk that starts at 0


@dataclass
class PdKvMetadata(KVConnectorMetadata):
    chunks: list[_Chunk] = field(default_factory=list)


@dataclass
class _Capture:
    tag: str
    prompt_len: int
    dir: str
    transport: str = "file"
    omlx_model: str | None = None
    hashes: list | None = None
    have: int = 0                                  # rows received, contiguous from 0
    pending: list = field(default_factory=list)    # tensors [L, 2, n, H, D] not yet cut into blocks (host: file, GPU: rdma)
    pending_rows: int = 0
    blocks: int = 0                                # blocks cut (and, for rdma, queued)
    sent: int = 0                                  # blocks the receiver acknowledged
    gaps: list = field(default_factory=list)
    steps: int = 0
    t_first: float = 0.0
    t_last: float = 0.0
    gather_sync_s: float = 0.0
    d2h_s: float = 0.0
    write_s: float = 0.0
    stack_s: float = 0.0
    enqueue_wait_s: float = 0.0
    copy_s: float = 0.0
    wire_s: float = 0.0
    ack_s: float = 0.0
    t_last_sent: float = 0.0
    bytes: int = 0
    error: str | None = None


class PdKvConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: "KVCacheConfig"):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        self._block_size = vllm_config.cache_config.block_size            # vLLM paged block
        self._omlx_block = int(extra.get("omlx_block", 256))              # decoder prefix-cache block
        self._capture_dir = os.path.expanduser(str(extra.get("capture_dir") or os.environ.get("PD_CAPTURE_DIR", "~/pd_capture")))
        self._rdma_cfg = {k: extra[k] for k in ("rdma_host", "rdma_port", "rdma_lib", "rdma_dev", "rdma_gid_index", "rdma_buf_mib") if k in extra}
        self._rdma_async = str(extra.get("rdma_async", "true")).lower() not in ("0", "false", "no")
        pc = vllm_config.parallel_config
        if pc.tensor_parallel_size != 1 or pc.pipeline_parallel_size != 1:
            # each TP rank holds only its shard of the KV heads; stitching shards is not implemented
            raise NotImplementedError("pd_kv_connector supports TP1/PP1 only")
        mc = vllm_config.model_config
        self._n_layers = mc.get_num_layers(pc)
        self._n_kv = mc.get_num_kv_heads(pc)
        self._head_dim = mc.get_head_size()
        # scheduler side
        self._reqs: dict[str, dict[str, Any]] = {}
        # worker side
        self._layer_index: dict[str, int] = {}
        self._layout: str | None = None
        self._device: torch.device | None = None
        self._step_rows: dict[int, list] = {}
        self._step_slots: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._caps: dict[str, _Capture] = {}
        self._tx: _RdmaTx | None = None
        self._tx_lock = threading.Lock()
        self._tx_error_logged = False
        self._sendq: queue.Queue | None = None
        self._sender: threading.Thread | None = None
        self._copy_stream = None
        if role == KVConnectorRole.WORKER:
            os.makedirs(self._capture_dir, exist_ok=True)
        logger.info("pd_kv: role=%s capture_dir=%s omlx_block=%d vllm_block=%d layers=%d kv_heads=%d head_dim=%d rdma=%s async=%s omlx_blocks=%s",
                    role.name, self._capture_dir, self._omlx_block, self._block_size, self._n_layers, self._n_kv,
                    self._head_dim, self._rdma_cfg.get("rdma_host") or "off", self._rdma_async,
                    "available" if pd_omlx_block else "unavailable")

    # ------------------------------------------------------------------ scheduler side
    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int | None, bool]:
        return 0, False   # producer only: never loads

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        pass

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = PdKvMetadata()
        n_sched = scheduler_output.num_scheduled_tokens
        for nr in scheduler_output.scheduled_new_reqs:
            params = _params_of(nr.sampling_params)
            tag = _tag_of(params)
            if tag is None:
                continue
            ids = nr.prompt_token_ids or []
            st = {"tag": tag, "prompt_len": len(ids), "blocks": list(nr.block_ids[0])}
            model = params.get("omlx_model")
            if model is not None:
                block = int(params.get("omlx_block") or self._omlx_block)
                if pd_omlx_block is None or not isinstance(model, str) or not _MODEL_RE.match(model) or block != self._omlx_block:
                    logger.warning("pd_kv: %s asks for oMLX blocks (model=%r block=%s) this engine cannot build (omlx_block=%d, module %s)",
                                   tag, model, block, self._omlx_block, "loaded" if pd_omlx_block else "missing")
                else:
                    st["omlx_model"] = model
                    st["hashes"] = [h.hex() for h in pd_omlx_block.chain_hashes(ids, model, self._omlx_block)]
            self._reqs[nr.req_id] = st
            self._emit(meta, st, nr.num_computed_tokens, n_sched.get(nr.req_id, 0))
        cr = scheduler_output.scheduled_cached_reqs
        for i, rid in enumerate(cr.req_ids):
            st = self._reqs.get(rid)
            if st is None:
                continue
            nb = cr.new_block_ids[i]
            if rid in cr.resumed_req_ids:
                if nb is not None:
                    st["blocks"] = list(nb[0])   # resumed after preemption: the full block list is re-sent
            elif nb is not None:
                st["blocks"].extend(nb[0])
            self._emit(meta, st, cr.num_computed_tokens[i], n_sched.get(rid, 0))
        for rid in scheduler_output.finished_req_ids:
            self._reqs.pop(rid, None)
        return meta

    def _emit(self, meta: PdKvMetadata, st: dict, computed: int, scheduled: int) -> None:
        P = st["prompt_len"]
        if scheduled <= 0 or computed >= P:
            return   # decode steps: nothing to capture
        end = min(computed + scheduled, P)
        pos = torch.arange(computed, end, dtype=torch.int64)
        blocks = torch.tensor(st["blocks"], dtype=torch.int64)
        slots = blocks[pos // self._block_size] * self._block_size + pos % self._block_size
        meta.chunks.append(_Chunk(st["tag"], P, computed, slots, end >= P, st.get("omlx_model"),
                                  st.get("hashes") if computed == 0 else None))

    def request_finished(self, request: "Request", block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        self._reqs.pop(request.request_id, None)
        return False, None

    # ------------------------------------------------------------------ worker side
    @staticmethod
    def _detect_layout(shape: tuple, block_size: int, n_kv: int, head_dim: int) -> str:
        """Logical KV tensor layouts, as vLLM hands them to register_kv_caches and to save_kv_layer."""
        if len(shape) == 4 and tuple(shape[1:]) == (n_kv, block_size, 2 * head_dim):
            return "BHN2D"   # vLLM 0.28 FLASH_ATTN: K and V packed in the content dim, K first (flash_attn.py splits on head_size)
        if len(shape) == 5 and shape[0] == 2 and tuple(shape[2:]) == (block_size, n_kv, head_dim):
            return "2BN"     # older FLASH_ATTN: [2, num_blocks, block, H, D]
        if len(shape) == 5 and shape[1] == 2 and tuple(shape[2:]) == (block_size, n_kv, head_dim):
            return "B2N"     # flashinfer / triton NHD: [num_blocks, 2, block, H, D]
        raise RuntimeError(f"pd_kv: unsupported KV cache shape {tuple(shape)} (block {block_size}, kv_heads {n_kv}, head_dim {head_dim})")

    @staticmethod
    def _gather_kv(kv_layer: torch.Tensor, layout: str, blk: torch.Tensor, off: torch.Tensor, head_dim: int):
        """K and V rows [n, H, D] for token slots (block index, offset in block). Advanced indexing copies only
        the requested rows, never the whole cache."""
        if layout == "BHN2D":
            x = kv_layer[blk, :, off]                    # [n, H, 2D]
            return x[..., :head_dim], x[..., head_dim:]
        if layout == "2BN":
            return kv_layer[0, blk, off], kv_layer[1, blk, off]
        return kv_layer[blk, 0, off], kv_layer[blk, 1, off]

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        names = sorted(kv_caches, key=lambda n: int(re.search(r"layers\.(\d+)\.", n).group(1)))
        self._layer_index = {n: i for i, n in enumerate(names)}
        t = kv_caches[names[0]]
        self._device = t.device
        self._layout = self._detect_layout(tuple(t.shape), self._block_size, self._n_kv, self._head_dim)
        logger.info("pd_kv: %d layers registered, layout %s, shape %s, dtype %s", len(names), self._layout, tuple(t.shape), t.dtype)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: "AttentionMetadata", **kwargs: Any) -> None:
        meta = self._get_connector_metadata()
        if not isinstance(meta, PdKvMetadata) or not meta.chunks:
            return
        li = self._layer_index[layer_name]
        for ci, ch in enumerate(meta.chunks):
            if ci not in self._step_slots:
                s = ch.slots.to(kv_layer.device, non_blocking=True)
                self._step_slots[ci] = (s // self._block_size, s % self._block_size)
                self._step_rows[ci] = [None] * self._n_layers
            b, o = self._step_slots[ci]
            k, v = self._gather_kv(kv_layer, self._layout, b, o, self._head_dim)
            self._step_rows[ci][li] = torch.stack((k, v))   # [2, n, H, D]; a copy, safe once the pages are reused

    def wait_for_save(self):
        meta = self._get_connector_metadata()
        if not isinstance(meta, PdKvMetadata) or not meta.chunks:
            return
        try:
            for ci, ch in enumerate(meta.chunks):
                rows = self._step_rows.get(ci)
                if rows is None or any(r is None for r in rows):
                    logger.warning("pd_kv: %s step missing layers; dropping capture", ch.tag)
                    self._caps.pop(ch.tag, None)
                    continue
                self._ingest(ch, rows)
        finally:
            self._step_rows.clear()
            self._step_slots.clear()

    # -- transport ---------------------------------------------------------------------------
    def _tx_get(self) -> _RdmaTx | None:
        """The RDMA sender, opened lazily on the first tagged request and reopened after a failure."""
        cfg = self._rdma_cfg
        if not cfg.get("rdma_host") or not cfg.get("rdma_lib"):
            return None
        if self._sendq is not None:
            self._sendq.join()                       # never swap the handle under an in-flight send
        with self._tx_lock:
            if self._tx is not None and self._tx.dead():
                self._tx_drop_locked()
            if self._tx is None:
                try:
                    self._tx = _RdmaTx(str(cfg["rdma_lib"]), str(cfg["rdma_host"]), int(cfg.get("rdma_port", 18516)),
                                       cfg.get("rdma_dev"), int(cfg.get("rdma_gid_index", 3)), int(cfg.get("rdma_buf_mib", 128)))
                    self._tx_error_logged = False
                    logger.info("pd_kv: rdma sender connected to %s:%s, buffer %d MiB, cuda_host_registered=%s, async=%s",
                                cfg["rdma_host"], cfg.get("rdma_port", 18516), self._tx.capacity >> 20,
                                self._tx.cuda_registered, self._rdma_async)
                except Exception as e:
                    if not self._tx_error_logged:
                        logger.warning("pd_kv: rdma sender unavailable (%s); captures fall back to files", e)
                        self._tx_error_logged = True
                    self._tx = None
            if self._tx is not None and self._rdma_async and self._sender is None:
                self._sendq = queue.Queue(maxsize=_SEND_QUEUE_BLOCKS)
                self._copy_stream = torch.cuda.Stream(device=self._device) if self._device is not None else None
                self._sender = threading.Thread(target=self._sender_loop, name="pd_kv_sender", daemon=True)
                self._sender.start()
            return self._tx

    def _tx_drop_locked(self) -> None:
        if self._tx is not None:
            try:
                self._tx.close()
            except Exception:
                pass
        self._tx = None

    def _tx_drop(self) -> None:
        with self._tx_lock:
            self._tx_drop_locked()

    def shutdown(self):
        if self._sendq is not None:
            self._sendq.put(None)
        self._tx_drop()

    def _sender_loop(self) -> None:
        if self._device is not None:
            torch.cuda.set_device(self._device)
        while True:
            item = self._sendq.get()
            try:
                if item is None:
                    return
                if item[0] == "finish":
                    self._finish_rdma(item[1])
                else:
                    self._send_block(*item)
            except Exception as e:  # never let the sender thread die silently
                logger.warning("pd_kv: sender thread error: %s", e)
            finally:
                self._sendq.task_done()

    def _send_block(self, kind: str, cap: _Capture, index: int, kv: torch.Tensor, hash_hex: str | None, ev) -> None:
        if cap.error:
            return                                  # an earlier block failed: nothing after it may land
        with gpu_sync_allowed():
            if ev is not None:
                ev.synchronize()                    # only this block's stack, not the whole forward queue
            with self._tx_lock:
                tx = self._tx
                try:
                    if tx is None:
                        raise RuntimeError("sender gone")
                    if kind == "oblk":
                        tc, tw, ta = tx.oblk(hash_hex, cap.omlx_model, kv, self._copy_stream)
                    else:
                        tc, tw, ta = tx.block(cap.tag, index, kv, self._copy_stream)
                except Exception as e:
                    cap.error = f"{cap.transport} block {index}: {e}"
                    cap.gaps.append([index * self._omlx_block, cap.have])   # nothing past this block reached the Mac
                    logger.warning("pd_kv: %s — capture %s stops at %d acknowledged blocks", cap.error, cap.tag, cap.sent)
                    self._tx_drop_locked()
                    return
        cap.copy_s += tc
        cap.wire_s += tw
        cap.ack_s += ta
        cap.sent += 1
        cap.t_last_sent = time.time()

    # -- capture -----------------------------------------------------------------------------
    def _ingest(self, ch: _Chunk, rows: list) -> None:
        now = time.time()
        cap = self._caps.get(ch.tag)
        if cap is None or ch.start == 0:
            if cap is not None:
                logger.warning("pd_kv: %s restarted from 0 (preempted and recomputed); discarding %d rows", ch.tag, cap.have)
            cap = _Capture(ch.tag, ch.prompt_len, os.path.join(self._capture_dir, ch.tag), t_first=now,
                           omlx_model=ch.omlx_model, hashes=ch.hashes)
            tx = self._tx_get()
            cap.transport = ("omlx" if ch.omlx_model and ch.hashes else "rdma2") if tx is not None else "file"
            os.makedirs(cap.dir, exist_ok=True)
            self._caps[ch.tag] = cap
        if ch.start != cap.have:
            cap.gaps.append([cap.have, ch.start])   # rows never delivered: nothing past `have` is trustworthy
            logger.warning("pd_kv: %s position gap have=%d step_start=%d", ch.tag, cap.have, ch.start)
        else:
            t0 = time.time()
            with gpu_sync_allowed():
                stacked = torch.stack(rows)        # [L, 2, n, H, D]
                if cap.transport == "file" or not self._rdma_async:
                    torch.cuda.synchronize(stacked.device)
                t1 = time.time()
                if cap.transport == "file":
                    stacked = stacked.to("cpu")
            cap.gather_sync_s += t1 - t0
            cap.d2h_s += time.time() - t1
            cap.pending.append(stacked)
            cap.pending_rows += stacked.shape[2]
            cap.have += stacked.shape[2]
            self._cut_blocks(cap)
        cap.steps += 1
        cap.t_last = time.time()
        if ch.last:
            self._finish(cap)
            self._caps.pop(ch.tag, None)

    def _cut_blocks(self, cap: _Capture) -> None:
        B = self._omlx_block
        if cap.gaps or cap.pending_rows < B:
            return
        buf = torch.cat(cap.pending, dim=2) if len(cap.pending) > 1 else cap.pending[0]
        n_full = buf.shape[2] // B
        t0 = time.time()
        with gpu_sync_allowed():
            for i in range(n_full):
                if cap.error:
                    break
                blk = buf[:, :, i * B:(i + 1) * B]                          # [L, 2, B, H, D]
                k = blk[:, 0].permute(0, 2, 1, 3)                           # [L, H, B, D], MLX KVCache order
                v = blk[:, 1].permute(0, 2, 1, 3)
                if cap.transport in ("rdma2", "omlx"):
                    ts = time.time()
                    if cap.transport == "omlx":
                        if cap.blocks >= len(cap.hashes):
                            cap.error = f"no chain hash for block {cap.blocks} ({len(cap.hashes)} computed)"
                            cap.gaps.append([cap.blocks * B, cap.have])
                            break
                        kind, hash_hex = "oblk", cap.hashes[cap.blocks]
                        kv = torch.stack((k, v), dim=1)                    # [L, 2, H, B, D] contiguous: the file's tensor order
                    else:
                        kind, hash_hex = "blk", None
                        kv = torch.stack((k, v))                           # [2, L, H, B, D] contiguous: k then v
                    cap.stack_s += time.time() - ts
                    if self._rdma_async and self._sendq is not None:
                        ev = torch.cuda.Event()
                        ev.record()
                        tq = time.time()
                        self._sendq.put((kind, cap, cap.blocks, kv, hash_hex, ev))   # blocks only if 16 blocks are in flight
                        cap.enqueue_wait_s += time.time() - tq
                    else:
                        self._send_block(kind, cap, cap.blocks, kv, hash_hex, None)
                else:
                    k, v = k.contiguous(), v.contiguous()
                    path = os.path.join(cap.dir, f"blk_{cap.blocks:06d}.safetensors")
                    save_file({"k": k, "v": v}, path + ".tmp")
                    os.replace(path + ".tmp", path)
                    cap.sent += 1
                cap.blocks += 1
                cap.bytes += k.numel() * k.element_size() * 2
        if cap.transport == "file":
            cap.write_s += time.time() - t0
        rest = buf[:, :, n_full * B:]
        cap.pending = [rest.clone()] if rest.shape[2] else []
        cap.pending_rows = rest.shape[2]

    def _manifest(self, cap: _Capture) -> dict:
        complete = not cap.gaps and cap.have == cap.prompt_len and cap.sent == cap.blocks and not cap.error
        return {"version": _VERSION, "tag": cap.tag, "transport": cap.transport,
                "landed": "omlx_cache" if cap.transport == "omlx" else "pull_dir" if cap.transport == "rdma2" else "spark_nvme",
                "omlx_model": cap.omlx_model, "async": bool(self._rdma_async and cap.transport != "file"),
                "T": cap.have if not cap.gaps else cap.gaps[0][0],
                "prompt_len": cap.prompt_len, "blocks": cap.blocks, "blocks_sent": cap.sent, "block": self._omlx_block,
                "layers": self._n_layers, "kv_heads": self._n_kv, "head_dim": self._head_dim, "dtype": "bfloat16",
                "tensor_layout": "k,v [layers, kv_heads, block, head_dim]", "complete": complete,
                "position_gaps": cap.gaps, "error": cap.error, "tail_rows_not_shipped": cap.pending_rows, "steps": cap.steps,
                "t_capture_span_s": round(cap.t_last - cap.t_first, 3),
                "t_forward_wait_s": round(cap.gather_sync_s, 3),   # sync modes: waits on the whole forward queue (mostly model compute)
                "t_d2h_s": round(cap.d2h_s, 3), "t_write_s": round(cap.write_s, 3),
                "t_stack_s": round(cap.stack_s, 3), "t_enqueue_wait_s": round(cap.enqueue_wait_s, 3),
                "t_copy_s": round(cap.copy_s, 3), "t_wire_s": round(cap.wire_s, 3), "t_ack_s": round(cap.ack_s, 3),
                "t_sender_lag_s": round(max(0.0, cap.t_last_sent - cap.t_last), 3) if cap.t_last_sent else None,
                "cuda_host_registered": bool(self._tx is not None and self._tx.cuda_registered),
                "bytes": cap.bytes, "finished_at": time.time()}

    def _write_local_manifest(self, cap: _Capture, meta: dict) -> bytes:
        blob = json.dumps(meta, indent=1).encode()
        tmp = os.path.join(cap.dir, "manifest.json.tmp")
        with open(tmp, "wb") as f:
            f.write(blob)
        os.replace(tmp, os.path.join(cap.dir, "manifest.json"))
        return blob

    def _log_finished(self, cap: _Capture, meta: dict) -> None:
        logger.info("pd_kv: finished %s via %s%s T=%d blocks=%d/%d complete=%s span=%.2fs stack=%.2fs enqueue_wait=%.2fs "
                    "copy=%.2fs wire=%.2fs ack=%.2fs sender_lag=%ss d2h=%.2fs write=%.2fs %.1f MB",
                    cap.tag, cap.transport, " async" if meta["async"] else "", meta["T"], cap.sent, cap.blocks,
                    meta["complete"], meta["t_capture_span_s"], cap.stack_s, cap.enqueue_wait_s, cap.copy_s, cap.wire_s,
                    cap.ack_s, meta["t_sender_lag_s"], cap.d2h_s, cap.write_s, cap.bytes / 1e6)

    def _finish(self, cap: _Capture) -> None:
        if cap.transport == "file":
            meta = self._manifest(cap)
            self._write_local_manifest(cap, meta)
            with open(os.path.join(cap.dir, "DONE"), "w") as f:
                f.write("complete" if meta["complete"] else "gaps")
            self._log_finished(cap, meta)
        elif self._rdma_async and self._sendq is not None:
            self._sendq.put(("finish", cap))         # behind the capture's last block
        else:
            self._finish_rdma(cap)

    def _tail_blob(self, cap: _Capture) -> tuple[bytes | None, int]:
        """O2b (docs/RDMA.md): rows after the last full block, except the prompt's last token, as a safetensors file
        in the decoder's own tensor naming (layer_i_state_0/1, [1, kv_heads, n, head_dim], bf16). oMLX only caches full
        blocks, so without these the decoder recomputes up to 255 tokens attending over the whole context (~2 s at 62K);
        with them it runs the last token only. ≤255 rows x 256 KiB fits the registered buffer."""
        n = cap.pending_rows - 1
        if cap.transport != "omlx" or cap.gaps or cap.error or n <= 0 or not cap.pending:
            return None, 0
        from safetensors.torch import save
        rest = cap.pending[0]                                          # [L, 2, pending_rows, H, D] on the GPU
        with gpu_sync_allowed():
            rows = rest[:, :, :n].permute(0, 1, 3, 2, 4).unsqueeze(2)   # [L, 2, 1, H, n, D]
            host = rows.to("cpu")
        tensors = {}
        for i in range(host.shape[0]):
            tensors[f"layer_{i}_state_0"] = host[i, 0].contiguous()
            tensors[f"layer_{i}_state_1"] = host[i, 1].contiguous()
        return save(tensors), n

    def _finish_rdma(self, cap: _Capture) -> None:
        tail, tail_rows = None, 0
        try:
            tail, tail_rows = self._tail_blob(cap)
        except Exception as e:                       # the tail is an optimization: never fail the capture over it
            logger.warning("pd_kv: %s tail rows not built: %s", cap.tag, e)
        meta = self._manifest(cap)                   # built after every block of this capture was sent or failed
        meta["tail_rows_shipped"] = 0
        with self._tx_lock:
            tx = self._tx
            if tail is not None and tx is not None:
                try:
                    tx.file(cap.tag, "tail.safetensors", tail)
                    meta["tail_rows_shipped"] = tail_rows
                except Exception as e:
                    logger.warning("pd_kv: %s tail rows over rdma failed: %s", cap.tag, e)
        blob = self._write_local_manifest(cap, meta)
        with self._tx_lock:
            tx = self._tx
            try:
                if tx is None:
                    raise RuntimeError("sender gone")
                tx.file(cap.tag, "manifest.json", blob)
                tx.done(cap.tag, "complete" if meta["complete"] else "gaps")
            except Exception as e:
                logger.warning("pd_kv: %s manifest/DONE over rdma failed: %s", cap.tag, e)
                self._tx_drop_locked()
        self._log_finished(cap, meta)
