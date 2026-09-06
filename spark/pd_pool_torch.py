#!/usr/bin/env python3
"""pd_pool_torch.py — pure-torch port of the oMLX/mlx-lm DeepSeek-V4 compressor pooling.

Source of truth for the math (oMLX omlx/patches/deepseek_v4/deepseek_v4_model.py + cache_extras.py):
  _overlap_compress_kv / _simple_compress_kv / DeepseekV4RoPE / Compressor.consume / PoolingCache.

Dtype discipline mirrors MLX op-for-op so the result is bf16-level identical:
  overlap (ratio 4):  gate = gate + ape.astype(gate.dtype)          (bf16 add)
                      lane split/shift, softmax(precise) -> f32 math, bf16 out
                      (kv * w) in bf16, sum over the 2R rows accumulated in f32 -> bf16
  simple  (ratio 128): w = softmax(gate.f32 + ape) in f32 -> bf16 ; (kv * w) bf16 ; sum f32 -> bf16
  rmsnorm:  mx.fast.rms_norm: normalizer in f32, out = w * bf16(x * normalizer)
  rope:     mx.fast.rope(traditional=True, freqs=...): interleaved pairs, theta = pos * (1/freqs),
            positions = offset // freq_scale + i, the leading (head_dim - dims)/2 pairs have freqs=inf
            (theta 0 => untouched). Only the LAST `dims` features rotate.

Chunk semantics (PoolingCache.accumulate_windows, prompt mode): remainder rows (< ratio) are buffered
across calls; pool_base for a call = start_pos - remainder; ratio-4 layers carry the previous completed
window (raw kv/gate, NO ape) and prepend it so lane-A of the first new window is real (first window of
the sequence gets zero lane-A / -inf gate). Chunked calls == one whole call, bit-exact (self-test).

Dependency-free (torch only). CPU and CUDA.
"""
from __future__ import annotations
import math
from typing import Dict, Optional, Tuple

import torch

# ----------------------------------------------------------------------------------------------
# model constants (DV4-Flash config.json — read live 2026-09-06 on the Mac decode node)
# ----------------------------------------------------------------------------------------------
HEAD_DIM = 512
QK_ROPE_HEAD_DIM = 64
INDEX_HEAD_DIM = 128
ROPE_THETA = 10000.0            # LocalAttention (layers 0,1) SWA kv, NO yarn
COMPRESS_ROPE_THETA = 160000.0  # CompressedAttention / SparseCompressedAttention SWA kv (yarn) + all compressors
ROPE_SCALING = {"type": "yarn", "factor": 16, "original_max_position_embeddings": 65536,
                "beta_fast": 32, "beta_slow": 1}
MAX_POSITION_EMBEDDINGS = 1048576
RMS_NORM_EPS = 1e-6
SLIDING_WINDOW = 128


# ----------------------------------------------------------------------------------------------
# RoPE
# ----------------------------------------------------------------------------------------------
class DSv4Rope:
    """DeepseekV4RoPE semantics (MLX). `dims` = number of rotated features (64); the tensor's
    head_dim may be larger (512 / 128): only the trailing `dims` features rotate, as interleaved
    (traditional) pairs.  Positions are `offset // freq_scale + i`; the angular frequency is
    `1 / (freqs / freq_scale)` so a pooled row is placed at its window-start token position."""

    def __init__(self, dims: int, base: float, yarn_cfg: Optional[dict] = None,
                 max_pos: int = MAX_POSITION_EMBEDDINGS, freq_scale: int = 1):
        self.dims = int(dims)
        self.base = float(base)
        self.freq_scale = int(freq_scale)
        self.yarn_cfg = yarn_cfg
        # --- mirror MLX f32 op sequence exactly ---
        ar = torch.arange(0, dims, 2, dtype=torch.float32) / float(dims)
        inv_freq = 1.0 / torch.pow(torch.tensor(base, dtype=torch.float32), ar)
        rope_type = None
        if yarn_cfg is not None:
            rope_type = yarn_cfg.get("type") or yarn_cfg.get("rope_type")
        if rope_type in ("yarn", "deepseek_yarn"):
            factor = yarn_cfg["factor"]
            omax = yarn_cfg["original_max_position_embeddings"]
            beta_fast = yarn_cfg.get("beta_fast", 32)
            beta_slow = yarn_cfg.get("beta_slow", 1)

            def correction_dim(num_rotations):
                return dims * math.log(omax / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

            low = max(math.floor(correction_dim(beta_fast)), 0)
            high = min(math.ceil(correction_dim(beta_slow)), dims - 1)
            if low == high:
                high += 0.001
            ramp = (torch.arange(dims // 2, dtype=torch.float32) - low) / (high - low)
            smooth = 1 - torch.clamp(ramp, 0, 1)
            inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth
        elif rope_type not in (None, "default"):
            raise ValueError(f"unsupported rope type {rope_type}")
        self._freqs = 1.0 / inv_freq                      # MLX: self._freqs = 1.0 / inv_freq
        f = self._freqs
        if self.freq_scale != 1:
            f = f / float(self.freq_scale)                # MLX: f = f / self.freq_scale
        self.freqs_scaled = f                             # length dims//2 (the real pairs)
        self.inv_freq_eff = 1.0 / f                       # kernel: inv_freq = 1.0 / freqs[i]
        self._cache: Dict[torch.device, torch.Tensor] = {}

    def _inv(self, device) -> torch.Tensor:
        t = self._cache.get(device)
        if t is None:
            t = self.inv_freq_eff.to(device)
            self._cache[device] = t
        return t

    def __call__(self, x: torch.Tensor, offset: int, inverse: bool = False) -> torch.Tensor:
        """x: [..., L, head_dim]. Returns same dtype/shape, rows i at position offset//freq_scale + i."""
        L = x.shape[-2]
        head_dim = x.shape[-1]
        assert head_dim >= self.dims and (head_dim - self.dims) % 2 == 0
        pos0 = offset // self.freq_scale if self.freq_scale != 1 else offset
        pos = (torch.arange(L, device=x.device, dtype=torch.float32) + float(pos0))  # [L]
        inv = self._inv(x.device)
        if inverse:
            inv = -inv
        theta = pos[:, None] * inv[None, :]                # [L, dims/2] f32 (kernel: L * inv_freq)
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        nope = head_dim - self.dims
        xr = x[..., nope:].to(torch.float32)               # [..., L, dims]
        x1 = xr[..., 0::2]
        x2 = xr[..., 1::2]
        r1 = x1 * cos - x2 * sin
        r2 = x1 * sin + x2 * cos
        out_r = torch.stack([r1, r2], dim=-1).reshape(xr.shape).to(x.dtype)
        if nope == 0:
            return out_r
        return torch.cat([x[..., :nope], out_r], dim=-1)


# ----------------------------------------------------------------------------------------------
# elementwise pieces
# ----------------------------------------------------------------------------------------------
def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """mx.fast.rms_norm: out = w * T(x * rsqrt(mean(x^2) + eps)), accumulation in f32."""
    xf = x.to(torch.float32)
    normalizer = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = (xf * normalizer).to(x.dtype)
    if w.dtype == x.dtype:
        return y * w
    return (y.to(torch.float32) * w.to(torch.float32)).to(x.dtype)


def _bf16_sum_rows(prod: torch.Tensor) -> torch.Tensor:
    """(kv * weights).sum(axis=-2) with MLX's (Metal) bf16 reduction semantics — probed bit-exact on the Mac decode node
    2026-09-06 (`pd_probe_mlx_sum*.py`): MLX does NOT accumulate bf16 sums in f32.
      R <= 8   (col_reduce_small):  one thread per column, serial bf16 accumulation in row order      (R=8 → 100%)
      R = 128  (col_reduce_looped, BM=32): 32 strided partials (rows j, j+32, j+64, ...) each accumulated
               serially in bf16, then the 32 partials combined in f32 and rounded once                (R=128 → 100%)
    Plain f32 accumulation matched only 50% (R=8) / 81% (R=128) of elements (all misses = 1 bf16 ulp).
    Other R fall back to f32 accumulation (never occurs for DV4-Flash: R is 8 or 128)."""
    R = prod.shape[-2]
    if prod.dtype == torch.float32:
        return prod.sum(dim=-2)
    if R <= 8:
        acc = prod[..., 0, :]
        for i in range(1, R):
            acc = acc + prod[..., i, :]                    # bf16 + bf16 → one RNE rounding, like Metal
        return acc
    if R % 32 == 0:
        parts = prod[..., 0:32, :]
        for i in range(1, R // 32):
            parts = parts + prod[..., i * 32:(i + 1) * 32, :]
        return parts.to(torch.float32).sum(dim=-2).to(prod.dtype)
    return prod.to(torch.float32).sum(dim=-2).to(prod.dtype)


def simple_compress(kv: torch.Tensor, gate: torch.Tensor, ape: torch.Tensor) -> torch.Tensor:
    """kv, gate: [W, R, D] ; ape: [R, D]. Returns [W, D] in kv.dtype."""
    weights = torch.softmax(gate.to(torch.float32) + ape.to(torch.float32), dim=-2)
    weights = weights.to(kv.dtype)
    return _bf16_sum_rows(kv * weights)


def overlap_compress(kv: torch.Tensor, gate: torch.Tensor, ape: torch.Tensor) -> torch.Tensor:
    """kv, gate: [W, R, D] (D = 2*head_dim) ; ape: [R, D]. Returns [W, D//2] in kv.dtype.
    Lane-A of window i comes from window i-1 (first window: kv 0 / gate -inf)."""
    W, R, D = kv.shape
    h = D // 2
    gate = gate + ape.to(gate.dtype)
    kv_a, kv_b = kv[..., :h], kv[..., h:]
    kv_a = torch.cat([torch.zeros((1, R, h), dtype=kv.dtype, device=kv.device), kv_a[:-1]], dim=0)
    kvc = torch.cat([kv_a, kv_b], dim=1)                   # [W, 2R, h]
    g_a, g_b = gate[..., :h], gate[..., h:]
    g_a = torch.cat([torch.full((1, R, h), float("-inf"), dtype=kv.dtype, device=kv.device), g_a[:-1]], dim=0)
    gc = torch.cat([g_a, g_b], dim=1)                      # [W, 2R, h]
    weights = torch.softmax(gc.to(torch.float32), dim=-2).to(kv.dtype)   # precise=True
    return _bf16_sum_rows(kvc * weights)


# ----------------------------------------------------------------------------------------------
# chunk-capable layer pooling (Compressor.consume + PoolingCache.accumulate_windows/prev carry)
# ----------------------------------------------------------------------------------------------
def new_carry() -> dict:
    return {"buf_kv": None, "buf_gate": None, "prev_kv": None, "prev_gate": None, "processed": 0}


def pool_layer(kv_score: torch.Tensor, ratio: int, head_dim: int, ape: torch.Tensor,
               norm_w: torch.Tensor, eps: float, rope: DSv4Rope, start_pos: int,
               carry: Optional[dict] = None) -> Tuple[torch.Tensor, dict]:
    """kv_score: [L, 2*out_dim] (first half kv, second half gate), out_dim = head_dim*(2 if ratio==4 else 1).
    start_pos: absolute token position of kv_score[0] (== PoolingCache caller's `offset`).
    Returns (pooled_new [P_new, head_dim] in kv dtype, carry). Chunked calls == one whole call."""
    if carry is None:
        carry = new_carry()
    overlap = ratio == 4
    out_dim = head_dim * (2 if overlap else 1)
    assert kv_score.shape[-1] == 2 * out_dim, (kv_score.shape, out_dim)
    assert ape.shape == (ratio, out_dim), (ape.shape, ratio, out_dim)
    kv = kv_score[:, :out_dim]
    gate = kv_score[:, out_dim:]
    L = kv.shape[0]
    dt, dev = kv.dtype, kv.device
    if carry.get("processed") is not None:
        assert carry["processed"] == start_pos, f"start_pos {start_pos} != processed {carry['processed']}"

    buf_kv = carry["buf_kv"]
    buf_gate = carry["buf_gate"]
    rem = 0 if buf_kv is None else buf_kv.shape[0]
    total = rem + L
    usable = (total // ratio) * ratio
    new_rem = total % ratio

    if usable > 0:
        take = usable - rem
        r_kv = kv[:take] if rem == 0 else torch.cat([buf_kv, kv[:take]], dim=0)
        r_gate = gate[:take] if rem == 0 else torch.cat([buf_gate, gate[:take]], dim=0)
        r_base = start_pos - rem
        nb_kv = kv[L - new_rem:] if new_rem > 0 else None
        nb_gate = gate[L - new_rem:] if new_rem > 0 else None
    else:
        r_kv = r_gate = None
        r_base = 0
        nb_kv = kv if rem == 0 else torch.cat([buf_kv, kv], dim=0)
        nb_gate = gate if rem == 0 else torch.cat([buf_gate, gate], dim=0)
        if nb_kv.shape[0] == 0:
            nb_kv = nb_gate = None

    new = {"buf_kv": nb_kv, "buf_gate": nb_gate, "prev_kv": carry["prev_kv"],
           "prev_gate": carry["prev_gate"], "processed": start_pos + L}

    if r_kv is None:
        return torch.zeros((0, head_dim), dtype=dt, device=dev), new

    W = usable // ratio
    kvw = r_kv.reshape(W, ratio, out_dim)
    gw = r_gate.reshape(W, ratio, out_dim)
    if overlap:
        prev_kv, prev_gate = carry["prev_kv"], carry["prev_gate"]
        if prev_kv is not None:
            pooled = overlap_compress(torch.cat([prev_kv[None], kvw], 0),
                                      torch.cat([prev_gate[None], gw], 0), ape)[1:]
        else:
            pooled = overlap_compress(kvw, gw, ape)
        new["prev_kv"] = kvw[-1].clone()
        new["prev_gate"] = gw[-1].clone()
    else:
        pooled = simple_compress(kvw, gw, ape)

    pooled = rmsnorm(pooled, norm_w, eps)
    pooled = rope(pooled, r_base)
    return pooled, new


# ----------------------------------------------------------------------------------------------
# projections: kv_norm(wkv(x)) + compressor / indexer-compressor (kv | gate)  — the Mac's weights, in torch
# ----------------------------------------------------------------------------------------------
def linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """MLX nn.Linear / QuantizedLinear semantics: bf16 in, f32 accumulate, bf16 out. w is [out, in]."""
    if x.device.type == "cuda":
        return torch.matmul(x, w.t())                      # tensor cores accumulate in f32, output bf16
    return torch.matmul(x.to(torch.float32), w.to(torch.float32).t()).to(x.dtype)


def project_layer(hidden: torch.Tensor, W: dict):
    """hidden: [L, 4096] bf16 (attention_impl input = attn_norm(attn_hc(h))).  W: per-layer dict from
    load_proj_weights() — keys wkv, kv_norm, ratio, eps, and for ratio>0: comp_wkv, comp_wgate, comp_ape,
    comp_norm, comp_out_dim; for ratio==4: idx_wkv, idx_wgate, idx_ape, idx_norm.
    Returns (kv_pre [L,512] pre-RoPE, kv_score [L, 2*out_dim] or None, idx_kv_score [L, 512] or None)."""
    kv_pre = rmsnorm(linear(hidden, W["wkv"]), W["kv_norm"], W["eps"])
    kv_score = idx_kv_score = None
    if W["ratio"] > 0:
        kv_score = torch.cat([linear(hidden, W["comp_wkv"]), linear(hidden, W["comp_wgate"])], dim=-1)
    if W["ratio"] == 4:
        idx_kv_score = torch.cat([linear(hidden, W["idx_wkv"]), linear(hidden, W["idx_wgate"])], dim=-1)
    return kv_pre, kv_score, idx_kv_score


def read_safetensors(path: str, keys=None, device="cpu") -> dict:
    """Minimal safetensors reader (BF16 / F32 / F16), no dependency."""
    import json, struct
    dt = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        for k, v in hdr.items():
            if k == "__metadata__" or (keys is not None and k not in keys):
                continue
            s, e = v["data_offsets"]
            f.seek(base + s)
            raw = bytearray(f.read(e - s))
            t = torch.frombuffer(raw, dtype=torch.uint8).view(dt[v["dtype"]]).reshape(v["shape"])
            out[k] = t.clone().to(device)
    return out


def load_proj_weights(path: str, layers=None, device="cpu") -> dict:
    """Load dv4_proj_weights.safetensors (+ .json) → {layer_idx: W dict for project_layer}."""
    import json, os
    meta = json.load(open(os.path.splitext(path)[0] + ".json"))
    ratios = meta["compress_ratios"]
    layers = list(range(len(ratios))) if layers is None else list(layers)
    want = set()
    for i in layers:
        want |= {f"layer_{i}.wkv.weight", f"layer_{i}.kv_norm.weight"}
        if ratios[i] > 0:
            want |= {f"layer_{i}.comp.{k}" for k in ("wkv.weight", "wgate.weight", "ape", "norm.weight")}
        if ratios[i] == 4:
            want |= {f"layer_{i}.idx.{k}" for k in ("wkv.weight", "wgate.weight", "ape", "norm.weight")}
    t = read_safetensors(path, want, device)
    out = {}
    for i in layers:
        W = {"ratio": ratios[i], "eps": meta["rms_norm_eps"], "head_dim": meta["head_dim"],
             "wkv": t[f"layer_{i}.wkv.weight"], "kv_norm": t[f"layer_{i}.kv_norm.weight"]}
        if ratios[i] > 0:
            W.update(comp_wkv=t[f"layer_{i}.comp.wkv.weight"], comp_wgate=t[f"layer_{i}.comp.wgate.weight"],
                     comp_ape=t[f"layer_{i}.comp.ape"], comp_norm=t[f"layer_{i}.comp.norm.weight"])
            W["comp_out_dim"] = W["comp_wkv"].shape[0]
        if ratios[i] == 4:
            W.update(idx_wkv=t[f"layer_{i}.idx.wkv.weight"], idx_wgate=t[f"layer_{i}.idx.wgate.weight"],
                     idx_ape=t[f"layer_{i}.idx.ape"], idx_norm=t[f"layer_{i}.idx.norm.weight"])
        out[i] = W
    out["meta"] = meta
    return out


def make_ropes(meta: dict):
    """The three RoPE instances the cache path needs, from the weights json meta.
    Returns dict: swa_local (layers 0,1: rope_theta, NO yarn), swa (layers>=2: compress_rope_theta + yarn),
    comp4 / comp128 (compressor, freq_scale = ratio), idx (indexer compressor, head_dim 128, freq_scale 4)."""
    d = meta["qk_rope_head_dim"]; mp = meta["max_position_embeddings"]
    return {"swa_local": DSv4Rope(d, meta["rope_theta"], None, mp, 1),
            "swa": DSv4Rope(d, meta["compress_rope_theta"], meta["rope_scaling"], mp, 1),
            "comp4": DSv4Rope(d, meta["compress_rope_theta"], meta["rope_scaling"], mp, 4),
            "comp128": DSv4Rope(d, meta["compress_rope_theta"], meta["rope_scaling"], mp, 128),
            "idx": DSv4Rope(d, meta["compress_rope_theta"], meta["rope_scaling"], mp, 4)}


def pool_chunks(kv_score: torch.Tensor, chunk_lens, ratio, head_dim, ape, norm_w, eps, rope, start_pos=0):
    """Helper: run pool_layer over consecutive chunks of the given lengths. Returns (pooled, carry)."""
    carry = new_carry()
    outs = []
    pos = start_pos
    for n in chunk_lens:
        p, carry = pool_layer(kv_score[pos - start_pos: pos - start_pos + n], ratio, head_dim, ape,
                              norm_w, eps, rope, pos, carry)
        outs.append(p)
        pos += n
    return torch.cat(outs, 0), carry


# ----------------------------------------------------------------------------------------------
# self-test: chunked == whole, bit-exact, both ratios, random chunk lengths
# ----------------------------------------------------------------------------------------------
def _self_test(device="cpu", seed=0):
    g = torch.Generator().manual_seed(seed)
    T = 5003  # 5003 % 4 == 3, 5003 % 128 == 11 → trailing remainder exercised
    for ratio, hd, rope_dims in ((4, HEAD_DIM, QK_ROPE_HEAD_DIM), (128, HEAD_DIM, QK_ROPE_HEAD_DIM),
                                 (4, INDEX_HEAD_DIM, QK_ROPE_HEAD_DIM)):
        out_dim = hd * (2 if ratio == 4 else 1)
        ks = (torch.randn((T, 2 * out_dim), generator=g) * 2).to(torch.bfloat16).to(device)
        ape = (torch.randn((ratio, out_dim), generator=g) * 0.5).to(device)
        w = (1 + 0.1 * torch.randn((hd,), generator=g)).to(torch.bfloat16).to(device)
        rope = DSv4Rope(rope_dims, COMPRESS_ROPE_THETA, ROPE_SCALING, freq_scale=ratio)
        whole, cw = pool_layer(ks, ratio, hd, ape, w, RMS_NORM_EPS, rope, 0, None)
        assert whole.shape == (T // ratio, hd), whole.shape
        for lens in ([2048] * (T // 2048) + [T % 2048], [1] * 300 + [T - 300], None, None):
            if lens is None:
                cuts = sorted(torch.randint(1, T, (17,), generator=g).tolist())
                lens = [b - a for a, b in zip([0] + cuts, cuts + [T])]
            chunked, cc = pool_chunks(ks, lens, ratio, hd, ape, w, RMS_NORM_EPS, rope)
            assert torch.equal(chunked, whole), f"ratio {ratio} hd {hd}: chunked != whole for lens {lens[:6]}..."
            # carries must match too
            for k in ("buf_kv", "buf_gate", "prev_kv", "prev_gate"):
                a, b = cw[k], cc[k]
                assert (a is None and b is None) or torch.equal(a, b), f"carry {k} differs"
        print(f"self-test ratio={ratio} head_dim={hd}: chunked == whole bit-exact "
              f"(pooled {tuple(whole.shape)}, remainder {T % ratio}) OK")
    # rope sanity: nope pairs untouched, offset//freq_scale semantics
    rope = DSv4Rope(QK_ROPE_HEAD_DIM, COMPRESS_ROPE_THETA, ROPE_SCALING, freq_scale=4)
    x = torch.randn((3, HEAD_DIM)).to(torch.bfloat16)
    y = rope(x, 8)
    assert torch.equal(x[:, :HEAD_DIM - QK_ROPE_HEAD_DIM], y[:, :HEAD_DIM - QK_ROPE_HEAD_DIM])
    assert not torch.equal(x[:, -QK_ROPE_HEAD_DIM:], y[:, -QK_ROPE_HEAD_DIM:])
    assert torch.equal(rope(x, 8), rope(x, 11))  # 8//4 == 11//4
    print("rope: nope pairs untouched, offset//freq_scale OK")
    print("ALL SELF-TESTS PASSED")


if __name__ == "__main__":
    import sys
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    _self_test(dev)
