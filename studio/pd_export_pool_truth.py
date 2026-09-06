#!/usr/bin/env python3
"""pd_export_pool_truth.py — runs ON Studio 3. Exports MLX ground truth for the torch pooling port.

For each requested compressor (layers 2, 3, 42 main compressor + layer 2 indexer compressor):
  <out>/<name>_kv_c.npy      [T, out_dim] f32   compressor.project(x) kv   (per-2048-chunk projections, concatenated)
  <out>/<name>_gate.npy      [T, out_dim] f32   compressor.project(x) gate
  <out>/<name>_ape.npy       [ratio, out_dim] f32
  <out>/<name>_norm_w.npy    [head_dim] f32
  <out>/<name>_pooled.npy    [P, head_dim] f32  POOLED TRUTH: real Compressor.consume + fresh PoolingCache, 2048-token chunks
  <out>/<name>_buf_kv.npy / _buf_gate.npy        remainder rows (if any)
  <out>/<name>_prev_win_kv.npy / _prev_win_gate.npy   [ratio, out_dim] (ratio-4 only)
  <out>/<name>_rope_freqs.npy   MLX DeepseekV4RoPE._freqs (unscaled)
  <out>/<name>_rope_probe_in.npy / _rope_probe_out.npy   random bf16 [64, head_dim] rotated by comp.rope at probe offset
  <out>/<name>_meta.json     ratio, head_dim, out_dim, eps, dtypes, rope params, T, chunk, probe offset
SWA kv for layers 0 and 2:
  <out>/layer_XX_kv_win_prerope.npy [128, 512] f32 = kv_norm(wkv(x)) rows [T-128, T)
  <out>/layer_XX_kv_win_rope.npy    [128, 512] f32 = attn.rope(kv, offset=T-128)
  <out>/layer_XX_kv_meta.json       rope base / yarn / freqs
usage: $OMLX_PYTHON pd_export_pool_truth.py --model DIR --captures attn_inputs.safetensors --out DIR
"""
import argparse, json, os, time
import numpy as np
import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="$PD_MODEL")
ap.add_argument("--captures", default=os.path.expanduser("$PD_HOME/cap23k/attn_inputs.safetensors"))
ap.add_argument("--out", default=os.path.expanduser("$PD_HOME/pool_truth"))
ap.add_argument("--chunk", type=int, default=2048)
ap.add_argument("--layers", default="2,3,42")
ap.add_argument("--kv-layers", default="0,2")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
logf = open(os.path.join(args.out, "export.log"), "a")
def L(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(map(str, a)); print(s, flush=True); logf.write(s + "\n"); logf.flush()

from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch; apply_deepseek_v4_patch()
from mlx_lm import load
from mlx_lm.models.cache import PoolingCache  # injected by the patch

t0 = time.time(); model, tok = load(args.model, lazy=True)
cfg = model.args
layers = [int(s) for s in args.layers.split(",")]
kv_layers = [int(s) for s in args.kv_layers.split(",")]
for i in sorted(set(layers) | set(kv_layers)):
    mx.eval(model.model.layers[i].attn.parameters())
L(f"model (lazy) + attention params for layers {sorted(set(layers)|set(kv_layers))} in {time.time()-t0:.1f}s")
caps = mx.load(args.captures)
T = caps["layer_00"].shape[0]; CH = args.chunk
L(f"captures T={T} chunk={CH}")

def f32(a): return np.array(a.astype(mx.float32), copy=False)
def save(name, a): np.save(os.path.join(args.out, name + ".npy"), f32(a) if isinstance(a, mx.array) else a)
def rope_meta(r):
    return {"dims": r.dims, "freq_scale": r.freq_scale, "freqs_len": int(r._freqs.shape[0])}

def export_compressor(name, comp, x_full, rope_base, rope_scaling):
    ratio, hd = comp.compress_ratio, comp.head_dim
    cache = PoolingCache(ratio)
    kvs, gates = [], []
    for s in range(0, T, CH):
        x = x_full[s:s + CH][None]
        kv, gate = comp.project(x); mx.eval(kv, gate)
        pooled = comp.consume(kv, gate, cache, s); mx.eval(pooled)
        if cache.buf_kv is not None: mx.eval(cache.buf_kv, cache.buf_gate)
        if cache.prev_win_kv is not None: mx.eval(cache.prev_win_kv, cache.prev_win_gate)
        kvs.append(kv[0]); gates.append(gate[0])
    kv_all = mx.concatenate(kvs, 0); gate_all = mx.concatenate(gates, 0); mx.eval(kv_all, gate_all)
    st = cache.state  # (buf_kv, buf_gate, pooled, prev_win_kv, prev_win_gate)
    save(f"{name}_kv_c", kv_all); save(f"{name}_gate", gate_all)
    save(f"{name}_ape", comp.ape); save(f"{name}_norm_w", comp.norm.weight)
    save(f"{name}_pooled", st[2][0])
    if st[0] is not None: save(f"{name}_buf_kv", st[0][0]); save(f"{name}_buf_gate", st[1][0])
    if st[3] is not None: save(f"{name}_prev_win_kv", st[3][0, 0]); save(f"{name}_prev_win_gate", st[4][0, 0])
    save(f"{name}_rope_freqs", comp.rope._freqs)
    # rope probe: isolates the rope kernel (positions = probe_offset//ratio + i)
    probe_offset = (T // ratio) * ratio - 64 * ratio
    pin = mx.random.normal((1, 1, 64, hd), key=mx.random.key(7)).astype(kv_all.dtype)
    pout = comp.rope(pin, offset=probe_offset); mx.eval(pout)
    save(f"{name}_rope_probe_in", pin[0, 0]); save(f"{name}_rope_probe_out", pout[0, 0])
    meta = {"ratio": ratio, "head_dim": hd, "out_dim": comp.out_dim, "eps": comp.norm.eps, "T": T, "chunk": CH,
            "kv_dtype": str(kv_all.dtype), "gate_dtype": str(gate_all.dtype), "ape_dtype": str(comp.ape.dtype),
            "norm_w_dtype": str(comp.norm.weight.dtype), "pooled_dtype": str(st[2].dtype),
            "pooled_shape": list(st[2].shape), "remainder": cache.remainder,
            "rope": {"base": rope_base, "scaling": rope_scaling, **rope_meta(comp.rope)},
            "rope_probe_offset": probe_offset, "rope_probe_dtype": str(pin.dtype)}
    json.dump(meta, open(os.path.join(args.out, f"{name}_meta.json"), "w"), indent=1)
    L(f"{name}: ratio {ratio} hd {hd} out_dim {comp.out_dim} pooled {st[2].shape} rem {cache.remainder} "
      f"ape {comp.ape.dtype} norm_w {comp.norm.weight.dtype} kv {kv_all.dtype}")

for li in layers:
    attn = model.model.layers[li].attn
    x_full = caps[f"layer_{li:02d}"]
    t1 = time.time()
    export_compressor(f"layer_{li:02d}", attn.compressor, x_full, cfg.compress_rope_theta, cfg.rope_scaling)
    if getattr(attn, "indexer", None) is not None and li == 2:
        export_compressor(f"layer_{li:02d}_indexer", attn.indexer.compressor, x_full, cfg.compress_rope_theta, cfg.rope_scaling)
    L(f"layer {li} done in {time.time()-t1:.1f}s")

for li in kv_layers:
    attn = model.model.layers[li].attn
    x_full = caps[f"layer_{li:02d}"]
    s = (T // CH) * CH if T % CH else T - CH          # last prefill chunk start
    x = x_full[s:T][None]
    kv = attn.kv_norm(attn.wkv(x)).reshape(1, 1, T - s, attn.head_dim); mx.eval(kv)
    win = kv[:, :, -128:]
    roped = attn.rope(win, offset=T - 128); mx.eval(roped)
    save(f"layer_{li:02d}_kv_win_prerope", win[0, 0]); save(f"layer_{li:02d}_kv_win_rope", roped[0, 0])
    save(f"layer_{li:02d}_kv_rope_freqs", attn.rope._freqs)
    is_local = type(attn).__name__ == "LocalAttention"
    meta = {"layer": li, "attn_class": type(attn).__name__, "offset": T - 128, "T": T, "kv_dtype": str(kv.dtype),
            "kv_norm_w_dtype": str(attn.kv_norm.weight.dtype), "eps": attn.kv_norm.eps,
            "rope": {"base": cfg.rope_theta if is_local else cfg.compress_rope_theta,
                     "scaling": None if is_local else cfg.rope_scaling, **rope_meta(attn.rope)}}
    json.dump(meta, open(os.path.join(args.out, f"layer_{li:02d}_kv_meta.json"), "w"), indent=1)
    L(f"layer {li} kv window: {type(attn).__name__} rope base {meta['rope']['base']} scaling {meta['rope']['scaling'] is not None}")

json.dump({"T": T, "chunk": CH, "layers": layers, "kv_layers": kv_layers, "model": args.model,
           "compress_ratios": list(cfg.compress_ratios), "rope_theta": cfg.rope_theta,
           "compress_rope_theta": cfg.compress_rope_theta, "rope_scaling": cfg.rope_scaling,
           "rms_norm_eps": cfg.rms_norm_eps}, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
L("DONE")
