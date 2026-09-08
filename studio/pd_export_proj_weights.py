#!/usr/bin/env python3
"""pd_export_proj_weights.py — runs ON the Mac decode node. Exports the DEQUANTIZED bf16 attention-projection weights the
pooled-cache path needs (wkv/kv_norm, compressor wkv/wgate/ape/norm, indexer-compressor same) for ALL 43
layers into ONE safetensors (+ json meta), and extracts a few cap23k hidden-state layers for validation.
usage: $OMLX_PYTHON pd_export_proj_weights.py --out $PD_HOME/proj_weights [--hidden-layers 0,2,3,42]
"""
import argparse, json, os, time
import mlx.core as mx, mlx.nn as nn
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="$PD_MODEL")
ap.add_argument("--captures", default=os.path.expanduser("$PD_HOME/cap23k/attn_inputs.safetensors"))
ap.add_argument("--out", default=os.path.expanduser("$PD_HOME/proj_weights"))
ap.add_argument("--hidden-layers", default="0,2,3,42")
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch; apply_deepseek_v4_patch()
from mlx_lm import load
t0 = time.time(); model, _ = load(args.model, lazy=True); cfg = model.args
def deq(lin):
    if isinstance(lin, nn.QuantizedLinear):
        w = mx.dequantize(lin.weight, lin.scales, getattr(lin, "biases", None), lin.group_size, lin.bits, mode=lin.mode)
        kind = f"{lin.mode}-g{lin.group_size}-b{lin.bits}"
    else:
        w, kind = lin.weight, "linear"
    return w.astype(mx.bfloat16), kind
out = {}; meta = {"rms_norm_eps": cfg.rms_norm_eps, "compress_ratios": list(cfg.compress_ratios), "head_dim": cfg.head_dim,
                  "qk_rope_head_dim": cfg.qk_rope_head_dim, "index_head_dim": cfg.index_head_dim, "hidden_size": cfg.hidden_size,
                  "rope_theta": cfg.rope_theta, "compress_rope_theta": cfg.compress_rope_theta, "rope_scaling": cfg.rope_scaling,
                  "max_position_embeddings": cfg.max_position_embeddings, "sliding_window": cfg.sliding_window, "layers": {}}
for i, layer in enumerate(model.model.layers):
    a = layer.attn; mx.eval(a.parameters()); r = cfg.compress_ratios[i]; lm = {"ratio": r, "attn_class": type(a).__name__}
    out[f"layer_{i}.wkv.weight"], lm["wkv_kind"] = deq(a.wkv)
    out[f"layer_{i}.kv_norm.weight"] = a.kv_norm.weight.astype(mx.bfloat16)
    if r > 0:
        c = a.compressor
        out[f"layer_{i}.comp.wkv.weight"], lm["comp_wkv_kind"] = deq(c.wkv)
        out[f"layer_{i}.comp.wgate.weight"], lm["comp_wgate_kind"] = deq(c.wgate)
        out[f"layer_{i}.comp.ape"] = c.ape.astype(mx.float32)
        out[f"layer_{i}.comp.norm.weight"] = c.norm.weight.astype(mx.bfloat16)
        lm["comp_out_dim"] = c.out_dim; lm["comp_head_dim"] = c.head_dim
        lm["orig_dtypes"] = {"kv_norm": str(a.kv_norm.weight.dtype), "comp_norm": str(c.norm.weight.dtype), "ape": str(c.ape.dtype),
                             "comp_wkv": str(c.wkv.weight.dtype)}
    if r == 4:
        c = a.indexer.compressor
        out[f"layer_{i}.idx.wkv.weight"], lm["idx_wkv_kind"] = deq(c.wkv)
        out[f"layer_{i}.idx.wgate.weight"], lm["idx_wgate_kind"] = deq(c.wgate)
        out[f"layer_{i}.idx.ape"] = c.ape.astype(mx.float32)
        out[f"layer_{i}.idx.norm.weight"] = c.norm.weight.astype(mx.bfloat16)
        lm["idx_out_dim"] = c.out_dim; lm["idx_head_dim"] = c.head_dim
    meta["layers"][str(i)] = lm
mx.eval(*out.values())
p = os.path.join(args.out, "dv4_proj_weights.safetensors"); mx.save_safetensors(p, out, metadata={"model": "DV4-Flash-MXFP4-MLX"})
json.dump(meta, open(os.path.join(args.out, "dv4_proj_weights.json"), "w"), indent=1)
print(f"weights: {len(out)} tensors, {os.path.getsize(p)/1e9:.3f} GB, {time.time()-t0:.1f}s; layer2 wkv kind {meta['layers']['2']['wkv_kind']}, comp {meta['layers']['2']['comp_wkv_kind']}", flush=True)
caps = mx.load(args.captures); hl = [int(s) for s in args.hidden_layers.split(",")]
h = {f"layer_{i:02d}": caps[f"layer_{i:02d}"] for i in hl}; mx.eval(*h.values())
p2 = os.path.join(args.out, "hidden_23k.safetensors"); mx.save_safetensors(p2, h, metadata={"T": str(h[f'layer_{hl[0]:02d}'].shape[0]), "layers": ",".join(map(str, hl))})
print(f"hidden: layers {hl} shape {h[f'layer_{hl[0]:02d}'].shape} {os.path.getsize(p2)/1e9:.3f} GB", flush=True)
