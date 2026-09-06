#!/usr/bin/env python3
"""pd_rebuild_mlx.py — rebuild DV4-Flash oMLX caches from captured per-layer attention inputs.
Loads the model LAZILY and evaluates only each layer's attention weights (~GBs, not 156 GB),
then replays attn_inputs through layer.attn(...) in oMLX-sized chunks with the model's own masks.
The attention output is discarded; the side effect is the cache list, identical in structure to a
real prefill. Dumps cache state to <out>/cache_state.safetensors (same layout as the capture dump)
so the two can be diffed. This is the Studio-side half of the P/D bridge.
usage: pd_rebuild_mlx.py --model PATH --captures attn_inputs.safetensors --ids token_ids.json --out DIR [--chunk 2048]
"""
import argparse, json, os, time
import mlx.core as mx
ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True); ap.add_argument("--captures",required=True); ap.add_argument("--ids",required=True); ap.add_argument("--out",required=True); ap.add_argument("--chunk",type=int,default=2048); ap.add_argument("--fast",action="store_true",help="projection-only rebuild (no attention compute)")
args=ap.parse_args(); os.makedirs(args.out,exist_ok=True)
def L(*a):
    s=time.strftime("%H:%M:%S ")+" ".join(map(str,a)); print(s,flush=True); open(os.path.join(args.out,"rebuild.log"),"a").write(s+"\n")
from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch; apply_deepseek_v4_patch()
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
t0=time.time(); model,tok=load(args.model, lazy=True)
layers=model.model.layers
for i,layer in enumerate(layers): mx.eval(layer.attn.parameters())
L(f"attention weights loaded (lazy model) in {time.time()-t0:.1f}s; active mem {mx.get_active_memory()/1e9:.1f} GB")
ids=json.load(open(args.ids))["token_ids"]; T=len(ids)
caps=mx.load(args.captures); L(f"captures: {len(caps)} layers, T={caps['layer_00'].shape[0]} (ids {T})")
assert caps["layer_00"].shape[0]==T

def fast_update(attn, x, c):
    """Projection-only cache update: exactly the cache-touching ops of LocalAttention /
    CompressedAttention / SparseCompressedAttention.__call__ (same call order, same classes)."""
    B,L,_=x.shape
    if attn.compress_ratio==0:            # LocalAttention: cache is a RotatingKVCache
        local=c; offset=local.offset
        kv=attn.kv_norm(attn.wkv(x)).reshape(B,1,L,attn.head_dim); kv=attn.rope(kv,offset)
        kv,_=local.update_and_fetch(kv, mx.zeros((B,1,L,0))); mx.eval(kv); return
    local=c[0]; offset=local.offset
    kv=attn.kv_norm(attn.wkv(x)).reshape(B,1,L,attn.head_dim); kv=attn.rope(kv,offset)
    kv,_=local.update_and_fetch(kv, mx.zeros((B,1,L,0)))
    pooled=attn.compressor(x, c[1], offset)
    outs=[kv,pooled]
    if len(c.caches)>2:                   # SparseCompressedAttention: indexer pool
        ipooled=attn.indexer.compressor(x, c[2], offset); outs.append(ipooled)
    mx.eval(*outs)

cache=model.make_cache()
from mlx_lm.models.cache import CacheList
first=cache[0]; mask_cache=first[0] if isinstance(first,CacheList) else first
t0=time.time(); done=0
while done<T:
    n=min(args.chunk,T-done)
    # same mask the model builds: from the first layer's local cache + sliding window
    dummy=mx.zeros((1,n,1),dtype=mx.bfloat16)
    mask=create_attention_mask(dummy, mask_cache, window_size=model.args.sliding_window, return_array=True)
    for i,layer in enumerate(layers):
        x=caps[f"layer_{i:02d}"][done:done+n][None]  # (1,n,4096) bf16
        if args.fast:
            fast_update(layer.attn, x, cache[i])
        else:
            y=layer.attn(x, mask=mask, cache=cache[i], _standard_mask=True)
            mx.eval(y)
    done+=n; L(f"rebuild {done}/{T} {(time.time()-t0):.1f}s {done/(time.time()-t0):.0f} tok/s")
L(f"rebuild done: {T} tokens in {time.time()-t0:.1f}s")
st={}; meta={}
def dump(prefix,c):
    if hasattr(c,"caches"):
        for j,sub in enumerate(c.caches): dump(f"{prefix}_sub_{j}",sub)
        return
    s=c.state
    if not isinstance(s,(tuple,list)): s=(s,)
    for k,e in enumerate(s):
        if isinstance(e,mx.array):
            if e.size==0: meta[f"{prefix}_state_{k}_zero_dim"]=",".join(map(str,e.shape))
            else: st[f"{prefix}_state_{k}"]=e
        else: meta[f"{prefix}_state_{k}"]=str(e)
    ms=getattr(c,"meta_state",None)
    if ms is not None: meta[f"{prefix}_meta_state"]=json.dumps(list(map(str,ms)) if isinstance(ms,(tuple,list)) else str(ms))
for i,c in enumerate(cache): dump(f"layer_{i}",c)
mx.save_safetensors(os.path.join(args.out,"cache_state.safetensors"),st,metadata=meta)
L(f"saved cache_state.safetensors: {len(st)} arrays, {len(meta)} meta"); L("DONE")
