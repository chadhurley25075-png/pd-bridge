#!/usr/bin/env python3
"""pd_make_v3_from_mlx.py — produce a v3-format pooled capture from captured hidden states, computed with MLX itself.
Ground truth for the Spark hook and a test input for pd_assemble_blocks. Runs on a Studio.
usage: pd_make_v3_from_mlx.py --model PATH --captures attn_inputs.safetensors --out DIR [--chunk 2048]"""
import argparse, json, os, time
import mlx.core as mx
ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True); ap.add_argument("--captures",required=True); ap.add_argument("--out",required=True); ap.add_argument("--chunk",type=int,default=2048)
a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
def L(*x): print(time.strftime("%H:%M:%S "),*x,flush=True)
from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch; apply_deepseek_v4_patch()
from mlx_lm import load
model,_=load(a.model,lazy=True); layers=model.model.layers
for l in layers: mx.eval(l.attn.parameters())
caps=mx.load(a.captures); T=caps["layer_00"].shape[0]; NL=len(layers); L(f"T={T} layers={NL}")
cache=model.make_cache()
def sq(x): return None if x is None else mx.contiguous(x.reshape(x.shape[-2],x.shape[-1]) if x.ndim>2 else x)
out=[{} for _ in range(NL)]; bounds=[]; lastkv=[None]*NL
def snap(b, tag):
    for i,l in enumerate(layers):
        attn=l.attn; out[i][f"kvwin_{tag}"]=mx.contiguous(lastkv[i][-128:])   # pre-RoPE rows from the chunk computation
        if attn.compress_ratio==4:
            c=cache[i]; pk,pg=c[1].prev_win_kv,c[1].prev_win_gate
            out[i][f"prev_kv_{tag}"]=sq(pk); out[i][f"prev_gate_{tag}"]=sq(pg)
            if len(c.caches)>2: out[i][f"idx_prev_kv_{tag}"]=sq(c[2].prev_win_kv); out[i][f"idx_prev_gate_{tag}"]=sq(c[2].prev_win_gate)
    mx.eval([v for d in out for v in d.values() if v is not None])
done=0; t0=time.time()
while done<T:
    n=min(a.chunk,T-done)
    for i,l in enumerate(layers):
        attn=l.attn; x=caps[f"layer_{i:02d}"][done:done+n][None]; c=cache[i]; B,Ln,_=x.shape
        kvpre=attn.kv_norm(attn.wkv(x)).reshape(B,1,Ln,attn.head_dim); mx.eval(kvpre)
        rows=kvpre.reshape(Ln,attn.head_dim); lastkv[i]=rows if lastkv[i] is None or Ln>=128 else mx.concatenate([lastkv[i],rows],axis=0)[-128:]
        if attn.compress_ratio==0:
            off=c.offset; kv=attn.rope(kvpre,off); kv,_=c.update_and_fetch(kv,mx.zeros((B,1,Ln,0))); mx.eval(kv); continue
        off=c[0].offset; kv=attn.rope(kvpre,off); kv,_=c[0].update_and_fetch(kv,mx.zeros((B,1,Ln,0)))
        o=[kv, attn.compressor(x,c[1],off)]
        if len(c.caches)>2: o.append(attn.indexer.compressor(x,c[2],off))
        mx.eval(*o)
    done+=n
    if done%a.chunk==0: bounds.append(done); snap(done,str(done))
    L(f"{done}/{T}")
snap(T,"end")
for i,l in enumerate(layers):
    attn=l.attn; c=cache[i]
    if attn.compress_ratio==0: continue
    st=c[1].state; out[i]["pooled"]=sq(st[2])
    if st[0] is not None: out[i]["buf_kv"]=sq(st[0]); out[i]["buf_gate"]=sq(st[1])
    if len(c.caches)>2:
        st=c[2].state; out[i]["idx_pooled"]=sq(st[2])
        if st[0] is not None: out[i]["idx_buf_kv"]=sq(st[0]); out[i]["idx_buf_gate"]=sq(st[1])
tot=0
for i in range(NL):
    d={k:v for k,v in out[i].items() if v is not None}; mx.eval(list(d.values()))
    mx.save_safetensors(os.path.join(a.out,f"layer_{i:02d}.safetensors"),d); tot+=sum(v.nbytes for v in d.values())
json.dump({"T":T,"num_layers":NL,"ratios":[l.attn.compress_ratio for l in layers],"boundaries":bounds,"end":T,"source":"mlx-truth"},open(os.path.join(a.out,"manifest.json"),"w"))
open(os.path.join(a.out,"DONE"),"w").write("ok\n"); L(f"wrote {NL} layers, {tot/1e6:.1f} MB total, {time.time()-t0:.1f}s")
