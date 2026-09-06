#!/usr/bin/env python3
"""pd_assemble_blocks.py — build oMLX SSD prefix-cache blocks directly from a v3 pooled capture (no rebuild compute).
Library: assemble_and_write(cap_dir, ids, model, model_name, cache_dir) -> list of block paths.
CLI test: pd_assemble_blocks.py --model PATH --cap DIR --ids token_ids.json --out DIR"""
import json, os, time
import mlx.core as mx
def L(*x): print(time.strftime("%H:%M:%S "),*x,flush=True)
_W={}
def _load_cap(cap_dir, NL):
    man=json.load(open(os.path.join(cap_dir,"manifest.json")))
    lay=[mx.load(os.path.join(cap_dir,f"layer_{i:02d}.safetensors")) for i in range(NL)]
    return man, lay
def _set_layer(attn, c, d, b, tag):
    """Set one layer's cache list to the state at boundary b from capture dict d."""
    kvw=d[f"kvwin_{tag}"]; kv=attn.rope(kvw[None,None].astype(mx.bfloat16), b-128)
    rot=c if attn.compress_ratio==0 else c[0]
    rot.state=(kv, mx.zeros((1,1,128,0),dtype=mx.bfloat16)); rot.meta_state=(0,128,b,128)
    if attn.compress_ratio==0: return
    r=attn.compress_ratio; pooled=d["pooled"][:b//r][None]
    if r==4: c[1].state=(None,None,pooled,d[f"prev_kv_{tag}"][None,None],d[f"prev_gate_{tag}"][None,None])
    else:    c[1].state=(None,None,pooled,None,None)
    if len(c.caches)>2:
        c[2].state=(None,None,d["idx_pooled"][:b//4][None],d[f"idx_prev_kv_{tag}"][None,None],d[f"idx_prev_gate_{tag}"][None,None])
def assemble_and_write(cap_dir, ids, model, model_name, cache_dir):
    from omlx_block_writer import BlockWriter
    layers=model.model.layers; NL=len(layers); t0=time.time(); man,lay=_load_cap(cap_dir,NL); L(f"capture load {time.time()-t0:.2f}s")
    T=len(ids); assert man["T"]==T, (man["T"],T)
    global _W
    t0=time.time()
    key=(model_name,str(cache_dir))
    if _W.get("key")!=key:
        _W={"key":key,"w":BlockWriter(model_name=model_name, out_dir=cache_dir, cache_list_factory=model.make_cache)}; L(f"writer init {time.time()-t0:.2f}s (cache dir scan, once per process)")
    w=_W["w"]; w.snapshots.clear()
    t0=time.time()
    for b in man["boundaries"]:
        if b>T: break
        cache=model.make_cache()
        for i,l in enumerate(layers): _set_layer(l.attn, cache[i], lay[i], b, str(b))
        w.snapshot(cache,b)
    L(f"assembled {len(man['boundaries'])} boundaries in {time.time()-t0:.1f}s")
    t0=time.time(); paths=w.finalize(ids); L(f"wrote {len(paths)} blocks in {time.time()-t0:.1f}s")
    return paths
if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True); ap.add_argument("--cap",required=True); ap.add_argument("--ids",required=True); ap.add_argument("--out",required=True); ap.add_argument("--name",default="DV4-Flash-MXFP4-MLX")
    a=ap.parse_args()
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch; apply_deepseek_v4_patch()
    from mlx_lm import load
    model,_=load(a.model,lazy=True)
    for l in model.model.layers: mx.eval(l.attn.parameters())
    ids=json.load(open(a.ids)); ids=ids["token_ids"] if isinstance(ids,dict) else ids
    paths=assemble_and_write(a.cap, ids, model, a.name, a.out)
    for p in paths: print(p)
