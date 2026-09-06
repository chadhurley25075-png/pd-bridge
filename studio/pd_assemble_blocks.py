#!/usr/bin/env python3
"""pd_assemble_blocks.py — build oMLX SSD prefix-cache blocks directly from a v3 pooled capture (no rebuild compute).
Library: assemble_and_write(cap_dir, ids, model, model_name, cache_dir) -> (paths, info).
CLI test: pd_assemble_blocks.py --model PATH --cap DIR --ids token_ids.json --out DIR

9/6 (qwenmax seat, FINDING-bench4-cold-fallback.md): a capture's manifest can promise boundaries its
layer files do not contain (a lost tail chunk still yields T-correct manifests — boundaries are computed
from T, not from the data). Assemble is now SALVAGE-SAFE: boundaries are attempted in order, the first
missing key stops the loop, and the longest contiguous good prefix is written. oMLX then prefix-hits at
B and natively prefills only the tail — a 93%-coverage capture becomes a near-win instead of a full
native fallback. info carries the coverage so the front door (and the X-PD-Bridge header, and the bench
JSON) can never again present a partial or failed bridge as a complete one.
Snapshots are stored INCREMENTALLY (BlockWriter.begin_stream/store_boundary) so peak memory is one
boundary, not the quadratic N(N+1)/2 accumulation — this is what lifts the ~82K ceiling.
"""
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
    T=len(ids)
    bounds=[b for b in man["boundaries"] if 0<b<=T]
    # 9/6: a per-process cached writer left 47/48 blocks unwritten on the third request (its background SSD writer/paged state
    # went stale between requests) → fresh writer per request; the cache-dir scan costs ~1-2 s and is safe.
    t0=time.time(); w=BlockWriter(model_name=model_name, out_dir=cache_dir, cache_list_factory=model.make_cache); L(f"writer init {time.time()-t0:.2f}s (cache dir scan)")
    w.begin_stream(ids)
    t0=time.time(); ok=0; missing=None
    for b in bounds:
        cache=model.make_cache()
        try:
            for i,l in enumerate(layers): _set_layer(l.attn, cache[i], lay[i], b, str(b))
        except KeyError as e:
            missing=[b,repr(e)]; L(f"boundary {b}: key missing from capture ({e!r}) — stopping with {ok} good boundaries"); break
        w.snapshot(cache,b)
        try:
            w.store_boundary(b)          # incremental: peak memory = one boundary snapshot
        except Exception as e:
            missing=[b,f"store: {e!r}"]; L(f"boundary {b}: store failed ({e!r}) — stopping with {ok} good boundaries"); break
        ok+=1
    B=bounds[ok-1] if ok else 0
    info={"boundaries_ok":ok,"boundaries_claimed":len(bounds),"B":B,"coverage":round(B/T,4),"T_manifest":man.get("T"),"T_request":T}
    if missing: info["missing_at"]=missing
    L(f"assembled {ok}/{len(bounds)} boundaries (B={B}, coverage {B/T:.1%}) in {time.time()-t0:.1f}s")
    if not ok:
        return [], info
    t0=time.time(); paths=w.finalize(ids); L(f"wrote {len(paths)} blocks in {time.time()-t0:.1f}s")
    info["blocks"]=len(paths)
    return paths, info
if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True); ap.add_argument("--cap",required=True); ap.add_argument("--ids",required=True); ap.add_argument("--out",required=True); ap.add_argument("--name",default="DV4-Flash-MXFP4-MLX")
    a=ap.parse_args()
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch; apply_deepseek_v4_patch()
    from mlx_lm import load
    model,_=load(a.model,lazy=True)
    for l in model.model.layers: mx.eval(l.attn.parameters())
    ids=json.load(open(a.ids)); ids=ids["token_ids"] if isinstance(ids,dict) else ids
    paths,info=assemble_and_write(a.cap, ids, model, a.name, a.out)
    print(json.dumps(info))
    for p in paths: print(p)
