#!/usr/bin/env python3
"""pd_assemble_blocks.py — build oMLX SSD prefix-cache blocks directly from a v3 pooled capture (no rebuild compute).
Library: assemble_and_write(cap_dir, ids, model, model_name, cache_dir) -> (paths, info).
CLI test: pd_assemble_blocks.py --model PATH --cap DIR --ids token_ids.json --out DIR

9/6 (FINDING-bench4-cold-fallback.md): a capture's manifest can promise boundaries its
layer files do not contain (a lost tail chunk still yields T-correct manifests — boundaries are computed
from T, not from the data). Assemble is now SALVAGE-SAFE: boundaries are attempted in order, the first
missing key stops the loop, and the longest contiguous good prefix is written. oMLX then prefix-hits at
B and natively prefills only the tail — a 93%-coverage capture becomes a near-win instead of a full
native fallback. info carries the coverage so the front door (and the X-PD-Bridge header, and the bench
JSON) can never again present a partial or failed bridge as a complete one.
Snapshots are stored INCREMENTALLY (BlockWriter.begin_stream/store_boundary) so peak memory is one
boundary, not the quadratic N(N+1)/2 accumulation — this is what lifts the ~82K ceiling.
StreamAssembler (below) is the same per-boundary build fed from seg_<b> files WHILE the Spark prefills
(PD_STREAM=1, docs/STREAMING-CAPTURE.md); pd_stream_assembler.py holds the mlx-free ordering/manifest logic.
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
class StreamAssembler:
    """Incremental twin of assemble_and_write for a STREAMING capture (docs/STREAMING-CAPTURE.md): feed it
    seg_<b>.safetensors files in boundary order as they arrive; each one becomes the decoder state at b (via the
    same _set_layer) and is stored through oMLX's own writer immediately. finalize(manifest) drains the writer and
    reports coverage + the manifest cross-check. Same salvage law as assemble_and_write: the first boundary that
    cannot be built stops the run and the longest good prefix is what gets written."""
    def __init__(self, ids, model, model_name, cache_dir, block=2048):
        from omlx_block_writer import BlockWriter
        from pd_stream_assembler import SegmentStream
        self.ids=list(ids); self.T=len(ids); self.model=model; self.layers=model.model.layers; self.block=block
        t0=time.time(); self.w=BlockWriter(model_name=model_name, out_dir=cache_dir, cache_list_factory=model.make_cache)
        L(f"stream writer init {time.time()-t0:.2f}s (cache dir scan)"); self.w.begin_stream(self.ids)
        def _cat(xs):
            a=mx.concatenate(xs,0); mx.eval(a); return a          # materialise the prefix once per boundary (O(b), like a snapshot)
        self.stream=SegmentStream(block, _cat, num_layers=len(self.layers))
        self.ok=0; self.B=0; self.missing=None; self.stopped=False; self.t_store=0.0
    def next_b(self): return self.stream.next_b
    def consumed(self): return self.stream.consumed
    def on_segment(self, b, path):
        """Store boundary b from its local segment file. Returns True if the block was written."""
        if self.stopped or b>self.T: return False
        t0=time.time()
        try: views=self.stream.feed(b, mx.load(path))
        except (KeyError, ValueError) as e:
            self.missing=[b,repr(e)]; self.stopped=True; L(f"stream boundary {b}: segment rejected ({e!r}) — stopping with {self.ok} good boundaries"); return False
        cache=self.model.make_cache()
        try:
            for i,l in enumerate(self.layers): _set_layer(l.attn, cache[i], views[i], b, str(b))
        except KeyError as e:
            self.missing=[b,repr(e)]; self.stopped=True; L(f"stream boundary {b}: key missing ({e!r}) — stopping with {self.ok} good boundaries"); return False
        self.w.snapshot(cache,b)
        try: self.w.store_boundary(b)
        except Exception as e:
            self.missing=[b,f"store: {e!r}"]; self.stopped=True; L(f"stream boundary {b}: store failed ({e!r}) — stopping with {self.ok} good boundaries"); return False
        self.ok+=1; self.B=b; self.t_store+=time.time()-t0
        return True
    def finalize(self, manifest=None):
        bounds=[b for b in range(self.block, self.T+1, self.block)]
        info={"boundaries_ok":self.ok,"boundaries_claimed":len(bounds),"B":self.B,"coverage":round(self.B/self.T,4) if self.T else 0.0,
              "T_manifest":(manifest or {}).get("T"),"T_request":self.T,"segments_consumed":len(self.stream.consumed),"t_store_s":round(self.t_store,2)}
        if self.missing: info["missing_at"]=self.missing
        if manifest is not None:
            try: info["stream_check"]=self.stream.check_manifest(manifest, self.T)
            except ValueError as e: info["stream_check"]={"error":repr(e)}
        L(f"stream assembled {self.ok}/{len(bounds)} boundaries (B={self.B}, coverage {info['coverage']:.1%}), store {self.t_store:.1f}s")
        if not self.ok: return [], info
        t0=time.time(); paths=self.w.finalize(self.ids); L(f"wrote/verified {len(paths)} blocks in {time.time()-t0:.1f}s")
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
