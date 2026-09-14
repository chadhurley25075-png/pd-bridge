#!/usr/bin/env python3
"""pd_front.py — the heterogeneous P/D front door for DeepSeek-V4-Flash (2026-09-06).
Runs ON the decode Studio next to oMLX. OpenAI-compatible /v1/chat/completions on :8012.
Per request:  tokenize (oMLX's own patched tokenizer/template) → POST ids to the Spark prefill
engine (vLLM TP2, capture hook) → fetch the per-layer attention-input capture (rsync over LAN) →
rebuild oMLX caches locally with the resident attention-only model → write blocks into oMLX's
SSD prefix cache (omlx_block_writer) → forward the ORIGINAL chat request to oMLX (:8011), which
hits the prefix and only decodes. Falls back to plain oMLX on any bridge error (never breaks a reply).
Env: PD_MODEL (MLX model dir), PD_MODEL_NAME (oMLX model id), PD_SPARK (http://PREFILL_HOST:8000),
     PD_SPARK_SSH (PREFILL_USER@PREFILL_HOST), PD_SPARK_CAPDIR (~/pd_capture),
     PD_OMLX (http://127.0.0.1:8011), PD_CACHE_DIR (~/.omlx/cache), PD_MIN_TOKENS (default 4096: below this, skip the bridge)
"""
import json, os, sys, time, glob, subprocess, threading, traceback
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
import threading, queue as _queue
import urllib.request
import mlx.core as mx

PD_MODEL=os.environ.get("PD_MODEL", os.path.expanduser("~/models/DV4-Flash-MXFP4-MLX"))
PD_MODEL_NAME=os.environ.get("PD_MODEL_NAME","DV4-Flash-MXFP4-MLX")
PD_SPARK=os.environ.get("PD_SPARK","http://PREFILL_HOST:8000")
PD_SPARK_SSH=os.environ.get("PD_SPARK_SSH","PREFILL_USER@PREFILL_HOST")
PD_SPARK_CAPDIR=os.environ.get("PD_SPARK_CAPDIR","~/pd_capture")
PD_OMLX=os.environ.get("PD_OMLX","http://127.0.0.1:8011")
PD_CACHE_DIR=os.environ.get("PD_CACHE_DIR",os.path.expanduser("~/.omlx/cache"))
PD_MIN_TOKENS=int(os.environ.get("PD_MIN_TOKENS","4096"))
PD_MIN_TAIL=int(os.environ.get("PD_MIN_TAIL","8192"))   # pooled mode: bridge when the uncached tail is at least this many tokens
PD_MODE=os.environ.get("PD_MODE","hidden")   # "hidden" = v2 hidden-state capture+rebuild, "pooled" = v3 pooled capture+assemble, "kv" = plain-attention K/V blocks (docs/RDMA.md)
PD_PULL_DIR=os.path.expanduser(os.environ.get("PD_PULL_DIR","~/pd_lab/pd_pull"))
PD_SPARK_MODEL=os.environ.get("PD_SPARK_MODEL","deepseek-v4-flash")   # served model name on the prefill engine
PD_SHARE_PORT=int(os.environ.get("PD_SHARE_PORT","8010"))               # capture share on the prefill head
PD_BLOCK=int(os.environ.get("PD_BLOCK","256" if PD_MODE=="kv" else "2048"))   # decoder's paged_cache_block_size
PD_TRANSPORT=os.environ.get("PD_TRANSPORT","tcp10")                       # recorded in every verdict: a number without its transport is not a number
PD_STREAM=os.environ.get("PD_STREAM","0")=="1"                             # pooled mode: consume seg_<b> files WHILE the Spark prefills (docs/STREAMING-CAPTURE.md); the hook must run with PD_STREAM=1 too
PD_STREAM_ACK=os.environ.get("PD_STREAM_ACK","0")=="1"                     # tell the share each segment is stored (with PD_SHARE_ACK_DELETE=1 there it frees the Spark's disk)
# 9/6 (FINDING-bench4-cold-fallback.md): capture-trust knobs. The hook can flush MID-REQUEST during
# chunked prefill (DONE + manifest T < request T) and a T-correct capture can still miss tail-boundary
# data. The front validates every DONE capture, prefers a complete one, salvages the best contiguous
# prefix, or declines LOUDLY — a native fallback can never again masquerade as a bridged measurement.
PD_CAPTURE_GRACE=float(os.environ.get("PD_CAPTURE_GRACE","45"))    # after engine returns: wait this long for the final capture flush
PD_STAMP_TIMEOUT=float(os.environ.get("PD_STAMP_TIMEOUT","300"))   # FLOOR only — see stamp_deadline()
# 9/7: a flat 300s silently broke every request above ~375K tokens. The prefill itself takes T/rate seconds, so a
# fixed deadline is not "the share is dead", it is "the prompt is big". Measured bridged prefill ~1250 tok/s; 700 is
# a deliberately pessimistic floor so a slow-but-healthy run is never mistaken for a dead hook. 900K -> ~1586s.
PD_STAMP_TOK_RATE=float(os.environ.get("PD_STAMP_TOK_RATE","700"))
# 9/7: the engine call, the capture fetch and the block push all carried a flat 900s. At 1.25M tokens the prefill
# alone needs ~1400s, so the bridge died at 15 min and every request above ~800K silently fell back to the decoder
# alone. One knob, generous, env-overridable. This is a ceiling for a hang, not a pacing budget.
PD_LONG_TIMEOUT=float(os.environ.get("PD_LONG_TIMEOUT","10800"))
def stamp_deadline(T):
    return PD_STAMP_TIMEOUT + (T/PD_STAMP_TOK_RATE if T else 0.0)
PD_MIN_COVERAGE=float(os.environ.get("PD_MIN_COVERAGE","0.5"))     # salvage floor: contiguous captured prefix / request tokens
import shutil
PD_PORT=int(os.environ.get("PD_PORT","8012"))
LOCAL_CAP=os.path.expanduser("~/pd_capture_in"); os.makedirs(LOCAL_CAP,exist_ok=True)
LOG=open(os.path.expanduser("~/pd_front.log"),"a")
def L(*a):
    s=time.strftime("%H:%M:%S ")+" ".join(map(str,a)); print(s,flush=True); LOG.write(s+"\n"); LOG.flush()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if PD_MODE!="kv":   # DV4 only; a plain-attention model needs neither the patch nor resident weights
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch; apply_deepseek_v4_patch()
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import CacheList
import omlx_block_writer  # sister B's writer: write_blocks(cache_list, token_ids, model_name, out_dir) -> paths

try: PD_MODEL_TYPE=json.load(open(os.path.join(PD_MODEL,"config.json"))).get("model_type") or "deepseek_v4"
except Exception: PD_MODEL_TYPE="deepseek_v4"
t0=time.time(); MODEL,TOK=load(PD_MODEL, lazy=True)
if PD_MODE=="kv":
    L(f"kv mode: tokenizer + cache geometry only, weights stay lazy ({time.time()-t0:.1f}s, model_type {PD_MODEL_TYPE})")
else:
    for layer in MODEL.model.layers: mx.eval(layer.attn.parameters())
    L(f"attention-only model resident in {time.time()-t0:.1f}s, active mem {mx.get_active_memory()/1e9:.1f} GB")
LOCK=threading.Lock()

# ---- prompt rendering: oMLX's OWN chain (server.py create_chat_completion → VLMBatchedEngine._apply_chat_template),
# so the token ids we hash are byte-identical to the ones oMLX will look up. ----
from omlx.server import ChatCompletionRequest
from omlx.api.utils import (extract_multimodal_content, detect_and_strip_partial, prepare_system_messages_for_template,
                            merge_reasoning_effort_chat_template_kwargs, uses_native_reasoning_content)
from omlx.api.tool_calling import convert_tools_for_template
from omlx.model_settings import merge_chat_template_request_kwargs, ModelSettingsManager
from omlx.reasoning_effort import apply_chat_template_with_reasoning_effort_fallback
from omlx.utils.image import extract_images_from_messages
try:
    from pathlib import Path as _P; _MSM=ModelSettingsManager(_P(os.environ.get('PD_OMLX_HOME', '~/.omlx')).expanduser())
except Exception as _e: _MSM=None; L("model settings manager unavailable:",_e)

def render_request(raw_json):
    """raw OpenAI chat request (dict) -> (token_ids, model_id) exactly as oMLX renders it."""
    req=ChatCompletionRequest.model_validate(raw_json)
    ms=None
    if _MSM is not None:
        try: ms=_MSM.get_settings(req.model)
        except Exception: ms=None
    merged=merge_chat_template_request_kwargs(ms, merge_reasoning_effort_chat_template_kwargs(req.chat_template_kwargs, req.reasoning_effort))
    native=uses_native_reasoning_content(req.model, config_model_type=PD_MODEL_TYPE, engine_model_type=PD_MODEL_TYPE,
                                         preserve_thinking_default=(ms.preserve_thinking if ms else None))
    msgs=extract_multimodal_content(req.messages, (ms.max_tool_result_tokens if ms else None), TOK,
                                    native_reasoning_content=native, consolidate_system_messages=False)
    is_partial=detect_and_strip_partial(msgs)
    tools=None if req.tool_choice=="none" else req.tools
    tools_t=convert_tools_for_template(tools) if tools else None
    msgs=prepare_system_messages_for_template(msgs, TOK, tools=tools_t, chat_template_kwargs=(merged or None),
                                              is_partial=is_partial, merge_consecutive_roles=False, unsupported_mid_system_policy="strict")
    text_msgs,_,_=extract_images_from_messages(msgs)
    tk={"tokenize":False,"add_generation_prompt":not is_partial}
    if is_partial: tk["continue_final_message"]=True
    if tools_t: tk["tools"]=convert_tools_for_template(tools_t)   # the engine converts again; mirror it
    if ms is not None and ms.enable_thinking is not None: tk["enable_thinking"]=ms.enable_thinking
    if merged: tk.update(merged)
    prompt=apply_chat_template_with_reasoning_effort_fallback(TOK, text_msgs, tk, is_harmony=False)
    return list(map(int, TOK.encode(prompt))), req.model

def cached_prefix_tokens(ids):
    """How many leading tokens oMLX already holds as SSD blocks for this exact prompt (chain hashes, PD_BLOCK/block)."""
    try:
        from omlx_block_writer import chain_hashes_for
        hs=chain_hashes_for(ids, PD_MODEL_NAME, block_size=PD_BLOCK)
    except Exception as e:
        L("chain_hashes_for unavailable:",e); return 0
    n=0
    for h in hs:
        hx=h.hex() if isinstance(h,(bytes,bytearray)) else str(h)
        if os.path.isfile(os.path.join(PD_CACHE_DIR,hx[0],hx+".safetensors")): n+=1
        else: break
    return n*PD_BLOCK

def _http_listing(base):
    """Newest .json sidecar via the capture HTTP share (no ssh)."""
    import re, html
    txt=urllib.request.urlopen(base+"/",timeout=10).read().decode("utf-8","replace")
    names=[html.unescape(n) for n in re.findall(r'href="([^"]+\.json)"',txt)]
    return sorted(names)

def spark_prefill(ids):
    base=PD_SPARK.rsplit(":",1)[0]+f":{PD_SHARE_PORT}"
    before=set(_http_listing(base))
    body=json.dumps({"model":PD_SPARK_MODEL,"prompt":ids,"max_tokens":1,"temperature":0}).encode()
    t=time.time(); r=urllib.request.urlopen(urllib.request.Request(PD_SPARK+"/v1/completions",body,{"Content-Type":"application/json"}),timeout=PD_LONG_TIMEOUT); r.read()
    t_engine=time.time()-t
    # wait for the sidecar whose token count matches (capture flushes on idle after the engine returns)
    name=None
    for _ in range(600):
        for n in reversed(_http_listing(base)):
            if n in before: continue
            try:
                m=json.load(urllib.request.urlopen(base+"/"+n,timeout=10))
                if int(m.get("tokens",-1))==len(ids): name=n; break
            except Exception: pass
        if name: break
        time.sleep(0.25)
    if not name: raise RuntimeError("capture sidecar never matched token count")
    t_prefill=time.time()-t; L(f"bridge: engine {t_engine:.1f}s, capture ready +{t_prefill-t_engine:.1f}s")
    fname=name[:-5]+".safetensors"
    t=time.time(); local=os.path.join(LOCAL_CAP,fname)
    with urllib.request.urlopen(base+"/"+fname,timeout=PD_LONG_TIMEOUT) as rr, open(local,"wb") as f:
        while True:
            b=rr.read(1<<24)
            if not b: break
            f.write(b)
    t_xfer=time.time()-t
    return local, t_prefill, t_xfer

def fast_update(attn, x, c):
    """Projection-only cache update — proven bit-exact vs the full forward (S3, 313/313 arrays, 23,217 tok in 2.0 s)."""
    B,L,_=x.shape
    if attn.compress_ratio==0:
        local=c; offset=local.offset
        kv=attn.kv_norm(attn.wkv(x)).reshape(B,1,L,attn.head_dim); kv=attn.rope(kv,offset)
        kv,_=local.update_and_fetch(kv, mx.zeros((B,1,L,0))); mx.eval(kv); return
    local=c[0]; offset=local.offset
    kv=attn.kv_norm(attn.wkv(x)).reshape(B,1,L,attn.head_dim); kv=attn.rope(kv,offset)
    kv,_=local.update_and_fetch(kv, mx.zeros((B,1,L,0)))
    pooled=attn.compressor(x, c[1], offset); outs=[kv,pooled]
    if len(c.caches)>2: outs.append(attn.indexer.compressor(x, c[2], offset))
    mx.eval(*outs)

_WRITER=None
def rebuild(local_cap, ids, chunk=2048):
    """Fast (projection-only) rebuild with BlockWriter snapshots at every 2048 boundary."""
    global _WRITER
    from omlx_block_writer import BlockWriter
    caps=mx.load(local_cap); T=len(ids); assert caps["layer_00"].shape[0]==T, (caps["layer_00"].shape, T)
    w=BlockWriter(model_name=PD_MODEL_NAME, out_dir=PD_CACHE_DIR, cache_list_factory=MODEL.make_cache); _WRITER=w
    cache=MODEL.make_cache(); layers=MODEL.model.layers; done=0
    while done<T:
        n=min(chunk,T-done)
        for i,layer in enumerate(layers):
            fast_update(layer.attn, caps[f"layer_{i:02d}"][done:done+n][None], cache[i])
        done+=n
        if done % chunk == 0: w.snapshot(cache, done)
    return cache
def write_blocks_from_cache(cache, ids):
    return _WRITER.finalize(ids)

import numpy as np
def _ls(base, sub=""):
    return json.load(urllib.request.urlopen(f"{base}/_ls/{sub}", timeout=10))

def _fetch_range(base, path, start, end):
    """bytes [start, end) via HTTP Range."""
    rq=urllib.request.Request(f"{base}/{path}", headers={"Range": f"bytes={start}-{end-1}"})
    with urllib.request.urlopen(rq, timeout=PD_LONG_TIMEOUT) as r:
        out=bytearray()
        while True:
            b=r.read(1<<24)
            if not b: break
            out+=b
    return bytes(out)

def spark_prefill_pipelined(ids, chunk=2048):
    """Run the Spark prefill and, while it runs, pull each layer's rows as they land (hook v2 appends
    per-layer .bin files), rebuild the caches boundary-by-boundary, and snapshot for the block writer.
    Returns (cache, writer, timings)."""
    from omlx_block_writer import BlockWriter
    base=PD_SPARK.rsplit(":",1)[0]+f":{PD_SHARE_PORT}"; T=len(ids); ROW=4096*2
    before={e["name"] for e in _ls(base) if e["dir"]}
    tm={"t0":time.time()}
    eng={}
    def _engine():
        body=json.dumps({"model":PD_SPARK_MODEL,"prompt":ids,"max_tokens":1,"temperature":0}).encode()
        try:
            r=urllib.request.urlopen(urllib.request.Request(PD_SPARK+"/v1/completions",body,{"Content-Type":"application/json"}),timeout=PD_LONG_TIMEOUT); r.read(); eng["t"]=time.time()-tm["t0"]
        except Exception as e: eng["err"]=repr(e)
    th=threading.Thread(target=_engine,daemon=True); th.start()
    stamp=None
    while stamp is None:
        if "err" in eng: raise RuntimeError("spark engine: "+eng["err"])
        new=[e["name"] for e in _ls(base) if e["dir"] and e["name"] not in before]
        if new: stamp=sorted(new)[-1]; tm["stamp_seen"]=time.time()-tm["t0"]
        else: time.sleep(0.1)
    L(f"bridge: capture dir {stamp} appeared +{tm['stamp_seen']:.1f}s")
    w=BlockWriter(model_name=PD_MODEL_NAME, out_dir=PD_CACHE_DIR, cache_list_factory=MODEL.make_cache)
    cache=MODEL.make_cache(); layers=MODEL.model.layers; NL=len(layers)
    pulled=[0]*NL; bufs=[[] for _ in range(NL)]  # per layer: list of np.uint16 [rows,4096]
    nb=(T+chunk-1)//chunk; next_k=0; pulled_bytes=0; t_first=None
    while next_k<nb:
        prog=False
        try: sizes={e["name"]:e["size"] for e in _ls(base, stamp)}
        except Exception: sizes={}
        for li in range(NL):
            avail=min(sizes.get(f"layer_{li:02d}.bin",0)//ROW, T)
            if avail>pulled[li]:
                data=_fetch_range(base, f"{stamp}/layer_{li:02d}.bin", pulled[li]*ROW, avail*ROW)
                if t_first is None: t_first=time.time()-tm["t0"]
                bufs[li].append(np.frombuffer(data,dtype=np.uint16).reshape(-1,4096)); pulled[li]=avail; pulled_bytes+=len(data); prog=True
        # rebuild every boundary whose rows every layer has
        while next_k<nb and all(pulled[li]>=min((next_k+1)*chunk,T) for li in range(NL)):
            s0,s1=next_k*chunk,min((next_k+1)*chunk,T)
            for li in range(NL):
                buf=bufs[li] if len(bufs[li])==1 else [np.concatenate(bufs[li])]; bufs[li]=buf
                x=mx.array(buf[0][s0:s1]).view(mx.bfloat16)[None]
                fast_update(layers[li].attn, x, cache[li])
            next_k+=1
            if s1%chunk==0: w.snapshot(cache, s1)
            prog=True
        if not prog:
            if "err" in eng: raise RuntimeError("spark engine: "+eng["err"])
            time.sleep(0.05)
    th.join(timeout=600)
    tm.update({"t_engine":round(eng.get("t",-1),2),"t_first_bytes":round(t_first or -1,2),"t_rebuild_done":round(time.time()-tm["t0"],2),"pulled_gb":round(pulled_bytes/1e9,2)})
    return cache, w, tm

def _cap_manifest(base, stamp):
    with urllib.request.urlopen(f"{base}/{stamp}/manifest.json", timeout=30) as r:
        return json.loads(r.read())

def _cap_usable_prefix(man):
    """Tokens of contiguous-from-0 capture data this manifest promises. 0 = untrustworthy (a mid-request
    flush fragment or a capture with position gaps can't seed blocks from token 0)."""
    if man.get("partial_start"): return 0
    if man.get("position_gaps"): return 0
    return int(man.get("T") or 0)

def spark_prefill_pooled(ids):
    """v3: run the Spark prefill; the hook writes a pooled capture (see DESIGN-v3-pooled.md).
    Scan every new DONE stamp, validate its manifest against the request, and use the FIRST COMPLETE
    capture (manifest T == request T, no partial_start, no position_gaps). If the engine finishes and no
    complete capture exists after PD_CAPTURE_GRACE, salvage the best contiguous-prefix candidate (the
    assemble step verifies keys per boundary and writes the longest good prefix, so a capture that lost
    its tail chunk still yields a prefix hit) or decline loudly. Returns (writer_paths, timings); raises
    on decline — the caller falls back to native and X-PD-Bridge carries the reason."""
    from pd_assemble_blocks import assemble_and_write
    base=PD_SPARK.rsplit(":",1)[0]+f":{PD_SHARE_PORT}"; T=len(ids)
    before={e["name"] for e in _ls(base) if e["dir"]}
    tm={"t0":time.time(),"transport":PD_TRANSPORT}; eng={}
    def _engine():
        body=json.dumps({"model":PD_SPARK_MODEL,"prompt":ids,"max_tokens":1,"temperature":0}).encode()
        try:
            r=urllib.request.urlopen(urllib.request.Request(PD_SPARK+"/v1/completions",body,{"Content-Type":"application/json"}),timeout=PD_LONG_TIMEOUT); r.read(); eng["t"]=time.time()-tm["t0"]
        except Exception as e: eng["err"]=repr(e)
    th=threading.Thread(target=_engine,daemon=True); th.start()
    scanned={}; chosen=None; t_eng_done=None
    while True:
        try: dirs=sorted(e["name"] for e in _ls(base) if e["dir"] and e["name"] not in before)
        except Exception as e: dirs=[]; L(f"bridge: capture share unreachable ({e!r}) — retrying")
        if dirs and "stamp_seen" not in tm: tm["stamp_seen"]=round(time.time()-tm["t0"],2)
        for stamp in dirs:
            if stamp in scanned: continue
            try:
                names={e["name"] for e in _ls(base, stamp)}
                if "DONE" not in names or "manifest.json" not in names: continue
                man=_cap_manifest(base, stamp)
            except Exception: continue
            up=_cap_usable_prefix(man)
            scanned[stamp]=(up,man)
            L(f"bridge: capture {stamp}: manifest T={man.get('T')} usable_prefix={up} flush={man.get('flush_reason')} calls={man.get('calls')} bytes={man.get('bytes')} gaps={bool(man.get('position_gaps'))} partial_start={man.get('partial_start')}")
            if man.get("T")==T and up==T:
                chosen=stamp; break
        if chosen:
            tm["t_done_seen"]=round(time.time()-tm["t0"],2); break
        if "err" in eng: raise RuntimeError("spark engine: "+eng["err"])
        if eng.get("t") is not None:
            if t_eng_done is None:
                t_eng_done=time.time(); tm["t_engine"]=round(eng["t"],2)
                L(f"bridge: engine done +{eng['t']:.1f}s; signaling FLUSH_NOW; waiting up to {PD_CAPTURE_GRACE:.0f}s for the final capture flush")
                # STRUCTURAL FIX (9/6, bench5 seed 713): the hook's idle watcher cannot distinguish an
                # inter-chunk scheduler gap (GPU idle >2s, guards aligned and satisfied) from request
                # completion — only WE know the prefill is over, because our engine call returned. Say so.
                try:
                    urllib.request.urlopen(base+"/_flush", timeout=5).read(); tm["t_flush_signal"]=round(time.time()-tm["t0"],2)
                except Exception as e:
                    L(f"bridge: /_flush signal failed ({e!r}) — hook idle/new-request backstops apply")
            elif time.time()-t_eng_done>PD_CAPTURE_GRACE: break
        elif time.time()-tm["t0"]>stamp_deadline(T):
            raise RuntimeError(f"no complete capture {stamp_deadline(T):.0f}s after request start (T={T}, floor {PD_STAMP_TIMEOUT:.0f}s + T/{PD_STAMP_TOK_RATE:.0f}) while the engine still runs — capture share down or hook dead?")
        time.sleep(0.2)
    verdict="complete"; stamp=chosen
    if stamp is None:
        cands=sorted(((up,s) for s,(up,man) in scanned.items() if up>0))
        best_up,best_s=(cands[-1] if cands else (0,None))
        if best_s is None or best_up < PD_MIN_COVERAGE*T:
            detail="; ".join(f"{s}: T={man.get('T')} up={up} flush={man.get('flush_reason')}" for s,(up,man) in sorted(scanned.items())) or "no captures at all"
            raise RuntimeError(f"bridge declined: no usable capture for T={T} after engine completion + {PD_CAPTURE_GRACE:.0f}s grace. Candidates — {detail}. Likely the hook's mid-request idle flush during chunked prefill (FINDING-bench4-cold-fallback.md root cause A).")
        stamp=best_s; verdict=f"salvage {best_up}/{T}"; tm["t_done_seen"]=round(time.time()-tm["t0"],2)
        L(f"bridge: no complete capture; salvaging {stamp} (usable prefix {best_up}/{T} = {best_up/T:.0%})")
    return _pull_and_assemble(base, stamp, ids, tm, eng, th, verdict)

def _pull_and_assemble(base, stamp, ids, tm, eng, th, verdict):
    """Pull a DONE pooled capture's layer files and assemble every boundary (the one-shot path)."""
    from pd_assemble_blocks import assemble_and_write
    T=len(ids)
    local=os.path.join(PD_PULL_DIR, stamp); os.makedirs(local, exist_ok=True); pulled=0
    for e in _ls(base, stamp):
        if e["dir"] or e["name"]=="DONE" or e["name"].startswith("seg_"): continue
        with urllib.request.urlopen(f"{base}/{stamp}/{e['name']}", timeout=PD_LONG_TIMEOUT) as r, open(os.path.join(local,e["name"]),"wb") as f:
            while True:
                b=r.read(1<<24)
                if not b: break
                f.write(b); pulled+=len(b)
    tm["t_pulled"]=round(time.time()-tm["t0"],2); tm["pulled_gb"]=round(pulled/1e9,3)
    paths,ainfo=assemble_and_write(local, ids, MODEL, PD_MODEL_NAME, PD_CACHE_DIR)
    th.join(timeout=60)
    tm.update({"t_engine":round(eng.get("t",-1),2),"t_assembled":round(time.time()-tm["t0"],2),
               "boundaries_ok":ainfo["boundaries_ok"],"B":ainfo["B"],"coverage":ainfo["coverage"]})
    if "missing_at" in ainfo: tm["missing_at"]=ainfo["missing_at"]
    cov=ainfo["coverage"]
    if not paths or cov<PD_MIN_COVERAGE:
        raise RuntimeError(f"bridge declined: assembled {ainfo['B']}/{T} tokens ({cov:.0%}) < {PD_MIN_COVERAGE:.0%} minimum — capture incomplete (missing_at={ainfo.get('missing_at')})")
    full_B=(T//PD_BLOCK)*PD_BLOCK  # B<full_B means genuinely missing boundaries; B==full_B with a sub-boundary tail is the NORMAL complete case (oMLX prefills the tail natively by design)
    if ainfo["B"]<full_B: verdict=f"partial {ainfo['B']}/{T}"
    tm["verdict"]=verdict
    try: shutil.rmtree(local)
    except Exception: pass
    return paths, tm

def spark_prefill_stream(ids):
    """PD_STREAM=1 (docs/STREAMING-CAPTURE.md): the hook ships seg_<b>.safetensors for every finished 2048-token
    boundary WHILE the prefill runs; we pull each one the moment it is listed, build that boundary's decoder state
    and store its block through oMLX's own writer — so the Spark never holds more than one chunk of capture and the
    block assembly overlaps the prefill instead of following it. manifest.json + DONE still arrive last; the manifest's
    `stream.segments` must match what we consumed (check_manifest) or the verdict says so. If the capture that appears
    is NOT a streaming one (hook launched without PD_STREAM=1) this falls back to the one-shot pull+assemble."""
    from pd_assemble_blocks import StreamAssembler
    from pd_stream_assembler import ready_segments, seg_name
    base=PD_SPARK.rsplit(":",1)[0]+f":{PD_SHARE_PORT}"; T=len(ids)
    before={e["name"] for e in _ls(base) if e["dir"]}
    tm={"t0":time.time(),"transport":PD_TRANSPORT,"stream":True}; eng={}
    def _engine():
        body=json.dumps({"model":PD_SPARK_MODEL,"prompt":ids,"max_tokens":1,"temperature":0}).encode()
        try:
            r=urllib.request.urlopen(urllib.request.Request(PD_SPARK+"/v1/completions",body,{"Content-Type":"application/json"}),timeout=PD_LONG_TIMEOUT); r.read(); eng["t"]=time.time()-tm["t0"]
        except Exception as e: eng["err"]=repr(e)
    th=threading.Thread(target=_engine,daemon=True); th.start()
    # the capture dir appears with the first ingested chunk; its start manifest says whether the hook streams
    stamp=None; streaming=None
    while stamp is None:
        if "err" in eng: raise RuntimeError("spark engine: "+eng["err"])
        try: new=sorted(e["name"] for e in _ls(base) if e["dir"] and e["name"] not in before)
        except Exception as e: new=[]; L(f"bridge: capture share unreachable ({e!r}) — retrying")
        for cand in reversed(new):
            try: man0=_cap_manifest(base, cand)
            except Exception: continue
            stamp=cand; streaming=bool(man0.get("stream")); tm["stamp_seen"]=round(time.time()-tm["t0"],2); break
        if stamp is None:
            if time.time()-tm["t0"]>stamp_deadline(T):
                raise RuntimeError(f"no capture directory {stamp_deadline(T):.0f}s after request start (T={T}) while the engine still runs — capture share down or hook dead?")
            time.sleep(0.2)
    L(f"bridge(stream): capture {stamp} appeared +{tm['stamp_seen']}s, hook streaming={streaming}")
    if not streaming:
        L("bridge(stream): the hook is not in PD_STREAM mode — falling back to the one-shot pooled path for this request")
        tm["stream"]=False
        return _wait_done_then_pull(base, stamp, ids, tm, eng, th)
    local=os.path.join(PD_PULL_DIR, stamp); os.makedirs(local, exist_ok=True); pulled=0
    asm=StreamAssembler(ids, MODEL, PD_MODEL_NAME, PD_CACHE_DIR, block=PD_BLOCK)
    man=None; t_eng_done=None; last_log=0
    while True:
        try: names=[e["name"] for e in _ls(base, stamp) if not e["dir"]]
        except Exception as e: names=[]; L(f"bridge(stream): share listing failed ({e!r}) — retrying")
        prog=False
        for b,name in ready_segments(names, asm.next_b(), PD_BLOCK):
            if b>T or asm.stopped: break
            dst=os.path.join(local,name); t1=time.time()
            with urllib.request.urlopen(f"{base}/{stamp}/{name}", timeout=PD_LONG_TIMEOUT) as r, open(dst+".tmp","wb") as f:
                while True:
                    chunk=r.read(1<<24)
                    if not chunk: break
                    f.write(chunk); pulled+=len(chunk)
            os.replace(dst+".tmp", dst)
            if "t_first_segment" not in tm: tm["t_first_segment"]=round(time.time()-tm["t0"],2)
            t2=time.time(); stored=asm.on_segment(b, dst); t3=time.time()
            tm["t_pull_s"]=round(tm.get("t_pull_s",0)+(t2-t1),3); tm["t_store_s"]=round(tm.get("t_store_s",0)+(t3-t2),3)
            try: os.unlink(dst)                       # the block is in the oMLX cache; the segment stays on the Spark until acked/pruned
            except Exception: pass
            if stored and PD_STREAM_ACK:
                try: urllib.request.urlopen(f"{base}/_ack/{stamp}/{name}", timeout=5).read()
                except Exception as e: L(f"bridge(stream): ack {name} failed ({e!r})")
            prog=True
            if time.time()-last_log>15: last_log=time.time(); L(f"bridge(stream): boundary {b}/{T} stored (+{time.time()-tm['t0']:.1f}s, pull {tm['t_pull_s']}s, store {tm['t_store_s']}s)")
        if man is None and "DONE" in names and "manifest.json" in names:
            man=_cap_manifest(base, stamp); tm["t_done_seen"]=round(time.time()-tm["t0"],2)
            st=man.get("stream") or {}
            for k in ("segments","emitted_T","segment_bytes","segment_write_s","error"):
                if k in st: tm["spark_stream_"+k]=(len(st[k]) if k=="segments" else st[k])
            tm["spark_T"]=man.get("T"); tm["spark_flush_reason"]=man.get("flush_reason")
        if man is not None:
            pending=[b for b in ((man.get("stream") or {}).get("segments") or []) if b not in asm.consumed() and b<=T and not asm.stopped]
            if not pending: break
            if not prog and seg_name(pending[0]) not in names: break        # sealed, listed, gone from the share: report it
        if "err" in eng: raise RuntimeError("spark engine: "+eng["err"])
        if eng.get("t") is not None and t_eng_done is None:
            t_eng_done=time.time(); tm["t_engine"]=round(eng["t"],2)
            L(f"bridge(stream): engine done +{eng['t']:.1f}s; signaling FLUSH_NOW; {asm.ok} boundaries already stored")
            try: urllib.request.urlopen(base+"/_flush", timeout=5).read(); tm["t_flush_signal"]=round(time.time()-tm["t0"],2)
            except Exception as e: L(f"bridge(stream): /_flush signal failed ({e!r}) — hook idle/new-request backstops apply")
        if man is None and t_eng_done is not None and time.time()-t_eng_done>PD_CAPTURE_GRACE:
            L(f"bridge(stream): no DONE {PD_CAPTURE_GRACE:.0f}s after the engine returned — salvaging the {asm.ok} boundaries already stored"); break
        if not prog: time.sleep(0.2)
    tm["pulled_gb"]=round(pulled/1e9,3)
    paths,ainfo=asm.finalize(man)
    th.join(timeout=60)
    if "t_engine" not in tm and eng.get("t") is not None: tm["t_engine"]=round(eng["t"],2)
    tm.update({"t_assembled":round(time.time()-tm["t0"],2),"boundaries_ok":ainfo["boundaries_ok"],"B":ainfo["B"],"coverage":ainfo["coverage"],
               "segments_consumed":ainfo.get("segments_consumed")})
    for k in ("missing_at","stream_check"):
        if k in ainfo: tm[k]=ainfo[k]
    try: shutil.rmtree(local)
    except Exception: pass
    cov=ainfo["coverage"]
    if not paths or cov<PD_MIN_COVERAGE:
        raise RuntimeError(f"bridge declined: streamed {ainfo['B']}/{T} tokens ({cov:.0%}) < {PD_MIN_COVERAGE:.0%} minimum (manifest={'sealed' if man else 'never arrived'}, missing_at={ainfo.get('missing_at')}, stream_error={(man or {}).get('stream',{}).get('error')})")
    full_B=(T//PD_BLOCK)*PD_BLOCK
    if man is None: verdict=f"salvage {ainfo['B']}/{T}"
    elif ainfo["B"]<full_B: verdict=f"partial {ainfo['B']}/{T}"
    else: verdict="complete"
    tm["verdict"]=verdict
    return paths, tm

def _wait_done_then_pull(base, stamp, ids, tm, eng, th):
    """Streaming front, non-streaming hook: wait for this stamp's DONE (FLUSH_NOW when the engine returns, grace,
    salvage rules as in spark_prefill_pooled) then run the one-shot pull+assemble."""
    T=len(ids); t_eng_done=None; man=None
    while True:
        try: names={e["name"] for e in _ls(base, stamp)}
        except Exception: names=set()
        if "DONE" in names and "manifest.json" in names:
            man=_cap_manifest(base, stamp); tm["t_done_seen"]=round(time.time()-tm["t0"],2); break
        if "err" in eng: raise RuntimeError("spark engine: "+eng["err"])
        if eng.get("t") is not None and t_eng_done is None:
            t_eng_done=time.time(); tm["t_engine"]=round(eng["t"],2)
            try: urllib.request.urlopen(base+"/_flush", timeout=5).read(); tm["t_flush_signal"]=round(time.time()-tm["t0"],2)
            except Exception as e: L(f"bridge: /_flush signal failed ({e!r})")
        if t_eng_done is not None and time.time()-t_eng_done>PD_CAPTURE_GRACE:
            raise RuntimeError(f"bridge declined: engine returned {PD_CAPTURE_GRACE:.0f}s ago and capture {stamp} never sealed")
        time.sleep(0.2)
    up=_cap_usable_prefix(man)
    if up<PD_MIN_COVERAGE*T:
        raise RuntimeError(f"bridge declined: capture {stamp} usable prefix {up}/{T} (flush={man.get('flush_reason')})")
    return _pull_and_assemble(base, stamp, ids, tm, eng, th, "complete" if up==T else f"salvage {up}/{T}")

_RDMA=None
class _RdmaClient:
    """R1 (docs/RDMA.md): a long-lived `pd_rdma client` co-process. It registers its arena once and keeps one
    QP to `pd_rdma serve` on the Spark; each pull is one line in, one JSON line out. A dead client is restarted once,
    and a failed pull declines the bridge — an RDMA row must never silently become a TCP row."""
    def __init__(self):
        here=os.path.dirname(os.path.abspath(__file__))
        self.cmd=[os.environ.get("PD_RDMA_BIN", os.path.join(here,"..","rdma","pd_rdma")), "client",
                  "--host", os.environ.get("PD_RDMA_HOST", PD_SPARK.split("//")[-1].rsplit(":",1)[0]),
                  "--port", os.environ.get("PD_RDMA_PORT","18515"), "--dev", os.environ.get("PD_RDMA_DEV","mlx5_0"),
                  "--gid-index", os.environ.get("PD_RDMA_GID_INDEX","0"), "--arena-mib", os.environ.get("PD_RDMA_ARENA_MIB","128")]
        self.p=None
    def _start(self):
        self.p=subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=LOG, text=True, bufsize=1)
        first=self.p.stdout.readline()
        if not first or not json.loads(first).get("ready"):
            raise RuntimeError(f"pd_rdma client did not come up: {first.strip()!r} (entitlement signed? MELONDMA_* set? serve running?)")
        L("pd_rdma client ready:", first.strip())
    def pull(self, tag, outdir):
        for attempt in (0,1):
            if self.p is None or self.p.poll() is not None: self._start()
            try:
                self.p.stdin.write(f"PULL {tag} {outdir}\n"); self.p.stdin.flush()
                line=self.p.stdout.readline()
                if line: return json.loads(line)
            except (BrokenPipeError, ValueError) as e:
                L(f"pd_rdma client pull failed ({e!r}), attempt {attempt}")
            self.p=None
        raise RuntimeError("pd_rdma client died twice")

def spark_prefill_kv(ids):
    """R0 (docs/RDMA.md): plain-attention capture from spark/pd_kv_connector.py. The request carries its own
    pd_tag, so the capture directory is known up front — no before/after listing race, no stamp guessing. Poll
    <tag>/ for DONE, validate the manifest against the request, pull every block, assemble into KVCache and store
    through oMLX's own writer. Raises on decline; the caller serves natively and X-PD-Bridge carries the reason."""
    from pd_assemble_kv import assemble_and_write_kv
    import uuid
    base=PD_SPARK.rsplit(":",1)[0]+f":{PD_SHARE_PORT}"; T=len(ids); tag="pd"+uuid.uuid4().hex[:16]
    tm={"t0":time.time(),"tag":tag,"transport":PD_TRANSPORT}; eng={}
    def _engine():
        body=json.dumps({"model":PD_SPARK_MODEL,"prompt":ids,"max_tokens":1,"temperature":0,"kv_transfer_params":{"pd_tag":tag}}).encode()
        try:
            r=urllib.request.urlopen(urllib.request.Request(PD_SPARK+"/v1/completions",body,{"Content-Type":"application/json"}),timeout=PD_LONG_TIMEOUT); r.read(); eng["t"]=time.time()-tm["t0"]
        except Exception as e: eng["err"]=repr(e)
    th=threading.Thread(target=_engine,daemon=True); th.start()
    polls=0; t_eng_done=None; man=None
    while True:
        try: names={e["name"] for e in _ls(base, tag)}
        except Exception: names=set()          # the directory appears with the first captured step
        polls+=1
        if names and "t_first_seen" not in tm: tm["t_first_seen"]=round(time.time()-tm["t0"],2)
        if "DONE" in names and "manifest.json" in names:
            man=_cap_manifest(base, tag); tm["t_done_seen"]=round(time.time()-tm["t0"],2); break
        if "err" in eng: raise RuntimeError("spark engine: "+eng["err"])
        if eng.get("t") is not None and t_eng_done is None:
            t_eng_done=time.time(); tm["t_engine"]=round(eng["t"],2)
        if t_eng_done is not None and time.time()-t_eng_done>PD_CAPTURE_GRACE:
            raise RuntimeError(f"bridge declined: engine returned {PD_CAPTURE_GRACE:.0f}s ago and capture {tag} never committed (connector not loaded? prefix caching on?)")
        if t_eng_done is None and not names and time.time()-tm["t0"]>PD_STAMP_TIMEOUT:
            raise RuntimeError(f"bridge declined: no capture directory for {tag} after {PD_STAMP_TIMEOUT:.0f}s")
        time.sleep(0.2)
    tm["polls"]=polls
    for k in ("t_capture_span_s","t_forward_wait_s","t_gather_sync_s","t_d2h_s","t_write_s","steps","complete","position_gaps"):   # t_gather_sync_s: pre-rename captures
        if k in man: tm["spark_"+k]=man[k]
    if man.get("prompt_len")!=T:
        raise RuntimeError(f"bridge declined: capture {tag} is for {man.get('prompt_len')} tokens, request has {T} (render parity broken?)")
    t0=time.time(); local=os.path.join(PD_PULL_DIR, tag); os.makedirs(local, exist_ok=True); pulled=0
    if PD_TRANSPORT=="rdma":
        global _RDMA
        if _RDMA is None: _RDMA=_RdmaClient()
        rr=_RDMA.pull(tag, local)
        if not rr.get("ok"):
            raise RuntimeError(f"bridge declined: rdma pull of {tag} failed ({rr.get('error')}) — not falling back to TCP")
        pulled=int(rr.get("bytes",0))
        for k in ("t_server_read_s","t_wire_s","t_client_write_s","wire_gb_per_s","files"): tm["rdma_"+k]=rr.get(k)
    else:
        for e in _ls(base, tag):
            if e["dir"] or e["name"]=="DONE": continue
            with urllib.request.urlopen(f"{base}/{tag}/{e['name']}", timeout=600) as r, open(os.path.join(local,e["name"]),"wb") as f:
                while True:
                    b=r.read(1<<24)
                    if not b: break
                    f.write(b); pulled+=len(b)
    tm["t_pull"]=round(time.time()-t0,2); tm["pulled_gb"]=round(pulled/1e9,3)
    try:
        paths,ainfo=assemble_and_write_kv(local, ids, MODEL, PD_MODEL_NAME, PD_CACHE_DIR)
    finally:
        shutil.rmtree(local, ignore_errors=True)
    th.join(timeout=60)
    tm.update(ainfo); tm["t_assembled"]=round(time.time()-tm["t0"],2)
    if "t_engine" not in tm and eng.get("t") is not None: tm["t_engine"]=round(eng["t"],2)
    cov=ainfo["coverage"]
    if not paths or cov<PD_MIN_COVERAGE:
        raise RuntimeError(f"bridge declined: assembled {ainfo['B']}/{T} tokens ({cov:.0%}) < {PD_MIN_COVERAGE:.0%} minimum ({ainfo.get('stopped_at')})")
    full_B=(T//PD_BLOCK)*PD_BLOCK   # the sub-block tail is always prefilled natively; B<full_B means blocks are genuinely missing
    tm["verdict"]="complete" if ainfo["B"]==full_B else f"partial {ainfo['B']}/{T}"
    return paths, tm

def spark_prefill_kv_stream(ids):
    """R2/R4 (docs/RDMA.md): the connector RDMA-writes the capture to `pd_rdma recvd` on this Mac while the
    prefill runs; waiting is a local stat, the manifest arrives after the last block and DONE after the manifest.
      rdma2  blocks land in PD_PULL_DIR/<tag>/ and are assembled + stored through oMLX's own writer here.
      rdma4  the Spark builds oMLX-native block files (chain hash, header, layer tensors) and the receiver lands them
             straight in PD_CACHE_DIR: nothing is assembled or re-stored — verify the hashes are on disk and forward."""
    from pd_assemble_kv import assemble_and_write_kv
    import uuid
    T=len(ids); tag="pd"+uuid.uuid4().hex[:16]; local=os.path.join(PD_PULL_DIR, tag); omlx=PD_TRANSPORT=="rdma4"
    tm={"t0":time.time(),"tag":tag,"transport":PD_TRANSPORT}; eng={}
    params={"pd_tag":tag}
    if omlx: params.update({"omlx_model":PD_MODEL_NAME,"omlx_block":PD_BLOCK})   # the connector hashes the ids exactly as oMLX will
    if omlx and os.environ.get("PD_OMLX_STAGE")=="1":
        # O2c: the oMLX hook (studio/pd_omlx_hooks.py) allocates the request's cache and fills it while the Spark prefills
        try:
            from omlx_block_writer import chain_hashes_for
            hs=[h if isinstance(h,str) else h.hex() for h in chain_hashes_for(ids, PD_MODEL_NAME, block_size=PD_BLOCK)][:T//PD_BLOCK]
            cfg=json.load(open(os.path.join(PD_MODEL,"config.json")))
            body=json.dumps({"hashes":hs,"layers":cfg["num_hidden_layers"],"kv_heads":cfg["num_key_value_heads"],
                             "head_dim":cfg.get("head_dim") or cfg["hidden_size"]//cfg["num_attention_heads"],"block":PD_BLOCK,
                             "reserve":int(os.environ.get("PD_OMLX_RESERVE_TOKENS","1024"))}).encode()
            tm["stage"]=json.load(urllib.request.urlopen(urllib.request.Request(PD_OMLX+"/pd/stage",body,{"Content-Type":"application/json"}),timeout=10))
        except Exception as e:
            tm["stage_error"]=repr(e); L(f"bridge: /pd/stage failed ({e!r}); the decoder restores from disk")
    def _engine():
        body=json.dumps({"model":PD_SPARK_MODEL,"prompt":ids,"max_tokens":1,"temperature":0,"kv_transfer_params":params}).encode()
        try:
            r=urllib.request.urlopen(urllib.request.Request(PD_SPARK+"/v1/completions",body,{"Content-Type":"application/json"}),timeout=PD_LONG_TIMEOUT); r.read(); eng["t"]=time.time()-tm["t0"]
        except Exception as e: eng["err"]=repr(e)
    th=threading.Thread(target=_engine,daemon=True); th.start()
    t_eng_done=None
    while not os.path.isfile(os.path.join(local,"DONE")):
        if "t_first_seen" not in tm and os.path.isdir(local): tm["t_first_seen"]=round(time.time()-tm["t0"],2)
        if "err" in eng: raise RuntimeError("spark engine: "+eng["err"])
        if eng.get("t") is not None and t_eng_done is None:
            t_eng_done=time.time(); tm["t_engine"]=round(eng["t"],2)
        if t_eng_done is not None and time.time()-t_eng_done>PD_CAPTURE_GRACE:
            shutil.rmtree(local, ignore_errors=True)
            raise RuntimeError(f"bridge declined: engine returned {PD_CAPTURE_GRACE:.0f}s ago and no RDMA capture for {tag} arrived (pd_rdma recvd down? connector fell back to files?)")
        time.sleep(0.02)
    tm["t_done_seen"]=round(time.time()-tm["t0"],2)
    try:
        man=json.load(open(os.path.join(local,"manifest.json")))
        for k in ("transport","landed","async","t_capture_span_s","t_forward_wait_s","t_stack_s","t_enqueue_wait_s","t_copy_s","t_wire_s","t_ack_s",
                  "t_sender_lag_s","t_d2h_s","steps","blocks_sent","complete","position_gaps","error","cuda_host_registered"):
            if k in man: tm["spark_"+k]=man[k]
        if man.get("prompt_len")!=T:
            raise RuntimeError(f"bridge declined: capture {tag} is for {man.get('prompt_len')} tokens, request has {T} (render parity broken?)")
        if omlx:
            if man.get("transport")!="omlx":
                raise RuntimeError(f"bridge declined: connector sent {man.get('transport')} blocks, not oMLX-native ones (omlx_model/omlx_block rejected? see the vLLM log)")
            # nothing to assemble: the blocks are already the decoder's own files — prove they are where it will look
            t1=time.time(); B=cached_prefix_tokens(ids); tm["t_verify_on_disk"]=round(time.time()-t1,3)
            n=B//PD_BLOCK
            from omlx_block_writer import chain_hashes_for
            paths=[os.path.join(PD_CACHE_DIR,h[0],h+".safetensors") for h in chain_hashes_for(ids, PD_MODEL_NAME, block_size=PD_BLOCK)[:n]]
            ainfo={"blocks_ok":n,"blocks_claimed":int(man.get("blocks",0)),"B":B,"coverage":round(B/T,4) if T else 0.0,
                   "T_manifest":man.get("T"),"T_request":T,"landed_gb":round(sum(os.path.getsize(p) for p in paths)/1e9,3)}
            # O2b: rows after the last full block (all but the last token) -> pd_tail/<sha256(ids)> for the oMLX hook
            tail_src=os.path.join(local,"tail.safetensors"); tm["tail_rows"]=int(man.get("tail_rows_shipped") or 0)
            if tm["tail_rows"] and os.path.isfile(tail_src) and B==n*PD_BLOCK and B+tm["tail_rows"]==T-1:
                import hashlib, struct
                tail_dir=os.path.join(PD_CACHE_DIR,"pd_tail"); os.makedirs(tail_dir, exist_ok=True)
                os.replace(tail_src, os.path.join(tail_dir, hashlib.sha256(struct.pack(f"<{T}i",*ids)).hexdigest()+".safetensors"))
                tm["tail_landed"]=True
        else:
            tm["pulled_gb"]=round(sum(os.path.getsize(os.path.join(local,f)) for f in os.listdir(local))/1e9,3)
            paths,ainfo=assemble_and_write_kv(local, ids, MODEL, PD_MODEL_NAME, PD_CACHE_DIR)
    finally:
        shutil.rmtree(local, ignore_errors=True)
    th.join(timeout=60)
    tm.update(ainfo); tm["t_assembled"]=round(time.time()-tm["t0"],2)
    if "t_engine" not in tm and eng.get("t") is not None: tm["t_engine"]=round(eng["t"],2)
    cov=ainfo["coverage"]
    if not paths or cov<PD_MIN_COVERAGE:
        raise RuntimeError(f"bridge declined: assembled {ainfo['B']}/{T} tokens ({cov:.0%}) < {PD_MIN_COVERAGE:.0%} minimum ({ainfo.get('stopped_at')})")
    full_B=(T//PD_BLOCK)*PD_BLOCK
    tm["verdict"]="complete" if ainfo["B"]==full_B else f"partial {ainfo['B']}/{T}"
    return paths, tm

def bridge(raw_json):
    """Returns timing dict; raises on failure (caller falls back). Single-threaded server: MLX lives on this thread."""
    with LOCK, mx.stream(mx.default_stream(mx.Device(mx.gpu))):
        t=time.time(); ids,_m=render_request(raw_json); t_tok=time.time()-t
        cached=cached_prefix_tokens(ids); L(f"bridge: {len(ids)} tokens (render {t_tok:.2f}s), cached prefix {cached}")
        if len(ids)<PD_MIN_TOKENS: return {"skipped":True,"tokens":len(ids),"why":"short"}
        tail=len(ids)-cached
        if PD_MODE=="kv" and not os.environ.get("PD_IGNORE_CACHED") and tail>=PD_MIN_TAIL:
            paths,tm=(spark_prefill_kv_stream if PD_TRANSPORT in ("rdma2","rdma4") else spark_prefill_kv)(ids)   # rdma2/rdma4: blocks arrive during prefill
            moved=f"landed {tm.get('landed_gb')} GB in the oMLX cache" if PD_TRANSPORT=="rdma4" else f"received {tm.get('pulled_gb')} GB" if PD_TRANSPORT=="rdma2" else f"pulled {tm.get('pulled_gb')} GB in {tm.get('t_pull')}s"
            L(f"bridge(kv/{PD_TRANSPORT}): engine {tm.get('t_engine')}s, DONE +{tm.get('t_done_seen')}s, {moved}, blocks {len(paths)}, done +{tm.get('t_assembled')}s, verdict {tm.get('verdict')}")
            return {"skipped":False,"mode":"kv","tokens":len(ids),"cached_prefix":cached,"t_tokenize":round(t_tok,2),**{k:v for k,v in tm.items() if k!="t0"},"blocks":len(paths)}
        if not os.environ.get("PD_IGNORE_CACHED"):
            if PD_MODE=="kv":
                return {"skipped":True,"tokens":len(ids),"cached_prefix":cached,"tail":tail,"transport":PD_TRANSPORT,"why":f"warm — new tail {tail} < {PD_MIN_TAIL}, oMLX prefills it natively"}
            if PD_MODE=="pooled":
                # v3: the Spark re-prefills the whole prompt cheaply and ships ~10 KB/token, so bridge whenever the NEW part is big.
                if tail<PD_MIN_TAIL: return {"skipped":True,"tokens":len(ids),"cached_prefix":cached,"tail":tail,"why":f"warm — new tail {tail} < {PD_MIN_TAIL}, oMLX prefills it natively"}
            elif cached>0: return {"skipped":True,"tokens":len(ids),"cached_prefix":cached,"tail":tail,"why":"warm — oMLX prefills only the tail natively (hidden-state mode bridges cold prompts only)"}
        if PD_MODE=="pooled":
            paths,tm=(spark_prefill_stream if PD_STREAM else spark_prefill_pooled)(ids)   # PD_STREAM=1: blocks land while the Spark prefills
            L(f"bridge(pooled{'/stream' if tm.get('stream') else ''}): engine {tm.get('t_engine')}s, DONE +{tm.get('t_done_seen')}s, "
              f"{'first segment +'+str(tm.get('t_first_segment'))+'s, ' if tm.get('stream') else ''}pulled {tm.get('pulled_gb')} GB by +{tm.get('t_assembled')}s, blocks {len(paths)}, verdict {tm.get('verdict')} via {tm.get('transport')}")
            return {"skipped":False,"mode":"pooled","tokens":len(ids),"t_tokenize":round(t_tok,2),**{k:v for k,v in tm.items() if k!="t0"},"blocks":len(paths)}
        cache,w,tm=spark_prefill_pipelined(ids)
        L(f"bridge: engine {tm['t_engine']}s, first bytes +{tm['t_first_bytes']}s, rebuild done +{tm['t_rebuild_done']}s, pulled {tm['pulled_gb']} GB")
        t=time.time(); paths=w.finalize(ids); t_write=time.time()-t; L(f"bridge: wrote {len(paths)} blocks {t_write:.1f}s")
        return {"skipped":False,"tokens":len(ids),"t_tokenize":round(t_tok,2),"t_engine":tm["t_engine"],"t_first_bytes":tm["t_first_bytes"],"t_rebuild_done":tm["t_rebuild_done"],"t_write":round(t_write,2),"blocks":len(paths),"pulled_gb":tm["pulled_gb"]}

class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def do_GET(self):
        if self.path.startswith("/health"): self._json(200,{"ok":True,"front":"pd","model":PD_MODEL_NAME})
        else: self._proxy_get()
    def _json(self,code,obj):
        b=json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(b))); self.end_headers(); self.wfile.write(b)
    def _proxy_get(self):
        r=urllib.request.urlopen(PD_OMLX+self.path,timeout=60); b=r.read(); self.send_response(r.status); self.send_header("Content-Type",r.headers.get("Content-Type","application/json")); self.send_header("Content-Length",str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        n=int(self.headers.get("Content-Length","0")); raw=self.rfile.read(n)
        info={}
        if self.path.startswith("/v1/chat/completions"):
            try:
                req=json.loads(raw)
                t=time.time(); info=bridge_via_main(req); info["t_bridge_total"]=round(time.time()-t,2); L("bridge",json.dumps(info))
            except Exception as e:
                info={"bridge_error":str(e)[:200]}; L("bridge FAILED, falling back:",traceback.format_exc()[-600:])
                try:
                    dbg=json.loads(raw); open(os.path.expanduser("~/pd_front_lastreq.json"),"w").write(json.dumps({k:(v if k!="messages" else [{"role":m.get("role"),"content_type":type(m.get("content")).__name__,"content_head":(m.get("content") if isinstance(m.get("content"),str) else json.dumps(m.get("content"))[:300])[:300]} for m in v]) for k,v in dbg.items()},indent=1)[:20000])
                except Exception: pass
            info["t_bridge_total"]=round(time.time()-t,2)
        # forward to oMLX (streaming passthrough)
        t=time.time(); rq=urllib.request.Request(PD_OMLX+self.path,raw,{"Content-Type":"application/json"})
        try: r=urllib.request.urlopen(rq,timeout=PD_LONG_TIMEOUT)
        except urllib.error.HTTPError as e: r=e
        self.send_response(r.status)
        self.send_header("Content-Type", r.headers.get("Content-Type","application/json")); self.send_header("Connection","close")
        self.send_header("X-PD-Bridge",json.dumps(info)); self.end_headers()
        first=None
        while True:
            # read1, not read: read(65536) blocks until 64 KiB or EOF, so a short SSE answer reached the client in one
            # burst at the end and every TTFT measured through the front equalled total time (found 2026-09-14)
            chunk=r.read1(65536) if hasattr(r,"read1") else r.read(65536)
            if not chunk: break
            if first is None: first=time.time()-t
            self.wfile.write(chunk); self.wfile.flush()
        try: self.close_connection=True
        except Exception: pass
        L("omlx done", json.dumps({"ttfb_omlx":round(first or 0,2),"total_omlx":round(time.time()-t,2)}))

# ── CONCURRENCY (9/6): threads for HTTP, ONE main thread for MLX ────────────────────────────
# The model and its MLX streams live on the main thread and must stay there — but nothing else does.
# So: handler threads serve /health and /v1/models instantly and stream oMLX passthroughs in parallel
# (oMLX batches concurrent decodes itself, max_concurrent_requests=8), while every bridge() call is
# marshalled to the main thread through this queue and executed one at a time, exactly as before.
# Net effect: a second caller no longer waits for the first to finish decoding; only the cache-building
# step serializes, which is the part that must.
_BRIDGE_Q: "_queue.Queue[tuple]" = _queue.Queue()

def bridge_via_main(req, timeout=PD_LONG_TIMEOUT):
    """Called from a handler thread: run bridge(req) on the main thread, return its result or raise."""
    done = threading.Event(); box = {}
    _BRIDGE_Q.put((req, box, done))
    if not done.wait(timeout): raise RuntimeError("bridge timed out waiting for the main thread")
    if "err" in box: raise box["err"]
    return box["out"]

if __name__=="__main__":
    L(f"pd_front listening :{PD_PORT} → oMLX {PD_OMLX}, Spark {PD_SPARK} (threaded HTTP, MLX on main thread)")
    srv = ThreadingHTTPServer(("0.0.0.0", PD_PORT), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, name="pd_front_http", daemon=True).start()
    while True:                      # the main thread does nothing but MLX work, forever
        req, box, done = _BRIDGE_Q.get()
        try: box["out"] = bridge(req)
        except Exception as e: box["err"] = e
        finally: done.set()
