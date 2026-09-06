#!/usr/bin/env python3
"""pd_front.py — the heterogeneous P/D front door for DeepSeek-V4-Flash.
Runs ON the decode machine next to oMLX. OpenAI-compatible /v1/chat/completions on :8012.
Per request:  tokenize (oMLX's own patched tokenizer/template) → POST ids to the Spark prefill
engine (vLLM TP2, capture hook) → fetch the per-layer attention-input capture (rsync over LAN) →
rebuild oMLX caches locally with the resident attention-only model → write blocks into oMLX's
SSD prefix cache (omlx_block_writer) → forward the ORIGINAL chat request to oMLX (:8011), which
hits the prefix and only decodes. Falls back to plain oMLX on any bridge error (never breaks a reply).
Env: PD_MODEL (MLX model dir), PD_MODEL_NAME (oMLX model id), PD_SPARK ($PD_SPARK),
     PD_SPARK_SSH (user@<prefill-head>), PD_SPARK_CAPDIR (capture dir on the prefill head),
     PD_OMLX (http://127.0.0.1:8011), PD_CACHE_DIR (~/.omlx/cache), PD_MIN_TOKENS (default 4096: below this, skip the bridge),
     PD_MAX_BRIDGE_TOKENS (default 81920: above this, skip the bridge — see the note by its definition)
"""
import json, os, sys, time, glob, subprocess, threading, traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.request
import mlx.core as mx

PD_MODEL=os.environ.get("PD_MODEL", os.path.expanduser("~/models/DV4-Flash-MXFP4-MLX"))
PD_MODEL_NAME=os.environ.get("PD_MODEL_NAME","DV4-Flash-MXFP4-MLX")
PD_SPARK=os.environ["PD_SPARK"]                 # required, e.g. http://10.0.0.11:8000
PD_SPARK_SSH=os.environ["PD_SPARK_SSH"]             # required, e.g. user@10.0.0.11
PD_SPARK_CAPDIR=os.environ["PD_SPARK_CAPDIR"]          # required: capture dir on the prefill head
PD_OMLX=os.environ.get("PD_OMLX","http://127.0.0.1:8011")
PD_CACHE_DIR=os.environ.get("PD_CACHE_DIR",os.path.expanduser("~/.omlx/cache"))
PD_MIN_TOKENS=int(os.environ.get("PD_MIN_TOKENS","4096"))
PD_MIN_TAIL=int(os.environ.get("PD_MIN_TAIL","8192"))   # pooled mode: bridge when the uncached tail is at least this many tokens
# Upper bound on what we will bridge. The block writer holds one materialised
# cumulative cache snapshot per 2048-token boundary until finalize(), so peak
# memory grows QUADRATICALLY with prompt length: ~sum(k)*block*KB_per_token.
# Measured on a 256 GB M3 Ultra with a 156 GB model resident: 39 boundaries
# (81,024 tok, ~16 GB of snapshots) succeeds; 47 boundaries (97,848 tok,
# ~23 GB) exhausts headroom, thrashes for minutes, and writes nothing.
# Above this ceiling we decline the bridge and let the decoder serve natively.
# Raise it only if you have measured the headroom on YOUR machine.
PD_MAX_BRIDGE_TOKENS=int(os.environ.get("PD_MAX_BRIDGE_TOKENS","81920"))  # 40 boundaries
PD_MODE=os.environ.get("PD_MODE","hidden")   # "hidden" = v2 hidden-state capture+rebuild, "pooled" = v3 pooled capture+assemble
PD_PULL_DIR=os.path.expanduser(os.environ.get("PD_PULL_DIR","~/pd_pull"))
# Capture-trust knobs (see docs/FINDING-bench4-cold-fallback.md): the hook can flush MID-REQUEST during
# chunked prefill (DONE + manifest T < request T) and a T-correct capture can still miss tail-boundary
# data. The front validates every DONE capture, prefers a complete one, salvages the best contiguous
# prefix, or declines LOUDLY — a native fallback can never masquerade as a bridged measurement.
PD_CAPTURE_GRACE=float(os.environ.get("PD_CAPTURE_GRACE","45"))    # after engine returns: wait this long for the final capture flush
PD_STAMP_TIMEOUT=float(os.environ.get("PD_STAMP_TIMEOUT","300"))   # no capture stamp at all by here (engine still running) = share/hook dead
PD_MIN_COVERAGE=float(os.environ.get("PD_MIN_COVERAGE","0.5"))     # salvage floor: contiguous captured prefix / request tokens
import shutil
PD_PORT=int(os.environ.get("PD_PORT","8012"))
LOCAL_CAP=os.path.expanduser("~/pd_capture_in"); os.makedirs(LOCAL_CAP,exist_ok=True)
LOG=open(os.path.expanduser("~/pd_front.log"),"a")
def L(*a):
    s=time.strftime("%H:%M:%S ")+" ".join(map(str,a)); print(s,flush=True); LOG.write(s+"\n"); LOG.flush()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch; apply_deepseek_v4_patch()
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import CacheList
import omlx_block_writer  # the block writer: write_blocks(cache_list, token_ids, model_name, out_dir) -> paths

t0=time.time(); MODEL,TOK=load(PD_MODEL, lazy=True)
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
    native=uses_native_reasoning_content(req.model, config_model_type="deepseek_v4", engine_model_type="deepseek_v4",
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
    """How many leading tokens oMLX already holds as SSD blocks for this exact prompt (chain hashes, 2048/block)."""
    try:
        from omlx_block_writer import chain_hashes_for
        hs=chain_hashes_for(ids, PD_MODEL_NAME)
    except Exception as e:
        L("chain_hashes_for unavailable:",e); return 0
    n=0
    for h in hs:
        hx=h.hex() if isinstance(h,(bytes,bytearray)) else str(h)
        if os.path.isfile(os.path.join(PD_CACHE_DIR,hx[0],hx+".safetensors")): n+=1
        else: break
    return n*2048

def _http_listing(base):
    """Newest .json sidecar via the capture HTTP share (no ssh)."""
    import re, html
    txt=urllib.request.urlopen(base+"/",timeout=10).read().decode("utf-8","replace")
    names=[html.unescape(n) for n in re.findall(r'href="([^"]+\.json)"',txt)]
    return sorted(names)

def spark_prefill(ids):
    base=PD_SPARK.rsplit(":",1)[0]+":8010"
    before=set(_http_listing(base))
    body=json.dumps({"model":"deepseek-v4-flash","prompt":ids,"max_tokens":1,"temperature":0}).encode()
    t=time.time(); r=urllib.request.urlopen(urllib.request.Request(PD_SPARK+"/v1/completions",body,{"Content-Type":"application/json"}),timeout=900); r.read()
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
    with urllib.request.urlopen(base+"/"+fname,timeout=900) as rr, open(local,"wb") as f:
        while True:
            b=rr.read(1<<24)
            if not b: break
            f.write(b)
    t_xfer=time.time()-t
    return local, t_prefill, t_xfer

def fast_update(attn, x, c):
    """Projection-only cache update — proven bit-exact vs the full forward (on the Mac decode node, 313/313 arrays, 23,217 tok in 2.0 s)."""
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
    with urllib.request.urlopen(rq, timeout=900) as r:
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
    base=PD_SPARK.rsplit(":",1)[0]+":8010"; T=len(ids); ROW=4096*2
    before={e["name"] for e in _ls(base) if e["dir"]}
    tm={"t0":time.time()}
    eng={}
    def _engine():
        body=json.dumps({"model":"deepseek-v4-flash","prompt":ids,"max_tokens":1,"temperature":0}).encode()
        try:
            r=urllib.request.urlopen(urllib.request.Request(PD_SPARK+"/v1/completions",body,{"Content-Type":"application/json"}),timeout=900); r.read(); eng["t"]=time.time()-tm["t0"]
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
    base=PD_SPARK.rsplit(":",1)[0]+":8010"; T=len(ids)
    before={e["name"] for e in _ls(base) if e["dir"]}
    tm={"t0":time.time()}; eng={}
    def _engine():
        body=json.dumps({"model":"deepseek-v4-flash","prompt":ids,"max_tokens":1,"temperature":0}).encode()
        try:
            r=urllib.request.urlopen(urllib.request.Request(PD_SPARK+"/v1/completions",body,{"Content-Type":"application/json"}),timeout=900); r.read(); eng["t"]=time.time()-tm["t0"]
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
                L(f"bridge: engine done +{eng['t']:.1f}s; waiting up to {PD_CAPTURE_GRACE:.0f}s for the final capture flush")
            elif time.time()-t_eng_done>PD_CAPTURE_GRACE: break
        elif time.time()-tm["t0"]>PD_STAMP_TIMEOUT:
            raise RuntimeError(f"no complete capture {PD_STAMP_TIMEOUT:.0f}s after request start while the engine still runs — capture share down or hook dead?")
        time.sleep(0.2)
    verdict="complete"; stamp=chosen
    if stamp is None:
        cands=sorted(((up,s) for s,(up,man) in scanned.items() if up>0))
        best_up,best_s=(cands[-1] if cands else (0,None))
        if best_s is None or best_up < PD_MIN_COVERAGE*T:
            detail="; ".join(f"{s}: T={man.get('T')} up={up} flush={man.get('flush_reason')}" for s,(up,man) in sorted(scanned.items())) or "no captures at all"
            raise RuntimeError(f"bridge declined: no usable capture for T={T} after engine completion + {PD_CAPTURE_GRACE:.0f}s grace. Candidates — {detail}. Likely the hook's mid-request idle flush during chunked prefill (FINDING-bench4-cold-fallback.md root cause A).")
        stamp=best_s; verdict=f"salvage {best_up}/{T}"
        L(f"bridge: no complete capture; salvaging {stamp} (usable prefix {best_up}/{T} = {best_up/T:.0%})")
    local=os.path.join(PD_PULL_DIR, stamp); os.makedirs(local, exist_ok=True); pulled=0
    for e in _ls(base, stamp):
        if e["dir"] or e["name"]=="DONE": continue
        with urllib.request.urlopen(f"{base}/{stamp}/{e['name']}", timeout=600) as r, open(os.path.join(local,e["name"]),"wb") as f:
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
    full_B=(T//2048)*2048          # B<full_B means genuinely missing boundaries; B==full_B with a sub-boundary tail is the NORMAL complete case (oMLX prefills the tail natively by design)
    if ainfo["B"]<full_B: verdict=f"partial {ainfo['B']}/{T}"
    tm["verdict"]=verdict
    try: shutil.rmtree(local)
    except Exception: pass
    return paths, tm

def bridge(raw_json):
    """Returns timing dict; raises on failure (caller falls back). Single-threaded server: MLX lives on this thread."""
    with LOCK, mx.stream(mx.default_stream(mx.Device(mx.gpu))):
        t=time.time(); ids,_m=render_request(raw_json); t_tok=time.time()-t
        cached=cached_prefix_tokens(ids); L(f"bridge: {len(ids)} tokens (render {t_tok:.2f}s), cached prefix {cached}")
        if len(ids)<PD_MIN_TOKENS: return {"skipped":True,"tokens":len(ids),"why":"short"}
        if len(ids)>PD_MAX_BRIDGE_TOKENS:
            return {"skipped":True,"tokens":len(ids),"limit":PD_MAX_BRIDGE_TOKENS,
                    "why":f"over ceiling — {len(ids)} > PD_MAX_BRIDGE_TOKENS {PD_MAX_BRIDGE_TOKENS}; snapshot memory grows quadratically, serving natively"}
        tail=len(ids)-cached
        if not os.environ.get("PD_IGNORE_CACHED"):
            if PD_MODE=="pooled":
                # v3: the Spark re-prefills the whole prompt cheaply and ships ~10 KB/token, so bridge whenever the NEW part is big.
                if tail<PD_MIN_TAIL: return {"skipped":True,"tokens":len(ids),"cached_prefix":cached,"tail":tail,"why":f"warm — new tail {tail} < {PD_MIN_TAIL}, oMLX prefills it natively"}
            elif cached>0: return {"skipped":True,"tokens":len(ids),"cached_prefix":cached,"tail":tail,"why":"warm — oMLX prefills only the tail natively (hidden-state mode bridges cold prompts only)"}
        if PD_MODE=="pooled":
            paths,tm=spark_prefill_pooled(ids)
            L(f"bridge(pooled): engine {tm['t_engine']}s, DONE +{tm['t_done_seen']}s, pulled {tm['pulled_gb']} GB by +{tm['t_pulled']}s, blocks {len(paths)} by +{tm['t_assembled']}s")
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
                t=time.time(); info=bridge(req); info["t_bridge_total"]=round(time.time()-t,2); L("bridge",json.dumps(info))
            except Exception as e:
                info={"bridge_error":str(e)[:200]}; L("bridge FAILED, falling back:",traceback.format_exc()[-600:])
                try:
                    dbg=json.loads(raw); open(os.path.expanduser("~/pd_front_lastreq.json"),"w").write(json.dumps({k:(v if k!="messages" else [{"role":m.get("role"),"content_type":type(m.get("content")).__name__,"content_head":(m.get("content") if isinstance(m.get("content"),str) else json.dumps(m.get("content"))[:300])[:300]} for m in v]) for k,v in dbg.items()},indent=1)[:20000])
                except Exception: pass
            info["t_bridge_total"]=round(time.time()-t,2)
        # forward to oMLX (streaming passthrough)
        t=time.time(); rq=urllib.request.Request(PD_OMLX+self.path,raw,{"Content-Type":"application/json"})
        try: r=urllib.request.urlopen(rq,timeout=3600)
        except urllib.error.HTTPError as e: r=e
        self.send_response(r.status)
        self.send_header("Content-Type", r.headers.get("Content-Type","application/json")); self.send_header("Connection","close")
        self.send_header("X-PD-Bridge",json.dumps(info)); self.end_headers()
        first=None
        while True:
            chunk=r.read(65536)
            if not chunk: break
            if first is None: first=time.time()-t
            self.wfile.write(chunk); self.wfile.flush()
        try: self.close_connection=True
        except Exception: pass
        L("omlx done", json.dumps({"ttfb_omlx":round(first or 0,2),"total_omlx":round(time.time()-t,2)}))

if __name__=="__main__":
    L(f"pd_front listening :{PD_PORT} → oMLX {PD_OMLX}, Spark {PD_SPARK}")
    HTTPServer(("0.0.0.0",PD_PORT),H).serve_forever()  # single-threaded on purpose: MLX streams are thread-local and the model lives on the main thread
