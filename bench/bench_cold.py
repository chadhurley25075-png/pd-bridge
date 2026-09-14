#!/usr/bin/env python3
"""bench_cold.py — NEUTRAL cold-prefill benchmark for the P/D bridge (no application prompt, no hooks, no cache warmth).
Builds a deterministic synthetic document from Python's own stdlib sources (seeded shuffle, so every seed is a genuinely
cold prompt), asks one short question about it, and measures time-to-first-token and total time through any OpenAI-compatible
endpoint (front door :8012 = bridge, :8011 = native oMLX).
usage: bench_cold.py --chars 330000 --seed 1 [--url http://<decoder>:8012] [--model DV4-Flash-MXFP4-MLX] [--max-tokens 64]"""
import argparse, glob, json, os, random, sys, sysconfig, time, urllib.request
ap=argparse.ArgumentParser(); ap.add_argument("--chars",type=int,default=330000); ap.add_argument("--seed",type=int,default=1)
ap.add_argument("--url",default=os.environ.get("PD_BRIDGE_URL","http://127.0.0.1:8012")); ap.add_argument("--model",default="DV4-Flash-MXFP4-MLX"); ap.add_argument("--max-tokens",type=int,default=300)
a=ap.parse_args()
files=sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"],"*.py"))); random.Random(a.seed).shuffle(files)
doc=[]; n=0
for f in files:
    try: s=open(f,encoding="utf-8",errors="ignore").read()
    except Exception: continue
    doc.append(f"\n\n### FILE {os.path.basename(f)} (seed {a.seed})\n"+s); n+=len(s)
    if n>=a.chars: break
doc="".join(doc)[:a.chars]
marker=f"ZEBRA-{a.seed:04d}-{random.Random(a.seed*7).randint(1000,9999)}"
doc=doc[:len(doc)//2]+f"\n# The secret marker is {marker}.\n"+doc[len(doc)//2:]
msgs=[{"role":"user","content":doc+"\n\nAnswer in one line: what is the secret marker written in the middle of the document above?"}]
# include_usage: the final chunk carries prompt_tokens_details.cached_tokens — direct evidence that the decoder hit the
# bridged prefix, independent of any timing (2026-09-14: timings alone once hid a front door that buffered the stream)
body=json.dumps({"model":a.model,"messages":msgs,"max_tokens":a.max_tokens,"temperature":0,"stream":True,"stream_options":{"include_usage":True}}).encode()
t0=time.time(); first=None; first_content=None; out=[]; usage=None
bridge=None
with urllib.request.urlopen(urllib.request.Request(a.url+"/v1/chat/completions",body,{"Content-Type":"application/json"}),timeout=3600) as r:
    # record the front door's verdict so a native fallback can NEVER enter a results table as "bridged"
    hdr=r.headers.get("X-PD-Bridge")
    if hdr:
        try: bridge=json.loads(hdr)
        except Exception: bridge=hdr[:400]
    for line in r:
        if not line.startswith(b"data: ") or line.strip()==b"data: [DONE]": continue
        try: d=json.loads(line[6:])
        except Exception: continue
        if d.get("usage"): usage=d["usage"]
        choices=d.get("choices") or []          # the usage chunk has an empty choices list
        delta=choices[0].get("delta",{}) if choices else {}
        c=delta.get("content"); rc=delta.get("reasoning_content") or delta.get("reasoning")
        if (c or rc) and first is None: first=time.time()-t0   # first token of any kind = the decoder's TTFT
        if c:
            if first_content is None: first_content=time.time()-t0
            out.append(c)
tot=time.time()-t0; ans="".join(out).strip()
ptd=(usage or {}).get("prompt_tokens_details") or {}
# THE THREE FIELDS EVERY ROW MUST CARRY (bench/BENCHMARK-PROTOCOL.md): verdict — the front door's own word for what
# happened (complete / partial B/T / salvage / skipped / declined / native); transport — the wire the capture crossed
# (tcp10, rdma2, rdma4, or none); cached_tokens — the decoder's own count of prompt tokens it served from cache. A row
# with a fast time and no verdict, or a verdict but cached_tokens ~0, did not measure the bridge.
b=bridge if isinstance(bridge,dict) else {}
if not bridge: verdict="native"                       # no X-PD-Bridge header at all: this endpoint is the decoder itself
elif b.get("bridge_error"): verdict="declined: "+str(b.get("bridge_error"))[:80]
elif b.get("skipped"): verdict="skipped: "+str(b.get("why",""))[:60]
else: verdict=b.get("verdict") or "bridged (no verdict field — old front door?)"
transport=b.get("transport") or ("none" if not bridge or b.get("skipped") or b.get("bridge_error") else "unknown")
print(json.dumps({"url":a.url,"seed":a.seed,"chars":len(doc),"ttft_s":round(first or -1,2),"ttft_content_s":round(first_content or -1,2),"total_s":round(tot,2),
                  "prompt_tokens":(usage or {}).get("prompt_tokens"),"cached_tokens":ptd.get("cached_tokens"),"completion_tokens":(usage or {}).get("completion_tokens"),
                  "verdict":verdict,"transport":transport,"stream":bool(b.get("stream")),
                  "marker":marker,"found":marker in ans,"answer":ans[:120],"bridge":bridge}))
