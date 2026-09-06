#!/usr/bin/env python3
"""pd_quality_eval.py — judged quality gate: same long document + checkable questions, native (S3 mlx_lm) vs bridge (S1 pd_front).
Doc = first N chars of a public source file (default: any public source file) so answers are verifiable by grep.
usage: pd_quality_eval.py --doc PATH --chars 60000 --out results.json  [--front http://DECODER_HOST:8012] [--model DV4-Flash-MXFP4-MLX]
Runs the BRIDGE leg here; prints the messages JSON so the same prompt can be run natively on S3 with pd_native_ref_doc.py.
"""
import argparse, os, json, time, urllib.request
ap=argparse.ArgumentParser(); ap.add_argument("--doc",required=True); ap.add_argument("--chars",type=int,default=60000); ap.add_argument("--out",required=True); ap.add_argument("--front",default=os.environ.get("PD_FRONT","http://DECODER_HOST:8012")); ap.add_argument("--model",default="DV4-Flash-MXFP4-MLX"); ap.add_argument("--max-tokens",type=int,default=400)
a=ap.parse_args(); doc=open(a.doc).read()[:a.chars]
QS=json.load(open(os.path.join(os.path.dirname(os.path.abspath(a.doc)),"eval_questions.json")))["questions"]
res=[]
for q in QS:
    msgs=[{"role":"user","content":f"Read this Python source and answer precisely.\n\n<file>\n{doc}\n</file>\n\nQuestion: {q}\nAnswer concisely with exact quotes where asked. /nothink"}]
    body=json.dumps({"model":a.model,"messages":msgs,"max_tokens":a.max_tokens,"temperature":0}).encode()
    t=time.time(); r=urllib.request.urlopen(urllib.request.Request(a.front+"/v1/chat/completions",body,{"Content-Type":"application/json"}),timeout=3600); d=json.load(r); dt=time.time()-t
    m=d["choices"][0]["message"]; txt=m.get("content") or m.get("reasoning") or ""
    res.append({"q":q,"bridge":txt,"wall":round(dt,1),"bridge_hdr":r.headers.get("X-PD-Bridge")}); print(f"[{dt:.1f}s] Q: {q[:60]}\n   -> {txt[:200]!r}")
json.dump({"doc":a.doc,"chars":a.chars,"model":a.model,"results":res},open(a.out,"w"),indent=1); print("saved",a.out)
