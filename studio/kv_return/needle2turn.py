#!/usr/bin/env python3
"""needle2turn.py — 2-turn needle test through the front door (:8012) or the ring front (:8015). Turn 1: ~W1 words of filler with the needle
planted at ~65% depth (>65K tokens); turn 2: the reply + ≥9K new words of filler (so the door BRIDGES the tail, PD_MIN_TAIL=8192)
+ the question. Prints X-PD-Ring / X-PD-Bridge timings per turn and the needle verdict. Writes JSON to --out."""
import json, random, sys, time, urllib.request, argparse, re
ap=argparse.ArgumentParser(); ap.add_argument("--url",required=True,help="front door, e.g. http://DECODER_HOST:8012/v1/chat/completions (door) or :8015 (ring)"); ap.add_argument("--w1",type=int,default=62000)
ap.add_argument("--w2",type=int,default=8000); ap.add_argument("--out",default="/tmp/needle2turn.json"); ap.add_argument("--seed",type=int,default=918); a=ap.parse_args()
rnd=random.Random(a.seed); NEEDLE=f"The secret harbor code is LANTERN-{rnd.randint(1000,9999)}-OAK."
WORDS="river stone lantern harbor willow journey silver morning archive cadence marble orchard signal ledger meadow granite tidal beacon canvas ember".split()
def filler(n): return " ".join(rnd.choice(WORDS) for _ in range(n))
f1=filler(a.w1).split(" "); pos=int(len(f1)*0.65); f1.insert(pos,NEEDLE); doc=" ".join(f1)
msgs=[{"role":"system","content":"You are a careful assistant. Answer briefly."},{"role":"user","content":"Read this document and reply with just the word READY.\n\n"+doc}]
def turn(msgs,tag):
    body=json.dumps({"model":"DV4-Flash-MXFP4-MLX","messages":msgs,"max_tokens":200,"temperature":0}).encode()
    t0=time.time(); rq=urllib.request.Request(a.url,body,{"Content-Type":"application/json"}); r=urllib.request.urlopen(rq,timeout=7200)
    raw=r.read(); t=time.time()-t0; ring=r.headers.get("X-PD-Ring"); txt=""
    try: txt=json.loads(raw)["choices"][0]["message"]["content"]
    except Exception: txt=raw[:200].decode(errors="replace")
    ring=json.loads(ring) if ring else {}
    print(f"[{tag}] {t:.1f}s reply={txt[:80]!r}\n  ring={json.dumps(ring)[:1500]}",flush=True); return {"s":round(t,1),"reply":txt,"ring":ring}
r1=turn(msgs,"turn1")
msgs2=msgs+[{"role":"assistant","content":r1["reply"] or "READY"},{"role":"user","content":"Here is an addendum, then a question.\n\n"+filler(a.w2)+"\n\nQuestion: what is the secret harbor code stated in the first document? Reply with the code only."}]
r2=turn(msgs2,"turn2")
digits=re.search(r"\d{4}",NEEDLE).group(0); ok=digits in (r2["reply"] or "")
print(f"NEEDLE {'PASS' if ok else 'FAIL'}: expected {NEEDLE!r} got {r2['reply']!r}")
json.dump({"needle":NEEDLE,"pass":ok,"turn1":r1,"turn2":r2},open(a.out,"w"),indent=1)
