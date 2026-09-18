#!/usr/bin/env python3
"""hetero_ring.py — the literal 4-machine loop, measured per hop, per turn.
   Turn k:  prompt_k = prior turns + new text
     1. Sparks 6+7 prefill prompt_k (vLLM TP2)            -> capture on spark-06
     2. capture -> S1 over MCDMA RDMA WRITE (rdma_pulldir)  -> S1 assembles oMLX blocks (pd_front bridge does 1+2)
     3. S1 decodes reply_k (oMLX, prefix hit)                [door decode]
     4. S1 blocks -> S2 over TB5 RDMA (tbsend UC SEND)       -> S2 holds the library; S2 decodes reply_k too [library decode]
     5. reply_k -> back into prompt_{k+1} -> Sparks prefill again (carry = text; KV-carry is the next build)
   Controls each turn: S2-alone (native oMLX, no Sparks), Spark-alone (vLLM generate).
   Runs on Cerebro. Writes JSONL to ~/pd_lab/rdma/ring_results.jsonl"""
import json,time,subprocess,sys,os,urllib.request
S1="chadhurley@192.168.86.68"; S2="chadhurley@192.168.0.76"; S06="chad-hurley@192.168.0.107"
S1_FRONT="http://192.168.0.68:8012/v1/chat/completions"; S1_OMLX="http://192.168.0.68:8011/v1/chat/completions"; S2_OMLX="http://127.0.0.1:18014/v1/chat/completions"; SPARK="http://192.168.0.107:8000"
MODEL="DV4-Flash-MXFP4-MLX"; OUT=os.path.expanduser("~/pd_lab/rdma/ring_results.jsonl")
def ssh(h,c,t=300): return subprocess.run(["ssh","-o","BatchMode=yes",h,c],capture_output=True,text=True,timeout=t)
def post(url,body,t=900):
    t0=time.time(); rq=urllib.request.Request(url,json.dumps(body).encode(),{"Content-Type":"application/json"})
    try: r=urllib.request.urlopen(rq,timeout=t)
    except urllib.error.HTTPError as e: raise RuntimeError(f"{url} -> HTTP {e.code}: {e.read()[:200]!r}")
    hdr={k.lower():v for k,v in r.headers.items()}; d=json.loads(r.read()); return time.time()-t0,d,hdr
def chat(url,msgs,max_tokens,model=MODEL):
    w,d,h=post(url,{"model":model,"messages":msgs,"max_tokens":max_tokens,"temperature":0})
    txt=d["choices"][0]["message"]["content"]; u=d.get("usage",{}); return dict(wall=round(w,2),text=txt,usage=u,bridge=h.get("x-pd-bridge"))
def sync_blocks_s1_to_s2():
    """Ship S1's newest oMLX blocks to S2's cache over TB5 RDMA (tar of files newer than marker), report Gb/s."""
    ls=ssh(S1,'cd ~/.omlx/cache && find . -name "*.safetensors" -newer /tmp/.ring_marker 2>/dev/null | sort').stdout.split()
    if not ls: return {"files":0}
    ssh(S1,'cd ~/.omlx/cache && find . -name "*.safetensors" -newer /tmp/.ring_marker | tar -cf /tmp/ring_blocks.tar -T -; touch /tmp/.ring_marker')
    size=int(ssh(S1,'stat -f%z /tmp/ring_blocks.tar').stdout.strip())
    r=subprocess.run(["python3","/tmp/tbrun.py",S2,"rdma_en3","/tmp/ring_blocks.tar",S1,"rdma_en4","/tmp/ring_blocks.tar","1","2"],capture_output=True,text=True,timeout=300,env=dict(os.environ,TB_FRAME="1048576",TB_RING="3"))
    res=[l for l in r.stdout.splitlines() if "role=recv" in l]
    ssh(S2,'cd ~/.omlx/cache && tar -xf /tmp/ring_blocks.tar && rm /tmp/ring_blocks.tar')
    m={}
    if res: m=dict(kv.split("=") for kv in res[0].split("recv: TBRESULT ")[-1].split() if "=" in kv)
    return {"files":len(ls),"bytes":size,"tb5_seconds":float(m.get("seconds",0)),"tb5_gbit":float(m.get("gbit",0))}
def spark_generate(prompt,max_tokens):
    w,d,_=post(SPARK+"/v1/completions",{"model":"deepseek-v4-flash","prompt":prompt,"max_tokens":max_tokens,"temperature":0})
    return dict(wall=round(w,2),text=d["choices"][0]["text"],usage=d.get("usage",{}))
if __name__=="__main__":
    seed=open(sys.argv[1]).read() if len(sys.argv)>1 and sys.argv[1] else ""
    turns=int(sys.argv[2]) if len(sys.argv)>2 else 3; add_chars=int(sys.argv[3]) if len(sys.argv)>3 else 60000
    vol=open('/opt/cerebro/vaultline_local/Vaultline_Volume_10.txt',errors='replace').read(); cursor=int(len(vol)*0.5)
    ssh(S1,'touch /tmp/.ring_marker')
    history=[]; convo_text=seed
    for k in range(1,turns+1):
        chunk=vol[cursor:cursor+add_chars]; cursor+=add_chars
        q=f"[Turn {k}] Here is more of the scroll:\n\n{chunk}\n\nIn 2-3 sentences, what is the most important thing that happened in this passage, and how does it connect to the previous turns?"
        history.append({"role":"user","content":q}); rec={"turn":k,"prompt_chars":sum(len(m["content"]) for m in history)}
        # 1-3: hetero door (Sparks prefill -> RDMA -> S1 decode)
        h=chat(S1_FRONT,history,120); rec["hetero_s1"]={"wall":h["wall"],"usage":h["usage"],"bridge":h["bridge"],"text":h["text"][:200]}
        # 4: S1 blocks -> S2 library over TB5 RDMA, then S2 decodes the same turn (prefix should hit)
        rec["tb5_sync"]=sync_blocks_s1_to_s2()
        s2h=chat(S2_OMLX,history,120); rec["library_s2"]={"wall":s2h["wall"],"usage":s2h["usage"],"text":s2h["text"][:200]}
        # controls
        s2a=chat(S2_OMLX,[{"role":"user","content":"CONTROL-S2ALONE "+q}],120); rec["ctrl_s2_alone"]={"wall":s2a["wall"],"usage":s2a["usage"]}
        try:
            sp=spark_generate("CONTROL-SPARK "+q,120); rec["ctrl_spark_alone"]={"wall":sp["wall"],"usage":sp["usage"]}
        except Exception as e: rec["ctrl_spark_alone"]={"error":str(e)[:120]}
        # 5: the reply goes back into the conversation -> next turn's prefill on the Sparks carries it
        history.append({"role":"assistant","content":h["text"]})
        open(OUT,"a").write(json.dumps(rec)+"\n"); print(json.dumps({k2:(v if k2 in("turn","prompt_chars","tb5_sync") else {kk:vv for kk,vv in v.items() if kk!="text"}) for k2,v in rec.items()}),flush=True)
