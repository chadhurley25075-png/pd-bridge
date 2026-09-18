#!/usr/bin/env python3
"""pd_ring_front.py — the hetero-deep4 FRONT DOOR. Runs on S1. OpenAI-compatible /v1/chat/completions on :8015.

Every request circulates the ring:
  ① Sparks 6+7 prefill (via S1's pd_front bridge :8012, which pulls the capture over MCDMA RDMA and assembles blocks into S1's cache)
  ② new S1 blocks -> S2's cache over TB5 RDMA (tbsend, UC SEND)
  ③ S2 (512 GB library) decodes the reply with the full prefix already resident  (S2 oMLX :8014 via ssh tunnel)
  ④ the reply is returned; and — BOTH WAYS — S2's own new blocks (the reply tokens it decoded) are shipped BACK to S1 over TB5,
     so S1's cache (the door) also holds the reply. The Sparks re-prefill from text on the next turn (KV return arrow = next build);
     the ring keeps the two Studios' caches identical so either can answer.
Falls back to S1-only if S2 is unreachable. Every hop timed into the X-PD-Ring header and ~/pd_ring.log.
Env: RING_S1_FRONT (http://127.0.0.1:8012), RING_S2_OMLX (http://127.0.0.1:18014), RING_S2_SSH (chadhurley@192.168.0.76),
     RING_S1_TBDEV (rdma_en4), RING_S2_TBDEV (rdma_en3), RING_S1_GID (2), RING_S2_GID (1), RING_PORT (8015)"""
import json,os,sys,time,subprocess,threading,urllib.request,glob
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
E=os.environ.get
S1_FRONT=E("RING_S1_FRONT","http://127.0.0.1:8012"); S2_OMLX=E("RING_S2_OMLX","http://127.0.0.1:18014"); S2_SSH=E("RING_S2_SSH","chadhurley@192.168.0.76")
S1_TB=E("RING_S1_TBDEV","rdma_en4"); S2_TB=E("RING_S2_TBDEV","rdma_en3"); S1_GID=E("RING_S1_GID","2"); S2_GID=E("RING_S2_GID","1")
CACHE=os.path.expanduser("~/.omlx/cache"); S2_CACHE="/Users/chadhurley/.omlx/cache"; PORT=int(E("RING_PORT","8015"))
LOG=open(os.path.expanduser("~/pd_ring.log"),"a"); LOCK=threading.Lock()
def L(*a):
    with LOCK: LOG.write(time.strftime("%H:%M:%S ")+" ".join(str(x) for x in a)+"\n"); LOG.flush()
def ssh(h,c,t=120): return subprocess.run(["ssh","-o","BatchMode=yes",h,c],capture_output=True,text=True,timeout=t)
def post(url,body,t=1800,extra=None):
    hd={"Content-Type":"application/json"}; hd.update(extra or {})
    rq=urllib.request.Request(url,json.dumps(body).encode(),hd); r=urllib.request.urlopen(rq,timeout=t)
    return {k.lower():v for k,v in r.headers.items()}, r.read()
def tb_send(local_tar,remote_tar,src_host,src_dev,src_gid,dst_host,dst_dev,dst_gid):
    """one UC-SEND session S1<->S2 via tbrun.py protocol (both binaries live at /tmp/tbsend)."""
    env=dict(os.environ,TB_FRAME="1048576",TB_RING="3")
    r=subprocess.run(["python3",os.path.expanduser("~/bin/tbrun.py"),dst_host,dst_dev,remote_tar,src_host,src_dev,local_tar,dst_gid,src_gid],capture_output=True,text=True,timeout=300,env=env)
    for l in r.stdout.splitlines():
        if "role=recv" in l: return dict(kv.split("=") for kv in l.split("TBRESULT ")[-1].split() if "=" in kv)
    raise RuntimeError("tb_send: "+(r.stderr or r.stdout)[-300:])
def ship_s1_to_s2(marker):
    files=subprocess.run(["bash","-c",f'cd {CACHE} && find . -name "*.safetensors" -newer {marker} | sort'],capture_output=True,text=True).stdout.split()
    if not files: return {"files":0}
    subprocess.run(["bash","-c",f'cd {CACHE} && find . -name "*.safetensors" -newer {marker} | tar -cf /tmp/ring_s1s2.tar -T -'],check=True)
    size=os.path.getsize("/tmp/ring_s1s2.tar"); t0=time.time()
    m=tb_send("/tmp/ring_s1s2.tar","/tmp/ring_s1s2.tar","chadhurley@127.0.0.1",S1_TB,S1_GID,S2_SSH,S2_TB,S2_GID)
    ssh(S2_SSH,f"cd {S2_CACHE} && tar -xf /tmp/ring_s1s2.tar && rm -f /tmp/ring_s1s2.tar")
    return {"files":len(files),"bytes":size,"wire_s":float(m.get("seconds",0)),"gbit":float(m.get("gbit",0)),"total_s":round(time.time()-t0,2)}
def ship_s2_to_s1(marker_iso):
    """BACK: S2's blocks newer than marker -> S1 cache over TB5 (S2 sends, S1 receives)."""
    r=ssh(S2_SSH,f'cd {S2_CACHE} && find . -name "*.safetensors" -newer /tmp/.ring_s2_marker | sort | tee /tmp/ring_list.txt | tar -cf /tmp/ring_s2s1.tar -T - 2>/dev/null; touch /tmp/.ring_s2_marker; wc -l < /tmp/ring_list.txt; stat -f%z /tmp/ring_s2s1.tar 2>/dev/null')
    parts=r.stdout.split(); n=int(parts[0]) if parts else 0
    if n==0: return {"files":0}
    size=int(parts[1]); t0=time.time()
    m=tb_send("/tmp/ring_s2s1.tar","/tmp/ring_s2s1.tar",S2_SSH,S2_TB,S2_GID,"chadhurley@127.0.0.1",S1_TB,S1_GID)
    subprocess.run(["bash","-c",f"cd {CACHE} && tar -xf /tmp/ring_s2s1.tar && rm -f /tmp/ring_s2s1.tar"],check=True)
    return {"files":n,"bytes":size,"wire_s":float(m.get("seconds",0)),"gbit":float(m.get("gbit",0)),"total_s":round(time.time()-t0,2)}
class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def do_GET(self):
        if self.path.startswith("/v1/models"):
            b=urllib.request.urlopen(S1_FRONT+"/v1/models",timeout=10).read()
            self.send_response(200); self.send_header("Content-Type","application/json"); self.end_headers(); self.wfile.write(b); return
        self.send_response(404); self.end_headers()
    def do_POST(self):
        n=int(self.headers.get("Content-Length","0")); raw=self.rfile.read(n); T0=time.time(); ring={}
        if not self.path.startswith("/v1/chat/completions"): self.send_response(404); self.end_headers(); return
        req=json.loads(raw); want_stream=bool(req.get("stream")); req["stream"]=False
        marker="/tmp/.ring_s1_marker"; open(marker,"a").close(); os.utime(marker,None)
        # ① Sparks prefill -> RDMA -> S1 blocks (bridge only; S1 does not decode)
        t=time.time()
        try: h1,b1=post(S1_FRONT+"/v1/chat/completions",req,extra={"X-PD-Bridge-Only":"1"}); ring["door"]={"s":round(time.time()-t,2),"bridge":json.loads(h1.get("x-pd-bridge","{}") or "{}")}
        except Exception as e: ring["door"]={"s":round(time.time()-t,2),"error":str(e)[:160]}
        target=None
        try:
            # ② new blocks -> S2 library
            t=time.time(); ring["tb5_s1_to_s2"]=ship_s1_to_s2(marker); ring["tb5_s1_to_s2"]["s"]=round(time.time()-t,2)
            ssh(S2_SSH,"touch /tmp/.ring_s2_marker"); target=S2_OMLX
        except Exception as e:
            ring["library_error"]=str(e)[:200]; L("tb5 leg failed, decoding at the door:",e)
        # ③ decode: stream straight through from the library (or the door as fallback)
        dec_url=(target or S1_FRONT)+"/v1/chat/completions"; ring["decoder"]="S2-library" if target else "S1-door"
        req["stream"]=want_stream; t=time.time()
        rq=urllib.request.Request(dec_url,json.dumps(req).encode(),{"Content-Type":"application/json"})
        try: r=urllib.request.urlopen(rq,timeout=1800)
        except Exception as e:
            ring["decode_error"]=str(e)[:200]; L(json.dumps(ring)); self.send_response(502); self.end_headers(); return
        self.protocol_version="HTTP/1.1"; self.send_response(200)
        self.send_header("Content-Type",r.headers.get("Content-Type","application/json"))
        self.send_header("Cache-Control","no-cache"); self.send_header("X-Accel-Buffering","no")
        self.send_header("Connection","close")   # we relay decoded bytes; end-of-body = connection close (no chunked framing of our own)
        self.send_header("X-PD-Ring",json.dumps(ring)); self.end_headers()
        first=None; cached=None; tail=b""
        try:
            while True:
                chunk=r.read(1<<14)      # raw read; works for both chunked SSE and a single JSON body
                if not chunk: break
                if first is None: first=time.time()-t
                tail=(tail+chunk)[-4096:]
                self.wfile.write(chunk); self.wfile.flush()
        except BrokenPipeError: pass
        try:
            import re as _re; m=_re.search(rb'"cached_tokens":\s*(\d+)',tail); cached=int(m.group(1)) if m else None
        except Exception: pass
        ring["decode"]={"ttfb":round(first or 0,2),"s":round(time.time()-t,2),"cached":cached}
        # ④ BACK: the decoder's new blocks -> the other Studio, so both caches hold the reply
        try:
            t=time.time()
            if target: ring["tb5_s2_to_s1"]=ship_s2_to_s1(None)
            else: ring["tb5_s1_to_s2_after"]=ship_s1_to_s2(marker)
            ring["back_s"]=round(time.time()-t,2)
        except Exception as e: ring["back_error"]=str(e)[:160]
        ring["total_s"]=round(time.time()-T0,2); L(json.dumps(ring))
        try: self.wfile.flush(); self.connection.close()
        except Exception: pass
if __name__=="__main__":
    L(f"pd_ring_front listening :{PORT} door={S1_FRONT} library={S2_OMLX}")
    ThreadingHTTPServer(("0.0.0.0",PORT),H).serve_forever()
