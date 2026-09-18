#!/usr/bin/env python3
"""rdma_pull.py — pull a file from a Spark into this Mac over MCDMA RDMA WRITE (Spark initiates the WRITE).
   usage: rdma_pull.py SPARK_SSH SPARK_PATH LOCAL_PATH [--spark-dev rocep1s0f0 --spark-gid 1 --mac-dev rdma_mcrdma0 --mac-gid 0]
   Prints JSON {bytes, seconds, gbit}. No HTTP, no TCP on the data path."""
import subprocess,sys,json,time,os,argparse
ap=argparse.ArgumentParser(); ap.add_argument("spark_ssh"); ap.add_argument("spark_path"); ap.add_argument("local_path")
ap.add_argument("--spark-dev",default="rocep1s0f0"); ap.add_argument("--spark-gid",default="1")
ap.add_argument("--mac-dev",default="rdma_mcrdma0"); ap.add_argument("--mac-gid",default="0")
ap.add_argument("--spark-bin",default="/home/chad-hurley/bin/rdma_file"); ap.add_argument("--mac-bin",default=os.path.expanduser("~/bin/rdma_file"))
a=ap.parse_args()
size=int(subprocess.check_output(["ssh","-o","BatchMode=yes",a.spark_ssh,f"stat -c %s {a.spark_path}"],text=True).strip())
env=dict(os.environ,MCDMA_CQ_MAP="2",MCDMA_USER_POST="1",MCDMA_USER_BF="64")  # Ash's fast path (mapped CQ + user doorbell + BlueFlame-64)
R=subprocess.Popen([a.mac_bin,"recv",a.mac_dev,a.mac_gid,a.local_path,str(size)],env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
ep=R.stdout.readline().strip(); nmr=int(ep.split()[-1]); mrs=[R.stdout.readline().strip() for _ in range(nmr)]
S=subprocess.Popen(["ssh","-o","BatchMode=yes",a.spark_ssh,f"{'RF_NOREAD=1 ' if os.environ.get('RF_NOREAD') else ''}{a.spark_bin} send {a.spark_dev} {a.spark_gid} {a.spark_path}"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
S.stdin.write(ep+"\n"+"\n".join(mrs)+"\n"); S.stdin.flush()
sep=S.stdout.readline().strip()
if not sep.startswith("EP"): print(json.dumps({"error":"spark: "+S.stderr.read()[:300]})); sys.exit(2)
R.stdin.write(sep+"\n"); R.stdin.flush()
ready=R.stdout.readline().strip(); S.stdin.write(ready+"\n"); S.stdin.flush()
lines=[]
while True:
    l=R.stdout.readline()
    if not l: break
    l=l.strip(); lines.append(l)
    if l.startswith("ACK"): S.stdin.write(l+"\n"); S.stdin.flush()
    if l.startswith("RESULT"): break
so,se=S.communicate(timeout=600); re=R.stderr.read(); R.wait()
res={}
for l in (so.splitlines()+lines):
    if l.startswith("RESULT"): 
        d=dict(kv.split("=") for kv in l.split()[1:]); res[d["role"]]=d
if "recv" not in res: print(json.dumps({"error":(se+re)[:400]})); sys.exit(2)
r=res["recv"]; import re as _re; sp=_re.search(r"send_profile[^\n]*",se); print(sp.group(0) if sp else "", file=sys.stderr); m=_re.search(r"wire_only_gbit=([0-9.]+)",re); print(json.dumps({"bytes":int(r["bytes"]),"seconds":float(r["seconds"]),"gbit":float(r["gbit"]),"wire_gbit":float(m.group(1)) if m else None}))
