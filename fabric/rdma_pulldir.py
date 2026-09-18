#!/usr/bin/env python3
"""rdma_pulldir.py — pull every regular file in a Spark directory into a local dir over ONE MCDMA RDMA session.
   Files are concatenated on the Spark side into a stream (tar, no compression) and RDMA-WRITTEN; local side untars.
   usage: rdma_pulldir.py SPARK_SSH SPARK_DIR LOCAL_DIR   -> JSON {bytes, seconds, gbit, files}"""
import subprocess,sys,json,os,time,argparse,tarfile,io
ap=argparse.ArgumentParser(); ap.add_argument("spark_ssh"); ap.add_argument("spark_dir"); ap.add_argument("local_dir")
ap.add_argument("--spark-dev",default="rocep1s0f0"); ap.add_argument("--spark-gid",default="1")
ap.add_argument("--mac-dev",default="rdma_mcrdma0"); ap.add_argument("--mac-gid",default="0")
ap.add_argument("--spark-bin",default="/home/chad-hurley/bin/rdma_file"); ap.add_argument("--mac-bin",default=os.path.expanduser("~/bin/rdma_file"))
a=ap.parse_args(); t_all=time.time()
# 1) Spark: tar the dir's regular files (skip DONE/seg_*) into /dev/shm, report size
tarcmd=(f"cd {a.spark_dir} && find . -maxdepth 1 -type f ! -name DONE ! -name 'seg_*' -printf '%P\\n' | sort | "
        f"tar -cf /dev/shm/pd_pull.tar -T - && stat -c %s /dev/shm/pd_pull.tar")
size=int(subprocess.check_output(["ssh","-o","BatchMode=yes",a.spark_ssh,tarcmd],text=True).strip()); t_tar=time.time()-t_all
# 2) one RDMA session for the tar
os.makedirs(a.local_dir,exist_ok=True); tarpath=os.path.join(a.local_dir,"_pull.tar")
env=dict(os.environ,MCDMA_CQ_MAP="2",MCDMA_USER_POST="1",MCDMA_USER_BF="64")
R=subprocess.Popen([a.mac_bin,"recv",a.mac_dev,a.mac_gid,tarpath,str(size)],env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
ep=R.stdout.readline().strip(); nmr=int(ep.split()[-1]); mrs=[R.stdout.readline().strip() for _ in range(nmr)]
S=subprocess.Popen(["ssh","-o","BatchMode=yes",a.spark_ssh,f"{a.spark_bin} send {a.spark_dev} {a.spark_gid} /dev/shm/pd_pull.tar"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
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
so,se=S.communicate(timeout=900); re_=R.stderr.read(); R.wait()
res={}
for l in so.splitlines()+lines:
    if l.startswith("RESULT"): d=dict(kv.split("=") for kv in l.split()[1:]); res[d["role"]]=d
if "recv" not in res: print(json.dumps({"error":(se+re_)[:400]})); sys.exit(2)
t_wire=float(res["recv"]["seconds"])
# 3) untar locally, drop the tar
t_u=time.time()
with tarfile.open(tarpath) as tf: names=tf.getnames(); tf.extractall(a.local_dir)
os.remove(tarpath); t_untar=time.time()-t_u
subprocess.run(["ssh","-o","BatchMode=yes",a.spark_ssh,"rm -f /dev/shm/pd_pull.tar"],check=False)
print(json.dumps({"bytes":size,"files":len(names),"seconds_total":round(time.time()-t_all,3),"seconds_wire":round(t_wire,3),"gbit_wire":round(size*8/t_wire/1e9,2),"t_tar":round(t_tar,2),"t_untar":round(t_untar,2)}))
