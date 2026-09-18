import subprocess,sys,time,select
recv_host,recv_dev,recv_out = sys.argv[1],sys.argv[2],sys.argv[3]
send_host,send_dev,send_file = sys.argv[4],sys.argv[5],sys.argv[6]
rgid = sys.argv[7] if len(sys.argv)>7 else "1"
sgid = sys.argv[8] if len(sys.argv)>8 else rgid
import os
ENV=" ".join(f"{k}={v}" for k,v in os.environ.items() if k.startswith("TB_"))
def ssh(h,cmd): return subprocess.Popen(["ssh","-o","BatchMode=yes",h,ENV+" "+cmd],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
R=ssh(recv_host,f"export LC_ALL=C; ~/bin/tbsend recv {recv_dev} {rgid} {recv_out}")
ep=R.stdout.readline().strip(); print("recv:",ep)
if not ep.startswith("ENDPOINT"): print("recv stderr:",R.stderr.read()); sys.exit(2)
S=ssh(send_host,f"export LC_ALL=C; ~/bin/tbsend send {send_dev} {sgid} {send_file}")
S.stdin.write(ep+"\n"); S.stdin.flush()
sep=S.stdout.readline().strip(); print("send:",sep)
if not sep: print("send stderr:",S.stderr.read()); sys.exit(2)
R.stdin.write(sep+"\n"); R.stdin.flush()
ready=R.stdout.readline().strip(); print("recv:",ready)
S.stdin.write(ready+"\n"); S.stdin.flush()
for p,name in ((S,"send"),(R,"recv")):
    try: out,err=p.communicate(timeout=60)
    except subprocess.TimeoutExpired: p.kill(); out,err=p.communicate()
    for l in out.strip().splitlines(): print(f"{name}: {l}")
    if err.strip(): print(f"{name} stderr: {err.strip()[:300]}")
