#!/usr/bin/env python3
"""rdma_pushfile.py — push ONE local file from the door Mac INTO a Spark over one MCDMA RDMA session
(the reverse of rdma_pulldir.py: the Spark runs `rdma_file recv`, the Mac runs `rdma_file send`).
Used by the KV return to land the previous turn's kvstate back on the prefill head before the next prefill.
   usage: rdma_pushfile.py SPARK_SSH LOCAL_FILE SPARK_OUT_PATH [--spark-dev rocep1s0f0 --spark-gid 1 --mac-dev rdma_mcrdma0 --mac-gid 0]
   -> JSON {bytes, seconds_wire, gbit_wire, seconds_total, transport}.  Falls back to scp (exit 0, transport="scp")
      if the RDMA session fails, so a missing/half-up lane never blocks a turn."""
import subprocess, sys, json, os, time, argparse
ap = argparse.ArgumentParser(); ap.add_argument("spark_ssh"); ap.add_argument("local_file"); ap.add_argument("spark_out")
ap.add_argument("--spark-dev", default="rocep1s0f0"); ap.add_argument("--spark-gid", default="1")
ap.add_argument("--mac-dev", default="rdma_mcrdma0"); ap.add_argument("--mac-gid", default="0")
ap.add_argument("--spark-bin", default="~/bin/rdma_file", help="path on the prefill node (expanded by its shell)"); ap.add_argument("--mac-bin", default=os.path.expanduser("~/bin/rdma_file"))
ap.add_argument("--no-rdma", action="store_true", help="scp only")
a = ap.parse_args(); t_all = time.time()
size = os.path.getsize(a.local_file); tmp = "/dev/shm/kvr_push.bin"
def scp():
    t0 = time.time()
    subprocess.run(["ssh", "-o", "BatchMode=yes", a.spark_ssh, f"mkdir -p {os.path.dirname(a.spark_out)}"], check=True)
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", a.local_file, f"{a.spark_ssh}:{a.spark_out}"], check=True)
    t = time.time() - t0
    print(json.dumps({"bytes": size, "seconds_wire": round(t, 3), "gbit_wire": round(size * 8 / t / 1e9, 2), "seconds_total": round(time.time() - t_all, 3), "transport": "scp"}))
if a.no_rdma or not os.path.exists(a.mac_bin):
    scp(); sys.exit(0)
try:
    env = dict(os.environ, MCDMA_CQ_MAP="2", MCDMA_USER_POST="1", MCDMA_USER_BF="64")
    # 1) Spark = receiver: prints "EP ... <nmr>" then nmr MR lines
    R = subprocess.Popen(["ssh", "-o", "BatchMode=yes", a.spark_ssh, f"{a.spark_bin} recv {a.spark_dev} {a.spark_gid} {tmp} {size}"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    ep = R.stdout.readline().strip()
    if not ep.startswith("EP"): raise RuntimeError("spark recv: " + R.stderr.read()[:300])
    nmr = int(ep.split()[-1]); mrs = [R.stdout.readline().strip() for _ in range(nmr)]
    # 2) Mac = sender
    S = subprocess.Popen([a.mac_bin, "send", a.mac_dev, a.mac_gid, a.local_file], env=env,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    S.stdin.write(ep + "\n" + "\n".join(mrs) + "\n"); S.stdin.flush()
    sep = S.stdout.readline().strip()
    if not sep.startswith("EP"): raise RuntimeError("mac send: " + S.stderr.read()[:300])
    R.stdin.write(sep + "\n"); R.stdin.flush()
    ready = R.stdout.readline().strip(); S.stdin.write(ready + "\n"); S.stdin.flush()
    lines = []
    while True:
        l = R.stdout.readline()
        if not l: break
        l = l.strip(); lines.append(l)
        if l.startswith("ACK"): S.stdin.write(l + "\n"); S.stdin.flush()
        if l.startswith("RESULT"): break
    so, se = S.communicate(timeout=300); R.wait(timeout=60)
    res = {}
    for l in so.splitlines() + lines:
        if l.startswith("RESULT"):
            d = dict(kv.split("=") for kv in l.split()[1:]); res[d["role"]] = d
    if "recv" not in res: raise RuntimeError((se + R.stderr.read())[:400])
    t_wire = float(res["recv"]["seconds"])
    # 3) move into place on the Spark (same filesystem as /dev/shm? no — cp then rm)
    subprocess.run(["ssh", "-o", "BatchMode=yes", a.spark_ssh, f"mkdir -p {os.path.dirname(a.spark_out)} && cp {tmp} {a.spark_out}.tmp && mv {a.spark_out}.tmp {a.spark_out} && rm -f {tmp}"], check=True)
    print(json.dumps({"bytes": size, "seconds_wire": round(t_wire, 3), "gbit_wire": round(size * 8 / t_wire / 1e9, 2), "seconds_total": round(time.time() - t_all, 3), "transport": "rdma"}))
except Exception as e:
    sys.stderr.write(f"rdma push failed ({e!r}) — scp fallback\n")
    try: subprocess.run(["ssh", "-o", "BatchMode=yes", a.spark_ssh, f"rm -f {tmp}"], check=False, timeout=30)
    except Exception: pass
    scp()
