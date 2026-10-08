#!/usr/bin/env python3
"""pd_kv_return.py — decoder (door) side of the KV return (2026-09-18). Imported by pd_front.py; every entry
point is a no-op unless PD_KV_RETURN=1, and every failure returns a verdict instead of raising, so the
bridge's existing paths are untouched when the arrow is off or broken.

The arrow, end to end (see spark/capture_sitecustomize_v3.kvreturn.py "KV RETURN" and studio/kv_return/README.md):
  turn 1  Sparks prefill 0..T1 (vLLM prefix caching ON) -> hook writes layer files + kvstate.{safetensors,json}
          -> door pulls the capture dir (RDMA) -> assembles blocks -> remember(): the pulled dir is KEPT as
          the parent (PD_PULL_DIR/_kvr_last/<stamp>) instead of deleted.
  turn 2  push_state(): if the Spark no longer has the parent's kvstate (janitor 30 min / prune keep=3), push
          it back into <capdir>/_kvreturn/<stamp>/ over MCDMA (rdma_pushfile.py; scp fallback) BEFORE the
          engine call. vLLM hits its prefix cache, computes only [S,T2), the hook resumes from the state and
          writes a capture holding [S,T2) + the end state, manifest kv_return.resume_from = S.
          usable_prefix(): the front accepts that capture (it used to reject any partial_start).
          merge_prefix(): pooled = cat(parent.pooled[:S//ratio], new.pooled); kvwin_b / prev_*_b for b<=S come
          from the parent; everything else (b>S, *_end, buf_*) from the new capture. Written back in place, the
          normal assemble_and_write then runs on a capture that is complete from 0. Proven bit-exact by
          spark/test_kv_return.py (115 checks, CPU).
Env: PD_KV_RETURN=1 · PD_PULL_DIR (pd_front's) · PD_KVR_PUSH_BIN (default: rdma_pushfile.py next to this file) · PD_KVR_NO_RDMA=1 (scp)
"""
import json, os, re, shutil, subprocess, sys, time

E = os.environ.get
ENABLED = E("PD_KV_RETURN", "0") == "1"
PULL_DIR = os.path.expanduser(E("PD_PULL_DIR", "~/pd_lab/pd_pull"))
LAST_DIR = os.path.join(PULL_DIR, "_kvr_last")
LAST_JSON = os.path.join(PULL_DIR, "_kvr_last.json")
PUSH_BIN = os.path.expanduser(E("PD_KVR_PUSH_BIN", os.path.join(os.path.dirname(os.path.abspath(__file__)), "rdma_pushfile.py")))
KVR_DIR = "_kvreturn"; STATE = "kvstate.safetensors"; META = "kvstate.json"
_LOG = None
def L(*a):
    global _LOG
    try:
        if _LOG is None: _LOG = open(os.path.expanduser("~/pd_kv_return.log"), "a")
        _LOG.write(time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a) + "\n"); _LOG.flush()
    except Exception: pass

def enabled(): return ENABLED

def last():
    try:
        with open(LAST_JSON) as f: j = json.load(f)
        if os.path.isdir(j.get("dir", "")) and os.path.isfile(os.path.join(j["dir"], STATE)): return j
    except Exception: pass
    return None

def remember(stamp, local_dir, T, man=None):
    """Keep this pulled capture as the parent for the next turn (replaces the previous parent). Returns dict."""
    if not ENABLED: return {"kept": False}
    try:
        if man and man.get("stream", {}).get("enabled"):
            L(f"remember: {stamp} is a STREAMED capture (end-state layer files) — cannot be a merge parent; not kept")
            return {"kept": False, "why": "streamed"}
        if not os.path.isfile(os.path.join(local_dir, STATE)):
            L(f"remember: {stamp} has no {STATE} (hook not armed with PD_KV_RETURN=1?) — not kept")
            return {"kept": False, "why": "no state file"}
        shutil.rmtree(LAST_DIR, ignore_errors=True); os.makedirs(LAST_DIR, exist_ok=True)
        dst = os.path.join(LAST_DIR, stamp); shutil.move(local_dir, dst)
        j = {"stamp": stamp, "T": int(T), "dir": dst, "saved_at": time.time(),
             "bytes": sum(os.path.getsize(os.path.join(dst, f)) for f in os.listdir(dst))}
        with open(LAST_JSON, "w") as f: json.dump(j, f)
        L(f"remember: parent = {stamp} T={T} ({j['bytes']/1e9:.2f} GB kept at {dst})")
        return {"kept": True, "stamp": stamp, "T": int(T)}
    except Exception as e:
        L(f"remember failed: {e!r}"); return {"kept": False, "why": repr(e)}

def push_state(spark_ssh, spark_capdir):
    """Before the engine call: make sure the Spark holds the parent's kvstate (own stamp dir OR _kvreturn copy)."""
    if not ENABLED: return None
    j = last()
    if j is None: return {"pushed": False, "why": "no parent"}
    t0 = time.time(); stamp = j["stamp"]; cap = spark_capdir.rstrip("/")
    try:
        chk = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", spark_ssh,
                              f"test -f {cap}/{stamp}/{STATE} -a -f {cap}/{stamp}/{META} && echo own; "
                              f"test -f {cap}/{KVR_DIR}/{stamp}/{STATE} -a -f {cap}/{KVR_DIR}/{stamp}/{META} && echo copy; "
                              f"mkdir -p {cap}/{KVR_DIR} && touch {cap}/{KVR_DIR}"],   # touch: keep the janitor (-mmin +30) off it
                             capture_output=True, text=True, timeout=30).stdout.split()
        if chk:
            return {"pushed": False, "present": chk[0], "stamp": stamp, "seconds": round(time.time() - t0, 3)}
        out = {"pushed": True, "stamp": stamp, "files": {}}
        for name in (STATE, META):
            src = os.path.join(j["dir"], name); dst = f"{cap}/{KVR_DIR}/{stamp}/{name}"
            if name == META or E("PD_KVR_NO_RDMA") == "1" or not os.path.exists(PUSH_BIN):
                subprocess.run(["ssh", "-o", "BatchMode=yes", spark_ssh, f"mkdir -p {cap}/{KVR_DIR}/{stamp}"], check=True, timeout=30)
                subprocess.run(["scp", "-q", "-o", "BatchMode=yes", src, f"{spark_ssh}:{dst}"], check=True, timeout=300)
                out["files"][name] = {"transport": "scp", "bytes": os.path.getsize(src)}
            else:
                r = subprocess.run([sys.executable, PUSH_BIN, spark_ssh, src, dst], capture_output=True, text=True, timeout=600)
                line = (r.stdout.strip().splitlines() or ["{}"])[-1]
                try: out["files"][name] = json.loads(line)
                except Exception: out["files"][name] = {"error": (r.stderr or r.stdout)[-300:]}
        out["seconds"] = round(time.time() - t0, 3)
        L(f"push_state: {out}")
        return out
    except Exception as e:
        L(f"push_state failed: {e!r}"); return {"pushed": False, "why": repr(e)}

def usable_prefix(man):
    """A resumed capture (kv_return.resume_from = S) is complete-from-0 AFTER merge_prefix — but only if its
    parent is the capture we kept. Returns T, or 0."""
    if not ENABLED: return 0
    kr = man.get("kv_return") or {}
    S = kr.get("resume_from")
    if not S: return 0
    j = last()
    if j is None or kr.get("parent_stamp") != j["stamp"] or j["T"] < S:
        L(f"usable_prefix: resumed from {S} but parent {kr.get('parent_stamp')} != kept {j and j['stamp']} — declining")
        return 0
    if man.get("position_gaps"): return 0
    return int(man.get("T") or 0)

_INT_B = re.compile(r"^(?:idx_)?(?:kvwin|prev_kv|prev_gate)_(\d+)$")
def merge_prefix(local_dir, man):
    """Rewrite local_dir's layer files so they are complete from token 0 (see module doc). Returns dict."""
    if not ENABLED: return None
    kr = man.get("kv_return") or {}; S = kr.get("resume_from")
    if not S: return {"merged": False, "why": "not a resumed capture"}
    j = last()
    if j is None or kr.get("parent_stamp") != j["stamp"]: return {"merged": False, "why": "parent mismatch"}
    import mlx.core as mx
    t0 = time.time(); ratios = man.get("ratios") or []; n = 0; nbytes = 0
    try:
        for f in sorted(os.listdir(local_dir)):
            m = re.match(r"^layer_(\d+)\.safetensors$", f)
            if not m: continue
            li = int(m.group(1)); r = ratios[li] if li < len(ratios) else None
            new = mx.load(os.path.join(local_dir, f)); par = mx.load(os.path.join(j["dir"], f))
            out = {}
            for k, v in par.items():
                mb = _INT_B.match(k)
                if mb and int(mb.group(1)) <= S: out[k] = v
            for k, v in new.items():
                mb = _INT_B.match(k)
                if mb and int(mb.group(1)) <= S: continue        # the hook never re-exports these; belt+braces
                out[k] = v
            if r and "pooled" in par:
                keep = S // r; head = par["pooled"][:keep]
                out["pooled"] = mx.concatenate([head, new["pooled"]], 0) if "pooled" in new else head
            if "idx_pooled" in par:
                keep = S // 4; head = par["idx_pooled"][:keep]
                out["idx_pooled"] = mx.concatenate([head, new["idx_pooled"]], 0) if "idx_pooled" in new else head
            mx.eval(*out.values())
            tmp = os.path.join(local_dir, "merge_tmp_" + f)    # must END in .safetensors: mx.save_safetensors appends the suffix otherwise
            mx.save_safetensors(tmp, out); os.replace(tmp, os.path.join(local_dir, f))
            n += 1; nbytes += sum(int(v.nbytes) for v in out.values())
        res = {"merged": True, "layers": n, "S": S, "parent": j["stamp"], "bytes": nbytes, "seconds": round(time.time() - t0, 2)}
        L(f"merge_prefix: {res}"); return res
    except Exception as e:
        L(f"merge_prefix failed: {e!r}"); return {"merged": False, "why": repr(e)}
