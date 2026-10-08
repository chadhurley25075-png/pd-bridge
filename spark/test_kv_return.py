#!/usr/bin/env python3
"""test_kv_return.py — proves the KV RETURN bookkeeping in capture_sitecustomize_v3.kvreturn.py (PD_KV_RETURN=1).

CPU-only, synthetic hidden states, REAL projection weights (layers 0..3 = ratios 0,0,4,128; layer 2 has an
indexer). No model, no vLLM, nothing touches a running service.

  1. REFERENCE: one capture fed 0..T2 in chunks -> layer files.
  2. TURN 1:    a capture fed 0..T1, finished -> layer files + kvstate.safetensors/json (the saved end state).
  3. TURN 2:    a fresh _Capture over the same root whose first chunk starts at S (256-aligned, S <= T1) must
                find the saved state, seed itself, and continue to T2. The result must satisfy the MERGE RULE
                the Mac applies:  pooled_ref == cat(turn1.pooled[:S//ratio], turn2.pooled)  and every
                kvwin_b / prev_*_b with b > S plus kvwin_end / buf_* / prev_*_end must equal the reference.
  4. Same again with S = T1 - 1 (vLLM's "whole prompt cached -> recompute the last token" case).
  5. A start with NO covering state must fall back to the old partial path (resume_from None).
Usage: PD_KV_RETURN=1 python3 test_kv_return.py [--weights dv4_proj_weights.safetensors] [--hook FILE]
(needs torch + safetensors; the weights are what `make weights` exports on the decoder)
"""
import argparse, json, os, shutil, sys, tempfile, time
os.environ.setdefault("PD_KV_RETURN", "1")
os.environ["PD_CAPTURE_V3_NOARM"] = "1"
os.environ["PD_CAPTURE_DIR"] = os.environ.get("PD_CAPTURE_DIR") or tempfile.mkdtemp(prefix="kvr_")
os.environ.setdefault("PD_CAPTURE_MIN_T", "64")
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
os.environ.setdefault("PD_POOL_PATH", here)
ap = argparse.ArgumentParser(); ap.add_argument("--weights", default=os.path.join(here, "dv4_proj_weights.safetensors"))
ap.add_argument("--hook", default=None, help="hook file (default: capture_sitecustomize_v3.kvreturn.py next to this test)")
ap.add_argument("--T1", type=int, default=4700); ap.add_argument("--T2", type=int, default=9100)
args = ap.parse_args()
os.environ["PD_PROJ_WEIGHTS"] = args.weights
import torch
from safetensors.torch import load_file
import importlib.util
hook_file = args.hook or os.path.join(here, "capture_sitecustomize_v3.kvreturn.py")
spec = importlib.util.spec_from_file_location("hook", hook_file)
H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
assert H._KV_RETURN, "PD_KV_RETURN=1 not seen by the hook"
DEV = torch.device("cpu"); LAYERS = [0, 1, 2, 3]; HID = 4096
ROOT = os.environ["PD_CAPTURE_DIR"]
fails = []
def check(c, m):
    print(("  ok   " if c else "  FAIL ") + m)
    if not c: fails.append(m)

W = H._Weights(args.weights); assert W.load_cpu(), "weights"
g = torch.Generator().manual_seed(11)
hidden = torch.randn((args.T2, HID), generator=g).to(torch.bfloat16)
POS = torch.arange(args.T2, dtype=torch.int64)

def feed(cap, lo, hi, chunks):
    pos = lo
    while pos < hi:
        n = min(chunks, hi - pos)
        for li in LAYERS:
            Wl = cap.weights.layer(li, DEV)
            kv_pre, kv_score, idx = H.project_layer_torch(hidden[pos:pos + n], Wl)
            cap.ingest(li, Wl["ratio"], kv_pre, kv_score, idx, pos)
        pos += n

def capture(root, lo, hi, chunks, tag):
    cap = H._Capture(root=root, idle_s=30.0, weights=W, start_threads=False); cap.rank = 0
    feed(cap, lo, hi, chunks)
    d = cap.flush(tag)
    man = json.load(open(os.path.join(d, "manifest.json")))
    files = {f: load_file(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.startswith("layer_")}
    return d, man, files, cap

def eq(a, b): return a.shape == b.shape and torch.equal(a, b)

def run_case(S, tag):
    print(f"\n=== case {tag}: T1={args.T1} S={S} T2={args.T2}")
    root = os.path.join(ROOT, tag); os.makedirs(root, exist_ok=True)
    d_ref, man_ref, ref, _ = capture(os.path.join(root, "ref"), 0, args.T2, 2048, "ref")
    d1, man1, t1, _ = capture(root, 0, args.T1, 2048, "turn1")
    st = os.path.join(d1, H._KVR_STATE); mt = os.path.join(d1, H._KVR_META)
    check(os.path.isfile(st) and os.path.isfile(mt), f"turn 1 saved {H._KVR_STATE} + {H._KVR_META} ({os.path.getsize(st)/1e6:.1f} MB for {len(LAYERS)} layers)")
    meta = json.load(open(mt)); check(meta["T"] == args.T1 and meta["keep_extra"] == H._KEEP_EXTRA, f"state meta T={meta['T']} keep_extra={meta['keep_extra']}")
    raw = load_file(st)
    check(raw["layer_02.kv"].shape[0] == H._KV_TAIL and raw["layer_02.main"].shape[0] == H._POOL_TAIL, f"tail rows kv={raw['layer_02.kv'].shape[0]} main={raw['layer_02.main'].shape[0]}")
    per43 = sum(v.numel() * 2 for v in raw.values()) / len(LAYERS) * 43
    print(f"  info state size scaled to 43 layers ~ {per43/1e6:.0f} MB (layers 0/1 have no compressor; real mix is ~half ratio-4)")
    d2, man2, t2, cap2 = capture(root, S, args.T2, 2048, "turn2")
    kr = man2.get("kv_return", {})
    check(kr.get("resume_from") == S and kr.get("parent_stamp") == man1["stamp"], f"turn 2 manifest resume_from={kr.get('resume_from')} parent={kr.get('parent_stamp')}")
    check(man2["T"] == args.T2 and not man2.get("position_gaps"), f"turn 2 T={man2['T']} gaps={man2.get('position_gaps')}")
    check(cap2.errors == 0, "no worker errors")
    ratios = man_ref["ratios"]
    for li in LAYERS:
        f = f"layer_{li:02d}.safetensors"; R, A, B = ref[f], t1[f], t2[f]
        r = ratios[li]
        # kv windows: b <= S from turn 1, b > S from turn 2; end from turn 2
        for k in R:
            if k == "pooled" or k == "idx_pooled": continue
            src = None
            if k.startswith("kvwin_") or k.startswith("prev_") or k.startswith("idx_prev_"):
                b = k.rsplit("_", 1)[1]
                src = B if b == "end" or int(b) > S else A
            else:
                src = B     # buf_* etc = end state
            check(k in src and eq(R[k], src[k]), f"L{li} {k} == {'turn2' if src is B else 'turn1'}")
        if r:
            merged = torch.cat([A["pooled"][:S // r], B["pooled"]], 0)
            check(eq(R["pooled"], merged), f"L{li} pooled: ref == cat(turn1[:S//{r}]={S//r}, turn2={B['pooled'].shape[0]})")
        if "idx_pooled" in R:
            merged = torch.cat([A["idx_pooled"][:S // 4], B["idx_pooled"]], 0)
            check(eq(R["idx_pooled"], merged), f"L{li} idx_pooled: ref == cat(turn1[:S//4], turn2)")
        extra = set(B) - set(R)
        check(not extra, f"L{li} turn 2 has no unexpected keys {sorted(extra)[:4]}")
    # turn 2 saved its own state for turn 3
    check(os.path.isfile(os.path.join(d2, H._KVR_STATE)), "turn 2 saved its own end state")
    # Mac-pushed copy location is honoured: move the state into _kvreturn/<stamp>/ and resume again
    kvr = os.path.join(root, H._KVR_DIR, man2["stamp"]); os.makedirs(kvr, exist_ok=True)
    for n in (H._KVR_STATE, H._KVR_META): shutil.move(os.path.join(d2, n), os.path.join(kvr, n))
    shutil.rmtree(d2)
    S3 = (args.T2 // 256) * 256
    cap3 = H._Capture(root=root, idle_s=30.0, weights=W, start_threads=False); cap3.rank = 0
    feed(cap3, S3, args.T2 + 300, 2048); d3 = cap3.flush("turn3"); man3 = json.load(open(os.path.join(d3, "manifest.json")))
    check(man3["kv_return"]["resume_from"] == S3 and H._KVR_DIR in man3["kv_return"]["state_src"], f"turn 3 resumed at {S3} from the Mac-pushed copy {man3['kv_return']['state_src']}")

run_case((args.T1 // 256) * 256, "aligned")
run_case(args.T1 - 1, "lastminus1")
# no covering state -> old partial path
root = os.path.join(ROOT, "nostate"); os.makedirs(root, exist_ok=True)
cap = H._Capture(root=root, idle_s=30.0, weights=W, start_threads=False); cap.rank = 0
feed(cap, 3000, 5000, 2048); d = cap.flush("x"); man = json.load(open(os.path.join(d, "manifest.json")))
check(man["partial_start"] == 3000 and man["kv_return"]["resume_from"] is None, "no saved state -> partial_start set, resume_from None (old behaviour)")
print("\nRESULT:", "PASS" if not fails else f"FAIL ({len(fails)}): " + "; ".join(fails[:5]))
shutil.rmtree(ROOT, ignore_errors=True)
sys.exit(1 if fails else 0)
