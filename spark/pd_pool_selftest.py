#!/usr/bin/env python3
"""pd_pool_selftest.py — exercises capture_sitecustomize_v3's bookkeeping with SYNTHETIC tensors (no model).
(AGENT B, 2026-09-06)

Runs inside the vllm_pd container (or anywhere with torch + safetensors + a GPU; falls back to CPU):
  /opt/env/bin/python /pd_capture/_selftest_v3/pd_pool_selftest.py [--T 5000]
  /opt/env/bin/python /pd_capture/_selftest_v3/pd_pool_selftest.py --smoke --weights /pd_capture/_selftest_v3/dv4_proj_weights.safetensors

Default mode (synthetic weights, layers 0..3 = ratios 0,0,4,128; layer 2 also carries an indexer):
  1. Random hidden states [T,4096] fed through the hook's REAL path (record() -> side-stream projections ->
     worker -> ingest -> flush) as chunks [2048,2048,904], as [1000,3000,1000] (boundaries inside chunks)
     and as one shot [T]  =>  every tensor in every layer_XX.safetensors must be IDENTICAL.
  2. Reference: one-shot pooled/kvwin/prev/buf recomputed directly with pd_pool_torch on the same
     projections must equal the files (torch.equal); manifest must match the v3 format; key sets exact.
  3. Sizes: sum of tensor bytes == closed-form byte estimate == manifest["bytes"]; the estimate is printed
     for T=102,595 / 43 layers (design says ~1.0 GB).
--smoke mode (REAL dv4_proj_weights.safetensors, all 43 layers, random hidden, chunks [8192, T-8192]):
  loads the real file through the hook's loader, runs the full hook path, reports forward-thread enqueue
  cost, pooling cost, write time, bytes vs estimate, and that manifest ratios == config compress_ratios.
Exit status non-zero on any failure. Nothing here touches the vLLM engine.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--T", type=int, default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--device", default=None)
ap.add_argument("--keep", action="store_true")
ap.add_argument("--smoke", action="store_true", help="real weights, 43 layers")
ap.add_argument("--weights", default=os.path.join(HERE, "dv4_proj_weights.safetensors"))
args = ap.parse_args()

ROOT = args.out or tempfile.mkdtemp(prefix="pd_v3_selftest_", dir=os.environ.get("PD_SELFTEST_TMP", None))
os.makedirs(ROOT, exist_ok=True)
WPATH = args.weights if args.smoke else os.path.join(ROOT, "dv4_proj_weights.safetensors")
os.environ["PD_CAPTURE_V3_NOARM"] = "1"      # import the module without installing the import hook
os.environ["PD_CAPTURE_DIR"] = os.path.join(ROOT, "cap")
os.environ["PD_PROJ_WEIGHTS"] = WPATH
os.environ["PD_PROJ_PRELOAD"] = "0"
os.environ.setdefault("PD_CAPTURE_MIN_T", "64")
sys.path.insert(0, HERE)

import torch  # noqa: E402
from safetensors.torch import save_file, load_file  # noqa: E402
import pd_pool_torch as P  # noqa: E402
import capture_sitecustomize_v3 as H  # noqa: E402

DEV = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
HID = 4096
BLOCK = H._BLOCK
CFG_RATIOS = [0, 0] + [4, 128] * 20 + [4]   # 43 real layers (config.json's 44th entry is the MTP layer); in --smoke the sidecar's list is used
fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


def estimate(T, ratios, block=BLOCK, win=128):
    nb = T // block
    total = 0
    for r in ratios:
        total += (nb + 1) * min(win, T) * 512 * 2                        # kvwin_{b} + kvwin_end
        if r:
            od = 512 * (2 if r == 4 else 1)
            total += (T // r) * 512 * 2                                 # pooled
            total += (T % r) * od * 2 * 2                               # buf_kv + buf_gate
        if r == 4:
            total += (nb + 1) * 2 * 4 * 1024 * 2                        # prev_kv/gate per boundary + end
            total += (T // 4) * 128 * 2                                 # idx_pooled
            total += (T % 4) * 256 * 2 * 2                              # idx_buf
            total += (nb + 1) * 2 * 4 * 256 * 2                         # idx_prev per boundary + end
    return total


def run(cap_weights, layers, hidden, chunks, tag):
    """Drive the hook exactly as attention_impl would (one record() per layer per chunk)."""
    T = hidden.shape[0]
    positions_all = torch.arange(T, dtype=torch.int64, device=DEV)
    root = os.path.join(ROOT, "cap", tag)
    cap = H._Capture(root=root, idle_s=30.0, weights=cap_weights, start_threads=True)
    cap.rank = 0
    if DEV.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        m0 = torch.cuda.memory_allocated()
    t0 = time.time()
    pos = 0
    for n in chunks:
        for li in layers:
            name = f"model.layers.{li}.self_attn.deepseek_v4_multi_head_latent_attention"
            if DEV.type == "cuda":
                cap.record(name, hidden[pos:pos + n], positions_all[pos:pos + n])
            else:  # CPU fallback: no streams — project here and ingest directly
                Wl = cap.weights.layer(li, DEV)
                kv_pre, kv_score, idx = H.project_layer_torch(hidden[pos:pos + n], Wl)
                cap.ingest(li, Wl["ratio"], kv_pre, kv_score, idx, pos)
        pos += n
    t_enq = time.time() - t0
    if DEV.type == "cuda":  # an MTP-named call must be ignored
        cap.record("model.mtp.layers.0.self_attn.x", hidden[:8], positions_all[:8])
    d = cap.flush("selftest")
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    t_all = time.time() - t0
    man = json.load(open(os.path.join(d, "manifest.json"))) if d else {}
    mem = f", peak GPU alloc +{(torch.cuda.max_memory_allocated()-m0)/1e9:.2f} GB" if DEV.type == "cuda" else ""
    print(f"[{tag}] T={T} chunks={chunks} layers={len(layers)} -> {d}\n"
          f"       forward-thread time in record() {cap.enqueue_s:.3f}s (wall to enqueue all {t_enq:.3f}s), worker pooling "
          f"{cap.pool_s:.3f}s, write {man.get('write_s')}s, total {t_all:.2f}s, {man.get('bytes',0)/1e6:.1f} MB, "
          f"worker_errors={cap.errors}, projector={cap.projector}{mem}")
    check(cap.errors == 0, f"[{tag}] no worker/capture errors")
    check(d is not None and os.path.isfile(os.path.join(d, "DONE")), f"[{tag}] DONE marker written")
    files = {f: load_file(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.endswith(".safetensors")}
    return files, man


# =============================================================================== smoke: real weights
if args.smoke:
    T = args.T or 8192 + 904
    check(os.path.isfile(WPATH) and os.path.isfile(os.path.splitext(WPATH)[0] + ".json"), f"real weights present: {WPATH}")
    Wd = H._Weights(WPATH)
    t0 = time.time()
    check(Wd.load_cpu(), f"loader accepted the real file ({time.time()-t0:.1f}s, {len(Wd.cpu)} layers, eps={Wd.eps})")
    side = json.load(open(os.path.splitext(WPATH)[0] + ".json")).get("compress_ratios")
    check(side == CFG_RATIOS, f"sidecar compress_ratios (len {len(side)}) == the 43-layer config pattern")
    CFG_RATIOS = side
    check([Wd.ratio(i) for i in range(43)] == CFG_RATIOS, "loader ratios == sidecar compress_ratios")
    g = torch.Generator().manual_seed(7)
    hidden = (torch.randn((T, HID), generator=g)).to(torch.bfloat16).to(DEV)
    files, man = run(Wd, list(range(43)), hidden, [8192, T - 8192] if T > 8192 else [T], "smoke_real")
    check(man.get("ratios") == CFG_RATIOS, "manifest ratios == sidecar compress_ratios")
    check(man.get("num_layers") == 43 and len(files) == 43, f"43 layer files ({len(files)}) / num_layers {man.get('num_layers')}")
    actual = sum(v.numel() * 2 for f in files.values() for v in f.values())
    est = estimate(T, CFG_RATIOS)
    check(actual == est == man.get("bytes"), f"bytes {actual:,} == estimate {est:,} == manifest {man.get('bytes'):,}")
    r4 = files["layer_02.safetensors"]
    check(r4["pooled"].shape == (T // 4, 512) and r4["idx_pooled"].shape == (T // 4, 128), f"layer 2: pooled {tuple(r4['pooled'].shape)} idx_pooled {tuple(r4['idx_pooled'].shape)}")
    check(torch.isfinite(r4["pooled"].float()).all().item() and torch.isfinite(files["layer_03.safetensors"]["pooled"].float()).all().item(), "pooled rows finite (real weights)")
    print(f"  info estimate for T=102,595 / 43 layers: {estimate(102595, CFG_RATIOS)/1e9:.3f} GB (design: ~1.0 GB)")
    print(f"  info rope: {man.get('rope_source')} {man.get('rope')}")

# =============================================================================== default: synthetic
else:
    T = args.T or 5000
    LAYERS = {0: 0, 1: 0, 2: 4, 3: 128}
    g = torch.Generator().manual_seed(1234)

    def rn(*shape, scale=1.0, dtype=torch.bfloat16):
        return (torch.randn(shape, generator=g) * scale).to(dtype)

    W = {}
    for li, r in LAYERS.items():
        W[f"layer_{li}.wkv.weight"] = rn(512, HID, scale=HID ** -0.5)
        W[f"layer_{li}.kv_norm.weight"] = (1 + 0.1 * torch.randn(512, generator=g)).to(torch.bfloat16)
        if r:
            od = 512 * (2 if r == 4 else 1)
            W[f"layer_{li}.comp.wkv.weight"] = rn(od, HID, scale=HID ** -0.5)
            W[f"layer_{li}.comp.wgate.weight"] = rn(od, HID, scale=HID ** -0.5)
            W[f"layer_{li}.comp.ape"] = rn(r, od, scale=0.5, dtype=torch.float32)
            W[f"layer_{li}.comp.norm.weight"] = (1 + 0.1 * torch.randn(512, generator=g)).to(torch.bfloat16)
        if r == 4:
            W[f"layer_{li}.idx.wkv.weight"] = rn(256, HID, scale=HID ** -0.5)
            W[f"layer_{li}.idx.wgate.weight"] = rn(256, HID, scale=HID ** -0.5)
            W[f"layer_{li}.idx.ape"] = rn(4, 256, scale=0.5, dtype=torch.float32)
            W[f"layer_{li}.idx.norm.weight"] = (1 + 0.1 * torch.randn(128, generator=g)).to(torch.bfloat16)
    save_file(W, WPATH)
    json.dump({"rms_norm_eps": 1e-6, "compress_ratios": [LAYERS[i] for i in range(4)], "head_dim": 512,
               "qk_rope_head_dim": 64, "index_head_dim": 128, "hidden_size": HID, "rope_theta": 10000,
               "compress_rope_theta": 160000, "rope_scaling": P.ROPE_SCALING, "max_position_embeddings": 1048576,
               "sliding_window": 128, "note": "SYNTHETIC selftest weights"}, open(os.path.splitext(WPATH)[0] + ".json", "w"))
    hidden = rn(T, HID).to(DEV)
    print(f"device={DEV} T={T} layers={LAYERS} block={BLOCK} weights={WPATH}")
    A, manA = run(H._Weights(WPATH), list(LAYERS), hidden, [2048, 2048, T - 4096] if T > 4096 else [T], "chunked_2048")
    B, manB = run(H._Weights(WPATH), list(LAYERS), hidden, [T], "oneshot")
    C, manC = run(H._Weights(WPATH), list(LAYERS), hidden, [1000, 3000, T - 4000] if T > 4000 else [T], "chunked_inside")

    # ---- 1. chunked == one-shot
    for tag, X in (("chunked_2048", A), ("chunked_inside", C)):
        n_before = len(fails)
        check(set(X) == set(B), f"[{tag}] same layer files as one-shot: {sorted(B)}")
        for f in sorted(B):
            ka, kb = set(X.get(f, {})), set(B[f])
            check(ka == kb, f"[{tag}] {f} same keys ({len(kb)}): missing={sorted(kb-ka)} extra={sorted(ka-kb)}")
            for k in sorted(kb & ka):
                a, b = X[f][k], B[f][k]
                if not (a.shape == b.shape and torch.equal(a, b)):
                    d = (a.float() - b.float()).abs().max().item() if a.shape == b.shape else float("nan")
                    check(False, f"[{tag}] {f}:{k} differs (shapes {tuple(a.shape)} vs {tuple(b.shape)}, max|d|={d:.3g})")
        if len(fails) == n_before:
            print(f"  ok   [{tag}] every tensor in every layer identical to one-shot")

    # ---- 2. reference recompute (one-shot files vs direct math)
    bounds = list(range(BLOCK, T + 1, BLOCK))
    check(manB["T"] == T and manB["end"] == T and manB["boundaries"] == bounds and manB["num_layers"] == 4
          and manB["ratios"] == [0, 0, 4, 128], f"manifest v3 fields T/end/boundaries/num_layers/ratios = "
          f"{manB['T']}/{manB['end']}/{manB['boundaries']}/{manB['num_layers']}/{manB['ratios']}")
    Wd = H._Weights(WPATH)
    Wd.load_cpu()
    proj = P.project_layer if hasattr(P, "project_layer") else H.project_layer_torch
    for li, r in LAYERS.items():
        f = B[f"layer_{li:02d}.safetensors"]
        Wl = {k: (v.to(DEV) if hasattr(v, "to") else v) for k, v in Wd.cpu[li].items()}
        kv_pre, kv_score, idx = proj(hidden, Wl)
        exp_keys = {f"kvwin_{b}" for b in bounds} | {"kvwin_end"}
        for b in bounds:
            check(torch.equal(f[f"kvwin_{b}"].to(DEV), kv_pre[b - 128:b]), f"layer {li} kvwin_{b} == kv_pre[{b-128}:{b}]")
        check(torch.equal(f["kvwin_end"].to(DEV), kv_pre[T - 128:]), f"layer {li} kvwin_end == kv_pre[T-128:T]")
        if r:
            rope = P.DSv4Rope(64, 160000.0, P.ROPE_SCALING, 1048576, r)
            pooled, carry = P.pool_layer(kv_score, r, 512, Wl["comp_ape"], Wl["comp_norm"], Wl["eps"], rope, 0, None)
            check(f["pooled"].shape == (T // r, 512) and torch.equal(f["pooled"].to(DEV), pooled), f"layer {li} pooled [{T//r},512] == pd_pool_torch one-shot")
            exp_keys |= {"pooled"}
            rem = T % r
            od = 512 * (2 if r == 4 else 1)
            if rem:
                exp_keys |= {"buf_kv", "buf_gate"}
                check(torch.equal(f["buf_kv"].to(DEV), kv_score[T - rem:, :od]) and torch.equal(f["buf_gate"].to(DEV), kv_score[T - rem:, od:]),
                      f"layer {li} buf_kv/buf_gate == raw rows [{T-rem},{T}) (rem {rem})")
                check(torch.equal(f["buf_kv"].to(DEV), carry["buf_kv"]), f"layer {li} buf_kv == pool_layer carry buf_kv")
            if r == 4:
                for b in bounds + ["end"]:
                    e = T - rem if b == "end" else b
                    exp_keys |= {f"prev_kv_{b}", f"prev_gate_{b}"}
                    check(torch.equal(f[f"prev_kv_{b}"].to(DEV), kv_score[e - 4:e, :od]) and torch.equal(f[f"prev_gate_{b}"].to(DEV), kv_score[e - 4:e, od:]),
                          f"layer {li} prev_kv/gate_{b} == raw rows [{e-4},{e})")
                check(torch.equal(f["prev_kv_end"].to(DEV), carry["prev_kv"]), f"layer {li} prev_kv_end == pool_layer carry prev_kv")
                irope = P.DSv4Rope(64, 160000.0, P.ROPE_SCALING, 1048576, 4)
                ipooled, icarry = P.pool_layer(idx, 4, 128, Wl["idx_ape"], Wl["idx_norm"], Wl["eps"], irope, 0, None)
                check(f["idx_pooled"].shape == (T // 4, 128) and torch.equal(f["idx_pooled"].to(DEV), ipooled), f"layer {li} idx_pooled [{T//4},128] == pd_pool_torch one-shot")
                exp_keys |= {"idx_pooled"}
                if T % 4:
                    exp_keys |= {"idx_buf_kv", "idx_buf_gate"}
                for b in bounds + ["end"]:
                    e = T - T % 4 if b == "end" else b
                    exp_keys |= {f"idx_prev_kv_{b}", f"idx_prev_gate_{b}"}
                    check(torch.equal(f[f"idx_prev_kv_{b}"].to(DEV), idx[e - 4:e, :256]), f"layer {li} idx_prev_kv_{b} == raw idx rows [{e-4},{e})")
        check(set(f) == exp_keys, f"layer {li} key set exactly per v3 format: extra={sorted(set(f)-exp_keys)} missing={sorted(exp_keys-set(f))}")
        bad = [k for k, v in f.items() if not (v.dtype == torch.bfloat16 and v.dim() == 2)]
        check(not bad, f"layer {li} all tensors bf16 2-D (no batch dims){'' if not bad else ': ' + str(bad)}")

    # ---- 3. byte estimate
    actual = sum(v.numel() * 2 for f in B.values() for v in f.values())
    est = estimate(T, [LAYERS[i] for i in range(4)])
    check(actual == est == manB["bytes"], f"tensor bytes {actual:,} == closed-form estimate {est:,} == manifest bytes {manB['bytes']:,}")
    print(f"  info estimate for T=102,595 / 43 layers (config ratios): {estimate(102595, CFG_RATIOS)/1e9:.3f} GB  (design: ~1.0 GB)")
    cdir = os.path.join(ROOT, "cap", "oneshot", manB["stamp"])
    disk = sum(os.path.getsize(os.path.join(cdir, f)) for f in os.listdir(cdir))
    print(f"  info on-disk one-shot capture dir: {disk/1e6:.2f} MB (tensors {actual/1e6:.2f} MB + headers/manifest); files: {sorted(os.listdir(cdir))}")

print()
if fails:
    print(f"SELFTEST FAILED: {len(fails)} check(s)")
    for f in fails:
        print("   -", f)
    rc = 1
else:
    print("SELFTEST PASSED" + ("" if args.smoke else ": chunked == one-shot (all tensors identical), reference math matches, v3 key set + byte estimate exact"))
    rc = 0
if not args.keep and not args.out:
    shutil.rmtree(ROOT, ignore_errors=True)
else:
    print(f"kept {ROOT}")
sys.exit(rc)
