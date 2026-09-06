#!/usr/bin/env python3
"""pd_pool_validate.py — validate pd_pool_torch against MLX truth exported by studio/pd_export_pool_truth.py.

Runs pool_layer whole and chunked (2048 = oMLX schedule, plus odd chunk lengths) on the exported
kv_c/gate and compares to the MLX pooled truth (real Compressor + PoolingCache, 2048 chunks).
Also validates the carry (remainder rows + prev window), the RoPE probe, the SWA kv window RoPE,
and cross-checks against the v3_truth_23k capture if present. Writes POOL-VALIDATION.md.
usage: $PYTHON pd_pool_validate.py [--truth $PD_HOME/pool_truth] [--device cpu]
"""
import argparse, json, os, struct, time
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pd_pool_torch import DSv4Rope, pool_layer, pool_chunks, new_carry, rmsnorm

ap = argparse.ArgumentParser()
ap.add_argument("--truth", default=os.path.expanduser("$PD_HOME/pool_truth"))
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "POOL-VALIDATION.md"))
args = ap.parse_args()
D = args.truth
dev = torch.device(args.device)
lines = []
def P(s=""):
    print(s, flush=True); lines.append(s)

def npy(name):
    return np.load(os.path.join(D, name + ".npy"))
def bf(name):
    return torch.from_numpy(npy(name)).to(torch.bfloat16).to(dev)   # exported values are bf16-exact
def f32(name):
    return torch.from_numpy(npy(name)).to(torch.float32).to(dev)
def meta(name):
    return json.load(open(os.path.join(D, name + ".json")))

def stats(a: torch.Tensor, b: torch.Tensor):
    """a = torch, b = truth. Returns dict of max|Δ|, mean|Δ|, cosine, rel (max|Δ| / max|b|), exact fraction."""
    af, bfl = a.to(torch.float32).flatten(), b.to(torch.float32).flatten()
    d = (af - bfl).abs()
    cos = torch.nn.functional.cosine_similarity(af, bfl, dim=0).item()
    return {"max": d.max().item(), "mean": d.mean().item(), "cos": cos,
            "rel": (d.max() / bfl.abs().max()).item(), "exact": (d == 0).float().mean().item(),
            "n": af.numel()}
def fmt(s):
    return (f"max|Δ| {s['max']:.3e}  mean|Δ| {s['mean']:.3e}  rel {s['rel']:.2e}  cos {s['cos']:.6f}  "
            f"bit-exact {100*s['exact']:.2f}%  (n={s['n']})")

def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, v in hdr.items():
            if k == "__metadata__": continue
            s, e = v["data_offsets"]; f.seek(base + s); raw = f.read(e - s)
            assert v["dtype"] == "BF16", (k, v["dtype"])
            t = torch.frombuffer(bytearray(raw), dtype=torch.uint16).view(torch.bfloat16).reshape(v["shape"])
            out[k] = t.clone()
        return out

man = meta("manifest")
T, CH = man["T"], man["chunk"]
P(f"# POOL-VALIDATION — torch port (`pd_pool_torch.py`) vs MLX truth (`pd_export_pool_truth.py` on the Mac decode node)")
P(f"Run {time.strftime('%Y-%m-%d %H:%M:%S')} on {os.uname().nodename} · torch {torch.__version__} · device {dev} · "
  f"truth T={T}, MLX chunk={CH}, model {os.path.basename(man['model'])}")
P(f"Config: compress_rope_theta {man['compress_rope_theta']}, rope_theta {man['rope_theta']}, "
  f"rope_scaling {man['rope_scaling']}, rms_norm_eps {man['rms_norm_eps']}")
P()
P("bf16 ulp at |x|~1 is 7.8e-3; 'rel' = max|Δ| / max|truth|. bit-exact = fraction of elements identical.")
P()

odd_lens = [1000, 3, 777, 2048, 1, 129, 4097, 2]
def odd_schedule(T):
    lens = []; s = 0; i = 0
    while s < T:
        n = min(odd_lens[i % len(odd_lens)], T - s); lens.append(n); s += n; i += 1
    return lens

results = {}
for name in ["layer_02", "layer_03", "layer_42", "layer_02_indexer"]:
    m = meta(name + "_meta")
    ratio, hd, out_dim, eps = m["ratio"], m["head_dim"], m["out_dim"], m["eps"]
    kv_c, gate = bf(name + "_kv_c"), bf(name + "_gate")
    ape, norm_w = f32(name + "_ape"), bf(name + "_norm_w")
    truth = bf(name + "_pooled")
    rope = DSv4Rope(m["rope"]["dims"], m["rope"]["base"], m["rope"]["scaling"], freq_scale=ratio)
    P(f"## {name}  (ratio {ratio}, head_dim {hd}, out_dim {out_dim}, eps {eps}, rope base {m['rope']['base']} "
      f"yarn={'yes' if m['rope']['scaling'] else 'no'} freq_scale {ratio}; MLX dtypes: kv {m['kv_dtype'].split('.')[-1]}, "
      f"ape {m['ape_dtype'].split('.')[-1]}, norm_w {m['norm_w_dtype'].split('.')[-1]})")
    # rope freqs
    fr = f32(name + "_rope_freqs")
    P(f"- rope `_freqs` (unscaled) torch vs MLX: max rel diff {((rope._freqs.to(dev)-fr).abs()/fr.abs()).max().item():.2e}")
    # rope probe
    pin, pout = bf(name + "_rope_probe_in"), bf(name + "_rope_probe_out")
    s = stats(rope(pin, m["rope_probe_offset"]), pout)
    P(f"- rope probe (random bf16 [64,{hd}] at offset {m['rope_probe_offset']} → pooled pos {m['rope_probe_offset']//ratio}): {fmt(s)}")
    kv_score = torch.cat([kv_c, gate], dim=-1)
    t0 = time.time()
    whole, cw = pool_layer(kv_score, ratio, hd, ape, norm_w, eps, rope, 0, None)
    tw = time.time() - t0
    assert whole.shape == truth.shape, (whole.shape, truth.shape)
    sw = stats(whole, truth)
    P(f"- WHOLE (one call, {T} rows → {tuple(whole.shape)} in {tw:.2f}s): {fmt(sw)}")
    lens2048 = [CH] * (T // CH) + ([T % CH] if T % CH else [])
    ch2048, c2048 = pool_chunks(kv_score, lens2048, ratio, hd, ape, norm_w, eps, rope)
    s2 = stats(ch2048, truth)
    P(f"- CHUNKED 2048 (MLX schedule, {len(lens2048)} chunks): {fmt(s2)}  · == whole: {torch.equal(ch2048, whole)}")
    lens_odd = odd_schedule(T)
    chodd, codd = pool_chunks(kv_score, lens_odd, ratio, hd, ape, norm_w, eps, rope)
    P(f"- CHUNKED odd lengths ({len(lens_odd)} chunks, e.g. {lens_odd[:6]}): == whole: {torch.equal(chodd, whole)}"
      f"  · vs truth: {fmt(stats(chodd, truth))}")
    # where is the error? split nope vs rope features
    nope = hd - m["rope"]["dims"]
    sn = stats(whole[:, :nope], truth[:, :nope]); sr = stats(whole[:, nope:], truth[:, nope:])
    P(f"- error split: nope features [0,{nope}) {fmt(sn)}")
    P(f"                rope features [{nope},{hd}) {fmt(sr)}")
    # carry
    rem = m["remainder"]
    if rem:
        tb_kv, tb_gate = bf(name + "_buf_kv"), bf(name + "_buf_gate")
        okb = torch.equal(cw["buf_kv"], tb_kv) and torch.equal(cw["buf_gate"], tb_gate)
        P(f"- carry remainder rows: {rem} (torch {cw['buf_kv'].shape[0]}) — buf_kv/buf_gate bit-exact vs MLX state: {okb}"
          f" · chunked carries == whole: {torch.equal(c2048['buf_kv'], cw['buf_kv']) and torch.equal(codd['buf_kv'], cw['buf_kv'])}")
    else:
        P(f"- carry remainder rows: 0 (torch {None if cw['buf_kv'] is None else cw['buf_kv'].shape[0]})")
    if ratio == 4:
        tp_kv, tp_gate = bf(name + "_prev_win_kv"), bf(name + "_prev_win_gate")
        okp = torch.equal(cw["prev_kv"], tp_kv) and torch.equal(cw["prev_gate"], tp_gate)
        P(f"- carry prev window [4,{out_dim}]: bit-exact vs MLX prev_win_kv/gate: {okp}"
          f" · chunked == whole: {torch.equal(c2048['prev_kv'], cw['prev_kv']) and torch.equal(codd['prev_kv'], cw['prev_kv'])}")
    # v3_truth cross-check
    v3 = os.path.join(D, "v3_truth_23k", f"{name[:8]}.safetensors")
    if os.path.exists(v3):
        st = load_safetensors(v3)
        key = "idx_pooled" if name.endswith("indexer") else "pooled"
        if key in st:
            v3p = st[key].to(dev)
            P(f"- v3_truth_23k `{key}` {tuple(v3p.shape)} vs pool_truth pooled: bit-exact {torch.equal(v3p, truth)}"
              f" · torch vs v3_truth: {fmt(stats(whole, v3p))}")
            pk = "idx_prev_kv_end" if name.endswith("indexer") else "prev_kv_end"
            if ratio == 4 and pk in st:
                pg = pk.replace("kv", "gate")
                P(f"- v3_truth {pk}/{pg} vs torch carry: bit-exact "
                  f"{torch.equal(st[pk].to(dev), cw['prev_kv']) and torch.equal(st[pg].to(dev), cw['prev_gate'])}")
    results[name] = sw
    P()

# ---- SWA kv window rope ----
P("## SWA kv window RoPE (kv = kv_norm(wkv(x)) rows [T-128,T), torch rope vs attn.rope(kv, offset=T-128))")
for li in man["kv_layers"]:
    name = f"layer_{li:02d}"
    m = meta(name + "_kv_meta")
    pre, post = bf(name + "_kv_win_prerope"), bf(name + "_kv_win_rope")
    rope = DSv4Rope(m["rope"]["dims"], m["rope"]["base"], m["rope"]["scaling"], freq_scale=1)
    fr = f32(name + "_kv_rope_freqs")
    s = stats(rope(pre, m["offset"]), post)
    nope = 512 - m["rope"]["dims"]
    sr = stats(rope(pre, m["offset"])[:, nope:], post[:, nope:])
    P(f"- {name} ({m['attn_class']}, rope base {m['rope']['base']}, yarn={'yes' if m['rope']['scaling'] else 'no'}, "
      f"offset {m['offset']}): {fmt(s)}")
    P(f"    rope features only: {fmt(sr)} · `_freqs` max rel diff {((rope._freqs.to(dev)-fr).abs()/fr.abs()).max().item():.2e}")
    v3 = os.path.join(D, "v3_truth_23k", f"{name}.safetensors")
    if os.path.exists(v3):
        st = load_safetensors(v3)
        if "kvwin_end" in st:
            P(f"    v3_truth kvwin_end vs pool_truth prerope window: bit-exact {torch.equal(st['kvwin_end'].to(dev), pre)}")
    results[name + "_kvwin"] = s
P()
# ---- hidden-state path: cap23k hidden → project_layer (Mac's dequantized weights) → pool_layer → v3_truth ----
from pd_pool_torch import project_layer, load_proj_weights, make_ropes, read_safetensors
WP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dv4_proj_weights.safetensors")
HP = os.path.join(D, "hidden_23k.safetensors")
if os.path.exists(WP) and os.path.exists(HP):
    P("## HIDDEN-STATE PATH: cap23k attention inputs → `project_layer` (dequantized Mac weights, torch) → `pool_layer` → v3_truth_23k")
    P(f"weights `{os.path.basename(WP)}` {os.path.getsize(WP)/1e9:.3f} GB · hidden `{os.path.basename(HP)}`")
    hidden = read_safetensors(HP, device=dev)
    hl = sorted(int(k.split("_")[1]) for k in hidden)
    Wall = load_proj_weights(WP, layers=hl, device=dev)
    ropes = make_ropes(Wall["meta"])
    for li in hl:
        x = hidden[f"layer_{li:02d}"]; W = Wall[li]; ratio = W["ratio"]; name = f"layer_{li:02d}"
        v3 = load_safetensors(os.path.join(D, "v3_truth_23k", f"{name}.safetensors"))
        t0 = time.time(); kv_pre, kv_score, idx_score = project_layer(x, W); tproj = time.time() - t0
        P(f"### {name} (ratio {ratio}, {Wall['meta']['layers'][str(li)]['attn_class']}, wkv {Wall['meta']['layers'][str(li)]['wkv_kind']}; "
          f"project_layer {T} rows in {tproj:.2f}s on {dev})")
        s = stats(kv_pre[-128:], v3["kvwin_end"].to(dev))
        P(f"- kv_pre rows [T-128,T) vs v3 `kvwin_end` (pre-RoPE): {fmt(s)}"); results[f"hidden→{name} kvwin_pre"] = s
        rope_swa = ropes["swa_local"] if li < 2 else ropes["swa"]
        if os.path.exists(os.path.join(D, name + "_kv_win_rope.npy")):
            s = stats(rope_swa(kv_pre[-128:], T - 128), bf(name + "_kv_win_rope"))
            P(f"- + torch RoPE (base {rope_swa.base:.0f}, yarn={'yes' if rope_swa.yarn_cfg else 'no'}) vs MLX attn.rope(kv, {T-128}): {fmt(s)}")
            results[f"hidden→{name} kvwin_rope"] = s
        if ratio > 0:
            od = W["comp_out_dim"]
            if os.path.exists(os.path.join(D, name + "_kv_c.npy")):
                P(f"- projection only: torch kv_c vs MLX project(x) kv_c: {fmt(stats(kv_score[:, :od], bf(name + '_kv_c')))}")
                P(f"                   torch gate vs MLX project(x) gate: {fmt(stats(kv_score[:, od:], bf(name + '_gate')))}")
            rope_c = ropes["comp4"] if ratio == 4 else ropes["comp128"]
            t0 = time.time(); whole, cw = pool_layer(kv_score, ratio, 512, W["comp_ape"], W["comp_norm"], W["eps"], rope_c, 0, None); tpool = time.time() - t0
            lens2048 = [CH] * (T // CH) + ([T % CH] if T % CH else [])
            ch, cc = pool_chunks(kv_score, lens2048, ratio, 512, W["comp_ape"], W["comp_norm"], W["eps"], rope_c)
            s = stats(whole, v3["pooled"].to(dev)); results[f"hidden→{name} pooled"] = s
            P(f"- pooled {tuple(whole.shape)} vs v3 `pooled` (pool {tpool:.2f}s): {fmt(s)} · chunked2048 == whole: {torch.equal(ch, whole)}")
            okb = torch.equal(cw["buf_kv"], v3["buf_kv"].to(dev)) and torch.equal(cw["buf_gate"], v3["buf_gate"].to(dev)) if "buf_kv" in v3 else None
            sb = stats(cw["buf_kv"], v3["buf_kv"].to(dev)) if "buf_kv" in v3 else None
            P(f"- carry remainder {v3['buf_kv'].shape[0] if 'buf_kv' in v3 else 0} rows vs v3 `buf_kv`: all-elements-identical {okb}" + (f" · {fmt(sb)}" if sb else ""))
            if ratio == 4:
                sp = stats(cw["prev_kv"], v3["prev_kv_end"].to(dev)); sg = stats(cw["prev_gate"], v3["prev_gate_end"].to(dev))
                P(f"- carry prev window vs v3 `prev_kv_end`: {fmt(sp)}"); P(f"                     vs v3 `prev_gate_end`: {fmt(sg)}")
                results[f"hidden→{name} prev_kv"] = sp
                t0 = time.time(); iw, ic = pool_layer(idx_score, 4, 128, W["idx_ape"], W["idx_norm"], W["eps"], ropes["idx"], 0, None); tpi = time.time() - t0
                ich, _ = pool_chunks(idx_score, lens2048, 4, 128, W["idx_ape"], W["idx_norm"], W["eps"], ropes["idx"])
                s = stats(iw, v3["idx_pooled"].to(dev)); results[f"hidden→{name} idx_pooled"] = s
                P(f"- indexer pooled {tuple(iw.shape)} vs v3 `idx_pooled` (pool {tpi:.2f}s): {fmt(s)} · chunked2048 == whole: {torch.equal(ich, iw)}")
                if os.path.exists(os.path.join(D, name + "_indexer_kv_c.npy")):
                    P(f"- indexer projection only: torch vs MLX kv_c: {fmt(stats(idx_score[:, :256], bf(name + '_indexer_kv_c')))}")
                P(f"- indexer carry: prev_kv_end {fmt(stats(ic['prev_kv'], v3['idx_prev_kv_end'].to(dev)))}; "
                  f"buf bit-exact {torch.equal(ic['buf_kv'], v3['idx_buf_kv'].to(dev))}")
        P()
else:
    P(f"(hidden-state path skipped: missing {WP} or {HP})")
P()
P("## Summary")
P("| tensor | max abs Δ | mean abs Δ | rel | cosine | bit-exact |")
P("|---|---|---|---|---|---|")
for k, s in results.items():
    P(f"| {k} | {s['max']:.3e} | {s['mean']:.3e} | {s['rel']:.2e} | {s['cos']:.6f} | {100*s['exact']:.1f}% |")
worst = max(s["rel"] for s in results.values())
P()
P(f"Worst relative max|Δ| across all tensors: {worst:.2e}  → {'PASS (bf16-level)' if worst < 2e-2 else 'FAIL — port bug'}")
P()
P("## MLX semantics established by probes on the Mac decode node (`studio/pd_probe_mlx_semantics.py`, `pd_probe_mlx_sum.py`, `pd_probe_mlx_sum2.py`)")
P("- Projections: `wkv` is QuantizedLinear **mxfp8** (group 32, 8-bit, no biases) on every layer — not MXFP4 despite the model-dir name;")
P("  compressor / indexer `wkv`,`wgate` are plain bf16 `nn.Linear`; norm weights bf16; `ape` f32. torch f32-matmul → bf16 reproduces")
P("  MLX's projections bit-exactly on layers 0/2/42 (layer 3: rare 1-ulp flips from accumulation order).")
P("- `_overlap_compress_kv`: gate + ape.astype(bf16) in bf16; softmax(precise) → bf16 weights; product in bf16; sum = MLX bf16 reduction.")
P("- `_simple_compress_kv`: weights = softmax(gate.f32 + ape) in f32 → bf16; product bf16; sum = MLX bf16 reduction.")
P("- **MLX `sum` over bf16 does NOT accumulate in f32.** 8 rows (col_reduce_small): serial bf16 in row order (100% match).")
P("  128 rows (col_reduce_looped BM=32): 32 strided serial-bf16 partials (rows j, j+32, j+64, j+96) combined in f32 (100% match).")
P("  f32 accumulation matched only 50% (R=8) / 81% (R=128) of elements — that was the whole 1-ulp mismatch in the first run.")
P("- RMSNorm (`mx.fast.rms_norm`): out = w * bf16(x * rsqrt(mean(x²)+eps)) with the normalizer in f32 → two roundings (100% match).")
P("- RoPE (`mx.fast.rope`, traditional=True, freqs): theta = (offset//freq_scale + i) * (freq_scale / freqs_i) in f32, precise cos/sin,")
P("  interleaved pairs, only the trailing 64 features; leading pairs have freqs=inf ⇒ untouched (100% match at pos 5740 and 23089).")
P("- SWA-kv RoPE base: layers 0,1 (LocalAttention) = rope_theta 10000 with NO yarn; layers ≥2 = compress_rope_theta 160000 WITH yarn")
P("  (design doc v3 §1 says 'yarn, base 10000' for the window — wrong for every layer: 0/1 have no yarn, ≥2 use base 160000).")
P("- Indexer compressor = Compressor(config, 4, 128): rope dims 64 on head_dim 128, base 160000, yarn, freq_scale 4 (design §3 CHECK confirmed).")
P("- PoolingCache prompt mode: pool_base = offset − remainder; remainder rows carried raw; prev window (ratio 4) = raw last window (no ape);")
P("  first window of a sequence pools with zero lane-A / −inf gate. Chunked == whole is bit-exact in the port for any chunk lengths.")
open(args.out, "w").write("\n".join(lines) + "\n")
print(f"\nwrote {args.out}")
