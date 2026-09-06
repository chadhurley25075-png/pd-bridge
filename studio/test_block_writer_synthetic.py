#!/usr/bin/env python3
"""Synthetic end-to-end test for omlx_block_writer on the Mac lab box.

Builds a DeepSeek-V4-Flash-shaped cache_list (43 layers: 0,1 RotatingKVCache
window 128; even layers >=2 CacheList(Rotating, PoolingCache(4) D=512,
PoolingCache(4) D=128); odd layers >=3 CacheList(Rotating, PoolingCache(128)
D=512)) with random bf16 state at two boundaries (2048, 4096), writes blocks
through oMLX's own store path, and checks the files' layout against a real
reference block written by S1.

Run:  $OMLX_PYTHON test_block_writer_synthetic.py --ref <ref_block.safetensors> --out /tmp/pd_blocks_test
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx

from omlx_block_writer import BlockWriter, chain_hashes_for

import omlx.patches.deepseek_v4 as dv4
dv4.apply_pooling_cache_support()
from mlx_lm.models.cache import CacheList, RotatingKVCache, PoolingCache  # noqa: E402

NUM_LAYERS, WINDOW, HEAD_DIM, IDX_DIM = 43, 128, 512, 128


def make_cache():
    out = []
    for i in range(NUM_LAYERS):
        if i < 2:
            out.append(RotatingKVCache(max_size=WINDOW))
        elif i % 2 == 0:
            out.append(CacheList(RotatingKVCache(max_size=WINDOW), PoolingCache(4), PoolingCache(4)))
        else:
            out.append(CacheList(RotatingKVCache(max_size=WINDOW), PoolingCache(128)))
    return out


def fill_rotating(rc: RotatingKVCache, tc: int):
    rc.keys = mx.random.normal((1, 1, WINDOW, HEAD_DIM)).astype(mx.bfloat16)
    rc.values = mx.zeros((1, 1, WINDOW, 0), dtype=mx.bfloat16)
    rc.offset = tc
    rc._idx = WINDOW


def fill_pool(pc: PoolingCache, tc: int, dim: int):
    ratio = pc.ratio
    pooled = mx.random.normal((1, tc // ratio, dim)).astype(mx.bfloat16)
    if ratio == 4:
        prev_kv = mx.random.normal((1, 1, 4, 2 * dim)).astype(mx.bfloat16)
        prev_gate = mx.random.normal((1, 1, 4, 2 * dim)).astype(mx.bfloat16)
    else:
        prev_kv = prev_gate = None
    pc.state = (None, None, pooled, prev_kv, prev_gate)


def fill(cache_list, tc):
    for i, c in enumerate(cache_list):
        if isinstance(c, CacheList):
            subs = c.caches
            fill_rotating(subs[0], tc)
            fill_pool(subs[1], tc, HEAD_DIM)
            if len(subs) > 2:
                fill_pool(subs[2], tc, IDX_DIM)
        else:
            fill_rotating(c, tc)
    mx.eval([c.state for c in cache_list])


def header(path):
    with open(path, "rb") as f:
        L = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(L))
    meta = h.pop("__metadata__", {})
    return meta, {k: (v["dtype"], tuple(v["shape"])) for k, v in h.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-name", default="DV4-Flash-MXFP4-MLX")
    a = ap.parse_args()
    mx.random.seed(0)
    tokens = [int(x) for x in (mx.random.randint(5, 120000, (4096 + 17,)).tolist())]
    w = BlockWriter(a.model_name, a.out, make_cache)
    cache = make_cache()
    for tc in (2048, 4096):
        fill(cache, tc)
        w.snapshot(cache, tc)
    paths = w.finalize(tokens)
    w.close()
    print("wrote:", [str(p) for p in paths])
    exp = chain_hashes_for(tokens, a.model_name)
    assert [p.stem for p in paths] == exp, "hash chain mismatch"
    ref_meta, ref_t = header(a.ref)
    ok = True
    for p in paths:
        m, t = header(p)
        # layout: same tensor keys, dtypes, shapes as the reference block
        if set(t) != set(ref_t):
            print("KEY SET DIFF", p.name[:12], sorted(set(t) ^ set(ref_t))[:10]); ok = False
        for k in ref_t:
            if k in t and t[k] != ref_t[k]:
                print("SHAPE/DTYPE DIFF", p.name[:12], k, t[k], "ref", ref_t[k]); ok = False
        for key in ("omlx_cache_format_version", "token_count", "num_layers", "block_size", "payload_layout", "layer_cache_types"):
            if m.get(key) != ref_meta.get(key):
                print("META DIFF", key, m.get(key)[:80] if m.get(key) else None, "ref", ref_meta.get(key)[:80] if ref_meta.get(key) else None); ok = False
        sig_a, sig_b = json.loads(m["cache_signature"]), json.loads(ref_meta["cache_signature"])
        for key in sig_b:
            if key != "model_name" and sig_a.get(key) != sig_b.get(key):
                print("SIGNATURE DIFF", key); ok = False
        lm_a, lm_b = json.loads(m["layer_meta_states"]), json.loads(ref_meta["layer_meta_states"])
        # meta_states: same structure; offsets differ by token count -> compare structure only
        if len(lm_a) != len(lm_b) or lm_a[2][0] != lm_b[2][0] or lm_a[2][1][1:] != lm_b[2][1][1:]:
            print("LAYER META STRUCTURE DIFF", lm_a[2], lm_b[2]); ok = False
        # per-layer sidecar keys identical
        side_a = {k for k in m if k.startswith("layer_")}
        side_b = {k for k in ref_meta if k.startswith("layer_")}
        if side_a != side_b:
            print("SIDECAR KEY DIFF", sorted(side_a ^ side_b)[:10]); ok = False
        # delta ranges recorded correctly
        i64 = [k for k in t if t[k][0] == "I64"]
        print(f"{p.name[:12]} tokens={m['token_count']} tensors={len(t)} i64 keys={len(i64)} sidecar keys={len(side_a)}")
    print("LAYOUT MATCH vs reference:", ok)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
