#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
verify_blocks.py — compare oMLX paged-SSD cache block files key-by-key.

Pure Python + numpy (no MLX): runs anywhere.

    verify_blocks.py A.safetensors B.safetensors          # two files
    verify_blocks.py dirA dirB                             # match by hash filename
    verify_blocks.py --layout A.safetensors B.safetensors  # shapes/dtypes/metadata only
    verify_blocks.py --self A.safetensors                  # sanity: A vs A -> all zeros

Reports, per tensor key: dtype/shape equality and max |a-b| (bf16 decoded via
uint16<<16 -> float32). Metadata keys are compared except the volatile ones
(block_hash, created_at, model_name unless --strict-meta).
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

_ST_TO_NP = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16, "I64": np.int64, "I32": np.int32, "U32": np.uint32, "U8": np.uint8, "BOOL": np.bool_}
_VOLATILE_META = {"block_hash", "created_at"}


def read_block(path: Path):
    with open(path, "rb") as f:
        L = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(L))
        data_start = 8 + L
        meta = header.pop("__metadata__", {})
        tensors = {}
        for name, info in header.items():
            s, e = info["data_offsets"]
            f.seek(data_start + s)
            raw = f.read(e - s)
            dt = info["dtype"]
            arr = np.frombuffer(raw, dtype=_ST_TO_NP[dt]).reshape(info["shape"]) if raw else np.zeros(info["shape"], dtype=_ST_TO_NP[dt])
            tensors[name] = (dt, tuple(info["shape"]), arr)
    return meta, tensors


def to_float(dt, arr):
    if dt == "BF16":
        return (arr.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    return arr.astype(np.float64)


def compare_files(a: Path, b: Path, layout_only=False, strict_meta=False, quiet=False):
    ma, ta = read_block(a)
    mb, tb = read_block(b)
    rows = []
    ok = True
    # metadata
    skip = set() if strict_meta else _VOLATILE_META | {"model_name"}
    keys = sorted(set(ma) | set(mb))
    meta_diff = [k for k in keys if k not in skip and ma.get(k) != mb.get(k)]
    if meta_diff:
        ok = False
    # tensors
    tkeys = sorted(set(ta) | set(tb))
    worst = 0.0
    for k in tkeys:
        if k not in ta or k not in tb:
            rows.append((k, "MISSING", "", "", ""))
            ok = False
            continue
        dta, sha, aa = ta[k]
        dtb, shb, bb = tb[k]
        if dta != dtb or sha != shb:
            rows.append((k, f"{dta}{list(sha)}", f"{dtb}{list(shb)}", "SHAPE/DTYPE MISMATCH", ""))
            ok = False
            continue
        if layout_only:
            rows.append((k, f"{dta}{list(sha)}", "=", "", "layout-ok"))
            continue
        if dta in ("I64", "I32", "U32", "U8", "BOOL"):
            eq = np.array_equal(aa, bb)
            rows.append((k, f"{dta}{list(sha)}", "=", "0" if eq else "INT DIFF", "" if eq else f"{aa.ravel()[:4]} vs {bb.ravel()[:4]}"))
            ok = ok and eq
            continue
        fa, fb = to_float(dta, aa), to_float(dtb, bb)
        d = np.abs(fa - fb)
        mx_ = float(d.max()) if d.size else 0.0
        worst = max(worst, mx_)
        rows.append((k, f"{dta}{list(sha)}", "=", f"{mx_:.3e}", "bit-exact" if mx_ == 0.0 else f"mean {float(d.mean()):.3e}"))
        if mx_ != 0.0:
            ok = False
    if not quiet:
        print(f"A: {a}\nB: {b}")
        if meta_diff:
            print("METADATA DIFFS:")
            for k in meta_diff:
                print(f"  {k}: A={str(ma.get(k))[:100]!r}  B={str(mb.get(k))[:100]!r}")
        else:
            print(f"metadata: identical ({len(keys)} keys, ignoring {sorted(skip)})")
        print(f"{'key':40s} {'dtype/shape':22s} {'B':3s} {'max|a-b|':12s} note")
        for r in rows:
            print(f"{r[0]:40s} {r[1]:22s} {r[2]:3s} {r[3]:12s} {r[4]}")
        print(f"tensors: {len(tkeys)}  worst max|a-b| = {worst:.3e}  -> {'IDENTICAL' if ok else 'DIFFERENT'}")
    return ok, worst, meta_diff, rows


def compare_dirs(da: Path, db: Path, **kw):
    fa = {p.name: p for p in da.rglob("*.safetensors")}
    fb = {p.name: p for p in db.rglob("*.safetensors")}
    common = sorted(set(fa) & set(fb))
    only_a, only_b = sorted(set(fa) - set(fb)), sorted(set(fb) - set(fa))
    print(f"dir A {da}: {len(fa)} blocks · dir B {db}: {len(fb)} blocks · common {len(common)} · only-A {len(only_a)} · only-B {len(only_b)}")
    all_ok = True
    for name in common:
        ok, worst, md, _ = compare_files(fa[name], fb[name], quiet=True, **kw)
        print(f"  {name[:16]}  {'OK ' if ok else 'DIFF'}  worst={worst:.3e}  meta_diffs={md}")
        all_ok &= ok
    for n in only_a[:5]:
        print(f"  only in A: {n[:16]}")
    for n in only_b[:5]:
        print(f"  only in B: {n[:16]}")
    return all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--layout", action="store_true", help="compare shapes/dtypes/metadata only")
    ap.add_argument("--strict-meta", action="store_true")
    ap.add_argument("--self", action="store_true", help="compare a to itself")
    a = ap.parse_args()
    pa = Path(a.a)
    pb = pa if a.self else Path(a.b)
    if pa.is_dir():
        ok = compare_dirs(pa, pb, layout_only=a.layout, strict_meta=a.strict_meta)
    else:
        ok, *_ = compare_files(pa, pb, layout_only=a.layout, strict_meta=a.strict_meta)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
