#!/usr/bin/env python3
"""Tensor-level hash manifest for a safetensors file.

Whole-file hashes are NOT portable for this artifact: the safetensors header embeds
__metadata__, which in our exporter carried the absolute source model path -- a
machine-specific string. Two byte-identical tensor sets therefore produce different
file sizes and different file hashes. Hash the tensors, not the file.

usage: tensor_manifest.py FILE.safetensors > manifest.txt
"""
import hashlib, json, struct, sys

path = sys.argv[1]
with open(path, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    data_start = 8 + n
    entries = sorted((k, v) for k, v in hdr.items() if k != "__metadata__")
    print(f"# tensor-level manifest for {path.split('/')[-1]}")
    print(f"# tensors: {len(entries)}   header_bytes: {n}")
    print(f"# __metadata__: {json.dumps(hdr.get('__metadata__'))}")
    print("# NOTE: __metadata__ is machine-specific (absolute model path) and is EXCLUDED")
    print("#       from the hashes below. Compare these, not the file hash.")
    print("# name\tdtype\tshape\tnbytes\tsha256(raw bytes)")
    agg = hashlib.sha256()
    for name, meta in entries:
        s, e = meta["data_offsets"]
        f.seek(data_start + s)
        h = hashlib.sha256()
        remaining = e - s
        while remaining:
            chunk = f.read(min(1 << 20, remaining))
            if not chunk:
                break
            h.update(chunk); remaining -= len(chunk)
        d = h.hexdigest()
        agg.update(name.encode()); agg.update(d.encode())
        print(f"{name}\t{meta['dtype']}\t{tuple(meta['shape'])}\t{e-s}\t{d}")
    print(f"# AGGREGATE (sha256 over name+tensorhash, sorted): {agg.hexdigest()}")
