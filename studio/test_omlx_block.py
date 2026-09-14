#!/usr/bin/env python3
"""test_omlx_block.py — byte-exact check of spark/pd_omlx_block.py against blocks oMLX wrote itself (R4 gate).

For every sampled block file: rebuild the header from its own block_hash and created_at and require the bytes to
match the file's header exactly, and the file size to match header + tensor bytes. Optionally, with a token-id file
for a prompt oMLX has served, require chain_hashes() to equal oMLX's compute_block_hash chain.
Run in the oMLX venv:
    ~/omlx/bin/python studio/test_omlx_block.py ~/.omlx/cache ~/pd_verify/ids_901.json
"""
import glob
import json
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "spark"))
import pd_omlx_block as ob  # noqa: E402


def main():
    cache = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/.omlx/cache")
    files = sorted(glob.glob(os.path.join(cache, "*", "*.safetensors")))
    if not files:
        sys.exit(f"no block files under {cache}")
    sample = files[:: max(1, len(files) // 40)][:40]
    exact = 0
    for f in sample:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            raw = fh.read(n)
        doc = json.loads(raw)
        meta = doc["__metadata__"]
        shape = doc["layer_0_state_0"]["shape"]
        hdr, data_len = ob.header(bytes.fromhex(meta["block_hash"]), meta["model_name"], int(meta["num_layers"]),
                                  shape[1], int(meta["block_size"]), shape[3], created_at=meta["created_at"])
        same = hdr == struct.pack("<Q", n) + raw
        size_ok = os.path.getsize(f) == 8 + n + data_len
        exact += same and size_ok
        if not (same and size_ok):
            mine = json.loads(hdr[8:])
            diff = [k for k in set(meta) | set(mine["__metadata__"]) if meta.get(k) != mine["__metadata__"].get(k)]
            print(f"MISMATCH {os.path.basename(f)[:16]} header_equal={same} size_ok={size_ok} metadata_diffs={diff[:5]} "
                  f"len native={n} mine={len(hdr) - 8}")
    print(f"headers byte-identical: {exact}/{len(sample)} sampled blocks ({len(files)} in cache)")

    if len(sys.argv) > 2:
        from omlx.cache.paged_cache import compute_block_hash
        ids = json.load(open(os.path.expanduser(sys.argv[2])))
        ids = ids["token_ids"] if isinstance(ids, dict) else ids
        name = json.loads(open(sample[0], "rb").read(8 + struct.unpack("<Q", open(sample[0], "rb").read(8))[0])[8:])["__metadata__"]["model_name"]
        mine = ob.chain_hashes(ids, name, 256)
        ref, parent = [], None
        for i in range(len(ids) // 256):
            parent = compute_block_hash(parent, ids[i * 256:(i + 1) * 256], extra_keys=None, model_name=name)
            ref.append(parent)
        print(f"chain hashes equal to oMLX: {sum(a == b for a, b in zip(mine, ref))}/{len(ref)}")
        ok = exact == len(sample) and mine == ref
    else:
        ok = exact == len(sample)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
