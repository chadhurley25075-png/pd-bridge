#!/usr/bin/env python3
"""pd_assemble_kv.py — decode-side assembly for a plain-attention capture (spark/pd_kv_connector.py), R0 of
docs/RDMA.md. Per-block K/V files -> one KVCache per layer -> oMLX's OWN store pipeline
(BlockAwarePrefixCache.store_cache), which slices the sliceable KVCache into paged blocks and writes them
exactly as the server does. This is the honest baseline the byte-level fast path (R4) is verified against.

Native oMLX 0.6.4 block for Qwen3-32B (read off a server-written file 2026-09-14): 64 MiB, 128 tensors
layer_{i}_state_{0,1} BF16 [1, 8, 256, 128], format version 5, payload_layout split_recurrent_v1.

Library: assemble_and_write_kv(cap_dir, ids, model, model_name, cache_dir) -> (paths, info)
CLI test: pd_assemble_kv.py --model PATH --cap DIR --ids token_ids.json --out DIR --name MODEL_NAME
"""
import json, os, time
import mlx.core as mx


def L(*x): print(time.strftime("%H:%M:%S "), *x, flush=True)


def _load_blocks(cap_dir, man, n_blocks):
    """Contiguous blocks from 0. Stops at the first missing or malformed file (a truncated capture still
    seeds a valid prefix; the decoder prefills the rest natively)."""
    NL, H, D, blk = int(man["layers"]), int(man["kv_heads"]), int(man["head_dim"]), int(man["block"])
    ks, vs, ok, why = [[] for _ in range(NL)], [[] for _ in range(NL)], 0, None
    for i in range(n_blocks):
        p = os.path.join(cap_dir, f"blk_{i:06d}.safetensors")
        if not os.path.isfile(p):
            why = f"missing blk_{i:06d}"; break
        d = mx.load(p)
        k, v = d.get("k"), d.get("v")
        if k is None or v is None or tuple(k.shape) != (NL, H, blk, D) or tuple(v.shape) != (NL, H, blk, D):
            why = f"blk_{i:06d} malformed: {None if k is None else tuple(k.shape)}"; break
        for li in range(NL):
            ks[li].append(k[li]); vs[li].append(v[li])
        ok += 1
    return ks, vs, ok, why


def assemble_and_write_kv(cap_dir, ids, model, model_name, cache_dir):
    from omlx_block_writer import BlockWriter, extract_cache_states
    tm = {}
    t0 = time.time()
    man = json.load(open(os.path.join(cap_dir, "manifest.json")))
    T, blk, NL = len(ids), int(man["block"]), int(man["layers"])
    want = min(int(man["blocks"]), T // blk)       # blocks we both have and can hash against the request
    ks, vs, ok, why = _load_blocks(cap_dir, man, want)
    tm["t_load"] = round(time.time() - t0, 3)
    B = ok * blk
    info = {"blocks_ok": ok, "blocks_claimed": int(man["blocks"]), "B": B, "coverage": round(B / T, 4) if T else 0.0,
            "T_manifest": man.get("T"), "T_request": T}
    if why:
        info["stopped_at"] = why
    if not ok:
        return [], {**info, **tm}

    t0 = time.time()
    from mlx_lm.models.cache import make_prompt_cache
    factory = lambda: make_prompt_cache(model)     # plain models (Qwen3) have no model.make_cache(); this is what oMLX uses too
    cache = factory()
    if len(cache) != NL:
        raise RuntimeError(f"capture has {NL} layers, model cache has {len(cache)}")
    for li, c in enumerate(cache):
        kk = mx.concatenate(ks[li], axis=1) if ok > 1 else ks[li][0]
        vv = mx.concatenate(vs[li], axis=1) if ok > 1 else vs[li][0]
        c.state = (kk[None], vv[None])             # KVCache.state setter also sets offset = B
    mx.eval([c.keys for c in cache] + [c.values for c in cache])
    del ks, vs
    tm["t_assemble"] = round(time.time() - t0, 3)

    t0 = time.time()
    w = BlockWriter(model_name=model_name, out_dir=cache_dir, cache_list_factory=factory, block_size=blk)
    tm["t_writer_init"] = round(time.time() - t0, 3)
    t0 = time.time()
    extracted, cfg = extract_cache_states(cache, model_name)
    tokens = list(ids[:B])
    paths = [w.ssd._get_file_path(h) for h in w.chain_hashes(tokens)]
    w._expected_paths = paths
    table = w.prefix.store_cache("pd-bridge-kv", tokens, extracted, model_cache_config=cfg,
                                 boundary_snapshots=None, extra_keys=None, extra_key_token_start=None,
                                 extra_key_ranges=None, hot_cache_write_back=True)
    if table is None:
        raise RuntimeError("store_cache returned None")
    w._drain()
    missing = [p for p in paths if not p.exists()]
    w.close()
    tm["t_store"] = round(time.time() - t0, 3)
    if missing:
        raise RuntimeError(f"{len(missing)}/{len(paths)} block files not written: {missing[:2]}")
    info["blocks"] = len(paths)
    L(f"kv assemble: {ok} blocks B={B}/{T} load {tm['t_load']}s assemble {tm['t_assemble']}s store {tm['t_store']}s")
    return paths, {**info, **tm}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--cap", required=True)
    ap.add_argument("--ids", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    from mlx_lm import load
    model, _ = load(a.model, lazy=True)            # weights never materialize: only make_cache() is used
    ids = json.load(open(a.ids)); ids = ids["token_ids"] if isinstance(ids, dict) else ids
    paths, info = assemble_and_write_kv(a.cap, ids, model, a.name, a.out)
    print(json.dumps(info))
