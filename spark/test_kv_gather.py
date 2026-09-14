#!/usr/bin/env python3
"""test_kv_gather.py — pins pd_kv_connector's KV layout detection and row gather without a GPU or a model.
Every cache cell holds a value that encodes (block, head, offset, dim, K|V), so a wrong axis, a swapped K/V
half or a wrong slot formula shows up as a concrete mismatch. Run with the vLLM venv (the module imports vLLM):
    PYTHONPATH=spark ~/vllm-env/bin/python spark/test_kv_gather.py
"""
import torch

from pd_kv_connector import PdKvConnector as C

NB, H, N, D = 6, 3, 4, 5      # blocks, kv heads, tokens per paged block, head dim


def cell(b, h, o, d, which):
    return 1000 * b + 100 * h + 10 * o + d + (0.5 if which == "v" else 0.0)


def truth(blocks, positions):
    k = torch.zeros(len(positions), H, D); v = torch.zeros(len(positions), H, D)
    for i, p in enumerate(positions):
        b, o = blocks[p // N], p % N
        for h in range(H):
            for d in range(D):
                k[i, h, d] = cell(b, h, o, d, "k"); v[i, h, d] = cell(b, h, o, d, "v")
    return k, v


def packed_cache():
    """vLLM 0.28 FLASH_ATTN logical view (num_blocks, H, N, 2D), K in the first half — built from an NHD memory
    layout and permuted, like attn_utils._reshape_kv_cache, so the test also covers a non-contiguous view."""
    mem = torch.zeros(NB, N, H, 2 * D)
    for b in range(NB):
        for o in range(N):
            for h in range(H):
                for d in range(D):
                    mem[b, o, h, d] = cell(b, h, o, d, "k"); mem[b, o, h, D + d] = cell(b, h, o, d, "v")
    return mem.permute(0, 2, 1, 3)


def five_d_caches():
    two_bn = torch.zeros(2, NB, N, H, D); b2n = torch.zeros(NB, 2, N, H, D)
    for b in range(NB):
        for o in range(N):
            for h in range(H):
                for d in range(D):
                    two_bn[0, b, o, h, d] = b2n[b, 0, o, h, d] = cell(b, h, o, d, "k")
                    two_bn[1, b, o, h, d] = b2n[b, 1, o, h, d] = cell(b, h, o, d, "v")
    return two_bn, b2n


def check(cache, want_layout):
    layout = C._detect_layout(tuple(cache.shape), N, H, D)
    assert layout == want_layout, (layout, want_layout)
    blocks = [4, 1, 5]                        # a request's block table, deliberately out of order
    positions = list(range(2, 11))            # crosses two paged-block boundaries
    slots = torch.tensor([blocks[p // N] * N + p % N for p in positions])
    k, v = C._gather_kv(cache, layout, slots // N, slots % N, D)
    tk, tv = truth(blocks, positions)
    assert tuple(k.shape) == (len(positions), H, D), k.shape
    assert torch.equal(k, tk), "K rows wrong"
    assert torch.equal(v, tv), "V rows wrong"
    print(f"ok  {want_layout:6s} shape {tuple(cache.shape)}")


def main():
    check(packed_cache(), "BHN2D")
    two_bn, b2n = five_d_caches()
    check(two_bn, "2BN")
    check(b2n, "B2N")
    try:
        C._detect_layout((NB, H, N, 2 * D + 1), N, H, D)
    except RuntimeError:
        print("ok  unknown layout rejected")
    else:
        raise AssertionError("an unknown layout must be rejected, not guessed")


if __name__ == "__main__":
    main()
