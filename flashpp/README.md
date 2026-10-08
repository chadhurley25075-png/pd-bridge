# flashpp — one model, layers split across NVIDIA (MLX-CUDA) and Apple (MLX-Metal) (2026-10-05)

**The other direction of heterogeneity.** Everything else in this repo splits a request by *phase* (prefill on
NVIDIA, decode on Apple). flashpp splits the *model* by *layers*: the same MLX model code runs a contiguous slice
of decoder layers on a GB10 under **MLX's CUDA backend**, and the rest on a Mac Studio under **MLX-Metal**. Only
the hidden state crosses machines, over plain TCP.

```
token ids ─▶ [GB10, MLX-CUDA: embed + layers 0..k)] ─hidden─▶ [M3 Ultra, MLX-Metal: layers k..N + norm + lm_head] ─▶ logits
                                            32 KB/token (GLM-5.3-Flash) · 12 KB/token (GLM-5.3)
```

A stage owns layers `[start, end)`, exactly their caches, and only the safetensors shards that hold its tensors —
a machine never stores layers it does not run.

## What ran (all 2026-10-05, greedy, results in `results/`)

| run | stages | result |
|---|---|---|
| split logic, one box | GLM-5.3-Flash 8-layer stub: 1 stage vs 2 stages on one GB10 | **identical 12 tokens** |
| cross-family exactness | same stub: CUDA layers 0–3 → Metal 4–7 + head, vs Metal only | **identical 12 tokens** |
| **full GLM-5.3-Flash** (45 layers) | GB10 (CUDA) layers 0–5 → M3 Ultra 256 GB (Metal) layers 6–44 + head | coherent English · **19.8 tok/s decode** · prefill 29 tokens in 2.23 s · per token: CUDA stage 13.0 ms, Metal stage 28.8 ms, ~3 ms network+overhead |
| **full GLM-5.3** (78 layers, `glm_moe_dsa`, `mlx-community` indexerBF16-q4) | 3× GB10 (CUDA) layers 0–9 / 10–25 / 26–41 → M3 Ultra 256 GB (Metal) layers 42–77 + head | coherent ("The capital of France is Paris. Famous landmarks include the Eiffel Tower, the Louvre, Notre-Dame…") · **3.95 tok/s decode** · prefill 32 tokens in 3.84 s · per token 33.3 / 46.9 / 47.6 / 49.6 ms compute per stage |
| the same 4 stages, earlier that evening | before the last MLX-CUDA workaround | **garbled** output (kept: `glm53_full_4stage_earlier_garbled_…json`) |

These are proofs that the split is *correct*, not that it is *fast*. Per layer the GB10 under MLX-CUDA was ~3×
slower than the M3 Ultra under Metal for Flash (~2.2 ms vs ~0.74 ms per layer), because the CUDA path ran plain
fallbacks for the linear-attention and indexer ops. 3.95 tok/s for the full model across four machines is a
pipeline with one request and no overlap.

## The MLX-CUDA bugs we hit (mlx 0.32.0, CUDA 13, GB10) — and the workarounds in `patches/`

Measured against exact fp32 references. **None of these is reported upstream yet.**

1. **`gather_qmm` returns wrong rows once ≥ 8 (row, expert) pairs go in one call at K = 6144** (GLM-5.3's
   gate/up projection); ≤ 4 pairs per call is exact, and the down projection (K = 2048) is fine. One MoE block
   (layer 42) vs exact fp32: **Metal 1.3 %, CUDA 71 %**. Workaround: on CUDA, feed `gather_qmm` 4 pairs at a time
   (`omlx-glm_moe_dsa-switch_layers-gather_qmm-4-per-call.patch`) — the MoE module then matches a per-expert loop
   exactly. Repro: `repro/gq_batch.py <model_dir> <layer>` sweeps the batch size.
2. **RoPE / MultiLinear on strided split views.** After `mx.split` of q into `q_nope`/`q_pe`, CUDA was 52 % / 56 %
   off; RoPE on the `k_pe` split view was intermittently garbage (pe scores 6.6× off in about 1 run in 5). A
   contiguous copy fixes it: attention then landed at 0.35–0.48 % vs fp32 on 12 of 12 runs. Patches:
   `omlx-glm_moe_dsa-model-contiguous-rope.patch`, `omlx-glm_moe_dsa-deepseek_v32-contiguous-indexer-rope.patch`
   (indexer q/k), `mlx_lm-mla-contiguous-multilinear.patch`. `repro/bmm_test.py` is the bmm-on-a-strided-view probe.
3. **Batched `quantized_matmul` (H = 64, `transpose=True`) ~100 % wrong on CUDA** — found by a parallel session the
   same evening and noted; *no repro script is included here and we have not re-checked it.* It matters for any
   build with quantized attention.

Not bugs (checked): fused SDPA (≤ 1.6 %), broadcast bmm, row gather — all exact. And one trap in the *reference*:
**MLX's CPU bf16 `quantized_matmul` is the inaccurate one** (7 % vs exact fp32; CUDA 0.5 %; CPU with fp32 inputs
0.5 %) — never use the CPU backend as ground truth. `repro/qmm_truth.py`.

One Flash-specific loader fix: `sanitize()` in oMLX's vendored mlx-vlm `glm5_next` moved only the forget-gate
`.weight`s and left the quantized `.scales/.biases` behind (136 orphan parameters on a 4-bit build with quantized
forget-gate projections) — `omlx-glm5_next-sanitize-forget-gate.patch`.

The patches are against oMLX 0.6.4's Python package and its bundled `mlx_lm` (paths relative to `site-packages`):
`cd $SITE_PACKAGES && patch -p1 < .../flashpp/patches/<file>.patch`. Apply them on the CUDA stages only; they are
harmless but unnecessary on Metal.

## Running it

**Environment.** CUDA nodes: a venv with `mlx[cuda13]==0.32.0` (match the Mac's MLX version), plus oMLX 0.6.4's
Python package with its patched `mlx_lm`/`mlx_vlm` (we copied them file-for-file from the Mac's oMLX install; no
`.so`/`.metallib` needed), then the patches above. Mac: oMLX 0.6.4's own venv. The same model build on every
stage (check a few tensors byte for byte — ours matched on five fingerprinted tensors).

```bash
# each stage: model_dir start end bind_ip port [--dsa for full GLM-5.3] [--head to end a truncated test early]
python stage_server.py $MODEL 0  6  0.0.0.0 7301            # CUDA box: embed + layers 0-5
python stage_server.py $MODEL 6  45 0.0.0.0 7302            # Mac: layers 6-44 + norm + lm_head (end == n_layers)
python driver.py CUDA_HOST:7301,MAC_HOST:7302 60 --text "your question" --tok $MODEL --out res.json
```

`needed_shards()` in `stage.py` / `stage_dsa.py` lists the shard files a stage needs, so each machine can hold
only its slice. For full GLM-5.3 (`--dsa`), **cut only on a layer that runs its own indexer** — layers that
share the previous layer's top-k (IndexShare) cannot start a stage; `DsaStage` refuses and names the valid cuts.

## Traps

- **Metal: set the wired limit.** Without `mx.set_wired_limit(max_recommended_working_set_size)` macOS paged a
  208 GB stage every step: 2,200 ms → 36.5 ms per step. `stage_server.py` does it.
- **One model process per GB10.** A second weight-loading check process next to a 45 GB stage drove a 128 GB box
  into `NV_ERR_NO_MEMORY` and an OOM kill. Do reference checks on a different machine.
- A stage that receives part of a neighbour's layer from a shared shard must drop it *before* `sanitize()` stacks
  the MoE experts (`stage.py` filters on load).

## Status, honestly

- **Correctness proven on short prompts only** (12–60 greedy tokens, prompts of 20–32 tokens). No long-context,
  no quality eval, no needle test through the pipeline.
- **Slow by design of this version:** one request at a time, no pipelining, plain TCP, CUDA fallbacks. The
  speed levers we listed (and later partly pursued outside this folder) are an RDMA hop, compiled/native CUDA
  kernels for the linear-attention and indexer ops, MTP drafting on the last stage, and overlapping stages.
- The workarounds are workarounds. The real fixes belong in MLX's CUDA backend.

## Files

`stage.py` (GLM-5.3-Flash stage) · `stage_dsa.py` (full GLM-5.3 stage) · `stage_server.py` (one stage over TCP:
`reset` / `forward` / `info` / `bye`) · `net.py` (wire: 4-byte header length + JSON header + raw array bytes) ·
`driver.py` (prefill then greedy decode through an ordered list of stages) · `patches/` · `repro/` · `results/`.

## Credits

MLX and its CUDA backend (Apple ml-explore); oMLX (the patched model code these stages load, incl. its vendored
mlx-vlm `glm5_next`); `mlx-community` for the GLM-5.3 indexerBF16-q4 conversion. Related prior work: Ash Hart's
mixed Metal + CUDA pool support in oMLX.
