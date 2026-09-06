# pd-bridge — heterogeneous prefill/decode for DeepSeek-V4-Flash

**Prefill a 284B model on NVIDIA. Decode it on Apple Silicon. Over plain 10GbE.**

Two production inference engines that share no cache format, no framework, no vendor and no
quantization, serving one request together. DeepSeek-V4-Flash is 284B total / 13B active,
256 routed experts, MLA + sparse attention, 149 GB resident on the prefill side and 156 GB on
the decode side:

- **Prefill:** 2× NVIDIA DGX Spark (GB10), vLLM TP2, official `deepseek-ai/DeepSeek-V4-Flash` **FP8**
- **Decode:** 1× Mac Studio M3 Ultra, oMLX, `DV4-Flash-MXFP4-MLX` (**MXFP4**)
- **Link:** ordinary 10 gigabit Ethernet. No RDMA, no Thunderbolt.

```
81,024-token cold prompt          time to answer     decode
  Mac Studio alone                    195.1 s        23.7 tok/s
  Sparks prefill -> Mac decode         63.6 s        23.5 tok/s      3.07x
```

Reproduced on a second cold seed: 78,504 tokens, 65.6 s. **Validated envelope is roughly
20K-82K tokens** — see *Known limits* below, which is where you should look before believing
anything above.

Full numbers and methodology: [RESULTS.md](RESULTS.md) · [bench/BENCHMARK-PROTOCOL.md](bench/BENCHMARK-PROTOCOL.md)

---

## The idea

Prefill/decode disaggregation is well established, and so is the hardware argument for it: prefill is
compute-bound, decode is memory-bandwidth-bound, so run each phase where it is cheapest. Existing
systems do this by **transferring the KV cache** from the prefill worker to the decode worker.

That requires both ends to agree on a cache format. Ours never can. One side is CUDA/vLLM with an
FP8 paged cache; the other is Metal/MLX with its own block layout. Worse, DeepSeek-V4-Flash does not
have "a KV cache" — each layer carries a rotating 128-token window, a compressor pool (ratio 4 with
overlap carry, and ratio 128), and an indexer pool, all with layer-dependent RoPE.

**So we don't transfer a cache. We compute the decoder's finished cache on the prefill machine,
using the decoder's own weights, and write it straight into the decoder's prefix-cache store.**

The prefill engine already computes the exact tensor those pools are a pure function of — the
attention input. A hook takes it there, applies the *Mac's* projection and pooling math on the GPU,
and emits the finished pools. The decode side assembles them into MLX cache objects and hands them
to oMLX's own block writer. oMLX then sees a normal prefix-cache hit and only decodes. Neither
engine is modified in its hot path; the decoder does not know a bridge exists.

Payload: **~10 KB per token** — 0.80 GB for an 81K-token prompt, pulled in 1.08 s. The network
stopped being the bottleneck; the prefill engine is now 65% of wall time, which is where you want it.

### Why the reconstruction is trustworthy

The whole design rests on the pooled tensors being *the same tensors* the decoder would have
computed. That is tested, not assumed:

| check | result |
|---|---|
| Cache arrays rebuilt on the Mac vs. a full native forward | **313/313 bit-exact** |
| Blocks written by the bridge vs. blocks oMLX writes itself | **11/11 identical** (only the `created_at` stamp differs) |
| Torch pooling port vs. MLX ground truth (T=23,217) | projections, window, carries **bit-exact**; pooled tensors 99.95–99.96% identical, worst delta **one bf16 ulp** |
| In-container hook selftest (chunked == one-shot) | **52/52** |
| Needle retrieval through a fully reconstructed 81K cache | correct on every benchmark run |

`studio/verify_blocks.py`, `studio/pd_diff_state.py`, `spark/pd_pool_validate.py` and
`spark/pd_pool_selftest.py` reproduce these. Compare tensors, never file hashes — `created_at` means
a bridge-written block can never be byte-identical as a *file*.

---

## Known limits — read this before you run it

**There is a hard ceiling at ~82K tokens, and it is a real bug, not caution.**

`omlx_block_writer` holds one *materialised cumulative* cache snapshot per 2048-token boundary
until `finalize()`. Peak memory therefore grows **quadratically** with prompt length —
holding N boundaries costs roughly `N(N+1)/2` blocks' worth of arrays:

| boundaries | prompt | snapshots held | result on a 256 GB M3 Ultra (156 GB model resident) |
|---|---|---|---|
| 39 | 81,024 tok | ~16 GB | works, 63.6 s |
| 47 | 97,848 tok | ~23 GB | **exhausts headroom, writes 0 blocks** |

`PD_MAX_BRIDGE_TOKENS` (default 81920) declines the bridge above the validated envelope and lets
the decoder serve natively — a clean decision instead of a failure. Raise it only after measuring
headroom on your own machine; a 512 GB box will go further.

**Fixing this properly is contribution #1** (see *Contributing*): stream each boundary to disk and
release it, instead of batching every snapshot into one `store_cache` call.

Two related behaviours worth knowing:

- **The fallback works.** When the writer failed at 97,848 tokens the reply still came back correct
  — `pd_front` caught it and served natively. You lose the speedup, never the answer.
- **Cold-start variance is real.** The decoder's own first-request model load lands inside whichever
  leg runs first and can double a measured time. Warm both legs before comparing anything, and see
  the *Superseded* note in RESULTS.md for what that mistake looked like.

## Status, honestly

This is a **reference implementation, not a library.** It is pinned hard and it is young.

- **One model.** DeepSeek-V4-Flash. The pooling math is specific to its sparse attention.
- **Pinned stacks.** oMLX 0.6.4; vLLM 0.21.1rc1 with the DeepSeek-V4 plugin (sparkrun image).
- **It monkey-patches private internals of both engines** — a `sitecustomize` hook onto
  `DeepseekV4MultiHeadLatentAttentionWrapper.attention_impl` on the vLLM side, and a
  filesystem-fallback patch to oMLX's `PagedSSDCacheIndex` on the MLX side (oMLX indexes SSD blocks
  at model load only, so externally written blocks are otherwise invisible). **Expect this to break
  when either project moves.**
- **The judged quality eval is not finished.** Needle retrieval passes on every run, and the native
  leg scores 5/5 on the question set, but the bridged leg has not been scored against it. Prefill
  runs FP8 weights and decode runs MXFP4, so bridged output is *not* token-identical to native. Until
  that eval lands, treat quality as "looks right, not yet proven".
- Only cold, long prompts benefit. Warm turns bypass the bridge by design and are served natively.

**The transferable idea is bigger than this code:** when two engines cannot share a cache format,
compute the *consumer's* finished cache on the *producer*, using the consumer's weights. That
generalizes past this model and this hardware, and it is the part worth stealing.

---

## Layout

```
spark/    prefill side (NVIDIA / vLLM)
  capture_sitecustomize_v3.py   the hook: projections + pooling on the GPU, per-layer safetensors
  pd_pool_torch.py              torch port of the decoder's pooling math (RoPE, compress, rmsnorm)
  pd_pool_selftest.py           in-container selftest (chunked == one-shot)
  pd_pool_validate.py           validate the port against MLX ground truth
  pd-launch-v3.sh               launch vLLM with the hook (PD_HOOK=off for a control run)
  pd_capture_http.py            Range-capable server so the decoder can stream captures
  pd-hf-layout.sh               lay the checkpoint out as an HF hub dir inside the container mount
  POOL-VALIDATION.md            what the validation numbers mean

studio/   decode side (Apple Silicon / oMLX)
  pd_front.py                   OpenAI-compatible front door; orchestrates a request end to end
  omlx_block_writer.py          drive oMLX's own store pipeline to emit prefix-cache blocks
  pd_assemble_blocks.py         build MLX cache objects from a pooled capture, snapshot per boundary
  pd_export_proj_weights.py     export the MLX projection weights the prefill hook needs
  pd_export_pool_truth.py       MLX-computed ground truth for validating the torch port
  pd_make_v3_from_mlx.py        build a v3 capture entirely in MLX (acceptance harness)
  pd_capture_mlx.py             capture attention inputs natively (test fixture)
  pd_rebuild_mlx.py             attention-only replay (the v1 path, kept for comparison)
  verify_blocks.py              directory-vs-directory block comparison
  pd_diff_state.py              cache-array diff against a full forward
  test_block_writer_synthetic.py

bench/    bench_cold.py, BENCHMARK-PROTOCOL.md
docs/     DESIGN-v3-pooled.md — the pooling math and the hook points, derived from oMLX's own code
```

## Running it

```bash
cp config.example.env config.env && $EDITOR config.env   # nothing has a working default
source config.env
```

**1. Export the decoder's projection weights** (on the Mac, in the oMLX venv). These are what the
prefill hook uses, so that the pooled tensors match the decoder's arithmetic rather than the
prefill engine's:

```bash
$OMLX_PYTHON studio/pd_export_proj_weights.py --model "$PD_MODEL" --out "$PD_V3"
```

Copy `$PD_V3` (`pd_pool_torch.py`, `dv4_proj_weights.*`, `capture_sitecustomize_v3.py`) to **both**
prefill nodes.

**2. Start the prefill pair** (rank 0 = TP head, rank 1 = worker):

```bash
./spark/pd-launch-v3.sh 1     # worker first
./spark/pd-launch-v3.sh 0     # then head
python3 spark/pd_capture_http.py --root "$PD_CAPTURE_DIR" --port 8010
```

**3. Patch and start oMLX**, then the front door (on the Mac):

```bash
# oMLX must index externally written blocks; see docs/ for the PagedSSDCacheIndex fallback.
$OMLX_PYTHON studio/pd_front.py
```

**4. Benchmark:**

```bash
python3 bench/bench_cold.py --chars 330000 --seed 301 --url http://<decoder>:8012   # bridged
python3 bench/bench_cold.py --chars 330000 --seed 302 --url http://<decoder>:8011   # native
```

## Gotchas that cost us hours

- **Prefix caching must be OFF on the prefill engine.** With it on, vLLM skips a repeated document
  prefix, the hook sees `43 layers x 0 tokens`, and the decoder waits forever for rows that will
  never arrive. The decoder owns the caches; the prefill engine must compute every token it pools.
- **`--enforce-eager`.** Prefill-only engine: CUDA graphs buy nothing and cost ~13 min of boot. The
  first async version of the hook also invalidated vLLM's graph capture at startup
  (`cudaErrorStreamCaptureInvalidated`); guard any hook with `torch.cuda.is_current_stream_capturing()`.
- **Keep exactly one model build resident on the decoder.** Two builds in oMLX's pool exceeded the
  admission target and cost a ~33 s evict-and-reload on every request that targeted the other one.
  It looks exactly like a bridge regression and is not.
- **Restart order matters.** Stop the old oMLX server and *wait for its shutdown line* before
  starting a new one, or the new server's first load hits a memory settle barrier and aborts.
- **`NCCL_IB_GID_INDEX` is fabric-specific.** Ours is 5; the common recipe says 3. Check `show_gids`.
- **Same math is not the same bits.** Computing the 128-row window on a 128-row slice differs by one
  bf16 ulp from taking those rows out of the 2048-row chunk matmul — MXFP4 kernel tiling. Slice from
  the chunk computation. Relatedly, MLX's bf16 `sum` is serial for 8 rows and 32 strided bf16
  partials combined in f32 for 128 rows; plain f32 accumulation matched only ~50% of elements.
- **Benchmark with a real token budget.** A 64-token cap truncated answers mid-reasoning and read as
  a retrieval failure on *both* paths.

## Contributing

The most useful things anyone could add, roughly in order:

1. **Kill the quadratic snapshot memory** (see *Known limits*). Boundaries should stream to disk and
   be released as they go, rather than all being held for one `store_cache` call. This is what
   currently caps the bridge at ~82K tokens on a 256 GB machine, and it is the single most valuable
   thing anyone could fix.
2. **A second model.** The bridge shape should generalize to any MLA/sparse-attention model whose
   caches are a pure function of the attention input. Porting the pooling math is the work.
3. **Stream the capture over a socket** instead of staging it on the prefill node's NVMe.
4. **Ship the decoder's carry state to the prefill engine** so warm-but-extended prompts can prefill
   only the tail instead of the whole thing.
5. **Finish the judged quality eval** on the bridged leg (`bench/BENCHMARK-PROTOCOL.md`).
6. **Make the engine patches survive upstream.** Both would be better as small upstream hooks than
   as monkey patches — a cache-rescan API on the oMLX side especially.

Benchmark numbers in a PR must follow `bench/BENCHMARK-PROTOCOL.md`, including a native baseline on
the same hardware and a warm engine on both legs.

## Credits

oMLX for the decoder and its cache format; the vLLM DeepSeek-V4 plugin and the sparkrun GB10 image;
EXO Labs, whose DGX Spark + Mac Studio prefill/decode result set the reference point this builds on;
and the Spark↔Mac USB4/RDMA work that made joining the two silicon families look worth trying.

Apache-2.0.
