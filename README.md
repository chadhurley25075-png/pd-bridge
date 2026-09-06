# pd-bridge — heterogeneous prefill/decode for DeepSeek-V4-Flash

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![status: reference implementation](https://img.shields.io/badge/status-reference%20implementation-orange.svg)](#status-honestly)
[![model: DeepSeek-V4-Flash](https://img.shields.io/badge/model-DeepSeek--V4--Flash-8A2BE2.svg)](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
[![prefill: vLLM / CUDA](https://img.shields.io/badge/prefill-vLLM%20%2F%20CUDA-76b900.svg)](#running-it)
[![decode: oMLX / Metal](https://img.shields.io/badge/decode-oMLX%20%2F%20Metal-lightgrey.svg)](#running-it)

**Prefill a 284B model on NVIDIA. Decode it on Apple Silicon. Over plain 10GbE.**

![How the bridge works](docs/architecture.svg)

Two production inference engines that share no cache format, no framework, no vendor and no
quantization, serving one request together. DeepSeek-V4-Flash is 284B total / 13B active,
256 routed experts, MLA + sparse attention, 149 GB resident on the prefill side and 156 GB on
the decode side:

- **Prefill:** 2× NVIDIA DGX Spark (GB10), vLLM TP2, official `deepseek-ai/DeepSeek-V4-Flash` **FP8**
- **Decode:** 1× Mac Studio M3 Ultra, oMLX, `DV4-Flash-MXFP4-MLX` (**MXFP4**)
- **Link:** ordinary 10 gigabit Ethernet. No RDMA, no Thunderbolt.

```
cold prompt        Mac Studio alone    Sparks prefill -> Mac decode
   ~25K tokens          42.6 s               28.2 s     1.5x
   ~82K tokens         205.8 s               72.9 s     2.8x
  ~105K tokens         245.6 s               75.5 s     3.3x
  ~241K tokens         732.3 s              200.3 s     3.7x   (the prefill pair's 262K window — the ceiling)
decode rate unchanged (23-25 tok/s both ways); warm turns bypass the bridge (4.9 s / 19.3 s at 241K)
```

Every row verdict-checked, 2026-09-06. The 17K/79K/236K rows are hook v5, after the flush-signal fix landed
the same afternoon (`docs/FINDING-flush-signal-three-watchers.md`); the 105K row predates it.
The bridged leg scores **5/5 on the judged quality eval**, same as native. **Validated envelope is
~20K-105K tokens** — see *Known limits* below, which is where you should look before believing
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

**The ~82K-token ceiling was a real bug. It is fixed.** (History kept because the failure mode is
instructive and the arithmetic still matters on smaller machines.)

`omlx_block_writer` used to hold one *materialised cumulative* cache snapshot per 2048-token
boundary until `finalize()`: peak memory grew **quadratically** with prompt length (~`N(N+1)/2`
blocks' worth of arrays — ~16 GB at 39 boundaries, ~23 GB at 47, which exhausted a 256 GB M3 Ultra
holding a 156 GB model and wrote zero blocks at 97,848 tokens).

The writer now **streams**: each boundary is stored through oMLX's own pipeline and released the
moment it is snapshotted (`begin_stream`/`store_boundary`; `finalize` drains and verifies). Peak
memory is ONE boundary snapshot (~20 MB × boundary index / N — tens of MB, not tens of GB). It is
validated against a real oMLX reference block (synthetic layout match) and live at 52 boundaries /
109,085 tokens — see RESULTS.md. `PD_STREAM_BOUNDARIES=0` restores the batched path for comparison.

`PD_MAX_BRIDGE_TOKENS` remains as a configurable envelope guard, not a bug workaround: raise it to
your machine's measured headroom.

Two related behaviours worth knowing:

- **The fallback works, and it can no longer lie.** When a bridge fails the reply still comes back
  correct — the decoder serves natively. Since the bench4 autopsy (docs/FINDING-bench4-cold-fallback.md),
  every response carries an `X-PD-Bridge` verdict (`complete` / `partial B/T` / declined with reason),
  and `bench_cold.py` records it — a silent native fallback can never again enter a results table
  as a bridged number.
- **The "cold-start variance" at 20K was not variance.** The 25.5 s vs 55.4 s spread was the capture
  hook flushing *mid-request* during chunked prefill (see the FINDING): the 55 s runs were native
  fallbacks wearing a bridge label. The hook now guards its idle flush with a CUDA-event query and a
  chunk-alignment check, and the front validates every capture manifest before trusting it.

## Status, honestly

This is a **reference implementation, not a library.** It is pinned hard and it is young.

- **One model.** DeepSeek-V4-Flash. The pooling math is specific to its sparse attention.
- **Pinned stacks.** oMLX 0.6.4; vLLM 0.21.1rc1 with the DeepSeek-V4 plugin (sparkrun image).
- **It monkey-patches private internals of both engines** — a `sitecustomize` hook onto
  `DeepseekV4MultiHeadLatentAttentionWrapper.attention_impl` on the vLLM side, and a
  filesystem-fallback patch to oMLX's `PagedSSDCacheIndex` on the MLX side (oMLX indexes SSD blocks
  at model load only, so externally written blocks are otherwise invisible). **Expect this to break
  when either project moves.**
- **The judged quality eval is five questions on one document.** The bridged leg scores 5/5 on it,
  twice (once from a fresh cold v3 bridge), same as native. Prefill runs FP8 weights and decode runs
  MXFP4, so bridged output is *not* token-identical to native; it is factually faithful on what we
  checked, which is a smaller claim than "equivalent".
- **The flush signal was unreliable until 2026-09-06 17:52 — fixed.** The hook's watcher ran in three
  processes and two of them deleted the signal before the capturing worker saw it (~1 in 3 hit rate). Hook v5
  fixes it; captures now close 0.6–0.9 s after the engine returns (4.8 s at 236K, which is the block write).
  Autopsy: `docs/FINDING-flush-signal-three-watchers.md`. Unit test: `spark/test_flush_decision.py`.
- **The front door is single-threaded.** One request at a time; a second caller queues behind a bridge
  in flight and `/health` goes silent while the port stays open. Busy is not down.
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
  pd_share.py                   the threaded share the pooled path actually runs (SimpleHTTP drops connections under poll+fetch)
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

bench/    bench_cold.py (records the X-PD-Bridge verdict), hetero (the one-command demo client),
          BENCHMARK-PROTOCOL.md
docs/     DESIGN-v3-pooled.md — the pooling math and the hook points, derived from oMLX's own code
          FINDING-bench4-cold-fallback.md — the mid-request-flush autopsy; what broke and what it taught
```

## Don't have this hardware? Start on the rung you can reach

Everything in this repo is written against the exact machines we ran, on purpose: if you have the same
gear you get an exact replica and the numbers in RESULTS.md. If you don't, the idea is the same and the
recipe scales down. Three rungs, honestly labeled:

| rung | prefill side | decode side | model | status |
|---|---|---|---|---|
| **A · exact replica** | 2× DGX Spark, TP2 over their direct 200G cable | Mac Studio M3 Ultra 256 GB | DeepSeek-V4-Flash (284B / 13B active) | **measured** — everything in RESULTS.md |
| **B · one Spark + any Apple Silicon Mac** | 1× DGX Spark (128 GB) | Mac Studio / Mac mini / **iMac** with ≥32 GB unified memory | a model that fits *both* boxes: the FP8 V4-Flash does **not** fit one Spark, so pick an MLA-latent model that does — DeepSeek-V2-Lite (16B) is the obvious first | recipe only, **unmeasured** |
| **C · the kid's stack** | one used CUDA gaming card (8–24 GB) in a beat PC, vLLM or sglang | an M-series iMac / MacBook with 16 GB | the smallest MLA-latent model that fits both | recipe only, **unmeasured** |

What is identical across all three rungs: the front door, the verdict header, the cold/warm decision, the
block writer path, the benchmark protocol, the wire (ordinary Ethernet — ~10 KB/token means even 1 GbE
moves a 30K-token prompt in ~0.3 s). What changes when you move down: the model, and therefore the pooling
math in the capture hook (DeepSeek-V4-Flash's hook is specific to its sparse attention; a plain-MLA model
like V2-Lite is *simpler* — its per-layer cache is the K/V rows themselves). `docs/PORTING.md` names the
four seams you touch and has a two-question feasibility test that takes ten minutes.

What to expect at the bottom rung, honestly: the win is the ratio of prefill speeds. A used 3090 prefills a
16B MLA model far faster than a 16 GB Mac does, so the shape of the result should hold; the absolute
numbers will be smaller because the prompts and models are smaller. **We have not run rungs B or C
ourselves.** They are the first ports we want to see, a negative result is a result, and we will feature
whoever lands one. Open an issue.

The point of this repo is that the privilege travels down. Take it apart.

## Running it

### Prerequisites

| side | hardware | software | model |
|---|---|---|---|
| prefill | 2× NVIDIA DGX Spark (GB10, 128 GB each) on a 200G RoCE link (TP2) | Docker + the sparkrun vLLM image with the DeepSeek-V4 plugin (`aidendle94/sparkrun-vllm-ds4-gb10:production-ready`, vLLM 0.21.1rc1) | `deepseek-ai/DeepSeek-V4-Flash` (official FP8 checkpoint, ~149 GB) |
| decode | 1× Mac Studio M3 Ultra, 256 GB | oMLX 0.6.4 in a venv + the one-file patch in `studio/` | an MLX MXFP4-experts / MXFP8-attention conversion of `deepseek-ai/DeepSeek-V4-Flash-0731` (~156 GB; any bit-exact conversion works — ours keeps the DSpark MTP heads) |
| link | any Ethernet ≥10 GbE between the two | SSH key from the Mac to the prefill head; Python 3.10+ on both | — |

The official FP8 checkpoint is ~149 GB, so it does **not** fit one 128 GB Spark: prefill is tensor-parallel
across two Sparks over their direct ConnectX-7 link (the standard two-Spark cable — box to box, no switch
involved). The only traffic that crosses to the Mac is HTTP over ordinary Ethernet, through whatever switch
you have. The numbers in RESULTS.md are TP2 over a 10 GbE LAN.

**The Mac-side model.** We run a local, bit-exact MLX conversion of `deepseek-ai/DeepSeek-V4-Flash-0731`
(MXFP4 experts, MXFP8 attention, DSpark MTP heads kept). There is no single published id to point at, so
produce your own; the closest one-liner is

```bash
mlx_lm.convert --hf-path deepseek-ai/DeepSeek-V4-Flash-0731 --mlx-path ~/models/DV4-Flash-MXFP4-MLX \
  -q --q-mode mxfp4 --q-bits 4 --q-group-size 32
```

(unverified by us end to end — our build was a mixed conversion). What matters for the bridge is
**self-consistency**, not which conversion: `make weights` exports the attention-projection weights from
*your* MLX model, and the prefill hook uses exactly those, so the pooled tensors match whatever the decoder
actually runs.

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
python3 spark/pd_share.py "$PD_CAPTURE_DIR" 8010        # pooled mode (the one the numbers use)
# or: python3 spark/pd_capture_http.py --root "$PD_CAPTURE_DIR" --port 8010   # Range-capable; needed by the older 'hidden' pipelined mode
```

**3. Patch and start oMLX**, then the front door (on the Mac):

```bash
# oMLX must notice blocks written after model load — one small patch, applied once, in the oMLX venv:
OMLX_PKG=$($OMLX_PYTHON -c 'import omlx,os;print(os.path.dirname(omlx.__file__))')   # .../site-packages/omlx
patch -p0 -d "$OMLX_PKG/cache" < "$OLDPWD/studio/omlx-0.6.4-paged_ssd_cache-disk-index-fallback.patch"
# then (re)start oMLX serving $PD_MODEL on :8011, and start the front door:
$OMLX_PYTHON studio/pd_front.py          # listens on $PD_PORT (8012), OpenAI-compatible
curl -s localhost:8012/health             # {"ok": true, "front": "pd", ...}
```

Point any OpenAI-compatible client at `:8012`. Prompts under `PD_MIN_TOKENS` or with fewer than
`PD_MIN_TAIL` uncached tokens go straight to oMLX; longer cold prompts are prefilled on the Sparks. The
`X-PD-Bridge` response header says which happened. `make doctor` checks every link in the chain.

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
- **An idle-flush capture hook and chunked prefill are a dangerous pair.** With
  `--max-num-batched-tokens 8192` the engine grinds ~4 s per chunk; a 2 s idle watcher fires
  *between* chunks (or inside a chunk's enqueue burst) and ships a DONE manifest for a request that
  is still running. Guard the flush with a CUDA-event query (GPU busy = mid-request) and a
  chunk-alignment check (`calls % n_layers == 0`), and make the consumer validate `manifest T ==
  request T` before trusting any DONE. This cost us a whole benchmark round: docs/FINDING-bench4-cold-fallback.md.
- **Benchmark with a real token budget.** A 64-token cap truncated answers mid-reasoning and read as
  a retrieval failure on *both* paths.

## Contributing

The most useful things anyone could add, roughly in order:

1. ~~Kill the quadratic snapshot memory~~ — **done 2026-09-06** (streaming boundary store; see
   *Known limits* and RESULTS.md). The bridge is validated to 109K tokens on a 256 GB machine.
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
