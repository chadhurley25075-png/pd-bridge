# How we ran a ~290B model with NVIDIA prefill and Apple Silicon decode

*Technical detail behind the 3.07× number. Everything measured on two DGX Sparks and one Mac Studio
over ordinary 10GbE.*

---

## 1. The setup, and why it shouldn't work

Prefill and decode want opposite hardware. Prefill is a big batched matmul — compute-bound. Decode is
one token at a time against the whole cache — memory-bandwidth-bound. A DGX Spark (GB10) has roughly
4× the FP16 compute of an M3 Ultra; the M3 Ultra has roughly 3× the memory bandwidth. Split the
phases and each box does what it's good at. That argument is well established and EXO Labs published
a clean demonstration of it on this exact hardware pairing.

The way everyone does it is: **prefill builds a KV cache, you send the KV cache over the network,
decode continues from it.** vLLM has NIXL connectors for this. AMD shipped MORI-IO. It's a solved
problem — *when both ends agree on what a KV cache is.*

Ours never can:

| | prefill side | decode side |
|---|---|---|
| engine | vLLM 0.21.1rc1 + DeepSeek-V4 plugin | oMLX 0.6.4 |
| runtime | CUDA | Metal / MLX |
| weights | official `deepseek-ai/DeepSeek-V4-Flash`, **FP8** (e4m3, block 128) | `DV4-Flash-MXFP4-MLX`, **MXFP4** |
| cache | fp8 paged, CUDA kernel writes it directly | MLX block objects, SSD prefix store |

Two vendors, two frameworks, two quantizations, two cache layouts. There is no wire format to agree on.

And DeepSeek-V4-Flash makes it worse, because it doesn't have "a KV cache." Per layer it carries:

- a **rotating window** of the last 128 KV rows,
- a **compressor pool** at ratio 4 (with overlap carry between chunks) and at ratio 128,
- an **indexer pool** (64 heads × 128 dim, top-k 512),

each with layer-dependent RoPE — layers 0–1 use `rope_theta` 10000 with **no** yarn; layers ≥2 use
`compress_rope_theta` 160000 **with** yarn, and that applies to the SWA kv too. Translating that
between an fp8 CUDA paged cache and MLX block objects is not a serialization problem, it's a
reimplementation problem.

Model shape, for scale: 43 layers, hidden 4096, 256 routed experts + 1 shared, 6 active per token,
MLA with 1 KV head at head_dim 512. **~290B parameters** (283.5B of fp4-packed experts plus ~7.4B
attention/dense — summed from the checkpoint's safetensors headers). 149 GB resident on the prefill
side, 156 GB on the decode side.

## 2. The insight

Don't transfer a cache. **Compute the decoder's finished cache on the prefill machine, using the
decoder's own weights, and write it into the decoder's own prefix-cache store.**

This works because of one property: every one of those cache structures is a **pure function of the
per-layer attention input and the token position.** Not of the engine, not of the kernel, not of the
paged layout — of one tensor the prefill engine already computes and then throws away.

So the handoff becomes: intercept that tensor, apply the decoder's projection and pooling math to it,
ship the *results*. The decoder never learns that anything unusual happened — it looks up its prefix
cache, gets a hit, and decodes.

Two consequences fall out immediately:

- **The quantization mismatch stops mattering for the cached tensors.** We run the *Mac's* projection
  weights on the NVIDIA GPU, so the pooled output is the Mac's arithmetic, not the Spark's.
- **The payload collapses.** Finished pools are ~10 KB/token. Raw hidden states are ~360 KB/token.

That second point was learned the expensive way — see §5.

## 3. Implementation

### 3a. Prefill side — the capture hook

A `sitecustomize.py` mounted into the vLLM container, so it loads before anything else and survives
`torch.compile`. It patches exactly one method:

```
DeepseekV4MultiHeadLatentAttentionWrapper.attention_impl
```

taking `hidden_states` on the way in. From there, on the GPU, using projection weights exported from
the *MLX* model:

1. `project_layer` — wkv → kv_norm → the compressor/indexer linears
2. RoPE (correct theta and yarn per layer, `freq_scale = ratio`)
3. pooling: `softmax(gate + ape)`-weighted mean over each pool window, with overlap lanes for ratio 4
4. RMSNorm
5. slice the 128-row rotating window **out of the 2048-row chunk computation** (see §6)

Output: per-layer safetensors written incrementally, plus a manifest and a DONE marker, with
boundary snapshots every 2048 tokens.

Cost, measured at 81,024 tokens: **41.67 s** engine time with the hook on, and the pooling runs on a
side stream so the forward thread pays ~0.66 s total. Capture flush 2.62 s. Output **0.801 GB**.

Two engine flags are load-bearing:

- **`--no-enable-prefix-caching`.** With prefix caching on, vLLM skips a repeated document prefix, the
  hook observes `43 layers × 0 tokens`, and the decoder waits forever for rows that will never
  arrive. The decoder owns the caches; the prefill engine must compute every token it is asked to pool.
- **`--enforce-eager`.** Prefill-only engine — CUDA graphs buy nothing and cost ~13 min of startup.
  Our first async hook also invalidated vLLM's graph capture at boot
  (`cudaErrorStreamCaptureInvalidated`, then NCCL "previous error during capture"). If you hook a
  compiled engine, guard with `torch.cuda.is_current_stream_capturing()` and skip decode-shaped calls.

### 3b. Transport

Plain HTTP with Range support, pulled by the decoder, pipelined boundary-by-boundary while the
prefill is still running. **0.801 GB in 1.08 s** (~740 MB/s effective on 10GbE). We started with rsync
over ssh — 19 s for the same class of payload. HTTP was 7.1 s on that payload and is now ~1 s on the
pooled one.

### 3c. Decode side — assemble and inject

`pd_assemble_blocks.py` builds MLX cache objects from the capture through the cache classes' own
`.state` / `.meta_state` setters (RotatingKVCache, PoolingCache), snapshots at each 2048 boundary, and
hands them to `omlx_block_writer.write_blocks()` — which drives **oMLX's own store pipeline**, not a
reimplementation of it. That's the reason the output is byte-for-byte what oMLX writes: we're calling
its writer.

**39 blocks assembled and written in 4.59 s** at 81K.

One patch was needed on the decoder. oMLX indexes its SSD cache blocks into memory **at model load
only** — there's no rescan API — so blocks dropped in afterward are invisible. We added a
filesystem fallback in `PagedSSDCacheIndex`: on an index miss, stat the hash path, validate it with
the scan's own reader plus the compatibility check, and index it. After that the server log reads:

```
[pd] indexed externally written block dc8ef116bd7b7210… from disk
Preloaded 9/9 blocks into hot cache (time=158.5ms)
```

The front door (`pd_front.py`) is OpenAI-compatible and does the orchestration: render the prompt
through **oMLX's own template chain** (this matters — calling the model's chat template directly gave
a different token count; going through oMLX's chain produced 102,498 ids against oMLX's logged
`prompt: 102498`), chain-hash the ids to check what's already cached, bridge only if the uncached tail
is long enough, and **fall back to plain oMLX on any error** so a bridge failure can never cost a reply.

## 4. Numbers

Inputs are deterministic synthetic documents built from CPython's stdlib with a seeded shuffle — each
seed is a genuinely cold prompt for both engines. Native baseline on the same box, same prompt shape,
warm engine on both legs.

**81K tokens:**

| | prompt | time to answer | decode |
|---|---|---|---|
| Mac Studio alone, cold | 80,768 tok | **195.1 s** | 23.7 tok/s |
| Sparks → Mac | 81,024 tok | **63.6 s** | 23.5 tok/s |

**3.07×.** Decode rate identical, as designed — the bridge hands off before the first token.

Breakdown of the 63.6 s:

```
prefill engine (vLLM TP2, hook on)   41.67 s   ← 65% of wall time
capture flush                         2.62 s
pull 0.801 GB over 10GbE              1.08 s   ← 1.7%
assemble + write 39 blocks            4.59 s
decoder: tail prefill + 179 tokens   13.6  s
```

**19K tokens:** 44.7 s native → 25.5 s bridged (**1.76×**). Warm repeat: the bridge detects the cache
in 0.02 s, skips itself, decoder serves natively in 10.5 s.

**Prefill throughput:** 1,944 tok/s (2× GB10, TP2, FP8, hook on) vs **423 tok/s** (M3 Ultra, MXFP4) —
**4.6×**, consistent at both sizes.

**Payload:** 9.9–10.4 KB/token.

## 5. What we validated, and how

The design is only as good as the claim that reconstructed pools *are* the decoder's pools. Tested:

**Read the provenance column before the result column.** Three of these checks are *same-input*: both
sides start from one set of captured attention inputs, so they test our reconstruction and writing
math, not the end-to-end effect of prefilling in FP8 and decoding in MXFP4. That end-to-end difference
is real, it is flagged in Limits, and as of this writing it has **only behavioural evidence** — no
numeric one. We say which is which rather than let a strong number stand in for a claim it cannot make.

| check | provenance | result |
|---|---|---|
| rebuilt cache arrays vs. the caches the same MLX prefill produced | same-input — both sides replay one capture, on the Mac, in MLX | **313/313 bit-exact** |
| bridge-written blocks vs. oMLX's own blocks for the same prompt | same-input — blocks assembled from an MLX-computed capture | **11/11 identical** (only the `created_at` stamp differs) |
| torch pooling port vs. MLX ground truth, T=23,217 | same-input, cross-framework — MLX truth exported from that capture | projections, pre-RoPE window, carries **bit-exact**; pooled r=4 rel 2.6e-3 (99.96% of elements identical), r=128 rel 2.5e-4, indexer rel ≤7.4e-3 — worst case **one bf16 ulp** |
| in-container hook selftest, chunked vs. one-shot | same-input | **52/52** |
| marker retrieval through a fully reconstructed 81K cache | **end-to-end** — real FP8 prefill → real MXFP4 decode | correct on every run |
| judged answer quality, 5 questions, bridged vs. native leg | **end-to-end** | 5/5 both legs, two passes |
| Spark FP8 attention inputs vs. Mac MXFP4 attention inputs, identical tokens | **end-to-end, numeric** | **not yet measured — open** |

Compare tensors, not file hashes — `created_at` means a bridge-written block can never be
byte-identical as a *file*.

**The open row is the one worth running.** The bridge's prefill runs FP8 on CUDA; the decoder runs
MXFP4 on Metal. Those two forwards cannot produce bit-identical hidden states — which is exactly why
the bit-exact rows above had to be same-input to mean anything. The honest characterisation of the
bridge is the layer-by-layer divergence between those two forwards on the same tokens, and we own
every piece of machinery needed to measure it: capture on both sides, diff with `pd_diff_state.py`.
Until that number exists, treat the retrieval and judged-eval rows as the only evidence that the
quantisation gap does not matter in practice.

Credit for pushing on this: a reader asked whether the 313/313 started from the same captured hidden
states or from independent native and bridged forwards. It was the former, the distinction matters,
and the table above now says so.

### The payload mistake, since it's the most useful thing here

v1 shipped **raw hidden states**: 43 layers × [T, 4096] bf16. That was deliberate — it made
bit-exactness easy to prove and avoided translating vLLM's fp8 paged format at all. It worked, and it
served the first heterogeneous completion.

Then we measured what the decoder actually consumes. Our own written blocks for a 102,595-token
prompt totalled **1.01 GB**. We had shipped **36.14 GB** to produce them. **36× the payload the
decoder needed.**

The diagnosis that mattered: *transport was never the wall — the tensor choice was.* At 36 GB the
Spark's own NVMe write was 53.5 s of a 72 s span, i.e. write-bound, not compute-bound. Moving the
projection and pooling onto the prefill node deleted the disk stage and the pull in one change. The
`hidden` mode is still in the front door if you want to compare.

Generalizable version: **before optimizing a transfer, measure the size of what the consumer actually
ingests.** If your payload is a large multiple of that, you're shipping the wrong tensor.

## 6. Gotchas that cost real hours

- **Same math ≠ same bits across shapes, even on one device.** Computing the 128-row window on a
  128-row slice differs by one bf16 ulp from taking those rows out of the 2048-row chunk matmul —
  MXFP4 kernel tiling. Slice from the chunk computation.
- **MLX's bf16 `sum` is not f32 accumulation.** It's serial bf16 for 8 rows, and 32 strided bf16
  partials combined in f32 for 128 rows. Plain f32 accumulation matched only ~50% of elements. If
  you're porting MLX math to torch and getting "close but not exact," look here first.
- **Keep exactly one model build resident on the decoder.** Two DV4 builds in oMLX's pool exceeded the
  admission target; every request targeting the other one paid a ~33 s evict-and-reload. It reads
  exactly like a bridge regression and isn't.
- **Restart order.** Stop the old oMLX server and *wait for its shutdown line* before starting a new
  one, or the new server's first load hits the memory settle barrier and aborts.
- **`NCCL_IB_GID_INDEX` is fabric-specific.** Ours is 5; the common recipe says 3.
- **MLX streams are thread-local** — the front door has to be single-threaded.
- **Don't forward `Transfer-Encoding` on a passthrough proxy.**
- **Benchmark with a real token budget.** Our first run capped at 64 tokens, which truncated answers
  mid-reasoning and read as a retrieval failure — on *both* paths. It was a harness bug, not a result.
- **Warm up both legs before comparing.** Our first published-internally number was 1.19×, because the
  decoder's one-time model load sat inside the bridged leg while the native leg ran warm. Same runs,
  warm engine: 3.07×.

## 7. What is not proven

- **No judged quality score on the bridged leg yet.** Native scores 5/5 on our question set; the
  bridged leg hasn't been run against it. Prefill computes hidden states with FP8 weights and decode
  runs MXFP4, so bridged output is **not token-identical** to native — same meaning, different
  wording, on a greedy A/B. Marker retrieval passing every run is evidence, not proof.
- **Single stream** (`--max-num-seqs 1`). No concurrency data.
- **One model, pinned stacks.** The pooling math is specific to DV4-Flash's sparse attention. It
  monkey-patches private internals of both engines and will break when either moves.
- **Warm turns gain nothing.** By design.

## 8. Where the remaining time is

The prefill engine is now 65% of wall time, which is a good place for a bottleneck — it responds to
kernel and batching work rather than to cable you can't buy. Next levers, in order: stream the capture
over a socket instead of staging on NVMe (~1 s), tune vLLM prefill throughput, and ship the decoder's
carry state to the prefill engine so warm-but-extended prompts prefill only the tail.

---

**The transferable idea, one sentence:** when two inference engines can't share a cache format,
compute the *consumer's* finished cache on the *producer* using the consumer's weights, and write it
into the consumer's own cache store. Nothing about that is specific to these two boxes.

Code, benchmark protocol, and validation harnesses: Apache-2.0.
