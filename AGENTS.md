# AGENTS.md — if you are an AI agent (or a human in a hurry), start here

This repo is a **reference implementation**, not a library. It proves one idea on one model and two
specific machines, and it is written so you can tear it apart and rebuild the idea on hardware you
actually have. Read this file, then `README.md`, then `docs/PORTING.md`. In that order.

## The one idea

Two inference engines that share no cache format (CUDA/vLLM on one box, Metal/oMLX on another) serve
one request together: the fast box prefills, the cheap-per-token box decodes. Instead of transferring
a KV cache neither side can read, **the prefill box computes the decoder's finished cache with the
decoder's own weights and writes it straight into the decoder's on-disk prefix cache.** The decoder
sees a normal cache hit and only decodes. Payload is ~10 KB/token, so any Ethernet works.

## Do this first, always

```bash
cp config.example.env config.env && $EDITOR config.env   # nothing has a working default
source config.env
make doctor          # names the first broken link in the chain; fix it; run again
```

`make doctor` failing with `set PD_SPARK` means you have not configured anything yet. That is the
correct first result.

## Rules that will save you days (each one cost us one)

1. **A benchmark row without a verdict is not a number.** Every reply from the front door carries an
   `X-PD-Bridge` header: `complete`, `partial B/T`, or declined-with-reason. `bench/bench_cold.py`
   records it. If you see a fast time and no verdict, the decoder served natively and you measured
   nothing. We lost a whole round this way — `docs/FINDING-bench4-cold-fallback.md`.
2. **Compare tensors, never file hashes.** Bridge-written blocks carry a timestamp; byte-identical
   files are impossible even when the contents are exact. `studio/verify_blocks.py` is the test.
3. **Warm prompts bypass the bridge by design.** A short question, or a document the decoder has seen,
   goes straight to the decoder. To exercise the bridge you need ≥ `PD_MIN_TAIL` (default 8192)
   *uncached* tokens. `bench/bench_cold.py` builds guaranteed-cold prompts from a seed.
4. **The front door is single-threaded.** One request at a time. `/health` goes silent while a bridge
   is in flight but the port stays open. Busy is not down.
5. **Run the hook-off control** (`PD_HOOK=off ./spark/pd-launch-v3.sh`) before you believe any prefill
   number. The capture hook's cost is only knowable against the engine alone.
6. **Never commit `dv4_proj_weights.safetensors` or any model derivative.** `make weights` regenerates
   it from your own copy of the model.

## Where to cut when porting

`docs/PORTING.md` names the four seams. In short: the hook point (where the prefill engine exposes the
tensor the decoder's cache is a pure function of), the pooling math (the decoder's, ported to the
prefill side), the block writer (the decoder's own on-disk format), and the cache-index patch (so the
decoder notices files written after model load). Everything else — the front door's control flow, the
verdict header, the cold/warm decision, the fallback — is reusable as-is.

**Feasibility test before you write code:** does your decode engine keep an on-disk, content-addressed
prefix cache? And is your model's per-layer cache state a pure function of tensors available at prefill
time? Two yeses and this repo is a skeleton for you. One no and it is a different, harder project.

## A bug we fixed the same day, kept here because it teaches the shape of this system

The front door signals the prefill hook the moment the engine returns (`FLUSH_NOW`). For most of
2026-09-06 the hook missed it on ~2 of 3 runs: the watcher thread ran in every process that imported
the hook (API server, engine core, worker), and the two that never own a capture unlinked the one-shot
file on sight. Two agents found it independently from the logs; hook v5 starts the watcher only in the
capturing worker and never consumes a signal it cannot act on. `docs/FINDING-flush-signal-three-watchers.md`
has the timeline; `spark/test_flush_decision.py` pins the behavior without a GPU. Lesson: a sitecustomize
hook runs everywhere Python starts — design every side effect as if three copies of you are watching.

## What we would love to see next (in order)

1. **Rung B or C from the README ladder.** One Spark + an iMac, or a used gaming GPU + an M-series
   Mac, with a small MLA-latent model (DeepSeek-V2-Lite is the obvious candidate). Same recipe, your
   numbers, either way. See README → *Don't have this hardware? Start on the rung you can reach*.
2. A second decode engine (vLLM's CPU/disk KV connector, or llama.cpp's prompt cache — the latter is
   harder, see PORTING).
3. Multi-stream: the front door is single-threaded today.

Open an issue with your verdict-tagged numbers. Negative results are results.
