# Results — 2026-09-06

All figures measured on the hardware described in [bench/BENCHMARK-PROTOCOL.md](bench/BENCHMARK-PROTOCOL.md).
Nothing here is derived or extrapolated. Every bridged run has a native cold run on the same box, same
prompt shape, same engine state.

Inputs are `bench/bench_cold.py`: a deterministic synthetic document built from CPython's stdlib
sources with a seeded shuffle, an embedded marker, and one question. **Each seed is a genuinely cold
prompt** — no cache clearing, no force switches, nothing either engine has seen.

## After the flush fix — the current numbers (2026-09-06 17:52–17:57, hook v5)

The flush-lag bug described below was root-caused and fixed the same afternoon (two agents, independent, same
finding: `docs/FINDING-flush-signal-three-watchers.md`). The capture hook's watcher thread ran in three processes
inside the vLLM container; the two that never own a capture deleted the front door's one-shot `FLUSH_NOW` signal on
sight, so the capturing process saw it on about one run in three and fell back to its 15 s idle timer otherwise.
Hook v5 starts the watcher only in the capturing worker, never consumes a signal it cannot act on, and tags the
manifest with the real reason (`flush_now` vs `idle`). One seed per size, fresh, verdict recorded:

| cold prompt | native (earlier today) | **bridged, hook v5** | signal → capture closed | gain |
|---|---|---|---|---|
| 17,095 tok | 42.6 s | **17.2 s** | 0.6 s | 2.5× |
| 79,314 tok | 205.8 s | **56.4 s** | 0.9 s | 3.65× |
| 236,377 tok | 732.3 s | **187.6 s** | 4.8 s (4.5 s of it is the 2.3 GB write — the floor) | 3.9× |

Every capture closed on `flush_now`, from the capturing worker only, 0.03–0.08 s after the signal landed.

## The ceiling — 241K tokens, the largest prompt the prefill pair accepts (2026-09-06 16:10–16:26, hook v4 — before the flush fix)

vLLM on the Spark pair is built with a 262,144-token window, so ~241K tokens of document plus the question is the
biggest cold prompt this stack can bridge. One seed each way, decoder otherwise idle, verdict recorded.

| configuration | prompt | time to answer | marker |
|---|---|---|---|
| native cold (decoder alone) | 241,155 tok | **732.3 s** | found |
| **bridged cold** | 241,416 tok | **200.3 s** | found |
| bridged, warm rerun (bridge self-skips, 239,616 cached) | 241,416 tok | 19.3 s | found |

**3.66× faster to an answer at the ceiling.** Where the 200.3 s went: prefill engine 137.6 s (~1,755 tok/s at this length),
capture closed +17.4 s after the engine returned (the flush-lag bug), 2.38 GB pulled in 2.7 s, 117 blocks assembled and written by
+176.7 s, decoder first token +23 s. The streaming block writer held 117 boundaries; the decoder went from 4.5 GB free / 106 GB
inactive to 0.1 GB free / 70 GB inactive during the write and recovered. No memory guard, no swap.

## The matrix — current code, one sitting (bench6, 2026-09-06 13:31–13:43)

Hook v4 (explicit flush signal + 15 s idle backstop), validating front door, streaming block writer.
Every row below carries the front door's own `X-PD-Bridge` verdict; a native fallback cannot appear here
as a bridged number. The decoder was otherwise idle (the prefix-cache warmer that shares it was paused).

| prompt | native cold (decoder alone) | **bridged cold** | gain | verdict | marker |
|---|---|---|---|---|---|
| ~25K tok (24,924) | 42.6 s | **28.2 s** | 1.51× | complete, 12 blocks | found / found |
| ~82K tok (82,505) | 205.8 s | **72.9 s** | 2.82× | complete, 40 blocks | found / found |
| ~105K tok (105,401) | 245.6 s | **75.5 s** | 3.25× | complete, 51 blocks | found / found |
| ~25K warm (bridge self-skips) | — | 4.9 s | — | skipped: 24,576 cached | found |

Decode rate was unchanged on every pair (23.1–25.5 tok/s bridged vs 23.7–25.4 native). Native and
bridged legs use different seeds, so token counts differ by ≤3%.

### Where the bridged time goes (bench6)

| stage | 25K | 82K | 105K |
|---|---|---|---|
| prefill engine (vLLM TP2, hook on) | 17.1 s | 42.2 s | 53.7 s |
| capture flush landed (DONE seen) | +0.6 s | **+15.5 s** | +1.4 s |
| pull over 10GbE | 0.25 GB, 0.5 s | 0.82 GB, 1.0 s | 1.04 GB, 1.2 s |
| assemble + write blocks | 3.8 s | 7.5 s | 8.6 s |
| **bridge total** | **22.0 s** | **66.3 s** | **65.1 s** |
| decoder: tail prefill + first token | 6.1 s | 6.5 s | 10.4 s |

**The 82K row paid 15 s it did not need.** The front signals the hook the moment the prefill engine
returns; on the 25K and 105K rows the capture closed within ~1 s of that signal, on the 82K row the hook
missed the signal and closed on its 15 s idle backstop instead. The same lag showed on a 14.8K eval
prompt (+13.2 s). It is the largest remaining inefficiency in the pipeline and it is a hook-side bug, not
a physics limit — see *Known limits* in the README. With the signal landing, 82K bridged is ~58 s (3.5×).

### Prior samples, same sizes (for n)

| prompt | bridged cold | note |
|---|---|---|
| 81,024 tok | 63.6 s | hook v3.1, 2026-09-06 morning — flush landed immediately |
| 78,504 tok | 65.6 s | hook v3.1, second cold seed |
| 109,085 tok | 83.5 s | first 100K+ bridge; tail salvaged (52/53 boundaries) |
| 18,553 tok | 25.5 s | hook v3.1 |
| 18,694 tok | 19.7 s | 2026-09-06 13:04, fixed front, decoder idle; native at this size 40–46 s |

So the ~80K figure now has three cold samples (63.6 / 65.6 / 72.9 s) against natives of 195–206 s, and
the 100K figure two (83.5 salvaged / 75.5 complete) against 246–257 s.

## Judged quality — the bridged leg scores 5/5

Same 14.7K-token source document, same five checkable questions (`bench/eval_questions.json`), same
lenient exact-token scoring as the native leg (which scored 5/5).

| pass | how the cache was made | score |
|---|---|---|
| native (decoder alone) | decoder's own prefill | **5/5** |
| bridged, pass 1 | blocks written by the bridge earlier that day (hidden-state mode) | **5/5** |
| bridged, pass 2 | fresh cold prompt → v3 pooled bridge, verdict `complete`, 7 blocks; Q2–Q5 served from those blocks | **5/5** |

Bridged answers are not token-identical to native (FP8 prefill weights vs MXFP4 decode weights) but
every checked fact matched. This closes the "looks right, not yet proven" caveat from the first release.

## Prefill throughput

| | tok/s |
|---|---|
| decoder alone (Apple Silicon, MXFP4) | 423 |
| prefill pair (2× GB10, TP2, FP8, hook on) | 1,944–2,077 |
| prefill pair, **hook off** (control run, 9/6) | 2,017–2,077 (median ~2,050) |

**~4.6–4.9×** over the decoder. The control run — the pair restarted with `PD_HOOK=off` and the same
neutral prompts — puts the honest prefill floor at **~2,050 tok/s**: the capture hook now costs
single-digit percent (39.35 s hook-off vs 39.63–40.77 s hook-on at ~78–81K). The earlier v3.0 hook
cost ~33 s per request; v3.1's launch-lean data path removed it. An earlier "~8 s pure prefill at
81K" estimate was wrong by 5× and never measured — the control run is the measurement.

## Payload

| prompt | capture shipped | KB/token | pull time |
|---|---|---|---|
| 18,553 tok | 0.193 GB | 10.4 | 0.48 s |
| 81,024 tok | 0.801 GB | 9.9 | 1.08 s |

An earlier design shipped raw hidden states — 36.14 GB for a 102K prompt, to produce 1.01 GB of
cache. **36× the payload the decoder actually consumes.** Moving the projection and pooling math onto
the prefill node removed it. The `hidden` mode is still in `pd_front.py` for comparison.

## Correctness

Provenance matters more than the numbers here, so it gets its own column. *Same-input* means both
sides start from one captured set of attention inputs — those rows test reconstruction and writing
math. *End-to-end* means a real FP8 prefill fed a real MXFP4 decode.

| check | provenance | result |
|---|---|---|
| rebuilt cache arrays vs. the caches the same MLX prefill produced | same-input (both sides on the Mac, in MLX) | 313/313 bit-exact |
| bridge-written blocks vs. oMLX's own blocks | same-input (blocks from an MLX-computed capture) | 11/11 identical (only `created_at` differs) |
| torch pooling port vs. MLX truth, T=23,217 | same-input, cross-framework | projections / window / carries bit-exact; pooled 99.95–99.96% of elements identical, worst 1 bf16 ulp |
| hook selftest, chunked vs. one-shot | same-input | 52/52 |
| marker retrieval, bridged and native | end-to-end | 6/6 (earlier) + 7/7 (bench6) |
| judged quality eval, 5 questions, bridged leg | end-to-end | 5/5 on two passes (native 5/5) |
| Spark FP8 attention inputs vs. Mac MXFP4 attention inputs, same tokens | end-to-end, numeric | **not yet measured — open** |

The bit-exact rows cannot speak to the FP8-vs-MXFP4 difference noted under Limits, because two
different quantisations cannot produce bit-identical tensors — the exactness is evidence the inputs
were shared, not that the gap is zero. That gap currently has behavioural evidence only. Measuring it
directly is the open item.

## The ceiling — found, root-caused, and cured

A third benchmark round pushed past 81K and hit a hard wall:

| prompt | path | result |
|---|---|---|
| 78,504 tok | bridged | **65.6 s** — reproduces the 81K figure on a second cold seed |
| 97,848 tok | bridged | **fails.** `RuntimeError: 47 block files not written`; fell back to native. Client saw 905 s |
| 100,083 tok | native cold | 259.9 s (decode degraded to 4.6 tok/s — the machine was out of headroom) |

Cause: the block writer held one materialised cumulative snapshot per boundary until `finalize()`,
so peak memory was quadratic in prompt length (~16 GB at 39 boundaries, ~23 GB at 47). On a 256 GB
machine with a 156 GB model resident, 47 boundaries exhausted the headroom.

**The cure shipped the same day.** The writer now streams: each boundary is stored through oMLX's
own pipeline and released the moment it is snapshotted, and the assemble path stores incrementally,
so peak memory is ONE boundary snapshot instead of N(N+1)/2 of them. Validated two ways before going
live: the synthetic writer test (two boundaries through the streaming path, hash chain + tensor
layout matching a real oMLX reference block), and live at **52 boundaries / 109,085 tokens — the
first 100K+ bridge to work** (83.5 s to first token against 256.8 s native, with the tail salvaged
from a capture that lost its last boundary — see the autopsy below). `PD_MAX_BRIDGE_TOKENS` stays as
a configurable envelope guard; the bug ceiling is gone.

## The bench4 autopsy — why a whole round of "bridge" numbers was actually native

The fourth benchmark round reported cold bridge runs of 55 s (20K), 220 s (80K) and 327 s (100K).
Every one of them was a **native fallback wearing a bridge label**: the decoder's own server log
shows it prefilling the full prompt (46.2 s / 202.2 s / 269.7 s), and the front door's failure lines
show why. Two bugs, one disease — the capture hook's 2-second idle flush racing a chunked prefill
(`--max-num-batched-tokens 8192`, ~4 s of GPU per chunk):

- **A — mid-request flush.** The watcher fired between chunks and wrote DONE with `manifest T` =
  tokens-so-far (16,384 of 18,424; 32,768 of 82,703). The front correctly refused the mismatched
  capture, then silently served natively.
- **B — split final chunk.** The flush sentinel landed *inside* the final chunk's 43-item enqueue
  burst, splitting one chunk across two captures: a T-correct manifest with missing tail-boundary
  windows (`KeyError: kvwin_108544` live) and 41 orphaned-item assertions per event.

Four fixes, all in this tree: the hook guards its idle flush with a CUDA-event query (GPU busy =
mid-request) and a chunk-alignment check; the front validates every DONE manifest (`T` match, no
`partial_start`, no `position_gaps`) and keeps scanning until the engine returns + a grace window;
a missing tail is **salvaged** — the longest contiguous boundary prefix is assembled and written, so
the decoder prefix-hits at B and natively prefills only the remainder (this turned a 327 s failure
into an 83.5 s partial win at 109K); and a bridge that cannot be trusted **declines loudly**, with
the verdict (`complete` / `partial B/T` / declined + reason) in the `X-PD-Bridge` response header,
which `bench_cold.py` now records. Full detail: docs/FINDING-bench4-cold-fallback.md.

The same autopsy explains the "unexplained 20K variance" from the previous round (25.5 s vs 55.4 s):
the slow runs were this bug. The 25.5 s sample was genuine.

## What these numbers do not show

- **The quality eval is five questions on one document.** Bridged 5/5 twice is evidence the cache is
  faithful, not a benchmark suite. Prefill uses FP8 weights and decode uses MXFP4, so bridged output is
  not token-identical to native.
- **Warm turns gain nothing**, by design.
- **Single stream only** (`--max-num-seqs 1`). No concurrency numbers.
- **Small n.** ~80K has three cold bridged samples, ~100K two, ~20–25K two at different token counts.
  The protocol asks for 3 seeds per size in one sitting; bench6 is one seed per size.
- **Three machines vs one.** Two prefill nodes plus a decoder against a decoder alone. This is a
  latency result on hardware you already own, not a throughput-per-dollar or per-watt claim.
- **13B active of 284B.** Compute per token is modest; what makes the problem hard is the cache
  structure and the 149/156 GB resident footprint, not the FLOPs.
- **One decoder, one prefill pair.** No scaling data.
- The two legs of each pair use different seeds and therefore differ slightly in token count
  (≤0.3%). That is deliberate — reusing a seed would warm the decoder's cache and invalidate the
  cold baseline.

## Superseded

An earlier run measured 1.19× at 19K. That run put the decoder's one-time model load *inside* the
bridged leg while the native leg ran warm, and its decode rate came in at 13.0 tok/s against native's
24.7. Re-running with a warm engine on both legs produced the numbers above and closed the decode
gap. **Do not cite the 1.19×.**
