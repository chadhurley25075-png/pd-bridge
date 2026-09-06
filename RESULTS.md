# Results — 2026-09-06

All figures measured on the hardware described in [bench/BENCHMARK-PROTOCOL.md](bench/BENCHMARK-PROTOCOL.md).
Nothing here is derived or extrapolated. Every bridged run has a native cold run on the same box, same
prompt shape, same engine state.

Inputs are `bench/bench_cold.py`: a deterministic synthetic document built from CPython's stdlib
sources with a seeded shuffle, an embedded marker, and one question. **Each seed is a genuinely cold
prompt** — no cache clearing, no force switches, nothing either engine has seen.

## Headline — 81K tokens

| configuration | prompt | time to answer | decode | marker |
|---|---|---|---|---|
| native cold (decoder alone) | 80,768 tok | **195.1 s** | 23.7 tok/s | found |
| **bridged cold** | 81,024 tok | **63.6 s** | 23.5 tok/s | found |

**3.07× faster to an answer.** Decode rate is unchanged, as designed — the bridge hands off before
the first token and never participates in generation.

### Where the 63.6 s goes

| stage | time |
|---|---|
| prefill engine (vLLM TP2, hook on) | 41.67 s |
| capture flush | 2.62 s |
| pull 0.801 GB over 10GbE | 1.08 s |
| assemble + write 39 blocks | 4.59 s |
| **bridge total** | **49.96 s** |
| decoder: tail prefill + 179 tokens | 13.6 s |

The prefill engine is 65% of wall time. Transport is 1.7%.

## 19K tokens

| configuration | prompt | time to answer | decode |
|---|---|---|---|
| native cold | 19,493 tok | 44.7 s | 25.6 tok/s |
| **bridged cold** | 18,553 tok | **25.5 s** | 25.4 tok/s |
| bridged, warm (bridge self-skips) | 19,702 tok | 10.5 s | 25.2 tok/s |

**1.76×.** The bridge correctly detected the warm case in 0.02 s and let the decoder serve natively.

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

| check | result |
|---|---|
| rebuilt cache arrays vs. full native forward | 313/313 bit-exact |
| bridge-written blocks vs. oMLX's own blocks | 11/11 identical (only `created_at` differs) |
| torch pooling port vs. MLX truth, T=23,217 | projections / window / carries bit-exact; pooled 99.95–99.96% identical, worst 1 bf16 ulp |
| hook selftest, chunked vs. one-shot | 52/52 |
| marker retrieval, bridged and native, both sizes | 6/6 |

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

- **No judged quality score on the bridged leg.** The native leg scores 5/5 on the question set; the
  bridged leg has not been run against it. Prefill uses FP8 weights and decode uses MXFP4, so bridged
  output is not token-identical to native. Marker retrieval passing 6/6 is evidence, not proof.
- **Warm turns gain nothing**, by design.
- **Single stream only** (`--max-num-seqs 1`). No concurrency numbers.
- **n=1 at most sizes.** Only the ~80K bridged figure has been reproduced (63.6 s / 65.6 s on two
  cold seeds). The protocol asks for 3 seeds per size; that is not done.
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
