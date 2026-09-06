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
| prefill pair (2× GB10, TP2, FP8, hook on) | 1,944 |

**4.6×**, consistent at 19K and 81K.

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

## The ceiling, and the run that found it

A third benchmark round pushed past 81K and hit a hard wall:

| prompt | path | result |
|---|---|---|
| 78,504 tok | bridged | **65.6 s** — reproduces the 81K figure on a second cold seed |
| 97,848 tok | bridged | **fails.** `RuntimeError: 47 block files not written`; fell back to native. Client saw 905 s |
| 100,083 tok | native cold | 259.9 s (decode degraded to 4.6 tok/s — the machine was out of headroom) |

Cause: the block writer holds one materialised cumulative snapshot per boundary until `finalize()`,
so peak memory is quadratic in prompt length (~16 GB at 39 boundaries, ~23 GB at 47). On a 256 GB
machine with a 156 GB model resident, 47 boundaries exhausts the headroom.

Two fixes are in the code as shipped: `PD_MAX_BRIDGE_TOKENS` (default 81920) declines the bridge
above the validated envelope, and `_drain` now aborts after 30 s without progress instead of burning
its full timeout — that timeout is what turned a failed write into an 11-minute stall.

**The same round also showed cold-start variance at 20K**: 25.5 s bridged in one run, 55.4 s in
another, against a stable ~44 s native. In the slow run the decoder spent 43.7 s despite holding a
valid prefix hit, with decode healthy at 26.1 tok/s. That is not explained yet. Treat the 19-20K
figures as provisional; the ~80K figures reproduced twice and are the ones we stand behind.

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
