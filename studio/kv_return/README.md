# KV return — the second turn stops re-prefilling (2026-09-18)

**Default OFF.** Nothing in the main path changes unless `PD_KV_RETURN=1` is set on both sides.

## The problem

In pooled mode every turn re-prefilled the *whole* prompt on the Sparks, even when the decoder already held
the first 98% of it as cached blocks: a 713K-token conversation cost ~615 s of Spark prefill per turn.
Two things forced that:

1. `pd-launch-v3.sh` runs vLLM with `--no-enable-prefix-caching` (with it on, a repeated prefix is skipped and
   the hook sees nothing to pool — README *Gotchas*).
2. The capture hook treated a request whose first chunk starts at position `S > 0` as poison
   (`partial_start` → the front's `_cap_usable_prefix` returns 0 → decline → native fallback).

## The idea

Let vLLM's prefix cache do what it is good at (skip the prefix it still holds), and teach the hook to
*continue* the pooled capture from `S` instead of starting at 0. Continuing needs only a little state per
layer — the rows just before `S`:

| piece | rows kept | why |
|---|---|---|
| SWA window (pre-RoPE kv) | last 128 (+ `PD_KV_RETURN_KEEP`) | rows `[S-128, S)` |
| compressor raw rows | last 136 (+ keep) | remainder ≤127 + previous 4 |
| indexer raw rows (ratio-4 layers) | last 136 (+ keep) | same |
| pool carry | — | derivable from the raw rows + `S`, so not saved |

vLLM resumes at a 256-aligned `S` (block size 256) or at `T-1` (fully cached prompt), so the hook keeps
`PD_KV_RETURN_KEEP=256` extra rows. The state is written per finished capture as
`<stamp>/kvstate.safetensors` + `kvstate.json` (≈75 MB for 43 layers estimated from row counts; the 4-layer
test scaled to 47 MB — not a measured file size). The decoder then *merges*: `pooled = cat(parent.pooled[:S//ratio], new.pooled)`,
boundary snapshots `≤ S` from the parent, everything after `S` from the new capture — and the normal
assemble/write runs on a capture that is complete from token 0.

If the Spark no longer has the parent's state (capture janitor, prune), the decoder pushes it back over
RDMA (`rdma_pushfile.py`, scp fallback) into `<capture root>/_kvreturn/<stamp>/` before the engine call.

## Files

| file | side | what |
|---|---|---|
| `../../spark/capture_sitecustomize_v3.kvreturn.py` | prefill | the hook with the resume path (`_KVState.seed` / `_PoolState.seed`, state save in `_finish`, `ingest()` loads a covering state when `start > 0`). **A side-car, not a replacement:** it is built on the hook from the `stream-capture` branch (adds the 2026-09-08 memory/disk survival guards and `PD_STREAM`, both of which it carries; `PD_STREAM` stays off). |
| `../../spark/test_kv_return.py` | anywhere (CPU) | the proof — see below |
| `../../spark/pd-launch-v3.sh` | prefill | `PD_KV_RETURN=1` → `--enable-prefix-caching`, mounts the side-car hook, passes `PD_KV_RETURN`/`PD_KV_RETURN_KEEP` |
| `pd_kv_return.py` | decoder | `remember` (keep the parent capture) · `push_state` · `usable_prefix` · `merge_prefix` — every entry point is a no-op when off, and returns a verdict instead of raising |
| `pd_front.kvreturn.patch` | decoder | the five small hooks in `studio/pd_front.py` (apply: `patch -p1 < studio/kv_return/pd_front.kvreturn.patch` from the repo root) |
| `rdma_pushfile.py` | decoder | one file Mac → Spark over one MCDMA RDMA session (the reverse of `fabric/rdma_pulldir.py`), scp fallback |
| `needle2turn.py` | client | two-turn needle test: turn 1 ≥65K tokens with a needle at 65%, turn 2 = reply + ≥8K new tokens + the question |

## Proof

`spark/test_kv_return.py` — CPU only, real projection weights (layers 0–3: ratios 0, 0, 4, 128; layer 2 has an
indexer), synthetic hidden states. A one-shot reference capture over `[0, T2)` versus turn 1 `[0, T1)` + turn 2
resumed at `S`, merged by the decoder's rule: **every** `kvwin` / `prev` / `buf` / `pooled` / `idx_pooled` tensor
must be `torch.equal` to the reference, for `S` 256-aligned and `S = T1-1`, from the Spark's own state and from
a decoder-pushed `_kvreturn/` copy; and a start with no covering state must fall back to the old partial path.

```
PD_KV_RETURN=1 python3 spark/test_kv_return.py --weights /path/to/dv4_proj_weights.safetensors
...
RESULT: PASS        # 115 checks — first run 2026-09-18, re-run on this branch 2026-10-07
```

## Measured live (2026-09-18, one run pair, DeepSeek-V4-Flash, Spark pair TP2 → MCDMA door Studio → TB5 library Studio)

Needle test straight to the door (`:8012`), seed 921: turn 1 = 68,289 tokens with the needle at 65%, turn 2 =
+8,850 new tokens + the question. **Needle: PASS, exact.** vLLM resumed at position 68,096.

| stage | turn 1 (cold) | turn 2 (KV return) |
|---|---|---|
| Spark engine | 33.7 s | **5.4 s** (hook span 4.48 s) |
| capture size / RDMA pull | 0.754 GB / 0.68 s wire | 0.171 GB / 0.30 s |
| kvstate push | — | 0.26–0.48 s (check only: the state was still on the Spark) |
| prefix merge on the door | — | 0.67 s (43 layers) |
| **door total (bridge)** | 47–50 s | **16.9–17.7 s** |
| ring: TB5 door → library | 2.0 s (33 files) | 0.85 s (6 files, 120 MB) |
| ring: library decode first token / cached | — | 5.5 s / 75,776 of 77,035 |
| **ring total** | 358–460 s (fell back to door decode — see limits) | **48.5 s** |

Same morning, warm turns without KV return: first token 36–62 s. The Spark column is the point: at 68K the
second turn's prefill went 33.7 s → 5.4 s, and it should scale with the *new* tokens, not the conversation
(expected from the mechanism; one run does not prove the scaling).

## Honest limits

- **Measured once, at one size.** One two-turn needle pair at 68K→77K. The 713K case in the problem statement
  is the motivation, not a measurement — we have not run KV return at 700K.
- **The RDMA push path was not exercised live.** In the live run the parent state was still on the Spark, so
  `push_state` only checked. The push (and its scp fallback) is covered by the CPU test's `_kvreturn/` case,
  not by a live run.
- **One conversation at a time.** The hook picks a saved state by *position* (`T-256 ≤ S ≤ T`, newest first);
  the decoder's `parent_stamp` check is what stops a different document's state from being merged. Interleaved
  conversations need a content hash on the state.
- **Prefix caching is a trade.** With it on, an unrelated cold prompt that shares a prefix with an earlier one
  (a common system prompt) also starts at `S > 0` but has no covering state → partial → the front declines →
  native fallback. Safe, but that request is not bridged. We have not measured how often this happens.
- **Streamed captures (`PD_STREAM=1`) cannot be a merge parent** (`remember()` refuses them).
- The ring's first-turn fallback in the table is a separate, pre-existing flake (the library's oMLX restart
  over ssh timed out at 180 s), not part of this change.
- The merge re-reads the whole parent capture on the door (fine at 0.75 GB; at 7 GB it will cost seconds) and
  the assembler still re-assembles from token 0. Merging only blocks the decoder does not already hold is the
  obvious next step.
