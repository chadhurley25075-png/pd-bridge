# Streaming capture — push blocks during prefill (`PD_STREAM=1`)

**Status:** implemented on branch `stream-capture`, CPU-tested, **not yet deployed to a live pair.** Default
behaviour is unchanged; everything here is behind `PD_STREAM=1` on both ends. Deploy plan and rollback:
[`DEPLOY-STAGED.md`](../DEPLOY-STAGED.md).

## The problem this removes

The pooled capture (hook v5, `spark/capture_sitecustomize_v3.py`) holds the whole request's decoder state on the
prefill box until the request ends, then writes 43 layer files. That state costs ~9.9 KB per token of unified
memory on rank 0. On a GB10 with the model up, free memory is ~17–22 GB, and the 2026-09-08 guard seals the
capture when `MemAvailable` crosses 5 GB — so 1M-token prompts come back **`partial 772,096 / 710,656 / 491,520`**
(70–78 % complete, see RESULTS.md *Known limits*). The seal is correct; the O(T) footprint that triggers it is the
design flaw.

The kv-mode connector (Ben's `spark/pd_kv_connector.py`, merged in #1) never had this problem: it cuts a block the
moment its rows exist, ships it, and drops it. Its footprint is O(one step). This note ports that shape to the
pooled DV4 path.

## Why the pooled capture is streamable at all

The Mac assembler (`studio/pd_assemble_blocks.py::_set_layer`) builds the decoder state at boundary `b` from, per
layer:

| tensor | what it is | depends on |
|---|---|---|
| `kvwin_b` `[128,512]` | pre-RoPE SWA rows `[b-128, b)` | tokens `< b` |
| `pooled[:b//ratio]` | compressor cache rows | tokens `< b` (ratio 4 or 128) |
| `prev_kv_b`, `prev_gate_b` `[4, out_dim]` | raw rows `[b-4, b)` (ratio-4 layers) | tokens `< b` |
| `idx_pooled[:b//4]`, `idx_prev_*_b` | the indexer's, same shapes | tokens `< b` |

Nothing at boundary `b` depends on tokens `≥ b`. Every boundary's contribution beyond the previous one is a fixed
small set of tensors (**one segment ≈ 20 MB for 2048 tokens**), and once shipped the Spark never needs them again:
the accumulators (`_KVState.snaps[b]`, `_PoolState.prev[b]`, the pooled rows `[prev_b//ratio, b//ratio)`) can be
released. The only state that must stay resident is the per-layer *tail* — the last 128 kv rows, the last ≤136 raw
compressor rows, `pool_layer`'s carry — which is O(1) per layer.

## Design (10 lines)

1. `PD_STREAM=1` on the hook. Nothing else changes for a hook without it (`manifest.stream.enabled=false`).
2. The worker thread ingests one layer-chunk at a time, in layer order. When **every layer has ingested the chunk**
   (`len(r.kv) == num_layers and r.calls % num_layers == 0`; layer count = the projection weights' layer count, so a
   partially-ingested first chunk can never emit), `E = min(next_pos)` is the position all layers have reached.
3. For every boundary `emitted_b < b ≤ E`: build `seg_<b:08d>.safetensors` holding, for every layer, `kvwin_b`,
   `pooled_b` (= rows `[prev_b//ratio, b//ratio)`), `prev_kv_b/prev_gate_b`, `idx_pooled_b/idx_prev_*_b`.
4. Pass 1 proves every tensor exists **without mutating state**; pass 2 detaches them. D2H copies run on the side
   stream (the worker's current stream). Write `.tmp`, `os.replace`, and only then pop the device snapshots.
5. The file lands in the capture dir served by the existing `spark/pd_share.py` (`PD_TRANSPORT=tcp10`), which never
   lists `.tmp` files — so the Mac only ever sees complete segments.
6. The Mac front door (`PD_STREAM=1`, `studio/pd_front.py::spark_prefill_stream`) polls `/_ls/<stamp>`, pulls each
   segment the moment it is listed **in boundary order**, and `StreamAssembler.on_segment` builds that boundary's
   cache (same `_set_layer`) and stores it through oMLX's own writer (`BlockWriter.store_boundary`) immediately. The
   per-layer `pooled` prefix is concatenated once per boundary (O(b), the same order as the writer's own snapshot).
7. At the end the hook emits any boundaries the last chunk completed, writes the layer files with the **end state
   only** (`kvwin_end`, `buf_*`, `prev_*_end`, `*pooled_tail`), then `manifest.json` (with `stream.segments`,
   `emitted_T`, `segment_bytes`, `error`), then `DONE` — last, as before.
8. The Mac cross-checks: `SegmentStream.check_manifest` refuses a manifest whose `segments` are not the contiguous
   run `block, 2·block, …, emitted_T`, or that lists a boundary beyond the request; the verdict names anything
   listed-but-never-consumed.
9. **The seal condition changes.** The 5 GB `MemAvailable` guard is untouched, but its estimate becomes
   `(T - emitted_b) × 9876 B` — the *unshipped backlog*, never more than one chunk (~40 MB at 4096) on a healthy
   stream, far below the 0.5 GB `PD_CAPTURE_MIN_ABORT_GB` threshold. The seal now fires only when shipping has
   actually stalled. A segment write failure seals at `emitted_b` itself (`DONE = stream-error`).
10. Verdicts stay honest: `complete` if `B == (T//2048)·2048`, `partial B/T` if the manifest sealed early, `salvage B/T`
    if DONE never came but boundaries were stored; every row carries `transport` and (from usage) `cached_tokens`.

## Wire format

```
<capture_root>/<stamp>/
  manifest.json                # written at start ({"T": null, "stream": true, "segment_file": "seg_{b:08d}.safetensors"})
  seg_00002048.safetensors     # appears ~1 chunk after token 2048 was prefilled
  seg_00004096.safetensors
  …
  layer_00.safetensors …       # END STATE ONLY in stream mode: kvwin_end, buf_kv/buf_gate, prev_*_end, pooled_tail, idx_*
  manifest.json                # rewritten sealed: T, boundaries, stream{enabled, segments[], emitted_T, segment_bytes, error}
  DONE                         # last; content = flush reason (flush_now | idle | new-request | mem-floor-seal | stream-error)
```

Segment keys: `layer_XX.kvwin_<b>`, `layer_XX.pooled_<b>`, `layer_XX.prev_kv_<b>`, `layer_XX.prev_gate_<b>`,
`layer_XX.idx_pooled_<b>`, `layer_XX.idx_prev_kv_<b>`, `layer_XX.idx_prev_gate_<b>` — all bf16, 2-D. The one-shot
files are unchanged when `PD_STREAM` is off; `spark/test_stream_capture.py` proves the replayed segments equal them
tensor-for-tensor at every boundary.

Share routes added (`spark/pd_share.py`): `/_flush` (restored — the front door has called it since 2026-09-06; the
live share had it, the repo copy had lost it) and `/_ack/<stamp>/<seg>` (optional; with `PD_SHARE_ACK_DELETE=1` the
Spark unlinks a stored segment).

## Failure modes

| what breaks | what happens | what the verdict says |
|---|---|---|
| **Segment write fails on the Spark** (disk full, I/O error, a hole in the pooled rows) | hook seals at `emitted_b`: end-state layer files + manifest (`stream.error`, `segments` so far) + `DONE=stream-error`. The pass-1 check means a refused boundary leaves the accumulators intact, so the tail export is still correct. Later chunks form a *new partial-start request* with streaming disabled (front door discards it: `usable_prefix=0`). | `partial emitted_b/T` if ≥ `PD_MIN_COVERAGE`, else declined with the error |
| **Partial stream: DONE never arrives** (hook died, engine crashed mid-prefill) | front door keeps storing whatever segments exist; after `PD_CAPTURE_GRACE` past engine return (or engine error) it finalizes what it has | `salvage B/T` — never `complete` |
| **Share unreachable mid-stream** | puller retries each tick; segments stay on the Spark (never deleted unless acked with `PD_SHARE_ACK_DELETE=1`) | as above if it never comes back |
| **Mac front door restarts mid-prefill** | the in-flight request is lost (as today). The segments are still on the share; a fresh consumer can replay them from `seg_00002048` (tested: `test_stream_assembler.py` K) — already-cached blocks are prefix hits for oMLX. With ack-delete ON this replay is impossible; that is why it is off by default. | the next request over the same prompt hits the stored prefix |
| **Restart mid-prefill on the Spark** (container restart) | the capture dir keeps its segments; no manifest/DONE; the front door's grace timer expires → salvage | `salvage B/T` |
| **Hook not streaming, front is** | front sees `manifest.stream=false` on the start manifest and runs the one-shot pull+assemble for that request | normal pooled verdict, `stream=false` in X-PD-Bridge |
| **Front not streaming, hook is** | the one-shot front ignores `seg_*` files, waits for DONE, pulls the layer files — but those now hold the END STATE only, so `assemble_and_write` finds no `kvwin_2048` and writes 0 boundaries | **declined** (coverage 0). Deploy both sides together — see DEPLOY-STAGED.md |
| **Prefix hit / request starts at position > 0** | streaming disabled for that request (`stream.note`), one-shot export, `partial_start` set | discarded by the front (as today) |
| **Memory floor still crossed** (something else eating the box) | estimate = backlog (< 0.5 GB) → no seal, a WARNING is logged; the box's health is the guard's job only for our own footprint | unaffected |

## What this costs

- **Spark worker:** per boundary, 43 layers × a handful of small D2H copies + one ~20 MB safetensors write, under the
  capture lock. At ~1,100 tok/s a boundary arrives every ~1.9 s; the write is tens of ms. The forward thread never
  waits on the worker (unchanged).
- **Spark disk:** segments accumulate in the capture dir until the capture is pruned (`PD_CAPTURE_KEEP=3` newest) or
  acked-and-deleted: ~10 GB per 1M-token capture, so up to ~30 GB resident. `PD_CAPTURE_DISK_FLOOR_GB=40` still
  refuses new captures below 40 GB free. Check `df` on the prefill head before a 1M run.
- **Mac:** the pooled prefix per layer is concatenated per boundary — same O(b)-per-boundary order as the batch
  path's snapshot materialisation, but now overlapped with the prefill instead of after it. Memory: the running
  prefixes (~10 KB/token total) + one boundary snapshot.
- **Wall clock:** the pull+assemble step that used to follow the prefill (~60 s at 700K) now runs concurrently; the
  tail after engine return is the last chunk's segments + finalize.

## What is NOT verified yet (needs the live pair — Chad's go)

- Real `pd_pool_torch.pool_layer` row-availability at chunk ends on CUDA (the CPU test uses a fake pooler; pass 1 of
  `_emit_segment` refuses and seals rather than guess if rows are short).
- Worker-thread cost of the per-boundary D2H + write against a real 4096-token chunk cadence.
- The Mac-side `StreamAssembler` under mlx (the ordering/manifest logic is tested under numpy; `_set_layer` and
  `store_boundary` are the existing, benchmarked calls).
- End-to-end: a 1M-token cold run coming back `complete` with a flat `MemAvailable` on the prefill head.

## Tests

- `spark/test_stream_capture.py` — 23 checks, CPU torch, fake pooler/weights: default unchanged; segments emitted
  during ingest; replay == one-shot at every boundary; manifest/DONE order; no early emission; write-failure seal;
  memory-floor estimate tracks the backlog (streaming completes where one-shot seals).
- `studio/test_stream_assembler.py` — 16 checks, numpy: ordering, refusals, manifest integrity (holes, beyond-T,
  behind), resume from the share, `consume_dir` over a filling directory with acks, partial stream.
- Both run in `make test`.
