# Reproducibility pack — conversion, runtime lock, and the 24,576-token receipts

Answering three requests directly. Where a request contains an assumption we cannot support,
we correct it rather than answer around it.

---

## 1. Mac conversion recipe + manifest

**Correction first: we did not convert the model.** The decode checkpoint is a *published*
MXFP4 MLX build of DeepSeek-V4-Flash-0731. There is no conversion recipe of ours to hand you,
and inventing one would misrepresent the provenance.

The one property of that checkpoint the bridge depends on: **it retains the DSpark MTP heads.**
Many MLX conversions strip auxiliary heads during quantization; oMLX's Lightning MTP path is
inert without the embedded `dspark_*` / `mtp.*` weights. If you rebuild the checkpoint yourself,
verify `mtp.*` tensors survive, or speculative decode silently does nothing.

Quantization actually present in `config.json`:

```json
"quantization": {"group_size": 32, "bits": 4, "mode": "mxfp4", "embed": false,
                 "layers.0.attn.wkv":  {"group_size": 32, "bits": 8, "mode": "mxfp8"},
                 "layers.0.attn.wo_a": {"group_size": 32, "bits": 8, "mode": "mxfp8"},
                 "layers.0.attn.wq_a": {"group_size": 32, "bits": 8, "mode": "mxfp8"}, ...}
```
Mixed precision: MXFP4 bulk, **MXFP8 on layer-0 attention projections**, embeddings unquantized.

### What IS produced on the Mac — and it is the artifact that matters

The bridge's correctness claim is *"the Spark computes the cache with the **Mac's own** weights."*
That is made concrete by one export, run **on the decode node**:

```bash
$OMLX_PYTHON studio/pd_export_proj_weights.py --out $PD_HOME/proj_weights
```

It writes the **dequantized bf16** attention-projection weights for **all 43 layers** — `wkv`,
`kv_norm`, compressor `wkv`/`wgate`/`ape`/`norm`, and the indexer-compressor equivalents — into one
safetensors plus a JSON meta. This file is then staged on the prefill node; the Spark uses it so the
pooled cache is arithmetically the Mac's, not an approximation of it.

**Manifest (this is the file our published numbers were produced with):**

| file | bytes | sha256 |
|---|---:|---|
| `dv4_proj_weights.safetensors` | 794,325,527 | `db8511b6ab8024637948ffc57714858a6298913f962bed911ec9f61a59a923ca` |
| `dv4_proj_weights.json` | 18,776 | `ed92dc74b036748dce92a1e0218667a3fd91414dc79af4d86f685f01baeae140` |

JSON meta carries the geometry the pooling math must agree on:
`rms_norm_eps 1e-06` · `head_dim 512` · `qk_rope_head_dim` · `index_head_dim` · `hidden_size` ·
`compress_ratios` (the per-layer `[…4, 128, 4, 128…]` pattern).

Ground truth for validating a rebuild: `studio/pd_export_pool_truth.py` (MLX truth) and
`spark/pd_pool_validate.py` (torch port vs that truth).

---

## 2. oMLX patch + complete runtime lock

### The patch
`studio/omlx-0.6.4-paged_ssd_cache-disk-index-fallback.patch` — one function against
`paged_ssd_cache.py`:

```python
-        return self._index.get(block_hash)
+        found = self._index.get(block_hash)
+        if found is None:
+            found = self._pd_index_from_disk(block_hash)
+        return found
```

**Why it is needed:** upstream indexes prefix-cache blocks only by a **startup scan**. Blocks
written by an external producer mid-run are therefore invisible until restart. On an index miss the
patch stats the expected path, validates the file with **the same reader and compatibility check
the startup scan uses**, and indexes it.

**Verify it is actually applied** — `grep -c _pd_index_from_disk .../omlx/cache/paged_ssd_cache.py`
must return 3. We shipped this patch while one of our own decode nodes was running **unpatched**;
without it the bridge can write valid blocks and the decoder will still prefill natively, because
the block is invisible until restart. A silent 5x slowdown with no error. Check the grep.

**Scope, stated plainly:** it does not change block format, hashing, eviction, or the cache's
correctness rules; it only makes an already-valid on-disk block visible without a restart. Cost is
one `stat` on a true miss. It is additive — remove it and the system still runs, just without
external block visibility.

### Complete runtime lock (live, as measured)

**Decode node — Mac Studio (M3 Ultra)**

⚠ **Correction, and it matters for reproduction:** the published results were produced on our
**256 GB** Studio running **python 3.13.11**, not the 512 GB box. Package versions are identical
across both, but the interpreter differs — and `bench_cold.py` builds its fixture from the
interpreter's own stdlib, so **the Python version is part of the fixture, not just the runtime.**

| | |
|---|---|
| macOS | 26.5.1 |
| python | **3.13.11** (results box) |
| omlx | 0.6.4 |
| mlx | 0.32.0 |
| mlx-metal | 0.32.0 |
| mlx-lm | 0.31.3 |
| dflash-mlx | 0.1.10+omlx.7 |
| numpy | 2.3.5 |
| safetensors | 0.8.0 |

**Prefill nodes — 2× DGX Spark (GB10), tensor-parallel 2**
| | |
|---|---|
| OS | Ubuntu 24.04.4 LTS |
| kernel | 6.17.0-1029-nvidia |
| NVIDIA driver | 580.173.02 |
| engine image | `aidendle94/sparkrun-vllm-ds4-gb10:production-ready` |

⚠ **TP=2 is not an optimization.** The prefill weights are ~149 GB and a Spark has 121 GB unified;
the model does not fit on one node. A single-Spark reproduction is not possible with this build.

⚠ **This vLLM build rejects `--rope-scaling`.** Set the YaRN factor in each prefill node's model
`config.json` instead; passing the flag kills the launch, and a supervisor that restarts *without*
it will silently serve a different rope factor than the decoder. See
`FINDING-stale-limits-after-a-window-change.md`.

---

## 3. The exact 24,576-token fixture + raw receipts

### The fixture — nothing to download

`24,576` is not the prompt length; it is **B**, the block-aligned cached prefix (12 blocks × 2048).
The prompt is 24,924 tokens. It is generated deterministically:

```bash
python bench/bench_cold.py --chars 75041 --seed 721 --url http://<decoder>:8012
```

`bench_cold.py` builds the document from **Python's own stdlib sources**, shuffled with
`random.Random(seed)`, concatenated to `--chars`, with a retrieval marker
`ZEBRA-{seed:04d}-{Random(seed*7).randint(1000,9999)}` → for seed 721, **`ZEBRA-0721-9199`**.

⚠ **Honest reproducibility caveat:** the corpus is your interpreter's stdlib, so byte-identical
reproduction requires the same Python version. The *shape* (length, block alignment, marker)
reproduces anywhere; the exact bytes do not. If you need byte-identity, pin the interpreter or
substitute your own corpus — the harness only requires a deterministic document plus a marker.

### Raw receipt — COLD, bridge engaged

```json
{"seed":721,"chars":75041,"ttft_s":28.17,"marker":"ZEBRA-0721-9199","found":true,
 "answer":"ZEBRA-0721-9199",
 "bridge":{"skipped":false,"mode":"pooled","tokens":24924,
           "t_tokenize":0.06,"stamp_seen":0.22,
           "t_engine":17.1,"t_flush_signal":17.24,"t_done_seen":17.69,
           "t_pulled":18.15,"pulled_gb":0.254,"t_assembled":21.91,
           "boundaries_ok":12,"B":24576,"coverage":0.986,
           "verdict":"complete","blocks":12}}
```

Derived from those raw fields:
- **transfer: 0.254 GB in 0.46 s** (`t_pulled − t_done_seen` = 18.15 − 17.69)
- **assemble on the Mac: 3.76 s** (21.91 − 18.15)
- **12 of 12 block boundaries validated**, `verdict: complete`
- `coverage 0.986` — the 348-token tail beyond B is not block-aligned and is prefilled natively by
  design. Coverage is deliberately **not** 1.0 and we do not round it up.

### Raw receipt — ZERO REPLAY

The claim "the decoder does not recompute what the bridge delivered" is evidenced by the *second*
run of the identical prompt:

```json
{"seed":721,"chars":75041,"ttft_s":4.87,"marker":"ZEBRA-0721-9199","found":true,
 "bridge":{"skipped":true,"tokens":24924,"cached_prefix":24576,"tail":348,
           "why":"warm — new tail 348 < 8192, oMLX prefills it natively",
           "t_bridge_total":0.03}}
```

**`cached_prefix: 24576` of `tokens: 24924` → the decoder prefilled 348 tokens, not 24,924.** The
24,576 tokens the bridge wrote were read back from the prefix cache and **not replayed**. The front
door correctly *disqualifies itself* here (`skipped: true`) because there is nothing left worth
bridging — which is also the honest reason 4.87 s is a cache result and **not** a bridge speedup.
We label it as such in `RESULTS.md` rather than quoting it as a 5.8× win.

### Independent block-level verification

`studio/verify_blocks.py` compares bridge-written cache blocks against natively written ones
key-by-key (dtype, shape, max |Δ|, bf16 decoded via `uint16<<16`). Use `--self` as a sanity control
(a file against itself must be all zeros).
