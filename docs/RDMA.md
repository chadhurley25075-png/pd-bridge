# Plain-attention models over RDMA — the KV bridge

The DeepSeek-V4 path in this repo ships a *computed* decoder cache over plain Ethernet, because V4's cache is a
pooled function of the prefill tensors and is only ~10 KB/token. A dense GQA model (Qwen3, Llama, Mistral, …) is the
opposite case: the decoder's cache **is** the prefill engine's post-RoPE K/V rows, nothing has to be recomputed, but
the payload is ~25× larger (Qwen3-32B: 256 KiB/token, 8 GB at 32K). This path moves those rows over RoCEv2 while the
prefill runs, writes them as the decoder's own prefix-cache files, and hands the decoder a finished cache.

Reference pair (all numbers below): one DGX Spark (GB10, vLLM 0.28, TP1) → one Mac Studio M2 Ultra (oMLX 0.6.4),
Qwen3-32B bf16, ConnectX-7 ↔ ConnectX-4 Lx direct RoCEv2 link. On the Mac the NIC is driven by
[MelonDMA](https://github.com/denmrnngp-cloud/MelonDMA) (DriverKit, libibverbs-compatible user space).

## What happens to one request

```
client ─► front door (studio/pd_front.py, PD_MODE=kv PD_TRANSPORT=rdma4)
            │ render + tokenize exactly as oMLX will
            │ POST /pd/stage {chain hashes, geometry}  ─► oMLX hook allocates the whole cache, starts a fill thread
            ▼
          vLLM /v1/completions {prompt ids, kv_transfer_params: pd_tag, omlx_model, omlx_block}
            │ spark/pd_kv_connector.py (official v1 KV connector, kv_producer)
            │   every full 256-token block: stack on the GPU in oMLX tensor order, one contiguous copy into a
            │   registered buffer, RDMA WRITE → `pd_rdma recvd` lands <cache>/<h0>/<hash>.safetensors
            │   (spark/pd_omlx_block.py builds the byte-exact oMLX header and chain hash)
            │   after the last block: tail rows (all but the last prompt token), manifest, DONE
            ▼
          front door: DONE seen → tail → <cache>/pd_tail/<sha256(ids)>.safetensors → forward the chat request
            ▼
          oMLX (studio/pd_omlx_hooks.py): prefix walk finds every block → restore = views of the staged arrays
          (no read, no concat) → tail rows appended in place → one token left to prefill → decode
```

Every reply still carries `X-PD-Bridge` (verdict, transport, blocks, per-stage timings). A bridged run whose
`cached_tokens` is not `T − 1` did not take this path.

## Components

| file | side | role |
|---|---|---|
| `spark/pd_kv_connector.py` | prefill | vLLM v1 KV connector: gathers K/V rows per step, cuts decoder-sized blocks, async RDMA sender, tail rows, manifest |
| `spark/pd_omlx_block.py` | prefill | oMLX chain hashes + block file header (stdlib only) |
| `spark/pd-launch-kv.sh` | prefill | vLLM launch with the connector (`PD_HOOK=off` = control run; `PD_EAGER`, `PD_ATTN_BACKEND`, `PD_QUANT` variants) |
| `spark/pd_memguard.sh` | prefill | stops the engine before unified memory runs out (started by the launch script) |
| `rdma/pd_rdma.c`, `rdma/pd_rdma_tx.c`, `rdma/pd_rdma_common.h` | both | transport: `pd_rdma recvd` (Mac), `libpd_rdma_tx.so` (loaded by the connector), R1 `serve`/`client` pull mode |
| `studio/pd-rdma-recvd.sh` | decode | starts the receiver (async block writes) |
| `studio/pd_front.py` + `studio/pd-front-kv.sh` | decode | front door in kv mode (`PD_TRANSPORT` = `tcp10`, `rdma`, `rdma2`, `rdma4`) |
| `studio/pd_omlx_hooks.py` | decode | oMLX hooks: staged restore, tail install, restore reserve, timing |
| `studio/omlx-0.6.4-paged_ssd_cache-disk-index-fallback.patch` | decode | oMLX sees blocks written after model load (`has_block` and `get_block_metadata`) |
| `bench/engine_only.py`, `bench/bridge_direct.py`, `bench/run_r0.sh` | client | hook-off control, one bridged request without the front door, native-vs-bridged matrix |
| `spark/test_kv_gather.py`, `studio/test_omlx_block.py`, `studio/pd_verify_kv.py` | tests | layout/gather without a GPU; byte-exact block headers vs real oMLX blocks; tensor-level bridged-vs-native check |

## Wire protocol (control over TCP, bytes over RDMA)

The receiver registers one arena (default 128 MiB) and publishes it in the handshake; the sender RDMA-writes a
payload into it and then names it on the control line. Control TCP rides whatever IP link both boxes share (the Mac
side of a DriverKit NIC has no macOS network interface).

```
S->C READY
C->S OBLK <hash> <len>          arena = complete oMLX block file; lands at <omlx-cache>/<hash[0]>/<hash>.safetensors
C->S BLK <tag> <i> <L> <H> <B> <D>   arena = k then v (R2 layout, assembled on the Mac)
C->S FILE <tag> <name> <len>    arena = file bytes (tail.safetensors, manifest.json) → <root>/<tag>/<name>
C->S DONE <tag> <status>        written last
S->C OK | ERR <reason>          after each
```

`pd_rdma recvd` acks `OBLK` after one memcpy into a pre-touched pool buffer and writes files on a thread; `FILE` and
`DONE` wait for the queue to drain, so the tail, the manifest and DONE still land after every block, and a failed
background write comes back as `ERR` on the next command (`PD_RDMA_ASYNC_WRITES=0` = synchronous writes).

## Setup

```bash
cp config.example.env config.env && cp config.rdma.example.env config.rdma.env   # edit both
source config.env && source config.rdma.env
```

**Prefill box** (rdma-core / libibverbs installed, vLLM ≥ 0.28 in a venv, `ulimit -l` ≥ 128 MiB):

```bash
make -C rdma                                      # pd_rdma + libpd_rdma_tx.so
PYTHONPATH=spark $PD_VLLM_PYTHON spark/test_kv_gather.py
PD_RDMA_STREAM_HOST=$PD_RDMA_RECV_BIND PD_KV_MODEL=$PD_KV_MODEL PD_KV_SERVED=$PD_SPARK_MODEL ./spark/pd-launch-kv.sh
```

**Mac** (MelonDMA installed and approved, ConnectX cabled to the prefill box):

```bash
MELONDMA_DEXT=/path/to/MelonDMA/dev/src/dext make -C rdma     # builds and signs pd_rdma with the userclient entitlement
# RoCE addressing for the MelonDMA provider, kept out of the repo:
#   ~/.config/pd-bridge/rdma.env  →  export MELONDMA_LOCAL_IP=… MELONDMA_LOCAL_MAC=… MELONDMA_REMOTE_MAC=…
bash studio/pd-rdma-recvd.sh &                                # start before the engine's first bridged request

OMLX_PKG=$($OMLX_PYTHON -c 'import omlx,os;print(os.path.dirname(omlx.__file__))')
patch -p0 -d "$OMLX_PKG/cache" < studio/omlx-0.6.4-paged_ssd_cache-disk-index-fallback.patch
# hooks, loaded only when PD_OMLX_HOOKS=1 (the one-line .pth never affects other users of the venv):
echo "import os, sys; exec(\"if os.environ.get('PD_OMLX_HOOKS') == '1':\\n    sys.path.insert(0, '$PWD/studio')\\n    import pd_omlx_hooks\")" \
  > "$(dirname "$OMLX_PKG")/pd_omlx_hooks.pth"
PD_OMLX_HOOKS=1 PD_OMLX_STAGE=1 PD_OMLX_TAIL=1 PD_OMLX_RESERVE_TOKENS=1024 PD_CACHE_DIR=$PD_CACHE_DIR \
  "$(dirname "$OMLX_PYTHON")/omlx" serve &                    # serving $PD_MODEL, cache dir = $PD_CACHE_DIR

PD_TRANSPORT=rdma4 PD_OMLX_STAGE=1 bash studio/pd-front-kv.sh &
```

The first bridged request after `pd_rdma recvd` restarts fails once and is declined: the sender learns that its old
queue pair is gone on its first send, and the next request reconnects.

## Verify before you measure

1. `$OMLX_PYTHON studio/test_omlx_block.py "$PD_CACHE_DIR" ids.json` — rebuilt headers must be byte-identical to
   blocks oMLX wrote itself, and `chain_hashes()` must equal oMLX's `compute_block_hash`.
2. One bridged request: verdict `complete`, `spark_transport` `omlx`, `tail_landed` true, `cached_tokens = T − 1`.
3. Retrieval: `bench/bench_cold.py --max-tokens 1024` through the front door must find the marker. CUDA and Metal
   are not bit-identical at bf16, so compare answers and tensors with a tolerance, never file hashes.

## Measure

```bash
python3 bench/bench_cold.py --chars 142000 --seed 7401 --url http://127.0.0.1:$PD_PORT --model $PD_MODEL_NAME --max-tokens 16
# hook-off control on the same token ids (tokenize in the venv, send from any python that can reach the engine):
$OMLX_PYTHON bench/engine_only.py --chars 142000 --seed 7401 --tokenizer $PD_MODEL --save-ids ids.json
python3 bench/engine_only.py --chars 142000 --seed 7401 --ids-file ids.json
```

Report the bridge overhead as TTFT minus the engine time, not TTFT alone: the prefill box dominates and its speed
moves with temperature. A GB10 spends most of a long prefill in thermal slowdown (`nvidia-smi -q -d PERFORMANCE`); a
run started cold is ~8% faster at 32K than the same run started hot. Record the throttle counters per run.

## Porting to another model

Feasibility, in order:

1. **Plain attention in the decoder.** Every oMLX layer must be a `KVCache` (no sliding window, no recurrent or
   pooled state). Then the decoder's cache is the prefill engine's post-RoPE K/V rows and nothing else is needed.
2. **Same RoPE on both sides.** For long windows set `rope_scaling` in each side's `config.json` rather than a launch
   flag, and give the scaled model its own directory and name on both boxes: the oMLX chain hash is keyed by model
   name and tokens, so a distinct name keeps unscaled blocks from ever being served to the scaled model.
3. **The connector sees a layout it knows.** `_detect_layout` accepts vLLM's packed FLASH_ATTN view
   `(blocks, kv_heads, 16, 2·head_dim)` and the 5-D `2BN` / `B2N` layouts; `spark/test_kv_gather.py` pins all three.
4. **A block fits the registered buffers.** Block bytes = layers × 2 × kv_heads × block × head_dim × 2 (bf16) plus a
   small header. Qwen3-32B: 64 MiB in a 128 MiB arena. Raise `PD_RDMA_ARENA_MIB` / `PD_RDMA_BUF_MIB` together when a
   block is larger (MelonDMA allows 512 MiB pinned per user client), or use a smaller decoder block size.
5. **The decoder block size** (`PD_OMLX_BLOCK` on the engine, `PD_BLOCK` in the front door) equals oMLX's
   `paged_cache_block_size` for that model.

Geometry for `/pd/stage` comes from the decoder model's `config.json` (`num_hidden_layers`, `num_key_value_heads`,
`head_dim`); nothing in the pipeline is sized for Qwen3 specifically.

## Knobs

| variable | where | default | meaning |
|---|---|---|---|
| `PD_RDMA_STREAM_HOST`, `PD_RDMA_STREAM_PORT` | engine launch | —, 18516 | where `pd_rdma recvd` listens; unset = file capture |
| `PD_RDMA_DEV`, `PD_RDMA_GID_INDEX`, `PD_RDMA_BUF_MIB` | engine launch | rocep1s0f1, 3, 128 | sender device, RoCEv2 GID, registered buffer |
| `PD_RDMA_RECV_BIND`, `PD_RDMA_ARENA_MIB`, `PD_RDMA_ASYNC_WRITES` | receiver | —, 128, 1 | control address, arena, async file writes |
| `PD_TRANSPORT` | front door | tcp10 | `rdma4` = oMLX-native blocks straight into the decoder cache |
| `PD_OMLX_STAGE` | front door + oMLX | off | staged restore (`/pd/stage`) |
| `PD_OMLX_TAIL`, `PD_OMLX_RESERVE_TOKENS` | oMLX | off, 0 | install the shipped tail; spare positions in restored caches |
| `PD_OMLX_TIMING` | oMLX | off | log the restore split (`[pd-timing]`, `[pd-stage]`, `[pd-tail]`) |
| `PD_MAX_BATCHED`, `PD_EAGER`, `PD_ATTN_BACKEND`, `PD_QUANT` | engine launch | 8192, on, —, — | prefill variants (none beat the default on a GB10) |

## Reference numbers (Qwen3-32B bf16, one GB10 → M2 Ultra)

Bridge overhead = TTFT − engine time, through the front door at ~33K tokens:

| stage of this work | overhead |
|---|---|
| files over 10GbE, Mac assembles and re-stores (R0) | ~89 s of a 131.6 s TTFT |
| oMLX-native blocks built on the GPU, pushed during prefill, landed in the decoder cache (R4c) | ~3.4 s |
| + tail rows shipped + staged restore | 1.96–2.69 s |
| + async receiver writes + warm staged buffers | **1.32 s** |

| tokens | native oMLX TTFT | bridged TTFT | engine alone | verdict | cached |
|---|---|---|---|---|---|
| 33,207 | ~196 s (derived: native measured at 169 tok/s on ~32K) | **43.93 s** | 42.61 s | complete | 33,206 |
| 33,771 (retrieval, 1,024 tokens) | — | 45.09 s | 43.72 s | complete, marker found | 33,770 |
| 62,337 (YaRN ×4) | 601.27 s measured at 68,616 tokens | 117.93 s | 108.57 s | complete, marker found | 62,208 (before tail rows) |

## Known limits

- The oMLX hooks patch private internals of oMLX 0.6.4 (`BlockAwarePrefixCache`, `Scheduler`, `KVCacheHandler`).
  Pin the version.
- The stage thread issues Metal work while the decoder is idle. A decoder busy with other requests during a bridged
  prefill is untested.
- Metal buffers go cold when idle: after a prefill step's gap the first GPU read of a staged buffer cost up to 1.5 s.
  The stage reads a small slice of every buffer every 0.5 s; single-element reads or unrelated GPU work do not help.
- On a GB10, `max_num_batched_tokens` 16K/32K, `torch.compile`, FLASHINFER and online FP8 weights did not beat eager
  FLASH_ATTN at 8,192; FP8 was ~45% slower.
