# Porting the bridge to another model or decode engine

## The feasibility test — answer this first

**Does your decode engine have an on-disk, content-addressed prefix cache?**

The trick that makes this design cheap is: write the decoder's *own* cache files, exactly as it
would write them, and let it get a normal cache hit — one small patch so it notices files that
appeared after model load. That only works where the cache is a hash-chained block store on disk
(oMLX's PagedSSDCache is; vLLM's CPU/disk connectors are close in spirit). A single-file prompt
snapshot (llama.cpp's `--prompt-cache`) means load-order surgery instead of block injection — a
different, harder project. If your answer is no, stop here and save a week.

Second question: **is your model's per-layer cache state a pure function of tensors available at
prefill time?** For plain attention, K/V rows are. For hybrid/pooled caches (DV4's rotating +
pooling + indexer), the pooled states are pure functions of the attention input — that is the whole
insight this repo is built on, and it is why the prefill node can compute the *decoder's* finished
cache using the *decoder's own weights*.

## What is reusable as-is (the skeleton)

- `studio/pd_front.py`'s control flow: render with the decoder's own template → chain-hash the
  cold/warm decision → trigger prefill → validate the capture manifest → pull → assemble → write
  decoder-native blocks → forward the original request → fall back natively on any decline. Nothing
  in this *needs* DV4 except through the four seams below.
- `spark/pd_share.py` / `spark/pd_capture_http.py` and the capture protocol (`<stamp>/` dir,
  `manifest.json`, `DONE` written last). Model-agnostic.
- The hook *skeleton* in `spark/capture_sitecustomize_v3.py`: sitecustomize delivery (survives
  torch.compile), side-stream + worker-thread split, `is_current_stream_capturing()` guard,
  rank-0-only gating, position-0 request reset — and both mid-flush guards (CUDA-event-query +
  chunk-alignment). The failure history encoded in those guards is CUDA-vs-serving-engine wisdom,
  not DV4 wisdom.
- `studio/omlx_block_writer.py`, `studio/verify_blocks.py`, chain-hash warm detection —
  oMLX-specific but model-agnostic *within* oMLX.
- `bench/` entirely (bench_cold's verdict recording, hetero, the protocol).

## The four DV4-specific seams (what an adapter fills)

1. **Capture point + per-layer math.** Where to hook (for DV4: the plugin's `attention_impl`) and
   what to compute — projections, pooling compressors with overlap carries, the indexer, yarn-RoPE
   quirks. `spark/POOL-VALIDATION.md` is the record of establishing MLX semantics by probe; a new
   model's adapter needs an equivalent truth file.
2. **Capture file format.** The binding keys in `docs/DESIGN-v3-pooled.md` (`pooled`, `idx_pooled`,
   `kvwin_{b}`, `prev_*`, `buf_*`) plus the manifest fields the front's validator trusts
   (`T`, `partial_start`, `position_gaps`, `boundaries`).
3. **Decode-side assembly.** `studio/pd_assemble_blocks.py::_set_layer` knows oMLX's
   `RotatingKVCache`/`PoolingCache` state tuples (including the zero-width-values quirk).
4. **Projection-weights export.** Which decoder weights must exist on the prefiller
   (`studio/pd_export_proj_weights.py`).

## The interface (target shape — the front door should eventually be factored against this)

```python
class CacheAdapter(Protocol):
    """One per (model, decode-engine) pair. The front door, capture server, and bench never look past this."""
    name: str                      # e.g. "dv4-flash/omlx", "llama3.1-8b/mlx-lm"
    block_size: int                # decode engine's prefix-cache block (oMLX: 2048)

    # decode side
    def render(self, request: dict, tok) -> list[int]: ...   # exact decoder tokenization (parity is load-bearing)
    def chain_hashes(self, ids: list[int]) -> list[str]: ... # decoder's cache addressing
    def cached_prefix_tokens(self, ids) -> int: ...
    def assemble(self, cap_dir: str, boundary: int) -> object: ...  # capture -> decoder cache object
    def write_blocks(self, states, ids) -> list[str]: ...    # persist as decoder-native prefix-cache files

    # prefill side
    def hook_targets(self) -> list[str]: ...                 # module/class/method to patch + layer-id regex
    def project_and_pool(self, hidden, layer_i, weights) -> dict: ...  # torch; the sufficient-statistic step
    def capture_keys(self, layer_i) -> dict[str, tuple]: ... # file format: key -> shape/dtype per boundary
    def export_weights(self, model_dir) -> str: ...          # which decoder weights the prefiller needs
```

## Worked example: a plain-KV model (Llama-3.1-8B), to prove the interface isn't DV4-shaped

- `hook_targets`: stock `LlamaAttention.forward` — capture post-RoPE `key_states`, `value_states`.
  The sufficient statistic is *smaller* than DV4's: no pooling, no indexer, no carries.
- `project_and_pool`: identity — ship K/V rows directly. (You *could* capture the attention input
  and re-project with the decoder's k/v weights for the same "decoder's own numerics" property; for
  a plain model it is cheaper to accept the prefiller's numerics — document the trade.)
- Payload honesty: 2 × n_layers × kv_dim × dtype = **~128 KB/token** for Llama-3.1-8B FP16 vs this
  repo's 9.9 KB/token for DV4. A plain-KV adapter is *worse* per token and only makes sense on
  small models or fast links. That is the honest reason this repo starts with a hybrid-cache model.
- `capture_keys`: `k_{b} [n_kv,128]`, `v_{b} [n_kv,128]` per boundary; no prev/buf.
- `assemble`: `KVCache.state = (keys[:b].transpose(0,1), values[:b].transpose(0,1))` in mlx_lm's
  layout — about ten lines.
- Everything else (front door, share server, doctor, bench, verify) is inherited unchanged.

Estimated port effort for someone who knows both engines: **1–2 days**, against the ~2 weeks the
DV4 adapter took (design → validated numbers, as a multi-seat sprint — don't promise strangers that
pace).

## A second decode engine

The seam is `chain_hashes`/`write_blocks`: you must reproduce *that engine's* on-disk cache
addressing exactly, then verify with its own truth (the equivalent of `verify_blocks.py` against
natively-written blocks — tensor-identical except timestamps). Everything upstream of assembly is
untouched.
