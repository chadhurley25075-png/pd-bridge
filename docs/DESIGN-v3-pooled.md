# DESIGN v3 — POOLED CAPTURE (ship finished caches, not hidden states)   

## Why
v2 ships 43 × [T,4096] bf16 hidden states (36.1 GB at T=102,595) so the Mac can rebuild caches. The caches themselves are 1.01 GB
(measured from our written oMLX blocks: 9.9 KB/token). Capture write on the Spark (53 s) + pull (~30 s) dominate a 19 s prefill.
v3 computes the pooled caches ON THE SPARK in torch (same math as MLX) and ships ~1 GB. Target cold TTFT at 102K ≈ 30 s (native 273 s).

## The exact math (from oMLX omlx/patches/deepseek_v4/deepseek_v4_model.py + cache_extras.py — copies in scratch; vLLM plugin in $PD_HOME/spark/plugin/dsv4_plugin/)
Per layer l with compress_ratio r_l ∈ {0,4,128} (config: [0,0,4,128,4,128,…] → layers 0,1 SWA-only; even≥2 → 4; odd≥3 → 128):
1. SWA kv (every layer):  kv = kv_norm(wkv(x)) ∈ [T,512] bf16.  vLLM: `kv` after `fused_q_kv_rmsnorm` in attention_impl (pre-RoPE).
   MLX RotatingKVCache(max 128) stores the LAST 128 rows, RoPE'd (rope on last 64 dims, interleaved/traditional, yarn, base 10000).
   → ship rows [b-128, b) at every 2048 boundary b and at T (pre-RoPE; Mac applies attn.rope(kv, offset=b-128) — exact).
2. Compressor (r_l ∈ {4,128}): project: kv_c = wkv_c(x), gate = wgate_c(x), each [T, out_dim], out_dim = 512·(2 if r=4 else 1).
   vLLM: `kv_score = fused_wkv_wgate(x)` = [kv_c | gate] (same order), input to DeepseekCompressor.forward(kv_score, positions, rotary).
   Pool per window of r rows:  r=128: w = softmax(gate.f32 + ape, over the r axis).bf16; pooled = Σ kv_c·w   (→ [T//128, 512])
                                r=4 (overlap): gate += ape; split kv_c/gate into lanes A|B (last-dim halves, 512 each);
                                   lane-A of window i is taken from window i-1 (first window: kv_a=0, gate_a=-inf); concat along the
                                   row axis → 8 rows × 512; w = softmax(gate, rows, precise); pooled = Σ kv·w  (→ [T//4, 512]).
   then pooled = RMSNorm(512, eps)(pooled) ; then RoPE with compress_rope_theta=160000, yarn, freq_scale=r (offset=pool_base//r).
   ape: Parameter [r, out_dim] f32 (vLLM: compressor.ape ; MLX: same values).
   Cache state at a 2048 boundary b (2048 % 4 == 2048 % 128 == 0 → remainder 0): pooled[: b//r] ; prev window (r=4 only) = raw
   kv_c/gate rows [b-4, b) each [1,1,4,1024] (PoolingCacheDelta state_3/state_4).  At T: remainder rows T % r of raw kv_c/gate (buf).
3. Indexer (only r=4 layers): same Compressor with head_dim 128, ratio 4, out_dim 256; vLLM input `indexer_kv_score` [T,512] =
   [kv | gate]; own ape/norm; RoPE dims = index_head_dim rope? → CHECK: MLX Indexer.compressor = Compressor(config, 4, 128): rope dims
   = qk_rope_head_dim (64) on last 64 of 128, compress_rope_theta, freq_scale 4.  pooled → [T//4, 128].
4. TP2: fused_wkv_wgate has disable_tp=True → replicated; capture on rank 0 only (v2 hook already gates on rank).

## Payload at T=102,595 (43 layers): pooled 21×26.3 MB + indexer 21×6.6 MB + r128 20×0.8 MB + kv windows 43×51×131 KB ≈ 1.0 GB. ✓ = block total.

## Files / owners
- `$PD_HOME/spark/pd_pool_torch.py` (AGENT A): pure-torch `overlap_compress`, `simple_compress`, `rmsnorm`, `DSv4Rope(dims, base, yarn_cfg, freq_scale)`,
  `pool_layer(kv_score, ratio, head_dim, ape, norm_w, eps, rope, start_pos, carry) -> (pooled, new_carry)` supporting CHUNKED input
  (chunks are arbitrary lengths; carry = remainder rows + prev window). Validated against MLX truth exported from the Mac decode node (cap23k) — report
  max|Δ| and cosine per layer for 4 layers (one r=4, one r=128, one indexer, plus layer 42).
- `$PD_HOME/studio/pd_export_pool_truth.py` (AGENT A, runs ON the Mac decode node via ssh $DECODER_SSH, python $OMLX_PYTHON, model
  $PD_MODEL, capture $PD_HOME/cap23k/attn_inputs.safetensors [layer_XX keys? check pd_capture_mlx.py]):
  for chosen layers: x → attn.compressor.project(x) (kv_c, gate) saved as .npy bf16→f32, plus pooled truth via the real Compressor + PoolingCache
  in 2048-token chunks (matches oMLX prefill_step_size), plus ape/norm weights, kv=kv_norm(wkv(x)) rows [T-128,T). Save to $PD_HOME/pool_truth/.
- `$PD_HOME/spark/capture_sitecustomize_v3.py` (AGENT B): from v2, replace hidden-state capture with: hook DeepseekCompressor.forward
  (both main and indexer instances — identify by head_dim 512/128) to grab kv_score (+positions) per chunk; hook attention_impl to grab kv
  (post fused_q_kv_rmsnorm) per chunk; per layer keep: pooled accumulators, carries, rolling last-128 kv rows, boundary snapshots at every
  2048 tokens; on idle (2 s) write ONE safetensors per layer: pooled[P,512] bf16, idx_pooled[P,128] bf16 (r=4), prev_kv/prev_gate [4,out_dim],
  idx_prev_*, buf_* remainder rows, kvwin_{b} [128,512] for each boundary b and final, + manifest.json {T, layers, ratios, boundaries}.
  Use pd_pool_torch. GPU work only on rank 0; never block the forward (copy to a side stream/CPU asynchronously like v2). Guard CUDA-graph
  capture as v2 does (torch.cuda.is_current_stream_capturing()). Keep `--enforce-eager` launch.
- `$PD_HOME/studio/pd_assemble_blocks.py` (operator): build MLX cache objects from the per-layer files (RotatingKVCache state+meta,
  PoolingCache state (buf_kv, buf_gate, pooled, prev_win_kv, prev_win_gate) via .state setter) at each 2048 boundary and call
  omlx_block_writer.BlockWriter.snapshot/finalize → blocks; pd_front v3 switches to this path (env PD_MODE=pooled).
## Validation ladder
1. torch port vs MLX truth (cap23k): max|Δ| ≤ bf16 noise (~1e-2 relative) — AGENT A reports numbers.
2. Blocks from pooled path vs native oMLX blocks at 23K: compare per-tensor max|Δ| (will NOT be bit-identical: Spark FP8 weights vs Mac MXFP4).
3. Judged quality eval (pd_quality_eval.py) bridged vs native 5/5. 4. 102K cold TTFT.

## v3 CAPTURE FILE FORMAT (binding — the assembler reads exactly this)   [added 09:40]
Directory PD_CAPTURE_DIR/<stamp>/ with `manifest.json` = {"T":int,"num_layers":43,"ratios":[...],"boundaries":[2048,4096,...,<=T],"end":T}
and `layer_XX.safetensors` (XX = 00..42) with keys (all bf16 unless noted; NO batch dims):
  pooled        [P,512]      P = T//ratio   (ratio 4 or 128 layers only; SWA-only layers 0,1 have no pooling keys)
  idx_pooled    [Pi,128]     Pi = T//4      (ratio-4 layers only)
  kvwin_{b}     [128,512]    PRE-RoPE kv = kv_norm(wkv(x)) rows [b-128, b), for every boundary b in manifest AND b=end (key kvwin_end)
  prev_kv_{b}, prev_gate_{b}         [4,1024]  raw compressor kv_c / gate rows [b-4, b) (ratio-4 layers; also *_end)
  idx_prev_kv_{b}, idx_prev_gate_{b} [4,256]   same for the indexer compressor (ratio-4 layers; also *_end)
  buf_kv, buf_gate       [rem, out_dim]  remainder rows T % ratio at end (ratio 4: out_dim 1024; ratio 128: 512); omitted when rem==0
  idx_buf_kv, idx_buf_gate [rem, 256]
The Mac assembler builds, at each boundary b: RotatingKVCache.state = (rope(kvwin_b, offset=b-128)[None,None], zeros[1,1,128,0]),
meta (0,128,b,128); PoolingCache.state = (None, None, pooled[:b//r][None], prev_kv_b[None,None], prev_gate_b[None,None]) (prev None for ratio 128),
then BlockWriter.snapshot(cache, b); finalize(ids). Blocks produced from MLX-computed inputs must be byte-identical to native oMLX blocks.

## DECISION 09:55 (from an outside review, adopted): project with the MAC's weights, on the Spark
The pooled caches must be what the Mac would have computed. So the Spark hook keeps v2's hook point (attention_impl INPUT `hidden_states`, bf16)
and runs the cache projections itself in torch on the GPU with a DEQUANTIZED COPY OF THE STUDIO'S MXFP4 ATTENTION-PROJECTION WEIGHTS
(~0.8 GB bf16 — only wkv/kv_norm/compressor/indexer-compressor, not q/o), then pools with pd_pool_torch. vLLM's own kv_score / fp8 caches
are NOT used. Only ~1 GB of pooled results ever leaves the GPU. Drop v2's `cur.wait_event(ev)` (it serialized D2H onto the compute stream).
Weight file (AGENT A exports from the MLX model): `$PD_HOME/spark/dv4_proj_weights.safetensors` + `dv4_proj_weights.json` (eps, ratios):
  layer_{i}.wkv.weight [512,4096] bf16 · layer_{i}.kv_norm.weight [512] · (ratio>0) layer_{i}.comp.wkv.weight [out_dim,4096] ·
  layer_{i}.comp.wgate.weight [out_dim,4096] · layer_{i}.comp.ape [ratio,out_dim] f32 · layer_{i}.comp.norm.weight [512] ·
  (ratio==4) layer_{i}.idx.wkv.weight [256,4096] · layer_{i}.idx.wgate.weight [256,4096] · layer_{i}.idx.ape [4,256] f32 · layer_{i}.idx.norm.weight [128]
  (torch Linear layout [out,in], same as MLX nn.Linear; dequantize QuantizedLinear with mx.dequantize).
pd_pool_torch gains `project_layer(hidden[L,4096], W_i) -> (kv_pre[L,512], kv_score[L,2*out_dim], idx_kv_score[L,512])` so the hook and
the validator share one code path. Validation: hidden states (cap23k) → torch project+pool → compare with the MLX-truth v3 capture: expect bf16 noise.
