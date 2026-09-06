# POOL-VALIDATION — torch port (`pd_pool_torch.py`) vs MLX truth (`pd_export_pool_truth.py` on the Mac decode node)
Run on a DGX Spark head node · torch 2.5.1 · device cpu · truth T=23217, MLX chunk=2048, model DV4-Flash-MXFP4-MLX
Config: compress_rope_theta 160000, rope_theta 10000, rope_scaling {'beta_fast': 32, 'beta_slow': 1, 'factor': 16, 'original_max_position_embeddings': 65536, 'type': 'yarn'}, rms_norm_eps 1e-06

bf16 ulp at |x|~1 is 7.8e-3; 'rel' = max|Δ| / max|truth|. bit-exact = fraction of elements identical.

## layer_02  (ratio 4, head_dim 512, out_dim 1024, eps 1e-06, rope base 160000 yarn=yes freq_scale 4; MLX dtypes: kv bfloat16, ape float32, norm_w bfloat16)
- rope `_freqs` (unscaled) torch vs MLX: max rel diff 1.45e-07
- rope probe (random bf16 [64,512] at offset 22960 → pooled pos 5740): max|Δ| 1.562e-02  mean|Δ| 1.100e-06  rel 3.88e-03  cos 1.000004  bit-exact 99.93%  (n=32768)
- WHOLE (one call, 23217 rows → (5804, 512) in 0.15s): max|Δ| 1.562e-02  mean|Δ| 4.329e-07  rel 2.63e-03  cos 0.999917  bit-exact 99.96%  (n=2971648)
- CHUNKED 2048 (MLX schedule, 12 chunks): max|Δ| 1.562e-02  mean|Δ| 4.329e-07  rel 2.63e-03  cos 0.999917  bit-exact 99.96%  (n=2971648)  · == whole: True
- CHUNKED odd lengths (23 chunks, e.g. [1000, 3, 777, 2048, 1, 129]): == whole: True  · vs truth: max|Δ| 1.562e-02  mean|Δ| 4.329e-07  rel 2.63e-03  cos 0.999917  bit-exact 99.96%  (n=2971648)
- error split: nope features [0,448) max|Δ| 1.562e-02  mean|Δ| 5.861e-08  rel 2.63e-03  cos 0.999906  bit-exact 100.00%  (n=2600192)
                rope features [448,512) max|Δ| 1.562e-02  mean|Δ| 3.053e-06  rel 3.73e-03  cos 0.999999  bit-exact 99.65%  (n=371456)
- carry remainder rows: 1 (torch 1) — buf_kv/buf_gate bit-exact vs MLX state: True · chunked carries == whole: True
- carry prev window [4,1024]: bit-exact vs MLX prev_win_kv/gate: True · chunked == whole: True
- v3_truth_23k `pooled` (5804, 512) vs pool_truth pooled: bit-exact True · torch vs v3_truth: max|Δ| 1.562e-02  mean|Δ| 4.329e-07  rel 2.63e-03  cos 0.999917  bit-exact 99.96%  (n=2971648)
- v3_truth prev_kv_end/prev_gate_end vs torch carry: bit-exact True

## layer_03  (ratio 128, head_dim 512, out_dim 512, eps 1e-06, rope base 160000 yarn=yes freq_scale 128; MLX dtypes: kv bfloat16, ape float32, norm_w bfloat16)
- rope `_freqs` (unscaled) torch vs MLX: max rel diff 1.45e-07
- rope probe (random bf16 [64,512] at offset 14976 → pooled pos 117): max|Δ| 7.812e-03  mean|Δ| 1.393e-06  rel 2.01e-03  cos 1.000004  bit-exact 99.92%  (n=32768)
- WHOLE (one call, 23217 rows → (181, 512) in 0.06s): max|Δ| 1.953e-03  mean|Δ| 3.985e-08  rel 2.51e-04  cos 1.000004  bit-exact 99.95%  (n=92672)
- CHUNKED 2048 (MLX schedule, 12 chunks): max|Δ| 1.953e-03  mean|Δ| 3.985e-08  rel 2.51e-04  cos 1.000004  bit-exact 99.95%  (n=92672)  · == whole: True
- CHUNKED odd lengths (23 chunks, e.g. [1000, 3, 777, 2048, 1, 129]): == whole: True  · vs truth: max|Δ| 1.953e-03  mean|Δ| 3.985e-08  rel 2.51e-04  cos 1.000004  bit-exact 99.95%  (n=92672)
- error split: nope features [0,448) max|Δ| 4.883e-04  mean|Δ| 9.967e-09  rel 6.28e-05  cos 1.000002  bit-exact 99.99%  (n=81088)
                rope features [448,512) max|Δ| 1.953e-03  mean|Δ| 2.490e-07  rel 5.87e-04  cos 1.000002  bit-exact 99.63%  (n=11584)
- carry remainder rows: 49 (torch 49) — buf_kv/buf_gate bit-exact vs MLX state: True · chunked carries == whole: True
- v3_truth_23k `pooled` (181, 512) vs pool_truth pooled: bit-exact True · torch vs v3_truth: max|Δ| 1.953e-03  mean|Δ| 3.985e-08  rel 2.51e-04  cos 1.000004  bit-exact 99.95%  (n=92672)

## layer_42  (ratio 4, head_dim 512, out_dim 1024, eps 1e-06, rope base 160000 yarn=yes freq_scale 4; MLX dtypes: kv bfloat16, ape float32, norm_w bfloat16)
- rope `_freqs` (unscaled) torch vs MLX: max rel diff 1.45e-07
- rope probe (random bf16 [64,512] at offset 22960 → pooled pos 5740): max|Δ| 1.562e-02  mean|Δ| 1.100e-06  rel 3.88e-03  cos 1.000004  bit-exact 99.93%  (n=32768)
- WHOLE (one call, 23217 rows → (5804, 512) in 0.14s): max|Δ| 1.562e-02  mean|Δ| 2.275e-07  rel 1.74e-03  cos 1.000291  bit-exact 99.96%  (n=2971648)
- CHUNKED 2048 (MLX schedule, 12 chunks): max|Δ| 1.562e-02  mean|Δ| 2.275e-07  rel 1.74e-03  cos 1.000291  bit-exact 99.96%  (n=2971648)  · == whole: True
- CHUNKED odd lengths (23 chunks, e.g. [1000, 3, 777, 2048, 1, 129]): == whole: True  · vs truth: max|Δ| 1.562e-02  mean|Δ| 2.275e-07  rel 1.74e-03  cos 1.000291  bit-exact 99.96%  (n=2971648)
- error split: nope features [0,448) max|Δ| 1.562e-02  mean|Δ| 2.965e-08  rel 1.74e-03  cos 1.000267  bit-exact 100.00%  (n=2600192)
                rope features [448,512) max|Δ| 1.562e-02  mean|Δ| 1.612e-06  rel 4.63e-03  cos 1.000005  bit-exact 99.66%  (n=371456)
- carry remainder rows: 1 (torch 1) — buf_kv/buf_gate bit-exact vs MLX state: True · chunked carries == whole: True
- carry prev window [4,1024]: bit-exact vs MLX prev_win_kv/gate: True · chunked == whole: True
- v3_truth_23k `pooled` (5804, 512) vs pool_truth pooled: bit-exact True · torch vs v3_truth: max|Δ| 1.562e-02  mean|Δ| 2.275e-07  rel 1.74e-03  cos 1.000291  bit-exact 99.96%  (n=2971648)
- v3_truth prev_kv_end/prev_gate_end vs torch carry: bit-exact True

## layer_02_indexer  (ratio 4, head_dim 128, out_dim 256, eps 1e-06, rope base 160000 yarn=yes freq_scale 4; MLX dtypes: kv bfloat16, ape float32, norm_w bfloat16)
- rope `_freqs` (unscaled) torch vs MLX: max rel diff 1.45e-07
- rope probe (random bf16 [64,128] at offset 22960 → pooled pos 5740): max|Δ| 7.812e-03  mean|Δ| 3.795e-06  rel 2.16e-03  cos 1.000001  bit-exact 99.76%  (n=8192)
- WHOLE (one call, 23217 rows → (5804, 128) in 0.04s): max|Δ| 1.562e-02  mean|Δ| 2.024e-06  rel 2.48e-03  cos 0.999954  bit-exact 99.83%  (n=742912)
- CHUNKED 2048 (MLX schedule, 12 chunks): max|Δ| 1.562e-02  mean|Δ| 2.024e-06  rel 2.48e-03  cos 0.999954  bit-exact 99.83%  (n=742912)  · == whole: True
- CHUNKED odd lengths (23 chunks, e.g. [1000, 3, 777, 2048, 1, 129]): == whole: True  · vs truth: max|Δ| 1.562e-02  mean|Δ| 2.024e-06  rel 2.48e-03  cos 0.999954  bit-exact 99.83%  (n=742912)
- error split: nope features [0,64) max|Δ| 7.812e-03  mean|Δ| 7.361e-08  rel 1.34e-03  cos 0.999970  bit-exact 100.00%  (n=371456)
                rope features [64,128) max|Δ| 1.562e-02  mean|Δ| 3.974e-06  rel 2.48e-03  cos 0.999979  bit-exact 99.67%  (n=371456)
- carry remainder rows: 1 (torch 1) — buf_kv/buf_gate bit-exact vs MLX state: True · chunked carries == whole: True
- carry prev window [4,256]: bit-exact vs MLX prev_win_kv/gate: True · chunked == whole: True
- v3_truth_23k `idx_pooled` (5804, 128) vs pool_truth pooled: bit-exact True · torch vs v3_truth: max|Δ| 1.562e-02  mean|Δ| 2.024e-06  rel 2.48e-03  cos 0.999954  bit-exact 99.83%  (n=742912)
- v3_truth idx_prev_kv_end/idx_prev_gate_end vs torch carry: bit-exact True

## SWA kv window RoPE (kv = kv_norm(wkv(x)) rows [T-128,T), torch rope vs attn.rope(kv, offset=T-128))
- layer_00 (LocalAttention, rope base 10000, yarn=no, offset 23089): max|Δ| 1.562e-02  mean|Δ| 1.631e-06  rel 2.76e-03  cos 0.999999  bit-exact 99.95%  (n=65536)
    rope features only: max|Δ| 1.562e-02  mean|Δ| 1.305e-05  rel 2.76e-03  cos 1.000001  bit-exact 99.63%  (n=8192) · `_freqs` max rel diff 1.16e-07
    v3_truth kvwin_end vs pool_truth prerope window: bit-exact True
- layer_02 (SparseCompressedAttention, rope base 160000, yarn=yes, offset 23089): max|Δ| 1.562e-02  mean|Δ| 1.464e-06  rel 3.70e-03  cos 1.000000  bit-exact 99.92%  (n=65536)
    rope features only: max|Δ| 1.562e-02  mean|Δ| 1.171e-05  rel 3.70e-03  cos 1.000001  bit-exact 99.39%  (n=8192) · `_freqs` max rel diff 1.45e-07
    v3_truth kvwin_end vs pool_truth prerope window: bit-exact True

## HIDDEN-STATE PATH: cap23k attention inputs → `project_layer` (dequantized Mac weights, torch) → `pool_layer` → v3_truth_23k
weights `dv4_proj_weights.safetensors` 0.794 GB · hidden `hidden_23k.safetensors`
### layer_00 (ratio 0, LocalAttention, wkv mxfp8-g32-b8; project_layer 23217 rows in 0.28s on cpu)
- kv_pre rows [T-128,T) vs v3 `kvwin_end` (pre-RoPE): max|Δ| 0.000e+00  mean|Δ| 0.000e+00  rel 0.00e+00  cos 1.000000  bit-exact 100.00%  (n=65536)
- + torch RoPE (base 10000, yarn=no) vs MLX attn.rope(kv, 23089): max|Δ| 1.562e-02  mean|Δ| 1.631e-06  rel 2.76e-03  cos 0.999999  bit-exact 99.95%  (n=65536)

### layer_02 (ratio 4, SparseCompressedAttention, wkv mxfp8-g32-b8; project_layer 23217 rows in 1.37s on cpu)
- kv_pre rows [T-128,T) vs v3 `kvwin_end` (pre-RoPE): max|Δ| 0.000e+00  mean|Δ| 0.000e+00  rel 0.00e+00  cos 1.000001  bit-exact 100.00%  (n=65536)
- + torch RoPE (base 160000, yarn=yes) vs MLX attn.rope(kv, 23089): max|Δ| 1.562e-02  mean|Δ| 1.464e-06  rel 3.70e-03  cos 1.000000  bit-exact 99.92%  (n=65536)
- projection only: torch kv_c vs MLX project(x) kv_c: max|Δ| 0.000e+00  mean|Δ| 0.000e+00  rel 0.00e+00  cos 1.002311  bit-exact 100.00%  (n=23774208)
                   torch gate vs MLX project(x) gate: max|Δ| 0.000e+00  mean|Δ| 0.000e+00  rel 0.00e+00  cos 1.001529  bit-exact 100.00%  (n=23774208)
- pooled (5804, 512) vs v3 `pooled` (pool 0.13s): max|Δ| 1.562e-02  mean|Δ| 4.329e-07  rel 2.63e-03  cos 0.999917  bit-exact 99.96%  (n=2971648) · chunked2048 == whole: True
- carry remainder 1 rows vs v3 `buf_kv`: all-elements-identical True · max|Δ| 0.000e+00  mean|Δ| 0.000e+00  rel 0.00e+00  cos 1.000000  bit-exact 100.00%  (n=1024)
- carry prev window vs v3 `prev_kv_end`: max|Δ| 0.000e+00  mean|Δ| 0.000e+00  rel 0.00e+00  cos 1.000001  bit-exact 100.00%  (n=4096)
                     vs v3 `prev_gate_end`: max|Δ| 0.000e+00  mean|Δ| 0.000e+00  rel 0.00e+00  cos 1.000001  bit-exact 100.00%  (n=4096)
- indexer pooled (5804, 128) vs v3 `idx_pooled` (pool 0.03s): max|Δ| 4.688e-02  mean|Δ| 5.852e-06  rel 7.43e-03  cos 0.999954  bit-exact 99.78%  (n=742912) · chunked2048 == whole: True
- indexer projection only: torch vs MLX kv_c: max|Δ| 7.812e-03  mean|Δ| 9.678e-08  rel 3.98e-03  cos 1.000351  bit-exact 99.98%  (n=5943552)
- indexer carry: prev_kv_end max|Δ| 5.960e-08  mean|Δ| 5.821e-11  rel 3.72e-08  cos 1.000000  bit-exact 99.90%  (n=1024); buf bit-exact True

### layer_03 (ratio 128, CompressedAttention, wkv mxfp8-g32-b8; project_layer 23217 rows in 0.69s on cpu)
- kv_pre rows [T-128,T) vs v3 `kvwin_end` (pre-RoPE): max|Δ| 0.000e+00  mean|Δ| 0.000e+00  rel 0.00e+00  cos 0.999999  bit-exact 100.00%  (n=65536)
- projection only: torch kv_c vs MLX project(x) kv_c: max|Δ| 1.953e-03  mean|Δ| 9.238e-10  rel 8.01e-04  cos 1.001148  bit-exact 100.00%  (n=11887104)
                   torch gate vs MLX project(x) gate: max|Δ| 1.562e-02  mean|Δ| 8.278e-09  rel 4.42e-03  cos 1.002324  bit-exact 100.00%  (n=11887104)
- pooled (181, 512) vs v3 `pooled` (pool 0.03s): max|Δ| 1.953e-03  mean|Δ| 3.993e-08  rel 2.51e-04  cos 1.000004  bit-exact 99.95%  (n=92672) · chunked2048 == whole: True
- carry remainder 49 rows vs v3 `buf_kv`: all-elements-identical False · max|Δ| 1.953e-03  mean|Δ| 9.052e-08  rel 1.76e-03  cos 1.000002  bit-exact 99.96%  (n=25088)

### layer_42 (ratio 4, SparseCompressedAttention, wkv mxfp8-g32-b8; project_layer 23217 rows in 1.30s on cpu)
- kv_pre rows [T-128,T) vs v3 `kvwin_end` (pre-RoPE): max|Δ| 0.000e+00  mean|Δ| 0.000e+00  rel 0.00e+00  cos 0.999997  bit-exact 100.00%  (n=65536)
- projection only: torch kv_c vs MLX project(x) kv_c: max|Δ| 0.000e+00  mean|Δ| 0.000e+00  rel 0.00e+00  cos 1.003232  bit-exact 100.00%  (n=23774208)
                   torch gate vs MLX project(x) gate: max|Δ| 0.000e+00  mean|Δ| 0.000e+00  rel 0.00e+00  cos 1.002424  bit-exact 100.00%  (n=23774208)
- pooled (5804, 512) vs v3 `pooled` (pool 0.14s): max|Δ| 1.562e-02  mean|Δ| 2.275e-07  rel 1.74e-03  cos 1.000291  bit-exact 99.96%  (n=2971648) · chunked2048 == whole: True
- carry remainder 1 rows vs v3 `buf_kv`: all-elements-identical True · max|Δ| 0.000e+00  mean|Δ| 0.000e+00  rel 0.00e+00  cos 1.000000  bit-exact 100.00%  (n=1024)
- carry prev window vs v3 `prev_kv_end`: max|Δ| 0.000e+00  mean|Δ| 0.000e+00  rel 0.00e+00  cos 1.000001  bit-exact 100.00%  (n=4096)
                     vs v3 `prev_gate_end`: max|Δ| 0.000e+00  mean|Δ| 0.000e+00  rel 0.00e+00  cos 1.000001  bit-exact 100.00%  (n=4096)
- indexer pooled (5804, 128) vs v3 `idx_pooled` (pool 0.04s): max|Δ| 1.562e-02  mean|Δ| 2.265e-06  rel 2.82e-03  cos 0.999990  bit-exact 99.80%  (n=742912) · chunked2048 == whole: True
- indexer carry: prev_kv_end max|Δ| 3.906e-03  mean|Δ| 3.815e-06  rel 4.37e-03  cos 1.000000  bit-exact 99.80%  (n=1024); buf bit-exact True


## Summary
| tensor | max abs Δ | mean abs Δ | rel | cosine | bit-exact |
|---|---|---|---|---|---|
| layer_02 | 1.562e-02 | 4.329e-07 | 2.63e-03 | 0.999917 | 100.0% |
| layer_03 | 1.953e-03 | 3.985e-08 | 2.51e-04 | 1.000004 | 99.9% |
| layer_42 | 1.562e-02 | 2.275e-07 | 1.74e-03 | 1.000291 | 100.0% |
| layer_02_indexer | 1.562e-02 | 2.024e-06 | 2.48e-03 | 0.999954 | 99.8% |
| layer_00_kvwin | 1.562e-02 | 1.631e-06 | 2.76e-03 | 0.999999 | 100.0% |
| layer_02_kvwin | 1.562e-02 | 1.464e-06 | 3.70e-03 | 1.000000 | 99.9% |
| hidden→layer_00 kvwin_pre | 0.000e+00 | 0.000e+00 | 0.00e+00 | 1.000000 | 100.0% |
| hidden→layer_00 kvwin_rope | 1.562e-02 | 1.631e-06 | 2.76e-03 | 0.999999 | 100.0% |
| hidden→layer_02 kvwin_pre | 0.000e+00 | 0.000e+00 | 0.00e+00 | 1.000001 | 100.0% |
| hidden→layer_02 kvwin_rope | 1.562e-02 | 1.464e-06 | 3.70e-03 | 1.000000 | 99.9% |
| hidden→layer_02 pooled | 1.562e-02 | 4.329e-07 | 2.63e-03 | 0.999917 | 100.0% |
| hidden→layer_02 prev_kv | 0.000e+00 | 0.000e+00 | 0.00e+00 | 1.000001 | 100.0% |
| hidden→layer_02 idx_pooled | 4.688e-02 | 5.852e-06 | 7.43e-03 | 0.999954 | 99.8% |
| hidden→layer_03 kvwin_pre | 0.000e+00 | 0.000e+00 | 0.00e+00 | 0.999999 | 100.0% |
| hidden→layer_03 pooled | 1.953e-03 | 3.993e-08 | 2.51e-04 | 1.000004 | 99.9% |
| hidden→layer_42 kvwin_pre | 0.000e+00 | 0.000e+00 | 0.00e+00 | 0.999997 | 100.0% |
| hidden→layer_42 pooled | 1.562e-02 | 2.275e-07 | 1.74e-03 | 1.000291 | 100.0% |
| hidden→layer_42 prev_kv | 0.000e+00 | 0.000e+00 | 0.00e+00 | 1.000001 | 100.0% |
| hidden→layer_42 idx_pooled | 1.562e-02 | 2.265e-06 | 2.82e-03 | 0.999990 | 99.8% |

Worst relative max|Δ| across all tensors: 7.43e-03  → PASS (bf16-level)

## MLX semantics established by probes on the Mac decode node (`studio/pd_probe_mlx_semantics.py`, `pd_probe_mlx_sum.py`, `pd_probe_mlx_sum2.py`)
- Projections: `wkv` is QuantizedLinear **mxfp8** (group 32, 8-bit, no biases) on every layer — not MXFP4 despite the model-dir name;
  compressor / indexer `wkv`,`wgate` are plain bf16 `nn.Linear`; norm weights bf16; `ape` f32. torch f32-matmul → bf16 reproduces
  MLX's projections bit-exactly on layers 0/2/42 (layer 3: rare 1-ulp flips from accumulation order).
- `_overlap_compress_kv`: gate + ape.astype(bf16) in bf16; softmax(precise) → bf16 weights; product in bf16; sum = MLX bf16 reduction.
- `_simple_compress_kv`: weights = softmax(gate.f32 + ape) in f32 → bf16; product bf16; sum = MLX bf16 reduction.
- **MLX `sum` over bf16 does NOT accumulate in f32.** 8 rows (col_reduce_small): serial bf16 in row order (100% match).
  128 rows (col_reduce_looped BM=32): 32 strided serial-bf16 partials (rows j, j+32, j+64, j+96) combined in f32 (100% match).
  f32 accumulation matched only 50% (R=8) / 81% (R=128) of elements — that was the whole 1-ulp mismatch in the first run.
- RMSNorm (`mx.fast.rms_norm`): out = w * bf16(x * rsqrt(mean(x²)+eps)) with the normalizer in f32 → two roundings (100% match).
- RoPE (`mx.fast.rope`, traditional=True, freqs): theta = (offset//freq_scale + i) * (freq_scale / freqs_i) in f32, precise cos/sin,
  interleaved pairs, only the trailing 64 features; leading pairs have freqs=inf ⇒ untouched (100% match at pos 5740 and 23089).
- SWA-kv RoPE base: layers 0,1 (LocalAttention) = rope_theta 10000 with NO yarn; layers ≥2 = compress_rope_theta 160000 WITH yarn
  (design doc v3 §1 says 'yarn, base 10000' for the window — wrong for every layer: 0/1 have no yarn, ≥2 use base 160000).
- Indexer compressor = Compressor(config, 4, 128): rope dims 64 on head_dim 128, base 160000, yarn, freq_scale 4 (design §3 CHECK confirmed).
- PoolingCache prompt mode: pool_base = offset − remainder; remainder rows carried raw; prev window (ratio 4) = raw last window (no ape);
  first window of a sequence pools with zero lane-A / −inf gate. Chunked == whole is bit-exact in the port for any chunk lengths.
