# BENCHMARK PROTOCOL — heterogeneous prefill/decode, DeepSeek-V4-Flash (stock builds, neutral inputs)   — 2026-09-06

Purpose: the numbers we publish. No application prompt, no hooks, no memory fabric, no identity — a stranger with a Spark pair and a Mac Studio
must be able to reproduce every line. "Best case" = the fully optimized pipeline (v3 pooled capture, single resident build, no disk staging),
but every figure is MEASURED on this hardware, never derived, and the derived floors are labeled as such.

## Hardware / software under test
- Prefill: 2× NVIDIA DGX Spark (GB10, 128 GB each), TP2 over 200G RoCE. vLLM 0.21.1rc1 + DeepSeek-V4 plugin (sparkrun image
  `aidendle94/sparkrun-vllm-ds4-gb10:production-ready`), official `deepseek-ai/DeepSeek-V4-Flash` FP8 (e4m3, block 128), `--enforce-eager`,
  `--kv-cache-dtype fp8`, `--max-num-seqs 1`, `--gpu-memory-utilization 0.75`, `--max-model-len 262144`.
- Decode: 1× Mac Studio M3 Ultra 256 GB, oMLX 0.6.4, `DV4-Flash-MXFP4-MLX` (stock MXFP4), SSD prefix cache on, ONE model resident.
- Link: plain 10GbE (measured pull 1.15 GB/s). No RDMA, no Thunderbolt.

## Inputs
- `bench_cold.py --chars C --seed S`: deterministic synthetic document from CPython's stdlib sources (seeded shuffle) with an embedded marker
  and a one-line question. Each seed is a genuinely cold prompt (never seen by either engine) — no cache-clearing, no force switches.
- Sizes: ~23K, ~50K, ~100K tokens (chars ≈ 3.2× tokens for code; report the exact token count oMLX logs).
- Seeds: 3 per size per configuration. Report median and the individual runs.

## Configurations (same prompt, same seeds)
| id | path | what it measures |
|---|---|---|
| N-cold | Mac alone (:8011), cold | native cold TTFT |
| N-warm | Mac alone, same prompt again | native prefix-hit TTFT (floor for the bridged first token) |
| B-v3 | bridge (:8012, PD_MODE=pooled) | Spark prefill + pooled capture + assemble + Mac prefix-hit decode |
| S-only | Spark pair (:8000) with the hook OFF, same ids, max_tokens=1 | pure engine prefill time (the "compute floor" — MEASURED, not derived) |
| S-hook | Spark pair with the v3 hook ON, max_tokens=1 | hook overhead |

## Metrics (all from the client, streaming)
TTFT (first content token), total time to a 64-token answer, decode tok/s (tokens/(total−TTFT)), marker found (yes/no) — plus the front
door's own breakdown line (engine, DONE seen, pulled GB, assembled) and oMLX's server-log TTFB for the same request.

## Quality gate (separate from speed)
`pd_quality_eval.py` on `eval_doc.py` (neutral code document) with `eval_questions.json` gold answers; native scored 5/5; bridged leg must
match. Also: marker-found must be 3/3 at every size on the bridged path.

## What we will NOT publish
Derived compute floors as if measured; the (operator) seat prompt as a benchmark input; any number from a run with the build swap present;
warm-turn numbers as if they were bridged (warm turns bypass the bridge by design).
