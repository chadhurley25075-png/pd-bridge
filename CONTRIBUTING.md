# Contributing

Two kinds of contributions matter here: **making this stack faster/honest-er**, and **porting the
shape to another model or decode engine** (see [docs/PORTING.md](docs/PORTING.md) for the second).

## Ground rules (learned the hard way — every one of these cost real hours)

1. **A benchmark number without a verdict is not a number.** Every bridged run must carry the
   front door's `X-PD-Bridge` verdict (`complete` / `partial B/T` / declined-with-reason), and
   `bench_cold.py` records it. We once lost an entire benchmark round to native fallbacks wearing
   bridge labels — `docs/FINDING-bench4-cold-fallback.md` is the autopsy. Never let a client-side
   timer be the only evidence of which path ran.
2. **Follow [bench/BENCHMARK-PROTOCOL.md](bench/BENCHMARK-PROTOCOL.md)**: neutral synthetic docs,
   fresh seeds per leg, warm engines on both legs, a native baseline on the same hardware, and the
   hook-off control (`PD_HOOK=off`) when you touch anything on the prefill side.
3. **Compare tensors, never file hashes.** Bridge-written blocks carry a `created_at` stamp, so a
   byte-identical *file* is impossible even when the contents are exact. `studio/verify_blocks.py`
   is the acceptance test for the project's central claim.
4. **`make scrub-check` must pass.** No personal paths, LAN addresses, or credentials in tracked
   files. Config lives in `hetero.env` (gitignored); `config.example.env` shows the shape.
5. **Restart order matters** (see README *Gotchas*): the prefill pair takes ~12 minutes; the oMLX
   server must fully shut down before a new one starts. Don't restart anything while someone else's
   benchmark is in flight.

## The test tiers

| tier | where | what |
|---|---|---|
| `make test` | any CPU box (CI) | hook pooling selftest (synthetic weights, chunked == one-shot, 52/52), `scrub-check` |
| `make test-mac` | the decode Mac, oMLX venv | synthetic block writer vs a REAL oMLX reference block (layout + hash chain), streaming-store paths |
| `make test-spark` | inside the running vllm_pd container | pooling selftest against the real exported weights (`--smoke`) |
| `make bench-quick` | the full fabric | cold 20K/80K/100K bridged + native, verdicts recorded |

The judged quality eval on the bridged leg (the protocol's Tier D) is still open — native scores
5/5; the bridged leg has passed needle retrieval on every run but has not been scored. Landing that
eval is contribution #5 in the README list and the last thing standing between "looks right" and
"proven".

## Where the sharp edges are

- The hook monkey-patches private internals of both engines. Expect breakage when either moves;
  pin versions and re-run the validation ladder (`spark/pd_pool_validate.py` against
  `studio/pd_export_pool_truth.py` output) after any engine upgrade.
- The capture format is binding: `docs/DESIGN-v3-pooled.md` defines it. If you change a key, you
  change the manifest `version` and the front's validator together.
- Idle-flush + chunked prefill is a dangerous pair; both mid-flush guards in the hook
  (CUDA-event-query, chunk-alignment) are load-bearing. If you touch `_watch`, read the FINDING first.

## Ports we want first

1. **The cheap pair**: DeepSeek-V2-Lite (or any small MLA-latent model) on a used gaming GPU + a 16–32 GB
   Apple Silicon Mac. Same recipe, your numbers. This is the port that makes the idea useful to people
   without a rack, and we will feature it here.
2. A second decode engine (vLLM CPU/disk KV connector; llama.cpp prompt-cache — harder, see PORTING).
3. Further front-door concurrency (the MLX cache-assembly step still serializes).

If you are an AI agent working on this: read `AGENTS.md` first. If you are a kid with an old iMac and a
beat gaming PC: that is exactly who this is for. Say so in the issue.
