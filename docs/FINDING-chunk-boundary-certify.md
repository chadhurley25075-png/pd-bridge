# FINDING — KV frames are exact only at chunk boundaries; certify every frame before serving (2026-10-12)

**A KV frame is bit-equal to a cold prefill only where the engine computed it inside a full prefill chunk.**
Rows computed in a partial chunk, in decode, or by a one-row probe are a different computation of the same
tokens: deterministic, repeatable, and not what a cold read of that prompt produces. A frame cut at an
unaligned position restores without error and answers differently. The chunk size is a property of the
engine, not of the model, so the general rule is "only aligned frames are servable" and the number comes
from the engine you run.

Seen on two engines, two chunk sizes, the same month (frames captured for a prefix bank, restored on a later
boot of the same engine and weights):

| engine | chunk | aligned frames | unaligned frames |
|---|---|---|---|
| vLLM on a GB10 pair, GLM-5.3-Flash (hybrid attention; prefill split at recurrent-state boundaries) | 4,608 tokens | 4,608 / 9,216 / 13,824 / 18,432: first-token logprob Δ = 0.0 on 12/12 probes, full greedy answers identical 12/12, same boot and across a restart | frame at 12,288: Δ +0.25 / −0.20 / −0.14 nats on three probes, identical on recapture (a different chunk shape, not noise) |
| TensorFold on four Mac Studios, full GLM-5.3 | 256 tokens | rows inside a full chunk bit-identical to cold, including across an engine rebuild | partial-chunk / decode / probe rows differ; answers drift after 3–28 tokens |

Older frames that predated the rule (3,5xx–3,7xx tokens, cut wherever the capture ended, from earlier boots)
loaded 5/5, matched the native first token 1/5, and a poisoned copy changed the answer 0/5. That is undefined
behaviour wearing a cache hit; the loader now refuses them.

One engine quirk worth knowing: vLLM's own prefix cache never hit when the prompt was *exactly* one block
(0 of 4,608 tokens reused); it hits with at least one token past the boundary.

## The recipe

1. **Cut at the boundary.** Serve a frame only up to the last position that is a multiple of the engine's
   prefill chunk (`T % chunk == 0`). If the capture ends mid-chunk, keep the aligned prefix and let the engine
   re-read the tail: the TensorFold front attaches up to `exact_upto` and prefills the remainder; the vLLM
   loader refuses anything unaligned (`bank_require_block_aligned`). Both are exact; "pad to the boundary"
   was not tested and is not recommended.
2. **Certify every frame against a cold read.** Restore the frame, ask a few probe questions, and compare
   the first-token logprob and top-5 (and, for production frames, a 64-token greedy run) to the same
   questions on the same engine and weights with the cache cold. Serve only on bit-equality. This is the
   gate that also catches the cold-engine capture in `FINDING-warm-capture.md`.
3. **Poison-probe the path.** Corrupt a copy of the frame and confirm the answer changes (11/12 and 3/3
   here). It is the only check that proves the bytes were read; a loader that silently recomputes passes
   every other test.
4. **Same weights or nothing.** A frame is exact only within identical weights and KV format. The same
   model as NVFP4 with fp8 KV on the GB10s and as a 4-bit MLX build on the Studios do not share frames
   (fp8 KV alone flipped 3/3); a translated frame can be a draft, never a served cache.
5. **Refuse loudly.** The refusals seen in testing were all worth having: "sampled sha256 differs
   (corrupted)", "uncertified: no recorded next token", "size 390,451,200 != n × 95,232" (truncated), a frame
   claiming another engine's namespace, one prefix token changed (no hit, by construction). One attempt per
   request, then native; never retry a failed load inside the scheduler (one loop of that produced 22,742
   failed loads in 90 s and hung every request).

## What this means for KV return in this repo

The DeepSeek-V4 path resumes a second turn from a saved per-layer state at a position vLLM chose: a
256-token block boundary, or `T − 1` when the whole prompt was cached (`studio/kv_return/README.md`). The CPU
proof covers both (`S` aligned and `S = T1 − 1`), tensor by tensor, against a one-shot capture. What it does
not cover is the GPU prefill-chunk effect above: whether the engine's resumed prefill produces the same
rows a cold prefill of the whole prompt would. Before trusting a resumed capture in a results table, check
it the same way: first-token logprob of the merged cache against a cold bridge of the same prompt, on the
same decoder. We have not run that check; the one live KV-return run measured a needle, not bit-equality.
