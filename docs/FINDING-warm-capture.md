# FINDING — capture a KV frame only on a WARM engine (2026-10-12)

**The first long prefill after an engine boot can compute differently from every later prefill of the
same prompt.** A KV frame captured from that first prefill loads without error and fails certification.
A frame captured after the engine has served a few prefills certifies. Loading a certified frame *into* a
cold engine is exact.

This came out of a frame bank built for a different model than the one this repo is about: GLM-5.3-Flash
(hybrid attention: a few MLA layers, the rest recurrent) served by vLLM on a GB10 pair, with prefix frames
captured from the engine and restored into a later boot. It is written up here because the mechanism is
engine-side, not model-side, and the repo's KV-return path (`PD_KV_RETURN=1`, merged 2026-10-07) saves
engine state the same way.

## What was seen

Two independent sightings, both caught by certification (restore the frame, ask probe questions, compare
the first-token logprob and top-5 to a cold read on the same engine and weights):

1. **Test lane, 13 captures.** One of the 13 — the first capture after a fresh boot — differed from a
   recapture of the same prompt in every layer past layer 0. First-token deltas against cold: 0.08–0.51
   nats. The unaligned-frame effect (see `FINDING-chunk-boundary-certify.md`) was ruled out separately.
2. **Production, same day.** The first capture after the 12:23 boot loaded, then failed 3/3 probes: first-token
   logprob, top-5, 64 greedy tokens and the chat endpoint all differed from native. Byte diff against the
   recapture: every MLA row from row 0, layer 3 onward. The recapture, taken after two cold reads of the
   exact prompt shape, certified 3/3 (64-token reply: 16.7–17.1 s native, 2.8–3.2 s from the frame).

The other direction is fine: four frames captured in one boot and restored into the next boot certified
4/4 bit-exact, both before and after the new boot warmed up, and the first real restore after a restart
(36,864 tokens) loaded with 0 failures.

**The cause is not proven.** We did not find which kernel or allocation path differs on the first call;
only that the frame it produces is not the frame every later prefill produces.

## The rule

- Capture on a warm engine. The capture tool now runs two cold reads of the exact prompt shape before it
  captures; the cost is two prefills.
- Certify every frame against a cold read before it is served (recipe in `FINDING-chunk-boundary-certify.md`).
  Certification is what caught both sightings; without it both frames would have been served.
- Treat a frame captured in the first minutes after a boot as suspect until it certifies.

## What this means for KV return in this repo

`PD_KV_RETURN=1` saves a small per-layer state at the end of each capture and resumes the next turn from it.
The saved state is a product of the prefill that produced it. If the prefill pair was just restarted, the
first turn's state is the first long prefill after boot — the case above. Nothing in the repo today checks
for it: the CPU proof (`spark/test_kv_return.py`, 115 checks) is engine-free, and the one live measurement
(2026-09-18, `studio/kv_return/README.md`) was a needle test, which checks retrieval, not bit-equality.

We have **not** tested whether the DeepSeek-V4 pooled capture shows the same first-prefill effect. The
warning is precautionary: return to a warm pair, and if you build a bank of saved states, certify them the
way the frames above were certified before any state is reused across a restart.
