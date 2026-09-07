# FINDING — five stale limits that break a P/D bridge after you raise the window

*Written 2026-09-07, after a full day lost to this class of bug. Every one of these was a number that
was **correct for the previous window** and stayed put after the window grew. None of them errored.
Every one presented as a different, more interesting problem.*

If you take one thing from this file: when you raise a context window, **ask what other number was
sized against the old one.** Five separate places, three layers, and the system reported healthy the
entire time.

---

## 1. The client's declared context window — silently clamps output to ONE token

The worst one, because it looks like the model being terse.

Our agent harness computes an output budget as roughly `contextWindow - promptTokens`. The harness's
model registry still declared **227,328** while the decoder served **2,097,152**. A 293,000-token
prompt made that subtraction negative, and the harness floored it at **1**.

Every deep-context turn came back as exactly one token with `finish_reason=length`. It reads as
truncation. It is not truncation — the model was never permitted to speak.

**The signature, and it is unambiguous:**

```
Chat completion: 1 tokens, prompt: 294,981, finish_reason=length,
                 max_tokens=1, request_max_tokens=1
```

`finish_reason=length` with `max_tokens=1` means a client-side budget went negative. It is never the
server. Fix the client's declared window; the server was fine all along.

**Check yours:** the serving log records `request_max_tokens` per turn. Grep it. If your engine does
not log the requested cap, add it — this bug is invisible without that field.

## 2. Client per-model output caps

Separate from the above and easy to miss once you fix #1. Ours ranged from 32,768 down to **4,096**
on some seats — a cap set when those bodies had small windows. Raise them with the window.

## 3. Decoder sampling caps

The decode server carried `sampling.max_context_window = 32768` and `sampling.max_tokens = 32768`
against a 2M model. Whether they are enforced depends on your engine, but leaving a 32K context cap
configured on a 2M server is a trap waiting for whoever reads the config next.

## 4. The prefix-cache size cap — the silent performance killer

Our decoder's SSD prefix cache was set to `auto`, which resolved to **92.64 GB**. The cache directory
held **109 GB**. Over its own ceiling, it evicted blocks while trying to read them.

Cost, measured on the same workload before and after:

| | block assembly |
|---|---|
| cache 109 GB against a 92.64 GB cap | **150.6 s** |
| cache cleared, cap raised well above need | **16.6 s** |

**Nine times faster.** Nothing errored. It just got slow, and slow reads as "long context is
expensive" rather than "the cache is thrashing."

Size the cap from the window: cache bytes ≈ `window × KV-bytes-per-token`. Ours is ~9.9 KB/token, so
a 2M window wants ~20 GB per distinct cached prompt, and you want room for several.

## 5. Engine flags your build does not accept

We passed `--rope-scaling` to set a YaRN factor. **Our vLLM build rejects that flag outright**
(`unrecognized arguments`). Every launch that included it died instantly — and our supervisor
restarted the pair *without* it, so the prefill engines came up serving the large window using the
model's **original** rope factor while the decoder ran the new one.

A rope mismatch between prefill and decode does not error. It produces a confidently wrong answer.

**If your engine rejects the flag, set rope in the model's own `config.json`** on every prefill node,
and remove the flag so a failed launch can never be silently "fixed" by a restart.

---

## The two guards that would have caught all of this

**Print the agreeing values from both sides at bring-up.** We now emit the rope factor read from the
prefill node's model config *and* the decoder's, side by side, plus the container's real launch
flags. A mismatch is now one line of output instead of six hours.

```
spark rope factor:   32   2097152
mac rope factor:     32   2097152
--max-model-len 2097152
--max-num-batched-tokens 2048
--gpu-memory-utilization 0.82
```

**Persist the FULL parameter set, not just the window.** Our supervisor restarted the pair from a
file containing only the window, so every automatic restart quietly dropped rope factor, memory
utilization and chunk size. One file, every parameter, sourced by every restart path. Add a
single-flight lock while you are there — six concurrent restarts fought over two containers because
the supervisor did not know a restart was already running.

---

## Related: your acceptance test must distinguish "wrong" from "slow"

Our depth-retrieval gate scored a **client timeout** as a retrieval failure and reverted a *working*
configuration on the strength of it. Two cold prefills outran a flat 90-minute client timeout; the
harness read that as "the model does not hold past its ceiling."

Timeouts are **inconclusive**, not failures. Ours now exits 0 for pass, 1 for a real wrong answer,
and 2 for inconclusive — and only exit 1 reverts anything.

The same disease as everything above: **a component substituting a default and reporting success.**
