# FINDING — FLUSH_NOW is eaten by watcher threads in processes that never own a request

Agent B · 2026-09-06 · read-only investigation (lab files + live logs on the prefill head / the decoder; nothing touched).
Files: hook `~/pd_lab/spark/capture_sitecustomize_v3.py` (md5 66874bcf…, == deployed), `spark/pd_share.py`, `studio/pd_front.py`.

## Root cause (one paragraph)

`_Capture.__init__` starts a watcher thread unconditionally (lines 389-391), and `_Capture` is instantiated in **every
process that imports the attention module**, not just the capturing worker: `_patch_attention` ends with
`threading.Thread(target=_cap().weights.load_cpu, …)` (line 782-783), and `_cap()` builds the singleton. The live log proves
three such processes exist — `projection weights loaded to host` is printed by **APIServer pid=1, EngineCore pid=89 and
Worker_TP0 pid=113** (that line is only reachable via `_cap().weights.load_cpu`). All three watchers poll
`/pd_capture/FLUSH_NOW` every 0.1 s, and the tick body (lines 625-632) **unlinks the file first and checks for an open request
second**:

```python
625    demanded = os.path.exists(flush_now)
626    if demanded:
627        try: os.unlink(flush_now)          # <- consumed by WHOEVER ticks first
628        except OSError: pass
629    with self.lock:
630        r = self.req
631        if r is None:
632            continue                        # <- pid 1 / pid 89 are ALWAYS here: signal gone, nothing done
```

Only pid 113 ever has `self.req`. Three independent 0.1 s pollers race for one file: the owner wins ~1/3 of the time. When it
loses, `demanded` (a per-tick local, never latched) is False on every later tick, the code falls to the `elif self.idle_s > 0`
branch (line 651-652) and the capture closes at `r.last_t + 15 s`. That is the bimodal "0.5–1.5 s or 13–17 s, never in between"
signature, independent of prompt size. Observed today: 3 of 8 runs consumed the signal (37.5 %) — consistent with 1/3.

A second, smaller defect in the same lines compounds it inside the owner: because the file is unlinked *before* `settled` is
evaluated (line 648), a signal that lands while the last chunk is still draining (`q` non-empty, `last_ev` pending, or
`r.calls % 43 != 0` mid-burst) is also thrown away — `fire` is False on that tick and the next tick has no file. Same
fall-through to idle. (Suspect (a) in the brief — real, but the log timelines below show it is NOT what fired today.)

## Evidence — the two runs that live in the current container's log (armed 20:47:03Z = 15:47 CDT)

All hook timestamps UTC (`docker logs --timestamps`), front timestamps CDT (+5 h). Stamp = `first_t` (ms precision).

**Run 16:13 ceiling, 241,416 tokens** (front `t_engine 137.59 · t_flush_signal 137.74 · t_done_seen 155.01`):
```
hook   stamp 20260906-211026-940  -> first_t = 21:10:26.940Z ; front stamp_seen 0.21 => t0 = 21:10:26.73Z
hook   span=135.474s              -> last_t  = 21:12:42.414Z   (last ingest: worker DRAINED here)
front  engine done +137.59        -> 21:12:44.32Z
front  /_flush 200 OK  +137.74    -> 21:12:44.47Z   FLUSH_NOW written, 2.05 s AFTER the worker drained
hook   2026-09-06T21:13:01.770Z finished 20260906-211026-940: … calls=1290 … write=4.263s … (idle)
       21:13:01.77 - 4.263 (write) = 21:12:57.5Z = last_t + 15.0 s  -> the IDLE timer fired, to the tick
```
On the signal tick the owner was fully settled: `q` empty for 2 s, `calls=1290 = 30×43` aligned, forward finished (engine
had returned, so `last_ev` complete). Had pid 113 seen the file, `fire = settled = True`. It never saw it — another process's
watcher had already unlinked it. This rules out the "worker not drained" story for this run and leaves only the
competing-consumer race. (Path mismatch is also ruled out: `pd_share.py` pid 3892 serves `~/pd_capture`,
the container binds that same dir at `/pd_capture`, and the same server lists the captures the front fetched — so the touched
file and the polled file are one inode.)

**Run 16:00 hetero-pi, 17,524 tokens** (front `t_engine 16.05 · t_flush_signal 16.09 · t_done_seen 30.83`):
```
hook   stamp 20260906-205937-568  -> first_t 20:59:37.568Z ; span 15.424 -> last_t 20:59:52.992Z
front  signal +16.09              -> ~20:59:53.44Z   (0.45 s after the last ingest; calls=129 = 3×43)
hook   2026-09-06T21:00:08.209Z finished … (idle)   = last_t + 15.0 + 0.139 write  -> idle fired again
```
Same shape. Also visible in this container: the startup dummy runs (`non-contiguous positions 0..0 n=8192`, 20:59:13/17) each
became a junk capture and the second one's idle flush at ~20:59:33 put pid 113's watcher into `time.sleep(15.0)` (line 657)
until ~20:59:48 — a 15 s blind window that a chained request can finish inside (the eval-pass2 7 s prefill with a 13.2 s lag
is the likely victim of this secondary defect; its log is in the previous container and is gone).

**Why the manifests can't tell us more:** the sentinel is a bare `object()` and the worker always calls `_finish("idle")`
(line 571) — a FLUSH_NOW-triggered flush is ALSO labelled `flush=idle`. Every manifest today says `idle`, including the three
fast runs. That is the observability gap the brief asks to close; the patch labels them `flush_now` vs `idle` and logs the
decision with what the tick saw.

## Suspects from the brief, ruled in / out

| | suspect | verdict | evidence |
|---|---|---|---|
| (a) | flag consumed before the last chunk is enqueued/aligned, never re-checked | **real, secondary** — same lines 625-628, but not what fired today | 241K: drained 2.05 s before the signal; 17.5K: 0.45 s before |
| (b) | stale flag / stale mtime from a previous request | out — the file is unlinked on sight; nothing checks mtime | code; and a stale flag would cause an EARLY flush, not a late one |
| (c) | aligned / drained precondition false at signal, no re-check | out for today's runs (calls 1290 = 30×43, 129 = 3×43, queue idle for seconds) | timelines above |
| (d) | host/container path mismatch | out | pd_share ROOT `~/pd_capture` (ps), bind `-v …pd_capture:/pd_capture` (docker inspect), same server lists the captures |
| (e) | poll interval / sleep swallows the signal | **secondary** — `time.sleep(self.idle_s)` after ANY fire = 15 s blind window (line 657) | dummy-run flush at ~20:59:33 → asleep until ~20:59:48 |
| **new** | **competing watcher threads in APIServer + EngineCore unlink the flag** | **ROOT CAUSE** | `weights loaded` from pids 1, 89, 113; `_patch_attention` → `_cap()`; `__init__` starts `_watch` unconditionally; unlink precedes `r is None` |

If two causes are weighed: the evidence favours the multi-process race over (a) because in both logged runs the owner was
provably settled when the signal arrived, so an owner-only race could not have produced a miss; and the ~1/3 success rate is
exactly what three equal pollers predict.

## The fix (patch.diff — hook only; pd_share.py and pd_front.py unchanged)

1. **Threads start lazily on the first `record()`** (`_start_threads`). Only the process that actually records — the rank-0
   worker — ever runs a watcher. APIServer/EngineCore keep the weight preload but get no threads.
2. **The flag is never consumed by a process with no open request** (`_flush_decision`: `if r is None: return`), and the owner
   consumes it **only when it fires on it** or when it is **provably stale** (`mtime < r.first_t`: written before this request
   began, i.e. left over from a request that idle/new-request-flushed). A signal that arrives while the worker is still draining
   stays on disk and is re-read next tick — nothing latched, nothing lost. This also fixes (a) and gives (b) a real definition.
3. **Post-fire sleep `idle_s` → `min(idle_s, 1.0)`**: `_finish` nulls `self.req` under the lock so a duplicate sentinel is a
   no-op; the 15 s blind window (e) goes away.
4. **Observability**: the sentinel carries its reason; manifests now say `flush_reason: flush_now | idle | new-request` plus
   `flush_why`; the watcher logs ONE line at fire time:
   `flush decision: fire=1 reason=flush_now — signal=0.07s old q_empty=True gpu_busy=False aligned=True calls=1290 nl=43 idle_for=2.1s …`
   (and `cleared stale FLUSH_NOW — …` in the rare leftover case).

Verified: `python3 -m py_compile` on the patched copy; `patch -p1` applies cleanly to a pristine copy and reproduces it
byte-for-byte; `test_flush_decision.py` (in this folder, no torch/GPU) passes 8 scenarios — non-owner leaves the flag, stale flag
cleared, signal-while-draining survives and fires on the next tick, mid-burst survives, idle backstop still fires, worker passes
the reason through — and shows one tick of the ORIGINAL `_watch` in a request-less process deleting the flag.

Known limit (pre-existing, unchanged): any FLUSH_NOW with `mtime >= first_t` is trusted as "this request is over". A second
client hitting the Spark directly mid-request could still trigger an inter-chunk flush; `--max-num-seqs 1` + the front's LOCK
exclude that today.
