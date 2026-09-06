# FINDING — bench4's cold "bridge" numbers are NATIVE FALLBACKS. Do not publish them as bridged.
Written 2026-09-06 from the ACTUAL logs: the decoder's pd_front.out, the Spark hook lines in the
chain log, and omlx.server's own completion lines in the bench results. Not from any summary.

## What bench4 actually measured
| run | label | ttft | what really happened |
|---|---|---|---|
| seed 601 20K | "bridge cold" | 55.2s | bridge FAILED 10:48:43 → oMLX native prefill 18,424 tok in 46.27s |
| seed 602 20K | native cold | 40.5s | native ✓ (honest) |
| seed 601 20K | bridge warm | 11.02s | genuine — skip-bridge warm prefix hit ✓ (honest, the "eleven") |
| seed 603 80K | "bridge cold" | 219.7s | bridge FAILED 10:50:39 → native 82,703 tok in 202.23s |
| seed 605 100K | "bridge cold" | 327.1s | bridge FAILED 10:54:59 → native 105,878 tok in 269.67s |
| seed 606 100K | native cold | 256.8s | native ✓ (honest) |
| hetero pass1 | "cold bridge" | 256.3s | bridge FAILED 11:04:28 → native 214.58s |
| hetero pass2 | "warm" | 54.9s | GENUINE FULL COLD BRIDGE: 78,891 tok, engine 40.77s, 0.782 GB pulled, 38 blocks, t_bridge_total 44.68s |
| hetero --native | native | 8.6s | warm native ✓ |

Pass2 is labeled "warm" but `cached prefix 2048` — it is a COLD bridge (pass1's failed attempt left
no usable capture; the engine re-prefilled: 40.77s). **The real bench4 headline: 78,891 tokens cold,
first token 54.9s vs 214.6s native on the same doc = 3.9x.** Warm 20K = 11.0s vs 40.5s = 3.7x.

## Root cause A — mid-request idle flush (kills seeds 601, 603, hetero pass1)
Launcher runs `--max-num-batched-tokens 8192` → every prompt prefills in 8192-token chunks (~4s GPU each).
Hook v3.1's forward thread enqueues each chunk's 43 records in a tight burst, then goes quiet while the
GPU grinds. The idle watcher (`q.empty() and now-last_t >= PD_CAPTURE_IDLE_S=2.0`) fires BETWEEN chunks,
mid-request, and `_finish()` writes manifest+DONE with `T = max(next_pos)` so far:
- seed 601: `AssertionError: (16384, 18424)` — T=16384 = exactly 2 chunks. DONE at +9s.
- seed 603: `AssertionError: (32768, 82703)` — T=32768 = exactly 4 chunks. DONE at +17s.
The continuation chunks then start a NEW request state at position>0 → "WARNING request starts at
position N (prefix hit?) — capture will be partial" → the tail fragment (stamp 155104: T=82703,
calls=2, 5.6 MB, span=0.0s) is useless alone. Front asserts (good) then falls back to full native (bad:
silent in the bench table — only visible in pd_front.out).
bench2/3 succeeded with the SAME launcher because worker-queue contention kept `q.empty()` false through
the inter-chunk gaps. It was always a race; v3.1's leaner forward thread made the race lose.

## Root cause B — tail-chunk data loss with a T-correct manifest (kills seed 605, hetero pass1)
seed 605 capture (stamp 155402): T=105878 ✓, boundaries=51, 975.1 MB, span 50.4s — but assemble died
`KeyError: 'kvwin_100352'` (boundary 49/51). calls=517 = 12×43+1 → exactly ONE chunk's worth of
records missing (chunk 12: tokens 98304–106496). `_finish()` computes `boundaries=_boundaries(0,T)`
from T, NOT from the data actually exported → manifest promises windows that were never captured.
hetero pass1 same: `KeyError: 'kvwin_75776'` (boundary 37/38) at T=78891.

## What is NOT broken
- The pooled capture path itself: pass2 proves it end-to-end at 78,891 tokens (engine 40.77s ≈ the
  hook-off control floor of 39.35s @81K — v3.1's contention cost is now ~1-2s, was ~33s in v3.0).
- Control run (hook off): prefill floor **2020–2077 tok/s** (81K in 39.35s; 106K in 52.59s). The old
  "~8s pure prefill at 81K" estimate is DEAD — README math must use ~2050 tok/s.
- Warm path, fallback correctness (answers all correct, markers found), decode 23–25 tok/s everywhere.
- This is NOT the bench3 98K quadratic-memory bug (that one wrote zero blocks at finalize; different).

## Fixes (front-side land without a pair restart; hook-side needs one ~12min restart)
1. FRONT: validate every DONE capture before trusting it — manifest T==len(ids), no partial_start, no
   position_gaps; keep scanning newer stamps until the engine thread returns + grace; loud decline otherwise.
2. FRONT/ASSEMBLE: salvage longest contiguous boundary prefix (per-boundary try/except in _set_layer);
   if coverage ≥ PD_MIN_COVERAGE, write those blocks — oMLX prefix-hits at B, natively prefills only the
   tail. Turns root-cause-B failures (93% coverage at 100K) into near-wins instead of full native.
3. HOOK: idle watcher must not fire while the last chunk's CUDA event is still pending (GPU busy =
   mid-request). Event-query guard in `_watch`. Kills root cause A structurally.
4. BENCH: bench_cold.py records the X-PD-Bridge header so a fallback can NEVER enter a table again.
5. The 20K variance (bench2 25.5s vs bench3/4 55.2s) — bench4's 55.2 was native fallback (46.3s oMLX +
   ~9s failed-bridge overhead), so bench3's "55.4s slow run" is likely the SAME mid-flush bug, not a
   mystery. bench2's 25.5s remains the one genuine cold-20K bridge sample (n=1).
