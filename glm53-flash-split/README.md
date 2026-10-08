# glm53-flash-split — one GLM-5.3-Flash across three machines (2026-10-03/04)

**Two DGX-Spark-class boxes read the prompt. A Mac Studio writes the answer. The state crosses an RDMA door.**

```
GB10 pair (vLLM TP2, EXL3 4bpw)  ──prefill + Glm53HandoffConnector──▶  /dev/shm handoff blobs
        │                                                                  │
        └──────────── MCDMA RDMA READ (ConnectX-4 Lx door, switched RoCE) ─┘──▶ Studio handoff daemon
                                                                                  │
            tf_split.py: Ash's assembly → TensorFold KDACache / MLACache → TensorFold prompt store
                                                                                  │
                                         TensorFold (Mac Studio M3 Ultra) prefills only the tail and decodes
```

This is the pd-bridge idea on a different model and a different decode engine. The DeepSeek-V4 bridge in the
rest of this repo computes the decoder's cache on the prefill side; here **Ash Hart's** handoff kit already
exports GLM-5.3-Flash's state from vLLM (KDA linear-attention state for the linear layers; latent KV + pooled
index keys for the sparse layers), and what we wrote is the last mile: `tf_split.py` turns that state into
**TensorFold's own per-layer caches** and stores it as a prompt prefix, so TensorFold sees a normal prefix hit,
prefills only what is left, and decodes with its own kernels and MTP drafting.

## Why bother: the numbers (2026-10-03/04, needle found on every run)

Same prompt three ways. "First word" = time to first streamed token; prompts are a long document + a question.

| prompt | Sparks alone (vLLM TP2) | TensorFold alone (Studio) | **split** |
|---|---|---|---|
| 17,962 tokens — first word | 11.42 s | 32.01 s | **13.56 s** |
| 33,903 tokens — first word | 21.14 s | 59.68 s | **23.84 s** → **23.09 s** after the fp8 table (below) |
| decode | 19.7–21.3 tok/s | 48.3–50.6 tok/s | **45.4–49.8 tok/s** |
| follow-up turn on the same context (17,983 tokens, 17,956 cached) | — | — | **0.19 s** |

The split reads at Spark speed and writes at TensorFold speed: within ~2 s of the Sparks' first word, at
more than twice their decode rate. Bigger prompts (split only):

| prompt | first word | breakdown |
|---|---|---|
| 109,851 tokens | **73.8 s** (was 186.6 s before the door fix below) | Spark prefill 68.4 s (1,606 tok/s) · door 0.35 s for 1,035 MiB · assemble 3.6 s · cache build 1.1 s · decode 47.2 tok/s |
| 97,504 tokens | 71.3 s | decode 41.2 tok/s |
| 30K–70K, 10 back-to-back (soak) | 22.0–48.2 s | 10/10 needle · decode 41.5–54.5 tok/s, one run 8.6 tok/s while the same engine prefilled another client's prompts |
| 17,320 tokens, 12 health probes 20 min apart overnight | 12.71–12.85 s | 12/12 needle · decode 44.3–53.4 tok/s |

At 33,903 tokens the stages were: Spark prefill 21.1 s (1,607 tok/s) · door 0.23 s (491 MiB) · assemble 1.2 s
(0.54 s with the fp8 lookup table) · TensorFold cache build 1.1 s.

**Is the handed-off state faithful?** `compare` mode (`results/compare_8884_2026-10-03.json`, 8,884 tokens):
KL over the prefill engine's own top-20 next-token distribution — **split 0.026, TensorFold's own prefill 0.060**,
top-1 identical for both. The split is *closer* to the reference than the decoder's own reading of the same
tokens. What remains is quantization: EXL3 4 bpw on the Sparks vs MLX 4-bit on the Studio, so the per-layer
latent differs from TensorFold's own by 0.10 relative L2 at layer 3, rising (not monotonically) to 0.38 at layer 43. Every
alternative layout and a ±1 position shift were checked and are far worse (≥1.0), so this is not a layout bug —
but it does mean **split output is not token-identical to either engine alone.**

**Lock step.** One stream decoding while a 34K split ran: worst inter-chunk gap on the decoding stream 0.18 s,
p99 0.05 s — after the two-phase fix below (before it, that stream fell to 6.5 tok/s). Two splits submitted
together (25,963 + 21,445 tokens): 31.5 s and 19.4 s first word — the pair's shared prefill rate is the limit.

All rows: `results/split_runs_2026-10-03.json` (answers stripped; see its notes). The measuring harness was
`bench_split.py` with a private document; the public `bench_split.py` uses synthetic text and **has not been
re-run** against the split.

## Hardware and software we ran

| role | hardware | software |
|---|---|---|
| prefill | 2× GB10 (ASUS Ascent GX10, the DGX Spark design), TP2 over their 200G link | vLLM via [MiaAI-Lab's GLM-5.3-Flash EXL3 two-Spark recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) (EXL3 4 bpw, stock weights) + Ash Hart's `Glm53HandoffConnector` (`kv_role: kv_producer`), window 196,608 |
| door | ConnectX-4 Lx in a Thunderbolt 5 enclosure on the decode Studio, 40G port on the same switched RoCE fabric as `fabric/` | [MCDMA](https://github.com/ashhart/MCDMA) + Ash's handoff daemon (one RDMA READ puller) |
| decode | Mac Studio M3 Ultra, 512 GB | [TensorFold](https://github.com/ashhart/TensorFold) **0.6.2**, an MLX 4-bit GLM-5.3-Flash build, + `tf_split.py` |

## Running it

You need Ash Hart's GLM-5.3-Flash handoff kit (`glm53_handoff_connector.py` for vLLM, the Studio handoff daemon,
and `glm53_split.py`). **It was not public when this was written** — ask Ash. Nothing of his is vendored here;
`tf_split.py` imports `glm53_split` from your checkout (`TF_SPLIT_ASH_SRC`).

1. **Prefill pair:** `prefill_pair_up.example.sh` shows the connector arguments we passed to MiaAI-Lab's `start.sh`.
2. **Door:** start Ash's handoff daemon on the Studio with the pair as peers **in TP rank order** (rank 0 first).
   Read 256 KB chunks (`HANDOFF_CHUNK_KB=256`, see *Traps*).
3. **TensorFold:** put this directory on the `tensorfold serve` process's `PYTHONPATH`. `sitecustomize.py` installs
   the hook only inside `tensorfold serve`; `TF_SPLIT=0` disables it.
4. **Config:** copy `tf_split.example.json` to `~/.glm53/tf_split.json` and set `url` (the pair's vLLM endpoint),
   `peers` (the daemon's peer names, rank order). It is hot-reloaded, and so is `tf_split.py` itself.
   `mode`: `off` · `on` · `compare` (also runs TensorFold's own prefill and reports the per-component difference
   and the KL above — slow, for checking only).
5. `python3 bench_split.py http://STUDIO:PORT/v1 glm-5.3-flash 32000` — compare against the pair's own endpoint
   and against TensorFold with `mode: off`.

**What the hook does.** `Scheduler.submit` (the request's own thread): pick the handoff boundary (the start of
the last message, rounded down to 4, capped at `max_handoff_tokens`) → reserve the pair's KV budget → the Sparks
prefill it → take the door lock → pull. `PromptFill._start_fill` (TensorFold's scheduler thread): Ash's
assembly → `build_cache` → insert into TensorFold's prompt store → add the boundary to the prompt's chunk starts.
Any failure at any step is logged and TensorFold prefills the prompt itself; a request never fails because of
the split.

## Traps (each one cost us an hour or a night)

- **Big pulls collapsed** (960 MiB took 112 s, repeats got worse). A 2 MiB RDMA READ reply arrives as a 200G
  burst into a 40G door; the switch dropped packets toward the Studio and ignored its pause frames (flow control
  off on that port), and RoCE sat in 67–537 ms retransmit timeouts. Fix without touching the switch: the
  daemon reads **256 KB chunks** → 960 MiB in 0.33 s (3,839/3,839 chunks fast, ~24 Gb/s), 80 MB in 27 ms.
  Flow control on the door port would also help.
- **Do not run the network phase on the decoder's scheduler thread.** The first version did; every other stream
  froze for ~21 s. Network work on the request thread, only assemble + build (~2.3 s at 34K) on the scheduler.
- **Livelock near KV capacity.** After a restart the pair's KV held ~207–211K tokens and a handoff needs about 2×
  its length. A 120,000-token handoff spun 15 minutes; a 100K one livelocked when a second consumer captured on
  the pair at the same time. Fixes: `max_handoff_tokens` 100,000 (longer prompts hand off the first 100K and
  TensorFold prefills the rest) · a Spark-prefill timeout of 60 s + tokens/600 (the disconnect makes vLLM abort) ·
  a KV budget across concurrent splits · big handoffs (≥40K) and the other consumer take turns via marker files.
- **The handoff boundary must be the start of the last message**, not the history length: otherwise a new question
  on the same document re-does the whole handoff. After the fix, a different question on an 18,108-token handed-off
  document answered its first word in 0.23 s.
- **After a reboot the prefill boxes forget the door's IPv6 link-local neighbour** → QP stuck (`RTR 110`). Pin it
  (`ip -6 neigh replace <fe80::…> lladdr <mac> dev <if> nud permanent`) in the unit that starts the connector side.
- **`arena registration errno=12` on boot** = systemd's memlock limit. `LimitMEMLOCK=infinity` in that unit.
- **macOS Local Network Privacy:** a server launched through `ssh host "nohup …"` from a launch agent got
  `EHOSTUNREACH` to LAN peers; the same binary started with `nohup … & disown` from an interactive ssh session
  reached them. If the split silently never happens, check this first — the hook falls back to local prefill.
- **The first two requests after a pair restart prefill at ~1/3 speed** (~450–700 tok/s vs ~1,350–1,600). Warm up.
- fp8 e4m3 → fp32 through a 256-entry table instead of elementwise decode: identical values, assemble at 34K
  1.2 s → 0.54 s.

## Status, honestly

- **Pinned to TensorFold 0.6.2 private internals** (`Scheduler.submit`, `PromptFill._start_fill`,
  `LaneEngine.prompt_chunks`, `KDACache`/`MLACache`). TensorFold has since released 1.0; we have not tried this
  against it. Expect it to break.
- **Depends on a kit that was not public** (Ash Hart's handoff connector, daemon and `glm53_split`).
- **One pair, one door, one puller.** A single lock serializes pulls on the MCDMA device.
- **Measured over two nights on one setup.** No error bars; the soak and the overnight probes are the
  repeatability evidence we have.
- `frozen_marker_ids` (keep a TensorFold checkpoint where a long fixed prefix ends) is off by default; it was
  built for a 700K-token fixed prefix and is described, not benchmarked, here.

## Credits

**Ash Hart** — MCDMA, TensorFold, and the GLM-5.3-Flash handoff kit (`Glm53HandoffConnector`, the handoff daemon,
`glm53_split`): the transport, the export and the state assembly are all his; this folder is the TensorFold end.
**MiaAI-Lab** — the GLM-5.3-Flash EXL3 two-Spark vLLM recipe the prefill pair runs. vLLM, MLX.
