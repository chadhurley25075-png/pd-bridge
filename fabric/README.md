# fabric/ — the switched RDMA layer (2026-09-17)

Everything in `pd-bridge` above this directory moved the cache over **10GbE HTTP**. This directory is what
changed when a Mac Studio joined the Sparks' RoCE fabric through a switch and the Studios' Thunderbolt mesh
started carrying the same blocks. It is the difference between *one Spark pair → one Mac* and a **ring**:

```
spark-06 + spark-07  ──prefill──▶  capture  ──MCDMA RDMA WRITE (MikroTik CRS812)──▶  S1 (the door)  ──assemble──▶  oMLX blocks
                                                                                     S1 blocks ──TB5 RDMA (UC SEND)──▶  S2 (the library, 512 GB) ──▶ decodes
                                                                                     S2's reply blocks ──TB5 RDMA──▶ S1        (both Studios hold the conversation)
                                                        reply text ──▶ next prompt ──▶ Sparks prefill again  (KV return arrow: not yet)
```

## Hardware that made it possible (credit where it belongs)
- **[MCDMA](https://github.com/ashhart/MCDMA) by Ash Hart** — the macOS ConnectX RDMA driver. We changed three constants
  ([PR #3](https://github.com/ashhart/MCDMA/pull/3)) so it binds a ConnectX-4 Lx; the driver did the rest.
  Ash said it would work as-is. It did.
- **[MelonDMA](https://github.com/) by Ben** — the ConnectX-4 Lx driver whose bench numbers told us what the Gen3 tunnel
  could do before we had one, and whose `iommu.passthrough` and CQ-mapping findings are on our list.
- **Apple's RDMA over Thunderbolt** (TN3205) — UC queue pairs and `IBV_WR_SEND` only; no one-sided ops. `tbsend.c` is
  written to exactly that contract.
- A **$40 used ConnectX-4 Lx** in a **Sonnet Echo SE I T5**, a **Cisco 40G DAC**, one cage on a **MikroTik CRS812**.

## What is in here
| file | what it is |
|---|---|
| `rdma_file.c` | one file Spark→Mac (or Mac→Spark) by RDMA WRITE over MCDMA/RoCE. Windowed (Mac MR cap ~128 MiB total, ≤4 MiB per region), 2 slots, flag word + ACK. Builds on Linux (`-libverbs`) and macOS (`-lrdma`). |
| `rdma_pull.py` / `rdma_pulldir.py` | orchestrators. `rdma_pulldir` tars a capture directory on the Spark and moves it in **one** RDMA session — 44 files, 416 MB in 2.2 s; the same pull per-file took 42 s. |
| `tbsend.c` / `tbrun.py` | Studio↔Studio file transfer over Apple Thunderbolt RDMA. Same-size frames both ends, receives posted **after** RTR, ≤~4 MiB posted. **72 Gb/s** measured S1→S4 on a 1 GiB payload. |
| `pd_ring_front.py` | the **hetero-deep4 front door** (`:8015` on the door Studio). Every request: door bridges (Sparks→RDMA→blocks, no decode) → new blocks → library over TB5 → library streams the reply → library's new blocks → back to the door. Falls back to door decode. Per-hop timings in the `X-PD-Ring` header. |
| `hetero_ring.py` | the multi-turn benchmark that produced `ring_results-2026-09-17.jsonl`. |
| `../studio/pd_front.py` | the door, with `PD_RDMA=1` (RDMA dir pull, HTTP fallback) and `X-PD-Bridge-Only: 1` (ingest without decoding). |
| `../spark/pd-launch-v3.sh` | now resolves `NCCL_IB_GID_INDEX` at launch. Hard-coding it broke the pair when MCDMA's link-local neighbours shifted the GID table. |

## Numbers (all 2026-09-17, one M3 Ultra door, one M3 Ultra 512 GB library, one Spark pair, temperature 0)

**Transport**
| hop | measured | ceiling |
|---|---|---|
| Spark → S1, RDMA WRITE, real DV4 bytes | 21–24.6 Gbit/s · 4 KiB WRITE **6.9 µs** median (p99 7.0) | PCIe Gen3 x4 tunnel (~26 G payload) — 93% of it |
| S1 → S2, TB5 RDMA (UC SEND) | 13–16 Gbit/s on block batches · **72 Gbit/s** on 1 GiB | TB5 80 G |
| same hops over TCP yesterday | 9.3 G (10GbE) · 18 G (TB5 TCP) | |

**The ring, four turns, ~15K new tokens each**
| turn | prompt tok | Sparks prefill | RDMA→S1 | TB5→S2 | **S2 library reply** (cached) | S2 alone | Sparks alone |
|---|---|---|---|---|---|---|---|
| 2 | 31,419 | 15.6 s | 0.31 s | 16.2 G | **7.7 s** (30,720) | 29.8 s | 17.8 s |
| 3 | 47,457 | 23.2 s | 0.45 s | 16.3 G | **7.4 s** (47,104) | 31.0 s | 19.2 s |
| 4 | 63,434 | 31.4 s | 0.47 s | 16.4 G | **10.5 s** (61,440) | 30.9 s | 18.4 s |

The library never prefills: it answers a 63K-token conversation in 10 s because the Sparks computed the attention
state and two RDMA fabrics carried it. S2-alone stays flat at ~31 s (re-prefills everything). **3–4× vs one Studio,
~2× vs the Sparks, widening with context.** The 713,666-token cold wake bridged end to end: 615.6 s Spark prefill,
**7.03 GB over RDMA in 5.4 s wire**, 348 blocks, `complete`.

## Honest limits
- The carry back to the Sparks is **text**: turn k+1 re-prefills turn k. The KV return arrow (~10 MB of rotating-window +
  compressor state written into the Spark by RDMA, and the capture hook accepting `start > 0`) is the next build.
- Mac-side `ibv_reg_mr` caps near 4 MiB per region and ~128 MiB total under MCDMA today; the writer pools regions.
- Apple TB RDMA has no RDMA WRITE/READ; the Studio hop is a two-sided message, not a one-sided write. Still kernel-bypass.
- EXO/JACCL owns the Thunderbolt RDMA peers while it runs (`launchctl bootout system/com.exo.node` to borrow the mesh; `pkill` respawns it).
  Our first "front ports are slow" and "one peer per controller" readings were this collision, not hardware — retracted.
- `pd_ring_front` ships blocks newer than a marker; blocks that land while no request is in flight are not shipped until the next one. Known, on the list.
- A door reboot loses the MCDMA static neighbours, the S2 tunnel and the ring front; not yet in the keeper.

See `../docs/SPIDERSENSE-DISTRIBUTION-2026-09-16.md` for the lane-by-lane design and the tests behind it.
