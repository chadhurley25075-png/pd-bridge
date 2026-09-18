# SPIDERSENSE DISTRIBUTION — multi-lane inference transport
*Design + measured baseline. Compass (Fable seat) for Chad, 2026-09-16 night. All numbers measured tonight unless marked.*

## The idea in one paragraph
The fleet has four kinds of wire. Don't bond them — **assign traffic to the lane built for it.** Bulk (KV, weights,
captures) is chunkable and bandwidth-bound: stripe it across lanes. Per-token collectives (tensor-parallel) are
latency-bound: keep them on ONE homogeneous fabric. The Studio TB5 mesh is the *distribution layer*: a door Studio
ingests from the Sparks and redistributes to peer Studios, so every Studio is fed at door speed without a card.

## Lanes (as wired 9/16)
| Lane | Speed | RDMA | Notes |
|---|---|---|---|
| Spark↔Spark QSFP RoCE (.220.x) | 200G | yes | 7 Sparks + Cerebro, flat L2 across m1/m2 (200G ISL) |
| Spark→Studio door (Sonnet + CX-4 Lx) | **26.4G** iperf / 23G single HTTP | after SIP | S1 live (.220.21). 3 CX-5 Ex cards inbound → ~38G each |
| Spark 10GbE → Studio 10GbE (Omada LAN) | 9.3G | no | every node |
| Studio↔Studio TB5 mesh (rdma_en*, jaccl) | 80G RDMA link; **18.4G TCP** measured | yes (jaccl) | S1–S4 K4-ish mesh; TCP over it is kernel-limited |

## Measured tonight (4,057 MiB payload = Ash's largest KV)
| Test | Result | Verdict |
|---|---|---|
| Door only, Spark→S1 | 1.44–1.48s = **23.2G** | baseline |
| LAN only, Spark→S1 | 3.67s = 9.3G | old path |
| **STRIPE door+LAN, 72/28 split, separate server sockets** | **1.27s = 26.8G** | **+15%** over door alone — PROVEN |
| Stripe 50/50 | 2.76s | slow lane gates — split must match lane ratio |
| Stripe, both lanes on ONE python http.server | 3.74s | single-threaded server serialises; use one server per lane or threaded |
| TB5 mesh S3→S1, HTTP 1-stream | 13.8G | TCP over TB5 well below 80G RDMA |
| TB5 mesh S1→S3, memory→memory TCP, 4MiB sends | **18.4G** | TCP ceiling on TB5 ≈ 18–20G; RDMA (jaccl) is the 80G path |
| **GATEWAY two-hop Spark→S1→S3, pipelined TCP relay** | 3.05s = **11.2G end-to-end** | works; 4× faster than... no: 1.2× faster than LAN direct (9.3G). Bound by TB5-TCP hop. |
| Gateway via ssh pipe | 6.05s = 5.6G | SSH crypto halves it — never relay through ssh |

## What this proves / disproves
1. **Striping is real and cheap** — +15% today, and the gain scales with lane count (add door #2 = +26G). Rule: split
   bytes in proportion to lane speed; one server socket per lane.
2. **Gateway pattern works but TCP-over-TB5 is the choke (18G).** The 80G needs RDMA verbs on the mesh — that's jaccl
   (already used by EXO for TP) or Apple `rdma_en*` via libibverbs. Relay must be RDMA WRITE, not a socket copy.
   → With RDMA relay, S2 gets fed at door speed (26→38G) with zero S2 config change (S2 law intact).
3. **Never mix lane classes inside a collective.** TP stays: Sparks on RoCE, Studios on jaccl/TB5.
4. **Spark side is not the bottleneck** — 200G/port vs 26–38G/door. One Spark can feed 3 doors at once.

## The build (in order)
1. `pd_front.py` striped pull: N block sockets bound per-lane, byte split by measured lane speed. ~20 lines. Do first.
2. Three doors (S1, S3, S4) with CX-5 Ex → ~38G each, ~114G aggregate Spark→Studio ingress. S2 untouched.
3. RDMA relay on door Studios: receive blocks → `ibv_post_send` WRITE into peer Studio memory over `rdma_en*`.
   Nobody has this yet (Ben's pd_rdma writes local; jaccl is TP-only). This is ours to build. Unlocks 80G distribution.
4. After SIP + MelonDMA: door hop becomes RDMA too (≈9 µs) → end-to-end RDMA Spark→door→any Studio.

## Gotchas learned tonight
- en16 grabbed a DHCP lease + default route from Omada over the flat bridge. Set Manual BEFORE link-up on S3/S4.
- Deleting routes on macOS: `route delete -net X -ifscope IF` can remove the on-link route too. Verify `route get` after.
- Studio TB5 links carry only link-local by default; add /30 aliases for TCP tests, remove after.
- CNS bus caps a command at 30s — split long benches.
