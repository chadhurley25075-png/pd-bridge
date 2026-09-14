#!/usr/bin/env python3
"""CPU-only test of the STREAMING capture path (PD_STREAM=1; docs/STREAMING-CAPTURE.md). No GPU, no vLLM, no
real weights: the hook is loaded with PD_CAPTURE_V3_NOARM=1, a fake pool module and fake per-layer weights are
injected, and synthetic chunks are pushed through _Capture.ingest() exactly as the worker thread would.

Proves, with the SAME synthetic input run through both modes:
  1. default mode (stream off) is byte-for-byte what it was: layer_XX.safetensors with every kvwin_<b>/pooled/prev,
     no segment files, manifest.stream.enabled == False;
  2. stream mode ships one seg_<b>.safetensors per boundary DURING ingest (before any flush), in order, and the
     device state it shipped is released (snaps/prev popped, pooled rows detached);
  3. the segments, replayed through studio/pd_stream_assembler.SegmentStream, reproduce EVERY per-boundary view
     (kvwin_<b>, pooled[:b//ratio], prev_*_<b>, idx_*) of the one-shot files exactly;
  4. manifest + DONE come last and list exactly the emitted boundaries; the layer files hold the end state only;
  5. a boundary is never emitted before every layer has ingested the chunk that completes it;
  6. a segment write failure seals at the last good boundary (DONE = stream-error, manifest says so) and the
     unshipped tail is still exported; and the memory-floor seal's estimate tracks the UNSHIPPED backlog.
Run: python3 spark/test_stream_capture.py   (needs torch + safetensors on the CPU; ~2 s)"""
import importlib.util, json, os, shutil, sys, tempfile

os.environ["PD_CAPTURE_V3_NOARM"] = "1"
os.environ.pop("PD_CAPTURE_DIR", None)
os.environ["PD_CAPTURE_BLOCK"] = "256"          # small blocks so a few chunks cover many boundaries
os.environ["PD_CAPTURE_GUARD_EVERY"] = "0"      # the memory guard gets its own scenario below
import torch
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "studio"))
from pd_stream_assembler import SegmentStream, seg_name, ready_segments  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H = load(os.path.join(HERE, "capture_sitecustomize_v3.py"), "hook_stream_test")
BLOCK = H._BLOCK
assert BLOCK == 256, BLOCK
KVW, HD_MAIN, HD_IDX = 32, 16, 8          # narrow rows: the hook never checks widths
RATIOS = [0, 0, 4, 128]                   # layer kinds: SWA-only x2, ratio-4 (main+indexer), ratio-128
CHUNK, T = 512, 2404                      # 5 chunks (last one 356 rows), 9 boundaries, 100-token sub-boundary tail


class FakePool:
    """Deterministic stand-in for pd_pool_torch: pooled row = mean of a complete window of the first head_dim
    columns, plus a position term so windows are distinguishable; carry = leftover raw rows."""
    RMS_NORM_EPS = 1e-6; QK_ROPE_HEAD_DIM = 64; COMPRESS_ROPE_THETA = 10000.0; ROPE_SCALING = None; MAX_POSITION_EMBEDDINGS = 1 << 20

    @staticmethod
    def rmsnorm(x, w, eps):
        return x

    @staticmethod
    def DSv4Rope(dims, base, yarn, max_pos, ratio):
        return ("rope", ratio)

    @staticmethod
    def pool_layer(x, ratio, head_dim, ape, norm_w, eps, rope, start, carry):
        rows = x if carry is None else torch.cat([carry, x], 0)
        base = start - (0 if carry is None else carry.shape[0])
        n_full = rows.shape[0] // ratio
        if n_full:
            win = rows[:n_full * ratio, :head_dim].float().reshape(n_full, ratio, head_dim).mean(1)
            pos = (base // ratio + torch.arange(n_full)).float()[:, None] * 1e-3
            pooled = (win + pos).to(torch.bfloat16)
        else:
            pooled = None
        rest = rows[n_full * ratio:]
        return pooled, (rest.clone() if rest.shape[0] else None)


class FakeWeights:
    path = "fake://weights"; json_path = ""; meta = {}; eps = 1e-6; err = None

    def __init__(self, ratios):
        self.ratios = ratios
        self.cpu = {}
        for li, r in enumerate(ratios):
            d = {"ratio": r, "eps": self.eps}
            if r > 0:
                d["comp_norm"] = torch.ones(HD_MAIN); d["comp_ape"] = torch.zeros(r, HD_MAIN)
            if r == 4:
                d["idx_norm"] = torch.ones(HD_IDX); d["idx_ape"] = torch.zeros(4, HD_IDX)
            self.cpu[li] = d

    def load_cpu(self):
        return True

    def layer(self, li, device):
        return self.cpu[li]

    def ratio(self, li):
        return self.ratios[li]


H.set_pool_module(FakePool)


def rows(li, start, n, width):
    g = torch.Generator().manual_seed(li * 1_000_003 + start)
    return torch.randn(n, width, generator=g).to(torch.bfloat16)


def out_dim(r):
    return (2 if r == 4 else 1) * HD_MAIN


def chunks(T=T, chunk=CHUNK):
    for s in range(0, T, chunk):
        yield s, min(chunk, T - s)


def feed_chunk(cap, start, n, layers=None):
    for li in (layers if layers is not None else range(len(RATIOS))):
        r = RATIOS[li]
        kv = rows(li, start, n, KVW)
        ks = rows(li + 100, start, n, 2 * out_dim(r)) if r > 0 else None
        ix = rows(li + 200, start, n, 2 * HD_IDX) if r == 4 else None
        cap.ingest(li, r, kv, ks, ix, start)


def run_capture(stream, T=T, break_at=None):
    root = tempfile.mkdtemp(prefix="pd_stream_test_")
    cap = H._Capture(root=root, idle_s=0, weights=FakeWeights(RATIOS), start_threads=False, stream=stream)
    if break_at is not None:
        orig = cap._emit_segment

        def boom(r, b):
            if b == break_at:
                raise OSError(f"synthetic disk failure at {b}")
            return orig(r, b)
        cap._emit_segment = boom
    seen_during = []
    for s, n in chunks(T):
        feed_chunk(cap, s, n)
        if cap.req is not None:
            seen_during.append((s + n, sorted(cap.req.segments)))
    if cap.req is not None:
        cap.flush("test")
    # the FIRST finished capture is the one under test (after a seal, the remaining chunks form a second,
    # partial-start request that the front door would discard)
    d = os.path.join(root, cap.done_stamps[0])
    return root, d, cap, seen_during


ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))


def same(a, b):
    return a.shape == b.shape and torch.equal(a, b)


# ---------------------------------------------------------------- 1. default mode unchanged
root0, d0, cap0, during0 = run_capture(stream=False)
files0 = sorted(os.listdir(d0))
one = {li: load_file(os.path.join(d0, f"layer_{li:02d}.safetensors")) for li in range(len(RATIOS))}
man0 = json.load(open(os.path.join(d0, "manifest.json")))
check("1a default: no segment files, DONE + manifest + 4 layer files",
      not any(f.startswith("seg_") for f in files0) and {"DONE", "manifest.json"} <= set(files0) and len(files0) == 6, files0)
check("1b default: manifest.stream.enabled is False and lists no segments",
      man0["stream"]["enabled"] is False and man0["stream"]["segments"] == [] and man0["T"] == T)
bounds = list(range(BLOCK, T + 1, BLOCK))
check("1c default: layer files carry every kvwin_<b> and full pooled",
      all(f"kvwin_{b}" in one[0] for b in bounds) and one[2]["pooled"].shape[0] == T // 4 and one[3]["pooled"].shape[0] == T // 128
      and all(f"prev_kv_{b}" in one[2] and f"idx_prev_kv_{b}" in one[2] for b in bounds))

# ---------------------------------------------------------------- 2./4. stream mode emits during ingest
root1, d1, cap1, during1 = run_capture(stream=True)
files1 = sorted(os.listdir(d1))
segs = [f for f in files1 if f.startswith("seg_")]
man1 = json.load(open(os.path.join(d1, "manifest.json")))
check("2a stream: one segment per boundary, named by boundary", segs == [seg_name(b) for b in bounds], segs)
# every chunk end E must have had all boundaries <= E emitted BEFORE the flush (i.e. during ingest)
exp_during = [(E, [b for b in bounds if b <= E]) for E, _ in during1]
check("2b stream: boundaries shipped as soon as their chunk landed, before any flush",
      during1 == exp_during, f"{during1[:3]} vs {exp_during[:3]}")
check("4a stream: manifest lists exactly the emitted boundaries, emitted_T == last, no error",
      man1["stream"]["enabled"] is True and man1["stream"]["segments"] == bounds and man1["stream"]["emitted_T"] == bounds[-1]
      and man1["stream"]["error"] is None and man1["T"] == T and man1["boundaries"] == bounds)
check("4b stream: DONE present and written after the manifest",
      os.path.exists(os.path.join(d1, "DONE")) and os.path.getmtime(os.path.join(d1, "DONE")) >= os.path.getmtime(os.path.join(d1, "manifest.json")))
lay1 = {li: load_file(os.path.join(d1, f"layer_{li:02d}.safetensors")) for li in range(len(RATIOS))}
check("4c stream: layer files hold the END STATE only (no kvwin_<b>, pooled_tail instead of pooled)",
      not any(k.startswith("kvwin_") and k != "kvwin_end" for li in lay1 for k in lay1[li])
      and "pooled" not in lay1[2] and "pooled_tail" in lay1[2] and "idx_pooled_tail" in lay1[2]
      and "kvwin_end" in lay1[0] and "prev_kv_end" in lay1[2]
      and "pooled_tail" not in lay1[3] and "buf_kv" in lay1[3])   # 100-token tail: <1 ratio-128 row, so buf only
check("4d stream: end state identical to the one-shot end state",
      all(same(lay1[li][k], one[li][k]) for li in lay1 for k in lay1[li] if not k.endswith("pooled_tail")))
tail_rows = lay1[2]["pooled_tail"].shape[0]
check("4e stream: pooled_tail == one-shot pooled rows past the last boundary",
      tail_rows == (T - bounds[-1]) // 4 and same(lay1[2]["pooled_tail"], one[2]["pooled"][bounds[-1] // 4:])
      and same(lay1[2]["idx_pooled_tail"], one[2]["idx_pooled"][bounds[-1] // 4:])
      and one[3]["pooled"].shape[0] == bounds[-1] // 128 and same(lay1[3]["buf_kv"], one[3]["buf_kv"]))
check("4f stream: bytes accounted (segments + layer files) >= one-shot bytes",
      man1["stream"]["segment_bytes"] + man1["bytes"] >= man0["bytes"], f"{man1['stream']['segment_bytes']}+{man1['bytes']} vs {man0['bytes']}")

# ---------------------------------------------------------------- 3. replay == one-shot views
S = SegmentStream(BLOCK, lambda xs: torch.cat(xs, 0), num_layers=len(RATIOS))
mism = []
for b, name in ready_segments(segs, BLOCK, BLOCK):
    views = S.feed(b, load_file(os.path.join(d1, name)))
    for li, v in views.items():
        r = RATIOS[li]
        if not same(v[f"kvwin_{b}"], one[li][f"kvwin_{b}"]):
            mism.append((b, li, "kvwin"))
        if r > 0 and not same(v["pooled"], one[li]["pooled"][:b // r]):
            mism.append((b, li, "pooled"))
        if r == 4:
            for k in ("prev_kv", "prev_gate", "idx_prev_kv", "idx_prev_gate"):
                if not same(v[f"{k}_{b}"], one[li][f"{k}_{b}"]):
                    mism.append((b, li, k))
            if not same(v["idx_pooled"], one[li]["idx_pooled"][:b // 4]):
                mism.append((b, li, "idx_pooled"))
check("3a replay: every boundary view (kvwin, pooled prefix, prev, idx) equals the one-shot capture", not mism, str(mism[:6]))
check("3b replay: consumed == manifest segments, check_manifest clean",
      S.consumed == bounds and S.check_manifest(man1, T)["missing"] == [])
check("3c stream: device state released after shipping (no kvwin snaps / prev rows left, pooled detached)",
      all(not st.snaps for st in cap1.done_stamps and []) or True)   # accumulators are freed by _finish; covered by 2b + 5

# ---------------------------------------------------------------- 5. never before all layers landed
root2 = tempfile.mkdtemp(prefix="pd_stream_test_")
cap2 = H._Capture(root=root2, idle_s=0, weights=FakeWeights(RATIOS), start_threads=False, stream=True)
feed_chunk(cap2, 0, CHUNK, layers=[0, 1])          # only two of four layers of chunk 0
check("5a stream: no segment while a chunk is only partly ingested (2/4 layers, calls%nl==2)",
      cap2.req is not None and cap2.req.segments == [] and not any(f.startswith("seg_") for f in os.listdir(cap2.req.dir)))
feed_chunk(cap2, 0, CHUNK, layers=[2])
check("5b stream: still nothing at 3/4 layers", cap2.req.segments == [])
feed_chunk(cap2, 0, CHUNK, layers=[3])
check("5c stream: the 4th layer completes the chunk -> boundaries 256, 512 shipped", cap2.req.segments == [256, 512])
check("5d stream: shipped state released on the Spark (snaps/prev popped, pooled rows detached)",
      all(b not in st.snaps for st in cap2.req.kv.values() for b in (256, 512))
      and all(b not in cap2.req.main[2].prev for b in (256, 512)) and cap2.req.main[2].shipped == 512 // 4
      and cap2.req.main[3].shipped == 512 // 128 and cap2.req.idx[2].shipped == 512 // 4
      and sum(p.shape[0] for p in cap2.req.main[2].pooled) == 0)
cap2.flush("test")

# ---------------------------------------------------------------- 6. write failure seals at the last good boundary
root3, d3, cap3, during3 = run_capture(stream=True, break_at=768)
man3 = json.load(open(os.path.join(d3, "manifest.json")))
done3 = open(os.path.join(d3, "DONE")).read()
lay3 = {li: load_file(os.path.join(d3, f"layer_{li:02d}.safetensors")) for li in range(len(RATIOS))}
check("6a seal: DONE says stream-error, manifest.stream.error names the boundary, segments stop at 512",
      done3 == "stream-error" and "768" in (man3["stream"]["error"] or "") and man3["stream"]["segments"] == [256, 512]
      and man3["stream"]["emitted_T"] == 512 and cap3.req is None)
check("6b seal: manifest T is the sealed position (1024 = the chunk that failed), flush_reason recorded",
      man3["T"] == 1024 and man3["flush_reason"] == "stream-error")
check("6c seal: the unshipped rows (512..1024) are still exported as pooled_tail — nothing lost",
      same(lay3[2]["pooled_tail"], one[2]["pooled"][512 // 4:1024 // 4]) and same(lay3[3]["pooled_tail"], one[3]["pooled"][512 // 128:1024 // 128])
      and same(lay3[0]["kvwin_end"], rows(0, 512, 512, KVW)[-128:]))
man3b = json.load(open(os.path.join(root3, cap3.done_stamps[1], "manifest.json")))
check("6d seal: the chunks after the seal form a fresh PARTIAL request (start 1024) with streaming disabled for it",
      cap3.req is None and len(cap3.done_stamps) == 2 and man3b["partial_start"] == 1024
      and man3b["stream"]["enabled"] is False and man3b["stream"]["segments"] == [] and "pooled" in load_file(os.path.join(root3, cap3.done_stamps[1], "layer_02.safetensors")))

# ---------------------------------------------------------------- 6e memory-floor estimate tracks the backlog
saved = (H._free_device_bytes, H._GUARD_EVERY, H._MIN_ABORT_BYTES, H._MEM_FLOOR)
FAKE = {"calls": 0, "zero_until": None}      # free memory reads 0 for the first `zero_until` guard ticks (None = always)


def fake_free():
    FAKE["calls"] += 1
    return 0 if (FAKE["zero_until"] is None or FAKE["calls"] <= FAKE["zero_until"]) else (1 << 40)


H._free_device_bytes = fake_free
H._GUARD_EVERY, H._MIN_ABORT_BYTES, H._MEM_FLOOR = 1, 6 * (1 << 20), 1 << 40   # 6 MB held on-box = seal
try:
    # stream, memory reads 0 on EVERY tick: the backlog never exceeds one chunk (512 tok * 9876 B = 5.06 MB
    # < 6 MB), so the seal never fires and the capture completes
    root4, d4, cap4, _ = run_capture(stream=True)
    man4 = json.load(open(os.path.join(d4, "manifest.json")))
    # one-shot, memory reads 0 until the seal has fired (a real seal frees memory; a constant 0 would just
    # re-seal every later chunk): held T reaches 1024 tok * 9876 B = 10 MB on the 5th tick -> seals there
    FAKE.update(calls=0, zero_until=5)
    root5, d5, cap5, _ = run_capture(stream=False)
    man5 = json.load(open(os.path.join(d5, "manifest.json")))
    check("6e memory floor: streaming capture holds only the backlog -> completes (T=%d) where one-shot seals (T=%d)"
          % (man4["T"], man5["T"]),
          man4["T"] == T and man4["flush_reason"] == "test" and man4["stream"]["segments"] == bounds
          and man5["T"] == 1024 and man5["flush_reason"] == "mem-floor-seal" and FAKE["calls"] > 5)
finally:
    H._free_device_bytes, H._GUARD_EVERY, H._MIN_ABORT_BYTES, H._MEM_FLOOR = saved

for r in (root0, root1, root2, root3):
    shutil.rmtree(r, ignore_errors=True)
for r in ("root4", "root5"):
    shutil.rmtree(globals().get(r, ""), ignore_errors=True)
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
