#!/usr/bin/env python3
"""CPU-only test of studio/pd_stream_assembler.py under numpy (no mlx, no model): ordering, resume, manifest
integrity, and a consumer driving a directory that fills up over time (the front door's situation).
Run: python3 studio/test_stream_assembler.py   (needs numpy + safetensors; ~1 s)"""
import json, os, shutil, sys, tempfile, threading, time
import numpy as np
from safetensors.numpy import save_file, load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pd_stream_assembler import SegmentStream, consume_dir, ready_segments, seg_name, seg_boundary  # noqa: E402

BLOCK, N, RATIOS = 256, 9, [0, 4, 128]     # 9 boundaries, three layer kinds
HD = 16
T = N * BLOCK + 100                          # sub-boundary tail, like every real prompt
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))


def truth():
    """Ground truth per layer: the full pooled arrays and per-boundary rows a one-shot capture would hold."""
    rng = np.random.default_rng(7)
    full = {}
    for li, r in enumerate(RATIOS):
        d = {"kvwin": {b: rng.standard_normal((128, 32)).astype(np.float32) for b in range(BLOCK, T + 1, BLOCK)}}
        if r > 0:
            d["pooled"] = rng.standard_normal((T // r, HD)).astype(np.float32)
        if r == 4:
            d["idx_pooled"] = rng.standard_normal((T // 4, HD // 2)).astype(np.float32)
            for k in ("prev_kv", "prev_gate", "idx_prev_kv", "idx_prev_gate"):
                d[k] = {b: rng.standard_normal((4, HD)).astype(np.float32) for b in range(BLOCK, T + 1, BLOCK)}
        full[li] = d
    return full


def segment(full, b):
    out = {}
    for li, r in enumerate(RATIOS):
        pre = f"layer_{li:02d}."
        out[pre + f"kvwin_{b}"] = full[li]["kvwin"][b]
        if r > 0:
            out[pre + f"pooled_{b}"] = full[li]["pooled"][(b - BLOCK) // r: b // r]
        if r == 4:
            out[pre + f"idx_pooled_{b}"] = full[li]["idx_pooled"][(b - BLOCK) // 4: b // 4]
            for k in ("prev_kv", "prev_gate", "idx_prev_kv", "idx_prev_gate"):
                out[pre + f"{k}_{b}"] = full[li][k][b]
    return out


def views_ok(full, b, views):
    for li, r in enumerate(RATIOS):
        v = views[li]
        if not np.array_equal(v[f"kvwin_{b}"], full[li]["kvwin"][b]):
            return f"kvwin b={b} li={li}"
        if r > 0 and not np.array_equal(v["pooled"], full[li]["pooled"][: b // r]):
            return f"pooled b={b} li={li}"
        if r == 4:
            if not np.array_equal(v["idx_pooled"], full[li]["idx_pooled"][: b // 4]):
                return f"idx_pooled b={b} li={li}"
            for k in ("prev_kv", "prev_gate", "idx_prev_kv", "idx_prev_gate"):
                if not np.array_equal(v[f"{k}_{b}"], full[li][k][b]):
                    return f"{k} b={b} li={li}"
    return None


full = truth()
bounds = list(range(BLOCK, T + 1, BLOCK))
cat = lambda xs: np.concatenate(xs, 0)   # noqa: E731

# ---------------------------------------------------------------- ordering + views
s = SegmentStream(BLOCK, cat, num_layers=len(RATIOS))
bad = [views_ok(full, b, s.feed(b, segment(full, b))) for b in bounds]
check("A in-order feed reproduces every boundary view (pooled prefix grows, prev/kvwin per boundary)", not any(bad), str([x for x in bad if x][:3]))
check("B consumed list and next_b advance", s.consumed == bounds and s.next_b == bounds[-1] + BLOCK)
s2 = SegmentStream(BLOCK, cat)
try:
    s2.feed(2 * BLOCK, segment(full, 2 * BLOCK)); check("C out-of-order boundary refused", False)
except ValueError as e:
    check("C out-of-order boundary refused (ValueError)", "out of order" in str(e))
s3 = SegmentStream(BLOCK, cat)
seg = segment(full, BLOCK); seg.pop("layer_01.kvwin_256")
try:
    s3.feed(BLOCK, seg); check("D missing kvwin refused", False)
except KeyError as e:
    check("D missing tensor refused (KeyError)", "kvwin" in str(e))
s4 = SegmentStream(BLOCK, cat)
s4.feed(BLOCK, segment(full, BLOCK))
seg = segment(full, 2 * BLOCK); seg = {k: v for k, v in seg.items() if not k.startswith("layer_02.")}
try:
    s4.feed(2 * BLOCK, seg); check("E layer set change refused", False)
except KeyError as e:
    check("E layer set change refused (KeyError)", "layer set" in str(e))
seg = segment(full, BLOCK); seg["layer_00.kvwin_512"] = seg.pop("layer_00.kvwin_256")
try:
    SegmentStream(BLOCK, cat).feed(BLOCK, seg); check("F foreign boundary inside a segment refused", False)
except KeyError as e:
    check("F foreign boundary inside a segment refused (KeyError)", "boundary" in str(e))

# ---------------------------------------------------------------- manifest integrity
good = {"T": T, "stream": {"enabled": True, "segments": bounds, "emitted_T": bounds[-1], "error": None}}
info = s.check_manifest(good, T)
check("G check_manifest: clean manifest -> no missing, no extra", info["missing"] == [] and info["extra"] == [] and info["segments_manifest"] == N)
hole = {"T": T, "stream": {"segments": [b for b in bounds if b != 3 * BLOCK], "emitted_T": bounds[-1]}}
try:
    s.check_manifest(hole, T); check("H check_manifest: hole refused", False)
except ValueError as e:
    check("H check_manifest: hole in segments refused", "holes" in str(e))
try:
    s.check_manifest({"T": T, "stream": {"segments": bounds + [bounds[-1] + BLOCK]}}, T); check("I beyond-T refused", False)
except ValueError as e:
    check("I check_manifest: boundary beyond T refused", "beyond" in str(e))
short = SegmentStream(BLOCK, cat)
for b in bounds[:4]:
    short.feed(b, segment(full, b))
info = short.check_manifest(good, T)
check("J check_manifest: consumer behind the manifest reports the missing boundaries", info["missing"] == bounds[4:])

# ---------------------------------------------------------------- resume: a fresh consumer replays from the first segment
d = tempfile.mkdtemp(prefix="pd_stream_asm_")
for b in bounds:
    save_file(segment(full, b), os.path.join(d, seg_name(b)))
first = SegmentStream(BLOCK, cat)
for b in bounds[:5]:
    first.feed(b, load_file(os.path.join(d, seg_name(b))))
del first                                   # "the Mac process died" — its accumulators are gone
resumed = SegmentStream(BLOCK, cat)
bad = [views_ok(full, b, resumed.feed(b, load_file(os.path.join(d, seg_name(b))))) for b, name in ready_segments(os.listdir(d), BLOCK, BLOCK)]
check("K resume: a new consumer replays all segments from the share and rebuilds identical views", not any(bad) and resumed.consumed == bounds)
check("L ready_segments: contiguous run only (a missing middle file stops the run)",
      [b for b, _ in ready_segments([seg_name(b) for b in bounds if b != 4 * BLOCK], BLOCK, BLOCK)] == bounds[:3]
      and seg_boundary("seg_00000256.safetensors") == 256 and seg_boundary("layer_00.safetensors") is None)
shutil.rmtree(d)

# ---------------------------------------------------------------- consume_dir over a directory that fills up
d = tempfile.mkdtemp(prefix="pd_stream_asm_")
stored, acked = [], []
state = {"done": False}


def producer():
    for b in bounds:
        time.sleep(0.03)
        p = os.path.join(d, seg_name(b))
        save_file(segment(full, b), p + ".tmp"); os.replace(p + ".tmp", p)      # atomic, like the hook
    time.sleep(0.03)
    json.dump(good, open(os.path.join(d, "manifest.json"), "w"))
    open(os.path.join(d, "DONE"), "w").write("test")
    state["done"] = True


th = threading.Thread(target=producer); th.start()


def on_boundary(b, views):
    err = views_ok(full, b, views)
    if err:
        raise AssertionError(err)
    stored.append(b)


def manifest_done():
    if os.path.exists(os.path.join(d, "DONE")):
        return json.load(open(os.path.join(d, "manifest.json")))
    return None


t0 = time.time()
stream, man, info = consume_dir(d, BLOCK, load_file, cat, on_boundary, manifest_done=manifest_done, poll_s=0.01,
                                wait_for=lambda: not state["done"], ack=lambda b, name: acked.append(name))
th.join()
check("M consume_dir: stored every boundary in order as files appeared, then saw the sealed manifest",
      stored == bounds and man is not None and stream.check_manifest(man, T)["missing"] == [])
check("N consume_dir: acked every segment by name", acked == [seg_name(b) for b in bounds])
check("O consume_dir: first segment consumed before the producer finished (overlap, not a post-hoc pull)",
      info["t_first_segment"] is not None and info["t_first_segment"] - t0 < 0.03 * N)
shutil.rmtree(d)

# ---------------------------------------------------------------- consume_dir: sealed manifest lists a segment that never arrives
d = tempfile.mkdtemp(prefix="pd_stream_asm_")
for b in bounds[:6]:
    save_file(segment(full, b), os.path.join(d, seg_name(b)))
json.dump(good, open(os.path.join(d, "manifest.json"), "w")); open(os.path.join(d, "DONE"), "w").write("test")
stream, man, info = consume_dir(d, BLOCK, load_file, cat, lambda b, v: None, manifest_done=lambda: json.load(open(os.path.join(d, "manifest.json"))),
                                poll_s=0.01, wait_for=lambda: False)
info = stream.check_manifest(man, T)
check("P partial stream: consumer stops at the last present segment and check_manifest names the missing ones",
      stream.consumed == bounds[:6] and info["missing"] == bounds[6:])
shutil.rmtree(d)

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
