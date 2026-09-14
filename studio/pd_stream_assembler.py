#!/usr/bin/env python3
"""pd_stream_assembler.py — consume a STREAMING pooled capture (docs/STREAMING-CAPTURE.md) boundary by boundary.

Backend-agnostic: no mlx import here. The caller supplies `load(path) -> {key: array}` and
`concat(list_of_arrays) -> array`, so the same ordering/resume/integrity logic runs under numpy in the CPU
tests and under mlx.core inside pd_front / pd_assemble_blocks on the Mac.

Segment file <stamp>/seg_<b:08d>.safetensors carries, for every layer li, the tensors that the decoder state
at boundary b needs beyond what earlier segments carried (spark/capture_sitecustomize_v3.py::_emit_segment):
  layer_XX.kvwin_<b>, layer_XX.pooled_<b> (rows [prev_b//ratio, b//ratio)), layer_XX.prev_kv_<b>/prev_gate_<b>
  (ratio 4), layer_XX.idx_pooled_<b> / idx_prev_kv_<b> / idx_prev_gate_<b>.

SegmentStream.feed(b, tensors) returns a per-layer dict in the SAME shape the one-shot layer files have
(kvwin_<b>, pooled = the whole prefix so far, prev_kv_<b>, prev_gate_<b>, idx_*), so pd_assemble_blocks._set_layer
works unchanged on it. Boundaries must be fed in order; a gap is refused (the consumer waits for the missing
segment instead of guessing). `pooled` is the running prefix concatenated once per boundary — O(b) per boundary,
the same order of cost as the block writer's own per-boundary snapshot, and it overlaps the Spark prefill.
"""
import json, os, re, time

SEG_RE = re.compile(r"^seg_(\d{8})\.safetensors$")


def seg_name(b):
    return f"seg_{b:08d}.safetensors"


def seg_boundary(name):
    m = SEG_RE.match(name)
    return int(m.group(1)) if m else None


def parse_key(key):
    """'layer_07.idx_prev_kv_4096' -> (7, 'idx_prev_kv', 4096); None for anything else."""
    m = re.match(r"^layer_(\d+)\.(.+)_(\d+)$", key)
    if not m:
        return None
    return int(m.group(1)), m.group(2), int(m.group(3))


class SegmentStream:
    """Ordered consumer of streamed boundaries with per-layer pooled accumulators."""

    def __init__(self, block, concat, num_layers=None):
        self.block = int(block)
        self.concat = concat
        self.num_layers = num_layers
        self.next_b = self.block
        self.consumed = []                 # boundaries fed, in order
        self.acc = {}                      # li -> {"pooled": [chunks], "idx_pooled": [chunks]}
        self.prefix = {}                   # li -> {"pooled": array | None, "idx_pooled": array | None}
        self.layers_seen = None

    # -- feeding -------------------------------------------------------------------------------
    def feed(self, b, tensors):
        """Ingest one segment. Returns {li: view_dict} for pd_assemble_blocks._set_layer(…, view, b, str(b)).
        Raises ValueError on an out-of-order boundary, KeyError on a segment missing a layer/tensor."""
        if b != self.next_b:
            raise ValueError(f"segment {b} out of order: expected {self.next_b} (consumed {len(self.consumed)})")
        per = {}
        for k, v in tensors.items():
            p = parse_key(k)
            if p is None:
                continue
            li, kind, kb = p
            if kb != b:
                raise KeyError(f"{k}: boundary {kb} inside segment {b}")
            per.setdefault(li, {})[kind] = v
        if not per:
            raise KeyError(f"segment {b} carries no layer tensors")
        layers = sorted(per)
        if self.layers_seen is None:
            self.layers_seen = layers
            if self.num_layers is not None and len(layers) != self.num_layers:
                raise KeyError(f"segment {b}: {len(layers)} layers, expected {self.num_layers}")
        elif layers != self.layers_seen:
            raise KeyError(f"segment {b}: layer set changed ({len(layers)} vs {len(self.layers_seen)})")
        views = {}
        for li in layers:
            d = per[li]
            if "kvwin" not in d:
                raise KeyError(f"segment {b} layer {li}: kvwin missing")
            view = {f"kvwin_{b}": d["kvwin"]}
            a = self.acc.setdefault(li, {"pooled": [], "idx_pooled": []})
            pf = self.prefix.setdefault(li, {"pooled": None, "idx_pooled": None})
            for kind in ("pooled", "idx_pooled"):
                if kind in d:
                    parts = ([pf[kind]] if pf[kind] is not None else []) + [d[kind]]
                    pf[kind] = parts[0] if len(parts) == 1 else self.concat(parts)
                    view[kind] = pf[kind]
            for kind in ("prev_kv", "prev_gate", "idx_prev_kv", "idx_prev_gate"):
                if kind in d:
                    view[f"{kind}_{b}"] = d[kind]
            # a compressor layer must carry pooled rows whenever the segment says it has prev rows
            if ("prev_kv" in d) != ("prev_gate" in d) or ("idx_prev_kv" in d) != ("idx_prev_gate" in d):
                raise KeyError(f"segment {b} layer {li}: prev_kv/prev_gate pair incomplete")
            views[li] = view
        self.consumed.append(b)
        self.next_b = b + self.block
        return views

    # -- integrity -----------------------------------------------------------------------------
    def check_manifest(self, man, T):
        """Compare what we consumed against the sealed manifest. Returns an info dict; raises on a lie
        (manifest lists a boundary we never saw, or claims a boundary beyond T)."""
        st = man.get("stream") or {}
        segs = list(st.get("segments") or [])
        if segs != sorted(segs) or any(s % self.block for s in segs):
            raise ValueError(f"manifest segments not block-ordered: {segs[:5]}…")
        if segs and segs[-1] > T:
            raise ValueError(f"manifest emitted_T {segs[-1]} beyond request T={T}")
        expected = list(range(self.block, (segs[-1] if segs else 0) + 1, self.block))
        if segs != expected:
            raise ValueError(f"manifest segments have holes: {len(segs)} listed, {len(expected)} expected up to {segs[-1] if segs else 0}")
        missing = [s for s in segs if s not in self.consumed]
        extra = [c for c in self.consumed if c not in segs]
        return {"segments_manifest": len(segs), "segments_consumed": len(self.consumed), "emitted_T": st.get("emitted_T"),
                "missing": missing, "extra": extra, "stream_error": st.get("error"), "T_manifest": man.get("T")}


def ready_segments(names, next_b, block):
    """From a directory listing, the contiguous run of segment names starting at next_b (in order)."""
    have = {}
    for n in names:
        b = seg_boundary(n)
        if b is not None:
            have[b] = n
    out = []
    b = next_b
    while b in have:
        out.append((b, have[b]))
        b += block
    return out


def consume_dir(cap_dir, block, load, concat, on_boundary, manifest_done=None, poll_s=0.2, wait_for=None,
                num_layers=None, log=None, ack=None):
    """Drive a SegmentStream over a LOCAL directory that fills up over time (files appear as the Spark
    emits them; a puller copies them here). `on_boundary(b, views)` builds+stores the decoder state.
    `wait_for()` returns False when nothing more will arrive (DONE seen or engine failed); until then we poll.
    `manifest_done()` -> manifest dict once DONE is present (or None). `ack(b, name)` after a boundary is stored.
    Returns (stream, manifest, info)."""
    s = SegmentStream(block, concat, num_layers=num_layers)
    man = None
    t_first = None
    while True:
        try:
            names = os.listdir(cap_dir)
        except FileNotFoundError:
            names = []
        prog = False
        for b, name in ready_segments(names, s.next_b, block):
            path = os.path.join(cap_dir, name)
            views = s.feed(b, load(path))
            if t_first is None:
                t_first = time.time()
            on_boundary(b, views)
            if ack is not None:
                try:
                    ack(b, name)
                except Exception as e:
                    if log: log(f"stream: ack {name} failed ({e!r}) — continuing")
            prog = True
        if man is None and manifest_done is not None:
            man = manifest_done()
        if man is not None:
            # the manifest is sealed; anything it lists that we have not consumed must still be on disk
            st = man.get("stream") or {}
            pending = [b for b in (st.get("segments") or []) if b not in s.consumed]
            if not pending:
                break
            if not prog:
                names_now = set(os.listdir(cap_dir)) if os.path.isdir(cap_dir) else set()
                if seg_name(pending[0]) not in names_now and (wait_for is None or not wait_for()):
                    break                    # sealed, listed, never arrived: the caller sees it in check_manifest
        elif not prog and wait_for is not None and not wait_for():
            break
        if not prog:
            time.sleep(poll_s)
    return s, man, {"t_first_segment": t_first}


if __name__ == "__main__":
    import argparse
    import numpy as np
    from safetensors.numpy import load_file
    ap = argparse.ArgumentParser(description="list a streamed capture's segments and check them against its manifest")
    ap.add_argument("--cap", required=True); ap.add_argument("--block", type=int, default=2048); ap.add_argument("--T", type=int, required=True)
    a = ap.parse_args()
    s = SegmentStream(a.block, lambda xs: np.concatenate(xs, 0))
    for b, name in ready_segments(os.listdir(a.cap), a.block, a.block):
        s.feed(b, load_file(os.path.join(a.cap, name)))
    man = json.load(open(os.path.join(a.cap, "manifest.json")))
    print(json.dumps(s.check_manifest(man, a.T), indent=1))
