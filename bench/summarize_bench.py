#!/usr/bin/env python3
"""summarize_bench.py — one line per bench_cold.py row, with the three fields that make a row a NUMBER:
verdict · transport · cached_tokens (bench/BENCHMARK-PROTOCOL.md). Reads any file that contains bench_cold JSON lines
(a bench log, a .jsonl, the run_r0.sh wrapper format {"leg":…,"result":{…}}).
usage: summarize_bench.py <file> [<file>…]   |   some_command | summarize_bench.py -"""
import json, sys


def rows(fh):
    for line in fh:
        line = line.strip()
        if not (line.startswith("{") and '"ttft_s"' in line):
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        leg = d.get("leg")
        if "result" in d and isinstance(d["result"], dict):
            d = d["result"]
        d["_leg"] = leg
        yield d


def describe(d):
    b = d.get("bridge") if isinstance(d.get("bridge"), dict) else {}
    verdict = d.get("verdict")
    if verdict is None:                                     # rows from a bench_cold older than 2026-09-14
        if not d.get("bridge"): verdict = "native"
        elif b.get("bridge_error"): verdict = "declined: " + str(b.get("bridge_error"))[:60]
        elif b.get("skipped"): verdict = "skipped: " + str(b.get("why", ""))[:40]
        else: verdict = b.get("verdict") or "bridged?"
    transport = d.get("transport") or b.get("transport") or "-"
    toks = d.get("prompt_tokens") or b.get("tokens")
    cached = d.get("cached_tokens")
    eng = b.get("t_engine")
    return (f"{(d.get('_leg') or ''):>7} {str(toks):>9} tok  ttft {d.get('ttft_s'):>8}s  engine {str(eng):>7}s  "
            f"cached {str(cached):>9}  {transport:<6} {'stream' if (d.get('stream') or b.get('stream')) else '      '} "
            f"{'HIT ' if d.get('found') else 'MISS'}  {verdict}")


if __name__ == "__main__":
    files = sys.argv[1:] or ["-"]
    n = 0
    print(f"{'leg':>7} {'tokens':>9}      {'ttft':>8}   {'engine':>7}   {'cached_tok':>9}  wire   stream marker verdict")
    for f in files:
        fh = sys.stdin if f == "-" else open(f)
        for d in rows(fh):
            print(describe(d)); n += 1
    if not n:
        print("(no bench_cold rows found)", file=sys.stderr); sys.exit(1)
