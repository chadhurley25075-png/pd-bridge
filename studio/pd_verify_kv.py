#!/usr/bin/env python3
"""pd_verify_kv.py — end-to-end correctness check for the plain-attention bridge (R0 of docs/RDMA.md).

Sends the exact token ids of a request to the prefill engine with a pd_tag, pulls the capture, assembles it into
oMLX blocks in a SEPARATE directory (never the live decoder cache), then compares those blocks tensor by tensor
with the blocks oMLX wrote natively for the same prompt. Run the prompt natively through oMLX first.

vLLM CUDA and MLX Metal are not bit-identical even at bf16, so the verdict is a measured difference, not a hash.

    pd_verify_kv.py --ids ~/pd_verify/ids_901.json --native-cache ~/.omlx/cache --out ~/pd_verify/bridged_901
"""
import argparse, json, os, shutil, sys, threading, time, urllib.request, uuid
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _ls(base, sub):
    return json.load(urllib.request.urlopen(f"{base}/_ls/{sub}", timeout=10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--native-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.path.expanduser("~/models/Qwen3-32B"))
    ap.add_argument("--name", default="Qwen3-32B")
    ap.add_argument("--spark", default=os.environ.get("PD_SPARK"), required=not os.environ.get("PD_SPARK"))
    ap.add_argument("--spark-model", default="qwen3-32b")
    ap.add_argument("--share", default=os.environ.get("PD_SHARE"), required=not os.environ.get("PD_SHARE"))
    ap.add_argument("--pull", default=os.path.expanduser("~/pd_verify/pull"))
    a = ap.parse_args()

    ids = json.load(open(os.path.expanduser(a.ids)))
    tag = "vf" + uuid.uuid4().hex[:16]
    t0 = time.time(); eng = {}

    def _engine():
        body = json.dumps({"model": a.spark_model, "prompt": ids, "max_tokens": 1, "temperature": 0,
                           "kv_transfer_params": {"pd_tag": tag}}).encode()
        try:
            urllib.request.urlopen(urllib.request.Request(a.spark + "/v1/completions", body, {"Content-Type": "application/json"}), timeout=3600).read()
            eng["t"] = time.time() - t0
        except Exception as e:
            eng["err"] = repr(e)

    th = threading.Thread(target=_engine, daemon=True); th.start()
    while True:
        try:
            names = {e["name"] for e in _ls(a.share, tag)}
        except Exception:
            names = set()
        if "DONE" in names:
            break
        if "err" in eng:
            sys.exit(f"engine failed: {eng['err']}")
        if eng.get("t") is not None and time.time() - t0 > eng["t"] + 60:
            sys.exit(f"engine returned but capture {tag} never committed")
        time.sleep(0.2)
    th.join(timeout=60)
    man = json.load(urllib.request.urlopen(f"{a.share}/{tag}/manifest.json", timeout=30))
    print(f"capture {tag}: engine {eng.get('t', -1):.1f}s, DONE +{time.time() - t0:.1f}s, T={man['T']}/{len(ids)} blocks={man['blocks']} "
          f"complete={man['complete']} d2h={man['t_d2h_s']}s write={man['t_write_s']}s")

    local = Path(a.pull) / tag; local.mkdir(parents=True, exist_ok=True)
    for e in _ls(a.share, tag):
        if e["dir"] or e["name"] == "DONE":
            continue
        with urllib.request.urlopen(f"{a.share}/{tag}/{e['name']}", timeout=600) as r, open(local / e["name"], "wb") as f:
            shutil.copyfileobj(r, f, 1 << 24)

    out = Path(os.path.expanduser(a.out))
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    from mlx_lm import load
    from pd_assemble_kv import assemble_and_write_kv
    model, _ = load(a.model, lazy=True)
    paths, info = assemble_and_write_kv(str(local), ids, model, a.name, str(out))
    shutil.rmtree(local, ignore_errors=True)
    print("assembled:", json.dumps(info))

    from verify_blocks import compare_files
    native = Path(os.path.expanduser(a.native_cache))
    worst, n_ok, n_cmp, meta_bad = 0.0, 0, 0, []
    for p in paths:
        ref = native / p.name[0] / p.name
        if not ref.exists():
            print(f"  no native block for {p.name[:16]} — run the prompt through oMLX first"); continue
        ok, w, md, _ = compare_files(ref, p, quiet=True)
        n_cmp += 1; worst = max(worst, w); n_ok += ok
        if md:
            meta_bad.append((p.name[:16], md))
    print(f"compared {n_cmp}/{len(paths)} blocks: bit-exact {n_ok}, worst max|native-bridged| = {worst:.3e}, metadata diffs: {meta_bad or 'none'}")


if __name__ == "__main__":
    main()
