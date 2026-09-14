#!/usr/bin/env python3
"""engine_only.py — the prefill engine alone: the hook-off control of docs/RDMA.md.

Builds exactly the prompt bench_cold.py builds for a (--chars, --seed) pair — same stdlib files (taken from the same
`python3` bench_cold.py runs under), same marker, same question — renders it with the decoder's chat template (proven
token-identical to the front door's render) and sends the token ids to vLLM /v1/completions with max_tokens=1. With the
same seed as a bridged run, the engine sees the identical token sequence, so the two times compare directly.

Run in the oMLX venv (it has transformers), against an engine started with PD_HOOK=off:
    ~/omlx/bin/python bench/engine_only.py --chars 142000 --seed 7403 --url $PD_SPARK
"""
import argparse, glob, json, os, random, subprocess, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--chars", type=int, required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--url", default=os.environ.get("PD_SPARK"), required=not os.environ.get("PD_SPARK"))
ap.add_argument("--model", default=os.environ.get("PD_SPARK_MODEL", "qwen3-32b"))
ap.add_argument("--tokenizer", default=os.path.expanduser("~/models/Qwen3-32B"))
ap.add_argument("--stdlib-python", default="python3", help="the interpreter bench_cold.py runs under (its stdlib is the document)")
# Tokenizing needs the venv (transformers); sending does not. Split them when the venv's interpreter cannot reach the
# prefill box (macOS Local Network privacy is granted per interpreter):
#   ~/omlx/bin/python bench/engine_only.py --chars C --seed S --save-ids ids.json
#   python3 bench/engine_only.py --chars C --seed S --ids-file ids.json
ap.add_argument("--save-ids", help="tokenize only: write the token ids to this file and exit")
ap.add_argument("--ids-file", help="send only: read the token ids from this file (no transformers needed)")
a = ap.parse_args()

if a.ids_file:
    saved = json.load(open(a.ids_file))
    if saved.get("seed") != a.seed or saved.get("chars") != a.chars:
        raise SystemExit(f"{a.ids_file} holds seed {saved.get('seed')} chars {saved.get('chars')}, not {a.seed}/{a.chars}")
    ids = saved["ids"]
    body = json.dumps({"model": a.model, "prompt": ids, "max_tokens": 1, "temperature": 0}).encode()
    t0 = time.time()
    with urllib.request.urlopen(urllib.request.Request(a.url + "/v1/completions", body, {"Content-Type": "application/json"}), timeout=3600) as r:
        r.read()
    dt = time.time() - t0
    print(json.dumps({"leg": "engine_only", "seed": a.seed, "chars": a.chars, "tokens": len(ids), "t_request_s": round(dt, 2),
                      "tok_per_s": round(len(ids) / dt, 1), "stdlib": saved.get("stdlib")}))
    raise SystemExit(0)

stdlib = subprocess.check_output([a.stdlib_python, "-c", "import sysconfig; print(sysconfig.get_paths()['stdlib'])"], text=True).strip()
files = sorted(glob.glob(os.path.join(stdlib, "*.py"))); random.Random(a.seed).shuffle(files)
doc = []; n = 0
for f in files:
    try: s = open(f, encoding="utf-8", errors="ignore").read()
    except Exception: continue
    doc.append(f"\n\n### FILE {os.path.basename(f)} (seed {a.seed})\n" + s); n += len(s)
    if n >= a.chars: break
doc = "".join(doc)[:a.chars]
marker = f"ZEBRA-{a.seed:04d}-{random.Random(a.seed * 7).randint(1000, 9999)}"
doc = doc[:len(doc) // 2] + f"\n# The secret marker is {marker}.\n" + doc[len(doc) // 2:]
msgs = [{"role": "user", "content": doc + "\n\nAnswer in one line: what is the secret marker written in the middle of the document above?"}]

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.tokenizer)
ids = tok.encode(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
if a.save_ids:
    json.dump({"seed": a.seed, "chars": a.chars, "stdlib": stdlib, "ids": ids}, open(a.save_ids, "w"))
    print(json.dumps({"saved": a.save_ids, "seed": a.seed, "tokens": len(ids)}))
    raise SystemExit(0)

body = json.dumps({"model": a.model, "prompt": ids, "max_tokens": 1, "temperature": 0}).encode()
t0 = time.time()
with urllib.request.urlopen(urllib.request.Request(a.url + "/v1/completions", body, {"Content-Type": "application/json"}), timeout=3600) as r:
    r.read()
dt = time.time() - t0
print(json.dumps({"leg": "engine_only", "seed": a.seed, "chars": a.chars, "tokens": len(ids), "t_request_s": round(dt, 2),
                  "tok_per_s": round(len(ids) / dt, 1), "stdlib": stdlib}))
