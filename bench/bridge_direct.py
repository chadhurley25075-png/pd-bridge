#!/usr/bin/env python3
"""bridge_direct.py — one rdma4 bridged request without the front door (docs/RDMA.md, O2b).

Does exactly what studio/pd_front.py does for PD_TRANSPORT=rdma4, stdlib only, so it runs under any python3 that can
reach the prefill box (macOS Local Network privacy is granted per interpreter):
  1. POST the token ids to vLLM with kv_transfer_params {pd_tag, omlx_model, omlx_block} (the connector hashes the ids
     exactly as oMLX will and `pd_rdma recvd --omlx-cache` lands the blocks in the decoder's cache during prefill);
  2. wait for <pull>/<tag>/DONE, read the manifest, move tail.safetensors to <cache>/pd_tail/<sha256(ids)>;
  3. send the same chat messages to oMLX, streaming, and time the first token.
TTFT here = engine + DONE + oMLX first token, measured from the POST, i.e. the front door's TTFT minus its render.

Token ids come from `engine_only.py --save-ids` (same render as the front door, proven token-identical).
    python3 bench/bridge_direct.py --ids-file ids.json --chars 142000 --seed 7401 --max-tokens 16 --stage
"""
import argparse, glob, hashlib, json, os, random, shutil, struct, sysconfig, threading, time, urllib.request, uuid

ap = argparse.ArgumentParser()
ap.add_argument("--ids-file", required=True)
ap.add_argument("--chars", type=int, required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--max-tokens", type=int, default=16)
ap.add_argument("--spark", default=os.environ.get("PD_SPARK"), required=not os.environ.get("PD_SPARK"))
ap.add_argument("--spark-model", default=os.environ.get("PD_SPARK_MODEL"), required=not os.environ.get("PD_SPARK_MODEL"))
ap.add_argument("--omlx", default=os.environ.get("PD_OMLX", "http://127.0.0.1:8011"))
ap.add_argument("--omlx-model", default=os.environ.get("PD_MODEL_NAME"), required=not os.environ.get("PD_MODEL_NAME"))
ap.add_argument("--pull", default=os.path.expanduser(os.environ.get("PD_PULL_DIR", "~/pd_pull")))
ap.add_argument("--cache", default=os.path.expanduser(os.environ.get("PD_CACHE_DIR", "~/.omlx/cache")))
ap.add_argument("--model-dir", default=os.environ.get("PD_MODEL"), help="decoder model dir: its config.json sizes the /pd/stage buffers")
ap.add_argument("--block", type=int, default=256)
ap.add_argument("--grace", type=float, default=30.0)
ap.add_argument("--no-tail", action="store_true", help="A/B control: drop the shipped tail rows, the decoder recomputes the tail")
ap.add_argument("--stage", action="store_true", help="O2c: POST the block hashes to oMLX /pd/stage before prefill (needs PD_OMLX_STAGE=1)")
a = ap.parse_args()

saved = json.load(open(a.ids_file))
if saved.get("seed") != a.seed or saved.get("chars") != a.chars:
    raise SystemExit(f"{a.ids_file} is seed {saved.get('seed')} chars {saved.get('chars')}, not {a.seed}/{a.chars}")
ids = saved["ids"]
T = len(ids)

# the same document bench_cold.py / engine_only.py build (stdlib of THIS interpreter must match saved["stdlib"])
if saved.get("stdlib") and os.path.realpath(saved["stdlib"]) != os.path.realpath(sysconfig.get_paths()["stdlib"]):
    raise SystemExit(f"run under the interpreter whose stdlib is {saved['stdlib']}")
files = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py"))); random.Random(a.seed).shuffle(files)
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

tag = "pd" + uuid.uuid4().hex[:16]
local = os.path.join(a.pull, tag)
eng = {}
t0 = time.time()

stage = None
if a.stage:
    # O2c: tell the oMLX hook which blocks are coming, so it allocates and fills the cache while the Spark prefills
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "spark"))
    import pd_omlx_block
    hashes = [h.hex() for h in pd_omlx_block.chain_hashes(ids, a.omlx_model, a.block)][:T // a.block]
    if not a.model_dir:
        raise SystemExit("--stage needs --model-dir (or PD_MODEL): the stage is sized from the decoder model's config.json")
    cfg = json.load(open(os.path.join(os.path.expanduser(a.model_dir), "config.json")))
    geometry = {"layers": cfg["num_hidden_layers"], "kv_heads": cfg["num_key_value_heads"],
                "head_dim": cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]}
    body = json.dumps({"hashes": hashes, **geometry, "block": a.block, "reserve": 1024}).encode()
    stage = json.load(urllib.request.urlopen(urllib.request.Request(a.omlx + "/pd/stage", body, {"Content-Type": "application/json"}), timeout=30))


def _engine():
    body = json.dumps({"model": a.spark_model, "prompt": ids, "max_tokens": 1, "temperature": 0,
                       "kv_transfer_params": {"pd_tag": tag, "omlx_model": a.omlx_model, "omlx_block": a.block}}).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(a.spark + "/v1/completions", body, {"Content-Type": "application/json"}), timeout=3600).read()
        eng["t"] = time.time() - t0
    except Exception as e:
        eng["err"] = repr(e)


th = threading.Thread(target=_engine, daemon=True); th.start()
while not os.path.isfile(os.path.join(local, "DONE")):
    if "err" in eng: raise SystemExit("spark engine: " + eng["err"])
    if "t" in eng and time.time() - t0 - eng["t"] > a.grace:
        raise SystemExit(f"engine returned {a.grace:.0f}s ago and no DONE for {tag}")
    time.sleep(0.02)
t_done = time.time() - t0
man = json.load(open(os.path.join(local, "manifest.json")))
n_full = (T // a.block)
tail_rows = int(man.get("tail_rows_shipped") or 0)
tail_landed = False
src = os.path.join(local, "tail.safetensors")
if tail_rows and not a.no_tail and os.path.isfile(src) and n_full * a.block + tail_rows == T - 1:
    os.makedirs(os.path.join(a.cache, "pd_tail"), exist_ok=True)
    os.replace(src, os.path.join(a.cache, "pd_tail", hashlib.sha256(struct.pack(f"<{T}i", *ids)).hexdigest() + ".safetensors"))
    tail_landed = True
shutil.rmtree(local, ignore_errors=True)
t_handoff = time.time() - t0

# decoder: stream, first token of any kind = TTFT (bench_cold.py semantics)
body = json.dumps({"model": a.omlx_model, "messages": msgs, "max_tokens": a.max_tokens, "temperature": 0, "stream": True,
                   "stream_options": {"include_usage": True}}).encode()
first = None; out = []; usage = None
t_req = time.time()
with urllib.request.urlopen(urllib.request.Request(a.omlx + "/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=3600) as r:
    for line in r:
        if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]": continue
        try: d = json.loads(line[6:])
        except Exception: continue
        if d.get("usage"): usage = d["usage"]
        ch = d.get("choices") or []
        delta = ch[0].get("delta", {}) if ch else {}
        c = delta.get("content"); rc = delta.get("reasoning_content") or delta.get("reasoning")
        if (c or rc) and first is None: first = time.time() - t0
        if c: out.append(c)
th.join(timeout=60)
ptd = (usage or {}).get("prompt_tokens_details") or {}
ans = "".join(out).strip()
print(json.dumps({"leg": "bridge_direct", "seed": a.seed, "chars": a.chars, "tokens": T, "tag": tag,
                  "ttft_s": round(first or -1, 2), "t_engine_s": round(eng.get("t", -1), 2), "t_done_s": round(t_done, 2),
                  "t_handoff_s": round(t_handoff, 2), "decoder_after_handoff_s": round((first or 0) - t_handoff, 2),
                  "prompt_tokens": (usage or {}).get("prompt_tokens"), "cached_tokens": ptd.get("cached_tokens"),
                  "tail_rows": tail_rows, "tail_landed": tail_landed, "marker": marker, "found": marker in ans,
                  "answer": ans[:120],
                  "spark": {k: man.get(k) for k in ("transport", "complete", "blocks", "blocks_sent", "T", "error", "t_copy_s",
                                                    "t_wire_s", "t_ack_s", "t_sender_lag_s", "tail_rows_not_shipped")}}))
