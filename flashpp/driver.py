"""flashpp driver: push a prompt through an ordered list of stage servers, then decode greedily.

usage: python driver.py host:port[,host:port...] <n_new_tokens> [--out result.json]
The first stage receives token ids; each later stage receives the previous stage's hidden state.
"""
import json
import socket
import sys
import time

import mlx.core as mx
import numpy as np

import net

stages = [s.split(":") for s in sys.argv[1].split(",")]
n_new = int(sys.argv[2])
out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
PROMPT = [151331, 151333, 9707, 11, 1246, 525, 498, 30, 3555, 374, 279, 6722, 315, 9625, 30, 13, 576, 4226, 1879, 13]
tok = None
if "--text" in sys.argv:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(sys.argv[sys.argv.index("--tok") + 1])
    msgs = [{"role": "user", "content": sys.argv[sys.argv.index("--text") + 1]}]
    PROMPT = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, enable_thinking=False)
    if hasattr(PROMPT, "input_ids"):
        PROMPT = PROMPT["input_ids"]
    PROMPT = [int(t) for t in PROMPT]

socks = []
for host, port in stages:
    s = socket.create_connection((host, int(port)))
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    net.send(s, {"op": "reset"}); net.recv(s)
    socks.append(s)


def step(ids):
    x, kind, per = mx.array([ids], dtype=mx.int32), "ids", []
    for s in socks:
        t0 = time.perf_counter()
        net.send(s, {"op": "forward", "kind": kind}, x)
        hdr, x = net.recv(s)
        per.append(((time.perf_counter() - t0) * 1e3, hdr["ms"]))
        kind = "hidden"
    return hdr["top"], np.array(x), per


t0 = time.perf_counter()
top, logits0, per = step(PROMPT)
prefill_s = time.perf_counter() - t0
tokens, tops, timings = [top[0]], [top], [per]
t1 = time.perf_counter()
for _ in range(n_new - 1):
    top, _, per = step([tokens[-1]])
    tokens.append(top[0]); tops.append(top); timings.append(per)
decode_s = time.perf_counter() - t1
for s in socks:
    net.send(s, {"op": "bye"}); net.recv(s)

res = {"stages": sys.argv[1], "tokens": tokens, "top5": tops, "prefill_s": prefill_s,
       "decode_tok_s": (n_new - 1) / decode_s if n_new > 1 else None,
       "per_stage_ms_last": timings[-1], "first_logits_head": logits0[0][:8].tolist()}
if tok is not None:
    res["text"] = tok.decode(tokens)
    res["prompt_tokens"] = len(PROMPT)
print(json.dumps({k: v for k, v in res.items() if k not in ("top5", "first_logits_head")}, indent=1))
if out:
    json.dump(res, open(out, "w"))
    np.save(out.replace(".json", "_logits0.npy"), logits0)
