#!/usr/bin/env python3
"""pd_capture_mlx.py — DV4-Flash heterogeneous P/D, Studio-side capture harness (M0).
Loads the MLX model through oMLX's own deepseek_v4 patch, tokenizes the reference prompt
with the SAME tokenizer/template oMLX uses, prefills in oMLX-sized chunks (2048), and
captures each layer's attention input (attn_norm(attn_hc(h))) — the tensor every cache
(rotating window, compressor pool, indexer pool) is computed from. Writes:
  <out>/attn_inputs.safetensors  keys layer_00..layer_42, each [T,4096] bf16
  <out>/token_ids.json           the exact ids oMLX would hash
  <out>/caches_ref.pkl-less: the final cache objects are kept in-process only; the block
  writer consumes cache objects, so this script also saves each layer's cache
  `state`/`meta_state` to <out>/cache_state.safetensors for inspection.
Usage: pd_capture_mlx.py --model PATH --out DIR [--words 22000] [--chunk 2048]
"""
import argparse, json, os, random, sys, time
import mlx.core as mx
import mlx.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--words", type=int, default=22000); ap.add_argument("--chunk", type=int, default=2048)
ap.add_argument("--prompt-file", default=None, help="optional: raw text prompt instead of the synthetic note")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
log = open(os.path.join(args.out, "capture.log"), "a")
def L(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()

from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch
apply_deepseek_v4_patch()
from mlx_lm import load
t0 = time.time(); model, tok = load(args.model); mx.eval(model.parameters()); L(f"model loaded in {time.time()-t0:.1f}s")
import mlx_lm.models.deepseek_v4 as dv4mod
Block = dv4mod.DeepseekV4Block

# ---- prompt + ids (same recipe as the S1 reference bench, seed 11) ----
if args.prompt_file:
    p = open(args.prompt_file).read()
else:
    random.seed(11)
    words = "river stone lantern harbor beacon quiet ember meadow signal orbit copper willow thunder cobalt velvet garden anchor summit silver".split()
    body = " ".join(random.choice(words) for _ in range(args.words))
    p = f"Here is a long note:\n{body}\n\nSummarize the note in one sentence."
msgs = [{"role": "user", "content": p}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
if isinstance(ids, str): ids = tok.encode(ids)
ids = list(map(int, ids)); L(f"token ids: {len(ids)} first8={ids[:8]} last8={ids[-8:]}")
json.dump({"token_ids": ids, "n": len(ids), "template_kwargs": {}}, open(os.path.join(args.out, "token_ids.json"), "w"))

# ---- capture hook: re-implement the block call with attn_input capture (same ops) ----
from mlx_lm.models.hyper_connection import hc_expand  # registered by the patch
captured = {}  # layer_idx -> list of [L,4096] bf16 chunks
_orig_call = Block.__call__
def _capturing_call(self, h, mask, cache, input_ids, *, _standard_mask=False):
    residual = h
    x, post, comb = self.attn_hc(h)
    attn_input = self.attn_norm(x)
    captured.setdefault(self._pd_idx, []).append(attn_input[0].astype(mx.bfloat16))
    x = self.attn(attn_input, mask=mask, cache=cache, _standard_mask=_standard_mask)
    h = hc_expand(x, residual, post, comb)
    residual = h
    x, post, comb = self.ffn_hc(h)
    x = self.ffn_norm(x)
    x = self.ffn(x, input_ids)
    return hc_expand(x, residual, post, comb)
for i, layer in enumerate(model.model.layers): layer._pd_idx = i
Block.__call__ = _capturing_call

# ---- prefill in oMLX-sized chunks ----
cache = model.make_cache()
arr = mx.array(ids)[None]
t0 = time.time(); done = 0
while done < len(ids):
    n = min(args.chunk, len(ids) - done)
    out = model(arr[:, done:done+n], cache=cache)
    mx.eval(out)
    for i in captured: mx.eval(captured[i][-1])
    done += n
    L(f"prefill {done}/{len(ids)}  {(time.time()-t0):.1f}s  {done/(time.time()-t0):.0f} tok/s")
L(f"prefill done: {len(ids)} tokens in {time.time()-t0:.1f}s")

# ---- save captures ----
tensors = {f"layer_{i:02d}": mx.concatenate(captured[i], axis=0) for i in sorted(captured)}
mx.save_safetensors(os.path.join(args.out, "attn_inputs.safetensors"), tensors, metadata={"n_tokens": str(len(ids)), "chunk": str(args.chunk)})
sz = os.path.getsize(os.path.join(args.out, "attn_inputs.safetensors")) / 1e9
L(f"saved attn_inputs.safetensors {sz:.2f} GB, layers={len(tensors)}, shape0={tensors['layer_00'].shape}")

# ---- save final cache state for inspection (state + meta_state per layer / sub) ----
st = {}; meta = {}
def dump(prefix, c):
    if hasattr(c, "caches"):  # CacheList
        for j, sub in enumerate(c.caches): dump(f"{prefix}_sub_{j}", sub)
        return
    s = c.state
    if not isinstance(s, (tuple, list)): s = (s,)
    for k, e in enumerate(s):
        if isinstance(e, mx.array):
            if e.size == 0: meta[f"{prefix}_state_{k}_zero_dim"] = ",".join(map(str, e.shape))
            else: st[f"{prefix}_state_{k}"] = e
        else: meta[f"{prefix}_state_{k}"] = str(e)
    ms = getattr(c, "meta_state", None)
    if ms is not None: meta[f"{prefix}_meta_state"] = json.dumps(list(map(str, ms)) if isinstance(ms, (tuple, list)) else str(ms))
for i, c in enumerate(cache): dump(f"layer_{i}", c)
mx.save_safetensors(os.path.join(args.out, "cache_state.safetensors"), st, metadata=meta)
L(f"saved cache_state.safetensors: {len(st)} arrays, {len(meta)} meta entries")
L("DONE")
