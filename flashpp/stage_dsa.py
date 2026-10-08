"""flashpp stage for the FULL GLM-5.3 (model_type glm_moe_dsa: 78 MLA + DSA-indexer decoder layers, routed MoE).

Same contract as stage.py (Flash): a stage owns layers [start, end) and exactly their caches (per layer a
CacheList(latent KV, indexer keys)); only the hidden state crosses machines: [B, S, 6144] bf16 = 12 KB per token.
No linear-attention state anywhere, so rollback for drafting is a plain cache trim.

Weights come from mlx-community--GLM-5.3-indexerBF16-q4 (4-bit experts, BF16 attention + indexer); only the shards
holding this stage's tensors need to be present.
"""
from __future__ import annotations

import glob
import json
import re
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from omlx.utils.model_loading import _patch_mlx_lm_load_config, maybe_apply_pre_load_patches

_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _wanted(name: str, start: int, end: int, first: bool, last: bool) -> bool:
    if "mtp." in name:
        return False
    m = _LAYER.search(name)
    if m:
        return start <= int(m.group(1)) < end
    if "embed_tokens" in name:
        return first
    if name.endswith("norm.weight") or "lm_head" in name:
        return last
    return False


def needed_shards(model_dir: str, start: int, end: int, first: bool, last: bool) -> list[str]:
    idx = json.load(open(Path(model_dir) / "model.safetensors.index.json"))["weight_map"]
    return sorted({f for k, f in idx.items() if _wanted(k, start, end, first, last)})


class DsaStage:
    def __init__(self, model_dir: str, start: int, end: int, head: bool | None = None):
        model_dir = str(model_dir)
        cfg = json.load(open(Path(model_dir) / "config.json"))
        n = cfg["num_hidden_layers"]
        self.start, self.end = start, end
        self.first = start == 0
        self.last = (end == n) if head is None else head
        t0 = time.time()
        _patch_mlx_lm_load_config()
        maybe_apply_pre_load_patches(model_dir, None, for_vlm=False)
        self.model = self._load(model_dir)
        self.inner = self.model.model
        self.layers = self.inner.layers[start:end]
        if getattr(self.layers[0].self_attn, "skip_topk", False):
            raise ValueError(f"layer {start} shares the previous layer's top-k (IndexShare): start a stage on a layer "
                             f"that runs its own indexer (full layers: {self.full_layers()})")
        for i in range(len(self.inner.layers)):
            if not (start <= i < end):
                self.inner.layers[i] = None
        self.reset()
        self.load_s = time.time() - t0

    def _load(self, model_dir: str):
        from mlx_lm import utils as U

        present = {Path(p).name for p in glob.glob(str(Path(model_dir) / "model*.safetensors"))}
        start, end, first, last = self.start, self.end, self.first, self.last
        orig_glob, orig_load = U.glob.glob, U.mx.load

        def only_present(pattern, *a, **k):
            return [p for p in orig_glob(pattern, *a, **k) if Path(p).name in present]

        def only_mine(path, *a, **k):
            w = orig_load(path, *a, **k)
            return {k2: v for k2, v in w.items() if _wanted(k2, start, end, first, last)} if isinstance(w, dict) else w

        U.glob.glob, U.mx.load = only_present, only_mine
        try:
            model, _ = U.load_model(Path(model_dir), lazy=True, strict=False)
        finally:
            U.glob.glob, U.mx.load = orig_glob, orig_load
        own = [p for name, p in nn.utils.tree_flatten(model.parameters())
               if _wanted(name, start, end, first, last)]
        mx.eval(own)
        return model

    def full_layers(self):
        return [i for i, l in enumerate(self.inner.layers) if l is not None and not getattr(l.self_attn, "skip_topk", False)]

    def reset(self):
        from mlx_lm.models.cache import CacheList, KVCache

        # shared (IndexShare) layers run no indexer, so they get no indexer cache — same as the model's make_cache
        self.cache = [CacheList(KVCache()) if getattr(l.self_attn, "skip_topk", False) else CacheList(KVCache(), KVCache())
                      for l in self.layers]

    def embed(self, ids: mx.array) -> mx.array:
        return self.inner.embed_tokens(ids)

    def forward(self, h: mx.array) -> mx.array:
        from mlx_lm.models.base import create_attention_mask

        mask = create_attention_mask(h, self.cache[0][0] if self.cache else None, return_array=True)
        prev_topk = None   # a stage always starts on a full-indexer layer, so nothing crosses for this
        for layer, c in zip(self.layers, self.cache):
            h, prev_topk = layer(h, mask, c, prev_topk)
        return h

    def head(self, h: mx.array) -> mx.array:
        return self.model.lm_head(self.inner.norm(h[:, -1:, :]))[:, -1, :]
