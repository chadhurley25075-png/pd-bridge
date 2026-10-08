"""flashpp stage: one contiguous slice of GLM-5.3-Flash's decoder, runnable on MLX-CUDA (Spark) or MLX-Metal (Studio).

A stage owns layers [start, end) and the caches for exactly those layers (KDA state for linear-attention layers,
KV + index pool for sparse-attention layers). Nothing but the hyper-connection hidden state crosses machines:
shape [B, S, hc_mult, hidden] = [1, S, 4, 4096] bf16 (32 KB per token).

The first stage also owns the token embedding; the last stage owns the final norm and lm_head.

Weights: only the safetensors shards that hold this stage's tensors need to be on disk (`needed_shards`), so a
machine never stores layers it does not run.
"""
from __future__ import annotations

import glob
import json
import re
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from omlx.patches.mlx_vlm_glm5_next_compat import apply_mlx_vlm_glm5_next_compat_patch
from omlx.utils.model_loading import _patch_mlx_lm_load_config, maybe_apply_pre_load_patches

apply_mlx_vlm_glm5_next_compat_patch()
_patch_mlx_lm_load_config()

from mlx_vlm.models.base import create_attention_mask, create_ssm_mask  # noqa: E402

_LAYER = re.compile(r"\.layers\.(\d+)\.")
_TOP = ("embed_tokens", ".norm.weight", "lm_head")


def _tensor_wanted(name: str, start: int, end: int, first: bool, last: bool) -> bool:
    if "mtp." in name or "vision" in name or "visual" in name:
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
    """The shard files a stage needs, from the checkpoint's index."""
    idx = json.load(open(Path(model_dir) / "model.safetensors.index.json"))["weight_map"]
    return sorted({f for k, f in idx.items() if _tensor_wanted(k, start, end, first, last)})


class Stage:
    def __init__(self, model_dir: str, start: int, end: int, n_layers_total: int = 45, head: bool | None = None):
        self.start, self.end = start, end
        self.first = start == 0
        self.last = (end == n_layers_total) if head is None else head   # head=True lets a truncated test end early
        t0 = time.time()
        model_dir = str(model_dir)
        maybe_apply_pre_load_patches(model_dir, None, for_vlm=True)
        self.model = self._load(model_dir)
        self.lm = self.model.language_model
        self.inner = self.lm.model
        self.layers = self.inner.layers[start:end]
        # drop every layer this stage does not own, so nothing else can ever be materialized here
        for i in range(len(self.inner.layers)):
            if not (start <= i < end):
                self.inner.layers[i] = None
        self.cache = None
        self.reset()
        self.load_s = time.time() - t0

    def _load(self, model_dir: str):
        """mlx-vlm's loader, but over only the shards present for this stage (non-strict)."""
        from mlx_vlm import utils as U

        present = {Path(p).name for p in glob.glob(str(Path(model_dir) / "*.safetensors"))}
        orig_glob = U.glob.glob

        def only_present(pattern, *a, **k):
            return [p for p in orig_glob(pattern, *a, **k) if Path(p).name in present]

        orig_load = U._load_safetensors
        start, end, first, last = self.start, self.end, self.first, self.last

        def only_mine(path):
            # keep only this stage's tensors: a shard can hold part of a neighbour's layer, and sanitize()
            # stacks MoE experts per layer, so a half-present layer must never reach it
            return {k: v for k, v in orig_load(path).items() if _tensor_wanted(k, start, end, first, last)}

        U.glob.glob = only_present
        U._load_safetensors = only_mine
        try:
            model = U.load_model(Path(model_dir), lazy=True, strict=False)
        finally:
            U.glob.glob = orig_glob
            U._load_safetensors = orig_load
        # materialize exactly this stage's parameters now (fail fast if a shard is missing)
        own = [p for n, p in nn.utils.tree_flatten(model.language_model.parameters())
               if _tensor_wanted("language_model." + n.replace("model.", "model.", 1), self.start, self.end, self.first, self.last)]
        mx.eval(own)
        return model

    def reset(self):
        full = self.lm.make_cache() if all(l is not None for l in self.inner.layers) else self._make_cache()
        self.cache = full

    def _make_cache(self):
        from mlx_vlm.models.cache import ArraysCache, CacheList, KVCache
        from mlx_lm.models.cache import PoolingCache

        caches = []
        for layer in self.layers:
            if layer.is_linear:
                caches.append(ArraysCache(size=2))
            else:
                caches.append(CacheList(KVCache(), PoolingCache(layer.self_attn.indexer.index_kpool)))
        return caches

    def embed(self, ids: mx.array) -> mx.array:
        h = self.inner.embed_tokens(ids)
        return mx.contiguous(mx.broadcast_to(h[:, :, None, :], (h.shape[0], h.shape[1], self.inner.hc_mult, h.shape[2])))

    def forward(self, h: mx.array) -> mx.array:
        """Run this stage's layers on hidden state h [B,S,hc,D]; updates this stage's caches."""
        flat = h[:, :, 0, :]  # masks only need B and S
        fa = next((i for i, l in enumerate(self.layers) if not l.is_linear), None)
        ssm = next((i for i, l in enumerate(self.layers) if l.is_linear), None)
        fa_mask = create_attention_mask(flat, self.cache[fa][0] if fa is not None else None, return_array=True)
        ssm_mask = create_ssm_mask(flat, self.cache[ssm] if ssm is not None else None)
        for layer, c in zip(self.layers, self.cache):
            h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=c)
        return h

    def head(self, h: mx.array) -> mx.array:
        """Final norm + lm_head on the last position: logits [B, vocab]."""
        out = self.inner.norm(h[:, -1:, :, :].mean(axis=2))
        from mlx_vlm.models.glm5_next.linear import linear_forward

        logits = self.model.language_model.model.embed_tokens.as_linear(out) if self.lm.args.tie_word_embeddings \
            else linear_forward(self.lm.lm_head, out)
        return logits[:, -1, :]
