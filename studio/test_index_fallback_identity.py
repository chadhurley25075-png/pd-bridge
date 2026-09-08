#!/usr/bin/env python3
"""Falsifier for the cache-index disk fallback.

The fallback reaches a block file by PATH (derived from the requested hash) while the
startup scan derives identity from CONTENT. Without an explicit check those two notions
of identity can disagree, and the fallback will index a file as a block it is not --
serving wrong KV as correct cache. That failure is silent: the model stays fluent.

This test builds exactly that disagreement and asserts the fallback refuses it.
Reported by an external reviewer, 2026-09-08.

usage: $OMLX_PYTHON studio/test_index_fallback_identity.py
"""
import sys
from pathlib import Path


class _Meta:
    def __init__(self, block_hash): self.block_hash = block_hash


class _Index:
    def __init__(self): self.added = []
    def get(self, h): return None
    def add(self, m): self.added.append(m)


class _CacheUnderTest:
    """Mirrors the patched _pd_index_from_disk contract with the file layer stubbed."""
    def __init__(self, cache_dir, file_meta):
        self._cache_dir = Path(cache_dir); self._index = _Index(); self._file_meta = file_meta
    def _read_file_metadata(self, p): return self._file_meta
    def _is_compatible_block(self, m): return True

    def _pd_index_from_disk(self, block_hash):
        hex_hash = bytes(block_hash).hex()
        file_path = self._cache_dir / hex_hash[0] / f"{hex_hash}.safetensors"
        if not file_path.is_file():
            return None
        metadata = self._read_file_metadata(file_path)
        if metadata is None or not self._is_compatible_block(metadata):
            return None
        if bytes(metadata.block_hash) != bytes(block_hash):   # the fix under test
            return None
        self._index.add(metadata)
        return metadata


def main():
    import tempfile
    requested = bytes.fromhex("aa" * 32)
    impostor  = bytes.fromhex("bb" * 32)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / requested.hex()[0]; p.mkdir(parents=True, exist_ok=True)
        (p / f"{requested.hex()}.safetensors").write_bytes(b"stub")

        # 1. A file whose own metadata names a DIFFERENT block must be refused.
        c = _CacheUnderTest(d, _Meta(impostor))
        assert c._pd_index_from_disk(requested) is None, \
            "FAIL: indexed a file whose metadata.block_hash != requested hash"
        assert not c._index.added, "FAIL: impostor block reached the index"

        # 2. The honest case must still work, or the fallback is useless.
        c2 = _CacheUnderTest(d, _Meta(requested))
        assert c2._pd_index_from_disk(requested) is not None, \
            "FAIL: refused a block whose metadata matches the request"
        assert len(c2._index.added) == 1

        # 3. A true miss stays a miss.
        c3 = _CacheUnderTest(d, _Meta(requested))
        assert c3._pd_index_from_disk(impostor) is None, "FAIL: indexed a nonexistent block"

    print("ok  identity mismatch refused / matching block indexed / true miss stays a miss")
    return 0


if __name__ == "__main__":
    sys.exit(main())
