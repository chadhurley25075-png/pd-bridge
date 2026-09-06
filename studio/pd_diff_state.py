#!/usr/bin/env python3
"""pd_diff_state.py A.safetensors B.safetensors — key-by-key compare of two cache_state dumps."""
import sys, mlx.core as mx
A,ma=mx.load(sys.argv[1],return_metadata=True); B,mb=mx.load(sys.argv[2],return_metadata=True)
ka,kb=set(A),set(B); print(f"arrays: A={len(ka)} B={len(kb)} onlyA={len(ka-kb)} onlyB={len(kb-ka)}")
worst=[]; exact=0
for k in sorted(ka&kb):
    a,b=A[k],B[k]
    if a.shape!=b.shape or a.dtype!=b.dtype: print("SHAPE/DTYPE MISMATCH",k,a.shape,a.dtype,b.shape,b.dtype); continue
    d=float(mx.max(mx.abs(a.astype(mx.float32)-b.astype(mx.float32)))); ref=float(mx.max(mx.abs(a.astype(mx.float32))))+1e-9
    if d==0: exact+=1
    worst.append((d/ref,d,k,tuple(a.shape)))
worst.sort(reverse=True); print(f"bit-exact arrays: {exact}/{len(worst)}"); print("worst 12 (rel, abs, key, shape):"); [print(f"  {r:.3e} {d:.3e} {k} {s}") for r,d,k,s in worst[:12]]
mm=[k for k in set(ma)|set(mb) if ma.get(k)!=mb.get(k)]; print(f"meta mismatches: {len(mm)}", mm[:8])
