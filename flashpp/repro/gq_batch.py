import sys, numpy as np, mlx.core as mx
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # stage_dsa.py lives one level up
mx.set_default_device(mx.gpu)
from stage_dsa import DsaStage
st = DsaStage(sys.argv[1], int(sys.argv[2]), int(sys.argv[2]) + 1)
sw = st.layers[0].mlp.switch_mlp
rng = np.random.default_rng(2)
def rel(a, b): return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
for name, p, D in (("gate_up", sw["gate_up_proj"], 6144), ("down", sw["down_proj"], 2048)):
    for B in (1, 2, 3, 4, 8, 16, 64):
        x = mx.array(rng.standard_normal((B, 1, D)).astype(np.float32) * 0.05).astype(mx.bfloat16)
        idx = mx.array(rng.choice(256, B).astype(np.uint32))
        g = mx.gather_qmm(x, p["weight"], p["scales"], p["biases"], rhs_indices=idx, transpose=True, group_size=p.group_size, bits=p.bits)
        r = mx.concatenate([mx.quantized_matmul(x[b], p["weight"][int(idx[b].item())], p["scales"][int(idx[b].item())], p["biases"][int(idx[b].item())], transpose=True, group_size=p.group_size, bits=p.bits) for b in range(B)])
        mx.eval(g, r)
        ga = np.array(g.astype(mx.float32)).reshape(B, -1); ra = np.array(r.astype(mx.float32)).reshape(B, -1)
        bad = [b for b in range(B) if rel(ga[b], ra[b]) > 0.01]
        same = mx.array(np.full(B, int(idx[0].item()), np.uint32))
        g2 = mx.gather_qmm(x, p["weight"], p["scales"], p["biases"], rhs_indices=same, transpose=True, group_size=p.group_size, bits=p.bits)
        r2 = mx.quantized_matmul(x.reshape(B, D), p["weight"][int(idx[0].item())], p["scales"][int(idx[0].item())], p["biases"][int(idx[0].item())], transpose=True, group_size=p.group_size, bits=p.bits)
        mx.eval(g2, r2)
        print(f"{name} B={B:3d}: rel {rel(ga, ra):.4f}  bad rows {bad[:8]}{'…' if len(bad)>8 else ''}  | same-expert rel {rel(np.array(g2.astype(mx.float32)).reshape(B,-1), np.array(r2.astype(mx.float32))):.4f}", flush=True)
