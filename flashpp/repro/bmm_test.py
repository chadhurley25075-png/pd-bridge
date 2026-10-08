import mlx.core as mx, numpy as np
rng = np.random.default_rng(0)
x = mx.array(rng.standard_normal((1, 64, 24, 192)).astype(np.float32)).astype(mx.bfloat16)   # [B,H,L,D]
w = mx.array(rng.standard_normal((64, 512, 192)).astype(np.float32) * 0.05).astype(mx.bfloat16)  # [H,O,D]
def rel(a, b):
    a = np.array(a.astype(mx.float32)).astype(np.float64); b = np.array(b.astype(mx.float32)).astype(np.float64)
    return np.linalg.norm(a - b) / np.linalg.norm(b)
exact = np.einsum("bhld,hod->bhlo", np.array(x.astype(mx.float32)).astype(np.float64), np.array(w.astype(mx.float32)).astype(np.float64))
with mx.stream(mx.gpu):
    a = x @ w.swapaxes(-1, -2); mx.eval(a)                       # MultiLinear(transpose=True) as written
    b = x @ mx.contiguous(w.swapaxes(-1, -2)); mx.eval(b)        # contiguous transposed weight
    c = (x.astype(mx.float32) @ w.swapaxes(-1, -2).astype(mx.float32)); mx.eval(c)
print("strided-view bf16 bmm vs exact:", round(rel(a, mx.array(exact.astype(np.float32))), 4))
print("contiguous bf16 bmm vs exact:  ", round(rel(b, mx.array(exact.astype(np.float32))), 4))
print("fp32 strided bmm vs exact:     ", round(rel(c, mx.array(exact.astype(np.float32))), 4))
