import mlx.core as mx, numpy as np
rng=np.random.default_rng(0)
x=mx.array(rng.standard_normal((20,4096)).astype(np.float32)).astype(mx.bfloat16)
w=mx.array(rng.standard_normal((2048,4096)).astype(np.float32)*0.02).astype(mx.bfloat16)
def rel(a,b):
    a=np.array(a.astype(mx.float32)).astype(np.float64); b=np.array(b.astype(mx.float32)).astype(np.float64); return np.linalg.norm(a-b)/np.linalg.norm(b)
for bits in (4,8):
    with mx.stream(mx.cpu):
        wq,s,b=mx.quantize(w,group_size=64,bits=bits); mx.eval(wq,s,b)
        wd=mx.dequantize(wq,s,b,group_size=64,bits=bits).astype(mx.float32)
        truth=x.astype(mx.float32)@wd.T; mx.eval(truth)
    res={}
    for name,dev in (("gpu",mx.gpu),("cpu",mx.cpu)):
        with mx.stream(dev):
            r=mx.quantized_matmul(x,wq,s,b,transpose=True,group_size=64,bits=bits); mx.eval(r); res[name]=r
            r2=mx.quantized_matmul(x.astype(mx.float32),wq,s.astype(mx.float32),b.astype(mx.float32),transpose=True,group_size=64,bits=bits); mx.eval(r2); res[name+"_f32"]=r2
    for k,v in res.items(): print(f"{bits}-bit {k:8s} vs exact fp32: {rel(v,truth):.5f}")
    with mx.stream(mx.gpu):
        wd_g=mx.dequantize(wq,s,b,group_size=64,bits=bits); mx.eval(wd_g)
    print(f"{bits}-bit dequantize gpu vs cpu: {rel(wd_g, wd):.6f}")
