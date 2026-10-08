"""Serve one flashpp stage over TCP.

usage: python stage_server.py <model_dir> <start> <end> <bind_ip> <port> [--head]
ops: {"op":"reset"} · {"op":"forward", ids|hidden} -> hidden, or (if this stage has the head) {"top": [...], "logits": f32[vocab]} ·
     {"op":"info"} · {"op":"bye"}
"""
import socket
import sys
import time

import mlx.core as mx

import net

model_dir, start, end, ip, port = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
head = True if "--head" in sys.argv else None
mx.set_default_device(mx.gpu)
if mx.metal.is_available():
    # keep every weight resident on the GPU: without a wired limit macOS pages a 200 GB stage in and out each step
    _rec = mx.metal.device_info().get("max_recommended_working_set_size", 0)
    print("metal wired limit ->", mx.set_wired_limit(int(_rec)), "->", _rec, flush=True)
if "--dsa" in sys.argv:
    from stage_dsa import DsaStage
    st = DsaStage(model_dir, start, end, head=head)
else:
    from stage import Stage
    st = Stage(model_dir, start, end, head=head)
print(f"stage [{start},{end}) first={st.first} head={st.last} device={mx.default_device()} loaded in {st.load_s:.1f}s", flush=True)

srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind((ip, port))
srv.listen(4)
print(f"listening {ip}:{port}", flush=True)
while True:
    conn, peer = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print("client", peer, flush=True)
    try:
        while True:
            hdr, arr = net.recv(conn)
            op = hdr.get("op")
            if op == "reset":
                st.reset()
                net.send(conn, {"ok": True})
            elif op == "info":
                net.send(conn, {"start": start, "end": end, "first": st.first, "head": st.last, "device": str(mx.default_device())})
            elif op == "forward":
                t0 = time.perf_counter()
                h = st.embed(arr) if hdr.get("kind") == "ids" else arr
                h = st.forward(h)
                if st.last:
                    logits = st.head(h).astype(mx.float32)
                    mx.eval(logits)
                    top = mx.argsort(-logits[0])[:5]
                    net.send(conn, {"ms": (time.perf_counter() - t0) * 1e3, "top": [int(t) for t in top]}, logits)
                else:
                    mx.eval(h)
                    net.send(conn, {"ms": (time.perf_counter() - t0) * 1e3}, h)
            elif op == "bye":
                net.send(conn, {"ok": True})
                break
    except ConnectionError:
        pass
    finally:
        conn.close()
