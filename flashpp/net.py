"""Wire format between stages: 4-byte header length + JSON header + raw array bytes (row-major)."""
import json
import socket
import struct

import mlx.core as mx
import numpy as np

_DT = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32, "int32": mx.int32, "uint32": mx.uint32}


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 22))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return bytes(buf)


def send(sock: socket.socket, header: dict, arr: mx.array | None = None) -> None:
    payload = b""
    if arr is not None:
        dtype = str(arr.dtype).split(".")[-1]
        a = arr.view(mx.uint16) if arr.dtype == mx.bfloat16 else arr
        payload = memoryview(np.array(a, copy=False)).tobytes()
        header = {**header, "shape": list(arr.shape), "dtype": dtype, "nbytes": len(payload)}
    h = json.dumps(header).encode()
    sock.sendall(struct.pack("<I", len(h)) + h + payload)


def recv(sock: socket.socket) -> tuple[dict, mx.array | None]:
    (n,) = struct.unpack("<I", _recv_exact(sock, 4))
    header = json.loads(_recv_exact(sock, n))
    if "nbytes" not in header:
        return header, None
    raw = _recv_exact(sock, header["nbytes"])
    dt = header["dtype"]
    if dt == "bfloat16":
        arr = mx.array(np.frombuffer(raw, dtype=np.uint16).reshape(header["shape"])).view(mx.bfloat16)
    else:
        arr = mx.array(np.frombuffer(raw, dtype=np.dtype(dt)).reshape(header["shape"]))
    return header, arr
