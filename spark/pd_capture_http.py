#!/usr/bin/env python3
"""pd_capture_http.py — tiny capture file server with byte-Range support and a JSON listing.
GET /_ls/<subpath>  -> [{"name","size","dir"}]     GET /<path> (Range: bytes=a-b supported)
usage: pd_capture_http.py --root $PD_CAPTURE_DIR --port 8010
"""
import argparse, json, os, re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
ap=argparse.ArgumentParser(); ap.add_argument("--root",required=True); ap.add_argument("--port",type=int,default=8010); a=ap.parse_args()
ROOT=os.path.abspath(a.root)
class H(BaseHTTPRequestHandler):
    protocol_version="HTTP/1.1"
    def log_message(self,*x): pass
    def _p(self,sub): p=os.path.abspath(os.path.join(ROOT,sub.lstrip("/"))); return p if p.startswith(ROOT) else None
    def do_GET(self):
        if self.path.startswith("/_ls"):
            p=self._p(self.path[4:] or "/")
            if p is None or not os.path.isdir(p): self.send_error(404); return
            out=[]
            for n in sorted(os.listdir(p)):
                fp=os.path.join(p,n); st=os.stat(fp); out.append({"name":n,"size":st.st_size,"dir":os.path.isdir(fp),"mtime":st.st_mtime})
            b=json.dumps(out).encode(); self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(b))); self.end_headers(); self.wfile.write(b); return
        p=self._p(self.path.split("?")[0])
        if p is None or not os.path.isfile(p): self.send_error(404); return
        size=os.path.getsize(p); start,end=0,size-1; rng=self.headers.get("Range")
        if rng:
            m=re.match(r"bytes=(\d*)-(\d*)",rng)
            if m:
                if m.group(1): start=int(m.group(1))
                if m.group(2): end=min(int(m.group(2)),size-1)
                if start>end: self.send_response(416); self.send_header("Content-Length","0"); self.end_headers(); return
        n=end-start+1
        self.send_response(206 if rng else 200); self.send_header("Content-Type","application/octet-stream"); self.send_header("Content-Length",str(n))
        if rng: self.send_header("Content-Range",f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(p,"rb") as f:
            f.seek(start); left=n
            while left>0:
                b=f.read(min(1<<24,left))
                if not b: break
                self.wfile.write(b); left-=len(b)
ThreadingHTTPServer(("0.0.0.0",a.port),H).serve_forever()
