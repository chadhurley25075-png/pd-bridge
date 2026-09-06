#!/usr/bin/env python3
"""pd_share.py — threaded read-only file share for the P/D capture dir (replaces `python3 -m http.server`,
which is single-threaded and drops connections under the front door's poll+fetch pattern → RemoteDisconnected)."""
import http.server, os, socketserver, sys, json
ROOT=sys.argv[1] if len(sys.argv)>1 else os.path.expanduser("~/pd_capture"); PORT=int(sys.argv[2]) if len(sys.argv)>2 else 8010
class H(http.server.SimpleHTTPRequestHandler):
    protocol_version="HTTP/1.1"
    def __init__(self,*a,**k): super().__init__(*a,directory=ROOT,**k)
    def log_message(self,*a): pass
    def do_GET(self):
        if self.path.startswith("/_ls/") or self.path=="/_ls":
            sub=self.path[5:].strip("/"); d=os.path.join(ROOT,sub)
            try:
                out=[]
                for e in os.scandir(d):
                    if e.name.endswith(".tmp"): continue   # hook writes .tmp then renames: never list partials (9/6)
                    try: out.append({"name":e.name,"dir":e.is_dir(),"size":(e.stat().st_size if e.is_file() else 0)})
                    except FileNotFoundError: continue      # entry vanished mid-listing (janitor/rename race): skip, don't 404
            except Exception as e: self.send_error(404,str(e)); return
            b=json.dumps(out).encode(); self.send_response(200); self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(b))); self.end_headers(); self.wfile.write(b); return
        return super().do_GET()
class S(socketserver.ThreadingMixIn, http.server.HTTPServer): daemon_threads=True; allow_reuse_address=True
S(("0.0.0.0",PORT),H).serve_forever()
