#!/usr/bin/env python3
"""pd_share.py — threaded read-only file share for the P/D capture dir (replaces `python3 -m http.server`,
which is single-threaded and drops connections under the front door's poll+fetch pattern → RemoteDisconnected).

Routes (all GET, all JSON except plain file reads):
  /<stamp>/<file>          the capture files themselves (Range supported by SimpleHTTPRequestHandler)
  /_ls  /_ls/<stamp>       directory listing as JSON [{name, dir, size}] — never lists *.tmp (atomic writes)
  /_flush                  touch <ROOT>/FLUSH_NOW: the front door's "my engine call returned, flush now" signal
                           (docs/FINDING-flush-signal-three-watchers.md). Idempotent, write-only.
  /_ack/<stamp>/<seg>      streaming (docs/STREAMING-CAPTURE.md): the consumer has stored this segment. With
                           PD_SHARE_ACK_DELETE=1 the segment file is unlinked so the prefill box's disk holds
                           only the un-consumed backlog; default keeps it (resume after a Mac restart needs it).
Env: PD_SHARE_HOST (bind address, default 0.0.0.0 — bind to the bridge link only; captures are prompt KV),
     PD_SHARE_ACK_DELETE (0/1)."""
import http.server, os, re, socketserver, sys, json, time
ROOT=sys.argv[1] if len(sys.argv)>1 else os.path.expanduser("~/pd_capture"); PORT=int(sys.argv[2]) if len(sys.argv)>2 else 8010
HOST=os.environ.get("PD_SHARE_HOST","0.0.0.0")
ACK_DELETE=os.environ.get("PD_SHARE_ACK_DELETE","0")=="1"
_STAMP=re.compile(r"^\d{8}-\d{6}-\d{3}$"); _SEG=re.compile(r"^seg_\d{8}\.safetensors$")
class H(http.server.SimpleHTTPRequestHandler):
    protocol_version="HTTP/1.1"
    def __init__(self,*a,**k): super().__init__(*a,directory=ROOT,**k)
    def log_message(self,*a): pass
    def _json(self,code,obj):
        b=json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(b))); self.end_headers(); self.wfile.write(b)
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
            self._json(200,out); return
        if self.path.startswith("/_flush"):
            # the front door signals "my engine call returned — the prefill is truly over, flush now".
            # The hook's watcher consumes FLUSH_NOW within 0.1 s. Write-only signal; idempotent.
            try:
                open(os.path.join(ROOT,"FLUSH_NOW"),"w").write(str(time.time())); self._json(200,{"ok":True})
            except Exception as e: self.send_error(500,str(e))
            return
        if self.path.startswith("/_ack/"):
            parts=self.path[6:].strip("/").split("/")
            if len(parts)!=2 or not _STAMP.match(parts[0]) or not _SEG.match(parts[1]):
                self.send_error(400,"expected /_ack/<stamp>/seg_XXXXXXXX.safetensors"); return
            p=os.path.join(ROOT,parts[0],parts[1]); deleted=False
            if ACK_DELETE:
                try: os.unlink(p); deleted=True
                except FileNotFoundError: pass
                except Exception as e: self.send_error(500,str(e)); return
            self._json(200,{"ok":True,"deleted":deleted}); return
        return super().do_GET()
class S(socketserver.ThreadingMixIn, http.server.HTTPServer): daemon_threads=True; allow_reuse_address=True
if __name__=="__main__":
    S((HOST,PORT),H).serve_forever()
