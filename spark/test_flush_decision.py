"""Logic test for the FLUSH_NOW watcher (no torch, no GPU): loads the hook with PD_CAPTURE_V3_NOARM=1 and
drives one watcher tick at a time. Runs the PATCHED copy through 6 scenarios and shows the ORIGINAL's
non-owner tick eating the flag."""
import importlib.util, os, sys, tempfile, time, types
os.environ["PD_CAPTURE_V3_NOARM"] = "1"; os.environ.pop("PD_CAPTURE_DIR", None)
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
class KV:  # stands in for _KVState
    def __init__(self, next_pos): self.next_pos = next_pos
def owner(H, root, T=17524, calls=129, nl=43, last_t_ago=0.5):
    cap = H._Capture(root=root, idle_s=15.0, weights=object(), start_threads=False)
    r = H._Req("20260906-205937-568", root); r.kv = {li: KV(T) for li in range(nl)}; r.calls = calls
    r.first_t = time.time() - 16.0; r.last_t = time.time() - last_t_ago; cap.req = r; return cap
def touch(path, mtime=None):
    open(path, "w").write(str(time.time()))
    if mtime is not None: os.utime(path, (mtime, mtime))
ok = True
def check(name, cond):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name)

P = load(os.path.join(os.path.dirname(__file__), "capture_sitecustomize_v3.py"), "hook_patched")
root = tempfile.mkdtemp(); flag = os.path.join(root, "FLUSH_NOW")
print("== PATCHED ==")
# A: non-owner process (APIServer/EngineCore: req is None forever) must leave the flag alone
cap0 = P._Capture(root=root, idle_s=15.0, weights=object(), start_threads=False); touch(flag)
fire, reason, why = cap0._flush_decision(flag); check("A non-owner: no fire, flag survives", not fire and os.path.exists(flag))
# B: owner sees a flag OLDER than this request (left over from a previous request) -> cleared, no fire
cap = owner(P, root); touch(flag, mtime=cap.req.first_t - 30)
fire, reason, why = cap._flush_decision(flag); check("B stale flag: no fire, flag cleared  [" + why.split(' q_empty')[0] + "]", not fire and not os.path.exists(flag))
# C: owner, fresh flag, but the worker is still draining the last chunk (q non-empty) -> no fire, flag STAYS
cap = owner(P, root); touch(flag); cap.q.put(("pending-item",))
fire, reason, why = cap._flush_decision(flag); check("C signal while draining: no fire, flag survives", not fire and os.path.exists(flag))
# D: next tick, worker drained -> fires flush_now and consumes the flag
cap.q.get(); cap.q.task_done()
fire, reason, why = cap._flush_decision(flag); check(f"D drained: fire={fire} reason={reason}, flag consumed", fire and reason == "flush_now" and not os.path.exists(flag))
print("   why =", why)
# C2: fresh flag but mid-burst (calls not a multiple of nl) -> no fire, flag survives
cap = owner(P, root, calls=130); touch(flag)
fire, reason, why = cap._flush_decision(flag); check("C2 signal mid-burst (calls=130): no fire, flag survives", not fire and os.path.exists(flag)); os.unlink(flag)
# E: no flag, idle 15.2 s -> idle backstop
cap = owner(P, root, last_t_ago=15.2); fire, reason, why = cap._flush_decision(flag); check(f"E idle backstop: fire={fire} reason={reason}", fire and reason == "idle")
# F: no flag, idle 2 s -> nothing
cap = owner(P, root, last_t_ago=2.0); fire, reason, why = cap._flush_decision(flag); check("F quiet tick: no fire", not fire)
# G: worker honours the sentinel reason (flush_now reaches _finish) — drive _worker one step with a stub _finish
sys.modules.setdefault("torch", types.ModuleType("torch"))   # _worker imports torch at entry; the sentinel branch never uses it
cap = owner(P, root); seen = {}
cap._finish = lambda reason, why=None: seen.update(reason=reason, why=why)
cap.q.put(P._Capture._Flush("flush_now", "test"))
class Stop(BaseException): pass
orig_get = cap.q.get
def get_once(): 
    if seen: raise Stop()
    return orig_get()
cap.q.get = get_once
try: cap._worker()
except Stop: pass
check(f"G worker passes reason to _finish: {seen}", seen.get("reason") == "flush_now")

print("== ORIGINAL (lab canonical) — one tick of _watch in a process with NO open request ==")
O = load("./capture_sitecustomize_v3.py", "hook_orig")
root2 = tempfile.mkdtemp(); flag2 = os.path.join(root2, "FLUSH_NOW"); touch(flag2)
cap = O._Capture(root=root2, idle_s=15.0, weights=object(), start_threads=False)   # req is None, like pid 1 / pid 89
n = {"calls": 0}
def fake_sleep(s):
    n["calls"] += 1
    if n["calls"] > 1: raise Stop()
O.time.sleep = fake_sleep
try: cap._watch()
except Stop: pass
check("ORIGINAL non-owner tick DELETES the flag (the bug)", not os.path.exists(flag2))
print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
