#!/usr/bin/env python3
"""bench_split.py — one long prompt -> time to first token + decode tok/s against an OpenAI-compatible endpoint
(streaming), with a needle so correctness is checked on every run.

usage: bench_split.py <base_url> <model> <approx_tokens> [--tag T] [--max N] [--seed S] [--think] [--followup] [--doc FILE]

The document is neutral synthetic text (seeded), or your own text with --doc, cut to ~approx_tokens (3.6 chars per
token), with a needle sentence at the midpoint and a question about it. A fresh salt line at the top makes every run
uncached unless --seed repeats. Prints one JSON line per turn; `needle_ok` says whether the answer found it.

NOTE: the numbers in README.md were measured with the same harness shape but a different (private) document; this
public version swaps in synthetic text and has not been re-run against the split.
"""
import argparse, json, random, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("url"); ap.add_argument("model"); ap.add_argument("tokens", type=int)
ap.add_argument("--tag", default=""); ap.add_argument("--max", type=int, default=300)
ap.add_argument("--seed", default=None); ap.add_argument("--think", action="store_true")
ap.add_argument("--followup", action="store_true", help="second turn on the same context")
ap.add_argument("--doc", default=None, help="use this text file instead of synthetic text")
a = ap.parse_args()

want = int(a.tokens * 3.6)
if a.doc:
    text = open(a.doc, errors="ignore").read()
    while len(text) < want:
        text += "\n\n" + text
else:
    rnd = random.Random(7)
    words = ("river stone harbor willow journey silver morning archive cadence marble orchard signal meadow granite "
             "tidal beacon canvas ember valley copper thread lantern window season").split()
    paras = []
    while sum(len(p) for p in paras) < want:
        paras.append(" ".join(rnd.choice(words) for _ in range(rnd.randint(40, 120))).capitalize() + ".")
    text = "\n\n".join(paras)
text = text[:want]
mid = len(text) // 2
needle = "\n\nNEEDLE: the brass lantern on the porch was painted teal on Thursday by the caretaker.\n\n"
text = text[:mid] + needle + text[mid:]
salt = a.seed if a.seed is not None else str(time.time_ns())
msgs = [{"role": "system", "content": f"run {salt}. You are a careful reader. Answer from the document only."},
        {"role": "user", "content": "DOCUMENT:\n" + text},
        {"role": "assistant", "content": "I have read the document."},
        {"role": "user", "content": "What color was the porch lantern painted, on which day, and by whom? Then write "
                                    "four sentences about why careful reading matters."}]
body = {"model": a.model, "messages": msgs, "max_tokens": a.max, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True}}
if not a.think:
    body["chat_template_kwargs"] = {"enable_thinking": False}


def run(body):
    req = urllib.request.Request(a.url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.perf_counter(); first = None; n = 0; out = []; usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            j = json.loads(d)
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices") or []:
                delta = ch.get("delta") or {}
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if piece:
                    if first is None:
                        first = time.perf_counter()
                    n += 1; out.append(delta.get("content") or "")
    t1 = time.perf_counter()
    toks = (usage or {}).get("completion_tokens") or n
    answer = "".join(out)
    return {"ttft_s": round((first or t1) - t0, 2), "total_s": round(t1 - t0, 2),
            "decode_tok_s": round((toks - 1) / max(t1 - (first or t1), 1e-6), 1), "completion_tokens": toks,
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "cached": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
            "needle_ok": all(w in answer.lower() for w in ("teal", "thursday", "caretaker")),
            "answer": answer[:160].replace("\n", " ")}


res = run(body)
res.update({"tag": a.tag, "approx": a.tokens})
print(json.dumps(res))
if a.followup:
    body2 = dict(body)
    body2["messages"] = msgs + [{"role": "assistant", "content": "Teal, Thursday, the caretaker."},
                                {"role": "user", "content": "Name one other word that appears in the document."}]
    r2 = run(body2); r2.update({"tag": a.tag + "+followup"}); r2.pop("needle_ok", None)
    print(json.dumps(r2))
