#!/usr/bin/env python3
"""pd_score.py results.json key — lenient gold check: every gold token (split on / , and 'and', quotes/backticks stripped) must appear in the final (post-</think>) answer."""
import json,re,sys
import os; g=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'eval_questions.json')))['gold']; r=json.load(open(sys.argv[1]))['results']; key=sys.argv[2]
def final(t): return t.split('</think>')[-1].strip()
def norm(s): return re.sub(r'[`"\s]','',s)
score=0
for x,gg in zip(r,g):
    a=norm(final(x[key])); toks=[norm(t) for t in re.split(r'\s*/\s*|,\s*|\s+and\s+|\s*=\s*',gg) if t.strip()]
    ok=all(t in a for t in toks); score+=ok; print(f"{'OK ' if ok else 'MISS'} [{x.get('wall')}s] {x['q'][:55]} -> {final(x[key])[:110]!r}")
print(f"{key.upper()} score {score}/5")
