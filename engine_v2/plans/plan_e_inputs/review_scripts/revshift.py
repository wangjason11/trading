"""Reversal-candle shift between two random_tail_search outputs (same seed/stream): python revshift.py A.jsonl B.jsonl"""
import json, sys
from collections import Counter

A = [json.loads(l) for l in open(sys.argv[1])]
B = [json.loads(l) for l in open(sys.argv[2])]
c, ex = Counter(), {}
for x, y in zip(A, B):
    if "err" in x or "err" in y:
        k = "error " + ("A" if "err" in x else "") + ("B" if "err" in y else "")
    elif x["sig"] == y["sig"]:
        continue
    else:
        ra, rb = x["rev"], y["rev"]
        if ra == rb:
            k = "same reversal (path detail)"
        elif not ra:
            edge = (x["end"] if x.get("end") is not None else x["n"] - 1)
            k = "A none -> B reverses " + ("AT the data edge" if rb[0] == edge else "inside")
        elif not rb:
            edge = (x["end"] if x.get("end") is not None else x["n"] - 1)
            k = "A reverses -> B none " + ("(A's AT the data edge)" if ra[0] == edge else "(inside)")
        else:
            d = rb[0] - ra[0]
            k = f"B reverses {abs(d)} candle{'s' if abs(d) != 1 else ''} {'earlier' if d < 0 else 'later'}"
    c[k] += 1
    ex.setdefault(k, x["t"])
n = len(A)
print(f"n={n} differing={sum(c.values())}")
for k, v in c.most_common():
    print(f"  {k:36} {v:6}  ({100 * v / n:.2f}% of trials; e.g. t={ex[k]})")
