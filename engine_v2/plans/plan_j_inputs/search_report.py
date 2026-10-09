"""PLAN_J: summarise the random-tail corpus of each tree (`search_tree.sh` outputs) against the base tree.

    python search_report.py BASE_DIR NAME=DIR [NAME=DIR ...]

Per tree and stream (WB = WAVES-B, W = WAVES, R = REAL): trials, errors, MS order-tripwire trials (`trip`),
shape-(2) trials (`f14.s`, emission != processing order), F17 skips (`f14.skip`: an EST whose BOS sorts after it —
the downstream dispatch raises on it since S2b), and vs the base: trials whose signature (events + MS output columns)
differs, of them with a different reversal list.
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        stream, seed = os.path.basename(f)[:-6].split("_")
        for line in open(f):
            r = json.loads(line)
            if "t" in r:
                out[(stream, int(seed), int(r["t"]))] = r
    return out


base_dir, variants = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
base = load(base_dir)
trees = [("base", base)] + [(n, load(d)) for n, d in variants]
print(f"{'tree':7s} {'stream':6s} {'trials':>7s} {'err':>5s} {'trip':>5s} {'shape2':>6s} {'skip':>5s} "
      f"{'sig!=base':>8s} {'rev!=base':>8s}")
for name, recs in trees:
    by = defaultdict(Counter)
    errs = Counter()
    for k, r in recs.items():
        c = by[k[0]]
        c["trials"] += 1
        if r.get("err"):
            c["err"] += 1
            errs[r["err"][:90]] += 1
        if r.get("trip"):
            c["trip"] += 1
        f = r.get("f14") or {}
        if f.get("s"):
            c["shape2"] += 1
        if f.get("skip"):
            c["skip"] += 1
        b = base.get(k)
        if name != "base" and b is not None:
            if b.get("sig") != r.get("sig"):
                c["sig"] += 1
                if b.get("rev") != r.get("rev"):
                    c["rev"] += 1
    for s in ("WB", "W", "R"):
        c = by[s]
        print(f"{name:7s} {s:6s} {c['trials']:7d} {c['err']:5d} {c['trip']:5d} {c['shape2']:6d} {c['skip']:5d} "
              f"{c['sig'] if name != 'base' else '-':>8} {c['rev'] if name != 'base' else '-':>8}")
    for e, n in errs.most_common(5):
        print(f"        err x{n}: {e}")
