"""PLAN_J: compare `core_sig.py` outputs of a variant tree with the base tree.

    python core_report.py DIR VARIANT [VARIANT ...]     (files DIR/<tree>_<stream>_<seed>.jsonl)

Per stream: trials, errors, trips; vs base: the event MULTISET differs (`ev`; order-free), the structural decisions
differ (`core`), and of those, which kinds changed (EST / BOS / CONF / REV entries added or removed), plus the first
few trial ids per kind (to inspect with `trial_detail.py`).
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

d, variants = sys.argv[1], sys.argv[2:]


def load(tree):
    out = {}
    for f in glob.glob(os.path.join(d, f"{tree}_*.jsonl")):
        _, stream, seed = os.path.basename(f)[:-6].split("_")
        for line in open(f):
            r = json.loads(line)
            out[(stream, int(seed), r["t"])] = r
    return out


base = load("base")
for v in variants:
    var = load(v)
    by = defaultdict(Counter)
    ex = defaultdict(list)
    for k, b in base.items():
        r = var.get(k)
        if r is None:
            continue
        c = by[k[0]]
        c["trials"] += 1
        c["err"] += bool(r.get("err"))
        c["trip_base"] += bool(b.get("trip"))
        c["trip"] += bool(r.get("trip"))
        if b.get("err") or r.get("err"):
            continue
        if b["ev"] != r["ev"]:
            c["ev_diff"] += 1
        cb, cr = Counter(map(repr, b["core"])), Counter(map(repr, r["core"]))
        if cb != cr:
            c["core_diff"] += 1
            kinds = sorted({x.split(",")[0].strip("(' ") for x in (cb - cr) + (cr - cb)})
            key = "+".join(kinds)
            c["core:" + key] += 1
            if len(ex[key]) < 4:
                ex[key].append(f"{k[0]}/{k[1]}/{k[2]}")
    print(f"== {v} vs base")
    for s, c in by.items():
        kinds = {k[5:]: n for k, n in c.items() if k.startswith("core:")}
        print(f"  {s:6s} trials {c['trials']} err {c['err']} trip {c['trip_base']}->{c['trip']} "
              f"events-multiset differ {c['ev_diff']} core differ {c['core_diff']} {kinds}")
    for key, ids in ex.items():
        print(f"    e.g. {key}: {ids}")
