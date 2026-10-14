"""Leaf-level diff of two runs' three charts (positional traces; equal counts asserted)."""
import sys, glob, os, json
_a = sys.argv[:]; sys.argv = [sys.argv[0]]
from engine_v2.debug.chart_census import load_fig
sys.argv = _a
A, B = sys.argv[1], sys.argv[2]
def leaves(x, p=""):
    if isinstance(x, dict):
        for k in x: yield from leaves(x[k], f"{p}.{k}")
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x): yield from leaves(v, f"{p}[{i}]")
    else:
        yield p, x
for suf in ("rk2-5.html", "rk2-5_M15_confluence.html", "rk2-5_M15_counter.html"):
    fa = glob.glob(os.path.join(A, "*" + suf)); fb = glob.glob(os.path.join(B, "*" + suf))
    da, la = load_fig(fa[0]); db, lb = load_fig(fb[0])
    assert len(da) == len(db), (suf, len(da), len(db))
    diffs = []
    for i, (ta, tb) in enumerate(zip(da, db)):
        if json.dumps(ta, sort_keys=True, default=str) == json.dumps(tb, sort_keys=True, default=str):
            continue
        ma, mb = dict(leaves(ta)), dict(leaves(tb))
        for k in sorted(set(ma) | set(mb)):
            if ma.get(k, "<absent>") != mb.get(k, "<absent>"):
                diffs.append((i, ta.get("name"), k, ma.get(k, "<absent>"), mb.get(k, "<absent>")))
    lay_eq = json.dumps(la, sort_keys=True, default=str) == json.dumps(lb, sort_keys=True, default=str)
    print(f"{suf}: {len(diffs)} differing leaves; layout equal={lay_eq}")
    for d in diffs: print("   ", d)
