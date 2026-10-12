"""Keyed H1 trace diff + range-hover dump (the PLAN_N hover follow-up's `/compare`, 2026-10-12; output vs S_N:
`followup_h1_diff_vs_SN.txt`). From the repo root; a DIR holds ONE H1 chart (a save folder, or copy the current
`artifacts/charts` H1 html to a temp dir — or pass the window PREFIX):

    PYTHONPATH=. python engine_v2/plans/plan_n_inputs/h1_trace_diff.py diff OLD_DIR NEW_DIR [PREFIX]
    PYTHONPATH=. python engine_v2/plans/plan_n_inputs/h1_trace_diff.py dump DIR LO HI [PREFIX]

diff: per trace NAME, the points keyed by candle (customdata[0] when the trace carries a candle idx there, else the
x value); reports traces only on one side, then per common trace the added / removed / changed keys (y + customdata).
Shapes: compared as a sorted multiset. dump: every range:top/bottom trace's points with LO <= candle <= HI.
"""
import glob
import json
import os
import sys

_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from engine_v2.debug.chart_census import load_fig  # noqa: E402
sys.argv = _argv


def _h1(d, prefix=""):
    got = glob.glob(os.path.join(d, prefix + "*rk2-5.html"))
    assert len(got) == 1, (d, got)
    return load_fig(got[0])


def _points(tr):
    xs = tr.get("x") or []
    ys = tr.get("y") or [None] * len(xs)
    cd = tr.get("customdata")
    out = {}
    for i, x in enumerate(xs):
        c = cd[i] if cd is not None and i < len(cd) else None
        key = c[0] if isinstance(c, (list, tuple)) and c and isinstance(c[0], (int, float)) else x
        out.setdefault(json.dumps(key, default=str), []).append(
            (ys[i] if i < len(ys) else None, json.dumps(c, default=str)))
    return out


def _by_name(data):
    out = {}
    for tr in data:
        out.setdefault(tr.get("name") or "<noname>", []).append(tr)
    return out


def diff(old_d, new_d, prefix=""):
    (od, ol), (nd, nl) = _h1(old_d, prefix), _h1(new_d, prefix)
    print(f"traces old {len(od)} new {len(nd)}; shapes old {len(ol.get('shapes', []))} new {len(nl.get('shapes', []))}")
    so = sorted(json.dumps(s, sort_keys=True, default=str) for s in ol.get("shapes", []))
    sn = sorted(json.dumps(s, sort_keys=True, default=str) for s in nl.get("shapes", []))
    print("shapes", "IDENTICAL" if so == sn else "DIFFER")
    lo = {k: v for k, v in dict(ol).items() if k != "shapes"}
    ln = {k: v for k, v in dict(nl).items() if k != "shapes"}
    print("layout (sans shapes)", "IDENTICAL" if json.dumps(lo, sort_keys=True, default=str)
          == json.dumps(ln, sort_keys=True, default=str) else "DIFFER")
    bo, bn = _by_name(od), _by_name(nd)
    for n in sorted(set(bo) - set(bn)):
        print("ONLY OLD:", n, len(bo[n]))
    for n in sorted(set(bn) - set(bo)):
        print("ONLY NEW:", n, len(bn[n]))
    changed = 0
    for n in sorted(set(bo) & set(bn)):
        a, b = bo[n], bn[n]
        ja = sorted(json.dumps(t, sort_keys=True, default=str) for t in a)
        jb = sorted(json.dumps(t, sort_keys=True, default=str) for t in b)
        if ja == jb:
            continue
        changed += 1
        if len(a) != 1 or len(b) != 1:
            print(f"CHANGED {n}: {len(a)} vs {len(b)} traces of that name (multi-trace; not keyed)")
            continue
        pa, pb = _points(a[0]), _points(b[0])
        na, nb = len(a[0].get("x") or []), len(b[0].get("x") or [])
        print(f"CHANGED {n}: points {na} -> {nb}")
        for k in sorted(set(pa) - set(pb), key=lambda s: json.loads(s) if s[0] != '"' else s):
            print(f"   removed @ {k}: {pa[k]}")
        for k in sorted(set(pb) - set(pa), key=lambda s: json.loads(s) if s[0] != '"' else s):
            print(f"   added   @ {k}: {pb[k]}")
        for k in sorted(set(pa) & set(pb), key=lambda s: json.loads(s) if s[0] != '"' else s):
            if pa[k] != pb[k]:
                print(f"   changed @ {k}: {pa[k]} -> {pb[k]}")
        rest_a = {kk: vv for kk, vv in a[0].items() if kk not in ("x", "y", "customdata")}
        rest_b = {kk: vv for kk, vv in b[0].items() if kk not in ("x", "y", "customdata")}
        if json.dumps(rest_a, sort_keys=True, default=str) != json.dumps(rest_b, sort_keys=True, default=str):
            print("   (non-point attributes differ too)")
    print(f"changed traces: {changed}")


def dump(d, lo, hi, prefix=""):
    data, _ = _h1(d, prefix)
    for tr in data:
        n = tr.get("name") or ""
        if not n.startswith("range:"):
            continue
        for i, c in enumerate(tr.get("customdata") or []):
            if lo <= c[0] <= hi:
                print(n, "y", tr["y"][i], "cd", c)


if __name__ == "__main__":
    if sys.argv[1] == "diff":
        diff(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "")
    else:
        dump(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5] if len(sys.argv) > 5 else "")
