"""PLAN_W §4 landing check — the KL zone hover's `kind=` line, FIGURE-ONLY vs BASE (S_W9c):

  - every CSV md5-equal to BASE; every PNG md5-equal;
  - each chart: the same traces in the same order, the layouts (shapes included) equal; a trace differs ONLY if it is a
    KL zone hover (`KL Zone<br>` in its template) and then ONLY in `hovertemplate` (= BASE's with the one line
    `kind=%{customdata[10]}<br>` after `KL Zone<br>`) and `customdata` (each row = BASE's + [the kind]); the kind is
    CTS exactly on the hidden `sub_cts_zones` traces and on the H1 CTS zones (the H1 / overlay hovers of a CTS zone).
    Prints the changed-trace count per chart and the kinds seen.

    PYTHONIOENCODING=utf-8 PYTHONPATH=. python engine_v2/plans/plan_w_inputs/w9d_landing_check.py BASE CUR_DEBUG CUR_CHARTS
"""
import collections
import glob
import hashlib
import json
import os
import sys

_argv = sys.argv[:]
sys.argv = sys.argv[:1]
from engine_v2.debug.chart_census import load_fig  # noqa: E402
sys.argv = _argv

BASE, CUR_DEBUG, CUR_CHARTS = sys.argv[1:4]
ok = True


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


bad = [os.path.basename(p) for p in glob.glob(os.path.join(BASE, "*.csv"))
       if md5(p) != md5(os.path.join(CUR_DEBUG, os.path.basename(p)))]
ok &= not bad
print(f"CSVs: {len(glob.glob(os.path.join(BASE, '*.csv')))} compared, differing {bad}")
for p in sorted(glob.glob(os.path.join(BASE, "*.png"))):
    good = md5(p) == md5(os.path.join(CUR_CHARTS, os.path.basename(p)))
    ok &= good
    print(f"PNG {os.path.basename(p)} equal: {good}")
for p in sorted(glob.glob(os.path.join(BASE, "*.html"))):
    n = os.path.basename(p)
    (bd, bl), (cd, cl) = load_fig(p), load_fig(os.path.join(CUR_CHARTS, n))
    good = len(bd) == len(cd) and json.dumps(bl, sort_keys=True, default=str) == json.dumps(cl, sort_keys=True, default=str)
    changed, kinds, problems = 0, collections.Counter(), []
    for i, (a, b) in enumerate(zip(bd, cd)):
        keys = {k for k in set(a) | set(b) if json.dumps(a.get(k), default=str) != json.dumps(b.get(k), default=str)}
        if not keys:
            if "KL Zone<br>" in (a.get("hovertemplate") or ""):
                problems.append(f"trace {i} {a.get('name')}: a KL hover that did not change")
            continue
        changed += 1
        ta = a.get("hovertemplate") or ""
        if "KL Zone<br>" not in ta or keys - {"hovertemplate", "customdata"}:
            problems.append(f"trace {i} {a.get('name')}: differs in {sorted(keys)}")
            continue
        if b["hovertemplate"] != ta.replace("KL Zone<br>", "KL Zone<br>kind=%{customdata[10]}<br>"):
            problems.append(f"trace {i}: template not = BASE + the kind line")
        rows_a, rows_b = a.get("customdata") or [], b.get("customdata") or []
        if len(rows_a) != len(rows_b) or any(list(rb[:-1]) != list(ra) for ra, rb in zip(rows_a, rows_b)):
            problems.append(f"trace {i}: customdata not = BASE + one slot")
        ks = {rb[-1] for rb in rows_b}
        if len(ks) != 1 or not ks <= {"BOS", "CTS"}:
            problems.append(f"trace {i}: kinds {ks}")
        kind = next(iter(ks))
        kinds[(a.get("name"), kind, a.get("legendgroup") == "sub_cts_zones")] += 1
        if (a.get("legendgroup") == "sub_cts_zones") != (kind == "CTS") and a.get("name") == "M15 KL zone":
            problems.append(f"trace {i}: a sub zone hover {kind} but legendgroup {a.get('legendgroup')}")
    good &= not problems
    ok &= good
    print(f"{'OK ' if good else 'BAD'} {n}: traces {len(bd)}/{len(cd)}, layout equal, {changed} changed — "
          f"{dict(kinds)} (name, kind, hidden)")
    for pr in problems[:8]:
        print("   ", pr)
print("ALL AS PREDICTED" if ok else "NOT AS PREDICTED")
sys.exit(0 if ok else 1)
