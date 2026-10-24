"""Name-keyed figure diff (BASE vs CUR chart HTMLs): traces matched by (name, k-th occurrence of that name),
so an inserted trace does not cascade. Prints counts (traces / shapes), layout-equal, added / removed traces and
every matched trace whose content differs (with the differing keys).

    PYTHONPATH=. python engine_v2/plans/qa_chart_edits_inputs/fig_named_diff.py BASE_DIR CUR_DIR

Use it when a change ADDS or REMOVES a trace (`fig_scope.py` is positional — one inserted trace mismatches every
later name). Written for the replacement-break edit, 2026-10-24 (+1 PB-dot trace per M15 chart).
"""
import glob
import json
import os
import sys
from collections import Counter

_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from engine_v2.debug.chart_census import load_fig  # noqa: E402
sys.argv = _argv

BASE, CUR = sys.argv[1], sys.argv[2]
STEM = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_"


def canon(v):
    return json.dumps(v, sort_keys=True, default=str)


def keyed(traces):
    seen, out = Counter(), {}
    for t in traces:
        n = str(t.get("name"))
        out[(n, seen[n])] = t
        seen[n] += 1
    return out


for suffix in ("rk2-5.html", "rk2-5_M15_counter.html", "rk2-5_M15_confluence.html"):
    a, la = load_fig(glob.glob(os.path.join(BASE, STEM + suffix))[0])
    b, lb = load_fig(glob.glob(os.path.join(CUR, STEM + suffix))[0])
    sa, sb = len(la.get("shapes") or []), len(lb.get("shapes") or [])
    print(f"== {suffix}: traces {len(a)} -> {len(b)}; shapes {sa} -> {sb}; layout equal: {canon(la) == canon(lb)}")
    ka, kb = keyed(a), keyed(b)
    for k in sorted(set(kb) - set(ka)):
        print("   + added  ", k)
    for k in sorted(set(ka) - set(kb)):
        print("   - removed", k)
    for k in sorted(set(ka) & set(kb)):
        ta, tb = ka[k], kb[k]
        if canon(ta) != canon(tb):
            diff = sorted(x for x in set(ta) | set(tb) if canon(ta.get(x)) != canon(tb.get(x)))
            print("   ~ changed", k, diff)
