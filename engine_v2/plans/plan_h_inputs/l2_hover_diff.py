"""PLAN_L landing 2 — the changed hover texts vs a save (S_L1): every POI / fib hover trace present only in one figure,
with its rendered status cell (fib) / pattern line (POI), so the L2 labels can be read against PLAN_L §4.1.

    PYTHONPATH=. python engine_v2/plans/plan_h_inputs/l2_hover_diff.py BASE_SAVE_DIR [CUR_CHARTS]
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

BASE = sys.argv[1]
CHARTS = sys.argv[2] if len(sys.argv) > 2 else "artifacts/charts"


def cell(tr):
    """The fib status cell or the POI pattern line + the trace's identifying cells."""
    cd = (tr.get("customdata") or [[]])[0]
    name = tr.get("name") or ""
    strs = [str(x) for x in cd if isinstance(x, str) and ("<br>" in x or x.startswith(("IP ", "No pattern", "IC")))]
    ident = [x for x in cd[:4]]
    return f"{name} {ident} :: " + " || ".join(strs)


for bh in sorted(glob.glob(os.path.join(BASE, "*.html"))):
    ch = os.path.join(CHARTS, os.path.basename(bh))
    (da, _), (db, _) = load_fig(bh), load_fig(ch)
    ka = Counter(json.dumps(t, sort_keys=True, default=str) for t in da)
    kb = Counter(json.dumps(t, sort_keys=True, default=str) for t in db)
    print(f"== {os.path.basename(bh)}")
    for k in sorted((ka - kb).elements()):
        print("  - ", cell(json.loads(k)))
    for k in sorted((kb - ka).elements()):
        print("  + ", cell(json.loads(k)))
