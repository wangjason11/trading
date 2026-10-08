"""PLAN_I build: the by-name trace multiset diff §9.5 calls for (positional trace alignment shifts once a trace is
replaced). Per chart: traces as (name, x, y) — removed / added counted by name — and shapes as a multiset.

    PYTHONPATH=. python engine_v2/plans/plan_i_inputs/fig_multiset.py BASE_DIR [CUR_CHARTS]

BASE_DIR holds the baseline `.html` charts (e.g. a flat copy of a replay's artifacts/charts)."""
import glob
import json
import os
import sys
from collections import Counter

_argv = sys.argv[:]
sys.argv = [sys.argv[0]]  # chart_census runs a census on import when given args
from engine_v2.debug.chart_census import load_fig  # noqa: E402
sys.argv = _argv

BASE = sys.argv[1]
CUR = sys.argv[2] if len(sys.argv) > 2 else "artifacts/charts"


def dj(o):
    return json.dumps(o, sort_keys=True, default=str)


for bh in sorted(glob.glob(os.path.join(BASE, "*.html"))):
    name = os.path.basename(bh)
    ch = os.path.join(CUR, name)
    (bd, bl), (cd, cl) = load_fig(bh), load_fig(ch)
    bt = Counter((t.get("name"), dj(t.get("x")), dj(t.get("y"))) for t in bd)
    ct = Counter((t.get("name"), dj(t.get("x")), dj(t.get("y"))) for t in cd)
    rem, add = bt - ct, ct - bt
    bs = Counter(dj(s) for s in bl.get("shapes", []))
    cs = Counter(dj(s) for s in cl.get("shapes", []))
    print(f"{name}: traces {len(bd)} -> {len(cd)}  removed {sum(rem.values())} {dict(Counter(k[0] for k in rem.elements()))}"
          f"  added {sum(add.values())} {dict(Counter(k[0] for k in add.elements()))}")
    print(f"   shapes {len(bl.get('shapes', []))} -> {len(cl.get('shapes', []))}  removed {sum((bs - cs).values())}"
          f"  added {sum((cs - bs).values())}")
