"""PLAN_X X1 landing check: the H1 'reversal candidate' trace's customdata, BASE save vs CUR charts — which points
differ and how (slot 3 = the candidate's `pattern`; `none` where no candidate). The M15 charts draw no such trace.

    PYTHONPATH=. python engine_v2/plans/plan_x_inputs/x1_landing_check.py BASE_SAVE_DIR CUR_CHARTS_DIR
"""
import glob
import os
import sys

_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from engine_v2.debug.chart_census import load_fig  # noqa: E402
sys.argv = _argv


# The H1 chart's file name is taken from the BASE save (artifacts/charts also holds other windows' stale files).
_H1_NAME = [os.path.basename(p) for p in glob.glob(os.path.join(sys.argv[1], "*.html")) if "_M15_" not in p]
assert len(_H1_NAME) == 1, _H1_NAME


def h1_html(d):
    return os.path.join(d, _H1_NAME[0])


def trace(path):
    data, _layout = load_fig(h1_html(path))
    ts = [t for t in data if t.get("name") == "reversal candidate"]
    assert len(ts) == 1, len(ts)
    return ts[0]


a, b = trace(sys.argv[1]), trace(sys.argv[2])
ca, cb = a["customdata"], b["customdata"]
print(f"points {len(ca)} -> {len(cb)}; keys differing besides customdata: "
      f"{sorted(k for k in set(a) | set(b) if k != 'customdata' and a.get(k) != b.get(k))}")
for i, (x, y) in enumerate(zip(ca, cb)):
    if x != y:
        print(f"  point {i}: idx {x[0]} sid {x[4]}: {x} -> {y}")
print("unchanged points:", sum(1 for x, y in zip(ca, cb) if x == y), "pattern values now:",
      sorted({str(r[3]) for r in cb}))
