"""Print the sub swing / PB->BOS / dot traces of chosen subs on both M15 charts, x as M15 idx.

    PYTHONPATH=. python engine_v2/plans/qa_chart_edits_inputs/sub_lines.py SAVE_DIR SUB_ID [SUB_ID ...]

SAVE_DIR = a save folder or `artifacts/charts` (the M15 chart file names). Every point prints as `idx@price`
(x mapped through `m15_frame.frame`); `dash=dot` marks a prior run. Written for the replacement-break edit, 2026-10-24.
"""
import glob
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from engine_v2.debug.chart_census import load_fig  # noqa: E402
from m15_frame import frame  # noqa: E402
sys.argv = _argv

SAVE = sys.argv[1]
SUBS = [int(s) for s in sys.argv[2:]]
STEM = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"

for lens in ("confluence", "counter"):
    path = glob.glob(os.path.join(SAVE, f"{STEM}_M15_{lens}.html"))[0]
    traces, _ = load_fig(path)
    f = frame(path)
    pos = {t: i for i, t in zip(f.index, f["time"])}
    names = [str(t.get("name")) for t in traces]
    print(f"===== {lens}: {len(traces)} traces")
    for t in traces:
        name = str(t.get("name"))
        m = re.search(r"sub(\d+)", name)
        if not m or int(m.group(1)) not in SUBS or not re.match(r"^M15 (swing|PB|CTS|BOS)", name):
            continue
        pts = []
        for x, y in zip(t.get("x") or [], t.get("y") or []):
            if x is None:
                pts.append("|")
                continue
            ts = pd.Timestamp(x)
            ts = ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo else ts
            pts.append(f"{pos.get(ts)}@{y:.5f}")
        dash = (t.get("line") or {}).get("dash")
        print(f"  {name!r:40} {t.get('mode')}{' dash=' + dash if dash else ''}: {' -> '.join(pts)}")
