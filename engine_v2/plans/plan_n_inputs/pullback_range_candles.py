"""PLAN_N §3.2 (the cold review's finding 3): the M15 candles of one pullback-created RANGE_STARTED on S_M2 (read-only).

    python engine_v2/plans/plan_n_inputs/pullback_range_candles.py      (from the repo root)

M15 candles from the confluence chart's candlestick trace (as `plan_m_inputs/cl8_candles.py`).
"""
import ast
import base64
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
SAVE = "artifacts/commits/week9-entries-decisions/20261012_023137_10e4c6f"
P = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"
keep = sys.argv
sys.argv = [sys.argv[0]]
from engine_v2.debug.chart_census import load_fig  # noqa: E402
sys.argv = keep


def arr(a):
    if isinstance(a, dict) and "bdata" in a:
        return np.frombuffer(base64.b64decode(a["bdata"]), dtype=np.dtype(a["dtype"]))
    return np.asarray(a)


traces, _ = load_fig(f"{SAVE}/{P}_M15_confluence.html")
parts = [pd.DataFrame({"time": arr(t["x"]), "o": arr(t["open"]).astype(float), "h": arr(t["high"]).astype(float),
                       "l": arr(t["low"]).astype(float), "c": arr(t["close"]).astype(float)})
         for t in traces if t.get("type") == "candlestick"]
m = pd.concat(parts).drop_duplicates("time").sort_values("time").reset_index(drop=True)
m["time"] = pd.to_datetime(m["time"]).dt.strftime("%Y-%m-%d %H:%M")
ev = pd.read_csv(f"{SAVE}/{P}_M15_confluence_structure_events.csv")
ev["m"] = ev.meta.map(ast.literal_eval)
pb = [r for r in ev.itertuples() if r.type == "RANGE_STARTED" and r.m.get("reason") == "pullback_created_range"]
r = pb[0]
print("RANGE_STARTED", r.idx, {k: r.m[k] for k in ("reason", "cts_anchor_idx", "cts_price", "pullback_apply_idx", "hi", "lo",
                                                   "sub_id", "cycle_id" if "cycle_id" in r.m else "structure_id")})
for i in range(int(r.m["cts_anchor_idx"]) - 1, int(r.idx) + 2):
    c = m.iloc[i]
    print(i, c.time, c.o, c.h, c.l, c.c)
