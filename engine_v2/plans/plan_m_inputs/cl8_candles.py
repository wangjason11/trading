"""CL-8 (PLAN_M §17): extract the M15 / H1 candles of the sub 3 cycle 2 vs H1 sid 1 cycle 1 case from S_M2 (read-only).

    python engine_v2/plans/plan_m_inputs/cl8_candles.py      (from the repo root)

No M15 OHLC CSV is exported: the M15 candles come from the confluence chart's candlestick trace (as
`plan_h_inputs/icref_census.py` `m15_ohlc`); the H1 candles from the save's `*_raw.csv` (columns time,o,h,l,c,volume).
"""
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
m["time"] = pd.to_datetime(m["time"]).dt.strftime("%m-%d %H:%M")
raw = pd.read_csv(f"{SAVE}/NZD_USD_H1_2025-11-15_2026-01-20_raw.csv")
tc = [c for c in raw.columns if c.lower() in ("time", "timestamp", "datetime")][0]
raw["t"] = pd.to_datetime(raw[tc]).dt.strftime("%m-%d %H:%M")
print("M15 rows", len(m), "first", m.time.iloc[0], "last", m.time.iloc[-1])
for i in [2834, 2837, 2838, 2919, 2639, 2736, 2758, 2817, 2819, 2821, 2824, 2828, 2829, 2842, 2928, 2929, 2930, 2931, 2967, 3047]:
    r = m.iloc[i]
    print(f"M15 {i} {r.time} o {r.o:.5f} h {r.h:.5f} l {r.l:.5f} c {r.c:.5f}")
for i in [684, 689, 702, 703, 710, 728, 732, 741, 761]:
    r = raw.iloc[i]
    print(f"H1 {i} {r.t} o {r['o']:.5f} h {r['h']:.5f} l {r['l']:.5f} c {r['c']:.5f}")
