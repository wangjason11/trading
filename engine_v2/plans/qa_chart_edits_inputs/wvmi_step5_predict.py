"""Step 5 prediction ("WVMI back on the chart", 2026-10-26; was a scratchpad script): the marks each chart should
draw, from S_lp_rows' per-LP CSVs.

    PYTHONIOENCODING=utf-8 python engine_v2/plans/qa_chart_edits_inputs/wvmi_step5_predict.py
Step 5 prediction: the dots each chart should draw, from S_lp_rows' per-LP CSVs.

- sub dots per M15 chart = the lens's *_wvmi_lps.csv rows (side counts; candles with 2+ dots)
- H1 circles = the H1 *_wvmi_lps.csv rows; on the M15 charts at the hour's last M15 candle
  (the M15 frame decoded from the saved chart); same-candle with a sub dot (same / other side)
"""
import os
import sys
from collections import Counter, defaultdict

import pandas as pd

ROOT = r"C:\Users\wangj\OneDrive\Documents\codingproj\Project Retire\forex_engine_v2"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "engine_v2", "plans", "qa_chart_edits_inputs"))
_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from m15_frame import frame  # noqa: E402
sys.argv = _argv

S = os.path.join(ROOT, "artifacts", "commits", "week9-entries-decisions", "20261026_220225_588014a")
STEM = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"

h1 = pd.read_csv(os.path.join(S, STEM + "_wvmi_lps.csv"))
h1raw = pd.read_csv(os.path.join(S, "NZD_USD_H1_2025-11-15_2026-01-20_raw.csv"))
h1raw["time"] = pd.to_datetime(h1raw["time"], utc=True)
print("H1 rows", len(h1), dict(Counter(h1["zone_side"])), "status", dict(Counter(h1["lp_status"])))
print("H1 records", h1.groupby(["bos_structure_id", "bos_cycle_id"]).size().to_dict())
c = Counter(h1["lp_idx"])
print("H1 candles with 2+ rows:", {k: v for k, v in c.items() if v > 1})

for lens in ("counter", "confluence"):
    sub = pd.read_csv(os.path.join(S, f"{STEM}_M15_{lens}_wvmi_lps.csv"))
    f = frame(os.path.join(S, f"{STEM}_M15_{lens}.html"))
    f["time"] = pd.to_datetime(f["time"], utc=True)
    print(f"\n== {lens}: sub rows {len(sub)} {dict(Counter(sub['zone_side']))} status {dict(Counter(sub['lp_status']))}")
    print("   M15 frame", f.index.min(), f.index.max(), f["time"].iloc[0], f["time"].iloc[-1])
    by_idx = defaultdict(list)
    for _, r in sub.iterrows():
        by_idx[int(r.lp_idx)].append((r.zone_side, int(r.sub_id), int(r.bos_cycle_id), r.lp_status))
    multi = {k: v for k, v in by_idx.items() if len(v) > 1}
    same_side = {k: v for k, v in multi.items() if len(set(x[0] for x in v)) < len(v)}
    pairs = {k: v for k, v in multi.items() if len(set(x[0] for x in v)) == 2}
    print("   candles with 2+ sub dots:", len(multi), "| same side:", same_side, "| buy+sell:", pairs)
    # H1 rows -> M15 hour end
    mapped = []
    for _, r in h1.iterrows():
        t = h1raw.loc[int(r.lp_idx), "time"]
        m = f[(f["time"] >= t) & (f["time"] < t + pd.Timedelta(hours=1))]
        if m.empty:
            before = f[f["time"] < t]
            m15 = int(before.index[-1]) if len(before) else None
        else:
            m15 = int(m.index[-1])
        mapped.append((int(r.lp_idx), m15, r.zone_side, int(r.bos_structure_id), int(r.bos_cycle_id)))
    none = [m for m in mapped if m[1] is None]
    print("   H1 rows mapped:", len(mapped), "unmapped:", none)
    hit_same = [(h, m, s, by_idx[m]) for h, m, s, _, _ in mapped if m in by_idx and any(x[0] == s for x in by_idx[m])]
    hit_other = [(h, m, s, by_idx[m]) for h, m, s, _, _ in mapped if m in by_idx and not any(x[0] == s for x in by_idx[m])]
    print("   H1 circle on a same-side sub dot:", hit_same)
    print("   H1 circle on a candle with only other-side sub dots:", hit_other)
    # buy+sell pairs including H1 circles
    sides_at = defaultdict(set)
    for k, v in by_idx.items():
        for x in v:
            sides_at[k].add(x[0])
    for h, m, s, _, _ in mapped:
        sides_at[m].add(s)
    print("   candles with a buy and a sell mark (sub or H1):", sorted(k for k, v in sides_at.items() if len(v) == 2))
