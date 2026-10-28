"""The floor's history read from the saves (read-only): H1 candle types + big-maru flags per save,
and the M15 counter events around idx 4047 either side of the first floor (804d19d).
Run from the repo root (the argument = the repo root):
    PYTHONIOENCODING=utf-8 python engine_v2/plans/qa_pinbar_floor_inputs/floor_history_from_saves.py .
Findings (2026-10-27): memory `_INBOX.md` until filed into `features/CANDLE_BODY_FLOOR_NOTES.md`."""
from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = sys.argv[1]
W8 = os.path.join(ROOT, "artifacts", "commits", "week8-volmom-multitf")
W9 = os.path.join(ROOT, "artifacts", "commits", "week9-entries-decisions")
H1 = "NZD_USD_H1_2025-11-15_2026-01-20_final.csv"
CTR = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5_M15_counter_structure_events.csv"
SAVES = [
    ("no floor (b46bec8, 05-28 a.m.)", os.path.join(W8, "20260528_081805_b46bec8")),
    ("big-maru floor 3/2/1 (804d19d)", os.path.join(W8, "20260528_195756_804d19d")),
    ("big-maru floor, last (18a6b32)", os.path.join(W8, "20260608_171007_18a6b32")),
    ("pinbar floor 2.2/1.0 (2391597)", os.path.join(W8, "20260616_154122_2391597")),
    ("today S_controls (5dfb5b3)", os.path.join(W9, "20261027_164527_5dfb5b3")),
]

frames = {}
for name, d in SAVES:
    df = pd.read_csv(os.path.join(d, H1))
    frames[name] = df
    bm = int(df["is_big_maru_as0"].astype(bool).sum()) if "is_big_maru_as0" in df else -1
    bn = int(df["is_big_normal_as0"].astype(bool).sum()) if "is_big_normal_as0" in df else -1
    print(f"{name:34s} n={len(df)} types={df['candle_type'].value_counts().to_dict()} "
          f"big_maru_as0={bm} big_normal_as0={bn}")

base = frames[SAVES[0][0]]
for name, _ in SAVES[1:]:
    df = frames[name]
    same_time = (df["time"].values == base["time"].values).all()
    dt = (df["candle_type"] != base["candle_type"])
    db = (df["is_big_maru_as0"].astype(bool) != base["is_big_maru_as0"].astype(bool))
    print(f"vs no floor: {name:34s} same candles={same_time} type changed={int(dt.sum())} "
          f"big_maru_as0 changed={int(db.sum())}")

# the big-maru floor's H1 victims: big_maru_as0 True without, False with
a, b = frames[SAVES[0][0]], frames[SAVES[1][0]]
lost = a["is_big_maru_as0"].astype(bool) & ~b["is_big_maru_as0"].astype(bool)
print("\nH1 big-maru flags removed by the 3-pip big-maru floor:")
cols = ["time", "o", "h", "l", "c", "body_len", "candle_len", "body_pct", "candle_type", "big_ratio_as0"]
cols = [c for c in cols if c in a.columns]
t = a.loc[lost, cols].copy()
t["body_p"] = (t["body_len"] / 1e-4).round(2)
print(t.to_string())

print("\nM15 counter events with idx in 4030..4070 (no floor vs big-maru floor):")
for name, d in SAVES[:2]:
    ev = pd.read_csv(os.path.join(d, CTR))
    sel = ev[(ev["idx"] >= 4030) & (ev["idx"] <= 4070)]
    keep = [c for c in ["structure_id", "sub_sid", "idx", "event_type", "price", "time"] if c in sel.columns]
    print(f"-- {name}  ({len(ev)} events total)")
    print(sel[keep].to_string())
