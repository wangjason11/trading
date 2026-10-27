"""Step 5 check ("WVMI back on the chart", 2026-10-26; was a scratchpad script): the rendered WVMI LP marks vs the
per-LP CSVs (independent of the chart code). 0 problems on the compared replay.

    PYTHONIOENCODING=utf-8 python engine_v2/plans/qa_chart_edits_inputs/wvmi_step5_check.py BASE_DIR CUR_CHARTS_DIR
Step 5 check: the rendered WVMI LP marks vs the per-LP CSVs (independent of the chart code).

For each chart: the layout differs from BASE only by yaxis2.range + yaxis3; each WVMI trace's (x, y, hover) set =
the one built from the CSV rows (the H1 rows on the M15 charts at the hour's last M15 candle, from the M15 frame
decoded from the chart).
"""
import json
import os
import sys

import pandas as pd

ROOT = r"C:\Users\wangj\OneDrive\Documents\codingproj\Project Retire\forex_engine_v2"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "engine_v2", "plans", "qa_chart_edits_inputs"))
BASE, CUR = sys.argv[1], sys.argv[2]
sys.argv = [sys.argv[0]]
from engine_v2.debug.chart_census import load_fig  # noqa: E402
from m15_frame import frame  # noqa: E402

STEM = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"
DBG = os.path.join(ROOT, "artifacts", "debug", STEM)
ROWS = {"sell": 0.80, "buy": 0.91}
bad = 0


def m3(v):
    return "NA (not computed)" if pd.isna(v) else f"{float(v):.3f}"


def status(r):
    if pd.isna(r.lp_status_idx):
        return f"{r.lp_status} ({r.lp_status_reason})"
    return f"{r.lp_status} ({r.lp_status_reason} at {int(r.lp_status_idx)})"


def idxf(r, suf=""):
    out = [f"lp_idx{suf}={int(r.lp_idx)}"]
    if int(r.as_of_idx) != int(r.lp_idx):
        out.append(f"as_of_idx{suf}={int(r.as_of_idx)}")
    return out


def hover(tf, owner, idl, fields, r):
    return "<br>".join([f"TF={tf}", f"<b>{owner} WVMI</b>", idl, " | ".join([*fields, f"lp_status={status(r)}"]),
                        f"zone_side={r.zone_side} | breakout_momentum={m3(r.breakout_momentum)} | "
                        f"pullback_momentum={m3(r.pullback_momentum)}"])


def layout_delta(a, b):
    ka, kb = set(a), set(b)
    d = sorted(k for k in ka | kb if a.get(k) != b.get(k))
    out = {}
    for k in d:
        if isinstance(a.get(k), dict) and isinstance(b.get(k), dict):
            out[k] = sorted(x for x in set(a[k]) | set(b[k]) if a[k].get(x) != b[k].get(x))
        else:
            out[k] = "whole"
    return out


h1raw = pd.read_csv(os.path.join(ROOT, "artifacts", "debug", "NZD_USD_H1_2025-11-15_2026-01-20_raw.csv"))
h1raw["time"] = pd.to_datetime(h1raw["time"], utc=True)
h1rows = pd.read_csv(DBG + "_wvmi_lps.csv")


def got_marks(traces, name):
    tr = [t for t in traces if t.get("name") == name]
    if not tr:
        return set()
    t, = tr
    assert t.get("yaxis") == "y3", name
    return {(str(pd.Timestamp(x)), float(y), h) for x, y, h in zip(t["x"], t["y"], t["customdata"])}


for chart, lens in (("", None), ("_M15_counter", "counter"), ("_M15_confluence", "confluence")):
    path = os.path.join(CUR, STEM + chart + ".html")
    traces, lay = load_fig(path)
    _, lay0 = load_fig(os.path.join(BASE, STEM + chart + ".html"))
    print(f"== {chart or 'H1'}: layout delta {layout_delta(lay0, lay)}")
    print("   yaxis2.range", lay["yaxis2"].get("range"), "| yaxis3", lay.get("yaxis3"))
    if lens is None:
        exp = {s: set() for s in ROWS}
        for _, r in h1rows.iterrows():
            t = h1raw.loc[int(r.lp_idx), "time"]
            exp[r.zone_side].add((str(pd.Timestamp(t)), ROWS[r.zone_side], hover(
                "1H", "Main", f"sid={int(r.bos_structure_id)} | cycle_id={int(r.bos_cycle_id)}", idxf(r), r)))
        for s in ROWS:
            g = got_marks(traces, f"WVMI LP {s}")
            ok = g == exp[s]
            bad += not ok
            print(f"   WVMI LP {s}: {len(g)} marks, expected {len(exp[s])}: {'OK' if ok else 'MISMATCH'}")
        continue
    f = frame(path)
    f["time"] = pd.to_datetime(f["time"], utc=True)
    sub = pd.read_csv(f"{DBG}_M15_{lens}_wvmi_lps.csv")
    exp = {s: set() for s in ROWS}
    for _, r in sub.iterrows():
        exp[r.zone_side].add((str(pd.Timestamp(f.loc[int(r.lp_idx), "time"])), ROWS[r.zone_side], hover(
            "15M", "Sub", f"sub_id={int(r.sub_id)} | cycle_id={int(r.bos_cycle_id)} | relative_dir={r.relative_dir}",
            idxf(r), r)))
    for s in ROWS:
        g = got_marks(traces, f"M15 WVMI LP {s}")
        ok = g == exp[s]
        bad += not ok
        print(f"   M15 WVMI LP {s}: {len(g)} marks, expected {len(exp[s])}: {'OK' if ok else 'MISMATCH'}")
    exp = {s: set() for s in ROWS}
    for _, r in h1rows.iterrows():
        t = h1raw.loc[int(r.lp_idx), "time"]
        m = f[(f["time"] >= t) & (f["time"] < t + pd.Timedelta(hours=1))]
        m15 = int(m.index[-1])
        exp[r.zone_side].add((str(pd.Timestamp(f.loc[m15, "time"])), ROWS[r.zone_side], hover(
            "1H>>15M", "Main", f"sid={int(r.bos_structure_id)} | cycle_id={int(r.bos_cycle_id)}",
            [*idxf(r, "_1H"), f"lp_idx={m15}"], r)))
    for s in ROWS:
        g = got_marks(traces, f"H1 WVMI LP {s}")
        ok = g == exp[s]
        bad += not ok
        print(f"   H1 WVMI LP {s}: {len(g)} marks, expected {len(exp[s])}: {'OK' if ok else 'MISMATCH'}")
print("problems:", bad)
