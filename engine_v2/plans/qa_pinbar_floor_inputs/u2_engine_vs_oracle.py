"""PLAN_U U2 build check: the ENGINE's U2 candle types (its own fetch path — `fetch_history_with_auto_extend`,
`run_replay.fetch_floor_seed`, `fetch_lower_tf_data`, `apply_candle_classification(floor_seed=…)`) against the
prediction (`plan_u_prediction.txt`: every candle whose type differs between today / U2 / U3, with its U2 column).

A candle in the list must have the list's U2 type; every other candle must keep today's type (today = body-% + the
fixed floor H1 2.2 / M15 1.0 pips, recomputed here with the same EPS rule). Prints the seed lines, the type counts,
the `is_floor_pinbar` counts by body-% type, the change counts vs today and every mismatch.

    PYTHONIOENCODING=utf-8 python engine_v2/plans/qa_pinbar_floor_inputs/u2_engine_vs_oracle.py
"""
from __future__ import annotations

import re
import sys

import _frames  # noqa: F401  (sys.path)
import numpy as np
import pandas as pd

from _frames import HERE, out_path
from engine_v2.config import CONFIG
from engine_v2.features.candle_classifier import apply_candle_classification
from engine_v2.features.candle_params import CandleParams
from engine_v2.features.candles_v2 import EPS
from engine_v2.multitf.data_bridge import fetch_lower_tf_data
from engine_v2.run_replay import _floor_to_day_start, fetch_floor_seed, fetch_history_with_auto_extend

FIXED_PIPS = {"H1": 2.2, "M15": 1.0}
PIP = 0.0001


def predicted_u2(path):
    """{TF: {idx: U2 type}} from the prediction's per-candle lists."""
    out, tf = {}, None
    row = re.compile(r"^\s+(\d+)\s+\S+ \S+\s+(maru|normal|pinbar)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+(\w+)\s+(\w+)\s+(\w+)")
    for line in open(path, encoding="utf-8"):
        m = re.match(r"^## (H1|M15):", line)
        if m:
            tf = m.group(1)
            out[tf] = {}
            continue
        m = row.match(line)
        if m and tf:
            out[tf][int(m.group(1))] = m.group(4)
    return out


def main():
    lines = []
    pr = predicted_u2(HERE / "plan_u_prediction.txt")
    h1, _, _ = fetch_history_with_auto_extend(
        pair=CONFIG.pair, timeframe=CONFIG.timeframe, start=_floor_to_day_start(CONFIG.start),
        end=CONFIG.end, lookback_days=183, min_history=50, extend_days=4, max_extend_iters=8)
    h1 = h1.reset_index(drop=True)
    h1.attrs["pair"] = CONFIG.pair
    p = CandleParams()
    need = p.pinbar_floor_seed_windows * p.pinbar_floor_window
    first = h1["time"].iloc[0]
    m15 = fetch_lower_tf_data(CONFIG.pair, "M15", pd.to_datetime(first, utc=True),
                              pd.to_datetime(h1["time"].iloc[-1], utc=True))
    ok = True
    for tf, raw in (("H1", h1), ("M15", m15)):
        seed = fetch_floor_seed(CONFIG.pair, tf, first, min_pinbars=need)
        df = apply_candle_classification(raw, timeframe=tf, floor_seed=seed).df
        before = df["candle_type_before_floor"].to_numpy()
        body = df["body_len"].to_numpy(float)
        today = np.where((before != "pinbar") & (body < FIXED_PIPS[tf] * PIP - EPS), "pinbar", before)
        u2 = df["candle_type"].to_numpy()
        exp = today.copy()
        for i, t in pr[tf].items():
            exp[i] = t
        bad = np.flatnonzero(u2 != exp)
        isfp = df["is_floor_pinbar"].to_numpy(bool)
        ch = u2 != today
        vc = pd.Series(u2).value_counts()
        vt = pd.Series(today).value_counts()
        lines += [
            f"## {tf}: {len(df)} candles; seed {len(seed)} candles {seed['time'].iloc[0]} -> {seed['time'].iloc[-1]}",
            f"   floor NaN on {int(np.isnan(df['pinbar_body_floor'].to_numpy(float)).sum())} candles; "
            f"floor min {np.nanmin(df['pinbar_body_floor']) / PIP:.4f} pips",
            f"   today types maru/normal/pinbar {vt.get('maru', 0)}/{vt.get('normal', 0)}/{vt.get('pinbar', 0)}"
            f" -> U2 {vc.get('maru', 0)}/{vc.get('normal', 0)}/{vc.get('pinbar', 0)}",
            f"   today -> U2: {int(ch.sum())} change ({int((ch & (u2 == 'pinbar')).sum())} become, "
            f"{int((ch & (today == 'pinbar')).sum())} stop); from maru {int((ch & (before == 'maru')).sum())}, "
            f"normal {int((ch & (before == 'normal')).sum())}; with a pin direction "
            f"{int((ch & (df['pinbar_dir'].to_numpy() != 0)).sum())}",
            f"   is_floor_pinbar {int(isfp.sum())} (maru {int((isfp & (before == 'maru')).sum())}, "
            f"normal {int((isfp & (before == 'normal')).sum())})",
            f"   vs the prediction's per-candle U2 list ({len(pr[tf])} listed): {len(bad)} mismatches"
            + ("" if len(bad) == 0 else " — " + ", ".join(f"{i}: engine {u2[i]} expected {exp[i]}" for i in bad[:20])),
            "",
        ]
        ok &= len(bad) == 0
    lines.append("ALL MATCH" if ok else "MISMATCH")
    text = "\n".join(lines)
    print(text)
    open(out_path(__file__), "w", encoding="utf-8").write(text + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
