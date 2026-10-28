"""PLAN_U U3 build check: the ENGINE's U3 candle types (final window — its own fetch path,
`fetch_history_with_auto_extend`, `run_replay.fetch_floor_seed`, `fetch_lower_tf_data`,
`apply_candle_classification(floor_seed=…)`) against the prediction's U3 column (`plan_u_prediction.txt`: every candle
whose type differs between today / U2 / U3). Same method as `u2_engine_vs_oracle.py`.

Run it where `CandleParams().pinbar_floor_window_source == "final"` (asserted): the U3 variant tree (copy this
script + `_frames.py` + `u2_engine_vs_oracle.py` into its `engine_v2/plans/qa_pinbar_floor_inputs/`), or the repo
after a flip. A candle in the list must have the list's U3 type; every other candle must keep today's type (body-% +
the fixed floor H1 2.2 / M15 1.0 pips, recomputed here with the same EPS rule). Prints the seed lines, the type
counts, the `is_floor_pinbar` counts by body-% type, the change counts vs today and vs U2 (the list's U2 column) and
every mismatch.

    PYTHONIOENCODING=utf-8 python engine_v2/plans/qa_pinbar_floor_inputs/u3_engine_vs_oracle.py [OUT.txt]
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
from engine_v2.features.candles_v2 import EPS, classify_candles
from engine_v2.multitf.data_bridge import fetch_lower_tf_data
from engine_v2.run_replay import _floor_to_day_start, fetch_floor_seed, fetch_history_with_auto_extend

FIXED_PIPS = {"H1": 2.2, "M15": 1.0}
PIP = 0.0001
COLS = {"today": 3, "U2": 4, "U3": 5}


def predicted(path, col):
    """{TF: {idx: type in column `col`}} from the prediction's per-candle lists."""
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
            out[tf][int(m.group(1))] = m.group(COLS[col])
    return out


def counts(a):
    v = pd.Series(a).value_counts()
    return f"{v.get('maru', 0)}/{v.get('normal', 0)}/{v.get('pinbar', 0)}"


def change_line(label, a, b, before, pdir):
    ch = a != b
    return (f"   {label}: {int(ch.sum())} change ({int((ch & (b == 'pinbar')).sum())} become, "
            f"{int((ch & (a == 'pinbar')).sum())} stop); from maru {int((ch & (before == 'maru')).sum())}, "
            f"normal {int((ch & (before == 'normal')).sum())}; with a pin direction {int((ch & (pdir != 0)).sum())}")


def main():
    p = CandleParams()
    assert p.pinbar_floor_window_source == "final", p.pinbar_floor_window_source
    lines = [f"engine: pinbar_floor_window_source={p.pinbar_floor_window_source!r}, window {p.pinbar_floor_window}, "
             f"p{p.pinbar_floor_percentile:g}, seed windows {p.pinbar_floor_seed_windows}", ""]
    pr3 = predicted(HERE / "plan_u_prediction.txt", "U3")
    pr2 = predicted(HERE / "plan_u_prediction.txt", "U2")
    h1, _, _ = fetch_history_with_auto_extend(
        pair=CONFIG.pair, timeframe=CONFIG.timeframe, start=_floor_to_day_start(CONFIG.start),
        end=CONFIG.end, lookback_days=183, min_history=50, extend_days=4, max_extend_iters=8)
    h1 = h1.reset_index(drop=True)
    h1.attrs["pair"] = CONFIG.pair
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
        # the pin direction a candle has AS a pinbar (as the prediction's pin_dir column): the engine's own wick rule
        # (`classify_candles`, after step 4) with every maru / normal made a floor pinbar — a candle that stops
        # being a pinbar in U3 has pinbar_dir 0 in the U3 frame
        pdir = classify_candles(df, p, body_floor=np.full(len(df), np.inf))["pinbar_dir"].to_numpy()
        today = np.where((before != "pinbar") & (body < FIXED_PIPS[tf] * PIP - EPS), "pinbar", before)
        u3 = df["candle_type"].to_numpy()
        exp3, exp2 = today.copy(), today.copy()
        for i, t in pr3[tf].items():
            exp3[i] = t
        for i, t in pr2[tf].items():
            exp2[i] = t
        bad = np.flatnonzero(u3 != exp3)
        isfp = df["is_floor_pinbar"].to_numpy(bool)
        fl = df["pinbar_body_floor"].to_numpy(float)
        lines += [
            f"## {tf}: {len(df)} candles; seed {len(seed)} candles {seed['time'].iloc[0]} -> {seed['time'].iloc[-1]}",
            f"   floor NaN on {int(np.isnan(fl).sum())} candles; floor min {np.nanmin(fl) / PIP:.4f} "
            f"max {np.nanmax(fl) / PIP:.4f} pips",
            f"   types maru/normal/pinbar: today {counts(today)} -> U2 (list) {counts(exp2)} -> U3 engine {counts(u3)}",
            change_line("U2 -> U3", exp2, u3, before, pdir),
            change_line("today -> U3", today, u3, before, pdir),
            f"   is_floor_pinbar {int(isfp.sum())} (maru {int((isfp & (before == 'maru')).sum())}, "
            f"normal {int((isfp & (before == 'normal')).sum())})",
            f"   vs the prediction's per-candle U3 list ({len(pr3[tf])} listed): {len(bad)} mismatches"
            + ("" if len(bad) == 0 else " — " + ", ".join(f"{i}: engine {u3[i]} expected {exp3[i]}" for i in bad[:20])),
            "",
        ]
        ok &= len(bad) == 0
    lines.append("ALL MATCH" if ok else "MISMATCH")
    text = "\n".join(lines)
    print(text)
    dest = sys.argv[1] if len(sys.argv) > 1 else out_path(__file__)
    open(dest, "w", encoding="utf-8").write(text + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
