"""Exact demotions of the CURRENT pinbar body-pip floor on the replay window (H1 + M15).

Classifies each TF twice: natural (floor 0) and with the engine's per-TF params
(candle_params_for_timeframe), then lists every candle whose type differs.
Read-only; fetches like the calibration tool. Run from the repo root:
    PYTHONIOENCODING=utf-8 python engine_v2/plans/qa_pinbar_floor_inputs/current_floor_demotions.py OUT.txt
"""
from __future__ import annotations

import sys

import _frames  # noqa: F401  (puts the repo root on sys.path)
import pandas as pd

from engine_v2.config import CONFIG
from engine_v2.features.candle_classifier import candle_params_for_timeframe
from engine_v2.features.candle_params import CandleParams
from engine_v2.features.candles_v2 import compute_candle_metrics, classify_candles
from engine_v2.multitf.data_bridge import fetch_lower_tf_data
from engine_v2.run_replay import fetch_history_with_auto_extend, _floor_to_day_start

OUT = sys.argv[1]


def classify(df_raw, params):
    d = df_raw.copy().reset_index(drop=True)
    d.attrs["pair"] = CONFIG.pair
    return classify_candles(compute_candle_metrics(d), params)


def main():
    pip = 0.01 if "JPY" in CONFIG.pair.upper() else 0.0001
    df_h1, _, _ = fetch_history_with_auto_extend(
        pair=CONFIG.pair, timeframe=CONFIG.timeframe, start=_floor_to_day_start(CONFIG.start),
        end=CONFIG.end, lookback_days=183, min_history=50, extend_days=4, max_extend_iters=8)
    tmin = pd.to_datetime(df_h1["time"]).min()
    tmax = pd.to_datetime(df_h1["time"]).max()
    frames = {"H1": df_h1, "M15": fetch_lower_tf_data(CONFIG.pair, "M15", tmin, tmax)}
    with open(OUT, "w", encoding="utf-8") as fh:
        for tf, raw in frames.items():
            nat = classify(raw, CandleParams())
            cur = classify(raw, candle_params_for_timeframe(tf))
            diff = nat["candle_type"] != cur["candle_type"]
            rows = nat[diff].copy()
            rows["body_p"] = (rows["body_len"] / pip).round(2)
            rows["len_p"] = (rows["candle_len"] / pip).round(2)
            rows["nat"] = nat.loc[diff, "candle_type"]
            rows["cur"] = cur.loc[diff, "candle_type"]
            rows["pdir"] = cur.loc[diff, "pinbar_dir"]
            n = len(nat)
            vc = rows["nat"].value_counts().to_dict()
            fh.write(f"## {tf} floor={candle_params_for_timeframe(tf).pinbar_body_pip_floor} n={n} "
                     f"demoted={len(rows)} ({len(rows)/n*100:.1f}%) by natural type={vc}\n")
            fh.write(f"   natural counts={nat['candle_type'].value_counts().to_dict()} "
                     f"current counts={cur['candle_type'].value_counts().to_dict()}\n")
            # also the 0.2-pip-step demotion counts around the current floor
            for fl in [round(0.1 * k, 2) for k in range(5, 31)]:
                m = nat["candle_type"].isin(["maru", "normal"]) & (nat["body_len"] < fl * pip - 1e-12)
                fh.write(f"   floor {fl:4.1f}: {int(m.sum()):4d} demoted "
                         f"(maru {int((m & (nat.candle_type=='maru')).sum())}, "
                         f"normal {int((m & (nat.candle_type=='normal')).sum())})\n")
            cols = ["time", "o", "h", "l", "c", "body_p", "len_p", "body_pct", "nat", "cur", "pdir"]
            cols = [c for c in cols if c in rows.columns]
            fh.write(rows[cols].to_string() + "\n\n")


if __name__ == "__main__":
    main()
