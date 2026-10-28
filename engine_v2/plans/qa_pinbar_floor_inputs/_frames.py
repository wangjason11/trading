"""Shared by the item 7 (pinbar body-pip floor) measurement scripts in this folder.

The frames are fetched exactly like `debug/candle_size_distribution.py` (and so like a real replay):
H1 via `fetch_history_with_auto_extend`, M15 / M5 via `fetch_lower_tf_data` over the H1 span.
M5 is fetch + classify only (the pipeline never runs it).

Run any script here from the REPO ROOT, e.g.
    PYTHONIOENCODING=utf-8 python engine_v2/plans/qa_pinbar_floor_inputs/deciles.py
Each writes `<script stem>.txt` next to itself unless an output path is given.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from engine_v2.config import CONFIG  # noqa: E402
from engine_v2.features.candles_v2 import classify_candles, compute_candle_metrics  # noqa: E402
from engine_v2.multitf.data_bridge import fetch_lower_tf_data  # noqa: E402
from engine_v2.run_replay import _floor_to_day_start, fetch_history_with_auto_extend  # noqa: E402

PIP = 0.01 if "JPY" in CONFIG.pair.upper() else 0.0001


def fetch_frames(tfs=("H1", "M15", "M5")) -> dict:
    """Raw OHLCV frames per TF, positional index 0..n-1 (= the replay's candle idx)."""
    h1, _, _ = fetch_history_with_auto_extend(
        pair=CONFIG.pair, timeframe=CONFIG.timeframe, start=_floor_to_day_start(CONFIG.start),
        end=CONFIG.end, lookback_days=183, min_history=50, extend_days=4, max_extend_iters=8)
    tmin, tmax = pd.to_datetime(h1["time"]).min(), pd.to_datetime(h1["time"]).max()
    out = {}
    for tf in tfs:
        raw = h1 if tf == CONFIG.timeframe else fetch_lower_tf_data(CONFIG.pair, tf, tmin, tmax)
        out[tf] = raw.reset_index(drop=True)
    return out


def classify(raw: pd.DataFrame, params) -> pd.DataFrame:
    """Candle metrics + type (no big flags / volume — this folder reads types and lengths only)."""
    d = raw.copy()
    d.attrs["pair"] = CONFIG.pair
    return classify_candles(compute_candle_metrics(d), params)


def out_path(script_file: str) -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    return os.path.join(os.path.dirname(os.path.abspath(script_file)), Path(script_file).stem + ".txt")
