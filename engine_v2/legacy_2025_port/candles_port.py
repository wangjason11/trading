from __future__ import annotations

import numpy as np
import pandas as pd

from engine_v2.common.types import COL_C, COL_H, COL_L, COL_O, COL_TIME, COL_V, REQUIRED_CANDLE_COLS


def compute_candle_features(
    df: pd.DataFrame,
    *,
    pinbar_max_body_ratio: float = 0.35,
    pinbar_min_wick_ratio: float = 0.55,
    maru_max_wick_ratio: float = 0.10,
) -> pd.DataFrame:
    """
    Adds core candle-derived features needed by patterns/structure.

    Output columns (minimum):
      - direction: int in {-1, 0, 1}
      - candle_length: float
      - body_length: float
      - upper_wick: float
      - lower_wick: float
      - body_ratio: float
      - upper_wick_ratio: float
      - lower_wick_ratio: float
      - candle_type: {"pinbar","maru","normal"}
    """
    _validate(df)

    out = df.copy()

    o = out[COL_O].astype(float)
    h = out[COL_H].astype(float)
    l = out[COL_L].astype(float)
    c = out[COL_C].astype(float)

    # Direction: bullish (1) if c>o, bearish (-1) if c<o, neutral (0) if equal
    out["direction"] = np.where(c > o, 1, np.where(c < o, -1, 0)).astype(int)

    out["candle_length"] = (h - l).astype(float)
    out["body_length"] = (c - o).abs().astype(float)

    # Avoid division by zero
    cl = out["candle_length"].replace(0.0, np.nan)

    out["upper_wick"] = (h - np.maximum(o, c)).clip(lower=0).astype(float)
    out["lower_wick"] = (np.minimum(o, c) - l).clip(lower=0).astype(float)

    out["body_ratio"] = (out["body_length"] / cl).fillna(0.0)
    out["upper_wick_ratio"] = (out["upper_wick"] / cl).fillna(0.0)
    out["lower_wick_ratio"] = (out["lower_wick"] / cl).fillna(0.0)

    # Candle type rules
    is_pinbar = (out["body_ratio"] <= pinbar_max_body_ratio) & (
        (out["upper_wick_ratio"] >= pinbar_min_wick_ratio)
        | (out["lower_wick_ratio"] >= pinbar_min_wick_ratio)
    )

    is_maru = (
        (out["upper_wick_ratio"] <= maru_max_wick_ratio)
        & (out["lower_wick_ratio"] <= maru_max_wick_ratio)
        & (out["body_ratio"] >= (1.0 - 2 * maru_max_wick_ratio))
    )

    out["candle_type"] = np.where(is_pinbar, "pinbar", np.where(is_maru, "maru", "normal"))

    # Ensure volume exists and is numeric (we made it required)
    out[COL_V] = out[COL_V].astype(float)

    return out


def _validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_CANDLE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[candles_port] Missing required columns: {missing}")
    if df.empty:
        raise ValueError("[candles_port] Input df is empty")
