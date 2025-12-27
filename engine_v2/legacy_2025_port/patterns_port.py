from __future__ import annotations

import numpy as np
import pandas as pd

from engine_v2.common.types import COL_C, COL_H, COL_L, COL_O, REQUIRED_CANDLE_COLS


def apply_basic_entry_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a small starter set of entry pattern columns.
    (Week 1 goal is runnable + sanity-checkable, not completeness.)

    Columns added:
      - engulfing: int in {-1,0,1}
      - star: int in {-1,0,1}   (simple pinbar-based star concept)
    """
    _validate(df)

    out = df.copy()

    # Engulfing based on OC engulf (your requested definition earlier):
    # current candle open+close engulf previous open+close
    prev_o = out[COL_O].shift(1)
    prev_c = out[COL_C].shift(1)

    curr_o = out[COL_O]
    curr_c = out[COL_C]

    prev_hi = np.maximum(prev_o, prev_c)
    prev_lo = np.minimum(prev_o, prev_c)

    curr_hi = np.maximum(curr_o, curr_c)
    curr_lo = np.minimum(curr_o, curr_c)

    # bullish engulf: current direction = 1 and its body engulfs previous body
    bullish = (out["direction"] == 1) & (out["direction"].shift(1) == -1) & (curr_hi > prev_hi) & (curr_lo < prev_lo)
    bearish = (out["direction"] == -1) & (out["direction"].shift(1) == 1) & (curr_hi > prev_hi) & (curr_lo < prev_lo)

    out["engulfing"] = 0
    out.loc[bullish, "engulfing"] = 1
    out.loc[bearish, "engulfing"] = -1

    # "Star" (minimal usable definition for now):
    # treat as pinbar + direction, with breakout confirmation later (Week 2+)
    out["star"] = 0
    is_pinbar = out["candle_type"] == "pinbar"
    out.loc[is_pinbar & (out["direction"] == 1), "star"] = 1
    out.loc[is_pinbar & (out["direction"] == -1), "star"] = -1

    return out


def _validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_CANDLE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[patterns_port] Missing required columns: {missing}")
    if df.empty:
        raise ValueError("[patterns_port] Input df is empty")
    # candle features must exist (created by candles_port)
    for col in ["direction", "candle_type"]:
        if col not in df.columns:
            raise ValueError(f"[patterns_port] Missing candle feature column '{col}'. Run candle features first.")
