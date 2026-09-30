# engine_v2/structure/identify_start.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd


@dataclass
class StartDecision:
    start_idx: int
    struct_direction: int     # +1 up, -1 down
    reason: str
    meta: dict


# def _require_min_history(start_idx: int, *, min_history: int) -> int:
#     """
#     Ensure there are at least `min_history` candles available BEFORE start_idx.
#     If not, push start_idx forward to min_history.
#     """
#     if start_idx < min_history:
#         return min_history
#     return start_idx

def _enforce_min_history(start_idx: int, *, min_history: int) -> tuple[int, bool]:
    """
    Require at least `min_history` candles BEFORE the chosen start_idx.
    Returns: (effective_start_idx, too_early)

    NOTE: We intentionally DO NOT silently "fix" the underlying data sufficiency issue.
    Callers can detect `too_early=True` via StartDecision.meta and decide to fetch more history.
    """
    if start_idx < min_history:
        return int(min_history), True
    return int(start_idx), False


def identify_start_scenario_1(
    df: pd.DataFrame,
    *,
    input_idx: int,
    lookback_days: int = 182,
    min_history: int = 50,
) -> StartDecision:
    """
    Scenario 1 (HTF): from input_idx (input candle), go back ~half-year, find extremes in the window:
      - high extreme: max(h)
      - low extreme: min(l)

    Shape cases based on:
      - order of extremes (low before high vs high before low)
      - whether current candle is itself the high extreme / low extreme

    Regardless of shape: start candle is ALWAYS the furthest-back extreme.
      - low before high => start at low extreme, direction +1
      - high before low => start at high extreme, direction -1
    """
    if df.empty:
        raise ValueError("[identify_start] df is empty")

    if input_idx not in df.index:
        raise ValueError(f"[identify_start] input_idx={input_idx} not in df.index")

    if "time" not in df.columns or "h" not in df.columns or "l" not in df.columns:
        raise ValueError("[identify_start] df must have columns: time, h, l")

    end_time = pd.to_datetime(df.loc[input_idx, "time"], utc=True)
    start_time = end_time - pd.Timedelta(days=int(lookback_days))

    # window start index: first candle >= start_time
    w = df[df["time"].astype("datetime64[ns, UTC]") >= start_time]
    if w.empty:
        # fallback: use earliest available
        win_start_idx = int(df.index.min())
    else:
        win_start_idx = int(w.index.min())

    # clamp and ensure input_idx included
    win = df.loc[win_start_idx:input_idx].copy()
    if win.empty:
        raise ValueError("[identify_start] lookback window is empty after slicing")

    # extremes
    hi_idx = int(win["h"].astype(float).idxmax())
    lo_idx = int(win["l"].astype(float).idxmin())
    hi_price = float(df.loc[hi_idx, "h"])
    lo_price = float(df.loc[lo_idx, "l"])

    # "current price is extreme" means end candle matches the extreme candle index
    current_is_hi = (input_idx == hi_idx)
    current_is_lo = (input_idx == lo_idx)

    if lo_idx < hi_idx:
        # ascending or lambda
        start_idx = lo_idx
        struct_direction = 1
        shape = "ascending" if current_is_hi else "lambda"
    else:
        # descending or V
        start_idx = hi_idx
        struct_direction = -1
        shape = "descending" if current_is_lo else "v"

    # start_idx = _require_min_history(start_idx, min_history=min_history)
    raw_start_idx = int(start_idx)
    start_idx, too_early = _enforce_min_history(start_idx, min_history=min_history)

    return StartDecision(
        start_idx=start_idx,
        struct_direction=struct_direction,
        reason=f"scenario_1:{shape}",
        meta={
            "input_idx": int(input_idx),
            "lookback_days": int(lookback_days),
            "window_start_idx": int(win_start_idx),
            "hi_idx": int(hi_idx),
            "hi_price": float(hi_price),
            "lo_idx": int(lo_idx),
            "lo_price": float(lo_price),
            "current_is_hi": bool(current_is_hi),
            "current_is_lo": bool(current_is_lo),
            "raw_start_idx": int(raw_start_idx),
            "min_history": int(min_history),
            "too_early": bool(too_early),
        },
    )
