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


def _require_min_history(start_idx: int, *, min_history: int) -> int:
    """
    Ensure there are at least `min_history` candles available BEFORE start_idx.
    If not, push start_idx forward to min_history.
    """
    if start_idx < min_history:
        return min_history
    return start_idx


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

    start_idx = _require_min_history(start_idx, min_history=min_history)

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
        },
    )


def identify_start_scenario_2_after_reversal(
    df: pd.DataFrame,
    *,
    reversal_idx: int,
    prev_structure_id: int,
    prev_struct_direction: int,
    min_history: int = 50,
) -> StartDecision:
    """
    Scenario 2: after reversal, next structure starts at last CONFIRMED CTS prior to reversal.
    Direction flips vs prior structure.

    Exception #1:
      - If prev_struct_direction == +1 (uptrend before reversal):
          if any candle AFTER last confirmed CTS idx but BEFORE reversal has high > confirmed CTS price,
          start at that candle (highest high) instead.
      - If prev_struct_direction == -1 (downtrend before reversal):
          if any candle AFTER last confirmed CTS idx but BEFORE reversal has low < confirmed CTS price,
          start at that candle (lowest low) instead.
    """
    if reversal_idx not in df.index:
        raise ValueError(f"[identify_start] reversal_idx={reversal_idx} not in df.index")

    # locate last confirmed CTS before reversal for the previous structure
    mask = (
        (df.index < reversal_idx)
        & (df["cts_event"].astype(str) == "CTS_CONFIRMED")
        & (df["structure_id"].astype(int) == int(prev_structure_id))
    )
    cts_conf = df.loc[mask]
    if cts_conf.empty:
        raise ValueError(
            f"[identify_start] No CTS_CONFIRMED found before reversal_idx={reversal_idx} for structure_id={prev_structure_id}"
        )

    last_conf_row = cts_conf.iloc[-1]
    cts_idx = int(last_conf_row["cts_idx"])
    cts_price = float(last_conf_row["cts_price"])

    # default start
    start_idx = cts_idx
    reason = "scenario_2:base_last_cts_confirmed"

    # Exception #1
    if cts_idx < reversal_idx - 1:
        span = df.loc[cts_idx + 1 : reversal_idx - 1]

        if int(prev_struct_direction) == 1:
            # look for higher high than confirmed CTS price
            if not span.empty:
                hh = span["h"].astype(float)
                best_idx = int(hh.idxmax())
                best_price = float(df.loc[best_idx, "h"])
                if best_price > cts_price:
                    start_idx = best_idx
                    reason = "scenario_2:exception1_higher_high_after_last_cts"
        else:
            # look for lower low than confirmed CTS price
            if not span.empty:
                ll = span["l"].astype(float)
                best_idx = int(ll.idxmin())
                best_price = float(df.loc[best_idx, "l"])
                if best_price < cts_price:
                    start_idx = best_idx
                    reason = "scenario_2:exception1_lower_low_after_last_cts"

    start_idx = _require_min_history(start_idx, min_history=min_history)

    return StartDecision(
        start_idx=start_idx,
        struct_direction=int(-1 * int(prev_struct_direction)),
        reason=reason,
        meta={
            "reversal_idx": int(reversal_idx),
            "prev_structure_id": int(prev_structure_id),
            "prev_struct_direction": int(prev_struct_direction),
            "last_cts_confirmed_idx": int(cts_idx),
            "last_cts_confirmed_price": float(cts_price),
        },
    )
