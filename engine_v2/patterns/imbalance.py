# engine_v2/patterns/imbalance.py
"""
Imbalance (FVG) pattern detection for POI Zones and Fib activation.

Two concepts:
- Imbalance candle: a c2 (middle) candle whose neighbors (c1, c3) form a gap
  and whose own direction matches the gap direction. Flagged per-candle via
  `is_imbalance == 1` on the dataframe (charting consumes this).
- Imbalance instance: one or more consecutive same-direction imbalance candles
  merged into a single entity with merged gap bounds. Stored as
  ImbalanceInstance objects in `df.attrs["imbalances"]` (Fib/POI consume this).

Detection rules:
- Bullish FVG: c1.high < c3.low AND c2 direction == +1
- Bearish FVG: c1.low  > c3.high AND c2 direction == -1
- Consecutive same-direction imbalance candles form one merged instance.

Merged gap bounds:
- Bullish: gap_bottom = first c1.high, gap_top = last c3.low
- Bearish: gap_bottom = last c3.high,  gap_top = first c1.low
"""
from __future__ import annotations

from typing import List

import pandas as pd

from engine_v2.common.types import ImbalanceInstance


def compute_imbalance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute imbalance candles (column flag) and imbalance instances (attrs list).

    Sets:
    - df["is_imbalance"] (0/1): 1 for every c2 candle that forms an FVG with
      matching c2 direction.
    - df.attrs["imbalances"]: List[ImbalanceInstance] with merged bounds.

    Requires df to already have `direction` column from candle classification.
    """
    df = df.copy()
    df["is_imbalance"] = 0

    if len(df) < 3 or "direction" not in df.columns:
        df.attrs["imbalances"] = []
        return df

    # ---- Pass 1: flag individual imbalance candles ----
    h_vals = df["h"].values
    l_vals = df["l"].values
    dir_vals = df["direction"].values
    is_imb_col = df.columns.get_loc("is_imbalance")

    for idx in range(1, len(df) - 1):
        c1_h = h_vals[idx - 1]
        c1_l = l_vals[idx - 1]
        c3_h = h_vals[idx + 1]
        c3_l = l_vals[idx + 1]
        c2_dir = int(dir_vals[idx])

        # Bullish FVG + bullish c2
        if c1_h < c3_l and c2_dir == 1:
            df.iloc[idx, is_imb_col] = 1
        # Bearish FVG + bearish c2
        elif c1_l > c3_h and c2_dir == -1:
            df.iloc[idx, is_imb_col] = 1

    # ---- Pass 2: merge consecutive same-direction flags into instances ----
    instances: List[ImbalanceInstance] = []
    is_imb_vals = df["is_imbalance"].values

    i = 1
    while i < len(df) - 1:
        if is_imb_vals[i] != 1:
            i += 1
            continue

        direction = int(dir_vals[i])
        run_start = i
        run_end = i
        j = i + 1
        while (
            j < len(df) - 1
            and is_imb_vals[j] == 1
            and int(dir_vals[j]) == direction
        ):
            run_end = j
            j += 1

        if direction == 1:
            gap_bottom = float(h_vals[run_start - 1])  # first c1.high
            gap_top = float(l_vals[run_end + 1])       # last c3.low
        else:
            gap_bottom = float(h_vals[run_end + 1])    # last c3.high
            gap_top = float(l_vals[run_start - 1])     # first c1.low

        instances.append(
            ImbalanceInstance(
                start_idx=run_start,
                end_idx=run_end,
                direction=direction,
                gap_top=gap_top,
                gap_bottom=gap_bottom,
                gap_size=gap_top - gap_bottom,
            )
        )
        i = j

    df.attrs["imbalances"] = instances
    return df


# ---------------------------------------------------------------------------
# Instance-based aggregation helpers
# ---------------------------------------------------------------------------

def has_imbalance_in_range(df: pd.DataFrame, start_idx: int, end_idx: int) -> bool:
    """True if any imbalance instance overlaps [start_idx, end_idx]."""
    return any(
        inst.overlaps(start_idx, end_idx)
        for inst in df.attrs.get("imbalances", [])
    )


def has_unfilled_imbalance(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    check_to_idx: int,
    fill_threshold: float = 0.70,
) -> bool:
    """True if at least one imbalance instance overlapping [start_idx, end_idx]
    is unfilled as of `check_to_idx`."""
    for inst in df.attrs.get("imbalances", []):
        if not inst.overlaps(start_idx, end_idx):
            continue
        if not inst.is_filled(df, check_to_idx, fill_threshold):
            return True
    return False


def has_unfilled_imbalance_in_direction(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    direction: int,
    fill_threshold: float = 0.70,
) -> bool:
    """True if at least one same-direction imbalance instance overlapping
    [start_idx, end_idx] is unfilled as of `end_idx`.

    Used by POI IC candidate validation (unfilled imbalance in struct direction
    must exist AFTER the IC candidate).
    """
    for inst in df.attrs.get("imbalances", []):
        if inst.direction != direction:
            continue
        if not inst.overlaps(start_idx, end_idx):
            continue
        if not inst.is_filled(df, end_idx, fill_threshold):
            return True
    return False


def get_unfilled_imbalances(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    check_to_idx: int,
    fill_threshold: float = 0.70,
) -> List[ImbalanceInstance]:
    """Return all unfilled imbalance instances overlapping [start_idx, end_idx]."""
    return [
        inst
        for inst in df.attrs.get("imbalances", [])
        if inst.overlaps(start_idx, end_idx)
        and not inst.is_filled(df, check_to_idx, fill_threshold)
    ]
