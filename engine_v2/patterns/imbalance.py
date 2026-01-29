# engine_v2/patterns/imbalance.py
"""
Imbalance (FVG-like) pattern detection for POI Zones.

An imbalance is a 3-candle pattern where there's a gap between
candle 1's wick and candle 3's wick, indicating strong directional movement.

This module computes imbalance as DataFrame columns (pattern-style),
not as separate events. The middle candle (c2) receives the flag.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def compute_imbalance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute imbalance pattern flags for the DataFrame.

    Adds columns:
    - is_imbalance: 1 if imbalance detected (middle candle), 0 otherwise
    - imbalance_gap_size: size of the gap (0 if no imbalance)

    Direction is determined by the existing 'direction' column.

    Standard FVG logic:
    - Bullish: c1.high < c3.low (gap between them)
    - Bearish: c1.low > c3.high (gap between them)

    Parameters
    ----------
    df : DataFrame
        OHLC data with columns: h, l, direction

    Returns
    -------
    DataFrame with is_imbalance and imbalance_gap_size columns added
    """
    df = df.copy()

    # Initialize columns
    df["is_imbalance"] = 0
    df["imbalance_gap_size"] = 0.0

    # Need at least 3 candles
    if len(df) < 3:
        return df

    # Get arrays for vectorized comparison
    h = df["h"].values
    l = df["l"].values

    # c1 = idx-1, c2 = idx (middle), c3 = idx+1
    # We check indices 1 to len-2 (so c1 and c3 are valid)

    for idx in range(1, len(df) - 1):
        c1_h = h[idx - 1]
        c1_l = l[idx - 1]
        c3_h = h[idx + 1]
        c3_l = l[idx + 1]

        # Bullish imbalance: gap between c1.high and c3.low
        if c1_h < c3_l:
            gap_size = c3_l - c1_h
            df.iloc[idx, df.columns.get_loc("is_imbalance")] = 1
            df.iloc[idx, df.columns.get_loc("imbalance_gap_size")] = gap_size

        # Bearish imbalance: gap between c1.low and c3.high
        elif c1_l > c3_h:
            gap_size = c1_l - c3_h
            df.iloc[idx, df.columns.get_loc("is_imbalance")] = 1
            df.iloc[idx, df.columns.get_loc("imbalance_gap_size")] = gap_size

    return df


def has_imbalance_in_range(df: pd.DataFrame, start_idx: int, end_idx: int) -> bool:
    """
    Check if any imbalance exists in the given index range.

    Parameters
    ----------
    df : DataFrame
        Must have 'is_imbalance' column computed
    start_idx : int
        Start of range (inclusive)
    end_idx : int
        End of range (inclusive)

    Returns
    -------
    bool
        True if at least one imbalance exists in range
    """
    if "is_imbalance" not in df.columns:
        return False

    mask = (df.index >= start_idx) & (df.index <= end_idx)
    return df.loc[mask, "is_imbalance"].sum() > 0
