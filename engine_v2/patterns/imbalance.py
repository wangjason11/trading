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


def get_imbalance_gap_bounds(df: pd.DataFrame, imb_idx: int) -> tuple[float, float, int]:
    """
    Get the FVG gap bounds for an imbalance candle.

    Parameters
    ----------
    df : DataFrame
        OHLC data with h, l columns
    imb_idx : int
        Index of the middle candle (the one flagged as imbalance)

    Returns
    -------
    tuple of (gap_bottom, gap_top, direction)
        direction: 1 for bullish, -1 for bearish
        Returns (0, 0, 0) if not a valid imbalance
    """
    # imb_idx is the middle candle (c2), so c1 = imb_idx-1, c3 = imb_idx+1
    if imb_idx - 1 not in df.index or imb_idx + 1 not in df.index:
        return (0.0, 0.0, 0)

    c1_h = float(df.loc[imb_idx - 1, "h"])
    c1_l = float(df.loc[imb_idx - 1, "l"])
    c3_h = float(df.loc[imb_idx + 1, "h"])
    c3_l = float(df.loc[imb_idx + 1, "l"])

    # Bullish: gap between c1.high and c3.low
    if c1_h < c3_l:
        return (c1_h, c3_l, 1)  # gap_bottom, gap_top, bullish

    # Bearish: gap between c3.high and c1.low
    if c1_l > c3_h:
        return (c3_h, c1_l, -1)  # gap_bottom, gap_top, bearish

    return (0.0, 0.0, 0)


def is_imbalance_filled(
    df: pd.DataFrame,
    imb_idx: int,
    check_to_idx: int,
    fill_threshold: float = 0.70,
) -> bool:
    """
    Check if an imbalance at imb_idx is filled by candles from imb_idx+1 to check_to_idx.

    Filled means price retraced >= fill_threshold (default 70%) into the FVG gap.

    Parameters
    ----------
    df : DataFrame
        OHLC data
    imb_idx : int
        Index of the imbalance candle (middle candle)
    check_to_idx : int
        End index to check (inclusive)
    fill_threshold : float
        Percentage of gap that must be filled (0.70 = 70%)

    Returns
    -------
    bool
        True if imbalance is filled, False if unfilled
    """
    gap_bottom, gap_top, direction = get_imbalance_gap_bounds(df, imb_idx)

    if direction == 0:
        return True  # Not a valid imbalance, treat as filled

    gap_size = gap_top - gap_bottom
    if gap_size <= 0:
        return True  # Invalid gap

    # For bullish imbalance: price needs to come DOWN into the gap
    # Fill level = gap_top - (gap_size * fill_threshold)
    # Filled if any candle's low <= fill level

    # For bearish imbalance: price needs to go UP into the gap
    # Fill level = gap_bottom + (gap_size * fill_threshold)
    # Filled if any candle's high >= fill level

    if direction == 1:  # Bullish
        fill_level = gap_top - (gap_size * fill_threshold)
        # Check if any candle from imb_idx+1 to check_to_idx has low <= fill_level
        for idx in range(imb_idx + 1, check_to_idx + 1):
            if idx not in df.index:
                continue
            if float(df.loc[idx, "l"]) <= fill_level:
                return True
    else:  # Bearish
        fill_level = gap_bottom + (gap_size * fill_threshold)
        # Check if any candle from imb_idx+1 to check_to_idx has high >= fill_level
        for idx in range(imb_idx + 1, check_to_idx + 1):
            if idx not in df.index:
                continue
            if float(df.loc[idx, "h"]) >= fill_level:
                return True

    return False


def has_unfilled_imbalance(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    check_to_idx: int,
    fill_threshold: float = 0.70,
) -> bool:
    """
    Check if at least one unfilled imbalance exists in the range [start_idx, end_idx].

    Parameters
    ----------
    df : DataFrame
        Must have 'is_imbalance' column computed
    start_idx : int
        Start of imbalance search range (inclusive)
    end_idx : int
        End of imbalance search range (inclusive)
    check_to_idx : int
        Index to check fill status against (typically CTS anchor)
    fill_threshold : float
        Percentage threshold for fill (default 0.70 = 70%)

    Returns
    -------
    bool
        True if at least one imbalance in range is unfilled
    """
    if "is_imbalance" not in df.columns:
        return False

    # Find all imbalance indices in range
    mask = (df.index >= start_idx) & (df.index <= end_idx) & (df["is_imbalance"] == 1)
    imbalance_indices = df.index[mask].tolist()

    if not imbalance_indices:
        return False  # No imbalances in range

    # Check each imbalance - return True if ANY is unfilled
    for imb_idx in imbalance_indices:
        if not is_imbalance_filled(df, imb_idx, check_to_idx, fill_threshold):
            return True

    return False  # All imbalances are filled


def get_unfilled_imbalances(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    check_to_idx: int,
    fill_threshold: float = 0.70,
) -> list[int]:
    """
    Get list of unfilled imbalance indices in the range.

    Parameters
    ----------
    df : DataFrame
        Must have 'is_imbalance' column computed
    start_idx : int
        Start of imbalance search range (inclusive)
    end_idx : int
        End of imbalance search range (inclusive)
    check_to_idx : int
        Index to check fill status against
    fill_threshold : float
        Percentage threshold for fill

    Returns
    -------
    list of int
        Indices of unfilled imbalances
    """
    if "is_imbalance" not in df.columns:
        return []

    mask = (df.index >= start_idx) & (df.index <= end_idx) & (df["is_imbalance"] == 1)
    imbalance_indices = df.index[mask].tolist()

    unfilled = []
    for imb_idx in imbalance_indices:
        if not is_imbalance_filled(df, imb_idx, check_to_idx, fill_threshold):
            unfilled.append(imb_idx)

    return unfilled
