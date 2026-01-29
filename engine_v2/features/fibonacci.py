# engine_v2/features/fibonacci.py
"""
Fibonacci retracement level calculation for POI Zones.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

import pandas as pd


# Default Fib levels (can be overridden)
DEFAULT_FIB_LEVELS = [50.0, 61.8, 70.5, 78.6]


@dataclass(frozen=True)
class FibLevel:
    """A single Fibonacci level."""
    pct: float      # Percentage (e.g., 61.8)
    price: float    # Calculated price at this level


@dataclass(frozen=True)
class FibRetracement:
    """
    Fibonacci retracement between two anchor points.

    For a bullish move (anchor_low to anchor_high), retracement levels
    measure how far price pulls back from the high.

    For a bearish move (anchor_high to anchor_low), retracement levels
    measure how far price pulls back from the low.
    """
    anchor_high: float
    anchor_low: float
    direction: int  # +1 = bullish swing (low to high), -1 = bearish swing (high to low)
    levels: Tuple[FibLevel, ...]
    anchor_high_idx: int = -1
    anchor_low_idx: int = -1
    meta: Dict[str, Any] = field(default_factory=dict)

    def price_at_pct(self, pct: float) -> float:
        """Calculate price at a given retracement percentage."""
        return calculate_fib_price(
            self.anchor_high,
            self.anchor_low,
            pct,
            self.direction,
        )

    def is_price_between(self, price: float, pct_low: float, pct_high: float) -> bool:
        """Check if price is between two retracement levels."""
        p_low = self.price_at_pct(pct_low)
        p_high = self.price_at_pct(pct_high)
        return min(p_low, p_high) <= price <= max(p_low, p_high)


def calculate_fib_price(
    anchor_high: float,
    anchor_low: float,
    pct: float,
    direction: int,
) -> float:
    """
    Calculate the price at a given Fibonacci retracement percentage.

    Parameters
    ----------
    anchor_high : float
        The high anchor point
    anchor_low : float
        The low anchor point
    pct : float
        Retracement percentage (e.g., 61.8)
    direction : int
        +1 for bullish swing (retracement from high toward low)
        -1 for bearish swing (retracement from low toward high)

    Returns
    -------
    float
        Price at the given Fib level
    """
    range_size = anchor_high - anchor_low
    retracement = range_size * (pct / 100.0)

    if direction == 1:
        # Bullish: price moved up, retracement pulls back from high
        return anchor_high - retracement
    else:
        # Bearish: price moved down, retracement pulls back from low
        return anchor_low + retracement


def create_fib_retracement(
    anchor_high: float,
    anchor_low: float,
    direction: int,
    *,
    levels: List[float] = None,
    anchor_high_idx: int = -1,
    anchor_low_idx: int = -1,
    meta: Dict[str, Any] = None,
) -> FibRetracement:
    """
    Create a FibRetracement object with calculated levels.

    Parameters
    ----------
    anchor_high : float
        High price anchor
    anchor_low : float
        Low price anchor
    direction : int
        +1 bullish, -1 bearish
    levels : list of float, optional
        Fib percentages to calculate. Defaults to DEFAULT_FIB_LEVELS.
    anchor_high_idx : int
        DataFrame index of the high anchor
    anchor_low_idx : int
        DataFrame index of the low anchor
    meta : dict, optional
        Additional metadata

    Returns
    -------
    FibRetracement
    """
    if levels is None:
        levels = DEFAULT_FIB_LEVELS

    fib_levels = []
    for pct in levels:
        price = calculate_fib_price(anchor_high, anchor_low, pct, direction)
        fib_levels.append(FibLevel(pct=pct, price=price))

    return FibRetracement(
        anchor_high=anchor_high,
        anchor_low=anchor_low,
        direction=direction,
        levels=tuple(fib_levels),
        anchor_high_idx=anchor_high_idx,
        anchor_low_idx=anchor_low_idx,
        meta=meta or {},
    )


def fib_from_swing(
    df: pd.DataFrame,
    high_idx: int,
    low_idx: int,
    *,
    levels: List[float] = None,
) -> FibRetracement:
    """
    Create FibRetracement from swing high/low indices in a DataFrame.

    Direction is inferred: if high_idx < low_idx, it's a bearish swing;
    if low_idx < high_idx, it's a bullish swing.

    Parameters
    ----------
    df : DataFrame
        OHLC data with 'h' and 'l' columns
    high_idx : int
        Index of the swing high candle
    low_idx : int
        Index of the swing low candle
    levels : list of float, optional
        Fib percentages

    Returns
    -------
    FibRetracement
    """
    anchor_high = float(df.loc[high_idx, "h"])
    anchor_low = float(df.loc[low_idx, "l"])

    # Determine direction based on time sequence
    # If high came before low: bearish swing (price moved down)
    # If low came before high: bullish swing (price moved up)
    direction = -1 if high_idx < low_idx else 1

    return create_fib_retracement(
        anchor_high=anchor_high,
        anchor_low=anchor_low,
        direction=direction,
        levels=levels,
        anchor_high_idx=high_idx,
        anchor_low_idx=low_idx,
        meta={"source": "swing"},
    )
