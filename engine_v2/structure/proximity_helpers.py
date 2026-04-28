# engine_v2/structure/proximity_helpers.py
"""Helpers for inline zone-proximity check inside MarketStructure.

Used by the structure state machine to detect first sd zone proximity
trigger after CTS_ESTABLISHED, which can confirm CTS as an alternative
to the pullback pattern.

Stage 1: BOS-only proximity reference. POI inclusion comes in Stage 2.

This module is a backward-dependency from `structure/` into `zones/` —
the major refactor is expected to clean this up.
"""
from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from engine_v2.zones.kl_zones_v1 import identify_base_pattern, zone_thresholds


def compute_bos_inner_from_event(
    df: pd.DataFrame,
    bos_idx: int,
    struct_direction: int,
    length_threshold: float = 0.7,
) -> Optional[float]:
    """Derive the BOS zone's inner price for proximity checking.

    Mirrors the BOS-zone derivation logic used by `derive_kl_zones_v1`:
    1) Identify base pattern at the BOS anchor
    2) Resolve threshold(s) via zone_thresholds(...)
    3) Return the inner bound

    Parameters
    ----------
    df : DataFrame
        OHLC + candle classification columns must be present.
    bos_idx : int
        Anchor candle of the BOS (the BOS extreme idx — `bos_event.idx`).
    struct_direction : int
        +1 bullish, -1 bearish.
    length_threshold : float
        Pattern threshold (default 0.7), passed through to identify_base_pattern.

    Returns
    -------
    Optional[float]
        Inner price of the BOS zone, or None if it can't be computed.
    """
    if bos_idx not in df.index:
        return None

    try:
        zone_pattern, base_idx = identify_base_pattern(
            df,
            anchor_idx=int(bos_idx),
            struct_direction=int(struct_direction),
            bos=True,
            length_threshold=length_threshold,
        )
        if base_idx is None or base_idx not in df.index:
            return None

        _outer, inner = zone_thresholds(
            df,
            base_idx=int(base_idx),
            struct_direction=int(struct_direction),
            zone_pattern=zone_pattern,
            bos=True,
        )
        if inner is None:
            return None
        return float(inner)
    except Exception:
        # Defensive: if base pattern derivation fails for any reason,
        # fall back to no proximity reference (no proximity check fires).
        return None


def check_sd_proximity_at_candle(
    df: pd.DataFrame,
    candle_idx: int,
    struct_direction: int,
    bos_inner: float,
    threshold: float,
) -> Optional[Tuple[float, str]]:
    """Check if a candle wick comes within `threshold` of the sd-direction
    inner bound.

    Stage 1 considers only the BOS inner. POIs will be added in Stage 2.

    For sd=+1 (bullish): trigger when candle.low <= bos_inner + threshold
                         (price approaching from above).
    For sd=-1 (bearish): trigger when candle.high >= bos_inner - threshold
                         (price approaching from below).

    Parameters
    ----------
    threshold : float
        Pip threshold * pip_size (e.g., 20 * 0.0001 = 0.0020 for H1).

    Returns
    -------
    Optional[Tuple[float, str]]
        (trigger_inner, zone_kind) on hit; None otherwise.
        zone_kind is currently always "BOS" for Stage 1.
    """
    if candle_idx not in df.index:
        return None

    if struct_direction == 1:
        candle_low = float(df.loc[candle_idx, "l"])
        if candle_low <= bos_inner + threshold:
            return (bos_inner, "BOS")
    else:
        candle_high = float(df.loc[candle_idx, "h"])
        if candle_high >= bos_inner - threshold:
            return (bos_inner, "BOS")

    return None
