# engine_v2/structure/proximity_helpers.py
"""Helpers for inline zone-proximity check inside MarketStructure.

Used by the structure state machine to detect first sd zone proximity
trigger after CTS_ESTABLISHED, which can confirm CTS as an alternative
to the pullback pattern.

Stage 1: BOS-only proximity reference.
Stage 2: BOS inner + active POI inners (full spec compliance).

This module is a backward-dependency from `structure/` into `zones/` —
the major refactor is expected to clean this up.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd

from engine_v2.features.fibonacci import create_fib_retracement, DEFAULT_FIB_LEVELS
from engine_v2.patterns.imbalance import has_unfilled_imbalance_in_direction
from engine_v2.zones.fib_tracker import FibState
from engine_v2.zones.kl_zones_v1 import identify_base_pattern, zone_thresholds
from engine_v2.zones.poi_zones import (
    POIConfig,
    find_ic_candidates,
    select_ic_variants,
)


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


def compute_poi_inners_for_cycle(
    df: pd.DataFrame,
    bos_idx: int,
    bos_price: float,
    cts_idx: int,
    cts_price: float,
    struct_direction: int,
    structure_id: int = 0,
    cycle_id: int = 0,
    fill_threshold: float = 0.70,
) -> List[float]:
    """Derive POI zone inner prices for the cycle's current Fib state.

    Constructs a FibState inline from BOS_n + CTS_n, runs IC candidate
    scan + variant selection, and returns the list of POI inner prices
    (top of IC for buy zones, bottom for sell). Caller invokes this at
    CTS_ESTABLISHED + each CTS_UPDATED to refresh the per-cycle snapshot.

    Note: POIs are always sd-direction by Fib construction (Fib spans
    BOS->CTS), so all returned inners belong to sd zones.

    Parameters
    ----------
    bos_idx, bos_price : the BOS_n level
    cts_idx, cts_price : the current CTS_n extreme (may extend over time)
    struct_direction : +1 / -1
    fill_threshold : POI's unfilled-imbalance fill threshold (default 0.70)

    Returns
    -------
    List[float]
        POI inner prices. Empty list if no qualifying IC candidates.
        Returns [] gracefully on any error (defensive — proximity check
        falls back to BOS-only).
    """
    if cts_idx <= bos_idx:
        return []

    try:
        sd = int(struct_direction)
        if sd == 1:
            anchor_high, anchor_high_idx = float(cts_price), int(cts_idx)
            anchor_low, anchor_low_idx = float(bos_price), int(bos_idx)
        else:
            anchor_high, anchor_high_idx = float(bos_price), int(bos_idx)
            anchor_low, anchor_low_idx = float(cts_price), int(cts_idx)

        fib = create_fib_retracement(
            anchor_high=anchor_high,
            anchor_low=anchor_low,
            direction=sd,
            levels=DEFAULT_FIB_LEVELS,
            anchor_high_idx=anchor_high_idx,
            anchor_low_idx=anchor_low_idx,
            meta={"structure_id": structure_id, "cycle_id": cycle_id},
        )
        fib_state = FibState(
            structure_id=structure_id,
            cycle_id=cycle_id,
            struct_direction=sd,
            bos_idx=int(bos_idx),
            bos_price=float(bos_price),
            cts_idx=int(cts_idx),
            cts_price=float(cts_price),
            active=True,
            locked=False,
            fib=fib,
        )

        config = POIConfig(fill_threshold=fill_threshold)
        candidates = find_ic_candidates(df, fib_state, config)
        if not candidates:
            return []
        ic_variants = select_ic_variants(candidates, df, fib_state, config)
        if not ic_variants:
            return []

        inners: List[float] = []
        for ic_idx in ic_variants.keys():
            if ic_idx not in df.index:
                continue
            if sd == 1:
                inners.append(float(df.loc[ic_idx, "h"]))  # buy POI inner = IC top
            else:
                inners.append(float(df.loc[ic_idx, "l"]))  # sell POI inner = IC bottom
        return inners
    except Exception:
        # Defensive: if inline POI derivation fails, return empty list
        # — proximity check falls back to BOS-only without crashing.
        return []


def check_sd_proximity_at_candle(
    df: pd.DataFrame,
    candle_idx: int,
    struct_direction: int,
    bos_inner: float,
    threshold: float,
    poi_inners: Optional[List[float]] = None,
) -> Optional[Tuple[float, str]]:
    """Check if a candle wick comes within `threshold` of the closest
    sd-direction inner bound (BOS or POI).

    Stage 2: includes POI inners alongside BOS inner. Picks the closest
    inner to current price ("max for buy / min for sell").

    For sd=+1 (bullish): trigger when candle.low <= chosen_inner + threshold
                         (price approaching from above).
    For sd=-1 (bearish): trigger when candle.high >= chosen_inner - threshold
                         (price approaching from below).

    Parameters
    ----------
    bos_inner : float
        BOS zone inner bound (always present).
    threshold : float
        Pip threshold * pip_size (e.g., 20 * 0.0001 = 0.0020 for H1).
    poi_inners : Optional[List[float]]
        Active POI zone inner bounds for the cycle. None or empty list
        falls back to BOS-only behavior.

    Returns
    -------
    Optional[Tuple[float, str]]
        (trigger_inner, zone_kind) on hit; None otherwise.
        zone_kind is "BOS" or "POI" depending on which inner won.
    """
    if candle_idx not in df.index:
        return None

    # Build candidate inners with their zone-kind labels
    inners: List[Tuple[float, str]] = [(float(bos_inner), "BOS")]
    if poi_inners:
        for v in poi_inners:
            inners.append((float(v), "POI"))

    # Pick closest-to-current-price inner: max for buy, min for sell
    if struct_direction == 1:
        chosen_inner, zone_kind = max(inners, key=lambda t: t[0])
        candle_low = float(df.loc[candle_idx, "l"])
        if candle_low <= chosen_inner + threshold:
            return (chosen_inner, zone_kind)
    else:
        chosen_inner, zone_kind = min(inners, key=lambda t: t[0])
        candle_high = float(df.loc[candle_idx, "h"])
        if candle_high >= chosen_inner - threshold:
            return (chosen_inner, zone_kind)

    return None
