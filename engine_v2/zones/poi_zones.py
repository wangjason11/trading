# engine_v2/zones/poi_zones.py
"""
POI Zone derivation using Fibonacci retracement and Institutional Candle identification.

POI Zones are created when:
1. An imbalance pattern is detected
2. Fib retracement levels are drawn from anchor points
3. An Institutional Candle (IC) is identified within specific Fib bounds
4. The IC's high/low define the POI Zone boundaries
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Dict, Any, Literal

import pandas as pd

from engine_v2.features.fibonacci import (
    FibRetracement,
    create_fib_retracement,
    DEFAULT_FIB_LEVELS,
)
from engine_v2.patterns.imbalance import has_imbalance_in_range
from engine_v2.structure.market_structure import StructureEvent


@dataclass(frozen=True)
class POIZone:
    """
    Point of Interest Zone derived from Fibonacci + Institutional Candle.

    Similar structure to KLZone for consistency.
    """
    start_time: "pd.Timestamp"
    end_time: Optional["pd.Timestamp"]  # None = extends to end of chart
    side: Literal["buy", "sell"]
    top: float
    bottom: float

    # Source information
    ic_idx: int  # Index of the Institutional Candle
    fib: FibRetracement  # The Fib retracement used
    has_imbalance: bool  # Whether imbalance exists between anchors

    # Metadata
    strength: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


# Configuration for POI Zone detection
@dataclass
class POIConfig:
    """Configuration for POI Zone detection."""
    fib_levels: List[float] = field(default_factory=lambda: DEFAULT_FIB_LEVELS.copy())

    # IC search bounds (as Fib percentages)
    ic_fib_min: float = 62.0  # IC must be at or above this retracement
    ic_fib_max: float = 79.0  # IC must be at or below this retracement

    # Imbalance detection
    require_imbalance: bool = True
    min_imbalance_gap: float = 0.0

    # Additional filters (TBD)
    # ic_candle_types: List[str] = field(default_factory=lambda: ["maru", "normal"])


def find_institutional_candle(
    df: pd.DataFrame,
    fib: FibRetracement,
    config: POIConfig,
    *,
    search_start_idx: int,
    search_end_idx: int,
) -> Optional[int]:
    """
    Find the Institutional Candle (IC) within the Fib retracement bounds.

    The IC is a candle whose body (or wick) falls within the specified
    Fib retracement zone (ic_fib_min to ic_fib_max).

    Parameters
    ----------
    df : DataFrame
        OHLC data
    fib : FibRetracement
        The Fib retracement to use for bounds
    config : POIConfig
        Configuration with ic_fib_min and ic_fib_max
    search_start_idx : int
        Start of search range
    search_end_idx : int
        End of search range (exclusive)

    Returns
    -------
    int or None
        Index of the IC, or None if not found
    """
    # Calculate price bounds for IC search
    fib_min_price = fib.price_at_pct(config.ic_fib_min)
    fib_max_price = fib.price_at_pct(config.ic_fib_max)

    # Ensure min/max are ordered correctly
    zone_low = min(fib_min_price, fib_max_price)
    zone_high = max(fib_min_price, fib_max_price)

    # Search for IC within bounds
    for i in range(search_start_idx, min(search_end_idx, len(df))):
        row = df.iloc[i]
        candle_low = float(row.l)
        candle_high = float(row.h)

        # Check if candle overlaps with the Fib zone
        # IC is valid if any part of the candle is within the zone
        if candle_high >= zone_low and candle_low <= zone_high:
            return i

    return None


def derive_poi_zones(
    df: pd.DataFrame,
    events: List[StructureEvent],
    *,
    config: Optional[POIConfig] = None,
) -> List[POIZone]:
    """
    Derive POI Zones from structure events.

    This is the main entry point for POI Zone detection.

    Parameters
    ----------
    df : DataFrame
        OHLC data with required columns
    events : list of StructureEvent
        Structure events (BOS_CONFIRMED, CTS_CONFIRMED, etc.)
    config : POIConfig, optional
        Configuration for POI detection

    Returns
    -------
    List of POIZone
    """
    if config is None:
        config = POIConfig()

    zones: List[POIZone] = []

    # Helper to convert idx to time
    def _time(i: int):
        return pd.to_datetime(df.loc[i, "time"], utc=True)

    # Process structure events to find POI Zone candidates
    for ev in events:
        if ev.type not in ("BOS_CONFIRMED", "CTS_CONFIRMED"):
            continue

        # Get structure direction and confirmed index
        sd = int((ev.meta or {}).get("struct_direction", 0))
        confirmed_idx = int((ev.meta or {}).get("confirmed_at", ev.idx))
        structure_id = int((ev.meta or {}).get("structure_id", -1))

        # Skip if no valid direction
        if sd == 0:
            continue

        # Determine Fib anchor points based on event type and direction
        # This is a simplified implementation - actual anchors TBD
        # For now: use ev.idx as one anchor and search backward for the other

        # Find swing points for Fib (simplified)
        anchor_idx = int(ev.idx)
        lookback = 20  # Configurable

        if sd == 1:  # Bullish structure
            # For bullish: anchor_low is the swing low before the move
            search_start = max(0, anchor_idx - lookback)
            swing_low_idx = df.loc[search_start:anchor_idx, "l"].idxmin()
            swing_high_idx = anchor_idx

            fib = create_fib_retracement(
                anchor_high=float(df.loc[swing_high_idx, "h"]),
                anchor_low=float(df.loc[swing_low_idx, "l"]),
                direction=1,
                levels=config.fib_levels,
                anchor_high_idx=swing_high_idx,
                anchor_low_idx=swing_low_idx,
                meta={"event_type": ev.type, "structure_id": structure_id},
            )
        else:  # Bearish structure
            # For bearish: anchor_high is the swing high before the move
            search_start = max(0, anchor_idx - lookback)
            swing_high_idx = df.loc[search_start:anchor_idx, "h"].idxmax()
            swing_low_idx = anchor_idx

            fib = create_fib_retracement(
                anchor_high=float(df.loc[swing_high_idx, "h"]),
                anchor_low=float(df.loc[swing_low_idx, "l"]),
                direction=-1,
                levels=config.fib_levels,
                anchor_high_idx=swing_high_idx,
                anchor_low_idx=swing_low_idx,
                meta={"event_type": ev.type, "structure_id": structure_id},
            )

        # Check for imbalance if required (now uses column-based check)
        has_imbalance = False
        if config.require_imbalance:
            # Check if imbalance exists between anchor points
            anchor_start = min(fib.anchor_high_idx, fib.anchor_low_idx)
            anchor_end = max(fib.anchor_high_idx, fib.anchor_low_idx)
            has_imbalance = has_imbalance_in_range(df, anchor_start, anchor_end)

            if not has_imbalance:
                continue  # Skip if imbalance required but not found
        else:
            # If not required, still check if it exists (for metadata)
            anchor_start = min(fib.anchor_high_idx, fib.anchor_low_idx)
            anchor_end = max(fib.anchor_high_idx, fib.anchor_low_idx)
            has_imbalance = has_imbalance_in_range(df, anchor_start, anchor_end)

        # Find Institutional Candle within Fib bounds
        ic_idx = find_institutional_candle(
            df, fib, config,
            search_start_idx=confirmed_idx,
            search_end_idx=confirmed_idx + 20,  # Configurable search window
        )

        if ic_idx is None:
            continue  # No IC found

        # Create POI Zone from IC
        ic_row = df.iloc[ic_idx]
        zone_top = float(ic_row.h)
        zone_bottom = float(ic_row.l)

        # Determine side based on structure direction
        side = "buy" if sd == 1 else "sell"

        zone = POIZone(
            start_time=_time(ic_idx),
            end_time=None,
            side=side,
            top=zone_top,
            bottom=zone_bottom,
            ic_idx=ic_idx,
            fib=fib,
            has_imbalance=has_imbalance,
            meta={
                "structure_id": structure_id,
                "struct_direction": sd,
                "event_type": ev.type,
                "confirmed_idx": confirmed_idx,
            },
        )
        zones.append(zone)

    return zones
