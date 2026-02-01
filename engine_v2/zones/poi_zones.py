# engine_v2/zones/poi_zones.py
"""
POI Zone derivation using Fibonacci retracement and Institutional Candle identification.

POI Zones are created when:
1. A Fib is active for a cycle (from FibTracker)
2. IC candidates are identified (base + scenario conditions)
3. IC variants are selected based on overlap thresholds (V30/V60/V90)
4. POI zones are created from unique ICs with their high/low as bounds

IC Candidate Base Conditions (ALL required):
1. Within Fib bounds (inclusive): BOS_idx <= candle_idx <= CTS_idx
2. Opposite direction of struct_direction: candle.direction == -struct_direction
3. Unfilled imbalance (in sd direction) AFTER candidate, up to CTS idx
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal

import pandas as pd

from engine_v2.features.fibonacci import FibRetracement
from engine_v2.patterns.imbalance import has_unfilled_imbalance_in_direction
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.zones.fib_tracker import FibTracker, FibState


@dataclass(frozen=True)
class POIZone:
    """
    Point of Interest Zone derived from Fibonacci + Institutional Candle.

    One zone per unique IC (not per variant). Variants are stored in meta.
    """
    start_time: "pd.Timestamp"
    end_time: Optional["pd.Timestamp"]  # None = extends to end of chart
    side: Literal["buy", "sell"]        # "buy" if sd=+1, "sell" if sd=-1
    top: float                          # IC candle high
    bottom: float                       # IC candle low

    # Source information
    ic_idx: int                         # Index of the Institutional Candle

    # Metadata
    meta: Dict[str, Any] = field(default_factory=dict)
    # meta contains:
    #   structure_id, struct_direction, cycle_id
    #   confirmed_idx (first activation idx, always > ic_idx)
    #   versions: ["V30", "V60", "V90"]
    #   status: "active" | "inactive" | "disappeared"


# Configuration for POI Zone detection
@dataclass
class POIConfig:
    """Configuration for POI Zone detection."""
    # IC search bounds (as Fib percentages)
    ic_fib_min: float = 61.8  # IC overlap zone starts here
    ic_fib_max: float = 80.0  # IC overlap zone ends here

    # Variant overlap thresholds
    v30_threshold: float = 0.30
    v60_threshold: float = 0.60
    v90_threshold: float = 0.90

    # Imbalance detection
    fill_threshold: float = 0.70  # 70% = filled


def calculate_candle_overlap_pct(
    candle_high: float,
    candle_low: float,
    fib_zone_top: float,
    fib_zone_bottom: float,
) -> float:
    """
    Calculate what percentage of candle falls within the Fib zone.

    Parameters
    ----------
    candle_high : float
        The candle's high price
    candle_low : float
        The candle's low price
    fib_zone_top : float
        Top of the Fib zone (61.8% level)
    fib_zone_bottom : float
        Bottom of the Fib zone (80% level)

    Returns
    -------
    float
        Percentage of candle within Fib zone (0.0 to 1.0)
    """
    candle_range = candle_high - candle_low
    if candle_range <= 0:
        return 0.0

    # Ensure zone bounds are ordered correctly
    zone_top = max(fib_zone_top, fib_zone_bottom)
    zone_bottom = min(fib_zone_top, fib_zone_bottom)

    overlap_top = min(candle_high, zone_top)
    overlap_bottom = max(candle_low, zone_bottom)
    overlap = max(0.0, overlap_top - overlap_bottom)

    return overlap / candle_range


def passes_price_constraint(
    candle_idx: int,
    df: pd.DataFrame,
    struct_direction: int,
    reference_price: float,
) -> bool:
    """
    Check if candle passes the scenario price constraint.

    For sd=+1 (bullish): Entire candle (HIGH) must be BELOW reference price
    For sd=-1 (bearish): Entire candle (LOW) must be ABOVE reference price

    Parameters
    ----------
    candle_idx : int
        Index of the candle to check
    df : DataFrame
        OHLC data
    struct_direction : int
        +1 for bullish, -1 for bearish
    reference_price : float
        The price threshold to compare against

    Returns
    -------
    bool
        True if candle passes the price constraint
    """
    if candle_idx not in df.index:
        return False

    candle_high = float(df.loc[candle_idx, "h"])
    candle_low = float(df.loc[candle_idx, "l"])

    if struct_direction == 1:  # Bullish - entire candle must be BELOW reference
        return candle_high < reference_price
    else:  # Bearish - entire candle must be ABOVE reference
        return candle_low > reference_price


def find_ic_candidates(
    df: pd.DataFrame,
    fib_state: FibState,
    config: POIConfig,
    scenario_context: Optional[Dict] = None,
) -> List[int]:
    """
    Find IC candidates that meet base conditions.

    Base conditions (ALL required):
    1. Within Fib bounds (inclusive): BOS_idx <= candle_idx <= CTS_idx
    2. Opposite direction of struct_direction: candle.direction == -struct_direction
    3. Unfilled imbalance (in sd direction) AFTER candidate, up to CTS idx

    Parameters
    ----------
    df : DataFrame
        OHLC data with direction and is_imbalance columns
    fib_state : FibState
        The active Fib state for this cycle
    config : POIConfig
        Configuration for POI detection
    scenario_context : dict, optional
        Additional context for scenario-specific conditions:
        - scenario: 1, 2, or 3
        - reversal_confirmed_idx: idx where reversal was confirmed
        - prev_bos_price: previous structure's last BOS price
        - cts_established_idx: idx where CTS was established
        - prev_cts_price: CTS_N-1 price (for cycle 1+)
        - cross_cycle: bool (True if this is a cross-cycle Fib)

    Returns
    -------
    list of int
        Indices of candles that qualify as IC candidates
    """
    # A Fib is valid for IC detection if it's active OR locked
    # Locked means CTS was confirmed - bounds are finalized but still valid
    if not (fib_state.active or fib_state.locked):
        return []

    candidates = []
    sd = fib_state.struct_direction
    bos_idx = fib_state.bos_idx
    cts_idx = fib_state.cts_idx

    # Ensure we have the direction column
    if "direction" not in df.columns:
        return []

    # Search all candles within Fib bounds (inclusive)
    for idx in range(bos_idx, cts_idx + 1):
        if idx not in df.index:
            continue

        # Condition 1: Already satisfied by range

        # Condition 2: Opposite direction of struct_direction
        candle_dir = int(df.loc[idx, "direction"])
        if candle_dir != -sd:
            continue

        # Condition 3: Unfilled imbalance (in sd direction) AFTER candidate
        # Check range (candidate_idx + 1, CTS_idx] for unfilled imbalance in sd direction
        has_unfilled_after = has_unfilled_imbalance_in_direction(
            df,
            start_idx=idx + 1,  # strictly after
            end_idx=cts_idx,    # inclusive
            direction=sd,
            fill_threshold=config.fill_threshold,
        )

        if not has_unfilled_after:
            continue

        # All base conditions met
        candidates.append(idx)

    return candidates


def select_ic_variants(
    candidates: List[int],
    df: pd.DataFrame,
    fib_state: FibState,
    config: POIConfig,
) -> Dict[int, List[str]]:
    """
    From IC candidates, find the most recent candle for each variant threshold.

    For each variant (V30/V60/V90), scan candidates from most recent (highest idx)
    and pick the first that meets the threshold.

    Parameters
    ----------
    candidates : list of int
        IC candidate indices
    df : DataFrame
        OHLC data
    fib_state : FibState
        The active Fib state
    config : POIConfig
        Configuration with overlap thresholds

    Returns
    -------
    dict
        {ic_idx: ["V30", "V60", "V90"], ...} - grouped by unique IC
    """
    if not candidates or fib_state.fib is None:
        return {}

    fib = fib_state.fib

    # Get the 61.8-80% zone bounds
    fib_zone_top = fib.price_at_pct(config.ic_fib_min)
    fib_zone_bottom = fib.price_at_pct(config.ic_fib_max)

    thresholds = {
        "V30": config.v30_threshold,
        "V60": config.v60_threshold,
        "V90": config.v90_threshold,
    }
    variant_ic: Dict[str, int] = {}  # {variant: ic_idx}

    # Sort candidates by idx descending (most recent first)
    sorted_candidates = sorted(candidates, reverse=True)

    for variant, min_pct in thresholds.items():
        for idx in sorted_candidates:
            candle_high = float(df.loc[idx, "h"])
            candle_low = float(df.loc[idx, "l"])

            overlap = calculate_candle_overlap_pct(
                candle_high, candle_low,
                fib_zone_top, fib_zone_bottom
            )

            if overlap >= min_pct:
                variant_ic[variant] = idx
                break  # Found most recent for this variant

    # Group by unique IC, store versions as list
    ic_versions: Dict[int, List[str]] = defaultdict(list)
    for variant, idx in variant_ic.items():
        ic_versions[idx].append(variant)

    # Sort versions for consistency
    for idx in ic_versions:
        ic_versions[idx].sort()

    return dict(ic_versions)


def derive_poi_zones(
    df: pd.DataFrame,
    structure_events: List[StructureEvent],
    fib_tracker: Optional[FibTracker] = None,
    config: Optional[POIConfig] = None,
) -> List[POIZone]:
    """
    Derive POI Zones from Fib states and IC identification.

    This is the main entry point for POI Zone detection.

    Parameters
    ----------
    df : DataFrame
        OHLC data with required columns
    structure_events : list of StructureEvent
        Structure events for context (CTS_ESTABLISHED, etc.)
    fib_tracker : FibTracker, optional
        FibTracker with active Fib states
    config : POIConfig, optional
        Configuration for POI detection

    Returns
    -------
    List of POIZone
    """
    if config is None:
        config = POIConfig()

    zones: List[POIZone] = []

    if fib_tracker is None:
        return zones

    # Helper to convert idx to time
    def _time(i: int):
        if i in df.index:
            return pd.to_datetime(df.loc[i, "time"], utc=True)
        return None

    # Get all Fib states from tracker (for charting)
    fib_states = fib_tracker.get_fibs_for_charting()

    # Build lookup for CTS_ESTABLISHED events (for confirmed_idx and end_time)
    # Key: (sid, cycle_id) -> event
    cts_established_by_key = {}
    for ev in structure_events:
        if ev.type == "CTS_ESTABLISHED":
            sid = int(ev.meta.get("structure_id", 0))
            cycle_id = int(ev.meta.get("cycle_id", 0))
            key = (sid, cycle_id)
            cts_established_by_key[key] = ev

    # Build lookup for reversal_confirmed_idx per structure_id
    # Reversal ends ALL zones for that structure
    # Use STATE_CHANGED events where to='reversal'
    reversal_idx_by_sid: Dict[int, int] = {}
    for ev in structure_events:
        if ev.type == "STATE_CHANGED" and ev.meta.get("to") == "reversal":
            sid = int(ev.meta.get("structure_id", 0))
            idx = int(ev.idx)
            # Keep the MAX idx for each structure_id (last reversal candle)
            if sid not in reversal_idx_by_sid or idx > reversal_idx_by_sid[sid]:
                reversal_idx_by_sid[sid] = idx

    # Process each Fib state
    for fib_state in fib_states:
        if not fib_state.active and not fib_state.locked:
            # Skip deactivated Fibs that were never locked
            # (they were invalidated and shouldn't produce zones)
            continue

        sid = fib_state.structure_id
        cycle_id = fib_state.cycle_id
        sd = fib_state.struct_direction
        key = (sid, cycle_id)

        # Find IC candidates
        candidates = find_ic_candidates(df, fib_state, config)

        if not candidates:
            continue

        # Select IC variants
        ic_variants = select_ic_variants(candidates, df, fib_state, config)

        if not ic_variants:
            continue

        # Get confirmed_idx from CTS_ESTABLISHED event
        cts_event = cts_established_by_key.get(key)
        confirmed_idx = int(cts_event.idx) if cts_event else fib_state.cts_idx

        # Determine zone end_time (priority order)
        # 1. Reversal: all zones for this structure end at reversal_confirmed_idx
        # 2. New CTS: cycle N zones end when CTS_N+1 is established
        # 3. No event: zone extends to chart end (end_time = None)
        end_time = None
        end_idx = None

        # Check for reversal first (highest priority)
        if sid in reversal_idx_by_sid:
            end_idx = reversal_idx_by_sid[sid]
            end_time = _time(end_idx)

        # Check for next cycle CTS (only if no reversal, or if next CTS comes before reversal)
        next_cycle_key = (sid, cycle_id + 1)
        if next_cycle_key in cts_established_by_key:
            next_cts_idx = int(cts_established_by_key[next_cycle_key].idx)
            # Use next CTS if no reversal, or if next CTS comes before reversal
            if end_idx is None or next_cts_idx < end_idx:
                end_idx = next_cts_idx
                end_time = _time(end_idx)

        # Determine zone status
        # Active only if: no end_idx (zone hasn't ended) AND is current cycle AND fib is active
        # If zone has an end_idx, it has ended and should be "inactive"
        if end_idx is not None:
            status = "inactive"
        else:
            current_cycle = fib_tracker._current_cycle.get(sid)
            is_current = (current_cycle == cycle_id)
            status = "active" if is_current and fib_state.active else "inactive"

        # Create a zone for each unique IC
        for ic_idx, versions in ic_variants.items():
            ic_time = _time(ic_idx)
            if ic_time is None:
                continue

            ic_high = float(df.loc[ic_idx, "h"])
            ic_low = float(df.loc[ic_idx, "l"])

            zone = POIZone(
                start_time=ic_time,
                end_time=end_time,
                side="buy" if sd == 1 else "sell",
                top=ic_high,
                bottom=ic_low,
                ic_idx=ic_idx,
                meta={
                    "structure_id": sid,
                    "struct_direction": sd,
                    "cycle_id": cycle_id,
                    "confirmed_idx": confirmed_idx,
                    "end_idx": end_idx,  # None if extends to chart end
                    "versions": versions,
                    "status": status,
                    "bos_idx": fib_state.bos_idx,
                    "cts_idx": fib_state.cts_idx,
                },
            )
            zones.append(zone)

    print(f"[poi_zones] total candidates checked across all fibs, zones created={len(zones)}")
    for z in zones:
        print(f"[poi_zones] sid={z.meta.get('structure_id')} cycle={z.meta.get('cycle_id')} "
              f"ic_idx={z.ic_idx} end_idx={z.meta.get('end_idx')} status={z.meta.get('status')}")

    return zones
