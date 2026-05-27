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

import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal, Tuple

import numpy as np
import pandas as pd

from engine_v2.common.types import ImbalanceInstance
from engine_v2.features.fibonacci import FibRetracement
from engine_v2.patterns.imbalance import has_unfilled_imbalance
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.zones.fib_tracker import FibTracker, FibState, select_fib_anchor_for_cycle


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
        # Check range (candidate_idx + 1, CTS_idx] for unfilled imbalance in sd direction.
        # IC validation is strict: the same-direction filter (direction=sd) is the
        # whole point — an IC's role is to anchor a same-direction continuation, so
        # a counter-direction imbalance in the tail wouldn't justify it. (Fib
        # activation, by contrast, calls this same primitive without `direction` to
        # stay permissive — see fib_tracker.py call sites.)
        has_unfilled_after = has_unfilled_imbalance(
            df,
            start_idx=idx + 1,        # strictly after
            end_idx=cts_idx,          # inclusive
            check_to_idx=cts_idx,     # evaluate fill as of fib's current CTS upper bound
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
    lifecycle_floor: Optional[int] = None,
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

    # Structure lifecycle-start per sid (Phase 3 Commit 2, 2026-05-26): the idx
    # a structure first becomes active — sid 0 = first structural anchor (min
    # event idx); sid N>=1 = reversal-confirmation idx of sid N-1; subordinate
    # = trigger idx supplied via `lifecycle_floor`. POI activation is floored
    # here so a post-reversal cycle-0 POI (whose CTS_ESTABLISHED can precede the
    # reversal) cannot activate before its structure is alive. Mirrors the KL
    # clamp; see PART4_REFACTOR_SPEC §5 + ARCHITECTURE "Activation floor".
    struct_start_by_sid: Dict[int, int] = {}
    for ev in structure_events:
        s = ev.meta.get("structure_id")
        if s is None:
            continue
        s = int(s)
        i = int(ev.idx)
        if s not in struct_start_by_sid or i < struct_start_by_sid[s]:
            struct_start_by_sid[s] = i
    for s in list(struct_start_by_sid):
        if (s - 1) in reversal_idx_by_sid:
            struct_start_by_sid[s] = int(reversal_idx_by_sid[s - 1])
    if lifecycle_floor is not None:
        for s in struct_start_by_sid:
            struct_start_by_sid[s] = max(struct_start_by_sid[s], int(lifecycle_floor))

    # Pre-group CTS_ESTABLISHED + CTS_UPDATED events by (sid, cycle_id) so the
    # per-POI activation scan doesn't re-iterate the full event list for every
    # IC. Sort once in ascending idx order.
    cts_events_by_key: Dict[tuple, List[StructureEvent]] = defaultdict(list)
    for ev in structure_events:
        if ev.type not in ("CTS_ESTABLISHED", "CTS_UPDATED"):
            continue
        sid_ev = ev.meta.get("structure_id")
        cycle_ev = ev.meta.get("cycle_id")
        if sid_ev is None or cycle_ev is None:
            continue
        cts_events_by_key[(int(sid_ev), int(cycle_ev))].append(ev)
    for key in cts_events_by_key:
        cts_events_by_key[key].sort(key=lambda e: int(e.idx))

    # Precompute each imbalance instance's fill_idx (the first candle after
    # inst.end_idx where ≥70% retrace into the merged gap fires). One linear
    # scan per instance — shared across every POI's activation sweep.
    imbalances: List[ImbalanceInstance] = df.attrs.get("imbalances", [])
    fill_idx_cache = _compute_fill_idx_cache(df, imbalances, config.fill_threshold)

    # Variant thresholds are constant across POIs.
    variant_thresholds = {
        "V30": config.v30_threshold,
        "V60": config.v60_threshold,
        "V90": config.v90_threshold,
    }

    # Process each Fib state
    for fib_state in fib_states:
        # Lifecycle gate (FIB_LIFECYCLE_SPEC §11, Session 2). Process a fib iff
        # it is the live condition-active version (active AND its cycle has not
        # ended) OR it is locked (a confirmed historical record — still feeds IC
        # detection, including ended-locked fibs). `disappeared` records never
        # reach here (filtered by get_fibs_for_charting).
        #
        # This is the byte-identical-PRESERVING per-record form, NOT the
        # cycle-level `status == "active" OR locked`: `status` is shared by all
        # version records of a live cycle, so a status-keyed gate would wrongly
        # process superseded (dead) cross versions of a still-live cycle (same
        # reason the chart gate, §9.1, uses raw per-record flags). `active` is
        # condition-only post-Session-2, so we add `end_idx is None` (cycle not
        # ended) to reproduce today's behavior (today's `active` was also False
        # once a cycle ended).
        if not ((fib_state.active and fib_state.end_idx is None) or fib_state.locked):
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

        # CTS_ESTABLISHED idx for the cycle (needed for activation condition 1).
        cts_event = cts_established_by_key.get(key)
        cts_established_idx = (
            int(cts_event.idx) if cts_event else int(fib_state.cts_idx)
        )

        # Determine end_idx + end_reason (terminal state — irreversible).
        # Priority:
        #   1. Reversal (highest)
        #   2. Next cycle CTS_ESTABLISHED
        #   3. None → zone extends to chart end
        end_idx: Optional[int] = None
        end_reason: Optional[str] = None
        end_time = None

        if sid in reversal_idx_by_sid:
            end_idx = reversal_idx_by_sid[sid]
            end_reason = "reversal"
            end_time = _time(end_idx)

        next_cycle_key = (sid, cycle_id + 1)
        if next_cycle_key in cts_established_by_key:
            next_cts_idx = int(cts_established_by_key[next_cycle_key].idx)
            if end_idx is None or next_cts_idx < end_idx:
                end_idx = next_cts_idx
                end_reason = "next_cycle"
                end_time = _time(end_idx)

        # Last candle for the activation scan window.
        last_candle = int(df.index.max())
        scan_end = min(end_idx - 1, last_candle) if end_idx is not None else last_candle

        # Create a zone for each unique IC
        for ic_idx, versions in ic_variants.items():
            ic_time = _time(ic_idx)
            if ic_time is None:
                continue

            ic_high = float(df.loc[ic_idx, "h"])
            ic_low = float(df.loc[ic_idx, "l"])

            # Per-POI activation lifecycle. POI is "active" at candle t iff
            # ALL these condition-state requirements hold simultaneously (the
            # IC's identification conditions, re-evaluated dynamically against
            # the fib's time-varying state):
            #   1. fib's `cts_at(t)` has extended to >= ic_idx — i.e., IC lies
            #      within the fib's bounds at t. The fib's cts only grows, so
            #      becomes monotonic once reached.
            #   2. ic_idx <= t  (the IC candle exists in data — implied by t
            #      starting at first_active = max(cts_established_idx, ic_idx)).
            #   3. `has_unfilled_imbalance(df, ic_idx+1, t, check_to_idx=t,
            #      direction=sd)` — POI-specific sd-direction imbalance check
            #      decoupled from Fib's `active` flag.
            #   5. Variant qualification: at the fib's bounds [bos_price,
            #      cts_price_at(t)], the IC candle's overlap with the 61.8-80%
            #      Fib zone meets at least the V30 (30%) threshold. Variants
            #      can downgrade (V90 → V60 → V30) or vanish entirely as the
            #      fib extends.
            # (Condition 4 — scenario idx/price constraints — is gated at IC
            # identification and not re-checked per candle: once the
            # constraining event has fired, the comparison is fixed.)
            # End-state (`end_idx`) takes precedence: once ended, `status` is
            # "ended" regardless of activation conditions.
            cts_events_for_fib = cts_events_by_key.get((int(sid), int(cycle_id)), [])
            activation_history = _compute_poi_activation_history(
                df=df,
                ic_idx=int(ic_idx),
                cts_established_idx=cts_established_idx,
                sd=int(sd),
                scan_end=scan_end,
                fill_threshold=config.fill_threshold,
                bos_price=float(fib_state.bos_price),
                cts_events=cts_events_for_fib,
                fib_min_pct=config.ic_fib_min,
                fib_max_pct=config.ic_fib_max,
                variant_thresholds=variant_thresholds,
                imbalances=imbalances,
                fill_idx_cache=fill_idx_cache,
                lifecycle_floor_idx=struct_start_by_sid.get(int(sid)),
            )

            # confirmed_idx = idx of the most recent ACTIVATE event (if any).
            # current_versions = variants recorded at that activate event (or
            # empty if POI is currently deactivated). With per-candle variant
            # re-check, the active variant tier can downgrade (V90 -> V60 ->
            # V30) or vanish as the fib's `cts_price` shifts.
            confirmed_idx: Optional[int] = None
            current_versions: List[str] = []
            currently_active = False
            for ev in activation_history:
                if ev["active"]:
                    confirmed_idx = ev["idx"]
                    current_versions = list(ev.get("versions", []))
                    currently_active = True
                else:
                    current_versions = []
                    currently_active = False

            # Derived 3-state status at end of data (or end_idx if reached).
            if end_idx is not None and end_idx <= last_candle:
                status = "ended"
            elif currently_active:
                status = "active"
            else:
                status = "inactive"

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
                    "end_idx": end_idx,
                    "end_reason": end_reason,
                    # `versions` = peak variants ever achieved (snapshot from
                    # `select_ic_variants` using final fib state). For the
                    # live state at end of scan, see `current_versions`.
                    "versions": versions,
                    "current_versions": current_versions,
                    "status": status,
                    "activation_history": activation_history,
                    "bos_idx": fib_state.bos_idx,
                    "cts_idx": fib_state.cts_idx,
                    "cts_established_idx": cts_established_idx,
                },
            )
            zones.append(zone)

    # Per-POI lifecycle dump, opt-in via env var (silent by default).
    # Re-enable when reconstructing the per-POI table — see
    # `memory/project_item_3_poi_lifecycle.md` for the table this dump fed.
    # Usage: `POI_LIFECYCLE_DEBUG=1 python -m engine_v2.run_replay`.
    if os.environ.get("POI_LIFECYCLE_DEBUG"):
        print(f"[poi_zones] total candidates checked across all fibs, zones created={len(zones)}")
        for z in zones:
            ah = z.meta.get("activation_history", [])
            flips_summary = ",".join(f"{e['idx']}:{'A' if e['active'] else 'D'}" for e in ah) or "-"
            # Per-POI most-recent deact + first-react-after-deact (filtered to
            # POI lifecycle, i.e., events in activation_history).
            deact_idx = None
            react_after = None
            for e in ah:
                if not e["active"]:
                    deact_idx = e["idx"]
                    react_after = None
                elif deact_idx is not None and react_after is None:
                    react_after = e["idx"]
            peak = z.meta.get("versions", [])
            cur = z.meta.get("current_versions", [])
            print(f"[poi_zones] sid={z.meta.get('structure_id')} cycle={z.meta.get('cycle_id')} "
                  f"bos_idx={z.meta.get('bos_idx')} cts_est={z.meta.get('cts_established_idx')} "
                  f"confirmed={z.meta.get('confirmed_idx')} ic={z.ic_idx} cts={z.meta.get('cts_idx')} "
                  f"deact={deact_idx} react_after={react_after} end={z.meta.get('end_idx')} "
                  f"end_reason={z.meta.get('end_reason')} status={z.meta.get('status')} "
                  f"dir={z.meta.get('struct_direction')} peak={peak} cur={cur} hist=[{flips_summary}]")

            # Per-relevant-imbalance armed/confirmed dump. Same filter shape
            # as the activation sweep so the printed set matches what drove
            # the history. armed_idx + confirmed_fill_idx come straight from
            # the cache (which stashed them on inst.meta during the sweep).
            z_sd = z.meta.get("struct_direction")
            z_ic_idx = z.ic_idx
            z_end_idx = z.meta.get("end_idx")
            z_scan_end = (z_end_idx - 1) if z_end_idx is not None else None
            z_relevant = [
                inst for inst in imbalances
                if inst.direction == z_sd
                and inst.gap_size > 0
                and inst.end_idx > z_ic_idx
                and (z_scan_end is None or inst.start_idx <= z_scan_end)
            ]
            for k, inst in enumerate(z_relevant):
                print(f"[poi_zones]   imb#{k} start={inst.start_idx} end={inst.end_idx} "
                      f"dir={inst.direction:+d} gap_top={inst.gap_top:.5f} "
                      f"gap_bottom={inst.gap_bottom:.5f} "
                      f"armed={inst.meta.get('armed_idx')} "
                      f"confirmed={inst.meta.get('confirmed_fill_idx')}")

    return zones


def _compute_fill_idx_cache(
    df: pd.DataFrame,
    imbalances: List[ImbalanceInstance],
    fill_threshold: float,
) -> Dict[int, Tuple[Optional[int], Optional[int]]]:
    """Precompute each imbalance instance's `(armed_idx, confirmed_fill_idx)`
    pair for the two-stroke fill state machine.

    ``armed_idx`` — first candle in ``(end_idx, end-of-df]`` reaching the
    stroke-1 retrace level (``>= fill_threshold`` into the gap).
    ``confirmed_fill_idx`` — first candle at idx ``>= armed_idx`` where the
    close passes the gap outer in the instance's direction (stroke 2).
    Either or both can be ``None`` (never armed / armed but never confirmed).
    Both can be equal (same-candle stroke 1+2 — rare but legal).

    Mirrors :meth:`ImbalanceInstance.is_filled` exactly: a caller that asked
    ``is_filled(check_to_idx=t)`` would get the same boolean as
    ``confirmed_fill_idx is not None and confirmed_fill_idx <= t``. Used by
    the POI activation sweep so per-POI scans don't repeat the
    ``(end_idx, t]`` walk for every candidate t.

    Side effect: stashes both indices on ``inst.meta`` (``armed_idx`` and
    ``confirmed_fill_idx`` keys) so debug exporters can read them without
    rebuilding the cache. The instance dataclass is frozen but its ``meta``
    dict is mutable.

    Key: ``id(inst)`` → ``(armed_idx, confirmed_fill_idx)``. Both are
    entity-absolute idx (or ``None``). Stable across the call's lifetime;
    not persisted.
    """
    cache: Dict[int, Tuple[Optional[int], Optional[int]]] = {}
    if not imbalances or len(df) == 0:
        return cache

    # Numpy positional indexing is ~35x faster than pandas .loc per element
    # (microbench: 0.02ms vs 0.70ms for a 3000-candle scan with 60 starts).
    # For a contiguous RangeIndex 0..N (which df has after reset_index in the
    # pipeline) idx_to_pos is the identity map, and pos == idx. We compute it
    # explicitly so this stays correct for any df.index — sliced sub-entity
    # frames that have non-contiguous indices route through the dict lookup.
    n = len(df)
    required_cols = ("h", "l", "c")
    if any(col not in df.columns for col in required_cols):
        empty = (None, None)
        for inst in imbalances:
            inst.meta["armed_idx"] = None
            inst.meta["confirmed_fill_idx"] = None
        return {id(inst): empty for inst in imbalances}
    h_arr = df["h"].to_numpy(dtype=float, copy=False)
    l_arr = df["l"].to_numpy(dtype=float, copy=False)
    c_arr = df["c"].to_numpy(dtype=float, copy=False)
    df_idx_arr = df.index.to_numpy()
    if df_idx_arr.dtype.kind in "iu" and n > 0 and df_idx_arr[0] == 0 and df_idx_arr[-1] == n - 1:
        # Contiguous RangeIndex: pos == idx.
        idx_to_pos = None
    else:
        idx_to_pos = {int(v): i for i, v in enumerate(df_idx_arr)}

    def _pos(idx: int) -> int:
        if idx_to_pos is None:
            return idx if 0 <= idx < n else -1
        return idx_to_pos.get(idx, -1)

    def _idx_from_pos(pos: int) -> int:
        return int(df_idx_arr[pos]) if idx_to_pos is not None else pos

    for inst in imbalances:
        # Degenerate gap → treat as already filled (is_filled returns True
        # unconditionally). Represent as (None, None) and exclude from the
        # relevant set in the sweep.
        if inst.gap_size <= 0:
            cache[id(inst)] = (None, None)
            inst.meta["armed_idx"] = None
            inst.meta["confirmed_fill_idx"] = None
            continue

        start_pos = _pos(inst.end_idx + 1)
        if start_pos < 0:
            # End of instance is at or past the last candle — no candles
            # available to retrace into.
            cache[id(inst)] = (None, None)
            inst.meta["armed_idx"] = None
            inst.meta["confirmed_fill_idx"] = None
            continue

        armed_idx: Optional[int] = None
        confirmed_fill_idx: Optional[int] = None
        if inst.direction == 1:
            stroke1_level = inst.gap_top - inst.gap_size * fill_threshold
            stroke2_level = inst.gap_top
            slice_l = l_arr[start_pos:]
            stroke1_hits = np.where(slice_l <= stroke1_level)[0]
            if stroke1_hits.size:
                armed_pos = start_pos + int(stroke1_hits[0])
                armed_idx = _idx_from_pos(armed_pos)
                # Stroke 2 scan starts AT armed_pos (same-candle fill is legal).
                slice_c = c_arr[armed_pos:]
                stroke2_hits = np.where(slice_c >= stroke2_level)[0]
                if stroke2_hits.size:
                    confirmed_pos = armed_pos + int(stroke2_hits[0])
                    confirmed_fill_idx = _idx_from_pos(confirmed_pos)
        elif inst.direction == -1:
            stroke1_level = inst.gap_bottom + inst.gap_size * fill_threshold
            stroke2_level = inst.gap_bottom
            slice_h = h_arr[start_pos:]
            stroke1_hits = np.where(slice_h >= stroke1_level)[0]
            if stroke1_hits.size:
                armed_pos = start_pos + int(stroke1_hits[0])
                armed_idx = _idx_from_pos(armed_pos)
                slice_c = c_arr[armed_pos:]
                stroke2_hits = np.where(slice_c <= stroke2_level)[0]
                if stroke2_hits.size:
                    confirmed_pos = armed_pos + int(stroke2_hits[0])
                    confirmed_fill_idx = _idx_from_pos(confirmed_pos)

        cache[id(inst)] = (armed_idx, confirmed_fill_idx)
        inst.meta["armed_idx"] = armed_idx
        inst.meta["confirmed_fill_idx"] = confirmed_fill_idx

    return cache


def _compute_poi_activation_history(
    df: pd.DataFrame,
    *,
    ic_idx: int,
    cts_established_idx: int,
    sd: int,
    scan_end: int,
    fill_threshold: float,
    bos_price: float,
    cts_events: List[StructureEvent],
    fib_min_pct: float,
    fib_max_pct: float,
    variant_thresholds: Dict[str, float],
    imbalances: List[ImbalanceInstance],
    fill_idx_cache: Dict[int, Tuple[Optional[int], Optional[int]]],
    lifecycle_floor_idx: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Produce the POI's activation history via an event-driven sweep.

    POI is "active" at candle `t` iff all condition-state requirements hold
    simultaneously (decoupled from FibState.active):

      Condition 1 — `cts_at(t) >= ic_idx`. The fib's `cts_idx` only grows
                    via CTS_UPDATED events, so once met, monotonic.
      Condition 2 — `ic_idx <= t` (implicit: scan starts at first_active
                    = max(cts_established_idx, ic_idx)).
      Condition 3 — sd-direction imbalance overlaps `(ic_idx, t]` that is
                    not yet committed-filled (two-stroke: stroke 1 = 70%
                    retrace, stroke 2 = close past gap outer). Flips as
                    imbalances form / commit-fill.
      Condition 5 — Variant qualification: the IC candle overlaps the
                    61.8-80% Fib zone (computed from `bos_price` and the
                    time-varying `cts_price_at(t)`) by at least V30.

    Why event-driven: state can change only at three kinds of idx —
    imbalance "enter unfilled set" (`inst.start_idx`), imbalance "leave
    unfilled set" (cached `confirmed_fill_idx` — the stroke-2 candle),
    and CTS event idx (cts_price / cts_idx update → cond5 may flip).
    Between these idx state is constant, so the per-candle loop over
    `[first_active, scan_end]` is wasted work. Transitions are enumerated,
    sorted, and swept once; the unfilled-count is maintained by +1/−1 at
    enter/leave events, and current_versions is recomputed only at CTS
    transitions.

    `fill_idx_cache` entries are `(armed_idx, confirmed_fill_idx)` tuples;
    the sweep uses ``confirmed_fill_idx`` exclusively for the leave event.
    ``armed_idx`` is informational (stashed on `inst.meta` by the cache
    builder for debug exporters).

    Returns `[{"idx", "active", "reason", "versions"?}, ...]`. Empty list
    means POI never activated in the window.
    """
    # Floor at the structure lifecycle-start (Phase 3 Commit 2): a POI cannot
    # activate before its structure is alive (post-reversal cycle-0 / sub case).
    first_active = max(cts_established_idx, ic_idx)
    if lifecycle_floor_idx is not None:
        first_active = max(first_active, int(lifecycle_floor_idx))
    if first_active > scan_end or ic_idx not in df.index:
        return []

    ic_high = float(df.loc[ic_idx, "h"])
    ic_low = float(df.loc[ic_idx, "l"])

    sorted_cts_events = sorted(cts_events, key=lambda e: int(e.idx))

    # State carried through the sweep.
    cts_idx_at_t = -1
    cts_price_at_t = 0.0
    unfilled_count = 0
    current_versions: List[str] = []

    def _compute_versions() -> List[str]:
        # Retracement math is direction-aware (mirrors
        # `features.fibonacci.calculate_fib_price`).
        if cts_idx_at_t < ic_idx or cts_price_at_t <= 0:
            return []
        if sd == 1:
            anchor_high = cts_price_at_t
            anchor_low = bos_price
        else:
            anchor_high = bos_price
            anchor_low = cts_price_at_t
        range_size = anchor_high - anchor_low
        if range_size <= 0:
            return []
        if sd == 1:
            fib_level_min = anchor_high - range_size * (fib_min_pct / 100.0)
            fib_level_max = anchor_high - range_size * (fib_max_pct / 100.0)
        else:
            fib_level_min = anchor_low + range_size * (fib_min_pct / 100.0)
            fib_level_max = anchor_low + range_size * (fib_max_pct / 100.0)
        overlap = calculate_candle_overlap_pct(
            ic_high, ic_low, fib_level_min, fib_level_max,
        )
        return sorted(v for v, th in variant_thresholds.items() if overlap >= th)

    # --- Pre-window state: apply CTS events strictly before first_active so
    # current_versions reflects the entering state at first_active. (In
    # practice ic_idx <= cts_extreme < cts_established_idx, so first_active
    # equals cts_established_idx and this loop is a no-op — kept for
    # correctness if that invariant ever loosens.)
    in_window_cts: List[StructureEvent] = []
    for ev in sorted_cts_events:
        ev_idx = int(ev.idx)
        if ev_idx < first_active:
            cts_idx_at_t = ev_idx
            try:
                cts_price_at_t = float(ev.price)
            except (TypeError, ValueError):
                pass
        elif ev_idx <= scan_end:
            in_window_cts.append(ev)
    current_versions = _compute_versions()

    # --- Transitions inside the window.
    # Each is (idx, priority, kind, payload). Priority disambiguates within
    # same idx so CTS updates fire before evaluation (matters when no enter
    # / leave shares the idx).
    PRIO_IMB = 0
    PRIO_CTS = 1
    transitions: List[tuple] = []

    relevant_imbalances = [
        inst for inst in imbalances
        if inst.direction == sd
        and inst.gap_size > 0
        and inst.end_idx > ic_idx          # overlaps (ic_idx, t] only when end_idx > ic_idx
        and inst.start_idx <= scan_end     # could enter the window at all
    ]
    for inst in relevant_imbalances:
        enter_idx = max(inst.start_idx, first_active)
        if enter_idx > scan_end:
            continue
        _armed_cached, confirmed_cached = fill_idx_cache.get(id(inst), (None, None))
        # Instance contributes "unfilled" on [inst.start_idx, confirmed_fill_idx - 1]
        # (or forever if stroke 2 never confirms). Clip to scan window.
        # ``armed_idx`` alone doesn't drive transitions — the imbalance only
        # leaves the unfilled set when stroke 2 latches (cf. is_filled).
        if confirmed_cached is not None and confirmed_cached <= enter_idx:
            continue  # already committed-filled by the time it would enter
        leave_idx = confirmed_cached if (confirmed_cached is not None and confirmed_cached <= scan_end) else None
        transitions.append((enter_idx, PRIO_IMB, "enter", None))
        if leave_idx is not None:
            transitions.append((leave_idx, PRIO_IMB, "leave", None))

    for ev in in_window_cts:
        transitions.append((int(ev.idx), PRIO_CTS, "cts", ev))

    transitions.sort(key=lambda x: (x[0], x[1]))

    history: List[Dict[str, Any]] = []
    was_active = False
    i = 0
    while i < len(transitions):
        cur_idx = transitions[i][0]
        # Apply all transitions at this idx atomically before evaluating.
        cts_changed = False
        while i < len(transitions) and transitions[i][0] == cur_idx:
            _, _, kind, payload = transitions[i]
            if kind == "enter":
                unfilled_count += 1
            elif kind == "leave":
                unfilled_count -= 1
            elif kind == "cts":
                cts_idx_at_t = int(payload.idx)
                try:
                    cts_price_at_t = float(payload.price)
                except (TypeError, ValueError):
                    pass
                cts_changed = True
            i += 1
        if cts_changed:
            current_versions = _compute_versions()

        # Evaluate is_active at cur_idx. Skip rows not present in df.index
        # (defensive — entity dfs are contiguous, but mirror writes can
        # leave NA placeholder rows we don't want to evaluate).
        if cur_idx not in df.index:
            continue

        cond1 = cts_idx_at_t >= ic_idx
        cond3 = unfilled_count > 0
        cond5 = len(current_versions) > 0
        is_active = cond1 and cond3 and cond5

        if is_active and not was_active:
            history.append({
                "idx": cur_idx,
                "active": True,
                "reason": "initial" if not history else "conditions_re-met",
                "versions": list(current_versions),
            })
            was_active = True
        elif not is_active and was_active:
            reasons = []
            if not cond1:
                reasons.append("ic_out_of_fib_bounds")
            if not cond3:
                reasons.append("imbalance_filled")
            if not cond5:
                reasons.append("variant_below_v30")
            history.append({
                "idx": cur_idx,
                "active": False,
                "reason": "+".join(reasons) if reasons else "unknown",
            })
            was_active = False

    return history


# ---------------------------------------------------------------------------
# Single-cycle POI primitives — used by MarketStructure's dual CTS proximity
# check (Stage 2). Wired from `structure/structure_engine.py` as the
# `poi_inners_resolver` callable passed to MarketStructure.__init__ (Part 4
# §13.5.b). Defined here so structure/ never imports zones/.
# ---------------------------------------------------------------------------

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
    c0_data: Optional[Dict[str, Any]] = None,
) -> List[float]:
    """Derive POI zone inner prices for the cycle's current Fib state.

    Picks anchors via the shared
    :func:`zones.fib_tracker.select_fib_anchor_for_cycle` decision utility
    so this in-flight resolver agrees with FibTracker's downstream
    Scenario 2 cross-cycle decision. Constructs the FibState from the
    chosen anchors, runs IC candidate scan + variant selection, returns
    inner prices (top of IC for buy zones, bottom for sell). POIs are
    always sd-direction by Fib construction. Returns [] gracefully on any
    error so the proximity check falls back to BOS-only without crashing.

    Parameters
    ----------
    c0_data : optional dict
        Cycle-0 snapshot for the structure_id. When provided and the
        utility selects Scenario 2 (cross-cycle), the Fib anchors flip
        to ``(BOS_0, CTS_1)`` instead of ``(BOS_1, CTS_1)``. When None,
        the utility falls back to intra-cycle anchors regardless of
        cycle_id, matching the pre-Scenario-2-aware behaviour.
    """
    from engine_v2.features.fibonacci import (
        DEFAULT_FIB_LEVELS,
        create_fib_retracement,
    )

    if cts_idx <= bos_idx:
        return []

    try:
        sd = int(struct_direction)

        anchor_bos_idx, anchor_bos_price, anchor_cts_idx, anchor_cts_price, _label = (
            select_fib_anchor_for_cycle(
                df,
                int(structure_id),
                int(cycle_id),
                int(bos_idx),
                float(bos_price),
                int(cts_idx),
                float(cts_price),
                c0_data,
                float(fill_threshold),
                struct_direction=sd,
            )
        )
        if anchor_cts_idx <= anchor_bos_idx:
            return []

        if sd == 1:
            anchor_high, anchor_high_idx = anchor_cts_price, anchor_cts_idx
            anchor_low, anchor_low_idx = anchor_bos_price, anchor_bos_idx
        else:
            anchor_high, anchor_high_idx = anchor_bos_price, anchor_bos_idx
            anchor_low, anchor_low_idx = anchor_cts_price, anchor_cts_idx

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
            bos_idx=int(anchor_bos_idx),
            bos_price=float(anchor_bos_price),
            cts_idx=int(anchor_cts_idx),
            cts_price=float(anchor_cts_price),
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
        return []
