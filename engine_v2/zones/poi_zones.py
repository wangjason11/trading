# engine_v2/zones/poi_zones.py
"""
POI Zone derivation using Fibonacci retracement and Institutional Candle identification.

ONE POI per fib record (PLAN_H, Week 9 Part 1 item 1):
1. A fib record passes the lifecycle gate and its cycle has a CTS_ESTABLISHED
   (`select_ip_records`)
2. The IC candidates are identified (the base conditions below, with the AGREED
   direction)
3. The ONE IC = the institutional-pattern pick among the candidates touching the
   61.8-80 zone (`select_ip_ic`); no pattern -> no POI
4. The POI = the IC's high / low; it is active while the activation sweep's
   conditions hold (`_compute_poi_activation_history`)
MarketStructure's in-flight read (`poi_inner_for_fib`, on its embedded
FibTracker's representative fib — PLAN_I S5) calls the same `select_ip_ic`
(PLAN_H D2). The V30 / V60 / V90 selection (`select_ic_variants`)
is kept unwired (PLAN_H X1).

IC Candidate Base Conditions (ALL required):
1. Within Fib bounds (inclusive): BOS_idx <= candle_idx <= CTS_idx
2. Opposite direction of struct_direction: direction == -struct_direction (the
   IP path passes the AGREED direction, `ic_direction=`; the candle direction is
   the default)
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
from engine_v2.features.candle_classifier import candle_params_for_timeframe
from engine_v2.features.candles_v2 import agreed_direction
from engine_v2.features.fibonacci import FibRetracement, calculate_fib_price
from engine_v2.patterns.imbalance import has_unfilled_imbalance, iter_unfilled_imbalances
from engine_v2.patterns.institutional_pattern import IC_TYPE_PREFERENCE, IPResult, classify_ics
from engine_v2.structure import event_fields as ef
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.zones.cross_cycle_fib import ICReference
from engine_v2.zones.fib_tracker import FibTracker, FibState
from engine_v2.zones.structure_lifecycle import (
    compute_cycle_lifecycle,
    compute_reversal_idx_by_sid,
    compute_struct_start_by_sid,
)


@dataclass(frozen=True)
class POIZone:
    """
    Point of Interest Zone derived from Fibonacci + Institutional Candle.

    At most one zone per fib record: the institutional-pattern pick (its rank,
    legs and tiebreak in meta).
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
    # meta: the field list (confirmed_idx = the LAST activate idx, end_idx /
    # end_reason, activation_history, status, the institutional pattern —
    # ip_rank, ip_leg_candle_counts, pattern_anchor_idx, opp_leg_first_idx,
    # pattern_end_idx, pattern_confirmed_idx, the ip_tb_* tiebreak —,
    # cts_established_idx = the cycle's CTS-established moment (always: a cycle
    # without CTS_ESTABLISHED builds no POI, 2026-09-29), ...) is
    # canonical in POI_ZONES_SPEC.md "Zone Data Fields".


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

    # Institutional-pattern leg cap per timeframe (PLAN_H §1 item 6: the initial leg = its run's most recent `cap`
    # candles, the continuation = its first `cap`; H1 decision; M5 2 arrives with the M5 lens). An unknown TF
    # raises (KeyError) — no default cap.
    ip_leg_cap_by_tf: Dict[str, int] = field(default_factory=lambda: {"H1": 3, "M15": 3})


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
    Check if candle passes the price side of the IC reference condition (POI_ZONES_SPEC §3.2; PLAN_K —
    `ic_reference_reason`; written for the retired Week-7 scenario constraint, Appendix A).

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


# The IC reference condition (PLAN_K; PLAN_H §8.9 RX1) — landing A computes and EXPORTS it (`IPCandidate.
# ic_ref_reason`, the CSV column) with the filter OFF: the pick ignores it. Landing B turns it on (and removes this
# switch); the variant replay V between them = A with this True (PLAN_K §4).
_IC_REFERENCE_FILTER = False

IC_REF_PAST_REFERENCE = "ic_past_reference"
IC_REF_NOT_BEFORE_BREAK = "ic_not_before_break"


def ic_reference_reason(df: pd.DataFrame, candle_idx: int, ref: ICReference, struct_direction: int) -> Optional[str]:
    """The IC reference condition on one §3.1 candidate (PLAN_K §3.2; PLAN_H RX1 — ONE bundled condition, RX1-B):
    None = passes; `ic_past_reference` = the price side fails (sd +1: the candle's high is not strictly below the
    reference; sd −1: its low not strictly above — `passes_price_constraint`), reported FIRST when both fail;
    `ic_not_before_break` = the candle is not strictly before the break pattern's first candle. No reference
    (`ICReference.NONE`) → passes (RX1-4); a reference with no known break → the price side only (RX1-2a). `ref` is
    already AS OF its reader's moment (`FibTracker.ic_reference`)."""
    if ref.price is None:
        return None
    if not passes_price_constraint(int(candle_idx), df, int(struct_direction), float(ref.price)):
        return IC_REF_PAST_REFERENCE
    if ref.break_pattern_anchor_idx is not None and int(candle_idx) >= int(ref.break_pattern_anchor_idx):
        return IC_REF_NOT_BEFORE_BREAK
    return None


def find_ic_candidates(
    df: pd.DataFrame,
    fib_state: FibState,
    config: POIConfig,
    ic_direction: Optional[pd.Series] = None,
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
    ic_direction : Series, optional
        The direction condition 2 compares with `-struct_direction`, indexed
        like `df`. None = the candle `direction` (the V30 / V60 / V90 path);
        the institutional pattern passes the AGREED direction (PLAN_H VD — a
        volume-only doji can be an IC, a split candle cannot).

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
    dir_col = df["direction"] if ic_direction is None else ic_direction

    # Search all candles within Fib bounds (inclusive)
    for idx in range(bos_idx, cts_idx + 1):
        if idx not in df.index:
            continue

        # Condition 1: Already satisfied by range

        # Condition 2: Opposite direction of struct_direction
        candle_dir = int(dir_col.loc[idx])
        if candle_dir != -sd:
            continue

        # Condition 3: Unfilled imbalance (in sd direction) AFTER candidate
        # Check range (candidate_idx + 1, CTS_idx] for unfilled imbalance in sd direction.
        # IC validation is strict: the same-direction filter (direction=sd) is the
        # whole point — an IC's role is to anchor a same-direction continuation, so
        # a counter-direction imbalance in the tail wouldn't justify it. (Every Fib /
        # scenario check is sd-strict too since 2026-05-23 — IMBALANCE_FILL_SEMANTICS.)
        # evaluated_at=None (no knowability cut, Plan F) for both callers:
        # derive_poi_zones identifies ICs retrospectively on the final fib (the
        # activation SWEEP enforces when a POI can go live; Plan E §2.4 item 8),
        # and MS's in-flight read (`poi_inner_for_fib`, PLAN_I S5) is used only at
        # candles after the CTS, where every gap it counts has formed
        # (MARKET_STRUCTURE_SPEC "Snapshot vs per-candle").
        has_unfilled_after = has_unfilled_imbalance(
            df,
            start_idx=idx + 1,        # strictly after
            end_idx=cts_idx,          # inclusive
            check_to_idx=cts_idx,     # evaluate fill as of fib's current CTS upper bound
            direction=sd,
            fill_threshold=config.fill_threshold,
            evaluated_at=None,
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

    UNWIRED since PLAN_H landing B (2026-10-06): the one POI per record is the
    institutional-pattern pick (`select_ip_ic`). Kept with its tests as a
    possible no-pattern fallback (PLAN_H X1; register D.2).

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


def fib_zone_bounds(
    anchor_high: float,
    anchor_low: float,
    sd: int,
    config: POIConfig,
) -> Tuple[float, float]:
    """The IC zone `(zone_lo, zone_hi)` = the fib's `ic_fib_min`–`ic_fib_max`
    (61.8–80) band, from its anchors (`features.fibonacci.calculate_fib_price` —
    `FibRetracement.price_at_pct`'s formula). ONE helper for the institutional
    pattern's touch and (PLAN_H §8.5) its as-of re-check."""
    a = calculate_fib_price(anchor_high, anchor_low, config.ic_fib_min, sd)
    b = calculate_fib_price(anchor_high, anchor_low, config.ic_fib_max, sd)
    return min(a, b), max(a, b)


def ic_touches_zone(high: float, low: float, zone_lo: float, zone_hi: float) -> bool:
    """The IC touches the zone — inclusive at both edges (PLAN_H §1 item 1)."""
    return high >= zone_lo and low <= zone_hi


@dataclass(frozen=True)
class IPCandidate:
    """One §3.1 candidate of a fib record with its institutional pattern; `ic_ref_reason` = the IC reference
    condition's outcome (`ic_reference_reason`; None = passes — PLAN_K A2)."""
    ip: IPResult
    touches_zone: bool
    ic_ref_reason: Optional[str] = None


@dataclass(frozen=True)
class IPSelection:
    """`select_ip_ic`'s answer for one fib: `candidates` (index order; with
    `classify_misses` also the §3.1 candidates that miss the touch), `pick` (the
    one IC, or None — no pattern, no POI), `tiebreak` = `(rule, runner_up_ic_idx,
    tied_count, imbalance_first_c2_idx)` when TB1 / TB2 decided, else None —
    `tied_count` = the size of the group the DECIDING rule broke (TB1: the
    candidates tied on (rank, shape); TB2: the TB1 survivors)."""
    candidates: Tuple[IPCandidate, ...]
    pick: Optional[IPResult]
    tiebreak: Optional[Tuple[str, int, int, Optional[int]]]


def select_ip_ic(
    df: pd.DataFrame,
    fib_state: FibState,
    config: POIConfig,
    *,
    ic_reference: ICReference,
    classify_misses: bool = False,
) -> IPSelection:
    """ONE IC per fib, chosen by the institutional pattern (PLAN_H §1; D2: the
    drawn POI and MS's in-flight read call this same function).

    §3.1 candidates (`find_ic_candidates` with the AGREED direction) → the touch
    of the 61.8–80 zone (inclusive) → each candidate's pattern
    (`patterns.institutional_pattern.classify_ics`) → the best (rank, shape) →
    TB1 (the IC's candle type, maru > normal > pinbar) → TB2a (the reference =
    the earliest unfilled sd imbalance after the OLDEST tied IC — IC condition
    3's own test, `iter_unfilled_imbalances`, `evaluated_at=None`; the pick = the
    most recent tied IC before the reference's first c2 — an IC may be its c1).
    TB1's runner-up = the most recent IC of the best type among its losers; TB2's
    = the most recent other TB1 survivor.

    The candle params come from the FRAME (`df.attrs["timeframe"]` →
    `candle_params_for_timeframe`; a KeyError when unstamped — fail loudly), as
    does the leg cap (`config.ip_leg_cap_by_tf`, KeyError on an unknown TF).
    `classify_misses` (the debug CSV only) also classifies the candidates that
    miss the touch — never picked.

    `ic_reference` (REQUIRED — PLAN_K A1; built by `FibTracker.ic_reference` as of the reader's moment): the IC
    reference condition, applied per §3.1 candidate BEFORE the touch and the pattern (RX1-E); each candidate carries
    its outcome (`IPCandidate.ic_ref_reason`). While `_IC_REFERENCE_FILTER` is off (landing A) it is reported only.
    """
    timeframe = df.attrs["timeframe"]
    params = candle_params_for_timeframe(timeframe)
    cap = config.ip_leg_cap_by_tf[timeframe]
    fib = fib_state.fib
    if fib is None or not (fib_state.active or fib_state.locked):
        return IPSelection((), None, None)
    sd = int(fib_state.struct_direction)
    agreed = agreed_direction(df["direction"].to_numpy(), df["vol_dir"].to_numpy())
    candidates = find_ic_candidates(df, fib_state, config, ic_direction=pd.Series(agreed, index=df.index))
    zone_lo, zone_hi = fib_zone_bounds(fib.anchor_high, fib.anchor_low, sd, config)
    touches = {
        i: ic_touches_zone(float(df.loc[i, "h"]), float(df.loc[i, "l"]), zone_lo, zone_hi) for i in candidates
    }
    ref_reasons = {i: ic_reference_reason(df, i, ic_reference, sd) for i in candidates}
    eligible = {i: touches[i] and (ref_reasons[i] is None or not _IC_REFERENCE_FILTER) for i in candidates}
    to_classify = [i for i in candidates if classify_misses or eligible[i]]
    results = classify_ics(
        df, to_classify, sd,
        bos_anchor_idx=fib_state.bos_idx, cts_anchor_idx=fib_state.cts_idx,
        agreed=agreed, cap=cap, params=params,
    )
    cands = tuple(IPCandidate(ip=r, touches_zone=touches[r.ic_idx], ic_ref_reason=ref_reasons[r.ic_idx])
                  for r in results)

    patterns = [c.ip for c in cands if eligible[c.ip.ic_idx] and c.ip.rank is not None]
    if not patterns:
        return IPSelection(cands, None, None)
    best_key = min(r.order_key for r in patterns)
    tied = [r for r in patterns if r.order_key == best_key]
    if len(tied) == 1:
        return IPSelection(cands, tied[0], None)

    def _type_rank(r: IPResult) -> int:
        return IC_TYPE_PREFERENCE.index(str(df.loc[r.ic_idx, "candle_type"]))

    best_type = min(_type_rank(r) for r in tied)
    tb1 = [r for r in tied if _type_rank(r) == best_type]
    if len(tb1) == 1:
        losers = [r for r in tied if r is not tb1[0]]
        runner_type = min(_type_rank(r) for r in losers)
        runner_up = max(r.ic_idx for r in losers if _type_rank(r) == runner_type)
        return IPSelection(cands, tb1[0], ("TB1", int(runner_up), len(tied), None))

    oldest = min(r.ic_idx for r in tb1)
    reference = min(
        iter_unfilled_imbalances(
            df, oldest + 1, fib_state.cts_idx, fib_state.cts_idx, config.fill_threshold,
            direction=sd, evaluated_at=None,
        ),
        key=lambda inst: inst.start_idx,
    )
    # Structural: a c2's candle direction is sd (`compute_imbalance`) and a merged run is contiguous, while an IC's
    # candle direction is -sd or 0 — no reference gap can straddle the oldest tied IC (PLAN_H §8.8 PR13).
    assert reference.start_idx > oldest, (
        f"[poi_zones] TB2 reference imbalance {reference.start_idx} not after the oldest tied IC {oldest}")
    pick = max((r for r in tb1 if r.ic_idx < reference.start_idx), key=lambda r: r.ic_idx)
    runner_up = max(r.ic_idx for r in tb1 if r is not pick)
    return IPSelection(cands, pick, ("TB2", int(runner_up), len(tb1), int(reference.start_idx)))


def _poi_record_gate(fib_state: FibState) -> bool:
    """The per-record lifecycle gate of the POI records (FIB_LIFECYCLE_SPEC §11): the
    live condition-active version (active AND its cycle not ended) OR a locked
    record.

    This is the byte-identical-PRESERVING per-record form, NOT the cycle-level
    `status == "active" OR locked`: `status` is shared by all version records of
    a live cycle, so a status-keyed gate would wrongly process superseded (dead)
    cross versions of a still-live cycle (same reason the chart gate, §9.1, uses
    raw per-record flags). `active` is condition-only post-Session-2, so
    `end_idx is None` (cycle not ended) is added (the pre-Session-2 `active` was
    also False once a cycle ended). Locked records — confirmed historical ones,
    ended-locked included — always pass; `disappeared` records never reach here
    (filtered by `get_fibs_for_charting`)."""
    return bool((fib_state.active and fib_state.end_idx is None) or fib_state.locked)


@dataclass(frozen=True)
class IPRecordSelection:
    """One POI record (`select_ip_records`): a gated fib record, its cycle's
    CTS-established MOMENT and its institutional-pattern selection — the ONE
    input of the drawn POIs (`derive_poi_zones`) and of the debug CSV
    (`derive_ip_candidates`)."""
    fib_state: FibState
    cts_established_idx: int
    selection: IPSelection


def select_ip_records(
    df: pd.DataFrame,
    structure_events: List[StructureEvent],
    fib_tracker: Optional[FibTracker],
    config: POIConfig,
    *,
    lifecycle_cap: Optional[int] = None,
) -> List[IPRecordSelection]:
    """The POI records in `get_fibs_for_charting` order, each with its selection
    (PLAN_H §8.5: computed ONCE per pipeline run, handed to `derive_poi_zones` and
    `derive_ip_candidates`). Per record, in this order:

    1. the lifecycle gate (`_poi_record_gate`);
    2. the cycle's CTS_ESTABLISHED: its MOMENT (`meta["confirmed_at"]`, the
       canonical cycle lifecycle-start — `structure_lifecycle.compute_cycle_lifecycle`)
       is the activation floor's cycle term and the exported
       `cts_established_idx`, NOT the CTS anchor (Plan D, 2026-09-23). Direct
       index, no `.get` fallback: a missing moment fails loudly. A cycle with no
       CTS_ESTABLISHED gets NO record (user decision 2026-09-29, zones-audit
       "fallback POI" option N: its POIs could never activate) — reached only by
       a still-LIVE pre-created cross fib (an open-ended sub). A LOCKED fib there
       breaks the event contract (a fib locks at its cycle's CTS_CONFIRMED, after
       the establishment): AssertionError — checked BEFORE the selection, so it
       fires whatever the pattern;
    3. `select_ip_ic(..., classify_misses=True)` — the CSV's candidates; the pick
       is the one without `classify_misses` (a candidate missing the touch is
       never picked) — with the record's FINAL IC reference, as of the structure's end
       (`fib_tracker.ic_reference(..., evaluated_at=lifecycle_cap)`, the call
       `FibTracker.stamp_ic_references` makes: the drawn pick and the exported meta are one value, PLAN_K A3;
       the watch on the whole frame, never cut at the cap — §11.2 LA1).
    """
    if fib_tracker is None:
        return []
    cts_established_by_key: Dict[Tuple[int, int], StructureEvent] = {}
    for ev in structure_events:
        if ev.type == "CTS_ESTABLISHED":
            # Direct index (LANDMINES "Event Contract Rules"): the emitter always
            # stamps both; a default 0 would silently file a stray event under
            # (0, 0) while `compute_cycle_lifecycle` skips it (2026-09-28).
            cts_established_by_key[(int(ev.meta["structure_id"]), int(ev.meta["cycle_id"]))] = ev
    out: List[IPRecordSelection] = []
    for fib_state in fib_tracker.get_fibs_for_charting():
        if not _poi_record_gate(fib_state):
            continue
        key = (int(fib_state.structure_id), int(fib_state.cycle_id))
        cts_event = cts_established_by_key.get(key)
        if cts_event is None:
            assert not fib_state.locked, (
                f"[poi_zones] locked fib {key} has no CTS_ESTABLISHED (event contract)")
            continue
        out.append(IPRecordSelection(
            fib_state=fib_state,
            cts_established_idx=int(cts_event.meta["confirmed_at"]),
            selection=select_ip_ic(
                df, fib_state, config, classify_misses=True,
                ic_reference=fib_tracker.ic_reference(fib_state, df, evaluated_at=lifecycle_cap),
            ),
        ))
    return out


def derive_poi_zones(
    df: pd.DataFrame,
    structure_events: List[StructureEvent],
    fib_tracker: Optional[FibTracker] = None,
    config: Optional[POIConfig] = None,
    lifecycle_floor: Optional[int] = None,
    lifecycle_cap: Optional[int] = None,
    cap_reason: str = "lifecycle_end",
    ip_records: Optional[List[IPRecordSelection]] = None,
) -> List[POIZone]:
    """
    Derive the POI zones: at most ONE per fib record — the institutional-pattern
    pick (PLAN_H; no pattern -> no POI, D7) — with its activation history.

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
    ip_records : list of IPRecordSelection, optional
        The records with their selections (`select_ip_records`), computed once
        by the pipeline and shared with the debug CSV; None = selected here.

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

    # Reversal-confirmed idx per structure_id (reversal ends ALL zones for that
    # structure). Shared helper (B2 dedup, 2026-05-27) — was duplicated verbatim
    # here and in kl_zones_v1.
    reversal_idx_by_sid: Dict[int, int] = compute_reversal_idx_by_sid(structure_events)

    # Structure lifecycle-start per sid (Phase 3 Commit 2, 2026-05-26): the idx
    # a structure first becomes active — sid 0 = its first CTS_ESTABLISHED
    # moment (Plan E E3f); sid N>=1 = reversal-confirmation idx of sid N-1; subordinate
    # = `lifecycle_floor` = the unique sub's real-time `start_idx`
    # (= max(probe_finalize, trigger, parent floor) on its first live record,
    # PART4 §17.4), supplied (slice-local) by the projection
    # (`render_sub_projection`) — this layer stays parent-agnostic. POI activation is floored here so a
    # post-reversal cycle-0 POI (whose CTS_ESTABLISHED can precede the reversal)
    # cannot activate before its structure is alive. Mirrors the KL clamp; see
    # PART4_REFACTOR_SPEC §5 + ARCHITECTURE "Activation floor".
    # Start resolution is the shared pure-leaf helper (B1 unification, 2026-05-27).
    struct_start_by_sid = compute_struct_start_by_sid(
        structure_events, reversal_idx_by_sid, lifecycle_floor,
    )
    # Cycle lifecycle table for END inheritance (B2 pass-through, 2026-05-27):
    # end = min(next-cycle clamped start, reversal, lifecycle_cap). POI inherits
    # its (sid, cycle) end from here instead of recomputing it per fib. cap=None
    # for main; subs supply the slice-local cap (+ cap_reason) via the projection.
    cycle_life = compute_cycle_lifecycle(
        structure_events, reversal_idx_by_sid, lifecycle_floor, lifecycle_cap, cap_reason,
    )

    # Pre-group CTS_ESTABLISHED + CTS_UPDATED events by (sid, cycle_id) so the
    # per-POI activation scan doesn't re-iterate the full event list for every
    # IC. Sort once in ascending order of each event's MOMENT (the sweep's time
    # order; Plan E E3g-1 — stable, so ties keep the processing order).
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
        cts_events_by_key[key].sort(key=ef.event_moment)

    # Precompute each imbalance instance's fill_idx (the first candle after
    # inst.end_idx where ≥70% retrace into the merged gap fires). One linear
    # scan per instance — shared across every POI's activation sweep.
    imbalances: List[ImbalanceInstance] = df.attrs.get("imbalances", [])
    fill_idx_cache = _compute_fill_idx_cache(df, imbalances, config.fill_threshold)

    if ip_records is None:
        ip_records = select_ip_records(df, structure_events, fib_tracker, config, lifecycle_cap=lifecycle_cap)

    # One record = one fib record that passed the gate and has a CTS_ESTABLISHED
    # (`select_ip_records`); its pick (or None — no pattern, no POI: PLAN_H D7).
    for rec in ip_records:
        fib_state = rec.fib_state
        pick = rec.selection.pick
        if pick is None:
            continue

        sid = fib_state.structure_id
        cycle_id = fib_state.cycle_id
        sd = fib_state.struct_direction
        cts_established_idx = rec.cts_established_idx

        # end_idx + end_reason inherited from the cycle (B2 pass-through):
        # min(next-cycle clamped start, reversal, lifecycle_cap). Terminal /
        # irreversible. None → zone extends to chart end.
        life = cycle_life.get((sid, cycle_id))
        end_idx: Optional[int] = life[1] if life is not None else None
        end_reason: Optional[str] = life[2] if life is not None else None
        end_time = _time(end_idx) if end_idx is not None else None

        # Last candle for the activation scan window.
        last_candle = int(df.index.max())
        scan_end = min(end_idx - 1, last_candle) if end_idx is not None else last_candle

        ic_idx = int(pick.ic_idx)
        ic_time = _time(ic_idx)
        assert ic_time is not None, f"[poi_zones] IP pick {ic_idx} is not a row of the frame"
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
        #      starting at first_active = max(cts_established_idx, ic_idx,
        #      lifecycle floor, pattern_confirmed_idx)).
        #   3. `has_unfilled_imbalance(df, ic_idx+1, t, check_to_idx=t,
        #      direction=sd, evaluated_at=t)` — POI-specific sd-direction
        #      imbalance check, counting only gaps FORMED by t (Plan F: an
        #      instance exists from its first c3, `inst.formed_at`),
        #      decoupled from Fib's `active` flag.
        #   5. The touch: the IC candle touches the AS-OF 61.8-80% zone of the
        #      fib's bounds [bos_price, cts_price_at(t)] (`fib_zone_bounds` +
        #      `ic_touches_zone`, the selection's own helpers — PLAN_H D1a / R3).
        # (Condition 4 — the IC reference condition of POI_ZONES_SPEC §3.2,
        # PLAN_K — is evaluated at IC SELECTION only (`select_ip_ic`), never in
        # this sweep (PLAN_H RX1-E): computed + exported since PLAN_K landing A,
        # the pick filtered by it from landing B.)
        # End-state (`end_idx`) takes precedence: once ended, `status` is
        # "ended" regardless of activation conditions.
        cts_events_for_fib = cts_events_by_key.get((int(sid), int(cycle_id)), [])
        activation_history = _compute_poi_activation_history(
            df=df,
            ic_idx=ic_idx,
            cts_established_idx=cts_established_idx,
            sd=int(sd),
            scan_end=scan_end,
            bos_price=float(fib_state.bos_price),
            cts_events=cts_events_for_fib,
            config=config,
            imbalances=imbalances,
            fill_idx_cache=fill_idx_cache,
            lifecycle_floor_idx=struct_start_by_sid.get(int(sid)),
            pattern_confirmed_idx=int(pick.pattern_confirmed_idx),
        )

        # confirmed_idx = idx of the most recent ACTIVATE event (if any).
        confirmed_idx: Optional[int] = None
        currently_active = False
        for ev in activation_history:
            if ev["active"]:
                confirmed_idx = ev["idx"]
                currently_active = True
            else:
                currently_active = False

        # Derived 3-state status. Mirrors KL: a POI that never activated
        # (empty activation_history — e.g. a collapsed cycle) is "inactive",
        # not "ended" (it never began). Standardized 2026-05-27.
        if not activation_history:
            status = "inactive"
        elif end_idx is not None and end_idx <= last_candle:
            status = "ended"
        elif currently_active:
            status = "active"
        else:
            status = "inactive"

        tb = rec.selection.tiebreak or (None, None, None, None)
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
                # The institutional pattern of the IC (PLAN_H D6 / HV): flat
                # str / int / None — the index keys are in the mirror's
                # `_ZONE_META_IDX_KEYS`, `ip_tied_count` is a count.
                "ip_rank": pick.ip_rank,
                "ip_leg_candle_counts": pick.ip_leg_candle_counts,
                "pattern_anchor_idx": int(pick.pattern_anchor_idx),
                "opp_leg_first_idx": int(pick.opp_leg_first_idx),
                "pattern_end_idx": int(pick.pattern_end_idx),
                "pattern_confirmed_idx": int(pick.pattern_confirmed_idx),
                "ip_tiebreak": tb[0],
                "ip_tb_runner_up_ic_idx": tb[1],
                "ip_tied_count": tb[2],
                "ip_tb_imbalance_first_c2_idx": tb[3],
                "status": status,
                "activation_history": activation_history,
                "bos_anchor_idx": fib_state.bos_idx,
                "cts_anchor_idx": fib_state.cts_idx,
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
            print(f"[poi_zones] sid={z.meta.get('structure_id')} cycle={z.meta.get('cycle_id')} "
                  f"bos_anchor_idx={z.meta['bos_anchor_idx']} cts_est={z.meta.get('cts_established_idx')} "
                  f"confirmed={z.meta.get('confirmed_idx')} ic={z.ic_idx} cts_anchor_idx={z.meta['cts_anchor_idx']} "
                  f"deact={deact_idx} react_after={react_after} end={z.meta.get('end_idx')} "
                  f"end_reason={z.meta.get('end_reason')} status={z.meta.get('status')} "
                  f"dir={z.meta.get('struct_direction')} ip_rank={z.meta['ip_rank']} "
                  f"pattern_confirmed={z.meta['pattern_confirmed_idx']} hist=[{flips_summary}]")

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
                and (z_scan_end is None or inst.formed_at <= z_scan_end)
            ]
            for k, inst in enumerate(z_relevant):
                print(f"[poi_zones]   imb#{k} start={inst.start_idx} end={inst.end_idx} "
                      f"dir={inst.direction:+d} gap_top={inst.gap_top:.5f} "
                      f"gap_bottom={inst.gap_bottom:.5f} "
                      f"armed={inst.meta.get('armed_idx')} "
                      f"confirmed={inst.meta.get('confirmed_fill_idx')}")

    return zones


@dataclass(frozen=True)
class IPCandidateRecord:
    """One row of the institutional-pattern debug export (`*_ip_candidates.csv`,
    PLAN_H §8.3; the picked rows are the drawn POIs since landing B): a §3.1
    candidate of a POI record, or the record's `IP: none` row (`ic_idx` None)
    when it has no pattern. Every index field is a dataclass field (the sub mirror shifts
    `entity_df_mutation._IP_CANDIDATE_IDX_FIELDS`); `meta` = the attribution
    (`structure_path_id`; a sub's mirror adds `sub_id` and the rest). The
    tiebreak fields sit on the picked row."""
    structure_id: int
    cycle_id: int
    struct_direction: int
    bos_anchor_idx: int
    cts_anchor_idx: int
    ic_idx: Optional[int]
    candle_type: Optional[str]
    agreed_dir: Optional[int]
    touches_zone: Optional[bool]
    ip_rank: Optional[str]
    reason: Optional[str]
    pattern_anchor_idx: Optional[int]
    opp_leg_first_idx: Optional[int]
    pattern_end_idx: Optional[int]
    pattern_confirmed_idx: Optional[int]
    ip_leg_candle_counts: Optional[str]
    picked: bool
    ip_tiebreak: Optional[str] = None
    ip_tb_runner_up_ic_idx: Optional[int] = None
    ip_tied_count: Optional[int] = None
    ip_tb_imbalance_first_c2_idx: Optional[int] = None
    ic_ref_reason: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __deepcopy__(self, memo):
        """Shared, never copied: these records sit in `df.attrs["ip_candidates"]`,
        which pandas deep-copies on every `__finalize__` (the
        `ImbalanceInstance.__deepcopy__` reasoning). Nothing mutates one in place
        — the mirror builds shifted copies with `dataclasses.replace`."""
        return self


def derive_ip_candidates(
    df: pd.DataFrame,
    structure_events: List[StructureEvent],
    fib_tracker: Optional[FibTracker],
    config: POIConfig,
    *,
    structure_path_id: str,
    ip_records: Optional[List[IPRecordSelection]] = None,
    lifecycle_cap: Optional[int] = None,
) -> List[IPCandidateRecord]:
    """The institutional-pattern selection of every POI record (`select_ip_records`
    — the records and selections `derive_poi_zones` draws from) as debug records
    (PLAN_H §8.3). One record per §3.1 candidate, the ones missing the touch
    included (`select_ip_ic(..., classify_misses=True)`), + one `IP: none` record
    per fib record without a pick. `ip_records` None = selected here.
    """
    if ip_records is None:
        ip_records = select_ip_records(df, structure_events, fib_tracker, config, lifecycle_cap=lifecycle_cap)
    out: List[IPCandidateRecord] = []
    for rec in ip_records:
        fib_state, sel = rec.fib_state, rec.selection
        sid, cycle_id = int(fib_state.structure_id), int(fib_state.cycle_id)
        base = (sid, cycle_id, int(fib_state.struct_direction), int(fib_state.bos_idx), int(fib_state.cts_idx))
        for c in sel.candidates:
            r = c.ip
            picked = sel.pick is not None and r.ic_idx == sel.pick.ic_idx
            tb = sel.tiebreak if picked and sel.tiebreak is not None else (None, None, None, None)
            out.append(IPCandidateRecord(
                *base,
                ic_idx=int(r.ic_idx),
                candle_type=str(df.loc[r.ic_idx, "candle_type"]),
                agreed_dir=int(agreed_direction(df.loc[r.ic_idx, "direction"], df.loc[r.ic_idx, "vol_dir"])),
                touches_zone=bool(c.touches_zone),
                ip_rank=r.ip_rank,
                reason=r.reason,
                pattern_anchor_idx=r.pattern_anchor_idx,
                opp_leg_first_idx=r.opp_leg_first_idx,
                pattern_end_idx=r.pattern_end_idx,
                pattern_confirmed_idx=r.pattern_confirmed_idx,
                ip_leg_candle_counts=r.ip_leg_candle_counts,
                picked=picked,
                ip_tiebreak=tb[0],
                ip_tb_runner_up_ic_idx=tb[1],
                ip_tied_count=tb[2],
                ip_tb_imbalance_first_c2_idx=tb[3],
                ic_ref_reason=c.ic_ref_reason,
                detail=dict(r.detail),
                meta={"structure_path_id": structure_path_id},
            ))
        if sel.pick is None:
            out.append(IPCandidateRecord(
                *base,
                ic_idx=None, candle_type=None, agreed_dir=None, touches_zone=None, ip_rank=None,
                reason="no_pattern" if sel.candidates else "no_candidates",
                pattern_anchor_idx=None, opp_leg_first_idx=None, pattern_end_idx=None,
                pattern_confirmed_idx=None, ip_leg_candle_counts=None, picked=False,
                meta={"structure_path_id": structure_path_id},
            ))
    return out


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
    bos_price: float,
    cts_events: List[StructureEvent],
    config: POIConfig,
    imbalances: List[ImbalanceInstance],
    fill_idx_cache: Dict[int, Tuple[Optional[int], Optional[int]]],
    lifecycle_floor_idx: Optional[int] = None,
    pattern_confirmed_idx: int,
) -> List[Dict[str, Any]]:
    """Produce the POI's activation history via an event-driven sweep.

    POI is "active" at candle `t` iff all condition-state requirements hold
    simultaneously (decoupled from FibState.active):

      Condition 1 — `cts_at(t) >= ic_idx`. The fib's `cts_idx` only grows
                    via CTS_UPDATED events, so once met, monotonic.
      Condition 2 — `ic_idx <= t` (implicit: scan starts at first_active
                    = max(cts_established_idx, ic_idx, lifecycle_floor_idx,
                    pattern_confirmed_idx), with `cts_established_idx` = the
                    cycle's CTS-established moment and `pattern_confirmed_idx`
                    = the moment the IC's continuation leg is known complete
                    (PLAN_H R1) — POI_ZONES_SPEC.md §4 "Activation floor").
      Condition 3 — an sd-direction imbalance FORMED by t (its first c3 has
                    closed: `inst.formed_at <= t` — Plan F) overlaps
                    `(ic_idx, t]` and is not yet committed-filled (two-stroke:
                    stroke 1 = 70% retrace, stroke 2 = close past gap outer).
                    Flips as imbalances form / commit-fill.
      Condition 5 — The touch: the IC candle touches the 61.8-80% zone
                    (inclusive edges) of the fib AS OF t — from `bos_price` and
                    the time-varying `cts_price_at(t)` (`fib_zone_bounds` +
                    `ic_touches_zone`, the selection's helpers; PLAN_H D1a).
                    Deactivation reason `ic_not_touching_fib_zone` (R3).

    Why event-driven: state can change only at three kinds of idx —
    imbalance "enter unfilled set" (`inst.formed_at`, the first c3), imbalance "leave
    unfilled set" (cached `confirmed_fill_idx` — the stroke-2 candle),
    and CTS event idx (cts_price / cts_idx update → cond5 may flip).
    Between these idx state is constant, so the per-candle loop over
    `[first_active, scan_end]` is wasted work. Transitions are enumerated,
    sorted, and swept once; the unfilled-count is maintained by +1/−1 at
    enter/leave events, and the touch is recomputed only at CTS transitions.

    `fill_idx_cache` entries are `(armed_idx, confirmed_fill_idx)` tuples;
    the sweep uses ``confirmed_fill_idx`` exclusively for the leave event.
    ``armed_idx`` is informational (stashed on `inst.meta` by the cache
    builder for debug exporters).

    Returns `[{"idx", "active", "reason"}, ...]`. Empty list means POI never
    activated in the window.
    """
    # First-active = max(the cycle's established moment, the IC candle, the
    # structure lifecycle-start floor, the pattern's confirmation moment): a POI
    # cannot activate before its cycle is knowable (Plan D), before its structure
    # is alive (post-reversal cycle-0 / sub case, Phase 3 Commit 2), nor before its
    # institutional pattern is known complete (PLAN_H R1 — may lie past the frame:
    # then never active).
    first_active = max(cts_established_idx, ic_idx, int(pattern_confirmed_idx))
    if lifecycle_floor_idx is not None:
        first_active = max(first_active, int(lifecycle_floor_idx))
    if first_active > scan_end or ic_idx not in df.index:
        return []

    ic_high = float(df.loc[ic_idx, "h"])
    ic_low = float(df.loc[ic_idx, "l"])

    sorted_cts_events = sorted(cts_events, key=ef.event_moment)   # the time order (E3g-1)

    # State carried through the sweep.
    cts_anchor_idx_at_t = -1
    cts_price_at_t = 0.0
    unfilled_count = 0
    touching = False

    def _touches() -> bool:
        # The as-of fib (anchors from `bos_price` + `cts_price_at_t`, direction-
        # aware like `features.fibonacci.calculate_fib_price`); no fib yet / a
        # CTS before the IC / a degenerate range -> no touch.
        if cts_anchor_idx_at_t < ic_idx or cts_price_at_t <= 0:
            return False
        if sd == 1:
            anchor_high = cts_price_at_t
            anchor_low = bos_price
        else:
            anchor_high = bos_price
            anchor_low = cts_price_at_t
        if anchor_high - anchor_low <= 0:
            return False
        zone_lo, zone_hi = fib_zone_bounds(anchor_high, anchor_low, sd, config)
        return ic_touches_zone(ic_high, ic_low, zone_lo, zone_hi)

    # --- Pre-window state: apply CTS events KNOWN strictly before first_active
    # (their moment < first_active) so the touch (and condition 1) reflect
    # the entering state at first_active; events known in [first_active,
    # scan_end] are in-window transitions AT their moment (Plan E E3g-1, PLAN_E
    # §7.1 T1 — before it both keyed on the stamped idx, the CTS anchor for
    # CTS_ESTABLISHED / pattern-path CTS_UPDATED). The establishing
    # CTS_ESTABLISHED is known at cts_established_idx <= first_active: pre-window
    # when strictly before, else a transition at first_active itself, applied
    # atomically before that candle's evaluation (same entering state). (The IC
    # CAN lie past the anchor, even past the moment: IC candidates range up to
    # the fib's FINAL cts_idx, which CTS_UPDATED advances — 12/48 POIs on the
    # reference window.) Each CTS event has two roles: WHEN it applies (its
    # moment) and WHERE the CTS is (cond1 — the CTS anchor, a location).
    in_window_cts: List[StructureEvent] = []
    for ev in sorted_cts_events:
        ev_moment_idx = ef.event_moment(ev)
        if ev_moment_idx < first_active:
            cts_anchor_idx_at_t = ef.cts_anchor_idx(ev)
            try:
                cts_price_at_t = float(ev.price)
            except (TypeError, ValueError):
                pass
        elif ev_moment_idx <= scan_end:
            in_window_cts.append(ev)
    touching = _touches()

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
        and inst.formed_at <= scan_end     # could enter the window at all (formed by scan_end)
    ]
    for inst in relevant_imbalances:
        # An FVG exists from its first c3 (`formed_at` = start_idx + 1), not its
        # c2 — Plan F; IMBALANCE_FILL_SEMANTICS "Knowability — the c3 rule".
        enter_idx = max(inst.formed_at, first_active)
        if enter_idx > scan_end:
            continue
        _armed_cached, confirmed_cached = fill_idx_cache.get(id(inst), (None, None))
        # Instance contributes "unfilled" on [inst.formed_at, confirmed_fill_idx - 1]
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
        transitions.append((ef.event_moment(ev), PRIO_CTS, "cts", ev))

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
                cts_anchor_idx_at_t = ef.cts_anchor_idx(payload)
                try:
                    cts_price_at_t = float(payload.price)
                except (TypeError, ValueError):
                    pass
                cts_changed = True
            i += 1
        if cts_changed:
            touching = _touches()

        # Evaluate is_active at cur_idx. Skip rows not present in df.index
        # (defensive — entity dfs are contiguous, but mirror writes can
        # leave NA placeholder rows we don't want to evaluate).
        if cur_idx not in df.index:
            continue

        cond1 = cts_anchor_idx_at_t >= ic_idx
        cond3 = unfilled_count > 0
        cond5 = touching
        is_active = cond1 and cond3 and cond5

        if is_active and not was_active:
            history.append({
                "idx": cur_idx,
                "active": True,
                "reason": "initial" if not history else "conditions_re-met",
            })
            was_active = True
        elif not is_active and was_active:
            reasons = []
            if not cond1:
                reasons.append("ic_out_of_fib_bounds")
            if not cond3:
                reasons.append("imbalance_filled")
            if not cond5:
                reasons.append("ic_not_touching_fib_zone")
            history.append({
                "idx": cur_idx,
                "active": False,
                "reason": "+".join(reasons) if reasons else "unknown",
            })
            was_active = False

    return history


# ---------------------------------------------------------------------------
# MarketStructure's in-flight POI read (the dual CTS proximity check, Stage 2):
# `poi_inner_for_fib`, wired from `structure/structure_engine.py` with the
# embedded FibTracker (PLAN_I S5 / D5; Part 4 §13.5.b — structure/ never
# imports zones/).
# ---------------------------------------------------------------------------

class IPSelectionError(RuntimeError):
    """An error inside `select_ip_ic` during the MS in-flight read, re-raised so
    no narrow `except (ValueError, IndexError)` above the MS run can swallow it
    (PLAN_H §8.5 step 3)."""


def poi_inner_for_fib(
    df: pd.DataFrame,
    fib_state: Optional[FibState],
    fill_threshold: float,
    *,
    tracker: FibTracker,
    evaluated_at: int,
) -> Optional[float]:
    """MarketStructure's in-flight POI inner (PLAN_I S5 / D5, Q7): the institutional-pattern pick
    (`select_ip_ic` — the SAME function as the drawn POI, PLAN_H D2) on `fib_state`, the cycle's
    `FibTracker.representative_fib` of MS's embedded tracker — one fib decision for the drawn POI and MS.
    Returns the pick's inner (a buy POI's top = the IC's high, a sell POI's bottom = its low), or None — no fib,
    or no pattern (D7): the proximity check then reads the BOS inner only. Wired by `structure_engine` (MS
    imports nothing from `zones/`).

    Fails loudly (PLAN_H §8.8 PR1, LANDMINES "The POI Resolver Fails Loudly"; PLAN_I §9.2 F11): the frame's
    candle stamp `df.attrs["timeframe"]` is read FIRST on every call — a refresh with no fib included — and any
    error of the stamp read or the pick is re-raised as `IPSelectionError` (a RuntimeError), so neither a missing
    stamp nor an IP bug can pass as "no pattern", and the sub geometry's narrow `except (ValueError,
    IndexError)` (`entity_df_mutation._build_geometry`) cannot drop the sub. `fill_threshold` = the tracker's
    (`FIB_FILL_THRESHOLD`; the drawn POI's `POIConfig` default is the same value — pinned).

    The fib's IC reference is built HERE, inside the re-raising `try` (PLAN_K A4, §9 C01): `tracker.ic_reference(
    fib_state, df, evaluated_at=evaluated_at)` — MS's embedded tracker at the fed event's moment, the frame = MS's
    resolver frame (already cut at the run's data edge) — so an error there surfaces as `IPSelectionError` and never
    reaches the sub geometry's narrow `except (ValueError, IndexError)`. A None fib returns before the call."""
    try:
        df.attrs["timeframe"]
        if fib_state is None:
            return None
        ref = tracker.ic_reference(fib_state, df, evaluated_at=int(evaluated_at))
        sel = select_ip_ic(df, fib_state, POIConfig(fill_threshold=fill_threshold), ic_reference=ref)
    except Exception as exc:
        where = ("no fib" if fib_state is None else
                 f"sid {fib_state.structure_id}, cycle {fib_state.cycle_id}, fib {fib_state.bos_idx}..{fib_state.cts_idx}")
        raise IPSelectionError(f"[poi_zones] the MS in-flight IP read failed ({where}): {exc!r}") from exc
    if sel.pick is None:
        return None
    ic_idx = int(sel.pick.ic_idx)
    return float(df.loc[ic_idx, "h"]) if int(fib_state.struct_direction) == 1 else float(df.loc[ic_idx, "l"])
