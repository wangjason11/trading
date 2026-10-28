# engine_v2/zones/poi_zones.py
"""
POI Zone derivation using Fibonacci retracement and Institutional Candle identification.

ONE POI per fib record (PLAN_H, Week 9 Part 1 item 1), the IC identified in the
user's steps (POI_ZONES_SPEC §3; PLAN_L):
0. A fib record passes the lifecycle gate and its cycle has a CTS_ESTABLISHED
   (`select_ic_records`)
1. Step 1 — the IC candidates (`check_ic_conditions` / `find_ic_candidates`):
   every candle of the fib's BOS → CTS window (the scope) whose AGREED direction
   is -sd (condition 1) checked against (2) the IC reference condition, (3) an
   unfilled sd imbalance IMMEDIATELY after it (its first gap candle at IC + 1 …
   min(IC + 2, the CTS anchor) — PLAN_P), (4) the touch of the 61.8-80 zone —
   ALL together (PLAN_H
   RX1-F). Step 1A (PLAN_P): no IC candidate -> Step 1 again with condition 4 on
   the 80-100 fallback zone (`ic_zone_band`). Step 1B (PLAN_S step 3) — the
   prior-imbalance preference: the IC candidates whose closest prior sd gap is
   unfilled through IC − 1 (`check_prior_imbalance`) go through Steps 2–3 first;
   no IC from them, or none passing -> Steps 2–3 on all the candidates
2. Step 2 — the institutional pattern among the candidates Steps 2–3 run on +
   its tiebreak rules (`select_institutional_pattern` / `apply_tiebreak_rules`);
   a pattern -> the IC (`ip_outcome` "pattern")
3. Step 3 — no pattern (`ip_outcome` "no_pattern") -> the ranking rules over the
   same, unfiltered list (`apply_ranking_rules`, PLAN_L L2 — a rule set of its
   own, ICR-S); every candidate a pinbar -> no IC. No IC (no IC candidate / all
   pinbars) -> no POI (Step 4, the lower-TF fallback, is deferred)
4. The POI = the IC's high / low; it is active while the activation sweep's
   conditions hold (`_compute_poi_activation_history`)
`select_ic` runs the steps. MarketStructure's in-flight read (`poi_inner_for_fib`,
on its embedded FibTracker's representative fib — PLAN_I S5) calls the same
`select_ic` (PLAN_H D2). The V30 / V60 / V90 selection (`select_ic_variants`,
its finder `find_ic_variant_candidates`) is kept unwired (PLAN_H X1, ICR-5).
"""
from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import List, Optional, Dict, Any, Literal, Sequence, Tuple

import numpy as np
import pandas as pd

from engine_v2.common.types import IMBALANCE_FILL_THRESHOLD, ImbalanceInstance, require_positional_index
from engine_v2.features.candle_params import CandleParams
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

    At most one zone per fib record: the IC `select_ic` picks — a pattern's IC
    or, with no pattern, a ranked one (`ip_outcome`, the pattern / tiebreak /
    ranking in meta).
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
    # end_reason, activation_history, status, ip_outcome, ic_zone_band (the
    # IC's zone: "61.8-80" / "80-100" — Step 1A, PLAN_P), the institutional
    # pattern — ip_rank, ip_leg_candle_counts, pattern_anchor_idx,
    # opp_leg_first_idx, pattern_end_idx, pattern_confirmed_idx (all None on a
    # ranked IC), the ip_tb_* tiebreak —, the ranking_* fields, Step 1B's
    # prior_imbalance_* fields + ic_candidate_count (PLAN_S step 3),
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
    # The fallback zone (PLAN_P, the user's change 3, "Step 1A"): IC condition 4's zone ONLY when no candle passes
    # every IC condition with the 61.8–80 zone; the other conditions unchanged (`ic_zone_config`).
    ic_fallback_fib_min: float = 80.0
    ic_fallback_fib_max: float = 100.0

    # Variant overlap thresholds
    v30_threshold: float = 0.30
    v60_threshold: float = 0.60
    v90_threshold: float = 0.90

    # Imbalance detection
    fill_threshold: float = IMBALANCE_FILL_THRESHOLD  # the stroke-1 retrace fraction (common/types.py)

    # Institutional-pattern leg cap per timeframe (PLAN_H §1 item 6: the initial leg = its run's most recent `cap`
    # candles, the continuation = its first `cap`; H1 decision; M5 2 arrives with the M5 lens). An unknown TF
    # raises (KeyError) — no default cap.
    ip_leg_cap_by_tf: Dict[str, int] = field(default_factory=lambda: {"H1": 3, "M15": 3})

    # IC condition 3's window since PLAN_P (the immediate unfilled imbalance; until PLAN_P the window of the tiebreak /
    # ranking rule of that name — PLAN_H §6 TBU-1 / TBU-2): the imbalance's first gap candle at IC + 1 … IC + this
    # many candles (raw offsets, a true doji takes a slot). Every TF.
    immediate_imbalance_candles: int = 2


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


# The IC reference condition (PLAN_K; PLAN_H §8.9 RX1) — IC condition 2 since landing B (computed and exported since
# landing A): a candle failing it is not an IC candidate (`ICCandleCheck.is_ic_candidate`).
IC_REF_PAST_REFERENCE = "ic_past_reference"
IC_REF_NOT_BEFORE_BREAK = "ic_not_before_break"


def ic_reference_reason(df: pd.DataFrame, candle_idx: int, ref: ICReference, struct_direction: int) -> Optional[str]:
    """The IC reference condition (IC condition 2) on one checked candle (§3.1; PLAN_K §3.2; PLAN_H RX1 — ONE bundled condition, RX1-B):
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


def _has_unfilled_imbalance_after(df: pd.DataFrame, idx: int, fib_state: FibState, config: POIConfig) -> bool:
    """At least one unfilled imbalance in the structure's direction in `(idx, CTS anchor]`, the fill judged through the
    fib's CTS anchor. IC condition 3 until PLAN_P (now the IMMEDIATE unfilled imbalance —
    `immediate_unfilled_imbalance_first_c2_idx`): RETIRED as a condition, kept as INFORMATION on every Step 1 check
    (`ICCandleCheck.has_unfilled_imbalance_after`, the IC CSV column — the user, 2026-10-13) and as the unwired V30 /
    V60 / V90 finder's filter (`find_ic_variant_candidates`).

    The test is strict: the same-direction filter (direction=sd) is the whole point — an IC's role is to anchor
    a same-direction continuation, so a counter-direction imbalance in the tail would not justify it (every Fib /
    scenario check is sd-strict too since 2026-05-23 — IMBALANCE_FILL_SEMANTICS). `evaluated_at=None` (no knowability
    cut, Plan F) for both consumers: the drawn POI identifies ICs retrospectively on the final fib (the activation
    SWEEP enforces when a POI can go live; Plan E §2.4 item 8), and MS's in-flight read (`poi_inner_for_fib`, PLAN_I
    S5) is used only at candles after the CTS, where every gap it counts has formed (MARKET_STRUCTURE_SPEC "Snapshot
    vs per-candle")."""
    cts_idx = fib_state.cts_idx
    return has_unfilled_imbalance(
        df,
        start_idx=idx + 1,        # strictly after
        end_idx=cts_idx,          # inclusive
        check_to_idx=cts_idx,     # evaluate fill as of fib's current CTS upper bound
        direction=fib_state.struct_direction,
        fill_threshold=config.fill_threshold,
        evaluated_at=None,
    )


def find_ic_variant_candidates(
    df: pd.DataFrame,
    fib_state: FibState,
    config: POIConfig,
    ic_direction: Optional[pd.Series] = None,
) -> List[int]:
    """
    The candidate finder of the UNWIRED V30 / V60 / V90 path (`select_ic_variants`; kept — PLAN_H X1 / §6 ICR-5).
    Was `find_ic_candidates` until PLAN_L L1 — that name is Step 1's answer now (`find_ic_candidates`).

    Conditions (ALL required):
    1. Within Fib bounds (inclusive): BOS_idx <= candle_idx <= CTS_idx
    2. Opposite direction of struct_direction: `ic_direction` (None = the candle `direction`) == -struct_direction
    3. Unfilled imbalance (in sd direction) AFTER the candle, up to the CTS anchor (`_has_unfilled_imbalance_after`)

    Returns the qualifying candle indices (index order).
    """
    # A Fib is valid for IC detection if it's active OR locked
    # Locked means CTS was confirmed - bounds are finalized but still valid
    if not (fib_state.active or fib_state.locked):
        return []

    if "direction" not in df.columns:
        return []
    sd = fib_state.struct_direction
    dir_col = df["direction"] if ic_direction is None else ic_direction
    candidates = []
    for idx in range(fib_state.bos_idx, fib_state.cts_idx + 1):
        if idx not in df.index:
            continue
        if int(dir_col.loc[idx]) != -sd:
            continue
        if not _has_unfilled_imbalance_after(df, idx, fib_state, config):
            continue
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
    IC `select_ic` picks. Kept with its tests (PLAN_H X1;
    "Keep it for now" — PLAN_H §6 ICR-5): the no-pattern role went to Step 3,
    the ranking rules (PLAN_L L2); the remaining fallback is register §D.2
    "Step 4 of IC identification".

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
    band (61.8–80; the 80–100 fallback through `ic_zone_config`, PLAN_P), from its
    anchors (`features.fibonacci.calculate_fib_price` —
    `FibRetracement.price_at_pct`'s formula). ONE helper for IC condition 4's touch
    and (PLAN_H §8.5) its as-of re-check in the activation sweep."""
    a = calculate_fib_price(anchor_high, anchor_low, config.ic_fib_min, sd)
    b = calculate_fib_price(anchor_high, anchor_low, config.ic_fib_max, sd)
    return min(a, b), max(a, b)


def ic_touches_zone(high: float, low: float, zone_lo: float, zone_hi: float) -> bool:
    """The IC touches the zone — inclusive at both edges (PLAN_H §1 item 1)."""
    return high >= zone_lo and low <= zone_hi


# The zone IC condition 4 was evaluated in (PLAN_P, the user's change 3): `ICSelection.ic_zone_band` → the POI meta,
# the debug CSV (every row of the record), the hovers. The 80–100 fallback = Step 1A (no IC candidate with 61.8–80).
# The POI's activation re-checks the touch against the SAME band (the user, 2026-10-13: "POI activation check should
# always reference the zone parameters the IC came from").
IC_ZONE_PRIMARY = "61.8-80"
IC_ZONE_FALLBACK = "80-100"


def ic_zone_config(config: POIConfig, band: str) -> POIConfig:
    """`config` with `ic_fib_min` / `ic_fib_max` set to `band` — the one place a band becomes zone parameters
    (Step 1A's touch, the activation's touch re-check); the primary band returns `config` itself."""
    if band == IC_ZONE_PRIMARY:
        return config
    assert band == IC_ZONE_FALLBACK, band
    return replace(config, ic_fib_min=config.ic_fallback_fib_min, ic_fib_max=config.ic_fallback_fib_max)


def immediate_unfilled_imbalance_first_c2_idx(
    df: pd.DataFrame, ic_idx: int, fib_state: FibState, config: POIConfig,
) -> Optional[int]:
    """IC condition 3 since PLAN_P (the user's change 1; the window and the test are PLAN_H §6 TBU-1 / TBU-2 / TBU-5's,
    when it was a tiebreak / ranking rule): the first gap candle (c2 — a merged imbalance's first) of the unfilled sd
    imbalance IMMEDIATELY after the candle, or None — the candle fails condition 3. Its first gap candle in IC + 1 …
    min(IC + `config.immediate_imbalance_candles`, the CTS anchor) — raw candle offsets, a true doji takes a slot;
    the candle may be the gap's c1. "Unfilled" = `iter_unfilled_imbalances`: sd, the two-stroke fill through the
    fib's CTS anchor, `evaluated_at=None` (no knowability cut — `_has_unfilled_imbalance_after`'s reasons) — a gap
    that fills only after the anchor counts as unfilled. Consecutive sd gap candles merge into one instance, so at
    most one imbalance sits in a 2-candle window."""
    hi = min(ic_idx + config.immediate_imbalance_candles, int(fib_state.cts_idx))
    inst = next(iter_unfilled_imbalances(
        df, ic_idx + 1, hi, fib_state.cts_idx, config.fill_threshold,
        direction=int(fib_state.struct_direction), evaluated_at=None,
    ), None)
    if inst is None:
        return None
    # Structural: a c2's candle direction is sd (`compute_imbalance`) and a merged run is contiguous, while a checked
    # candle's candle direction is -sd or 0 (its AGREED direction is -sd) — no gap can straddle it (PLAN_H §8.8 PR13).
    # Asserted here for EVERY Step 1 check (condition 3, both consumers — PLAN_P), so the most-recent-before-imbalance
    # rule's reference never straddles its oldest candidate either.
    assert inst.start_idx > ic_idx, (
        f"[poi_zones] imbalance {inst.start_idx}..{inst.end_idx} straddles the IC {ic_idx}")
    return int(inst.start_idx)


def closest_prior_imbalance(df: pd.DataFrame, ic_idx: int, fib_state: FibState) -> Optional[ImbalanceInstance]:
    """Step 1B's gap (PLAN_S step 3; the user's write-up: "only the closest imbalance prior to the IC"): the sd
    imbalance instance whose gap-candle (c2) run starts at or after the fib's BOS anchor and ends before the IC, the one
    ending LATEST — or None (no prior gap: the candidate fails 1B). The user's words fix the direction ("in direction of
    structure & opposite of IC"), the window ("BOS anchor to IC"), "A gap formed by IC counts" (its c3 is the IC:
    `end_idx = IC − 1`) and "no min size"; the author's round-1 reading P adds: a gap = an ENGINE instance
    (consecutive sd c2s merged — `compute_imbalance`) and a run straddling the BOS anchor does not count (PLAN_S §4.3).
    An sd run never contains the IC (a c2's candle direction is sd, a checked candle's −sd or 0), so `end_idx < IC`
    keeps every run before it; instances never overlap, so the latest-ending one is unique. Reads no candle."""
    sd = int(fib_state.struct_direction)
    best: Optional[ImbalanceInstance] = None
    for inst in df.attrs.get("imbalances", []):
        if inst.direction != sd or inst.start_idx < fib_state.bos_idx or inst.end_idx >= ic_idx:
            continue
        if best is None or inst.end_idx > best.end_idx:
            best = inst
    return best


def check_prior_imbalance(
    df: pd.DataFrame, check: "ICCandleCheck", fib_state: FibState, config: POIConfig,
) -> "ICCandleCheck":
    """Step 1B's test of ONE check (the prior-imbalance preference — PLAN_S step 3): `check` with
    `prior_imbalance_first_c2_idx` = its closest prior gap's first c2 (`closest_prior_imbalance`; None = no prior gap)
    and `prior_imbalance_unfilled` = that gap is NOT two-stroke-filled through the candle right BEFORE the IC
    (`is_filled(df, IC − 1, config.fill_threshold)` — the user, round 5: "the balance needs to be unfilled up to the
    candle right before the IC": the IC's own candle is never read, so it cannot disqualify itself; a gap whose c3 is
    the IC has an empty scan → unfilled, unless degenerate — `is_filled`'s safe default); no prior gap → False (the
    author's default, round 5's final spec). Every candle read is ≤ the IC — no look-ahead in either consumer."""
    inst = closest_prior_imbalance(df, check.ic_idx, fib_state)
    if inst is None:
        return replace(check, prior_imbalance_first_c2_idx=None, prior_imbalance_unfilled=False)
    return replace(
        check,
        prior_imbalance_first_c2_idx=int(inst.start_idx),
        prior_imbalance_unfilled=not inst.is_filled(df, check.ic_idx - 1, config.fill_threshold),
    )


# The institutional-pattern tiebreakers, in order (PLAN_H §6 TBU-1; descriptive names, TBU-3): the `ip_tiebreak`
# value of the rule that decided a tie (POI meta, the debug CSV, the hover). PLAN_P removed the first,
# `immediate_unfilled_imbalance` (it became IC condition 3 — every IC candidate passes it; a save before S_P can hold
# the value).
TIEBREAK_IC_CANDLE_TYPE = "ic_candle_type"
TIEBREAK_MOST_RECENT_BEFORE_IMBALANCE = "most_recent_before_imbalance"

# The ranking rules, in order (PLAN_L §4.1; PLAN_H §6 ICR-3): the `ranking_rule` value of the rule that decided Step 3
# (POI meta, the debug CSV, the hovers). Rules 2–3 take the same descriptive words as the tiebreak rules — the field
# (`ranking_rule` vs `ip_tiebreak`) tells the two rule sets apart; their own constants (ICR-S: separate rule sets).
# PLAN_P removed the second, `immediate_unfilled_imbalance` (as above).
RANKING_EXCLUDE_PINBAR = "exclude_pinbar"
RANKING_IC_CANDLE_TYPE = "ic_candle_type"
RANKING_MOST_RECENT_BEFORE_IMBALANCE = "most_recent_before_imbalance"

# Step 2's outcome (`ip_outcome`, PLAN_H §6 ICR-3): a pattern IC vs a ranked one (None — Step 2 never ran: no IC
# candidate). The no-IC reasons (`no_ic_reason`, the debug CSV's no-IC row, the fib hover's `IC: none (<reason>)`):
# Step 1 (and Step 1A, PLAN_P) found no IC candidate / ranking rule 1 removed every candidate — both where the deferred
# Step 4 would run.
IP_OUTCOME_PATTERN = "pattern"
IP_OUTCOME_NO_PATTERN = "no_pattern"
NO_IC_CANDIDATE = "no_ic_candidate"
ALL_IC_CANDIDATES_PINBAR = "all_ic_candidates_pinbar"

# Step 1B — the prior-imbalance preference (PLAN_S step 3; the user's write-up 2026-10-20 + Q&A rounds 2–7): its outcome
# (`prior_imbalance_outcome` — `ICSelection`, the POI meta, the debug CSV, the hovers; the user's names, round 6). None
# = no IC candidate (1B never ran).
PRIOR_IMBALANCE_PREFERRED = "preferred"                            # the IC came from the candidates passing 1B
PRIOR_IMBALANCE_FALLBACK_PASSERS_NO_IC = "fallback_passers_no_ic"  # passers, no IC from them -> all the candidates
PRIOR_IMBALANCE_FALLBACK_NO_PASSER = "fallback_no_passer"          # no candidate passes -> all the candidates


def prior_imbalance_narrowed(outcome: Optional[str], passer_count: Optional[int], candidate_count: Optional[int]) -> bool:
    """The preference NARROWED the field (PLAN_S §4.3.1, the user's round 6): the IC came from the passers and fewer
    candidates passed than there were — the ONE test of both hovers' 1B wording (the POI hover reads the POI meta, the
    fib hover the CSV's picked row; both carry the outcome and the two counts)."""
    return outcome == PRIOR_IMBALANCE_PREFERRED and int(passer_count) < int(candidate_count)


@dataclass(frozen=True)
class ICCandleCheck:
    """Step 1's check of ONE candle (POI_ZONES_SPEC §3 Step 1; PLAN_L §3.1): a candle of the fib's BOS → CTS window
    whose AGREED direction is −sd (IC condition 1 = the row set), with the other three conditions' outcomes —
    condition 2 the IC reference (`ic_ref_reason`, None = passes — `ic_reference_reason`), condition 3 the immediate
    unfilled imbalance (`immediate_imbalance_first_c2_idx`, its first gap candle; None = fails —
    `immediate_unfilled_imbalance_first_c2_idx`; PLAN_P), condition 4 `touches_zone` (the touch of the selection's
    `ic_zone_band`) — the retired condition 3 as information (`has_unfilled_imbalance_after`: any unfilled sd
    imbalance after it through the CTS anchor — never read by the pick, PLAN_P) and, once classified (`select_ic`),
    its institutional pattern (`pattern`, None = not classified) and its Step 1B test (`check_prior_imbalance`, PLAN_S
    step 3: `prior_imbalance_first_c2_idx` — the closest prior gap's first c2, None = no prior gap —, and
    `prior_imbalance_unfilled` — passes 1B; None = not evaluated). 1B is a PREFERENCE, never an IC condition:
    `is_ic_candidate` does not read it."""
    ic_idx: int
    has_unfilled_imbalance_after: bool
    ic_ref_reason: Optional[str]
    touches_zone: bool
    immediate_imbalance_first_c2_idx: Optional[int]
    pattern: Optional[IPResult] = None
    prior_imbalance_first_c2_idx: Optional[int] = None
    prior_imbalance_unfilled: Optional[bool] = None

    @property
    def is_ic_candidate(self) -> bool:
        """An IC CANDIDATE passes EVERY IC condition, applied TOGETHER (PLAN_H §6 RX1-F) — the ONE predicate of the
        pick (Steps 2 and 3 look among IC candidates only) and of the `no_ic_candidate` reason (`select_ic`):
        condition 3 = an unfilled sd imbalance IMMEDIATELY after the candle (PLAN_P — `has_unfilled_imbalance_after`
        is not a condition)."""
        return (self.immediate_imbalance_first_c2_idx is not None and self.ic_ref_reason is None
                and bool(self.touches_zone))


def check_ic_conditions(
    df: pd.DataFrame,
    fib_state: FibState,
    config: POIConfig,
    *,
    ic_reference: ICReference,
    agreed: np.ndarray,
) -> Tuple[ICCandleCheck, ...]:
    """Step 1's per-candle checks (PLAN_L §3.1; POI_ZONES_SPEC §3 Step 1): every candle of the fib's BOS → CTS window
    (inclusive — the scope, not a condition; a missing index skipped) whose AGREED direction (`agreed`, positional —
    `agreed_direction`) is −sd, each with the outcomes of IC conditions 2–4 (`ICCandleCheck`; condition 4 on the
    61.8–80 zone — Step 1A re-evaluates it, `select_ic`) + the retired condition 3 as information, in index order.
    The fib must have a `fib` (the caller's guard)."""
    sd = int(fib_state.struct_direction)
    fib = fib_state.fib
    zone_lo, zone_hi = fib_zone_bounds(fib.anchor_high, fib.anchor_low, sd, config)
    agreed_s = pd.Series(agreed, index=df.index)
    checks = []
    for idx in range(fib_state.bos_idx, fib_state.cts_idx + 1):
        if idx not in df.index:
            continue
        if int(agreed_s.loc[idx]) != -sd:
            continue
        checks.append(ICCandleCheck(
            ic_idx=int(idx),
            has_unfilled_imbalance_after=bool(_has_unfilled_imbalance_after(df, idx, fib_state, config)),
            ic_ref_reason=ic_reference_reason(df, idx, ic_reference, sd),
            touches_zone=ic_touches_zone(float(df.loc[idx, "h"]), float(df.loc[idx, "l"]), zone_lo, zone_hi),
            immediate_imbalance_first_c2_idx=immediate_unfilled_imbalance_first_c2_idx(df, idx, fib_state, config),
        ))
    return tuple(checks)


def find_ic_candidates(checks: Sequence[ICCandleCheck]) -> List[int]:
    """Step 1's answer: the IC CANDIDATES — the checked candles passing every IC condition (index order)."""
    return [c.ic_idx for c in checks if c.is_ic_candidate]


# ---------------------------------------------------------------------------------------------------------------
# The shared rule helpers (PLAN_H §6 ICR-9): a rule that is the SAME in Step 2's tiebreak rules and Step 3's ranking
# rules is ONE function with ONE parameter set, applied to each rule set's OWN input list; the rule sets stay separate
# and independent (ICR-S) — nothing one computes feeds the other. (The immediate-unfilled-imbalance rule's helper,
# `filter_immediate_unfilled_imbalance`, went with the rule — PLAN_P: it is IC condition 3.)
# ---------------------------------------------------------------------------------------------------------------

def prefer_candle_type(in_play: Sequence[int], df: pd.DataFrame, *, include_pinbar: bool) -> List[int]:
    """The candle-type rule — ONE function with an optional pinbar (ICR-9): the candidates in play of the best candle
    type present, maru > normal > pinbar (`include_pinbar` — the tiebreak rules' rule 1) or maru > normal (the
    ranking rules' rule 2, after their rule 1 excluded the pinbars — a pinbar here raises ValueError)."""
    order = IC_TYPE_PREFERENCE if include_pinbar else tuple(t for t in IC_TYPE_PREFERENCE if t != "pinbar")
    type_rank = {i: order.index(str(df.loc[i, "candle_type"])) for i in in_play}
    best = min(type_rank.values())
    return [i for i in in_play if type_rank[i] == best]


def most_recent_before_imbalance(
    in_play: Sequence[int], df: pd.DataFrame, fib_state: FibState, config: POIConfig,
) -> Tuple[int, int]:
    """The most-recent-before-the-imbalance rule (the tiebreak rules' rule 2; the ranking rules' rule 3):
    the reference = the earliest unfilled sd imbalance after the OLDEST candidate in play (`iter_unfilled_imbalances`,
    the two-stroke fill through the CTS anchor, `evaluated_at=None`); the pick = the most recent candidate in play
    before the reference's first gap candle (a candidate may be its c1 — of two neighbours sharing one gap, the one
    closer to it, ICR-1). Returns `(pick, reference_first_c2_idx)`. Since PLAN_P every candidate passed IC condition 3
    (the immediate unfilled imbalance), so the reference IS the oldest candidate's own immediate gap and the pick is
    that oldest candidate — or its neighbour sharing that gap, the closer one (the user, 2026-10-13: "Yes"). The
    reference never straddles the oldest candidate (`immediate_unfilled_imbalance_first_c2_idx` asserts the structure
    for every check)."""
    oldest = min(in_play)
    reference = min(
        iter_unfilled_imbalances(
            df, oldest + 1, fib_state.cts_idx, fib_state.cts_idx, config.fill_threshold,
            direction=int(fib_state.struct_direction), evaluated_at=None,
        ),
        key=lambda inst: inst.start_idx,
    )
    pick = max(i for i in in_play if i < reference.start_idx)
    return int(pick), int(reference.start_idx)


def apply_tiebreak_rules(
    df: pd.DataFrame,
    fib_state: FibState,
    config: POIConfig,
    tied: Sequence[IPResult],
) -> Tuple[IPResult, Tuple[str, int, int, Optional[int]]]:
    """Step 2's TIEBREAK RULES over the ICs of the patterns tied on the best (rank, shape) — their own rule set
    (PLAN_H §6 ICR-S), each rule over the ICs the previous one left in play, the first to leave exactly one deciding
    (PLAN_H §6 TBU-1; its first rule, the immediate unfilled imbalance, became IC condition 3 — PLAN_P):
    1. the IC's candle type, maru > normal > pinbar (`prefer_candle_type(include_pinbar=True)`); runner-up = the most
       recent IC of the best type among its losers;
    2. the most recent before the imbalance (`most_recent_before_imbalance`) — always their OLDEST: every IC passed
       condition 3, and a pattern's IC is the last candle of its opp leg, so no other tied IC sits before the oldest
       one's immediate gap (TBU-4); runner-up = the most recent other IC in play.
    Returns `(pick, (rule, runner_up_ic_idx, tied_count, imbalance_first_c2_idx))` — `rule` one of the `TIEBREAK_*`
    names; `tied_count` = the size of the group the DECIDING rule broke; `imbalance_first_c2_idx` = the reference
    imbalance (None for the candle type)."""
    by_idx = {int(r.ic_idx): r for r in tied}
    in_play = list(by_idx)

    # 1. The IC's candle type.
    by_type = prefer_candle_type(in_play, df, include_pinbar=True)
    if len(by_type) == 1:
        losers = [i for i in in_play if i != by_type[0]]
        runner_up = max(prefer_candle_type(losers, df, include_pinbar=True))
        return by_idx[by_type[0]], (TIEBREAK_IC_CANDLE_TYPE, int(runner_up), len(in_play), None)

    # 2. The most recent before the imbalance.
    pick, reference_c2 = most_recent_before_imbalance(by_type, df, fib_state, config)
    runner_up = max(i for i in by_type if i != pick)
    return by_idx[pick], (TIEBREAK_MOST_RECENT_BEFORE_IMBALANCE, int(runner_up), len(by_type), reference_c2)


def select_institutional_pattern(
    df: pd.DataFrame,
    fib_state: FibState,
    config: POIConfig,
    ic_candidates: Sequence[ICCandleCheck],
) -> Tuple[Optional[IPResult], Optional[Tuple[str, int, int, Optional[int]]]]:
    """Step 2 (POI_ZONES_SPEC §3.4): the best (rank, shape) among the patterns of `ic_candidates` (every IC candidate,
    or Step 1B's passers — `select_ic`, PLAN_S step 3; also over all of them for the outranked pattern); two or more tied → the
    tiebreak rules (`apply_tiebreak_rules`). No pattern → `(None, None)`. The IC candidates carry their patterns
    (classified by `select_ic`) — nothing re-derived."""
    patterns = [c.pattern for c in ic_candidates if c.pattern.rank is not None]
    if not patterns:
        return None, None
    best_key = min(r.order_key for r in patterns)
    tied = [r for r in patterns if r.order_key == best_key]
    if len(tied) == 1:
        return tied[0], None
    return apply_tiebreak_rules(df, fib_state, config, tied)


def apply_ranking_rules(
    df: pd.DataFrame,
    fib_state: FibState,
    config: POIConfig,
    ic_candidates: Sequence[ICCandleCheck],
) -> Tuple[Optional[IPResult], Optional[Tuple[str, Optional[int], int, Optional[int]]]]:
    """Step 3's RANKING RULES (POI_ZONES_SPEC §3 Step 3; PLAN_L §4.1) — run by `select_ic` ONLY when Step 2 found no
    pattern, over the same UNFILTERED list Step 2 ran on (every IC candidate, or Step 1B's passers — PLAN_S step 3;
    their CHECKS, nothing re-derived — A9). A rule set of its OWN (PLAN_H
    §6 ICR-S), each rule over the candidates the previous one left in play, the first to leave exactly one deciding
    (its second rule until PLAN_P, the immediate unfilled imbalance, became IC condition 3):
    1. exclude pinbars (`candle_type` "pinbar") — none left → no IC (`ALL_IC_CANDIDATES_PINBAR`; over Step 1B's
       passers `select_ic` then falls back to all the candidates); one left of two or more → it (rules 2–3 never run:
       no runner-up, ICR-14);
    2. the candle type, maru > normal (`prefer_candle_type(include_pinbar=False)`); runner-up = the most recent normal;
    3. the most recent before the imbalance (`most_recent_before_imbalance`): the reference = the earliest unfilled sd
       imbalance after the OLDEST candidate STILL IN PLAY after rules 1–2 (ICR-8) — its own immediate gap (every
       candidate passed condition 3), so the pick is that oldest candidate or its neighbour sharing the gap, the closer
       one (ICR-1; the user, 2026-10-13); always decides when reached; runner-up = the most recent other candidate in
       play.
    Returns `(pick, ranking)`: `pick` = the IC's check's `pattern` (an `IPResult` with no rank), or None (all
    pinbars); `ranking` = `(rule, runner_up_ic_idx, in_play_count, imbalance_first_c2_idx)` — `rule` one of the
    `RANKING_*` names, `in_play_count` = the candidates entering the deciding rule, `imbalance_first_c2_idx` = the
    reference gap (rule 3), else None — or None when nothing was ranked: exactly ONE candidate entered (Step 1's /
    Step 1A's only IC candidate, or Step 1B's only passer) and it is not a pinbar (ICR-10). Every value an `int` /
    `str` (the mirror shifts only `isinstance(v, int)` — B-m9)."""
    by_idx = {int(c.ic_idx): c for c in ic_candidates}
    assert by_idx, "[poi_zones] the ranking rules need IC candidates"

    # 1. Exclude pinbars.
    in_play = [i for i in by_idx if str(df.loc[i, "candle_type"]) != "pinbar"]
    if not in_play:
        return None, None
    if len(in_play) == 1:
        ranking = None if len(by_idx) == 1 else (RANKING_EXCLUDE_PINBAR, None, len(by_idx), None)
        return by_idx[in_play[0]].pattern, ranking

    # 2. The candle type, maru > normal.
    by_type = prefer_candle_type(in_play, df, include_pinbar=False)
    if len(by_type) == 1:
        losers = [i for i in in_play if i != by_type[0]]
        runner_up = max(prefer_candle_type(losers, df, include_pinbar=False))
        return by_idx[by_type[0]].pattern, (RANKING_IC_CANDLE_TYPE, int(runner_up), len(in_play), None)

    # 3. The most recent before the imbalance after the oldest candidate still in play.
    pick, reference_c2 = most_recent_before_imbalance(by_type, df, fib_state, config)
    runner_up = max(i for i in by_type if i != pick)
    return by_idx[pick].pattern, (RANKING_MOST_RECENT_BEFORE_IMBALANCE, int(runner_up), len(by_type),
                                  int(reference_c2))


@dataclass(frozen=True)
class ICSelection:
    """`select_ic`'s answer for one fib (PLAN_L §3.1 / §4.1): `checks` = Step 1's check of EVERY agreed-opposite
    candle of the fib's window (index order; `touches_zone` in `ic_zone_band`) — `pattern` set on the IC
    candidates, or on every check with `classify_all`; `pick` = the one IC (a pattern's or a ranked one), or None —
    no IC → no POI; `tiebreak` = Step 2's `(rule, runner_up_ic_idx, tied_count, imbalance_first_c2_idx)` when its
    tiebreak rules decided (`apply_tiebreak_rules`), else None; `ip_outcome` = Step 2's outcome (`IP_OUTCOME_PATTERN`
    / `IP_OUTCOME_NO_PATTERN`; None — Step 2 never ran, no IC candidate); `ranking` = Step 3's `(rule,
    runner_up_ic_idx, in_play_count, imbalance_first_c2_idx)` when a ranking rule decided (`apply_ranking_rules`),
    else None; `no_ic_reason` = why there is no IC (`NO_IC_CANDIDATE` / `ALL_IC_CANDIDATES_PINBAR`), None with a pick;
    `ic_zone_band` = the zone IC condition 4 was evaluated in (PLAN_P): `IC_ZONE_PRIMARY` (61.8–80), or
    `IC_ZONE_FALLBACK` (80–100) when Step 1A ran — no check passed every IC condition with 61.8–80 (a no-IC answer
    with the fallback band = Step 1A ran and found no IC either). `ic_candidates` = the checks passing every IC
    condition. `prior_imbalance_outcome` = Step 1B's outcome (PLAN_S step 3: `PRIOR_IMBALANCE_*`; None = no IC
    candidate, 1B never ran) — `tiebreak` / `ip_outcome` / `ranking` / `no_ic_reason` are those of the Steps 2–3 run
    that gave the answer: over the PASSERS when `preferred` (the user, round 7: `ip_outcome` = Step 2's outcome over
    the candidates Steps 2–3 ran on), else over all the candidates; `prior_imbalance_outranked` = when the IC came
    from the passers and the preference narrowed the field, the IC Step 2 would pick over ALL the candidates if that
    IC fails 1B (the pattern the preference outranked — round 7); else None (a fallback never has one).
    The two counts are derived from `checks` (properties). The fields' consistency is asserted at construction."""
    checks: Tuple[ICCandleCheck, ...]
    pick: Optional[IPResult]
    tiebreak: Optional[Tuple[str, int, int, Optional[int]]]
    ip_outcome: Optional[str]
    ranking: Optional[Tuple[str, Optional[int], int, Optional[int]]]
    no_ic_reason: Optional[str]
    ic_zone_band: str
    prior_imbalance_outcome: Optional[str]
    prior_imbalance_outranked: Optional[IPResult]

    def __post_init__(self):
        assert (self.pick is None) == (self.no_ic_reason is not None), (
            f"[poi_zones] pick {self.pick} vs no_ic_reason {self.no_ic_reason!r}")
        candidates = {c.ic_idx: c for c in self.ic_candidates}
        # The IC is always an IC candidate (plan review lens B, PLAN_S step 3: a Steps-2–3 run fed the checks instead of
        # the candidates would pick a non-candidate on the drawn path and crash on MS's — never silently).
        assert self.pick is None or int(self.pick.ic_idx) in candidates, (
            f"[poi_zones] pick {self.pick.ic_idx} is no IC candidate {sorted(candidates)}")
        # Step 1B (PLAN_S step 3): it runs exactly when there are IC candidates, and evaluates every one of them.
        assert self.prior_imbalance_outcome in (None, PRIOR_IMBALANCE_PREFERRED, PRIOR_IMBALANCE_FALLBACK_PASSERS_NO_IC,
                                                PRIOR_IMBALANCE_FALLBACK_NO_PASSER), self.prior_imbalance_outcome
        assert (self.prior_imbalance_outcome is None) == (self.ip_outcome is None), (
            self.prior_imbalance_outcome, self.ip_outcome)
        assert all(c.prior_imbalance_unfilled is not None for c in candidates.values()) or self.ip_outcome is None, (
            "[poi_zones] an IC candidate without its Step 1B test")
        for c in self.checks:
            if c.prior_imbalance_unfilled:
                assert c.prior_imbalance_first_c2_idx is not None and c.prior_imbalance_first_c2_idx < c.ic_idx, c
        passers = {i for i, c in candidates.items() if c.prior_imbalance_unfilled}
        if self.prior_imbalance_outcome == PRIOR_IMBALANCE_PREFERRED:
            assert self.pick is not None and int(self.pick.ic_idx) in passers, (self.pick, sorted(passers))
        elif self.prior_imbalance_outcome == PRIOR_IMBALANCE_FALLBACK_PASSERS_NO_IC:
            # every passer is a no-pattern pinbar, so the fallback's IC (if any) is a non-passer
            assert passers and (self.pick is None or int(self.pick.ic_idx) not in passers), (
                self.pick, sorted(passers))
        elif self.prior_imbalance_outcome == PRIOR_IMBALANCE_FALLBACK_NO_PASSER:
            assert not passers, sorted(passers)
        out = self.prior_imbalance_outranked
        assert out is None or (self.prior_imbalance_outcome == PRIOR_IMBALANCE_PREFERRED and out.rank is not None
                               and int(out.ic_idx) in candidates and int(out.ic_idx) not in passers), (
            f"[poi_zones] outranked pattern {out} vs outcome {self.prior_imbalance_outcome!r}, passers {sorted(passers)}")
        assert self.ip_outcome in (None, IP_OUTCOME_PATTERN, IP_OUTCOME_NO_PATTERN), self.ip_outcome
        assert self.no_ic_reason in (None, NO_IC_CANDIDATE, ALL_IC_CANDIDATES_PINBAR), self.no_ic_reason
        assert self.ic_zone_band in (IC_ZONE_PRIMARY, IC_ZONE_FALLBACK), self.ic_zone_band
        assert self.ic_zone_band == IC_ZONE_PRIMARY or self.checks, "[poi_zones] Step 1A ran without checks"
        assert self.tiebreak is None or self.ip_outcome == IP_OUTCOME_PATTERN, (self.tiebreak, self.ip_outcome)
        assert self.ranking is None or (self.ip_outcome == IP_OUTCOME_NO_PATTERN and self.pick is not None), (
            self.ranking, self.ip_outcome)
        assert (self.no_ic_reason == NO_IC_CANDIDATE) == (self.ip_outcome is None), (
            self.no_ic_reason, self.ip_outcome)
        # (all pinbars ⇒ `no_pattern` follows: no pick rules out `pattern` below, and `None` is `no_ic_candidate`'s)
        assert self.ip_outcome != IP_OUTCOME_PATTERN or (self.pick is not None and self.pick.rank is not None), (
            self.ip_outcome, self.pick)
        assert self.ip_outcome != IP_OUTCOME_NO_PATTERN or self.pick is None or self.pick.rank is None, (
            self.ip_outcome, self.pick)

    @property
    def ic_candidates(self) -> Tuple[ICCandleCheck, ...]:
        return tuple(c for c in self.checks if c.is_ic_candidate)

    @property
    def ic_candidate_count(self) -> Optional[int]:
        """The IC candidates (Step 1's or Step 1A's) — None when 1B never ran (with `prior_imbalance_outcome`)."""
        return None if self.prior_imbalance_outcome is None else len(self.ic_candidates)

    @property
    def prior_imbalance_passer_count(self) -> Optional[int]:
        """The IC candidates passing Step 1B — None when 1B never ran."""
        if self.prior_imbalance_outcome is None:
            return None
        return sum(1 for c in self.ic_candidates if c.prior_imbalance_unfilled)

    @property
    def pick_check(self) -> Optional[ICCandleCheck]:
        """The pick's own check (None without a pick)."""
        if self.pick is None:
            return None
        return next(c for c in self.checks if c.ic_idx == int(self.pick.ic_idx))


def select_ic(
    df: pd.DataFrame,
    fib_state: FibState,
    config: POIConfig,
    *,
    ic_reference: ICReference,
    classify_all: bool = False,
) -> ICSelection:
    """ONE IC per fib, identified in the user's steps (POI_ZONES_SPEC §3; PLAN_L; PLAN_H D2: the drawn POI and MS's
    in-flight read call this same function): Step 1 the IC candidates (`check_ic_conditions` — every IC condition
    together, PLAN_H §6 RX1-F) → none: Step 1A, condition 4 on the 80–100 fallback zone, conditions 1–3 unchanged
    (PLAN_P; `ic_zone_band`; none again → no IC, `NO_IC_CANDIDATE`) → Step 1B the prior-imbalance preference
    (PLAN_S step 3; `check_prior_imbalance`, `prior_imbalance_outcome`): the candidates passing it go through Steps
    2–3 — no IC from them, or no passer → Steps 2–3 on ALL the candidates (1B never adds or removes an IC, it can only
    swap it for a passer; it outranks the pattern) → Step 2 the institutional pattern among the list + its tiebreak
    rules (`select_institutional_pattern`) → no pattern: Step 3 the ranking rules over the same, unfiltered list
    (`apply_ranking_rules`; every one a pinbar → no IC, `ALL_IC_CANDIDATES_PINBAR`) — `_steps_2_and_3`. Step 1A
    runs ONLY when Step 1 finds no candidate — never after Steps 2–3 (an all-pinbar answer stays no IC; the user,
    2026-10-13). A fib neither active nor locked has no checks → `NO_IC_CANDIDATE` (PLAN_L A6).

    The leg cap comes from the FRAME (`df.attrs["timeframe"]` → `config.ip_leg_cap_by_tf`; a KeyError when unstamped or
    on an unknown TF — fail loudly — read BEFORE the active-or-locked guard); the stacked legs classify with
    `CandleParams()` and each stack's body floor is the frame's `pinbar_body_floor` at its LAST candle (PLAN_U §3.3). `classify_all` (the drawn record + the debug CSV) classifies the pattern of EVERY
    checked candle (PLAN_H §6 ICR-11 — never picked unless an IC candidate) and evaluates Step 1B on every check;
    without it only the IC candidates.

    `ic_reference` (REQUIRED — PLAN_K A1; built by `FibTracker.ic_reference` as of the reader's moment): IC condition
    2, evaluated per checked candle at IC selection (RX1-E)."""
    timeframe = df.attrs["timeframe"]
    params = CandleParams()
    cap = config.ip_leg_cap_by_tf[timeframe]
    fib = fib_state.fib
    if fib is None or not (fib_state.active or fib_state.locked):
        return ICSelection((), None, None, ip_outcome=None, ranking=None, no_ic_reason=NO_IC_CANDIDATE,
                           ic_zone_band=IC_ZONE_PRIMARY, prior_imbalance_outcome=None, prior_imbalance_outranked=None)
    sd = int(fib_state.struct_direction)
    agreed = agreed_direction(df["direction"].to_numpy(), df["vol_dir"].to_numpy())
    checks = check_ic_conditions(df, fib_state, config, ic_reference=ic_reference, agreed=agreed)
    band = IC_ZONE_PRIMARY
    if checks and not find_ic_candidates(checks):
        # Step 1A (PLAN_P, the user's change 3): no candidate with the 61.8–80 zone → condition 4 in the 80–100
        # fallback zone, conditions 1–3 unchanged (none depends on the zone — reused from the checks); runs ONLY
        # here, never after Steps 2–3.
        band = IC_ZONE_FALLBACK
        zone_lo, zone_hi = fib_zone_bounds(fib.anchor_high, fib.anchor_low, sd, ic_zone_config(config, band))
        checks = tuple(
            replace(c, touches_zone=ic_touches_zone(float(df.loc[c.ic_idx, "h"]), float(df.loc[c.ic_idx, "l"]),
                                                    zone_lo, zone_hi))
            for c in checks
        )
    ic_candidates = set(find_ic_candidates(checks))          # Step 1's answer
    to_classify = [c.ic_idx for c in checks if classify_all or c.ic_idx in ic_candidates]
    results = {
        int(r.ic_idx): r
        for r in classify_ics(
            df, to_classify, sd,
            bos_anchor_idx=fib_state.bos_idx, cts_anchor_idx=fib_state.cts_idx,
            agreed=agreed, cap=cap, params=params,
        )
    }
    checks = tuple(
        replace(c, pattern=results[c.ic_idx]) if c.ic_idx in results else c
        for c in checks
    )
    # Step 1B's test (PLAN_S step 3) — on the IC candidates, and with `classify_all` on every check (the CSV), as the
    # pattern: MS's read pays only for its candidates.
    checks = tuple(
        check_prior_imbalance(df, c, fib_state, config) if (classify_all or c.ic_idx in ic_candidates) else c
        for c in checks
    )
    candidate_checks = tuple(c for c in checks if c.ic_idx in ic_candidates)
    if not candidate_checks:
        return ICSelection(checks, None, None, ip_outcome=None, ranking=None, no_ic_reason=NO_IC_CANDIDATE,
                           ic_zone_band=band, prior_imbalance_outcome=None, prior_imbalance_outranked=None)
    # Step 1B — the prior-imbalance preference (PLAN_S step 3; the user's rounds 2 / 5 / 7): the passers (any number)
    # go through Steps 2–3; no IC from them (every passer a no-pattern pinbar) -> Steps 2–3 on ALL the candidates of
    # the step before (Step 1's or Step 1A's) = the pick without 1B; no passer -> all the candidates. 1B outranks the
    # institutional pattern, never adds or removes an IC — it can only swap the IC for a passer.
    passers = tuple(c for c in candidate_checks if c.prior_imbalance_unfilled)
    steps = _steps_2_and_3(df, fib_state, config, passers) if passers else None
    outranked = None
    if steps is not None and steps[0] is not None:
        outcome = PRIOR_IMBALANCE_PREFERRED
        if len(passers) < len(candidate_checks):
            # The pattern the preference outranked (round 7): Step 2's pick over ALL the candidates, if it fails 1B.
            best, _tb = select_institutional_pattern(df, fib_state, config, candidate_checks)
            if best is not None and best.ic_idx not in {c.ic_idx for c in passers}:
                outranked = best
    else:
        outcome = PRIOR_IMBALANCE_FALLBACK_PASSERS_NO_IC if passers else PRIOR_IMBALANCE_FALLBACK_NO_PASSER
        steps = _steps_2_and_3(df, fib_state, config, candidate_checks)
    pick, tiebreak, ip_outcome, ranking, no_ic_reason = steps
    return ICSelection(checks, pick, tiebreak, ip_outcome=ip_outcome, ranking=ranking, no_ic_reason=no_ic_reason,
                       ic_zone_band=band, prior_imbalance_outcome=outcome, prior_imbalance_outranked=outranked)


def _steps_2_and_3(
    df: pd.DataFrame, fib_state: FibState, config: POIConfig, ic_candidates: Sequence[ICCandleCheck],
) -> Tuple[Optional[IPResult], Optional[tuple], str, Optional[tuple], Optional[str]]:
    """Steps 2–3 over `ic_candidates` (all of Step 1's / Step 1A's, or Step 1B's passers — `select_ic`): Step 2 the
    institutional pattern + its tiebreak rules; no pattern -> Step 3 the ranking rules over the same list (every one a
    pinbar -> no IC). Returns `(pick, tiebreak, ip_outcome, ranking, no_ic_reason)`."""
    pick, tiebreak = select_institutional_pattern(df, fib_state, config, ic_candidates)     # Step 2
    if pick is not None:
        return pick, tiebreak, IP_OUTCOME_PATTERN, None, None
    pick, ranking = apply_ranking_rules(df, fib_state, config, ic_candidates)               # Step 3
    return pick, None, IP_OUTCOME_NO_PATTERN, ranking, None if pick is not None else ALL_IC_CANDIDATES_PINBAR


def _poi_record_gate(fib_state: FibState) -> bool:
    """The per-record lifecycle gate of the POI records (FIB_LIFECYCLE_SPEC §11): the
    cycle's condition-active version (`active`, its cycle ended or not) OR a locked
    record — `select_ic`'s own validity check.

    A cycle that ENDS before its fib locks (its CTS never confirmed) passes like a
    live one: its IC rows and its POI, the POI ending at its cycle's end like every
    POI (`compute_cycle_lifecycle`) — PLAN_W 9b, the user 2026-10-28: "Full POI
    record". Until then the gate was `active AND end_idx is None` (FibState Session
    2, `310c395`, keeping Week 7's effect: `active` then went False once a cycle
    ended), which gave such a cycle no IC and no POI at all.

    Per-record, NOT the cycle-level `status == "active" OR locked`: `status` is
    shared by all version records of a live cycle, so a status-keyed gate would
    wrongly process superseded (dead) cross versions, which carry per-record
    `active=False` (same reason the chart gate, §9.1, uses raw per-record flags).
    Locked records — ended-locked included — always pass; `disappeared` records
    never reach here (filtered by `get_fibs_for_charting`)."""
    return bool(fib_state.active or fib_state.locked)


@dataclass(frozen=True)
class ICRecordSelection:
    """One POI record (`select_ic_records`): a gated fib record, its cycle's
    CTS-established MOMENT and its IC selection — the ONE
    input of the drawn POIs (`derive_poi_zones`) and of the debug CSV
    (`derive_ic_candidates`)."""
    fib_state: FibState
    cts_established_idx: int
    selection: ICSelection


def select_ic_records(
    df: pd.DataFrame,
    structure_events: List[StructureEvent],
    fib_tracker: Optional[FibTracker],
    config: POIConfig,
    *,
    lifecycle_cap: Optional[int] = None,
) -> List[ICRecordSelection]:
    """The POI records in `get_fibs_for_charting` order, each with its selection
    (PLAN_H §8.5: computed ONCE per pipeline run, handed to `derive_poi_zones` and
    `derive_ic_candidates`). Per record, in this order:

    1. the lifecycle gate (`_poi_record_gate`);
    2. the cycle's CTS_ESTABLISHED: its MOMENT (`meta["confirmed_at"]`, the
       canonical cycle lifecycle-start — `structure_lifecycle.compute_cycle_lifecycle`)
       is the activation floor's cycle term and the exported
       `cts_established_idx`, NOT the CTS anchor (Plan D, 2026-09-23). Direct
       index, no `.get` fallback: a missing moment fails loudly. A cycle with no
       CTS_ESTABLISHED gets NO record (user decision 2026-09-29, zones-audit
       "fallback POI" option N: its POIs could never activate; kept by the user
       2026-10-28, PLAN_W 9b) — reached only by a pre-created cross fib whose
       cycle never established, live or ended. A LOCKED fib there
       breaks the event contract (a fib locks at its cycle's CTS_CONFIRMED, after
       the establishment): AssertionError — checked BEFORE the selection, so it
       fires whatever the pattern;
    3. `select_ic(..., classify_all=True)` — the CSV's checks, every one
       classified; the pick reads only the IC candidates (the same pick as
       without `classify_all` — MS's read) — with the record's FINAL IC reference, as of the structure's end
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
    out: List[ICRecordSelection] = []
    for fib_state in fib_tracker.get_fibs_for_charting():
        if not _poi_record_gate(fib_state):
            continue
        key = (int(fib_state.structure_id), int(fib_state.cycle_id))
        cts_event = cts_established_by_key.get(key)
        if cts_event is None:
            assert not fib_state.locked, (
                f"[poi_zones] locked fib {key} has no CTS_ESTABLISHED (event contract)")
            continue
        out.append(ICRecordSelection(
            fib_state=fib_state,
            cts_established_idx=int(cts_event.meta["confirmed_at"]),
            selection=select_ic(
                df, fib_state, config, classify_all=True,
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
    ic_records: Optional[List[ICRecordSelection]] = None,
) -> List[POIZone]:
    """
    Derive the POI zones: at most ONE per fib record — the IC picked by
    `select_ic` (a pattern's IC or, with no pattern, a ranked one — PLAN_L L2;
    no IC -> no POI, PLAN_H D7) — with its activation history.

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
    ic_records : list of ICRecordSelection, optional
        The records with their selections (`select_ic_records`), computed once
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

    # Precompute each imbalance instance's (armed_idx, confirmed_fill_idx) — the
    # two-stroke fill (`_compute_fill_idx_cache`). One linear scan per
    # instance — shared across every POI's activation sweep.
    imbalances: List[ImbalanceInstance] = df.attrs.get("imbalances", [])
    fill_idx_cache = _compute_fill_idx_cache(df, imbalances, config.fill_threshold)

    if ic_records is None:
        ic_records = select_ic_records(df, structure_events, fib_tracker, config, lifecycle_cap=lifecycle_cap)

    # One record = one fib record that passed the gate and has a CTS_ESTABLISHED
    # (`select_ic_records`); its pick — a pattern's IC or a ranked one — or None
    # (no IC: no IC candidate / every candidate a pinbar → no POI, PLAN_H D7).
    for rec in ic_records:
        fib_state = rec.fib_state
        pick = rec.selection.pick
        if pick is None:
            continue
        # A ranked IC (Step 3, PLAN_L L2) has no pattern: no pattern term in its
        # activation floor (ICR-2) and its pattern meta ALL None — `classify_ics`
        # leaves a no-pattern candidate its first failing variant's counts, so
        # they are nulled here explicitly (PLAN_L A1 / B-M3).
        ranked = rec.selection.ip_outcome == IP_OUTCOME_NO_PATTERN
        assert (pick.pattern_confirmed_idx is None) == ranked, (
            f"[poi_zones] IC {pick.ic_idx}: pattern_confirmed_idx {pick.pattern_confirmed_idx} "
            f"vs ip_outcome {rec.selection.ip_outcome!r}")

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
        assert ic_time is not None, f"[poi_zones] IC pick {ic_idx} is not a row of the frame"
        ic_high = float(df.loc[ic_idx, "h"])
        ic_low = float(df.loc[ic_idx, "l"])

        # Per-POI activation lifecycle. POI is "active" at candle t iff
        # ALL these condition-state requirements hold simultaneously
        # (re-evaluated dynamically against the fib's time-varying state):
        #   1. fib's `cts_at(t)` has extended to >= ic_idx — i.e., IC lies
        #      within the fib's bounds at t. The fib's cts only grows, so
        #      becomes monotonic once reached.
        #   2. ic_idx <= t  (the IC candle exists in data — implied by t
        #      starting at first_active = max(cts_established_idx, ic_idx,
        #      lifecycle floor[, pattern_confirmed_idx — a pattern IC only])).
        #   3. The POI's OWN imbalance condition — the cycle still has an
        #      unfilled sd imbalance after the IC: `has_unfilled_imbalance(df,
        #      ic_idx+1, t, check_to_idx=t, direction=sd, evaluated_at=t)`,
        #      counting only gaps FORMED by t (Plan F: an instance exists from
        #      its first c3, `inst.formed_at`), decoupled from Fib's `active`
        #      flag. DECOUPLED from IC identification since PLAN_P (IC
        #      condition 3 is the IMMEDIATE gap; this stays ANY gap after the
        #      IC — the user, 2026-10-13: "keeping a POI zone active centers
        #      more around that we currently still have unfilled imbalances
        #      in our current cycle").
        #   5. The touch: the IC candle touches the AS-OF zone of the fib's
        #      bounds [bos_price, cts_price_at(t)] in the band the IC was
        #      identified in — 61.8-80, or 80-100 for a Step 1A IC (PLAN_P,
        #      `ic_zone_config`) — via `fib_zone_bounds` + `ic_touches_zone`,
        #      the selection's own helpers (PLAN_H D1a / R3).
        # (The IC reference condition — IC condition 2 in POI_ZONES_SPEC §3.1's
        # numbering, §3.2, PLAN_K; this list skips its #4 slot — is evaluated at
        # IC SELECTION only (`select_ic`), never in
        # this sweep (PLAN_H RX1-E): computed + exported since PLAN_K landing A,
        # the pick filtered by it since landing B.)
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
            # Condition 5's touch in the band the IC was identified in (PLAN_P).
            config=ic_zone_config(config, rec.selection.ic_zone_band),
            imbalances=imbalances,
            fill_idx_cache=fill_idx_cache,
            lifecycle_floor_idx=struct_start_by_sid.get(int(sid)),
            pattern_confirmed_idx=None if ranked else int(pick.pattern_confirmed_idx),
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
        rk = rec.selection.ranking or (None, None, None, None)
        outranked = rec.selection.prior_imbalance_outranked
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
                # Step 2's outcome, the zone the IC was identified in (PLAN_P),
                # the institutional pattern of the IC (PLAN_H D6 / HV; all None
                # on a ranked IC) and its tiebreak, then Step 3's ranking
                # (PLAN_L L2): flat str / int / None — the index keys are in
                # the mirror's `_ZONE_META_IDX_KEYS`, `ip_tied_count` /
                # `ranking_in_play_count` are counts.
                "ip_outcome": rec.selection.ip_outcome,
                "ic_zone_band": rec.selection.ic_zone_band,
                "ip_rank": None if ranked else pick.ip_rank,
                "ip_leg_candle_counts": None if ranked else pick.ip_leg_candle_counts,
                "pattern_anchor_idx": None if ranked else int(pick.pattern_anchor_idx),
                "opp_leg_first_idx": None if ranked else int(pick.opp_leg_first_idx),
                "pattern_end_idx": None if ranked else int(pick.pattern_end_idx),
                "pattern_confirmed_idx": None if ranked else int(pick.pattern_confirmed_idx),
                "ip_tiebreak": tb[0],
                "ip_tb_runner_up_ic_idx": tb[1],
                "ip_tied_count": tb[2],
                "ip_tb_imbalance_first_c2_idx": tb[3],
                "ranking_rule": rk[0],
                "ranking_runner_up_ic_idx": rk[1],
                "ranking_in_play_count": rk[2],
                "ranking_imbalance_first_c2_idx": rk[3],
                # Step 1B, the prior-imbalance preference (PLAN_S step 3; the user's names, rounds 6–7): its outcome,
                # the IC's own closest prior gap, the two counts (the hover's "<passers> of <candidates>") and the
                # pattern the preference outranked (None unless it did) — `prior_imbalance_first_c2_idx` /
                # `prior_imbalance_outranked_ic_idx` are in the mirror's `_ZONE_META_IDX_KEYS`.
                "prior_imbalance_outcome": rec.selection.prior_imbalance_outcome,
                "prior_imbalance_first_c2_idx": rec.selection.pick_check.prior_imbalance_first_c2_idx,
                "prior_imbalance_passer_count": rec.selection.prior_imbalance_passer_count,
                "ic_candidate_count": rec.selection.ic_candidate_count,
                "prior_imbalance_outranked_ic_idx": None if outranked is None else int(outranked.ic_idx),
                "prior_imbalance_outranked_ip_rank": None if outranked is None else outranked.ip_rank,
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
class ICCandidateRecord:
    """One row of the IC-selection debug export (`*_ic_candidates.csv`,
    PLAN_H §8.3; the picked rows are the drawn POIs since landing B): a Step 1
    check of a POI record (PLAN_L L1), or the record's no-IC row (`ic_idx` None,
    `reason` = `no_ic_candidate` / `all_ic_candidates_pinbar`) when it has no IC.
    Every index field is a dataclass field (the sub mirror shifts
    `entity_df_mutation._IC_CANDIDATE_IDX_FIELDS`); `meta` = the attribution
    (`structure_path_id`; a sub's mirror adds `sub_id` and the rest).
    `ip_outcome` sits on every row of the record (Step 2's outcome, PLAN_L L2),
    and so does `ic_zone_band` (PLAN_P — `touches_zone` is the touch of THAT
    band: 80–100 on every row when Step 1A ran);
    the tiebreak and the ranking fields on the picked row. A ranked IC's picked
    row keeps its pattern DIAGNOSTICS (`reason` = the first failing pattern
    rule, the counts, the stacks) like every failing row (PLAN_H §6 ICR-15)."""
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
    immediate_imbalance_first_c2_idx: Optional[int] = None  # IC condition 3 since PLAN_P (None = fails / the no-IC row)
    has_unfilled_imbalance_after: Optional[bool] = None    # the RETIRED condition 3 — information only, PLAN_P
    is_ic_candidate: Optional[bool] = None                 # every IC condition passes (None on the no-IC row)
    ip_outcome: Optional[str] = None                       # Step 2's outcome — every row of the record (PLAN_L L2)
    ranking_rule: Optional[str] = None                     # Step 3 — the picked row of a ranked IC
    ranking_runner_up_ic_idx: Optional[int] = None
    ranking_in_play_count: Optional[int] = None
    ranking_imbalance_first_c2_idx: Optional[int] = None
    ic_zone_band: Optional[str] = None                     # the selection's band, every row of the record (PLAN_P)
    # Step 1B, the prior-imbalance preference (PLAN_S step 3): the check's test on every check row (empty on the no-IC
    # row); the outcome and the two counts on every row of the record (the no-IC row too — like `ip_outcome`)
    prior_imbalance_first_c2_idx: Optional[int] = None
    prior_imbalance_unfilled: Optional[bool] = None
    prior_imbalance_outcome: Optional[str] = None
    prior_imbalance_passer_count: Optional[int] = None
    ic_candidate_count: Optional[int] = None
    detail: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __deepcopy__(self, memo):
        """Shared, never copied: these records sit in `df.attrs["ic_candidates"]`,
        which pandas deep-copies on every `__finalize__` (the
        `ImbalanceInstance.__deepcopy__` reasoning). Nothing mutates one in place
        — the mirror builds shifted copies with `dataclasses.replace`."""
        return self


def derive_ic_candidates(
    df: pd.DataFrame,
    structure_events: List[StructureEvent],
    fib_tracker: Optional[FibTracker],
    config: POIConfig,
    *,
    structure_path_id: str,
    ic_records: Optional[List[ICRecordSelection]] = None,
    lifecycle_cap: Optional[int] = None,
) -> List[ICCandidateRecord]:
    """The institutional-pattern selection of every POI record (`select_ic_records`
    — the records and selections `derive_poi_zones` draws from) as debug records
    (PLAN_H §8.3). One record per Step 1 check — every agreed-opposite candle of
    the fib's window, each with its condition outcomes and its pattern (PLAN_H §6
    ICR-4 / ICR-11; `select_ic(..., classify_all=True)`) — + one no-IC record per
    fib record without a pick, `reason` = `ICSelection.no_ic_reason`:
    `no_ic_candidate` (no candle passes every IC condition in either zone —
    `ICCandleCheck.is_ic_candidate`; never attributed to one condition, PLAN_H §6
    RX1-N) or `all_ic_candidates_pinbar` (no pattern, and ranking rule 1 removed
    every IC candidate — PLAN_L L2). `ip_outcome` and `ic_zone_band` (PLAN_P) on
    every record of a fib; the tiebreak / ranking fields on the picked one.
    `ic_records` None = selected here.
    """
    if ic_records is None:
        ic_records = select_ic_records(df, structure_events, fib_tracker, config, lifecycle_cap=lifecycle_cap)
    out: List[ICCandidateRecord] = []
    for rec in ic_records:
        fib_state, sel = rec.fib_state, rec.selection
        sid, cycle_id = int(fib_state.structure_id), int(fib_state.cycle_id)
        base = (sid, cycle_id, int(fib_state.struct_direction), int(fib_state.bos_idx), int(fib_state.cts_idx))
        prior = dict(prior_imbalance_outcome=sel.prior_imbalance_outcome,
                     prior_imbalance_passer_count=sel.prior_imbalance_passer_count,
                     ic_candidate_count=sel.ic_candidate_count)
        for c in sel.checks:
            r = c.pattern
            assert r is not None, f"[poi_zones] check {c.ic_idx} of fib {base} is unclassified (classify_all)"
            assert c.prior_imbalance_unfilled is not None, (
                f"[poi_zones] check {c.ic_idx} of fib {base} has no Step 1B test (classify_all)")
            picked = sel.pick is not None and r.ic_idx == sel.pick.ic_idx
            tb = sel.tiebreak if picked and sel.tiebreak is not None else (None, None, None, None)
            rk = sel.ranking if picked and sel.ranking is not None else (None, None, None, None)
            out.append(ICCandidateRecord(
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
                immediate_imbalance_first_c2_idx=c.immediate_imbalance_first_c2_idx,
                has_unfilled_imbalance_after=bool(c.has_unfilled_imbalance_after),
                is_ic_candidate=bool(c.is_ic_candidate),
                ip_outcome=sel.ip_outcome,
                ranking_rule=rk[0],
                ranking_runner_up_ic_idx=rk[1],
                ranking_in_play_count=rk[2],
                ranking_imbalance_first_c2_idx=rk[3],
                ic_zone_band=sel.ic_zone_band,
                prior_imbalance_first_c2_idx=c.prior_imbalance_first_c2_idx,
                prior_imbalance_unfilled=bool(c.prior_imbalance_unfilled),
                **prior,
                detail=dict(r.detail),
                meta={"structure_path_id": structure_path_id},
            ))
        if sel.pick is None:
            out.append(ICCandidateRecord(
                *base,
                ic_idx=None, candle_type=None, agreed_dir=None, touches_zone=None, ip_rank=None,
                # No IC: no candle passing every IC condition (never attributed to one condition — they filter
                # together, PLAN_H §6 RX1-F / RX1-N) or every IC candidate a pinbar (ranking rule 1, PLAN_L L2)
                reason=sel.no_ic_reason,
                pattern_anchor_idx=None, opp_leg_first_idx=None, pattern_end_idx=None,
                pattern_confirmed_idx=None, ip_leg_candle_counts=None, picked=False,
                ip_outcome=sel.ip_outcome,
                ic_zone_band=sel.ic_zone_band,
                **prior,
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
    ``confirmed_fill_idx`` — first candle at idx ``> armed_idx`` (a separate,
    later candle — PLAN_S step 2) where the close is at or past the gap outer
    in the instance's direction (stroke 2). Either or both can be ``None``
    (never armed / armed but never confirmed); when both are set,
    ``confirmed_fill_idx > armed_idx``.

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

    ``df`` must be indexed 0..n−1 (``require_positional_index``; ValueError
    otherwise — PLAN_W 9a).
    """
    n = require_positional_index(df, "poi_zones._compute_fill_idx_cache")
    cache: Dict[int, Tuple[Optional[int], Optional[int]]] = {}
    if not imbalances or n == 0:
        return cache

    # Numpy positional indexing is ~35x faster than pandas .loc per element
    # (microbench: 0.02ms vs 0.70ms for a 3000-candle scan with 60 starts).
    # Label == position (checked above), so a hit's position IS its idx.
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

    for inst in imbalances:
        # Degenerate gap → treat as already filled (is_filled returns True
        # unconditionally). Represent as (None, None) and exclude from the
        # relevant set in the sweep.
        if inst.gap_size <= 0:
            cache[id(inst)] = (None, None)
            inst.meta["armed_idx"] = None
            inst.meta["confirmed_fill_idx"] = None
            continue

        start_pos = inst.end_idx + 1
        if not 0 <= start_pos < n:
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
                armed_idx = armed_pos
                # Stroke 2 scan starts AFTER armed_pos — the arming candle's own
                # close never counts (PLAN_S step 2, separate candles).
                slice_c = c_arr[armed_pos + 1:]
                stroke2_hits = np.where(slice_c >= stroke2_level)[0]
                if stroke2_hits.size:
                    confirmed_pos = armed_pos + 1 + int(stroke2_hits[0])
                    confirmed_fill_idx = confirmed_pos
        elif inst.direction == -1:
            stroke1_level = inst.gap_bottom + inst.gap_size * fill_threshold
            stroke2_level = inst.gap_bottom
            slice_h = h_arr[start_pos:]
            stroke1_hits = np.where(slice_h >= stroke1_level)[0]
            if stroke1_hits.size:
                armed_pos = start_pos + int(stroke1_hits[0])
                armed_idx = armed_pos
                slice_c = c_arr[armed_pos + 1:]
                stroke2_hits = np.where(slice_c <= stroke2_level)[0]
                if stroke2_hits.size:
                    confirmed_pos = armed_pos + 1 + int(stroke2_hits[0])
                    confirmed_fill_idx = confirmed_pos

        cache[id(inst)] = (armed_idx, confirmed_fill_idx)
        inst.meta["armed_idx"] = armed_idx
        inst.meta["confirmed_fill_idx"] = confirmed_fill_idx

    return cache


def poi_activation_floor_idx(
    cts_established_idx: int,
    ic_idx: int,
    *,
    pattern_confirmed_idx: Optional[int],
    lifecycle_floor_idx: Optional[int],
) -> int:
    """The earliest candle a POI can be active (POI_ZONES_SPEC §4 "Activation floor"): max(the cycle's established
    moment, the IC candle, the pattern's confirmation moment, the structure lifecycle-start floor). A POI cannot
    activate before its cycle is knowable (Plan D), before its institutional pattern is known complete (PLAN_H R1 —
    may lie past the frame: then never active; a ranked IC has no pattern term, `None` — ICR-2), nor before its
    structure is alive (post-reversal cycle-0 / sub case, Phase 3 Commit 2). Both optional terms keyword-required, no
    default (PLAN_L A7). Two callers: the activation sweep (`_compute_poi_activation_history`, every term) and
    `zone_proximity.poi_unclamped_activation_floor_idx` (`lifecycle_floor_idx=None` — PLAN_Q: the floor in the POI's
    own structure's timeline, which the proximity scan asserts is at or before the cycle's scan start)."""
    floor = max(int(cts_established_idx), int(ic_idx))
    if pattern_confirmed_idx is not None:
        floor = max(floor, int(pattern_confirmed_idx))
    if lifecycle_floor_idx is not None:
        floor = max(floor, int(lifecycle_floor_idx))
    return floor


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
    pattern_confirmed_idx: Optional[int],
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
                    (PLAN_H R1) — None for a RANKED IC (no pattern: no pattern
                    term, PLAN_H §6 ICR-2); keyword-required, no default, so an
                    IP caller cannot drop the term silently (PLAN_L A7) —
                    POI_ZONES_SPEC.md §4 "Activation floor").
      Condition 3 — the POI's OWN imbalance condition (not IC condition 3 —
                    decoupled from IC identification since PLAN_P, which made
                    that the IMMEDIATE gap): an sd-direction imbalance FORMED
                    by t (its first c3 has closed: `inst.formed_at <= t` —
                    Plan F) overlaps `(ic_idx, t]` and is not yet
                    committed-filled (two-stroke: stroke 1 = the fill_threshold retrace,
                    stroke 2 = a LATER candle's close at / past the gap outer).
                    Flips as imbalances
                    form / commit-fill.
      Condition 5 — The touch: the IC candle touches the zone (inclusive
                    edges) of the fib AS OF t in `config`'s band — the caller
                    passes the band the IC was identified in, 61.8-80 or the
                    80-100 fallback (`ic_zone_config`, PLAN_P) — from
                    `bos_price` and the time-varying `cts_price_at(t)`
                    (`fib_zone_bounds` + `ic_touches_zone`, the selection's
                    helpers; PLAN_H D1a). Deactivation reason
                    `ic_not_touching_fib_zone` (R3).

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
    first_active = poi_activation_floor_idx(
        cts_established_idx, ic_idx,
        pattern_confirmed_idx=pattern_confirmed_idx, lifecycle_floor_idx=lifecycle_floor_idx,
    )
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

class ICSelectionError(RuntimeError):
    """An error inside `select_ic` during the MS in-flight read, re-raised so
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
    """MarketStructure's in-flight POI inner (PLAN_I S5 / D5, Q7): the IC pick (`select_ic` — the SAME function as
    the drawn POI, PLAN_H D2: a pattern's IC, or with no pattern the ranked one — PLAN_L L2, ICR-12) on `fib_state`,
    the cycle's `FibTracker.representative_fib` of MS's embedded tracker — one fib decision for the drawn POI and MS.
    Returns the pick's inner (a buy POI's top = the IC's high, a sell POI's bottom = its low), or None — no fib,
    or no IC (no IC candidate / every one a pinbar, D7): the proximity check then reads the BOS inner only. Wired by
    `structure_engine` (MS imports nothing from `zones/`).

    Fails loudly (PLAN_H §8.8 PR1, LANDMINES "The POI Resolver Fails Loudly"; PLAN_I §9.2 F11): the frame's
    candle stamp `df.attrs["timeframe"]` is read FIRST on every call — a refresh with no fib included — and any
    error of the stamp read or the pick is re-raised as `ICSelectionError` (a RuntimeError), so neither a missing
    stamp nor an IP bug can pass as "no IC", and the sub geometry's narrow `except (ValueError,
    IndexError)` (`entity_df_mutation._build_geometry`) cannot drop the sub. `fill_threshold` = the tracker's
    (`IMBALANCE_FILL_THRESHOLD`; the drawn POI's `POIConfig` default is the same constant — pinned).

    The fib's IC reference is built HERE, inside the re-raising `try` (PLAN_K A4, §9 C01): `tracker.ic_reference(
    fib_state, df, evaluated_at=evaluated_at)` — MS's embedded tracker at the fed event's moment, the frame = MS's
    resolver frame (already cut at the run's data edge) — so an error there surfaces as `ICSelectionError` and never
    reaches the sub geometry's narrow `except (ValueError, IndexError)`. A None fib returns before the call."""
    try:
        df.attrs["timeframe"]
        if fib_state is None:
            return None
        ref = tracker.ic_reference(fib_state, df, evaluated_at=int(evaluated_at))
        sel = select_ic(df, fib_state, POIConfig(fill_threshold=fill_threshold), ic_reference=ref)
    except Exception as exc:
        where = ("no fib" if fib_state is None else
                 f"sid {fib_state.structure_id}, cycle {fib_state.cycle_id}, fib {fib_state.bos_idx}..{fib_state.cts_idx}")
        raise ICSelectionError(f"[poi_zones] the MS in-flight IP read failed ({where}): {exc!r}") from exc
    if sel.pick is None:
        return None
    ic_idx = int(sel.pick.ic_idx)
    return float(df.loc[ic_idx, "h"]) if int(fib_state.struct_direction) == 1 else float(df.loc[ic_idx, "l"])
