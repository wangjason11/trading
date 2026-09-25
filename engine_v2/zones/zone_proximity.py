# engine_v2/zones/zone_proximity.py
"""
Zone proximity trigger detection.

Per (sid, cycle_id) with a CTS_CONFIRMED event, walk the scan window and
log each candle that comes within `proximity_pips` of the relevant zone
inner bound. Triggers alternate: the first must be sd-direction (BOS/POI),
then opp_sd (CTS zone), then sd, etc. — capturing each leg of a V/lambda
movement within a structure cycle.

Replaces the old `check_proximity_activation` in `wvmi.py`. The single
"activated cycles" gate is no longer the unit of interest — instead this
returns the full alternating list of trigger candles per cycle.

Narrow-gap cycle restrictions (Rules 2 & 3)
-------------------------------------------
A cycle is "narrow" at a given candle when
`|cts_threshold − bos_threshold|` (reconstructed from the cycle's
`BOS_CONFIRMED` + `CTS_CONFIRMED` + `*_THRESHOLD_UPDATED` events,
evaluated at the start of that candle — events with `idx < current` are
applied; events at `idx == current` are NOT, so the candle's own wick
can't self-rescue) is below the per-TF threshold in
`DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS`. Reading from events (not
df columns) is mandatory because a scan window can extend past a
reversal into the next structure's rows, where the df threshold cols
are nulled / overwritten. When a cycle is narrow:

- **Rule 2:** triggers may only fire on/after the cycle's
  `CTS_CONFIRMED` candle, AND `CTS_CONFIRMED.meta["confirmation_method"]`
  must be `"pullback"`. If no pullback-confirmed CTS exists for this
  cycle, the scan returns no triggers. (Rule 1, enforced upstream in
  `structure/market_structure.py`, blocks sd-proximity from confirming
  CTS in narrow cycles, so the only way a narrow cycle gets a
  `CTS_CONFIRMED` is via pullback.)
- **Rule 3:** while narrow, the cycle gets at most one sd trigger and
  at most one opp_sd trigger. Alternation rules still apply (first sd,
  then opp_sd, then sd, …) but the caps short-circuit the scan after
  the second narrow-mode hit.

The gap is monotonically non-decreasing within a cycle (BOS extends in
struct direction via probe; CTS extends via range sync — both widen the
gap, neither shrinks it). So a cycle can transition narrow → wide
exactly once. After crossing, the caps lift and the existing default
alternation resumes for the rest of the cycle (Q1=A).

Per-TF threshold values are codified now even though M15/M5 paths are
not exercised during the current main-structure debug pivot
(`lower_timeframes=()`); when lower-TF paths re-enable later, the
thresholds are already wired.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd

from engine_v2.common.types import KLZone
from engine_v2.structure import event_fields as ef
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.zones.poi_lifecycle import poi_active_as_of
from engine_v2.zones.poi_zones import POIZone


# ---------------------------------------------------------------------------
# Default thresholds per timeframe
# ---------------------------------------------------------------------------
# Proximity threshold in pips. Caller may override per-call. Same values
# are used regardless of pair (pip_size handles pair scaling separately).
DEFAULT_PROXIMITY_PIPS: Dict[str, int] = {
    "H1": 8,
    "M15": 5,
    "M5": 3,
}

# Probe reset threshold in pips — used by the BOS_0 / Exception 2 reach-back
# probe to decide whether price has returned to the prior zone. Per Part 4
# spec §4.4, must be strictly less than DEFAULT_PROXIMITY_PIPS on the same
# TF so probe-reset and proximity-trigger semantics never overlap.
DEFAULT_PROBE_RESET_PIPS: Dict[str, float] = {
    "H1": 4,
    "M15": 3,
    "M5": 2,
}

# Probe reset WICK cap in pips — condition 2 of the unified probe's two-
# condition reset (Phase 1 design 2026-05-29). The candidate retrace candle
# must have its toward-zone wick (the side opposite probe_sd, body-bounded
# with body_top = max(o,c) / body_bottom = min(o,c)) no longer than this
# threshold. Rejects single-candle stab wicks that pass the within-X-pips
# proximity check (condition 1) but represent transient spikes rather than
# structural retraces. Must be strictly greater than DEFAULT_PROBE_RESET_PIPS
# on the same TF (a wick cap smaller than the proximity tolerance is
# self-contradictory — condition 1 would be unreachable).
DEFAULT_PROBE_RESET_WICK: Dict[str, int] = {
    "H1": 16,
    "M15": 12,
    "M5": 8,
}

# Narrow-cycle threshold in pips — minimum |cts_threshold − bos_threshold|
# required to allow unrestricted proximity-trigger generation within a
# cycle. When the gap is below this, Rules 1/2/3 apply:
#   - Rule 1 (structure/market_structure.py): sd-proximity cannot confirm CTS.
#   - Rule 2 (this file): scan only fires triggers on/after a pullback-confirmed CTS.
#   - Rule 3 (this file): at most 1 sd + 1 opp_sd trigger while narrow.
# Must be strictly greater than DEFAULT_PROXIMITY_PIPS on the same TF
# (otherwise the proximity buffer can overlap CTS even in "wide" cycles).
DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS: Dict[str, int] = {
    "H1": 50,
    "M15": 30,
    "M5": 15,
}

for _tf in DEFAULT_PROBE_RESET_PIPS:
    assert DEFAULT_PROXIMITY_PIPS[_tf] > DEFAULT_PROBE_RESET_PIPS[_tf], (
        f"{_tf}: probe reset {DEFAULT_PROBE_RESET_PIPS[_tf]} pips must be "
        f"strictly less than proximity trigger {DEFAULT_PROXIMITY_PIPS[_tf]} pips"
    )
for _tf in DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS:
    assert DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS[_tf] > DEFAULT_PROXIMITY_PIPS[_tf], (
        f"{_tf}: narrow-gap threshold "
        f"{DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS[_tf]} pips must be "
        f"strictly greater than proximity trigger {DEFAULT_PROXIMITY_PIPS[_tf]} pips"
    )
for _tf in DEFAULT_PROBE_RESET_WICK:
    assert DEFAULT_PROBE_RESET_WICK[_tf] > DEFAULT_PROBE_RESET_PIPS[_tf], (
        f"{_tf}: probe reset wick cap {DEFAULT_PROBE_RESET_WICK[_tf]} pips "
        f"must be strictly greater than probe reset proximity "
        f"{DEFAULT_PROBE_RESET_PIPS[_tf]} pips (a wick cap smaller than the "
        f"proximity tolerance makes condition 1 unreachable)"
    )
del _tf


@dataclass(frozen=True)
class ZoneProximityTrigger:
    """One trigger candle within a cycle's scan window.

    Direction "sd" = price came within threshold of an sd-direction zone
    (BOS or POI). "opp_sd" = price came within threshold of the CTS zone
    (the only opposite-direction zone, by definition).
    """
    structure_id: int
    cycle_id: int
    direction: Literal["sd", "opp_sd"]
    idx: int                                  # the trigger candle index
    trigger_inner: float                       # the inner price used as trigger level
    zone_kind: Literal["BOS", "CTS", "POI"]   # which zone matched
    proximity_pips: int
    pip_size: float
    timeframe: str
    meta: Dict[str, Any] = field(default_factory=dict)


def _find_kl_zone_for_cycle(
    kl_zones: List[KLZone],
    sid: int,
    cycle_id: int,
    source_kind: Literal["BOS", "CTS"],
) -> Optional[KLZone]:
    """Find the KL zone for (sid, cycle_id) of a specific source_kind."""
    for z in kl_zones:
        if (z.meta.get("structure_id") == sid
                and z.meta.get("cycle_id") == cycle_id
                and z.source_kind == source_kind):
            return z
    return None


def _build_cycle_threshold_timeline(
    sorted_events: List[StructureEvent],
    sid: int,
    cycle_id: int,
) -> List[StructureEvent]:
    """Return the threshold-defining events for `(sid, cycle_id)` sorted
    by `(moment, type)` (`ef.event_moment`; Plan E E3g-2 — a BOS_CONFIRMED's
    initial threshold is known at its `confirmed_at`, not its anchor). Used to
    drive the running cts/bos thresholds during the scan.

    Includes:
      - `BOS_CONFIRMED` (initial bos_threshold)
      - `CTS_CONFIRMED` (initial cts_threshold)
      - `BOS_THRESHOLD_UPDATED` / `CTS_THRESHOLD_UPDATED` (subsequent
        widening events)

    Reading from events (not df columns) is mandatory: df-level
    `cts_threshold`/`bos_threshold` columns are OVERWRITTEN by the next
    structure once it starts processing (see LANDMINES "DataFrame
    Column Overwrite Hazard"). A scan window can straddle a reversal —
    e.g. sid=0 cycle=2's window extends up to reversal apply_idx − 1,
    by which time sid=1's processing has already nulled out the
    threshold cols for the post-reversal rows.
    """
    relevant_types = (
        "BOS_CONFIRMED",
        "CTS_CONFIRMED",
        "BOS_THRESHOLD_UPDATED",
        "CTS_THRESHOLD_UPDATED",
    )
    out = [
        ev for ev in sorted_events
        if ev.type in relevant_types
        and ev.meta.get("structure_id") == sid
        and ev.meta.get("cycle_id") == cycle_id
    ]
    out.sort(key=lambda e: (ef.event_moment(e), e.type))  # the time order (Plan E E3g-2)
    return out


def _apply_threshold_event(
    ev: StructureEvent,
    running_cts: Optional[float],
    running_bos: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    """Fold a single threshold event into the running cts/bos values.
    Returns new (running_cts, running_bos). Pure."""
    if ev.type in ("CTS_CONFIRMED", "CTS_THRESHOLD_UPDATED"):
        if ev.price is not None:
            running_cts = float(ev.price)
    elif ev.type in ("BOS_CONFIRMED", "BOS_THRESHOLD_UPDATED"):
        if ev.price is not None:
            running_bos = float(ev.price)
    return running_cts, running_bos


def check_zone_proximity(
    df: pd.DataFrame,
    sorted_events: List[StructureEvent],
    kl_zones: List[KLZone],
    poi_zones: List[POIZone],
    pip_size: float,
    timeframe: str,
    proximity_pips: Optional[int] = None,
    min_gap_pips: Optional[int] = None,
) -> Dict[Tuple[int, int], List[ZoneProximityTrigger]]:
    """Find zone proximity trigger candles per (sid, cycle_id).

    Per cycle, walk the scan window and log alternating sd/opp_sd triggers:
    1. First trigger must be sd (price within threshold of BOS or active POI)
    2. After an sd trigger, only opp_sd is eligible (CTS zone)
    3. After an opp_sd trigger, only sd again — alternating

    Each "first match wins" applies per slot (i.e., once an sd trigger is
    logged, no more sd triggers fire until an opp_sd has been logged).

    Narrow-gap restrictions (Rules 2 & 3, per module docstring): at each
    candle, evaluate `|cts_threshold − bos_threshold|` reconstructed
    from the cycle's threshold-defining events (events with `idx <
    current_candle` applied; events at `idx == current_candle` NOT
    applied — no self-rescue). While < `min_gap_pips`: require a
    pullback-confirmed CTS_CONFIRMED to have already fired for this
    cycle, and cap at most 1 sd + 1 opp_sd. After a wide-mode crossing
    (gap ≥ `min_gap_pips`), caps lift and the existing alternation
    continues unchanged.

    Scan window per (sid, cycle_id):
      start = CTS_CONFIRMED.meta["confirmed_at"] (the confirmation candle)
      end   = min(next_BOS_CONFIRMED.confirmed_at - 1,
                  REVERSAL_CANDIDATE.apply_idx - 1,
                  end_of_df)

    Parameters
    ----------
    df : DataFrame
        OHLC data with structure columns.
    sorted_events : list of StructureEvent
        All structure events for the run, sorted by (idx, type).
    kl_zones : list of KLZone
        Both BOS- and CTS-derived zones (sd and opp_sd respectively).
    poi_zones : list of POIZone
        All POI zones (always sd direction by construction — Fib spans
        BOS→CTS so any POI within is in the structure direction).
    pip_size : float
        Pip size for the pair (typically 0.0001 / 0.01 depending on quote).
    timeframe : str
        "H1", "M15", "M5", etc. — used to look up default proximity_pips
        and min_gap_pips.
    proximity_pips : int, optional
        Threshold in pips. If None, looked up from DEFAULT_PROXIMITY_PIPS
        by `timeframe` (default fallback 20 for unknown timeframes).
    min_gap_pips : int, optional
        Narrow-cycle threshold in pips. If None, looked up from
        DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS by `timeframe`
        (default fallback to H1=50).

    Returns
    -------
    Dict[(sid, cycle_id), List[ZoneProximityTrigger]]
        Ordered list of triggers per cycle. Empty list / missing key when
        no triggers fired in the scan window.
    """
    if proximity_pips is None:
        proximity_pips = DEFAULT_PROXIMITY_PIPS.get(timeframe, 20)
    if min_gap_pips is None:
        min_gap_pips = DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS.get(
            timeframe, DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS["H1"]
        )
    threshold = proximity_pips * pip_size

    # Lookups
    cts_conf_by_key: Dict[tuple, StructureEvent] = {}
    for ev in sorted_events:
        if ev.type == "CTS_CONFIRMED":
            key = (ev.meta.get("structure_id", 0), ev.meta.get("cycle_id", 0))
            cts_conf_by_key[key] = ev

    bos_conf_idx_by_key: Dict[tuple, int] = {}
    for ev in sorted_events:
        if ev.type == "BOS_CONFIRMED":
            key = (ev.meta.get("structure_id", 0), ev.meta.get("cycle_id", 0))
            # confirmed_at = confirmation candle, not BOS extreme
            bos_conf_idx_by_key[key] = ef.event_moment(ev)

    reversal_idx_by_sid: Dict[int, int] = {}
    for ev in sorted_events:
        if ev.type == "REVERSAL_CANDIDATE":
            sid = ev.meta.get("structure_id", 0)
            apply_idx = ev.meta.get("apply_idx")
            if apply_idx is not None:
                reversal_idx_by_sid[sid] = apply_idx

    triggers_by_cycle: Dict[Tuple[int, int], List[ZoneProximityTrigger]] = {}

    for key, cts_ev in cts_conf_by_key.items():
        sid, cycle_id = key
        sd = int(cts_ev.meta.get("struct_direction", 0))
        if sd == 0:
            continue

        bos_zone = _find_kl_zone_for_cycle(kl_zones, sid, cycle_id, "BOS")
        if bos_zone is None:
            continue
        bos_inner = bos_zone.meta.get("inner")
        if bos_inner is None:
            continue

        cts_zone = _find_kl_zone_for_cycle(kl_zones, sid, cycle_id, "CTS")
        cts_inner = cts_zone.meta.get("inner") if cts_zone is not None else None

        cycle_poi_zones = [
            pz for pz in poi_zones
            if pz.meta.get("structure_id") == sid
            and pz.meta.get("cycle_id") == cycle_id
        ]

        # Scan window
        scan_start = ef.event_moment(cts_ev)
        scan_end = len(df) - 1
        next_bos_idx = bos_conf_idx_by_key.get((sid, cycle_id + 1))
        if next_bos_idx is not None:
            scan_end = min(scan_end, next_bos_idx - 1)
        reversal_idx = reversal_idx_by_sid.get(sid)
        if reversal_idx is not None:
            scan_end = min(scan_end, reversal_idx - 1)

        if scan_start > scan_end:
            continue

        # Rule 2 prerequisite: identify whether this cycle's CTS_CONFIRMED
        # was via pullback. Under Rule 1, narrow cycles can ONLY ever have
        # confirmation_method == "pullback" (sd-proximity is blocked
        # upstream in MarketStructure). A narrow cycle with no
        # pullback-confirmed CTS produces no CTS_CONFIRMED event at all
        # and won't reach this loop body.
        cts_confirmation_method = cts_ev.meta.get("confirmation_method", "pullback")
        cycle_has_pullback_cts = (cts_confirmation_method == "pullback")

        # Event-driven gap timeline (Rules 1/2/3 use this — NOT the df
        # cts_threshold/bos_threshold columns, which get overwritten by
        # the next structure once it starts processing).
        timeline = _build_cycle_threshold_timeline(sorted_events, sid, cycle_id)
        running_cts: Optional[float] = None
        running_bos: Optional[float] = None
        tl_ptr = 0
        # Initial pass: apply all events with moment ≤ scan_start so the gap
        # at idx = scan_start (the CTS_CONFIRMED candle) reflects the
        # post-CTS-confirmation state. Includes any same-moment
        # BOS_THRESHOLD_UPDATED events (sort order is `(moment, type)`,
        # alphabetical — BOS_* < CTS_*).
        # The walk is a TIME pointer: it compares the same MOMENT the timeline
        # is sorted on (Plan E E3g-2, PLAN_E §7.1 T3).
        while tl_ptr < len(timeline) and ef.event_moment(timeline[tl_ptr]) <= scan_start:
            running_cts, running_bos = _apply_threshold_event(
                timeline[tl_ptr], running_cts, running_bos
            )
            tl_ptr += 1

        triggers: List[ZoneProximityTrigger] = []
        expected_dir: Literal["sd", "opp_sd"] = "sd"
        # Rule 3: per-cycle caps that apply ONLY while the cycle is in
        # narrow mode. Once the gap crosses min_gap_pips, the caps are
        # ignored (Q1=A); alternation state continues seamlessly.
        narrow_sd_fired = False
        narrow_opp_fired = False

        for i in range(scan_start, scan_end + 1):
            if i not in df.index:
                continue

            # For i > scan_start, advance the timeline pointer through
            # events with moment < i (start-of-candle semantics — the
            # current candle's own threshold events do NOT influence its
            # own eligibility; see GOTCHAS "Narrow-Cycle Gap: Evaluate at
            # Start-of-Candle"). For i == scan_start the initial pass
            # above has already applied everything with moment ≤ scan_start.
            while tl_ptr < len(timeline) and ef.event_moment(timeline[tl_ptr]) < i:
                running_cts, running_bos = _apply_threshold_event(
                    timeline[tl_ptr], running_cts, running_bos
                )
                tl_ptr += 1

            if running_cts is not None and running_bos is not None:
                gap_pips = abs(running_cts - running_bos) / pip_size
                is_narrow = gap_pips < min_gap_pips
            else:
                # Can't evaluate (events missing) — fall back to default.
                is_narrow = False

            if is_narrow:
                # Rule 2: narrow-mode triggers require pullback-confirmed
                # CTS in this cycle. Without it, skip this candle.
                if not cycle_has_pullback_cts:
                    continue
                # Rule 3: cap same-kind triggers while narrow.
                if expected_dir == "sd" and narrow_sd_fired:
                    continue
                if expected_dir == "opp_sd" and narrow_opp_fired:
                    continue

            trigger = _try_trigger_at_candle(
                df=df,
                idx=i,
                expected_dir=expected_dir,
                sd=sd,
                bos_inner=float(bos_inner),
                cts_inner=cts_inner,
                cycle_poi_zones=cycle_poi_zones,
                threshold=threshold,
                sid=sid,
                cycle_id=cycle_id,
                proximity_pips=proximity_pips,
                pip_size=pip_size,
                timeframe=timeframe,
            )
            if trigger is None:
                # opp_sd path has no CTS inner available → no further
                # triggers possible for this cycle.
                if expected_dir == "opp_sd" and cts_inner is None:
                    break
                continue

            triggers.append(trigger)
            if is_narrow:
                if expected_dir == "sd":
                    narrow_sd_fired = True
                else:
                    narrow_opp_fired = True
            expected_dir = "opp_sd" if expected_dir == "sd" else "sd"

        if triggers:
            triggers_by_cycle[(sid, cycle_id)] = triggers

    return triggers_by_cycle


def _try_trigger_at_candle(
    df: pd.DataFrame,
    idx: int,
    expected_dir: Literal["sd", "opp_sd"],
    sd: int,
    bos_inner: float,
    cts_inner: Optional[float],
    cycle_poi_zones: List[POIZone],
    threshold: float,
    sid: int,
    cycle_id: int,
    proximity_pips: int,
    pip_size: float,
    timeframe: str,
) -> Optional[ZoneProximityTrigger]:
    """Return a ZoneProximityTrigger if candle `idx` qualifies for the
    expected direction, else None. Pure check — no state mutation.
    """
    if expected_dir == "sd":
        sd_inners: List[Tuple[float, str]] = [(bos_inner, "BOS")]
        for pz in cycle_poi_zones:
            # A POI can flap active/inactive within a cycle, so its live
            # state is NOT the scalar `confirmed_idx` (= last activate). Ask
            # the activation history whether it is active AS OF this candle.
            if not poi_active_as_of(pz, idx):
                continue
            if sd == 1:
                sd_inners.append((float(pz.top), "POI"))
            else:
                sd_inners.append((float(pz.bottom), "POI"))

        if sd == 1:
            trigger_inner, zone_kind = max(sd_inners, key=lambda t: t[0])
            candle_low = float(df.loc[idx, "l"])
            if candle_low <= trigger_inner + threshold:
                return ZoneProximityTrigger(
                    structure_id=sid, cycle_id=cycle_id,
                    direction="sd", idx=idx,
                    trigger_inner=trigger_inner, zone_kind=zone_kind,
                    proximity_pips=proximity_pips, pip_size=pip_size,
                    timeframe=timeframe,
                )
        else:
            trigger_inner, zone_kind = min(sd_inners, key=lambda t: t[0])
            candle_high = float(df.loc[idx, "h"])
            if candle_high >= trigger_inner - threshold:
                return ZoneProximityTrigger(
                    structure_id=sid, cycle_id=cycle_id,
                    direction="sd", idx=idx,
                    trigger_inner=trigger_inner, zone_kind=zone_kind,
                    proximity_pips=proximity_pips, pip_size=pip_size,
                    timeframe=timeframe,
                )
        return None

    # expected_dir == "opp_sd"
    if cts_inner is None:
        return None
    trigger_inner = float(cts_inner)
    if sd == 1:
        candle_high = float(df.loc[idx, "h"])
        if candle_high >= trigger_inner - threshold:
            return ZoneProximityTrigger(
                structure_id=sid, cycle_id=cycle_id,
                direction="opp_sd", idx=idx,
                trigger_inner=trigger_inner, zone_kind="CTS",
                proximity_pips=proximity_pips, pip_size=pip_size,
                timeframe=timeframe,
            )
    else:
        candle_low = float(df.loc[idx, "l"])
        if candle_low <= trigger_inner + threshold:
            return ZoneProximityTrigger(
                structure_id=sid, cycle_id=cycle_id,
                direction="opp_sd", idx=idx,
                trigger_inner=trigger_inner, zone_kind="CTS",
                proximity_pips=proximity_pips, pip_size=pip_size,
                timeframe=timeframe,
            )
    return None
