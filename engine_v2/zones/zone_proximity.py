# engine_v2/zones/zone_proximity.py
"""
Zone proximity trigger detection.

Per (sid, cycle_id) with a CTS_CONFIRMED event, walk the scan window and
log each candle that comes within `proximity_pips` of the relevant zone
inner bound. Triggers alternate: the first must be sd-direction (the sd
proximity reference: a BOS / POI zone), then opp_sd (the opp_sd proximity
reference: the CTS zone on parent zones), then sd, etc. — capturing each leg
of a V/lambda movement within a structure cycle.

Replaces the old `check_proximity_activation` in `wvmi.py`. The single
"activated cycles" gate is no longer the unit of interest — instead this
returns the full alternating list of trigger candles per cycle.

Zones are read by LOCATION, never by lifecycle (PLAN_Q, 2026-10-14)
-------------------------------------------------------------------
The BOS and CTS KL zones are read by their `meta["inner"]` alone (no
`confirmed_idx` / `end_idx` / `status` / `activation_history`), and each of
the cycle's POIs the same way — the user: "I would like POI to be treated
exactly like a KL zone" — at every candle of the scan; its activation history
(the lifecycle clamp, the imbalance-filled off-stretches) and its `end_idx`
are not read. Both bounds provably lie outside the scan and are asserted:
`poi_unclamped_activation_floor_idx` (its activation floor without the
structure lifecycle-start term) <= the scan start, its end > the scan's last
candle (PLAN_Q §3). The scan window
itself is in the structure's own timeline too (its CTS_CONFIRMED moment → the
next cycle's BOS_CONFIRMED moment / its realised reversal), with no
structure-lifecycle floor. Consequence: a COLLAPSED cycle (its lifecycle
window empty — e.g. a post-reversal structure's early, retroactively built
cycles) still fires triggers, drawn on the charts — reference window: H1
structure 1 cycle 1, lifecycle `[902, 902)`, fires 820 sd:BOS and 871
opp_sd:CTS — while the M15 subs those triggers would start are not built
(`lifecycle_sweep`: a collapsed parent cycle's triggers are logged unresolved;
LANDMINES "Collapsed Parent Cycles").

Proximity reference ladders (PLAN_R, the user 2026-10-19 → 20)
---------------------------------------------------------------
Each side triggers on its PROXIMITY REFERENCE — the closest usable, unpassed
zone inner on its PROXIMITY REFERENCE LADDER ("the smallest gap available"):
- sd: the cycle's POI(s) closer than its BOS inner, then the BOS (the LAST
  rung; a POI at or beyond the BOS is unreachable — the BOS locks first);
- opp_sd: the CTS (the last rung).
(PLAN_R step 2 adds the PROXIMITY OVERLAY — sub zones on each side.) A rung
is a PASSED PROXIMITY ZONE once an H1 close lies strictly beyond its outer
edge (the outer as of that close); it stays out for the rest of the cycle,
from the next candle; the pass test runs on EVERY candle, also one where a
gate blocks the trigger; the last rung is never tested. A side LOCKS when its
reference was passed and the next pick is its last rung (never by default).
Ties (equal inners): the parent zone first, then a sub's BOS before its POI.

The NARROW PROXIMITY-REFERENCE CAP: the signed PROXIMITY REFERENCE GAP
`sd × (opp_sd ref − sd ref) / pip`, rounded half-up, `<=`
`DEFAULT_MIN_PROXIMITY_REFERENCE_GAP_PIPS` (40 / 25 / 15) = narrow. Per-side
counters, evaluated at the START of each candle: wide → narrow (or narrow at
the scan start) starts both at 0; narrow → narrow, a side whose reference
moved OUTWARD (away from the other side, on that side's own inner) resets to
0, an inward move does nothing; narrow → wide drops them; while narrow a side
fires only if its counter < 1. It is AND-ed with Rules 1–3 below (their gap is
the BOS–CTS threshold gap — a different measure; Rule 2 is not part of the
cap). The per-cycle state machine is `CycleProximityScan` — ONE object the
batch scan and the lifecycle sweep both drive.

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
  then opp_sd, then sd, …); the caps block the trigger test only (the
  pass test still runs).

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

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple

import pandas as pd

from engine_v2.common.types import KLZone
from engine_v2.structure import event_fields as ef
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.zones.poi_zones import POIZone, poi_activation_floor_idx
from engine_v2.zones.structure_lifecycle import compute_reversal_idx_by_sid


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

# The narrow proximity-reference cap (PLAN_R §1, the user 2026-10-19: "1H: 40 15M: 25 5M: 15") — the minimum
# SIGNED proximity reference gap: the distance between the two zone inners a cycle triggers on, sd × (opp_sd
# reference − sd reference) / pip, rounded half-up. At or below it (`<=`; crossed references, ≤ 0, included) each
# side may fire ONE trigger per narrow spell — per-side counters that a side's own outward move resets
# (`CycleProximityScan`). Combined
# with Rules 1–3 (the BOS–CTS threshold gap, unchanged): a trigger fires only if BOTH allow it. Must be strictly
# greater than DEFAULT_PROXIMITY_PIPS on the same TF (two 8-pip bands closer than that overlap — the ping-pong).
DEFAULT_MIN_PROXIMITY_REFERENCE_GAP_PIPS: Dict[str, int] = {
    "H1": 40,
    "M15": 25,
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
for _tf in DEFAULT_MIN_PROXIMITY_REFERENCE_GAP_PIPS:
    assert DEFAULT_MIN_PROXIMITY_REFERENCE_GAP_PIPS[_tf] > DEFAULT_PROXIMITY_PIPS[_tf], (
        f"{_tf}: narrow proximity-reference gap "
        f"{DEFAULT_MIN_PROXIMITY_REFERENCE_GAP_PIPS[_tf]} pips must be "
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

    Direction "sd" = price came within threshold of the cycle's sd proximity
    reference (a BOS or POI zone inner); "opp_sd" = within threshold of its
    opp_sd proximity reference (the CTS zone inner on parent zones). `zone_kind`
    / `trigger_inner` = that reference (PLAN_R §1). `meta` (PLAN_R §4):
    `prox_ref_gap_pips` = the signed, rounded proximity reference gap at the
    candle's start (None when a side has no reference), `narrow_prox_ref` = it
    is at or below `DEFAULT_MIN_PROXIMITY_REFERENCE_GAP_PIPS`, `prox_ref_source`
    = "parent" (a parent zone) or "sub" (a proximity-overlay zone).
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


def poi_unclamped_activation_floor_idx(pz: POIZone) -> int:
    """A POI's activation floor WITHOUT the structure lifecycle-start term (PLAN_Q): max(the cycle's CTS-established
    moment, the IC[, the pattern-confirmed moment]) — the earliest candle its own conditions could make it active, in
    its own structure's timeline (`poi_zones.poi_activation_floor_idx`, the activation sweep's own floor). The scan
    asserts it is at or before the cycle's scan start, then counts the POI from the scan start like the BOS KL zone
    (PLAN_Q §3: the IC is at or before the fib's final CTS anchor, the pattern confirms by the anchor + 1, and the
    anchor precedes the CTS_CONFIRMED moment). The trigger hovers show it as a POI marker's `zone_confirmed_idx`."""
    return poi_activation_floor_idx(
        int(pz.meta["cts_established_idx"]), int(pz.ic_idx),
        pattern_confirmed_idx=pz.meta["pattern_confirmed_idx"],
        lifecycle_floor_idx=None,
    )


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
    Column Overwrite Hazard"). A scan window can straddle the next
    structure's rows — e.g. sid=0 cycle=2's window extends up to the
    realised reversal − 1, while sid=1 processes from its probe start (at or
    before the reversal), so its run has already overwritten the threshold
    cols on those rows.
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


@dataclass(frozen=True)
class ProximityRung:
    """One zone on a side's proximity reference ladder (PLAN_R §1).

    `key` = its identity in the cycle's passed set. `outer_at(i)` = its outer
    edge as of candle `i`'s CLOSE (a POI's fixed edge; an expanding zone's
    stepped outer). A `last` rung — the parent's BOS (sd side) / CTS (opp_sd
    side), the bottom rung of its ladder — is never pass-tested. `tie_rank`
    orders equal inners: the parent zone first (0), then a sub's BOS (1) before
    its POI (2). `source` / `sub_pool_idx` / `sub_cycle_id` name where the zone
    comes from (a sub zone: the proximity overlay, PLAN_R step 2).
    """
    key: Tuple[Any, ...]
    side: Literal["sd", "opp_sd"]
    zone_kind: Literal["BOS", "CTS", "POI"]
    inner: float
    last: bool = False
    outer_at: Optional[Callable[[int], float]] = None
    source: Literal["parent", "sub"] = "parent"
    tie_rank: int = 0
    sub_pool_idx: Optional[int] = None
    sub_cycle_id: Optional[int] = None


def _fixed_outer(outer: float) -> Callable[[int], float]:
    return lambda _i: outer


def _kl_outer_at(zone: KLZone) -> Callable[[int], float]:
    """A KL zone's outer edge as of candle `i`'s close (PLAN_R §1 "Which outer"): its INIT outer, then each expansion
    step whose moment (`expanded_idx`) is <= i — outers expand (H1 CTS zones), inners never move. A zone without
    `bounds_steps` (a synthetic fixture) keeps `meta["outer"]`."""
    edge = "bottom" if zone.side == "buy" else "top"
    steps = zone.meta.get("bounds_steps") or []
    if not steps:
        return _fixed_outer(float(zone.meta["outer"]))
    assert steps[0]["event"] == "INIT", f"[zone_proximity] bounds_steps[0] is {steps[0]['event']}, not INIT"
    init = float(steps[0][edge])
    expansions = [(int(s["expanded_idx"]), float(s[edge])) for s in steps[1:]]

    def outer_at(i: int) -> float:
        outer = init
        for moment, value in expansions:
            if moment <= i:
                outer = value
        return outer
    return outer_at


def proximity_reference_gap_pips(sd: int, sd_inner: float, opp_sd_inner: float, pip_size: float) -> int:
    """The SIGNED proximity reference gap, rounded half-up (PLAN_R §1): `sd × (opp_sd reference − sd reference) /
    pip`, so crossed references give ≤ 0 (narrow). Half-up = `floor(g + 0.5 + 1e-9)`: 40.5 → 41, 40.4 → 40."""
    g = int(sd) * (float(opp_sd_inner) - float(sd_inner)) / float(pip_size)
    return int(math.floor(g + 0.5 + 1e-9))


class CycleProximityScan:
    """The proximity scan of ONE parent cycle (PLAN_R §4): the alternating
    sd / opp_sd trigger state machine over the cycle's two proximity reference
    ladders, driven candle by candle — `step(i)` then `close(i)` for every
    candle of `[scan_start, scan_end]`. The batch scan (`check_zone_proximity`)
    drives it over the whole window; the lifecycle sweep (PLAN_R step 2) drives
    the SAME object tick by tick, so the two can never fork.

    `step(i)` — evaluated at the START of candle `i`:
      1. advance the threshold timeline (events with moment < i; the scan
         start's own were applied at construction);
      2. pick each side's proximity reference: the closest unpassed rung to
         the other side (ties: `tie_rank`); a LOCKED side keeps its last rung.
         A side locks when its previous reference was passed and the next pick
         is its last rung — being the reference by default never locks;
      3. the signed proximity reference gap → narrow (≤ `min_ref_gap_pips`) /
         wide → the per-side counters: wide → narrow (or narrow at the scan
         start) starts both at 0; narrow → narrow resets a side whose
         reference moved OUTWARD (away from the other side, judged on that
         side's own inner); narrow → wide drops them;
      4. Rules 2 + 3 (the BOS–CTS threshold gap, unchanged), 5. the narrow
         proximity-reference cap (a side fires while narrow only if its
         counter < 1) and alternation gate ONLY
      6. the trigger test against the expected side's reference.
    `close(i)` — the pass test on candle `i`'s close, effective from `i + 1`,
    on EVERY candle (a blocked one included): each unpassed non-last rung on
    either side whose outer (as of the close) the close is strictly beyond is
    passed for the rest of the cycle; one close can pass several.

    The proximity overlay (`set_overlay`, PLAN_R step 2) adds sub zones to a
    side; the batch scan never sets it (parent zones only).
    """

    def __init__(
        self,
        *,
        df: pd.DataFrame,
        sid: int,
        cycle_id: int,
        sd: int,
        scan_start: int,
        scan_end: int,
        parent_rungs: List[ProximityRung],
        timeline: List[StructureEvent],
        cycle_has_pullback_cts: bool,
        proximity_pips: int,
        pip_size: float,
        timeframe: str,
        min_gap_pips: int,
        min_ref_gap_pips: int,
    ) -> None:
        assert int(sd) in (1, -1), f"[zone_proximity] ({sid},{cycle_id}) sd={sd}"
        for side in ("sd", "opp_sd"):
            lasts = [r for r in parent_rungs if r.side == side and r.last]
            assert len(lasts) <= 1, f"[zone_proximity] ({sid},{cycle_id}) {side}: {len(lasts)} last rungs"
        self.df = df
        self.sid = int(sid)
        self.cycle_id = int(cycle_id)
        self.sd = int(sd)
        self.scan_start = int(scan_start)
        self.scan_end = int(scan_end)
        self.parent_rungs = list(parent_rungs)
        self.proximity_pips = proximity_pips
        self.pip_size = float(pip_size)
        self.timeframe = timeframe
        self.threshold = self.proximity_pips * self.pip_size
        self.min_gap_pips = min_gap_pips
        self.min_ref_gap_pips = int(min_ref_gap_pips)
        self.cycle_has_pullback_cts = bool(cycle_has_pullback_cts)

        self.overlay: Dict[str, List[ProximityRung]] = {"sd": [], "opp_sd": []}
        self.passed: Set[Tuple[Any, ...]] = set()
        self.locked: Dict[str, bool] = {"sd": False, "opp_sd": False}
        self.refs: Dict[str, Optional[ProximityRung]] = {"sd": None, "opp_sd": None}
        self.ref_gap_pips: Optional[int] = None
        self.narrow_prox_ref: bool = False
        self.counters: Optional[Dict[str, int]] = None
        self.expected_dir: Literal["sd", "opp_sd"] = "sd"
        self.triggers: List[ZoneProximityTrigger] = []
        self.gate: Optional[str] = None     # the gate that blocked the current candle's trigger test, if any
        # Rule 3: per-cycle caps that apply ONLY while the threshold gap is narrow.
        self._narrow_sd_fired = False
        self._narrow_opp_fired = False
        self._stepped_idx: Optional[int] = None
        self._closed_idx: Optional[int] = None

        # Event-driven threshold timeline (Rules 1/2/3 use this — NOT the df
        # cts_threshold/bos_threshold columns, which get overwritten by the next
        # structure once it starts processing). Initial pass: apply all events
        # with moment ≤ scan_start so the gap at idx = scan_start (the
        # CTS_CONFIRMED candle) reflects the post-CTS-confirmation state —
        # including same-moment BOS_THRESHOLD_UPDATED events (sort order is
        # `(moment, type)`, alphabetical — BOS_* < CTS_*). The walk is a TIME
        # pointer on the same MOMENT the timeline is sorted on (Plan E E3g-2).
        self._timeline = timeline
        self._running_cts: Optional[float] = None
        self._running_bos: Optional[float] = None
        self._tl_ptr = 0
        while self._tl_ptr < len(timeline) and ef.event_moment(timeline[self._tl_ptr]) <= self.scan_start:
            self._running_cts, self._running_bos = _apply_threshold_event(
                timeline[self._tl_ptr], self._running_cts, self._running_bos
            )
            self._tl_ptr += 1

    # --- the ladders ---
    def set_overlay(self, side: Literal["sd", "opp_sd"], rungs: List[ProximityRung]) -> None:
        """Replace `side`'s proximity-overlay rungs (sub zones, PLAN_R step 2)."""
        assert all(r.side == side and not r.last for r in rungs), (
            f"[zone_proximity] ({self.sid},{self.cycle_id}) overlay for {side}: wrong side or a last rung"
        )
        self.overlay[side] = list(rungs)

    def ladder(self, side: Literal["sd", "opp_sd"]) -> List[ProximityRung]:
        """`side`'s current candidate rungs: its parent zones, then its overlay."""
        return [r for r in self.parent_rungs if r.side == side] + self.overlay[side]

    def _toward(self, side: str) -> int:
        # The direction from a side's zones toward the other side: the sd zones
        # sit on the pullback side of the opp_sd zones.
        return self.sd if side == "sd" else -self.sd

    def _pick_reference(self, side: Literal["sd", "opp_sd"]) -> Optional[ProximityRung]:
        rungs = self.ladder(side)
        last = next((r for r in rungs if r.last), None)
        if self.locked[side]:
            return last
        toward = self._toward(side)
        best: Optional[ProximityRung] = None
        for r in sorted((r for r in rungs if r.key not in self.passed), key=lambda r: r.tie_rank):
            if best is None or toward * (r.inner - best.inner) > 0:
                best = r
        prev = self.refs[side]
        if prev is not None and prev.key in self.passed and best is not None and best.last:
            self.locked[side] = True        # locked by passing, never by default
        return best

    # --- per candle ---
    def step(self, i: int) -> Optional[ZoneProximityTrigger]:
        """Candle `i` (start-of-candle evaluation): references, gap, counters,
        gates, then the trigger test. Returns the trigger, or None."""
        i = int(i)
        assert self.scan_start <= i <= self.scan_end, (
            f"[zone_proximity] ({self.sid},{self.cycle_id}) step({i}) outside [{self.scan_start}, {self.scan_end}]"
        )
        assert self._stepped_idx is None or (i > self._stepped_idx and self._closed_idx == self._stepped_idx), (
            f"[zone_proximity] ({self.sid},{self.cycle_id}) step({i}) after step({self._stepped_idx}) / "
            f"close({self._closed_idx})"
        )
        self._stepped_idx = i
        self.gate = None

        # 1. For i > scan_start, advance the timeline pointer through events
        # with moment < i (start-of-candle semantics — the current candle's own
        # threshold events do NOT influence its own eligibility; see GOTCHAS
        # "Narrow-Cycle Gap: Evaluate at Start-of-Candle").
        while self._tl_ptr < len(self._timeline) and ef.event_moment(self._timeline[self._tl_ptr]) < i:
            self._running_cts, self._running_bos = _apply_threshold_event(
                self._timeline[self._tl_ptr], self._running_cts, self._running_bos
            )
            self._tl_ptr += 1
        if self._running_cts is not None and self._running_bos is not None:
            is_narrow = abs(self._running_cts - self._running_bos) / self.pip_size < self.min_gap_pips
        else:
            is_narrow = False               # can't evaluate (events missing) — default mode

        # 2. The references (start of candle i; a pass at close(i - 1) applies here).
        prev_refs = dict(self.refs)
        prev_narrow = self.narrow_prox_ref
        self.refs = {side: self._pick_reference(side) for side in ("sd", "opp_sd")}

        # 3. The signed proximity reference gap and the per-side counters.
        sd_ref, opp_ref = self.refs["sd"], self.refs["opp_sd"]
        if sd_ref is not None and opp_ref is not None:
            self.ref_gap_pips = proximity_reference_gap_pips(self.sd, sd_ref.inner, opp_ref.inner, self.pip_size)
            self.narrow_prox_ref = self.ref_gap_pips <= self.min_ref_gap_pips
        else:
            self.ref_gap_pips = None
            self.narrow_prox_ref = False
        if self.narrow_prox_ref and not prev_narrow:
            self.counters = {"sd": 0, "opp_sd": 0}          # wide → narrow, or narrow at the scan start
        elif self.narrow_prox_ref:
            for side in ("sd", "opp_sd"):
                old, new = prev_refs[side], self.refs[side]
                # OUTWARD = away from the other side, on THIS side's own inner (whatever the other side did).
                if old is not None and new is not None and -self._toward(side) * (new.inner - old.inner) > 0:
                    self.counters[side] = 0
        else:
            self.counters = None                            # wide: no cap

        # 4-5. The gates — they block ONLY the trigger test.
        side = self.expected_dir
        if is_narrow and not self.cycle_has_pullback_cts:
            self.gate = "rule2"
        elif is_narrow and side == "sd" and self._narrow_sd_fired:
            self.gate = "rule3"
        elif is_narrow and side == "opp_sd" and self._narrow_opp_fired:
            self.gate = "rule3"
        elif self.narrow_prox_ref and self.counters[side] >= 1:
            self.gate = "prox_ref_cap"                      # the narrow proximity-reference cap
        if self.gate is not None:
            return None

        # 6. The trigger test against the expected side's reference.
        ref = self.refs[side]
        if ref is None:
            return None
        trigger_inner = float(ref.inner)
        if (side == "sd") == (self.sd == 1):
            hit = float(self.df.loc[i, "l"]) <= trigger_inner + self.threshold
        else:
            hit = float(self.df.loc[i, "h"]) >= trigger_inner - self.threshold
        if not hit:
            return None
        trigger = ZoneProximityTrigger(
            structure_id=self.sid, cycle_id=self.cycle_id,
            direction=side, idx=i,
            trigger_inner=trigger_inner, zone_kind=ref.zone_kind,
            proximity_pips=self.proximity_pips, pip_size=self.pip_size,
            timeframe=self.timeframe,
            meta={
                "prox_ref_gap_pips": self.ref_gap_pips,
                "narrow_prox_ref": self.narrow_prox_ref,
                "prox_ref_source": ref.source,
            },
        )
        self.triggers.append(trigger)
        if is_narrow:
            if side == "sd":
                self._narrow_sd_fired = True
            else:
                self._narrow_opp_fired = True
        if self.narrow_prox_ref:
            self.counters[side] += 1
        self.expected_dir = "opp_sd" if side == "sd" else "sd"
        return trigger

    def close(self, i: int) -> List[ProximityRung]:
        """The pass test on candle `i`'s close (after `step(i)`), effective from
        `i + 1`. Returns the rungs passed by this close."""
        i = int(i)
        assert self._stepped_idx == i and self._closed_idx != i, (
            f"[zone_proximity] ({self.sid},{self.cycle_id}) close({i}) without step({i}) first"
        )
        self._closed_idx = i
        return self._pass_test(i)

    def _pass_test(self, i: int) -> List[ProximityRung]:
        c = float(self.df.loc[i, "c"])
        out: List[ProximityRung] = []
        for side in ("sd", "opp_sd"):
            # Strictly beyond the outer, AWAY from the other side (a close AT the outer is not a pass).
            beyond_below = (side == "sd") == (self.sd == 1)
            for r in self.ladder(side):
                if r.last or r.key in self.passed:
                    continue
                outer = float(r.outer_at(i))
                if (c < outer) if beyond_below else (c > outer):
                    self.passed.add(r.key)
                    out.append(r)
        return out


def _parent_rungs(
    sd: int,
    bos_zone: KLZone,
    cts_zone: Optional[KLZone],
    cycle_poi_zones: List[POIZone],
) -> List[ProximityRung]:
    """A parent cycle's rungs (PLAN_R §4): sd = each POI whose inner is strictly
    closer to the opp_sd side than the BOS inner (a POI at or beyond its BOS can
    never be reached — the BOS locks first; today's "closer of the two"), then
    the BOS (the last rung); opp_sd = the CTS (the last rung), when the cycle has
    a CTS zone with an inner. The last rungs carry their as-of outer too, though
    the pass test never reads it."""
    bos_inner = float(bos_zone.meta["inner"])
    cts_inner = cts_zone.meta.get("inner") if cts_zone is not None else None
    rungs: List[ProximityRung] = []
    for n, pz in enumerate(cycle_poi_zones):
        # Read like a KL zone (PLAN_Q): at every candle of the scan — never its
        # activation history (bounds asserted by the caller).
        inner, outer = (float(pz.top), float(pz.bottom)) if sd == 1 else (float(pz.bottom), float(pz.top))
        if sd * (inner - bos_inner) > 0:
            rungs.append(ProximityRung(
                key=("parent", "POI", n), side="sd", zone_kind="POI", inner=inner,
                outer_at=_fixed_outer(outer),
            ))
    rungs.append(ProximityRung(key=("parent", "BOS"), side="sd", zone_kind="BOS", inner=bos_inner, last=True,
                               outer_at=_kl_outer_at(bos_zone)))
    if cts_inner is not None:
        rungs.append(ProximityRung(key=("parent", "CTS"), side="opp_sd", zone_kind="CTS", inner=float(cts_inner),
                                   last=True, outer_at=_kl_outer_at(cts_zone)))
    return rungs


def build_cycle_proximity_scans(
    df: pd.DataFrame,
    sorted_events: List[StructureEvent],
    kl_zones: List[KLZone],
    poi_zones: List[POIZone],
    pip_size: float,
    timeframe: str,
    proximity_pips: Optional[int] = None,
    min_gap_pips: Optional[int] = None,
    min_proximity_reference_gap_pips: Optional[int] = None,
) -> Dict[Tuple[int, int], CycleProximityScan]:
    """One `CycleProximityScan` per (sid, cycle_id) with a CTS_CONFIRMED, a BOS
    zone and a non-empty scan window, in CTS_CONFIRMED order (the window and the
    skips: `check_zone_proximity`). Shared by the batch scan and the lifecycle
    sweep (PLAN_R §4)."""
    if proximity_pips is None:
        proximity_pips = DEFAULT_PROXIMITY_PIPS.get(timeframe, 20)
    if min_gap_pips is None:
        min_gap_pips = DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS.get(
            timeframe, DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS["H1"]
        )
    if min_proximity_reference_gap_pips is None:
        min_proximity_reference_gap_pips = DEFAULT_MIN_PROXIMITY_REFERENCE_GAP_PIPS.get(
            timeframe, DEFAULT_MIN_PROXIMITY_REFERENCE_GAP_PIPS["H1"]
        )

    # Lookups
    cts_conf_by_key: Dict[tuple, StructureEvent] = {}
    for ev in sorted_events:
        if ev.type == "CTS_CONFIRMED":
            key = (ev.meta.get("structure_id", 0), ev.meta.get("cycle_id", 0))
            cts_conf_by_key[key] = ev

    bos_confirmed_idx_by_key: Dict[tuple, int] = {}
    for ev in sorted_events:
        if ev.type == "BOS_CONFIRMED":
            key = (ev.meta.get("structure_id", 0), ev.meta.get("cycle_id", 0))
            # confirmed_at = confirmation candle, not the BOS anchor
            bos_confirmed_idx_by_key[key] = ef.event_moment(ev)

    # The realised reversal per sid, not a candidate's scheduled apply (check_zone_proximity's docstring).
    reversal_idx_by_sid: Dict[int, int] = compute_reversal_idx_by_sid(sorted_events)

    scans: Dict[Tuple[int, int], CycleProximityScan] = {}

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

        cycle_poi_zones = [
            pz for pz in poi_zones
            if pz.meta.get("structure_id") == sid
            and pz.meta.get("cycle_id") == cycle_id
        ]

        # Scan window
        scan_start = ef.event_moment(cts_ev)
        scan_end = len(df) - 1
        next_bos_confirmed_idx = bos_confirmed_idx_by_key.get((sid, cycle_id + 1))
        if next_bos_confirmed_idx is not None:
            scan_end = min(scan_end, next_bos_confirmed_idx - 1)
        reversal_idx = reversal_idx_by_sid.get(sid)
        if reversal_idx is not None:
            scan_end = min(scan_end, reversal_idx - 1)

        if scan_start > scan_end:
            continue

        # A POI is read like the KL zones (PLAN_Q): it counts at EVERY candle of the scan — neither its activation
        # history nor its end is read. Both bounds are provably outside the scan (PLAN_Q §3) and asserted, so a POI
        # that could not yet be active at the scan start, or ends inside it, fails loudly instead of being misread:
        # its unclamped floor <= the scan start (IC <= the fib's final CTS anchor < the CTS_CONFIRMED moment), its end
        # (its cycle's lifecycle end) > the scan's last candle (the next cycle's clamped start >= its BOS_CONFIRMED
        # moment; the same realised reversal).
        for poi in cycle_poi_zones:
            poi_floor_idx = poi_unclamped_activation_floor_idx(poi)
            assert poi_floor_idx <= scan_start, (
                f"[zone_proximity] POI IC {poi.ic_idx} of ({sid},{cycle_id}): unclamped activation floor "
                f"{poi_floor_idx} after the scan start {scan_start} — the location read (PLAN_Q) counts it from the "
                f"scan start"
            )
            poi_end_idx = poi.meta["end_idx"]
            assert poi_end_idx is None or int(poi_end_idx) > scan_end, (
                f"[zone_proximity] POI IC {poi.ic_idx} of ({sid},{cycle_id}) ends at {poi_end_idx}, inside the scan "
                f"[{scan_start}, {scan_end}] — the location read (PLAN_Q) would ignore that end"
            )

        # Rule 2 prerequisite: identify whether this cycle's CTS_CONFIRMED
        # was via pullback. Under Rule 1, narrow cycles can ONLY ever have
        # confirmation_method == "pullback" (sd-proximity is blocked
        # upstream in MarketStructure). A narrow cycle with no
        # pullback-confirmed CTS produces no CTS_CONFIRMED event at all
        # and won't reach this loop body.
        cts_confirmation_method = cts_ev.meta.get("confirmation_method", "pullback")

        scans[(sid, cycle_id)] = CycleProximityScan(
            df=df, sid=sid, cycle_id=cycle_id, sd=sd,
            scan_start=scan_start, scan_end=scan_end,
            parent_rungs=_parent_rungs(sd, bos_zone, cts_zone, cycle_poi_zones),
            timeline=_build_cycle_threshold_timeline(sorted_events, sid, cycle_id),
            cycle_has_pullback_cts=(cts_confirmation_method == "pullback"),
            proximity_pips=proximity_pips, pip_size=pip_size, timeframe=timeframe,
            min_gap_pips=min_gap_pips, min_ref_gap_pips=min_proximity_reference_gap_pips,
        )
    return scans


def check_zone_proximity(
    df: pd.DataFrame,
    sorted_events: List[StructureEvent],
    kl_zones: List[KLZone],
    poi_zones: List[POIZone],
    pip_size: float,
    timeframe: str,
    proximity_pips: Optional[int] = None,
    min_gap_pips: Optional[int] = None,
    min_proximity_reference_gap_pips: Optional[int] = None,
) -> Dict[Tuple[int, int], List[ZoneProximityTrigger]]:
    """Find zone proximity trigger candles per (sid, cycle_id) — the BATCH scan
    on parent zones (each cycle's `CycleProximityScan`, driven over its window).

    Per cycle, walk the scan window and log alternating sd/opp_sd triggers:
    1. First trigger must be sd (price within threshold of the sd proximity
       reference — a POI of the cycle, read like the BOS zone (PLAN_Q), or the
       BOS)
    2. After an sd trigger, only opp_sd is eligible (the opp_sd proximity
       reference — the CTS zone on parent zones)
    3. After an opp_sd trigger, only sd again — alternating

    Each "first match wins" applies per slot (i.e., once an sd trigger is
    logged, no more sd triggers fire until an opp_sd has been logged).

    The references come from each side's proximity reference ladder (PLAN_R
    §1): the closest unpassed rung; a rung is passed by a close strictly beyond
    its outer (from the next candle), the last rung (BOS / CTS) never; the
    narrow proximity-reference cap: while the signed reference gap is
    ≤ `min_proximity_reference_gap_pips`, each side fires once, until its own
    outward move resets its counter (`CycleProximityScan`).

    Narrow-gap restrictions (Rules 2 & 3, per module docstring): at each
    candle, evaluate `|cts_threshold − bos_threshold|` reconstructed
    from the cycle's threshold-defining events (events with `idx <
    current_candle` applied; events at `idx == current_candle` NOT
    applied — no self-rescue). While < `min_gap_pips`: require a
    pullback-confirmed CTS_CONFIRMED to have already fired for this
    cycle, and cap at most 1 sd + 1 opp_sd. After a wide-mode crossing
    (gap ≥ `min_gap_pips`), caps lift and the existing alternation
    continues unchanged. A trigger fires only if BOTH rule sets allow it.

    Scan window per (sid, cycle_id):
      start = CTS_CONFIRMED.meta["confirmed_at"] (the confirmation candle)
      end   = min(next_BOS_CONFIRMED.confirmed_at - 1,
                  reversal_idx - 1,
                  end_of_df)
    where `reversal_idx` = the sid's REALISED reversal, the
    `STATE_CHANGED(to="reversal")` candle (`compute_reversal_idx_by_sid`, the
    map KL / POI / fib / the orchestrator read) — never a
    `REVERSAL_CANDIDATE.meta["apply_idx"]`, a SCHEDULED apply: a watch expiry
    can discard it (a discarded LAST candidate used to cap every later cycle
    of the sid below its own scan start: no triggers, no WVMI gate record) and
    a step-anchor reversal can realise EARLIER than it (2026-09-28;
    `tests/test_zone_proximity_reversal_cap.py`).

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
        BOS→CTS so any POI within is in the structure direction). Each counts
        at every candle of its cycle's scan, like the BOS zone (PLAN_Q; its
        unclamped activation floor and its end asserted outside the scan).
    pip_size : float
        Pip size for the pair (typically 0.0001 / 0.01 depending on quote).
    timeframe : str
        "H1", "M15", "M5", etc. — used to look up default proximity_pips,
        min_gap_pips and min_proximity_reference_gap_pips.
    proximity_pips : int, optional
        Threshold in pips. If None, looked up from DEFAULT_PROXIMITY_PIPS
        by `timeframe` (default fallback 20 for unknown timeframes).
    min_gap_pips : int, optional
        Narrow-cycle threshold in pips. If None, looked up from
        DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS by `timeframe`
        (default fallback to H1=50).
    min_proximity_reference_gap_pips : int, optional
        The narrow proximity-reference cap's threshold in pips. If None, looked
        up from DEFAULT_MIN_PROXIMITY_REFERENCE_GAP_PIPS by `timeframe`
        (default fallback to H1=40).

    Returns
    -------
    Dict[(sid, cycle_id), List[ZoneProximityTrigger]]
        Ordered list of triggers per cycle. Empty list / missing key when
        no triggers fired in the scan window.
    """
    scans = build_cycle_proximity_scans(
        df, sorted_events, kl_zones, poi_zones, pip_size, timeframe,
        proximity_pips=proximity_pips, min_gap_pips=min_gap_pips,
        min_proximity_reference_gap_pips=min_proximity_reference_gap_pips,
    )
    triggers_by_cycle: Dict[Tuple[int, int], List[ZoneProximityTrigger]] = {}
    for key, scan in scans.items():
        for i in range(scan.scan_start, scan.scan_end + 1):
            if i not in df.index:
                continue
            scan.step(i)
            scan.close(i)
        if scan.triggers:
            triggers_by_cycle[key] = list(scan.triggers)
    return triggers_by_cycle
