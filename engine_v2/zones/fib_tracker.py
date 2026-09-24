# engine_v2/zones/fib_tracker.py
"""
Fibonacci level tracking for POI Zones.

Fib levels are drawn from BOS to CTS within each market structure cycle.
This module handles activation, updates, and lifecycle management.

Scenario Logic (sid 1+ only):
- Scenario 1: CTS_0 idx >= reversal_confirmed_idx → cycle 0 Fib unlocked
- Scenario 2: Cross-cycle Fib (BOS_0 → CTS_1) when all conditions met
- Scenario 3: Normal cycle 1 Fib when cross-cycle conditions fail
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import List, Optional, Dict, Any

import pandas as pd

from engine_v2.features.fibonacci import (
    FibRetracement,
    create_fib_retracement,
    DEFAULT_FIB_LEVELS,
)
from engine_v2.patterns.imbalance import has_unfilled_imbalance, get_unfilled_imbalances
from engine_v2.structure import event_fields as ef
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.zones.cross_cycle_fib import resolve_cross_cycle_eligibility
from engine_v2.zones.structure_lifecycle import (
    compute_cycle_lifecycle,
    compute_reversal_idx_by_sid,
    compute_struct_start_by_sid,
)

# Lifecycle convention (FIB_LIFECYCLE_SPEC.md section 10): terminal reasons
# that mean "invalidated / retracted" rather than ordinary historical end.
# A cycle whose terminal carries one of these derives status="disappeared"
# (terminal AND suppressed). Today only Scenario 1 revert qualifies.
_INVALIDATION_END_REASONS = frozenset({"scenario1_revert"})


@dataclass(frozen=True)
class FibState:
    """
    State of a Fibonacci retracement for a single cycle.

    Immutable - updates create new instances.
    """
    # Identity
    structure_id: int
    cycle_id: int
    struct_direction: int  # +1 bullish, -1 bearish

    # Anchors
    bos_idx: int
    bos_price: float
    cts_idx: int
    cts_price: float

    # State
    active: bool = True
    locked: bool = False  # True after CTS_CONFIRMED

    # The actual Fib retracement (computed from anchors)
    fib: Optional[FibRetracement] = None

    # Metadata for logging/debugging
    meta: Dict[str, Any] = field(default_factory=dict)

    # History of CTS anchor changes (for logging)
    cts_history: tuple = field(default_factory=tuple)

    # ------------------------------------------------------------------
    # Lifecycle convention fields (FIB_LIFECYCLE_SPEC.md §15 — scalar model).
    # CYCLE-IDENTITY artifacts stamped onto every version record of a cycle by
    # `_finalize_lifecycle_fields` (and the sub lifecycle-end cap). `status` is
    # the cycle-level label (all versions of a live cycle share it — §9.1).
    # See §3 (axes), §7 (Option A end), §15 (scalar logging).
    # ------------------------------------------------------------------
    # Lifecycle start (sticky): the cycle fib's first-active idx, clamped to the
    # cycle/structure lifecycle-start. None  ⟺  never active (collapsed). Set
    # ONCE — never moved on a later reactivation. Subsumes the removed
    # `activation_history`'s "ever active" signal (§15.3).
    start_idx: Optional[int] = None
    # Terminal axis (irreversible). None until the cycle ends. Earliest end among
    # candidates wins (set-if-absent); the cycle pass-through end is one candidate
    # alongside fib's own terminals (Option A early-end / scenario1_revert) — §15.4.
    end_idx: Optional[int] = None
    end_reason: Optional[str] = None
    # Derived cycle-identity label in {active, inactive, ended, disappeared} (§15.5).
    status: str = "active"


@dataclass
class FibTrackerConfig:
    """Configuration for FibTracker."""
    fib_levels: List[float] = field(default_factory=lambda: DEFAULT_FIB_LEVELS.copy())
    fill_threshold: float = 0.70  # 70% = filled


def select_fib_anchor_for_cycle(
    df: pd.DataFrame,
    sid: int,
    cycle_id: int,
    bos_idx: int,
    bos_price: float,
    cts_idx: int,
    cts_price: float,
    c0_data: Optional[Dict[str, Any]],
    fill_threshold: float = 0.70,
    struct_direction: int = 0,
    *,
    evaluated_at: Optional[int],
    fill_horizon_idx: int,
    snapshot_horizon_idx: int,
) -> tuple:
    """Pick the Fib anchor for a cycle. Pure function — no FibTracker state.

    Returns ``(anchor_bos_idx, anchor_bos_price, anchor_cts_idx,
    anchor_cts_price, scenario_label)``. Cross-cycle returns are signalled
    by an anchor_bos_idx that differs from the input bos_idx.

    Decision tree (mirrors FibTracker._handle_cycle1_scenarios for cycle 1):

    - ``sid == 0`` or ``cycle_id != 1`` or ``c0_data is None`` →
      intra-cycle, label ``"intra"``.
    - ``c0_data["scenario1"] is True`` → intra-cycle (cycle 1 stays normal),
      label ``"scenario_1"``.
    - Otherwise evaluate Scenario 2 conditions over the imbalance set:

        cond1 — cycle 1 has unfilled sd-direction imbalance in [BOS_1, CTS_1]
                as of cts_idx, counting only gaps formed by ``evaluated_at``
        cond2 — cycle 0 has unfilled sd-direction imbalance (cached on
                ``c0_data``; caller computed with the same direction filter)
        cond3 — BOS_1 has not filled cycle 0's sd-direction imbalances

      All three true → cross-cycle, label ``"scenario_2_cross"`` (anchor
      becomes BOS_0 → CTS_1). Else → intra-cycle, label ``"scenario_3"``.

    ``evaluated_at`` (keyword-only, REQUIRED) is the moment the decision is
    taken (Plan F; IMBALANCE_FILL_SEMANTICS.md "Knowability — the c3 rule"):
    FibTracker passes the CTS_1 ESTABLISHED moment; the MS in-flight resolver
    passes ``None`` (no cut — its snapshot is only read at candles after the
    CTS, where every gap it counts has formed). The two layers therefore agree
    on cond2 / cond3 but not always on cond1 — the accepted M1 divergence
    (LANDMINES "Scenario 2 anchor agreement").

    ``fill_horizon_idx`` / ``snapshot_horizon_idx`` (keyword-only, REQUIRED;
    Plan E E2b) are the fill horizons of cond1 (today the CTS_1 anchor) and of
    cond3 (today the BOS_1 anchor) — TIMES, split from the ``bos_idx`` /
    ``cts_idx`` locations so Plan E E3a / E3a′ can move them alone.

    ``struct_direction`` is the sd of the structure being evaluated (+1 / -1).
    Threads into cond1 / cond3 as the imbalance direction filter — counter-
    direction imbalances never become POIs, so they shouldn't influence the
    Scenario 2 decision. cond2 inherits this filter through whatever upstream
    computed ``c0_data["has_unfilled"]``. ``struct_direction == 0`` (the
    default) preserves legacy permissive behaviour if a caller hasn't
    migrated yet.

    Approximation (intentional): Scenario 1 REVERT — FibTracker reverts
    Scenario 1 from TRUE to FALSE if BOS_1 touches the prev structure's BOS
    zone outer, then evaluates Scenario 2/3. This utility cannot detect
    that itself because ``prev_bos_outer`` / ``prev_sd`` aren't inputs. A
    caller that tracks the revert state should flip ``c0_data["scenario1"]``
    from True to False/None BEFORE invoking this function and let the
    Scenario 2/3 path take over. MarketStructure's in-flight POI snapshot
    intentionally leaves ``scenario1`` as None (it doesn't track Scenario 1
    at all), so the utility always falls into the Scenario 2/3 evaluation
    path for cycle 1 — matching FibTracker's behavior whenever Scenario 1
    is False (the common case).
    """
    if sid < 1 or cycle_id != 1 or c0_data is None:
        return (int(bos_idx), float(bos_price), int(cts_idx), float(cts_price), "intra")

    if c0_data.get("scenario1") is True:
        return (int(bos_idx), float(bos_price), int(cts_idx), float(cts_price), "scenario_1")

    # Delegate the Scenario-2 cond1/cond2/cond3 decision to the shared routine
    # (CROSS_CYCLE_FIB_SPEC.md §11a). This is a single-step (target=1) cross
    # with the H1-main "snapshot" fill-as-of policy: the cycle-0 liveness is
    # cond3 (re-checked as of BOS_1 = own_imb_start) AND cond2 (the cached
    # cycle-0 @CTS_0 liveness in c0_data["has_unfilled"]). cond1 is the routine's
    # own-imbalance test over [BOS_1, CTS_1] as of CTS_1 (= current_candle).
    # The Scenario-1 outer gate (above) is main-only and stays here. A fresh
    # throwaway dead_cycles set is used — the wrapper never caches.
    elig = resolve_cross_cycle_eligibility(
        df=df,
        target_cycle=1,
        sd=int(struct_direction),
        own_window_end_idx=int(cts_idx),
        own_imb_start=int(bos_idx),
        anchor_idx=int(cts_idx),
        anchor_price=float(cts_price),
        bos_by_cycle={0: (int(c0_data["bos_idx"]), float(c0_data["bos_price"]))},
        cts_by_cycle={0: (int(c0_data["cts_idx"]), float(c0_data["cts_price"]))},
        dead_cycles=set(),
        fill_threshold=fill_threshold,
        fill_as_of="snapshot",
        prior_cached_liveness={0: bool(c0_data.get("has_unfilled", False))},
        evaluated_at=evaluated_at,
        fill_horizon_idx=int(fill_horizon_idx),
        snapshot_horizon_idx=int(snapshot_horizon_idx),
    )

    if elig.crosses:
        return (
            elig.bos_idx,
            elig.bos_price,
            int(cts_idx),
            float(cts_price),
            "scenario_2_cross",
        )
    return (int(bos_idx), float(bos_price), int(cts_idx), float(cts_price), "scenario_3")


class FibTracker:
    """
    Tracks Fibonacci levels across market structure cycles.

    Lifecycle:
    1. CTS_ESTABLISHED + unfilled imbalance → Fib activated
    2. CTS moves to new extreme → anchor 2 updates
    3. All imbalances filled → Fib deactivates
    4. CTS_CONFIRMED → Fib locked (stops updating)

    Scenario Logic (sid 1+ only):
    - Scenario 1 checked at each CTS_0 ESTABLISHED/UPDATED
    - If CTS_0 idx >= rv_idx → Scenario 1 TRUE (permanent), cycle 0 Fib unlocked
    - If still FALSE at CTS_0 CONFIRMED → Scenario 1 FALSE (permanent)
    - Scenario 2/3 determined at CTS_1 ESTABLISHED (only when Scenario 1 is FALSE)

    Only 1 active Fib per structure at a time (new cycle obsoletes previous).
    """

    def __init__(self, config: Optional[FibTrackerConfig] = None, fib_mode: str = "h1"):
        self.config = config or FibTrackerConfig()
        self.fib_mode = fib_mode  # "h1" (default) or "cross_cycle"
        # "cross_cycle" enables generalized cross-cycle fibs + pre_established
        # phase support. Used by all subordinate-structure pipelines (counter
        # and confluence, M15 today, deeper TFs in the future). The legacy
        # name was "m15_reverse" before subordinate structures generalized
        # beyond counter/reverse direction.

        # Fib states per key. Two key shapes coexist in this dict:
        #   - (sid, cycle_id)                          → single fib
        #   - (sid, cycle_id, "cross", version)        → cross fib (cross_cycle only)
        # `_fibs.values()` yields all fibs for charting iteration.
        self._fibs: Dict[tuple, FibState] = {}

        # Track which cycle is "current" per structure (for obsolescence)
        # {structure_id: cycle_id}
        self._current_cycle: Dict[int, int] = {}

        # Track cross-cycle Fib eligibility (H1 post-reversal, sid 1+ only)
        # {structure_id: {"cycle0": {...}, "normal_cycle1": FibState, "cross_cycle": FibState}}
        self._cross_cycle_data: Dict[int, Dict] = {}

        # Track Scenario 1 resolution per structure (H1 sid 1+ only)
        # {structure_id: True/False/None}
        # None = undetermined, True = CTS_0 >= rv_idx, False = resolved at CTS_0 CONFIRMED
        self._scenario1: Dict[int, Optional[bool]] = {}

        # Per sid: (prev_bos_outer, prev_sd) — the previous structure's last BOS
        # zone max-expanded OUTER (= _get_prev_bos_outer) and its direction.
        # Stashed from on_cts_established (H1 sid 1+). §11b uses it as P_rev for
        # the multi-cycle cross target CEILING M (CROSS_CYCLE_FIB_SPEC §4): the
        # post-reversal cross may target cycles 1..M, where M = earliest cycle
        # whose CTS clears P_rev. See `_cross_allowed_for_target`.
        self._prev_bos_outer: Dict[int, tuple] = {}

        # ------------------------------------------------------------------
        # M15 reverse mode: cross-fib state
        # ------------------------------------------------------------------
        # Per (sid, cycle_id): phase ∈ {"pre_established", "established", "confirmed"}.
        # Lifecycle: implicit → pre_established (on CTS_{n-1} CONFIRMED)
        #            → established (on CTS_n ESTABLISHED)
        #            → confirmed (on CTS_n CONFIRMED).
        # Cycle 0 has no pre_established phase.
        self._m15_phase: Dict[tuple, str] = {}

        # Per sid: set of cycle_ids whose [BOS_k, CTS_k] imbalances are all filled.
        # Once dead, a cycle stays dead (filling is permanent). Populated lazily
        # during walk-backward in cross-fib checks.
        self._dead_cycles: Dict[int, set] = {}

        # Per (sid, cycle_id): idx where CTS_k CONFIRMED fired. Used to compute
        # the prospective BOS for the NEXT cycle's pre-established phase.
        self._cts_confirmed_idx: Dict[tuple, int] = {}

        # Per (sid, cycle_id): (bos_idx, bos_price) captured on CTS_ESTABLISHED.
        # Used by the cross-fib walk-backward to look up each prior cycle's BOS.
        self._bos_by_cycle: Dict[tuple, tuple] = {}

        # Per (sid, cycle_id): (cts_idx, cts_price) captured on CTS_CONFIRMED
        # (the locked/final CTS). Only cycles with CONFIRMED CTS are eligible
        # for dead-cycle evaluation.
        self._cts_by_cycle: Dict[tuple, tuple] = {}

        # Per (sid, cycle_id): highest cross-fib version currently stored.
        # Cross keys are (sid, cycle_id, "cross", version). Extension replaces
        # in-place (same version); shrink bumps version by 1 and creates a
        # new entry, marking the old one inactive.
        self._cross_version: Dict[tuple, int] = {}

        # ------------------------------------------------------------------
        # Lifecycle convention layer (FIB_LIFECYCLE_SPEC.md §15 — scalar model).
        # Projected onto FibState records by `_finalize_lifecycle_fields`.
        # ------------------------------------------------------------------
        # Cycle-identity lifecycle-START, fed by BOTH subsystems (A: subordinate
        # cross_cycle; B: H1-main Scenario 2). Keyed (sid, cycle_id) -> first
        # active idx. Set-if-absent (sticky — the first activation wins, never
        # moved by a later reactivation). Replaces the removed activation_history:
        # "key present" ⟺ "was ever active" (§15.3).
        self._first_active: Dict[tuple, int] = {}

        # Cycle-identity terminal axis. Keyed (sid, cycle_id) -> (end_idx,
        # end_reason). Set-if-absent (first/earliest end wins, irreversible).
        # Populated for tracker-controlled terminals: new_cycle (obsolete) and
        # scenario1_revert; cross_cycle Option A early-end flows through the
        # obsolete path. Reversal / lifecycle_end are "passed-through" terminals
        # (spec §7): the H1-main reversal terminal is wired via
        # `set_reversal_terminals`; the sub lifecycle_end is set by the external
        # cap in entity_df_mutation (and, once §15.6/(b) lands, by the cycle
        # pass-through end fed in as a candidate).
        self._terminal: Dict[tuple, tuple] = {}

        # The MOMENT of the event being handled (Plan F — IMBALANCE_FILL_SEMANTICS
        # "Knowability — the c3 rule"): every imbalance question a handler asks
        # counts only gaps formed by then. Set by `_evaluating` around the three
        # handlers that reach an imbalance read; None outside a handler, and
        # inside one when the event records no moment (pattern-path CTS_UPDATED).
        self._evaluated_at: Optional[int] = None
        self._in_event = False

    # ------------------------------------------------------------------
    # Knowability (Plan F): the moment an imbalance question is asked.
    # ------------------------------------------------------------------
    @contextmanager
    def _evaluating(self, event: StructureEvent):
        """Scope `_evaluated_at` to one handler: set it to the event's moment
        (`event_fields.event_moment`), restore the previous value on exit.
        Handlers do not nest (no handler calls another); assert it."""
        assert not self._in_event, "FibTracker event handlers must not nest"
        # Resolve the moment BEFORE touching state: event_moment raises on a
        # malformed / unsupported event, and must not leave the scope flag set.
        moment = ef.event_moment(event)
        previous = self._evaluated_at
        self._in_event = True
        self._evaluated_at = moment
        try:
            yield
        finally:
            self._evaluated_at = previous
            self._in_event = False

    def _has_unfilled(self, df: pd.DataFrame, lo: int, hi: int, check_to: int, sd: int) -> bool:
        """`has_unfilled_imbalance` over [lo, hi] (fill horizon `check_to`,
        sd-direction strict), counting only gaps formed by the handled event's
        moment."""
        return has_unfilled_imbalance(
            df, lo, hi, check_to, self.config.fill_threshold,
            direction=sd, evaluated_at=self._evaluated_at,
        )

    def _c0_has_unfilled_now(self, c0: Dict[str, Any], df: pd.DataFrame, sd: int) -> bool:
        """The cycle-0 liveness cache asked AT the handled event's moment. The
        cache itself stays uncut (it is Scenario-2 cond2, judged at its later
        use); a decision taken now re-asks it with the cut. Absent c0 → False
        (today's `.get` default; `_handle_cycle0_cts_updated` can reach it)."""
        if not c0:
            return False
        lo = min(c0["bos_idx"], c0["cts_idx"])
        hi = max(c0["bos_idx"], c0["cts_idx"])
        return self._has_unfilled(df, lo, hi, c0["cts_idx"], sd)

    # ------------------------------------------------------------------
    # Lifecycle convention helpers (FIB_LIFECYCLE_SPEC.md §15).
    # ------------------------------------------------------------------
    def _mark_first_active(self, sid: int, cycle_id: int, idx: int) -> None:
        """Record the cycle's first-active idx (lifecycle start), set-if-absent.

        Sticky: the first activation wins and is never moved by a later
        reactivation, mirroring how a cycle/structure lifecycle-start doesn't
        shift when a zone flickers inactive→active (§15.3). Called only from the
        two creation sites (`_activate_fib`, `_m15_create_cross`) — the first
        time a cycle's fib goes active. "key present" ⟺ "was ever active".
        """
        self._first_active.setdefault((sid, cycle_id), int(idx))

    def _set_terminal(
        self,
        sid: int,
        cycle_id: int,
        end_idx: int,
        end_reason: str,
    ) -> None:
        """Record the cycle's terminal (end_idx, end_reason), set-if-absent.

        First/earliest end wins (terminals are irreversible — spec section 7).
        e.g. a cross_cycle Option A early-end set at pre-established creation is
        not overwritten by a later same-cycle obsolete call.
        """
        key = (sid, cycle_id)
        if key not in self._terminal:
            self._terminal[key] = (int(end_idx), end_reason)

    def _cycle_currently_active(self, sid: int, cycle_id: int) -> bool:
        """Whether the cycle's representative fib is active right now.

        Representative = active cross version if present, else the single-fib
        entry (mirrors get_active_fib). Used only for the condition half of the
        derived `status` when no terminal is set.
        """
        latest = self._get_latest_cross(sid, cycle_id)
        if latest is not None and latest[1].active:
            return True
        single = self._fibs.get((sid, cycle_id))
        return bool(single is not None and single.active)

    def set_reversal_terminals(
        self, reversal_confirmed_by_sid: Dict[int, int],
    ) -> None:
        """Wire the reversal terminal (FIB_LIFECYCLE_SPEC §7 passed-through end).

        `reversal_confirmed_by_sid` maps NEW sid -> reversal apply idx (the
        reversal that birthed the new sid, ending the PREVIOUS sid =
        new_sid - 1). For each, stamp the reversal terminal on every cycle of
        the ended (prev) sid. `_set_terminal` is set-if-absent, so cycles
        already terminated by `new_cycle` keep their earlier (correct) end and
        only the sid's still-open final cycle picks up "reversal".

        Closes the Session-1 deferral (reversal-ended H1-main fibs previously
        kept end_idx=None). Must run BEFORE `_finalize_lifecycle_fields` so the
        derived `status` reflects the terminal. Harmless for subordinate
        structures (their open cycle is also capped by the entity_df
        lifecycle-end cap at the same slice-local idx).
        """
        if not reversal_confirmed_by_sid:
            return
        cycles_by_sid: Dict[int, set] = {}
        for key in self._fibs:
            if isinstance(key, tuple) and len(key) >= 2:
                cycles_by_sid.setdefault(int(key[0]), set()).add(int(key[1]))
        for new_sid, rv_idx in reversal_confirmed_by_sid.items():
            prev_sid = int(new_sid) - 1
            if prev_sid < 0:
                continue
            for cyc in cycles_by_sid.get(prev_sid, ()):
                self._set_terminal(prev_sid, cyc, int(rv_idx), "reversal")

    def _finalize_lifecycle_fields(
        self,
        events: Optional[List[StructureEvent]] = None,
        lifecycle_floor: Optional[int] = None,
        lifecycle_cap: Optional[int] = None,
        cap_reason: str = "lifecycle_end",
    ) -> None:
        """Project the scalar lifecycle axes onto every FibState record in `_fibs`.

        Stamps, per cycle identity (sid, cycle_id): the sticky lifecycle
        `start_idx`, the terminal `end_idx`/`end_reason`, and the derived
        `status`. `status` is a CYCLE-level label shared by all version records
        of the cycle (spec §3.3 / §9.1 / §15.5). Only the lifecycle fields are
        written (via dataclasses.replace) — `active`/`locked`/`meta`/`fib`/
        anchors are untouched, so consumers see no change.

        When `events` is supplied (§15.6, part b), fib is put on the shared
        `structure_lifecycle` helper exactly like KL/POI:
          - `struct_floor[sid]` (`compute_struct_start_by_sid`) CLAMPS each
            cycle's `start_idx` up to the structure/parent floor (§15.3). NB the
            clamp is to the STRUCTURE floor, not the full cycle-start, so the §6
            pre-established early start survives.
          - the cycle pass-through end (`compute_cycle_lifecycle`) is fed into
            `_set_terminal` as an earliest-wins candidate (§15.4): set-if-absent,
            so fib's own earlier terminals (new_cycle / Option A / scenario1_revert
            / reversal) win, and the table end only fills cycles with no earlier
            terminal (the open last cycle / the subordinate cap).
        `lifecycle_floor`/`lifecycle_cap` are the same ints KL/POI receive
        (`None` for main; slice-local for subs). When `events` is None the clamp
        and pass-through are skipped (no caller does this today).

        Idempotent and safe to call once at end of the tracker's event stream
        (the orchestrator calls it before get_fibs_for_charting). Does NOT add
        or remove entries in `_fibs`.
        """
        from collections import defaultdict

        struct_floor: Dict[int, int] = {}
        if events is not None:
            rev = compute_reversal_idx_by_sid(events)
            struct_floor = compute_struct_start_by_sid(events, rev, lifecycle_floor)
            cycle_life = compute_cycle_lifecycle(
                events, rev, lifecycle_floor, lifecycle_cap, cap_reason,
            )
            # Feed the cycle pass-through end as an earliest-wins candidate.
            for (s, c), (_cstart, cend, creason) in cycle_life.items():
                if cend is not None:
                    self._set_terminal(s, c, cend, creason or cap_reason)

        keys_by_cycle: Dict[tuple, list] = defaultdict(list)
        for key in self._fibs:
            keys_by_cycle[(key[0], key[1])].append(key)

        for (sid, cycle_id), keys in keys_by_cycle.items():
            start_idx = self._first_active.get((sid, cycle_id))
            # Clamp to the structure floor (§15.3): raises a start only when it
            # precedes the parent/structure floor; the §6 pre-established early
            # start (after the floor) is left intact.
            if start_idx is not None:
                sfloor = struct_floor.get(sid)
                if sfloor is not None and int(sfloor) > start_idx:
                    start_idx = int(sfloor)

            term = self._terminal.get((sid, cycle_id))
            end_idx = term[0] if term else None
            end_reason = term[1] if term else None

            # Collapse (§15.5): a clamped start at/after the cycle end means the
            # cycle never had an active window → start_idx None, status inactive
            # (KL/POI collapse rule).
            if start_idx is not None and end_idx is not None and start_idx >= end_idx:
                start_idx = None

            # Status derivation (§15.5). The `start_idx is None` clause (before
            # the `ended` check) makes a collapsed/never-active cycle read
            # `inactive` even with an end_idx — the scalar start_idx replaces the
            # old "ended if activation_history else inactive" cap discriminator.
            if end_reason in _INVALIDATION_END_REASONS:
                status = "disappeared"
            elif start_idx is None:
                status = "inactive"
            elif end_idx is not None:
                status = "ended"
            elif self._cycle_currently_active(sid, cycle_id):
                status = "active"
            else:
                status = "inactive"

            for key in keys:
                fib = self._fibs[key]
                self._fibs[key] = replace(
                    fib,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    end_reason=end_reason,
                    status=status,
                )

    def on_cts_established(
        self,
        event: StructureEvent,
        df: pd.DataFrame,
        bos_idx: int,
        bos_price: float,
        reversal_confirmed_idx: Optional[int] = None,
        prev_bos_outer: Optional[float] = None,
        prev_sd: Optional[int] = None,
    ) -> Optional[FibState]:
        """Handle CTS_ESTABLISHED (body: `_on_cts_established`). Its imbalance
        questions are asked at the event's moment, `meta["confirmed_at"]`."""
        with self._evaluating(event):
            return self._on_cts_established(
                event, df, bos_idx, bos_price,
                reversal_confirmed_idx, prev_bos_outer, prev_sd,
            )

    def _on_cts_established(
        self,
        event: StructureEvent,
        df: pd.DataFrame,
        bos_idx: int,
        bos_price: float,
        reversal_confirmed_idx: Optional[int] = None,
        prev_bos_outer: Optional[float] = None,
        prev_sd: Optional[int] = None,
    ) -> Optional[FibState]:
        """
        Handle CTS_ESTABLISHED event - potentially activate a new Fib.

        Parameters
        ----------
        event : StructureEvent
            The CTS_ESTABLISHED event
        df : DataFrame
            OHLC data with imbalance columns
        bos_idx : int
            Index of the confirmed BOS (anchor 1)
        bos_price : float
            Price of the confirmed BOS
        reversal_confirmed_idx : int, optional
            Index where the reversal from previous structure was confirmed.
            Required for sid 1+ Scenario 1/2/3 logic.
        prev_bos_outer : float, optional
            The max expanded outer threshold of sid N-1's last BOS zone.
            Used for Scenario 1 revert check at CTS_1 ESTABLISHED.
        prev_sd : int, optional
            Direction of previous structure (+1 bullish, -1 bearish).
            Used for Scenario 1 revert check at CTS_1 ESTABLISHED.

        Returns
        -------
        FibState or None
            New FibState if activated, None otherwise
        """
        sid = int(event.meta.get("structure_id", 0))
        cycle_id = int(event.meta.get("cycle_id", 0))
        sd = int(event.meta.get("struct_direction", 0))
        # The CTS ANCHOR (a location): the fib 100% point, the IC / imbalance
        # range end, the cross anchor, the cycle-0 snapshot.
        cts_idx = ef.cts_anchor_idx(event)
        # The TIME half (Plan E E2b split): activation stamps, fill horizons,
        # Scenario 1, the revert terminal. Today's value (the anchor); Plan E
        # E3a switches it to the moment (`self._evaluated_at`).
        cts_established_idx = ef.cts_anchor_idx(event)  # Plan E E3a → moment
        cts_price = float(event.price) if event.price else 0.0

        # Get CTS price from event or df
        if cts_price == 0.0 and cts_idx in df.index:
            if sd == 1:
                cts_price = float(df.loc[cts_idx, "h"])
            else:
                cts_price = float(df.loc[cts_idx, "l"])

        # Check for unfilled imbalance between BOS and CTS, counting only gaps
        # formed by the moment (Plan F). sd-direction filter: Fibs only ever
        # produce sd-direction POIs, so counter-direction imbalances in the
        # BOS->CTS swing don't justify activation.
        start_idx = min(bos_idx, cts_idx)
        end_idx = max(bos_idx, cts_idx)
        has_unfilled = self._has_unfilled(df, start_idx, end_idx, cts_established_idx, sd)

        # Populate BOS lookup (used by cross-fib walk-backward in cross_cycle)
        self._bos_by_cycle[(sid, cycle_id)] = (bos_idx, bos_price)

        # §11b: stash P_rev (prev-BOS-outer) for the multi-cycle cross ceiling M.
        # The orchestrator passes it for H1 sid >= 1 (any cycle now). Set-once.
        if prev_bos_outer is not None and prev_sd is not None and sid not in self._prev_bos_outer:
            self._prev_bos_outer[sid] = (float(prev_bos_outer), int(prev_sd))

        # ============================================================
        # BRANCH: fib_mode determines logic
        # ============================================================
        if self.fib_mode == "cross_cycle":
            return self._handle_cross_cycle_cts_established(
                sid, cycle_id, sd, bos_idx, bos_price, cts_idx, cts_price, has_unfilled, df,
                cts_established_idx=cts_established_idx,
            )

        if sid == 0:
            return self._handle_sid0_cts_established(
                sid, cycle_id, sd, bos_idx, bos_price, cts_idx, cts_price, has_unfilled, df,
                cts_established_idx=cts_established_idx,
            )
        else:
            return self._handle_sid1plus_cts_established(
                sid, cycle_id, sd, bos_idx, bos_price, cts_idx, cts_price,
                has_unfilled, df, reversal_confirmed_idx, prev_bos_outer, prev_sd,
                cts_established_idx=cts_established_idx,
            )

    def _handle_cross_cycle_cts_established(
        self,
        sid: int,
        cycle_id: int,
        sd: int,
        bos_idx: int,
        bos_price: float,
        cts_idx: int,
        cts_price: float,
        has_unfilled: bool,
        df: pd.DataFrame,
        *,
        cts_established_idx: int,
    ) -> Optional[FibState]:
        """Handle CTS_ESTABLISHED for M15 reverse structure.

        - Cycle 0: single-fib only (no cross possible, no pre-established phase).
        - Cycle ≥1: phase flips pre_established → established. Run cross-fib
          check with anchor = CTS_n. If cross fails, fall back to single.
        """
        # Flip phase to established
        self._m15_phase[(sid, cycle_id)] = "established"

        if cycle_id == 0:
            # Cycle 0: no cross possible. Simple single-fib activation.
            if not has_unfilled:
                print(f"[fib] cross_cycle sid={sid} cycle=0 NO FIB: no unfilled imbalance (evaluated_at={self._evaluated_at})")
                return None
            print(f"[fib] cross_cycle sid={sid} cycle=0 ACTIVATED (single, no cross)")
            return self._activate_fib(
                sid=sid,
                cycle_id=0,
                sd=sd,
                bos_idx=bos_idx,
                bos_price=bos_price,
                cts_idx=cts_idx,
                cts_price=cts_price,
                meta={"activated_at": cts_established_idx, "fib_mode": "cross_cycle"},
            )

        # Cycle ≥1: run cross-fib check with CTS_n as anchor
        # own_imb_start = BOS_n (just-confirmed BOS for this cycle)
        print(f"[fib] cross_cycle sid={sid} cycle={cycle_id} CTS_ESTABLISHED: "
              f"running cross check (anchor=CTS_n)")
        self._m15_cross_check(
            sid=sid,
            target_cycle=cycle_id,
            df=df,
            sd=sd,
            own_window_end_idx=cts_idx,
            current_candle=cts_established_idx,
            anchor_idx=cts_idx,
            anchor_price=cts_price,
            own_imb_start=bos_idx,
        )
        # Return the currently active fib for this cycle (cross or single)
        latest = self._get_latest_cross(sid, cycle_id)
        if latest is not None and latest[1].active:
            return latest[1]
        return self._fibs.get((sid, cycle_id))

    def _handle_sid0_cts_established(
        self,
        sid: int,
        cycle_id: int,
        sd: int,
        bos_idx: int,
        bos_price: float,
        cts_idx: int,
        cts_price: float,
        has_unfilled: bool,
        df: pd.DataFrame,
        *,
        cts_established_idx: int,
    ) -> Optional[FibState]:
        """
        Handle CTS_ESTABLISHED for sid=0 (no prior reversal).

        Simple flow:
        - Cycle 0: No Fib
        - Cycle 1+: Normal Fib (if unfilled imbalance)
        - No cross-cycle logic
        """
        if cycle_id == 0:
            # sid=0, cycle=0: No Fib - just store data for reference
            print(f"[fib] sid=0 cycle=0 NO FIB (simple flow): BOS idx={bos_idx} -> CTS idx={cts_idx}")
            return None

        # sid=0, cycle 1+: Normal Fib activation
        if not has_unfilled:
            print(f"[fib] sid=0 cycle={cycle_id} NOT activated (simple flow): no unfilled imbalance (evaluated_at={self._evaluated_at})")
            return None

        print(f"[fib] sid=0 cycle={cycle_id} ACTIVATED (simple flow)")
        return self._activate_fib(
            sid=sid,
            cycle_id=cycle_id,
            sd=sd,
            bos_idx=bos_idx,
            bos_price=bos_price,
            cts_idx=cts_idx,
            cts_price=cts_price,
            meta={"activated_at": cts_established_idx, "flow": "simple"},
        )

    def _handle_sid1plus_cts_established(
        self,
        sid: int,
        cycle_id: int,
        sd: int,
        bos_idx: int,
        bos_price: float,
        cts_idx: int,
        cts_price: float,
        has_unfilled: bool,
        df: pd.DataFrame,
        reversal_confirmed_idx: Optional[int],
        prev_bos_outer: Optional[float] = None,
        prev_sd: Optional[int] = None,
        *,
        cts_established_idx: int,
    ) -> Optional[FibState]:
        """
        Handle CTS_ESTABLISHED for sid=1+ (post-reversal).

        Scenario logic:
        - Cycle 0: Check Scenario 1 (CTS_0 idx >= rv_idx)
          - If TRUE → cycle 0 Fib unlocked (can activate if unfilled imbalance)
          - If undetermined → store data, no Fib yet
        - Cycle 1: If Scenario 1 is TRUE, check revert condition; if FALSE check Scenario 2/3
        - Cycle 2+: Normal Fib
        """
        # --- Cycle 0: Scenario 1 check ---
        if cycle_id == 0:
            # The cycle-0 liveness CACHE is Scenario-2 cond2, judged at its later
            # use (CTS_1 ESTABLISHED > CTS_0: every gap in [BOS_0, CTS_0] has
            # formed) → stored UNCUT, which also keeps it equal to the MS
            # in-flight mirror (`_update_cycle0_data`). The decision taken NOW
            # uses the cut `has_unfilled` (Plan F §2).
            # Fill horizon: cond2 "@CTS_0" — a TIME, in lock-step with the MS
            # mirror `_update_cycle0_data`; both move in Plan E E3a′ (Q8).
            c0_fill_horizon_idx = cts_idx  # Plan E E3a′ → moment
            c0_has_unfilled_uncut = has_unfilled_imbalance(
                df, min(bos_idx, cts_idx), max(bos_idx, cts_idx), c0_fill_horizon_idx,
                self.config.fill_threshold, direction=sd, evaluated_at=None,
            )
            return self._handle_cycle0_scenario1(
                sid, sd, bos_idx, bos_price, cts_idx, cts_price,
                has_unfilled, c0_has_unfilled_uncut, reversal_confirmed_idx,
                cts_established_idx=cts_established_idx,
            )

        # --- Cycle 1: Depends on Scenario 1 resolution ---
        if cycle_id == 1:
            return self._handle_cycle1_scenarios(
                sid, sd, bos_idx, bos_price, cts_idx, cts_price,
                has_unfilled, df, reversal_confirmed_idx,
                prev_bos_outer, prev_sd,
                cts_established_idx=cts_established_idx,
            )

        # --- Cycle 2+: §11b multi-cycle cross (target <= M) or plain single ---
        # The cross only forms when (a) the M-ceiling allows it (no prior cycle
        # cleared P_rev) AND (b) the dead-cycle walk finds an eligible prior.
        # A read-only eligibility PEEK (dead-cycle COPY, so it can't pollute the
        # real cache) decides cross-vs-single, so a "no cross" outcome stays
        # byte-identical to the pre-11b plain-single path (same anchors + meta).
        if self._cross_allowed_for_target(sid, cycle_id, sd):
            cross = self._maybe_activate_main_cross(
                sid, cycle_id, sd, cts_idx, cts_price, df, cts_established_idx=cts_established_idx,
            )
            if cross is not None:
                return cross

        # Plain single (byte-identical with pre-11b).
        if not has_unfilled:
            print(f"[fib] sid={sid} cycle={cycle_id} NOT activated: no unfilled imbalance (evaluated_at={self._evaluated_at})")
            return None

        return self._activate_fib(
            sid=sid,
            cycle_id=cycle_id,
            sd=sd,
            bos_idx=bos_idx,
            bos_price=bos_price,
            cts_idx=cts_idx,
            cts_price=cts_price,
            meta={"activated_at": cts_established_idx},
        )

    def _handle_cycle0_scenario1(
        self,
        sid: int,
        sd: int,
        bos_idx: int,
        bos_price: float,
        cts_idx: int,
        cts_price: float,
        has_unfilled: bool,
        c0_has_unfilled_uncut: bool,
        reversal_confirmed_idx: Optional[int],
        *,
        cts_established_idx: int,
    ) -> Optional[FibState]:
        """
        Handle cycle 0 CTS_ESTABLISHED for sid 1+ - check Scenario 1.

        `has_unfilled` is asked at the event's moment (the Scenario-1 activation
        decision); `c0_has_unfilled_uncut` is what the cycle-0 cache stores.

        Scenario 1: CTS_0 idx >= reversal_confirmed_idx
        - If TRUE → cycle 0 Fib unlocked (permanent)
        - If FALSE/undetermined → store data for cross-cycle check
        """
        # Initialize cross-cycle data storage
        if sid not in self._cross_cycle_data:
            self._cross_cycle_data[sid] = {}

        # Store cycle 0 data
        self._cross_cycle_data[sid]["cycle0"] = {
            "bos_idx": bos_idx,
            "bos_price": bos_price,
            "cts_idx": cts_idx,
            "cts_price": cts_price,
            "struct_direction": sd,
            "has_unfilled": c0_has_unfilled_uncut,
            "locked": False,
        }

        # Check Scenario 1: CTS_0 idx >= rv_idx
        # A timing question (PLAN_E Q2 / §7.1 T5): has CTS_0 become known by the
        # reversal's confirmation?
        if reversal_confirmed_idx is not None and cts_established_idx >= reversal_confirmed_idx:
            # Scenario 1 TRUE (permanent) - cycle 0 Fib unlocked
            self._scenario1[sid] = True
            print(f"[fib] sid={sid} cycle=0 Scenario 1 TRUE at idx={cts_idx} (CTS >= rv_idx={reversal_confirmed_idx})")

            if has_unfilled:
                return self._activate_fib(
                    sid=sid,
                    cycle_id=0,
                    sd=sd,
                    bos_idx=bos_idx,
                    bos_price=bos_price,
                    cts_idx=cts_idx,
                    cts_price=cts_price,
                    meta={"activated_at": cts_established_idx, "scenario1": True},
                )
            else:
                print(f"[fib] sid={sid} cycle=0 NOT activated: Scenario 1 TRUE but no unfilled imbalance (evaluated_at={self._evaluated_at})")
                return None
        else:
            # Scenario 1 undetermined - store data, no Fib yet
            if sid not in self._scenario1:
                self._scenario1[sid] = None  # Undetermined
            print(f"[fib] sid={sid} cycle=0 STORED for cross-cycle check: BOS idx={bos_idx} -> CTS idx={cts_idx}, has_unfilled={c0_has_unfilled_uncut} (Scenario 1 undetermined)")
            return None

    def _handle_cycle1_scenarios(
        self,
        sid: int,
        sd: int,
        bos_idx: int,
        bos_price: float,
        cts_idx: int,
        cts_price: float,
        has_unfilled: bool,
        df: pd.DataFrame,
        reversal_confirmed_idx: Optional[int],
        prev_bos_outer: Optional[float] = None,
        prev_sd: Optional[int] = None,
        *,
        cts_established_idx: int,
    ) -> Optional[FibState]:
        """
        Handle cycle 1 CTS_ESTABLISHED for sid 1+.

        If Scenario 1 is TRUE → check revert condition first
          - If BOS_1 touches prev BOS zone outer → revert to FALSE, deactivate cycle 0 Fib
          - If no touch → stays TRUE, normal cycle 1 Fib
        If Scenario 1 is FALSE → check Scenario 2/3
        """
        scenario1 = self._scenario1.get(sid)

        # NEW: If Scenario 1 was TRUE, check revert condition
        if scenario1 is True and prev_bos_outer is not None and prev_sd is not None:
            if self._should_revert_scenario1(bos_price, prev_bos_outer, prev_sd):
                # Revert Scenario 1 to FALSE
                self._scenario1[sid] = False
                # Deactivate cycle 0 Fib if it exists (revert fires at this
                # cycle-1 CTS_ESTABLISHED idx = the cycle-0 fib's terminal)
                self._deactivate_cycle0_fib(sid, cts_established_idx)
                scenario1 = False
                print(f"[fib] sid={sid} Scenario 1 REVERTED to FALSE (BOS_1={bos_price:.5f} touched prev BOS zone outer={prev_bos_outer:.5f})")
            else:
                print(f"[fib] sid={sid} Scenario 1 stays TRUE (BOS_1={bos_price:.5f} did NOT touch prev BOS zone outer={prev_bos_outer:.5f})")

        # Scenario 1 TRUE: Normal cycle 1 Fib
        if scenario1 is True:
            if not has_unfilled:
                print(f"[fib] sid={sid} cycle=1 NOT activated (Scenario 1): no unfilled imbalance (evaluated_at={self._evaluated_at})")
                return None

            print(f"[fib] sid={sid} cycle=1 ACTIVATED (Scenario 1 TRUE, normal flow)")
            return self._activate_fib(
                sid=sid,
                cycle_id=1,
                sd=sd,
                bos_idx=bos_idx,
                bos_price=bos_price,
                cts_idx=cts_idx,
                cts_price=cts_price,
                meta={"activated_at": cts_established_idx, "scenario1": True},
            )

        # Scenario 1 FALSE: Check Scenario 2/3
        # (scenario1 is False or None - if None, it was resolved FALSE at CTS_0 CONFIRMED)
        if sid not in self._cross_cycle_data or "cycle0" not in self._cross_cycle_data[sid]:
            # No cycle 0 data - fallback to normal
            if not has_unfilled:
                print(f"[fib] sid={sid} cycle=1 NOT activated: no cycle 0 data, no unfilled imbalance (evaluated_at={self._evaluated_at})")
                return None

            return self._activate_fib(
                sid=sid,
                cycle_id=1,
                sd=sd,
                bos_idx=bos_idx,
                bos_price=bos_price,
                cts_idx=cts_idx,
                cts_price=cts_price,
                meta={"activated_at": cts_established_idx},
            )

        c0 = self._cross_cycle_data[sid]["cycle0"]

        # Route cross vs intra anchor selection through the shared utility so
        # MarketStructure's in-flight POI resolver and this downstream layer
        # agree on Scenario 2 (see select_fib_anchor_for_cycle for the
        # decision contract). They agree on cond2 / cond3; on cond1 this layer
        # asks at the CTS_1 moment while the in-flight resolver does not cut —
        # the accepted M1 divergence (Plan F §2). `scenario1` here is the post-revert value
        # (Scenario 1 TRUE was handled above); pass it explicitly so the
        # utility doesn't have to re-derive Scenario 1 from inputs.
        c0_for_utility = dict(c0)
        c0_for_utility["scenario1"] = scenario1

        anchor_bos_idx, anchor_bos_price, anchor_cts_idx, anchor_cts_price, label = (
            select_fib_anchor_for_cycle(
                df,
                sid,
                1,
                bos_idx,
                bos_price,
                cts_idx,
                cts_price,
                c0_for_utility,
                self.config.fill_threshold,
                struct_direction=sd,
                evaluated_at=self._evaluated_at,
                fill_horizon_idx=cts_established_idx,
                # cond3 "has BOS_1 filled cycle 0?" is asked as of BOS_1 (PLAN_E Q8).
                snapshot_horizon_idx=bos_idx,  # Plan E E3a′ → moment
            )
        )
        print(f"[fib] sid={sid} cycle=1 anchor decision: label={label} "
              f"(has_unfilled={has_unfilled})")

        # §11b M-ceiling: suppress the cycle-1 cross ONLY on a definite M == 0
        # (P_rev present AND cycle 0 already cleared it) → Scenario-3 single. The
        # `is not None` guard keeps pre-11b behavior when P_rev is absent
        # (degenerate sid>=1 with no prev BOS zone): no ceiling → don't suppress.
        # No-op when M >= 1 (cycle 0 has not cleared), so byte-identical here.
        if (
            label == "scenario_2_cross"
            and self._prev_bos_outer.get(sid) is not None
            and not self._cross_allowed_for_target(sid, 1, sd)
        ):
            print(f"[fib] sid={sid} cycle=1 cross suppressed (M-ceiling: cycle 0 cleared P_rev)")
            label = "scenario_3"
            anchor_bos_idx, anchor_bos_price = bos_idx, bos_price
            anchor_cts_idx, anchor_cts_price = cts_idx, cts_price

        if label == "scenario_2_cross":
            # Scenario 2: Cross-cycle Fib
            print(f"[fib] sid={sid} cycle=1 Scenario 2: CROSS-CYCLE ACTIVATED: "
                  f"BOS_0 idx={anchor_bos_idx} -> CTS_1 idx={anchor_cts_idx}")

            # §11a-ii: the cross is stored in versioned _fibs at
            # (sid, 1, "cross", 0) (retiring the _cross_cycle_data["cross_cycle"]
            # named slot). No fallback single is created upfront — create-on-fail
            # (see _update_cycle1_main): the normal single is materialized ONLY
            # if the cross later fails, mirroring the subordinate path. On this
            # window the cross wins throughout, so no single is ever created and
            # the result is byte-identical (the old normal_cycle1 lived only in
            # scratch and never reached _fibs/POI/chart). The BOS price for a
            # later fallback comes from _bos_by_cycle[(sid, 1)], so meta stays
            # exactly as before (no cycle1_bos_price key added).
            cross_fib = self._activate_fib(
                sid=sid,
                cycle_id=1,
                sd=sd,
                bos_idx=anchor_bos_idx,
                bos_price=anchor_bos_price,
                cts_idx=anchor_cts_idx,
                cts_price=anchor_cts_price,
                meta={
                    "cross_cycle": True,
                    "scenario": 2,
                    "activated_at": cts_established_idx,
                    "cycle1_bos_idx": bos_idx,
                },
                cross_version=0,
            )
            return cross_fib

        if has_unfilled:
            # Scenario 3: Normal cycle 1 Fib (utility selected intra anchors)
            print(f"[fib] sid={sid} cycle=1 Scenario 3: NORMAL ACTIVATED: "
                  f"BOS idx={anchor_bos_idx} -> CTS idx={anchor_cts_idx}")
            return self._activate_fib(
                sid=sid,
                cycle_id=1,
                sd=sd,
                bos_idx=anchor_bos_idx,
                bos_price=anchor_bos_price,
                cts_idx=anchor_cts_idx,
                cts_price=anchor_cts_price,
                meta={"activated_at": cts_established_idx, "scenario": 3},
            )

        # No unfilled imbalance in cycle 1
        print(f"[fib] sid={sid} cycle=1 NOT activated: no unfilled imbalance in cycle 1 (evaluated_at={self._evaluated_at})")
        return None

    def _create_fib_retracement(
        self,
        sd: int,
        bos_idx: int,
        bos_price: float,
        cts_idx: int,
        cts_price: float,
        sid: int,
        cycle_id: int,
    ) -> FibRetracement:
        """Helper to create a FibRetracement from anchors."""
        if sd == 1:  # Bullish
            anchor_high = cts_price
            anchor_low = bos_price
        else:  # Bearish
            anchor_high = bos_price
            anchor_low = cts_price

        return create_fib_retracement(
            anchor_high=anchor_high,
            anchor_low=anchor_low,
            direction=sd,
            levels=self.config.fib_levels,
            meta={"structure_id": sid, "cycle_id": cycle_id},
        )

    def _activate_fib(
        self,
        sid: int,
        cycle_id: int,
        sd: int,
        bos_idx: int,
        bos_price: float,
        cts_idx: int,
        cts_price: float,
        meta: Optional[Dict] = None,
        cross_version: Optional[int] = None,
    ) -> FibState:
        """Internal helper to create and store a FibState.

        Records the cycle's sticky lifecycle `start_idx` on first activation
        (§15.3) via `_mark_first_active`.

        `cross_version` (§11a-ii): when not None, the record is stored under the
        versioned key ``(sid, cycle_id, "cross", cross_version)`` and
        ``_cross_version[(sid, cycle_id)]`` is set — this is how the H1-main
        Scenario-2 cross now lives in versioned ``_fibs`` instead of the retired
        ``_cross_cycle_data`` named slot. ``None`` (default) keeps the legacy
        single key ``(sid, cycle_id)``. The FibState content (anchors, meta,
        lifecycle) is identical either way — only the storage key differs.
        """
        if sd == 1:  # Bullish
            anchor_high = cts_price
            anchor_low = bos_price
        else:  # Bearish
            anchor_high = bos_price
            anchor_low = cts_price

        fib = create_fib_retracement(
            anchor_high=anchor_high,
            anchor_low=anchor_low,
            direction=sd,
            levels=self.config.fib_levels,
            meta={"structure_id": sid, "cycle_id": cycle_id},
        )

        state = FibState(
            structure_id=sid,
            cycle_id=cycle_id,
            struct_direction=sd,
            bos_idx=bos_idx,
            bos_price=bos_price,
            cts_idx=cts_idx,
            cts_price=cts_price,
            active=True,
            locked=False,
            fib=fib,
            meta=meta or {},
            cts_history=((cts_idx, cts_price),),
        )

        activated_at = int((meta or {}).get("activated_at", cts_idx))

        # Mark previous cycle's Fib as obsolete (if exists)
        prev_cycle = self._current_cycle.get(sid)
        if prev_cycle is not None and prev_cycle != cycle_id:
            prev_key = (sid, prev_cycle)
            if prev_key in self._fibs:
                old_fib = self._fibs[prev_key]
                # Session 2 (FIB_LIFECYCLE_SPEC §3.4/§11): new_cycle is a CYCLE
                # TERMINAL, not a condition flip. Record it via end_idx/end_reason
                # (_set_terminal) and leave `active` (now condition-only)
                # untouched. The obsolete_reason meta is kept for debugging.
                obsolete = replace(old_fib, meta={**old_fib.meta, "obsolete_reason": "new_cycle"})
                self._fibs[prev_key] = obsolete
                # Lifecycle: prev cycle ends when this cycle activates.
                self._set_terminal(sid, prev_cycle, activated_at, "new_cycle")
            # §11a-ii: the prev cycle's fib may be a VERSIONED cross (H1-main
            # Scenario-2 cross now lives at (sid, prev, "cross", v), which the
            # single-key check above misses). Obsolete it the same way so the
            # new_cycle terminal still lands (else finalize's pass-through would
            # set a different end_reason). H1-MAIN ONLY: subordinate
            # (cross_cycle) crosses are obsoleted by _m15_create_cross's
            # _obsolete_prev_cycle_all_fibs at the next cross's creation — a sub
            # `cross_failed` single reaching _activate_fib must NOT also stamp
            # the prev cross here (that would flip its end_reason
            # next_cycle -> new_cycle), so this stays gated to "h1".
            if self.fib_mode == "h1":
                latest_prev_cross = self._get_latest_cross(sid, prev_cycle)
                if latest_prev_cross is not None:
                    cross_key, cross_state = latest_prev_cross
                    self._set_terminal(sid, prev_cycle, activated_at, "new_cycle")
                    if cross_state.meta.get("obsolete_reason") != "new_cycle":
                        self._fibs[cross_key] = replace(
                            cross_state,
                            meta={**cross_state.meta, "obsolete_reason": "new_cycle"},
                        )

        key = (sid, cycle_id, "cross", cross_version) if cross_version is not None else (sid, cycle_id)
        self._fibs[key] = state
        if cross_version is not None:
            self._cross_version[(sid, cycle_id)] = cross_version
        self._current_cycle[sid] = cycle_id
        # Lifecycle: record the cycle's sticky first-active idx (§15.3).
        self._mark_first_active(sid, cycle_id, activated_at)

        print(f"[fib] sid={sid} cycle={cycle_id} ACTIVATED: BOS idx={bos_idx} price={bos_price:.5f} -> CTS idx={cts_idx} price={cts_price:.5f}")

        return state

    def on_cts_updated(
        self,
        event: StructureEvent,
        df: pd.DataFrame,
        reversal_confirmed_idx: Optional[int] = None,
    ) -> Optional[FibState]:
        """Handle CTS_UPDATED (body: `_on_cts_updated`). Its imbalance questions
        are asked at the event's moment: `ev.idx` on the raw path; none recorded
        on the pattern path (no cut — Plan E)."""
        with self._evaluating(event):
            return self._on_cts_updated(event, df, reversal_confirmed_idx)

    def _on_cts_updated(
        self,
        event: StructureEvent,
        df: pd.DataFrame,
        reversal_confirmed_idx: Optional[int] = None,
    ) -> Optional[FibState]:
        """
        Handle CTS_UPDATED event - update CTS anchor if Fib is active.

        Parameters
        ----------
        event : StructureEvent
            The CTS_UPDATED event
        df : DataFrame
            OHLC data
        reversal_confirmed_idx : int, optional
            Index where the reversal was confirmed. Required for sid 1+ Scenario 1 check.

        Returns
        -------
        FibState or None
            Updated FibState, or None if no active Fib for this cycle
        """
        sid = int(event.meta.get("structure_id", 0))
        cycle_id = int(event.meta.get("cycle_id", 0))
        sd = int(event.meta.get("struct_direction", 0))
        # The updated CTS ANCHOR (a location). Its time half (activation stamps,
        # fill horizons) still reads the same value — the raw path's idx IS its
        # moment; the pattern path's moment is recorded from Plan E E3·0 and
        # switched in E3a (PLAN_E §6.7: the UPDATED time halves are not split in E2).
        cts_idx = ef.cts_anchor_idx(event)
        cts_price = float(event.price) if event.price else 0.0

        # Get CTS price from df if not in event
        if cts_price == 0.0 and cts_idx in df.index:
            if sd == 1:
                cts_price = float(df.loc[cts_idx, "h"])
            else:
                cts_price = float(df.loc[cts_idx, "l"])

        # ============================================================
        # BRANCH: fib_mode → cross_cycle / sid=0 / sid≥1
        # ============================================================
        if self.fib_mode == "cross_cycle":
            return self._handle_cross_cycle_cts_updated(sid, cycle_id, sd, cts_idx, cts_price, df)

        if sid == 0:
            return self._handle_sid0_cts_updated(sid, cycle_id, cts_idx, cts_price, df)
        else:
            return self._handle_sid1plus_cts_updated(
                sid, cycle_id, sd, cts_idx, cts_price, df, reversal_confirmed_idx
            )

    def _handle_cross_cycle_cts_updated(
        self,
        sid: int,
        cycle_id: int,
        sd: int,
        cts_idx: int,
        cts_price: float,
        df: pd.DataFrame,
    ) -> Optional[FibState]:
        """Handle CTS_UPDATED for cross_cycle. Re-run cross check with the
        extended CTS anchor. Cycle 0 uses simple single-fib update."""
        if cycle_id == 0:
            # Cycle 0: no cross logic. Single-fib update.
            key = (sid, 0)
            if key in self._fibs:
                return self._update_fib_cts(key, cts_idx, cts_price, df)
            # First-activation on a later CTS_UPDATED (2026-06-08): the cycle-0
            # Fib did NOT activate at CTS_ESTABLISHED (no unfilled imbalance
            # then), but an unfilled sd-direction imbalance may appear as the
            # CTS extends. Re-check over [BOS_0, cts_idx] as-of cts_idx and
            # activate now if present — mirrors the Scenario-1 main path
            # (_handle_cycle0_cts_updated). Fixes the cross_cycle cycle-0
            # one-shot asymmetry (subs were one-shot; main already re-checks).
            bos = self._bos_by_cycle.get((sid, 0))
            if bos is None:
                return None
            bos_idx, bos_price = bos
            start_idx = min(bos_idx, cts_idx)
            end_idx = max(bos_idx, cts_idx)
            if not self._has_unfilled(df, start_idx, end_idx, cts_idx, sd):  # UPD time half: Plan E E3·0/E3a
                return None
            print(f"[fib] cross_cycle sid={sid} cycle=0 ACTIVATED on update "
                  f"(unfilled imbalance found post-EST): BOS idx={bos_idx} -> "
                  f"CTS idx={cts_idx}")
            return self._activate_fib(
                sid=sid,
                cycle_id=0,
                sd=sd,
                bos_idx=bos_idx,
                bos_price=bos_price,
                cts_idx=cts_idx,
                cts_price=cts_price,
                meta={
                    "activated_at": cts_idx,  # UPD time half: Plan E E3·0/E3a
                    "fib_mode": "cross_cycle",
                    "activated_on": "update",
                },
            )

        # Cycle ≥1: should be in established phase. Re-run cross check.
        phase = self._m15_phase.get((sid, cycle_id))
        if phase != "established":
            # Shouldn't happen — CTS_UPDATED implies ESTABLISHED has already fired
            print(f"[fib][cross_cycle][warn] CTS_UPDATED for sid={sid} cycle={cycle_id} "
                  f"in unexpected phase={phase}")
            return None

        bos = self._bos_by_cycle.get((sid, cycle_id))
        if bos is None:
            return None

        self._m15_cross_check(
            sid=sid,
            target_cycle=cycle_id,
            df=df,
            sd=sd,
            own_window_end_idx=cts_idx,
            current_candle=cts_idx,  # UPD time half: Plan E E3·0/E3a
            anchor_idx=cts_idx,
            anchor_price=cts_price,
            own_imb_start=bos[0],
        )
        latest = self._get_latest_cross(sid, cycle_id)
        if latest is not None and latest[1].active:
            return latest[1]
        return self._fibs.get((sid, cycle_id))

    def _handle_sid0_cts_updated(
        self,
        sid: int,
        cycle_id: int,
        cts_idx: int,
        cts_price: float,
        df: pd.DataFrame,
    ) -> Optional[FibState]:
        """Handle CTS_UPDATED for sid=0 (simple flow)."""
        if cycle_id == 0:
            # sid=0, cycle=0: No Fib
            return None

        # sid=0, cycle 1+: Update normal Fib if exists
        key = (sid, cycle_id)
        if key not in self._fibs:
            return None

        return self._update_fib_cts(key, cts_idx, cts_price, df)

    def _handle_sid1plus_cts_updated(
        self,
        sid: int,
        cycle_id: int,
        sd: int,
        cts_idx: int,
        cts_price: float,
        df: pd.DataFrame,
        reversal_confirmed_idx: Optional[int],
    ) -> Optional[FibState]:
        """Handle CTS_UPDATED for sid=1+ (post-reversal)."""
        # --- Cycle 0: Scenario 1 re-check ---
        if cycle_id == 0:
            return self._handle_cycle0_cts_updated(
                sid, sd, cts_idx, cts_price, df, reversal_confirmed_idx
            )

        # --- Cycle 1: Update cross-cycle (versioned) or normal Fib ---
        # §11a-ii: the cross lives at (sid, 1, "cross", 0); detect via the
        # version map rather than the retired _cross_cycle_data["cross_cycle"].
        if cycle_id == 1 and self._get_latest_cross(sid, 1) is not None:
            return self._update_cycle1_main(sid, cts_idx, cts_price, df)

        # --- Cycle 2+: maintain a §11b multi-cycle cross if one formed at EST ---
        # (cross only born at EST — main cycle >= 2 keeps its one-shot semantics;
        # a cycle with no versioned cross falls through to the plain single
        # update, byte-identical with pre-11b).
        if cycle_id >= 2 and self._get_latest_cross(sid, cycle_id) is not None:
            return self._run_main_cross_check(
                sid, cycle_id, sd, cts_idx, cts_price, df, current_candle=cts_idx,  # UPD time half: E3·0/E3a
            )

        # --- Cycle 1+ normal Fib update ---
        key = (sid, cycle_id)
        if key not in self._fibs:
            return None

        return self._update_fib_cts(key, cts_idx, cts_price, df)

    def _handle_cycle0_cts_updated(
        self,
        sid: int,
        sd: int,
        cts_idx: int,
        cts_price: float,
        df: pd.DataFrame,
        reversal_confirmed_idx: Optional[int],
    ) -> Optional[FibState]:
        """
        Handle cycle 0 CTS_UPDATED for sid 1+ - re-check Scenario 1.

        If Scenario 1 is already TRUE, update the cycle 0 Fib.
        If Scenario 1 is still undetermined, check again.
        """
        scenario1 = self._scenario1.get(sid)

        # Update stored cycle 0 data
        if sid in self._cross_cycle_data and "cycle0" in self._cross_cycle_data[sid]:
            c0 = self._cross_cycle_data[sid]["cycle0"]
            if not c0.get("locked", False) and cts_idx > c0["cts_idx"]:
                c0["cts_idx"] = cts_idx
                c0["cts_price"] = cts_price
                # Re-check unfilled imbalance. sd-direction filter mirrors
                # the activation-time filter at on_cts_established so the
                # cycle-0 snapshot stays direction-consistent across updates.
                # UNCUT: the cache is Scenario-2 cond2, judged at its later use
                # (Plan F §2); a decision taken now re-asks it at the moment
                # (`_c0_has_unfilled_now`).
                start_idx = min(c0["bos_idx"], cts_idx)
                end_idx = max(c0["bos_idx"], cts_idx)
                c0["has_unfilled"] = has_unfilled_imbalance(
                    df, start_idx, end_idx, cts_idx, self.config.fill_threshold,
                    direction=sd, evaluated_at=None,
                )

        # If Scenario 1 is already TRUE, update the Fib
        if scenario1 is True:
            key = (sid, 0)
            if key in self._fibs:
                return self._update_fib_cts(key, cts_idx, cts_price, df)
            # No Fib but Scenario 1 is TRUE - check if we can activate now
            c0 = self._cross_cycle_data.get(sid, {}).get("cycle0", {})
            if self._c0_has_unfilled_now(c0, df, sd):
                print(f"[fib] sid={sid} cycle=0 ACTIVATED on update (Scenario 1 TRUE, unfilled imbalance found)")
                return self._activate_fib(
                    sid=sid,
                    cycle_id=0,
                    sd=sd,
                    bos_idx=c0["bos_idx"],
                    bos_price=c0["bos_price"],
                    cts_idx=cts_idx,
                    cts_price=cts_price,
                    meta={"activated_at": cts_idx, "scenario1": True, "activated_on": "update"},  # UPD time: E3·0/E3a
                )
            return None

        # Scenario 1 undetermined - check again (UPD time half: Plan E E3·0/E3a)
        if reversal_confirmed_idx is not None and cts_idx >= reversal_confirmed_idx:
            # Scenario 1 becomes TRUE
            self._scenario1[sid] = True
            print(f"[fib] sid={sid} cycle=0 Scenario 1 TRUE at idx={cts_idx} (CTS >= rv_idx={reversal_confirmed_idx}) on update")

            c0 = self._cross_cycle_data.get(sid, {}).get("cycle0", {})
            if self._c0_has_unfilled_now(c0, df, sd):
                return self._activate_fib(
                    sid=sid,
                    cycle_id=0,
                    sd=sd,
                    bos_idx=c0["bos_idx"],
                    bos_price=c0["bos_price"],
                    cts_idx=cts_idx,
                    cts_price=cts_price,
                    meta={"activated_at": cts_idx, "scenario1": True, "activated_on": "update"},  # UPD time: E3·0/E3a
                )
            else:
                print(f"[fib] sid={sid} cycle=0 NOT activated: Scenario 1 TRUE but no unfilled imbalance (evaluated_at={self._evaluated_at})")

        return None

    def _update_fib_cts(
        self,
        key: tuple,
        cts_idx: int,
        cts_price: float,
        df: pd.DataFrame,
    ) -> Optional[FibState]:
        """
        Internal helper to update a Fib's CTS anchor and check imbalance conditions.

        Used for both normal Fibs and cross-cycle Fibs.
        """
        state = self._fibs[key]
        if state.locked:
            return state

        # Only update if CTS moved to new extreme
        if cts_idx <= state.cts_idx:
            return state

        sid = state.structure_id
        cycle_id = state.cycle_id
        sd = state.struct_direction

        if sd == 1:
            anchor_high = cts_price
            anchor_low = state.bos_price
        else:
            anchor_high = state.bos_price
            anchor_low = cts_price

        new_fib = create_fib_retracement(
            anchor_high=anchor_high,
            anchor_low=anchor_low,
            direction=sd,
            levels=self.config.fib_levels,
            meta={"structure_id": sid, "cycle_id": cycle_id},
        )

        new_history = state.cts_history + ((cts_idx, cts_price),)
        new_state = replace(
            state,
            cts_idx=cts_idx,
            cts_price=cts_price,
            fib=new_fib,
            cts_history=new_history,
        )

        is_cross_cycle = state.meta.get("cross_cycle", False)
        label = "cross-cycle" if is_cross_cycle else f"cycle={cycle_id}"
        print(f"[fib] sid={sid} {label} UPDATED: CTS idx={cts_idx} price={cts_price:.5f}")

        # Check unfilled imbalance condition - can reactivate or deactivate
        if is_cross_cycle:
            # Cross-cycle Fib: check BOTH cycles' imbalance conditions.
            # sd-direction filter throughout (counter-direction imbalances
            # never produce POIs, so they don't influence cross-cycle eligibility).
            # Condition 1: Cycle 0 has unfilled imbalance (in cycle 0's locked range)
            c0 = self._cross_cycle_data.get(sid, {}).get("cycle0", {})
            c0_bos_idx = c0.get("bos_idx", new_state.bos_idx)
            c0_cts_idx = c0.get("cts_idx", new_state.bos_idx)  # Locked CTS_0
            c0_start = min(c0_bos_idx, c0_cts_idx)
            c0_end = max(c0_bos_idx, c0_cts_idx)
            cond1 = self._has_unfilled(df, c0_start, c0_end, c0_cts_idx, sd)

            # Condition 2: Cycle 1 has unfilled imbalance (BOS_1 to current CTS_1)
            cycle1_bos_idx = new_state.meta.get("cycle1_bos_idx", cts_idx)
            c1_start = min(cycle1_bos_idx, cts_idx)
            c1_end = max(cycle1_bos_idx, cts_idx)
            cond2 = self._has_unfilled(df, c1_start, c1_end, cts_idx, sd)

            # Condition 3: Cycle 1's BOS doesn't fill cycle 0's imbalances (static check)
            cond3 = self._has_unfilled(df, c0_start, c0_end, cycle1_bos_idx, sd)

            has_unfilled = cond1 and cond2 and cond3
            print(f"[fib] sid={sid} cross-cycle check: cond1={cond1} cond2={cond2} cond3={cond3}")
        else:
            # Normal Fib: check its own range (sd-direction; formed gaps only —
            # "all_imbalances_filled" below means "no FORMED unfilled imbalance").
            start_idx = min(new_state.bos_idx, new_state.cts_idx)
            end_idx = max(new_state.bos_idx, new_state.cts_idx)
            has_unfilled = self._has_unfilled(df, start_idx, end_idx, cts_idx, sd)

        if has_unfilled and not new_state.active:
            # Reactivate - unfilled imbalances now exist in expanded range
            new_state = replace(new_state, active=True, meta={**new_state.meta, "reactivated_at": cts_idx})
            print(f"[fib] sid={sid} {label} REACTIVATED: unfilled imbalance found at idx={cts_idx}")
        elif not has_unfilled and new_state.active:
            # Deactivate - all imbalances filled
            new_state = replace(new_state, active=False, meta={**new_state.meta, "deactivated_at": cts_idx, "reason": "all_imbalances_filled"})
            print(f"[fib] sid={sid} {label} DEACTIVATED: all imbalances filled at idx={cts_idx} "
                  f"(evaluated_at={self._evaluated_at})")

        self._fibs[key] = new_state
        return new_state

    def _update_cycle1_main(
        self,
        sid: int,
        cts_idx: int,
        cts_price: float,
        df: pd.DataFrame,
    ) -> Optional[FibState]:
        """Handle H1-main cycle-1 CTS update when a versioned Scenario-2 cross
        exists (§11a-ii — replaces the named-slot `_update_cycle1_fibs`).

        The cross lives at ``(sid, 1, "cross", 0)``; it is extended in place and
        its cond1/cond2/cond3 re-checked (computation identical to the prior
        named-slot path). When the cross is valid it stays the active record and
        NO single is created — byte-identical to before (the old `normal_cycle1`
        lived only in scratch and never reached `_fibs`). Only when the cross
        fails while the cycle-1 own imbalance is still unfilled is a single
        materialized at ``(sid, 1)`` (create-on-fail), mirroring the subordinate
        `cross_failed → single` fallback. In that (general-case, not exercised
        on this window) path the deactivated cross record persists in `_fibs`
        as a dead-version trail — a deliberate, accepted divergence from the
        old overwrite-the-mirror behavior (§11a-ii / CROSS_CYCLE_FIB_SPEC §8).
        """
        latest = self._get_latest_cross(sid, 1)
        if latest is None:
            return None
        cross_key, cross_fib = latest

        if cross_fib.locked:
            return cross_fib

        # Only update if CTS moved to new extreme
        if cts_idx <= cross_fib.cts_idx:
            return cross_fib

        sd = cross_fib.struct_direction

        # --- Update cross-cycle Fib (in place, same version) ---
        new_cross_fib = self._create_updated_fib_state(cross_fib, cts_idx, cts_price, sd, sid)

        print(f"[fib] sid={sid} cross-cycle UPDATED: CTS idx={cts_idx} price={cts_price:.5f}")

        # --- Check cross-cycle conditions --- sd-direction filter throughout.
        # Computation is unchanged from the prior named-slot path.
        c0 = self._cross_cycle_data.get(sid, {}).get("cycle0", {})
        c0_bos_idx = c0.get("bos_idx", new_cross_fib.bos_idx)
        c0_cts_idx = c0.get("cts_idx", new_cross_fib.bos_idx)
        c0_start = min(c0_bos_idx, c0_cts_idx)
        c0_end = max(c0_bos_idx, c0_cts_idx)
        cond1 = self._has_unfilled(df, c0_start, c0_end, c0_cts_idx, sd)

        cycle1_bos_idx = new_cross_fib.meta.get("cycle1_bos_idx", cts_idx)
        c1_start = min(cycle1_bos_idx, cts_idx)
        c1_end = max(cycle1_bos_idx, cts_idx)
        cond2 = self._has_unfilled(df, c1_start, c1_end, cts_idx, sd)

        cond3 = self._has_unfilled(df, c0_start, c0_end, cycle1_bos_idx, sd)

        cross_active = cond1 and cond2 and cond3
        print(f"[fib] sid={sid} cross-cycle check: cond1={cond1} cond2={cond2} cond3={cond3}")

        # --- Determine active state ---
        if cross_active and not new_cross_fib.active:
            new_cross_fib = replace(new_cross_fib, active=True, meta={**new_cross_fib.meta, "reactivated_at": cts_idx})
            print(f"[fib] sid={sid} cross-cycle REACTIVATED")
        elif not cross_active and new_cross_fib.active:
            new_cross_fib = replace(new_cross_fib, active=False, meta={**new_cross_fib.meta, "deactivated_at": cts_idx})
            print(f"[fib] sid={sid} cross-cycle DEACTIVATED")

        self._fibs[cross_key] = new_cross_fib

        if cross_active:
            return new_cross_fib

        # --- Create-on-fail: cross invalid → materialize the cycle-1 single iff
        # its own (BOS_1 -> CTS_1) imbalance is unfilled (the old "FALLBACK to
        # normal cycle=1"). BOS_1 comes from _bos_by_cycle so no extra meta key
        # is needed. ---
        bos1 = self._bos_by_cycle.get((sid, 1))
        if bos1 is None:
            return new_cross_fib
        bos1_idx, bos1_price = bos1
        normal_start = min(bos1_idx, cts_idx)
        normal_end = max(bos1_idx, cts_idx)
        normal_has_unfilled = self._has_unfilled(df, normal_start, normal_end, cts_idx, sd)
        single_key = (sid, 1)
        existing = self._fibs.get(single_key)
        if normal_has_unfilled:
            if existing is None or not existing.active:
                print(f"[fib] sid={sid} FALLBACK to normal cycle=1 (create-on-fail)")
                return self._activate_fib(
                    sid=sid,
                    cycle_id=1,
                    sd=sd,
                    bos_idx=bos1_idx,
                    bos_price=bos1_price,
                    cts_idx=cts_idx,
                    cts_price=cts_price,
                    meta={"activated_at": cts_idx, "scenario": 2, "role": "fallback"},
                )
            return self._update_fib_cts(single_key, cts_idx, cts_price, df)
        # Cross dead and own imbalance also filled → keep an existing single in
        # sync (deactivate it); otherwise nothing active (cross stays inactive).
        if existing is not None:
            return self._update_fib_cts(single_key, cts_idx, cts_price, df)
        return new_cross_fib

    def _create_updated_fib_state(
        self,
        state: FibState,
        cts_idx: int,
        cts_price: float,
        sd: int,
        sid: int,
    ) -> FibState:
        """Helper to create updated FibState with new CTS."""
        new_fib = self._create_fib_retracement(
            sd, state.bos_idx, state.bos_price, cts_idx, cts_price, sid, state.cycle_id
        )
        new_history = state.cts_history + ((cts_idx, cts_price),)
        return replace(
            state,
            cts_idx=cts_idx,
            cts_price=cts_price,
            fib=new_fib,
            cts_history=new_history,
        )

    def on_cts_confirmed(self, event: StructureEvent) -> Optional[FibState]:
        """
        Handle CTS_CONFIRMED event - lock the Fib.

        For sid 1+ cycle 0: Also resolves Scenario 1 as FALSE if still undetermined.

        Parameters
        ----------
        event : StructureEvent
            The CTS_CONFIRMED event

        Returns
        -------
        FibState or None
            Locked FibState, or None if no active Fib
        """
        sid = int(event.meta.get("structure_id", 0))
        cycle_id = int(event.meta.get("cycle_id", 0))

        # Populate CTS lookup (used by cross-fib walk-backward for dead-cycle
        # evaluation). CTS_anchor_idx is the CTS extreme; ev.idx is the
        # confirmation candle.
        cts_anchor_idx = int(event.meta.get("cts_anchor_idx", event.idx))
        cts_price_val = float(event.price) if event.price else 0.0
        self._cts_by_cycle[(sid, cycle_id)] = (cts_anchor_idx, cts_price_val)
        # Record confirmation candle (used to bound prospective-BOS search for n+1)
        self._cts_confirmed_idx[(sid, cycle_id)] = int(event.idx)

        # ============================================================
        # BRANCH: fib_mode → cross_cycle / sid=0 / sid≥1
        # ============================================================
        if self.fib_mode == "cross_cycle":
            return self._handle_cross_cycle_cts_confirmed(sid, cycle_id, event)

        if sid == 0:
            return self._handle_sid0_cts_confirmed(sid, cycle_id, event)
        else:
            return self._handle_sid1plus_cts_confirmed(sid, cycle_id, event)

    def _handle_cross_cycle_cts_confirmed(
        self,
        sid: int,
        cycle_id: int,
        event: StructureEvent,
    ) -> Optional[FibState]:
        """Handle CTS_CONFIRMED for cross_cycle: lock active fib (cross or
        single), flip phase to 'confirmed', set up pre_established for n+1."""
        # Lock active cross fib if present
        latest = self._get_latest_cross(sid, cycle_id)
        if latest is not None and latest[1].active and not latest[1].locked:
            key, state = latest
            locked = replace(
                state, locked=True,
                meta={**state.meta, "locked_at": event.idx},
            )
            self._fibs[key] = locked
            v = state.meta.get("version", 0)
            print(f"[fib] cross_cycle sid={sid} cycle={cycle_id} CROSS v{v} LOCKED")

        # Lock active single fib if present
        single_key = (sid, cycle_id)
        if single_key in self._fibs:
            state = self._fibs[single_key]
            if state.active and not state.locked:
                locked = replace(
                    state, locked=True,
                    meta={**state.meta, "locked_at": event.idx},
                )
                self._fibs[single_key] = locked
                print(f"[fib] cross_cycle sid={sid} cycle={cycle_id} SINGLE LOCKED")

        # Phase transitions
        self._m15_phase[(sid, cycle_id)] = "confirmed"
        self._m15_phase[(sid, cycle_id + 1)] = "pre_established"

        # Return whichever is currently active (preferring cross)
        if latest is not None and latest[1].locked:
            return self._fibs.get(latest[0])
        return self._fibs.get(single_key)

    def _handle_sid0_cts_confirmed(
        self,
        sid: int,
        cycle_id: int,
        event: StructureEvent,
    ) -> Optional[FibState]:
        """Handle CTS_CONFIRMED for sid=0 (simple flow)."""
        if cycle_id == 0:
            # sid=0, cycle=0: No Fib to lock
            print(f"[fib] sid=0 cycle=0 CONFIRMED (no Fib, simple flow)")
            return None

        # sid=0, cycle 1+: Lock normal Fib
        key = (sid, cycle_id)
        if key not in self._fibs:
            return None

        state = self._fibs[key]
        if state.locked:
            return state

        locked_state = replace(state, locked=True, meta={**state.meta, "locked_at": event.idx})
        self._fibs[key] = locked_state
        print(f"[fib] sid=0 cycle={cycle_id} LOCKED (simple flow): CTS idx={state.cts_idx}")
        return locked_state

    def _handle_sid1plus_cts_confirmed(
        self,
        sid: int,
        cycle_id: int,
        event: StructureEvent,
    ) -> Optional[FibState]:
        """Handle CTS_CONFIRMED for sid=1+ (post-reversal)."""
        # --- Cycle 0: Resolve Scenario 1 and lock ---
        if cycle_id == 0:
            return self._handle_cycle0_cts_confirmed(sid, event)

        key = (sid, cycle_id)

        # §11a-ii/§11b: a cycle-1 (Scenario-2) OR cycle-2..M (§11b) cross lives in
        # versioned _fibs. Lock the active cross version and/or the active single
        # (mirrors the subordinate lock in _handle_cross_cycle_cts_confirmed,
        # minus the phase transitions — main has no _m15_phase). Guard on "a
        # versioned cross exists" (not cycle_id == 1) so §11b cycles 2..M lock
        # too; byte-identical for prior cases (a cycle with no versioned cross
        # falls through to the single-key lock exactly as before).
        if self._get_latest_cross(sid, cycle_id) is not None:
            latest = self._get_latest_cross(sid, cycle_id)
            locked_any = None
            if latest is not None and latest[1].active and not latest[1].locked:
                ckey, cstate = latest
                locked_cross = replace(cstate, locked=True, meta={**cstate.meta, "locked_at": event.idx})
                self._fibs[ckey] = locked_cross
                print(f"[fib] sid={sid} cycle={cycle_id} cross-cycle LOCKED: CTS idx={cstate.cts_idx}")
                locked_any = locked_cross
            single = self._fibs.get(key)
            if single is not None and single.active and not single.locked:
                locked_single = replace(single, locked=True, meta={**single.meta, "locked_at": event.idx})
                self._fibs[key] = locked_single
                print(f"[fib] sid={sid} cycle={cycle_id} single LOCKED: CTS idx={single.cts_idx}")
                if locked_any is None:
                    locked_any = locked_single
            return locked_any if locked_any is not None else self._fibs.get(latest[0])

        if key not in self._fibs:
            return None

        state = self._fibs[key]
        if state.locked:
            return state

        locked_state = replace(state, locked=True, meta={**state.meta, "locked_at": event.idx})
        self._fibs[key] = locked_state

        is_cross_cycle = state.meta.get("cross_cycle", False)
        label = "cross-cycle" if is_cross_cycle else f"cycle={cycle_id}"
        print(f"[fib] sid={sid} {label} LOCKED: CTS idx={state.cts_idx} price={state.cts_price:.5f}")

        return locked_state

    def _handle_cycle0_cts_confirmed(
        self,
        sid: int,
        event: StructureEvent,
    ) -> Optional[FibState]:
        """
        Handle cycle 0 CTS_CONFIRMED for sid 1+.

        Resolves Scenario 1 as FALSE if still undetermined.
        Locks cycle 0 data and any cycle 0 Fib.
        """
        # Resolve Scenario 1 if still undetermined
        scenario1 = self._scenario1.get(sid)
        if scenario1 is None:
            # Scenario 1 never became TRUE → resolve as FALSE (permanent)
            self._scenario1[sid] = False
            c0_cts = self._cross_cycle_data.get(sid, {}).get("cycle0", {}).get("cts_idx", "?")
            print(f"[fib] sid={sid} cycle=0 Scenario 1 FALSE (resolved at CONFIRMED): CTS idx={c0_cts} never reached rv_idx")

        # Lock cycle 0 data in _cross_cycle_data
        if sid in self._cross_cycle_data and "cycle0" in self._cross_cycle_data[sid]:
            c0 = self._cross_cycle_data[sid]["cycle0"]
            c0["locked"] = True
            print(f"[fib] sid={sid} cycle=0 LOCKED in cross-cycle data: CTS idx={c0['cts_idx']}")

        # Lock cycle 0 Fib if it exists (Scenario 1 was TRUE)
        key = (sid, 0)
        if key in self._fibs:
            state = self._fibs[key]
            if not state.locked:
                locked_state = replace(state, locked=True, meta={**state.meta, "locked_at": event.idx})
                self._fibs[key] = locked_state
                print(f"[fib] sid={sid} cycle=0 Fib LOCKED: CTS idx={state.cts_idx}")
                return locked_state
            return state

        return None

    def _should_revert_scenario1(
        self,
        bos1_price: float,
        prev_bos_outer: float,
        prev_sd: int,
    ) -> bool:
        """
        Check if BOS_1 level price touches/crosses into the prev structure's last BOS zone.

        "Touch" means price reaches the outer edge OR goes beyond it into the zone.

        For bullish prev structure (sd=1): BOS zone is buy, outer is bottom, zone is ABOVE outer
          → BOS_1 touches if bos1_price >= outer (price at or above outer edge)
        For bearish prev structure (sd=-1): BOS zone is sell, outer is top, zone is BELOW outer
          → BOS_1 touches if bos1_price <= outer (price at or below outer edge)
        """
        if prev_sd == 1:  # Prev was bullish, buy zone sits above outer (bottom)
            return bos1_price >= prev_bos_outer
        else:  # Prev was bearish, sell zone sits below outer (top)
            return bos1_price <= prev_bos_outer

    def _deactivate_cycle0_fib(self, sid: int, revert_idx: int) -> None:
        """Deactivate cycle 0 Fib when Scenario 1 reverts to FALSE.

        `revert_idx` (= the CTS_1 ESTABLISHED idx where the revert fires) is
        the cycle-0 fib's terminal. This is an invalidation, not an ordinary
        end → status="disappeared" (spec section 10).
        """
        key = (sid, 0)
        if key in self._fibs:
            state = self._fibs[key]
            # Session 2 (FIB_LIFECYCLE_SPEC §3.4/§10): scenario1_revert is a
            # terminal INVALIDATION (-> derived status "disappeared"), not a
            # condition flip — set end_idx/end_reason and leave `active`
            # (condition-only) untouched. The deactivated_by meta is kept as a
            # belt-and-suspenders marker alongside the disappeared status.
            self._fibs[key] = replace(
                state,
                meta={**state.meta, "deactivated_by": "scenario1_revert"}
            )
            # Lifecycle terminal (invalidation class -> disappeared).
            self._set_terminal(sid, 0, revert_idx, "scenario1_revert")
            print(f"[fib] sid={sid} cycle=0 DEACTIVATED (Scenario 1 reverted)")

    # ------------------------------------------------------------------
    # M15 reverse mode helpers
    # ------------------------------------------------------------------

    def _find_prospective_bos(
        self,
        df: pd.DataFrame,
        cts_n_confirmed_idx: int,
        current_candle: int,
        sd: int,
    ) -> Optional[tuple]:
        """Find the running-extreme pullback candle after CTS_n CONFIRMED.

        For sd=+1: argmin low in [cts_n_confirmed_idx + 1, current_candle].
        For sd=-1: argmax high in same range.

        This is the "prospective BOS_{n+1}" — the candle that will become
        BOS_{n+1} if and when a new breakout establishes CTS_{n+1}. Used as
        the start anchor for cycle n+1's pre-established imbalance range.

        Returns (idx, price) or None if window is empty.
        """
        start = cts_n_confirmed_idx + 1
        end = current_candle + 1
        if start >= end or start not in df.index:
            return None
        window = df.iloc[start:end]
        if sd == 1:
            rel = int(window["l"].values.argmin())
            return (start + rel, float(window["l"].values[rel]))
        else:
            rel = int(window["h"].values.argmax())
            return (start + rel, float(window["h"].values[rel]))

    def _running_extreme_anchor(
        self,
        df: pd.DataFrame,
        cts_n_confirmed_idx: int,
        current_candle: int,
        sd: int,
    ) -> Optional[tuple]:
        """Find the running extreme past CTS_n (used as the CTS anchor during
        pre-established phase).

        For sd=+1: argmax high in [cts_n_confirmed_idx + 1, current_candle].
        For sd=-1: argmin low in same range.

        Returns (idx, price) or None if window is empty.
        """
        start = cts_n_confirmed_idx + 1
        end = current_candle + 1
        if start >= end or start not in df.index:
            return None
        window = df.iloc[start:end]
        if sd == 1:
            rel = int(window["h"].values.argmax())
            return (start + rel, float(window["h"].values[rel]))
        else:
            rel = int(window["l"].values.argmin())
            return (start + rel, float(window["l"].values[rel]))

    def _cross_allowed_for_target(self, sid: int, target_cycle: int, sd: int) -> bool:
        """H1-main multi-cycle cross CEILING M (CROSS_CYCLE_FIB_SPEC §4).

        Allow a cross at ``target_cycle`` iff **no cycle in [0, target_cycle-1]
        has yet cleared P_rev** (the prev structure's last-BOS-zone outer) in
        the new structure's sd direction. This is exactly "target <= M" where
        ``M`` = earliest cycle whose locked CTS clears P_rev — the clearing
        cycle is itself the last valid target; the next cycle falls back to a
        plain single.

        Edge cases (§4.4): no P_rev stashed (sid 0 / no prev structure) → no
        cross. ``M = 0`` (CTS_0 already clears) → cycle 0 cleared, so every
        target >= 1 is disallowed → no cross ever (cycle-0 single only).
        Not-yet-cleared (open structure) → allowed (ceiling not reached).
        Prior cycles are CONFIRMED before ``target_cycle`` so their CTS extremes
        are locked in ``_cts_by_cycle`` — stable for the life of target_cycle.
        """
        pbo = self._prev_bos_outer.get(sid)
        if pbo is None:
            return False
        p_rev, _prev_sd = pbo
        for k in range(target_cycle):
            cts = self._cts_by_cycle.get((sid, k))
            if cts is None:
                continue
            cts_price = cts[1]
            clears = (cts_price >= p_rev) if sd == 1 else (cts_price <= p_rev)
            if clears:
                return False
        return True

    def _run_main_cross_check(
        self, sid: int, target_cycle: int, sd: int, cts_idx: int, cts_price: float,
        df: pd.DataFrame, *, current_candle: int,
    ) -> Optional[FibState]:
        """Run the versioned cross machinery for an H1-main cycle (§11b).

        Reuses `_m15_cross_check` (mode-agnostic version transitions; main has no
        `_m15_phase`, so phase=None drives the established-style single fallback).
        Anchor = CTS_n (the just-seen extreme); own_imb_start = BOS_n. Returns the
        active fib (cross or single fallback) for the cycle.
        """
        bos = self._bos_by_cycle.get((sid, target_cycle))
        if bos is None:
            return None
        self._m15_cross_check(
            sid=sid, target_cycle=target_cycle, df=df, sd=sd,
            own_window_end_idx=cts_idx, current_candle=current_candle,
            anchor_idx=cts_idx, anchor_price=cts_price, own_imb_start=bos[0],
        )
        latest = self._get_latest_cross(sid, target_cycle)
        if latest is not None and latest[1].active:
            return latest[1]
        return self._fibs.get((sid, target_cycle))

    def _maybe_activate_main_cross(
        self, sid: int, target_cycle: int, sd: int, cts_idx: int, cts_price: float,
        df: pd.DataFrame, *, cts_established_idx: int,
    ) -> Optional[FibState]:
        """Decide (read-only) whether a §11b main cross forms at target_cycle and,
        if so, apply it. Returns the cross FibState, or None when no cross forms
        (caller falls back to the byte-identical plain-single path).

        The PEEK runs the shared routine on a COPY of the dead-cycle cache so a
        no-cross outcome leaves real state untouched — only an actual cross
        mutates `_fibs`/`_cross_version`/`_dead_cycles` (via `_m15_cross_check`).
        """
        bos = self._bos_by_cycle.get((sid, target_cycle))
        if bos is None:
            return None
        elig = resolve_cross_cycle_eligibility(
            df=df, target_cycle=target_cycle, sd=sd, own_window_end_idx=cts_idx,
            own_imb_start=bos[0], anchor_idx=cts_idx, anchor_price=cts_price,
            bos_by_cycle={
                k: self._bos_by_cycle[(sid, k)]
                for k in range(target_cycle) if (sid, k) in self._bos_by_cycle
            },
            cts_by_cycle={
                k: self._cts_by_cycle[(sid, k)]
                for k in range(target_cycle) if (sid, k) in self._cts_by_cycle
            },
            dead_cycles=set(self._dead_cycles.get(sid, set())),  # COPY (peek only)
            fill_threshold=self.config.fill_threshold, fill_as_of="current",
            evaluated_at=self._evaluated_at,
            fill_horizon_idx=cts_established_idx, snapshot_horizon_idx=None,
        )
        if not elig.crosses:
            return None
        print(f"[fib] sid={sid} cycle={target_cycle} §11b multi-cycle CROSS "
              f"(x={elig.earliest_x} -> {target_cycle})")
        return self._run_main_cross_check(
            sid, target_cycle, sd, cts_idx, cts_price, df, current_candle=cts_established_idx,
        )

    def _get_latest_cross(self, sid: int, cycle_id: int) -> Optional[tuple]:
        """Return (key, FibState) for the highest-version cross fib at
        (sid, cycle_id), or None."""
        version = self._cross_version.get((sid, cycle_id))
        if version is None:
            return None
        key = (sid, cycle_id, "cross", version)
        fib = self._fibs.get(key)
        if fib is None:
            return None
        return (key, fib)

    def _obsolete_prev_cycle_all_fibs(
        self, sid: int, new_cycle_id: int, end_idx: Optional[int] = None,
    ) -> None:
        """Mark every fib for cycle < new_cycle_id (single + all cross versions)
        as inactive via obsolete_reason='new_cycle'. Skips already-inactive
        fibs.

        `end_idx` (the creating cycle's birth idx) is the obsoleted cycles'
        lifecycle terminal. Set-if-absent implements Option A (spec section 7):
        when a next-cycle pre-established cross is created during cycle n's
        tail, cycle n's fib ends at that early creation idx; an older cycle
        already terminated keeps its earlier end.
        """
        for key in list(self._fibs.keys()):
            if not isinstance(key, tuple) or len(key) < 2:
                continue
            if key[0] != sid:
                continue
            k_cycle = key[1]
            if k_cycle >= new_cycle_id:
                continue
            if end_idx is not None:
                self._set_terminal(sid, k_cycle, end_idx, "new_cycle")
            state = self._fibs[key]
            # Session 2: new_cycle is a TERMINAL (recorded above via
            # _set_terminal), NOT a condition flip — leave `active` alone.
            # Stamp obsolete_reason for debugging; skip if already stamped.
            if state.meta.get("obsolete_reason") == "new_cycle":
                continue
            self._fibs[key] = replace(
                state,
                meta={**state.meta, "obsolete_reason": "new_cycle"},
            )

    # ------------------------------------------------------------------
    # M15 reverse: CTS_THRESHOLD_UPDATED dispatch (pre-established phase)
    # ------------------------------------------------------------------

    def on_cts_threshold_updated(
        self,
        event: StructureEvent,
        df: pd.DataFrame,
    ) -> None:
        """Handle CTS_THRESHOLD_UPDATED (body: `_on_cts_threshold_updated`). Its
        imbalance questions are asked at the event's moment, `ev.idx` (the
        processing candle)."""
        with self._evaluating(event):
            self._on_cts_threshold_updated(event, df)

    def _on_cts_threshold_updated(
        self,
        event: StructureEvent,
        df: pd.DataFrame,
    ) -> None:
        """Dispatch CTS_THRESHOLD_UPDATED for cross_cycle mode.

        Triggers a cross-fib check for the target cycle (= event.cycle_id + 1)
        ONLY when that target is in pre-established phase. In all other cases
        (h1 mode, or target not pre-established), this is a no-op.

        The anchor is the running extreme past CTS_n; own_imb_start is the
        prospective BOS_{n+1} (the deepest pullback since CTS_n CONFIRMED).
        """
        if self.fib_mode != "cross_cycle":
            return

        sid = int(event.meta.get("structure_id", 0))
        source_cycle = int(event.meta.get("cycle_id", 0))
        target_cycle = source_cycle + 1

        phase = self._m15_phase.get((sid, target_cycle))
        if phase != "pre_established":
            # Not in pre-established for the expected target: skip.
            return

        sd = int(event.meta.get("struct_direction", 0))
        if sd == 0:
            return
        current_candle = int(event.idx)

        # Lookup CTS_n confirmation candle to bound the prospective-BOS search
        cts_n_conf = self._cts_confirmed_idx.get((sid, source_cycle))
        if cts_n_conf is None:
            return

        anchor = self._running_extreme_anchor(df, cts_n_conf, current_candle, sd)
        if anchor is None:
            return
        anchor_idx, anchor_price = anchor

        prospective_bos = self._find_prospective_bos(df, cts_n_conf, current_candle, sd)
        if prospective_bos is None:
            return
        own_imb_start = prospective_bos[0]

        self._m15_cross_check(
            sid=sid,
            target_cycle=target_cycle,
            df=df,
            sd=sd,
            own_window_end_idx=current_candle,
            current_candle=current_candle,
            anchor_idx=anchor_idx,
            anchor_price=anchor_price,
            own_imb_start=own_imb_start,
        )

    # ------------------------------------------------------------------
    # M15 reverse cross-fib check + state transitions
    # ------------------------------------------------------------------

    def _m15_cross_check(
        self,
        sid: int,
        target_cycle: int,
        df: pd.DataFrame,
        sd: int,
        current_candle: int,
        anchor_idx: int,
        anchor_price: float,
        own_imb_start: int,
        *,
        own_window_end_idx: int,
    ) -> None:
        """Central cross-fib check routine for cross_cycle mode.

        `own_window_end_idx` (a LOCATION: the CTS anchor at CTS_ESTABLISHED /
        CTS_UPDATED, the processing candle on the threshold path) ends the own
        test's window; `current_candle` is the TIME — the fill horizon and every
        lifecycle stamp (activated_at / deactivated_at / terminal). Split in
        Plan E E2b so E3a moves only the time.

        1. Check target cycle's own imbalance (range [own_imb_start, own_window_end_idx]).
           If none: deactivate any active cross; no fallback in pre-established.
        2. Walk backward from target-1 down to 0 using _dead_cycles cache.
           For each live cycle k: check [BOS_k, CTS_k] imbalance with fill to
           current_candle (Interpretation B). If all filled, mark dead and stop.
        3. Determine earliest_x (smallest cycle in contiguous run ending at target).
        4. Apply state transition:
           - earliest_x == target: cross fails; deactivate active cross;
             activate single (established phase only).
           - earliest_x < target, no active cross: create new cross (v0).
           - earliest_x < target, active cross same x: extend anchor in place.
           - earliest_x < target, active cross earlier x: shrink (deactivate old,
             create new with version+1).
        """
        phase = self._m15_phase.get((sid, target_cycle))

        # Steps 1–2 (own-imbalance test + dead-cycle backward walk) are the
        # shared routine (CROSS_CYCLE_FIB_SPEC.md §11a). `fill_as_of="current"`
        # = subordinate semantics (each prior cycle checked to the current
        # candle, Interpretation B). The dead_cycles cache is passed by
        # reference so the walk's memo persists across candles exactly as
        # before. `bos_by_cycle`/`cts_by_cycle` are sliced to this sid over the
        # walk range [0, target_cycle).
        elig = resolve_cross_cycle_eligibility(
            df=df,
            target_cycle=target_cycle,
            sd=sd,
            own_window_end_idx=own_window_end_idx,
            own_imb_start=own_imb_start,
            anchor_idx=anchor_idx,
            anchor_price=anchor_price,
            bos_by_cycle={
                k: self._bos_by_cycle[(sid, k)]
                for k in range(target_cycle)
                if (sid, k) in self._bos_by_cycle
            },
            cts_by_cycle={
                k: self._cts_by_cycle[(sid, k)]
                for k in range(target_cycle)
                if (sid, k) in self._cts_by_cycle
            },
            dead_cycles=self._dead_cycles.setdefault(sid, set()),
            fill_threshold=self.config.fill_threshold,
            fill_as_of="current",
            evaluated_at=self._evaluated_at,
            fill_horizon_idx=current_candle,
            snapshot_horizon_idx=None,
        )

        if not elig.own_has:
            self._deactivate_active_cross(sid, target_cycle, current_candle, "own_imb_filled")
            # Pre-established: no fallback. Established: existing single fib
            # (if any) is managed by _update_fib_cts elsewhere — nothing to do here.
            return

        earliest_x = elig.earliest_x

        # Step 3: act on result
        active_cross = self._get_latest_cross(sid, target_cycle)

        if earliest_x == target_cycle:
            # No prior cycle eligible → cross fails
            if active_cross is not None and active_cross[1].active:
                self._deactivate_cross(active_cross[0], current_candle, "cross_failed")
            if phase == "pre_established":
                return
            # Established / confirmed: fall back to single fib
            self._activate_or_update_single_m15(
                sid, target_cycle, sd, anchor_idx, anchor_price, df, current_candle
            )
            return

        # Cross eligible (earliest_x < target_cycle)
        bos_x = self._bos_by_cycle.get((sid, earliest_x))
        if bos_x is None:
            return  # safety guard (shouldn't happen)

        # `active_cross` is the LATEST version (active or not — _get_latest_cross
        # ignores the active flag). Session 2 (FIB_LIFECYCLE_SPEC §4):
        # a version is identified by its START ANCHOR. Same-anchor revival
        # reactivates the existing version IN PLACE; a new version is created
        # ONLY when the start anchor actually moves (true shrink) or when no
        # version exists yet.
        if active_cross is None:
            # No version yet — create v0.
            self._m15_create_cross(
                sid, target_cycle, sd, bos_x, anchor_idx, anchor_price,
                earliest_x, current_candle
            )
            return

        active_x = int(active_cross[1].meta.get("cross_start_cycle", earliest_x))

        if not active_cross[1].active:
            # Revival of an inactive version (it was deactivated by
            # own_imb_filled). Same start anchor -> reactivate IN PLACE (no new
            # version). Start anchor moved while inactive -> the geometry truly
            # changed, so spawn a new version.
            if earliest_x == active_x:
                self._m15_reactivate_cross_in_place(
                    active_cross[0], active_cross[1],
                    anchor_idx, anchor_price, current_candle
                )
            else:
                self._m15_create_cross(
                    sid, target_cycle, sd, bos_x, anchor_idx, anchor_price,
                    earliest_x, current_candle
                )
            return

        # Active version: compare earliest_x to its start cycle.
        if earliest_x == active_x:
            # Same start — update anchor in place (extension)
            self._m15_extend_cross_anchor(
                active_cross[0], active_cross[1], anchor_idx, anchor_price
            )
        elif earliest_x > active_x:
            # Shrink — deactivate old version, create new
            self._deactivate_cross(active_cross[0], current_candle, "cross_shortened")
            self._m15_create_cross(
                sid, target_cycle, sd, bos_x, anchor_idx, anchor_price,
                earliest_x, current_candle
            )
        else:
            # earliest_x < active_x — shouldn't happen (dead cycles monotonic)
            print(f"[fib][cross_cycle][warn] unexpected earliest_x={earliest_x} "
                  f"< active_x={active_x} for sid={sid} cycle={target_cycle}")

    def _m15_create_cross(
        self,
        sid: int,
        target_cycle: int,
        sd: int,
        bos_x: tuple,
        anchor_idx: int,
        anchor_price: float,
        earliest_x: int,
        current_candle: int,
    ) -> None:
        """Create a new cross fib at (sid, target_cycle, 'cross', version)."""
        bos_x_idx, bos_x_price = bos_x
        version = self._cross_version.get((sid, target_cycle), -1) + 1
        key = (sid, target_cycle, "cross", version)

        if sd == 1:
            anchor_high = anchor_price
            anchor_low = bos_x_price
        else:
            anchor_high = bos_x_price
            anchor_low = anchor_price

        fib = create_fib_retracement(
            anchor_high=anchor_high,
            anchor_low=anchor_low,
            direction=sd,
            levels=self.config.fib_levels,
            meta={"structure_id": sid, "cycle_id": target_cycle},
        )

        state = FibState(
            structure_id=sid,
            cycle_id=target_cycle,
            struct_direction=sd,
            bos_idx=bos_x_idx,
            bos_price=bos_x_price,
            cts_idx=anchor_idx,
            cts_price=anchor_price,
            active=True,
            locked=False,
            fib=fib,
            meta={
                "fib_mode": "cross_cycle",
                "cross_cycle": True,
                "cross_start_cycle": earliest_x,
                "version": version,
                "activated_at": current_candle,
            },
            cts_history=((anchor_idx, anchor_price),),
        )
        self._fibs[key] = state
        self._cross_version[(sid, target_cycle)] = version
        self._current_cycle[sid] = target_cycle

        # Obsolete all prior-cycle fibs (single + cross versions). The creation
        # idx is the obsoleted cycles' terminal (Option A early-end via
        # set-if-absent).
        self._obsolete_prev_cycle_all_fibs(sid, target_cycle, end_idx=current_candle)

        # Lifecycle: record the cycle's sticky first-active idx (§15.3). v0 sets
        # it; a later version (shrink/revival) is a handoff that keeps the cycle
        # alive — set-if-absent keeps v0's earlier start.
        self._mark_first_active(sid, target_cycle, current_candle)

        print(f"[fib] cross_cycle sid={sid} CROSS ({earliest_x}->{target_cycle}) "
              f"v{version} ACTIVATED: bos_x_idx={bos_x_idx} -> anchor_idx={anchor_idx}")

    def _m15_extend_cross_anchor(
        self,
        key: tuple,
        state: FibState,
        new_anchor_idx: int,
        new_anchor_price: float,
    ) -> None:
        """Extend a cross fib's CTS anchor in place (same version)."""
        if new_anchor_idx <= state.cts_idx:
            return  # anchor not advancing

        sd = state.struct_direction
        if sd == 1:
            anchor_high = new_anchor_price
            anchor_low = state.bos_price
        else:
            anchor_high = state.bos_price
            anchor_low = new_anchor_price

        new_fib = create_fib_retracement(
            anchor_high=anchor_high,
            anchor_low=anchor_low,
            direction=sd,
            levels=self.config.fib_levels,
            meta={"structure_id": state.structure_id, "cycle_id": state.cycle_id},
        )
        new_history = state.cts_history + ((new_anchor_idx, new_anchor_price),)
        self._fibs[key] = replace(
            state,
            cts_idx=new_anchor_idx,
            cts_price=new_anchor_price,
            fib=new_fib,
            cts_history=new_history,
        )
        v = state.meta.get("version", 0)
        print(f"[fib] cross_cycle sid={state.structure_id} CROSS v{v} EXTENDED: "
              f"cts idx={new_anchor_idx}")

    def _m15_reactivate_cross_in_place(
        self,
        key: tuple,
        state: FibState,
        new_anchor_idx: int,
        new_anchor_price: float,
        current_candle: int,
    ) -> None:
        """Reactivate an inactive cross version in place (SAME version).

        Session 2 (FIB_LIFECYCLE_SPEC §4): a same-start-anchor revival (the
        version was deactivated by own_imb_filled and its imbalance condition
        reformed) reactivates the existing version rather than spawning a new
        one. The CTS anchor is advanced if the running extreme moved on. The
        cycle's sticky `start_idx` is unchanged (§15.3) — only per-record
        `active` flips back to True.
        """
        sd = state.struct_direction
        advanced = new_anchor_idx > state.cts_idx
        anchor_idx = new_anchor_idx if advanced else state.cts_idx
        anchor_price = new_anchor_price if advanced else state.cts_price

        if sd == 1:
            anchor_high = anchor_price
            anchor_low = state.bos_price
        else:
            anchor_high = state.bos_price
            anchor_low = anchor_price

        new_fib = create_fib_retracement(
            anchor_high=anchor_high,
            anchor_low=anchor_low,
            direction=sd,
            levels=self.config.fib_levels,
            meta={"structure_id": state.structure_id, "cycle_id": state.cycle_id},
        )

        new_history = (
            state.cts_history + ((anchor_idx, anchor_price),)
            if advanced else state.cts_history
        )
        self._fibs[key] = replace(
            state,
            active=True,
            cts_idx=anchor_idx,
            cts_price=anchor_price,
            fib=new_fib,
            cts_history=new_history,
            meta={**state.meta, "reactivated_at": current_candle},
        )
        v = state.meta.get("version", 0)
        # Reactivation does NOT move start_idx (sticky, §15.3) — the cycle's
        # first-active was recorded at creation.
        print(f"[fib] cross_cycle sid={state.structure_id} cycle={state.cycle_id} "
              f"CROSS v{v} REACTIVATED in place: cts idx={anchor_idx}")

    def _deactivate_cross(self, key: tuple, current_candle: int, reason: str) -> None:
        """Mark a cross fib version inactive (per-record `active=False`).

        Session 2 note (FIB_LIFECYCLE_SPEC §9.1/§9.2): unlike the cycle-TERMINAL
        paths (new_cycle / scenario1_revert / lifecycle_end / reversal — which
        stop setting `active=False` and use end_idx/end_reason instead), this
        helper DELIBERATELY keeps clearing the *per-version* `active` flag for
        ALL reasons:
          - `own_imb_filled` — a reversible CONDITION flip (the version's
            imbalance condition went false); reactivates in place later.
          - `cross_failed` / `cross_shortened` — a VERSION-INTERNAL supersede
            (the cycle stays alive via the single fallback / the next version).
            The superseded version must read `active=False` so the per-record
            chart gate `(active OR locked)` drops it (the "dead-version trail"
            in §9.2 cat 1). This is per-record version liveness, NOT a
            cycle-level terminal, so it does NOT call `_set_terminal`.
        """
        state = self._fibs.get(key)
        if state is None or not state.active:
            return
        self._fibs[key] = replace(
            state,
            active=False,
            meta={
                **state.meta,
                "deactivated_by": reason,
                "deactivated_at": current_candle,
            },
        )
        v = state.meta.get("version", 0)
        # Per-record `active=False` is the only state change here (it drives the
        # per-record chart gate / version distinction — §9.1). None of these
        # reasons moves the cycle's sticky start_idx.
        print(f"[fib] cross_cycle sid={state.structure_id} cycle={state.cycle_id} "
              f"CROSS v{v} DEACTIVATED: {reason}")

    def _deactivate_active_cross(
        self,
        sid: int,
        cycle_id: int,
        current_candle: int,
        reason: str,
    ) -> None:
        latest = self._get_latest_cross(sid, cycle_id)
        if latest is not None and latest[1].active:
            self._deactivate_cross(latest[0], current_candle, reason)

    def _activate_or_update_single_m15(
        self,
        sid: int,
        target_cycle: int,
        sd: int,
        anchor_idx: int,
        anchor_price: float,
        df: pd.DataFrame,
        current_candle: int,
    ) -> None:
        """Activate or update the single fib for target_cycle when cross fails
        (established phase fallback). BOS is the confirmed BOS_{target_cycle}."""
        bos = self._bos_by_cycle.get((sid, target_cycle))
        if bos is None:
            return  # shouldn't happen in established phase
        key = (sid, target_cycle)
        existing = self._fibs.get(key)
        if existing is None or not existing.active:
            # Activate new single fib. If a cross was created for this cycle
            # earlier, `_mark_first_active` (set-if-absent) keeps that earlier
            # start — a cross->single handoff keeps the cycle's original start
            # (§15.3 / §6 "start belongs to the identity").
            self._activate_fib(
                sid=sid,
                cycle_id=target_cycle,
                sd=sd,
                bos_idx=bos[0],
                bos_price=bos[1],
                cts_idx=anchor_idx,
                cts_price=anchor_price,
                meta={
                    "fib_mode": "cross_cycle",
                    "via": "cross_failed",
                    "activated_at": current_candle,
                },
            )
        else:
            # Extend existing single fib anchor + re-check imbalance
            self._update_fib_cts(key, anchor_idx, anchor_price, df)

    def get_active_fib(self, structure_id: int) -> Optional[FibState]:
        """Get the current active Fib for a structure.

        Prefers the highest-version active cross fib over the single-fib entry
        for the current cycle.
        """
        cycle_id = self._current_cycle.get(structure_id)
        if cycle_id is None:
            return None
        # Prefer active cross fib if present
        latest = self._get_latest_cross(structure_id, cycle_id)
        if latest is not None and latest[1].active:
            return latest[1]
        state = self._fibs.get((structure_id, cycle_id))
        if state and state.active:
            return state
        return None

    def get_all_fibs(self) -> List[FibState]:
        """Get all Fib states (including historical)."""
        return list(self._fibs.values())

    def get_fibs_for_charting(self) -> List[FibState]:
        """
        Get Fibs for charting / POI derivation - current state per record.

        Session 2 (FIB_LIFECYCLE_SPEC §10.3): excludes records whose derived
        `status` is "disappeared" — the uniform terminal-INVALIDATION filter
        (today only Scenario 1 revert qualifies). Replaces the old
        scenario1_revert-specific `deactivated_by` meta check; equivalent for
        the revert case (revert -> end_reason "scenario1_revert" -> status
        "disappeared") but generalizes to any future invalidation-class
        terminal, for both POI and chart consumers. Relies on
        `_finalize_lifecycle_fields()` having run first (the orchestrator calls
        it before this).
        """
        return [
            fib for fib in self._fibs.values()
            if fib.status != "disappeared"
        ]
