# engine_v2/zones/fib_tracker.py
"""
Fibonacci level tracking for POI Zones.

Fib levels are drawn from BOS to CTS within each market structure cycle.
This module handles activation, updates, and lifecycle management.

ONE logic for every structure — an H1 sid or a unique sub (PLAN_I S4, 2026-10-08; canonical
`CROSS_CYCLE_FIB_SPEC.md`): cycle 0 a single; a cycle >= 1 within the structure's cross-fib LIMIT
a cross back across the live prior cycles (the dead-cycle walk, versions, the pre-established
phase), else a single. The retired H1-main logic (Scenario 1 / 2 / 3, the revert, the ceiling on
the previous BOS zone's outer) is recorded in that spec's "Retired logic (history)" appendix.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import List, Mapping, Optional, Dict, Any

import pandas as pd

from engine_v2.features.fibonacci import (
    FibRetracement,
    create_fib_retracement,
    DEFAULT_FIB_LEVELS,
)
from engine_v2.patterns.imbalance import has_unfilled_imbalance
from engine_v2.structure import event_fields as ef
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.zones.cross_cycle_fib import CrossFibLimit, beyond, resolve_cross_cycle_eligibility
from engine_v2.zones.structure_lifecycle import (
    compute_cycle_lifecycle,
    compute_reversal_idx_by_sid,
    compute_struct_start_by_sid,
)

# Lifecycle convention (FIB_LIFECYCLE_SPEC.md section 10): terminal reasons
# that mean "invalidated / retracted" rather than ordinary historical end.
# A cycle whose terminal carries one of these derives status="disappeared"
# (terminal AND suppressed). EMPTY since PLAN_I S4 (the Scenario 1 revert, its
# only producer, is retired): `disappeared` stays as a generic hook (R6-3 (v)).
_INVALIDATION_END_REASONS = frozenset()


class FibEventOrderError(RuntimeError):
    """The tracker was fed cycle T's CTS_ESTABLISHED (T >= 1) before cycle T−1's CTS_CONFIRMED (PLAN_I §9.2
    F14 (1), R6-4 — the order tripwire RAISES so the shape is seen). MS establishes cycle T only after T−1's
    confirmation in EMISSION order; the downstream consumes `processing_order_key` order (CTS_ESTABLISHED at its
    ANCHOR), so a T−1 confirmation inside T's [anchor, moment) inverts them — the cross walk would miss T−1 and
    its confirmation would reset T's phase. 0 / 34 on the reference window. A `RuntimeError`, so a sub
    geometry's `except (ValueError, IndexError)` never swallows it.

    Also raised by the downstream dispatch (`orchestrator._run_downstream_pipeline`) for a CTS_ESTABLISHED handled
    before its own cycle's BOS_CONFIRMED (PLAN_I §9.2 F17, R8-2; it used to be dropped silently): a cycle whose BOS
    anchor lies after its CTS anchor — which every F14 (1) case found by the random-tail search was, so this is the
    check that sees that shape through the dispatch.

    And raised by `FibTracker.check_processing_order` at the end of every MarketStructure run (PLAN_I S5, R9-1): MS's
    embedded tracker saw an EMISSION order other than `processing_order_key` order — the F16 shape (a CTS confirmed
    with an anchor at or after its confirmation candle) and the F17 (i) / F14 (1) shape (a cycle established with an
    anchor before the previous cycle's confirmation). F17 (ii) — a BOS anchor after its own CTS anchor — does NOT trip
    MS (both events share their moment); only the dispatch raises on it. 0 on the reference window."""


class CrossFibLimitError(RuntimeError):
    """The tracker was asked whether a cross may target a cycle ≥ 1 of a structure it holds no
    cross-fib limit for (PLAN_I §9.3 D1: the limit is REQUIRED from S3 — there is no silent "no P"). A
    `RuntimeError`, so a sub geometry's `except (ValueError, IndexError)` never swallows it."""


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
    # alongside fib's own terminals (new_cycle / Option A early-end) — §15.4.
    end_idx: Optional[int] = None
    end_reason: Optional[str] = None
    # Derived cycle-identity label in {active, inactive, ended, disappeared} (§15.5).
    status: str = "active"


# The ONE imbalance fill threshold of the fib decision (70 % = filled): the downstream tracker
# (`orchestrator._run_downstream_pipeline`) and MarketStructure's embedded tracker
# (`structure_engine._make_market_structure`, PLAN_I S5 / D5) both build their config with it, and MS's in-flight
# IP pick reads the tracker's value (`poi_zones.poi_inner_for_fib`).
FIB_FILL_THRESHOLD = 0.70


@dataclass
class FibTrackerConfig:
    """Configuration for FibTracker."""
    fib_levels: List[float] = field(default_factory=lambda: DEFAULT_FIB_LEVELS.copy())
    fill_threshold: float = FIB_FILL_THRESHOLD  # 70% = filled


class FibTracker:
    """
    Tracks Fibonacci levels across market structure cycles.

    Lifecycle:
    1. CTS_ESTABLISHED + unfilled imbalance → Fib activated
    2. CTS moves to new extreme → anchor 2 updates
    3. All imbalances filled → Fib deactivates
    4. CTS_CONFIRMED → Fib locked (stops updating)

    ONE logic for every structure (PLAN_I S4; `CROSS_CYCLE_FIB_SPEC.md`): cycle 0 a single; a cycle >= 1 within
    the structure's cross-fib limit (`_cross_allowed`) a cross over the live prior cycles (the dead-cycle walk,
    versioning, the pre-established phase on CTS_THRESHOLD_UPDATED), falling back to a single (`cross_failed`); a
    cycle past the limit a single (`via: "cross_fib_limit"`).

    Only 1 active Fib per structure at a time (new cycle obsoletes previous).

    Two instances run per structure (PLAN_I S5 / D5): the downstream's (`orchestrator._run_downstream_pipeline`,
    fed in `processing_order_key` order — the drawn fib) and MarketStructure's EMBEDDED one
    (`structure_engine._make_market_structure`, `quiet=True`), fed each CTS event as MS emits it; MS's in-flight
    POI read takes its `representative_fib`, and `check_processing_order` (called at the end of every MS run) raises
    when the emission order it saw is not the processing order (PLAN_I §9.2 F14 / F16 — the two then agree).
    """

    def __init__(
        self,
        config: Optional[FibTrackerConfig] = None,
        *,
        cross_fib_limits: Optional[Mapping[int, CrossFibLimit]] = None,
        quiet: bool = False,
    ):
        self.config = config or FibTrackerConfig()
        # No `[fib]` log lines (MarketStructure's embedded tracker, PLAN_I D5: every geometry / probe MS would flood
        # run.log with them; the downstream tracker logs the same decisions).
        self._quiet = bool(quiet)
        # Every event a public handler was fed, in feed order — `check_processing_order`'s input.
        self._handled: List[StructureEvent] = []

        # Fib states per key. Two key shapes coexist in this dict:
        #   - (sid, cycle_id)                          → single fib
        #   - (sid, cycle_id, "cross", version)        → cross fib
        # `_fibs.values()` yields all fibs for charting iteration.
        self._fibs: Dict[tuple, FibState] = {}

        # Track which cycle is "current" per structure (for obsolescence)
        # {structure_id: cycle_id}
        self._current_cycle: Dict[int, int] = {}

        # ------------------------------------------------------------------
        # Cross-fib state
        # ------------------------------------------------------------------
        # Per (sid, cycle_id): phase ∈ {"pre_established", "established", "confirmed"}.
        # Lifecycle: implicit → pre_established (on CTS_{n-1} CONFIRMED)
        #            → established (on CTS_n ESTABLISHED)
        #            → confirmed (on CTS_n CONFIRMED).
        # Cycle 0 has no pre_established phase.
        self._phase: Dict[tuple, str] = {}

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
        # (sid, cycle) -> the cycle's BOS MOMENT (== its CTS_ESTABLISHED moment;
        # Plan E E3a′): where a late-starting fib of the cycle ends the previous
        # cycle (`_prev_cycle_end_at`, PLAN_I S3b).
        self._bos_moment_by_cycle: Dict[tuple, int] = {}

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
        # Cycle-identity lifecycle-START. Keyed (sid, cycle_id) -> first
        # active idx. Set-if-absent (sticky — the first activation wins, never
        # moved by a later reactivation). Replaces the removed activation_history:
        # "key present" ⟺ "was ever active" (§15.3).
        self._first_active: Dict[tuple, int] = {}

        # Cycle-identity terminal axis. Keyed (sid, cycle_id) -> (end_idx,
        # end_reason). Set-if-absent (first/earliest end wins, irreversible).
        # Populated for tracker-controlled terminals: new_cycle (obsolete); the
        # Option A early-end flows through the obsolete path. Reversal /
        # lifecycle_end are "passed-through" terminals (spec §7): the H1-main
        # reversal terminal is wired via
        # `set_reversal_terminals`; the sub lifecycle_end is set by the external
        # cap in entity_df_mutation (and, once §15.6/(b) lands, by the cycle
        # pass-through end fed in as a candidate).
        self._terminal: Dict[tuple, tuple] = {}

        # The MOMENT of the event being handled (Plan F — IMBALANCE_FILL_SEMANTICS
        # "Knowability — the c3 rule"): every imbalance question a handler asks
        # counts only gaps formed by then. Set by `_evaluating` around the three
        # handlers that reach an imbalance read; None outside a handler (every
        # handled event records a moment since Plan E E3·0 — before it, None
        # inside one on a pattern-path CTS_UPDATED).
        self._evaluated_at: Optional[int] = None
        self._in_event = False

        # The cross-fib LIMIT per structure (PLAN_I §9.3 D1 / D2): `{sid: CrossFibLimit}` — `P` + its source
        # (+ the sub that supplied it, + main's `started_by`), fixed at the structure's start. Stamped onto every
        # fib record at finalize (S2); APPLIED by `_cross_allowed` (subs since S3, main since S4). A sid without
        # an entry gets no `cross_fib_limit_*` keys — and raises `CrossFibLimitError` when a cycle >= 1 consults it.
        self._limits: Dict[int, CrossFibLimit] = dict(cross_fib_limits or {})
        # (sid, cycle_id) -> (idx, price, sd): the cycle's LATEST CTS anchor (§4 N-f) — written at its
        # CTS_ESTABLISHED, every CTS_UPDATED and its CTS_CONFIRMED (the confirmed anchor wins). `M` reads it.
        self._latest_anchor: Dict[tuple, tuple] = {}

    def _log(self, msg: str) -> None:
        if not self._quiet:
            print(msg)

    # ------------------------------------------------------------------
    # MarketStructure's in-flight read (PLAN_I S5 / D5, Q7).
    # ------------------------------------------------------------------
    def representative_fib(self, sid: int, cycle_id: int) -> Optional[FibState]:
        """The fib MS's in-flight POI read uses for `(sid, cycle_id)`: the cycle's ACTIVE latest cross version,
        else its ACTIVE single, else None (no fib → the proximity check reads the BOS inner only). The same
        decision the drawn fib records (one tracker logic, fed the same events)."""
        latest = self._get_latest_cross(int(sid), int(cycle_id))
        if latest is not None and latest[1].active:
            return latest[1]
        single = self._fibs.get((int(sid), int(cycle_id)))
        if single is not None and single.active:
            return single
        return None

    def check_processing_order(self) -> None:
        """The end-of-run order tripwire (PLAN_I §9.3 D5, R6-4, R9-1): the events this tracker was fed, in feed
        order, must be in `processing_order_key` order — the order the downstream tracker (the drawn fib) handles
        them in. MarketStructure feeds its embedded tracker in EMISSION order and calls this at the end of every
        run; the two orders differ on the F16 shape (a CTS confirmed with an anchor at or after its confirmation
        candle — a tie sorts the confirmation first) and the F17 (i) / F14 (1) shape (a cycle's establishment whose
        anchor precedes the previous cycle's confirmation); F17 (ii) — a BOS anchor after its own CTS anchor — does
        not trip it (only the downstream dispatch raises). 0 on the reference window, and on 400 random real slices
        (S5 landing review). Raises `FibEventOrderError` naming the first event out of place."""
        handled = self._handled
        expected = sorted(handled, key=ef.processing_order_key)
        for k, (got, want) in enumerate(zip(handled, expected)):
            if got is not want:
                raise FibEventOrderError(
                    f"[fib] emission order != processing order at handled event #{k}: fed {got.type} "
                    f"(cycle {got.meta.get('cycle_id')}, idx {int(got.idx)}, sort key {ef.processing_order_key(got)})"
                    f" where processing order has {want.type} (cycle {want.meta.get('cycle_id')}, idx "
                    f"{int(want.idx)}, sort key {ef.processing_order_key(want)}) — PLAN_I §9.2 F16 / F17 (i); the MS "
                    f"fix is planned (R9-1 / R9-2)")

    # ------------------------------------------------------------------
    # The cross-fib limit (PLAN_I §9.3 D1): M and the record stamp.
    # ------------------------------------------------------------------
    def cross_fib_limit_cycle(self, sid: int) -> Optional[int]:
        """`M` (`cross_fib_limit_cycle`, §4 N-a): the first cycle whose latest CTS anchor is STRICTLY beyond the
        structure's `P` (`beyond`, Q2) = the last cycle a cross may target; None without `P` or while no cycle
        has gone beyond. Read off the latest anchors as they stand — after the stream, the structure's FINAL `M`
        (R6-6; a CTS anchor never regresses, so the first cycle beyond stays the first)."""
        lim = self._limits.get(int(sid))
        if lim is None or lim.price is None:
            return None
        for c in sorted(c for (s, c) in self._latest_anchor if s == int(sid)):
            _idx, price, sd = self._latest_anchor[(int(sid), c)]
            if beyond(price, lim.price, sd):
                return c
        return None

    def _cross_allowed(self, sid: int, target_cycle: int) -> bool:
        """May a cross target `target_cycle` (≥ 1) of `sid`? (PLAN_I §9.3 D1, §1.4.) Only with a reference price
        `P`, and only up to `M` — the first cycle whose latest CTS anchor is strictly beyond `P` is the LAST cycle a
        cross may target: allowed ⟺ `P` set and no cycle k < `target_cycle` is beyond (M is None or target ≤ M).
        A cycle past it — and every cycle ≥ 1 of a structure without `P` — gets the single (`via:
        "cross_fib_limit"`). The cycles k < target are confirmed before the target exists (MS order; the S2
        tripwire), so the answer is fixed for the target's whole life. No limit entry → `CrossFibLimitError`."""
        lim = self._limits.get(int(sid))
        if lim is None:
            raise CrossFibLimitError(
                f"[fib] sid={sid} cycle={target_cycle}: no cross_fib_limit entry (PLAN_I D1: required from S3)")
        if lim.price is None:
            return False
        m = self.cross_fib_limit_cycle(int(sid))
        return m is None or int(target_cycle) <= m

    def _cross_fib_limit_meta(self, sid: int) -> Dict[str, Any]:
        """The keys every fib record of `sid` carries (PLAN_I S2, §9.3 D1): `cross_fib_limit_price` /
        `_source` / `_cycle` (the FINAL `M`, R6-6) / `_from_sub_id` (the start-numbered sub that supplied `P`),
        and main's `started_by` (a sub's comes from the mirror). Empty for a sid without a limit entry."""
        lim = self._limits.get(int(sid))
        if lim is None:
            return {}
        out: Dict[str, Any] = {
            "cross_fib_limit_price": lim.price,
            "cross_fib_limit_source": lim.source,
            "cross_fib_limit_cycle": self.cross_fib_limit_cycle(int(sid)),
            "cross_fib_limit_from_sub_id": lim.from_sub_id,
        }
        if lim.started_by is not None:
            out["started_by"] = lim.started_by
        return out

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

    def _moment(self) -> int:
        """The handled event's MOMENT (Plan E E3a): the TIME half of every
        handler — activation / reactivation / deactivation stamps, fill
        horizons. Only inside a handler."""
        if not self._in_event or self._evaluated_at is None:
            raise RuntimeError("FibTracker._moment(): no handled event")
        return self._evaluated_at


    # ------------------------------------------------------------------
    # Lifecycle convention helpers (FIB_LIFECYCLE_SPEC.md §15).
    # ------------------------------------------------------------------
    def _mark_first_active(self, sid: int, cycle_id: int, idx: int) -> None:
        """Record the cycle's first-active idx (lifecycle start), set-if-absent.

        Sticky: the first activation wins and is never moved by a later
        reactivation, mirroring how a cycle/structure lifecycle-start doesn't
        shift when a zone flickers inactive→active (§15.3). Called only from the
        two creation sites (`_activate_fib`, `_create_cross`) — the first
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
        self, reversal_idx_by_new_sid: Dict[int, int],
    ) -> None:
        """Wire the reversal terminal (FIB_LIFECYCLE_SPEC §7 passed-through end).

        `reversal_idx_by_new_sid` maps NEW sid -> the REALISED reversal idx
        (`STATE_CHANGED` to=reversal via `structure_lifecycle.
        compute_reversal_idx_by_sid`, shifted to the sid it birthed — never a
        `REVERSAL_CANDIDATE`'s scheduled apply, which a watch expiry can discard),
        ending the PREVIOUS sid = new_sid - 1. For each, stamp the reversal terminal on every cycle of
        the ended (prev) sid. `_set_terminal` is set-if-absent, so cycles
        already terminated by `new_cycle` keep their earlier (correct) end and
        only the sid's still-open final cycle picks up "reversal".

        Closes the Session-1 deferral (reversal-ended H1-main fibs previously
        kept end_idx=None). Must run BEFORE `_finalize_lifecycle_fields` so the
        derived `status` reflects the terminal. For a sub (one sid 0, events
        clipped at its cap) the terminal is its realised reversal R <= the
        lifecycle cap — R == the cap when the sub's `end_reason` is reversal
        (the finalize pass-through, FIB_LIFECYCLE_SPEC §15).
        """
        if not reversal_idx_by_new_sid:
            return
        cycles_by_sid: Dict[int, set] = {}
        for key in self._fibs:
            if isinstance(key, tuple) and len(key) >= 2:
                cycles_by_sid.setdefault(int(key[0]), set()).add(int(key[1]))
        for new_sid, rv_idx in reversal_idx_by_new_sid.items():
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
        written (via dataclasses.replace) — `active`/`locked`/`fib`/anchors are
        untouched — plus, since PLAN_I S2, the structure's `cross_fib_limit_*`
        keys (+ main `started_by`) appended to `meta` (`_cross_fib_limit_meta`).

        When `events` is supplied (§15.6, part b), fib is put on the shared
        `structure_lifecycle` helper exactly like KL/POI:
          - `struct_floor[sid]` (`compute_struct_start_by_sid`) CLAMPS each
            cycle's `start_idx` up to the structure/parent floor (§15.3). NB the
            clamp is to the STRUCTURE floor, not the full cycle-start, so the §6
            pre-established early start survives.
          - the cycle pass-through end (`compute_cycle_lifecycle`) is fed into
            `_set_terminal` as an earliest-wins candidate (§15.4): set-if-absent,
            so fib's own earlier terminals (new_cycle / Option A / reversal) win, and the table end only fills cycles with no earlier
            terminal (the open last cycle / the subordinate cap).
          - a fib whose cycle has NO table row (a pre-established cross for a next
            cycle that never establishes) gets `lifecycle_cap` as its candidate
            (§15.4 candidate 3, 2026-09-28); `None` cap = none.
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
            # A fib whose cycle has no CTS_ESTABLISHED has no `cycle_life` row: the
            # pre-established cross (FIB_LIFECYCLE_SPEC §6; every structure since PLAN_I S4) pre-created for a next cycle
            # that never establishes. The structure's end still ends it — the reversal via
            # `set_reversal_terminals`, the subordinate CAP here (§15.4 candidate 3;
            # user 2026-09-28, "cap like the reversal"; restores the cap the removed
            # post-hoc loop gave it, §15.6). `poi_zones` then builds no POIs on it (an
            # ended, unlocked fib), exactly as after a reversal.
            if lifecycle_cap is not None:
                for s, c in {(int(k[0]), int(k[1])) for k in self._fibs
                             if isinstance(k, tuple) and len(k) >= 2}:
                    if (s, c) not in cycle_life:
                        self._set_terminal(s, c, int(lifecycle_cap), cap_reason)

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

            limit_meta = self._cross_fib_limit_meta(int(sid))
            for key in keys:
                fib = self._fibs[key]
                self._fibs[key] = replace(
                    fib,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    end_reason=end_reason,
                    status=status,
                    meta={**fib.meta, **limit_meta},
                )

    def on_cts_established(
        self,
        event: StructureEvent,
        df: pd.DataFrame,
        bos_idx: int,
        bos_price: float,
    ) -> Optional[FibState]:
        """Handle CTS_ESTABLISHED (body: `_on_cts_established`). Its imbalance
        questions are asked at the event's moment, `meta["confirmed_at"]`."""
        self._handled.append(event)
        with self._evaluating(event):
            return self._on_cts_established(event, df, bos_idx, bos_price)

    def _on_cts_established(
        self,
        event: StructureEvent,
        df: pd.DataFrame,
        bos_idx: int,
        bos_price: float,
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
        # The TIME half (Plan E E2b split; the moment since E3a): activation
        # stamps, fill horizons.
        cts_established_idx = self._moment()
        cts_price = float(event.price) if event.price else 0.0

        # Get CTS price from event or df
        if cts_price == 0.0 and cts_idx in df.index:
            if sd == 1:
                cts_price = float(df.loc[cts_idx, "h"])
            else:
                cts_price = float(df.loc[cts_idx, "l"])

        # The order tripwire (PLAN_I §9.2 F14 (1), R6-4): cycle T's establishment needs T−1 confirmed FIRST.
        if cycle_id >= 1 and (sid, cycle_id - 1) not in self._cts_by_cycle:
            raise FibEventOrderError(
                f"[fib] sid={sid} cycle={cycle_id} CTS_ESTABLISHED (anchor {cts_idx}, moment "
                f"{cts_established_idx}) handled before cycle {cycle_id - 1}'s CTS_CONFIRMED"
            )
        self._latest_anchor[(sid, cycle_id)] = (int(cts_idx), float(cts_price), int(sd))

        # Check for unfilled imbalance between BOS and CTS, counting only gaps
        # formed by the moment (Plan F). sd-direction filter: Fibs only ever
        # produce sd-direction POIs, so counter-direction imbalances in the
        # BOS->CTS swing don't justify activation.
        start_idx = min(bos_idx, cts_idx)
        end_idx = max(bos_idx, cts_idx)
        has_unfilled = self._has_unfilled(df, start_idx, end_idx, cts_established_idx, sd)

        # Populate BOS lookup (used by cross-fib walk-backward)
        self._bos_by_cycle[(sid, cycle_id)] = (bos_idx, bos_price)
        # The BOS's MOMENT (== the cycle's CTS_ESTABLISHED moment, by definition —
        # BOS_CONFIRMED and CTS_ESTABLISHED of a cycle share `confirmed_at`): where a
        # late-starting fib of the cycle ends the previous cycle (`_prev_cycle_end_at`).
        self._bos_moment_by_cycle[(sid, cycle_id)] = cts_established_idx

        return self._handle_cross_cycle_cts_established(
            sid, cycle_id, sd, bos_idx, bos_price, cts_idx, cts_price, has_unfilled, df,
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
        """Handle CTS_ESTABLISHED (every structure: an H1 sid or a unique sub, PLAN_I S4).

        - Cycle 0: single-fib only (no cross possible, no pre-established phase).
        - Cycle ≥1 past the cross-fib limit (`_cross_allowed` False — no `P`, or a cycle before it beyond `P`;
          PLAN_I D1, R6-1 (d)): the cycle-0 single, labelled `via: "cross_fib_limit"` (R7-3).
        - Cycle ≥1 within the limit: phase flips pre_established → established. Run cross-fib
          check with anchor = CTS_n. If cross fails, fall back to single.
        """
        # Flip phase to established
        self._phase[(sid, cycle_id)] = "established"

        if cycle_id == 0 or not self._cross_allowed(sid, cycle_id):
            # Cycle 0 — and a cycle past the limit — no cross: the single, activated now if its own
            # [BOS, CTS] holds an unfilled sd imbalance, else late on a CTS_UPDATED that finds one.
            meta = {"activated_at": cts_established_idx, "fib_mode": "cross_cycle"}
            if cycle_id >= 1:
                meta["via"] = "cross_fib_limit"
            if not has_unfilled:
                self._log(f"[fib] cross_cycle sid={sid} cycle={cycle_id} NO FIB: no unfilled imbalance "
                      f"(evaluated_at={self._evaluated_at})")
                return None
            self._log(f"[fib] cross_cycle sid={sid} cycle={cycle_id} ACTIVATED (single, no cross)")
            return self._activate_fib(
                sid=sid,
                cycle_id=cycle_id,
                sd=sd,
                bos_idx=bos_idx,
                bos_price=bos_price,
                cts_idx=cts_idx,
                cts_price=cts_price,
                meta=meta,
            )

        # Cycle ≥1: run cross-fib check with CTS_n as anchor
        # own_imb_start = BOS_n (just-confirmed BOS for this cycle)
        self._log(f"[fib] cross_cycle sid={sid} cycle={cycle_id} CTS_ESTABLISHED: "
              f"running cross check (anchor=CTS_n)")
        self._cross_check(
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






    def _activate_fib(
        self,
        sid: int,
        cycle_id: int,
        sd: int,
        bos_idx: int,
        bos_price: float,
        cts_idx: int,
        cts_price: float,
        meta: Dict,
    ) -> FibState:
        """Internal helper to create and store a SINGLE FibState at ``(sid, cycle_id)``
        (crosses: `_create_cross`). `meta["activated_at"]` (the activating event's moment)
        is required.

        Records the cycle's sticky lifecycle `start_idx` on first activation
        (§15.3) via `_mark_first_active`.
        """
        self._assert_no_active_cross(sid, cycle_id)
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

        activated_at = int(meta["activated_at"])
        prev_end = self._prev_cycle_end_at(sid, cycle_id, activated_at)

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
                # Lifecycle: prev cycle ends when this cycle activates — at this cycle's establishment at the
                # latest (a late activation, PLAN_I S3b F15).
                self._set_terminal(sid, prev_cycle, prev_end, "new_cycle")
            # A previous cycle's CROSS is not stamped here: crosses are obsoleted by
            # `_create_cross`'s `_obsolete_prev_cycle_all_fibs` at the next cross's
            # creation, and a single activated after a cross must not flip that cross's
            # end_reason `next_cycle` -> `new_cycle` (the retired h1-only versioned-cross
            # obsolete did — PLAN_I S4, §9.5 c: main's (1,1) now ends `next_cycle`).

        self._fibs[(sid, cycle_id)] = state
        self._current_cycle[sid] = cycle_id
        # Lifecycle: record the cycle's sticky first-active idx (§15.3).
        self._mark_first_active(sid, cycle_id, activated_at)

        self._log(f"[fib] sid={sid} cycle={cycle_id} ACTIVATED: BOS idx={bos_idx} price={bos_price:.5f} -> CTS idx={cts_idx} price={cts_price:.5f}")

        return state

    def on_cts_updated(
        self,
        event: StructureEvent,
        df: pd.DataFrame,
    ) -> Optional[FibState]:
        """Handle CTS_UPDATED (body: `_on_cts_updated`). Its imbalance questions
        are asked at the event's moment: `ev.idx` on the raw path; the apply candle
        `meta["confirmed_at"]` on the pattern path (Plan E E3·0)."""
        self._handled.append(event)
        with self._evaluating(event):
            return self._on_cts_updated(event, df)

    def _on_cts_updated(
        self,
        event: StructureEvent,
        df: pd.DataFrame,
    ) -> Optional[FibState]:
        """
        Handle CTS_UPDATED event - update CTS anchor if Fib is active.

        Parameters
        ----------
        event : StructureEvent
            The CTS_UPDATED event
        df : DataFrame
            OHLC data

        Returns
        -------
        FibState or None
            Updated FibState, or None if no active Fib for this cycle
        """
        sid = int(event.meta.get("structure_id", 0))
        cycle_id = int(event.meta.get("cycle_id", 0))
        sd = int(event.meta.get("struct_direction", 0))
        # The updated CTS ANCHOR (a location). Its time half (activation stamps,
        # fill horizons) is the update's moment,
        # `self._moment()` (Plan E E3a; the raw path's idx, the pattern path's
        # `confirmed_at`).
        cts_idx = ef.cts_anchor_idx(event)
        cts_price = float(event.price) if event.price else 0.0

        # Get CTS price from df if not in event
        if cts_price == 0.0 and cts_idx in df.index:
            if sd == 1:
                cts_price = float(df.loc[cts_idx, "h"])
            else:
                cts_price = float(df.loc[cts_idx, "l"])

        # The confirmed anchor wins: an update of an already-confirmed cycle (PLAN_I §9.2 F16 — MS can confirm a
        # CTS with an anchor AFTER its confirmation candle, so that cycle's earlier raw updates sort after the
        # confirmation) must not move it back, or `M` — and `_cross_allowed` — would change inside a later
        # cycle's life (S3 landing review m1). 0 on the reference window.
        if (sid, cycle_id) not in self._cts_by_cycle:
            self._latest_anchor[(sid, cycle_id)] = (int(cts_idx), float(cts_price), int(sd))

        return self._handle_cross_cycle_cts_updated(sid, cycle_id, sd, cts_idx, cts_price, df)

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
        extended CTS anchor. Cycle 0 — and a cycle past the cross-fib limit (PLAN_I D1) — uses the simple
        single-fib update (late activation included)."""
        if cycle_id == 0 or not self._cross_allowed(sid, cycle_id):
            # Cycle 0 / past the limit: no cross logic. Single-fib update.
            key = (sid, cycle_id)
            if key in self._fibs:
                return self._update_fib_cts(key, cts_idx, cts_price, df)
            # First-activation on a later CTS_UPDATED (2026-06-08): the cycle-0
            # Fib did NOT activate at CTS_ESTABLISHED (no unfilled imbalance
            # then), but an unfilled sd-direction imbalance may appear as the
            # CTS extends. Re-check over [BOS_0, cts_idx] as of the update's moment and
            # activate now if present (was one-shot for subs until 2026-06-08).
            # The same for a past-the-limit cycle's single (PLAN_I D1).
            bos = self._bos_by_cycle.get((sid, cycle_id))
            if bos is None:
                return None
            bos_idx, bos_price = bos
            start_idx = min(bos_idx, cts_idx)
            end_idx = max(bos_idx, cts_idx)
            if not self._has_unfilled(df, start_idx, end_idx, self._moment(), sd):
                return None
            self._log(f"[fib] cross_cycle sid={sid} cycle={cycle_id} ACTIVATED on update "
                  f"(unfilled imbalance found post-EST): BOS idx={bos_idx} -> "
                  f"CTS idx={cts_idx}")
            meta = {"activated_at": self._moment(), "fib_mode": "cross_cycle", "activated_on": "update"}
            if cycle_id >= 1:
                meta["via"] = "cross_fib_limit"
            return self._activate_fib(
                sid=sid,
                cycle_id=cycle_id,
                sd=sd,
                bos_idx=bos_idx,
                bos_price=bos_price,
                cts_idx=cts_idx,
                cts_price=cts_price,
                meta=meta,
            )

        # Cycle ≥1: should be in established phase. Re-run cross check.
        phase = self._phase.get((sid, cycle_id))
        if phase != "established":
            # Shouldn't happen — CTS_UPDATED implies ESTABLISHED has already fired
            self._log(f"[fib][cross_cycle][warn] CTS_UPDATED for sid={sid} cycle={cycle_id} "
                  f"in unexpected phase={phase}")
            return None

        bos = self._bos_by_cycle.get((sid, cycle_id))
        if bos is None:
            return None

        self._cross_check(
            sid=sid,
            target_cycle=cycle_id,
            df=df,
            sd=sd,
            own_window_end_idx=cts_idx,
            current_candle=self._moment(),
            anchor_idx=cts_idx,
            anchor_price=cts_price,
            own_imb_start=bos[0],
        )
        latest = self._get_latest_cross(sid, cycle_id)
        if latest is not None and latest[1].active:
            return latest[1]
        return self._fibs.get((sid, cycle_id))




    def _update_fib_cts(
        self,
        key: tuple,
        cts_idx: int,
        cts_price: float,
        df: pd.DataFrame,
    ) -> Optional[FibState]:
        """
        Internal helper to update a Fib's CTS anchor and check imbalance conditions.

        Single-keyed fibs only — every caller passes `(sid, cycle)`. Cross fibs live
        at 4-tuple keys `(sid, cycle, "cross", v)` and are re-checked in `_cross_check`
        (this helper's dead cross branch was deleted 2026-09-28).
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

        label = f"cycle={cycle_id}"
        self._log(f"[fib] sid={sid} {label} UPDATED: CTS idx={cts_idx} price={cts_price:.5f}")

        # Check unfilled imbalance condition - can reactivate or deactivate: the fib's
        # own range (sd-direction; formed gaps only — "all_imbalances_filled" below
        # means "no FORMED unfilled imbalance").
        start_idx = min(new_state.bos_idx, new_state.cts_idx)
        end_idx = max(new_state.bos_idx, new_state.cts_idx)
        has_unfilled = self._has_unfilled(df, start_idx, end_idx, self._moment(), sd)

        if has_unfilled and not new_state.active:
            # Reactivate - unfilled imbalances now exist in expanded range
            self._assert_no_active_cross(sid, cycle_id)
            new_state = replace(new_state, active=True, meta={**new_state.meta, "reactivated_at": self._moment()})
            self._log(f"[fib] sid={sid} {label} REACTIVATED: unfilled imbalance found at idx={cts_idx}")
        elif not has_unfilled and new_state.active:
            # Deactivate - all imbalances filled
            new_state = replace(new_state, active=False, meta={**new_state.meta, "deactivated_at": self._moment(), "reason": "all_imbalances_filled"})
            self._log(f"[fib] sid={sid} {label} DEACTIVATED: all imbalances filled at idx={cts_idx} "
                  f"(evaluated_at={self._evaluated_at})")

        self._fibs[key] = new_state
        return new_state



    def on_cts_confirmed(self, event: StructureEvent) -> Optional[FibState]:
        """
        Handle CTS_CONFIRMED event - lock the Fib; the next cycle enters its pre-established phase.

        Parameters
        ----------
        event : StructureEvent
            The CTS_CONFIRMED event

        Returns
        -------
        FibState or None
            Locked FibState, or None if no active Fib
        """
        self._handled.append(event)
        sid = int(event.meta.get("structure_id", 0))
        cycle_id = int(event.meta.get("cycle_id", 0))

        # Populate CTS lookup (used by cross-fib walk-backward for dead-cycle
        # evaluation). `cts_anchor_idx` is the CTS anchor (a location); ev.idx is
        # the confirmation candle (the moment).
        cts_anchor_idx = ef.cts_anchor_idx(event)
        cts_price_val = float(event.price) if event.price else 0.0
        self._cts_by_cycle[(sid, cycle_id)] = (cts_anchor_idx, cts_price_val)
        self._latest_anchor[(sid, cycle_id)] = (
            int(cts_anchor_idx), float(cts_price_val), int(event.meta["struct_direction"]))
        # Record confirmation candle (used to bound prospective-BOS search for n+1)
        self._cts_confirmed_idx[(sid, cycle_id)] = int(event.idx)

        return self._handle_cross_cycle_cts_confirmed(sid, cycle_id, event)

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
            self._log(f"[fib] cross_cycle sid={sid} cycle={cycle_id} CROSS v{v} LOCKED")

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
                self._log(f"[fib] cross_cycle sid={sid} cycle={cycle_id} SINGLE LOCKED")

        # Phase transitions
        self._phase[(sid, cycle_id)] = "confirmed"
        self._phase[(sid, cycle_id + 1)] = "pre_established"

        # Return whichever is currently active (preferring cross)
        if latest is not None and latest[1].locked:
            return self._fibs.get(latest[0])
        return self._fibs.get(single_key)






    # ------------------------------------------------------------------
    # Pre-established phase helpers
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




    def _prev_cycle_end_at(self, sid: int, cycle_id: int, at: int) -> int:
        """Where a fib of `cycle_id` activated / created at `at` ends the earlier cycles (PLAN_I S3b, F15 / F18,
        R6-1b (ii) / R8-3): at `at`, but never later than `cycle_id`'s own establishment — a cycle's fib cannot
        outlive its cycle, so a single activated late on a CTS_UPDATED, or a cross created late, ends the previous
        cycle at this cycle's CTS_ESTABLISHED moment. A pre-established cross (created BEFORE the establishment —
        Option A) still ends it at its creation (`_bos_moment_by_cycle` has no entry for the cycle yet)."""
        est = self._bos_moment_by_cycle.get((int(sid), int(cycle_id)))
        return int(at) if est is None else min(int(at), int(est))

    # I8 (PLAN_I §9.3 D1): a single and a cross are never both ACTIVE on one cycle — checked at every single
    # activation / reactivation and every cross creation / revival. A superseded (inactive) record of the other
    # kind may remain: the `cross_failed` fallback deactivates the cross, then activates the single.
    def _assert_no_active_cross(self, sid: int, cycle_id: int) -> None:
        latest = self._get_latest_cross(sid, cycle_id)
        assert latest is None or not latest[1].active, (
            f"[fib] I8: sid={sid} cycle={cycle_id} single activated while cross {latest[0]} is active")

    def _assert_no_active_single(self, sid: int, cycle_id: int) -> None:
        single = self._fibs.get((sid, cycle_id))
        assert single is None or not single.active, (
            f"[fib] I8: sid={sid} cycle={cycle_id} cross created / revived while the single is active")

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
    # CTS_THRESHOLD_UPDATED dispatch (pre-established phase)
    # ------------------------------------------------------------------

    def on_cts_threshold_updated(
        self,
        event: StructureEvent,
        df: pd.DataFrame,
    ) -> None:
        """Handle CTS_THRESHOLD_UPDATED (body: `_on_cts_threshold_updated`). Its
        imbalance questions are asked at the event's moment, `ev.idx` (the
        processing candle)."""
        self._handled.append(event)
        with self._evaluating(event):
            self._on_cts_threshold_updated(event, df)

    def _on_cts_threshold_updated(
        self,
        event: StructureEvent,
        df: pd.DataFrame,
    ) -> None:
        """Dispatch CTS_THRESHOLD_UPDATED (the pre-established phase).

        Triggers a cross-fib check for the target cycle (= event.cycle_id + 1)
        ONLY when that target is in pre-established phase AND within the
        structure's cross-fib limit (`_cross_allowed`, PLAN_I S3). In all other
        cases (target not pre-established, target past the limit),
        this is a no-op.

        The anchor is the running extreme past CTS_n; own_imb_start is the
        prospective BOS_{n+1} (the deepest pullback since CTS_n CONFIRMED).
        """
        sid = int(event.meta.get("structure_id", 0))
        source_cycle = int(event.meta.get("cycle_id", 0))
        target_cycle = source_cycle + 1

        phase = self._phase.get((sid, target_cycle))
        if phase != "pre_established":
            # Not in pre-established for the expected target: skip.
            return
        if not self._cross_allowed(sid, target_cycle):
            # Past the cross-fib limit (PLAN_I D1): no pre-established cross — the target is a single.
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

        self._cross_check(
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
    # The cross-fib check + state transitions
    # ------------------------------------------------------------------

    def _cross_check(
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
        """Central cross-fib check routine (every structure since PLAN_I S4).

        `own_window_end_idx` (a LOCATION: the CTS anchor at CTS_ESTABLISHED /
        CTS_UPDATED, the processing candle on the threshold path) ends the own
        test's window; `current_candle` is the TIME — the fill horizon and every
        lifecycle stamp (activated_at / deactivated_at / terminal) — the handled
        event's moment (split in Plan E E2b, switched in E3a).

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
        phase = self._phase.get((sid, target_cycle))

        # Steps 1–2 (own-imbalance test + dead-cycle backward walk) are the
        # shared routine (CROSS_CYCLE_FIB_SPEC.md §7): each prior cycle checked
        # to the current candle (Interpretation B). The dead_cycles cache is passed by
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
            evaluated_at=self._evaluated_at,
            fill_horizon_idx=current_candle,
        )

        if not elig.own_has:
            self._deactivate_active_cross(sid, target_cycle, current_candle, "own_imb_filled")
            # Pre-established: no fallback. Established: this cycle's existing single (the
            # `cross_failed` fallback) re-checks its own window like every single on every update
            # (PLAN_I S3b, F5) — the larger own window just tested has no unfilled imbalance, so
            # the single's cannot have one either: it deactivates instead of staying active on
            # its older CTS.
            if phase != "pre_established" and (sid, target_cycle) in self._fibs:
                self._update_fib_cts((sid, target_cycle), anchor_idx, anchor_price, df)
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
            self._activate_or_update_single(
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
            self._create_cross(
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
                self._reactivate_cross_in_place(
                    active_cross[0], active_cross[1],
                    anchor_idx, anchor_price, current_candle
                )
            else:
                self._create_cross(
                    sid, target_cycle, sd, bos_x, anchor_idx, anchor_price,
                    earliest_x, current_candle
                )
            return

        # Active version: compare earliest_x to its start cycle.
        if earliest_x == active_x:
            # Same start — update anchor in place (extension)
            self._extend_cross_anchor(
                active_cross[0], active_cross[1], anchor_idx, anchor_price
            )
        elif earliest_x > active_x:
            # Shrink — deactivate old version, create new
            self._deactivate_cross(active_cross[0], current_candle, "cross_shortened")
            self._create_cross(
                sid, target_cycle, sd, bos_x, anchor_idx, anchor_price,
                earliest_x, current_candle
            )
        else:
            # earliest_x < active_x — shouldn't happen (dead cycles monotonic)
            self._log(f"[fib][cross_cycle][warn] unexpected earliest_x={earliest_x} "
                  f"< active_x={active_x} for sid={sid} cycle={target_cycle}")

    def _create_cross(
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
        self._assert_no_active_single(sid, target_cycle)
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
        # set-if-absent) — at the target's establishment at the latest: a cross
        # created late on a CTS_UPDATED ends them there (PLAN_I S3b, F18).
        self._obsolete_prev_cycle_all_fibs(
            sid, target_cycle, end_idx=self._prev_cycle_end_at(sid, target_cycle, current_candle))

        # Lifecycle: record the cycle's sticky first-active idx (§15.3). v0 sets
        # it; a later version (shrink/revival) is a handoff that keeps the cycle
        # alive — set-if-absent keeps v0's earlier start.
        self._mark_first_active(sid, target_cycle, current_candle)

        self._log(f"[fib] cross_cycle sid={sid} CROSS ({earliest_x}->{target_cycle}) "
              f"v{version} ACTIVATED: bos_x_idx={bos_x_idx} -> anchor_idx={anchor_idx}")

    def _extend_cross_anchor(
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
        self._log(f"[fib] cross_cycle sid={state.structure_id} CROSS v{v} EXTENDED: "
              f"cts idx={new_anchor_idx}")

    def _reactivate_cross_in_place(
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
        self._assert_no_active_single(state.structure_id, state.cycle_id)
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
        self._log(f"[fib] cross_cycle sid={state.structure_id} cycle={state.cycle_id} "
              f"CROSS v{v} REACTIVATED in place: cts idx={anchor_idx}")

    def _deactivate_cross(self, key: tuple, current_candle: int, reason: str) -> None:
        """Mark a cross fib version inactive (per-record `active=False`).

        Session 2 note (FIB_LIFECYCLE_SPEC §9.1/§9.2): unlike the cycle-TERMINAL
        paths (new_cycle / lifecycle_end / reversal — which
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
        self._log(f"[fib] cross_cycle sid={state.structure_id} cycle={state.cycle_id} "
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

    def _activate_or_update_single(
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
        `status` is "disappeared" — the uniform terminal-INVALIDATION filter, a
        generic hook with NO producer since PLAN_I S4 (`_INVALIDATION_END_REASONS`
        is empty: the Scenario 1 revert, its only producer, is retired — R6-3 (v)),
        for both POI and chart consumers. Relies on
        `_finalize_lifecycle_fields()` having run first (the orchestrator calls
        it before this).
        """
        return [
            fib for fib in self._fibs.values()
            if fib.status != "disappeared"
        ]
