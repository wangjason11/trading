"""Unified identify-start + probe primitive.

Replaces the four asymmetric paths (Scenario 1, Scenario 2 + Exc1 + Exc2
H1-main reversal, Scenario 3 iterative for subs, Scenario 2 + Exc1 sub
reversal) with a single primitive whose contract is:

    "Given a candidate start `input_idx` and a price reference (the
    `reference_zone`), produce a validated structural anchor for a new
    structure running in `direction` from that anchor, bounded above by
    `probe_end_idx`."

Design notes — full spec lives in
`memory/project_true_first_breakout_cycle0.md` (cycle-0 true-first-breakout,
DESIGN LOCKED 2026-06-07 + scan-from-start pivot) and
`memory/project_unified_identify_start_probe.md`. Highlights:

- Runs on the structure's own TF; no post-probe cross-TF mapping.
- **Deterministic method (`_run_phase1`, runs for every caller):** per
  iteration it locates the cycle-0 true-first-breakout via the shared
  `find_true_first_breakout` routine (mechanism B against the MOVING
  BOS_0 inner threshold — iter 1 = `reference_zone.inner`; iter 2+ = an
  ad-hoc `bos=True` BOS_0 at the reset `current_start`), then evaluates
  the 2-condition retrace reset on the deepest candidate in
  `[CTS_0_EST+1, probe_end_idx]` against the CONSTANT `reference_zone`. No MS.
  This is the entire probe for non-FC callers (first_counter /
  subsequent_* / reversal).
- **Phase 2 (`_run_phase2`, only when `enable_phase2=True`):** MS-based,
  used by `first_confluence` (whose real probe_end_idx is NULL → it needs MS to
  reach CTS_0_CONFIRMED → `cts_anchor` to bound the retrace). Finalized
  alongside the MS scan-from-start change.
- Two-condition reset (BOTH must hold to restart from the candidate):
    1. Wick extreme within X pips of `reference_zone.inner`
       (X = `DEFAULT_PROBE_RESET_PIPS[timeframe]`).
    2. Toward-zone wick size ≤ Y pips
       (Y = `DEFAULT_PROBE_RESET_WICK[timeframe]`). Wick is body-bounded.
- Max 10 iterations. The retrace-reset `reference_zone` is CONSTANT across
  iterations (only the BOS_0 *threshold* moves — "two zones, don't
  conflate").
- `probe_end_idx` is the SUPREME upper bound (LANDMINES: "Probe `end_idx` Is the
  Supreme Bound") for both the breakout search and the retrace window.

The probe hands MS a DECISION, not events: `{finalized current_start,
BOS_0 bounds, CTS_0_EST (sanity-assert)}`. Under scan-from-start MS
re-finds CTS_0 by re-running the same shared routine from `current_start`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import pandas as pd

from engine_v2.common.types import REQUIRED_CANDLE_COLS
from engine_v2.patterns.structure_patterns import BreakoutPatterns
from engine_v2.structure import event_fields as ef
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.structure.reference_zone import (
    ReferenceZone,
    build_ad_hoc_bos0_reference_zone,
)
from engine_v2.structure.true_first_breakout import find_true_first_breakout

# `_make_market_structure` is the canonical MarketStructure constructor —
# it wires proximity resolvers + TF-keyed proximity/min-gap pips. Reusing
# it (rather than constructing MarketStructure directly) keeps the wiring
# in one place and avoids the spec §13.5.b inversion regression.
from engine_v2.structure.structure_engine import (
    _make_market_structure,
    _pip_size_from_pair,
)
from engine_v2.zones.zone_proximity import (
    DEFAULT_PROBE_RESET_PIPS,
    DEFAULT_PROBE_RESET_WICK,
)


# Mirror columns written by MarketStructure that get persisted onto entity_df
# after each sub run. Must be dropped from the working copy fed to a fresh MS
# run so prior-run values don't leak into the new state-machine reads.
# Keep in sync with `_STRUCTURE_COLS` in `engine_v2/multitf/entity_df_mutation.py`.
STRUCTURE_MIRROR_COLS = (
    "structure_id",
    "cycle_id",
    "cts_phase",
    "market_state",
    "range_lo",
    "range_hi",
    "swing_dir",
    "cts_price",
    "bos_price",
)

# Additional MS-managed columns not mirrored back but still written during a
# run; same drop-on-input rule applies. Keep in sync with
# `_MS_AUX_STRUCTURE_COLS` in `engine_v2/multitf/entity_df_mutation.py`.
STRUCTURE_AUX_COLS = (
    "range_active",
    "range_start_idx",
    "range_confirm_idx",
    "breakout_th",
    "pullback_th",
    "range_break_frac",
    "cts_idx",
    "cts_event",
    "bos_idx",
    "bos_event",
    "cts_cycle_id",
    "cts_threshold",
    "bos_threshold",
    "cycle_stage",
    "cts_phase_debug",
    "reversal_watch_active",
    "reversal_bos_th_frozen",
    "pending_reversal_pattern_anchor_idx",
    "pending_reversal_apply_idx",
    "struct_direction",
    "last_breakout_pat_apply_idx",
)


FinalizeCondition = Literal[
    "no_retrace",            # ≥1 CTS_EST + candidate fails reset → finalized
    "reversal_in_probe",     # reversal before 2nd CTS_EST in Phase 2 → finalized
    "second_cts_reached",    # Phase 2 saw cycle-1 CTS_EST without a qualifying retrace → finalized
    "end_idx_reached",       # probe_end_idx reached without CTS_EST → finalized at caller bound
    "no_cts_pending",        # no CTS_EST + probe_end_idx None → pending (live mode only)
    "one_cts_pending",       # 1 CTS_EST + probe_end_idx None → pending (can't bound check window)
    "max_iterations",        # all iterations triggered reset → pending
]


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of one `unified_probe` invocation.

    `starting_idx` is the validated/finalized structural anchor (the pool
    key under §17 — a HISTORICAL field, never a lifecycle value) when
    `status="finalized"`, or the best current candidate when
    `status="pending"` (caller may re-invoke once more data lands).

    `original_ref_zone` is the same `ReferenceZone` passed in — the probe
    never refines its reference. Returned for caller convenience /
    debug attribution.

    `bos0_inner` / `bos0_outer` are the BOS_0 threshold bounds at the
    finalized `starting_idx` (iter 1 = the reference inner/outer; iter 2+ =
    the ad-hoc BOS_0 at the reset start). MS uses `bos0_inner` as the
    cycle-0 breakout gate so probe and MS gate on the EXACT same number.
    `cts0_est_idx` is the apply/confirm idx of the located true first
    breakout — carried as a sanity-assert (MS re-finds it via
    scan-from-start); None when no breakout was found in the window.
    Populated by the deterministic method; left None on the Phase-2 path
    until that path is wired to the MS scan-from-start change.

    `finalize_idx` is the candle at which the probe's terminal DECISION became
    determinable — the latest idx whose information the finalize condition
    relied on (same frame as `starting_idx`). It is the causally-correct
    lifecycle-start floor for a probe-resolved structure ("the structure isn't
    KNOWN until the probe finalized"): under §17 it is the record's
    `probe_finalize_idx`, one of the three terms of `start_idx =
    max(probe_finalize_idx, trigger_idx, parent_floor_idx)` (PART4 §17.4).
    Per `finalize_condition`:
      - Phase 1 `no_retrace` / `end_idx_reached`     → `probe_end_idx`
      - Phase 2 `second_cts_reached`                 → 2nd CTS_ESTABLISHED moment
                                                       (meta["confirmed_at"], Plan B)
      - Phase 2 `reversal_in_probe`                  → reversal apply idx
      - Phase 2 `no_retrace`                         → CTS_0_CONFIRMED idx
                                                       (else `probe_end_idx`)
      - Phase 2 `end_idx_reached`                    → `probe_end_idx`
      - pending (`*_pending`, `max_iterations`)      → None (sid never builds)
    None whenever `status="pending"`.
    """
    starting_idx: int
    status: Literal["finalized", "pending"]
    iterations: int
    original_ref_zone: ReferenceZone
    finalize_condition: FinalizeCondition
    notes: str = ""
    bos0_inner: Optional[float] = None
    bos0_outer: Optional[float] = None
    cts0_est_idx: Optional[int] = None
    finalize_idx: Optional[int] = None


@dataclass(frozen=True)
class _DetResult:
    """Internal return of the deterministic method (`_run_phase1`)."""
    starting_idx: int
    status: Literal["finalized", "pending"]
    finalize_condition: FinalizeCondition
    iterations: int
    bos0_inner: Optional[float]
    bos0_outer: Optional[float]
    cts0_est_idx: Optional[int]
    finalize_idx: Optional[int]


def _select_extreme_retrace_candidate(
    df: pd.DataFrame,
    lo: int,
    hi: int,
    direction: int,
) -> Optional[int]:
    """Pick the single most-extreme retrace candle in `[lo, hi]`.

    For `direction=+1` (uptrend probe, retrace is downward): candle with
    minimum `l`. For `direction=-1`: candle with maximum `h`. Tie-break
    earliest (`.idxmin()` / `.idxmax()` return first occurrence).

    Returns None when the window is empty or out-of-bounds.
    """
    if lo > hi:
        return None
    seg = df.loc[lo:hi]
    if seg.empty:
        return None
    if direction == 1:
        return int(seg["l"].astype(float).idxmin())
    return int(seg["h"].astype(float).idxmax())


def _evaluate_reset_conditions(
    df: pd.DataFrame,
    candidate_idx: int,
    reference_zone: ReferenceZone,
    direction: int,
    reset_tol: float,
    wick_cap: float,
) -> bool:
    """Return True iff BOTH reset conditions hold on `candidate_idx`.

    Condition 1: wick extreme within `reset_tol` of `reference_zone.inner`.
        sd=+1: candidate `l` ≤ inner + reset_tol
        sd=-1: candidate `h` ≥ inner - reset_tol
    Condition 2: toward-zone wick ≤ `wick_cap` (body-bounded).
        sd=+1: body_bottom - l ≤ wick_cap   (lower wick)
        sd=-1: h - body_top ≤ wick_cap      (upper wick)

    Condition 2 rejects single-candle stab wicks that pass condition 1
    but represent transient spikes rather than structural retraces.

    NOTE: `reference_zone` here is the CONSTANT retrace-reset reference —
    NOT the moving BOS_0 threshold (gotcha #9, "two zones don't conflate").
    """
    if candidate_idx not in df.index:
        return False
    o = float(df.loc[candidate_idx, "o"])
    h = float(df.loc[candidate_idx, "h"])
    low = float(df.loc[candidate_idx, "l"])
    c = float(df.loc[candidate_idx, "c"])
    body_top = max(o, c)
    body_bottom = min(o, c)
    inner = reference_zone.inner

    if direction == 1:
        cond1 = low <= inner + reset_tol
        cond2 = (body_bottom - low) <= wick_cap
    else:
        cond1 = h >= inner - reset_tol
        cond2 = (h - body_top) <= wick_cap
    return bool(cond1 and cond2)


def _collect_cts_established(
    events: List[StructureEvent],
    structure_id: int,
) -> List[StructureEvent]:
    """Return CTS_ESTABLISHED events for the given `structure_id`, sorted
    by CTS anchor ascending (= cycle order)."""
    out = [
        ev for ev in events
        if ev.type == "CTS_ESTABLISHED"
        and ev.meta.get("structure_id") == structure_id
    ]
    out.sort(key=ef.cts_anchor_idx)
    return out


def _collect_cts_confirmed_for_cycle(
    events: List[StructureEvent],
    structure_id: int,
    cycle_id: int,
) -> Optional[StructureEvent]:
    """Return the CTS_CONFIRMED event for `(structure_id, cycle_id)`, or
    None if not present. Used in Phase 2 to bound the retrace search
    window by the cycle's `cts_anchor_idx`."""
    for ev in events:
        if (
            ev.type == "CTS_CONFIRMED"
            and ev.meta.get("structure_id") == structure_id
            and ev.meta.get("cycle_id") == cycle_id
        ):
            return ev
    return None


def _has_reversal(df: pd.DataFrame, structure_id: int) -> bool:
    """Detect whether `structure_id` reached the reversal state inside
    `df`. Mirrors the predicate `compute_structure_scenario_3` uses."""
    rev_mask = (
        (df["market_state"].astype(str).str.lower() == "reversal")
        & (df["structure_id"].astype(int) == structure_id)
    )
    return bool(rev_mask.any())


def _validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_CANDLE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[unified_probe] Missing required columns: {missing}")
    if df.empty:
        raise ValueError("[unified_probe] Input df is empty")


def _bos0_inner_at_start(
    df: pd.DataFrame,
    current_start: int,
    direction: int,
) -> Tuple[float, float]:
    """The MOVING BOS_0 (inner, outer) at a reset `current_start`.

    iter 2+ of the deterministic method: derive a fresh ad-hoc `bos=True`
    BOS_0 at the new start (shared `build_ad_hoc_bos0_reference_zone`).
    Degenerate-derivation fallback → the candle's own body-facing extreme
    (the "two zones" note's rare fallback): high for +1, low for -1
    (`outer` then the opposite extreme).
    """
    z = build_ad_hoc_bos0_reference_zone(df, current_start, direction)
    if z is not None:
        return float(z.inner), float(z.outer)
    if direction == 1:
        return float(df.iloc[current_start]["h"]), float(df.iloc[current_start]["l"])
    return float(df.iloc[current_start]["l"]), float(df.iloc[current_start]["h"])


def _run_phase1(
    df: pd.DataFrame,
    bp: BreakoutPatterns,
    input_idx: int,
    direction: int,
    reference_zone: ReferenceZone,
    probe_end_idx: Optional[int],
    reset_tol: float,
    wick_cap: float,
    *,
    max_iterations: int,
) -> _DetResult:
    """Deterministic method — shared true-first-breakout routine, no MS.

    Per iteration: locate the cycle-0 true first breakout from
    `current_start` against the MOVING BOS_0 inner threshold
    (`find_true_first_breakout`); if found, look for the deepest retrace
    candidate in `[CTS_0_EST+1, probe_end_idx]` and evaluate the 2-condition
    reset against the CONSTANT `reference_zone`. A successful reset
    advances `current_start` (and re-derives the moving BOS_0); no
    qualifying retrace finalizes; no breakout finalizes (or stays pending
    in live mode with `probe_end_idx=None`).

    The new-extreme comparison re-anchors to `current_start` on every
    reset (handled inside the shared routine). `probe_end_idx` is the inclusive
    supreme upper bound for BOTH the breakout search and the retrace
    window, and for the MS run Phase 2 drives (no candle past it is read
    there — Plan A; the detector `bp` is bounded at it too). The ad-hoc
    BOS_0 zone derivation at a reset candidate (`_bos0_inner_at_start`)
    still reads up to 5 candles past it (Plan A §8, L5b — deliberately out
    of scope).
    """
    current_start = int(input_idx)
    bos0_inner = float(reference_zone.inner)   # iter 1 = reference inner (they coincide)
    bos0_outer = float(reference_zone.outer)
    cts0_est_idx: Optional[int] = None
    final_condition: FinalizeCondition = "max_iterations"
    final_status: Literal["finalized", "pending"] = "pending"
    iteration = 0

    upper = int(probe_end_idx) if probe_end_idx is not None else int(df.index[-1])

    for iteration in range(1, max_iterations + 1):
        tfb = find_true_first_breakout(
            bp, current_start, upper, direction, bos0_inner,
        )

        if tfb is None:
            # No true first breakout in window.
            cts0_est_idx = None
            if probe_end_idx is not None:
                final_condition = "end_idx_reached"
                final_status = "finalized"
            else:
                final_condition = "no_cts_pending"
                final_status = "pending"
            break

        cts0_est_idx = int(tfb.est_idx)

        # Can't bound the retrace without probe_end_idx (Phase 2 picks this up
        # for first_confluence callers).
        if probe_end_idx is None:
            final_condition = "one_cts_pending"
            final_status = "pending"
            break

        candidate_idx = _select_extreme_retrace_candidate(
            df, int(tfb.est_idx) + 1, int(probe_end_idx), direction,
        )
        if candidate_idx is None:
            final_condition = "no_retrace"
            final_status = "finalized"
            break

        passes = _evaluate_reset_conditions(
            df, candidate_idx, reference_zone, direction,
            reset_tol, wick_cap,
        )
        if not passes:
            final_condition = "no_retrace"
            final_status = "finalized"
            break

        # Reset: advance current_start AND re-derive the moving BOS_0
        # threshold at the new start (the retrace-reset reference is
        # unchanged — only the breakout threshold moves).
        print(
            f"[unified_probe deterministic] reset triggered: iter={iteration} "
            f"cts0_est={tfb.est_idx} candidate={candidate_idx} (was {current_start})"
        )
        current_start = int(candidate_idx)
        bos0_inner, bos0_outer = _bos0_inner_at_start(df, current_start, direction)

    # finalize_idx (lifecycle-start floor): both finalized Phase-1 conditions
    # (`no_retrace`, `end_idx_reached`) examined the breakout/retrace window up
    # to `probe_end_idx` (both require `probe_end_idx is not None`), so the decision
    # depended on candles through `probe_end_idx`. Pending → None (sid never builds).
    _finalize_idx = (
        int(probe_end_idx) if (final_status == "finalized" and probe_end_idx is not None)
        else None
    )

    return _DetResult(
        starting_idx=current_start,
        status=final_status,
        finalize_condition=final_condition,
        iterations=iteration,
        bos0_inner=bos0_inner,
        bos0_outer=bos0_outer,
        cts0_est_idx=cts0_est_idx,
        finalize_idx=_finalize_idx,
    )


def _second_cts_moment(cts_est: list) -> int:
    """finalize_idx for `second_cts_reached`: the MOMENT the 2nd CTS was established
    (meta["confirmed_at"] = its apply candle), not `.idx` (the extreme inside the pattern
    span). Plan B §3.3 — the same principle as every lifecycle value (`confirmed_at` for
    timing, `.idx` for where the extreme sits), and it is what the early stop keys on."""
    ev = cts_est[1]
    return ef.event_moment(ev)


def _run_phase2(
    df: pd.DataFrame,
    starting_idx: int,
    bos0_inner: float,
    bos0_outer: float,
    direction: int,
    reference_zone: ReferenceZone,
    probe_end_idx: Optional[int],
    reset_tol: float,
    wick_cap: float,
    *,
    max_iterations: int,
) -> _DetResult:
    """Phase 2 — MS-based, called only by first_confluence (per spec).

    Runs MS from `current_start` in **pre-CTS_0 scan-from-start mode**
    (`enforce_cts0_new_extreme=True` + the handed `bos0_inner`), so MS
    establishes cycle 0 at the shared routine's true-first-breakout. Phase
    2's job is to drive MS far enough to reach CTS_0_CONFIRMED → its
    `cts_anchor_idx`, which bounds the retrace search (FC's real probe_end_idx is
    NULL, so it uses this earlier signal rather than waiting) — and it stops
    at the 2nd `CTS_ESTABLISHED` (Plan B): the double-CTS rule is an early
    stop, not a classification at exit. The MS run is handed
    `stop_after_cts_established=2` and ends at the first quiescent point (no
    reversal watch / pending reversal / pending rewind) after the 2nd CTS;
    `ms.early_stop_idx` says whether it actually stopped early (a run whose
    2nd CTS lands on the last in-bound step reaches `probe_end_idx` without one).
    Runs that never reach a 2nd CTS still run to `probe_end_idx`.

    Retrace search window:
      - If CTS_0 confirmed before probe_end_idx: `[CTS_0_est+1, cts_anchor_idx-1]`
      - Else (CTS_0 not confirmed before probe_end_idx): `[CTS_0_est+1, probe_end_idx]`

    Termination conditions:
      - Reversal AND <2 CTS_EST → `reversal_in_probe`
      - 0 CTS_EST → `end_idx_reached` (backtest) / `no_cts_pending` (live)
      - Retrace candidate exists + passes reset → advance (re-derive the
        moving BOS_0 at the new start), iterate
      - Retrace candidate exists + fails (or window empty) → `no_retrace`
        if no 2nd CTS_EST; `second_cts_reached` if a 2nd CTS_EST is
        already present (cycle 0 completed without a qualifying retrace —
        the run stopped at the 2nd CTS_EST; finalize = its moment,
        `confirmed_at`)

    `bos0_inner`/`bos0_outer` are the deterministic method's finalized
    BOS_0 bounds (the threshold at `starting_idx`); they MOVE with each reset
    via `_bos0_inner_at_start`, mirroring the deterministic method.
    """
    current_start = int(starting_idx)
    cur_bos0_inner = float(bos0_inner)
    cur_bos0_outer = float(bos0_outer)
    cts0_est_idx: Optional[int] = None
    final_condition: FinalizeCondition = "max_iterations"
    final_status: Literal["finalized", "pending"] = "pending"
    finalize_idx: Optional[int] = None  # lifecycle-start floor (see ProbeResult)
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        df_probe = df.copy()
        # Drop any pre-existing MS-written cols so prior runs' values
        # don't leak into this probe's state-machine reads.
        _stale_cols = [
            c for c in (STRUCTURE_MIRROR_COLS + STRUCTURE_AUX_COLS)
            if c in df_probe.columns
        ]
        if _stale_cols:
            df_probe = df_probe.drop(columns=_stale_cols)
        # Bounded at `probe_end_idx` with truncation semantics: MS reads nothing
        # past it and asserts post-run that no event is stamped past it
        # (Plan A — `MarketStructure.run()`), so `probe_events` and the
        # `df_probe` rows below are reproducible from the stated bound.
        # Plan B: the double-CTS rule is a true early stop — MS ends at the first
        # quiescent point after the 2nd CTS_ESTABLISHED (in addition to the
        # bound, never instead of it; `n_cts <= 1` runs still reach `probe_end_idx`).
        ms = _make_market_structure(
            df_probe,
            struct_direction=direction,
            start_idx=current_start,
            structure_id=0,
            end_idx=probe_end_idx,
            enforce_cts0_new_extreme=True,
            bos0_inner=cur_bos0_inner,
            stop_after_cts_established=2,
        )
        ms.debug = True
        df_probe, probe_events, _ = ms.run()

        cts_est = _collect_cts_established(probe_events, structure_id=0)
        n_cts = len(cts_est)
        has_rev = _has_reversal(df_probe, structure_id=0)

        # "Stopped early" comes from `ms.early_stop_idx`, never from `n_cts >= 2`
        # (a 2nd CTS on the last in-bound step with a pending reversal reaches
        # `probe_end_idx` without stopping). The print also checks the §2
        # extreme-vs-moment pair on the probe's OWN run.
        if ms.early_stop_idx is not None:
            print(
                f"[unified_probe phase2] early stop: p2_iter={iteration} "
                f"stop_idx={ms.early_stop_idx} probe_end_idx={probe_end_idx} "
                f"cts1_anchor={ef.cts_anchor_idx(cts_est[1]) if n_cts >= 2 else None} "
                f"cts1_moment={_second_cts_moment(cts_est) if n_cts >= 2 else None}"
            )

        # Reversal before 2 cycles → structure not viable.
        if has_rev and n_cts < 2:
            if cts_est:
                cts0_est_idx = ef.event_moment(cts_est[0])
            final_condition = "reversal_in_probe"
            final_status = "finalized"
            # finalize idx = the reversal apply candle (the latest signal the
            # "reversal before 2nd CTS" decision relied on).
            _rev_mask = (
                (df_probe["market_state"].astype(str).str.lower() == "reversal")
                & (df_probe["structure_id"].astype(int) == 0)
            )
            finalize_idx = (
                int(df_probe.index[_rev_mask].min()) if bool(_rev_mask.any())
                else (int(probe_end_idx) if probe_end_idx is not None else None)
            )
            break

        # No CTS_0 emitted in window (scan mode found no true breakout).
        if n_cts == 0:
            cts0_est_idx = None
            if probe_end_idx is not None:
                final_condition = "end_idx_reached"
                final_status = "finalized"
                finalize_idx = int(probe_end_idx)
            else:
                final_condition = "no_cts_pending"
                final_status = "pending"
            break

        first_cts = cts_est[0]
        first_cycle_id = int(first_cts.meta.get("cycle_id", 0))
        cts0_est_idx = ef.event_moment(first_cts)

        # Try to find CTS_0_CONFIRMED — if present, bound by its
        # cts_anchor_idx. Else fall back to probe_end_idx.
        cycle_0_conf = _collect_cts_confirmed_for_cycle(
            probe_events, structure_id=0, cycle_id=first_cycle_id,
        )

        # A TIME bound: the retrace window opens after CTS_0 is KNOWN — its
        # moment + 1 (Plan E E3c; Phase 1's `tfb.est_idx + 1` already was).
        check_lo = cts0_est_idx + 1
        if cycle_0_conf is not None:
            cts0_anchor_idx = int(
                ef.cts_anchor_idx(cycle_0_conf)
            )
            check_hi = cts0_anchor_idx - 1
        elif probe_end_idx is not None:
            check_hi = int(probe_end_idx)
        else:
            # CTS_0 not confirmed AND probe_end_idx None → pending in live.
            final_condition = "one_cts_pending"
            final_status = "pending"
            break

        # finalize-idx candidates for the three "no qualifying retrace" exits
        # below (window empty / no candidate / candidate fails reset):
        #   - second_cts_reached → the 2nd CTS_ESTABLISHED's MOMENT
        #     (`confirmed_at`; cycle 0 completed + cycle 1 established — the
        #     "double CTS", earlier than the parent-CTS probe_end_idx) — Plan B.
        #   - no_retrace → CTS_0_CONFIRMED idx (the decision needed cycle 0's
        #     confirmation), else probe_end_idx when cycle 0 didn't confirm in-window.
        _second_cts_fin = _second_cts_moment(cts_est) if n_cts >= 2 else None
        _no_retrace_fin = (
            int(cycle_0_conf.idx) if cycle_0_conf is not None
            else (int(probe_end_idx) if probe_end_idx is not None else None)
        )

        if check_lo > check_hi:
            # Window empty — no retrace possible.
            if n_cts >= 2:
                final_condition = "second_cts_reached"
                finalize_idx = _second_cts_fin
            else:
                final_condition = "no_retrace"
                finalize_idx = _no_retrace_fin
            final_status = "finalized"
            break

        candidate_idx = _select_extreme_retrace_candidate(
            df_probe, check_lo, check_hi, direction,
        )
        if candidate_idx is None:
            if n_cts >= 2:
                final_condition = "second_cts_reached"
                finalize_idx = _second_cts_fin
            else:
                final_condition = "no_retrace"
                finalize_idx = _no_retrace_fin
            final_status = "finalized"
            break

        passes = _evaluate_reset_conditions(
            df_probe, candidate_idx, reference_zone, direction,
            reset_tol, wick_cap,
        )
        if not passes:
            if n_cts >= 2:
                final_condition = "second_cts_reached"
                finalize_idx = _second_cts_fin
            else:
                final_condition = "no_retrace"
                finalize_idx = _no_retrace_fin
            final_status = "finalized"
            break

        # Reset: advance current_start AND re-derive the moving BOS_0
        # threshold at the new start (mirrors the deterministic method).
        print(
            f"[unified_probe phase2] reset triggered: iter={iteration} "
            f"cts0_est={cts0_est_idx} candidate={candidate_idx} "
            f"(was {current_start})"
        )
        current_start = int(candidate_idx)
        cur_bos0_inner, cur_bos0_outer = _bos0_inner_at_start(
            df, current_start, direction,
        )

    return _DetResult(
        starting_idx=current_start,
        status=final_status,
        finalize_condition=final_condition,
        iterations=iteration,
        bos0_inner=cur_bos0_inner,
        bos0_outer=cur_bos0_outer,
        cts0_est_idx=cts0_est_idx,
        finalize_idx=finalize_idx,
    )


def unified_probe(
    df: pd.DataFrame,
    input_idx: int,
    direction: int,
    reference_zone: ReferenceZone,
    probe_end_idx: Optional[int],
    timeframe: str,
    *,
    max_iterations: int = 10,
    enable_phase2: bool = False,
) -> ProbeResult:
    """Iteratively validate / refine `input_idx` against `reference_zone`.

    **Deterministic method (`_run_phase1`, runs always):** locates the
    cycle-0 true first breakout via `find_true_first_breakout` (mechanism
    B against the MOVING BOS_0 inner; iter 1 = `reference_zone.inner`,
    iter 2+ = ad-hoc BOS_0 at the reset start), then evaluates the
    2-condition retrace reset on the deepest candidate in
    `[CTS_0_EST+1, probe_end_idx]`. Iterates on a successful reset. No MS run.

    **Phase 2 (`_run_phase2`, only when `enable_phase2=True`):** MS-based,
    invoked after the deterministic method for `first_confluence` callers
    (whose real probe_end_idx is NULL → MS reaches CTS_0_CONFIRMED → `cts_anchor`
    to bound the retrace).

    For first_counter, subsequent_*, reversal, `enable_phase2=False`
    (default) and only the deterministic method runs — no MS at all. This
    both matches the live-trading flow (deterministic patterns are
    available as each candle arrives) AND wins a big perf saving.

    Parameters
    ----------
    df : DataFrame
        Candle features + patterns applied. Pair derived from `df.attrs`.
        Assumed 0-based RangeIndexed (loc == iloc — the standard MS / slice
        convention the detectors + retrace selector both rely on).
    input_idx : int
        Initial candidate start idx.
    direction : int
        +1 (uptrend probe) or -1 (downtrend probe).
    reference_zone : ReferenceZone
        The CONSTANT retrace-reset reference (held across iterations) AND
        the iter-1 BOS_0 threshold.
    probe_end_idx : int, optional
        Inclusive supreme upper bound for the breakout search, the retrace
        window and the MS run Phase 2 drives (no candle past it is read
        there — Plan A). A COMPUTE bound (like the run cap) — nothing to do
        with the lifecycle `end_idx` of §17 (renamed from `end_idx` by Plan C
        for exactly that collision). The ad-hoc BOS_0 zone derivation at a
        reset candidate still reads up to 5 candles past it (Plan A §8, L5b).
        None enables live-mode pending paths.
    timeframe : str
        TF for threshold lookup (H1 / M15 / M5).
    max_iterations : int
        Cap on iterations within each phase (default 10).
    enable_phase2 : bool
        When True, run Phase 2 after the deterministic method. Used for
        `first_confluence` callers; default False keeps the probe
        deterministic-only for every other caller.
    """
    _validate_input(df)
    if direction not in (1, -1):
        raise ValueError(f"[unified_probe] direction must be ±1, got {direction}")

    pip_size = _pip_size_from_pair(df)
    reset_pips = DEFAULT_PROBE_RESET_PIPS.get(
        timeframe, DEFAULT_PROBE_RESET_PIPS["H1"]
    )
    wick_pips = DEFAULT_PROBE_RESET_WICK.get(
        timeframe, DEFAULT_PROBE_RESET_WICK["H1"]
    )
    reset_tol = float(reset_pips) * pip_size
    wick_cap = float(wick_pips) * pip_size

    # Bounded detector: candles past `probe_end_idx` do not exist for Phase 1's
    # breakout search (Plan A §3.3 — equivalent to the `est > hi` drop in
    # `find_true_first_breakout`, now by construction). None → whole frame.
    bp = BreakoutPatterns(df, end_idx=probe_end_idx)

    # --- Deterministic method ---
    det = _run_phase1(
        df, bp, int(input_idx), direction, reference_zone, probe_end_idx,
        reset_tol, wick_cap,
        max_iterations=max_iterations,
    )

    if not enable_phase2:
        notes = (
            f"unified_probe deterministic: start={det.starting_idx} "
            f"direction={direction} iter={det.iterations} "
            f"status={det.status} condition={det.finalize_condition} "
            f"bos0_inner={det.bos0_inner} cts0_est={det.cts0_est_idx} "
            f"probe_end_idx={probe_end_idx} timeframe={timeframe}"
        )
        return ProbeResult(
            starting_idx=det.starting_idx,
            status=det.status,
            iterations=det.iterations,
            original_ref_zone=reference_zone,
            finalize_condition=det.finalize_condition,
            notes=notes,
            bos0_inner=det.bos0_inner,
            bos0_outer=det.bos0_outer,
            cts0_est_idx=det.cts0_est_idx,
            finalize_idx=det.finalize_idx,
        )

    # --- Phase 2: MS-based, only for first_confluence ---
    # Phase 2 starts from the deterministic method's final current_start
    # and its finalized BOS_0 bounds (the MS pre-CTS_0 scan gate). It
    # re-derives the moving BOS_0 on each of its own resets.
    p2 = _run_phase2(
        df, det.starting_idx, det.bos0_inner, det.bos0_outer,
        direction, reference_zone, probe_end_idx,
        reset_tol, wick_cap,
        max_iterations=max_iterations,
    )

    notes = (
        f"unified_probe deterministic+phase2: start={p2.starting_idx} "
        f"direction={direction} det_iter={det.iterations} "
        f"det_cond={det.finalize_condition} p2_iter={p2.iterations} "
        f"status={p2.status} condition={p2.finalize_condition} "
        f"bos0_inner={p2.bos0_inner} cts0_est={p2.cts0_est_idx} "
        f"probe_end_idx={probe_end_idx} timeframe={timeframe}"
    )
    return ProbeResult(
        starting_idx=p2.starting_idx,
        status=p2.status,
        iterations=det.iterations + p2.iterations,
        original_ref_zone=reference_zone,
        finalize_condition=p2.finalize_condition,
        notes=notes,
        bos0_inner=p2.bos0_inner,
        bos0_outer=p2.bos0_outer,
        cts0_est_idx=p2.cts0_est_idx,
        finalize_idx=p2.finalize_idx,
    )
