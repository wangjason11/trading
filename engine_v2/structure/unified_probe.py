"""Unified identify-start + probe primitive (Phase 1).

Replaces the four asymmetric paths (Scenario 1, Scenario 2 + Exc1 + Exc2
H1-main reversal, Scenario 3 iterative for subs, Scenario 2 + Exc1 sub
reversal) with a single primitive whose contract is:

    "Given a candidate start `input_idx` and a price reference (the
    `reference_zone`), produce a validated structural anchor for a new
    structure running in `direction` from that anchor, bounded above by
    `end_idx`."

Design notes — full spec lives in
`memory/project_unified_identify_start_probe.md`. Highlights:

- Runs on the structure's own TF; no post-probe cross-TF mapping.
- Per iteration evaluates EXACTLY ONE candidate — the single most extreme
  retrace candle in the scan window (lowest low for `direction=+1`,
  highest high for `direction=-1`).
- Two-condition reset (BOTH must hold to restart from the candidate):
    1. Wick extreme within X pips of `reference_zone.inner`
       (X = `DEFAULT_PROBE_RESET_PIPS[timeframe]`).
    2. Toward-zone wick size ≤ Y pips
       (Y = `DEFAULT_PROBE_RESET_WICK[timeframe]`). Wick is body-bounded
       (`body_top = max(o,c)`, `body_bottom = min(o,c)`); wick side is
       opposite to `direction` (the side that points toward the zone
       outer).
- Max 10 iterations. The `reference_zone` is unchanged across iterations
  (the probe walks forward in time, not outward in reference).
- Scan window: `[first_CTS_ESTABLISHED.idx + 1, exc_upper]` where
  `exc_upper = end_idx` whenever defined (LANDMINES: "Probe `end_idx` Is
  the Supreme Bound"); fallback `cts_est[1].idx` only in live mode.

This module sits unused in Phase 1 — no caller migrates until Session 2+.
The primitive's existence + tests gate the per-trigger migrations that
follow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import pandas as pd

from engine_v2.common.types import REQUIRED_CANDLE_COLS
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.structure.reference_zone import ReferenceZone

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
    "pending_reversal_anchor_idx",
    "pending_reversal_apply_idx",
    "struct_direction",
    "last_breakout_pat_apply_idx",
)


FinalizeCondition = Literal[
    "no_retrace",            # ≥1 CTS_EST + candidate fails reset → finalized
    "reversal_in_probe",     # reversal before 2nd CTS_EST in Phase 2 → finalized
    "second_cts_reached",    # Phase 2 saw cycle-1 CTS_EST without a qualifying retrace → finalized
    "end_idx_reached",       # end_idx reached without CTS_EST → finalized at caller bound
    "no_cts_pending",        # no CTS_EST + end_idx None → pending (live mode only)
    "one_cts_pending",       # 1 CTS_EST + end_idx None → pending (can't bound check window)
    "max_iterations",        # all iterations triggered reset → pending
]


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of one `unified_probe` invocation.

    `start_idx` is the validated/finalized start when `status="finalized"`,
    or the best current candidate when `status="pending"` (caller may
    re-invoke once more data lands).

    `original_ref_zone` is the same `ReferenceZone` passed in — the probe
    never refines its reference. Returned for caller convenience /
    debug attribution.
    """
    start_idx: int
    status: Literal["finalized", "pending"]
    iterations: int
    original_ref_zone: ReferenceZone
    finalize_condition: FinalizeCondition
    notes: str = ""


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
    by idx ascending."""
    out = [
        ev for ev in events
        if ev.type == "CTS_ESTABLISHED"
        and ev.meta.get("structure_id") == structure_id
    ]
    out.sort(key=lambda e: int(e.idx))
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


def _find_true_first_breakout_via_pat(
    df: pd.DataFrame,
    current_start: int,
    upper_idx: int,
    direction: int,
) -> Optional[int]:
    """Phase 1 helper: walk `df.pat` forward from `current_start` to find
    the first deterministic breakout pattern in the probe's direction
    whose anchor's extreme is a new running max/min.

    Uses the pre-computed deterministic pattern columns
    (`pat`, `pat_dir`) from `pattern_engine.detect_patterns` — these are
    threshold-independent (computed with `break_threshold=None`), so
    they identify every breakout pattern candidate regardless of any
    structural threshold. The first candidate whose high (sd=+1) or low
    (sd=-1) beats the running max/min over `[current_start, idx - 1]`
    is the "true first breakout."

    Comparison anchor is fixed at `current_start` within this scan
    (the iteration's anchor — see PART4 §4.4). Each successful
    retrace-reset advances `current_start` and re-anchors the next
    scan to the new candidate.

    Returns the idx of the true first breakout, or None if no pattern
    in `[current_start, upper_idx]` qualifies. `pat_status` is not
    filtered — every emitted pattern (`SUCCESS` or `CONFIRMED`) counts.
    """
    if upper_idx < current_start:
        return None
    for idx in range(int(current_start), int(upper_idx) + 1):
        if idx not in df.index:
            continue
        pat_val = df.loc[idx, "pat"]
        if not isinstance(pat_val, str) or pat_val == "":
            continue
        try:
            pat_dir = int(df.loc[idx, "pat_dir"])
        except (TypeError, ValueError):
            continue
        if pat_dir != direction:
            continue
        if idx <= current_start:
            return idx
        prior = df.loc[current_start:idx - 1]
        if prior.empty:
            return idx
        if direction == 1:
            prior_max = float(prior["h"].astype(float).max())
            this_h = float(df.loc[idx, "h"])
            if this_h >= prior_max:
                return idx
        else:
            prior_min = float(prior["l"].astype(float).min())
            this_l = float(df.loc[idx, "l"])
            if this_l <= prior_min:
                return idx
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


def _run_phase1(
    df: pd.DataFrame,
    input_idx: int,
    direction: int,
    reference_zone: ReferenceZone,
    end_idx: Optional[int],
    reset_tol: float,
    wick_cap: float,
    *,
    max_iterations: int,
) -> Tuple[int, Literal["finalized", "pending"], FinalizeCondition, int]:
    """Phase 1 — deterministic walk through `df.pat`, no MS.

    Per iteration: walk `df.pat` from `current_start` to find the true
    first breakout (`_find_true_first_breakout_via_pat`). If found, look
    for the deepest retrace candidate in `[true_X+1, end_idx]` and
    evaluate the 2-condition reset. Successful reset advances
    `current_start`. No qualifying retrace finalizes. No true_X
    finalizes (or stays pending in live mode with end_idx=None).

    Comparison anchor for the new-extreme check = `current_start`
    (resets re-anchor). End_idx is the inclusive supreme upper bound
    for both the df.pat walk and the retrace window.
    """
    current_start = int(input_idx)
    final_condition: FinalizeCondition = "max_iterations"
    final_status: Literal["finalized", "pending"] = "pending"
    iteration = 0

    upper_for_walk = end_idx if end_idx is not None else int(df.index[-1])

    for iteration in range(1, max_iterations + 1):
        true_x = _find_true_first_breakout_via_pat(
            df, current_start, upper_for_walk, direction,
        )

        if true_x is None:
            # No true first breakout in window.
            if end_idx is not None:
                final_condition = "end_idx_reached"
                final_status = "finalized"
            else:
                final_condition = "no_cts_pending"
                final_status = "pending"
            break

        # Bound the retrace window. Phase 1 uses end_idx as supreme; if
        # end_idx is None, we can't bound (Phase 2 picks this up for
        # first_confluence callers).
        if end_idx is None:
            final_condition = "one_cts_pending"
            final_status = "pending"
            break

        candidate_idx = _select_extreme_retrace_candidate(
            df, int(true_x) + 1, int(end_idx), direction,
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

        # Reset: advance current_start to the retrace candidate.
        print(
            f"[unified_probe phase1] reset triggered: iter={iteration} "
            f"true_x={true_x} candidate={candidate_idx} (was {current_start})"
        )
        current_start = int(candidate_idx)

    return current_start, final_status, final_condition, iteration


def _run_phase2(
    df: pd.DataFrame,
    start_idx: int,
    direction: int,
    reference_zone: ReferenceZone,
    end_idx: Optional[int],
    reset_tol: float,
    wick_cap: float,
    *,
    max_iterations: int,
) -> Tuple[int, Literal["finalized", "pending"], FinalizeCondition, int]:
    """Phase 2 — MS-based, called only by first_confluence (per spec).

    Runs MS from `current_start` with `enforce_cts0_new_extreme=True` so
    the FIRST emitted CTS_ESTABLISHED satisfies the "new running max/min
    since current_start" rule. Subsequent CTS by MS construction break
    the prior CTS's threshold, so they're implicitly new extremes.

    Retrace search window:
      - If CTS_0 confirmed before end_idx: `[CTS_0_est+1, cts_anchor_idx-1]`
      - Else (CTS_0 not confirmed before end_idx): `[CTS_0_est+1, end_idx]`

    Termination conditions:
      - Reversal AND <2 CTS_EST → `reversal_in_probe`
      - 0 CTS_EST → `end_idx_reached` (backtest) / `no_cts_pending` (live)
      - Retrace candidate exists + passes reset → advance, iterate
      - Retrace candidate exists + fails (or window empty) → `no_retrace`
        if no 2nd CTS_EST; `second_cts_reached` if 2nd CTS_EST already
        present in MS output (= cycle 0 completed without a qualifying
        retrace, and cycle 1 has begun)
    """
    current_start = int(start_idx)
    final_condition: FinalizeCondition = "max_iterations"
    final_status: Literal["finalized", "pending"] = "pending"
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
        ms = _make_market_structure(
            df_probe,
            struct_direction=direction,
            start_idx=current_start,
            structure_id=0,
            end_idx=end_idx,
            enforce_cts0_new_extreme=True,
        )
        ms.debug = True
        df_probe, probe_events, _ = ms.run()

        cts_est = _collect_cts_established(probe_events, structure_id=0)
        n_cts = len(cts_est)
        has_rev = _has_reversal(df_probe, structure_id=0)

        # Reversal before 2 cycles → structure not viable.
        if has_rev and n_cts < 2:
            final_condition = "reversal_in_probe"
            final_status = "finalized"
            break

        # No CTS_0 emitted in window (MS rejected every breakout pattern
        # via the cycle-0 new-extreme gate).
        if n_cts == 0:
            if end_idx is not None:
                final_condition = "end_idx_reached"
                final_status = "finalized"
            else:
                final_condition = "no_cts_pending"
                final_status = "pending"
            break

        first_cts = cts_est[0]
        first_cycle_id = int(first_cts.meta.get("cycle_id", 0))

        # Try to find CTS_0_CONFIRMED — if present, bound by its
        # cts_anchor_idx. Else fall back to end_idx.
        cycle_0_conf = _collect_cts_confirmed_for_cycle(
            probe_events, structure_id=0, cycle_id=first_cycle_id,
        )

        check_lo = int(first_cts.idx) + 1
        if cycle_0_conf is not None:
            cts0_anchor_idx = int(
                cycle_0_conf.meta.get("cts_anchor_idx", first_cts.idx)
            )
            check_hi = cts0_anchor_idx - 1
        elif end_idx is not None:
            check_hi = int(end_idx)
        else:
            # CTS_0 not confirmed AND end_idx None → pending in live.
            final_condition = "one_cts_pending"
            final_status = "pending"
            break

        if check_lo > check_hi:
            # Window empty — no retrace possible.
            if n_cts >= 2:
                final_condition = "second_cts_reached"
            else:
                final_condition = "no_retrace"
            final_status = "finalized"
            break

        candidate_idx = _select_extreme_retrace_candidate(
            df_probe, check_lo, check_hi, direction,
        )
        if candidate_idx is None:
            if n_cts >= 2:
                final_condition = "second_cts_reached"
            else:
                final_condition = "no_retrace"
            final_status = "finalized"
            break

        passes = _evaluate_reset_conditions(
            df_probe, candidate_idx, reference_zone, direction,
            reset_tol, wick_cap,
        )
        if not passes:
            if n_cts >= 2:
                final_condition = "second_cts_reached"
            else:
                final_condition = "no_retrace"
            final_status = "finalized"
            break

        # Reset: advance current_start.
        print(
            f"[unified_probe phase2] reset triggered: iter={iteration} "
            f"cts0_est={first_cts.idx} candidate={candidate_idx} "
            f"(was {current_start})"
        )
        current_start = int(candidate_idx)

    return current_start, final_status, final_condition, iteration


def unified_probe(
    df: pd.DataFrame,
    input_idx: int,
    direction: int,
    reference_zone: ReferenceZone,
    end_idx: Optional[int],
    timeframe: str,
    *,
    max_iterations: int = 10,
    enable_phase2: bool = False,
) -> ProbeResult:
    """Iteratively validate / refine `input_idx` against `reference_zone`
    via a two-phase design (user spec 2026-05-30).

    **Phase 1 (deterministic, runs always):** walks `df.pat` from
    `current_start` to `end_idx`, finds the first breakout pattern whose
    anchor is a new running max/min, then evaluates the 2-condition
    retrace reset on the deepest candidate in `[true_X+1, end_idx]`.
    Iterates on successful reset. No MS run.

    **Phase 2 (MS-based, only when `enable_phase2=True`):** invoked after
    Phase 1 for `first_confluence` callers. Runs MS from Phase 1's final
    `current_start` with `enforce_cts0_new_extreme=True`. Retrace window
    becomes `[CTS_0_est+1, cts_anchor_idx-1]` once CTS_0 confirms (or
    `[CTS_0_est+1, end_idx]` if it doesn't). Terminates via successful
    reset, no-retrace, 2nd CTS_EST reached, reversal, or end_idx.

    For first_counter, subsequent_*, etc., `enable_phase2=False`
    (default) and only Phase 1 runs — no MS at all. This both matches
    the live-trading flow (deterministic patterns are available as
    each candle arrives) AND wins a big perf savings in backtest.

    Parameters
    ----------
    df : DataFrame
        Candle features + patterns applied. Must include the
        deterministic-pattern columns (`pat`, `pat_dir`) written by
        `pattern_engine.detect_patterns`. Pair derived from `df.attrs`.
    input_idx : int
        Initial candidate start idx.
    direction : int
        +1 (uptrend probe) or -1 (downtrend probe).
    reference_zone : ReferenceZone
        Price target for the proximity check. Held constant across
        iterations.
    end_idx : int, optional
        Inclusive supreme upper bound. In Phase 1 it bounds both the
        df.pat walk and the retrace window. In Phase 2 it bounds the
        MS run AND the retrace window when CTS_0 doesn't confirm before
        end_idx is reached. None enables live-mode pending paths.
    timeframe : str
        TF for threshold lookup (H1 / M15 / M5).
    max_iterations : int
        Cap on iterations within each phase (default 10).
    enable_phase2 : bool
        When True, run Phase 2 after Phase 1. Used for `first_confluence`
        callers; default False keeps the probe Phase-1-only for every
        other caller.
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

    # --- Phase 1: deterministic ---
    p1_start, p1_status, p1_cond, p1_iter = _run_phase1(
        df, int(input_idx), direction, reference_zone, end_idx,
        reset_tol, wick_cap,
        max_iterations=max_iterations,
    )

    if not enable_phase2:
        notes = (
            f"unified_probe phase1: start={p1_start} direction={direction} "
            f"iter={p1_iter} status={p1_status} condition={p1_cond} "
            f"end_idx={end_idx} timeframe={timeframe}"
        )
        return ProbeResult(
            start_idx=p1_start,
            status=p1_status,
            iterations=p1_iter,
            original_ref_zone=reference_zone,
            finalize_condition=p1_cond,
            notes=notes,
        )

    # --- Phase 2: MS-based, only for first_confluence ---
    # Phase 2 starts from Phase 1's final current_start. If Phase 1
    # advanced via retrace-reset, Phase 2 honors that advance.
    p2_start, p2_status, p2_cond, p2_iter = _run_phase2(
        df, p1_start, direction, reference_zone, end_idx,
        reset_tol, wick_cap,
        max_iterations=max_iterations,
    )

    notes = (
        f"unified_probe phase1+2: start={p2_start} direction={direction} "
        f"p1_iter={p1_iter} p1_cond={p1_cond} p2_iter={p2_iter} "
        f"status={p2_status} condition={p2_cond} end_idx={end_idx} "
        f"timeframe={timeframe}"
    )
    return ProbeResult(
        start_idx=p2_start,
        status=p2_status,
        iterations=p1_iter + p2_iter,
        original_ref_zone=reference_zone,
        finalize_condition=p2_cond,
        notes=notes,
    )
