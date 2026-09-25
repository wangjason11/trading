from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from engine_v2.common.types import REQUIRED_CANDLE_COLS, StructureLevel
from engine_v2.structure import event_fields as ef
from engine_v2.structure.market_structure import MarketStructure, StructureEvent
from engine_v2.structure.identify_start import (
    identify_start_scenario_1,
    identify_start_scenario_2_after_reversal,
)
from engine_v2.structure.reference_zone import (
    build_ad_hoc_bos0_reference_zone,
    build_reference_zone_from_cts_event,
)
from engine_v2.zones.kl_zones_v1 import (
    compute_bos_inner_from_event,
    derive_kl_zones_v1,
)
from engine_v2.zones.poi_zones import compute_poi_inners_for_cycle
from engine_v2.zones.zone_proximity import (
    DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS,
    DEFAULT_PROBE_RESET_PIPS,
    DEFAULT_PROXIMITY_PIPS,
)


def _probe_reset_pips(timeframe: str) -> float:
    """Look up the probe reset threshold (in pips) for a timeframe.

    Returns float because the per-TF table allows fractional pips
    (e.g. M15 = 2.5) so it can sit cleanly between adjacent thresholds.
    """
    return DEFAULT_PROBE_RESET_PIPS.get(timeframe, DEFAULT_PROBE_RESET_PIPS["H1"])


def _proximity_pips(timeframe: str) -> int:
    """Look up the zone-proximity threshold (in pips) for a timeframe."""
    return DEFAULT_PROXIMITY_PIPS.get(timeframe, DEFAULT_PROXIMITY_PIPS["H1"])


def _min_gap_pips(timeframe: str) -> int:
    """Look up the narrow-cycle gap threshold (in pips) for a timeframe.

    Used by the dual CTS proximity confirmation Rule 1 gate
    (`MarketStructure._apply_pattern_at_apply_idx`) and by Rules 2/3 in
    the post-facto scan (`zones/zone_proximity.check_zone_proximity`).
    """
    return DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS.get(
        timeframe, DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS["H1"]
    )


def _make_market_structure(
    df: pd.DataFrame,
    struct_direction: int,
    *,
    timeframe: str = "H1",
    **kwargs,
) -> MarketStructure:
    """Construct MarketStructure with proximity resolvers + TF-defaulted
    proximity_pips wired in. Per Part 4 §13.5.b, MarketStructure does not
    import from zones/; resolvers are passed at construction so the
    structure→zones import inversion stays gone.

    Parity carve-out (matches Step 1 baseline): probe-only call sites
    (Exception 2 probes, Scenario 3 Phase 1/2 probes) historically
    constructed MarketStructure without `timeframe`, falling back to the
    H1 default of `DEFAULT_PROXIMITY_PIPS["H1"]` proximity_pips regardless
    of the outer caller's TF. Those sites still call this helper without
    `timeframe`, preserving the H1 default. Spec §4.4 explicitly listed
    only the probe-RESET
    threshold for TF-aware lookup; proximity_pips for probes is a
    deferred cleanup. See the comment in `compute_structure_from_start`
    near pip_tolerance for the same carve-out applied to probe reset.
    """
    kwargs.setdefault("proximity_pips", _proximity_pips(timeframe))
    kwargs.setdefault("min_gap_pips", _min_gap_pips(timeframe))
    kwargs.setdefault("bos_inner_resolver", compute_bos_inner_from_event)
    kwargs.setdefault("poi_inners_resolver", compute_poi_inners_for_cycle)
    return MarketStructure(
        df,
        struct_direction=struct_direction,
        timeframe=timeframe,
        **kwargs,
    )


@dataclass
class StructureEngineResult:
    df: pd.DataFrame
    levels: List[StructureLevel]
    events: List[StructureEvent]
    struct_direction: int
    notes: str = ""


@dataclass
class BoundedStructureResult:
    """Result of a bounded SINGLE-structure run (Part 4 §5, REVISED 2026-05-25).

    Unlike StructureEngineResult (multi-structure), this represents exactly
    one directional structure (structure_id=0). `reversal_idx` is the
    boundary to the next sub sid — the candle where the run hit its first
    internal reversal — or None if no reversal occurred inside
    `[start_idx, end_idx]` (bounded out by end_idx, or no reversal at all).
    """
    df: pd.DataFrame
    levels: List[StructureLevel]
    events: List[StructureEvent]
    struct_direction: int
    start_idx: int
    end_idx: Optional[int]
    reversal_idx: Optional[int]   # reversal pattern apply idx; None if no reversal in bounds
    notes: str = ""


@dataclass
class Scenario3Result:
    df: pd.DataFrame
    levels: List[StructureLevel]
    events: List[StructureEvent]
    struct_direction: int
    start_idx: int                    # Final validated start (or current best if pending)
    original_bos0_bounds: Optional[Tuple[float, float, str]]
    probe_iterations: int             # How many probes were run in Phase 1
    status: str                       # "finalized" or "pending"
    notes: str = ""


def compute_structure(df: pd.DataFrame, *, timeframe: str = "H1") -> StructureEngineResult:
    """
    Adapter boundary: df(with patterns/features) -> MarketStructure outputs:
      - df2 (with structure columns)
      - ms_events (StructureEvent list)
      - levels (StructureLevel list)

    Multi-structure flow:
      1. Identify initial start candle and direction (Scenario 1)
      2. Run MarketStructure for structure_id=0
      3. On reversal, identify next start (Scenario 2 with Exception 1/2 handling)
      4. Run MarketStructure for structure_id=1, etc.
      5. Repeat until no more reversals or guard limit reached

    Exception handling after reversal:
      - Exception 1: If higher high/lower low exists after last CTS but before reversal, start there
      - Exception 2 (probe): Always runs regardless of Exception 1. Starts from Exception 1's
        override if triggered, or base last CTS idx otherwise. Runs probe to reversal_confirmed.
        If pullback confirmed and price reached near CTS zone, start from that candle.
    """
    _validate_input(df)

    pip_size = _pip_size_from_pair(df)

    # --- Scenario 1 (initial start identification) ---
    input_idx = int(df.index.max())
    d0 = identify_start_scenario_1(df, input_idx=input_idx, lookback_days=183, min_history=50)

    df2 = df
    all_events: List[StructureEvent] = []
    all_levels: List[StructureLevel] = []

    start_idx = int(d0.start_idx)
    struct_direction = int(d0.struct_direction)
    structure_id = 0

    # Cycle-0 true-first-breakout scan mode (project_true_first_breakout_cycle0.md).
    # A structure's CTS_0 is established only by the first true breakout past the
    # BOS_0 inner. MS's pre-CTS_0 scan mode (`enforce_cts0_new_extreme`) gates on
    # that inner threshold (`cur_bos0_inner`), which is carried per-sid through the
    # loop below:
    #   - sid=0  (trading open): the ad-hoc bos=True BOS_0 anchored at the
    #     Scenario-1 start, built here. Single-shot: no probe, no retrace-resets
    #     (current_start is fixed at the Scenario-1 start).
    #   - sid>=1 (reversals, Step 4): the BOS_0 inner returned by the reversal
    #     unified_probe inside the loop, mirroring the sub reversal path.
    # `.inner` mirrors the sub probe's `bos0_inner = reference_zone.inner`
    # (unified_probe.py) — single source of truth, NOT a re-derivation.
    #
    # None => the ad-hoc base can't be derived. In a full backtest all history
    # exists, so None can only be the DEGENERATE case -> Option A: skip scan mode +
    # warn (sid=0 establishes as before). The INSUFFICIENT-HISTORY cause (a needed
    # neighbor candle past the live edge) cannot occur in backtest; if this warn
    # ever fires at a live edge, the wait-for-candles logic hooks in here
    # (memory/project_ad_hoc_zone_wait_for_candles).
    bos0_ref = build_ad_hoc_bos0_reference_zone(df, start_idx, struct_direction)
    cur_bos0_inner = bos0_ref.inner if bos0_ref is not None else None
    if cur_bos0_inner is None:
        print(
            f"[structure_engine][warn] main sid=0 cycle-0 ad-hoc BOS_0 zone could "
            f"not be built at start_idx={start_idx} (sd={struct_direction}); "
            f"skipping pre-CTS_0 scan mode (Option A) — sid=0 establishes as before. "
            f"If this fires at a live edge, the wait-for-candles logic hooks in here "
            f"(memory/project_ad_hoc_zone_wait_for_candles)."
        )

    # Run multiple structure segments until no more reversals (or we hit end)
    max_structures_guard = 20  # safety guard against infinite loops
    for loop_iter in range(max_structures_guard):
        # Pre-CTS_0 scan mode runs for EVERY sid whose BOS_0 inner was derivable:
        # sid=0 from the Scenario-1 ad-hoc above; sid>=1 from the reversal
        # unified_probe below (Step 4). None -> scan off for that sid (cycle-0
        # establishes the ordinary way), matching the sub-parity convention
        # `enforce_cts0_new_extreme=(bos0_inner is not None)`.
        _use_cts0_scan = (cur_bos0_inner is not None)
        ms = _make_market_structure(
            df2, struct_direction=struct_direction, start_idx=start_idx,
            structure_id=structure_id, timeframe=timeframe, pip_size=pip_size,
            enforce_cts0_new_extreme=_use_cts0_scan,
            bos0_inner=cur_bos0_inner if _use_cts0_scan else None,
        )
        ms.debug = True
        df2, ms_events, levels = ms.run()

        all_events.extend(ms_events)
        all_levels.extend(levels)

        # Find reversal idx for THIS structure_id (MarketStructure stops when it
        # hits reversal). The mask matches exactly ONE candle — the reversal
        # apply candle — because the run sets REVERSAL state at a single candle
        # then breaks, and the terminal forward-stamp touches market_state only
        # (not structure_id). So .min() == .max() by construction; we WARN (not
        # crash) if that invariant is ever violated and proceed with the apply
        # idx. This single value supersedes the old reversal_start/confirmed pair
        # (vestigial names for the same candle — unification cleanup).
        rev_mask = (df2["market_state"].astype(str).str.lower() == "reversal") & (df2["structure_id"].astype(int) == structure_id)
        if not rev_mask.any():
            break

        reversal_apply_idx = int(df2.loc[rev_mask].index.min())
        _rev_max = int(df2.loc[rev_mask].index.max())
        if _rev_max != reversal_apply_idx:
            print(
                f"[structure_engine][warn] reversal mask spans >1 candle for "
                f"sid={structure_id} (min={reversal_apply_idx} max={_rev_max}); "
                f"proceeding with apply idx (.min())."
            )

        # --- Reversal handoff (Step 4): unified_probe replaces Scenario 2 +
        # Exception 1 + Exception 2 (project_unified_identify_start_probe.md /
        # project_true_first_breakout_cycle0.md). Mirrors the sub reversal path
        # (entity_df_mutation._resolve_reversal_start): reference = the prior sid's most
        # recent {CONF/UPD/EST} CTS; the probe runs in the flipped direction over
        # [the prior CTS anchor, reversal apply idx] and hands back a DECISION
        # (start + BOS_0 inner) — NOT events. The reversal structure is produced
        # by a fresh unbounded scan-from-start MS run in the next loop iteration.
        # `unified_probe` is imported locally: it imports _make_market_structure
        # from this module, so a top-level import would be circular.
        from engine_v2.structure.unified_probe import unified_probe

        probe_sd = -struct_direction
        # Prior sid's CTS zones for the CONFIRMED-zone reference lookup. Same
        # derivation the (now-removed) Exception-2 path used via
        # _get_last_cts_zone_bounds; build_reference_zone_from_cts_event falls
        # back to an ad-hoc CTS derivation when no confirmed zone exists.
        kl_zones = derive_kl_zones_v1(df2, all_events, struct_direction=struct_direction)
        ref_zone = build_reference_zone_from_cts_event(
            all_events,
            kl_zones,
            df2,
            sid=structure_id,
            probe_direction=probe_sd,
            idx_window=None,
        )
        if ref_zone is None:
            # Degenerate: a reversal with zero CTS events for the prior structure
            # (should not occur — a structure must establish a cycle to reverse).
            # Stop adding structures rather than guess a start (sub-parity).
            print(
                f"[structure_engine][warn] reversal reference zone unavailable "
                f"for sid={structure_id} — no CTS event for prior structure; "
                f"stopping (no reversal-born sid)."
            )
            break

        probe_input_idx = int(ref_zone.anchor_idx)
        if probe_input_idx >= reversal_apply_idx:
            # No forward scan window — degenerate (sub-parity).
            print(
                f"[structure_engine][warn] degenerate reversal probe window for "
                f"sid={structure_id} (input={probe_input_idx} "
                f"end={reversal_apply_idx}) — stopping (no reversal-born sid)."
            )
            break

        probe = unified_probe(
            df2.copy(),
            input_idx=probe_input_idx,
            direction=probe_sd,
            reference_zone=ref_zone,
            probe_end_idx=reversal_apply_idx,
            timeframe=timeframe,
            enable_phase2=False,
        )
        print(
            f"[structure_engine] unified_probe (reversal): sid={structure_id} "
            f"input={probe_input_idx} end={reversal_apply_idx} "
            f"ref={ref_zone.source} -> start={probe.starting_idx} "
            f"status={probe.status} cond={probe.finalize_condition} "
            f"iter={probe.iterations}"
        )
        if probe.status == "pending":
            # Only possible with end_idx=None; we always pass a defined end_idx,
            # so this is dormant. Guard for symmetry with the sub path.
            print(
                f"[structure_engine][warn] reversal unified_probe did not "
                f"finalize for sid={structure_id}; stopping (no reversal-born sid)."
            )
            break

        next_start_idx = int(probe.starting_idx)
        # Advance to the reversal-born structure. Its cycle-0 CTS_0 is
        # established by the next iteration's scan-from-start MS run gated on
        # this BOS_0 inner (a price -> slice-invariant; None -> scan off,
        # sub-parity).
        start_idx = next_start_idx
        struct_direction = probe_sd
        structure_id = structure_id + 1
        cur_bos0_inner = probe.bos0_inner

    notes = (
        f"MarketStructure v1: initial_start={d0.start_idx} initial_sd={d0.struct_direction} "
        f"structures={structure_id + 1} events={len(all_events)} levels={len(all_levels)} "
        f"start_reason={d0.reason}"
    )

    _validate_output(df2)

    return StructureEngineResult(
        df=df2,
        levels=all_levels,
        events=all_events,
        struct_direction=struct_direction,
        notes=notes,
    )


def compute_structure_scenario_3(
    df: pd.DataFrame,
    start_idx: int,
    struct_direction: int,
    *,
    pip_tolerance_pips: Optional[float] = None,
    max_probe_iterations: int = 10,
    end_idx: Optional[int] = None,
    run_continuation: bool = True,
    timeframe: str = "H1",
) -> Scenario3Result:
    """
    Scenario 3: Arbitrary start with iterative BOS_0 probe.

    Phase 1 — Iterative probing to validate/refine start_idx via BOS_0 zone
    proximity checking.  Phase 2 — Multi-structure continuation from the
    finalized probe (same logic as compute_structure lines 69-176).

    Note: Phase 2 is currently exercised only by tests. All production
    callers (only ``_run_h1_reverse_probe`` today) pass
    ``run_continuation=False`` for probe-only mode. The Phase 2 path
    remains available for future features that need multi-structure
    continuation from an arbitrary validated start.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with candle features/patterns already applied.
    start_idx : int
        Arbitrary start index to probe from.
    struct_direction : int
        +1 for uptrend, -1 for downtrend.
    pip_tolerance_pips : float, optional
        Pip tolerance for zone proximity checking. If None (default), looked
        up from DEFAULT_PROBE_RESET_PIPS by ``timeframe`` (H1=3, M15=2.5, M5=2).
    max_probe_iterations : int
        Maximum number of probe restarts before giving up (default 10).
    end_idx : int, optional
        Bound the probe window — passed through to MarketStructure.
        When the bound is reached without 2 CTS_EST events, the current
        start is accepted as finalized (Condition 4a).
        When end_idx is None and the probe runs out of available df data
        without 2 CTS_EST events, status is "pending" (Condition 4b) — the
        caller may re-run the probe later when more data arrives.
    run_continuation : bool
        Whether to run Phase 2 (multi-structure continuation) after probe
        finalization. Set False for probe-only use (e.g. H1 reverse probe).

    Returns
    -------
    Scenario3Result
        Contains df, levels, events, status ("finalized" or "pending"),
        original_bos0_bounds, probe_iterations, and notes. Status is
        "pending" when end_idx is None and the probe could not reach a
        terminal break condition with the available data; the caller can
        invoke the probe again with the same or advanced start_idx after
        more data becomes available.
    """
    _validate_input(df)

    if pip_tolerance_pips is None:
        pip_tolerance_pips = _probe_reset_pips(timeframe)

    # ===== Phase 1: Iterative BOS_0 Probing =====
    original_bos0_bounds: Optional[Tuple[float, float, str]] = None
    current_start = start_idx
    pip_size = _pip_size_from_pair(df)
    tolerance = pip_tolerance_pips * pip_size
    status = "pending"

    # Keep references to the last probe's outputs
    df_probe = df.copy()
    probe_events: List[StructureEvent] = []
    probe_levels: List[StructureLevel] = []
    iteration = 0

    for iteration in range(max_probe_iterations):
        # No timeframe passed → H1 proximity_pips fallback (Step 1 parity
        # carve-out, see _make_market_structure docstring).
        df_probe = df.copy()
        ms = _make_market_structure(df_probe, struct_direction,
                                    start_idx=current_start, structure_id=0,
                                    end_idx=end_idx)
        ms.debug = True
        df_probe, probe_events, probe_levels = ms.run()

        # Find CTS_ESTABLISHED events for structure_id=0
        cts_est = sorted(
            [ev for ev in probe_events
             if ev.type == "CTS_ESTABLISHED"
             and ev.meta.get("structure_id") == 0],
            key=ef.cts_anchor_idx,
        )
        # Test-only path (PLAN_E Q10): its CTS reads go through the anchor
        # accessor, behaviour kept (they read the CTS anchor today).

        # First time: extract original BOS_0 zone bounds
        if original_bos0_bounds is None and len(cts_est) >= 1:
            original_bos0_bounds = _get_bos0_zone_bounds(
                df_probe, probe_events, sid=0, sd=struct_direction)

        # --- Condition 3: Reversal before 2nd CTS_EST → finalized ---
        rev_mask = ((df_probe["market_state"].astype(str).str.lower() == "reversal")
                    & (df_probe["structure_id"].astype(int) == 0))
        has_reversal = rev_mask.any()

        if has_reversal and len(cts_est) < 2:
            status = "finalized"
            break

        # --- Condition 4: Insufficient data for exception check ---
        # We need at least 1 CTS_EST (to anchor the check window's lower bound
        # at its CTS anchor + 1) AND a captured BOS_0 zone (to define the
        # proximity target). Without these, no exception check is possible:
        #   - end_idx defined → finalize at caller's bound
        #   - end_idx None → pending (more candles may resolve later)
        if not cts_est or original_bos0_bounds is None:
            if end_idx is not None:
                status = "finalized"
            else:
                status = "pending"
            break

        # --- Conditions 1 & 2: Evaluate exception ---
        # Window: [ef.cts_anchor_idx(cts_est[0]) + 1, exc_upper]
        #   Lower bound excludes the CTS anchor candle itself — the breakout
        #   span's extreme, whose far wick still belongs to the breakout leg
        #   (GOTCHAS "Exception Check Must Exclude CTS_ESTABLISHED Candle").
        #   We only care if price returns to the zone AFTER the breakout.
        #
        #   Upper bound respects end_idx-supersedes principle (see
        #   MARKET_STRUCTURE_SPEC.md): end_idx is a caller-defined hard
        #   bound on the probe; inner rules like "2 CTS_EST seen" don't
        #   narrow the check window. Fall back to cts_est[1]'s anchor only when
        #   end_idx is None (live-mode probes without an explicit terminal).
        if end_idx is not None:
            exc_upper = end_idx
        elif len(cts_est) >= 2:
            exc_upper = ef.cts_anchor_idx(cts_est[1])
        else:
            # Live mode, only 1 CTS, no end_idx → can't bound the check.
            status = "pending"
            break

        outer, inner, zone_side = original_bos0_bounds
        exc_idx = _find_closest_candle_to_outer(
            df_probe, ef.cts_anchor_idx(cts_est[0]) + 1, exc_upper,
            outer, inner, tolerance, zone_side)

        if exc_idx is None:
            # Condition 1: No exception → finalized
            status = "finalized"
            break

        # Condition 2: Exception triggered → restart from exc_idx
        print(f"[scenario3] BOS_0 probe exception triggered: "
              f"iteration={iteration}, new_start={exc_idx}")
        current_start = exc_idx
        # loop continues with fresh probe...

    # Record the validated start for the result
    phase1_start = current_start

    # ===== Phase 2: Multi-Structure Continuation (only if finalized + requested) =====
    if status == "finalized" and run_continuation:
        df2 = df_probe
        all_events = list(probe_events)
        all_levels = list(probe_levels)
        structure_id = 0
        sd = struct_direction

        max_structures_guard = 20
        for _loop_iter in range(max_structures_guard):
            rev_mask = ((df2["market_state"].astype(str).str.lower() == "reversal")
                        & (df2["structure_id"].astype(int) == structure_id))
            if not rev_mask.any():
                break

            reversal_start_idx = int(df2.loc[rev_mask].index.min())
            reversal_confirmed_idx = int(df2.loc[rev_mask].index.max())

            # Scenario 2 for next structure
            d_next = identify_start_scenario_2_after_reversal(
                df2,
                reversal_idx=reversal_start_idx,
                prev_structure_id=structure_id,
                prev_struct_direction=sd,
                min_history=50,
            )

            next_start_idx = int(d_next.start_idx)
            next_sd = int(d_next.struct_direction)
            next_sid = structure_id + 1

            if next_start_idx == current_start and next_sid == structure_id:
                break

            # Exception 2 probe (same logic as compute_structure)
            # Always run — starts from Exception 1 override if triggered,
            # otherwise from base last CTS idx.
            zone_bounds = _get_last_cts_zone_bounds(
                df2, all_events, structure_id, sd)

            if zone_bounds is not None:
                zb_outer, zb_inner, zb_side = zone_bounds
                pip_tol = pip_tolerance_pips * _pip_size_from_pair(df2)

                # Iterative Exception 2 probing
                exc2_candidate = next_start_idx
                max_exc2_iterations = 10
                exc2_triggered = False

                for _exc2_iter in range(max_exc2_iterations):
                    # No timeframe → H1 proximity_pips fallback (Step 1
                    # parity carve-out, see _make_market_structure docstring).
                    df_exc2_probe = df2.copy()
                    ms_exc2 = _make_market_structure(
                        df_exc2_probe,
                        struct_direction=next_sd,
                        start_idx=exc2_candidate,
                        structure_id=next_sid,
                        end_idx=reversal_confirmed_idx,
                    )
                    ms_exc2.debug = True
                    df_exc2_probe, exc2_events, exc2_levels = ms_exc2.run()

                    exc2_cts = [
                        ev for ev in exc2_events
                        if ev.type == "CTS_ESTABLISHED"
                        and ev.meta.get("structure_id") == next_sid
                        and ef.cts_anchor_idx(ev) <= reversal_confirmed_idx  # test-only (Q10)
                    ]

                    if not exc2_cts:
                        break

                    cts0_anchor_idx = ef.cts_anchor_idx(exc2_cts[0])
                    # Start from CTS_EST + 1: exclude the pullback candle itself
                    exc2_idx = _find_closest_candle_to_outer(
                        df_exc2_probe, cts0_anchor_idx + 1,
                        reversal_confirmed_idx,
                        zb_outer, zb_inner, pip_tol, zb_side)

                    if exc2_idx is None:
                        break

                    print(f"[scenario3] Exception 2 triggered: "
                          f"iteration={_exc2_iter}, start_idx={exc2_idx}")
                    exc2_triggered = True
                    exc2_candidate = exc2_idx

                if exc2_triggered:
                    # At least one exception triggered — discard all probes,
                    # use settled candidate as start in the outer loop
                    current_start = exc2_candidate
                    structure_id = next_sid
                    sd = next_sd
                    continue

                # No exception ever triggered — keep the first probe's data
                df2 = df_exc2_probe
                all_events.extend(exc2_events)
                all_levels.extend(exc2_levels)

                if reversal_confirmed_idx + 1 > df2.index.max():
                    break

                current_start = reversal_confirmed_idx + 1
                sd = next_sd
                structure_id = next_sid
                continue

            # Default path (no zone bounds found). No timeframe → H1=20
            # proximity_pips fallback (Step 1 parity carve-out — this path
            # was probe-only previously, defaults preserved).
            ms_next = _make_market_structure(df2, next_sd,
                                             start_idx=next_start_idx,
                                             structure_id=next_sid)
            ms_next.debug = True
            df2, ms_events, ms_levels = ms_next.run()
            all_events.extend(ms_events)
            all_levels.extend(ms_levels)
            structure_id = next_sid
            sd = next_sd
            current_start = next_start_idx

        notes = (
            f"Scenario3: start={phase1_start} sd={struct_direction} "
            f"probe_iterations={iteration + 1} "
            f"structures={structure_id + 1} events={len(all_events)} "
            f"levels={len(all_levels)} status=finalized"
        )

        return Scenario3Result(
            df=df2,
            levels=all_levels,
            events=all_events,
            struct_direction=struct_direction,
            start_idx=phase1_start,
            original_bos0_bounds=original_bos0_bounds,
            probe_iterations=iteration + 1,
            status="finalized",
            notes=notes,
        )

    # Return probe data as-is (Phase 2 skipped or probe exhausted iterations)
    notes = (
        f"Scenario3: start={phase1_start} sd={struct_direction} "
        f"probe_iterations={iteration + 1} "
        f"events={len(probe_events)} levels={len(probe_levels)} "
        f"status={status}"
    )

    return Scenario3Result(
        df=df_probe,
        levels=list(probe_levels),
        events=list(probe_events),
        struct_direction=struct_direction,
        start_idx=current_start,
        original_bos0_bounds=original_bos0_bounds,
        probe_iterations=iteration + 1,
        status=status,
        notes=notes,
    )


def compute_structure_from_start(
    df: pd.DataFrame,
    start_idx: int,
    struct_direction: int,
    *,
    timeframe: str = "H1",
    end_idx: Optional[int] = None,
) -> StructureEngineResult:
    """Run multi-structure analysis from a known start (no Scenario 1 / no probes).

    Used for lower-TF structures where the start has already been validated
    by a higher-TF probe.

    Flow:
      1. Run MarketStructure for structure_id=0 from start_idx (capped at
         end_idx if provided)
      2. On reversal -> Scenario 2 (Exception 1/2) -> next structure
      3. Repeat until no more reversals, end_idx reached, or end of data

    `end_idx` (Part 4 §13.5.c.ii): inclusive upper bound on the run. When
    set, every MS construction in this function passes it through, so the
    new sid only writes structure cols / emits events within
    `[start_idx, end_idx]`. Outer loop also exits early if the running
    cursor passes `end_idx`. Used by the entity-direct compute path
    (`apply_trigger_to_entity_df`) to bound a sub's lifecycle.
    """
    _validate_input(df)

    df2 = df
    all_events: List[StructureEvent] = []
    all_levels: List[StructureLevel] = []

    cur_start = start_idx
    sd = struct_direction
    structure_id = 0
    pip_size = _pip_size_from_pair(df2)

    max_structures_guard = 20
    for _loop_iter in range(max_structures_guard):
        # Stop if we've already passed the lifecycle bound.
        if end_idx is not None and cur_start > end_idx:
            break
        ms = _make_market_structure(df2, struct_direction=sd, start_idx=cur_start, structure_id=structure_id, timeframe=timeframe, pip_size=pip_size, end_idx=end_idx)
        ms.debug = True
        df2, ms_events, levels = ms.run()

        all_events.extend(ms_events)
        all_levels.extend(levels)

        # Find reversal for this structure_id
        rev_mask = (df2["market_state"].astype(str).str.lower() == "reversal") & (df2["structure_id"].astype(int) == structure_id)
        if not rev_mask.any():
            break

        reversal_start_idx = int(df2.loc[rev_mask].index.min())
        reversal_confirmed_idx = int(df2.loc[rev_mask].index.max())

        # Scenario 2 for next structure
        d_next = identify_start_scenario_2_after_reversal(
            df2,
            reversal_idx=reversal_start_idx,
            prev_structure_id=structure_id,
            prev_struct_direction=sd,
            min_history=50,
        )

        next_start_idx = int(d_next.start_idx)
        next_sd = int(d_next.struct_direction)
        next_sid = structure_id + 1

        if next_start_idx == cur_start and next_sid == structure_id:
            break

        # Exception 2 probe (same logic as compute_structure).
        # NOTE: pip tolerance left hardcoded at 10 here for Part 4 Step 1
        # parity. Spec §4.4 only explicitly listed compute_structure and
        # compute_structure_scenario_3 for the TF-keyed table; this path
        # currently runs on M15 (lower-TF) where the previous 10-pip value
        # was used. Revisit when M5 / other TFs start invoking this.
        zone_bounds = _get_last_cts_zone_bounds(df2, all_events, structure_id, sd)

        if zone_bounds is not None:
            outer, inner, zone_side = zone_bounds
            pip_size = _pip_size_from_pair(df2)
            pip_tolerance = 10 * pip_size

            exc2_candidate = next_start_idx
            max_exc2_iterations = 10
            exc2_triggered = False

            for _exc2_iter in range(max_exc2_iterations):
                # No timeframe → H1 proximity_pips fallback (Step 1
                # parity carve-out, see _make_market_structure docstring).
                df_probe = df2.copy()
                ms_probe = _make_market_structure(
                    df_probe,
                    struct_direction=next_sd,
                    start_idx=exc2_candidate,
                    structure_id=next_sid,
                    end_idx=reversal_confirmed_idx,
                )
                ms_probe.debug = True
                df_probe, probe_events, probe_levels = ms_probe.run()

                probe_cts_events = [
                    ev for ev in probe_events
                    if ev.type == "CTS_ESTABLISHED"
                    and ev.meta.get("structure_id") == next_sid
                    and ef.cts_anchor_idx(ev) <= reversal_confirmed_idx  # test-only (Q10)
                ]

                if not probe_cts_events:
                    break

                cts0_anchor_idx = ef.cts_anchor_idx(probe_cts_events[0])
                exception_2_idx = _find_closest_candle_to_outer(
                    df_probe, cts0_anchor_idx + 1,
                    reversal_confirmed_idx,
                    outer, inner, pip_tolerance, zone_side,
                )

                if exception_2_idx is None:
                    break

                print(f"[structure_from_start] Exception 2 triggered: "
                      f"iteration={_exc2_iter}, start_idx={exception_2_idx}")
                exc2_triggered = True
                exc2_candidate = exception_2_idx

            if exc2_triggered:
                cur_start = exc2_candidate
                sd = next_sd
                structure_id = next_sid
                continue

            # No exception — keep probe data
            df2 = df_probe
            all_events.extend(probe_events)
            all_levels.extend(probe_levels)

            if reversal_confirmed_idx + 1 > df2.index.max():
                break

            cur_start = reversal_confirmed_idx + 1
            sd = next_sd
            structure_id = next_sid
            continue

        # Default path (no zone bounds found)
        cur_start = next_start_idx
        sd = next_sd
        structure_id = next_sid

    notes = (
        f"StructureFromStart: start={start_idx} sd={struct_direction} "
        f"structures={structure_id + 1} events={len(all_events)} "
        f"levels={len(all_levels)}"
    )

    _validate_output(df2)

    return StructureEngineResult(
        df=df2,
        levels=all_levels,
        events=all_events,
        struct_direction=struct_direction,
        notes=notes,
    )


def compute_bounded_structure(
    df: pd.DataFrame,
    start_idx: int,
    struct_direction: int,
    *,
    timeframe: str = "H1",
    end_idx: Optional[int] = None,
    enforce_cts0_new_extreme: bool = False,
    bos0_inner: Optional[float] = None,
) -> BoundedStructureResult:
    """Run a bounded SINGLE directional structure and stop at its first reversal.

    Part 4 §5 (REVISED 2026-05-25) — the bounded single-structure primitive.
    Each subordinate sid is a SINGLE directional structure whose first
    reversal is the boundary to the next sid. This runs exactly one
    MarketStructure (structure_id=0) over ``[start_idx, end_idx]`` and reports
    where it reversed.

    Unlike ``compute_structure_from_start``, this does NOT roll past the
    reversal into further structure_ids, and does NOT run Scenario 2 /
    Exception 2 probing — those are the multi-structure concerns this
    primitive deliberately omits. The stop-at-first-reversal behaviour is
    already intrinsic to ``MarketStructure.run()`` (its loop breaks on
    ``MarketState.REVERSAL``); this wrapper just wires the proximity/zone
    resolvers (via ``_make_market_structure``) and extracts the reversal idx.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with candle features/patterns already applied.
    start_idx : int
        Structural anchor to run from (already validated by the upstream
        probe / identify_start — no internal Scenario 1).
    struct_direction : int
        +1 or -1. The single structure's direction for its whole run.
    timeframe : str
        TF for proximity-pips lookup (H1/M15/M5).
    end_idx : int, optional
        Inclusive upper bound on the run (parent-cycle-end or next subsequent
        trigger). The run is identical to an unbounded run on the frame
        truncated at ``end_idx`` (nothing past it is read — Plan A §2, with
        the ``attrs["imbalances"]`` residual noted there); no event carries an
        ``idx`` past it (asserted in ``MarketStructure.run()``).

    Returns
    -------
    BoundedStructureResult
        ``reversal_idx`` = the first reversal-marked candle (the reversal
        pattern's apply idx), or None if no reversal occurred within
        ``[start_idx, end_idx]``.
    """
    _validate_input(df)

    df2 = df  # MarketStructure copies internally; caller's df is untouched.
    pip_size = _pip_size_from_pair(df2)

    ms = _make_market_structure(
        df2,
        struct_direction=struct_direction,
        start_idx=start_idx,
        structure_id=0,
        timeframe=timeframe,
        pip_size=pip_size,
        end_idx=end_idx,
        enforce_cts0_new_extreme=enforce_cts0_new_extreme,
        bos0_inner=bos0_inner,
    )
    ms.debug = True
    df2, events, levels = ms.run()

    # Reversal detection: same predicate compute_structure_from_start uses.
    # A single run only ever writes structure_id=0; the structure_id==0
    # filter also excludes any rows the terminal stamping marked "reversal"
    # past end_idx (those were never written, so keep structure_id=-1).
    rev_mask = (
        (df2["market_state"].astype(str).str.lower() == "reversal")
        & (df2["structure_id"].astype(int) == 0)
    )
    reversal_idx = int(df2.loc[rev_mask].index.min()) if rev_mask.any() else None

    _validate_output(df2)

    notes = (
        f"BoundedStructure: start={start_idx} sd={struct_direction} "
        f"end_idx={end_idx} reversal_idx={reversal_idx} "
        f"events={len(events)} levels={len(levels)}"
    )

    return BoundedStructureResult(
        df=df2,
        levels=levels,
        events=events,
        struct_direction=struct_direction,
        start_idx=start_idx,
        end_idx=end_idx,
        reversal_idx=reversal_idx,
        notes=notes,
    )


def _validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_CANDLE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[structure_engine] Missing required columns: {missing}")
    if df.empty:
        raise ValueError("[structure_engine] Input df is empty")


def _validate_output(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_CANDLE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[structure_engine] Output df missing required columns: {missing}")


# ---------------------------------------------------------------------
# Exception 2 Probe Helpers
# ---------------------------------------------------------------------

def _pip_size_from_pair(df: pd.DataFrame) -> float:
    """Get pip size (0.01 for JPY pairs, 0.0001 otherwise)."""
    pair = df.attrs.get("pair", "")
    return 0.01 if "JPY" in str(pair).upper() else 0.0001


def _get_last_cts_zone_bounds(
    df: pd.DataFrame,
    events: List[StructureEvent],
    sid: int,
    sd: int,
) -> Optional[Tuple[float, float, str]]:
    """
    Get (outer, inner, zone_side) bounds of last CTS zone for given structure_id.
    For uptrend (sd=+1): CTS zone is sell zone, outer=top, inner=bottom
    For downtrend (sd=-1): CTS zone is buy zone, outer=bottom, inner=top

    Returns: (outer, inner, zone_side) or None if no CTS zone found.
    """
    zones = derive_kl_zones_v1(df, events, struct_direction=sd)
    cts_zones = [
        z for z in zones
        if z.source_kind == "CTS" and z.meta.get("structure_id") == sid
    ]
    if not cts_zones:
        return None

    # Sort by confirmed_idx to get the last one
    cts_zones.sort(key=lambda z: z.meta.get("confirmed_idx", -1))
    last = cts_zones[-1]

    # For sell zone: outer=top, inner=bottom
    # For buy zone: outer=bottom, inner=top
    if last.side == "sell":
        return (float(last.top), float(last.bottom), "sell")
    else:
        return (float(last.bottom), float(last.top), "buy")


def _get_bos0_zone_bounds(
    df: pd.DataFrame,
    events: List[StructureEvent],
    sid: int,
    sd: int,
) -> Optional[Tuple[float, float, str]]:
    """
    Get (outer, inner, zone_side) of BOS_0 zone for given structure_id.
    BOS_0 exists after 1st CTS_ESTABLISHED.
    """
    zones = derive_kl_zones_v1(df, events, struct_direction=sd)
    bos0 = [z for z in zones
            if z.source_kind == "BOS"
            and z.meta.get("structure_id") == sid
            and z.meta.get("cycle_id") == 0]
    if not bos0:
        return None
    zone = bos0[0]
    if zone.side == "sell":
        return (float(zone.top), float(zone.bottom), "sell")
    else:
        return (float(zone.bottom), float(zone.top), "buy")


def _find_closest_candle_to_outer(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    outer: float,
    inner: float,
    tolerance: float,
    zone_side: str,
) -> Optional[int]:
    """
    Find candle whose price is closest to outer bound.
    If that price is within tolerance of inner or crosses into zone, return that idx.
    For sell zone: look at highs, threshold = inner - tolerance
    For buy zone: look at lows, threshold = inner + tolerance
    """
    seg = df.loc[start_idx:end_idx]
    if seg.empty:
        return None

    if zone_side == "sell":
        prices = seg["h"].astype(float)
        closest_idx = int((prices - outer).abs().idxmin())
        if float(df.loc[closest_idx, "h"]) >= inner - tolerance:
            return closest_idx
    else:
        prices = seg["l"].astype(float)
        closest_idx = int((prices - outer).abs().idxmin())
        if float(df.loc[closest_idx, "l"]) <= inner + tolerance:
            return closest_idx

    return None
