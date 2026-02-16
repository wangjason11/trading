from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from engine_v2.common.types import REQUIRED_CANDLE_COLS, StructureLevel
from engine_v2.structure.market_structure import MarketStructure, StructureEvent
from engine_v2.structure.identify_start import (
    identify_start_scenario_1,
    identify_start_scenario_2_after_reversal,
)
from engine_v2.zones.kl_zones_v1 import derive_kl_zones_v1


@dataclass
class StructureEngineResult:
    df: pd.DataFrame
    levels: List[StructureLevel]
    events: List[StructureEvent]
    struct_direction: int
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


def compute_structure(df: pd.DataFrame) -> StructureEngineResult:
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

    # --- Scenario 1 (initial start identification) ---
    input_idx = int(df.index.max())
    d0 = identify_start_scenario_1(df, input_idx=input_idx, lookback_days=183, min_history=50)

    df2 = df
    all_events: List[StructureEvent] = []
    all_levels: List[StructureLevel] = []

    start_idx = int(d0.start_idx)
    struct_direction = int(d0.struct_direction)
    structure_id = 0

    # Run multiple structure segments until no more reversals (or we hit end)
    max_structures_guard = 20  # safety guard against infinite loops
    for loop_iter in range(max_structures_guard):
        ms = MarketStructure(df2, struct_direction=struct_direction, start_idx=start_idx, structure_id=structure_id)
        ms.debug = True
        df2, ms_events, levels = ms.run()

        all_events.extend(ms_events)
        all_levels.extend(levels)

        # Find reversal idx for THIS structure_id (MarketStructure stops when it hits reversal)
        # We detect it from df to avoid depending on event details.
        rev_mask = (df2["market_state"].astype(str).str.lower() == "reversal") & (df2["structure_id"].astype(int) == structure_id)
        if not rev_mask.any():
            break

        reversal_start_idx = int(df2.loc[rev_mask].index.min())
        reversal_confirmed_idx = int(df2.loc[rev_mask].index.max())

        # --- Scenario 2 (next start after reversal) ---
        # identify_start_scenario_2_after_reversal handles Exception 1 internally
        d_next = identify_start_scenario_2_after_reversal(
            df2,
            reversal_idx=reversal_start_idx,
            prev_structure_id=structure_id,
            prev_struct_direction=struct_direction,
            min_history=50,
        )

        next_start_idx = int(d_next.start_idx)
        next_struct_direction = int(d_next.struct_direction)
        next_structure_id = int(structure_id + 1)

        # Guard: if next start doesn't advance logically, stop to avoid loops
        if next_start_idx == start_idx and next_structure_id == structure_id:
            break

        # ========== Exception 2 Probe Logic ==========
        # Always run — starts from Exception 1 override if triggered,
        # otherwise from base last CTS idx.

        # Get last CTS zone bounds for Exception 2 evaluation
        zone_bounds = _get_last_cts_zone_bounds(df2, all_events, structure_id, struct_direction)

        if zone_bounds is not None:
            outer, inner, zone_side = zone_bounds
            pip_size = _pip_size_from_pair(df2)
            pip_tolerance = 10 * pip_size

            # Iterative Exception 2 probing (re-probe when exception triggers)
            exc2_candidate = next_start_idx
            max_exc2_iterations = 10
            exc2_triggered = False

            for _exc2_iter in range(max_exc2_iterations):
                # Run probe (on COPY of df, separate events list)
                df_probe = df2.copy()
                ms_probe = MarketStructure(
                    df_probe,
                    struct_direction=next_struct_direction,
                    start_idx=exc2_candidate,
                    structure_id=next_structure_id,
                    end_idx=reversal_confirmed_idx,  # Stop at reversal confirmed (inclusive)
                )
                ms_probe.debug = True
                df_probe, probe_events, probe_levels = ms_probe.run()

                # Check for CTS established in probe (CTS_ESTABLISHED event)
                probe_cts_events = [
                    ev for ev in probe_events
                    if ev.type == "CTS_ESTABLISHED"
                    and ev.meta.get("structure_id") == next_structure_id
                    and ev.idx <= reversal_confirmed_idx
                ]

                if not probe_cts_events:
                    break

                # Get CTS established idx from the first CTS_ESTABLISHED in probe
                cts_established_idx = int(probe_cts_events[0].idx)

                # Evaluate Exception 2: find candle closest to outer bound
                exception_2_idx = _find_closest_candle_to_outer(
                    df_probe,
                    cts_established_idx,
                    reversal_confirmed_idx,
                    outer,
                    inner,
                    pip_tolerance,
                    zone_side,
                )

                if exception_2_idx is None:
                    break

                # Exception 2 triggered — discard probe, re-probe from new idx
                print(f"[structure_engine] Exception 2 triggered: "
                      f"iteration={_exc2_iter}, start_idx={exception_2_idx}")
                exc2_triggered = True
                exc2_candidate = exception_2_idx

            if exc2_triggered:
                # At least one exception triggered — discard all probes,
                # use settled candidate as start in the outer loop
                start_idx = exc2_candidate
                struct_direction = next_struct_direction
                structure_id = next_structure_id
                continue

            # No exception ever triggered — keep the first probe's data
            df2 = df_probe
            all_events.extend(probe_events)
            all_levels.extend(probe_levels)

            # Continue from next candle after reversal confirmed
            if reversal_confirmed_idx + 1 > df2.index.max():
                break  # No more data

            # Move to next structure, continuing from after reversal
            start_idx = reversal_confirmed_idx + 1
            struct_direction = next_struct_direction
            structure_id = next_structure_id
            continue

        # Default path (no zone bounds found)
        # Move to next structure - events will be written in the next loop iteration
        start_idx = next_start_idx
        struct_direction = next_struct_direction
        structure_id = next_structure_id

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
    pip_tolerance_pips: int = 10,
    max_probe_iterations: int = 10,
) -> Scenario3Result:
    """
    Scenario 3: Arbitrary start with iterative BOS_0 probe.

    Phase 1 — Iterative probing to validate/refine start_idx via BOS_0 zone
    proximity checking.  Phase 2 — Multi-structure continuation from the
    finalized probe (same logic as compute_structure lines 69-176).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with candle features/patterns already applied.
    start_idx : int
        Arbitrary start index to probe from.
    struct_direction : int
        +1 for uptrend, -1 for downtrend.
    pip_tolerance_pips : int
        Pip tolerance for zone proximity checking (default 15).
    max_probe_iterations : int
        Maximum number of probe restarts before giving up (default 10).

    Returns
    -------
    Scenario3Result
        Contains df, levels, events, status ("finalized" or "pending"),
        original_bos0_bounds, probe_iterations, and notes.
    """
    _validate_input(df)

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
        df_probe = df.copy()
        ms = MarketStructure(df_probe, struct_direction,
                             start_idx=current_start, structure_id=0)
        ms.debug = True
        df_probe, probe_events, probe_levels = ms.run()

        # Find CTS_ESTABLISHED events for structure_id=0
        cts_est = sorted(
            [ev for ev in probe_events
             if ev.type == "CTS_ESTABLISHED"
             and ev.meta.get("structure_id") == 0],
            key=lambda e: e.idx,
        )

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

        # --- Condition 4: Data ends before 2nd CTS_EST → pending ---
        if len(cts_est) < 2 or original_bos0_bounds is None:
            status = "pending"
            break

        # --- Conditions 1 & 2: Evaluate exception ---
        outer, inner, zone_side = original_bos0_bounds
        exc_idx = _find_closest_candle_to_outer(
            df_probe, cts_est[0].idx, cts_est[1].idx,
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

    # ===== Phase 2: Multi-Structure Continuation (only if finalized) =====
    if status == "finalized":
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
                    df_exc2_probe = df2.copy()
                    ms_exc2 = MarketStructure(
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
                        and ev.idx <= reversal_confirmed_idx
                    ]

                    if not exc2_cts:
                        break

                    cts_est_idx = int(exc2_cts[0].idx)
                    exc2_idx = _find_closest_candle_to_outer(
                        df_exc2_probe, cts_est_idx,
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

            # Default path (no zone bounds found)
            ms_next = MarketStructure(df2, next_sd,
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

    # Pending — return probe data as-is (accessible but not finalized)
    notes = (
        f"Scenario3: start={phase1_start} sd={struct_direction} "
        f"probe_iterations={iteration + 1} "
        f"events={len(probe_events)} levels={len(probe_levels)} "
        f"status=pending"
    )

    return Scenario3Result(
        df=df_probe,
        levels=list(probe_levels),
        events=list(probe_events),
        struct_direction=struct_direction,
        start_idx=current_start,
        original_bos0_bounds=original_bos0_bounds,
        probe_iterations=iteration + 1,
        status="pending",
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
