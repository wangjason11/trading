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
      - Exception 2 (probe): If Exception 1 not triggered, run probe to reversal_confirmed.
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
        # Only run probe if Exception 1 was NOT triggered (reason is base_last_cts_confirmed)
        exception_1_triggered = "exception1" in d_next.reason

        if not exception_1_triggered:
            original_candidate_idx = next_start_idx  # This is last CTS idx from Exception 1 check

            # Get last CTS zone bounds for Exception 2 evaluation
            zone_bounds = _get_last_cts_zone_bounds(df2, all_events, structure_id, struct_direction)

            if zone_bounds is not None:
                outer, inner, zone_side = zone_bounds
                pip_size = _pip_size_from_pair(df2)
                pip_tolerance = 15 * pip_size

                # Run probe (on COPY of df, separate events list)
                df_probe = df2.copy()
                ms_probe = MarketStructure(
                    df_probe,
                    struct_direction=next_struct_direction,
                    start_idx=original_candidate_idx,
                    structure_id=next_structure_id,
                    end_idx=reversal_confirmed_idx,  # Stop at reversal confirmed (inclusive)
                )
                ms_probe.debug = True
                df_probe, probe_events, probe_levels = ms_probe.run()

                # Check for pullback confirmed in probe (CTS_CONFIRMED event)
                probe_cts_events = [
                    ev for ev in probe_events
                    if ev.type == "CTS_CONFIRMED"
                    and ev.meta.get("structure_id") == next_structure_id
                    and ev.idx <= reversal_confirmed_idx
                ]

                if probe_cts_events:
                    # Get pullback confirmed idx from the first CTS_CONFIRMED in probe
                    pullback_confirmed_idx = int(probe_cts_events[0].idx)

                    # Evaluate Exception 2: find candle closest to outer bound
                    exception_2_idx = _find_closest_candle_to_outer(
                        df_probe,
                        pullback_confirmed_idx,
                        reversal_confirmed_idx,
                        outer,
                        inner,
                        pip_tolerance,
                        zone_side,
                    )

                    if exception_2_idx is not None:
                        # Exception 2 triggered - discard probe, start fresh from this idx
                        print(f"[structure_engine] Exception 2 triggered: start_idx={exception_2_idx}")
                        next_start_idx = exception_2_idx
                        # Continue to next loop iteration with the new start_idx
                        start_idx = next_start_idx
                        struct_direction = next_struct_direction
                        structure_id = next_structure_id
                        continue

                # No Exception 2 - keep probe data
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

        # Default path (Exception 1 triggered OR no zone bounds found)
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
