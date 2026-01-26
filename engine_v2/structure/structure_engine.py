from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from engine_v2.common.types import REQUIRED_CANDLE_COLS, StructureLevel
from engine_v2.patterns.structure_patterns import BreakoutPatterns
from engine_v2.structure.market_structure import MarketStructure, StructureEvent
from engine_v2.structure.identify_start import (
    identify_start_scenario_1,
    identify_start_scenario_2_after_reversal,
)
from engine_v2.zones.kl_zones_v1 import derive_kl_zones_v1
# from engine_v2.structure.market_structure import MarketStructure


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

    Week 6 note:
      - struct_direction is still hardcoded here for now (1) until Week 6 Part 2
        where we implement IdentifyStart / candle of analysis logic.
    """
    _validate_input(df)

    # --- LEGACY HOOK (edit this block only) -----------------------
    # from engine_v2.structure.market_structure import MarketStructure

    # # For now, we pass struct_direction from upstream later (Week 6 Part 2 will compute it).
    # # Temporary: assume struct_direction=1 for your current replay window (uptrend).
    # struct_direction = 1

    # ms = MarketStructure(df, struct_direction=struct_direction)
    # ms.debug = True  # temp: enable reversal debug prints
    # df2, ms_events, levels = ms.run()

    # notes = f"MarketStructure v1: struct_direction={struct_direction}, events={len(ms_events)}, levels={len(levels)}"
    # --------------------------------------------------------------

    # --- NEW UPDATED Starting direction & candle Identification -----------------------

    # --- Scenario 1 (initial) ---
    input_idx = int(df.index.max())
    d0 = identify_start_scenario_1(df, input_idx=input_idx, lookback_days=183, min_history=50)

    df2 = df
    all_events: List[StructureEvent] = []
    all_levels: List[StructureLevel] = []

    # start_idx = 13
    # struct_direction = 1
    
    start_idx = int(d0.start_idx)
    struct_direction = int(d0.struct_direction)
    structure_id = 0

    # Run multiple structure segments until no more reversals (or we hit end)
    max_structures_guard = 20  # safety guard against infinite loops
    for _ in range(max_structures_guard):
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

        reversal_idx = int(df2.loc[rev_mask].index.min())

        # --- Scenario 2 (next start after reversal) ---
        # d_next = identify_start_scenario_2_after_reversal(
        #     df2,
        #     reversal_idx=reversal_idx,
        #     prev_structure_id=structure_id,
        #     prev_struct_direction=struct_direction,
        #     min_history=50,
        # )

        # next_start_idx = int(d_next.start_idx)
        # next_struct_direction = int(d_next.struct_direction)
        # next_structure_id = int(structure_id + 1)

        # # Guard: if next start doesn't advance logically, stop to avoid loops
        # if next_start_idx == start_idx and next_structure_id == structure_id:
        #     break

        # # Move to next structure
        # start_idx = next_start_idx
        # struct_direction = next_struct_direction
        # structure_id = next_structure_id

        # --- Scenario 2 (next start after reversal) ---
        d_next = identify_start_scenario_2_after_reversal(
            df2,
            reversal_idx=reversal_idx,
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

        # -------------------------
        # Scenario 2 Exception #2 (your clarified rule)
        # -------------------------
        pip_size = _pip_size_from_pair(df2)
        tol = 15.0 * float(pip_size)

        # prior structure’s last CONFIRMED CTS zone inner bound
        prior_inner = _last_confirmed_cts_zone_inner(
            df2,
            all_events,
            sid=int(structure_id),
            prev_sd=int(struct_direction),
        )

        # Only attempt Exception #2 if we can resolve prior inner bound
        if prior_inner is not None:
            # PROBE run the next structure once (do not commit events yet)
            ms_probe = MarketStructure(df2, struct_direction=next_struct_direction, start_idx=next_start_idx, structure_id=next_structure_id)
            ms_probe.debug = True
            df_probe, probe_events, probe_levels = ms_probe.run()

            # Find FIRST CTS_ESTABLISHED for this new structure_id (authoritative = event meta)
            first_est = None
            for ev in probe_events:
                if ev.type != "CTS_ESTABLISHED":
                    continue
                sid_ev = int((ev.meta or {}).get("structure_id", -1))
                if sid_ev != next_structure_id:
                    continue
                first_est = ev
                break

            exception2_triggered = False
            revised_start_idx = None

            if first_est is not None:
                est_meta = first_est.meta or {}
                anchor_idx = int(est_meta.get("anchor_idx", -1))
                pattern_name = str(est_meta.get("pattern_name", ""))
                confirmed_at = int(est_meta.get("confirmed_at", first_est.idx))
                est_sd = int(est_meta.get("struct_direction", next_struct_direction))

                # 1) Check "price afterwards touches into (or within 15 pips of) prior CTS zone inner bound"
                # We interpret touch using wick relative to the side implied by the PRIOR CTS zone side.
                # For prior structure uptrend, CTS zone is typically sell; for downtrend, CTS zone typically buy.
                prior_cts_side = "sell" if int(struct_direction) == 1 else "buy"

                touch_found = False
                if confirmed_at + 1 in df_probe.index:
                    seg = df_probe.loc[confirmed_at + 1 :]
                    if not seg.empty:
                        if prior_cts_side == "sell":
                            # approach sell zone from below: highs get near inner
                            touch_found = bool((seg["h"].astype(float) >= (float(prior_inner) - tol)).any())
                        else:
                            # approach buy zone from above: lows get near inner
                            touch_found = bool((seg["l"].astype(float) <= (float(prior_inner) + tol)).any())

                # 2) Check "CTS_established breakout candle pattern breaks prior inner bound"
                # Per your instruction: rerun the EXACT SAME pattern on the SAME anchor candle, same direction,
                # but this time with break_threshold = prior_inner.
                breaks_prior_inner = False
                if anchor_idx >= 0 and pattern_name:
                    bp = BreakoutPatterns(df_probe)
                    fn = getattr(bp, pattern_name, None)
                    if callable(fn):
                        ev_check = fn(anchor_idx, est_sd, break_threshold=float(prior_inner), do_confirm=True)
                        if ev_check is not None and ev_check.name == pattern_name and ev_check.direction == est_sd:
                            # must "pass" (SUCCESS or CONFIRMED)
                            if str(ev_check.status) in ("PatternStatus.SUCCESS", "PatternStatus.CONFIRMED") or str(ev_check.status) in ("SUCCESS", "CONFIRMED"):
                                breaks_prior_inner = True

                if touch_found and breaks_prior_inner:
                    # 3) Revised start: closest wick to inner bound by absolute distance,
                    # must be within 15 pips if it doesn't cross into zone.
                    # Search between the ORIGINAL next start (next_start_idx) and the reversal candle.
                    revised = _closest_wick_idx_to_inner(
                        df2,
                        i0=int(next_start_idx),
                        i1=int(reversal_idx),
                        inner=float(prior_inner),
                        tol=float(tol),
                        zone_side=prior_cts_side,
                    )
                    if revised is not None:
                        exception2_triggered = True
                        revised_start_idx = int(revised)

            # If triggered: discard probe, rerun next structure from revised start (no pruning of sid=0 history)
            if exception2_triggered and revised_start_idx is not None:
                ms2 = MarketStructure(df2, struct_direction=next_struct_direction, start_idx=revised_start_idx, structure_id=next_structure_id)
                ms2.debug = True
                df2, ms_events, levels = ms2.run()

                all_events.extend(ms_events)
                all_levels.extend(levels)

                # advance to next structure after rerun
                start_idx = revised_start_idx
                struct_direction = next_struct_direction
                structure_id = next_structure_id
                continue

            # If not triggered: accept probe output and move on normally
            df2 = df_probe
            all_events.extend(probe_events)
            all_levels.extend(probe_levels)

            start_idx = next_start_idx
            struct_direction = next_struct_direction
            structure_id = next_structure_id
            continue

        # No prior_inner available: proceed normally without Exception #2
        start_idx = next_start_idx
        struct_direction = next_struct_direction
        structure_id = next_structure_id


    notes = (
        f"MarketStructure v1: initial_start={d0.start_idx} initial_sd={d0.struct_direction} "
        f"structures={structure_id + 1} events={len(all_events)} levels={len(all_levels)} "
        f"start_reason={d0.reason}"
    )
    # --------------------------------------------------------------


    _validate_output(df2)

    # return StructureEngineResult(
    #     df=df2,
    #     levels=levels,
    #     events=ms_events,
    #     struct_direction=struct_direction,
    #     notes=notes,
    # )

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

def _pip_size_from_pair(dfx: pd.DataFrame) -> float:
    pair = (dfx.attrs or {}).get("pair", "")
    return 0.01 if "JPY" in str(pair).upper() else 0.0001

def _last_confirmed_cts_zone_inner(
    dfx: pd.DataFrame,
    evs: List[StructureEvent],
    *,
    sid: int,
    prev_sd: int,
) -> float | None:
    # Build zones from all events; then grab latest CTS zone for prev structure_id
    zones = derive_kl_zones_v1(dfx, evs, struct_direction=int(prev_sd))
    cts_z = [z for z in zones if (z.source_kind == "CTS") and int((z.meta or {}).get("structure_id", -1)) == int(sid)]
    if not cts_z:
        return None
    # pick most recent by confirmed_idx if present, else by start_time
    cts_z.sort(key=lambda z: int((z.meta or {}).get("confirmed_idx", -1)))
    z_last = cts_z[-1]
    inner = (z_last.meta or {}).get("inner", None)
    return float(inner) if inner is not None else None

def _closest_wick_idx_to_inner(
    dfx: pd.DataFrame,
    *,
    i0: int,
    i1: int,
    inner: float,
    tol: float,
    zone_side: str,
) -> int | None:
    """
    Your clarified rule:
      - choose the candle whose WICK is closest to inner bound by absolute distance
      - must be at least within 15 pips to the inner bound if it doesn't cross into zone
    We interpret:
      - sell zone touch uses HIGH wick
      - buy zone touch uses LOW wick
      - eligible if wick is within tol of inner OR crosses inner
    """
    if i1 < i0:
        return None

    seg = dfx.loc[i0:i1]
    if seg.empty:
        return None

    if zone_side == "sell":
        wick = seg["h"].astype(float)
        eligible = wick >= (inner - tol)   # within tol OR crosses inner
    else:
        wick = seg["l"].astype(float)
        eligible = wick <= (inner + tol)

    seg2 = seg.loc[eligible]
    if seg2.empty:
        return None

    wick2 = (seg2["h"].astype(float) if zone_side == "sell" else seg2["l"].astype(float))
    dist = (wick2 - float(inner)).abs()
    best_idx = int(dist.idxmin())
    return best_idx


def _validate_output(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_CANDLE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[structure_engine] Output df missing required columns: {missing}")
