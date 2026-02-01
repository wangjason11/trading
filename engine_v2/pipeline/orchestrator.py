from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd

from engine_v2.common.types import PatternEvent, StructureLevel, REQUIRED_CANDLE_COLS
from engine_v2.features.candle_classifier import apply_candle_classification
from engine_v2.patterns.pattern_engine import detect_patterns
from engine_v2.structure.structure_engine import compute_structure

# NEW: KL base features computed BEFORE structure
from engine_v2.zones.kl_zones_v1 import compute_base_features, derive_kl_zones_v1

# Week 7: POI zones
from engine_v2.zones.poi_zones import derive_poi_zones, POIConfig
from engine_v2.patterns.imbalance import compute_imbalance

# Week 7: Fib tracking
from engine_v2.zones.fib_tracker import FibTracker, FibTrackerConfig


@dataclass
class PipelineResult:
    df: pd.DataFrame
    patterns: List[PatternEvent]
    structure: List[StructureLevel]
    meta: Dict[str, Any]


def run_pipeline(df: pd.DataFrame) -> PipelineResult:
    """
    Orchestrator (Week 6 KL Zones ordering):

      input df
        -> candle classification (candles_v2 features)
        -> pattern engine
        -> KL base feature enrichment (base_pattern/base bounds features)
        -> market structure (df + structure_events)
        -> KL zones derived from structure confirmation events
        -> attach df.attrs["kl_zones"] for charting

    Zones remain event-driven and do not add rewinds/waits.
    """
    _validate_input(df)

    # 1) Candle features
    c_res = apply_candle_classification(df)

    # 2) Pattern engine (structure patterns used by market structure)
    p_res = detect_patterns(c_res.df)

    # ✅ Debug: confirm structure-pattern markers exist
    print("[patterns]", p_res.notes)
    print(p_res.df["pat"].value_counts().head())

    # 3) Imbalance patterns (Week 7) - computed as columns, before base features
    df_with_imbalance = compute_imbalance(p_res.df)
    imbalance_count = int(df_with_imbalance["is_imbalance"].sum())
    print(f"[imbalance] total={imbalance_count}")

    # 5) KL base features (must occur BEFORE structure)
    # Default length_threshold locked to 0.7 (per Week 6 spec)
    df_with_base = compute_base_features(df_with_imbalance, length_threshold=0.7)

    # 6) Market structure (must return events + struct_direction)
    s_res = compute_structure(df_with_base)

    meta = {
        "notes": {
            "candles": c_res.notes,
            "patterns": p_res.notes,
            "structure": s_res.notes,
        }
    }

    # 7) KL zones consume structure events (not levels)
    kl_zones = derive_kl_zones_v1(
        s_res.df,
        s_res.events,
        struct_direction=s_res.struct_direction,
        length_threshold=0.7,
    )

    print("[kl_zones] total=", len(kl_zones))
    if kl_zones:
        from collections import Counter
        print("[kl_zones] base_pattern counts:", Counter([z.meta.get("base_pattern") for z in kl_zones]).most_common(10))
        print("[kl_zones] active buy:", sum(1 for z in kl_zones if z.side=="buy" and z.meta.get("active")))
        print("[kl_zones] active sell:", sum(1 for z in kl_zones if z.side=="sell" and z.meta.get("active")))

    meta["kl_zones"] = kl_zones

    # 8) Fib tracking (Week 7) - process events to track Fib levels
    fib_tracker = FibTracker(FibTrackerConfig(
        fib_levels=[30.0, 50.0, 61.8, 80.0],
        fill_threshold=0.70,
    ))

    # Process events to track Fib activations
    # Need to pair BOS_CONFIRMED with subsequent CTS_ESTABLISHED
    bos_by_cycle = {}  # {(sid, cycle_id): (bos_idx, bos_price)}

    # Find reversal_confirmed_idx per structure (from REVERSAL_CANDIDATE apply_idx)
    # This is the idx where the reversal from previous structure was confirmed
    reversal_confirmed_by_sid = {}  # {sid: apply_idx}
    for ev in s_res.events:
        if ev.type == "REVERSAL_CANDIDATE":
            prev_sid = ev.meta.get("structure_id", 0)
            apply_idx = ev.meta.get("apply_idx")
            if apply_idx is not None:
                # The reversal from structure N creates structure N+1
                # Store the apply_idx for the NEW structure (prev_sid + 1)
                reversal_confirmed_by_sid[prev_sid + 1] = apply_idx

    # Sort events by idx to ensure BOS_CONFIRMED comes before CTS_ESTABLISHED
    sorted_events = sorted(s_res.events, key=lambda e: (e.idx, e.type))

    # Track previous structure's direction (for Scenario 1 revert check)
    prev_sd_by_sid = {}  # {sid: prev_sd}
    for ev in s_res.events:
        if ev.type == "REVERSAL_CANDIDATE":
            prev_sid = ev.meta.get("structure_id", 0)
            prev_sd = ev.meta.get("struct_direction", 0)
            # The reversal from structure N creates structure N+1
            # Store the direction for the NEW structure (prev_sid + 1)
            prev_sd_by_sid[prev_sid + 1] = prev_sd

    def _get_prev_bos_outer(sid: int) -> tuple:
        """Get max expanded outer threshold of prev structure's last BOS zone.

        Returns (prev_bos_outer, prev_sd) or (None, None) if not available.
        """
        prev_sid = sid - 1
        if prev_sid < 0:
            return None, None

        # Find last BOS zone from prev structure
        prev_bos_zones = [
            z for z in kl_zones
            if z.meta.get("structure_id") == prev_sid and z.source_kind == "BOS"
        ]
        if not prev_bos_zones:
            return None, None

        # Take the last one (highest cycle_id)
        last_bos_zone = max(prev_bos_zones, key=lambda z: z.meta.get("cycle_id", 0))

        # Get prev_sd from zone side: buy zone = bullish (sd=1), sell zone = bearish (sd=-1)
        prev_sd = 1 if last_bos_zone.side == "buy" else -1

        # Get max expanded outer threshold
        bounds_steps = last_bos_zone.meta.get("bounds_steps", [])
        if not bounds_steps:
            return last_bos_zone.meta.get("outer"), prev_sd

        # For buy zone (bullish, outer is bottom): find min bottom (most expanded outward)
        # For sell zone (bearish, outer is top): find max top (most expanded outward)
        if prev_sd == 1:  # Buy zone - outer is bottom
            max_expanded_outer = min(step["bottom"] for step in bounds_steps)
        else:  # Sell zone - outer is top
            max_expanded_outer = max(step["top"] for step in bounds_steps)

        return max_expanded_outer, prev_sd

    for ev in sorted_events:
        sid = ev.meta.get("structure_id", 0)
        cycle_id = ev.meta.get("cycle_id", 0)
        key = (sid, cycle_id)

        if ev.type == "BOS_CONFIRMED":
            # Store BOS info for this cycle
            bos_by_cycle[key] = (ev.idx, ev.price)

        elif ev.type == "CTS_ESTABLISHED":
            # Try to activate Fib using stored BOS info
            if key in bos_by_cycle:
                bos_idx, bos_price = bos_by_cycle[key]
                # Pass reversal_confirmed_idx for Scenario 1/2/3 logic (sid 1+ only)
                reversal_idx = reversal_confirmed_by_sid.get(sid)

                # For cycle 1, get prev BOS zone outer for Scenario 1 revert check
                prev_bos_outer, prev_sd = None, None
                if sid >= 1 and cycle_id == 1:
                    prev_bos_outer, prev_sd = _get_prev_bos_outer(sid)

                fib_tracker.on_cts_established(
                    ev, s_res.df, bos_idx, bos_price, reversal_idx,
                    prev_bos_outer, prev_sd
                )

        elif ev.type == "CTS_UPDATED":
            # Update CTS anchor if Fib is active for this cycle
            reversal_idx = reversal_confirmed_by_sid.get(sid)
            fib_tracker.on_cts_updated(ev, s_res.df, reversal_idx)

        elif ev.type == "CTS_CONFIRMED":
            # Lock the Fib
            fib_tracker.on_cts_confirmed(ev)

    fib_states = fib_tracker.get_fibs_for_charting()
    print(f"[fib_tracker] total fibs={len(fib_states)}, active={sum(1 for f in fib_states if f.active)}")
    meta["fib_states"] = fib_states

    # 9) Prev BOS lines - horizontal lines showing last BOS price of N-1 structure after reversal
    # These help visualize the Scenario 1 revert check threshold
    prev_bos_lines = []

    # Find last BOS_CONFIRMED per structure
    last_bos_by_sid = {}  # {sid: (idx, price)}
    for ev in sorted_events:
        if ev.type == "BOS_CONFIRMED":
            sid = ev.meta.get("structure_id", 0)
            # Always update - we want the LAST BOS for each structure
            last_bos_by_sid[sid] = (ev.idx, ev.price)

    # For each structure N >= 1 (post-reversal), compute the prev BOS line
    for sid, rv_idx in reversal_confirmed_by_sid.items():
        prev_sid = sid - 1
        if prev_sid not in last_bos_by_sid:
            continue

        start_idx, price = last_bos_by_sid[prev_sid]

        # Find earliest CTS_ESTABLISHED or CTS_UPDATED for structure N at or after rv_idx
        end_idx = None
        for ev in sorted_events:
            ev_sid = ev.meta.get("structure_id", 0)
            if ev_sid != sid:
                continue
            if ev.type not in ("CTS_ESTABLISHED", "CTS_UPDATED"):
                continue
            if ev.idx >= rv_idx:
                end_idx = ev.idx
                break  # First one found (events are sorted by idx)

        if end_idx is not None:
            prev_bos_lines.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "price": price,
                "structure_id": sid,  # The NEW structure (post-reversal)
                "prev_structure_id": prev_sid,
            })
            print(f"[prev_bos_line] sid={sid}: start_idx={start_idx} end_idx={end_idx} price={price:.5f}")

    meta["prev_bos_lines"] = prev_bos_lines

    # 10) POI zones (Week 7) - consume Fib states + IC identification
    poi_config = POIConfig(
        ic_fib_min=61.8,
        ic_fib_max=80.0,
        v30_threshold=0.30,
        v60_threshold=0.60,
        v90_threshold=0.90,
        fill_threshold=0.70,
    )
    poi_zones = derive_poi_zones(
        s_res.df,
        s_res.events,
        fib_tracker=fib_tracker,
        config=poi_config,
    )
    print("[poi_zones] total=", len(poi_zones))
    meta["poi_zones"] = poi_zones
    meta["fib_tracker"] = fib_tracker

    # For chart overlay (export_plotly reads df.attrs)
    # Note: imbalance is now in columns (is_imbalance, imbalance_gap_size), not attrs
    s_res.df.attrs["kl_zones"] = kl_zones
    s_res.df.attrs["poi_zones"] = poi_zones
    s_res.df.attrs["structure_events"] = s_res.events
    s_res.df.attrs["fib_states"] = fib_states
    s_res.df.attrs["prev_bos_lines"] = prev_bos_lines

    return PipelineResult(
        df=s_res.df,
        patterns=p_res.events,
        structure=s_res.levels,
        meta=meta,
    )


def _validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_CANDLE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[pipeline] Missing required columns: {missing}")
    if df.empty:
        raise ValueError("[pipeline] Input df is empty")
