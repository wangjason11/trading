"""Lower-TF pipeline runner for multi-TF analysis.

Runs Scenario 3 structure + downstream pipeline on a slice of lower-TF data,
then injects attribution metadata into all results.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from typing import Optional

import pandas as pd

from engine_v2.multitf.types import MultiTFTrigger, LowerTFResult
from engine_v2.multitf.data_bridge import map_candle_to_lower_tf
from engine_v2.structure.structure_engine import compute_structure_scenario_3
from engine_v2.pipeline.orchestrator import _run_downstream_pipeline

# Scenario 3 BOS_0 probe pip tolerance by timeframe.
# Lower TFs need tighter tolerance because price movements are smaller.
_PIP_TOLERANCE_BY_TF = {
    "H1": 10,
    "M15": 3,
    "M5": 1,
}


def _find_m15_lifecycle_end(
    trigger: MultiTFTrigger,
    m15_df: pd.DataFrame,
    h1_df: pd.DataFrame,
) -> Optional[int]:
    """Find the M15 index where the lower-TF structure should stop.

    Based on the parent H1 lifecycle_end_idx:
    - Map H1 lifecycle end candle to the last M15 candle in that H1 hour
    - For H1 reversal: end is the 4th M15 candle (H1 candle close)
    - For H1 new BOS: end at last M15 candle in that H1 hour
    - If lifecycle_end_idx is None (H1 cycle still active): run to end of M15 data

    Returns M15 index or None (run to end).
    """
    if trigger.lifecycle_end_idx is None:
        return None

    end_h1_idx = trigger.lifecycle_end_idx
    if end_h1_idx not in h1_df.index:
        return None

    end_h1_time = pd.to_datetime(h1_df.loc[end_h1_idx, "time"], utc=True)
    end_h1_hour_end = end_h1_time + timedelta(hours=1)

    # Find last M15 candle in this H1 hour (the close of the H1 candle)
    m15_times = pd.to_datetime(m15_df["time"], utc=True)
    mask = (m15_times >= end_h1_time) & (m15_times < end_h1_hour_end)
    candidates = m15_df[mask]

    if candidates.empty:
        # No M15 candles in the lifecycle end hour, use time-based cutoff
        # Find last M15 candle before the end time
        before_mask = m15_times < end_h1_time
        if before_mask.any():
            return int(m15_df[before_mask].index[-1])
        return None

    # Use the last M15 candle in the H1 hour (H1 candle close)
    return int(candidates.index[-1])


def run_lower_tf_pipeline(
    trigger: MultiTFTrigger,
    m15_df_prepared: pd.DataFrame,
    h1_df: pd.DataFrame,
) -> Optional[LowerTFResult]:
    """Run the lower-TF pipeline for a single trigger.

    1. Map H1 CTS candle to M15 start index
    2. Determine M15 slice boundaries (start to lifecycle end)
    3. Run Scenario 3 on the M15 slice
    4. Run downstream pipeline (KL zones BOS-only, Fib m15_reverse, POI, WVMI)
    5. Inject attribution metadata

    Returns LowerTFResult or None if mapping fails.
    """
    # 1. Map H1 CTS to M15 start
    m15_start_idx = map_candle_to_lower_tf(
        trigger.start_time,
        trigger.start_price,
        trigger.parent_sd,  # Use parent sd to find extreme (CTS marks high for sd=+1)
        m15_df_prepared,
    )

    if m15_start_idx is None:
        print(f"[lower_tf] WARNING: Could not map H1 CTS to M15 for "
              f"sid={trigger.parent_sid} cycle={trigger.parent_cycle_id}")
        return None

    # 2. Determine M15 slice end
    m15_end_idx = _find_m15_lifecycle_end(trigger, m15_df_prepared, h1_df)
    if m15_end_idx is None:
        m15_end_idx = int(m15_df_prepared.index[-1])

    if m15_start_idx >= m15_end_idx:
        print(f"[lower_tf] WARNING: M15 slice too short: start={m15_start_idx} end={m15_end_idx}")
        return None

    # 3. Create isolated M15 slice with lookback buffer
    # Include at least 50 candles before the start for neighbor-dependent
    # calculations (find_base_threshold ±5, wave candle lookback, etc.)
    lookback = 50
    slice_begin = max(0, m15_start_idx - lookback)
    trigger_df = m15_df_prepared.iloc[slice_begin:m15_end_idx + 1].copy()
    trigger_df = trigger_df.reset_index(drop=True)

    # Start idx in the sliced df is offset by the actual lookback
    start_in_slice = m15_start_idx - slice_begin

    if len(trigger_df) - start_in_slice < 5:
        print(f"[lower_tf] WARNING: M15 slice too small ({len(trigger_df) - start_in_slice} candles after start)")
        return None

    print(f"[lower_tf] M15 slice: {len(trigger_df)} candles, "
          f"sd={trigger.lower_sd}, start_in_slice={start_in_slice}")

    # 4. Run Scenario 3 structure
    pip_tol = _PIP_TOLERANCE_BY_TF.get(trigger.lower_tf, 10)
    try:
        s3_result = compute_structure_scenario_3(
            trigger_df,
            start_idx=start_in_slice,
            struct_direction=trigger.lower_sd,
            pip_tolerance_pips=pip_tol,
        )
    except (ValueError, IndexError) as exc:
        print(f"[lower_tf] WARNING: Scenario 3 failed for "
              f"sid={trigger.parent_sid} cycle={trigger.parent_cycle_id}: {exc}")
        return None

    # 5. Run downstream pipeline with M15-specific settings
    downstream = _run_downstream_pipeline(
        s3_result.df,
        s3_result.events,
        s3_result.struct_direction,
        source_kinds=["BOS"],       # BOS-only KL zones for M15
        fib_mode="m15_reverse",     # Imbalance-gated cross-cycle Fib
        log_prefix=f"M15_sid{trigger.parent_sid}_c{trigger.parent_cycle_id}",
    )

    # 6. Inject attribution into events
    attribution = {
        "timeframe": trigger.lower_tf,
        "use_case": trigger.use_case,
        "parent_tf": trigger.parent_tf,
        "parent_sid": trigger.parent_sid,
        "parent_cycle_id": trigger.parent_cycle_id,
    }

    for ev in s3_result.events:
        ev.meta.update(attribution)

    for zone in downstream["kl_zones"]:
        zone.meta.update(attribution)

    # 7. Cap open-ended zones/POIs to the lifecycle boundary.
    #    The M15 structure ends when the parent H1 cycle ends — any zone
    #    still active at that point should not extend beyond it.
    last_time = pd.to_datetime(s3_result.df["time"].iloc[-1], utc=True)
    last_idx = int(s3_result.df.index[-1])
    capped_zones = []
    for zone in downstream["kl_zones"]:
        if zone.end_time is None:
            zone = replace(
                zone,
                end_time=last_time,
                meta={**zone.meta, "active": False,
                      "deactivated_by": "lifecycle_end"},
            )
        capped_zones.append(zone)

    capped_pois = []
    for poi in downstream["poi_zones"]:
        if poi.end_time is None:
            poi = replace(
                poi,
                end_time=last_time,
                meta={**poi.meta, "active": False,
                      "deactivated_by": "lifecycle_end"},
            )
        capped_pois.append(poi)

    result = LowerTFResult(
        trigger=trigger,
        df=s3_result.df,
        events=s3_result.events,
        kl_zones=capped_zones,
        wave_candles=downstream["wave_candles"],
        fib_states=downstream["fib_states"],
        poi_zones=capped_pois,
        wvmi_records=downstream["wvmi_records"],
        status=s3_result.status,
        meta={
            "m15_start_idx": m15_start_idx,
            "m15_end_idx": m15_end_idx,
            "m15_candle_count": len(trigger_df),
            "structure_status": s3_result.status,
            **attribution,
        },
    )

    print(f"[lower_tf] UC1 result: status={result.status}, "
          f"kl_zones={len(result.kl_zones)}, events={len(result.events)}")

    return result
