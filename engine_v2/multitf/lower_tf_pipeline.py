"""Lower-TF helpers used by the entity-direct compute path.

§13.5.c.ii deleted the slice-based `run_lower_tf_pipeline`; what
remains here is the parent-TF subordinate probe and the parent →
lower-TF lifecycle-end translator. Both are pure helpers consumed by
`multitf/entity_df_mutation.apply_trigger_to_entity_df`.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Optional

import pandas as pd

from engine_v2.multitf.types import MultiTFTrigger
from engine_v2.structure.structure_engine import compute_structure_scenario_3


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


def _run_subordinate_probe(
    trigger: MultiTFTrigger,
    parent_df: pd.DataFrame,
) -> Optional[int]:
    """Run the parent-TF Scenario 3 probe to find a validated start.

    Generic across `subordinate` variations — params come from
    `trigger.meta["probe_input_idx"]` / `["probe_end_idx"]`. Direction is
    `trigger.lower_sd`. Tolerance comes from `DEFAULT_PROBE_RESET_PIPS`
    by `trigger.parent_tf` (H1=10, M15=5, M5=3).

    Returns parent-TF index of validated start, or None on failure.
    """
    input_idx = trigger.meta.get("probe_input_idx")
    end_idx = trigger.meta.get("probe_end_idx")

    if input_idx is None or end_idx is None:
        print(f"[lower_tf] WARNING: Missing probe_input_idx/probe_end_idx in trigger meta "
              f"for sid={trigger.parent_sid} cycle={trigger.parent_cycle_id}")
        return None

    input_idx = int(input_idx)
    end_idx = int(end_idx)

    try:
        s3_result = compute_structure_scenario_3(
            parent_df,
            start_idx=input_idx,
            struct_direction=trigger.lower_sd,
            end_idx=end_idx,
            run_continuation=False,
            timeframe=trigger.parent_tf,
        )
    except (ValueError, IndexError) as exc:
        print(f"[lower_tf] WARNING: subordinate probe failed for "
              f"{trigger.use_case} sid={trigger.parent_sid} cycle={trigger.parent_cycle_id}: {exc}")
        return None

    print(f"[lower_tf] subordinate probe ({trigger.use_case} on {trigger.parent_tf}): "
          f"sid={trigger.parent_sid} cycle={trigger.parent_cycle_id} "
          f"-> start_idx={s3_result.start_idx} status={s3_result.status} "
          f"iterations={s3_result.probe_iterations}")

    # Pending status means the probe could not reach a terminal condition with
    # the available data. In live use, more candles may arrive that resolve the
    # probe — but for now we skip M15 for this trigger. (In current backtest
    # probe_end_idx — the first sd zone-proximity trigger candle — is always
    # defined, so this path is dormant.)
    if s3_result.status == "pending":
        print(f"[lower_tf] PENDING: subordinate probe did not finalize for "
              f"sid={trigger.parent_sid} cycle={trigger.parent_cycle_id}; "
              f"skipping lower-TF build")
        return None

    return s3_result.start_idx
