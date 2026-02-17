"""UC1 trigger detection: 15M reverse structure from 1H CTS.

Scans WVMI records for activated cycles (CTS_CONFIRMED + proximity activation).
For each activated cycle, creates a MultiTFTrigger to start a 15M reverse structure.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from engine_v2.common.types import KLZone, WVMIRecord
from engine_v2.multitf.types import MultiTFTrigger
from engine_v2.structure.market_structure import StructureEvent


def detect_uc1_triggers(
    sorted_events: List[StructureEvent],
    h1_df: pd.DataFrame,
    wvmi_records: List[WVMIRecord],
    kl_zones: List[KLZone],
) -> List[MultiTFTrigger]:
    """Detect UC1 triggers from H1 WVMI records.

    For each WVMI record (implying CTS_CONFIRMED + proximity activation):
    - Look up the CTS_CONFIRMED event for (sid, cycle_id)
    - lower_sd = opposite of H1 struct_direction
    - start_time = time of H1 CTS candle
    - start_price = CTS extreme (high for sd=+1, low for sd=-1)

    Returns list of MultiTFTrigger.
    """
    # Build lookup: (sid, cycle_id) -> CTS_CONFIRMED event
    cts_conf_by_key: Dict[tuple, StructureEvent] = {}
    for ev in sorted_events:
        if ev.type == "CTS_CONFIRMED":
            key = (ev.meta.get("structure_id", 0), ev.meta.get("cycle_id", 0))
            cts_conf_by_key[key] = ev

    # Build lookup: (sid, cycle_id) -> struct_direction from BOS_CONFIRMED
    sd_by_key: Dict[tuple, int] = {}
    for ev in sorted_events:
        if ev.type == "BOS_CONFIRMED":
            key = (ev.meta.get("structure_id", 0), ev.meta.get("cycle_id", 0))
            sd_by_key[key] = int(ev.meta.get("struct_direction", 0))

    # Build lifecycle end lookup: for each (sid, cycle_id),
    # find the next BOS_CONFIRMED or REVERSAL_CANDIDATE that ends this cycle
    bos_conf_idx_by_key: Dict[tuple, int] = {}
    for ev in sorted_events:
        if ev.type == "BOS_CONFIRMED":
            key = (ev.meta.get("structure_id", 0), ev.meta.get("cycle_id", 0))
            # Use confirmed_at (the confirmation candle), not ev.idx (the extreme)
            bos_conf_idx_by_key[key] = int(ev.meta.get("confirmed_at", ev.idx))

    reversal_idx_by_sid: Dict[int, int] = {}
    for ev in sorted_events:
        if ev.type == "REVERSAL_CANDIDATE":
            sid = ev.meta.get("structure_id", 0)
            apply_idx = ev.meta.get("apply_idx")
            if apply_idx is not None:
                reversal_idx_by_sid[sid] = apply_idx

    triggers: List[MultiTFTrigger] = []

    for rec in wvmi_records:
        sid = rec.bos_structure_id
        cycle_id = rec.bos_cycle_id
        key = (sid, cycle_id)

        cts_ev = cts_conf_by_key.get(key)
        if cts_ev is None:
            continue

        h1_sd = int(cts_ev.meta.get("struct_direction", sd_by_key.get(key, 0)))
        if h1_sd == 0:
            continue

        # CTS level index (the extreme candle, not the confirmation candle)
        # cts_ev.idx = confirmation candle; cts_anchor_idx = actual CTS extreme
        cts_idx = int(cts_ev.meta.get("cts_anchor_idx", cts_ev.idx))

        # Get CTS extreme price
        if cts_idx not in h1_df.index:
            continue
        if h1_sd == 1:
            start_price = float(h1_df.loc[cts_idx, "h"])
        else:
            start_price = float(h1_df.loc[cts_idx, "l"])

        start_time = pd.to_datetime(h1_df.loc[cts_idx, "time"], utc=True)

        # Lifecycle end: next BOS or reversal
        lifecycle_end_idx = None
        next_bos_idx = bos_conf_idx_by_key.get((sid, cycle_id + 1))
        rev_idx = reversal_idx_by_sid.get(sid)
        if next_bos_idx is not None and rev_idx is not None:
            lifecycle_end_idx = min(next_bos_idx, rev_idx)
        elif next_bos_idx is not None:
            lifecycle_end_idx = next_bos_idx
        elif rev_idx is not None:
            lifecycle_end_idx = rev_idx

        trigger = MultiTFTrigger(
            parent_tf="H1",
            parent_sid=sid,
            parent_cycle_id=cycle_id,
            parent_sd=h1_sd,
            use_case="uc1_reverse",
            lower_tf="M15",
            lower_sd=-1 * h1_sd,  # Opposite direction
            start_time=start_time,
            start_price=start_price,
            lifecycle_end_idx=lifecycle_end_idx,
            meta={
                "cts_idx": cts_idx,
                "activation_idx": rec.meta.get("activation_idx"),
            },
        )
        triggers.append(trigger)
        print(
            f"[uc1_trigger] sid={sid} cycle={cycle_id} h1_sd={h1_sd} "
            f"-> M15 sd={trigger.lower_sd} start_time={start_time} "
            f"lifecycle_end_idx={lifecycle_end_idx}"
        )

    return triggers
