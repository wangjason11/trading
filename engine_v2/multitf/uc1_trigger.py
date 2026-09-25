"""UC1 trigger detection: 15M reverse structure from 1H CTS.

Scans WVMI records for activated cycles (CTS_CONFIRMED + proximity activation).
For each activated cycle, creates a MultiTFTrigger to start a 15M reverse structure.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from engine_v2.structure import event_fields as ef
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
    - probe input = the CTS anchor (`CTS_CONFIRMED.meta["cts_anchor_idx"]`)

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
        cts_idx = ef.cts_anchor_idx(cts_ev)

        # The CTS anchor seeds the probe input: it must lie inside the parent frame.
        if cts_idx not in h1_df.index:
            continue

        trigger = MultiTFTrigger(
            parent_tf="H1",
            parent_sid=sid,
            parent_cycle_id=cycle_id,
            parent_sd=h1_sd,
            use_case="first_counter",
            lower_tf="M15",
            lower_sd=-1 * h1_sd,  # Opposite direction
            meta={
                # Informational H1 input (the sweep trigger's `probe_input_idx`,
                # exported on unresolved rows); the sibling-CTS probe co-sources
                # its own M15 input and ends at the sweep's `hi`.
                "probe_input_idx": cts_idx,
                # Parent-TF candle where this trigger fires (sd zone-prox).
                "trigger_event_idx": rec.meta.get("triggered_by_event_idx"),
            },
        )
        triggers.append(trigger)
        print(
            f"[uc1_trigger] sid={sid} cycle={cycle_id} h1_sd={h1_sd} "
            f"-> M15 sd={trigger.lower_sd}"
        )

    return triggers
