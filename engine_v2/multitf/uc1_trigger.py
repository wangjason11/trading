"""UC1 trigger detection (`first_counter`): a 15M reverse structure from an H1 cycle's first sd zone-proximity trigger.

A cycle whose FIRST zone-proximity trigger is sd (`zone_proximity.first_sd_proximity_triggers`) and whose CTS is
confirmed gets one MultiTFTrigger, starting a 15M structure in the opposite direction. Until 2026-10-26 the detector
read the main's WVMI records, which existed for exactly those cycles under the main's first-sd WVMI gate; it reads the
trigger itself since ("WVMI back on the chart" step 1), so the H1 WVMI can drop that gate.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from engine_v2.structure import event_fields as ef
from engine_v2.multitf.types import MultiTFTrigger
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.zones.zone_proximity import ZoneProximityTrigger, first_sd_proximity_triggers


def detect_uc1_triggers(
    sorted_events: List[StructureEvent],
    h1_df: pd.DataFrame,
    zone_proximity_triggers: Dict[Tuple[int, int], List[ZoneProximityTrigger]],
) -> List[MultiTFTrigger]:
    """Detect UC1 triggers from the H1 cycles' first sd zone-proximity triggers.

    For each cycle with a first sd trigger and a CTS_CONFIRMED, in CTS_CONFIRMED order:
    - lower_sd = opposite of H1 struct_direction
    - parent input (meta `parent_input_idx`, H1) = the CTS anchor
      (`CTS_CONFIRMED.meta["cts_anchor_idx"]`) — informational: the probe
      co-sources its M15 input from the sibling confluence CTS (PART4 §4.3.3)
    - meta `trigger_event_idx` = the first sd trigger's candle (H1)

    The main's batch lists and the sweep's final lists give the same triggers: both keep each cycle's first trigger
    (`orchestrator._final_proximity_triggers` asserts it).

    Returns list of MultiTFTrigger.
    """
    first_sd = first_sd_proximity_triggers(zone_proximity_triggers)

    # Build lookup: (sid, cycle_id) -> CTS_CONFIRMED event (iterated in first-seen order)
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

    for key, cts_ev in cts_conf_by_key.items():
        trig = first_sd.get(key)
        if trig is None:
            continue
        sid, cycle_id = key

        h1_sd = int(cts_ev.meta.get("struct_direction", sd_by_key.get(key, 0)))
        if h1_sd == 0:
            continue

        # CTS level index (the anchor candle, not the confirmation candle)
        # cts_ev.idx = confirmation candle (the moment); cts_anchor_idx = the CTS anchor
        cts_anchor_idx = ef.cts_anchor_idx(cts_ev)

        # The CTS anchor is this trigger's H1 input (the sweep's `parent_input_idx`;
        # informational — the sibling-CTS probe co-sources its M15 input); it must lie
        # inside the parent frame.
        if cts_anchor_idx not in h1_df.index:
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
                # Informational H1 input (-> the sweep trigger's `parent_input_idx`,
                # exported on unresolved rows); the sibling-CTS probe co-sources
                # its own M15 input and ends at the sweep's `hi`.
                "parent_input_idx": cts_anchor_idx,
                # Parent-TF candle where this trigger fires (the cycle's first sd zone-prox).
                "trigger_event_idx": trig.idx,
            },
        )
        triggers.append(trigger)
        print(
            f"[uc1_trigger] sid={sid} cycle={cycle_id} h1_sd={h1_sd} "
            f"-> M15 sd={trigger.lower_sd}"
        )

    return triggers
