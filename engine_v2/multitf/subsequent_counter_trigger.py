"""`subsequent_counter` (var 4) trigger detection.

Per spec §4.3.5:
  Trigger    parent sd-zone proximity AND prior trigger was CTS-zone
             proximity AND the trigger before that was sd-zone (forms Λ
             in bullish parent / V in bearish parent)
  Idx input  parent-TF candle in [prior_sd_idx, this_sd_prox_idx] with
             extreme toward parent CTS (highest high for bullish, lowest
             low for bearish — Λ apex / V trough)
  End idx    the sd-prox trigger candle
  Probe sd   -parent_sd (counter — opposite direction to parent)

Walks the alternating `zone_proximity_triggers[(sid, cycle_id)]` list and
emits one trigger per sd entry at index ≥ 2. By the alternation invariant
in `zones/zone_proximity.py`, every such sd trigger is preceded by an
opp_sd (CTS-prox, var 3) and that by an sd (var 2 or an earlier var 4) —
the [sd, opp_sd, sd] window the spec requires.

The parent-cycle end every record uses lives in `multitf/parent_tables.py`
(the retired `lifecycle_end_idx` was deleted in Plan E E1b).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from engine_v2.multitf.types import SubsequentCounterTrigger
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.zones.zone_proximity import ZoneProximityTrigger


def _input_idx_window_extreme_toward_cts(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    parent_sd: int,
) -> int:
    """Pick parent-TF candle in [start_idx, end_idx] with extreme toward CTS.

    Bullish parent (sd=+1): CTS is above → highest high (Λ apex);
    tie-break: earliest.
    Bearish parent (sd=-1): CTS is below → lowest low (V trough);
    tie-break: earliest.
    """
    if start_idx > end_idx:
        return start_idx
    seg = df.loc[start_idx:end_idx]
    if seg.empty:
        return start_idx
    if parent_sd == 1:
        target = seg["h"].astype(float).max()
        matches = seg[seg["h"].astype(float) == target]
    else:
        target = seg["l"].astype(float).min()
        matches = seg[seg["l"].astype(float) == target]
    return int(matches.index[0])


def detect_subsequent_counter_triggers(
    sorted_events: List[StructureEvent],
    zone_proximity_triggers: Dict[Tuple[int, int], List[ZoneProximityTrigger]],
    parent_df: pd.DataFrame,
    parent_tf: str = "H1",
) -> List[SubsequentCounterTrigger]:
    """Detect var 4 triggers from parent zone-proximity-trigger output.

    For each parent cycle, scan its alternating trigger list. Every sd
    trigger at index ≥ 2 emits one var 4 trigger; its [-1] predecessor is
    necessarily opp_sd and its [-2] predecessor is necessarily sd by the
    alternation invariant.
    """
    sd_by_key: Dict[Tuple[int, int], int] = {}

    for ev in sorted_events:
        if ev.type == "CTS_CONFIRMED":
            key = (
                int(ev.meta.get("structure_id", 0)),
                int(ev.meta.get("cycle_id", 0)),
            )
            sd_by_key.setdefault(key, int(ev.meta.get("struct_direction", 0)))

    out: List[SubsequentCounterTrigger] = []

    for (sid, cycle_id), trig_list in zone_proximity_triggers.items():
        parent_sd = sd_by_key.get((sid, cycle_id), 0)
        if parent_sd == 0:
            continue

        for i in range(2, len(trig_list)):
            this_t = trig_list[i]
            if this_t.direction != "sd":
                continue
            cts_prox_t = trig_list[i - 1]
            prior_sd_t = trig_list[i - 2]
            # Alternation invariant: [-1] is opp_sd, [-2] is sd.
            if cts_prox_t.direction != "opp_sd" or prior_sd_t.direction != "sd":
                continue

            input_idx = _input_idx_window_extreme_toward_cts(
                parent_df, prior_sd_t.idx, this_t.idx, parent_sd,
            )

            out.append(SubsequentCounterTrigger(
                parent_tf=parent_tf,
                parent_sid=sid,
                parent_cycle_id=cycle_id,
                parent_sd=parent_sd,
                input_idx=input_idx,
                trigger_event_idx=this_t.idx,
                meta={
                    "prior_sd_trigger_idx": prior_sd_t.idx,
                    "prior_sd_zone_kind": prior_sd_t.zone_kind,
                    "prior_cts_prox_idx": cts_prox_t.idx,
                    "this_sd_zone_kind": this_t.zone_kind,
                    "trigger_inner": this_t.trigger_inner,
                    "sequence_index_in_cycle": i,
                },
            ))

    out.sort(key=lambda t: t.trigger_event_idx)
    return out
