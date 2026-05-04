"""`subsequent_confluence` (var 3) trigger detection.

Per spec §4.3.4:
  Trigger    parent CTS-zone proximity AND prior parent proximity was sd
  Idx input  parent-TF candle in [prior_sd_idx, cts_prox_idx] with extreme
             toward parent BOS (lowest low for bullish, highest high for
             bearish)
  End idx    the CTS-prox trigger candle
  Probe sd   +parent_sd (confluence — same direction as parent)

Walks the alternating `zone_proximity_triggers[(sid, cycle_id)]` list and
emits one trigger per opp_sd entry whose immediate predecessor in the
same cycle's list is sd. By the alternation invariant in
`zones/zone_proximity.py`, every opp_sd trigger past index 0 satisfies
this — the predecessor is necessarily sd.

`lifecycle_end_idx` is computed as the next BOS_CONFIRMED for
`(sid, cycle_id+1)` (`meta["confirmed_at"]`) or REVERSAL_CANDIDATE
`apply_idx` for the same sid, whichever fires first. Mirrors the
first_counter / first_confluence convention.

Reference-zone resolution per spec §4.3.4 step 1-5 is descriptive only;
the probe in `compute_structure_scenario_3` derives its own BOS_0 zone
from the probe direction and doesn't take an explicit reference zone
input. We carry the `prior_sd_trigger_idx` on meta for diagnostics.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from engine_v2.multitf.types import SubsequentConfluenceTrigger
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.zones.zone_proximity import ZoneProximityTrigger


def _input_idx_window_extreme(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    parent_sd: int,
) -> int:
    """Pick parent-TF candle in [start_idx, end_idx] with extreme toward BOS.

    Bullish parent (sd=+1): BOS is below → lowest low; tie-break: earliest.
    Bearish parent (sd=-1): BOS is above → highest high; tie-break: earliest.
    """
    if start_idx > end_idx:
        return start_idx
    seg = df.loc[start_idx:end_idx]
    if seg.empty:
        return start_idx
    if parent_sd == 1:
        target = seg["l"].astype(float).min()
        matches = seg[seg["l"].astype(float) == target]
    else:
        target = seg["h"].astype(float).max()
        matches = seg[seg["h"].astype(float) == target]
    return int(matches.index[0])


def detect_subsequent_confluence_triggers(
    sorted_events: List[StructureEvent],
    zone_proximity_triggers: Dict[Tuple[int, int], List[ZoneProximityTrigger]],
    parent_df: pd.DataFrame,
    parent_tf: str = "H1",
) -> List[SubsequentConfluenceTrigger]:
    """Detect var 3 triggers from parent zone-proximity-trigger output.

    For each parent cycle, scan its alternating trigger list. Every opp_sd
    trigger past index 0 emits one var 3 trigger (its predecessor in the
    list is necessarily sd by the alternation invariant).
    """
    sd_by_key: Dict[Tuple[int, int], int] = {}
    bos_conf_idx_by_key: Dict[Tuple[int, int], int] = {}
    reversal_idx_by_sid: Dict[int, int] = {}

    for ev in sorted_events:
        if ev.type == "CTS_CONFIRMED":
            key = (
                int(ev.meta.get("structure_id", 0)),
                int(ev.meta.get("cycle_id", 0)),
            )
            sd_by_key.setdefault(key, int(ev.meta.get("struct_direction", 0)))
        elif ev.type == "BOS_CONFIRMED":
            key = (
                int(ev.meta.get("structure_id", 0)),
                int(ev.meta.get("cycle_id", 0)),
            )
            bos_conf_idx_by_key[key] = int(
                ev.meta.get("confirmed_at", ev.idx)
            )
        elif ev.type == "REVERSAL_CANDIDATE":
            sid = int(ev.meta.get("structure_id", 0))
            apply_idx = ev.meta.get("apply_idx")
            if apply_idx is not None:
                reversal_idx_by_sid[sid] = int(apply_idx)

    out: List[SubsequentConfluenceTrigger] = []

    for (sid, cycle_id), trig_list in zone_proximity_triggers.items():
        parent_sd = sd_by_key.get((sid, cycle_id), 0)
        if parent_sd == 0:
            continue

        next_bos = bos_conf_idx_by_key.get((sid, cycle_id + 1))
        rev = reversal_idx_by_sid.get(sid)
        if next_bos is not None and rev is not None:
            lifecycle_end_idx = min(next_bos, rev)
        elif next_bos is not None:
            lifecycle_end_idx = next_bos
        elif rev is not None:
            lifecycle_end_idx = rev
        else:
            lifecycle_end_idx = None

        for i in range(1, len(trig_list)):
            this_t = trig_list[i]
            if this_t.direction != "opp_sd":
                continue
            prior_t = trig_list[i - 1]
            # Alternation invariant: prior must be sd
            if prior_t.direction != "sd":
                continue

            input_idx = _input_idx_window_extreme(
                parent_df, prior_t.idx, this_t.idx, parent_sd,
            )

            out.append(SubsequentConfluenceTrigger(
                parent_tf=parent_tf,
                parent_sid=sid,
                parent_cycle_id=cycle_id,
                parent_sd=parent_sd,
                input_idx=input_idx,
                end_idx=this_t.idx,
                trigger_event_idx=this_t.idx,
                lifecycle_end_idx=lifecycle_end_idx,
                meta={
                    "prior_sd_trigger_idx": prior_t.idx,
                    "prior_sd_zone_kind": prior_t.zone_kind,
                    "cts_prox_zone_kind": this_t.zone_kind,
                    "trigger_inner": this_t.trigger_inner,
                    "sequence_index_in_cycle": i,
                },
            ))

    out.sort(key=lambda t: t.trigger_event_idx)
    return out
