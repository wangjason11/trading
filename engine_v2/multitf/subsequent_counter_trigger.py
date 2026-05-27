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

`lifecycle_end_idx` is the next cycle's lifecycle-start —
`CTS_ESTABLISHED.ev.idx` for `(sid, cycle_id+1)` (the CTS extreme,
canonical per PART4 §5; B2 Phase B re-pointed this from the prior next
BOS_CONFIRMED.confirmed_at — equal on H1) — or REVERSAL_CANDIDATE
`apply_idx` for the same sid, whichever fires first. Mirrors the
first_counter / first_confluence / subsequent_confluence convention.
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
    cts_est_idx_by_key: Dict[Tuple[int, int], int] = {}
    reversal_idx_by_sid: Dict[int, int] = {}

    for ev in sorted_events:
        if ev.type == "CTS_CONFIRMED":
            key = (
                int(ev.meta.get("structure_id", 0)),
                int(ev.meta.get("cycle_id", 0)),
            )
            sd_by_key.setdefault(key, int(ev.meta.get("struct_direction", 0)))
        elif ev.type == "CTS_ESTABLISHED":
            key = (
                int(ev.meta.get("structure_id", 0)),
                int(ev.meta.get("cycle_id", 0)),
            )
            # Cycle lifecycle-start = CTS_ESTABLISHED.ev.idx (CTS extreme),
            # canonical per PART4 §5 (B2 Phase B; was next BOS confirmed_at).
            cts_est_idx_by_key[key] = int(ev.idx)
        elif ev.type == "REVERSAL_CANDIDATE":
            sid = int(ev.meta.get("structure_id", 0))
            apply_idx = ev.meta.get("apply_idx")
            if apply_idx is not None:
                reversal_idx_by_sid[sid] = int(apply_idx)

    out: List[SubsequentCounterTrigger] = []

    for (sid, cycle_id), trig_list in zone_proximity_triggers.items():
        parent_sd = sd_by_key.get((sid, cycle_id), 0)
        if parent_sd == 0:
            continue

        next_cycle_start = cts_est_idx_by_key.get((sid, cycle_id + 1))
        rev = reversal_idx_by_sid.get(sid)
        if next_cycle_start is not None and rev is not None:
            lifecycle_end_idx = min(next_cycle_start, rev)
        elif next_cycle_start is not None:
            lifecycle_end_idx = next_cycle_start
        elif rev is not None:
            lifecycle_end_idx = rev
        else:
            lifecycle_end_idx = None

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
                end_idx=this_t.idx,
                trigger_event_idx=this_t.idx,
                lifecycle_end_idx=lifecycle_end_idx,
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
