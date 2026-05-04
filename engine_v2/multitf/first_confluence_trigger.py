"""`first_confluence` (var 1) trigger detection.

Per spec §4.3.2:
  Trigger: parent BOS_CONFIRMED
  Idx input: parent BOS extreme idx (== BOS_CONFIRMED.ev.idx)
  Probe end_idx: parent CTS_CONFIRMED idx in the same parent cycle
                 (None until that CTS_CONFIRMED fires — pending state)
  Probe sd: +parent_sd (confluence)
  Output: starting_idx for first confluence sub (sid=0) of that parent cycle

Cycle 0 of any sid has no preceding BOS_CONFIRMED (the structure begins
at the start CTS), so var 1 doesn't fire for cycle 0 — matching spec
§4.3.6's bootstrap rule.

3a is detection only — no consumer yet. Steps 3b–3d will build the
confluence sub from these triggers.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from engine_v2.multitf.types import FirstConfluenceTrigger
from engine_v2.structure.market_structure import StructureEvent


def detect_first_confluence_triggers(
    sorted_events: List[StructureEvent],
    parent_tf: str = "H1",
) -> List[FirstConfluenceTrigger]:
    """Walk sorted parent events and emit one trigger per BOS_CONFIRMED.

    Pairs each BOS_CONFIRMED with the matching CTS_CONFIRMED for the same
    (sid, cycle_id). If no CTS_CONFIRMED exists yet (parent cycle still
    open at end-of-data), the trigger is emitted with `end_idx=None` and
    `status="pending"` per spec §14.

    Returns triggers sorted by `trigger_event_idx`.
    """
    cts_conf_by_key: Dict[Tuple[int, int], StructureEvent] = {}
    for ev in sorted_events:
        if ev.type == "CTS_CONFIRMED":
            key = (
                int(ev.meta.get("structure_id", 0)),
                int(ev.meta.get("cycle_id", 0)),
            )
            cts_conf_by_key.setdefault(key, ev)

    triggers: List[FirstConfluenceTrigger] = []

    for ev in sorted_events:
        if ev.type != "BOS_CONFIRMED":
            continue

        sid = int(ev.meta.get("structure_id", 0))
        cycle_id = int(ev.meta.get("cycle_id", 0))
        parent_sd = int(ev.meta.get("struct_direction", 0))
        if parent_sd == 0:
            continue

        # BOS_CONFIRMED.ev.idx = BOS extreme; meta["confirmed_at"] = trigger candle
        input_idx = int(ev.idx)
        trigger_event_idx = int(ev.meta.get("confirmed_at", ev.idx))

        cts_conf = cts_conf_by_key.get((sid, cycle_id))
        if cts_conf is not None:
            end_idx = int(cts_conf.idx)
            status = "finalized"
        else:
            end_idx = None
            status = "pending"

        triggers.append(FirstConfluenceTrigger(
            parent_tf=parent_tf,
            parent_sid=sid,
            parent_cycle_id=cycle_id,
            parent_sd=parent_sd,
            input_idx=input_idx,
            end_idx=end_idx,
            trigger_event_idx=trigger_event_idx,
            status=status,
            meta={
                "bos_price": ev.price,
                "cts_confirmation_method": (
                    cts_conf.meta.get("confirmation_method") if cts_conf else None
                ),
            },
        ))

    triggers.sort(key=lambda t: t.trigger_event_idx)
    return triggers
