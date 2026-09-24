"""`first_confluence` (var 1) trigger detection.

Per spec §4.3.2:
  Trigger: parent BOS_CONFIRMED
  Idx input: parent BOS extreme idx (== BOS_CONFIRMED.ev.idx)
  Probe probe_end_idx: the confirmed CTS's EXTREME idx (`cts_anchor_idx`) in
                 the same parent cycle — NOT the confirmation candle. None until
                 that CTS_CONFIRMED fires (pending state): the confirmation
                 candle gates *knowing* the value; the CTS extreme IS the value.
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
    (sid, cycle_id). `end_idx` is set to that CTS's EXTREME idx
    (`meta["cts_anchor_idx"]`), which is earlier than the confirmation
    candle (`CTS_CONFIRMED.idx == confirmed_at`). If no CTS_CONFIRMED exists
    yet (parent cycle still open at end-of-data), the trigger is emitted with
    `probe_end_idx=None` and `status="pending"` per spec §14.

    The parent-cycle end every record uses is `multitf/parent_tables.py` (the
    moment-based clamped next-cycle start, else `STATE_CHANGED→reversal`); the
    retired `lifecycle_end_idx` was deleted in Plan E E1b.

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

        input_idx = int(ev.idx)
        trigger_event_idx = int(ev.meta.get("confirmed_at", ev.idx))

        cts_conf = cts_conf_by_key.get((sid, cycle_id))
        if cts_conf is not None:
            # Probe end_idx = the confirmed CTS's EXTREME idx, not the
            # confirmation candle. CTS_CONFIRMED.idx is the confirmation candle
            # (== confirmed_at, the later pullback / sd-prox candle); the CTS
            # extreme is meta["cts_anchor_idx"] (earlier). We still WAIT for
            # CTS_CONFIRMED to fire before the value is known (status flips to
            # finalized here), but the value bounding the probe is the CTS
            # extreme. (spec §4.3.2)
            end_idx = int(cts_conf.meta["cts_anchor_idx"])
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
            probe_end_idx=end_idx,
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
