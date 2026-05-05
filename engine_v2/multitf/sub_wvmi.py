"""Parent-driven sub WVMI computation (Part 4 §8.3 / §8.4).

For sub entities, WVMI is gated by parent events instead of entity-local
zone proximity triggers. This module provides
`compute_parent_driven_sub_wvmi` which runs the standard WVMI lifecycle
(CTS_CONFIRMED -> create record, BOS_CONFIRMED -> lock, update_temporary_lp)
on a sub `LowerTFResult`, with all records carrying §8.7 attribution:
`structure_path_id`, `triggered_by_event_idx`, `triggered_by_event_type`.

Cadence per §8.5 (var 4 not built yet):

  Main first sd-prox after CTS  -> confluence sub WVMI on var 1 sid
  Var 3 (parent CTS-prox-after-sd-prox) -> counter sub WVMI on first_counter sid

Var 3 confluence sids and var 4 counter sids get NO sub WVMI in 3d.iii;
those are wired up alongside §13.4 (subsequent_counter / var 4).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from engine_v2.common.types import WVMIRecord
from engine_v2.multitf.types import LowerTFResult
from engine_v2.zones.wvmi import WVMITracker


@dataclass(frozen=True)
class ParentTrigger:
    """Parent event that activated a sub WVMI sweep.

    `idx` is the parent-TF candle index of the trigger; `event_type` is
    the spec-§8.7 string that goes onto every record's meta; `parent_path_id`
    identifies the parent entity (e.g. "H1.main"). The sub entity's own
    `structure_path_id` is passed separately to
    `compute_parent_driven_sub_wvmi`.
    """
    idx: int
    event_type: str
    parent_path_id: str


def compute_parent_driven_sub_wvmi(
    result: LowerTFResult,
    sub_path_id: str,
    parent_trigger: ParentTrigger,
) -> List[WVMIRecord]:
    """Build WVMI records for one sub LowerTFResult.

    Mirrors the main-entity WVMI lifecycle but skips the entity-local
    zone-proximity gate. Every CTS_CONFIRMED in the sub's events list
    becomes a WVMI record (subject to wave-candle / volume guards in
    `WVMITracker.on_cts_confirmed`); every BOS_CONFIRMED locks the
    previous cycle's record.

    All records carry §8.7 attribution merged into `record.meta`:
      - `triggered_by_event_idx`     parent trigger candle idx
      - `triggered_by_event_type`    e.g. "ZONE_PROXIMITY_TRIGGER",
                                     "SUBSEQUENT_CONFLUENCE_TRIGGER"
      - `parent_path_id`             e.g. "H1.main"

    The record's `structure_path_id` (sub entity) is set by the tracker
    constructor, and `bos_structure_id` / `bos_cycle_id` come from the
    sub events themselves.
    """
    tracker = WVMITracker(structure_path_id=sub_path_id)

    sorted_events = sorted(result.events, key=lambda e: (e.idx, e.type))

    for ev in sorted_events:
        if ev.type == "CTS_CONFIRMED":
            rec = tracker.on_cts_confirmed(
                ev, result.df, result.wave_candles, result.kl_zones,
            )
            if rec is not None:
                rec.meta.update({
                    "triggered_by_event_idx": parent_trigger.idx,
                    "triggered_by_event_type": parent_trigger.event_type,
                    "parent_path_id": parent_trigger.parent_path_id,
                })

    for ev in sorted_events:
        if ev.type == "BOS_CONFIRMED":
            tracker.on_bos_confirmed(ev, result.df, result.wave_candles)

    tracker.update_temporary_lp(result.df, result.kl_zones)

    return tracker.get_records()
