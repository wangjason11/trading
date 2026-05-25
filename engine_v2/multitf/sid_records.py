"""Per-sid record building for entity dfs.

Spec §9.2: each entity df carries `df.attrs["sids"]` — one `SidRecord` per
sid in that entity. For `main`, sids increment on reversal. For
`subordinate`, today each lower-TF trigger result is its own sid (sid 0
within that sub instance); §6.2 will eventually merge them under one
entity df with parent_cycle_id meta.

3a wires the records into df.attrs but doesn't consume them yet — Steps
3b–3d will read them when building confluence subs and routing the
event bus.
"""
from __future__ import annotations

from typing import List

from engine_v2.multitf.types import LowerTFResult, SidRecord
from engine_v2.structure.market_structure import StructureEvent


def build_sid_records_for_main(events: List[StructureEvent]) -> List[SidRecord]:
    """Derive per-sid records for a main entity from its event stream.

    For each sid present in events:
      - creation_event_idx = min ev.idx among events for that sid
      - end_event_idx = REVERSAL_CANDIDATE.apply_idx for that sid (None if absent)
      - end_reason = "reversal" if reversal exists, else None
      - starting_sd = struct_direction from any event of that sid
    """
    sids: dict[int, dict] = {}

    for ev in events:
        sid = ev.meta.get("structure_id")
        if sid is None:
            continue
        sid = int(sid)

        rec = sids.setdefault(sid, {
            "creation_event_idx": ev.idx,
            "end_event_idx": None,
            "end_reason": None,
            "starting_sd": int(ev.meta.get("struct_direction", 0)),
        })
        if ev.idx < rec["creation_event_idx"]:
            rec["creation_event_idx"] = ev.idx
        if rec["starting_sd"] == 0 and ev.meta.get("struct_direction") is not None:
            rec["starting_sd"] = int(ev.meta["struct_direction"])

        if ev.type == "REVERSAL_CANDIDATE":
            apply_idx = ev.meta.get("apply_idx")
            if apply_idx is not None:
                rec["end_event_idx"] = int(apply_idx)
                rec["end_reason"] = "reversal"

    out: List[SidRecord] = []
    for sid in sorted(sids.keys()):
        rec = sids[sid]
        out.append(SidRecord(
            sid=sid,
            starting_sd=rec["starting_sd"],
            creation_event_idx=rec["creation_event_idx"],
            end_event_idx=rec["end_event_idx"],
            end_reason=rec["end_reason"],
            parent_sid=None,
            parent_cycle_id=None,
        ))
    return out


def build_sid_records_for_subordinate(
    lower_tf_results: List[LowerTFResult],
) -> List[SidRecord]:
    """Derive per-sid records for a subordinate entity from its trigger results.

    Each `LowerTFResult` corresponds to one parent cycle's run of the
    subordinate. We assign the entity-level sid by enumeration order
    (§6.2: one entity df, sids accumulate across parent cycles).

    Indices are recorded in the entity df's coordinate space — for the
    M15.counter entity that's `m15_df_prepared` indices, which the
    pipeline already stashed on `LowerTFResult.meta` as
    `m15_start_idx` / `m15_end_idx`.
    """
    out: List[SidRecord] = []
    for entity_sid, result in enumerate(lower_tf_results):
        trigger = result.trigger
        creation = result.meta.get("m15_start_idx")
        end = result.meta.get("m15_end_idx")
        # Phase 2 (§2 / §6.1, 2026-05-25): canonical per-parent-cycle identity
        # lives on result.meta. `SidRecord.sid` stays the entity-wide
        # enumeration rank (== entity_sid, the chart's display key); the
        # per-cycle `sid` + `started_by` + `start_trigger_idx` are recorded in
        # meta. end_reason is now the sid's resolved end cause.
        end_reason = result.meta.get(
            "end_reason", "lifecycle_end" if end is not None else None,
        )

        out.append(SidRecord(
            sid=entity_sid,
            starting_sd=int(trigger.lower_sd),
            creation_event_idx=int(creation) if creation is not None else None,
            end_event_idx=int(end) if end is not None else None,
            end_reason=end_reason,
            parent_sid=int(trigger.parent_sid),
            parent_cycle_id=int(trigger.parent_cycle_id),
            meta={
                "use_case": trigger.use_case,
                "validated_parent_start": result.meta.get("validated_h1_start"),
                "slice_begin": result.meta.get("slice_begin"),
                # Canonical identity: per-parent-cycle sid + what spawned it.
                "sid_in_cycle": result.meta.get("sid"),
                "started_by": result.meta.get("started_by"),
                "start_trigger_idx": result.meta.get("start_trigger_idx"),
                # Parent-TF candle idx where the trigger event fired
                # (BOS_CONFIRMED.confirmed_at for var1, sd-prox candle for
                # var3/var4, etc). Stored on the trigger object.
                "trigger_event_idx": getattr(
                    trigger, "trigger_event_idx",
                    trigger.meta.get("trigger_event_idx"),
                ),
            },
        ))
    return out
