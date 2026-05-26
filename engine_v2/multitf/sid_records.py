"""Per-sid record building for entity dfs.

Spec §9.2: each entity df carries `df.attrs["sids"]` — one `SidRecord` per
sid in that entity. For `main`, `sub_sid` == `structure_id` (increments on
reversal). For `subordinate`, `sub_sid` is the per-parent-cycle counter
(resets to 0 each parent cycle); the sub's full identity is the tuple
`(parent_sid, parent_cycle_id, sub_sid)`.
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
        # Main entity: sub_sid == structure_id (no parent), so the identity
        # tuple (None, None, sub_sid) reduces to the structure_id.
        out.append(SidRecord(
            sub_sid=sid,
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
    """Derive per-sid records for a subordinate entity from its sid results.

    Each `LowerTFResult` is one bounded sub sid (§6.1 merge-and-bound). Its
    identity tuple `(parent_sid, parent_cycle_id, sub_sid)` is read from the
    trigger + `result.meta["sub_sid"]`. Results arrive in build order
    (lexicographic by tuple), preserved here.

    Indices are recorded in the entity df's coordinate space — for the
    M15.counter entity that's `m15_df_prepared` indices, which the
    pipeline already stashed on `LowerTFResult.meta` as
    `m15_start_idx` / `m15_end_idx`.
    """
    out: List[SidRecord] = []
    # Results are in build order (lexicographic by identity tuple); preserve
    # that order so the chart renders sids in a stable sequence.
    for result in lower_tf_results:
        trigger = result.trigger
        creation = result.meta.get("m15_start_idx")
        end = result.meta.get("m15_end_idx")
        # Canonical per-parent-cycle identity (§2 / §6.1): the sub is the
        # tuple (parent_sid, parent_cycle_id, sub_sid). `SidRecord.sub_sid`
        # is the per-cycle counter (was `meta["sid"]` on the result).
        # end_reason is the sid's resolved end cause.
        end_reason = result.meta.get(
            "end_reason", "lifecycle_end" if end is not None else None,
        )

        out.append(SidRecord(
            sub_sid=int(result.meta["sub_sid"]),
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
