"""Per-sid record building for entity dfs.

Spec §9.2 / §17.9: each entity df carries `df.attrs["sids"]` — one `SidRecord`
per sid in that entity. For `main`, `sub_sid` == `structure_id` (increments on
reversal). For a subordinate lens df, one `SidRecord` per UNIQUE SUB rendered
on that lens (`sub_id` set, `sub_sid = None`); the record table
(`attrs["triggers"]`) carries the per-parent-cycle attribution.
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
    """Derive per-sub records for a subordinate lens df from the per-sub
    projections (§17.9; `entity_df_mutation.render_sub_projection`).

    One `SidRecord` per unique sub on this lens, in the order given (the
    orchestrator passes them in `start_idx` order). Read off `result.meta`:
    `sub_id`, `m15_start_idx` (= `starting_idx`, the structural anchor →
    `creation_event_idx`), `start_idx` / `end_idx` (the real-time lifecycle
    window → `start_idx` / `end_event_idx`; `end_idx` None while open),
    `end_reason` (None while open — no fallback), `lenses`,
    `relative_dir_segments`, plus `natural_reversal_idx`, `n_records`,
    `first_record`, `slice_begin`, `validated_h1_start` into `meta`.
    `sub_sid` and the parent fields are None: parent attribution lives on the
    TriggerRecord table.
    """
    out: List[SidRecord] = []
    for result in lower_tf_results:
        m = result.meta
        trigger = result.trigger
        creation = m.get("m15_start_idx")
        start = m.get("start_idx")
        end = m.get("end_idx")
        lenses = m.get("lenses") or ()
        segs = m.get("relative_dir_segments") or ()
        out.append(SidRecord(
            sub_sid=None,
            starting_sd=int(trigger.lower_sd),
            creation_event_idx=int(creation) if creation is not None else None,
            end_event_idx=int(end) if end is not None else None,
            end_reason=m.get("end_reason"),
            parent_sid=None,
            parent_cycle_id=None,
            meta={
                "natural_reversal_idx": m.get("natural_reversal_idx"),
                "n_records": m.get("n_records"),
                "first_record": m.get("first_record"),
                "slice_begin": m.get("slice_begin"),
                "validated_parent_start": m.get("validated_h1_start"),
            },
            sub_id=int(m["sub_id"]) if m.get("sub_id") is not None else None,
            start_idx=int(start) if start is not None else None,
            lenses=tuple(lenses),
            relative_dir_segments=tuple((int(a), str(b)) for a, b in segs),
        ))
    return out
