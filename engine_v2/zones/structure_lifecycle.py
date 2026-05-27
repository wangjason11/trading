# engine_v2/zones/structure_lifecycle.py
"""Shared structure/cycle lifecycle-start resolution (pure leaf).

Single home for the per-`structure_id` lifecycle-start used by BOTH zone
derivations (`kl_zones_v1`, `poi_zones`). Previously each recomputed this block
verbatim; this is the start-side unification of the pass-through lifecycle model
(`PART4_REFACTOR_SPEC.md §5`, plan "B1", 2026-05-27).

Pure leaf: imports nothing from `zones/`, `charting/`, or `structure/` — it only
reads duck-typed structure events (`ev.meta`, `ev.idx`). Both callers depend on
it without a cycle.

Scope is START-ONLY. The end resolution (reversal / next-cycle CTS-established)
and the reversal-dict construction stay in each caller for now; unifying them
(+ deduping the reversal dict, currently
`kl_zones_v1._get_reversal_confirmed_by_sid_from_events` vs POI inline
`reversal_idx_by_sid` — proven byte-identical) is the deferred "B2" step, to
follow end-condition verification. See
`memory/project_cycle_lifecycle_parent_cycle_floor.md`.
"""
from __future__ import annotations

from typing import Dict, List, Optional


def compute_struct_start_by_sid(
    events: List,
    reversal_idx_by_sid: Dict[int, int],
    lifecycle_floor: Optional[int] = None,
) -> Dict[int, int]:
    """Per-`structure_id` lifecycle-start idx (the first idx a structure is active).

    Rule (identical for main and subordinate):
      1. base = the structure's first structural anchor = min event idx for that sid.
      2. reversal handoff = sid N's lifecycle-start is the reversal-confirmation idx
         of sid N-1 (`reversal_idx_by_sid[N-1]`), overriding the min-idx base.
      3. floor = raise every sid to `lifecycle_floor` when given.

    `reversal_idx_by_sid` is passed in (each caller supplies its own, kept for the
    untouched end resolution) — `{sid: reversal_confirmed_idx}` from
    `STATE_CHANGED where to=="reversal"`, MAX `ev.idx` per sid.

    `lifecycle_floor` semantics:
      - main pipeline → `None` (no extra floor; cycles still floor against this
        structure-start via the caller's per-zone `max(confirmed_idx, struct_start)`).
      - subordinate → `max(start_trigger_idx, parent_sid_start, parent_cycle_start)`
        (slice-local), computed by `multitf/entity_df_mutation.build_one_sid`. The
        zones layer stays parent-agnostic — it only sees one int. See
        `PART4_REFACTOR_SPEC.md §5` (the parent floors and their H1->M15 mapping).
    """
    struct_start_by_sid: Dict[int, int] = {}
    for ev in events:
        s = (ev.meta or {}).get("structure_id")
        if s is None:
            continue
        s = int(s)
        i = int(ev.idx)
        if s not in struct_start_by_sid or i < struct_start_by_sid[s]:
            struct_start_by_sid[s] = i
    # Reversal handoff: sid N's lifecycle-start = reversal idx of sid N-1.
    for s in list(struct_start_by_sid):
        if (s - 1) in reversal_idx_by_sid:
            struct_start_by_sid[s] = int(reversal_idx_by_sid[s - 1])
    # Floor (subordinate: trigger + parent floors; main: None => no-op).
    if lifecycle_floor is not None:
        for s in struct_start_by_sid:
            struct_start_by_sid[s] = max(struct_start_by_sid[s], int(lifecycle_floor))
    return struct_start_by_sid
