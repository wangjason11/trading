# engine_v2/zones/structure_lifecycle.py
"""Shared structure/cycle lifecycle-start resolution (pure leaf).

Single home for the per-`structure_id` lifecycle-start used by BOTH zone
derivations (`kl_zones_v1`, `poi_zones`). Previously each recomputed this block
verbatim; this is the start-side unification of the pass-through lifecycle model
(`PART4_REFACTOR_SPEC.md §5`, plan "B1", 2026-05-27).

Pure leaf: imports nothing from `zones/`, `charting/`, or `structure/` — it only
reads duck-typed structure events (`ev.meta`, `ev.idx`). Both callers depend on
it without a cycle.

Start AND end resolution live here (B1 + B2, 2026-05-27): the per-cycle
lifecycle (`compute_cycle_lifecycle`) starts at the CTS-established MOMENT
(`meta["confirmed_at"]`, Plan C 2026-09-19 — not the CTS anchor `ev.idx`) and ends as a
pass-through of the next cycle's clamped start / the reversal / the cap. The
same rule serves the H1 main, every sub cycle, and the sub-structure parent
tables (`multitf/parent_tables.py`). See
`memory/project_cycle_lifecycle_parent_cycle_floor.md`.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from engine_v2.structure import event_fields as ef


def compute_reversal_idx_by_sid(events: List) -> Dict[int, int]:
    """Reversal-confirmation idx per `structure_id` from `STATE_CHANGED` events.

    `{sid: MAX ev.idx}` over `STATE_CHANGED where to=="reversal"`. Single home
    for what `kl_zones_v1._get_reversal_confirmed_by_sid_from_events` and POI's
    inline `reversal_idx_by_sid` each built verbatim (proven byte-identical).
    Events are duck-typed (`ev.type`, `ev.meta`, `ev.idx`); rows missing a
    `structure_id` are skipped (reversal events always carry one).

    Reliable across structures because `market_state` df columns get overwritten
    by later structures while events are append-only.
    """
    rev_by_sid: Dict[int, int] = {}
    for ev in events:
        if getattr(ev, "type", None) != "STATE_CHANGED":
            continue
        meta = ev.meta or {}
        if meta.get("to") != "reversal":
            continue
        sid = meta.get("structure_id")
        if sid is None:
            continue
        sid = int(sid)
        idx = int(ev.idx)
        if sid not in rev_by_sid or idx > rev_by_sid[sid]:
            rev_by_sid[sid] = idx
    return rev_by_sid


def compute_cycle_lifecycle(
    events: List,
    reversal_idx_by_sid: Dict[int, int],
    lifecycle_floor: Optional[int] = None,
    lifecycle_cap: Optional[int] = None,
    cap_reason: str = "lifecycle_end",
) -> Dict[Tuple[int, int], Tuple[int, Optional[int], Optional[str]]]:
    """Per-`(sid, cycle)` lifecycle `(start_idx, end_idx, end_reason)`.

    The end side of the pass-through model (`PART4_REFACTOR_SPEC.md §5`,
    "End resolution as start-passthrough", plan "B2", 2026-05-27). Ends are
    derived from starts — never computed independently:

      - `start` = `max(CTS_ESTABLISHED.meta["confirmed_at"]` (the MOMENT the
        cycle was established — the apply candle; canonical cycle-start idx,
        Plan C 2026-09-19)`, struct_start, floor)`. NOT `CTS_ESTABLISHED.ev.idx`,
        which is the CTS ANCHOR (the pattern's extreme candle) — a historical
        price location exactly like `BOS_CONFIRMED.idx` — and can precede the
        apply candle (bound `meta["pattern_anchor_idx"]` `<= ev.idx <=
        confirmed_at <= meta["pattern_anchor_idx"] + range_max_k`; CTS anchor == moment is
        the common case — ARCHITECTURE "`ev.idx` convention").
        `struct_start` from `compute_struct_start_by_sid` already embeds the
        reversal handoff and, for subs, both parent floors via `lifecycle_floor`.
      - `end` = `min(next-cycle clamped start, reversal_idx_by_sid[sid],
        lifecycle_cap)`. Each term is the lifecycle-START of whatever supersedes
        the cycle (next cycle on the same structure / structure-end via reversal /
        the parent-driven cap). `min` mirrors the start-side `max`: the first end
        condition wins.

    `lifecycle_cap` (mirror of `lifecycle_floor`): `None` for main; for a sub it
    is the slice-local end-cap (= parent-cycle/parent-sid end == `end_m15_abs`,
    also the M15 slice/run bound). `cap_reason` tags an end the cap produced
    (`"reversal"` when the sub reversed at the cap idx, else `"lifecycle_end"`);
    a reversal also present in `reversal_idx_by_sid` wins the tag on ties (the
    reversal term is applied first).

    Tie-break matches the prior per-zone code: reversal set first, next-cycle and
    cap override only when *strictly* earlier. End uses the next cycle's CLAMPED
    start (not its raw established moment); the two differ only for collapsed
    sub cycles. A collapsed cycle (clamped `start >= end`) is left for the caller to
    render inert (`status="inactive"`, empty activation_history).
    """
    struct_start = compute_struct_start_by_sid(events, reversal_idx_by_sid, lifecycle_floor)

    # CTS-established MOMENT per (sid, cycle) — `meta["confirmed_at"]`, the apply
    # candle (== BOS_CONFIRMED.confirmed_at, definitional). Last-seen wins. The
    # CTS anchor (`ev.idx`, the pattern's extreme candle) is a historical price
    # location and never a lifecycle value.
    cts_est_by_key: Dict[Tuple[int, int], int] = {}
    for ev in events:
        if getattr(ev, "type", None) != "CTS_ESTABLISHED":
            continue
        meta = ev.meta or {}
        esid = meta.get("structure_id")
        ecyc = meta.get("cycle_id")
        if esid is None or ecyc is None:
            continue
        moment = meta.get("confirmed_at")
        assert moment is not None, (
            f"CTS_ESTABLISHED ({esid},{ecyc}) @ {ev.idx} lacks meta['confirmed_at'] "
            f"(the established moment — market_structure always stamps it)"
        )
        cts_est_by_key[(int(esid), int(ecyc))] = int(moment)

    # Pass 1: clamped cycle starts.
    start_by_key: Dict[Tuple[int, int], int] = {}
    for (s, c), cts_idx in cts_est_by_key.items():
        ss = struct_start.get(s)
        start_by_key[(s, c)] = max(cts_idx, int(ss)) if ss is not None else cts_idx

    # Pass 2: ends = min(next-cycle clamped start, reversal, cap).
    out: Dict[Tuple[int, int], Tuple[int, Optional[int], Optional[str]]] = {}
    for (s, c), start in start_by_key.items():
        end_idx: Optional[int] = None
        end_reason: Optional[str] = None
        # Structure end via reversal (set first → wins ties).
        if s in reversal_idx_by_sid:
            end_idx = int(reversal_idx_by_sid[s])
            end_reason = "reversal"
        # Next cycle on the same structure (clamped start).
        nk = (s, c + 1)
        if nk in start_by_key:
            ns = start_by_key[nk]
            if end_idx is None or ns < end_idx:
                end_idx = ns
                end_reason = "next_cycle"
        # Parent-driven cap (sub only; None for main).
        if lifecycle_cap is not None:
            cap = int(lifecycle_cap)
            if end_idx is None or cap < end_idx:
                end_idx = cap
                end_reason = cap_reason
        out[(s, c)] = (start, end_idx, end_reason)
    return out


def compute_struct_start_by_sid(
    events: List,
    reversal_idx_by_sid: Dict[int, int],
    lifecycle_floor: Optional[int] = None,
) -> Dict[int, int]:
    """Per-`structure_id` lifecycle-start idx (the first idx a structure is active).

    Rule (identical for main and subordinate):
      1. base = the MOMENT the structure became known = its first
         `CTS_ESTABLISHED` moment (`ef.event_moment`; == its BOS_0 moment) —
         Plan E E3f / Q5 (user decision 2026-09-25; before it the first
         structural ANCHOR, BOS_0's, e.g. H1 sid 0 96 → 115). A sid that emitted
         events but never established (reachable only at the data edge) has no
         such moment: its base is its first stamped idx (`ef.stamped_idx`), which
         the reversal handoff then overrides.
      2. reversal handoff = sid N's lifecycle-start is the reversal-confirmation idx
         of sid N-1 (`reversal_idx_by_sid[N-1]`), overriding the min-idx base.
      3. floor = raise every sid to `lifecycle_floor` when given.

    `reversal_idx_by_sid` is passed in (each caller supplies its own, kept for the
    untouched end resolution) — `{sid: reversal_confirmed_idx}` from
    `STATE_CHANGED where to=="reversal"`, MAX `ev.idx` per sid.

    `lifecycle_floor` semantics:
      - main pipeline → `None` (no extra floor; cycles still floor against this
        structure-start via the caller's per-zone `max(confirmed_idx, struct_start)`).
      - subordinate → the unique sub's real-time `start_idx` (slice-local) —
        already `max(probe_finalize_idx, trigger_idx, parent_floor_idx)` on its
        first live record (PART4 §17.4/§17.9; passed by
        `entity_df_mutation.render_sub_projection`). The zones layer stays
        parent-agnostic — it only sees one int.
    """
    struct_start_by_sid: Dict[int, int] = {}
    established_by_sid: Dict[int, int] = {}
    for ev in events:
        s = (ev.meta or {}).get("structure_id")
        if s is None:
            continue
        s = int(s)
        i = ef.stamped_idx(ev)   # the never-established fallback (see rule 1)
        if s not in struct_start_by_sid or i < struct_start_by_sid[s]:
            struct_start_by_sid[s] = i
        if ev.type == "CTS_ESTABLISHED":
            m = ef.event_moment(ev)
            if s not in established_by_sid or m < established_by_sid[s]:
                established_by_sid[s] = m
    # The lifecycle-start base is a TIME: the first CTS_ESTABLISHED moment (E3f).
    struct_start_by_sid.update(established_by_sid)
    # Reversal handoff: sid N's lifecycle-start = reversal idx of sid N-1.
    for s in list(struct_start_by_sid):
        if (s - 1) in reversal_idx_by_sid:
            struct_start_by_sid[s] = int(reversal_idx_by_sid[s - 1])
    # Floor (subordinate: trigger + parent floors; main: None => no-op).
    if lifecycle_floor is not None:
        for s in struct_start_by_sid:
            struct_start_by_sid[s] = max(struct_start_by_sid[s], int(lifecycle_floor))
    return struct_start_by_sid
