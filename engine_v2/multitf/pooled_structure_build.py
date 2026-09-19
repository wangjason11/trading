"""Natural-end structure builder + window projection for the sub-structure pool
(Phase 2, PART4_REFACTOR_SPEC.md §17.5–§17.7).

Kept separate from `sub_structure_pool` (which stays pandas/MS-free and purely
unit-testable): this module does the actual MarketStructure run + downstream
derivation.

Two functions, both called ONCE per unique sub (§17.7 — a unique sub has one
lifecycle, so one MS run and one downstream projection; the N triggers pointing
at it share both, which is the dedup win):

- `build_structure_geometry` — run `compute_bounded_structure` to the sub's
  **natural end** (its first reversal), bounded only by a generous `run_cap`
  (`min(parent-structure-end, data-edge)`). Independent of any trigger boundary,
  so identical no matter which trigger created the sub.

- `project_to_window` — given the natural-end run + the finalized lifecycle
  window `[floor, cap]`, clip events by **knowable-at** (§17.11) and derive the
  downstream elements (KL/POI/fib/wave/prev-bos) with the window's
  `lifecycle_floor`/`lifecycle_cap`. Byte-identical to a Phase-1 window-bounded
  build (proven by the equivalence tests) because MS is causal in `end_idx`.

Caller status (2026-09-19): `project_to_window` / `clip_events_to_window` were
called only by `entity_df_mutation.render_unique_sub` (Stage 3.2b, `9fd3143`),
which is being reverted — after that this module again has NO live caller
until Plan C, which will call `project_to_window` ONCE per unique sub (unified
window) and mirror the result into each lens df. `build_structure_geometry` has
never had a live caller — production geometry is
`entity_df_mutation._build_or_get_sub_geometry`, which re-implements the same
`compute_bounded_structure` call plus slicing, 50-candle lookback, `reset_index`,
`compute_imbalance` and `is_range` re-derivation; the two have already drifted
and the TESTED one is the dead one. Plan C
(`memory/project_sub_structure_pool_architecture.md`) should collapse them.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from engine_v2.multitf.sub_structure_pool import knowable_at_idx


def build_structure_geometry(
    df: pd.DataFrame,
    *,
    starting_idx: int,
    direction: int,
    run_cap: Optional[int] = None,
    bos0_inner: Optional[float] = None,
    timeframe: str = "M15",
):
    """Run the sub's MarketStructure to its NATURAL end (§17.5).

    `run_cap` is the inclusive upper bound the run may search to — `min(
    parent-structure-end, data-edge)` in production, `None` (→ data edge) in
    tests. It is a COMPUTE bound only; it does not move the natural reversal and
    is distinct from the lifecycle end (§17.6). Returns the
    `BoundedStructureResult`; `.reversal_idx` is the sub's natural reversal
    (entity-absolute) or None if it ran to the edge without reversing.

    `bos0_inner` (a price → slice-invariant) enables the cycle-0
    scan-from-start gate exactly as `build_one_sid` does; None → scan off.
    """
    from engine_v2.structure.structure_engine import compute_bounded_structure

    bounded = compute_bounded_structure(
        df,
        start_idx=int(starting_idx),
        struct_direction=int(direction),
        end_idx=run_cap,
        timeframe=timeframe,
        enforce_cts0_new_extreme=(bos0_inner is not None),
        bos0_inner=bos0_inner,
    )
    # The downstream pipeline reads df.attrs["imbalances"]; MS's internal copy
    # may not carry it, so pin it from the input df (mirrors build_one_sid).
    bounded.df.attrs["imbalances"] = df.attrs.get("imbalances", [])
    return bounded


def clip_events_to_window(events: List[Any], cap: Optional[int]) -> List[Any]:
    """Return events whose **knowable-at** idx is <= `cap` (§17.11).

    `cap` None → no clip (open lifecycle to the data edge). `BOS_CONFIRMED` is
    keyed on `meta["confirmed_at"]`, every other event on `ev.idx`.
    """
    if cap is None:
        return list(events)
    out = []
    for ev in events:
        k = knowable_at_idx(ev.type, ev.idx, ev.meta.get("confirmed_at"))
        if k <= cap:
            out.append(ev)
    return out


def project_to_window(
    bounded,
    *,
    floor: Optional[int],
    cap: Optional[int],
    cap_reason: Optional[str],
    direction: int,
    timeframe: str = "M15",
    source_kinds: Tuple[str, ...] = ("BOS",),
    fib_mode: str = "cross_cycle",
    structure_path_id: str = "H1.main >> M15",
    log_prefix: str = "",
) -> Dict[str, Any]:
    """Derive the render snapshots for one lifecycle window over a natural-end run.

    Clips events by knowable-at (§17.11), then runs the downstream pipeline with
    this window's `lifecycle_floor`/`lifecycle_cap`. Intended to be called ONCE
    per unique sub (§17.7); at HEAD `9fd3143` it is called once per (sub, lens)
    with DIFFERENT floors (the 3.2b per-lens start) — superseded; Plan C returns
    to one call per sub (unified window) mirrored into each lens df. Returns the
    downstream dict (kl_zones, poi_zones, fib_states, wave_candles, wvmi_records,
    prev_bos_lines, ...) plus the clipped `events`.

    Known limits (2026-09-19): `clip_events_to_window` returns the SHARED event
    objects un-deepcopied (safe only while nothing downstream mutates `ev.meta`);
    `knowable_at_idx` special-cases only `BOS_CONFIRMED` — `CTS_ESTABLISHED`
    (`confirmed_at`) and `REVERSAL_CANDIDATE` (`apply_idx`) also straddle a cap,
    so a mid-pair clip yields a half-derived cycle.

    `skip_wvmi=True` (sub WVMI is parent-event-driven, computed later, §8.3/8.4).
    """
    from engine_v2.pipeline.orchestrator import _run_downstream_pipeline

    events = clip_events_to_window(bounded.events, cap)
    downstream = _run_downstream_pipeline(
        bounded.df,
        events,
        direction,
        source_kinds=list(source_kinds),
        fib_mode=fib_mode,
        log_prefix=log_prefix,
        timeframe=timeframe,
        structure_path_id=structure_path_id,
        skip_wvmi=True,
        lifecycle_floor=floor,
        lifecycle_cap=cap,
        cap_reason=cap_reason,
    )
    downstream["events"] = events
    return downstream
