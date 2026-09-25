"""Window projection for the sub-structure pool (PART4_REFACTOR_SPEC.md §17.9).

Kept separate from `sub_structure_pool` (which stays pandas/MS-free and purely
unit-testable): this module runs the downstream derivation over a natural-end
geometry.

`project_to_window` is called ONCE per unique sub (§17.9 — a unique sub has
one lifecycle, so one downstream projection; the N records pointing at it
share it, which is the dedup win): given the natural-end run (run cap = the
data edge, built by `entity_df_mutation.build_or_get_geometry`) and the sub's
lifecycle window `[floor, cap]`, clip events by **knowable-at** and derive the
downstream elements (KL/POI/fib/wave/prev-bos) with the window's
`lifecycle_floor` / `lifecycle_cap`. Byte-identical to a window-bounded build
whenever the window reaches the run's natural reversal (proven by
`tests/test_pooled_structure_build.py`); a window cut by a parent bound can
differ from a run BOUNDED at that candle in the last `range_max_k` candles
(MARKET_STRUCTURE_SPEC "Bounded runs" — truncation ≠ prefix equivalence),
which is why the pool clips ONE data-edge run instead of re-running MS per
bound.

Live caller: `entity_df_mutation.render_sub_projection` (Plan C).
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from engine_v2.structure import event_fields as ef
from engine_v2.multitf.sub_structure_pool import knowable_at_idx


def clip_events_to_window(events: List[Any], cap: Optional[int]) -> List[Any]:
    """Return DEEP COPIES of the events whose **knowable-at** idx is <= `cap`
    (§17.9). `cap` None → no clip (open lifecycle to the data edge).
    `BOS_CONFIRMED` / `CTS_ESTABLISHED` / pattern-path `CTS_UPDATED` are keyed on
    their moment `meta["confirmed_at"]` (Plan E E3b), every other event on its
    stamped idx (`ef.stamped_idx`, today's `ev.idx`).

    Deep-copied because the geometry's event objects are SHARED across every
    consumer of the pool (the mirror stamps attribution onto `ev.meta`).
    """
    out = []
    for ev in events:
        if cap is not None:
            # A TIME clip on the event's moment (Plan E E3b).
            k = knowable_at_idx(ev)
            if k > cap:
                continue
        out.append(deepcopy(ev))
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
    """Derive the render snapshots for ONE lifecycle window over a natural-end run.

    Clips events by knowable-at (§17.9), then runs the downstream pipeline with
    this window's `lifecycle_floor` / `lifecycle_cap` / `cap_reason` (the sub's
    `start_idx` / `end_idx` / `end_reason`, slice-local). Returns the
    downstream dict (kl_zones, poi_zones, fib_states, wave_candles,
    wvmi_records, prev_bos_lines, ...) plus the clipped `events`.

    Known limit (§17.12): `knowable_at_idx` keys CTS / BOS events on their moment
    (CTS_ESTABLISHED / CTS_UPDATED since Plan E E3b); only `REVERSAL_CANDIDATE`
    (`apply_idx`) still straddles a cap, so a mid-pair clip can yield a
    half-derived reversal.

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
