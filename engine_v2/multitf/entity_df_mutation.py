"""Entity-df mutation primitives + the sub build model for the sub-structure pool
(PART4_REFACTOR_SPEC.md §17.8–§17.9; landed by Plan C, 2026-09).

What lives here:

  - **Start resolvers** (§17.8) — `_resolve_trigger_m15_start` dispatches by
    `use_case`: `first_confluence` → own ad-hoc BOS_0 reference + unified probe
    Phase 1+2; `first_counter` / `subsequent_*` → the SIBLING's most recent
    qualifying CTS read **from the pool** (`_build_sibling_cts_ref_zone_from_pool`)
    + Phase 1. `_resolve_reversal_start` is the reversal handoff over the
    reversing sub's own geometry. Every probe goes through `_probe_with_cache`
    (the §17.8 probe cache). Each returns a `ResolvedStart` or a `ProbeFailure`.
  - **Geometry** (§17.8) — `build_or_get_geometry`: the single owner of pool
    lookup + MS run (run cap = the DATA EDGE) + `natural_reversal_idx` +
    `bos0_inner`. MS first; the pool entry is created only on success.
  - **Projection + mirror** (§17.9) — `render_sub_projection` runs
    `project_to_window` ONCE per unique sub over the sub's `[start_idx,
    end_idx]` and `mirror_lower_tf_result_to_entity_df` mirrors the
    slice-shape `LowerTFResult` into every lens df the sub belongs to,
    translating slice-local → entity-absolute and stamping `sub_id`.
  - `_map_parent_idx_to_m15_hour_end` (LOH — the mapper for every TIMING
    value) and `_synth_reversal_trigger`.

The lifecycle DRIVER is `multitf/lifecycle_sweep.py` (the §17.6 sweep); the
data model is `multitf/sub_structure_pool.py`; the parent tables are
`multitf/parent_tables.py`. The Phase-1 two-entity cadence chain
(`_ChainCursor`, `build_two_entity_parent_cycle`, `build_parent_cycle_chain`,
`build_one_sid`) and the legacy parent-TF probe escape hatch are gone.

NOTE: the slice + 50-lookback + `reset_index` machinery (and the mirror's
slice-local → entity-absolute translation) is retained — direct entity-df MS
compute needs an MS refactor that is deferred.
"""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from engine_v2.common.types import KLZone
from engine_v2.multitf.lifecycle_sweep import ProbeFailure, ResolvedStart
from engine_v2.structure import event_fields as ef
from engine_v2.multitf.sub_structure_pool import (
    LENS_CONFLUENCE,
    LENS_COUNTER,
    PooledStructure,
    ProbeCacheEntry,
    StructureKey,
    SubStructurePool,
)
from engine_v2.multitf.types import LowerTFResult, MultiTFTrigger


# Structure columns potentially written by MarketStructure; we mirror
# back any that exist on the result df. Missing columns are skipped.
_STRUCTURE_COLS = (
    "structure_id",
    "cycle_id",
    "cts_phase",
    "market_state",
    "range_lo",
    "range_hi",
    "swing_dir",
    "cts_price",
    "bos_price",
)

# Additional MarketStructure-managed columns (per
# `market_structure._ensure_output_cols`) that we do NOT mirror back
# into entity_df explicitly but DO need to drop from the working copy
# fed to MS so prior sids' values don't leak into the new sid's
# state-machine reads. MS will recreate them with proper defaults via
# `_ensure_output_cols`.
_MS_AUX_STRUCTURE_COLS = (
    "range_active",
    "range_start_idx",
    "range_confirm_idx",
    "breakout_th",
    "pullback_th",
    "range_break_frac",
    "cts_idx",
    "cts_event",
    "bos_idx",
    "bos_event",
    "cts_cycle_id",
    "cts_threshold",
    "bos_threshold",
    "cycle_stage",
    "cts_phase_debug",
    "reversal_watch_active",
    "reversal_bos_th_frozen",
    "pending_reversal_pattern_anchor_idx",
    "pending_reversal_apply_idx",
    "struct_direction",
    "last_breakout_pat_apply_idx",
)

# Event-meta keys whose values are entity-df indices and need translating
# from slice-local to entity-absolute.
_EVENT_META_IDX_KEYS = (
    "confirmed_at",
    "apply_idx",
    "pattern_anchor_idx",
    "confirmed_idx",
    "cts_anchor_idx",
    "bos_anchor_idx",
    "pb_reconfirm_idx",
    "deactivated_at",
)

# Zone-meta keys with entity-df indices.
_ZONE_META_IDX_KEYS = (
    "anchor_idx",
    "confirmed_idx",
    "cts_established_idx",
    "start_idx",
    "end_idx",
    "base_idx",
    "ic_idx",
    "deactivated_at",
)


def _attrs_setdefault_list(entity_df: pd.DataFrame, key: str) -> list:
    if key not in entity_df.attrs:
        entity_df.attrs[key] = []
    return entity_df.attrs[key]


def _shift_meta_indices(meta: dict, keys, offset: int) -> dict:
    """Return a new dict with int values at `keys` shifted by `offset`."""
    out = dict(meta)
    for k in keys:
        v = out.get(k)
        if isinstance(v, int):
            out[k] = v + offset
    return out



def _sub_attribution(result: LowerTFResult, structure_path_id: str) -> Dict[str, Any]:
    """§17.9 attribution for every mirrored snapshot: `structure_path_id`,
    `timeframe`, `parent_tf`, **`sub_id`** (identity) + informational
    `parent_sid` / `parent_cycle_id` / `use_case` / `started_by` from the
    sub's first live record (read off `result.trigger` / `result.meta`; no
    consumer may use them for identity). Legacy/test results without a
    `sub_id` stamp only the trigger-derived fields."""
    trig = result.trigger
    attribution: Dict[str, Any] = {
        "structure_path_id": structure_path_id,
        "use_case": getattr(trig, "use_case", None),
        "parent_sid": getattr(trig, "parent_sid", None),
        "parent_cycle_id": getattr(trig, "parent_cycle_id", None),
        "timeframe": getattr(trig, "lower_tf", result.meta.get("timeframe")),
        "parent_tf": getattr(trig, "parent_tf", result.meta.get("parent_tf")),
    }
    for _k in ("sub_id", "started_by"):
        if _k in result.meta:
            attribution[_k] = result.meta[_k]
    return attribution


def mirror_lower_tf_result_to_entity_df(
    entity_df: pd.DataFrame,
    result: LowerTFResult,
    *,
    structure_path_id: str,
) -> None:
    """Mirror a slice-shape `LowerTFResult` into `entity_df.attrs[...]`.

    Translates every event / zone / POI / fib / wave-candle / WVMI idx
    field from slice-local (the result is built on a sliced + reset_index
    df) to entity-absolute (slice_begin + offset). Stamps the PART4 §17.9
    attribution onto every snapshot's meta: `structure_path_id`, `timeframe`,
    `parent_tf`, **`sub_id`** (the identity) + informational `parent_sid` /
    `parent_cycle_id` / `use_case` / `started_by` from the sub's first live
    record (`_sub_attribution`).

    Also mirrors the structure columns from `result.df` back to
    `entity_df.iloc[slice_begin:slice_begin+len(result.df)]` so the
    entity df reflects the per-candle "current truth" view of this sub —
    the orchestrator mirrors subs in `start_idx` order so a later-live sub
    wins overlapping candles (§17.9).
    """
    slice_begin = int(result.meta.get("slice_begin", 0))

    attribution = _sub_attribution(result, structure_path_id)

    # 2. Mirror structure columns — the sub's LIVE rows only (§17.9 "later-live
    #    wins": subs are mirrored in start_idx order, so painting from the
    #    sub's `start_idx` rather than from `slice_begin` keeps a later sub's
    #    50-candle lookback / pre-start rows from overwriting an earlier sub's
    #    live rows). Results without a `start_idx` (legacy/test shapes) paint
    #    from `slice_begin` as before.
    n_slice_rows = len(result.df)
    n_entity = len(entity_df)
    end_in_entity = min(slice_begin + n_slice_rows - 1, n_entity - 1)
    _start_abs = result.meta.get("start_idx")
    first_in_entity = max(slice_begin, int(_start_abs)) if _start_abs is not None else slice_begin
    first_in_slice = first_in_entity - slice_begin
    rows_to_write = end_in_entity - first_in_entity + 1
    if rows_to_write > 0:
        slice_df = result.df.iloc[first_in_slice:first_in_slice + rows_to_write]
        for col in _STRUCTURE_COLS:
            if col not in slice_df.columns:
                continue
            if col not in entity_df.columns:
                # Initialize with appropriate fill — match dtype where possible
                entity_df[col] = pd.NA
            col_loc = entity_df.columns.get_loc(col)
            entity_df.iloc[first_in_entity:end_in_entity + 1, col_loc] = (
                slice_df[col].values
            )

    # 3. Events — translate idx + idx-bearing meta keys; append
    new_events = []
    for ev in result.events:
        new_ev = deepcopy(ev)
        new_ev.idx = ev.idx + slice_begin
        new_ev.meta = _shift_meta_indices(
            new_ev.meta, _EVENT_META_IDX_KEYS, slice_begin,
        )
        new_ev.meta.update(attribution)
        new_events.append(new_ev)
    _attrs_setdefault_list(entity_df, "events").extend(new_events)

    # 4. KL zones — idx only in meta (bounds_steps[*]["start_idx"] +
    #    activation_history[*]["idx"] under the Phase 3 convention)
    new_kl = []
    for z in result.kl_zones:
        new_meta = _shift_meta_indices(z.meta, _ZONE_META_IDX_KEYS, slice_begin)
        if "bounds_steps" in new_meta:
            steps = []
            for step in new_meta["bounds_steps"]:
                new_step = dict(step)
                if isinstance(new_step.get("start_idx"), int):
                    new_step["start_idx"] = new_step["start_idx"] + slice_begin
                steps.append(new_step)
            new_meta["bounds_steps"] = steps
        ah = new_meta.get("activation_history")
        if ah:
            new_meta["activation_history"] = [
                {**ev, "idx": int(ev["idx"]) + slice_begin}
                for ev in ah
            ]
        new_meta.update(attribution)
        new_kl.append(replace(z, meta=new_meta))
    _attrs_setdefault_list(entity_df, "kl_zones").extend(new_kl)

    # 5. POI zones — direct ic_idx + meta + activation_history list
    new_poi = []
    for z in result.poi_zones:
        new_meta = _shift_meta_indices(z.meta, _ZONE_META_IDX_KEYS, slice_begin)
        # Shift each idx inside activation_history (list of {idx, active, reason}).
        ah = new_meta.get("activation_history")
        if ah:
            new_meta["activation_history"] = [
                {**ev, "idx": int(ev["idx"]) + slice_begin}
                for ev in ah
            ]
        new_meta.update(attribution)
        new_poi.append(replace(z, ic_idx=z.ic_idx + slice_begin, meta=new_meta))
    _attrs_setdefault_list(entity_df, "poi_zones").extend(new_poi)

    # 6. Fib states — direct bos_idx / cts_idx + meta. The scalar lifecycle
    #    fields (FIB_LIFECYCLE_SPEC.md §15) carry slice-local indices: both
    #    `start_idx` and `end_idx` must shift by slice_begin alongside the
    #    anchors. `end_reason` / `status` are not indices -> pass through.
    new_fibs = []
    for fib in result.fib_states:
        new_meta = _shift_meta_indices(fib.meta, ("deactivated_at",), slice_begin)
        new_meta.update(attribution)
        new_start_idx = (
            fib.start_idx + slice_begin
            if isinstance(fib.start_idx, int)
            else fib.start_idx
        )
        new_end_idx = (
            fib.end_idx + slice_begin
            if isinstance(fib.end_idx, int)
            else fib.end_idx
        )
        new_fibs.append(replace(
            fib,
            bos_idx=fib.bos_idx + slice_begin,
            cts_idx=fib.cts_idx + slice_begin,
            start_idx=new_start_idx,
            end_idx=new_end_idx,
            meta=new_meta,
        ))
    _attrs_setdefault_list(entity_df, "fib_states").extend(new_fibs)

    # 7. Wave candles — direct optional idx fields
    new_wcs = []
    for wc in result.wave_candles:
        new_meta = dict(wc.meta)
        new_meta.update(attribution)
        new_wcs.append(replace(
            wc,
            last_wave_candle_idx=(
                wc.last_wave_candle_idx + slice_begin
                if wc.last_wave_candle_idx is not None else None
            ),
            first_wave_candle_idx=(
                wc.first_wave_candle_idx + slice_begin
                if wc.first_wave_candle_idx is not None else None
            ),
            meta=new_meta,
        ))
    _attrs_setdefault_list(entity_df, "wave_candles").extend(new_wcs)

    # 8. WVMI records — mutable dataclass; deepcopy then translate.
    # `triggered_by_event_idx` in meta is the PARENT trigger idx (parent
    # df coords, e.g. H1) — do NOT translate.
    new_wvmis = []
    for w in result.wvmi_records:
        nw = deepcopy(w)
        for attr in ("fb_idx", "lb_idx", "fp_idx", "lp_idx"):
            cur = getattr(nw, attr)
            if cur is not None:
                setattr(nw, attr, cur + slice_begin)
        nw.meta.update(attribution)
        new_wvmis.append(nw)
    _attrs_setdefault_list(entity_df, "wvmi").extend(new_wvmis)

    # 9. Prev BOS lines — list of dicts with slice-local start_idx /
    # end_idx. Translate both, embed the attribution under "meta" so the
    # chart can group by `sub_id` (§17.9).
    new_prev_bos = []
    for ln in result.prev_bos_lines:
        if not isinstance(ln, dict):
            new_prev_bos.append(ln)
            continue
        new_ln = dict(ln)
        if isinstance(new_ln.get("start_idx"), int):
            new_ln["start_idx"] = new_ln["start_idx"] + slice_begin
        if isinstance(new_ln.get("end_idx"), int):
            new_ln["end_idx"] = new_ln["end_idx"] + slice_begin
        new_ln["meta"] = dict(new_ln.get("meta") or {})
        new_ln["meta"].update(attribution)
        new_prev_bos.append(new_ln)
    _attrs_setdefault_list(entity_df, "prev_bos_lines").extend(new_prev_bos)


def persist_facade_wvmi_to_entity_df(
    entity_df: pd.DataFrame,
    facade: LowerTFResult,
    *,
    structure_path_id: str,
) -> None:
    """Translate slice-local WVMI records on a facade to entity-absolute
    idx and append to `entity_df.attrs["wvmi"]`.

    Sub WVMI is parent-event-driven (§8.3 / §8.4 / §17.10), computed by the
    orchestrator's per-sub pass AFTER the projections are mirrored. The
    orchestrator then calls this helper once per lens df the sub is on.

    Stamps the §17.9 attribution (`sub_id` + informational parent fields)
    onto each record's meta. `triggered_by_event_idx` in record meta is parent-df
    coords (LANDMINE "WVMI Records Carry Mixed-Coordinate Meta") — do NOT
    translate.
    """
    slice_begin = int(facade.meta.get("slice_begin", 0))
    attribution = _sub_attribution(facade, structure_path_id)
    new_records = []
    for w in facade.wvmi_records:
        nw = deepcopy(w)
        for attr in ("fb_idx", "lb_idx", "fp_idx", "lp_idx"):
            cur = getattr(nw, attr)
            if cur is not None:
                setattr(nw, attr, cur + slice_begin)
        nw.meta.update(attribution)
        new_records.append(nw)
    _attrs_setdefault_list(entity_df, "wvmi").extend(new_records)



def _build_first_confluence_ref_zone(
    m15_df: pd.DataFrame,
    m15_input_idx: int,
    probe_direction: int,
) -> Optional["ReferenceZone"]:
    """Ad-hoc BOS_0 reference zone on M15 for first_confluence.

    Anchor = the M15 input_idx candle (= the parent BOS extreme
    price-mapped to M15). Derives a BOS-style zone from that candle via
    the same `identify_base_pattern + zone_thresholds` machinery the
    legacy Scenario 3 used, just on the SUB's TF rather than the parent's.

    Held constant across probe iterations per the unified-probe design.

    Thin delegate (2026-06-07): the ad-hoc BOS_0 derivation now lives in
    `reference_zone.build_ad_hoc_bos0_reference_zone` (single source of
    truth — the unified probe's MOVING BOS_0 reuses the SAME machinery for
    its post-reset threshold). Behaviour is byte-identical to the prior
    inline implementation.
    """
    from engine_v2.structure.reference_zone import (
        build_ad_hoc_bos0_reference_zone,
    )

    return build_ad_hoc_bos0_reference_zone(
        m15_df, int(m15_input_idx), int(probe_direction),
    )



def _build_sibling_cts_ref_zone_from_pool(
    pool: SubStructurePool,
    other_lens: str,
    parent_sid: int,
    parent_cycle_id: int,
    probe_direction: int,
    idx_window: Tuple[int, int],
    m15_df: pd.DataFrame,
) -> Optional["ReferenceZone"]:
    """Sibling-CTS reference zone for the three sibling-referencing variations
    (`first_counter`, `subsequent_confluence`, `subsequent_counter`) — a POOL
    QUERY (§17.8; replaces the per-trigger scratch entity-df read).

    Reads "the most recent qualifying CTS of the opposite-direction sub on the
    OTHER lens in this parent cycle":

    - candidates = `pool.records_for(other_lens, S, C)` with `direction ==
      -probe_direction` (direction qualification, 2026-06-15: a sibling that
      reversed away from its expected direction is no longer a genuine
      confluence/counter relative to the parent), non-zero-length, and LIVE
      somewhere in the window (`start_idx <= hi` and not ended before `lo`);
    - each record's events come from its sub's geometry (SLICE-LOCAL → shifted
      by `slice_begin`), **clipped to the record's own live window ∩ `[lo,
      hi]`** — this is what reproduces the Phase-1 candidate set now that
      geometry runs to the data edge (a REPLACED sub's later `CTS_UPDATED`s
      must not compete: sub `3304/−1` is replaced at 3819 but its geometry
      continues; its post-3819 CTS events would otherwise move
      `subsequent_counter`(1,2)@4083's `starting_idx` 4027 — a pool key);
    - `hi` = the reading trigger's `trigger_idx`: the window is the ONLY thing
      keeping the read causal — never widen it;
    - the reference zone is built ad hoc from the winning CTS event with
      `kl_zones=[]` — behaviour-preserving, not a shortcut: subs only ever
      receive BOS zones (`source_kinds=["BOS"]`), so the primitive's
      CONFIRMED-zone branch has been dead for subs since the narrowing (GOTCHAS
      "`ref=cts_confirmed` in a Replay Log Does NOT Mean…"). `df` = the shared
      entity-absolute M15 frame (the winner's own `bounded.df` is slice-local).

    The returned zone's `anchor_idx` is the sibling CTS extreme
    (entity-absolute) — the caller uses it as BOTH the probe `input_idx` AND
    the reference zone (co-sourced). Returns None when no qualifying CTS exists
    in the window (caller → own-entity ad-hoc fallback).
    """
    from engine_v2.structure.reference_zone import (
        build_reference_zone_from_cts_event,
        _CTS_EVENT_TYPES,
    )

    lo, hi = int(idx_window[0]), int(idx_window[1])
    expected_sd = -int(probe_direction)
    recs = [
        r for r in pool.records_for(other_lens, parent_sid, parent_cycle_id)
        if r.direction == expected_sd
        and not r.is_zero_length
        and r.start_idx <= hi
        and (r.trigger_end_idx is None or r.trigger_end_idx >= lo)
    ]
    if not recs:
        return None

    events: List[Any] = []
    for r in recs:
        sub = pool.get_by_id(r.sub_id)
        if sub.geometry is None:
            continue
        bounded, slice_begin = sub.geometry
        r_lo = max(lo, int(r.start_idx))
        r_hi = min(hi, int(r.trigger_end_idx) if r.trigger_end_idx is not None else hi)
        for ev in bounded.events:
            if ev.type not in _CTS_EVENT_TYPES:
                continue
            # The clip is a TIME: the event's MOMENT (Plan E E3b — keyed so
            # while CTS_ESTABLISHED / pattern-path CTS_UPDATED were stamped at
            # their anchor, until Plan E E4a / E4c).
            clip_idx = ef.event_moment(ev) + int(slice_begin)
            if not (r_lo <= clip_idx <= r_hi):
                continue
            new_ev = deepcopy(ev)
            # The mirror keeps the RAW index (shifted to entity-absolute).
            new_ev.idx = int(ev.idx) + int(slice_begin)
            new_ev.meta = _shift_meta_indices(
                new_ev.meta, _EVENT_META_IDX_KEYS, int(slice_begin),
            )
            events.append(new_ev)
    if not events:
        return None
    return build_reference_zone_from_cts_event(
        events=events,
        kl_zones=[],
        df=m15_df,
        sid=0,
        probe_direction=int(probe_direction),
        idx_window=(lo, hi),
    )


def _probe_with_cache(
    pool: Optional[SubStructurePool],
    *,
    m15_df: pd.DataFrame,
    parent_path: str,
    sub_tf: str,
    direction: int,
    input_idx: int,
    reference_zone: "ReferenceZone",
    probe_end_idx: int,
    enable_phase2: bool,
    timeframe: str,
    label: str,
) -> Tuple[int, int, str, Optional[float], str, bool]:
    """Run `unified_probe` behind the §17.8 probe cache.

    Key `(parent_path, sub_tf, direction, input_idx)` — "same probe" = same
    direction + same initial input. **The first probe to finalize for a key is
    the truth for every later probe of that key, (a) regardless of its search
    bound `probe_end_idx` and (b) regardless of its reference zone** (accepted
    approximation, 2026-09-19). Tripwire for (b): the hitting trigger's
    reference inner is compared to the cached probe's OWN reference inner
    (`ProbeCacheEntry.ref_inner` — iteration 1's BOS_0 threshold; NOT the
    cached `bos0_inner`, which is the final iteration's threshold and moves on
    every reset) and `[probe_cache] REF-ZONE DIFFERS` is logged when they are
    not close. A hit skips `unified_probe` entirely; `APPROX hit` when the
    bound differs.

    Returns `(starting_idx, finalize_idx, finalize_condition, bos0_inner,
    status, cache_hit)`; `status == "pending"` only from a real run.
    `pool=None` → no cache (unit tests / diagnostics).
    """
    from engine_v2.structure.unified_probe import unified_probe

    if pool is not None:
        hit = pool.get_cached_probe(parent_path, sub_tf, direction, input_idx)
        if hit is not None:
            kind = "hit" if hit.probe_end_idx == int(probe_end_idx) else "APPROX hit"
            print(
                f"[probe_cache] {kind} {label} key=({direction},{input_idx}) "
                f"probe_end_idx={probe_end_idx} cached_end={hit.probe_end_idx} "
                f"-> starting_idx={hit.starting_idx} finalize={hit.finalize_idx} "
                f"cond={hit.finalize_condition}"
            )
            ref_inner = float(reference_zone.inner)
            cached_ref = hit.ref_inner if hit.ref_inner is not None else hit.bos0_inner
            if cached_ref is None or not math.isclose(ref_inner, float(cached_ref), abs_tol=1e-9):
                print(
                    f"[probe_cache] REF-ZONE DIFFERS {label} key=({direction},{input_idx}): "
                    f"ref_inner={ref_inner} cached_ref_inner={cached_ref} "
                    f"(first-probe-is-truth assumption applied)"
                )
            return (
                int(hit.starting_idx), int(hit.finalize_idx), str(hit.finalize_condition),
                hit.bos0_inner, "finalized", True,
            )
        print(f"[probe_cache] miss {label} key=({direction},{input_idx}) probe_end_idx={probe_end_idx}")

    result = unified_probe(
        m15_df,
        input_idx=int(input_idx),
        direction=int(direction),
        reference_zone=reference_zone,
        probe_end_idx=int(probe_end_idx),
        timeframe=timeframe,
        enable_phase2=enable_phase2,
    )
    print(
        f"[entity_compute] unified_probe ({label}): m15_input={input_idx} "
        f"probe_end_idx={probe_end_idx} ref={reference_zone.source} -> "
        f"starting_idx={result.starting_idx} status={result.status} "
        f"cond={result.finalize_condition} iter={result.iterations}"
    )
    if result.status == "pending":
        return (int(result.starting_idx), -1, str(result.finalize_condition),
                result.bos0_inner, "pending", False)
    assert result.finalize_idx is not None, (
        f"[entity_compute] finalized probe without finalize_idx ({label})"
    )
    if pool is not None:
        pool.record_probe(
            parent_path, sub_tf, direction, input_idx,
            ProbeCacheEntry(
                starting_idx=int(result.starting_idx),
                finalize_idx=int(result.finalize_idx),
                finalize_condition=str(result.finalize_condition),
                bos0_inner=result.bos0_inner,
                probe_end_idx=int(probe_end_idx),
                ref_inner=float(reference_zone.inner),
            ),
        )
    return (
        int(result.starting_idx), int(result.finalize_idx), str(result.finalize_condition),
        result.bos0_inner, "finalized", False,
    )


def _resolve_first_confluence_via_unified_probe(
    trigger: MultiTFTrigger,
    parent_df: pd.DataFrame,
    m15_df: pd.DataFrame,
    *,
    pool: Optional[SubStructurePool] = None,
    parent_path: str = "H1.main",
):
    """Unified-probe path for `first_confluence` (the one variation that
    anchors on its OWN ad-hoc BOS_0, not a sibling). Phase 2 (MS-based) runs
    because first_confluence's probe bound is the parent CTS extreme — the only
    variation needing the post-bound MS search (PART4 §4.4).

    Returns a `ResolvedStart` (`validated_parent_idx` = the parent BOS extreme
    that seeded the M15 input) or a `ProbeFailure`.
    """
    from engine_v2.multitf.data_bridge import map_candle_to_lower_tf

    label = f"first_confluence sid={trigger.parent_sid} cycle={trigger.parent_cycle_id}"
    raw_input = trigger.meta.get("probe_input_idx")
    raw_end = trigger.meta.get("probe_end_idx")
    if raw_input is None or raw_end is None:
        print(f"[entity_compute] WARNING: missing probe_input_idx/probe_end_idx in trigger meta for {label}")
        return ProbeFailure("missing probe_input_idx/probe_end_idx", None)
    parent_bos_anchor_idx = int(raw_input)
    parent_cts_anchor_idx = int(raw_end)
    if parent_bos_anchor_idx not in parent_df.index:
        print(f"[entity_compute] WARNING: probe_input_idx={parent_bos_anchor_idx} out of parent_df bounds for {label}")
        return ProbeFailure(f"probe_input_idx {parent_bos_anchor_idx} out of parent bounds", None)

    # Candle-semantics mapping rule (user spec 2026-05-31): BOS/CTS ANCHOR
    # candles are PRICE-mapped (they anchor a price level into the sub);
    # trigger/confirmation candles are TIME-mapped (temporal gates). Both of
    # first_confluence's bounds are ANCHOR candles, so BOTH are price-mapped:
    #   - input = parent BOS anchor, extreme on the -lower_sd side (base of the
    #     breakout — the price the probe measures retraces from).
    #   - probe_end_idx = parent CTS anchor, extreme on the +lower_sd side (the
    #     structure ceiling/floor). NOT a temporal cutoff: "once price hit the
    #     parent CTS extreme, no NEW extreme can form, so stop probing" — a
    #     price bound for the search; changing it moves `starting_idx` = the
    #     pool key (§17.8 "FC probe end mapping — unchanged").
    parent_bos_anchor_time = pd.to_datetime(parent_df.loc[parent_bos_anchor_idx, "time"], utc=True)
    m15_input_idx = map_candle_to_lower_tf(parent_bos_anchor_time, -trigger.lower_sd, m15_df)
    if m15_input_idx is None:
        print(f"[entity_compute] WARNING: parent→M15 input mapping failed for {label}")
        return ProbeFailure("parent→M15 input mapping failed", parent_bos_anchor_idx)
    if parent_cts_anchor_idx not in parent_df.index:
        print(f"[entity_compute] WARNING: probe_end_idx={parent_cts_anchor_idx} out of parent_df bounds for {label}")
        return ProbeFailure(f"probe_end_idx {parent_cts_anchor_idx} out of parent bounds", parent_bos_anchor_idx)
    parent_cts_anchor_time = pd.to_datetime(parent_df.loc[parent_cts_anchor_idx, "time"], utc=True)
    m15_probe_end_idx = map_candle_to_lower_tf(parent_cts_anchor_time, trigger.lower_sd, m15_df)
    if m15_probe_end_idx is None:
        print(f"[entity_compute] WARNING: parent→M15 end mapping failed for {label}")
        return ProbeFailure("parent→M15 end mapping failed", parent_bos_anchor_idx)
    if m15_probe_end_idx <= m15_input_idx:
        print(
            f"[entity_compute] WARNING: degenerate probe window for {label} "
            f"(m15_input={m15_input_idx} m15_end={m15_probe_end_idx})"
        )
        return ProbeFailure(f"degenerate probe window ({m15_input_idx} >= {m15_probe_end_idx})", int(m15_input_idx))

    ref_zone = _build_first_confluence_ref_zone(m15_df, int(m15_input_idx), int(trigger.lower_sd))
    if ref_zone is None:
        print(f"[entity_compute] WARNING: ad_hoc_bos_0 reference zone unavailable for {label} — skipping trigger")
        return ProbeFailure("ad-hoc BOS_0 reference zone unavailable", int(m15_input_idx))

    starting_idx, finalize_idx, cond, bos0_inner, status, cache_hit = _probe_with_cache(
        pool, m15_df=m15_df, parent_path=parent_path, sub_tf=trigger.lower_tf,
        direction=int(trigger.lower_sd), input_idx=int(m15_input_idx),
        reference_zone=ref_zone, probe_end_idx=int(m15_probe_end_idx),
        enable_phase2=True, timeframe=trigger.lower_tf, label=label,
    )
    if status == "pending":
        print(f"[entity_compute] PENDING: unified_probe did not finalize for {label}; skipping M15 build")
        return ProbeFailure("probe pending", int(m15_input_idx))
    return ResolvedStart(
        starting_idx=starting_idx, validated_parent_idx=parent_bos_anchor_idx,
        bos0_inner=bos0_inner, finalize_idx=finalize_idx, finalize_condition=cond,
        probe_input_idx=int(m15_input_idx), cache_hit=cache_hit,
    )

def _window_extreme_idx(
    df: pd.DataFrame, lo: int, hi: int, extreme_dir: int,
) -> Optional[int]:
    """Idx of the extreme candle in `[lo, hi]` on the `extreme_dir` side.

    `extreme_dir == +1` → highest high; `extreme_dir == -1` → lowest low.
    Tie-break earliest. Used by the sibling-CTS fallback (PART4 §4.3.4 step 5)
    to anchor an ad-hoc BOS_0 on the own entity when the sibling has no CTS.
    Returns None for an empty / out-of-bounds window.
    """
    lo = max(int(lo), int(df.index[0]))
    hi = min(int(hi), int(df.index[-1]))
    if lo > hi:
        return None
    seg = df.loc[lo:hi]
    if seg.empty:
        return None
    if extreme_dir == 1:
        return int(seg["h"].astype(float).idxmax())
    return int(seg["l"].astype(float).idxmin())


def _sibling_cts_idx_window(
    trigger: MultiTFTrigger,
    parent_df: pd.DataFrame,
    m15_df: pd.DataFrame,
    hi: int,
) -> Tuple[int, int]:
    """Inclusive entity-absolute M15 `(lo, hi)` window the sibling CTS must
    fall in, per use_case (PART4 §4.3.3/4/5, Session 3).

    - `first_counter`: `[0, hi]` — the whole parent cycle up to the first
      sd-prox trigger (the pool read is already scoped to this parent cycle,
      so 0 is a safe lower bound).
    - `subsequent_confluence`: `[last-M15 of prior_sd_prox hour, hi]`.
    - `subsequent_counter`: `[last-M15 of prior_cts_prox hour, hi]`.

    `hi` = the reading trigger's `trigger_idx` (the last M15 of its H1 hour) —
    the ONLY thing keeping the sibling read causal (§17.8). A missing prior-prox
    meta value degrades `lo` to 0 (safe — the parent-cycle scope still bounds
    the read).
    """
    hi = int(hi)
    uc = trigger.use_case
    prior_parent_idx: Optional[int] = None
    if uc == "subsequent_confluence":
        prior_parent_idx = trigger.meta.get("prior_sd_trigger_idx")
    elif uc == "subsequent_counter":
        prior_parent_idx = trigger.meta.get("prior_cts_prox_idx")
    # first_counter (and any fallback): lo = 0.
    lo = 0
    if prior_parent_idx is not None:
        mapped = _map_parent_idx_to_m15_hour_end(
            int(prior_parent_idx), parent_df, m15_df,
        )
        if mapped is not None:
            lo = int(mapped)
    return lo, hi



_SIBLING_LENS = {
    "first_counter": LENS_CONFLUENCE,          # counter reads the confluence sibling
    "subsequent_counter": LENS_CONFLUENCE,
    "subsequent_confluence": LENS_COUNTER,     # confluence reads the counter sibling
}


def _resolve_sibling_cts_via_unified_probe(
    trigger: MultiTFTrigger,
    parent_df: pd.DataFrame,
    m15_df: pd.DataFrame,
    *,
    pool: SubStructurePool,
    hi: int,
    parent_path: str = "H1.main",
):
    """Unified-probe path for the three sibling-referencing variations
    (`first_counter`, `subsequent_confluence`, `subsequent_counter`) — Session 3
    uniform rule (2026-05-31), sibling read from the POOL (§17.8).

    Both the probe `input_idx` AND the `reference_zone` are co-sourced from the
    sibling lens's most recent qualifying CTS in the trigger's M15 window
    `[lo, hi]` (`hi` = this trigger's `trigger_idx`, the last M15 of its H1
    hour). `enable_phase2=False`. Returns a `ResolvedStart`
    (`validated_parent_idx` = the sibling CTS extreme, M15) or a `ProbeFailure`.
    """
    label = f"{trigger.use_case} sid={trigger.parent_sid} cycle={trigger.parent_cycle_id}"
    other_lens = _SIBLING_LENS.get(trigger.use_case)
    if other_lens is None:
        raise ValueError(f"[entity_compute] not a sibling-referencing use_case: {trigger.use_case!r}")
    hi = int(hi)

    # 1. Sibling-CTS reference zone within the per-variation M15 window. Its
    #    `anchor_idx` IS the probe input_idx (co-sourced).
    idx_window = _sibling_cts_idx_window(trigger, parent_df, m15_df, hi)
    ref_zone = _build_sibling_cts_ref_zone_from_pool(
        pool, other_lens, int(trigger.parent_sid), int(trigger.parent_cycle_id),
        int(trigger.lower_sd), idx_window, m15_df,
    )
    if ref_zone is None:
        # Fallback (PART4 §4.3.4 step 5) — the sibling has no qualifying CTS in
        # the window. Anchor on the OWN frame: input = window extreme on the
        # `-lower_sd` side, reference = own ad-hoc BOS_0 from that candle.
        fallback_anchor_idx = _window_extreme_idx(m15_df, idx_window[0], idx_window[1], -int(trigger.lower_sd))
        if fallback_anchor_idx is not None:
            ref_zone = _build_first_confluence_ref_zone(m15_df, int(fallback_anchor_idx), int(trigger.lower_sd))
        if ref_zone is None:
            print(
                f"[entity_compute] WARNING: sibling-CTS reference zone AND ad-hoc fallback "
                f"both unavailable for {label} window={idx_window} — skipping trigger"
            )
            return ProbeFailure("sibling-CTS reference zone and ad-hoc fallback unavailable", None)
        print(
            f"[entity_compute] sibling-CTS unavailable for {label} window={idx_window} "
            f"— using ad-hoc BOS_0 fallback at {fallback_anchor_idx}"
        )

    m15_input_idx = int(ref_zone.anchor_idx)
    if m15_input_idx >= hi:
        print(
            f"[entity_compute] WARNING: degenerate sibling-CTS probe window for {label} "
            f"(m15_input={m15_input_idx} m15_end={hi})"
        )
        return ProbeFailure(f"degenerate sibling-CTS probe window ({m15_input_idx} >= {hi})", m15_input_idx)

    # 2. Probe on the shared M15 frame (Phase 1 only), behind the cache.
    starting_idx, finalize_idx, cond, bos0_inner, status, cache_hit = _probe_with_cache(
        pool, m15_df=m15_df, parent_path=parent_path, sub_tf=trigger.lower_tf,
        direction=int(trigger.lower_sd), input_idx=m15_input_idx,
        reference_zone=ref_zone, probe_end_idx=hi,
        enable_phase2=False, timeframe=trigger.lower_tf, label=label,
    )
    if status == "pending":
        print(f"[entity_compute] PENDING: unified_probe did not finalize for {label}; skipping M15 build")
        return ProbeFailure("probe pending", m15_input_idx)
    if not cache_hit:
        # §5.2: a Phase-1 probe bounded at `hi` finalizes AT `hi` by construction.
        assert finalize_idx == hi, (
            f"[entity_compute] {label}: finalize_idx={finalize_idx} != trigger_idx={hi}"
        )
    return ResolvedStart(
        starting_idx=starting_idx, validated_parent_idx=m15_input_idx,
        bos0_inner=bos0_inner, finalize_idx=finalize_idx, finalize_condition=cond,
        probe_input_idx=m15_input_idx, cache_hit=cache_hit,
    )


# Use-case routing for the start resolver (Session 3, 2026-05-31; legacy
# parent-TF probe escape hatch retired by Plan C).
_FIRST_CONFLUENCE_USE_CASES = frozenset({"first_confluence"})
_SIBLING_CTS_USE_CASES = frozenset(
    {"first_counter", "subsequent_confluence", "subsequent_counter"}
)


def _resolve_trigger_m15_start(
    trigger: MultiTFTrigger,
    parent_df: pd.DataFrame,
    m15_df: pd.DataFrame,
    *,
    pool: Optional[SubStructurePool] = None,
    hi: Optional[int] = None,
    parent_path: str = "H1.main",
):
    """Resolve a trigger's M15 `starting_idx` (entity-absolute) — dispatcher.

    - `first_confluence` → unified probe vs own ad-hoc BOS_0 (Phase 1+2).
    - `first_counter` / `subsequent_confluence` / `subsequent_counter` →
      unified probe vs the SIBLING lens's most recent qualifying CTS from the
      pool (input + reference co-sourced), Phase 1 only. `hi` (the trigger's
      `trigger_idx`) is required.
    - `reversal` never comes here (`_resolve_reversal_start`).

    Returns a `ResolvedStart` or a `ProbeFailure`.
    """
    uc = trigger.use_case
    if uc in _FIRST_CONFLUENCE_USE_CASES:
        return _resolve_first_confluence_via_unified_probe(
            trigger, parent_df, m15_df, pool=pool, parent_path=parent_path,
        )
    if uc in _SIBLING_CTS_USE_CASES:
        assert pool is not None and hi is not None, (
            f"[entity_compute] {uc} needs pool + hi"
        )
        return _resolve_sibling_cts_via_unified_probe(
            trigger, parent_df, m15_df, pool=pool, hi=int(hi), parent_path=parent_path,
        )
    raise ValueError(f"[entity_compute] unknown use_case {uc!r}")


def _resolve_reversal_start(
    pool: Optional[SubStructurePool],
    sub: PooledStructure,
    reversal_idx: int,
    probe_direction: int,
    m15_df: pd.DataFrame,
    *,
    timeframe: str = "M15",
    parent_path: str = "H1.main",
):
    """The reversal handoff (PART4 §6 / unified-probe §3 reversal row), run on
    the reversing sub's SLICE-LOCAL geometry with every returned idx shifted by
    `slice_begin` — lifted out of the old `build_one_sid`.

    Reference = the reversing structure's (sid=0) most recent {CONF/UPD/EST}
    CTS, read from its own events (`kl_zones=[]`: subs never receive CTS KL
    zones, so the CONFIRMED-zone branch is dead — behaviour-preserving);
    probe bound = the reversal candle (`bounded.reversal_idx`, the canonical
    single-source value). Failure branches (no CTS reference / input ≥
    reversal / pending) → `ProbeFailure`.
    """
    from engine_v2.structure.reference_zone import build_reference_zone_from_cts_event

    label = f"reversal sub_id={sub.sub_id} R={reversal_idx}"
    assert sub.geometry is not None, f"[entity_compute] {label}: no geometry"
    bounded, slice_begin = sub.geometry
    slice_begin = int(slice_begin)
    reversal_end_local = int(reversal_idx) - slice_begin
    assert bounded.reversal_idx is not None and int(bounded.reversal_idx) == reversal_end_local, (
        f"[entity_compute] {label}: geometry reversal {bounded.reversal_idx} != R-slice_begin "
        f"{reversal_end_local}"
    )
    ref_zone = build_reference_zone_from_cts_event(
        bounded.events, [], bounded.df, sid=0,
        probe_direction=int(probe_direction), idx_window=None,
    )
    if ref_zone is None:
        print(
            f"[entity_compute] WARNING: reversal reference zone unavailable for {label} — "
            f"no CTS event for the reversing structure; no successor"
        )
        return ProbeFailure("reversal reference zone unavailable (no CTS event)", None)
    probe_input_local = int(ref_zone.anchor_idx)
    if probe_input_local >= reversal_end_local:
        print(
            f"[entity_compute] WARNING: degenerate reversal probe window for {label} "
            f"(input={probe_input_local + slice_begin} end={reversal_idx}) — no successor"
        )
        return ProbeFailure(
            f"degenerate reversal probe window ({probe_input_local + slice_begin} >= {reversal_idx})",
            probe_input_local + slice_begin,
        )
    # The cache key must be entity-absolute (the pool is shared across subs);
    # the probe itself runs slice-local on `bounded.df`. Wrap so the cache sees
    # absolute idxs while the run sees local ones.
    input_abs = probe_input_local + slice_begin
    if pool is not None:
        hit = pool.get_cached_probe(parent_path, timeframe, int(probe_direction), input_abs)
        if hit is not None:
            kind = "hit" if hit.probe_end_idx == int(reversal_idx) else "APPROX hit"
            print(
                f"[probe_cache] {kind} {label} key=({probe_direction},{input_abs}) "
                f"probe_end_idx={reversal_idx} cached_end={hit.probe_end_idx} "
                f"-> starting_idx={hit.starting_idx} finalize={hit.finalize_idx}"
            )
            cached_ref = hit.ref_inner if hit.ref_inner is not None else hit.bos0_inner
            if cached_ref is None or not math.isclose(float(ref_zone.inner), float(cached_ref), abs_tol=1e-9):
                print(
                    f"[probe_cache] REF-ZONE DIFFERS {label} key=({probe_direction},{input_abs}): "
                    f"ref_inner={float(ref_zone.inner)} cached_ref_inner={cached_ref}"
                )
            return ResolvedStart(
                starting_idx=int(hit.starting_idx), validated_parent_idx=None,
                bos0_inner=hit.bos0_inner, finalize_idx=int(hit.finalize_idx),
                finalize_condition=str(hit.finalize_condition),
                probe_input_idx=input_abs, cache_hit=True,
            )
        print(f"[probe_cache] miss {label} key=({probe_direction},{input_abs}) probe_end_idx={reversal_idx}")

    from engine_v2.structure.unified_probe import unified_probe
    rev_probe = unified_probe(
        bounded.df,
        input_idx=probe_input_local,
        direction=int(probe_direction),
        reference_zone=ref_zone,
        probe_end_idx=reversal_end_local,
        timeframe=timeframe,
        enable_phase2=False,
    )
    print(
        f"[entity_compute] unified_probe ({label}): input={input_abs} "
        f"probe_end_idx={reversal_idx} ref={ref_zone.source} -> "
        f"starting_idx={int(rev_probe.starting_idx) + slice_begin} status={rev_probe.status} "
        f"cond={rev_probe.finalize_condition} iter={rev_probe.iterations}"
    )
    if rev_probe.status == "pending":
        print(f"[entity_compute] PENDING: reversal unified_probe did not finalize for {label}; no successor")
        return ProbeFailure("reversal probe pending", input_abs)
    assert rev_probe.finalize_idx is not None
    starting_abs = int(rev_probe.starting_idx) + slice_begin
    finalize_abs = int(rev_probe.finalize_idx) + slice_begin
    if pool is not None:
        pool.record_probe(
            parent_path, timeframe, int(probe_direction), input_abs,
            ProbeCacheEntry(
                starting_idx=starting_abs, finalize_idx=finalize_abs,
                finalize_condition=str(rev_probe.finalize_condition),
                bos0_inner=rev_probe.bos0_inner, probe_end_idx=int(reversal_idx),
                ref_inner=float(ref_zone.inner),
            ),
        )
    return ResolvedStart(
        starting_idx=starting_abs, validated_parent_idx=None,
        bos0_inner=rev_probe.bos0_inner, finalize_idx=finalize_abs,
        finalize_condition=str(rev_probe.finalize_condition),
        probe_input_idx=input_abs, cache_hit=False,
    )

def _map_parent_idx_to_m15_hour_end(
    parent_idx: int,
    parent_df: pd.DataFrame,
    m15_df: pd.DataFrame,
) -> Optional[int]:
    """Map a parent-TF candle idx to the LAST M15 candle of that parent hour.

    LOH — the mapper for EVERY timing / lifecycle value (§17.4): a trigger's
    `trigger_idx`, the parent floors and ends (`parent_tables`), the WVMI
    trigger candle. Never used for a structural (price) anchor — that is
    `data_bridge.map_candle_to_lower_tf`; do not unify them. Assumes the
    parent is H1 (the only configured parent today).
    """
    if parent_idx not in parent_df.index:
        return None
    p_time = pd.to_datetime(parent_df.loc[parent_idx, "time"], utc=True)
    p_hour_end = p_time + timedelta(hours=1)
    m15_times = pd.to_datetime(m15_df["time"], utc=True)
    mask = (m15_times >= p_time) & (m15_times < p_hour_end)
    cands = m15_df[mask]
    if cands.empty:
        before = m15_times < p_time
        if before.any():
            return int(m15_df[before].index[-1])
        return None
    return int(cands.index[-1])


def _synth_reversal_trigger(
    source_trigger: MultiTFTrigger,
    sd: int,
    reversal_apply_idx: int,
) -> MultiTFTrigger:
    """Build a minimal MultiTFTrigger for a reversal-born record (§17.5).

    A reversal-born record has no parent probe — its start came from the
    reversing sub's own handoff. We still need a well-formed ``trigger`` so
    the ``LowerTFResult`` / mirror / WVMI paths stay uniform. Inherits parent
    linkage + TF from the SPAWNING RECORD's own source trigger (not the
    cycle's bootstrap); ``use_case="reversal"``; ``lower_sd`` is the
    reversal-flipped direction.
    """
    return replace(
        source_trigger,
        use_case="reversal",
        lower_sd=sd,
        meta={**source_trigger.meta, "reversal_apply_idx": int(reversal_apply_idx)},
    )



def _build_geometry(
    m15_df: pd.DataFrame,
    *,
    sd: int,
    start_abs: int,
    run_cap_abs: int,
    bos0_inner: Optional[float],
    timeframe: str,
) -> Optional[Tuple[Any, int]]:
    """Pool-free core of the geometry build: slice `[start-50, run_cap]`,
    preprocess (compute_imbalance + drop inherited structure cols + re-derive
    is_range labels — LANDMINE guards), run ONE `compute_bounded_structure`
    bounded to `run_cap` (the sub's NATURAL end — it stops at its first
    reversal regardless). Returns `(bounded, slice_begin)` or None on a
    degenerate / failed run. `bounded.events` / `bounded.df` are SLICE-LOCAL.
    """
    n = len(m15_df)
    if start_abs >= n or start_abs >= run_cap_abs:
        return None

    lookback = 50
    slice_begin = max(0, int(start_abs) - lookback)
    trigger_df = m15_df.iloc[slice_begin:int(run_cap_abs) + 1].copy()
    trigger_df = trigger_df.reset_index(drop=True)

    from engine_v2.patterns.imbalance import compute_imbalance
    trigger_df = compute_imbalance(trigger_df)
    cols_to_drop = [
        c for c in (_STRUCTURE_COLS + _MS_AUX_STRUCTURE_COLS)
        if c in trigger_df.columns
    ]
    if cols_to_drop:
        trigger_df = trigger_df.drop(columns=cols_to_drop, errors="ignore")

    # Re-derive range labels on the slice (LANDMINE "Sub Slices Must Re-Derive
    # is_range_* Labels After reset_index").
    from engine_v2.patterns.range_label import apply_is_range_labels, RangeLabelConfig
    trigger_df = trigger_df.drop(
        columns=["is_range", "is_range_confirm_idx", "is_range_lag"],
        errors="ignore",
    )
    trigger_df = apply_is_range_labels(trigger_df, RangeLabelConfig())

    start_in_slice = int(start_abs) - slice_begin
    run_cap_in_slice = int(run_cap_abs) - slice_begin
    if len(trigger_df) - start_in_slice < 5:
        return None

    from engine_v2.structure.structure_engine import compute_bounded_structure
    try:
        bounded = compute_bounded_structure(
            trigger_df,
            start_idx=start_in_slice,
            struct_direction=int(sd),
            end_idx=run_cap_in_slice,
            timeframe=timeframe,
            enforce_cts0_new_extreme=(bos0_inner is not None),
            bos0_inner=bos0_inner,
        )
    except (ValueError, IndexError):
        return None

    bounded.df.attrs["imbalances"] = trigger_df.attrs.get("imbalances", [])
    return bounded, slice_begin


def build_or_get_geometry(
    pool: SubStructurePool,
    m15_df: pd.DataFrame,
    *,
    parent_path: str,
    sd: int,
    start_abs: int,
    bos0_inner: Optional[float],
    timeframe: str = "M15",
) -> Optional[Tuple[PooledStructure, bool]]:
    """Build — or reuse from the pool — the natural-end geometry for a unique sub
    (§17.8): the SINGLE owner of pool lookup + MS run + `natural_reversal_idx`
    + `bos0_inner`.

    Run cap = the DATA EDGE (`len(m15_df) - 1`) for every sub — a compute bound
    only (a record cannot outlive its parent structure; a unique sub can). On a
    miss it runs MS FIRST and creates the pool entry only on success (a failed
    build → None, no `sub_id` consumed); on a hit it returns the existing sub.
    `bos0_inner` is not in the key: the sweep WARNING-logs a mismatch.

    Returns `(sub, created)` or None. The attached geometry is SHARED across
    records — callers must not mutate `bounded.events` / `bounded.df` in
    place (the projection deepcopies the events it stamps).
    """
    key = StructureKey(parent_path, timeframe, int(sd), int(start_abs))
    cached = pool.get(key)
    if cached is not None and cached.geometry is not None:
        return cached, False
    assert cached is None, f"[entity_compute] pool entry {tuple(key)} without geometry"

    geom = _build_geometry(
        m15_df, sd=int(sd), start_abs=int(start_abs), run_cap_abs=len(m15_df) - 1,
        bos0_inner=bos0_inner, timeframe=timeframe,
    )
    if geom is None:
        print(
            f"[entity_compute] WARNING: geometry unavailable for key={tuple(key)} "
            f"n={len(m15_df)} — no pool entry"
        )
        return None
    bounded, slice_begin = geom
    sub, created = pool.get_or_create(key)
    assert created
    sub.geometry = (bounded, slice_begin)
    sub.natural_reversal_idx = (
        int(bounded.reversal_idx) + slice_begin if bounded.reversal_idx is not None else None
    )
    sub.bos0_inner = bos0_inner
    print(
        f"[entity_compute] geometry built sub_id={sub.sub_id} key={tuple(key)} "
        f"slice_begin={slice_begin} run_cap={len(m15_df) - 1} "
        f"natural_reversal_idx={sub.natural_reversal_idx} events={len(bounded.events)}"
    )
    return sub, True


def render_sub_projection(
    sub: PooledStructure,
    m15_df: pd.DataFrame,
    *,
    lens_paths: Dict[str, str],
    lens_dfs: Dict[str, pd.DataFrame],
    timeframe: str = "M15",
) -> LowerTFResult:
    """§17.9: ONE projection per unique sub over the sub's `[start_idx,
    end_idx]`, mirrored into every lens df in `sub.lenses()`.

    `project_to_window` clips the shared natural-end geometry by knowable-at
    at the sub's cap and derives the downstream elements with the sub's
    `lifecycle_floor` / `lifecycle_cap` / `cap_reason`. The returned
    `LowerTFResult.trigger` is the sub's FIRST live record's source trigger
    (its `use_case` / parent fields are informational only — the mirror,
    the WVMI persister and `build_sid_records_for_subordinate` read them).
    """
    from engine_v2.multitf.pooled_structure_build import project_to_window

    assert sub.start_idx is not None, f"[entity_compute] sub {sub.sub_id} has no start_idx"
    assert sub.geometry is not None, f"[entity_compute] sub {sub.sub_id} has no geometry"
    bounded, slice_begin = sub.geometry
    slice_begin = int(slice_begin)
    n_slice = len(bounded.df)
    edge_abs = slice_begin + n_slice - 1

    floor_local = int(sub.start_idx) - slice_begin
    cap_local = (int(sub.end_idx) - slice_begin) if sub.end_idx is not None else None
    assert cap_local is None or cap_local <= n_slice - 1, (
        f"[entity_compute] sub {sub.sub_id} cap {sub.end_idx} past its geometry edge {edge_abs}"
    )
    live = sub.live_records()
    first = min(live, key=lambda r: (r.start_idx, r.seq))
    lenses = sorted(sub.lenses())
    # One canonical structure_path_id for the projection's attribution: the
    # first live record's lens (the mirror stamps each lens's own path).
    first_path = lens_paths[first.lens]

    down = project_to_window(
        bounded,
        floor=floor_local,
        cap=cap_local,
        cap_reason=sub.end_reason,
        direction=int(sub.direction),
        timeframe=timeframe,
        log_prefix=f"M15_sub{sub.sub_id}_{first.trigger_type}",
        structure_path_id=first_path,
    )
    end_in_slice = cap_local if cap_local is not None else n_slice - 1
    first_record = {
        "lens": first.lens,
        "parent_sid": int(first.parent_sid),
        "parent_cycle_id": int(first.parent_cycle_id),
        "trigger_type": first.trigger_type,
        "trigger_idx": int(first.trigger_idx),
        "start_idx": int(first.start_idx),
    }
    result = LowerTFResult(
        trigger=first.source_trigger,
        df=bounded.df.iloc[0:end_in_slice + 1],
        events=down["events"],
        kl_zones=down["kl_zones"],
        wave_candles=down["wave_candles"],
        fib_states=down["fib_states"],
        poi_zones=down["poi_zones"],
        wvmi_records=down["wvmi_records"],   # empty (skip_wvmi=True)
        prev_bos_lines=down["prev_bos_lines"],
        status="finalized",
        meta={
            "sub_id": int(sub.sub_id),
            "m15_start_idx": int(sub.starting_idx),
            "start_idx": int(sub.start_idx),
            "end_idx": (int(sub.end_idx) if sub.end_idx is not None else None),
            "m15_end_idx": (int(sub.end_idx) if sub.end_idx is not None else int(edge_abs)),
            "end_reason": sub.end_reason,
            "natural_reversal_idx": sub.natural_reversal_idx,
            "slice_begin": slice_begin,
            "lenses": tuple(lenses),
            "relative_dir_segments": tuple(sub.relative_dir_segments),
            "n_records": len(sub.records),
            "first_record": first_record,
            "timeframe": timeframe,
            "use_case": first.trigger_type,
            "started_by": first.trigger_type,
            "parent_tf": getattr(first.source_trigger, "parent_tf", "H1"),
            "parent_sid": int(first.parent_sid),
            "parent_cycle_id": int(first.parent_cycle_id),
        },
    )
    for lens in lenses:
        mirror_lower_tf_result_to_entity_df(
            lens_dfs[lens], result, structure_path_id=lens_paths[lens],
        )
    print(
        f"[entity_compute] projection sub_id={sub.sub_id} key=({sub.direction},{sub.starting_idx}) "
        f"window=[{sub.start_idx},{sub.end_idx}] reason={sub.end_reason} lenses={lenses} "
        f"events={len(result.events)} kl={len(result.kl_zones)} poi={len(result.poi_zones)} "
        f"fib={len(result.fib_states)}"
    )
    return result
