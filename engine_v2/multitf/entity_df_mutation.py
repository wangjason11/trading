"""Entity-df mutation primitives for the subordinate sub-build.

**Phase 2 (merge-and-bound, REVISED 2026-05-25).** The per-trigger
`apply_trigger_to_entity_df` + cascade model (old §13.5.c) is replaced by a
per-parent-cycle sequential sid chain (spec §6.1):

  - `build_parent_cycle_chain` — the driver. Per `(entity, parent_sid,
    parent_cycle_id)`: resolve the bootstrap (`first_*`) start, resolve the
    cycle's `subsequent_*` triggers, then stitch a sequential, NON-overlapping
    chain of bounded single-structure sids. Each sid runs to the first of
    {its own reversal, the next subsequent trigger, parent-cycle-end}; that
    boundary starts sid+1 (subsequent → probe start; reversal → identify_start
    flipped). `sid` resets per parent cycle; `entity_sid` is the entity-wide
    running rank.

  - `build_one_sid` — builds ONE bounded sid: slice + 50-lookback +
    `reset_index` + re-`compute_imbalance`, `compute_bounded_structure`
    (single structure, stops at first reversal — NOT `compute_structure_from_start`),
    downstream pipeline, lifecycle-cap, mirror. Returns the reversal handoff
    (`identify_start_scenario_2`) for the next sid.

  - `mirror_lower_tf_result_to_entity_df` — translates the slice-shape
    `LowerTFResult` into entity-absolute snapshots and appends them to
    `entity_df.attrs["events" / "kl_zones" / "poi_zones" / "fib_states" /
    "wave_candles" / "wvmi" / "prev_bos_lines"]`, stamping the canonical
    identity (`(parent_sid, parent_cycle_id, sid)` + `started_by` +
    `start_trigger_idx`) alongside `entity_sid`.

Persistence model (spec §7): sids are sequential & non-overlapping, so each
candle is written once by its owning sid — no cross-sid overwrite. Zone/POI/
fib/WVMI snapshots persist append-only keyed by sid + cycle_id; open ones cap
at their sid's resolved end (`reversal` or `lifecycle_end`). (The old
cascade-overwrite model + `_tag_old_sid_on_overwrite` helper were deleted in
redesign Phase 4 — sids no longer overlap, so nothing to cascade.)

NOTE: the slice + lookback + `reset_index` machinery (and the mirror's
slice-local → entity-absolute translation) is retained — direct entity-df MS
compute needs an MS refactor that is deferred (Phase 4 / later). The
`entity_sid` → tuple-identity chart migration is also a deferred follow-up;
`entity_sid` remains the chart's display key for now.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from engine_v2.multitf.types import LowerTFResult, MultiTFTrigger


@dataclass
class SidBuildOutcome:
    """One sid built by the merge-and-bound chain (Part 4 §6.1, 2026-05-25).

    `result` is the slice-shape ``LowerTFResult`` already mirrored into
    ``entity_df.attrs``. `reversal_idx_abs` is the sid's first internal
    reversal (entity-absolute apply idx) or None if it bounded out without
    reversing. When a reversal occurred, `next_start_abs` / `next_sd` carry
    the reversal-scenario handoff (``identify_start_scenario_2``) for the
    next sid, in entity-absolute coords; both None otherwise.
    """
    result: LowerTFResult
    reversal_idx_abs: Optional[int]
    next_start_abs: Optional[int]
    next_sd: Optional[int]


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
    "pending_reversal_anchor_idx",
    "pending_reversal_apply_idx",
    "struct_direction",
    "last_breakout_pat_apply_idx",
)

# Event-meta keys whose values are entity-df indices and need translating
# from slice-local to entity-absolute.
_EVENT_META_IDX_KEYS = (
    "confirmed_at",
    "apply_idx",
    "anchor_idx",
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


def mirror_lower_tf_result_to_entity_df(
    entity_df: pd.DataFrame,
    result: LowerTFResult,
    *,
    new_sid_id: int,
    structure_path_id: str,
) -> None:
    """Mirror a slice-shape `LowerTFResult` into `entity_df.attrs[...]`.

    Translates every event / zone / POI / fib / wave-candle / WVMI idx
    field from slice-local (the result is built on a sliced + reset_index
    df) to entity-absolute (slice_begin + offset). Adds an `entity_sid`
    attribution alongside existing `parent_sid` / `parent_cycle_id`
    attribution.

    Also mirrors the structure columns from `result.df` back to
    `entity_df.iloc[slice_begin:slice_begin+len(result.df)]` so the
    entity df reflects the per-candle "current truth" view of this sid.
    """
    slice_begin = int(result.meta.get("slice_begin", 0))

    attribution = {
        "entity_sid": new_sid_id,
        "structure_path_id": structure_path_id,
        "use_case": result.trigger.use_case,
        "parent_sid": result.trigger.parent_sid,
        "parent_cycle_id": result.trigger.parent_cycle_id,
        "timeframe": result.trigger.lower_tf,
    }
    # Phase 2 (§2 / §6.1 REVISED 2026-05-25): canonical per-parent-cycle
    # identity. `sid` resets per parent cycle; `entity_sid` above is now an
    # entity-wide time-order rank (chart display key), no longer the
    # identity. `started_by` ∈ {first_confluence, subsequent_confluence,
    # first_counter, subsequent_counter, reversal}. These travel on
    # result.meta from build_one_sid; absent on legacy/test results.
    for _k in ("sid", "started_by", "start_trigger_idx"):
        if _k in result.meta:
            attribution[_k] = result.meta[_k]

    # 2. Mirror structure columns
    n_slice_rows = len(result.df)
    n_entity = len(entity_df)
    end_in_entity = min(slice_begin + n_slice_rows - 1, n_entity - 1)
    rows_to_write = end_in_entity - slice_begin + 1
    if rows_to_write > 0:
        slice_df = result.df.iloc[0:rows_to_write]
        for col in _STRUCTURE_COLS:
            if col not in slice_df.columns:
                continue
            if col not in entity_df.columns:
                # Initialize with appropriate fill — match dtype where possible
                entity_df[col] = pd.NA
            col_loc = entity_df.columns.get_loc(col)
            entity_df.iloc[slice_begin:end_in_entity + 1, col_loc] = (
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

    # 6. Fib states — direct bos_idx / cts_idx + meta
    new_fibs = []
    for fib in result.fib_states:
        new_meta = _shift_meta_indices(fib.meta, ("deactivated_at",), slice_begin)
        new_meta.update(attribution)
        new_fibs.append(replace(
            fib,
            bos_idx=fib.bos_idx + slice_begin,
            cts_idx=fib.cts_idx + slice_begin,
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
    # end_idx. Translate both, embed entity_sid + parent attribution
    # under "meta" so the chart can group by entity_sid (§13.5.c.iii).
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
    new_sid_id: int,
    structure_path_id: str,
) -> None:
    """Translate slice-local WVMI records on a facade to entity-absolute
    idx and append to `entity_df.attrs["wvmi"]`.

    Sub WVMI is parent-event-driven (§8.3 / §8.4), computed by the
    orchestrator AFTER `apply_trigger_to_entity_df` returns the facade.
    The orchestrator then calls this helper to keep the entity-df
    persistence model in sync.

    `triggered_by_event_idx` in record meta is parent-df coords (LANDMINE
    "WVMI Records Carry Mixed-Coordinate Meta") — do NOT translate.
    """
    slice_begin = int(facade.meta.get("slice_begin", 0))
    attribution = {
        "entity_sid": new_sid_id,
        "structure_path_id": structure_path_id,
        "use_case": facade.trigger.use_case,
        "parent_sid": facade.trigger.parent_sid,
        "parent_cycle_id": facade.trigger.parent_cycle_id,
        "timeframe": facade.trigger.lower_tf,
        "parent_tf": facade.trigger.parent_tf,
    }
    for _k in ("sid", "started_by", "start_trigger_idx"):
        if _k in facade.meta:
            attribution[_k] = facade.meta[_k]
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


def _sub_path_id_for_use_case(use_case: str, fallback: str) -> str:
    """Resolve the structure_path_id from a use_case (counter vs confluence)."""
    if use_case in ("first_counter", "subsequent_counter"):
        return "H1.main >> M15.counter"
    if use_case in ("first_confluence", "subsequent_confluence"):
        return "H1.main >> M15.confluence"
    # "reversal" sids inherit the entity's path from the bootstrap fallback.
    return fallback


def _resolve_trigger_m15_start(
    trigger: MultiTFTrigger,
    parent_df: pd.DataFrame,
    entity_df: pd.DataFrame,
) -> Tuple[Optional[int], Optional[int]]:
    """Run the parent-TF probe and map the validated start to entity-absolute
    M15 idx.

    Returns ``(m15_start_idx, validated_parent_idx)`` — either may be None on
    a pending/failed probe or a failed parent→M15 mapping. Pulled out of the
    old ``apply_trigger_to_entity_df`` steps 1–2 so the chain driver can reuse
    it for every trigger-born sid (bootstrap + subsequent_*). Reversal-born
    sids do NOT use this — their start comes from ``identify_start`` on the
    sub's own data inside ``build_one_sid``.
    """
    from engine_v2.multitf.data_bridge import map_candle_to_lower_tf
    from engine_v2.multitf.lower_tf_pipeline import _run_subordinate_probe

    validated_parent_idx = _run_subordinate_probe(trigger, parent_df)
    if validated_parent_idx is None:
        return None, None

    parent_start_time = pd.to_datetime(
        parent_df.loc[validated_parent_idx, "time"], utc=True,
    )
    mapping_sd = -trigger.lower_sd  # §4.3.1 unified rule
    if mapping_sd == 1:
        parent_start_price = float(parent_df.loc[validated_parent_idx, "h"])
    else:
        parent_start_price = float(parent_df.loc[validated_parent_idx, "l"])

    m15_start_idx = map_candle_to_lower_tf(
        parent_start_time, parent_start_price, mapping_sd, entity_df,
    )
    if m15_start_idx is None:
        print(
            f"[entity_compute] WARNING: parent→lower-TF mapping failed for "
            f"{trigger.use_case} sid={trigger.parent_sid} "
            f"cycle={trigger.parent_cycle_id}"
        )
        return None, validated_parent_idx
    return m15_start_idx, validated_parent_idx


def _map_parent_idx_to_m15_hour_end(
    parent_idx: int,
    parent_df: pd.DataFrame,
    m15_df: pd.DataFrame,
) -> Optional[int]:
    """Map a parent-TF candle idx to the LAST M15 candle of that parent hour.

    Used to translate a ``subsequent_*`` trigger's ``trigger_event_idx``
    (the parent CTS-prox / sd-prox candle — the sub's lifecycle-start, §2)
    into the M15 boundary that ends the prior sid's bounded run (§6.1).
    Same H1-hour windowing as ``_find_m15_lifecycle_end``; assumes the parent
    is H1 (the only configured parent today).
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
    bootstrap: MultiTFTrigger,
    sd: int,
    reversal_apply_idx: int,
) -> MultiTFTrigger:
    """Build a minimal MultiTFTrigger for a reversal-born sid (§6.3).

    A reversal-born sid has no parent probe — its start came from the sub's
    own ``identify_start``. We still need a well-formed ``trigger`` so the
    ``LowerTFResult`` / ``build_sid_records_for_subordinate`` / chart paths
    stay uniform. Inherits parent linkage + TF from the cycle's bootstrap;
    ``use_case="reversal"`` (a value the §8.3/§8.4 WVMI gate intentionally
    ignores — sub WVMI for reversal sids is a Phase 5 concern); ``lower_sd``
    is the reversal-flipped direction. ``start_time``/``start_price`` are
    inherited-but-unused (no mapping happens for reversal sids).
    """
    return replace(
        bootstrap,
        use_case="reversal",
        lower_sd=sd,
        meta={**bootstrap.meta, "reversal_apply_idx": int(reversal_apply_idx)},
    )


def build_one_sid(
    entity_df: pd.DataFrame,
    *,
    start_m15_abs: int,
    sd: int,
    end_m15_abs: int,
    sub_path_id: str,
    timeframe: str,
    trigger: MultiTFTrigger,
    entity_sid: int,
    sid: int,
    started_by: str,
    start_trigger_idx: int,
    validated_parent_idx: Optional[int] = None,
) -> Optional[SidBuildOutcome]:
    """Build ONE bounded single-structure sub sid and mirror it (Part 4 §5/§6.1).

    Replaces the old per-trigger ``apply_trigger_to_entity_df`` tail. The
    probe + parent→M15 mapping moved up into the chain driver; this takes an
    explicit ``[start_m15_abs, end_m15_abs]`` window + direction + identity
    and runs **`compute_bounded_structure`** (a SINGLE directional structure
    that stops at its first reversal) — NOT ``compute_structure_from_start``
    (which rolls past reversals). Steps:

    1. Slice ``[start - 50 lookback, end]``, ``reset_index``, re-compute
       imbalances, drop inherited structure cols (LANDMINE "Slice Copies
       Inherit Mirrored Structure Cols").
    2. ``compute_bounded_structure`` bounded to the window.
    3. Downstream pipeline (KL BOS-only, Fib cross_cycle, POI; WVMI deferred).
    4. If it reversed → ``identify_start_scenario_2`` for the next sid's start
       (entity-absolute), returned on the outcome.
    5. Lifecycle-cap open/late zones/POIs/fibs at the sid's effective end
       (its reversal if any, else the bound).
    6. Mirror into ``entity_df.attrs`` with the canonical identity
       (sids are sequential & non-overlapping, so there is no cascade).

    Returns a ``SidBuildOutcome`` or None on a degenerate window / structure
    failure.
    """
    from dataclasses import replace as _replace

    from engine_v2.patterns.imbalance import compute_imbalance
    from engine_v2.pipeline.orchestrator import _run_downstream_pipeline
    from engine_v2.structure.identify_start import (
        identify_start_scenario_2_after_reversal,
    )
    from engine_v2.structure.structure_engine import compute_bounded_structure

    n = len(entity_df)
    # start_idx >= n guard (LANDMINE "MarketStructure.run() Crashes When
    # start_idx >= n") + degenerate-window guard.
    if start_m15_abs >= n or start_m15_abs >= end_m15_abs:
        print(
            f"[entity_compute] WARNING: degenerate window for {started_by} "
            f"sid={trigger.parent_sid}.{trigger.parent_cycle_id}.{sid}: "
            f"start={start_m15_abs} end={end_m15_abs} n={n}"
        )
        return None

    lookback = 50
    slice_begin = max(0, start_m15_abs - lookback)
    trigger_df = entity_df.iloc[slice_begin:end_m15_abs + 1].copy()
    trigger_df = trigger_df.reset_index(drop=True)
    trigger_df = compute_imbalance(trigger_df)
    cols_to_drop = [
        c for c in (_STRUCTURE_COLS + _MS_AUX_STRUCTURE_COLS)
        if c in trigger_df.columns
    ]
    if cols_to_drop:
        trigger_df = trigger_df.drop(columns=cols_to_drop, errors="ignore")

    start_in_slice = start_m15_abs - slice_begin
    end_in_slice = end_m15_abs - slice_begin
    if len(trigger_df) - start_in_slice < 5:
        print(
            f"[entity_compute] WARNING: M15 slice too small "
            f"({len(trigger_df) - start_in_slice} candles after start) "
            f"for {started_by} sid={trigger.parent_sid}."
            f"{trigger.parent_cycle_id}.{sid}"
        )
        return None

    try:
        bounded = compute_bounded_structure(
            trigger_df,
            start_idx=start_in_slice,
            struct_direction=sd,
            end_idx=end_in_slice,
            timeframe=timeframe,
        )
    except (ValueError, IndexError) as exc:
        print(
            f"[entity_compute] WARNING: bounded structure failed for "
            f"{started_by} sid={trigger.parent_sid}."
            f"{trigger.parent_cycle_id}.{sid}: {exc}"
        )
        return None

    bounded.df.attrs["imbalances"] = trigger_df.attrs.get("imbalances", [])

    # Sub structure lifecycle-start (slice-local): the trigger idx. A sub's
    # structural anchor (start_m15_abs) can sit historically before its trigger
    # (subsequent / reversal sids), so floor zone/POI activation at the trigger
    # — no sub zone may be active before the sub structure is alive (Phase 3
    # Commit 2). For the bootstrap, start_trigger_idx == start_m15_abs, so the
    # floor equals the structural start (a no-op clamp).
    sub_lifecycle_floor_local = int(start_trigger_idx) - slice_begin

    downstream = _run_downstream_pipeline(
        bounded.df,
        bounded.events,
        bounded.struct_direction,
        source_kinds=["BOS"],
        fib_mode="cross_cycle",
        log_prefix=f"M15_sid{entity_sid}_{started_by}",
        timeframe=timeframe,
        structure_path_id=sub_path_id,
        skip_wvmi=True,
        lifecycle_floor=sub_lifecycle_floor_local,
    )

    # Reversal handoff (slice-local → entity-absolute). The bounded run only
    # ever reverses inside the window (end_idx bounds it), so reversal_idx,
    # when set, is < end_m15_abs.
    reversal_idx_abs: Optional[int] = None
    next_start_abs: Optional[int] = None
    next_sd: Optional[int] = None
    if bounded.reversal_idx is not None:
        reversal_idx_abs = bounded.reversal_idx + slice_begin
        d_next = identify_start_scenario_2_after_reversal(
            bounded.df,
            reversal_idx=bounded.reversal_idx,
            prev_structure_id=0,
            prev_struct_direction=sd,
            min_history=50,
        )
        next_start_abs = int(d_next.start_idx) + slice_begin
        next_sd = int(d_next.struct_direction)

    # Effective end = the reversal (the sid's real boundary) if it reversed,
    # else the window bound. Open/late artifacts cap here so a sid that
    # reversed mid-window doesn't extend zones into the next sid's territory.
    effective_end_abs = (
        reversal_idx_abs if reversal_idx_abs is not None else end_m15_abs
    )
    end_reason = "reversal" if reversal_idx_abs is not None else "lifecycle_end"
    cap_time = pd.to_datetime(
        entity_df.loc[effective_end_abs, "time"], utc=True,
    )
    cap_idx_local = effective_end_abs - slice_begin

    attribution: Dict[str, Any] = {
        "timeframe": timeframe,
        "use_case": trigger.use_case,
        "parent_tf": trigger.parent_tf,
        "parent_sid": trigger.parent_sid,
        "parent_cycle_id": trigger.parent_cycle_id,
    }
    for ev in bounded.events:
        ev.meta.update(attribution)
    for zone in downstream["kl_zones"]:
        zone.meta.update(attribution)

    # KL cap = the structure-end → open-cycle-end → zone-end pass-through for
    # subs (Phase 3 convention). Open/late KL zones inherit the sub's effective
    # end (reversal or parent lifecycle_end). cap_idx_local is slice-local; the
    # mirror translates end_idx to entity-absolute alongside confirmed_idx.
    capped_zones = []
    for zone in downstream["kl_zones"]:
        if zone.end_time is None or zone.end_time > cap_time:
            ah = zone.meta.get("activation_history") or []
            zone = _replace(
                zone,
                end_time=cap_time,
                meta={**zone.meta,
                      "end_idx": cap_idx_local,
                      "end_reason": end_reason,
                      "status": "ended" if ah else "inactive"},
            )
        capped_zones.append(zone)

    capped_pois = []
    for poi in downstream["poi_zones"]:
        if poi.end_time is None or poi.end_time > cap_time:
            poi = _replace(
                poi,
                end_time=cap_time,
                meta={**poi.meta, "active": False,
                      "deactivated_by": end_reason},
            )
        capped_pois.append(poi)

    capped_fibs = []
    for fib in downstream["fib_states"]:
        if fib.active and not fib.locked:
            fib = _replace(
                fib,
                active=False,
                meta={**fib.meta, "deactivated_by": end_reason,
                      "deactivated_at": cap_idx_local},
            )
        capped_fibs.append(fib)

    result = LowerTFResult(
        trigger=trigger,
        df=bounded.df,
        events=bounded.events,
        kl_zones=capped_zones,
        wave_candles=downstream["wave_candles"],
        fib_states=capped_fibs,
        poi_zones=capped_pois,
        wvmi_records=downstream["wvmi_records"],   # empty (skip_wvmi=True)
        prev_bos_lines=downstream["prev_bos_lines"],
        status="finalized",
        meta={
            "m15_start_idx": start_m15_abs,
            "m15_end_idx": effective_end_abs,
            "m15_bound_idx": end_m15_abs,
            "m15_candle_count": len(trigger_df),
            "validated_h1_start": validated_parent_idx,
            "slice_begin": slice_begin,
            # Canonical identity (§2 / §6.1 REVISED 2026-05-25).
            "sid": sid,
            "started_by": started_by,
            "start_trigger_idx": start_trigger_idx,
            "end_reason": end_reason,
            **attribution,
        },
    )

    # Mirror into entity_df.attrs with entity-absolute idx. Sids are
    # sequential & non-overlapping (§6.1), so each candle is written once by
    # its owning sid — no cross-sid cascade.
    mirror_lower_tf_result_to_entity_df(
        entity_df,
        result,
        new_sid_id=entity_sid,
        structure_path_id=sub_path_id,
    )

    print(
        f"[entity_compute] {started_by} entity_sid={entity_sid} "
        f"id=({trigger.parent_sid},{trigger.parent_cycle_id},{sid}) "
        f"start={start_m15_abs} end={effective_end_abs} bound={end_m15_abs} "
        f"reversed={reversal_idx_abs is not None} "
        f"events={len(bounded.events)} kl={len(capped_zones)} "
        f"poi={len(capped_pois)} fib={len(capped_fibs)}"
    )

    return SidBuildOutcome(
        result=result,
        reversal_idx_abs=reversal_idx_abs,
        next_start_abs=next_start_abs,
        next_sd=next_sd,
    )


def build_parent_cycle_chain(
    entity_df: pd.DataFrame,
    parent_df: pd.DataFrame,
    *,
    bootstrap: MultiTFTrigger,
    subsequents: List[MultiTFTrigger],
    sub_path_id: str,
    first_entity_sid: int,
    wvmi_hook: Optional[Callable[[LowerTFResult, int], None]] = None,
) -> Tuple[List[LowerTFResult], int]:
    """Build one parent cycle's subordinate sid chain (Part 4 §6.1 merge-and-bound).

    Per ``(entity, parent_sid, parent_cycle_id)``:

    - **sid=0** starts at the bootstrap's probe-validated start (``first_*``
      variation). If that probe is pending / mapping fails → no chain (no
      sid=0 ⇒ subsequents don't build either).
    - Each sid runs a bounded single structure (``build_one_sid``) to the
      first of {its own reversal, the next subsequent trigger, parent-cycle
      end}.
    - That boundary starts sid+1: a subsequent trigger → its probe-validated
      start + use_case direction; a reversal → ``identify_start`` (reversal
      scenario, flipped direction).
    - ``sid`` is per-cycle (resets to 0 here); ``entity_sid`` is the
      entity-wide running rank threaded in via ``first_entity_sid`` and
      returned advanced for the next cycle.

    ``subsequents`` must be this cycle's ``subsequent_*`` triggers
    (already converted to ``MultiTFTrigger``); order doesn't matter (resolved
    + sorted by M15 boundary here). ``wvmi_hook(result, entity_sid)`` — if
    given — is invoked per built sid (the orchestrator computes parent-driven
    sub WVMI there; it self-skips ``use_case="reversal"``).

    Returns ``(results, next_entity_sid)``.
    """
    results: List[LowerTFResult] = []

    # Bootstrap start (entity-absolute).
    m15_start_0, validated_parent_0 = _resolve_trigger_m15_start(
        bootstrap, parent_df, entity_df,
    )
    if m15_start_0 is None:
        print(
            f"[chain] no sid=0 for {bootstrap.use_case} "
            f"sid={bootstrap.parent_sid} cycle={bootstrap.parent_cycle_id} "
            f"(probe pending / mapping failed) — cycle skipped"
        )
        return results, first_entity_sid

    # Parent-cycle end (entity-absolute).
    from engine_v2.multitf.lower_tf_pipeline import _find_m15_lifecycle_end
    cycle_end_m15 = _find_m15_lifecycle_end(bootstrap, entity_df, parent_df)
    if cycle_end_m15 is None:
        cycle_end_m15 = int(entity_df.index[-1])

    # Resolve subsequents up front: keep only those whose probe finalizes
    # (a pending probe ⇒ no sid AND no boundary, so the prior sid runs
    # through — §6.1). boundary = trigger_event_idx mapped to M15 hour-end.
    resolved_subs: List[Dict[str, Any]] = []
    for sub in subsequents:
        s_start, s_valid = _resolve_trigger_m15_start(sub, parent_df, entity_df)
        if s_start is None:
            continue
        tei = sub.meta.get("trigger_event_idx")
        if tei is None:
            continue
        boundary = _map_parent_idx_to_m15_hour_end(int(tei), parent_df, entity_df)
        if boundary is None:
            continue
        resolved_subs.append({
            "trigger": sub, "start": s_start,
            "valid": s_valid, "boundary": boundary,
        })
    resolved_subs.sort(key=lambda x: x["boundary"])
    pending = list(resolved_subs)

    entity_sid = first_entity_sid
    sid = 0
    cur_start = m15_start_0
    cur_sd = int(bootstrap.lower_sd)
    cur_trigger = bootstrap
    cur_started_by = bootstrap.use_case
    cur_start_trig = m15_start_0
    cur_valid = validated_parent_0

    guard = 0
    max_chain = 50  # safety against pathological loops
    while guard < max_chain:
        guard += 1

        next_sub = next((s for s in pending if s["boundary"] > cur_start), None)
        bound = cycle_end_m15
        if next_sub is not None:
            bound = min(bound, next_sub["boundary"])

        outcome = build_one_sid(
            entity_df,
            start_m15_abs=cur_start, sd=cur_sd, end_m15_abs=bound,
            sub_path_id=sub_path_id, timeframe=cur_trigger.lower_tf,
            trigger=cur_trigger, entity_sid=entity_sid, sid=sid,
            started_by=cur_started_by, start_trigger_idx=cur_start_trig,
            validated_parent_idx=cur_valid,
        )
        if outcome is None:
            # Degenerate / failed sid ends the chain (conservative — a rare
            # case on real data; revisit if it drops legitimate later sids).
            break

        results.append(outcome.result)
        if wvmi_hook is not None:
            wvmi_hook(outcome.result, entity_sid)
        entity_sid += 1
        sid += 1

        # Decide the next sid. Reversal (earlier than the next subsequent by
        # construction) wins if its handoff start is usable; else fall to the
        # pending subsequent; else the cycle is done.
        if (outcome.reversal_idx_abs is not None
                and outcome.next_start_abs is not None
                and outcome.next_start_abs < bound
                and outcome.next_start_abs < len(entity_df)):
            cur_start = outcome.next_start_abs
            cur_sd = int(outcome.next_sd)
            cur_trigger = _synth_reversal_trigger(
                bootstrap, cur_sd, outcome.reversal_idx_abs,
            )
            cur_started_by = "reversal"
            cur_start_trig = outcome.reversal_idx_abs
            cur_valid = None
            continue

        if next_sub is not None and bound == next_sub["boundary"]:
            pending.remove(next_sub)
            cur_start = next_sub["start"]
            cur_sd = int(next_sub["trigger"].lower_sd)
            cur_trigger = next_sub["trigger"]
            cur_started_by = next_sub["trigger"].use_case
            cur_start_trig = next_sub["boundary"]
            cur_valid = next_sub["valid"]
            continue

        break

    return results, entity_sid
