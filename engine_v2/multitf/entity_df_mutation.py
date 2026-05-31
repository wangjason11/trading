"""Entity-df mutation primitives for the subordinate sub-build.

**Phase 2 (merge-and-bound, REVISED 2026-05-25).** The per-trigger
`apply_trigger_to_entity_df` + cascade model (old §13.5.c) is replaced by a
per-parent-cycle sequential sid chain (spec §6.1):

  - `build_parent_cycle_chain` — the driver. Per `(entity, parent_sid,
    parent_cycle_id)`: resolve the bootstrap (`first_*`) start, resolve the
    cycle's `subsequent_*` triggers, then stitch a sequential, NON-overlapping
    chain of bounded single-structure sids. Each sid runs to the first of
    {its own reversal, the next subsequent trigger, parent-cycle-end}; that
    boundary starts sub_sid+1 (subsequent → probe start; reversal →
    identify_start flipped). `sub_sid` resets per parent cycle; the sub's
    identity is the tuple `(parent_sid, parent_cycle_id, sub_sid)`.

  - `build_one_sid` — builds ONE bounded sid: slice + 50-lookback +
    `reset_index` + re-`compute_imbalance`, `compute_bounded_structure`
    (single structure, stops at first reversal — NOT `compute_structure_from_start`),
    downstream pipeline, lifecycle-cap, mirror. Returns the reversal handoff
    (`identify_start_scenario_2`) for the next sid.

  - `mirror_lower_tf_result_to_entity_df` — translates the slice-shape
    `LowerTFResult` into entity-absolute snapshots and appends them to
    `entity_df.attrs["events" / "kl_zones" / "poi_zones" / "fib_states" /
    "wave_candles" / "wvmi" / "prev_bos_lines"]`, stamping the canonical
    identity tuple (`(parent_sid, parent_cycle_id, sub_sid)` + `started_by` +
    `start_trigger_idx`) onto each snapshot's meta.

Persistence model (spec §7): sids are sequential & non-overlapping, so each
candle is written once by its owning sid — no cross-sid overwrite. Zone/POI/
fib/WVMI snapshots persist append-only keyed by sid + cycle_id; open ones cap
at their sid's resolved end (`reversal` or `lifecycle_end`). (The old
cascade-overwrite model + `_tag_old_sid_on_overwrite` helper were deleted in
redesign Phase 4 — sids no longer overlap, so nothing to cascade.)

NOTE: the slice + lookback + `reset_index` machinery (and the mirror's
slice-local → entity-absolute translation) is retained — direct entity-df MS
compute needs an MS refactor that is deferred (Phase 4 / later).
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from engine_v2.common.types import KLZone
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
    structure_path_id: str,
) -> None:
    """Mirror a slice-shape `LowerTFResult` into `entity_df.attrs[...]`.

    Translates every event / zone / POI / fib / wave-candle / WVMI idx
    field from slice-local (the result is built on a sliced + reset_index
    df) to entity-absolute (slice_begin + offset). Stamps the canonical
    identity tuple `(parent_sid, parent_cycle_id, sub_sid)` onto every
    snapshot's meta (as separate fields).

    Also mirrors the structure columns from `result.df` back to
    `entity_df.iloc[slice_begin:slice_begin+len(result.df)]` so the
    entity df reflects the per-candle "current truth" view of this sid.
    """
    slice_begin = int(result.meta.get("slice_begin", 0))

    attribution = {
        "structure_path_id": structure_path_id,
        "use_case": result.trigger.use_case,
        "parent_sid": result.trigger.parent_sid,
        "parent_cycle_id": result.trigger.parent_cycle_id,
        "timeframe": result.trigger.lower_tf,
    }
    # Canonical per-parent-cycle identity (§2 / §6.1): the sub is uniquely
    # identified by the tuple `(parent_sid, parent_cycle_id, sub_sid)`.
    # `sub_sid` resets per parent cycle; alone it is meaningless — the tuple
    # is the identity. `started_by` ∈ {first_confluence, subsequent_confluence,
    # first_counter, subsequent_counter, reversal}. These travel on
    # result.meta from build_one_sid; absent on legacy/test results.
    for _k in ("sub_sid", "started_by", "start_trigger_idx"):
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
    # end_idx. Translate both, embed the identity-tuple attribution
    # under "meta" so the chart can group by (parent_sid, parent_cycle_id,
    # sub_sid) (§13.5.c.iii).
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

    Sub WVMI is parent-event-driven (§8.3 / §8.4), computed by the
    orchestrator's trigger-centric pass AFTER the sid chain is built. The
    orchestrator then calls this helper to keep the entity-df persistence
    model in sync.

    Stamps the identity tuple `(parent_sid, parent_cycle_id, sub_sid)` onto
    each record's meta. `triggered_by_event_idx` in record meta is parent-df
    coords (LANDMINE "WVMI Records Carry Mixed-Coordinate Meta") — do NOT
    translate.
    """
    slice_begin = int(facade.meta.get("slice_begin", 0))
    attribution = {
        "structure_path_id": structure_path_id,
        "use_case": facade.trigger.use_case,
        "parent_sid": facade.trigger.parent_sid,
        "parent_cycle_id": facade.trigger.parent_cycle_id,
        "timeframe": facade.trigger.lower_tf,
        "parent_tf": facade.trigger.parent_tf,
    }
    for _k in ("sub_sid", "started_by", "start_trigger_idx"):
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


def _build_first_confluence_ref_zone(
    entity_df: pd.DataFrame,
    m15_input_idx: int,
    probe_direction: int,
) -> Optional["ReferenceZone"]:
    """Ad-hoc BOS_0 reference zone on M15 for first_confluence.

    Anchor = the M15 input_idx candle (= the parent BOS extreme
    price-mapped to M15). Derives a BOS-style zone from that candle via
    the same `identify_base_pattern + zone_thresholds` machinery the
    legacy Scenario 3 used, just on the SUB's TF rather than the parent's.

    Held constant across probe iterations per the unified-probe design.
    """
    from engine_v2.structure.reference_zone import (
        ReferenceZone,
        _derive_zone_ad_hoc,
    )

    # `direction` here is the SOURCE structure's direction (= the sub's
    # own direction for first_confluence, since the BOS_0 belongs to the
    # sub being built). For first_confluence, sub direction == lower_sd ==
    # +parent_sd. Pass lower_sd in: the ad-hoc derivation produces a zone
    # geographic + side keyed off that, then we re-interpret as a probe
    # reference below.
    derived = _derive_zone_ad_hoc(
        entity_df, m15_input_idx, probe_direction, bos=True,
    )
    if derived is None:
        return None
    outer_geo, inner_geo, _side_geo = derived

    # The probe reads `inner` per probe_direction (see ReferenceZone
    # docstring): for +1 probe body is above ref, inner = top of zone;
    # for -1 probe inner = bottom. The ad-hoc helper returns (outer,
    # inner) in "source-structure" orientation matching the sd=+1 ⇒
    # zone-above-body convention — for our case that's INVERTED to the
    # probe's geometry because the source structure direction == probe
    # direction and the legacy outer/inner convention put outer on the
    # source direction side (= AWAY from body). For probe direction == +1
    # the body grows up from the input candle and the zone (BOS_0)
    # surrounds the input candle from BELOW — so inner = TOP of the zone
    # (= ad-hoc's "inner" if we asked for the source structure direction).
    # The ad-hoc helper's return is (outer, inner) where for sd=+1 outer
    # = base_high and inner = some lower threshold (per zone_thresholds);
    # that's NOT the probe's outer/inner. We re-key to probe semantics by
    # taking min/max of the two values:
    z_top = max(outer_geo, inner_geo)
    z_bottom = min(outer_geo, inner_geo)
    if probe_direction == 1:
        ref_outer, ref_inner, side = z_bottom, z_top, "buy"
    else:
        ref_outer, ref_inner, side = z_top, z_bottom, "sell"
    return ReferenceZone(
        outer=float(ref_outer),
        inner=float(ref_inner),
        side=side,  # type: ignore[arg-type]
        source="ad_hoc_bos_0",
        source_event_idx=int(m15_input_idx),
    )


def _build_first_counter_ref_zone(
    sibling_entity_df: Optional[pd.DataFrame],
    parent_sid: int,
    parent_cycle_id: int,
    probe_direction: int,
    m15_end_idx: int,
) -> Optional["ReferenceZone"]:
    """Reference zone for first_counter — sibling first_confluence's most
    recent CTS event for the same (parent_sid, parent_cycle_id) sub_sid=0.

    Filters `sibling_entity_df.attrs["events"]` + `["kl_zones"]` to just
    the sibling first_confluence bootstrap (sub_sid=0 of the same parent
    cycle), then delegates to `build_reference_zone_from_cts_event` —
    same primitive subsequent_*/reversal will use in Sessions 3+.

    Returns None when the sibling sub has no events yet (e.g., the
    bootstrap couldn't build for this cycle — orchestrator-side
    confluence-first ordering should make this very rare).
    """
    from engine_v2.structure.reference_zone import (
        build_reference_zone_from_cts_event,
    )

    if sibling_entity_df is None:
        return None
    events = sibling_entity_df.attrs.get("events", [])
    kl_zones = sibling_entity_df.attrs.get("kl_zones", [])
    sibling_events = [
        ev for ev in events
        if ev.meta.get("parent_sid") == parent_sid
        and ev.meta.get("parent_cycle_id") == parent_cycle_id
        and ev.meta.get("sub_sid") == 0
    ]
    sibling_zones = [
        z for z in kl_zones
        if z.meta.get("parent_sid") == parent_sid
        and z.meta.get("parent_cycle_id") == parent_cycle_id
        and z.meta.get("sub_sid") == 0
    ]
    if not sibling_events:
        return None
    # Each sibling sub's MS run uses structure_id=0 locally. Filter window
    # upper = the counter probe's end so we don't anachronistically pick a
    # CTS from after the counter trigger.
    return build_reference_zone_from_cts_event(
        events=sibling_events,
        kl_zones=sibling_zones,
        df=sibling_entity_df,
        sid=0,
        probe_direction=int(probe_direction),
        idx_window=(0, int(m15_end_idx)),
    )


def _resolve_first_via_unified_probe(
    trigger: MultiTFTrigger,
    parent_df: pd.DataFrame,
    entity_df: pd.DataFrame,
    sibling_entity_df: Optional[pd.DataFrame] = None,
) -> Tuple[Optional[int], Optional[int]]:
    """Unified-probe path for `first_counter` / `first_confluence` triggers
    (Part 4 §4.3, Session 2 design pivot 2026-05-29).

    Reference-zone sources (after the post-Gate-1 pivot):

    - `first_confluence`: ad-hoc BOS_0 derived on M15 from the input_idx
      candle (the parent BOS extreme price-mapped to M15). No parent or
      sibling zone read — this is the bootstrap of the confluence entity
      for this parent cycle, so nothing exists yet.
    - `first_counter`: sibling first_confluence's most recent CTS event
      (CONFIRMED → existing CTS zone; UPDATED/EST → ad-hoc CTS) — same
      rule as subsequent_*. Requires `sibling_entity_df` (the confluence
      entity's M15 df) to be passed in. The orchestrator builds
      confluence before counter so this dependency resolves.

    Reset predicate is two-condition (proximity + wick cap) per Session 1
    design.

    Returns ``(m15_start_idx, parent_extreme_idx)``. The second slot is
    metadata only (consumed by ``sid_records.validated_parent_start``);
    its meaning is "parent event extreme that seeded the M15 input."
    Either may be None on a pending probe, failed mapping, or missing
    sibling state for first_counter.
    """
    from engine_v2.multitf.data_bridge import map_candle_to_lower_tf
    from engine_v2.structure.unified_probe import unified_probe

    # 1. Parent event extreme. counter sets cts_anchor_idx; confluence
    #    sets BOS_CONFIRMED.ev.idx (which IS the BOS extreme per §4.3.2).
    raw_input = trigger.meta.get("probe_input_idx")
    raw_end = trigger.meta.get("probe_end_idx")
    if raw_input is None or raw_end is None:
        print(
            f"[entity_compute] WARNING: missing probe_input_idx/probe_end_idx "
            f"in trigger meta for {trigger.use_case} sid={trigger.parent_sid} "
            f"cycle={trigger.parent_cycle_id}"
        )
        return None, None
    parent_extreme_idx = int(raw_input)
    parent_end_idx = int(raw_end)
    if parent_extreme_idx not in parent_df.index:
        print(
            f"[entity_compute] WARNING: probe_input_idx={parent_extreme_idx} "
            f"out of parent_df bounds for {trigger.use_case} "
            f"sid={trigger.parent_sid} cycle={trigger.parent_cycle_id}"
        )
        return None, None

    # 2. Price-map parent extreme → M15 input_idx. parent_extreme_dir =
    #    -lower_sd (universal rule, spec §4.3.1).
    parent_extreme_time = pd.to_datetime(
        parent_df.loc[parent_extreme_idx, "time"], utc=True,
    )
    parent_extreme_dir = -trigger.lower_sd
    m15_input_idx = map_candle_to_lower_tf(
        parent_extreme_time, parent_extreme_dir, entity_df,
    )
    if m15_input_idx is None:
        print(
            f"[entity_compute] WARNING: parent→M15 input mapping failed for "
            f"{trigger.use_case} sid={trigger.parent_sid} "
            f"cycle={trigger.parent_cycle_id}"
        )
        return None, parent_extreme_idx

    # 3. Time-map probe end_idx (parent gate candle → last M15 of its hour).
    m15_end_idx = _map_parent_idx_to_m15_hour_end(
        parent_end_idx, parent_df, entity_df,
    )
    if m15_end_idx is None:
        print(
            f"[entity_compute] WARNING: parent→M15 end mapping failed for "
            f"{trigger.use_case} sid={trigger.parent_sid} "
            f"cycle={trigger.parent_cycle_id}"
        )
        return None, parent_extreme_idx
    if m15_end_idx <= m15_input_idx:
        print(
            f"[entity_compute] WARNING: degenerate probe window for "
            f"{trigger.use_case} sid={trigger.parent_sid} "
            f"cycle={trigger.parent_cycle_id} "
            f"(m15_input={m15_input_idx} m15_end={m15_end_idx})"
        )
        return None, parent_extreme_idx

    # 4. Build the per-trigger reference zone.
    if trigger.use_case == "first_confluence":
        ref_zone = _build_first_confluence_ref_zone(
            entity_df, int(m15_input_idx), int(trigger.lower_sd),
        )
        ref_label = "ad_hoc_bos_0"
    elif trigger.use_case == "first_counter":
        ref_zone = _build_first_counter_ref_zone(
            sibling_entity_df,
            trigger.parent_sid, trigger.parent_cycle_id,
            int(trigger.lower_sd), int(m15_end_idx),
        )
        ref_label = "sibling_confluence_cts"
    else:
        raise ValueError(
            f"_resolve_first_via_unified_probe called with unsupported "
            f"use_case={trigger.use_case!r}"
        )
    if ref_zone is None:
        print(
            f"[entity_compute] WARNING: {ref_label} reference zone unavailable "
            f"for {trigger.use_case} sid={trigger.parent_sid} "
            f"cycle={trigger.parent_cycle_id} — skipping trigger"
        )
        return None, parent_extreme_idx

    # 5. Run the unified probe on M15.
    # enable_phase2 only for first_confluence (the trigger whose end_idx
    # is NULL until parent CTS_CONFIRMED, making the post-end_idx MS
    # search structurally necessary — see PART4 §4.4 / project memory).
    _enable_phase2 = (trigger.use_case == "first_confluence")
    result = unified_probe(
        entity_df,
        input_idx=int(m15_input_idx),
        direction=int(trigger.lower_sd),
        reference_zone=ref_zone,
        end_idx=int(m15_end_idx),
        timeframe="M15",
        enable_phase2=_enable_phase2,
    )
    print(
        f"[entity_compute] unified_probe ({trigger.use_case}): "
        f"sid={trigger.parent_sid} cycle={trigger.parent_cycle_id} "
        f"m15_input={m15_input_idx} m15_end={m15_end_idx} "
        f"ref={ref_zone.source} -> start={result.start_idx} "
        f"status={result.status} cond={result.finalize_condition} "
        f"iter={result.iterations}"
    )

    if result.status == "pending":
        print(
            f"[entity_compute] PENDING: unified_probe did not finalize for "
            f"{trigger.use_case} sid={trigger.parent_sid} "
            f"cycle={trigger.parent_cycle_id}; skipping M15 build"
        )
        return None, parent_extreme_idx

    return int(result.start_idx), parent_extreme_idx


def _resolve_via_legacy_probe(
    trigger: MultiTFTrigger,
    parent_df: pd.DataFrame,
    entity_df: pd.DataFrame,
) -> Tuple[Optional[int], Optional[int]]:
    """Legacy Scenario-3-on-parent-TF probe + post-probe parent→M15
    mapping. Retained for ``subsequent_*`` triggers until their migration
    lands in Session 3 (per the Phase 1 migration plan in
    ``memory/project_unified_identify_start_probe.md``).
    """
    from engine_v2.multitf.data_bridge import map_candle_to_lower_tf
    from engine_v2.multitf.lower_tf_pipeline import _run_subordinate_probe

    validated_parent_idx = _run_subordinate_probe(trigger, parent_df)
    if validated_parent_idx is None:
        return None, None

    parent_start_time = pd.to_datetime(
        parent_df.loc[validated_parent_idx, "time"], utc=True,
    )
    # Universal mapping rule (§4.3.1 + LANDMINE "Subordinate mapping must
    # use `-lower_sd`"): the side of the parent hour we anchor on is the
    # OUTER of the sub's reference zone, which is the -lower_sd side.
    parent_extreme_dir = -trigger.lower_sd

    m15_start_idx = map_candle_to_lower_tf(
        parent_start_time, parent_extreme_dir, entity_df,
    )
    if m15_start_idx is None:
        print(
            f"[entity_compute] WARNING: parent→lower-TF mapping failed for "
            f"{trigger.use_case} sid={trigger.parent_sid} "
            f"cycle={trigger.parent_cycle_id}"
        )
        return None, validated_parent_idx
    return m15_start_idx, validated_parent_idx


# Use cases routed through the new unified-probe path (Session 2 scope).
# Other use cases (subsequent_*) stay on the legacy path until their
# Session 3 migration.
_UNIFIED_PROBE_USE_CASES = frozenset({"first_counter", "first_confluence"})


def _resolve_trigger_m15_start(
    trigger: MultiTFTrigger,
    parent_df: pd.DataFrame,
    entity_df: pd.DataFrame,
    sibling_entity_df: Optional[pd.DataFrame] = None,
) -> Tuple[Optional[int], Optional[int]]:
    """Resolve a trigger's M15 starting idx (entity-absolute) — dispatcher.

    Routes by ``trigger.use_case``:

    - ``first_counter`` / ``first_confluence`` → unified-probe path (M15
      probe; first_confluence consults an ad-hoc BOS_0; first_counter
      consults sibling first_confluence's most recent CTS — Session 2
      post-pivot design).
    - ``subsequent_confluence`` / ``subsequent_counter`` → legacy
      parent-TF Scenario-3 + post-probe mapping (until Session 3
      migration).

    Reversal-born sids do NOT call this — their start comes from
    ``identify_start_scenario_2_after_reversal`` on the sub's own data
    inside ``build_one_sid``.

    ``sibling_entity_df`` is the OTHER multi-TF entity's df (confluence
    if this is counter, None otherwise). Only first_counter consumes it
    (to look up sibling first_confluence's CTS for its reference zone);
    every other use_case ignores it. The orchestrator builds confluence
    before counter so the confluence entity_df is populated when counter
    asks for it.

    Returns ``(m15_start_idx, validated_or_extreme_parent_idx)``. The
    second slot is consumed only as metadata
    (``sid_records.validated_parent_start``); semantics shift slightly
    between paths (legacy = probe-converged parent idx; unified = parent
    event extreme that seeded the M15 input).
    """
    if trigger.use_case in _UNIFIED_PROBE_USE_CASES:
        return _resolve_first_via_unified_probe(
            trigger, parent_df, entity_df,
            sibling_entity_df=sibling_entity_df,
        )
    return _resolve_via_legacy_probe(trigger, parent_df, entity_df)


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
    sub_sid: int,
    started_by: str,
    start_trigger_idx: int,
    validated_parent_idx: Optional[int] = None,
    parent_floor_m15: Optional[int] = None,
    cap_open: bool = False,
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
            f"sid=({trigger.parent_sid},{trigger.parent_cycle_id},{sub_sid}): "
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

    # Re-derive range labels on the slice.
    # `apply_is_range_labels` runs in `prepare_lower_tf_data` on the FULL
    # entity-wide M15 df and writes `is_range_confirm_idx` with
    # ENTITY-ABSOLUTE positional values. After `reset_index(drop=True)`
    # above, MarketStructure reads those values as if they were slice-local
    # (see `_is_range_candle_given_confirm` / `_finalize_range_candidate_offline`).
    # That pollution: (a) emits RANGE_STARTED at a wildly wrong idx that
    # `mirror_lower_tf_result_to_entity_df` then double-shifts, (b) breaks the
    # back-fill window bound `min_d = min(confirm_idx, D)` in `_step_anchor`
    # (the entity-absolute value far exceeds D, so back-fill runs the full
    # range_max_k window unconditionally), and (c) poisons the `cts_anchor_idx`
    # snapshot at CTS_CONFIRMED because the over-extended back-fill calls
    # `_maybe_update_cts_pre_confirm` past the eventual pullback apply_idx,
    # so `st.cts.idx` records a future extreme. Symptom: BOS_{n+1}.idx can
    # land BEFORE CTS_n.cts_anchor_idx (cycle-progression invariant
    # violation). H1 main is unaffected because it never slices.
    from engine_v2.patterns.range_label import apply_is_range_labels, RangeLabelConfig
    trigger_df = trigger_df.drop(
        columns=["is_range", "is_range_confirm_idx", "is_range_lag"],
        errors="ignore",
    )
    trigger_df = apply_is_range_labels(trigger_df, RangeLabelConfig())

    start_in_slice = start_m15_abs - slice_begin
    end_in_slice = end_m15_abs - slice_begin
    if len(trigger_df) - start_in_slice < 5:
        print(
            f"[entity_compute] WARNING: M15 slice too small "
            f"({len(trigger_df) - start_in_slice} candles after start) "
            f"for {started_by} sid=({trigger.parent_sid},"
            f"{trigger.parent_cycle_id},{sub_sid})"
        )
        return None

    import time as _t
    _t_bounded_start = _t.perf_counter()
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
            f"{started_by} sid=({trigger.parent_sid},"
            f"{trigger.parent_cycle_id},{sub_sid}): {exc}"
        )
        return None
    _t_bounded = _t.perf_counter() - _t_bounded_start

    bounded.df.attrs["imbalances"] = trigger_df.attrs.get("imbalances", [])

    # Sub structure lifecycle-start (slice-local). Floors zone/POI activation so
    # no sub artifact is active before the sub structure is alive (Phase 3 Commit
    # 2). A sub's structural anchor (start_m15_abs) can sit historically before
    # its trigger (subsequent / reversal sids); for the bootstrap
    # start_trigger_idx == start_m15_abs. The floor is the LATEST of
    # {trigger idx, parent-cycle lifecycle-start} — the parent-cycle floor
    # (entity-absolute M15, last-of-hour H1->M15) already embeds the parent_sid
    # floor (PART4 §5, plan B1). parent_floor_m15 is None for main / when not
    # supplied (clamp degrades to trigger-only). All entity-absolute -> slice-local.
    _floor_abs = int(start_trigger_idx)
    if parent_floor_m15 is not None:
        _floor_abs = max(_floor_abs, int(parent_floor_m15))
    sub_lifecycle_floor_local = _floor_abs - slice_begin

    # Sub end-cap (B2 pass-through, 2026-05-27) — the mirror of the floor.
    # Resolved BEFORE the run so it can be passed in as `lifecycle_cap`: KL + POI
    # + Fib then INHERIT it as their cycle end via the shared helper (this
    # replaces the prior post-hoc cap loops — incl. the fib one, removed once fib
    # joined the helper in §15.6 / part b).
    #
    # The cap is a REAL structural end ONLY (PART4 §5 — "the data/window boundary
    # is not a lifecycle terminator"):
    #   - the sub's own reversal (structure boundary, slice-local, < end_m15_abs);
    #   - else the parent-cycle / next-sub boundary (`end_m15_abs`) WHEN it is a
    #     genuine end (`cap_open` False);
    #   - else (`cap_open`: the sub runs to the OPEN data edge — its parent cycle
    #     is the open last one, no next parent cycle / parent reversal) → cap
    #     None, so KL/POI/Fib stay active to the edge exactly like the H1 main's
    #     open last cycle. `effective_end_abs` (run/metadata bound) stays the edge;
    #     only the lifecycle CAP is dropped.
    reversal_idx_abs: Optional[int] = (
        bounded.reversal_idx + slice_begin if bounded.reversal_idx is not None else None
    )
    effective_end_abs = (
        reversal_idx_abs if reversal_idx_abs is not None else end_m15_abs
    )
    sub_lifecycle_cap_local: Optional[int]
    if reversal_idx_abs is not None:
        end_reason = "reversal"
        sub_lifecycle_cap_local = effective_end_abs - slice_begin
    elif cap_open:
        end_reason = None
        sub_lifecycle_cap_local = None
    else:
        end_reason = "lifecycle_end"
        sub_lifecycle_cap_local = effective_end_abs - slice_begin

    _t_downstream_start = _t.perf_counter()
    downstream = _run_downstream_pipeline(
        bounded.df,
        bounded.events,
        bounded.struct_direction,
        source_kinds=["BOS"],
        fib_mode="cross_cycle",
        log_prefix=(
            f"M15_{trigger.parent_sid}.{trigger.parent_cycle_id}."
            f"{sub_sid}_{started_by}"
        ),
        timeframe=timeframe,
        structure_path_id=sub_path_id,
        skip_wvmi=True,
        lifecycle_floor=sub_lifecycle_floor_local,
        lifecycle_cap=sub_lifecycle_cap_local,
        cap_reason=end_reason,
    )
    _t_downstream = _t.perf_counter() - _t_downstream_start

    # Reversal handoff (slice-local → entity-absolute). reversal_idx_abs was
    # resolved above (for the end-cap); here we derive the NEXT sid's start.
    next_start_abs: Optional[int] = None
    next_sd: Optional[int] = None
    if bounded.reversal_idx is not None:
        d_next = identify_start_scenario_2_after_reversal(
            bounded.df,
            reversal_idx=bounded.reversal_idx,
            prev_structure_id=0,
            prev_struct_direction=sd,
            min_history=50,
        )
        next_start_abs = int(d_next.start_idx) + slice_begin
        next_sd = int(d_next.struct_direction)

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

    # KL + POI now INHERIT their cycle end from the derivation: the cap was
    # passed in as `lifecycle_cap` above (B2 pass-through), so the prior post-hoc
    # cap loops are gone. The mirror translates end_idx slice-local→entity-absolute
    # alongside confirmed_idx, exactly as before.
    capped_zones = downstream["kl_zones"]
    capped_pois = downstream["poi_zones"]

    # Fib now inherits its cycle end (the sub `lifecycle_cap`) via the shared
    # helper inside `_finalize_lifecycle_fields` (FIB_LIFECYCLE_SPEC §15.6) — the
    # earlier post-hoc fib cap loop here is gone, exactly as KL/POI's were in B2.
    # The mirror (mirror_lower_tf_result_to_entity_df) shifts start_idx/end_idx
    # slice-local→entity-absolute.
    capped_fibs = downstream["fib_states"]

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
            # Canonical identity (§2 / §6.1): the tuple
            # (parent_sid, parent_cycle_id, sub_sid) — parent_* travel in
            # `attribution`. sub_sid resets per parent cycle.
            "sub_sid": sub_sid,
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
        structure_path_id=sub_path_id,
    )

    _slice_len = len(trigger_df)
    print(
        f"[entity_compute] {started_by} "
        f"id=({trigger.parent_sid},{trigger.parent_cycle_id},{sub_sid}) "
        f"start={start_m15_abs} end={effective_end_abs} bound={end_m15_abs} "
        f"slice_len={_slice_len} "
        f"reversed={reversal_idx_abs is not None} "
        f"events={len(bounded.events)} kl={len(capped_zones)} "
        f"poi={len(capped_pois)} fib={len(capped_fibs)} "
        f"t_bounded={_t_bounded:.2f}s t_downstream={_t_downstream:.2f}s "
        f"t_total={(_t_bounded + _t_downstream):.2f}s"
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
    parent_cycle_floor_h1: Optional[Dict[tuple, int]] = None,
    sibling_entity_df: Optional[pd.DataFrame] = None,
) -> List[LowerTFResult]:
    """Build one parent cycle's subordinate sid chain (Part 4 §6.1 merge-and-bound).

    Per ``(entity, parent_sid, parent_cycle_id)``:

    - **sub_sid=0** starts at the bootstrap's probe-validated start (``first_*``
      variation). If that probe is pending / mapping fails → no chain (no
      sub_sid=0 ⇒ subsequents don't build either).
    - Each sid runs a bounded single structure (``build_one_sid``) to the
      first of {its own reversal, the next subsequent trigger, parent-cycle
      end}.
    - That boundary starts sub_sid+1: a subsequent trigger → its
      probe-validated start + use_case direction; a reversal →
      ``identify_start`` (reversal scenario, flipped direction).
    - ``sub_sid`` is the per-cycle counter (resets to 0 each parent cycle);
      the sub's full identity is the tuple ``(parent_sid, parent_cycle_id,
      sub_sid)``.

    ``subsequents`` must be this cycle's ``subsequent_*`` triggers
    (already converted to ``MultiTFTrigger``); order doesn't matter (resolved
    + sorted by M15 boundary here). Sub WVMI is NOT computed here — the
    orchestrator runs a per-cycle trigger-centric pass over the returned
    ``results`` (``_assign_trigger_centric_sub_wvmi``), gating each sid by
    whether a parent trigger lands in its active window.

    Returns the cycle's ``results`` (sids in build order).
    """
    results: List[LowerTFResult] = []

    # Bootstrap start (entity-absolute). `sibling_entity_df` (confluence
    # for counter, None for confluence) only matters for `first_counter`
    # which consults sibling first_confluence's CTS for its reference zone.
    m15_start_0, validated_parent_0 = _resolve_trigger_m15_start(
        bootstrap, parent_df, entity_df,
        sibling_entity_df=sibling_entity_df,
    )
    if m15_start_0 is None:
        print(
            f"[chain] no sid=0 for {bootstrap.use_case} "
            f"sid={bootstrap.parent_sid} cycle={bootstrap.parent_cycle_id} "
            f"(probe pending / mapping failed) — cycle skipped"
        )
        return results

    # Parent-cycle end (entity-absolute).
    from engine_v2.multitf.lower_tf_pipeline import _find_m15_lifecycle_end
    cycle_end_m15 = _find_m15_lifecycle_end(bootstrap, entity_df, parent_df)
    if cycle_end_m15 is None:
        cycle_end_m15 = int(entity_df.index[-1])

    # Parent-cycle lifecycle-start floor (M15, entity-absolute) for this whole
    # chain (all sids share this (parent_sid, parent_cycle_id)). The orchestrator
    # supplies the floor in H1 coords (parent cycle's clamped lifecycle-start);
    # map H1->M15 last-of-hour here. None if not supplied / not found — the clamp
    # then degrades to each sid's own trigger floor. PART4 §5, plan B1.
    parent_floor_m15: Optional[int] = None
    if parent_cycle_floor_h1:
        _floor_h1 = parent_cycle_floor_h1.get(
            (bootstrap.parent_sid, bootstrap.parent_cycle_id)
        )
        if _floor_h1 is not None:
            parent_floor_m15 = _map_parent_idx_to_m15_hour_end(
                int(_floor_h1), parent_df, entity_df,
            )

    # Resolve subsequents up front: keep only those whose probe finalizes
    # (a pending probe ⇒ no sid AND no boundary, so the prior sid runs
    # through — §6.1). boundary = trigger_event_idx mapped to M15 hour-end.
    resolved_subs: List[Dict[str, Any]] = []
    for sub in subsequents:
        s_start, s_valid = _resolve_trigger_m15_start(
            sub, parent_df, entity_df,
            sibling_entity_df=sibling_entity_df,
        )
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

    sub_sid = 0
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

        # cap_open: this sid runs to the OPEN data edge — no subsequent trigger
        # after it AND the parent cycle is still active (`lifecycle_end_idx None`,
        # which is exactly why `cycle_end_m15` fell back to `entity_df.index[-1]`).
        # The data edge is not a lifecycle terminator (PART4 §5), so the sid's
        # cap is dropped (None) and its zones stay active to the edge. Keyed on
        # `lifecycle_end_idx is None` (the semantic open-cycle signal), NOT on
        # `cycle_end_m15 == last idx`, so a rare H1->M15 mapping failure (which
        # also falls back to the edge) is still treated as a real cap.
        cap_open = (next_sub is None) and (bootstrap.lifecycle_end_idx is None)

        outcome = build_one_sid(
            entity_df,
            start_m15_abs=cur_start, sd=cur_sd, end_m15_abs=bound,
            sub_path_id=sub_path_id, timeframe=cur_trigger.lower_tf,
            trigger=cur_trigger, sub_sid=sub_sid,
            started_by=cur_started_by, start_trigger_idx=cur_start_trig,
            validated_parent_idx=cur_valid,
            parent_floor_m15=parent_floor_m15,
            cap_open=cap_open,
        )
        if outcome is None:
            # Degenerate / failed sid ends the chain (conservative — a rare
            # case on real data; revisit if it drops legitimate later sids).
            break

        results.append(outcome.result)
        sub_sid += 1

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

    return results
