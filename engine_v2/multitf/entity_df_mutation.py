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


def _build_sibling_cts_ref_zone(
    sibling_entity_df: Optional[pd.DataFrame],
    parent_sid: int,
    parent_cycle_id: int,
    probe_direction: int,
    idx_window: Tuple[int, int],
) -> Optional["ReferenceZone"]:
    """Sibling-CTS reference zone for the three sibling-referencing variations
    (`first_counter`, `subsequent_confluence`, `subsequent_counter`) — Session 3
    uniform rule (2026-05-31).

    Walks the SIBLING entity's events for this `(parent_sid, parent_cycle_id)`
    (across ALL the sibling's sub_sids — the time-bounded `idx_window` selects
    the right one), takes the most recent of {CTS_CONFIRMED, CTS_UPDATED,
    CTS_ESTABLISHED} via `build_reference_zone_from_cts_event`. The returned
    zone's `source_event_idx` is the sibling CTS extreme on the sub TF — the
    caller uses it as BOTH the probe `input_idx` AND the reference zone (they
    are co-sourced).

    **sub_sid disambiguation (idx-correctness, LANDMINE "Cross-entity sibling
    references"):** every sibling sub's MS run uses `structure_id=0` LOCALLY, so
    the CONFIRMED-branch KL-zone lookup `(structure_id=0, cycle_id)` inside the
    primitive could match a zone from a DIFFERENT sub_sid that happens to share
    that local cycle_id. We pre-select the winning CTS event (most-recent in
    window) here, read its `sub_sid`, and pass the primitive a `kl_zones` list
    filtered to that winning sub_sid only — so the zone lookup is unambiguous.
    The event list is unfiltered (the primitive re-selects the same winner by
    idx), keeping the selection logic single-sourced in the primitive.

    `idx_window` is the inclusive `(lo, hi)` entity-absolute M15 range the
    sibling CTS must fall in (per-variation; see `_resolve_sibling_cts_probe`).

    Returns None when the sibling df is absent or has no qualifying CTS in the
    window (caller skips the trigger — extremely rare given cadence ordering).
    """
    from engine_v2.structure.reference_zone import (
        build_reference_zone_from_cts_event,
        _CTS_EVENT_TYPES,
    )

    if sibling_entity_df is None:
        return None
    events = sibling_entity_df.attrs.get("events", [])
    kl_zones = sibling_entity_df.attrs.get("kl_zones", [])
    lo, hi = int(idx_window[0]), int(idx_window[1])

    sibling_events = [
        ev for ev in events
        if ev.meta.get("parent_sid") == parent_sid
        and ev.meta.get("parent_cycle_id") == parent_cycle_id
    ]
    if not sibling_events:
        return None

    # Pre-select the winning CTS event to disambiguate the CONFIRMED zone lookup
    # by sub_sid (see docstring). Tie-break order matches the primitive's.
    _TYPE_ORDER = {"CTS_CONFIRMED": 2, "CTS_UPDATED": 1, "CTS_ESTABLISHED": 0}
    cands = [
        ev for ev in sibling_events
        if ev.type in _CTS_EVENT_TYPES and lo <= int(ev.idx) <= hi
    ]
    if not cands:
        return None
    winner = max(cands, key=lambda e: (int(e.idx), _TYPE_ORDER[e.type]))
    win_sub = winner.meta.get("sub_sid")

    sibling_zones = [
        z for z in kl_zones
        if z.meta.get("parent_sid") == parent_sid
        and z.meta.get("parent_cycle_id") == parent_cycle_id
        and z.meta.get("sub_sid") == win_sub
    ]
    return build_reference_zone_from_cts_event(
        events=sibling_events,
        kl_zones=sibling_zones,
        df=sibling_entity_df,
        sid=0,
        probe_direction=int(probe_direction),
        idx_window=(lo, hi),
    )


def _resolve_first_confluence_via_unified_probe(
    trigger: MultiTFTrigger,
    parent_df: pd.DataFrame,
    entity_df: pd.DataFrame,
) -> Tuple[Optional[int], Optional[int]]:
    """Unified-probe path for `first_confluence` (the one variation that
    anchors on its OWN ad-hoc BOS_0, not a sibling).

    The confluence entity's sid=0 bootstrap has no prior/sibling structure
    yet, so its reference zone is an ad-hoc BOS_0 derived on M15 from the
    input_idx candle (the parent BOS extreme price-mapped to M15). Phase 2
    (MS-based) runs because first_confluence's end_idx is the parent CTS
    extreme — the only variation needing the post-end_idx MS search (PART4
    §4.4).

    Returns ``(m15_start_idx, parent_extreme_idx)``. The second slot is
    metadata only (`sid_records.validated_parent_start`) — the parent BOS
    extreme that seeded the M15 input.
    """
    from engine_v2.multitf.data_bridge import map_candle_to_lower_tf
    from engine_v2.structure.unified_probe import unified_probe

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

    # Candle-semantics mapping rule (user spec 2026-05-31): BOS/CTS ANCHOR
    # candles are PRICE-mapped (they anchor a price level into the sub);
    # trigger/confirmation candles are TIME-mapped (temporal gates). Both of
    # first_confluence's bounds are ANCHOR candles, so BOTH are price-mapped:
    #   - input  = parent BOS anchor, extreme on the -lower_sd side (base of
    #              the breakout — the price the probe measures retraces from).
    #   - end    = parent CTS anchor, extreme on the +lower_sd side (the
    #              structure ceiling/floor). NOT a temporal cutoff: it says
    #              "once price hit the parent CTS extreme, no NEW extreme can
    #              form, so stop probing." end_idx is unknown until parent CTS
    #              confirms (cts_anchor_idx only exists then), but the value
    #              that bounds the probe is the CTS extreme, not the (later)
    #              confirmation candle — hence price-mapped, not time-mapped.
    parent_extreme_time = pd.to_datetime(
        parent_df.loc[parent_extreme_idx, "time"], utc=True,
    )
    m15_input_idx = map_candle_to_lower_tf(
        parent_extreme_time, -trigger.lower_sd, entity_df,
    )
    if m15_input_idx is None:
        print(
            f"[entity_compute] WARNING: parent→M15 input mapping failed for "
            f"{trigger.use_case} sid={trigger.parent_sid} "
            f"cycle={trigger.parent_cycle_id}"
        )
        return None, parent_extreme_idx

    if parent_end_idx not in parent_df.index:
        print(
            f"[entity_compute] WARNING: probe_end_idx={parent_end_idx} out of "
            f"parent_df bounds for {trigger.use_case} sid={trigger.parent_sid} "
            f"cycle={trigger.parent_cycle_id}"
        )
        return None, parent_extreme_idx
    parent_cts_time = pd.to_datetime(
        parent_df.loc[parent_end_idx, "time"], utc=True,
    )
    m15_end_idx = map_candle_to_lower_tf(
        parent_cts_time, trigger.lower_sd, entity_df,
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

    ref_zone = _build_first_confluence_ref_zone(
        entity_df, int(m15_input_idx), int(trigger.lower_sd),
    )
    if ref_zone is None:
        print(
            f"[entity_compute] WARNING: ad_hoc_bos_0 reference zone unavailable "
            f"for {trigger.use_case} sid={trigger.parent_sid} "
            f"cycle={trigger.parent_cycle_id} — skipping trigger"
        )
        return None, parent_extreme_idx

    result = unified_probe(
        entity_df,
        input_idx=int(m15_input_idx),
        direction=int(trigger.lower_sd),
        reference_zone=ref_zone,
        end_idx=int(m15_end_idx),
        timeframe="M15",
        enable_phase2=True,
    )
    print(
        f"[entity_compute] unified_probe (first_confluence): "
        f"sid={trigger.parent_sid} cycle={trigger.parent_cycle_id} "
        f"m15_input={m15_input_idx} m15_end={m15_end_idx} "
        f"ref={ref_zone.source} -> start={result.start_idx} "
        f"status={result.status} cond={result.finalize_condition} "
        f"iter={result.iterations}"
    )
    if result.status == "pending":
        print(
            f"[entity_compute] PENDING: unified_probe did not finalize for "
            f"first_confluence sid={trigger.parent_sid} "
            f"cycle={trigger.parent_cycle_id}; skipping M15 build"
        )
        return None, parent_extreme_idx
    return int(result.start_idx), parent_extreme_idx


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
    entity_df: pd.DataFrame,
    m15_end_idx: int,
) -> Tuple[int, int]:
    """Inclusive entity-absolute M15 `(lo, hi)` window the sibling CTS must
    fall in, per use_case (PART4 §4.3.3/4/5, Session 3).

    - `first_counter`: `[0, m15_end_idx]` — the whole parent cycle up to the
      first sd-prox trigger (events are already filtered to this parent cycle,
      so 0 is a safe lower bound).
    - `subsequent_confluence`: `[last-M15 of prior_sd_prox hour, m15_end_idx]`.
    - `subsequent_counter`: `[last-M15 of prior_cts_prox hour, m15_end_idx]`.

    `hi` is always `m15_end_idx` (= the time-mapped trigger candle). A missing
    prior-prox meta value degrades `lo` to 0 (safe — the parent-cycle event
    filter still scopes the walk).
    """
    hi = int(m15_end_idx)
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
            int(prior_parent_idx), parent_df, entity_df,
        )
        if mapped is not None:
            lo = int(mapped)
    return lo, hi


def _resolve_sibling_cts_via_unified_probe(
    trigger: MultiTFTrigger,
    parent_df: pd.DataFrame,
    entity_df: pd.DataFrame,
    sibling_entity_df: Optional[pd.DataFrame] = None,
) -> Tuple[Optional[int], Optional[int]]:
    """Unified-probe path for the three sibling-referencing variations
    (`first_counter`, `subsequent_confluence`, `subsequent_counter`) — Session 3
    uniform rule (2026-05-31).

    Both the probe `input_idx` AND the `reference_zone` are co-sourced from the
    **sibling entity's** most recent CTS event in the trigger's sub-TF window
    (CONFIRMED → existing KL zone + `cts_anchor_idx`; UPDATED/EST → ad-hoc CTS
    zone + event idx). The sibling is whichever entity the driver passes as
    `sibling_entity_df` (confluence reads counter; counter reads confluence) —
    correct for all three by cadence (the read is always strictly earlier than
    this trigger). The probe runs on the structure's OWN M15 frame; the sibling
    CTS idx is entity-absolute M15, valid under the frame-alignment guard.

    `enable_phase2=False` — none of these has the NULL-end_idx problem that makes
    Phase 2 necessary (that's first_confluence only).

    Returns ``(m15_start_idx, sibling_cts_extreme_idx)``. The second slot is
    metadata only (`sid_records.validated_parent_start`); for these variations
    it is the sibling CTS extreme (M15 frame) that seeded the input.
    """
    from engine_v2.structure.unified_probe import unified_probe

    raw_end = trigger.meta.get("probe_end_idx")
    if raw_end is None:
        print(
            f"[entity_compute] WARNING: missing probe_end_idx in trigger meta "
            f"for {trigger.use_case} sid={trigger.parent_sid} "
            f"cycle={trigger.parent_cycle_id}"
        )
        return None, None

    # 1. Time-map probe end_idx (parent trigger candle → last M15 of its hour).
    m15_end_idx = _map_parent_idx_to_m15_hour_end(
        int(raw_end), parent_df, entity_df,
    )
    if m15_end_idx is None:
        print(
            f"[entity_compute] WARNING: parent→M15 end mapping failed for "
            f"{trigger.use_case} sid={trigger.parent_sid} "
            f"cycle={trigger.parent_cycle_id}"
        )
        return None, None

    # 2. Sibling-CTS reference zone within the per-variation M15 window. Its
    #    `source_event_idx` IS the probe input_idx (co-sourced).
    idx_window = _sibling_cts_idx_window(
        trigger, parent_df, entity_df, m15_end_idx,
    )
    ref_zone = _build_sibling_cts_ref_zone(
        sibling_entity_df,
        trigger.parent_sid, trigger.parent_cycle_id,
        int(trigger.lower_sd), idx_window,
    )

    if ref_zone is None:
        # Fallback (PART4 §4.3.4 step 5) — sibling has no CTS event in the
        # window (genuinely rare after the lazy-resolution fix, but specced, so
        # implemented). Anchor on the OWN entity: input = window extreme on the
        # `-lower_sd` side (the base the structure grows from, same convention
        # as first_confluence's BOS extreme), reference = own ad-hoc BOS_0 from
        # that candle.
        fb_idx = _window_extreme_idx(
            entity_df, idx_window[0], idx_window[1], -int(trigger.lower_sd),
        )
        if fb_idx is not None:
            ref_zone = _build_first_confluence_ref_zone(
                entity_df, int(fb_idx), int(trigger.lower_sd),
            )
        if ref_zone is None:
            print(
                f"[entity_compute] WARNING: sibling-CTS reference zone AND "
                f"ad-hoc fallback both unavailable for {trigger.use_case} "
                f"sid={trigger.parent_sid} cycle={trigger.parent_cycle_id} "
                f"window={idx_window} — skipping trigger"
            )
            return None, None
        print(
            f"[entity_compute] sibling-CTS unavailable for {trigger.use_case} "
            f"sid={trigger.parent_sid} cycle={trigger.parent_cycle_id} "
            f"window={idx_window} — using ad-hoc BOS_0 fallback at {fb_idx}"
        )

    m15_input_idx = int(ref_zone.source_event_idx)
    if m15_input_idx >= m15_end_idx:
        # The sibling CTS extreme is at/after the probe end — no forward scan
        # window. Degenerate; skip this trigger.
        print(
            f"[entity_compute] WARNING: degenerate sibling-CTS probe window for "
            f"{trigger.use_case} sid={trigger.parent_sid} "
            f"cycle={trigger.parent_cycle_id} "
            f"(m15_input={m15_input_idx} m15_end={m15_end_idx})"
        )
        return None, m15_input_idx

    # 3. Run the unified probe on the structure's own M15 frame (Phase 1 only).
    result = unified_probe(
        entity_df,
        input_idx=m15_input_idx,
        direction=int(trigger.lower_sd),
        reference_zone=ref_zone,
        end_idx=int(m15_end_idx),
        timeframe="M15",
        enable_phase2=False,
    )
    print(
        f"[entity_compute] unified_probe ({trigger.use_case}): "
        f"sid={trigger.parent_sid} cycle={trigger.parent_cycle_id} "
        f"m15_input={m15_input_idx} m15_end={m15_end_idx} window={idx_window} "
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
        return None, m15_input_idx
    return int(result.start_idx), m15_input_idx


def _resolve_via_legacy_probe(
    trigger: MultiTFTrigger,
    parent_df: pd.DataFrame,
    entity_df: pd.DataFrame,
) -> Tuple[Optional[int], Optional[int]]:
    """Legacy Scenario-3-on-parent-TF probe + post-probe parent→M15
    mapping. NO LONGER on any default path (all four variations migrated to
    the unified probe as of Session 3). Retained ONLY as the bisect escape
    hatch behind `_LEGACY_PROBE_USE_CASES` — add a use_case to that set to
    force it back onto this path when isolating a /compare regression.
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


# Use-case routing for the start resolver (Session 3, 2026-05-31).
#   - first_confluence → own ad-hoc BOS_0, Phase 1+2.
#   - first_counter / subsequent_* → sibling CTS (input + ref co-sourced),
#     Phase 1 only.
# `_LEGACY_PROBE_USE_CASES` is the bisect escape hatch: any use_case listed
# there falls back to the legacy parent-TF Scenario-3 probe. Empty by default
# (all migrated) — populate it temporarily to isolate a /compare regression.
_FIRST_CONFLUENCE_USE_CASES = frozenset({"first_confluence"})
_SIBLING_CTS_USE_CASES = frozenset(
    {"first_counter", "subsequent_confluence", "subsequent_counter"}
)
_LEGACY_PROBE_USE_CASES: frozenset = frozenset()


def _resolve_trigger_m15_start(
    trigger: MultiTFTrigger,
    parent_df: pd.DataFrame,
    entity_df: pd.DataFrame,
    sibling_entity_df: Optional[pd.DataFrame] = None,
) -> Tuple[Optional[int], Optional[int]]:
    """Resolve a trigger's M15 starting idx (entity-absolute) — dispatcher.

    Routes by ``trigger.use_case`` (Session 3 uniform rule):

    - ``first_confluence`` → unified probe vs own ad-hoc BOS_0 (Phase 1+2).
    - ``first_counter`` / ``subsequent_confluence`` / ``subsequent_counter`` →
      unified probe vs the **sibling** entity's most recent CTS, which supplies
      BOTH the input_idx and the reference zone (Phase 1 only).
    - Any use_case in ``_LEGACY_PROBE_USE_CASES`` (empty by default) → legacy
      parent-TF Scenario-3 probe — the bisect escape hatch.

    Reversal-born sids do NOT call this — their start comes from the sub's own
    reversal handoff inside ``build_one_sid``.

    ``sibling_entity_df`` is the OTHER multi-TF entity's df (counter when this
    is confluence, confluence when this is counter). The three sibling-CTS
    variations consume it; first_confluence ignores it. The two-entity cadence
    driver (`build_two_entity_parent_cycle`) guarantees the needed sibling sid
    is already built when this fires (LANDMINE "Cross-entity sibling references
    require cadence-order interleaving").

    Returns ``(m15_start_idx, metadata_idx)``. The second slot is consumed only
    as metadata (`sid_records.validated_parent_start`); its meaning varies by
    path (legacy = probe-converged parent idx; first_confluence = parent BOS
    extreme; sibling-CTS = sibling CTS extreme on M15).
    """
    uc = trigger.use_case
    if uc in _LEGACY_PROBE_USE_CASES:
        return _resolve_via_legacy_probe(trigger, parent_df, entity_df)
    if uc in _FIRST_CONFLUENCE_USE_CASES:
        return _resolve_first_confluence_via_unified_probe(
            trigger, parent_df, entity_df,
        )
    if uc in _SIBLING_CTS_USE_CASES:
        return _resolve_sibling_cts_via_unified_probe(
            trigger, parent_df, entity_df,
            sibling_entity_df=sibling_entity_df,
        )
    # Unknown use_case → legacy as a conservative default.
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
    4. If it reversed → the **unified probe** vs the just-reversed structure's
       most recent CTS reference (PART4 §6 / unified-probe §3 reversal row) for
       the next sid's start (entity-absolute), returned on the outcome. This
       replaces ``identify_start_scenario_2_after_reversal`` (Scenario 2 +
       Exception 1) on the sub path — the probe's furthest-retrace Phase-1 walk
       subsumes Exception 1. (Main's reversal handoff keeps Scenario 2 + Exc1
       until Step 4.)
    5. Lifecycle-cap open/late zones/POIs/fibs at the sid's effective end
       (its reversal if any, else the bound).
    6. Mirror into ``entity_df.attrs`` with the canonical identity
       (sids are sequential & non-overlapping, so there is no cascade).

    Returns a ``SidBuildOutcome`` or None on a degenerate window / structure
    failure.
    """
    from engine_v2.patterns.imbalance import compute_imbalance
    from engine_v2.pipeline.orchestrator import _run_downstream_pipeline
    from engine_v2.structure.reference_zone import (
        build_reference_zone_from_cts_event,
    )
    from engine_v2.structure.unified_probe import unified_probe
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
    # resolved above (for the end-cap); here we derive the NEXT sid's start via
    # the unified probe vs the just-reversed structure's most recent CTS
    # reference (PART4 §6 / unified-probe §3 reversal row). Replaces
    # identify_start_scenario_2_after_reversal (Scenario 2 + Exception 1) — the
    # probe's furthest-retrace Phase-1 walk + cond-2 wick filter subsume Exc1.
    # next_start_abs/next_sd are set TOGETHER only on a successful probe (the
    # cursor consumer requires next_start_abs not None); every failure branch
    # leaves both None so the chain falls through to any pending subsequent.
    next_start_abs: Optional[int] = None
    next_sd: Optional[int] = None
    if bounded.reversal_idx is not None:
        probe_sd = -sd
        # Reference = prior structure's (sid=0) most recent {CONF/UPD/EST} CTS.
        # downstream (with kl_zones) already ran above, so the CONFIRMED-zone
        # lookup is populated; the prior sid is NOT yet mirrored into
        # entity_df.attrs (that happens below), so we read bounded.events
        # directly. idx_window=None per the reversal contract.
        ref_zone = build_reference_zone_from_cts_event(
            bounded.events,
            downstream["kl_zones"],
            bounded.df,
            sid=0,
            probe_direction=probe_sd,
            idx_window=None,
        )
        if ref_zone is None:
            # Degenerate: a reversal with zero CTS events of any type for the
            # prior structure (should not occur — a structure must establish a
            # cycle to reverse). Terminate the chain rather than guess a start.
            print(
                f"[entity_compute] WARNING: reversal reference zone unavailable "
                f"for {started_by} sid=({trigger.parent_sid},"
                f"{trigger.parent_cycle_id},{sub_sid}) — no CTS event for prior "
                f"structure; no reversal-born sid (chain falls through)"
            )
        else:
            # Probe end_idx = the reversal candle (slice-local), per PART4 §4.
            # Use the value compute_bounded_structure ALREADY derived
            # (`bounded.reversal_idx` = the reversal-marked candle, identical to
            # the lifecycle-cap `reversal_idx_abs` resolved above) rather than
            # re-deriving from a market_state mask. Re-deriving via a `.max()`
            # mask drifts to the window edge: in a bounded single structure the
            # "reversal" state persists from the reversal candle to the end, and
            # terminal stamping past end_idx marks structure_id=-1 rows too (see
            # compute_bounded_structure) — a mask without the structure_id
            # filter sweeps those up. `bounded.reversal_idx` is the canonical
            # single-source value; don't recompute it.
            reversal_end_local = int(bounded.reversal_idx)
            probe_input_idx = int(ref_zone.source_event_idx)
            if probe_input_idx >= reversal_end_local:
                # No forward scan window — degenerate; no reversal-born sid.
                print(
                    f"[entity_compute] WARNING: degenerate reversal probe window "
                    f"for {started_by} sid=({trigger.parent_sid},"
                    f"{trigger.parent_cycle_id},{sub_sid}) "
                    f"(input={probe_input_idx} end={reversal_end_local}) — "
                    f"no reversal-born sid (chain falls through)"
                )
            else:
                rev_probe = unified_probe(
                    bounded.df,
                    input_idx=probe_input_idx,
                    direction=probe_sd,
                    reference_zone=ref_zone,
                    end_idx=reversal_end_local,
                    timeframe=timeframe,
                    enable_phase2=False,
                )
                print(
                    f"[entity_compute] unified_probe (reversal): "
                    f"sid=({trigger.parent_sid},{trigger.parent_cycle_id},"
                    f"{sub_sid}) input={probe_input_idx} "
                    f"end={reversal_end_local} ref={ref_zone.source} -> "
                    f"start={rev_probe.start_idx} status={rev_probe.status} "
                    f"cond={rev_probe.finalize_condition} "
                    f"iter={rev_probe.iterations}"
                )
                if rev_probe.status == "pending":
                    # Only possible with end_idx=None; we always pass a defined
                    # end_idx, so this is dormant. Guard for symmetry.
                    print(
                        f"[entity_compute] PENDING: reversal unified_probe did "
                        f"not finalize for {started_by} sid=("
                        f"{trigger.parent_sid},{trigger.parent_cycle_id},"
                        f"{sub_sid}); no reversal-born sid (chain falls through)"
                    )
                else:
                    next_start_abs = int(rev_probe.start_idx) + slice_begin
                    next_sd = probe_sd

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
    cursor = _ChainCursor(
        entity_df, parent_df,
        bootstrap=bootstrap, subsequents=subsequents,
        sub_path_id=sub_path_id,
        parent_cycle_floor_h1=parent_cycle_floor_h1,
        sibling_entity_df=sibling_entity_df,
    )
    while not cursor.done:
        cursor.step()
    return cursor.results


class _ChainCursor:
    """Resumable single-entity parent-cycle chain builder (Part 4 §6.1).

    Wraps the per-``(entity, parent_sid, parent_cycle_id)`` sid-chain state so
    two cursors (confluence + counter) can be advanced in LOCK-STEP by
    `build_two_entity_parent_cycle`, each reading the OTHER entity_df as its
    sibling. Within one entity the sids are built in the SAME order as the
    legacy single-shot loop; the only timing change is that the **bootstrap
    start is resolved lazily** (on the first `step`) so a sibling-reading
    bootstrap (``first_counter``) sees the sibling built through the cadence
    so far. Subsequents are still resolved eagerly here (Step 1: the legacy
    subsequent probe ignores the sibling, so eager resolution exactly
    reproduces the legacy drop-on-pending behavior and keeps output
    byte-identical).

    `next_boundary` is the M15 idx (entity-absolute) that the SIBLING must be
    built through before this cursor's current sid is built — the current
    sid's probe end (trigger candle) for trigger-born sids, or its reversal
    apply idx for reversal-born sids. The driver always steps whichever cursor
    has the smaller `next_boundary`; because the alternating cadence guarantees
    every sibling CTS a probe reads is strictly earlier than the probe's own
    end, that sibling sid is always already built.
    """

    def __init__(
        self,
        entity_df: pd.DataFrame,
        parent_df: pd.DataFrame,
        *,
        bootstrap: MultiTFTrigger,
        subsequents: List[MultiTFTrigger],
        sub_path_id: str,
        parent_cycle_floor_h1: Optional[Dict[tuple, int]] = None,
        sibling_entity_df: Optional[pd.DataFrame] = None,
    ) -> None:
        self.entity_df = entity_df
        self.parent_df = parent_df
        self.bootstrap = bootstrap
        self.sub_path_id = sub_path_id
        self._sibling_df = sibling_entity_df
        self.results: List[LowerTFResult] = []
        self.done = False

        # Parent-cycle end (entity-absolute). None ⇒ open cycle → data edge.
        from engine_v2.multitf.lower_tf_pipeline import _find_m15_lifecycle_end
        cend = _find_m15_lifecycle_end(bootstrap, entity_df, parent_df)
        self.cycle_end_m15 = (
            cend if cend is not None else int(entity_df.index[-1])
        )

        # Parent-cycle lifecycle-start floor (M15, entity-absolute), PART4 §5.
        self.parent_floor_m15: Optional[int] = None
        if parent_cycle_floor_h1:
            _floor_h1 = parent_cycle_floor_h1.get(
                (bootstrap.parent_sid, bootstrap.parent_cycle_id)
            )
            if _floor_h1 is not None:
                self.parent_floor_m15 = _map_parent_idx_to_m15_hour_end(
                    int(_floor_h1), parent_df, entity_df,
                )

        # Subsequent triggers: store trigger + boundary ONLY. The probe (which
        # reads the SIBLING entity for subsequent_*) is resolved LAZILY at
        # step-time, when the subsequent becomes the current sid — by then the
        # two-entity driver has built the sibling through this subsequent's
        # boundary (cadence guarantee). Resolving eagerly here would read an
        # EMPTY sibling (LANDMINE "Cross-entity sibling references require
        # cadence-order interleaving" — the Step-2 bug this fixes). `boundary`
        # comes from trigger meta (no probe needed), so merge-ordering via
        # `next_boundary` still works without resolving.
        self.pending: List[Dict[str, Any]] = []
        for sub in (subsequents or []):
            tei = sub.meta.get("trigger_event_idx")
            if tei is None:
                continue
            boundary = _map_parent_idx_to_m15_hour_end(
                int(tei), parent_df, entity_df,
            )
            if boundary is None:
                continue
            self.pending.append({"trigger": sub, "boundary": boundary})
        self.pending.sort(key=lambda x: x["boundary"])

        # Chain state for the NEXT sid to build. cur_start is resolved lazily
        # on the first step (bootstrap); subsequent/reversal sids set it when
        # advancing.
        self.sub_sid = 0
        self.cur_trigger: MultiTFTrigger = bootstrap
        self.cur_started_by: str = bootstrap.use_case
        self.cur_sd: int = int(bootstrap.lower_sd)
        self.cur_start: Optional[int] = None
        self.cur_start_trig: Optional[int] = None
        self.cur_valid: Optional[int] = None
        self.guard = 0
        self.max_chain = 50

    def _boundary_for_trigger(self, trig: MultiTFTrigger) -> Optional[int]:
        """M15 idx the probe reads the sibling through = its probe end."""
        pe = trig.meta.get("probe_end_idx")
        if pe is None:
            pe = trig.meta.get("trigger_event_idx")
        if pe is None:
            return None
        return _map_parent_idx_to_m15_hour_end(
            int(pe), self.parent_df, self.entity_df,
        )

    @property
    def next_boundary(self) -> Optional[int]:
        """Merge key: the sibling must be built through this before stepping."""
        if self.done:
            return None
        if self.cur_started_by == "reversal":
            # Reversal sids read their OWN entity's prior sid, never the sibling;
            # the apply idx is just a monotonic ordering key.
            return self.cur_start_trig
        b = self._boundary_for_trigger(self.cur_trigger)
        if b is not None:
            return b
        return self.cur_start_trig if self.cur_start_trig is not None else self.cur_start

    def step(self) -> None:
        """Build ONE sid (the current cur_* sid) and advance the cursor."""
        if self.done:
            return
        self.guard += 1
        if self.guard > self.max_chain:
            self.done = True
            return

        # Lazy start resolution (bootstrap AND subsequents read the sibling —
        # resolution is deferred to here, where the two-entity driver has built
        # the sibling through this sid's boundary). Reversal sids set cur_start
        # explicitly when advancing, so they skip this block.
        if self.cur_start is None:
            while True:
                m15_start, valid = _resolve_trigger_m15_start(
                    self.cur_trigger, self.parent_df, self.entity_df,
                    sibling_entity_df=self._sibling_df,
                )
                if m15_start is not None:
                    self.cur_start = m15_start
                    self.cur_valid = valid
                    if self.cur_start_trig is None:
                        # Bootstrap: the start candle IS the lifecycle trigger.
                        # (Subsequents already have cur_start_trig = boundary.)
                        self.cur_start_trig = m15_start
                    break
                # Resolution failed (mapping / degenerate / unavailable / pending).
                if self.sub_sid == 0:
                    # Bootstrap couldn't resolve → no chain for this cycle.
                    print(
                        f"[chain] no sid=0 for {self.cur_trigger.use_case} "
                        f"sid={self.cur_trigger.parent_sid} "
                        f"cycle={self.cur_trigger.parent_cycle_id} "
                        f"(probe pending / mapping failed) — cycle skipped"
                    )
                    self.done = True
                    return
                # A subsequent failed to resolve → skip it and try the NEXT
                # pending subsequent (don't kill the chain tail). The prior sid
                # keeps its already-applied bound at the skipped subsequent's
                # boundary (a small uncovered gap — logged, rare after the
                # sibling-CTS fix + ad-hoc fallback).
                print(
                    f"[chain] subsequent {self.cur_trigger.use_case} "
                    f"sid={self.cur_trigger.parent_sid} "
                    f"cycle={self.cur_trigger.parent_cycle_id} "
                    f"(boundary {self.cur_start_trig}) failed to resolve — "
                    f"skipping to next pending subsequent"
                )
                _floor = self.cur_start_trig if self.cur_start_trig is not None else -1
                nxt = next(
                    (s for s in self.pending if s["boundary"] > _floor), None
                )
                if nxt is None:
                    self.done = True
                    return
                self.pending.remove(nxt)
                self.cur_trigger = nxt["trigger"]
                self.cur_started_by = nxt["trigger"].use_case
                self.cur_sd = int(nxt["trigger"].lower_sd)
                self.cur_start_trig = nxt["boundary"]
                # loop to resolve the new cur

        next_sub = next(
            (s for s in self.pending if s["boundary"] > self.cur_start), None
        )
        bound = self.cycle_end_m15
        if next_sub is not None:
            bound = min(bound, next_sub["boundary"])

        # cap_open: open data edge — no subsequent after it AND parent cycle
        # still active (`lifecycle_end_idx is None`). PART4 §5.
        cap_open = (next_sub is None) and (self.bootstrap.lifecycle_end_idx is None)

        outcome = build_one_sid(
            self.entity_df,
            start_m15_abs=self.cur_start, sd=self.cur_sd, end_m15_abs=bound,
            sub_path_id=self.sub_path_id, timeframe=self.cur_trigger.lower_tf,
            trigger=self.cur_trigger, sub_sid=self.sub_sid,
            started_by=self.cur_started_by, start_trigger_idx=self.cur_start_trig,
            validated_parent_idx=self.cur_valid,
            parent_floor_m15=self.parent_floor_m15,
            cap_open=cap_open,
        )
        if outcome is None:
            # Degenerate / failed sid ends the chain (conservative).
            self.done = True
            return

        self.results.append(outcome.result)
        self.sub_sid += 1

        # Decide the next sid. Reversal (earlier than the next subsequent by
        # construction) wins if its handoff start is usable; else the pending
        # subsequent; else the cycle is done.
        if (outcome.reversal_idx_abs is not None
                and outcome.next_start_abs is not None
                and outcome.next_start_abs < bound
                and outcome.next_start_abs < len(self.entity_df)):
            self.cur_start = outcome.next_start_abs
            self.cur_sd = int(outcome.next_sd)
            self.cur_trigger = _synth_reversal_trigger(
                self.bootstrap, self.cur_sd, outcome.reversal_idx_abs,
            )
            self.cur_started_by = "reversal"
            self.cur_start_trig = outcome.reversal_idx_abs
            self.cur_valid = None
            return

        if next_sub is not None and bound == next_sub["boundary"]:
            self.pending.remove(next_sub)
            self.cur_trigger = next_sub["trigger"]
            self.cur_started_by = next_sub["trigger"].use_case
            self.cur_sd = int(next_sub["trigger"].lower_sd)
            # Lazy resolution: leave cur_start None so the top-of-step block
            # resolves this subsequent's probe NOW (sibling built through its
            # boundary by the driver). cur_start_trig = boundary (the sub's
            # lifecycle-start trigger candle); cur_valid filled on resolve.
            self.cur_start = None
            self.cur_start_trig = next_sub["boundary"]
            self.cur_valid = None
            return

        self.done = True


def _assert_m15_frames_aligned(
    a: Optional[pd.DataFrame], b: Optional[pd.DataFrame],
) -> None:
    """Guard: the two M15 entity dfs MUST share one entity-absolute index frame.

    The whole cross-entity sibling mechanism (subsequent_* reading the other
    entity's CTS by entity-absolute M15 idx) is only correct if the same idx
    means the same candle in both dfs. Both are fetched over the same
    ``(pair, M15, h1_start, h1_end)`` range, so this should always hold; assert
    it so a future fetch-range drift fails loudly instead of silently
    mis-referencing.
    """
    if a is None or b is None:
        return
    if len(a) != len(b):
        raise ValueError(
            f"[two_entity] M15 frame length mismatch: {len(a)} vs {len(b)}"
        )
    if a.empty or b.empty:
        return
    if (str(a["time"].iloc[0]) != str(b["time"].iloc[0])
            or str(a["time"].iloc[-1]) != str(b["time"].iloc[-1])):
        raise ValueError(
            "[two_entity] M15 frame time bounds mismatch "
            f"(conf {a['time'].iloc[0]}..{a['time'].iloc[-1]} vs "
            f"ctr {b['time'].iloc[0]}..{b['time'].iloc[-1]})"
        )


def build_two_entity_parent_cycle(
    parent_df: pd.DataFrame,
    *,
    conf_df: pd.DataFrame,
    conf_bootstrap: Optional[MultiTFTrigger],
    conf_subs: List[MultiTFTrigger],
    conf_path: str,
    ctr_df: pd.DataFrame,
    ctr_bootstrap: Optional[MultiTFTrigger],
    ctr_subs: List[MultiTFTrigger],
    ctr_path: str,
    parent_cycle_floor_h1: Optional[Dict[tuple, int]] = None,
) -> Tuple[List[LowerTFResult], List[LowerTFResult]]:
    """Build one parent cycle's confluence + counter chains INTERLEAVED in
    cadence order, each reading the other entity_df as sibling (Part 4 §6.1 +
    unified-probe Phase 1 Session 3).

    Two `_ChainCursor`s are advanced in lock-step: at each step the cursor with
    the smaller `next_boundary` builds one sid (mutating its own entity_df,
    reading the other as sibling). This guarantees that when a sid's probe
    reads the sibling's CTS within `[…, this_trigger]`, every sibling sid with
    an earlier boundary is already built — which is what makes the
    ``subsequent_*`` sibling-CTS reference resolvable (Step 2/3). A cycle may
    have a bootstrap in only one entity (the other cursor is None).

    Returns ``(conf_results, ctr_results)`` in per-entity build order.
    """
    _assert_m15_frames_aligned(conf_df, ctr_df)

    conf_cursor = (
        _ChainCursor(
            conf_df, parent_df,
            bootstrap=conf_bootstrap, subsequents=conf_subs,
            sub_path_id=conf_path,
            parent_cycle_floor_h1=parent_cycle_floor_h1,
            sibling_entity_df=ctr_df,
        )
        if conf_bootstrap is not None else None
    )
    ctr_cursor = (
        _ChainCursor(
            ctr_df, parent_df,
            bootstrap=ctr_bootstrap, subsequents=ctr_subs,
            sub_path_id=ctr_path,
            parent_cycle_floor_h1=parent_cycle_floor_h1,
            sibling_entity_df=conf_df,
        )
        if ctr_bootstrap is not None else None
    )

    cursors = [c for c in (conf_cursor, ctr_cursor) if c is not None]
    _INF = float("inf")
    # Stable tie-break (confluence before counter) — cadence makes ties
    # essentially impossible (sd-prox vs CTS-prox land on distinct candles),
    # but keep it deterministic. Index in `cursors` preserves that order.
    while True:
        active = [c for c in cursors if not c.done]
        if not active:
            break
        nxt = min(
            active,
            key=lambda c: (
                c.next_boundary if c.next_boundary is not None else _INF,
                cursors.index(c),
            ),
        )
        nxt.step()

    conf_results = conf_cursor.results if conf_cursor is not None else []
    ctr_results = ctr_cursor.results if ctr_cursor is not None else []
    return conf_results, ctr_results
