"""Entity-df mutation primitives for Part 4 §13.5.c (Option A — mutate
in place).

c.iii scope (current):

  - `apply_trigger_to_entity_df` — runs the parent-TF probe, maps the
    validated parent idx to entity-absolute lower-TF idx, builds a
    slice + lookback for MS compute, then calls
    `compute_structure_from_start` on the slice, runs the downstream
    pipeline, applies lifecycle cap, mirrors all artifacts back to
    `entity_df.attrs[...]` with entity-absolute idx and `entity_sid`
    attribution. Returns the slice-shape `LowerTFResult` so the
    orchestrator can compute parent-driven sub WVMI on it before
    persisting via `persist_facade_wvmi_to_entity_df`.

  - `mirror_lower_tf_result_to_entity_df` — translates the slice-shape
    `LowerTFResult` into entity-absolute snapshots and appends them to
    `entity_df.attrs["events" / "kl_zones" / "poi_zones" /
    "fib_states" / "wave_candles" / "wvmi" / "prev_bos_lines"]`.

  - `_tag_old_sid_on_overwrite` — cascade helper invoked from
    `mirror_lower_tf_result_to_entity_df` when `prior_sid_id` is set.

The c.ii-era facade builders (`_build_facade_lower_tf_result` +
`_shift_*` helpers) were removed in c.iii — the chart consumer reads
`entity_df.attrs[...]` directly and no longer needs a slice-shape
LowerTFResult-style projection of the new sid.

Persistence model (spec §7):
- df columns are "current truth" — overwritten in place by new sid.
- `entity_df.attrs["events"]` is append-only. New sid's events
  append; old sid's events stay with their original attribution.
- `entity_df.attrs["kl_zones" / "poi_zones" / "fib_states" /
  "wave_candles" / "wvmi" / "prev_bos_lines"]` keep all sids' entries;
  old-sid entries carry `deactivated_by="overwritten_by_sid_{N}"` after
  a cascade.

Cascade rules (spec §6.1, codified by `_tag_old_sid_on_overwrite`):
- KL / POI zones with `meta["entity_sid"] == prior_sid_id` get
  `meta["deactivated_by"]` and `meta["active"] = False`. If
  `end_time` is None or > boundary_time, cap at boundary_time.
- Fib states get `meta["deactivated_by"]` and `meta["deactivated_at"]
  = boundary_idx`; still-active-unlocked fibs flip `active=False`.
- WVMI records that are still `lp_locked=False` get
  `lp_locked=True` and `meta["lp_locked_by"]`.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, Dict, Optional

import pandas as pd

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


def _tag_old_sid_on_overwrite(
    entity_df: pd.DataFrame,
    prior_sid_id: int,
    new_sid_id: int,
    boundary_idx: int,
) -> None:
    """Tag prior-sid snapshots on `entity_df.attrs[...]` per spec §6.1.

    Cascade contents:
      - KL zones / POI zones with prior sid → `meta["deactivated_by"]`,
        `meta["active"] = False`, end_time capped at `boundary_time` if
        still open.
      - Fib states → `meta["deactivated_by"]`, `meta["deactivated_at"]`
        = `boundary_idx`; still-active-unlocked fibs flip
        `active=False`.
      - WVMI records still `lp_locked=False` → `lp_locked=True`,
        `meta["lp_locked_by"]`.

    No-op for snapshot lists not present on `entity_df.attrs`.

    For c.i pilot in production this is called with `prior_sid_id=None`
    (a no-op via the caller-side guard), so the only exercise is the
    unit test in `tests/test_entity_df_mutation.py`. Once c.ii lands and
    multiple sids accumulate per entity, this is the production cascade
    path.
    """
    boundary_time = pd.to_datetime(entity_df.loc[boundary_idx, "time"], utc=True)
    deactivated_tag = f"overwritten_by_sid_{new_sid_id}"

    # KL zones
    new_kl: list = []
    for z in entity_df.attrs.get("kl_zones", []):
        if z.meta.get("entity_sid") == prior_sid_id:
            new_meta = {**z.meta, "deactivated_by": deactivated_tag, "active": False}
            new_end = z.end_time
            if z.end_time is None or z.end_time > boundary_time:
                new_end = boundary_time
            z = replace(z, end_time=new_end, meta=new_meta)
        new_kl.append(z)
    entity_df.attrs["kl_zones"] = new_kl

    # POI zones
    new_poi: list = []
    for z in entity_df.attrs.get("poi_zones", []):
        if z.meta.get("entity_sid") == prior_sid_id:
            new_meta = {**z.meta, "deactivated_by": deactivated_tag, "active": False}
            new_end = z.end_time
            if z.end_time is None or z.end_time > boundary_time:
                new_end = boundary_time
            z = replace(z, end_time=new_end, meta=new_meta)
        new_poi.append(z)
    entity_df.attrs["poi_zones"] = new_poi

    # Fib states
    new_fibs: list = []
    for fib in entity_df.attrs.get("fib_states", []):
        if fib.meta.get("entity_sid") == prior_sid_id:
            new_meta = {
                **fib.meta,
                "deactivated_by": deactivated_tag,
                "deactivated_at": boundary_idx,
            }
            new_active = fib.active and fib.locked
            fib = replace(fib, active=new_active, meta=new_meta)
        new_fibs.append(fib)
    entity_df.attrs["fib_states"] = new_fibs

    # WVMI records (mutable dataclass — mutate in place)
    for w in entity_df.attrs.get("wvmi", []):
        if w.meta.get("entity_sid") == prior_sid_id and not w.lp_locked:
            w.lp_locked = True
            w.meta["lp_locked_by"] = deactivated_tag


def mirror_lower_tf_result_to_entity_df(
    entity_df: pd.DataFrame,
    result: LowerTFResult,
    *,
    new_sid_id: int,
    structure_path_id: str,
    prior_sid_id: Optional[int] = None,
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

    Cascade (§6.1) runs first if `prior_sid_id` is set — tags prior-sid
    snapshots already present on `entity_df.attrs` with `deactivated_by`
    and locks open WVMI.
    """
    slice_begin = int(result.meta.get("slice_begin", 0))

    # 1. Cascade — no-op if prior_sid_id is None
    if prior_sid_id is not None:
        boundary_idx = int(result.meta.get("m15_start_idx", slice_begin))
        _tag_old_sid_on_overwrite(
            entity_df, prior_sid_id, new_sid_id, boundary_idx,
        )

    attribution = {
        "entity_sid": new_sid_id,
        "structure_path_id": structure_path_id,
        "use_case": result.trigger.use_case,
        "parent_sid": result.trigger.parent_sid,
        "parent_cycle_id": result.trigger.parent_cycle_id,
        "timeframe": result.trigger.lower_tf,
    }

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

    # 4. KL zones — idx only in meta (and bounds_steps[*]["idx"])
    new_kl = []
    for z in result.kl_zones:
        new_meta = _shift_meta_indices(z.meta, _ZONE_META_IDX_KEYS, slice_begin)
        if "bounds_steps" in new_meta:
            steps = []
            for step in new_meta["bounds_steps"]:
                new_step = dict(step)
                if isinstance(new_step.get("idx"), int):
                    new_step["idx"] = new_step["idx"] + slice_begin
                steps.append(new_step)
            new_meta["bounds_steps"] = steps
        new_meta.update(attribution)
        new_kl.append(replace(z, meta=new_meta))
    _attrs_setdefault_list(entity_df, "kl_zones").extend(new_kl)

    # 5. POI zones — direct ic_idx + meta
    new_poi = []
    for z in result.poi_zones:
        new_meta = _shift_meta_indices(z.meta, _ZONE_META_IDX_KEYS, slice_begin)
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


def apply_trigger_to_entity_df(
    entity_df: pd.DataFrame,
    trigger: MultiTFTrigger,
    parent_df: pd.DataFrame,
    *,
    new_sid_id: int,
    structure_path_id: str,
    prior_sid_id: Optional[int] = None,
) -> Optional[LowerTFResult]:
    """Apply ONE trigger to the entity df: slice-compute → cascade → mirror.

    The c.ii core primitive. Achieves §6.1 / §6.2 in-place overwrite
    semantics on `entity_df.attrs[...]` while still using the slice +
    50-candle lookback + `reset_index` machinery internally for the MS
    compute. (MS has deep assumptions — BreakoutPatterns scanning the
    full df, rewind-from-0, state machine state evolving across the
    whole df — that prevent direct `compute_structure_from_start` on
    the entity df without a larger MS refactor; that refactor is
    deferred. The slice-elimination optimization stays open; the
    cascade semantics — the actual goal of c.ii — land here.)

    Flow:

    1. Parent-TF subordinate probe → validated parent idx.
    2. Map parent idx to entity-absolute lower-TF idx
       (`mapping_sd = -trigger.lower_sd` per §4.3.1).
    3. Determine entity-absolute lifecycle end idx.
    4. Build the slice with 50-candle lookback, `reset_index(drop=True)`,
       re-`compute_imbalance` (slice-local idx).
    5. Run `compute_structure_from_start` on the slice (no probes).
    6. Run downstream pipeline (KL zones BOS-only, Fib cross_cycle,
       POI; sub WVMI deferred to orchestrator per §8.3 / §8.4).
    7. Lifecycle-cap still-open zones / POIs / fibs.
    8. Cascade prior sid's snapshots on `entity_df.attrs[...]` if
       `prior_sid_id` is set (§6.1 same-type overwrite).
    9. Mirror everything back to `entity_df.attrs[...]` with
       entity-absolute idx via `mirror_lower_tf_result_to_entity_df`.
       Sub WVMI is filled in later by the orchestrator and persisted
       via `persist_facade_wvmi_to_entity_df`.
    10. Return the slice-shape `LowerTFResult` for the chart consumer.

    Returns the result or None on probe / mapping / structure failure.
    """
    from dataclasses import replace as _replace
    from datetime import timedelta

    from engine_v2.multitf.data_bridge import map_candle_to_lower_tf
    from engine_v2.multitf.lower_tf_pipeline import (
        _find_m15_lifecycle_end,
        _run_subordinate_probe,
    )
    from engine_v2.patterns.imbalance import compute_imbalance
    from engine_v2.pipeline.orchestrator import _run_downstream_pipeline
    from engine_v2.structure.structure_engine import compute_structure_from_start

    # 1. Parent-TF probe
    validated_parent_idx = _run_subordinate_probe(trigger, parent_df)
    if validated_parent_idx is None:
        return None

    # 2. Map validated parent idx to entity-absolute lower-TF idx
    parent_start_time = pd.to_datetime(
        parent_df.loc[validated_parent_idx, "time"], utc=True,
    )
    mapping_sd = -trigger.lower_sd
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
        return None

    # 3. Lifecycle end (entity-absolute)
    m15_end_idx = _find_m15_lifecycle_end(trigger, entity_df, parent_df)
    if m15_end_idx is None:
        m15_end_idx = int(entity_df.index[-1])
    if m15_start_idx >= m15_end_idx:
        print(
            f"[entity_compute] WARNING: degenerate window for "
            f"{trigger.use_case} sid={trigger.parent_sid} "
            f"cycle={trigger.parent_cycle_id}: "
            f"start={m15_start_idx} end={m15_end_idx}"
        )
        return None

    # 4. Build slice (50-candle lookback) and re-compute imbalances on it.
    #
    # Drop any structure cols that prior applies' mirror may have written
    # to entity_df. `mirror_lower_tf_result_to_entity_df` initializes new
    # cols with `pd.NA` outside the prior sid's range — those NA values
    # would survive the slice + reset_index and crash MS's debug
    # `float(self.df.at[prev_row, "range_hi"])` (NA passes the
    # `is not None` check). MS's `_ensure_output_cols` only initializes
    # cols that don't already exist, so we must drop them entirely before
    # passing the slice to MS.
    lookback = 50
    slice_begin = max(0, m15_start_idx - lookback)
    trigger_df = entity_df.iloc[slice_begin:m15_end_idx + 1].copy()
    trigger_df = trigger_df.reset_index(drop=True)
    trigger_df = compute_imbalance(trigger_df)
    cols_to_drop = [
        c for c in (_STRUCTURE_COLS + _MS_AUX_STRUCTURE_COLS)
        if c in trigger_df.columns
    ]
    if cols_to_drop:
        trigger_df = trigger_df.drop(columns=cols_to_drop, errors="ignore")

    start_in_slice = m15_start_idx - slice_begin
    if len(trigger_df) - start_in_slice < 5:
        print(
            f"[entity_compute] WARNING: M15 slice too small "
            f"({len(trigger_df) - start_in_slice} candles after start) "
            f"for {trigger.use_case} sid={trigger.parent_sid} "
            f"cycle={trigger.parent_cycle_id}"
        )
        return None

    # 5. Compute structure from validated start (no probes)
    try:
        m15_result = compute_structure_from_start(
            trigger_df,
            start_idx=start_in_slice,
            struct_direction=trigger.lower_sd,
            timeframe=trigger.lower_tf,
        )
    except (ValueError, IndexError) as exc:
        print(
            f"[entity_compute] WARNING: structure failed for "
            f"{trigger.use_case} sid={trigger.parent_sid} "
            f"cycle={trigger.parent_cycle_id}: {exc}"
        )
        return None

    m15_result.df.attrs["imbalances"] = trigger_df.attrs.get("imbalances", [])

    # 6. Downstream pipeline on the slice
    if trigger.use_case in ("first_counter", "subsequent_counter"):
        sub_path_id = "H1.main >> M15.counter"
    elif trigger.use_case in ("first_confluence", "subsequent_confluence"):
        sub_path_id = "H1.main >> M15.confluence"
    else:
        sub_path_id = structure_path_id

    downstream = _run_downstream_pipeline(
        m15_result.df,
        m15_result.events,
        m15_result.struct_direction,
        source_kinds=["BOS"],
        fib_mode="cross_cycle",
        log_prefix=f"M15_sid{new_sid_id}_{trigger.use_case}",
        timeframe=trigger.lower_tf,
        structure_path_id=sub_path_id,
        skip_wvmi=True,
    )

    # 7. Inject use-case-level attribution and lifecycle-cap open artifacts
    attribution: Dict[str, Any] = {
        "timeframe": trigger.lower_tf,
        "use_case": trigger.use_case,
        "parent_tf": trigger.parent_tf,
        "parent_sid": trigger.parent_sid,
        "parent_cycle_id": trigger.parent_cycle_id,
    }
    for ev in m15_result.events:
        ev.meta.update(attribution)
    for zone in downstream["kl_zones"]:
        zone.meta.update(attribution)

    last_time = pd.to_datetime(m15_result.df["time"].iloc[-1], utc=True)
    capped_zones = []
    for zone in downstream["kl_zones"]:
        if zone.end_time is None:
            zone = _replace(
                zone,
                end_time=last_time,
                meta={**zone.meta, "active": False,
                      "deactivated_by": "lifecycle_end"},
            )
        capped_zones.append(zone)

    capped_pois = []
    for poi in downstream["poi_zones"]:
        if poi.end_time is None:
            poi = _replace(
                poi,
                end_time=last_time,
                meta={**poi.meta, "active": False,
                      "deactivated_by": "lifecycle_end"},
            )
        capped_pois.append(poi)

    last_idx = int(m15_result.df.index[-1])
    capped_fibs = []
    for fib in downstream["fib_states"]:
        if fib.active and not fib.locked:
            fib = _replace(
                fib,
                active=False,
                meta={**fib.meta, "deactivated_by": "lifecycle_end",
                      "deactivated_at": last_idx},
            )
        capped_fibs.append(fib)

    result = LowerTFResult(
        trigger=trigger,
        df=m15_result.df,
        events=m15_result.events,
        kl_zones=capped_zones,
        wave_candles=downstream["wave_candles"],
        fib_states=capped_fibs,
        poi_zones=capped_pois,
        wvmi_records=downstream["wvmi_records"],   # empty (skip_wvmi=True)
        prev_bos_lines=downstream["prev_bos_lines"],
        status="finalized",
        meta={
            "m15_start_idx": m15_start_idx,
            "m15_end_idx": m15_end_idx,
            "m15_candle_count": len(trigger_df),
            "validated_h1_start": validated_parent_idx,
            "slice_begin": slice_begin,
            **attribution,
        },
    )

    # 8 + 9. Cascade prior sid + mirror new sid's snapshots into
    # entity_df.attrs with entity-absolute idx. The cascade is
    # mirror's responsibility when `prior_sid_id` is set.
    mirror_lower_tf_result_to_entity_df(
        entity_df,
        result,
        new_sid_id=new_sid_id,
        structure_path_id=structure_path_id,
        prior_sid_id=prior_sid_id,
    )

    print(
        f"[entity_compute] {trigger.use_case} entity_sid={new_sid_id} "
        f"parent_sid={trigger.parent_sid} cycle={trigger.parent_cycle_id} "
        f"start={m15_start_idx} end={m15_end_idx} "
        f"events={len(m15_result.events)} kl={len(capped_zones)} "
        f"poi={len(capped_pois)} fib={len(capped_fibs)}"
    )

    return result
