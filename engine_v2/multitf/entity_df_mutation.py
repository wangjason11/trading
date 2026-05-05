"""Entity-df mutation primitives for Part 4 §13.5.c (Option A — mutate
in place).

c.i scope (this module's first version): mirror infrastructure +
cascade helper. Pairs each fresh `LowerTFResult` (built by
`run_lower_tf_pipeline` on a slice with slice-local idx) with a write
to `entity_df.attrs[...]` that uses entity-absolute idx and carries
`entity_sid` attribution. The structure columns from `result.df` are
also mirrored back into the entity df (rows
`[slice_begin, slice_begin+len(result.df)-1]`).

c.ii will replace this with `apply_trigger_to_entity_df` running
`compute_structure_from_start(entity_df, start_idx, end_idx=...)`
directly on the entity df, eliminating the slice + 50-candle lookback +
`reset_index` + per-trigger `compute_imbalance` machinery.

Persistence model (spec §7):
- df columns are "current truth" — overwritten in place by new sid.
- `entity_df.attrs["events"]` is append-only. New sid's events
  append; old sid's events stay with their original attribution.
- `entity_df.attrs["kl_zones" / "poi_zones" / "fib_states" /
  "wave_candles" / "wvmi"]` keep all sids' entries; old-sid entries
  carry `deactivated_by="overwritten_by_sid_{N}"` after a cascade.

Cascade rules (spec §6.1, codified by `_tag_old_sid_on_overwrite`):
- KL / POI zones with `meta["structure_id"] == prior_sid_id` get
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
from typing import Optional

import pandas as pd

from engine_v2.multitf.types import LowerTFResult


# Structure columns potentially written by MarketStructure; we mirror
# back any that exist on the slice df. Missing columns are skipped.
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
