from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from engine_v2.common.types import WVMIRecord
from engine_v2.multitf.types import SidRecord, sub_relative_dir_at


# Explicit column order so an empty export still emits a usable header.
# Sub identity is `sub_id` (the unique sub, PART4 §17.9); `parent_sid` /
# `parent_cycle_id` / `started_by` are the sub's FIRST record's — informational.
_COLUMNS = [
    "structure_path_id",
    "parent_sid",
    "parent_cycle_id",
    "sub_id",
    "started_by",
    "bos_structure_id",
    "bos_cycle_id",
    "zone_side",
    "status",
    "lp_locked",
    "cycle_collapsed",   # the record's cycle has an empty lifecycle window (Plan G G2)
    "locked_by_cycle_id",
    "fb_idx",
    "lb_idx",
    "fp_idx",
    "lp_idx",
    "breakout_momentum",
    "pullback_momentum",
    "buy_momentum",
    "sell_momentum",
    # §8.7 attribution. triggered_by_event_idx is PARENT-df coords for subs
    # (LANDMINE "WVMI Records Carry Mixed-Coordinate Meta") — never translated;
    # on a sub row it is the lens's first WVMI-class trigger inside the sub's
    # window, None (an empty cell) when none lands there (Plan G G4).
    "triggered_by_event_idx",
    "triggered_by_event_type",
    "parent_path_id",
    "meta",
]


# The integer columns — any of them may hold None (an H1 row has no sub identity; an
# unlocked record no `locked_by_cycle_id`; a missing wave candle no idx), so each
# is written as nullable `Int64`, never as a float.
_INT_COLUMNS = (
    "parent_sid", "parent_cycle_id", "sub_id", "bos_structure_id", "bos_cycle_id",
    "locked_by_cycle_id", "fb_idx", "lb_idx", "fp_idx", "lp_idx", "triggered_by_event_idx",
)


def export_wvmi(records: list[WVMIRecord], path: str | Path) -> None:
    """Export WVMI records to CSV (main + sub entities).

    Flattens the §8.7 attribution + sub identity (`sub_id`; parent_sid /
    parent_cycle_id / started_by informational) out of `meta` into their own
    columns, keeping the full `meta` dict as the last column. Wave-candle idx
    fields are entity-df coords; `triggered_by_event_idx` is parent-df coords
    for subs. Every integer column (`_INT_COLUMNS`) is written as pandas
    nullable `Int64`: a column mixing ints and None would otherwise render every
    int as a float (`710` -> `710.0`, `locked_by_cycle_id` `1` -> `1.0` on every
    locked row); a None is an empty cell (2026-09-30, Plan G + the user's Int64
    decision).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in records:
        m = r.meta or {}
        rows.append({
            "structure_path_id": r.structure_path_id,
            "parent_sid": m.get("parent_sid"),
            "parent_cycle_id": m.get("parent_cycle_id"),
            "sub_id": m.get("sub_id"),
            "started_by": m.get("started_by"),
            "bos_structure_id": r.bos_structure_id,
            "bos_cycle_id": r.bos_cycle_id,
            "zone_side": r.zone_side,
            "status": r.status,
            "lp_locked": r.lp_locked,
            "cycle_collapsed": r.cycle_collapsed,
            "locked_by_cycle_id": r.locked_by_cycle_id,
            "fb_idx": r.fb_idx,
            "lb_idx": r.lb_idx,
            "fp_idx": r.fp_idx,
            "lp_idx": r.lp_idx,
            "breakout_momentum": r.breakout_momentum,
            "pullback_momentum": r.pullback_momentum,
            "buy_momentum": r.buy_momentum,
            "sell_momentum": r.sell_momentum,
            # strict reads (LANDMINES "Event Contract Rules" rule 3, Plan G's meaning change on sub rows):
            # every exported record carries both keys — the main's attribution stamp, a sub copy's per-lens stamp (None
            # values when no trigger); a missing key is an unstamped record, never "no trigger".
            "triggered_by_event_idx": m["triggered_by_event_idx"],
            "triggered_by_event_type": m["triggered_by_event_type"],
            "parent_path_id": m.get("parent_path_id"),
            "meta": m,
        })

    out = pd.DataFrame(rows, columns=_COLUMNS)
    for col in _INT_COLUMNS:
        out[col] = out[col].astype("Int64")
    out.to_csv(path, index=False)


# The per-LP CSV ("WVMI back on the chart" step 4 — the user's round 8, 2026-10-26): the record keys named as in
# `_wvmi.csv`, then the row. `relative_dir` = the sub's at the LP candle (M15; empty on an H1 row).
_LP_COLUMNS = [
    "structure_path_id",
    "parent_sid",
    "parent_cycle_id",
    "sub_id",
    "bos_structure_id",
    "bos_cycle_id",
    "relative_dir",
    "zone_side",
    "lp_idx",
    "as_of_idx",
    "lp_status",
    "lp_status_reason",
    "lp_status_idx",
    "breakout_momentum",
    "pullback_momentum",
]

_LP_INT_COLUMNS = (
    "parent_sid", "parent_cycle_id", "sub_id", "bos_structure_id", "bos_cycle_id",
    "lp_idx", "as_of_idx", "lp_status_idx",
)


def export_wvmi_lps(records: list[WVMIRecord], path: str | Path, sid_records: Iterable[SidRecord] = ()) -> None:
    """Export the per-LP rows (`WVMIRecord.lp_rows`) to CSV — one row per dot: every
    temporary LP and the locked one, in record order then LP order (WVMI_SPEC "Per-LP
    rows"). A COLLAPSED record (`cycle_collapsed`) gets no rows (the user 2026-10-26:
    "for now collapsed records don't need to get per-LP rows"; it keeps its `_wvmi.csv`
    row). `relative_dir` on a sub row = `sub_relative_dir_at` at the LP candle over
    that sub's `SidRecord` (THIS lens's `attrs["sids"]`) — `NA (before start_idx)`
    before the sub is live (the user 2026-10-26); empty on an H1 row. Indices are
    entity-df coords; integer columns nullable `Int64` (as `export_wvmi`)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sid_by_sub_id = {int(s.sub_id): s for s in sid_records if s.sub_id is not None}

    rows = []
    for r in records:
        if r.cycle_collapsed:
            continue
        m = r.meta or {}
        sub_id = m.get("sub_id")
        sid_rec = sid_by_sub_id[int(sub_id)] if sub_id is not None else None
        for lp in r.lp_rows:
            rows.append({
                "structure_path_id": r.structure_path_id,
                "parent_sid": m.get("parent_sid"),
                "parent_cycle_id": m.get("parent_cycle_id"),
                "sub_id": sub_id,
                "bos_structure_id": r.bos_structure_id,
                "bos_cycle_id": r.bos_cycle_id,
                "relative_dir": (
                    sub_relative_dir_at(lp.lp_idx, sid_rec.relative_dir_segments, sid_rec.start_idx, int(sub_id))
                    if sid_rec is not None else None
                ),
                "zone_side": r.zone_side,
                "lp_idx": lp.lp_idx,
                "as_of_idx": lp.as_of_idx,
                "lp_status": lp.lp_status,
                "lp_status_reason": lp.lp_status_reason,
                "lp_status_idx": lp.lp_status_idx,
                "breakout_momentum": r.breakout_momentum,
                "pullback_momentum": lp.pullback_momentum,
            })

    out = pd.DataFrame(rows, columns=_LP_COLUMNS)
    for col in _LP_INT_COLUMNS:
        out[col] = out[col].astype("Int64")
    out.to_csv(path, index=False)
