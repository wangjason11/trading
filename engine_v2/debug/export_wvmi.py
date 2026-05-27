from __future__ import annotations

from pathlib import Path
import pandas as pd

from engine_v2.common.types import WVMIRecord


# Explicit column order so an empty export still emits a usable header.
# Sub identity is the tuple (parent_sid, parent_cycle_id, sub_sid) — its
# three components are first-class columns (sub_sid alone is meaningless).
_COLUMNS = [
    "structure_path_id",
    "parent_sid",
    "parent_cycle_id",
    "sub_sid",
    "started_by",
    "bos_structure_id",
    "bos_cycle_id",
    "zone_side",
    # Lifecycle convention (derived status + scalar start/end) followed by the
    # computation axis (lp_status/lp_locked) — WVMI_SPEC "Lifecycle convention".
    "status",
    "start_idx",
    "end_idx",
    "end_reason",
    "lp_status",
    "lp_locked",
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
    # (LANDMINE "WVMI Records Carry Mixed-Coordinate Meta") — never translated.
    "triggered_by_event_idx",
    "triggered_by_event_type",
    "parent_path_id",
    "meta",
]


def export_wvmi(records: list[WVMIRecord], path: str | Path) -> None:
    """Export WVMI records to CSV (main + sub entities).

    Flattens the §8.7 attribution + identity tuple (parent_sid /
    parent_cycle_id / sub_sid / started_by) out of `meta` into their own
    columns, keeping the full `meta` dict as the last column. Wave-candle idx
    fields are entity-df coords; `triggered_by_event_idx` is parent-df coords
    for subs.
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
            "sub_sid": m.get("sub_sid"),
            "started_by": m.get("started_by"),
            "bos_structure_id": r.bos_structure_id,
            "bos_cycle_id": r.bos_cycle_id,
            "zone_side": r.zone_side,
            "status": r.status,
            "start_idx": r.start_idx,
            "end_idx": r.end_idx,
            "end_reason": r.end_reason,
            "lp_status": r.lp_status,
            "lp_locked": r.lp_locked,
            "locked_by_cycle_id": r.locked_by_cycle_id,
            "fb_idx": r.fb_idx,
            "lb_idx": r.lb_idx,
            "fp_idx": r.fp_idx,
            "lp_idx": r.lp_idx,
            "breakout_momentum": r.breakout_momentum,
            "pullback_momentum": r.pullback_momentum,
            "buy_momentum": r.buy_momentum,
            "sell_momentum": r.sell_momentum,
            "triggered_by_event_idx": m.get("triggered_by_event_idx"),
            "triggered_by_event_type": m.get("triggered_by_event_type"),
            "parent_path_id": m.get("parent_path_id"),
            "meta": m,
        })

    pd.DataFrame(rows, columns=_COLUMNS).to_csv(path, index=False)
