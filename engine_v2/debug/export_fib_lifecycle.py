from __future__ import annotations

from pathlib import Path
import pandas as pd


# Explicit column order so an empty export still emits a usable header.
# FibState has no `structure_path_id` field (unlike WVMIRecord), so the owning
# entity's path is passed in by the caller. For subs, parent identity lives in
# the mirrored `meta` (parent_sid / parent_cycle_id, added as attribution).
_COLUMNS = [
    "structure_path_id",
    "parent_sid",
    "parent_cycle_id",
    "structure_id",
    "cycle_id",
    "version",
    "start_idx",
    "end_idx",
    "end_reason",
    "status",
    "active",
    "locked",
    "bos_idx",
    "cts_idx",
    "fib_mode",
    "meta",
]


def export_fib_lifecycle(records, path, structure_path_id=None) -> None:
    """Export FibState lifecycle records to CSV (one entity per call).

    Surfaces the otherwise-LATENT fib lifecycle (no fib CSV existed; M15 fib
    lines are off) so the scalar `start_idx`/`end_idx`/`end_reason`/`status`
    axes (FIB_LIFECYCLE_SPEC §15) are inspectable. Idx fields are entity-df
    coords. One row per version record (cross fibs have multiple).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in records:
        m = f.meta or {}
        rows.append({
            "structure_path_id": structure_path_id,
            "parent_sid": m.get("parent_sid"),
            "parent_cycle_id": m.get("parent_cycle_id"),
            "structure_id": f.structure_id,
            "cycle_id": f.cycle_id,
            "version": m.get("version"),
            "start_idx": f.start_idx,
            "end_idx": f.end_idx,
            "end_reason": f.end_reason,
            "status": f.status,
            "active": f.active,
            "locked": f.locked,
            "bos_idx": f.bos_idx,
            "cts_idx": f.cts_idx,
            "fib_mode": m.get("fib_mode"),
            "meta": m,
        })

    pd.DataFrame(rows, columns=_COLUMNS).to_csv(path, index=False)
