from __future__ import annotations
from pathlib import Path
import pandas as pd
from engine_v2.common.types import KLZone


def export_poi_zones(zones: list, path: str | Path) -> None:
    """Export POI zones (one row per IC; variants listed in `versions`).

    POI zones are otherwise chart-only — this is the inspectable surface for
    `/compare` + debugging (mirrors `export_kl_zones`). Key meta fields are
    flattened out; the full meta is kept in the `meta` column. Columns are
    explicit so an empty list still writes a stable header.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for z in zones:
        m = getattr(z, "meta", {}) or {}
        rows.append({
            "ic_idx": getattr(z, "ic_idx", None),
            "start_time": getattr(z, "start_time", None),
            "end_time": getattr(z, "end_time", None),
            "side": getattr(z, "side", None),
            "top": getattr(z, "top", None),
            "bottom": getattr(z, "bottom", None),
            "structure_id": m.get("structure_id"),
            "cycle_id": m.get("cycle_id"),
            "parent_sid": m.get("parent_sid"),
            "parent_cycle_id": m.get("parent_cycle_id"),
            "confirmed_idx": m.get("confirmed_idx"),
            "versions": m.get("versions"),
            "status": m.get("status"),
            "meta": m,
        })
    cols = [
        "ic_idx", "start_time", "end_time", "side", "top", "bottom",
        "structure_id", "cycle_id", "parent_sid", "parent_cycle_id",
        "confirmed_idx", "versions", "status", "meta",
    ]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def export_kl_zones(zones: list[KLZone], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{
        "start_time": z.start_time,
        "end_time": z.end_time,
        "side": z.side,
        "top": z.top,
        "bottom": z.bottom,
        "source_kind": z.source_kind,
        "source_time": z.source_time,
        "source_price": z.source_price,
        "strength": z.strength,
        "meta": z.meta,
    } for z in zones])
    df.to_csv(path, index=False)
