from __future__ import annotations

from pathlib import Path
import pandas as pd

from engine_v2.structure.structure_v1 import SwingPoint
from engine_v2.common.types import StructureLevel


def export_swings(swings: list[SwingPoint], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        [
            {
                "time": s.time,
                "kind": s.kind,
                "price": s.price,
                "idx": s.idx,
            }
            for s in swings
        ]
    )
    df.to_csv(path, index=False)


def export_levels(levels: list[StructureLevel], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        [
            {
                "time": lv.time,
                "kind": lv.kind,
                "direction": lv.direction,
                "price": lv.price,
                "meta": lv.meta,
            }
            for lv in levels
        ]
    )
    df.to_csv(path, index=False)
