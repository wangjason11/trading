from __future__ import annotations

from pathlib import Path
import pandas as pd

from engine_v2.structure.market_structure import StructureEvent


def export_structure_events(events: list[StructureEvent], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([
        {
            "idx": ev.idx,
            "category": ev.category,
            "type": ev.type,
            "price": ev.price,
            "meta": ev.meta,
        }
        for ev in events
    ])
    df.to_csv(path, index=False)
