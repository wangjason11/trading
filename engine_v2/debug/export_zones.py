from __future__ import annotations
from pathlib import Path
import pandas as pd
from engine_v2.common.types import KLZone

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
