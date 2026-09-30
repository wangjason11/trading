from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from engine_v2.common.types import ImbalanceInstance


def export_imbalance_instances(
    instances: Iterable[ImbalanceInstance],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([
        {
            "start_idx": inst.start_idx,
            "end_idx": inst.end_idx,
            "direction": inst.direction,
            "gap_top": inst.gap_top,
            "gap_bottom": inst.gap_bottom,
            "gap_size": inst.gap_size,
            "meta": inst.meta,
        }
        for inst in instances
    ])
    df.to_csv(path, index=False)
