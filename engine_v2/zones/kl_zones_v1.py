from __future__ import annotations

from typing import List, Optional
import pandas as pd

from engine_v2.common.types import KLZone, StructureLevel
from engine_v2.common.types import COL_TIME, COL_O, COL_H, COL_L, COL_C


def derive_kl_zones_v1(df: pd.DataFrame, levels: List[StructureLevel]) -> List[KLZone]:
    """
    v1: derive KL zones from BOS/CTS structure levels.
    - BOS -> enter with trend direction (level.direction)
    - CTS -> enter against level.direction
    Zone height = candle segment at event (placeholder)
    """
    dfx = df.copy()
    dfx[COL_TIME] = pd.to_datetime(dfx[COL_TIME], utc=True)

    # Map timestamp -> row index for fast lookup
    # (We assume exact times match; if not, we'll use merge_asof later.)
    time_to_idx = {t: i for i, t in enumerate(dfx[COL_TIME])}

    zones: List[KLZone] = []

    for lv in levels:
        idx = time_to_idx.get(pd.to_datetime(lv.time, utc=True), None)
        if idx is None:
            continue  # skip if not found (we'll improve later)

        row = dfx.iloc[idx]
        o = float(row[COL_O]); h = float(row[COL_H]); l = float(row[COL_L]); c = float(row[COL_C])

        # Decide side based on BOS/CTS rule
        if lv.kind == "BOS":
            side = "buy" if lv.direction == 1 else "sell"
        else:  # CTS
            side = "sell" if lv.direction == 1 else "buy"

        # Zone bounds (placeholder)
        if side == "buy":
            bottom = l
            top = max(o, c)
        else:
            top = h
            bottom = min(o, c)

        zones.append(
            KLZone(
                start_time=pd.to_datetime(lv.time, utc=True),
                end_time=None,
                side=side,
                top=float(top),
                bottom=float(bottom),
                source_kind=lv.kind,
                source_time=pd.to_datetime(lv.time, utc=True),
                source_price=float(lv.price),
                strength=0.0,
                meta={"level_meta": lv.meta or {}},
            )
        )

    return zones
