from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from engine_v2.common.types import COL_H, COL_L, COL_TIME, Direction, StructureLevel


@dataclass(frozen=True)
class SwingPoint:
    time: pd.Timestamp
    kind: str  # "H" or "L"
    price: float
    idx: int


def detect_swings_fractal(df: pd.DataFrame, *, left: int = 2, right: int = 2) -> List[SwingPoint]:
    """
    Simple fractal swing detection:
      swing high if high is strictly greater than highs in [i-left, i+right]
      swing low  if low  is strictly lower  than lows  in [i-left, i+right]
    """
    h = df[COL_H].to_numpy(dtype=float)
    l = df[COL_L].to_numpy(dtype=float)
    t = pd.to_datetime(df[COL_TIME], utc=True)

    n = len(df)
    swings: List[SwingPoint] = []
    for i in range(left, n - right):
        hi = h[i]
        lo = l[i]

        if hi > np.max(h[i - left : i]) and hi >= np.max(h[i + 1 : i + 1 + right]):
            swings.append(SwingPoint(time=t.iloc[i], kind="H", price=float(hi), idx=i))

        if lo < np.min(l[i - left : i]) and lo <= np.min(l[i + 1 : i + 1 + right]):
            swings.append(SwingPoint(time=t.iloc[i], kind="L", price=float(lo), idx=i))

    # sort by index/time
    swings.sort(key=lambda s: s.idx)
    return swings


def swings_to_structure_levels(swings: List[SwingPoint]) -> List[StructureLevel]:
    """
    Very first pass:
      - BOS up: break above previous swing high
      - BOS down: break below previous swing low
      - CTS up: higher low after BOS up
      - CTS down: lower high after BOS down

    We output StructureLevel with kind BOS/CTS at the swing price.
    """
    levels: List[StructureLevel] = []
    last_high: SwingPoint | None = None
    last_low: SwingPoint | None = None

    trend: Direction = 0  # 1 up, -1 down, 0 unknown

    for sp in swings:
        if sp.kind == "H":
            if last_high is None:
                last_high = sp
                continue

            # BOS up if current high > previous high AND we weren't in downtrend
            if sp.price > last_high.price:
                levels.append(
                    StructureLevel(
                        time=sp.time,
                        kind="BOS",
                        direction=1,
                        price=sp.price,
                        meta={"from": last_high.price},
                    )
                )
                trend = 1

            # CTS down: in downtrend, lower high reinforces
            if trend == -1 and sp.price < last_high.price:
                levels.append(
                    StructureLevel(
                        time=sp.time,
                        kind="CTS",
                        direction=-1,
                        price=sp.price,
                        meta={"from": last_high.price},
                    )
                )

            last_high = sp

        else:  # "L"
            if last_low is None:
                last_low = sp
                continue

            # BOS down if current low < previous low
            if sp.price < last_low.price:
                levels.append(
                    StructureLevel(
                        time=sp.time,
                        kind="BOS",
                        direction=-1,
                        price=sp.price,
                        meta={"from": last_low.price},
                    )
                )
                trend = -1

            # CTS up: in uptrend, higher low reinforces
            if trend == 1 and sp.price > last_low.price:
                levels.append(
                    StructureLevel(
                        time=sp.time,
                        kind="CTS",
                        direction=1,
                        price=sp.price,
                        meta={"from": last_low.price},
                    )
                )

            last_low = sp

    return levels


def filter_swings_by_price_distance(swings: list[SwingPoint], *, min_dist: float) -> list[SwingPoint]:
    if not swings:
        return swings
    out = [swings[0]]
    for s in swings[1:]:
        if abs(s.price - out[-1].price) >= min_dist:
            out.append(s)
    return out


def dedupe_levels(levels: list[StructureLevel], *, band: float) -> list[StructureLevel]:
    out: list[StructureLevel] = []
    for lv in sorted(levels, key=lambda x: x.time):
        if not out:
            out.append(lv)
            continue
        if abs(lv.price - out[-1].price) >= band:
            out.append(lv)
    return out


def compute_structure_levels(df: pd.DataFrame, *, left: int = 2, right: int = 2) -> Tuple[List[SwingPoint], List[StructureLevel]]:
    swings = detect_swings_fractal(df, left=left, right=right)
    swings = filter_swings_by_price_distance(swings, min_dist=0.0004)  # start here for NZDUSD
    levels = swings_to_structure_levels(swings)
    levels = dedupe_levels(levels, band=0.0004)
    return swings, levels