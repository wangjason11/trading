from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class CandleParams:
    maru: float = 0.65
    pinbar: float = 0.4
    pinbar_distance: float = 0.5

    big_maru_threshold: float = 0.65
    big_normal_threshold: float = 0.5
    lookback: int = 5

    special_maru: float = 0.5
    special_maru_distance: float = 0.1
