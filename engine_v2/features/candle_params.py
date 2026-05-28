from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class CandleParams:
    maru: float = 0.65
    pinbar: float = 0.4
    pinbar_distance: float = 0.5
    normal_distance: float = 0.4

    big_maru_threshold: float = 0.65
    big_normal_threshold: float = 0.5
    lookback: int = 5

    # Additive absolute-pip floor on body length for the `is_big_maru` flag
    # ONLY (NOT `is_big_normal` — kept on the legacy ratio-only gate). For
    # is_big_maru: both the rolling-max ratio gate (>= big_maru_threshold)
    # AND this body-pip floor must pass. 0.0 = no floor (legacy behavior).
    # Per-TF defaults are wired by `apply_candle_classification(timeframe=...)`
    # via the `DEFAULT_BIG_BODY_PIP_FLOOR_BY_TF` table in
    # `candle_classifier.py`. See features/candles_v2.py:classify_big_flags
    # for the call site.
    big_body_pip_floor: float = 0.0

    special_maru: float = 0.5
    special_maru_distance: float = 0.1
