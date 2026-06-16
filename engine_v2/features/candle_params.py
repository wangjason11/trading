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

    # Absolute-pip floor on real-body length. Any candle whose body is SHORTER
    # than this floor is reclassified `pinbar` (trumping maru/normal/pinbar),
    # regardless of body_pct — a body that small carries no directional
    # conviction. Applied LAST in `classify_candles` (after the special-maru
    # promotion). 0.0 = disabled (legacy: candle_type purely body_pct-driven).
    # Per-TF defaults are wired by `apply_candle_classification(timeframe=...)`
    # via the `DEFAULT_PINBAR_BODY_PIP_FLOOR_BY_TF` table in
    # `candle_classifier.py`. Pip size is resolved per-pair (JPY=0.01 else
    # 0.0001) inside `classify_candles`. See features/candles_v2.py.
    #
    # NOTE: this replaced the former `big_body_pip_floor`, which gated the
    # `is_big_maru` flag only. That gate is now redundant — small-bodied
    # candles can never be maru in the first place, so they never reach the
    # big-maru ratio race nor the prior-maru pool. PROVISIONAL — see
    # `engine_v2/features/CANDLE_BODY_FLOOR_NOTES.md` (old vs new logic + open
    # questions; may be tuned/reworked).
    pinbar_body_pip_floor: float = 0.0

    special_maru: float = 0.5
    special_maru_distance: float = 0.1
