from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from engine_v2.common.types import REQUIRED_CANDLE_COLS


# Per-TF absolute body-pip floor for `is_big_maru` ONLY (NOT `is_big_normal` —
# kept on the legacy ratio-only gate by design; pattern alt-paths that gate on
# `c1.is_big_normal_as1` should keep qualifying even when c1's body is small).
# Additive with the ratio gate (CandleParams.big_maru_threshold): both must
# pass for `is_big_maru_as*` = True. Calibrated to typical body lengths at
# each TF — H1 candles range ~5-15 pips, M15 ~1-5, M5 ~0.5-2. The floor filters
# out candles that win the ratio race only because surrounding marus are even
# smaller (a recurring noise source in low-volatility stretches). Pip size is
# resolved per-pair (JPY=0.01 else 0.0001) inside `classify_big_flags`.
DEFAULT_BIG_BODY_PIP_FLOOR_BY_TF: dict[str, float] = {
    "H1": 3.0,
    "M15": 2.0,
    "M5": 1.0,
}


@dataclass
class CandleClassifierResult:
    df: pd.DataFrame
    notes: str = ""


def apply_candle_classification(
    df: pd.DataFrame,
    *,
    timeframe: str = "H1",
) -> CandleClassifierResult:
    """
    Adapter boundary: canonical df -> legacy candle classification.
    Expected behavior:
      - returns df with additional candle-type columns (whatever legacy produces)

    `timeframe` selects the per-TF absolute body-pip floor for the big_maru /
    big_normal flags (see `DEFAULT_BIG_BODY_PIP_FLOOR_BY_TF`). Unknown TFs
    degrade to no floor (legacy ratio-only gate). Default "H1" matches the
    H1-main call site which doesn't pass it.
    """
    _validate_input(df)

    # --- LEGACY HOOK (edit this block only) -----------------------
    #
    from engine_v2.features.candle_params import CandleParams
    from engine_v2.features.candles_v2 import compute_candle_features

    body_pip_floor = DEFAULT_BIG_BODY_PIP_FLOOR_BY_TF.get(str(timeframe), 0.0)
    params = CandleParams(big_body_pip_floor=body_pip_floor)

    df = compute_candle_features(df, params, anchor_shifts=(0,1,2))
    notes = "Candle features computed via features.candles_v2"
    # --------------------------------------------------------------

    _validate_output(df)
    return CandleClassifierResult(df=df, notes=notes)


def _validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_CANDLE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[candle_classifier] Missing required columns: {missing}")
    if df.empty:
        raise ValueError("[candle_classifier] Input df is empty")


def _validate_output(df: pd.DataFrame) -> None:
    # Must preserve canonical columns at least
    missing = [c for c in REQUIRED_CANDLE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[candle_classifier] Output df missing required columns: {missing}")
    

    required_out = ["candle_type", "pinbar_dir", "body_pct", "candle_len"]
    missing = [c for c in required_out if c not in df.columns]
    if missing:
        raise ValueError(f"[candle_classifier] Missing derived columns: {missing}")
