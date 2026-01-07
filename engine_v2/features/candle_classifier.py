from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from engine_v2.common.types import REQUIRED_CANDLE_COLS


@dataclass
class CandleClassifierResult:
    df: pd.DataFrame
    notes: str = ""


def apply_candle_classification(df: pd.DataFrame) -> CandleClassifierResult:
    """
    Adapter boundary: canonical df -> legacy candle classification.
    Expected behavior:
      - returns df with additional candle-type columns (whatever legacy produces)
    """
    _validate_input(df)

    # --- LEGACY HOOK (edit this block only) -----------------------
    #
    from engine_v2.features.candle_params import CandleParams
    from engine_v2.features.candles_v2 import compute_candle_features

    # params = CandleParams(
    #     maru=0.7,
    #     pinbar=0.5,
    #     pinbar_distance=0.5,
    #     big_maru_threshold=0.7,
    #     big_normal_threshold=0.5,
    #     lookback=5,
    #     special_maru=0.5,
    #     special_maru_distance=0.1,
    # )
    params = CandleParams()

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
