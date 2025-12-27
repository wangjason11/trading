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
    from engine_v2.legacy_2025_port.candles_port import compute_candle_features

    df2 = compute_candle_features(df)
    notes = "Candle features computed via legacy_2025_port.candles_port"
    # --------------------------------------------------------------

    _validate_output(df2)
    return CandleClassifierResult(df=df2, notes=notes)


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
