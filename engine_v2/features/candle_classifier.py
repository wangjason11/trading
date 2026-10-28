from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from engine_v2.common.types import REQUIRED_CANDLE_COLS


# The pinbar body floor of `classify_candles` step 4 is the ROLLING floor (PLAN_U U2,
# the user 2026-10-27 / 28): each candle's floor = the p90 of the bodies of the 90
# most recent pinbars strictly before it (`CandleParams.pinbar_floor_*`,
# `features/pinbar_body_floor.py`). It replaced the fixed per-TF table
# (H1 2.2 / M15 1.0 / M5 0.5 pips, calibrated 2026-06-09) and its reader
# `candle_params_for_timeframe` — the table's history: CANDLE_BODY_FLOOR_NOTES
# "Previous logic". Every frame is classified with `CandleParams()`.


@dataclass
class CandleClassifierResult:
    df: pd.DataFrame
    notes: str = ""


def apply_candle_classification(
    df: pd.DataFrame,
    *,
    timeframe: str = "H1",
    floor_seed: Optional[pd.DataFrame] = None,
    body_floor=None,
) -> CandleClassifierResult:
    """
    Adapter boundary: canonical df -> legacy candle classification.
    Expected behavior:
      - returns df with additional candle-type columns (whatever legacy produces)

    Classifies with `CandleParams()`; the pinbar body floor is the rolling one
    over `floor_seed` (the history fetched before the frame —
    `run_replay.fetch_floor_seed`; None -> frame-only windows: the candles before
    the frame's 90th pinbar get no floor) + this frame. `body_floor` (one floor
    per row, price units) overrides it — a test seam (PLAN_U §8 A5).

    Stamps `df.attrs["timeframe"]` (a str; default "H1" = the H1-main call site,
    which doesn't pass it) — the institutional pattern's leg cap reads it
    (`config.ip_leg_cap_by_tf`).
    """
    _validate_input(df)

    # --- LEGACY HOOK (edit this block only) -----------------------
    #
    from engine_v2.features.candle_params import CandleParams
    from engine_v2.features.candles_v2 import compute_candle_features

    df = compute_candle_features(df, CandleParams(), anchor_shifts=(0,1,2),
                                 floor_seed=floor_seed, body_floor=body_floor)
    df.attrs["timeframe"] = str(timeframe)
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
