from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from engine_v2.common.types import REQUIRED_CANDLE_COLS


# PROVISIONAL (2026-06-16) — both these values AND the reclassify-to-pinbar
# approach are a working iteration, not settled logic; expected to be tuned /
# split per pair / possibly reworked. Full design record (current logic, the
# previous `big_body_pip_floor` logic it replaced, and the open questions) lives
# in `engine_v2/features/CANDLE_BODY_FLOOR_NOTES.md`.
#
# Per-TF absolute body-pip floor for the `pinbar` reclassification: any candle
# whose real body is shorter than this floor is classified `pinbar` (strict `<`),
# regardless of its body_pct (a body that small carries no directional
# conviction). Applied LAST in `classify_candles`, so it trumps maru/normal/pinbar.
# Pip size is resolved per-pair (JPY=0.01 else 0.0001) inside `classify_candles`.
#
# Calibrated 2026-06-09 (NZD_USD) via `debug/candle_size_distribution.py`, which
# prints per-TF body distributions + the demotion impact of each candidate floor.
# Body medians are roughly H1 2.8 / M15 1.5 / M5 0.9 pips and halve per TF step
# down, so the floors scale similarly rather than staying flat. These values give
# a balanced, gentle proportional bite (~4-5% of all candles, ~3% of marus,
# ~13-15% of normals at each TF). They REPLACE the original 3/2/1 trial, which was
# too aggressive at the lower TFs (M15=2 demoted ~23% of marus, M5=1 ~20% of all
# candles) and pushed an H1 sid=0 reversal ~190 candles late. NOT the same as the
# old `is_big_maru` `big_body_pip_floor` (that gate was removed). Down the line
# these may be tuned and/or split per currency pair as the strategy is optimized.
DEFAULT_PINBAR_BODY_PIP_FLOOR_BY_TF: dict[str, float] = {
    "H1": 2.20,
    "M15": 1.0,
    "M5": 0.5,
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

    `timeframe` selects the per-TF absolute body-pip floor for the `pinbar`
    reclassification (see `DEFAULT_PINBAR_BODY_PIP_FLOOR_BY_TF`). Unknown TFs
    degrade to no floor (candle_type purely body_pct-driven). Default "H1"
    matches the H1-main call site which doesn't pass it.
    """
    _validate_input(df)

    # --- LEGACY HOOK (edit this block only) -----------------------
    #
    from engine_v2.features.candle_params import CandleParams
    from engine_v2.features.candles_v2 import compute_candle_features

    body_pip_floor = DEFAULT_PINBAR_BODY_PIP_FLOOR_BY_TF.get(str(timeframe), 0.0)
    params = CandleParams(pinbar_body_pip_floor=body_pip_floor)

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
