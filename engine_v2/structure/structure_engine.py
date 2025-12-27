from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from engine_v2.common.types import REQUIRED_CANDLE_COLS, StructureLevel


@dataclass
class StructureEngineResult:
    df: pd.DataFrame
    levels: List[StructureLevel]
    notes: str = ""


def compute_structure(df: pd.DataFrame) -> StructureEngineResult:
    """
    Adapter boundary: df(with patterns/features) -> BOS/CTS levels (and/or df columns).
    """
    _validate_input(df)

    # --- LEGACY HOOK (edit this block only) -----------------------
    df2 = df.copy()
    levels = []
    notes = "Structure engine is stubbed in Week 1 Day 4 (levels empty)."
    # --------------------------------------------------------------

    _validate_output(df2)
    return StructureEngineResult(df=df2, levels=levels, notes=notes)


def _validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_CANDLE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[structure_engine] Missing required columns: {missing}")
    if df.empty:
        raise ValueError("[structure_engine] Input df is empty")


def _validate_output(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_CANDLE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[structure_engine] Output df missing required columns: {missing}")
