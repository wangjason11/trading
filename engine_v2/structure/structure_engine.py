from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from engine_v2.common.types import REQUIRED_CANDLE_COLS, StructureLevel
from engine_v2.structure.market_structure import StructureEvent


@dataclass
class StructureEngineResult:
    df: pd.DataFrame
    levels: List[StructureLevel]
    events: List[StructureEvent]
    struct_direction: int
    notes: str = ""


def compute_structure(df: pd.DataFrame) -> StructureEngineResult:
    """
    Adapter boundary: df(with patterns/features) -> MarketStructure outputs:
      - df2 (with structure columns)
      - ms_events (StructureEvent list)
      - levels (StructureLevel list)

    Week 6 note:
      - struct_direction is still hardcoded here for now (1) until Week 6 Part 2
        where we implement IdentifyStart / candle of analysis logic.
    """
    _validate_input(df)

    # --- LEGACY HOOK (edit this block only) -----------------------
    from engine_v2.structure.market_structure import MarketStructure

    # For now, we pass struct_direction from upstream later (Week 6 Part 2 will compute it).
    # Temporary: assume struct_direction=1 for your current replay window (uptrend).
    struct_direction = 1

    ms = MarketStructure(df, struct_direction=struct_direction)
    ms.debug = True  # temp: enable reversal debug prints
    df2, ms_events, levels = ms.run()

    notes = f"MarketStructure v1: struct_direction={struct_direction}, events={len(ms_events)}, levels={len(levels)}"
    # --------------------------------------------------------------

    _validate_output(df2)

    return StructureEngineResult(
        df=df2,
        levels=levels,
        events=ms_events,
        struct_direction=struct_direction,
        notes=notes,
    )


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
