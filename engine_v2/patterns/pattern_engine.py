from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

from engine_v2.common.types import PatternEvent, REQUIRED_CANDLE_COLS


@dataclass
class PatternEngineResult:
    df: pd.DataFrame
    events: List[PatternEvent]
    notes: str = ""


def detect_patterns(df: pd.DataFrame) -> PatternEngineResult:
    """
    Adapter boundary: canonical+features df -> pattern events.
    In Week 1 we allow either:
      A) events list (preferred), or
      B) legacy columns on df (we can convert to events later)
    """
    _validate_input(df)

    # --- LEGACY HOOK (edit this block only) -----------------------
    from engine_v2.legacy_2025_port.patterns_port import apply_basic_entry_patterns

    df2 = apply_basic_entry_patterns(df)
    events = []  # Week 1: we store pattern columns; events come later
    notes = "Basic entry patterns applied via legacy_2025_port.patterns_port"
    # --------------------------------------------------------------

    _validate_output(df2)
    return PatternEngineResult(df=df2, events=events, notes=notes)


def _validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_CANDLE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[pattern_engine] Missing required columns: {missing}")
    if df.empty:
        raise ValueError("[pattern_engine] Input df is empty")


def _validate_output(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_CANDLE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[pattern_engine] Output df missing required columns: {missing}")
