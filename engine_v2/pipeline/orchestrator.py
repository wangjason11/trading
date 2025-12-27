from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd

from engine_v2.common.types import PatternEvent, StructureLevel, REQUIRED_CANDLE_COLS
from engine_v2.features.candle_classifier import apply_candle_classification
from engine_v2.patterns.pattern_engine import detect_patterns
from engine_v2.structure.structure_engine import compute_structure


@dataclass
class PipelineResult:
    df: pd.DataFrame
    patterns: List[PatternEvent]
    structure: List[StructureLevel]
    meta: Dict[str, Any]


def run_pipeline(df: pd.DataFrame) -> PipelineResult:
    """
    Week 1 orchestrator:
      input df -> candles -> patterns -> structure
    Zones/entries come later.
    """
    _validate_input(df)

    c_res = apply_candle_classification(df)
    p_res = detect_patterns(c_res.df)
    s_res = compute_structure(p_res.df)

    meta = {
        "notes": {
            "candles": c_res.notes,
            "patterns": p_res.notes,
            "structure": s_res.notes,
        }
    }

    return PipelineResult(
        df=s_res.df,
        patterns=p_res.events,
        structure=s_res.levels,
        meta=meta,
    )


def _validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_CANDLE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[pipeline] Missing required columns: {missing}")
    if df.empty:
        raise ValueError("[pipeline] Input df is empty")
