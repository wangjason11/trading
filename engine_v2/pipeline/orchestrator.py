from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd

from engine_v2.common.types import PatternEvent, StructureLevel, REQUIRED_CANDLE_COLS
from engine_v2.features.candle_classifier import apply_candle_classification
from engine_v2.patterns.pattern_engine import detect_patterns
from engine_v2.structure.structure_engine import compute_structure

# NEW: KL base features computed BEFORE structure
from engine_v2.zones.kl_zones_v1 import compute_base_features, derive_kl_zones_v1


@dataclass
class PipelineResult:
    df: pd.DataFrame
    patterns: List[PatternEvent]
    structure: List[StructureLevel]
    meta: Dict[str, Any]


def run_pipeline(df: pd.DataFrame) -> PipelineResult:
    """
    Orchestrator (Week 6 KL Zones ordering):

      input df
        -> candle classification (candles_v2 features)
        -> pattern engine
        -> KL base feature enrichment (base_pattern/base bounds features)
        -> market structure (df + structure_events)
        -> KL zones derived from structure confirmation events
        -> attach df.attrs["kl_zones"] for charting

    Zones remain event-driven and do not add rewinds/waits.
    """
    _validate_input(df)

    # 1) Candle features
    c_res = apply_candle_classification(df)

    # 2) Pattern engine (structure patterns used by market structure)
    p_res = detect_patterns(c_res.df)

    # ✅ Debug: confirm structure-pattern markers exist
    print("[patterns]", p_res.notes)
    print(p_res.df["pat"].value_counts().head())

    # 3) KL base features (must occur BEFORE structure)
    # Default length_threshold locked to 0.7 (per Week 6 spec)
    df_with_base = compute_base_features(p_res.df, length_threshold=0.7)

    # 4) Market structure (must return events + struct_direction)
    s_res = compute_structure(df_with_base)

    meta = {
        "notes": {
            "candles": c_res.notes,
            "patterns": p_res.notes,
            "structure": s_res.notes,
        }
    }

    # 5) KL zones consume structure events (not levels)
    kl_zones = derive_kl_zones_v1(
        s_res.df,
        s_res.events,
        struct_direction=s_res.struct_direction,
        length_threshold=0.7,
    )

    print("[kl_zones] total=", len(kl_zones))
    if kl_zones:
        from collections import Counter
        print("[kl_zones] base_pattern counts:", Counter([z.meta.get("base_pattern") for z in kl_zones]).most_common(10))
        print("[kl_zones] active buy:", sum(1 for z in kl_zones if z.side=="buy" and z.meta.get("active")))
        print("[kl_zones] active sell:", sum(1 for z in kl_zones if z.side=="sell" and z.meta.get("active")))

    meta["kl_zones"] = kl_zones

    # For chart overlay (export_plotly reads df.attrs["kl_zones"] and df.attrs["structure_events"])
    s_res.df.attrs["kl_zones"] = kl_zones
    s_res.df.attrs["structure_events"] = s_res.events

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
