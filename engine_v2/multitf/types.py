"""Multi-timeframe data types for UC1 (reverse structure) and future use cases."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from engine_v2.zones.fib_tracker import FibState
from engine_v2.zones.wave_candles import WaveCandleResult
from engine_v2.common.types import KLZone, WVMIRecord
from engine_v2.structure.market_structure import StructureEvent


@dataclass
class MultiTFTrigger:
    """A trigger from higher TF that starts a lower TF structure."""
    parent_tf: str                    # "H1"
    parent_sid: int                   # structure_id in parent
    parent_cycle_id: int              # cycle_id in parent
    parent_sd: int                    # struct_direction in parent
    use_case: str                     # "uc1_reverse"
    lower_tf: str                     # "M15"
    lower_sd: int                     # struct_direction for lower TF (opposite for UC1)
    start_time: pd.Timestamp          # H1 CTS candle time -> mapped to M15
    start_price: float                # H1 CTS extreme price
    lifecycle_end_idx: Optional[int]  # H1 index where this lower TF run ends (None = open)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LowerTFResult:
    """Result of running a lower TF pipeline."""
    trigger: MultiTFTrigger
    df: pd.DataFrame                  # Lower-TF DataFrame with pipeline columns
    events: List[StructureEvent]      # StructureEvents (with attribution)
    kl_zones: List[KLZone]
    wave_candles: List[WaveCandleResult]
    fib_states: List[FibState]
    poi_zones: list
    wvmi_records: List[WVMIRecord]
    prev_bos_lines: list              # Previous BOS lines from downstream pipeline
    status: str                       # "finalized" or "pending"
    meta: Dict[str, Any] = field(default_factory=dict)
