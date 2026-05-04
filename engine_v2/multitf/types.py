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
    use_case: str                     # "first_counter" (formerly "uc1_reverse")
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


@dataclass(frozen=True)
class SidRecord:
    """Per-sid record within an entity's df (`df.attrs["sids"]`).

    Spec §9.2: per-sid attribution lives in the df, not on EntityState. One
    record per `structure_id` (main) or per parent_cycle (subordinate).

    Indices use the entity df's own coordinate space.
    """
    sid: int
    starting_sd: int                  # +1 / -1
    creation_event_idx: Optional[int] # candle idx where this sid begins
    end_event_idx: Optional[int]      # candle idx where this sid ends (None = still active)
    end_reason: Optional[str]         # "reversal" | "lifecycle_end" | "overwritten_by_sid_..." | None
    parent_sid: Optional[int] = None         # None for main
    parent_cycle_id: Optional[int] = None    # None for main
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FirstConfluenceTrigger:
    """`first_confluence` (var 1) trigger — fires on parent BOS_CONFIRMED.

    Per spec §4.3.2: probe runs on parent TF with sd = +parent_sd,
    input_idx = parent BOS extreme, end_idx = parent CTS_CONFIRMED in the
    same parent cycle. `end_idx` is None (pending) until that CTS_CONFIRMED
    fires; per spec §14 / §16.6 a pending sub is hidden entirely until
    end_idx resolves.

    Detection only — 3a does not consume these triggers.
    """
    parent_tf: str
    parent_sid: int
    parent_cycle_id: int               # same cycle_id as the BOS_CONFIRMED
    parent_sd: int                     # parent struct_direction at trigger time
    input_idx: int                     # BOS extreme idx (== BOS_CONFIRMED.ev.idx)
    end_idx: Optional[int]             # CTS_CONFIRMED idx in same cycle; None = pending
    trigger_event_idx: int             # BOS_CONFIRMED.confirmed_at (candle when trigger fires)
    status: str = "finalized"          # "finalized" once end_idx is known, else "pending"
    meta: Dict[str, Any] = field(default_factory=dict)
