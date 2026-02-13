from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Sequence, TypedDict, Optional

# ---------------------------
# Canonical column names
# ---------------------------
# All OHLC dataframes in engine_v2 must use these column names.
COL_TIME = "time"
COL_O = "o"
COL_H = "h"
COL_L = "l"
COL_C = "c"
COL_V = "volume"

REQUIRED_CANDLE_COLS = (COL_TIME, COL_O, COL_H, COL_L, COL_C)

Direction = Literal[-1, 0, 1]  # -1 bearish, 1 bullish, 0 neutral/unknown


class PatternStatus(str, Enum):
    NONE = "NONE"
    SUCCESS = "SUCCESS"
    FAIL_NEEDS_CONFIRM = "FAIL_NEEDS_CONFIRM"
    CONFIRMED = "CONFIRMED"


@dataclass(frozen=True)
class PatternEvent:
    """
    Discrete multi-candle pattern occurrence.

    Backward-compatible:
      - time/meta still allowed
    Structure-pattern compatible (Week 4):
      - start_idx/end_idx/status/confirmation fields added
    """
    # Optional timestamp (keep for charting / later usage)
    time: Any | None = None

    name: str = ""
    direction: Direction = 0

    # Week 4 structure-pattern fields
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None
    status: PatternStatus = PatternStatus.NONE

    confirmation_threshold: Optional[float] = None
    confirmation_idx: Optional[int] = None
    break_threshold_used: Optional[float] = None

    meta: Dict[str, Any] = field(default_factory=dict)
    debug: Optional[Dict[str, Any]] = None

@dataclass(frozen=True)
class StructureLevel:
    """A BOS/CTS (or other structure) horizontal level."""

    time: Any
    kind: Literal["BOS", "CTS"]
    direction: Direction
    price: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Zone:
    """A zone (KL or POI) used for context and/or entries."""

    id: str
    zone_type: Literal["KL", "POI"]
    timeframe: str
    formed_at: Any
    low: float
    high: float
    status: Literal["active", "mitigated", "broken", "expired"] = "active"
    strength_score: float = 0.0
    strength_flags: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TradeIntent:
    """A planned trade (not necessarily executed yet)."""

    id: str
    timeframe: str
    formed_at: Any
    direction: Direction
    entry: float
    stop: float
    tps: Sequence[float]
    rr: float
    meta: Dict[str, Any] = field(default_factory=dict)


class ChartMarker(TypedDict, total=False):
    time: Any
    text: str
    position: Literal["aboveBar", "belowBar"]
    meta: Dict[str, Any]


class ChartLine(TypedDict, total=False):
    price: float
    text: str
    meta: Dict[str, Any]


class ChartRect(TypedDict, total=False):
    # rectangle spanning time interval [t0, t1] and price interval [low, high]
    t0: Any
    t1: Any
    low: float
    high: float
    text: str
    meta: Dict[str, Any]

@dataclass(frozen=True)
class KLZone:
    start_time: "pd.Timestamp"
    end_time: Optional["pd.Timestamp"]  # None = extends to end of chart
    side: Literal["buy", "sell"]
    top: float
    bottom: float
    source_kind: Literal["BOS", "CTS"]
    source_time: "pd.Timestamp"
    source_price: float
    strength: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WVMIRecord:
    """Wave Volume Momentum Indicator for a BOS zone."""
    # Attribution to main structure's BOS zone
    bos_structure_id: int
    bos_cycle_id: int
    zone_side: Literal["buy", "sell"]
    source: Literal["main", "scenario3"]

    # Wave candle indices
    fb_idx: Optional[int] = None      # First Breakout (from BOS_n)
    lb_idx: Optional[int] = None      # Last Breakout (from CTS_n)
    fp_idx: Optional[int] = None      # First Pullback (from CTS_n)
    lp_idx: Optional[int] = None      # Last Pullback (temporary or locked)

    # Raw volumes
    fb_volume: Optional[float] = None
    lb_volume: Optional[float] = None
    fp_volume: Optional[float] = None
    lp_volume: Optional[float] = None

    # Weights (last candles only)
    lb_weight: float = 1.0
    lp_weight: float = 1.0

    # Computed ratios
    breakout_momentum: Optional[float] = None
    pullback_momentum: Optional[float] = None

    # Direction-agnostic labels
    buy_momentum: Optional[float] = None
    sell_momentum: Optional[float] = None

    # Lifecycle
    status: Literal["created", "updated", "locked"] = "created"
    lp_locked: bool = False
    locked_by_cycle_id: Optional[int] = None

    # Metadata
    meta: Dict[str, Any] = field(default_factory=dict)
