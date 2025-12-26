from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Sequence, TypedDict

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

REQUIRED_CANDLE_COLS = (COL_TIME, COL_O, COL_H, COL_L, COL_C, COL_V)

Direction = Literal[-1, 0, 1]  # -1 bearish, 1 bullish, 0 neutral/unknown


@dataclass(frozen=True)
class PatternEvent:
    """Discrete multi-candle pattern occurrence."""

    time: Any  # typically pd.Timestamp; keep Any to avoid pandas hard dependency here
    name: str
    direction: Direction
    meta: Dict[str, Any] = field(default_factory=dict)


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
    """A zone (KL or OB) used for context and/or entries."""

    id: str
    zone_type: Literal["KL", "OB"]
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
