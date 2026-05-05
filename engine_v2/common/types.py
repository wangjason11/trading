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
class ImbalanceInstance:
    """One merged FVG instance: either a single imbalance candle or a run
    of consecutive same-direction imbalance candles.

    start_idx / end_idx refer to the c2 (middle) candles at the ends of the run.
    gap_top / gap_bottom are the merged bounds:
      bullish: gap_bottom = df[start_idx-1].h (first c1), gap_top = df[end_idx+1].l (last c3)
      bearish: gap_bottom = df[end_idx+1].h (last c3),    gap_top = df[start_idx-1].l (first c1)
    """
    start_idx: int
    end_idx: int
    direction: Direction  # +1 bullish, -1 bearish
    gap_top: float
    gap_bottom: float
    gap_size: float
    meta: Dict[str, Any] = field(default_factory=dict)

    def overlaps(self, start_idx: int, end_idx: int) -> bool:
        """True if this instance intersects the inclusive [start_idx, end_idx] range."""
        return self.start_idx <= end_idx and self.end_idx >= start_idx

    def is_filled(
        self,
        df: "pd.DataFrame",
        check_to_idx: int,
        fill_threshold: float = 0.70,
    ) -> bool:
        """Check if the merged gap is filled by candles in (end_idx, check_to_idx].

        Bullish: filled when any candle.low <= gap_top - gap_size*threshold
        Bearish: filled when any candle.high >= gap_bottom + gap_size*threshold
        Empty scan range (end_idx >= check_to_idx): returns False (unfilled).
        """
        if self.gap_size <= 0:
            return True  # invalid gap -> treat as filled (safe default)

        if self.direction == 1:
            fill_level = self.gap_top - self.gap_size * fill_threshold
            for idx in range(self.end_idx + 1, check_to_idx + 1):
                if idx not in df.index:
                    continue
                if float(df.loc[idx, "l"]) <= fill_level:
                    return True
        elif self.direction == -1:
            fill_level = self.gap_bottom + self.gap_size * fill_threshold
            for idx in range(self.end_idx + 1, check_to_idx + 1):
                if idx not in df.index:
                    continue
                if float(df.loc[idx, "h"]) >= fill_level:
                    return True

        return False


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
    # Attribution to entity (Part 4 §8.7) and that entity's BOS zone
    bos_structure_id: int
    bos_cycle_id: int
    zone_side: Literal["buy", "sell"]
    # Entity path identifier (e.g., "H1.main", "H1.main >> M15.confluence").
    # None on records created by code paths that haven't been migrated yet.
    structure_path_id: Optional[str] = None

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
