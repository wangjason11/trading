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

    @property
    def formed_at(self) -> int:
        """The moment the instance's gap exists: the close of its FIRST c3
        (`start_idx + 1`). A merged run keeps growing — its `end_idx` and merged
        bounds are final only at `end_idx + 1` — but from `formed_at` a live
        engine already sees its formed prefix. See IMBALANCE_FILL_SEMANTICS.md
        "Knowability — the c3 rule"."""
        return self.start_idx + 1

    def overlaps_formed_prefix(self, start_idx: int, end_idx: int, evaluated_at: int) -> bool:
        """`overlaps` as known at the moment `evaluated_at`: False before
        `formed_at`; otherwise the formed prefix `[start_idx, min(end_idx,
        evaluated_at - 1)]` (the c2s whose c3 has closed) must intersect the
        inclusive `[start_idx, end_idx]` window."""
        if self.formed_at > evaluated_at:
            return False
        formed_end_idx = min(self.end_idx, evaluated_at - 1)
        return self.start_idx <= end_idx and formed_end_idx >= start_idx

    def __deepcopy__(self, memo):
        """Return self instead of a deep copy — a deliberate performance escape.

        `ImbalanceInstance`s live in `df.attrs["imbalances"]`. pandas'
        `NDFrame.__finalize__` does `self.attrs = deepcopy(other.attrs)` on
        virtually every derived object (column boxing, astype, isna, Series
        arithmetic, slicing, ...), so the per-sub downstream pipeline was paying
        to deep-copy this whole instance list — recursing into each `meta` dict —
        on essentially every pandas operation (the dominant remaining cost after
        the range_label fix; see GOTCHAS "Per-cell `.iloc[]` ... deepcopy
        explosion" + the profile that motivated this).

        Returning `self` is byte-identical because engine code only ever
        reads/mutates the ORIGINAL instances: `compute_imbalance` creates them,
        and each working df pins the SAME list by reference
        (`entity_df_mutation.py`: `bounded.df.attrs["imbalances"] =
        trigger_df.attrs["imbalances"]`). pandas' transient `__finalize__`
        copies are never read by engine code, so sharing identity with them
        changes only cost, not values. The only in-place mutation of `meta`
        (`_compute_fill_idx_cache` stashing `armed_idx`/`confirmed_fill_idx`)
        happens on those originals either way. Validated byte-identical via
        `/compare` + the full test suite (POI activation reads that meta cache,
        so any leak would surface as a POI diff).
        """
        return self

    def is_filled(
        self,
        df: "pd.DataFrame",
        check_to_idx: int,
        fill_threshold: float = 0.70,
    ) -> bool:
        """Check if the merged gap is committed-filled by candles in
        (end_idx, check_to_idx].

        Two-stroke state machine:
          Stroke 1 (armed): first candle retracing >= fill_threshold into the gap.
            Bullish: candle.low  <= gap_top    - gap_size * fill_threshold
            Bearish: candle.high >= gap_bottom + gap_size * fill_threshold
          Stroke 2 (confirmed): first candle at idx >= armed_idx where the close
          passes the gap outer in the imbalance's direction.
            Bullish: candle.close >= gap_top
            Bearish: candle.close <= gap_bottom

        Filled iff both strokes occur within (end_idx, check_to_idx]. Both can
        fire on the same candle (rare). Each stroke latches on first occurrence;
        the state machine is monotonic (0 -> armed -> filled, no regression).

        Degenerate gap (gap_size <= 0): returns True (safe default — treat
          invalid gap as filled).
        Empty scan range (end_idx >= check_to_idx): returns False (unfilled) —
          correct for a FORMED prefix (its only scanned candle, its own c3,
          cannot arm its gap), but an instance that has not formed yet is not
          an imbalance at all: a caller asking at a moment must apply the
          `formed_at` cut first (`has_unfilled_imbalance(evaluated_at=...)`).
        """
        if self.gap_size <= 0:
            return True  # invalid gap -> treat as filled (safe default)

        # Fast path: positional numpy scan when the working df is a contiguous
        # 0..n-1 RangeIndex (always in practice — sub slices are reset_index'd
        # and the H1 main df is a 0..n-1 RangeIndex). Avoids the ~35x-slower
        # `df.loc[idx, col]` scalar lookups + `idx not in df.index`
        # (RangeIndex.__contains__) in the hot per-candle loop; is_filled is
        # called ~O(1e5) times per replay across the Fib/POI/scenario imbalance
        # checks. Byte-identical to the label-based loop in the fallback below
        # (same ascending (end_idx, check_to_idx] scan clamped to the data,
        # same two-stroke armed/confirmed latch, same inclusive comparisons).
        # Duck-typed RangeIndex check (types.py has no runtime `pd` import).
        n = len(df)
        _index = df.index
        contiguous = (
            getattr(_index, "step", None) == 1
            and getattr(_index, "start", None) == 0
            and getattr(_index, "stop", None) == n
        )
        if contiguous:
            lo = self.end_idx + 1
            if lo < 0:
                lo = 0
            hi = check_to_idx
            if hi > n - 1:
                hi = n - 1
            if lo > hi:
                return False
            if self.direction == 1:
                l_arr = df["l"].to_numpy(dtype=float)
                c_arr = df["c"].to_numpy(dtype=float)
                stroke1_level = self.gap_top - self.gap_size * fill_threshold
                stroke2_level = self.gap_top
                armed = False
                for pos in range(lo, hi + 1):
                    if not armed and l_arr[pos] <= stroke1_level:
                        armed = True
                    if armed and c_arr[pos] >= stroke2_level:
                        return True
            elif self.direction == -1:
                h_arr = df["h"].to_numpy(dtype=float)
                c_arr = df["c"].to_numpy(dtype=float)
                stroke1_level = self.gap_bottom + self.gap_size * fill_threshold
                stroke2_level = self.gap_bottom
                armed = False
                for pos in range(lo, hi + 1):
                    if not armed and h_arr[pos] >= stroke1_level:
                        armed = True
                    if armed and c_arr[pos] <= stroke2_level:
                        return True
            return False

        # Fallback: label-based loop for a non-contiguous index (rare/never in
        # practice). Verbatim pre-optimization implementation.
        if self.direction == 1:
            stroke1_level = self.gap_top - self.gap_size * fill_threshold
            stroke2_level = self.gap_top
            armed = False
            for idx in range(self.end_idx + 1, check_to_idx + 1):
                if idx not in df.index:
                    continue
                if not armed and float(df.loc[idx, "l"]) <= stroke1_level:
                    armed = True
                if armed and float(df.loc[idx, "c"]) >= stroke2_level:
                    return True
        elif self.direction == -1:
            stroke1_level = self.gap_bottom + self.gap_size * fill_threshold
            stroke2_level = self.gap_bottom
            armed = False
            for idx in range(self.end_idx + 1, check_to_idx + 1):
                if idx not in df.index:
                    continue
                if not armed and float(df.loc[idx, "h"]) >= stroke1_level:
                    armed = True
                if armed and float(df.loc[idx, "c"]) <= stroke2_level:
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

    # Lifecycle (LP-finalization computation state)
    status: Literal["created", "updated", "locked"] = "created"
    lp_locked: bool = False
    locked_by_cycle_id: Optional[int] = None

    # Metadata
    meta: Dict[str, Any] = field(default_factory=dict)
