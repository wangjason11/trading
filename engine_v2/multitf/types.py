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
    end_reason: Optional[str]         # "reversal" | "lifecycle_end" | None
    parent_sid: Optional[int] = None         # None for main
    parent_cycle_id: Optional[int] = None    # None for main
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SubsequentConfluenceTrigger:
    """`subsequent_confluence` (var 3) trigger — fires on parent CTS-prox
    after a parent sd-prox.

    Per spec §4.3.4:
      Trigger:    parent CTS-zone proximity AND prior trigger was sd-zone
      Idx input:  parent-TF candle in `[prior_sd_idx, this_cts_prox_idx]`
                  with extreme toward parent BOS (lowest low for bullish
                  parent, highest high for bearish)
      Probe end:  this CTS-prox trigger candle
      Probe sd:   +parent_sd (confluence)

    Reference-zone resolution (§4.3.4 step 1-5) is documented but not
    consumed by the probe — the probe derives its own BOS_0 internally
    from the data. The `meta` carries the prior_sd_idx for diagnostics.
    """
    parent_tf: str
    parent_sid: int
    parent_cycle_id: int               # parent cycle this trigger fires within
    parent_sd: int
    input_idx: int                     # parent-TF window extreme toward BOS
    end_idx: int                       # the CTS-prox trigger candle
    trigger_event_idx: int             # same as end_idx (CTS-prox candle)
    lifecycle_end_idx: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SubsequentCounterTrigger:
    """`subsequent_counter` (var 4) trigger — fires on parent sd-prox that
    forms a Λ (bullish parent) / V (bearish parent) with the prior CTS-prox
    and the sd-prox before that.

    Per spec §4.3.5:
      Trigger:    parent sd-zone proximity AND prior trigger was CTS-zone
                  proximity AND the proximity trigger before that was
                  sd-zone (sequence sd → CTS → sd in the cycle's
                  alternating list)
      Idx input:  parent-TF candle in `[prior_sd_idx, this_sd_prox_idx]`
                  with extreme toward parent CTS (highest high for
                  bullish parent / lowest low for bearish — Λ apex / V
                  trough)
      Probe end:  this sd-prox trigger candle
      Probe sd:   -parent_sd (counter — opposite direction to parent)
      Reference:  active parent CTS zone (descriptive; probe derives its
                  own BOS_0 internally)

    By the alternation invariant in `zones/zone_proximity.py`, every sd
    trigger past the first one (index ≥ 2 in the cycle's list) satisfies
    the [sd, opp_sd, sd] window — its immediate predecessor is opp_sd
    (== CTS-prox, var 3) and the one before that is sd (var 2 or an
    earlier var 4). `meta` carries `prior_sd_trigger_idx` and
    `prior_cts_prox_idx` for diagnostics.
    """
    parent_tf: str
    parent_sid: int
    parent_cycle_id: int
    parent_sd: int
    input_idx: int                     # parent-TF window extreme toward CTS
    end_idx: int                       # the sd-prox trigger candle
    trigger_event_idx: int             # same as end_idx (sd-prox candle)
    lifecycle_end_idx: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FirstConfluenceTrigger:
    """`first_confluence` (var 1) trigger — fires on parent BOS_CONFIRMED.

    Per spec §4.3.2: probe runs on parent TF with sd = +parent_sd,
    input_idx = parent BOS extreme, end_idx = parent CTS_CONFIRMED in the
    same parent cycle. `end_idx` is None (pending) until that CTS_CONFIRMED
    fires; per spec §14 / §16.6 a pending sub is hidden entirely until
    end_idx resolves.

    `lifecycle_end_idx` bounds the sub's lifetime — it ends at the next
    parent BOS for cycle_id+1 (the cycle this trigger belongs to ends
    when the next cycle's BOS confirms) or at the parent's reversal.
    """
    parent_tf: str
    parent_sid: int
    parent_cycle_id: int               # same cycle_id as the BOS_CONFIRMED
    parent_sd: int                     # parent struct_direction at trigger time
    input_idx: int                     # BOS extreme idx (== BOS_CONFIRMED.ev.idx)
    end_idx: Optional[int]             # CTS_CONFIRMED idx in same cycle; None = pending
    trigger_event_idx: int             # BOS_CONFIRMED.confirmed_at (candle when trigger fires)
    lifecycle_end_idx: Optional[int] = None  # next BOS (cycle+1) or reversal idx; None = parent cycle still open
    status: str = "finalized"          # "finalized" once end_idx is known, else "pending"
    meta: Dict[str, Any] = field(default_factory=dict)
