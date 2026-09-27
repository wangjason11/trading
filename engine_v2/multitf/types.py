"""Multi-timeframe data types for UC1 (reverse structure) and future use cases."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
    """Per-sid record within an entity's df (`df.attrs["sids"]`) — ONE class,
    two roles (PART4 §9.4 / §17.9).

    - **main** (`build_sid_records_for_main`): one record per MarketStructure
      `structure_id`; `sub_sid = structure_id`, `sub_id = None`, parent fields
      None. Chart identity = `sub_sid`.
    - **sub** (`build_sid_records_for_subordinate`): one record per UNIQUE SUB
      (`sub_id` set, `sub_sid = None`, parent fields None — parent attribution
      lives on the TriggerRecord table `attrs["triggers"]`);
      `creation_event_idx = starting_idx` (the structural anchor, historical),
      `start_idx` / `end_event_idx = end_idx` (the real-time lifecycle window),
      `end_reason`, `lenses`, `relative_dir_segments`. Chart identity = `sub_id`;
      ownership = the lifecycle window per direction.

    `end_reason` vocabulary: `"reversal" | "same_dir_replacement" |
    "parent_end" | None` (Plan C; the pre-pool `"lifecycle_end"` is gone —
    `next_cycle` never reaches a SidRecord). Indices use the entity df's own
    coordinate space.
    """
    sub_sid: Optional[int]
    starting_sd: int                  # +1 / -1
    creation_event_idx: Optional[int] # main: first event idx; sub: starting_idx (anchor)
    end_event_idx: Optional[int]      # main: reversal apply idx; sub: lifecycle end_idx (None = open)
    end_reason: Optional[str]         # "reversal" | "same_dir_replacement" | "parent_end" | None
    parent_sid: Optional[int] = None         # None for main AND for subs (see docstring)
    parent_cycle_id: Optional[int] = None    # None for main AND for subs
    meta: Dict[str, Any] = field(default_factory=dict)
    sub_id: Optional[int] = None             # subs only — the chart identity
    start_idx: Optional[int] = None          # subs only — lifecycle start (real-time)
    lenses: Tuple[str, ...] = ()             # subs only — charts this sub draws on
    relative_dir_segments: Tuple[Tuple[int, str], ...] = ()   # subs only — §17.3 step function


@dataclass(frozen=True)
class SubsequentConfluenceTrigger:
    """`subsequent_confluence` (var 3) trigger — fires on parent CTS-prox
    after a parent sd-prox.

    Per spec §4.3.4:
      Trigger:    parent CTS-zone proximity AND prior trigger was sd-zone
      Idx input:  `input_idx` = parent-TF candle in `[prior_sd_idx,
                  this_cts_prox_idx]` with extreme toward parent BOS (lowest
                  low for bullish parent, highest high for bearish) — the
                  H1 `parent_input_idx`, informational since Session 3
      Probe end:  this CTS-prox trigger candle (its LOH = the sweep's `hi`)
      Probe sd:   +parent_sd (confluence)

    The M15 probe input AND reference zone are co-sourced from the sibling
    counter lens's most recent qualifying CTS in the sub-TF window
    (§4.3.4 steps 1–4, `_resolve_sibling_cts_via_unified_probe`); `meta`
    carries `prior_sd_trigger_idx`, which sets that window's `lo`.
    """
    parent_tf: str
    parent_sid: int
    parent_cycle_id: int               # parent cycle this trigger fires within
    parent_sd: int
    input_idx: int                     # parent-TF window extreme toward BOS
    trigger_event_idx: int             # the CTS-prox trigger candle (the probe ends at its M15 `hi`)
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
      Idx input:  `input_idx` = parent-TF candle in `[prior_sd_idx,
                  this_sd_prox_idx]` with extreme toward parent CTS
                  (highest high for bullish parent / lowest low for
                  bearish — Λ apex / V trough) — the H1 `parent_input_idx`,
                  informational since Session 3
      Probe end:  this sd-prox trigger candle (its LOH = the sweep's `hi`)
      Probe sd:   -parent_sd (counter — opposite direction to parent)
      Reference:  the sibling confluence lens's most recent qualifying CTS,
                  co-sourced with the M15 probe input (§4.3.5; §4.3.4
                  steps 1–4)

    By the alternation invariant in `zones/zone_proximity.py`, every sd
    trigger past the first one (index ≥ 2 in the cycle's list) satisfies
    the [sd, opp_sd, sd] window — its immediate predecessor is opp_sd
    (== CTS-prox, var 3) and the one before that is sd (var 2 or an
    earlier var 4). `meta` carries `prior_cts_prox_idx` (sets the
    sibling-read window's `lo`) and `prior_sd_trigger_idx` (diagnostic).
    """
    parent_tf: str
    parent_sid: int
    parent_cycle_id: int
    parent_sd: int
    input_idx: int                     # parent-TF window extreme toward CTS
    trigger_event_idx: int             # the sd-prox trigger candle (the probe ends at its M15 `hi`)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FirstConfluenceTrigger:
    """`first_confluence` (var 1) trigger — fires on parent BOS_CONFIRMED.

    Per spec §4.3.2: probe sd = +parent_sd, `input_idx` = parent BOS anchor,
    `parent_cts_anchor_idx` = the confirmed CTS's ANCHOR (`cts_anchor_idx`) in
    the same parent cycle (H1) — the FC resolver price-maps it into the probe's
    M15 search bound `probe_end_idx` (a compute bound, unrelated to the
    lifecycle `end_idx` of PART4 §17). Named `probe_end_idx` until Plan E
    Post-E·3 (2026-09-27; `end_idx` before Plan C). None (pending) until that
    CTS_CONFIRMED fires; a pending trigger is logged as
    `UnresolvedTrigger(reason="pending")`.

    The parent-cycle end lives in `multitf/parent_tables.py` (one helper for
    every record).
    """
    parent_tf: str
    parent_sid: int
    parent_cycle_id: int               # same cycle_id as the BOS_CONFIRMED
    parent_sd: int                     # parent struct_direction at trigger time
    input_idx: int                     # the BOS anchor (BOS_CONFIRMED.meta["bos_anchor_idx"])
    parent_cts_anchor_idx: Optional[int]   # H1 CTS anchor (cts_anchor_idx) in same cycle; None = pending
    trigger_event_idx: int             # BOS_CONFIRMED.confirmed_at (candle when trigger fires)
    status: str = "finalized"          # "finalized" once parent_cts_anchor_idx is known, else "pending"
    meta: Dict[str, Any] = field(default_factory=dict)
