from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple

import numpy as np
import pandas as pd

from engine_v2.common.types import PatternEvent, PatternStatus, StructureLevel, COL_TIME, COL_O, COL_C
from engine_v2.patterns.imbalance import has_unfilled_imbalance
from engine_v2.patterns.structure_patterns import BreakoutPatterns


# ---------------------------------------------------------------------
# Path 2b: MarketStructure output-column schema for batched writes.
#
# `_write_df_row` writes 27 columns per candle. Instead of ~27 `df.at`
# scalar writes per candle (slow on a wide frame), we write into
# preallocated numpy arrays per column and bulk-assign them to the df once
# at end of `run()`. Each group's dtype + default reproduces the EXACT final
# dtype/value the old per-row `df.at` writes produced — VERIFIED against the
# saved baseline CSV (2026-06-20):
#   - int64 idx/id columns default -1 (incl. structure_id);
#   - int64 counter/flag columns default 0 (range_active, cts_cycle_id,
#     reversal_watch_active, struct_direction);
#   - range_hi/range_lo are created int -1 by `_ensure_output_cols` but upcast
#     to float64 on the first nan write, so their UNPROCESSED-row value is
#     -1.0 (NOT nan) — load-bearing for `range_break_frac` at the first
#     processed candle;
#   - threshold/price columns default nan;
#   - state/event string columns are object "" (written as empty CSV fields).
# Nothing reads these df output columns mid-run (pattern detection + the
# zone resolvers read only input columns), so deferring the write to
# end-of-run is safe.
_OUT_INT_NEG1 = (
    "range_start_idx", "range_confirm_idx", "cts_idx", "bos_idx",
    "pending_reversal_anchor_idx", "pending_reversal_apply_idx",
    "last_breakout_pat_apply_idx", "structure_id",
)
_OUT_INT_ZERO = (
    "range_active", "cts_cycle_id", "reversal_watch_active", "struct_direction",
)
_OUT_FLOAT_NEG1 = ("range_hi", "range_lo")
_OUT_FLOAT_NAN = (
    "breakout_th", "pullback_th", "range_break_frac", "cts_price",
    "bos_price", "cts_threshold", "bos_threshold", "reversal_bos_th_frozen",
)
_OUT_OBJ_EMPTY = (
    "market_state", "cts_event", "bos_event", "cycle_stage", "cts_phase_debug",
)


# Resolver protocol for the dual CTS proximity check (Part 4 §13.5.b).
# MarketStructure does not import from zones/ — instead it consumes these
# callables, wired by the orchestrator (`structure/structure_engine.py`)
# from the relocated derivation primitives in `zones/kl_zones_v1.py` /
# `zones/poi_zones.py`.
BosInnerResolver = Callable[[pd.DataFrame, int, int], Optional[float]]
# (df, bos_idx, struct_direction) -> inner_price | None

PoiInnersResolver = Callable[
    [pd.DataFrame, int, float, int, float, int, int, int, float, Optional[Dict[str, Any]]],
    List[float],
]
# (df, bos_idx, bos_price, cts_idx, cts_price, sd, sid, cycle_id,
#  fill_threshold, c0_data) -> [inner prices]
#
# `fill_threshold` matches the POIConfig default (0.70). Passed explicitly
# so MarketStructure controls the value its has_unfilled checks (in
# `_update_cycle0_data` and in the shared
# `zones/fib_tracker.select_fib_anchor_for_cycle` utility) agree with the
# resolver's downstream IC search.
#
# `c0_data` is the sid's cycle-0 snapshot tracked on MarketStructureState
# (bos_idx/price, cts_idx/price, has_unfilled, scenario1, locked). Used by
# `zones/poi_zones.compute_poi_inners_for_cycle` to drive the shared
# `zones/fib_tracker.select_fib_anchor_for_cycle` Scenario 2 decision so
# the in-flight POI snapshot agrees with FibTracker's downstream anchor
# selection. None when MS has not yet captured cycle-0 data
# (cts_cycle_id == -1 / cycle 0 itself / sid 0).


# Default fallback proximity threshold for direct/test callers that don't
# pass `proximity_pips`. Real callers go through `structure/structure_engine.py`
# wrappers which look up the TF-keyed value from `DEFAULT_PROXIMITY_PIPS`.
# Kept in sync with `zones/zone_proximity.py::DEFAULT_PROXIMITY_PIPS["H1"]`
# manually (a direct import would create a cycle — zone_proximity imports
# StructureEvent from this module).
_FALLBACK_PROXIMITY_PIPS = 9  # matches H1 default

# Default fallback narrow-cycle gap threshold for the dual CTS proximity
# confirmation gate (Rule 1). Below this, sd-proximity cannot confirm CTS.
# Same import-cycle reason as above — kept in sync manually with
# `zones/zone_proximity.py::DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS["H1"]`.
_FALLBACK_MIN_GAP_PIPS = 50  # matches H1 default


def _check_sd_proximity_at_candle(
    df: pd.DataFrame,
    candle_idx: int,
    struct_direction: int,
    bos_inner: float,
    threshold: float,
    poi_inners: Optional[List[float]] = None,
) -> Optional[Tuple[float, str]]:
    """Check if a candle wick comes within `threshold` of the closest
    sd-direction inner bound (BOS or POI). Returns (trigger_inner, zone_kind)
    on hit; None otherwise.

    Pure function — no zone-derivation deps. Moved into structure/ from the
    deleted `proximity_helpers.py` in Part 4 §13.5.b.
    """
    if candle_idx not in df.index:
        return None

    inners: List[Tuple[float, str]] = [(float(bos_inner), "BOS")]
    if poi_inners:
        for v in poi_inners:
            inners.append((float(v), "POI"))

    if struct_direction == 1:
        chosen_inner, zone_kind = max(inners, key=lambda t: t[0])
        candle_low = float(df.loc[candle_idx, "l"])
        if candle_low <= chosen_inner + threshold:
            return (chosen_inner, zone_kind)
    else:
        chosen_inner, zone_kind = min(inners, key=lambda t: t[0])
        candle_high = float(df.loc[candle_idx, "h"])
        if candle_high >= chosen_inner - threshold:
            return (chosen_inner, zone_kind)

    return None


# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------

class MarketState(str, Enum):
    NONE = "none"  # only at very beginning before first breakout/CTS
    BREAKOUT = "breakout"
    RANGE = "range"
    PULLBACK = "pullback"
    PULLBACK_RANGE = "pullback_range"
    REVERSAL = "reversal"


@dataclass(frozen=True)
class Point:
    idx: int
    price: float


@dataclass
class StructureEvent:
    idx: int
    category: Literal["STRUCTURE", "RANGE", "STATE"]
    type: str
    price: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketStructureState:
    # Structure unit id (increments on reversal)
    struct_direction: int # structure direction or starting_direction
    structure_id: int = 0
    state: MarketState = MarketState.NONE
    prev_state: MarketState = MarketState.NONE

    # CTS lifecycle
    cts: Optional[Point] = None
    cts_phase: Literal["NONE", "EST_OR_UPD", "CONFIRMED", "FALSE_BREAK"] = "NONE"
    cts_confirmed_for_idx: Optional[int] = None  # prevent duplicate CTS_CONFIRMED emits for same CTS
    cts_event: str = ""  # one-candle event marker written by _write_df_row

    # NEW (Part 2): CTS cycles + thresholds
    cts_cycle_id: int = -1  # increments to 0 on first CTS cycle (0-indexed)
    cts_threshold: Optional[float] = None  # mirrors active range bound after CTS confirmation (debug)

    # BOS lifecycle
    bos_confirmed: Optional[Point] = None
    bos_threshold: Optional[float] = None  # used for reversal checks when implemented
    bos_event: str = ""  # one-candle event marker written by _write_df_row

    # Range (active)
    range_active: bool = False
    range_hi: Optional[float] = None
    range_lo: Optional[float] = None
    range_start_idx: Optional[int] = None
    range_confirm_idx: Optional[int] = None

    # Breakout bookkeeping
    last_breakout_pat_apply_idx: Optional[int] = None  # j
    last_pullback_pat_apply_idx: Optional[int] = None

    # Zone-proximity-based CTS confirmation (per-cycle state)
    # confirmation_method indicates HOW the current cycle's CTS got confirmed.
    # "pullback" = via pullback pattern (existing behavior).
    # "sd_zone_proximity" = via first sd zone proximity trigger.
    # None = not yet confirmed.
    cts_confirmed_method: Optional[Literal["pullback", "sd_zone_proximity"]] = None
    # Whether a valid pullback pattern fired for the current cycle.
    pullback_fired_for_cycle: bool = False
    # Idx where proximity confirmed CTS for the current cycle (None if not).
    proximity_confirmed_idx: Optional[int] = None
    # Idx where CTS was confirmed (proximity or pullback, whichever first).
    # Used to bound BOS_n+1 max-retracement search.
    cts_confirmed_idx: Optional[int] = None
    # BOS inner price (computed once when BOS_CONFIRMED fires for the new cycle)
    # — used by the per-candle proximity check.
    bos_inner_for_cycle: Optional[float] = None
    # POI inner snapshot for the current cycle — refreshed at CTS_ESTABLISHED
    # and each CTS_UPDATED. Stays empty until first refresh.
    poi_inners_for_cycle: List[float] = field(default_factory=list)

    # Cycle-0 snapshot tracked for the shared
    # `zones.fib_tracker.select_fib_anchor_for_cycle` Scenario 2 decision.
    # Populated at cycle-0 CTS_ESTABLISHED, refreshed on each CTS_UPDATED
    # for cycle 0 (raw + pattern paths), locked at cycle-0 CTS_CONFIRMED so
    # any later raw-extreme CTS_UPDATEDs don't mutate it. Stays None for
    # cycle 1+ refreshes when sid never produced a cycle 0 (e.g. partial
    # runs / sid 0 itself doesn't need it). Schema: see
    # `_update_cycle0_data`. MS does not track Scenario 1; the resolver
    # treats `scenario1=None` as "evaluate Scenario 2/3" — see the utility's
    # docstring for the approximation contract.
    cycle0_data: Optional[Dict[str, Any]] = None

    # -------------------------------------------------
    # Week 5 Part 3A: BOS barrier semantics + reversal watch
    #
    # Reversal watch starts ONLY when a candle CLOSE breaks bos_threshold.
    # While active:
    #   - bos_threshold MUST NOT update
    #   - reversal candidates use reversal_bos_th_frozen (the barrier snapshot)
    # If no valid reversal pattern appears within watch window:
    #   - bos_threshold updates to the ANCHOR candle wick (close-break candle)
    #   - watch clears
    #   - execution rewinds to (anchor_idx + 1) to reprocess subsequent candles normally
    # -------------------------------------------------
     
    reversal_watch_active: bool = False
    reversal_watch_start_idx: Optional[int] = None
    reversal_bos_th_frozen: Optional[float] = None
    reversal_watch_expires_idx: Optional[int] = None

    # Pending reversal (scheduled on BOS close-break anchor candle)
    pending_reversal_ev: Optional[PatternEvent] = None
    pending_reversal_anchor_idx: Optional[int] = None
    pending_reversal_apply_idx: Optional[int] = None

    jump_to_idx: Optional[int] = None   # next anchor override
    jump_seed_state: Optional[dict] = None


# ---------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------

class MarketStructure:
    """
    Week 5: State-driven market structure labeling.

    Key properties:
      - Sequential (event-driven). No skipping.
      - Pattern evaluation uses BreakoutPatterns.detect_first_success(...) with dynamic break_threshold.
      - Range candidate anchored at current candle i
      - Activates a range starting at candidate candle j (bounds = high[j], low[j]). Expansion occurs later via incremental updates; replay handles threshold-correct pattern evaluation between j and the decision point.
      - A candle cannot become a range starter if it is a pattern apply candle (end_idx or confirmation_idx).
      - A candle also cannot become a range starter if it participated in the original pattern window excluding the last candle (e.g., continuous: start+middle; 2-candle patterns: first candle).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        struct_direction: int,
        *,
        eps: float = 0.0001,
        range_min_k: int = 2,
        range_max_k: int = 5,
        debug_invariants: bool = True,
        start_idx: int = 0,
        structure_id: int = 0,
        end_idx: int | None = None,
        timeframe: str = "H1",
        proximity_pips: Optional[int] = None,
        min_gap_pips: Optional[int] = None,
        pip_size: float = 0.0001,
        bos_inner_resolver: Optional[BosInnerResolver] = None,
        poi_inners_resolver: Optional[PoiInnersResolver] = None,
        fill_threshold: float = 0.70,
        enforce_cts0_new_extreme: bool = False,
        bos0_inner: Optional[float] = None,
    ):
        if struct_direction not in (1, -1):
            raise ValueError(f"[market_structure] struct_direction must be 1 or -1, got {struct_direction}")
        self.df = df.copy()
        # ---- Perf (Path 2a): positional numpy price views for the per-candle
        # hot loop. MS used to read o/h/l/c thousands of times via
        # `float(df.iloc[i][<col>])`, each of which builds a full-row Series
        # just to pull one scalar. These float arrays replace that with O(1)
        # indexed access. Prices are immutable for the whole run (incl. rewind),
        # so the views never go stale.
        #
        # INVARIANT: MS reads candles positionally (`.iloc[i]`) AND by label
        # (`.loc[candle_idx]`) with candle_idx sourced from the positional loop
        # var, so the existing code already requires a 0..n-1 RangeIndex. Assert
        # it so positional array access is provably equivalent (and so a future
        # caller passing a non-reset index fails loudly instead of silently
        # corrupting).
        if not self.df.index.equals(pd.RangeIndex(len(self.df))):
            raise ValueError(
                "[market_structure] requires a 0..n-1 RangeIndex df (got "
                f"{self.df.index!r}); MS mixes positional .iloc[i] and label "
                ".loc[candle_idx] access."
            )
        self._o = self.df["o"].to_numpy(dtype=float)
        self._h = self.df["h"].to_numpy(dtype=float)
        self._l = self.df["l"].to_numpy(dtype=float)
        self._c = self.df["c"].to_numpy(dtype=float)
        self.struct_direction = struct_direction
        self.start_idx = int(start_idx)
        self.end_idx = int(end_idx) if end_idx is not None else None  # optional stopping point (inclusive)
        self.eps = float(eps)
        self.debug_invariants = bool(debug_invariants)
        # Opt-in 'true first breakout' rule (see PART4_REFACTOR_SPEC §4.4
        # / project_unified_identify_start_probe memory). When True, MS
        # rejects a breakout pattern that would establish cycle 0 unless
        # its anchor's extreme is the running max/min over
        # [start_idx, cts_idx - 1]. Default False preserves existing
        # behavior — only Phase 2 of the unified probe passes True.
        #
        # REPURPOSED 2026-06-07 (scan-from-start, see
        # project_true_first_breakout_cycle0.md): when True this is the
        # "pre-CTS_0 scan mode" switch. While cycle 0 is unestablished MS
        # delegates the entire breakout search to the shared
        # `find_true_first_breakout` routine (mechanism B against the
        # handed `bos0_inner` + strict full-pattern extreme + cycle-0
        # tie-break), establishes CTS_0 at the located winner via the
        # normal cycle-0 path, then resumes. The probe and MS therefore
        # agree on CTS_0 by construction (same routine, same data, same
        # `bos0_inner`).
        self.enforce_cts0_new_extreme = bool(enforce_cts0_new_extreme)
        # The cycle-0 breakout gate threshold (= the BOS_0 inner the probe
        # finalized). REQUIRED in scan mode — condition 1 (anchor close
        # past the BOS_0 inner) is meaningless without it.
        self.bos0_inner = float(bos0_inner) if bos0_inner is not None else None
        if self.enforce_cts0_new_extreme and self.bos0_inner is None:
            raise ValueError(
                "[market_structure] enforce_cts0_new_extreme (pre-CTS_0 scan "
                "mode) requires bos0_inner (the cycle-0 breakout gate threshold)."
            )
        # Lazily-computed cycle-0 true-first-breakout decision (scan mode).
        self._cts0_tfb = None
        self._cts0_tfb_computed = False

        self.state = MarketStructureState(struct_direction=struct_direction, structure_id=0)
        self.state.struct_direction = int(struct_direction)
        self.state.structure_id = int(structure_id)
        self.events: List[StructureEvent] = []

        # Pattern detector (uses df feature columns)
        self._bp = BreakoutPatterns(self.df)

        self.range_min_k = int(range_min_k)
        self.range_max_k = int(range_max_k)

        # Zone-proximity-based CTS confirmation config. The TF→pips lookup
        # lives in `structure/structure_engine.py` (the orchestrator); when
        # a direct/test caller doesn't pass proximity_pips, we fall back to
        # the H1 default to keep the code path safe.
        self.timeframe = str(timeframe)
        self.proximity_pips = int(proximity_pips) if proximity_pips is not None else _FALLBACK_PROXIMITY_PIPS
        self.min_gap_pips = int(min_gap_pips) if min_gap_pips is not None else _FALLBACK_MIN_GAP_PIPS
        self.pip_size = float(pip_size)
        self._proximity_threshold = self.proximity_pips * self.pip_size
        self._min_gap_threshold = self.min_gap_pips * self.pip_size

        # Dual CTS resolver protocol (Part 4 §13.5.b). When unset, the
        # proximity check returns None — the dual CTS path is effectively
        # disabled. All real callers wire resolvers via structure_engine.py.
        self._bos_inner_resolver = bos_inner_resolver
        self._poi_inners_resolver = poi_inners_resolver

        # Imbalance fill threshold used by the cycle-0 snapshot's
        # `has_unfilled` computation. Kept in sync with POIConfig default
        # so MS-side has_unfilled matches FibTracker's view of the same
        # imbalance instances.
        self._fill_threshold = float(fill_threshold)

        self._ensure_output_cols()

        # ---- Profiling instrumentation (temporary) ----
        # Accumulator dict keyed by sub-step name; populated in
        # `_replay_step_no_patterns` + `_step_anchor` + helpers. Dumped at the
        # end of `run()`. Cheap (perf_counter only on slow paths); can be
        # stripped once we've identified the bottleneck.
        self._t_prof: Dict[str, float] = {
            "update_active_range": 0.0,
            "cts_pre_confirm": 0.0,
            "cts_proximity": 0.0,
            "bos_barrier": 0.0,
            "reversal_watch": 0.0,
            "pending_reversal": 0.0,
            "write_df_row": 0.0,
            "pattern_detect": 0.0,
            "step_anchor_other": 0.0,
            "replay_no_pat_total": 0.0,
        }
        self._t_prof_count: Dict[str, int] = {
            "replay_no_pat_calls": 0,
            "step_anchor_calls": 0,
            "pattern_detect_calls": 0,
        }

    # ----------------------------
    # Public API
    # ----------------------------

    def run(self) -> Tuple[pd.DataFrame, List[StructureEvent], List[StructureLevel]]:
        """
        Sequentially labels market_state, CTS/BOS/range fields, and emits StructureEvents.
        No skipping. Uses pending confirmation for patterns + internal range evaluation window (min/max lookahead) with rewind+replay.

        If end_idx is set, processing stops after that idx (inclusive).
        """
        n = len(self.df)
        # ---- Profiling (temporary): per-sub CPU time. process_time() is
        # immune to wall-clock / machine-load noise (the 38m/44m/6.3h swings);
        # ~16ms coarse on Windows, but bracketing the whole run keeps the
        # multi-second delta accurate. Headline metric for the MS perf sprint;
        # the perf_counter per-substep breakdown below stays for proportions. ----
        from time import process_time as _ptime
        _cpu_run_start = _ptime()
        # i = 0
        i = int(self.start_idx)
        if i < 0:
            i = 0
        if i >= n:
            return self.df, self.events, self.levels

        # Determine effective end index (inclusive)
        effective_end = n - 1
        if self.end_idx is not None and self.end_idx < effective_end:
            effective_end = self.end_idx

        # Path 2b: preallocate output arrays (reused across rewinds). Placed
        # after the start_idx>=n early returns so degenerate runs skip it.
        self._init_output_arrays(n)

        while i <= effective_end:
            if self.state.state == MarketState.REVERSAL:
                break

            next_i = self._step_anchor(i)
            self._dbg(f"[POST_STEP] i={i} next_i={next_i} jump_to={self.state.jump_to_idx}")

            # If reversal-watch expiry requested a rewind, honor it
            if self.state.jump_to_idx is not None:
                jump_to = int(self.state.jump_to_idx)
                seed = self.state.jump_seed_state  # capture before rewind resets state

                self._dbg(f"[REWIND] from={i} step_next={next_i} jump_to={jump_to}")

                # Rebuild state up to jump_to - 1, then restore the post-expire seed snapshot
                self._rewind_to(jump_to, seed=seed)

                # clear jump request + seed after using
                self.state.jump_to_idx = None
                self.state.jump_seed_state = None

                i = jump_to
                continue

            i = next_i

        # Path 2b: flush the batched per-candle output arrays to df columns
        # before the terminal reversal stamping + invariant checks below
        # (both read the df output columns).
        self._flush_output_arrays()

        levels = self._events_to_structure_levels()

        # --- Terminal reversal stamping ---
        # If a reversal was reached and we exited early, later rows were never updated.
        # Stamp market_state forward so invariants and charts are consistent.
        if "market_state" in self.df.columns:
            rev_mask = self.df["market_state"].astype(str) == "reversal"
            if rev_mask.any():
                first_rev_idx = int(rev_mask.idxmax())
                self.df.loc[first_rev_idx:, "market_state"] = "reversal"

        if self.debug_invariants:
            self._check_invariants_df()

        # ---- Profiling dump (temporary) ----
        # Only print for non-trivial runs so the H1 main + tiny probe runs
        # don't drown the log. Threshold: any sub that took > 5s total.
        _total_step_anchor = self._t_prof["step_anchor_other"]
        _cpu_total = _ptime() - _cpu_run_start
        if _total_step_anchor > 5.0:
            _rnp_calls = max(self._t_prof_count["replay_no_pat_calls"], 1)
            _anchor_calls = max(self._t_prof_count["step_anchor_calls"], 1)
            print(
                f"[ms_prof] sid={self.state.structure_id} sd={self.struct_direction} "
                f"n={len(self.df)} cpu_total={_cpu_total:.2f}s "
                f"anchors={_anchor_calls} rnp_calls={_rnp_calls} "
                f"step_anchor_total={_total_step_anchor:.2f}s "
                f"pattern_detect={self._t_prof['pattern_detect']:.2f}s "
                f"rnp_total={self._t_prof['replay_no_pat_total']:.2f}s "
                f"(per-substep avg ms over rnp_calls): "
                f"upd_range={self._t_prof['update_active_range']/_rnp_calls*1000:.3f} "
                f"cts_pre={self._t_prof['cts_pre_confirm']/_rnp_calls*1000:.3f} "
                f"cts_prox={self._t_prof['cts_proximity']/_rnp_calls*1000:.3f} "
                f"bos_bar={self._t_prof['bos_barrier']/_rnp_calls*1000:.3f} "
                f"rev_watch={self._t_prof['reversal_watch']/_rnp_calls*1000:.3f} "
                f"pend_rev={self._t_prof['pending_reversal']/_rnp_calls*1000:.3f} "
                f"write_row={self._t_prof['write_df_row']/_rnp_calls*1000:.3f} | "
                f"per-substep TOTAL (s): "
                f"upd_range={self._t_prof['update_active_range']:.2f} "
                f"cts_pre={self._t_prof['cts_pre_confirm']:.2f} "
                f"cts_prox={self._t_prof['cts_proximity']:.2f} "
                f"bos_bar={self._t_prof['bos_barrier']:.2f} "
                f"rev_watch={self._t_prof['reversal_watch']:.2f} "
                f"pend_rev={self._t_prof['pending_reversal']:.2f} "
                f"write_row={self._t_prof['write_df_row']:.2f}"
            )

        return self.df, self.events, levels

    
    def _rewind_to(self, jump_to: int, *, seed: Optional[dict] = None) -> None:
        """
        Rebuild state/events by replaying from the start up to (jump_to - 1),
        then restore a provided seed snapshot (used for reversal-watch false-break rewinds).
        """
        jump_to = int(jump_to)
        n = len(self.df)
        if jump_to <= 0:
            self.state = MarketStructureState(struct_direction=self.struct_direction)
            self.events = []
            return
        if jump_to >= n:
            jump_to = n - 1

        # Reset state + events (df price/features stay; output cols will be overwritten as we replay)
        self.state = MarketStructureState(struct_direction=self.struct_direction)
        self.events = []

        # Replay forward up to jump_to-1.
        # IMPORTANT: ignore any jump_to_idx requests that occur during this rebuild.
        self._in_rewind = True
        try:
            i = 0
            while i < jump_to:
                nxt = self._step_anchor(i)

                # During rebuild, ignore any jump requests
                if self.state.jump_to_idx is not None:
                    self.state.jump_to_idx = None
                if self.state.jump_seed_state is not None:
                    self.state.jump_seed_state = None

                i = nxt
        finally:
            self._in_rewind = False

        # Restore seed snapshot (post-expire BOS/CTS/range state)
        if seed is not None:
            self.state.state = seed.get("market_state", self.state.state)
            self.state.bos_threshold = seed.get("bos_threshold", self.state.bos_threshold)
            self.state.cts_threshold = seed.get("cts_threshold", self.state.cts_threshold)

            self.state.range_active = seed.get("range_active", self.state.range_active)
            self.state.range_hi = seed.get("range_hi", self.state.range_hi)
            self.state.range_lo = seed.get("range_lo", self.state.range_lo)
            self.state.range_start_idx = seed.get("range_start_idx", self.state.range_start_idx)
            self.state.range_confirm_idx = seed.get("range_confirm_idx", self.state.range_confirm_idx)

        # CRITICAL: the replay-to-(jump_to-1) may have started a reversal watch (e.g. at 105)
        # which must NOT remain active when resuming after an expiry-based rewind.
        self._clear_pending_reversal()
        self._clear_reversal_watch()

        # Also ensure no jump is still requested from the rebuild itself
        self.state.jump_to_idx = None
        self.state.jump_seed_state = None


    # ----------------------------
    # Per-candle step
    def _replay_step_no_patterns(self, i: int, *, freeze_range: bool = False) -> None:
        """Advance deterministic range/state fields without running pattern detection.

        IMPORTANT: When we're back-filling candles *before* a known pattern apply candle,
        we must NOT expand the active range bounds, otherwise the range can "absorb"
        breakout/pullback window candles and the pattern will no longer qualify.

        freeze_range=True means:
          - do not call _update_active_range(i)
          - do not transition RANGE->PULLBACK/PULLBACK_RANGE based on candle-by-candle expansion
          - just persist current state (typically RANGE) and write the row
        """
        # ---- Profiling: per-substep timing accumulator ----
        from time import perf_counter as _pc
        _t_entry = _pc()
        self._t_prof_count["replay_no_pat_calls"] += 1

        st = self.state
        if not freeze_range:
            # If an active range exists, evolve it candle-by-candle (unless frozen).
            if st.range_active:
                prev_hi = st.range_hi
                prev_lo = st.range_lo

                _t0 = _pc()
                self._update_active_range(i)
                self._t_prof["update_active_range"] += _pc() - _t0

                if st.state in (MarketState.PULLBACK, MarketState.PULLBACK_RANGE):
                    # Do not auto-downgrade on the same candle that *applied* the pullback pattern.
                    if st.last_pullback_pat_apply_idx is not None and i == st.last_pullback_pat_apply_idx:
                        self._set_state(MarketState.PULLBACK, i, meta={"reason": "pullback_apply_candle"})
                    else:
                        if self._expanded_pullback_side(i, prev_hi=prev_hi, prev_lo=prev_lo):
                            self._set_state(MarketState.PULLBACK, i, meta={"reason": "replay_expand_pullback_side"})
                        else:
                            self._set_state(MarketState.PULLBACK_RANGE, i, meta={"reason": "replay_no_expand_pullback_side"})

        # Option B: even when backfilling (freeze_range=True), CTS can update pre-confirm
        # based on raw new extremes in struct_direction.
        _t0 = _pc()
        self._maybe_update_cts_pre_confirm(i, via="replay_raw")
        self._t_prof["cts_pre_confirm"] += _pc() - _t0

        # Per-candle CTS confirmation via sd zone proximity. Runs on every
        # candle (anchor, back-fill, apply, fallthrough) — see LANDMINES
        # "Dual CTS Proximity Confirmation" for the load-bearing gates.
        _t0 = _pc()
        self._maybe_confirm_cts_via_proximity(i)
        self._t_prof["cts_proximity"] += _pc() - _t0

        # -------------------------------------------------
        # Week 5 Part 3A: BOS barrier semantics
        # -------------------------------------------------
        # NOTE:
        # - "cross" = wick crosses barrier
        # - "break" = close breaks barrier
        # - probe (wick cross, no close break) updates bos_threshold
        # - close break starts reversal watch and freezes barrier
        _t0 = _pc()
        self._bos_barrier_step(i)
        self._t_prof["bos_barrier"] += _pc() - _t0

        _t0 = _pc()
        self._maybe_expire_reversal_watch(i)
        self._t_prof["reversal_watch"] += _pc() - _t0

        # NEW: if reversal applies on this candle, it's terminal
        _t0 = _pc()
        _is_terminal = self._maybe_apply_pending_reversal(i)
        self._t_prof["pending_reversal"] += _pc() - _t0

        if _is_terminal:
            # write row after terminal apply state updates
            _t0 = _pc()
            self._write_df_row(i)
            self._t_prof["write_df_row"] += _pc() - _t0
            self._t_prof["replay_no_pat_total"] += _pc() - _t_entry
            return

        _t0 = _pc()
        self._write_df_row(i)
        self._t_prof["write_df_row"] += _pc() - _t0
        self._t_prof["replay_no_pat_total"] += _pc() - _t_entry

    # ----------------------------
    # Week 5 Part 3A: BOS barrier + reversal watch helpers
    # ----------------------------

    def _close_breaks_bos(self, i: int, bos: float) -> bool:
        c = float(self._c[i])
        if self.struct_direction == 1:
            return c < bos
        else:
            return c > bos

    def _bos_barrier_step(self, i: int) -> None:
        """
        BOS barrier semantics for candle i.
        Uptrend (struct_direction=+1):
        - If close breaks below bos_th => start reversal watch (freeze bos at pre-candle value)
            - If anchor fails immediately (no pending reversal), treat as false-break:
            bos_th := candle low, emit BOS_THRESHOLD_UPDATED, clear watch.
        - Else if wick crosses below bos_th but close does NOT => bos_th := candle low, emit BOS_THRESHOLD_UPDATED
        Downtrend symmetric:
        - If close breaks above bos_th => start reversal watch...
        - Else if wick crosses above bos_th but close does NOT => bos_th := candle high, emit BOS_THRESHOLD_UPDATED
        """
        st = self.state
        if st.bos_threshold is None:
            return

        # Don't process BOS barrier in REVERSAL state (terminal - structure is over)
        if st.state == MarketState.REVERSAL:
            return

        # While reversal watch is active, bos_th must not update.
        if st.reversal_watch_active:
            return

        bos = float(st.bos_threshold)
        h = float(self._h[i])
        l = float(self._l[i])
        c = float(self._c[i])

        if self.struct_direction == 1:
            # ---- close-break -> start watch (and maybe immediate false-break resolve)
            if c < bos:
                self._start_reversal_watch(i, bos_frozen=bos)
                self._schedule_reversal_from_anchor(i, bos_frozen=bos)

                # if this anchor produced no pending reversal -> false break resolution
                if self.state.pending_reversal_apply_idx is None:
                    prev = float(st.bos_threshold)
                    new_thr = float(l)
                    if new_thr < prev:
                        st.bos_threshold = new_thr
                        self._emit_bos_threshold_updated(
                            i,
                            float(new_thr),
                            meta={"prev": float(prev), "reason": "rv_anchor_failed"},
                        )
                    self._clear_reversal_watch()

                self._dbg(f"[BOS_CLOSE_BREAK] i={i} close={c:.5f} bos_prev={bos:.5f} dir={self.struct_direction}")
                return

            # ---- wick-cross probe (no close-break) -> update barrier to wick extreme
            if l < bos and c >= bos:
                prev = float(st.bos_threshold)
                new_thr = float(l)
                if new_thr < prev:
                    st.bos_threshold = new_thr
                    self._emit_bos_threshold_updated(
                        i,
                        float(new_thr),
                        meta={"prev": float(prev), "reason": "probe_no_break"},
                    )
                return

        else:
            # ---- close-break (downtrend is ABOVE bos_th)
            if c > bos:
                self._start_reversal_watch(i, bos_frozen=bos)
                self._schedule_reversal_from_anchor(i, bos_frozen=bos)

                if self.state.pending_reversal_apply_idx is None:
                    prev = float(st.bos_threshold)
                    new_thr = float(h)
                    if new_thr > prev:
                        st.bos_threshold = new_thr
                        self._emit_bos_threshold_updated(
                            i,
                            float(new_thr),
                            meta={"prev": float(prev), "reason": "rv_anchor_failed"},
                        )
                    self._clear_reversal_watch()

                self._dbg(f"[BOS_CLOSE_BREAK] i={i} close={c:.5f} bos_prev={bos:.5f} dir={self.struct_direction}")
                return

            # ---- wick-cross probe (no close-break)
            if h > bos and c <= bos:
                prev = float(st.bos_threshold)
                new_thr = float(h)
                if new_thr > prev:
                    st.bos_threshold = new_thr
                    self._emit_bos_threshold_updated(
                        i,
                        float(new_thr),
                        meta={"prev": float(prev), "reason": "probe_no_break"},
                    )
                return


    def _start_reversal_watch(self, i: int, *, bos_frozen: float) -> None:
        st = self.state
        if st.reversal_watch_active:
            return

        st.reversal_watch_active = True
        st.reversal_watch_start_idx = int(i)
        st.reversal_bos_th_frozen = float(bos_frozen)
        st.reversal_watch_expires_idx = min(int(i) + int(self.range_max_k), len(self.df) - 1)

        # Emit REVERSAL_WATCH_START event for all close-breaks (survives rewinds)
        # This captures the moment reversal watch begins, even if no pattern is found
        self.events.append(
            StructureEvent(
                idx=int(i),
                category="STRUCTURE",
                type="REVERSAL_WATCH_START",
                price=float(bos_frozen),
                meta={
                    "anchor_idx": int(i),
                    "bos_frozen": float(bos_frozen),
                    "expires_idx": int(st.reversal_watch_expires_idx),
                    "structure_id": int(st.structure_id),
                    "struct_direction": int(self.struct_direction),
                },
            )
        )


    def _clear_reversal_watch(self) -> None:
        st = self.state
        st.reversal_watch_active = False
        st.reversal_watch_start_idx = None
        st.reversal_bos_th_frozen = None
        st.reversal_watch_expires_idx = None

    def _maybe_expire_reversal_watch(self, i: int) -> None:
        """
        If reversal watch is active but no valid reversal pattern appears within the watch window,
        then this was a "false break" by your definition: update bos_threshold the anchor candle wick, then clear the watch.

        After expiry, we "rewind" execution to (anchor_idx + 1) so the next anchor starts there.
        """
        st = self.state
        if not st.reversal_watch_active:
            return
        if st.reversal_watch_expires_idx is None:
            return
        if int(i) < int(st.reversal_watch_expires_idx):
            return

        # Expire: failed reversal => bos_threshold expands to ANCHOR candle wick
        anchor = st.reversal_watch_start_idx
        if anchor is None:
            # defensive: shouldn't happen, but don't crash
            self._clear_pending_reversal()
            self._clear_reversal_watch()
            return

        anchor = int(anchor)
        if self.struct_direction == 1:
            # st.bos_threshold = float(self.df.iloc[anchor]["l"])
            prev = st.bos_threshold
            new_thr = float(self._l[anchor])
            st.bos_threshold = new_thr
            self._emit_bos_threshold_updated(
                i,
                float(new_thr),
                meta={"prev": None if prev is None else float(prev), "reason": "probe_no_break"},
            )
        else:
            # st.bos_threshold = float(self.df.iloc[anchor]["h"])
            prev = st.bos_threshold
            new_thr = float(self._h[anchor])
            st.bos_threshold = new_thr
            self._emit_bos_threshold_updated(
                i,
                float(new_thr),
                meta={"prev": None if prev is None else float(prev), "reason": "probe_no_break"},
            )

        # Rewind to anchor + 1 (do NOT jump to extremes)
        jump_to = min(anchor + 1, len(self.df) - 1)
        st.jump_to_idx = jump_to

        # Seed snapshot: this is the state we want to be true when we resume at jump_to
        # (because this BOS update is based on the full watch window lookahead).
        st.jump_seed_state = {
            "market_state": st.state,
            "bos_threshold": st.bos_threshold,
            "cts_threshold": st.cts_threshold,
            "range_active": st.range_active,
            "range_hi": st.range_hi,
            "range_lo": st.range_lo,
            "range_start_idx": st.range_start_idx,
            "range_confirm_idx": st.range_confirm_idx,
        }

        self._dbg(
            f"[RV_EXPIRE] i={i} expired_idx={st.reversal_watch_expires_idx} "
            f"anchor={anchor} new_bos={st.bos_threshold} jump_to={jump_to} "
            f"clearing_pending={st.pending_reversal_apply_idx is not None}"
        )

        # Pending reversal is no longer valid after watch expiry
        self._clear_pending_reversal()
        self._clear_reversal_watch()

    def _clear_pending_reversal(self) -> None:
        st = self.state
        st.pending_reversal_ev = None
        st.pending_reversal_anchor_idx = None
        st.pending_reversal_apply_idx = None

    def _schedule_reversal_from_anchor(self, anchor_idx: int, *, bos_frozen: float) -> None:
        """
        Called ONLY when a candle CLOSE-breaks the previously established bos_threshold.
        Schedules a reversal PatternEvent (if any) and its apply index.
        """
        st = self.state

        # Do not schedule reversals in NONE regime
        if st.state == MarketState.NONE:
            return

        ev_r = self._bp.detect_best_for_anchor(anchor_idx, -self.struct_direction, float(bos_frozen))
        if ev_r is None:
            return

        apply_r = self._apply_idx(ev_r)
        if apply_r is None:
            return

        # Only consider applies within the watch window (if defined)
        if st.reversal_watch_expires_idx is not None and int(apply_r) > int(st.reversal_watch_expires_idx):
            return

        # Keep earliest scheduled reversal apply
        if st.pending_reversal_apply_idx is None or int(apply_r) < int(st.pending_reversal_apply_idx):
            st.pending_reversal_ev = ev_r
            st.pending_reversal_anchor_idx = int(anchor_idx)
            st.pending_reversal_apply_idx = int(apply_r)

            pat = getattr(ev_r, "pat", None) or getattr(ev_r, "pattern", None) or "?"
            self._dbg(
                f"[RV_SCHEDULE] anchor={st.pending_reversal_anchor_idx} "
                f"apply={st.pending_reversal_apply_idx} pat={pat} "
                f"bos_frozen={bos_frozen:.5f} expires={st.reversal_watch_expires_idx}"
            )

            # Emit REVERSAL_CANDIDATE event for charting (survives rewinds)
            self.events.append(
                StructureEvent(
                    idx=int(anchor_idx),
                    category="STRUCTURE",
                    type="REVERSAL_CANDIDATE",
                    price=float(bos_frozen),
                    meta={
                        "anchor_idx": int(anchor_idx),
                        "apply_idx": int(apply_r),
                        "pattern": pat,
                        "bos_frozen": float(bos_frozen),
                        "expires_idx": int(st.reversal_watch_expires_idx) if st.reversal_watch_expires_idx is not None else None,
                        "structure_id": int(st.structure_id),
                        "struct_direction": int(self.struct_direction),
                    },
                )
            )

    def _maybe_apply_pending_reversal(self, i: int) -> bool:
        """
        If a pending reversal is scheduled to apply on candle i, apply it (terminal).
        Returns True if reversal was applied.
        """
        st = self.state
        if st.pending_reversal_ev is None or st.pending_reversal_apply_idx is None:
            return False
        if int(i) != int(st.pending_reversal_apply_idx):
            return False

        pat = getattr(st.pending_reversal_ev, "pat", None) or getattr(st.pending_reversal_ev, "pattern", None) or "?"
        self._dbg(
            f"[RV_APPLY] i={i} anchor={st.pending_reversal_anchor_idx} "
            f"apply={st.pending_reversal_apply_idx} pat={pat}"
        )

        # Terminal apply
        self._apply_pattern_at_apply_idx(st.pending_reversal_ev, i, "reversal")
        self._clear_pending_reversal()
        return True

    # ----------------------------

    def _step_anchor(self, i: int) -> int:
        from time import perf_counter as _pc
        _t_anchor_entry = _pc()
        self._t_prof_count["step_anchor_calls"] += 1

        n = len(self.df)
        D = min(i + self.range_max_k, n - 1)  # range_max_k is 5 by default

        st = self.state

        # If state NONE: we may still allow breakout patterns, but we don't allow ranges
        allow_range = (st.state != MarketState.NONE)

        # Determine applicable thresholds (if range active)
        breakout_th = None
        pullback_th = None
        if st.range_active:
            breakout_th = self._range_breakout_threshold()
            pullback_th = self._range_pullback_threshold()

        # 1) Evaluate patterns starting at i (offline computation), but only "act" at apply candle
        _t0 = _pc()
        self._t_prof_count["pattern_detect_calls"] += 1
        winner = self._best_bopb_pattern_at_anchor(
            i=i,
            breakout_th=breakout_th,
            pullback_th=pullback_th,
            D=D,
        )
        self._t_prof["pattern_detect"] += _pc() - _t0

        if winner is not None:
            ev, apply_idx, kind = winner  # kind in {"breakout","pullback", "reversal"}

            # Back-fill candles from anchor up to (but not including) apply candle.
            # Back-fill candles before the apply candle WITHOUT expanding range bounds.
            # (Prevents the active range from absorbing the breakout/pullback window.)
            for k in range(i, apply_idx):
                self._replay_step_no_patterns(k, freeze_range=True)

            # "live-like": we act as if apply candle just closed
            self._apply_pattern_at_apply_idx(ev, apply_idx, kind)

            # reversal check at apply_idx
            # self._maybe_trigger_reversal(apply_idx)

            # Always write the apply candle row (post-apply)
            self._replay_step_no_patterns(apply_idx)

            # IMPORTANT: do NOT attempt to "finalize" a new range anchored at the apply candle.
            # After a breakout, the next range candidate (if any) must begin at a later candle
            # (e.g. your example: breakout at 53, next range starts at 57), so we simply
            # continue sequentially from the next candle.
            next_i = apply_idx + 1
            if self.state.jump_to_idx is not None:
                next_i = int(self.state.jump_to_idx)
            self._t_prof["step_anchor_other"] += _pc() - _t_anchor_entry
            return next_i

        # 2b) No valid pattern by D:
        if allow_range and not st.range_active:
            range_confirmed, confirm_idx = self._is_range_candle_given_confirm(i)

            if range_confirmed:
                min_d = D if confirm_idx is None else min(confirm_idx, D)
                for k in range(i, min_d):
                    self._replay_step_no_patterns(k, freeze_range=True)

                self._finalize_range_candidate_offline(i)

        # Re-write the i candle row after finalize
        self._replay_step_no_patterns(i)

        # Next anchor always i+1 (the next candle after the candidate)
        next_i = i + 1
        if self.state.jump_to_idx is not None:
            next_i = int(self.state.jump_to_idx)
        self._t_prof["step_anchor_other"] += _pc() - _t_anchor_entry
        return next_i

    def _best_bopb_pattern_at_anchor(
        self,
        *,
        i: int,
        breakout_th: Optional[float],
        pullback_th: Optional[float],
        D: int,
    ) -> Optional[Tuple[PatternEvent, int, Literal["breakout", "pullback"]]]:
        st = self.state

        # Pre-CTS_0 scan-from-start mode: while cycle 0 is unestablished,
        # the ONLY breakout that may establish CTS_0 is the shared routine's
        # true first breakout. Suppress every other candidate (range is
        # already disabled at state NONE; pullback/reversal detection gate
        # on st.cts/state, so they're dormant pre-CTS_0). MS then drives the
        # located winner through its normal establishment path — no
        # duplicated cycle-0 logic. See project_true_first_breakout_cycle0.
        if self.enforce_cts0_new_extreme and st.cts is None:
            tfb = self._get_cts0_tfb()
            if tfb is None:
                return None
            if int(i) == int(tfb.pattern.start_idx):
                # est_idx is the apply/confirm idx == MS's _apply_idx
                # (asserted) — the candle that establishes cycle-0 CTS.
                assert int(self._apply_idx(tfb.pattern)) == int(tfb.est_idx)
                return (tfb.pattern, int(tfb.est_idx), "breakout")
            return None

        # If state NONE: only breakout direction checks (no pullback)
        candidates = []

        # Breakout
        ev_b = self._bp.detect_best_for_anchor(i, self.struct_direction, breakout_th)

        # Breakout Pattern Debugging Print
        # if i in (387, 388):
        #     omo = self._bp.one_maru_opposite(i, self.struct_direction, breakout_th, do_confirm=False)
        #     omc = self._bp.one_maru_continuous(i, self.struct_direction, breakout_th, do_confirm=False)
        #     dm  = self._bp.double_maru(i, self.struct_direction, breakout_th, do_confirm=False)
        #     print(f"[DBG] i={i} breakout_th={breakout_th} "
        #         f"OMO={None if omo is None else omo.status} "
        #         f"OMC={None if omc is None else omc.status} "
        #         f"DM={None if dm is None else dm.status}")
        #     if omc is not None:
        #         print(f"[DBG] OMC start={omc.start_idx} end={omc.end_idx} conf={omc.confirmation_idx}")
        #         print(f"[DBG] candle+1 close={self.df.iloc[i+1]['c']} high={self.df.iloc[i+1]['h']}")

        if ev_b is not None:
            apply_b = self._apply_idx(ev_b)
            if apply_b is not None and apply_b <= D:
                candidates.append((ev_b, apply_b, "breakout"))

        # Pullback (only after first breakout/CTS regime AND not already in pullback modes)
        allow_pullback_detection = (
            st.state not in (MarketState.NONE, MarketState.PULLBACK, MarketState.PULLBACK_RANGE)
        )
        if allow_pullback_detection:
            ev_p = self._bp.detect_best_for_anchor(i, -self.struct_direction, pullback_th)

            # Pullback Pattern Debugging Print
            if i in (102, 103):
                omo = self._bp.one_maru_opposite(i, -self.struct_direction, pullback_th, do_confirm=False)
                omc = self._bp.one_maru_continuous(i, -self.struct_direction, pullback_th, do_confirm=False)
                dm  = self._bp.double_maru(i, -self.struct_direction, pullback_th, do_confirm=False)
                print(f"[DBG] i={i} pullback_th={pullback_th} "
                    f"OMO={None if omo is None else omo.status} "
                    f"OMC={None if omc is None else omc.status} "
                    f"DM={None if dm is None else dm.status}")
                if omc is not None:
                    print(f"[DBG] OMC start={omc.start_idx} end={omc.end_idx} conf={omc.confirmation_idx}")
                    print(f"[DBG] candle+1 close={self.df.iloc[i+1]['c']} high={self.df.iloc[i+1]['h']}")

            if ev_p is not None:
                apply_p = self._apply_idx(ev_p)
                if apply_p is not None and apply_p <= D:
                    candidates.append((ev_p, apply_p, "pullback"))

        # Reversal:
        # - Primary mode: if reversal_watch_active, use frozen BOS barrier.
        # - NEW: also allow reversal detection on the SAME candle that CLOSE-breaks the current bos_threshold
        #        (even though the watch gets activated later in _bos_barrier_step during replay).
        bos_frozen_for_anchor = None

        if st.reversal_watch_active and st.reversal_bos_th_frozen is not None and st.state != MarketState.NONE:
            bos_frozen_for_anchor = float(st.reversal_bos_th_frozen)
        elif (st.state != MarketState.NONE) and (st.bos_threshold is not None) and (not st.reversal_watch_active):
            bos_cur = float(st.bos_threshold)
            if self._close_breaks_bos(i, bos_cur):
                bos_frozen_for_anchor = bos_cur  # snapshot pre-candle barrier (what watch will freeze)

        if bos_frozen_for_anchor is not None:
            ev_r = self._bp.detect_best_for_anchor(i, -self.struct_direction, bos_frozen_for_anchor)
            if ev_r is not None:
                apply_r = self._apply_idx(ev_r)
                if apply_r is not None and apply_r <= D:
                    candidates.append((ev_r, apply_r, "reversal"))

        if not candidates:
            return None

        # Choose earliest apply time; tie-breaker: breakout wins over pullback
        # candidates.sort(key=lambda x: (x[1], 0 if x[2] == "breakout" else 1))
        priority = {"reversal": 0, "breakout": 1, "pullback": 2}
        candidates.sort(key=lambda x: (x[1], priority.get(x[2], 9)))
        return candidates[0]


    # ----------------------------
    # Range candidate creation & activation
    # ----------------------------

    def _is_range_candle_given_confirm(self, i: int | None) -> tuple[bool, int | None]:
        """
        Range candle test for the *candidate start candle i*,
        using the same rule everywhere.

        Must have confirm_idx plus base range label from range_label.py, and then:
        (pinbar and direction == struct_direction) OR (direction != struct_direction)
        """
        if i is None:
            return (False, None)

        base = int(self.df.iloc[i].get("is_range", 0)) == 1
        if not base:
            return (False, None)

        confirm_idx = int(self.df.iloc[i].get("is_range_confirm_idx", -1))
        if confirm_idx < 0:
            return (False, None)

        candle_i = str(self.df.iloc[i].get("candle_type", ""))
        dir_i = int(self.df.iloc[i].get("direction", 0))

        qualifies = (candle_i == "pinbar" and dir_i == self.struct_direction) or (dir_i != self.struct_direction)
        return (qualifies, confirm_idx)

    def _finalize_range_candidate_offline(self, i: int) -> None:
        st = self.state

        # If we already have an active range, this anchor-range-candidate logic probably shouldn't run.
        # In your simplified model, range-active mode is handled by per-candle expansion instead.

        lo_i = float(self._l[i])
        hi_i = float(self._h[i])
        confirm_idx = int(self.df.iloc[i]["is_range_confirm_idx"])

        # NEW: seed range bound using prior CTS extreme (if exists)
        if st.cts is not None:
            cts_price = float(st.cts.price)
            if self.struct_direction == 1:
                hi_i = max(hi_i, cts_price)   # ensure range_hi reaches prior CTS high
            else:
                lo_i = min(lo_i, cts_price)   # ensure range_lo reaches prior CTS low

        # Activate range anchored at i, decision at confirm_idx
        st.range_active = True
        st.range_start_idx = i
        st.range_confirm_idx = confirm_idx
        st.range_hi = hi_i
        st.range_lo = lo_i

        self.events.append(
            StructureEvent(
                idx=confirm_idx,
                category="RANGE",
                type="RANGE_STARTED",
                meta={
                    "start_idx": i,
                    "confirm_idx": confirm_idx,
                    "hi": hi_i,
                    "lo": lo_i,
                    "cts_idx": None if st.cts is None else st.cts.idx,
                    "cts_price": None if st.cts is None else float(st.cts.price),
                    "structure_id": int(st.structure_id),
                    "struct_direction": int(self.struct_direction),
                },
            )
        )
        self._set_state(MarketState.RANGE, confirm_idx, meta={"reason": "range_confirmed", "effective_idx": i})

    def _update_active_range(self, i: int) -> None:
        """
        Expand range_hi/lo using candle i.
        Emit RANGE_UPDATED when expanded.
        """
        st = self.state
        if not st.range_active:
            return

        hi0 = float(st.range_hi) if st.range_hi is not None else float("-inf")
        lo0 = float(st.range_lo) if st.range_lo is not None else float("inf")

        hi1 = max(hi0, float(self._h[i]))
        lo1 = min(lo0, float(self._l[i]))

        if hi1 != hi0 or lo1 != lo0:
            st.range_hi = hi1
            st.range_lo = lo1
            self.events.append(
                StructureEvent(
                    idx=i,
                    category="RANGE",
                    type="RANGE_UPDATED",
                    meta={
                        "hi": hi1,
                        "lo": lo1,
                        "structure_id": int(st.structure_id),
                        "struct_direction": int(self.struct_direction),
                    },
                )
            )

            # CTS_UPDATED can happen while in range per your rule (only if in EST_OR_UPD track)
            # if st.cts is not None and st.cts_phase == "EST_OR_UPD":
            #     cts_ext = self._cts_price_at(i)
            #     if self._is_new_cts_extreme(cts_ext):
            #         self._emit_cts_updated(i, cts_ext, meta={"via": "range_expand"})
            #         st.cts = Point(idx=i, price=cts_ext)
            
        # keep thresholds aligned whenever range is active
        self._sync_thresholds_from_range(i)

    def _deactivate_range(self, i: int, meta: Optional[dict] = None) -> None:
        st = self.state
        if not st.range_active:
            return
        m = dict(meta or {})
        m["structure_id"] = int(st.structure_id)
        m["struct_direction"] = int(self.struct_direction)
        self.events.append(
            StructureEvent(
                idx=i,
                category="RANGE",
                type="RANGE_RESET",
                meta=m,
            )
        )
        st.range_active = False
        st.range_hi = None
        st.range_lo = None
        st.range_start_idx = None
        st.range_confirm_idx = None

    def _ensure_range_on_pullback(self, i: int, pullback_ev: PatternEvent) -> None:
        """
        Week 5 rule update:
        - A valid pullback pattern NEVER resets range.
        - If no active range exists, the first pullback should CREATE a range:
            * struct_direction = +1: range_hi = confirmed CTS price, range_lo = lowest low of the original pullback pattern window (excluding confirmation candle)
            * struct_direction = -1: range_lo = confirmed CTS price, range_hi = highest high of the original pullback pattern window (excluding confirmation candle)
        - If range already exists, expand it to include the pullback apply candle extremes if needed.
        """
        st = self.state
        apply_idx = self._apply_idx(pullback_ev)
        if apply_idx is None:
            return

        if st.cts is None:
            # Should not happen in normal flow (pullback after at least one breakout/CTS),
            # but keep safe.
            return

        cts_price = float(st.cts.price)
        
        # Use extremes from the ORIGINAL pattern window; exclude confirmation candle (but include full original pattern window).
        L = self._pattern_len(pullback_ev)
        end = int(pullback_ev.end_idx)
        start = max(0, end - (L - 1))

        hi_c = float(self._h[start : end + 1].max())
        lo_c = float(self._l[start : end + 1].min())

        # apply_idx = self._apply_idx(pullback_ev)  # keep for logging

        if not st.range_active:
            # Create range anchored to CTS and pullback low/high
            st.range_active = True
            st.range_start_idx = st.cts.idx  # range starts from the CTS that is now confirmed by pullback
            st.range_confirm_idx = apply_idx

            if self.struct_direction == 1:
                st.range_hi = cts_price
                st.range_lo = lo_c
            else:
                st.range_lo = cts_price
                st.range_hi = hi_c

            self.events.append(
                StructureEvent(
                    idx=i,
                    category="RANGE",
                    type="RANGE_STARTED",
                    price=None,
                    meta={
                        "reason": "pullback_created_range",
                        "cts_idx": st.cts.idx,
                        "cts_price": cts_price,
                        "pullback_apply_idx": apply_idx,
                        "hi": float(st.range_hi),
                        "lo": float(st.range_lo),
                        "pat": pullback_ev.name,
                        "structure_id": int(st.structure_id),
                        "struct_direction": int(self.struct_direction),
                    },
                )
            )
            self._sync_thresholds_from_range(i)
            return

        # Range exists: expand if needed (do NOT reset)
        hi0 = float(st.range_hi) if st.range_hi is not None else float("-inf")
        lo0 = float(st.range_lo) if st.range_lo is not None else float("inf")

        hi1 = max(hi0, hi_c)
        lo1 = min(lo0, lo_c)

        if hi1 != hi0 or lo1 != lo0:
            st.range_hi = hi1
            st.range_lo = lo1
            self.events.append(
                StructureEvent(
                    idx=i,
                    category="RANGE",
                    type="RANGE_UPDATED",
                    meta={
                        "reason": "pullback_expand",
                        "hi": hi1,
                        "lo": lo1,
                        "pat": pullback_ev.name,
                        "structure_id": int(st.structure_id),
                        "struct_direction": int(self.struct_direction),
                    },
                )
            )
            self._sync_thresholds_from_range(i)


    # ----------------------------
    # Pattern helpers
    # ----------------------------

    def _pattern_len(self, ev: PatternEvent) -> int:
        """
        Returns the length of the ORIGINAL pattern (not including confirmation candle).
        Assumptions (current Week 4 set):
          - continuous is a 3-candle pattern
          - double_maru, one_maru_continuous, one_maru_opposite are 2-candle patterns
        """
        # Use ev.name because that's what you're already logging (meta={"via": ev.name})
        return 3 if ev.name == "continuous" else 2


    def _apply_idx(self, ev: PatternEvent) -> Optional[int]:
        if ev.status == PatternStatus.CONFIRMED:
            return ev.confirmation_idx
        return ev.end_idx
    
    def _apply_pattern_at_apply_idx(self, ev: PatternEvent, apply_idx: int, kind: str) -> None:
        st = self.state

        # Record pattern ownership/disqualification if you keep it (you said you want to drop most of it)
        # You can keep ONLY apply-candle ownership if desired.
        apply_idx = self._apply_idx(ev)
        if apply_idx is None:
            return
        
        # Reversal is terminal
        if kind == "reversal":
            self._set_state(
                MarketState.REVERSAL,
                apply_idx,
                meta={
                    "reason": "reversal_pattern",
                    "pat": ev.name,
                    "bos_frozen": None if st.reversal_bos_th_frozen is None else float(st.reversal_bos_th_frozen),
                },
            )
            self._clear_reversal_watch()
            return

        # if kind == "breakout":
        #     # BOS confirmation (Part 2, Option B):
        #     # Confirm BOS on the first breakout after CTS has been CONFIRMED (i.e., after a pullback pattern applied).
        #     # BOS price is the pullback extreme between pullback apply and this breakout apply.
        #     if st.cts_phase == "CONFIRMED" and st.last_pullback_pat_apply_idx is not None:
        #         bos_price = self._select_bos_price_on_breakout(apply_idx)
        #         self._emit_bos_confirmed(apply_idx, bos_price, meta={"via": ev.name, "pb_start": st.last_pullback_pat_apply_idx})

        #         # reset pullback-cycle bookkeeping
        #         st.bos_candidate = None
        #         st.false_break_active = False
        #         st.reentered_pullback_after_false_break = False
        #         st.trend_leg_id += 1
        #         st.last_pullback_pat_apply_idx = None


        #     # Breakout breaks range (if active)
        #     if st.range_active:
        #         self._deactivate_range(apply_idx, meta={"reason": "range_breakout", "pat": ev.name})

        #     # cts_price = self._cts_price_at(apply_idx)
        #     # if st.state == MarketState.BREAKOUT:
        #     #     self._emit_cts_updated(apply_idx, cts_price, meta={"via": ev.name})
        #     # else:
        #     #     self._emit_cts_established(apply_idx, cts_price, meta={"via": ev.name})

        #     cts_idx, cts_price = self._cts_from_breakout_event(ev)

        #     # emit CTS event using cts_idx/cts_price
        #     if st.cts is not None:
        #         self._emit_cts_updated(cts_idx, cts_price, meta={"via": ev.name})
        #     else:
        #         self._emit_cts_established(cts_idx, cts_price, meta={"via": ev.name})

        #     st.cts = Point(idx=cts_idx, price=cts_price)
        #     st.cts_phase = "EST_OR_UPD"
        #     st.last_breakout_pat_apply_idx = apply_idx
        #     self._set_state(MarketState.BREAKOUT, apply_idx, meta={"reason": "breakout_pattern", "pat": ev.name})
        #     self._post_apply_range_check(apply_idx)
        #     return

        if kind == "breakout":
            # CTS from breakout window extreme (already correct helper).
            # Resolve before any state mutation so the cycle-0 new-extreme
            # check below can compare against an untouched state.
            cts_idx, cts_price = self._cts_from_breakout_event(ev)

            # Cycle-0 scan-from-start mode (`enforce_cts0_new_extreme`):
            # the breakout reaching this block in scan mode is ALREADY the
            # shared routine's true-first-breakout (selected in
            # `_best_bopb_pattern_at_anchor`), so no gate is needed here —
            # MS establishes it through the normal cycle-0 path below.
            # (The old partial `_cts0_new_extreme_passes` anchor-extreme
            # gate is removed; selection-time gating replaced it.)

            # Breakout breaks range (if active)
            if st.range_active:
                self._deactivate_range(apply_idx, meta={"reason": "range_breakout", "pat": ev.name})

            # Establishing a NEW CTS cycle only if:
            #   - CTS is None (first ever), OR
            #   - previous CTS is CONFIRMED (i.e., we've seen at least one pullback since last CTS cycle start)
            establishing_new_cycle = (st.cts is None) or (st.cts_phase == "CONFIRMED")

            if establishing_new_cycle:
                # advance CTS cycle id for the new CTS
                st.cts_cycle_id += 1
                # self._emit_cts_established(cts_idx, cts_price, meta={"via": ev.name})
                self._emit_cts_established(
                    cts_idx,
                    cts_price,
                    meta={
                        "via": ev.name,
                        # NEW: required for Scenario 2 Exception #2
                        "anchor_idx": int(ev.start_idx) if ev.start_idx is not None else int(apply_idx),
                        "pattern_name": str(ev.name),
                        "confirmed_at": int(apply_idx),  # apply candle that established CTS
                    },
                )

                # Create + confirm BOS simultaneously with CTS establishment
                if st.cts_cycle_id == 0:
                    # bos_price = self._initial_bos_before_first_cts(cts_idx)
                    # self._emit_bos_confirmed(apply_idx, bos_price, meta={"source": "initial_prior_extreme"})
                    bos_idx, bos_price = self._initial_bos_before_first_cts(cts_idx)
                    self._emit_bos_confirmed(
                        bos_idx,
                        bos_price,
                        meta={
                            "source": "initial_prior_extreme",
                            "confirmed_at": apply_idx,
                            "pb_start": self.state.last_pullback_pat_apply_idx,
                        },
                    )
                else:
                    # BOS from pullback extreme (window-based)
                    # Uses existing helper _select_bos_price_on_breakout which references last_pullback_pat_apply_idx
                    # bos_price = self._select_bos_price_on_breakout(apply_idx)
                    # self._emit_bos_confirmed(apply_idx, bos_price, meta={"source": "pullback_extreme", "pb_start": st.last_pullback_pat_apply_idx})
                    bos_idx, bos_price = self._select_bos_on_breakout(apply_idx)
                    self._emit_bos_confirmed(
                        bos_idx,
                        bos_price,
                        meta={
                            "source": "pullback_extreme",
                            "confirmed_at": apply_idx,
                            "pb_start": self.state.last_pullback_pat_apply_idx,
                        },
                    )

                # New cycle => reset CTS confirmation guard & threshold
                st.cts_confirmed_for_idx = None
                st.cts_threshold = None

                # New cycle => reset proximity-based confirmation state
                st.cts_confirmed_method = None
                st.pullback_fired_for_cycle = False
                st.proximity_confirmed_idx = None
                st.cts_confirmed_idx = None
                # Compute BOS inner for the new cycle's proximity check via
                # the resolver wired by structure_engine.py (Part 4 §13.5.b).
                if self._bos_inner_resolver is not None:
                    st.bos_inner_for_cycle = self._bos_inner_resolver(
                        self.df,
                        int(bos_idx),
                        int(self.struct_direction),
                    )
                else:
                    st.bos_inner_for_cycle = None

                # After consuming the pullback to create BOS for the new cycle, clear pullback anchor
                st.last_pullback_pat_apply_idx = None
            else:
                # Not allowed to create a new CTS cycle yet => this breakout just updates CTS (pre-confirm)
                self._emit_cts_updated(cts_idx, cts_price, meta={"via": ev.name})

            # Update current CTS point (always)
            st.cts = Point(idx=cts_idx, price=cts_price)
            st.cts_phase = "EST_OR_UPD"
            st.last_breakout_pat_apply_idx = apply_idx

            # Refresh POI inner snapshot for the cycle (Stage 2). New cycle:
            # uses fresh BOS_n + CTS_n. Continuation breakout (CTS_UPDATED):
            # same BOS, extended CTS — POIs may shift as Fib bounds expand.
            self._refresh_poi_inners_for_cycle()

            self._set_state(MarketState.BREAKOUT, apply_idx, meta={"reason": "breakout_pattern", "pat": ev.name})
            self._post_apply_range_check(apply_idx)
            return


        # pullback
        if kind == "pullback":
            # Mark that a pullback fired for this cycle (used for BOS_n+1 derivation)
            st.pullback_fired_for_cycle = True

            # Ensure range exists or expand it based on pullback pattern.
            # If proximity already created the range, this expands it.
            self._ensure_range_on_pullback(apply_idx, ev)  # note: pass apply_idx (time) not anchor i

            if st.cts_confirmed_method == "sd_zone_proximity":
                # CTS already confirmed via proximity. Emit CTS_RECONFIRMED to
                # log the pullback as the "stronger" reaffirmation. Do not
                # re-emit CTS_CONFIRMED. Phase stays CONFIRMED.
                self._emit_cts_reconfirmed(apply_idx, meta={"via": ev.name})
            else:
                # Standard pullback-based confirmation (existing behavior)
                self._emit_cts_confirmed_once(
                    apply_idx,
                    meta={"via": ev.name},
                    confirmation_method="pullback",
                )
                st.cts_phase = "CONFIRMED"

                # initialize cts_threshold to the confirmed CTS value at confirmation time.
                # NOTE: do NOT reset bos_threshold here — BOS locked at BOS_CONFIRMED and
                # may have legitimately expanded via barrier probes in [BOS_CONFIRMED,
                # CTS_CONFIRMED]; re-initing it to bos_confirmed.price would discard that
                # expansion (see GOTCHAS "bos_threshold is reset to the ORIGINAL BOS").
                if st.cts is not None:
                    st.cts_threshold = float(st.cts.price)

            st.last_pullback_pat_apply_idx = apply_idx
            self._set_state(MarketState.PULLBACK, apply_idx, meta={"reason": "pullback_pattern", "pat": ev.name})
            self._sync_thresholds_from_range(apply_idx)
            return

    def _post_apply_range_check(self, i: int) -> None:
        n = len(self.df)
        D = min(i + self.range_max_k, n - 1)  # range_max_k is 5 by default

        st = self.state

        range_confirmed, confirm_idx = self._is_range_candle_given_confirm(i)

        if range_confirmed:
            # Preserve one-candle event markers across the back-fill loop.
            # These get cleared by _write_df_row, but we need them to be written
            # to the apply candle row by _step_anchor after this function returns.
            saved_bos_event = st.bos_event
            saved_cts_event = st.cts_event

            min_d = D if confirm_idx is None else min(confirm_idx, D)
            for k in range(i, min_d):
                self._replay_step_no_patterns(k, freeze_range=True)

            self._finalize_range_candidate_offline(i)

            # Restore event markers for the final write in _step_anchor
            st.bos_event = saved_bos_event
            st.cts_event = saved_cts_event


    # ----------------------------
    # CTS/BOS helpers & emits
    # ----------------------------

    def _cts_from_breakout_event(self, ev: PatternEvent) -> tuple[int, float]:
        """
        CTS for breakout = extreme of the full confirmed pattern span.
        For SUCCESS patterns: [start_idx..end_idx] (pattern candles only).
        For CONFIRMED patterns: [start_idx..confirmation_idx] (includes confirming candle).
        Returns (cts_idx, cts_price).
        """
        s = int(ev.start_idx)
        e = int(ev.end_idx)
        # Extend span to include confirmation candle when pattern needed confirmation
        if ev.confirmation_idx is not None:
            e = max(e, int(ev.confirmation_idx))
        if e < s:
            s, e = e, s

        if self.struct_direction == 1:
            # bullish structure -> CTS is max high in pattern span
            highs = self._h[s : e + 1]
            k = int(highs.argmax())
            cts_idx = s + k
            cts_price = float(highs[k])
            return cts_idx, cts_price
        else:
            # bearish structure -> CTS is min low in pattern span
            lows = self._l[s : e + 1]
            k = int(lows.argmin())
            cts_idx = s + k
            cts_price = float(lows[k])
            return cts_idx, cts_price

    def _cts_price_at(self, idx: int) -> float:
        if self.struct_direction == 1:
            return float(self._h[idx])
        return float(self._l[idx])

    def _is_new_cts_extreme(self, new_price: float) -> bool:
        st = self.state
        if st.cts is None:
            return True
        if self.struct_direction == 1:
            return new_price > float(st.cts.price)
        return new_price < float(st.cts.price)
    
    def _maybe_update_cts_pre_confirm(self, i: int, *, via: str = "raw") -> None:
        """
        Option B: Before the current CTS is confirmed (cts_phase != CONFIRMED),
        update CTS whenever price makes a new extreme in struct_direction,
        regardless of whether a breakout pattern fired.
        """
        st = self.state
        if st.cts is None:
            return
        if st.cts_phase == "CONFIRMED":
            return

        if self.struct_direction == 1:
            new_price = float(self._h[i])
            if new_price > float(st.cts.price):
                self._emit_cts_updated(i, new_price, meta={"via": via})
                st.cts = Point(idx=i, price=new_price)
                self._refresh_poi_inners_for_cycle()
        else:
            new_price = float(self._l[i])
            if new_price < float(st.cts.price):
                self._emit_cts_updated(i, new_price, meta={"via": via})
                st.cts = Point(idx=i, price=new_price)
                self._refresh_poi_inners_for_cycle()

    # def _initial_bos_before_first_cts(self, cts_idx: int) -> float:
    #     """
    #     Cycle 1 BOS: extreme prior to the first CTS.
    #       - Uptrend: min low in [0 .. cts_idx-1]
    #       - Downtrend: max high in [0 .. cts_idx-1]
    #     """
    #     if cts_idx <= 0:
    #         return float(self.df.iloc[0]["l"]) if self.struct_direction == 1 else float(self.df.iloc[0]["h"])

    #     if self.struct_direction == 1:
    #         return float(self.df.iloc[0:cts_idx]["l"].astype(float).min())
    #     return float(self.df.iloc[0:cts_idx]["h"].astype(float).max())

    def _initial_bos_before_first_cts(self, cts_idx: int) -> tuple[int, float]:
        """
        Cycle 1 BOS: extreme prior to the first CTS.
        - Uptrend: min low in [start_idx .. cts_idx-1]
        - Downtrend: max high in [start_idx .. cts_idx-1]
        Returns (bos_idx, bos_price) where bos_idx is a *positional* index.
        """
        start = self.start_idx

        if cts_idx <= start:
            bos_idx = start
            bos_price = float(self._l[start]) if self.struct_direction == 1 else float(self._h[start])
            return bos_idx, bos_price

        if self.struct_direction == 1:
            series = self._l[start:cts_idx]
            rel = int(series.argmin())          # position within window
            bos_idx = start + rel               # offset by start_idx
            bos_price = float(series[rel])
            return bos_idx, bos_price

        else:
            series = self._h[start:cts_idx]
            rel = int(series.argmax())          # position within window
            bos_idx = start + rel               # offset by start_idx
            bos_price = float(series[rel])
            return bos_idx, bos_price

    def _sync_thresholds_from_range(self, i: int) -> None:
        """
        Threshold behavior (your spec):
        - cts_threshold mirrors the range breakout bound (range_hi for uptrend, range_lo for downtrend).
        Part 3A update:
        - DO NOT update bos_threshold here.
        - NOTE: This sync is not a probe/no-break threshold update, so it does NOT emit threshold-update events.
        """
        st = self.state
        if not st.range_active or st.range_hi is None or st.range_lo is None:
            return

        # st.cts_threshold = float(st.range_hi) if self.struct_direction == 1 else float(st.range_lo)
        new_thr = float(st.range_hi) if self.struct_direction == 1 else float(st.range_lo)
        prev = st.cts_threshold
        if prev is not None and prev != new_thr:
            st.cts_threshold = new_thr

            # This is the ONLY source of CTS threshold change -> emit for zone expansion
            self._emit_cts_threshold_updated(
                i,
                new_thr,
                meta={"prev": None if prev is None else float(prev), "reason": "range_sync"},
            )

    # def _emit_cts_established(self, idx: int, price: float, meta: Optional[dict] = None) -> None:
    #     self.events.append(
    #         StructureEvent(idx=idx, category="STRUCTURE", type="CTS_ESTABLISHED", price=price, meta=meta or {})
    #     )
    #     self.state.cts_event = "CTS_ESTABLISHED"  # written to df row via _write_df_row

    def _get_cts0_tfb(self):
        """Lazily compute (and cache) the cycle-0 true-first-breakout
        decision for scan-from-start mode.

        Delegates to the ONE shared `find_true_first_breakout` routine over
        `[start_idx, effective_end]` against the handed `bos0_inner`
        threshold — the SAME call the unified probe made, so MS re-finds
        the identical CTS_0 by construction. Returns a `TrueFirstBreakout`
        or None (no qualifying breakout in the window — MS then establishes
        no cycle 0, writing NONE-state rows forward).
        """
        if not self._cts0_tfb_computed:
            from engine_v2.structure.true_first_breakout import (
                find_true_first_breakout,
            )
            effective_end = (
                int(self.end_idx) if self.end_idx is not None
                else int(len(self.df) - 1)
            )
            self._cts0_tfb = find_true_first_breakout(
                self._bp,
                int(self.start_idx),
                effective_end,
                int(self.struct_direction),
                self.bos0_inner,
            )
            self._cts0_tfb_computed = True
        return self._cts0_tfb

    def _emit_cts_established(self, idx: int, price: float, meta: Optional[dict] = None) -> None:
        meta2 = dict(meta or {})
        # meta2.setdefault("cycle_id", int(self.state.cts_cycle_id))
        meta2["cycle_id"] = int(self.state.cts_cycle_id)
        meta2["structure_id"] = int(self.state.structure_id)
        meta2["struct_direction"] = int(self.state.struct_direction)
        self.events.append(
            StructureEvent(idx=idx, category="STRUCTURE", type="CTS_ESTABLISHED", price=price, meta=meta2)
        )
        self.state.cts_event = "CTS_ESTABLISHED"

    # def _emit_cts_updated(self, idx: int, price: float, meta: Optional[dict] = None) -> None:
    #     self.events.append(
    #         StructureEvent(idx=idx, category="STRUCTURE", type="CTS_UPDATED", price=price, meta=meta or {})
    #     )
    #     self.state.cts_event = "CTS_UPDATED"

    def _emit_cts_updated(self, idx: int, price: float, meta: Optional[dict] = None) -> None:
        meta2 = dict(meta or {})
        # meta2.setdefault("cycle_id", int(self.state.cts_cycle_id))
        meta2["cycle_id"] = int(self.state.cts_cycle_id)
        meta2["structure_id"] = int(self.state.structure_id)
        meta2["struct_direction"] = int(self.state.struct_direction)
        self.events.append(
            StructureEvent(idx=idx, category="STRUCTURE", type="CTS_UPDATED", price=price, meta=meta2)
        )
        self.state.cts_event = "CTS_UPDATED"

    def _emit_cts_confirmed_once(
        self,
        idx: int,
        meta: Optional[dict] = None,
        confirmation_method: str = "pullback",
    ) -> None:
        st = self.state
        cts_anchor = st.cts.idx if st.cts is not None else None
        if cts_anchor is not None and st.cts_confirmed_for_idx == cts_anchor:
            return

        meta2 = dict(meta or {})
        meta2["cycle_id"] = int(self.state.cts_cycle_id)
        meta2["structure_id"] = int(self.state.structure_id)
        meta2["struct_direction"] = int(self.state.struct_direction)

        # ✅ add these
        meta2["confirmed_at"] = int(idx)                 # pullback candle (confirmation candle)
        meta2["cts_anchor_idx"] = int(cts_anchor) if cts_anchor is not None else None  # CTS being confirmed
        # confirmation_method: "pullback" (existing path) or "sd_zone_proximity" (new path)
        meta2["confirmation_method"] = str(confirmation_method)

        # Carry the CTS extreme price on the event so chart consumers don't have
        # to fall back to the df's cts_price column (which is NaN at the CTS-
        # established candle's own row when confirmed_at > anchor_idx — see the
        # sd_zone_proximity path that can fire 1–2 candles after CTS_ESTABLISHED).
        cts_price_val = float(st.cts.price) if st.cts is not None else None

        self.events.append(
            StructureEvent(idx=idx, category="STRUCTURE", type="CTS_CONFIRMED", price=cts_price_val, meta=meta2)
        )
        st.cts_confirmed_for_idx = cts_anchor
        st.cts_confirmed_method = confirmation_method
        st.cts_confirmed_idx = int(idx)
        st.cts_event = "CTS_CONFIRMED"

        # Lock the cycle-0 snapshot once CTS_0 is confirmed. Subsequent
        # raw-extreme CTS_UPDATEDs (which can keep firing between CTS_0
        # CONFIRMED and BOS_1 CONFIRMED while range is still active) must
        # not mutate `cycle0_data`. Mirrors FibTracker's
        # `_handle_cycle0_cts_confirmed`'s `c0["locked"] = True`.
        if int(st.cts_cycle_id) == 0:
            self._lock_cycle0_data()

    def _emit_cts_reconfirmed(self, idx: int, meta: Optional[dict] = None) -> None:
        """Emit CTS_RECONFIRMED — fires when a valid pullback pattern fires
        AFTER CTS was already confirmed via sd zone proximity. Does not change
        the original CTS_CONFIRMED's idx or method; CTS zone metadata can use
        this event to upgrade its confirmation_method to "pullback" and record
        pb_reconfirm_idx.
        """
        st = self.state
        cts_anchor = st.cts.idx if st.cts is not None else None

        meta2 = dict(meta or {})
        meta2["cycle_id"] = int(self.state.cts_cycle_id)
        meta2["structure_id"] = int(self.state.structure_id)
        meta2["struct_direction"] = int(self.state.struct_direction)
        meta2["confirmed_at"] = int(idx)
        meta2["cts_anchor_idx"] = int(cts_anchor) if cts_anchor is not None else None
        meta2["confirmation_method"] = "pullback"  # the upgrade method

        cts_price_val = float(st.cts.price) if st.cts is not None else None

        self.events.append(
            StructureEvent(idx=idx, category="STRUCTURE", type="CTS_RECONFIRMED", price=cts_price_val, meta=meta2)
        )

    # def _emit_bos_confirmed(self, idx: int, price: float, meta: Optional[dict] = None) -> None:
    #     self.events.append(
    #         StructureEvent(idx=idx, category="STRUCTURE", type="BOS_CONFIRMED", price=price, meta=meta or {})
    #     )
    #     self.state.bos_event = "BOS_CONFIRMED"
    #     self.state.bos_confirmed = Point(idx=idx, price=float(price))
    #     self.state.bos_threshold = float(price)

    def _emit_bos_confirmed(self, idx: int, price: float, meta: Optional[dict] = None) -> None:
        meta2 = dict(meta or {})
        # meta2.setdefault("cycle_id", int(self.state.cts_cycle_id))
        meta2["cycle_id"] = int(self.state.cts_cycle_id)
        meta2["structure_id"] = int(self.state.structure_id)
        meta2["struct_direction"] = int(self.state.struct_direction)
        self.events.append(
            StructureEvent(idx=idx, category="STRUCTURE", type="BOS_CONFIRMED", price=price, meta=meta2)
        )
        self.state.bos_event = "BOS_CONFIRMED"
        self.state.bos_confirmed = Point(idx=idx, price=float(price))
        self.state.bos_threshold = float(price)

    # ----------------------------
    # Zone proximity (Stage 1: BOS-only)
    # ----------------------------

    def _check_proximity_at_candle(self, i: int) -> Optional[Tuple[float, str]]:
        """Check if candle i wicks within proximity threshold of the closest
        sd-direction inner (BOS or active POI). Returns (trigger_inner,
        zone_kind) on hit; None otherwise."""
        st = self.state
        if st.bos_inner_for_cycle is None:
            return None
        return _check_sd_proximity_at_candle(
            self.df,
            candle_idx=i,
            struct_direction=self.struct_direction,
            bos_inner=st.bos_inner_for_cycle,
            threshold=self._proximity_threshold,
            poi_inners=list(st.poi_inners_for_cycle) if st.poi_inners_for_cycle else None,
        )

    def _maybe_confirm_cts_via_proximity(self, i: int) -> None:
        """Per-candle CTS confirmation via sd zone proximity.

        Called from `_replay_step_no_patterns`, so it runs on every candle
        (anchor, back-fill, apply, fallthrough). Two gates are load-bearing —
        see LANDMINES "Dual CTS Proximity Confirmation" (cycle-0 carve-out +
        Rule 1 narrow-gap gate).
        """
        st = self.state
        if st.cts is None or st.bos_threshold is None:
            return
        cycle_gap_ok = (
            abs(float(st.cts.price) - float(st.bos_threshold))
            >= self._min_gap_threshold
        )
        if not (
            st.cts_phase == "EST_OR_UPD"
            and st.cts_cycle_id > 0
            and cycle_gap_ok
            and st.bos_inner_for_cycle is not None
            and i > st.cts.idx
        ):
            return
        hit = self._check_proximity_at_candle(i)
        if hit is None:
            return
        trigger_inner, zone_kind = hit
        self._fire_cts_confirmation_via_proximity(i, trigger_inner, zone_kind)

    def _refresh_poi_inners_for_cycle(self) -> None:
        """Recompute the cycle's POI inner snapshot using current BOS_n + CTS_n.
        Called at CTS_ESTABLISHED (new cycle) and each CTS_UPDATED (CTS
        extended). Stage 2 — adds POI awareness to the proximity check.
        Uses the resolver wired by structure_engine.py (Part 4 §13.5.b)."""
        st = self.state
        if st.cts is None or st.bos_confirmed is None or self._poi_inners_resolver is None:
            st.poi_inners_for_cycle = []
            return
        # While we're on cycle 0, keep the cycle-0 snapshot in sync with
        # the current BOS_0/CTS_0 + imbalance fill state. The snapshot is
        # consumed when cycle 1+ POIs refresh.
        if st.cts_cycle_id == 0:
            self._update_cycle0_data()
        st.poi_inners_for_cycle = self._poi_inners_resolver(
            self.df,
            int(st.bos_confirmed.idx),
            float(st.bos_confirmed.price),
            int(st.cts.idx),
            float(st.cts.price),
            int(self.struct_direction),
            int(st.structure_id),
            int(st.cts_cycle_id),
            float(self._fill_threshold),
            st.cycle0_data,
        )


    def _update_cycle0_data(self) -> None:
        """Refresh the cycle-0 snapshot from current state. Skips when
        cycle 0 has been locked (CTS_0 CONFIRMED already fired) or when
        BOS_0 / CTS_0 aren't both available.

        Snapshot schema (consumed by
        :func:`zones.fib_tracker.select_fib_anchor_for_cycle`):

        - ``bos_idx`` / ``bos_price`` — BOS_0 anchor (immutable once captured)
        - ``cts_idx`` / ``cts_price`` — latest cycle-0 CTS extreme
        - ``has_unfilled`` — at least one imbalance in [BOS_0, CTS_0]
          remains unfilled as of ``cts_idx``
        - ``scenario1`` — always ``None`` (MS does not track Scenario 1;
          see ``select_fib_anchor_for_cycle`` docstring for the contract)
        - ``locked`` — flipped True by ``_lock_cycle0_data`` at CTS_0
          CONFIRMED
        """
        st = self.state
        if st.cts_cycle_id != 0 or st.cts is None or st.bos_confirmed is None:
            return
        existing = st.cycle0_data
        if existing is not None and existing.get("locked"):
            return
        bos_idx = int(st.bos_confirmed.idx)
        bos_price = float(st.bos_confirmed.price)
        cts_idx = int(st.cts.idx)
        cts_price = float(st.cts.price)
        c0_lo = min(bos_idx, cts_idx)
        c0_hi = max(bos_idx, cts_idx)
        # sd-direction filter mirrors FibTracker's cycle-0 snapshot (and the
        # filter applied to Scenario 2 cond1/cond3 in select_fib_anchor_for_cycle).
        # Counter-direction imbalances in [BOS_0, CTS_0] never become POIs, so
        # they shouldn't influence the in-flight Scenario 2 decision either.
        has_unfilled = has_unfilled_imbalance(
            self.df, c0_lo, c0_hi, cts_idx, self._fill_threshold,
            direction=int(st.struct_direction),
        )
        st.cycle0_data = {
            "bos_idx": bos_idx,
            "bos_price": bos_price,
            "cts_idx": cts_idx,
            "cts_price": cts_price,
            "has_unfilled": has_unfilled,
            "scenario1": None,
            "locked": False,
        }

    def _lock_cycle0_data(self) -> None:
        """Flip the cycle-0 snapshot to locked. Called by
        ``_emit_cts_confirmed_once`` when CTS_0 CONFIRMED fires; further
        raw-extreme CTS_UPDATEDs (which can still arrive between CTS_0
        CONFIRMED and BOS_1 CONFIRMED) no longer mutate ``cycle0_data``,
        matching FibTracker's behaviour."""
        if self.state.cycle0_data is not None:
            self.state.cycle0_data["locked"] = True

    def _fire_cts_confirmation_via_proximity(
        self,
        candle_idx: int,
        trigger_inner: float,
        zone_kind: str,
    ) -> None:
        """Fire CTS_CONFIRMED via sd zone proximity. Mirrors the state
        transitions of pullback-based confirmation, including range creation
        with range_lo (sd=+1) / range_hi (sd=-1) seeded by the proximity
        candle's wick (Option B)."""
        st = self.state
        if st.cts_phase == "CONFIRMED":
            return  # already confirmed
        if st.cts is None:
            return  # no CTS to confirm

        # Emit event with method label
        self._emit_cts_confirmed_once(
            candle_idx,
            meta={
                "via": "sd_zone_proximity",
                "trigger_inner": float(trigger_inner),
                "zone_kind": str(zone_kind),
                "proximity_pips": int(self.proximity_pips),
            },
            confirmation_method="sd_zone_proximity",
        )
        st.cts_phase = "CONFIRMED"
        st.proximity_confirmed_idx = int(candle_idx)

        # Create range (Option B: range_lo = proximity candle's low for sd=+1)
        # Mirror the logic in _ensure_range_on_pullback's "create" branch.
        cts_price = float(st.cts.price)
        if not st.range_active:
            st.range_active = True
            st.range_start_idx = int(st.cts.idx)
            st.range_confirm_idx = int(candle_idx)
            if self.struct_direction == 1:
                st.range_hi = cts_price
                st.range_lo = float(self._l[candle_idx])
            else:
                st.range_lo = cts_price
                st.range_hi = float(self._h[candle_idx])
            self.events.append(
                StructureEvent(
                    idx=int(candle_idx),
                    category="RANGE",
                    type="RANGE_STARTED",
                    price=None,
                    meta={
                        "reason": "proximity_created_range",
                        "cts_idx": int(st.cts.idx),
                        "cts_price": cts_price,
                        "proximity_apply_idx": int(candle_idx),
                        "hi": float(st.range_hi),
                        "lo": float(st.range_lo),
                        "structure_id": int(st.structure_id),
                        "struct_direction": int(self.struct_direction),
                    },
                )
            )
            # Proximity CREATED the range (prior state is provably BREAKOUT here:
            # gated on `not range_active`, and no pullback could have confirmed
            # without setting cts_phase=CONFIRMED, which blocks proximity). Set a
            # coherent RANGE state to match _finalize_range_candidate_offline and
            # _ensure_range_on_pullback. Dispatch eligibility is UNCHANGED — line
            # ~992 treats BREAKOUT and RANGE identically (both allow pullback +
            # breakout detection); only the no-create branch leaves state as-is.
            self._set_state(
                MarketState.RANGE,
                candle_idx,
                meta={"reason": "proximity_created_range"},
            )

        # Initialize cts_threshold (mirrors pullback path).
        # NOTE: do NOT reset bos_threshold here — see the matching note on the
        # pullback path and GOTCHAS "bos_threshold is reset to the ORIGINAL BOS".
        st.cts_threshold = cts_price
        # Sync cts_threshold to the live range bound, mirroring the pullback
        # path's inline _sync_thresholds_from_range(apply_idx) call. Without
        # this the proximity path lagged one candle (relied on the next
        # candle's _expand_range). No-op when proximity just created the range
        # (range bound == cts_price); only fires when the range already
        # expanded before confirmation.
        self._sync_thresholds_from_range(candle_idx)

        # Note: when proximity CREATES the range above, state is set to RANGE
        # (coherent with the pullback / offline-finalize paths). When the range
        # already existed, state is left unchanged. Either way pattern dispatch
        # is preserved — both pullback and breakout patterns remain eligible for
        # subsequent candles (line ~992 buckets BREAKOUT and RANGE together).

    def _emit_cts_threshold_updated(self, idx: int, price: float, meta: Optional[dict] = None) -> None:
        meta2 = dict(meta or {})
        meta2["cycle_id"] = int(self.state.cts_cycle_id)
        meta2["structure_id"] = int(self.state.structure_id)
        meta2["struct_direction"] = int(self.state.struct_direction)
        self.events.append(
            StructureEvent(idx=idx, category="STRUCTURE", type="CTS_THRESHOLD_UPDATED", price=float(price), meta=meta2)
        )

    def _emit_bos_threshold_updated(self, idx: int, price: float, meta: Optional[dict] = None) -> None:
        meta2 = dict(meta or {})
        meta2["cycle_id"] = int(self.state.cts_cycle_id)
        meta2["structure_id"] = int(self.state.structure_id)
        meta2["struct_direction"] = int(self.state.struct_direction)
        self.events.append(
            StructureEvent(idx=idx, category="STRUCTURE", type="BOS_THRESHOLD_UPDATED", price=float(price), meta=meta2)
        )

    # def _select_bos_price_on_breakout(self, breakout_apply_idx: int) -> float:
    #     """
    #     Part 2 (Option B): BOS price is computed at confirmation time as the pullback extreme
    #     between the pullback apply candle and the breakout apply candle.

    #     - struct_direction == +1: BOS price = min(low) over [pb_start .. breakout_apply_idx]
    #     - struct_direction == -1: BOS price = max(high) over [pb_start .. breakout_apply_idx]

    #     Fallbacks (should be rare):
    #     - Else fall back to range pullback bound if range is active
    #     - Else fall back to current candle extreme
    #     """
    #     st = self.state

    #     pb_start = st.last_pullback_pat_apply_idx
    #     if pb_start is not None:
    #         s = int(pb_start)
    #         e = int(breakout_apply_idx)
    #         if e < s:
    #             s, e = e, s

    #         if self.struct_direction == 1:
    #             return float(self.df.iloc[s : e + 1]["l"].astype(float).min())
    #         else:
    #             return float(self.df.iloc[s : e + 1]["h"].astype(float).max())

    #     # fallback 1: range pullback side
    #     if st.range_active and st.range_lo is not None and st.range_hi is not None:
    #         return float(st.range_lo) if self.struct_direction == 1 else float(st.range_hi)

    #     # fallback 2: current candle extreme
    #     if self.struct_direction == 1:
    #         return float(self.df.iloc[breakout_apply_idx]["l"])
    #     return float(self.df.iloc[breakout_apply_idx]["h"])

    def _select_bos_on_breakout(self, breakout_apply_idx: int) -> tuple[int, float]:
        """
        Cycle k>1 BOS: select the BOS extreme from the cycle's retracement window.

        Window selection:
        - If a pullback fired for the just-completed cycle: use
          [last_pullback_pat_apply_idx, breakout_apply_idx] (existing behavior).
        - Else if cycle was confirmed via sd zone proximity: use
          [cts_confirmed_idx, breakout_apply_idx] (max retracement across the
          full proximity-to-breakout window).

        Returns (bos_idx, bos_price).
        """
        st = self.state

        # Determine search window start
        if st.pullback_fired_for_cycle and st.last_pullback_pat_apply_idx is not None:
            window_start = st.last_pullback_pat_apply_idx
        elif st.cts_confirmed_idx is not None:
            # Proximity-only confirmation — search from confirmed candle onward
            window_start = st.cts_confirmed_idx
        else:
            # Neither pullback nor proximity confirmed (shouldn't happen if
            # we got here, but safe fallback)
            return self._initial_bos_before_first_cts(breakout_apply_idx)

        s = int(window_start)
        e = int(breakout_apply_idx)
        if e < s:
            s, e = e, s

        if self.struct_direction == 1:
            series = self._l[s : e + 1]
            # bos idx in positional coordinates
            rel = int(series.argmin())
            bos_idx = s + rel
            bos_price = float(series.min())
            return bos_idx, bos_price
        else:
            series = self._h[s : e + 1]
            rel = int(series.argmax())
            bos_idx = s + rel
            bos_price = float(series.max())
            return bos_idx, bos_price

    def _maybe_trigger_reversal(self, i: int) -> None:
        """
        Part 2 v1 reversal rule:
        - Once a BOS threshold exists, if price breaks it in the opposite direction by eps,
          mark REVERSAL and stop the run loop.
        """
        st = self.state
        if st.bos_threshold is None:
            return

        bos = float(st.bos_threshold)
        c = float(self._c[i])
        h = float(self._h[i])
        l = float(self._l[i])

        # Conservative check uses close; you can switch to wick-based later if desired.
        if self.struct_direction == 1:
            # Uptrend: reversal if close breaks below BOS by eps
            if c < bos - self.eps:
                self._set_state(MarketState.REVERSAL, i, meta={"reason": "bos_threshold_broken", "bos": bos})
        else:
            # Downtrend: reversal if close breaks above BOS by eps
            if c > bos + self.eps:
                self._set_state(MarketState.REVERSAL, i, meta={"reason": "bos_threshold_broken", "bos": bos})


    # ----------------------------
    # Pullback-side expansion decision
    # ----------------------------

    def _expanded_pullback_side(
        self,
        i: int,
        prev_hi: float | None = None,
        prev_lo: float | None = None,
    ) -> bool:
        st = self.state

        if not st.range_active or st.range_hi is None or st.range_lo is None:
            return False

        # ✅ Use previous bounds if provided, otherwise fall back to current
        hi = st.range_hi if prev_hi is None else prev_hi
        lo = st.range_lo if prev_lo is None else prev_lo

        if self.struct_direction == 1:
            return float(self._l[i]) < float(lo)
        return float(self._h[i]) > float(hi)

    # ----------------------------
    # Range threshold helpers
    # ----------------------------

    def _range_breakout_threshold(self) -> float:
        st = self.state
        if st.range_hi is None or st.range_lo is None:
            raise ValueError("[market_structure] range thresholds requested but range is not initialized")
        return float(st.range_hi) if self.struct_direction == 1 else float(st.range_lo)

    def _range_pullback_threshold(self) -> float:
        st = self.state
        if st.range_hi is None or st.range_lo is None:
            raise ValueError("[market_structure] range thresholds requested but range is not initialized")
        return float(st.range_lo) if self.struct_direction == 1 else float(st.range_hi)

    # ----------------------------
    # State + df writing
    # ----------------------------

    def _set_state(self, new_state: MarketState, i: int, meta: Optional[dict] = None) -> None:
        st = self.state

        # Uniform schema: always include effective_idx.
        # idx=i is the observed candle; effective_idx defaults to i unless overridden.
        m = dict(meta or {})
        m.setdefault("effective_idx", i)

        if new_state != st.state:
            self.events.append(
                StructureEvent(
                    idx=i,
                    category="STATE",
                    type="STATE_CHANGED",
                    meta={
                        "from": st.state.value,
                        "to": new_state.value,
                        "structure_id": int(st.structure_id),
                        "struct_direction": int(self.struct_direction),
                        **m,
                    },
                )
            )
        st.state = new_state


    def _init_output_arrays(self, n: int) -> None:
        """Path 2b: preallocate the per-column output arrays. `_write_df_row`
        writes into these per candle; `run()` bulk-assigns them to the df at
        the end (one vectorized assignment per column instead of ~27 `df.at`
        scalar writes per candle). Created once per run and reused across
        rewinds (which overwrite cells).

        CRITICAL — init each array FROM the existing df column, NOT from fresh
        defaults. `compute_structure` chains MULTIPLE structures through the
        SAME df (`df2` is reassigned to each structure's run() output and fed to
        the next). The old per-row `df.at` writes only touched THIS structure's
        processed rows; a whole-column flush of default-filled arrays would
        clobber a PRIOR structure's rows back to defaults. Seeding from the df
        means rows this structure never processes flush back unchanged. For the
        first/only structure the df holds `_ensure_output_cols` defaults, so
        `to_numpy(dtype)` reproduces the schema defaults exactly (the `_OUT_*`
        dtypes match those defaults' post-write dtype — verified vs saved CSV).
        `.copy()` so per-candle writes never mutate the df in place pre-flush."""
        self._out: Dict[str, np.ndarray] = {}
        for c in _OUT_INT_NEG1 + _OUT_INT_ZERO:
            self._out[c] = self.df[c].to_numpy(dtype=np.int64).copy()
        for c in _OUT_FLOAT_NEG1 + _OUT_FLOAT_NAN:
            self._out[c] = self.df[c].to_numpy(dtype=np.float64).copy()
        for c in _OUT_OBJ_EMPTY:
            self._out[c] = self.df[c].to_numpy(dtype=object).copy()

    def _flush_output_arrays(self) -> None:
        """Path 2b: bulk-assign the per-column output arrays to df columns.
        Runs after the processing loop, BEFORE terminal reversal stamping +
        invariant checks (which read the df output columns)."""
        for col, arr in self._out.items():
            self.df[col] = arr

    def _ensure_output_cols(self) -> None:
        out = self.df
        # market_state
        if "market_state" not in out.columns:
            out["market_state"] = ""
        # range
        for c in ("range_active", "range_hi", "range_lo", "range_start_idx", "range_confirm_idx"):
            if c not in out.columns:
                out[c] = 0 if c == "range_active" else -1
        # thresholds (derived, but nice for debugging)
        for c in ("breakout_th", "pullback_th", "range_break_frac"):
            if c not in out.columns:
                out[c] = float("nan")
        # CTS/BOS columns
        for c in ("cts_idx", "cts_price", "cts_event", "bos_idx", "bos_price", "bos_event"):
            if c not in out.columns:
                out[c] = -1 if c.endswith("_idx") else ("" if c.endswith("_event") else float("nan"))
        # Part 2 extras
        for c in ("cts_cycle_id", "cts_threshold", "bos_threshold"):
            if c not in out.columns:
                out[c] = float("nan") if c.endswith("_threshold") else 0
        # Part 3 (C): trend progression / cycle stage debug
        for c in ("cycle_stage", "cts_phase_debug"):
            if c not in out.columns:
                out[c] = ""
        # Part 3 (B/D): reversal watch debug columns (for chart + invariants)
        for c in ("reversal_watch_active", "reversal_bos_th_frozen"):
            if c not in out.columns:
                if c == "reversal_watch_active":
                    out[c] = 0
                else:
                    out[c] = float("nan")
        # pending reversal columns
        for c in ("pending_reversal_anchor_idx", "pending_reversal_apply_idx"):
            if c not in out.columns:
                out[c] = -1
        # structure
        for c in ("structure_id", "struct_direction"):
            if c not in out.columns:
                if c == "structure_id":
                    out[c] = -1
                else:
                    out[c] = 0
        # last breakout pat idx
        if "last_breakout_pat_apply_idx" not in out.columns:
            out["last_breakout_pat_apply_idx"] = -1


    def _write_df_row(self, i: int) -> None:
        # Path 2b: write into the preallocated per-column output arrays at
        # positional index `i` (bulk-assigned to df at end of run()). Mirrors
        # the prior per-row `self.df.at[row, col] = ...` writes 1:1 — same
        # values, same None->default handling.
        st = self.state
        out = self._out
        # prev_i mirrors the old `prev_row = index[i-1] if i>0 else row`: for
        # range_break_frac we read the PREVIOUS candle's range bounds (i>0),
        # or the just-written current row for i==0.
        prev_i = i - 1 if i > 0 else i

        out["market_state"][i] = st.state.value
        out["range_active"][i] = int(st.range_active)
        out["range_hi"][i] = float(st.range_hi) if st.range_hi is not None else float("nan")
        out["range_lo"][i] = float(st.range_lo) if st.range_lo is not None else float("nan")
        out["range_start_idx"][i] = int(st.range_start_idx) if st.range_start_idx is not None else -1
        out["range_confirm_idx"][i] = int(st.range_confirm_idx) if st.range_confirm_idx is not None else -1

        if st.range_active and st.range_hi is not None and st.range_lo is not None:
            out["breakout_th"][i] = self._range_breakout_threshold()
            out["pullback_th"][i] = self._range_pullback_threshold()
        else:
            out["breakout_th"][i] = float("nan")
            out["pullback_th"][i] = float("nan")

        out["cts_idx"][i] = int(st.cts.idx) if st.cts is not None else -1
        out["cts_price"][i] = float(st.cts.price) if st.cts is not None else float("nan")
        out["cts_event"][i] = st.cts_event
        out["cts_cycle_id"][i] = int(st.cts_cycle_id)
        out["cts_threshold"][i] = float(st.cts_threshold) if st.cts_threshold is not None else float("nan")
        out["bos_threshold"][i] = float(st.bos_threshold) if st.bos_threshold is not None else float("nan")

        # Part 3 (B/D): reversal watch debug (for chart + invariants)
        out["reversal_watch_active"][i] = int(st.reversal_watch_active)
        out["reversal_bos_th_frozen"][i] = (
            float(st.reversal_bos_th_frozen) if st.reversal_bos_th_frozen is not None else float("nan")
        )

        out["pending_reversal_anchor_idx"][i] = (
            int(st.pending_reversal_anchor_idx) if st.pending_reversal_anchor_idx is not None else -1
        )
        out["pending_reversal_apply_idx"][i] = (
            int(st.pending_reversal_apply_idx) if st.pending_reversal_apply_idx is not None else -1
        )


        # ----------------------------
        # Part 3 (C): trend progression / cycle stage
        # ----------------------------
        out["cts_phase_debug"][i] = str(st.cts_phase)

        if st.cts is None:
            out["cycle_stage"][i] = "NONE"
        else:
            # If CTS is confirmed, we're in the "wait for next breakout" portion of the cycle.
            # Otherwise, we're still in the "wait for pullback confirmation" portion.
            out["cycle_stage"][i] = "SEEK_BREAKOUT" if st.cts_phase == "CONFIRMED" else "SEEK_PULLBACK"

        out["bos_idx"][i] = int(st.bos_confirmed.idx) if st.bos_confirmed is not None else -1
        out["bos_price"][i] = float(st.bos_confirmed.price) if st.bos_confirmed is not None else float("nan")
        out["bos_event"][i] = st.bos_event

        out["last_breakout_pat_apply_idx"][i] = (
            int(st.last_breakout_pat_apply_idx) if st.last_breakout_pat_apply_idx is not None else -1
        )

        # Debug: range body-break fraction on this candle (close-breaks only)
        # If candle CLOSE breaks above range_hi / below range_lo, compute the fraction of the
        # candle's real body that lies beyond the breached threshold.
        # NOTE: the old `is not None` guard on the prev range bounds was always
        # true (df values are never None post-_ensure_output_cols; array values
        # are never None either), so it's dropped — body runs unconditionally,
        # identical to before. prev bounds read from the arrays (range_hi/lo
        # default -1.0 on an unprocessed prev row, matching the int->float
        # upcast of the old df default).
        frac = float("nan")
        prev_hi = out["range_hi"][prev_i]
        prev_lo = out["range_lo"][prev_i]
        o = float(self._o[i])
        c = float(self._c[i])
        body_low = min(o, c)
        body_high = max(o, c)
        body_len = body_high - body_low

        if body_len > 0:
            # Close-break above / below range
            if c > float(prev_hi):
                th = float(prev_hi)
                above = max(0.0, body_high - max(th, body_low))
                frac = above / body_len
            elif c < float(prev_lo):
                th = float(prev_lo)
                below = max(0.0, min(th, body_high) - body_low)
                frac = below / body_len

        out["range_break_frac"][i] = frac

        # Clear one-candle event fields so they don't smear across rows
        st.cts_event = ""
        st.bos_event = ""

        # Structure ID + direction
        out["structure_id"][i] = int(st.structure_id)
        out["struct_direction"][i] = int(st.struct_direction)


    # ----------------------------
    # Week 5 (B): Invariant checks (df-level, low-noise)
    # ----------------------------

    def _check_invariants_df(self) -> None:
        df = self.df

        def _isnan(x) -> bool:
            try:
                return pd.isna(x)
            except Exception:
                return x != x

        # 1) Range bounds consistency when active
        if all(c in df.columns for c in ("range_active", "range_lo", "range_hi")):
            active = df["range_active"].astype(bool)
            lo = df["range_lo"].astype(float)
            hi = df["range_hi"].astype(float)

            bad = active & (lo > hi)
            if bad.any():
                i = int(bad.idxmax())
                raise AssertionError(f"[INV] range_lo > range_hi at idx={i}: lo={lo.loc[i]} hi={hi.loc[i]}")

        # 2) CTS confirmed rows must be coherent
        if all(c in df.columns for c in ("cts_event", "cts_phase_debug", "cycle_stage", "cts_idx", "cts_price")):
            conf = df["cts_event"].astype(str) == "CTS_CONFIRMED"
            if conf.any():
                bad_phase = conf & (df["cts_phase_debug"].astype(str) != "CONFIRMED")
                if bad_phase.any():
                    i = int(bad_phase.idxmax())
                    raise AssertionError(f"[INV] CTS_CONFIRMED but cts_phase_debug!=CONFIRMED at idx={i}")

                bad_stage = conf & (df["cycle_stage"].astype(str) != "SEEK_BREAKOUT")
                if bad_stage.any():
                    i = int(bad_stage.idxmax())
                    raise AssertionError(f"[INV] CTS_CONFIRMED but cycle_stage!=SEEK_BREAKOUT at idx={i}")

                bad_idx = conf & (df["cts_idx"].astype(int) < 0)
                if bad_idx.any():
                    i = int(bad_idx.idxmax())
                    raise AssertionError(f"[INV] CTS_CONFIRMED but cts_idx<0 at idx={i}")

                bad_price = conf & df["cts_price"].apply(_isnan)
                if bad_price.any():
                    i = int(bad_price.idxmax())
                    raise AssertionError(f"[INV] CTS_CONFIRMED but cts_price is NaN at idx={i}")

        # 3) BOS confirmed rows must be coherent
        if all(c in df.columns for c in ("bos_event", "bos_idx", "bos_price")):
            conf = df["bos_event"].astype(str) == "BOS_CONFIRMED"
            if conf.any():
                bad_idx = conf & (df["bos_idx"].astype(int) < 0)
                if bad_idx.any():
                    i = int(bad_idx.idxmax())
                    raise AssertionError(f"[INV] BOS_CONFIRMED but bos_idx<0 at idx={i}")

                bad_price = conf & df["bos_price"].apply(_isnan)
                if bad_price.any():
                    i = int(bad_price.idxmax())
                    raise AssertionError(f"[INV] BOS_CONFIRMED but bos_price is NaN at idx={i}")

        # 4) Reversal watch: frozen must exist; bos_threshold must not change while active
        if all(c in df.columns for c in ("reversal_watch_active", "reversal_bos_th_frozen", "bos_threshold")):
            active = df["reversal_watch_active"].astype(bool)

            if active.any():
                bad_frozen = active & df["reversal_bos_th_frozen"].apply(_isnan)
                if bad_frozen.any():
                    i = int(bad_frozen.idxmax())
                    raise AssertionError(f"[INV] reversal_watch_active but reversal_bos_th_frozen is NaN at idx={i}")

                bos = df["bos_threshold"].astype(float)
                bos_prev = bos.shift(1)
                prev_active = active.shift(1).fillna(False).astype(bool)
                active_b = active.astype(bool)
                changed = active_b & prev_active & (bos != bos_prev)
                if changed.any():
                    i = int(changed.idxmax())
                    raise AssertionError(
                        f"[INV] bos_threshold changed during reversal watch at idx={i}: prev={bos_prev.loc[i]} now={bos.loc[i]}"
                    )

        # 5) Terminal reversal: once reversal appears, all later states must be reversal
        if "market_state" in df.columns:
            rev = df["market_state"].astype(str) == "reversal"
            if rev.any():
                first = int(rev.idxmax())
                later_nonrev = df.loc[first:, "market_state"].astype(str) != "reversal"
                if later_nonrev.any():
                    i = int(later_nonrev.idxmax())
                    raise AssertionError(f"[INV] market_state left reversal after idx={first}, non-reversal at idx={i}")

    # ----------------------------
    # Convert events -> StructureLevel (for downstream consumers like KL zones)
    # ----------------------------

    def _events_to_structure_levels(self) -> List[StructureLevel]:
        """
        Keep the downstream interface stable: produce BOS/CTS levels list.
        We'll emit:
          - CTS: CTS_ESTABLISHED + CTS_UPDATED
          - BOS: BOS_CONFIRMED
        """
        levels: List[StructureLevel] = []
        t = pd.to_datetime(self.df[COL_TIME], utc=True)

        for ev in self.events:
            if ev.category != "STRUCTURE":
                continue
            if ev.type in ("CTS_ESTABLISHED", "CTS_UPDATED") and ev.price is not None:
                levels.append(
                    StructureLevel(
                        time=t.iloc[ev.idx],
                        kind="CTS",
                        direction=self.struct_direction,
                        price=float(ev.price),
                        meta={"event": ev.type, **(ev.meta or {})},
                    )
                )
            if ev.type == "BOS_CONFIRMED" and ev.price is not None:
                levels.append(
                    StructureLevel(
                        time=t.iloc[ev.idx],
                        kind="BOS",
                        direction=self.struct_direction,
                        price=float(ev.price),
                        meta={"event": ev.type, **(ev.meta or {})},
                    )
                )

        return levels

    # ----------------------------
    # Debug
    # ----------------------------
    def _dbg(self, msg: str) -> None:
        # Print only when debug is enabled AND we're not inside a rewind rebuild
        if bool(getattr(self, "debug", False)) and not bool(getattr(self, "_in_rewind", False)):
            print(msg)
