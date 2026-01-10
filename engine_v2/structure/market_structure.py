from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import pandas as pd

from engine_v2.common.types import PatternEvent, PatternStatus, StructureLevel, COL_TIME
from engine_v2.patterns.structure_patterns import BreakoutPatterns


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
    struct_direction: int # structure direction or starting_direction
    state: MarketState = MarketState.NONE
    prev_state: MarketState = MarketState.NONE

    # CTS lifecycle
    cts: Optional[Point] = None
    cts_phase: Literal["NONE", "EST_OR_UPD", "CONFIRMED", "FALSE_BREAK"] = "NONE"
    cts_confirmed_for_idx: Optional[int] = None  # prevent duplicate CTS_CONFIRMED emits for same CTS
    cts_event: str = ""  # one-candle event marker written by _write_df_row

    # BOS lifecycle
    bos_confirmed: Optional[Point] = None
    bos_threshold: Optional[float] = None  # used for reversal checks when implemented
    bos_event: str = ""  # one-candle event marker written by _write_df_row

    # Range (active)
    range_active: bool = False
    range_hi: Optional[float] = None
    range_lo: Optional[float] = None
    range_start_idx: Optional[int] = None

    # Pending range candidate (range evaluation)
    range_candidate_start_idx: Optional[int] = None      # j
    range_candidate_decision_idx: Optional[int] = None   # d = confirm idx if confirmed else fail idx (j+max_k)
    range_candidate_confirm_idx: Optional[int] = None    # c if confirmed else None
    range_candidate_will_confirm: bool = False
    range_candidate_disqualified: bool = False

    # Breakout bookkeeping
    last_breakout_pat_apply_idx: Optional[int] = None  # j

    # False-break bookkeeping
    false_break_active: bool = False
    bos_floor_pre_false_break: Optional[float] = None
    bos_floor_post_false_break: Optional[float] = None
    reentered_pullback_after_false_break: bool = False

    # BOS candidate tracking during pullback
    bos_candidate: Optional[float] = None  # min(low) if struct_direction=+1, max(high) if struct_direction=-1


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

    def __init__(self, df: pd.DataFrame, struct_direction: int, *, eps: float = 0.0001, range_min_k: int = 2, range_max_k: int = 5):
        if struct_direction not in (1, -1):
            raise ValueError(f"[market_structure] struct_direction must be 1 or -1, got {struct_direction}")
        self.df = df.copy()
        self.struct_direction = struct_direction
        self.eps = float(eps)

        self.state = MarketStructureState(struct_direction=struct_direction)
        self.events: List[StructureEvent] = []


        # If a candle is in here, it can never be a range starter.
        # Apply candle indices (end or confirmation candle) for any pattern event.
        self._pattern_owned_apply_idxs: Set[int] = set()

        # Original pattern non-last candles (e.g. for continuous: start and middle; for 2-candle: first candle).
        self._range_disqualify_idxs: Set[int] = set()

        # Pattern detector (uses df feature columns)
        self._bp = BreakoutPatterns(self.df)

        self.range_min_k = int(range_min_k)
        self.range_max_k = int(range_max_k)

        self._in_replay = False

        self._ensure_output_cols()

    # ----------------------------
    # Public API
    # ----------------------------

    def run(self) -> Tuple[pd.DataFrame, List[StructureEvent], List[StructureLevel]]:
        """
        Sequentially labels market_state, CTS/BOS/range fields, and emits StructureEvents.
        No skipping. Uses pending confirmation for patterns + internal range evaluation window (min/max lookahead) with rewind+replay.
        """
        n = len(self.df)
        for i in range(n):
            self._step(i)
            if self.state.state == MarketState.REVERSAL:
                break

        levels = self._events_to_structure_levels()
        return self.df, self.events, levels

    # ----------------------------
    # Per-candle step
    # ----------------------------

    def _step(self, i: int) -> None:
        st = self.state
        st.prev_state = st.state

        # Ensure when state is pullback or pullback_range there is always an active range
        if st.state in (MarketState.PULLBACK, MarketState.PULLBACK_RANGE) and not st.range_active:
            raise RuntimeError(
                f"Invariant violated at i={i}: state={st.state.value} but range_active=False"
            )
        
        # Invariant: RANGE state is only valid when a range is active.
        # If we ever observe state == RANGE while range_active == False,
        # this indicates a state-transition or replay-ordering bug and must fail fast.
        if st.state == MarketState.RANGE and not st.range_active:
            raise RuntimeError(
                "Invalid market state: state=RANGE but range_active=False. "
                "RANGE state must only exist while an active range is present. "
                "This indicates a state-transition or replay ordering bug."
            )

        # 1) Main state machine
        if st.range_active:
            self._step_with_active_range(i)
        else:
            self._step_without_active_range(i)

        # 2) Finalize pending range candidate at decision point (confirm OR fail); 
        # if confirmed, activate range using only candle j hi/lo, then rewind+replay.
        self._finalize_range_candidate_if_due(i)

        # 3) Write df debug columns for this candle
        self._write_df_row(i)

    # ----------------------------
    # State transitions - no active range
    # ----------------------------

    def _step_without_active_range(self, i: int) -> None:
        st = self.state

        # ------------------------------------------------------------
        # ORDERING RULE: patterns always first, range only if no pattern
        # ------------------------------------------------------------

        # 1) Always check breakout patterns
        breakout_ev = self._check_pattern(i, direction=self.struct_direction, break_threshold=None)
        if breakout_ev is not None:
            self._clear_pending_range_candidate()

            self._consume_pattern_apply(breakout_ev)
            apply_idx = self._apply_idx(breakout_ev)
            if apply_idx is None:
                return

            cts_price = self._cts_price_at(apply_idx)

            # CTS established vs updated:
            # - updated only if we were already in BREAKOUT
            # - otherwise established
            if st.state == MarketState.BREAKOUT:
                self._emit_cts_updated(apply_idx, cts_price, meta={"via": breakout_ev.name})
            else:
                self._emit_cts_established(apply_idx, cts_price, meta={"via": breakout_ev.name})

            st.cts = Point(idx=apply_idx, price=cts_price)
            st.cts_phase = "EST_OR_UPD"
            st.last_breakout_pat_apply_idx = apply_idx

            self._set_state(MarketState.BREAKOUT, i, meta={"reason": "breakout_pattern", "pat": breakout_ev.name})
            return

        # 2) If state is not NONE, PULLBACK, or PULLBACK_RANGE, we also check pullback patterns
        if st.state not in (MarketState.NONE, MarketState.PULLBACK, MarketState.PULLBACK_RANGE):
            pullback_ev = self._check_pattern(i, direction=-self.struct_direction, break_threshold=None)
            if pullback_ev is not None:
                self._clear_pending_range_candidate()

                self._consume_pattern_apply(pullback_ev)

                # NEW: pullback should create/expand range; never resets range
                self._ensure_range_on_pullback(i, pullback_ev)

                self._emit_cts_confirmed_once(i, meta={"via": pullback_ev.name})
                st.cts_phase = "CONFIRMED"
                st.bos_candidate = self._init_bos_candidate(i)

                self._set_state(MarketState.PULLBACK, i, meta={"reason": "pullback_pattern", "pat": pullback_ev.name})
                return

        # 3) No patterns at i -> consider range (fallback)
        if st.state == MarketState.NONE:
            # still pre-first-breakout: no ranges created
            return

        # If no pattern fired, we may start (or keep) a pending range candidate anchored at i.
        # Pending-ness is tracked via range_candidate_* fields (not a separate state).
        self._maybe_set_range_candidate_here(i)

        # if st.range_candidate_start_idx is not None and not st.range_active and not st.range_candidate_disqualified:
        #     self._set_state(st.state, i, meta={"reason": "no_pattern"})
        # else:
        #     # Keep current state (BREAKOUT or PULLBACK/PULLBACK_RANGE) if no candidate
        #     self._set_state(st.state, i, meta={"reason": "no_pattern_no_range"})


    # ----------------------------
    # State transitions - active range
    # ----------------------------

    def _step_with_active_range(self, i: int) -> None:
        st = self.state

        # Thresholds depend on starting direction
        breakout_th = self._range_breakout_threshold()
        pullback_th = self._range_pullback_threshold()

        # ------------------------------------------------------------
        # ORDERING RULE: patterns first, range update only if no pattern
        # ------------------------------------------------------------

        # 1) Breakout attempt (break range in starting direction) -> resets range
        breakout_ev = self._check_pattern(i, direction=self.struct_direction, break_threshold=breakout_th)
        if breakout_ev is not None:
            self._clear_pending_range_candidate()

            self._consume_pattern_apply(breakout_ev)
            apply_idx = self._apply_idx(breakout_ev)
            if apply_idx is None:
                return

            # Range reset on breakout in starting direction
            self._deactivate_range(i, meta={"reason": "range_breakout", "pat": breakout_ev.name})

            cts_price = self._cts_price_at(apply_idx)
            self._emit_cts_established(apply_idx, cts_price, meta={"via": breakout_ev.name, "break_threshold": breakout_th})

            st.cts = Point(idx=apply_idx, price=cts_price)
            st.cts_phase = "EST_OR_UPD"
            st.last_breakout_pat_apply_idx = apply_idx

            self._set_state(MarketState.BREAKOUT, i, meta={"reason": "range_breakout"})
            return

        # 2) Pullback attempt (break range opposite direction) -> does NOT reset range
        if st.state not in (MarketState.NONE, MarketState.PULLBACK, MarketState.PULLBACK_RANGE):
            pullback_ev = self._check_pattern(i, direction=-self.struct_direction, break_threshold=pullback_th)
            if pullback_ev is not None:
                self._clear_pending_range_candidate()

                self._consume_pattern_apply(pullback_ev)
                self._emit_cts_confirmed_once(i, meta={"via": pullback_ev.name, "break_threshold": pullback_th})
                st.cts_phase = "CONFIRMED"
                st.bos_candidate = self._init_bos_candidate(i)

                self._set_state(MarketState.PULLBACK, i, meta={"reason": "range_pullback", "pat": pullback_ev.name})
                return

        # 3) No pattern -> now update/expand range bounds
        self._update_active_range(i)

        # 4) Maintain pullback / pullback_range behavior if we are in pullback regime
        if st.state in (MarketState.PULLBACK, MarketState.PULLBACK_RANGE):
            if self._expanded_pullback_side(i):
                if st.false_break_active:
                    st.reentered_pullback_after_false_break = True
                st.bos_candidate = self._update_bos_candidate(i, st.bos_candidate)
                self._set_state(MarketState.PULLBACK, i, meta={"reason": "expand_pullback_side"})
            else:
                self._set_state(MarketState.PULLBACK_RANGE, i, meta={"reason": "no_expand_pullback_side"})
            return

        # Otherwise remain in range
        self._set_state(MarketState.RANGE, i, meta={"reason": "range_persist"})


    # ----------------------------
    # Range candidate creation & activation
    # ----------------------------

    def _clear_pending_range_candidate(self) -> None:
        st = self.state
        st.range_candidate_start_idx = None
        st.range_candidate_decision_idx = None
        st.range_candidate_confirm_idx = None
        st.range_candidate_will_confirm = False
        st.range_candidate_disqualified = False

    def _maybe_set_range_candidate_here(self, i: int) -> None:
        """
        Week 5 RANGE evaluation (updated):

        - Only consider range AFTER we checked patterns at i and found none.
        - A "range candidate" starts an evaluation window of [i+min_k .. i+max_k].
        It either:
            * CONFIRMS at candle c (first close within candidate candle [low_i, high_i])
            * FAILS at decision candle d=i+max_k if no such close occurs
        - Regardless of confirm or fail, when the evaluation ends we will rewind+replay from j+1
        (handled in _finalize_range_candidate_if_due).
        - State must be != NONE (no ranges before first breakout).
        - Do NOT start range evaluation if a range is already active.
        """
        st = self.state

        if st.state == MarketState.NONE:
            return
        if st.range_active:
            return
        if st.range_candidate_start_idx is not None and not st.range_candidate_disqualified:
            # already waiting on a candidate; don't replace it
            return

        # Disqualify if this candle participates in any pattern (apply or original-window non-last)
        if i in self._pattern_owned_apply_idxs:
            return
        if i in self._range_disqualify_idxs:
            return

        # Start RANGE evaluation window (confirm or fail)
        d, c, will_confirm = self._range_decision_for_candidate(i)

        st.range_candidate_start_idx = int(i)               # j
        st.range_candidate_decision_idx = int(d)            # d
        st.range_candidate_confirm_idx = int(c) if c is not None else None  # c or None
        st.range_candidate_will_confirm = bool(will_confirm)
        st.range_candidate_disqualified = False


    def _finalize_range_candidate_if_due(self, i: int) -> None:
        if getattr(self, "_in_replay", False):
            return
        st = self.state
        if st.range_candidate_start_idx is None or st.range_candidate_decision_idx is None:
            return
        if st.range_candidate_disqualified:
            # Even if disqualified, the evaluation ended at decision point; we still rewind+replay.
            if i == st.range_candidate_decision_idx:
                j = int(st.range_candidate_start_idx)
                d = int(st.range_candidate_decision_idx)
                self._clear_pending_range_candidate()
                self._rewind_replay(j=j, d=d)
            return

        d = int(st.range_candidate_decision_idx)
        if i != d:
            return

        j = int(st.range_candidate_start_idx)

        # If confirmed, activate range using ONLY candle j hi/lo
        if st.range_candidate_will_confirm and st.range_candidate_confirm_idx is not None:
            c = int(st.range_candidate_confirm_idx)
            self._activate_range_at_confirm(i=i, j=j, c=c)
        else:
            # fail: no active range; state remains whatever it already is (usually BREAKOUT)
            self._clear_pending_range_candidate()

        # ALWAYS rewind + replay from j+1 after range evaluation ends (confirm or fail)
        self._rewind_replay(j=j, d=d)


    def _activate_range_at_confirm(self, i: int, j: int, c: int) -> None:
        st = self.state

        # Disqualify if the starter candle later became pattern apply candle
        if j in self._pattern_owned_apply_idxs:
            st.range_candidate_disqualified = True
            self._clear_pending_range_candidate()
            return
        
        # NEW: disqualify if the starter candle participated in the original portion of any pattern
        if j in self._range_disqualify_idxs:
            st.range_candidate_disqualified = True
            self._clear_pending_range_candidate()
            return

        hi = float(self.df.iloc[j]["h"])
        lo = float(self.df.iloc[j]["l"])

        st.range_active = True
        st.range_start_idx = j
        st.range_hi = hi
        st.range_lo = lo

        self.events.append(
            StructureEvent(
                idx=i,
                category="RANGE",
                type="RANGE_STARTED",
                price=None,
                meta={"start_idx": j, "confirm_idx": c, "hi": hi, "lo": lo},
            )
        )
        self._set_state(MarketState.RANGE, i, meta={"reason": "range_confirmed", "effective_idx": j})
        self._clear_pending_range_candidate()

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

        hi1 = max(hi0, float(self.df.iloc[i]["h"]))
        lo1 = min(lo0, float(self.df.iloc[i]["l"]))

        if hi1 != hi0 or lo1 != lo0:
            st.range_hi = hi1
            st.range_lo = lo1
            self.events.append(
                StructureEvent(
                    idx=i,
                    category="RANGE",
                    type="RANGE_UPDATED",
                    meta={"hi": hi1, "lo": lo1},
                )
            )

            # CTS_UPDATED can happen while in range per your rule (only if in EST_OR_UPD track)
            if st.cts is not None and st.cts_phase == "EST_OR_UPD":
                cts_ext = self._cts_price_at(i)
                if self._is_new_cts_extreme(cts_ext):
                    self._emit_cts_updated(i, cts_ext, meta={"via": "range_expand"})
                    st.cts = Point(idx=i, price=cts_ext)

    def _deactivate_range(self, i: int, meta: Optional[dict] = None) -> None:
        st = self.state
        if not st.range_active:
            return
        self.events.append(
            StructureEvent(
                idx=i,
                category="RANGE",
                type="RANGE_RESET",
                meta=meta or {},
            )
        )
        st.range_active = False
        st.range_hi = None
        st.range_lo = None
        st.range_start_idx = None

        # clear pending range candidate too
        self._clear_pending_range_candidate()

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
        
        # Use extremes from the ORIGINAL pattern window (exclude confirmation candle)
        L = self._pattern_len(pullback_ev)
        end = int(pullback_ev.end_idx)
        start = max(0, end - (L - 1))

        hi_c = float(self.df.iloc[start : end + 1]["h"].max())
        lo_c = float(self.df.iloc[start : end + 1]["l"].min())

        # apply_idx = self._apply_idx(pullback_ev)  # keep for logging

        if not st.range_active:
            # Create range anchored to CTS and pullback low/high
            st.range_active = True
            st.range_start_idx = st.cts.idx  # range starts from the CTS that is now confirmed by pullback

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
                    },
                )
            )
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
                    meta={"reason": "pullback_expand", "hi": hi1, "lo": lo1, "pat": pullback_ev.name},
                )
            )

    def _range_decision_for_candidate(self, j: int) -> Tuple[int, Optional[int], bool]:
        """
        Returns (decision_idx d, confirm_idx c_or_None, will_confirm).
        Confirm rule: exists t in [j+min_k .. j+max_k] s.t. close[t] in [low[j], high[j]].
        If none, decision_idx = min(j+max_k, n-1) and will_confirm=False.
        """
        n = len(self.df)
        lo_j = float(self.df.iloc[j]["l"])
        hi_j = float(self.df.iloc[j]["h"])

        start = j + self.range_min_k
        end = min(j + self.range_max_k, n - 1)

        for t in range(start, end + 1):
            c_t = float(self.df.iloc[t]["c"])
            if lo_j <= c_t <= hi_j:
                return t, t, True

        return end, None, False

    def _drop_events_in_idx_range(self, a: int, b: int) -> None:
        """Remove previously-emitted events with idx in [a..b]."""
        self.events = [e for e in self.events if not (a <= int(e.idx) <= b)]


    def _rewind_replay(self, j: int, d: int) -> None:
        """
        Week 5 rule:
        After RANGE evaluation ends (confirm OR fail), ALWAYS rewind+replay from candle j+1
        through the decision candle d (inclusive).

        This is the "go back in time but quickly processable" behavior:
        we re-process candles that were previously processed without the now-known outcome
        of the range evaluation.
        """
        if d <= j:
            return

        # Remove events that were emitted for the segment we will replay.
        # We'll re-emit correct events during replay.
        self._drop_events_in_idx_range(j + 1, d)

        # Re-run the state machine for candles j+1..d.
        # Important: during this replay, we should NOT finalize range candidates again
        # at the same indices to avoid recursion / duplicated rewinds.
        # We'll guard using a flag.

        prev = self._in_replay
        self._in_replay = True
        try:
            for k in range(j + 1, d + 1):
                # Replay the step logic, but skip finalize to prevent nested rewinds.
                self._step_replay(k)
        finally:
            self._in_replay = prev

    def _step_replay(self, i: int) -> None:
        """
        Replay version of _step:
        - runs the same step_with/without_range logic and writes df row
        - DOES NOT call _finalize_range_candidate_if_due (prevents recursive rewinds)
        """
        st = self.state
        st.prev_state = st.state

        # Ensure when state is pullback or pullback_range there is always an active range
        if st.state in (MarketState.PULLBACK, MarketState.PULLBACK_RANGE) and not st.range_active:
            raise RuntimeError(
                f"Invariant violated at i={i}: state={st.state.value} but range_active=False"
            )
        
        # Invariant: RANGE state is only valid when a range is active.
        # If we ever observe state == RANGE while range_active == False,
        # this indicates a state-transition or replay-ordering bug and must fail fast.
        if st.state == MarketState.RANGE and not st.range_active:
            raise RuntimeError(
                "Invalid market state: state=RANGE but range_active=False. "
                "RANGE state must only exist while an active range is present. "
                "This indicates a state-transition or replay ordering bug."
            )

        if st.range_active:
            self._step_with_active_range(i)
        else:
            self._step_without_active_range(i)

        self._write_df_row(i)


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

    def _original_pattern_non_last_idxs(self, ev: PatternEvent) -> List[int]:
        """
        Returns indices of candles in the original pattern EXCLUDING the last candle.
        - Uses ev.end_idx as the last candle of the original pattern.
        - Does NOT include confirmation candle even if ev.status == CONFIRMED.
        """
        L = self._pattern_len(ev)
        end = int(ev.end_idx)
        start = end - (L - 1)
        # exclude the last original candle (end)
        return [k for k in range(start, end) if k >= 0]


    def _check_pattern(self, i: int, direction: int, break_threshold: Optional[float]) -> Optional[PatternEvent]:
        # Sequential use: call Week 4 detector dynamically with threshold
        ev = self._bp.detect_first_success(i, direction, break_threshold)
        return ev

    def _apply_idx(self, ev: PatternEvent) -> Optional[int]:
        if ev.status == PatternStatus.CONFIRMED:
            return ev.confirmation_idx
        return ev.end_idx

    def _consume_pattern_apply(self, ev: PatternEvent) -> None:
        """
        Record that the apply candle is owned by a breakout/pullback pattern,
        disqualifying it from being a range starter later.
        """
        apply_idx = self._apply_idx(ev)
        if apply_idx is None:
            return
        self._pattern_owned_apply_idxs.add(int(apply_idx))

        # If we are currently evaluating a range candidate and it later becomes
        # the apply candle of a pattern, retroactively disqualify it.
        st = self.state
        if (
            st.range_candidate_start_idx is not None
            and int(st.range_candidate_start_idx) == int(apply_idx)
        ):
            st.range_candidate_disqualified = True

        # NEW: disqualify any candle that participated in the ORIGINAL pattern window
        # (excluding the last candle of the original pattern).
        for k in self._original_pattern_non_last_idxs(ev):
            self._range_disqualify_idxs.add(int(k))

        # Retroactively disqualify pending range candidate if it matches
        if st.range_candidate_start_idx is not None and int(st.range_candidate_start_idx) in self._range_disqualify_idxs:
            st.range_candidate_disqualified = True


    # ----------------------------
    # CTS/BOS helpers & emits
    # ----------------------------

    def _cts_price_at(self, idx: int) -> float:
        if self.struct_direction == 1:
            return float(self.df.iloc[idx]["h"])
        return float(self.df.iloc[idx]["l"])

    def _is_new_cts_extreme(self, new_price: float) -> bool:
        st = self.state
        if st.cts is None:
            return True
        if self.struct_direction == 1:
            return new_price > float(st.cts.price)
        return new_price < float(st.cts.price)

    def _emit_cts_established(self, idx: int, price: float, meta: Optional[dict] = None) -> None:
        self.events.append(
            StructureEvent(idx=idx, category="STRUCTURE", type="CTS_ESTABLISHED", price=price, meta=meta or {})
        )
        self.state.cts_event = "CTS_ESTABLISHED"  # written to df row via _write_df_row

    def _emit_cts_updated(self, idx: int, price: float, meta: Optional[dict] = None) -> None:
        self.events.append(
            StructureEvent(idx=idx, category="STRUCTURE", type="CTS_UPDATED", price=price, meta=meta or {})
        )
        self.state.cts_event = "CTS_UPDATED"

    def _emit_cts_confirmed_once(self, idx: int, meta: Optional[dict] = None) -> None:
        st = self.state
        # confirm only once per CTS anchor idx
        cts_anchor = st.cts.idx if st.cts is not None else None
        if cts_anchor is not None and st.cts_confirmed_for_idx == cts_anchor:
            return
        self.events.append(
            StructureEvent(idx=idx, category="STRUCTURE", type="CTS_CONFIRMED", price=None, meta=meta or {})
        )
        st.cts_confirmed_for_idx = cts_anchor
        st.cts_event = "CTS_CONFIRMED"

    def _emit_bos_confirmed(self, idx: int, price: float, meta: Optional[dict] = None) -> None:
        self.events.append(
            StructureEvent(idx=idx, category="STRUCTURE", type="BOS_CONFIRMED", price=price, meta=meta or {})
        )
        self.state.bos_event = "BOS_CONFIRMED"
        self.state.bos_confirmed = Point(idx=idx, price=float(price))
        self.state.bos_threshold = float(price)

    def _init_bos_candidate(self, i: int) -> float:
        # start tracking from current candle forward (cheap v1)
        if self.struct_direction == 1:
            return float(self.df.iloc[i]["l"])
        return float(self.df.iloc[i]["h"])

    def _update_bos_candidate(self, i: int, cur: Optional[float]) -> float:
        if self.struct_direction == 1:
            val = float(self.df.iloc[i]["l"])
            return val if cur is None else min(float(cur), val)
        val = float(self.df.iloc[i]["h"])
        return val if cur is None else max(float(cur), val)

    def _select_bos_price_on_breakout(self) -> float:
        """
        Implements your false-break BOS selection nuance (v1 subset).

        For now (since we haven't implemented CTS_FALSE_BREAK_UPDATED fully yet),
        we use the current tracked bos_candidate, else fallback to active range pullback bound.
        """
        st = self.state
        if st.bos_candidate is not None:
            return float(st.bos_candidate)
        # fallback: use range pullback side
        if st.range_active and st.range_lo is not None and st.range_hi is not None:
            if self.struct_direction == 1:
                return float(st.range_lo)
            return float(st.range_hi)
        # ultimate fallback: current candle extreme
        if self.struct_direction == 1:
            return float(self.df.iloc[-1]["l"])
        return float(self.df.iloc[-1]["h"])

    # ----------------------------
    # Pullback-side expansion decision
    # ----------------------------

    def _expanded_pullback_side(self, i: int) -> bool:
        st = self.state
        if not st.range_active or st.range_hi is None or st.range_lo is None:
            # without a range, use bos_candidate progression as proxy
            if st.bos_candidate is None:
                return True
            if self.struct_direction == 1:
                return float(self.df.iloc[i]["l"]) < float(st.bos_candidate)
            return float(self.df.iloc[i]["h"]) > float(st.bos_candidate)

        # With active range:
        if self.struct_direction == 1:
            return float(self.df.iloc[i]["l"]) < float(st.range_lo)
        return float(self.df.iloc[i]["h"]) > float(st.range_hi)

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
                    meta={"from": st.state.value, "to": new_state.value, **m},
                )
            )
        st.state = new_state


    def _ensure_output_cols(self) -> None:
        out = self.df
        # market_state
        if "market_state" not in out.columns:
            out["market_state"] = ""
        # range
        for c in ("range_active", "range_hi", "range_lo", "range_start_idx"):
            if c not in out.columns:
                out[c] = 0 if c == "range_active" else -1
        # thresholds (derived, but nice for debugging)
        for c in ("breakout_th", "pullback_th"):
            if c not in out.columns:
                out[c] = float("nan")
        # CTS/BOS columns
        for c in ("cts_idx", "cts_price", "cts_event", "bos_idx", "bos_price", "bos_event"):
            if c not in out.columns:
                out[c] = -1 if c.endswith("_idx") else ("" if c.endswith("_event") else float("nan"))

        # range candidate
        idx_cols = (
            "range_candidate_start_idx",
            "range_candidate_decision_idx",
            "range_candidate_confirm_idx",
        )
        bool_cols = (
            "range_candidate_will_confirm",
            "range_candidate_disqualified",
        )

        for c in idx_cols:
            if c not in out.columns:
                out[c] = -1

        for c in bool_cols:
            if c not in out.columns:
                out[c] = 0

        # last breakout pat idx
        if "last_breakout_pat_apply_idx" not in out.columns:
            out["last_breakout_pat_apply_idx"] = -1

    def _write_df_row(self, i: int) -> None:
        st = self.state
        row = self.df.index[i]

        self.df.at[row, "market_state"] = st.state.value
        self.df.at[row, "range_active"] = int(st.range_active)
        self.df.at[row, "range_hi"] = float(st.range_hi) if st.range_hi is not None else float("nan")
        self.df.at[row, "range_lo"] = float(st.range_lo) if st.range_lo is not None else float("nan")
        self.df.at[row, "range_start_idx"] = int(st.range_start_idx) if st.range_start_idx is not None else -1

        if st.range_active and st.range_hi is not None and st.range_lo is not None:
            self.df.at[row, "breakout_th"] = self._range_breakout_threshold()
            self.df.at[row, "pullback_th"] = self._range_pullback_threshold()
        else:
            self.df.at[row, "breakout_th"] = float("nan")
            self.df.at[row, "pullback_th"] = float("nan")

        self.df.at[row, "cts_idx"] = int(st.cts.idx) if st.cts is not None else -1
        self.df.at[row, "cts_price"] = float(st.cts.price) if st.cts is not None else float("nan")
        self.df.at[row, "cts_event"] = st.cts_event

        self.df.at[row, "bos_idx"] = int(st.bos_confirmed.idx) if st.bos_confirmed is not None else -1
        self.df.at[row, "bos_price"] = float(st.bos_confirmed.price) if st.bos_confirmed is not None else float("nan")
        self.df.at[row, "bos_event"] = st.bos_event

        self.df.at[row, "range_candidate_start_idx"] = (
            int(st.range_candidate_start_idx) if st.range_candidate_start_idx is not None else -1
        )

        self.df.at[row, "range_candidate_decision_idx"] = (
            int(st.range_candidate_decision_idx) if st.range_candidate_decision_idx is not None else -1
        )

        self.df.at[row, "range_candidate_confirm_idx"] = (
            int(st.range_candidate_confirm_idx) if st.range_candidate_confirm_idx is not None else -1
        )

        self.df.at[row, "range_candidate_will_confirm"] = int(bool(st.range_candidate_will_confirm))

        self.df.at[row, "range_candidate_disqualified"] = int(bool(st.range_candidate_disqualified))

        self.df.at[row, "last_breakout_pat_apply_idx"] = (
            int(st.last_breakout_pat_apply_idx) if st.last_breakout_pat_apply_idx is not None else -1
        )

        # Clear one-candle event fields so they don't smear across rows
        st.cts_event = ""
        st.bos_event = ""

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

