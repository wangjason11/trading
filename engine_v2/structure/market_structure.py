from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import pandas as pd

from engine_v2.common.types import PatternEvent, PatternStatus, StructureLevel, COL_TIME, COL_O, COL_C
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

    # NEW (Part 2): CTS cycles + thresholds
    cts_cycle_id: int = 0  # increments only when we ESTABLISH a NEW CTS cycle
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

        # Pattern detector (uses df feature columns)
        self._bp = BreakoutPatterns(self.df)

        self.range_min_k = int(range_min_k)
        self.range_max_k = int(range_max_k)

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
        i = 0
        while i < n:
            # Stop on reversal
            if self.state.state == MarketState.REVERSAL:
                break

            i = self._step_anchor(i)

        levels = self._events_to_structure_levels()
        return self.df, self.events, levels

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
        st = self.state
        if not freeze_range:
            # If an active range exists, evolve it candle-by-candle (unless frozen).
            if st.range_active:
                prev_hi = st.range_hi
                prev_lo = st.range_lo
                
                self._update_active_range(i)

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
        self._maybe_update_cts_pre_confirm(i, via="replay_raw")

        self._write_df_row(i)

    # ----------------------------

    def _step_anchor(self, i: int) -> int:
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
        winner = self._best_bopb_pattern_at_anchor(
            i=i,
            breakout_th=breakout_th,
            pullback_th=pullback_th,
            D=D,
        )

        if winner is not None:
            ev, apply_idx, kind = winner  # kind in {"breakout","pullback"}

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
            return apply_idx + 1

        # 2) No valid pattern by D:
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
        return i + 1

    def _best_bopb_pattern_at_anchor(
        self,
        *,
        i: int,
        breakout_th: Optional[float],
        pullback_th: Optional[float],
        D: int,
    ) -> Optional[Tuple[PatternEvent, int, Literal["breakout", "pullback"]]]:
        st = self.state

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
            # if i in (169, 170):
            #     omo = self._bp.one_maru_opposite(i, -self.struct_direction, pullback_th, do_confirm=False)
            #     omc = self._bp.one_maru_continuous(i, -self.struct_direction, pullback_th, do_confirm=False)
            #     dm  = self._bp.double_maru(i, -self.struct_direction, pullback_th, do_confirm=False)
            #     print(f"[DBG] i={i} pullback_th={pullback_th} "
            #         f"OMO={None if omo is None else omo.status} "
            #         f"OMC={None if omc is None else omc.status} "
            #         f"DM={None if dm is None else dm.status}")
            #     if omc is not None:
            #         print(f"[DBG] OMC start={omc.start_idx} end={omc.end_idx} conf={omc.confirmation_idx}")
            #         print(f"[DBG] candle+1 close={self.df.iloc[i+1]['c']} high={self.df.iloc[i+1]['h']}")

            if ev_p is not None:
                apply_p = self._apply_idx(ev_p)
                if apply_p is not None and apply_p <= D:
                    candidates.append((ev_p, apply_p, "pullback"))

        if not candidates:
            return None

        # Choose earliest apply time; tie-breaker: breakout wins over pullback
        candidates.sort(key=lambda x: (x[1], 0 if x[2] == "breakout" else 1))
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

        lo_i = float(self.df.iloc[i]["l"])
        hi_i = float(self.df.iloc[i]["h"])
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
            # if st.cts is not None and st.cts_phase == "EST_OR_UPD":
            #     cts_ext = self._cts_price_at(i)
            #     if self._is_new_cts_extreme(cts_ext):
            #         self._emit_cts_updated(i, cts_ext, meta={"via": "range_expand"})
            #         st.cts = Point(idx=i, price=cts_ext)
            
        # keep thresholds aligned whenever range is active
        self._sync_thresholds_from_range()

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

        hi_c = float(self.df.iloc[start : end + 1]["h"].max())
        lo_c = float(self.df.iloc[start : end + 1]["l"].min())

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
                    },
                )
            )
            self._sync_thresholds_from_range()
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
            self._sync_thresholds_from_range()


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
            # Breakout breaks range (if active)
            if st.range_active:
                self._deactivate_range(apply_idx, meta={"reason": "range_breakout", "pat": ev.name})

            # CTS from breakout window extreme (already correct helper)
            cts_idx, cts_price = self._cts_from_breakout_event(ev)

            # Establishing a NEW CTS cycle only if:
            #   - CTS is None (first ever), OR
            #   - previous CTS is CONFIRMED (i.e., we've seen at least one pullback since last CTS cycle start)
            establishing_new_cycle = (st.cts is None) or (st.cts_phase == "CONFIRMED")

            if establishing_new_cycle:
                # advance CTS cycle id for the new CTS
                st.cts_cycle_id += 1
                self._emit_cts_established(cts_idx, cts_price, meta={"via": ev.name})

                # Create + confirm BOS simultaneously with CTS establishment
                if st.cts_cycle_id == 1:
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

                # After consuming the pullback to create BOS for the new cycle, clear pullback anchor
                st.last_pullback_pat_apply_idx = None
            else:
                # Not allowed to create a new CTS cycle yet => this breakout just updates CTS (pre-confirm)
                self._emit_cts_updated(cts_idx, cts_price, meta={"via": ev.name})

            # Update current CTS point (always)
            st.cts = Point(idx=cts_idx, price=cts_price)
            st.cts_phase = "EST_OR_UPD"
            st.last_breakout_pat_apply_idx = apply_idx

            self._set_state(MarketState.BREAKOUT, apply_idx, meta={"reason": "breakout_pattern", "pat": ev.name})
            self._post_apply_range_check(apply_idx)
            return


        # pullback
        if kind == "pullback":
            # Ensure range exists or expand it based on pullback pattern
            self._ensure_range_on_pullback(apply_idx, ev)  # note: pass apply_idx (time) not anchor i

            self._emit_cts_confirmed_once(apply_idx, meta={"via": ev.name})
            st.cts_phase = "CONFIRMED"

            # initialize thresholds to the confirmed CTS/BOS values at confirmation time
            if st.cts is not None:
                st.cts_threshold = float(st.cts.price)
            if st.bos_confirmed is not None:
                st.bos_threshold = float(st.bos_confirmed.price)

            st.last_pullback_pat_apply_idx = apply_idx
            self._set_state(MarketState.PULLBACK, apply_idx, meta={"reason": "pullback_pattern", "pat": ev.name})
            self._sync_thresholds_from_range()
            return

    def _post_apply_range_check(self, i: int) -> None:
        n = len(self.df)
        D = min(i + self.range_max_k, n - 1)  # range_max_k is 5 by default

        st = self.state

        range_confirmed, confirm_idx = self._is_range_candle_given_confirm(i)

        if range_confirmed:
            min_d = D if confirm_idx is None else min(confirm_idx, D)
            for k in range(i, min_d):
                self._replay_step_no_patterns(k, freeze_range=True)

            self._finalize_range_candidate_offline(i)


    # ----------------------------
    # CTS/BOS helpers & emits
    # ----------------------------

    def _cts_from_breakout_event(self, ev: PatternEvent) -> tuple[int, float]:
        """
        CTS for breakout = extreme of the breakout pattern candle span [start_idx..end_idx].
        Returns (cts_idx, cts_price).
        """
        s = int(ev.start_idx)
        e = int(ev.end_idx)
        if e < s:
            s, e = e, s

        if self.struct_direction == 1:
            # bullish structure -> CTS is max high in pattern span
            highs = self.df.iloc[s : e + 1]["h"].astype(float).values
            k = int(highs.argmax())
            cts_idx = s + k
            cts_price = float(highs[k])
            return cts_idx, cts_price
        else:
            # bearish structure -> CTS is min low in pattern span
            lows = self.df.iloc[s : e + 1]["l"].astype(float).values
            k = int(lows.argmin())
            cts_idx = s + k
            cts_price = float(lows[k])
            return cts_idx, cts_price

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
            new_price = float(self.df.iloc[i]["h"])
            if new_price > float(st.cts.price) + self.eps:
                self._emit_cts_updated(i, new_price, meta={"via": via})
                st.cts = Point(idx=i, price=new_price)
        else:
            new_price = float(self.df.iloc[i]["l"])
            if new_price < float(st.cts.price) - self.eps:
                self._emit_cts_updated(i, new_price, meta={"via": via})
                st.cts = Point(idx=i, price=new_price)

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
        - Uptrend: min low in [0 .. cts_idx-1]
        - Downtrend: max high in [0 .. cts_idx-1]
        Returns (bos_idx, bos_price) where bos_idx is a *positional* index.
        """
        if cts_idx <= 0:
            bos_idx = 0
            bos_price = float(self.df.iloc[0]["l"]) if self.struct_direction == 1 else float(self.df.iloc[0]["h"])
            return bos_idx, bos_price

        window = self.df.iloc[0:cts_idx]

        if self.struct_direction == 1:
            series = window["l"].astype(float)
            rel = int(series.values.argmin())   # position within window
            bos_idx = rel                       # window starts at 0 => absolute position == rel
            bos_price = float(series.values[rel])
            return bos_idx, bos_price

        else:
            series = window["h"].astype(float)
            rel = int(series.values.argmax())   # position within window
            bos_idx = rel
            bos_price = float(series.values[rel])
            return bos_idx, bos_price

    def _sync_thresholds_from_range(self) -> None:
        """
        Threshold behavior (your spec):
          - cts_threshold mirrors the range breakout bound (range_hi for uptrend, range_lo for downtrend).
          - bos_threshold expands using the *more extreme* of (current bos_threshold, pullback-side range bound).
            * Uptrend: BOS is a low -> more extreme means lower -> compare against range_lo
            * Downtrend: BOS is a high -> more extreme means higher -> compare against range_hi
        """
        st = self.state
        if not st.range_active or st.range_hi is None or st.range_lo is None:
            return

        # mirror CTS threshold to range breakout bound
        st.cts_threshold = float(st.range_hi) if self.struct_direction == 1 else float(st.range_lo)

        # expand BOS threshold using pullback-side bound
        if st.bos_threshold is None:
            return
        if self.struct_direction == 1:
            st.bos_threshold = min(float(st.bos_threshold), float(st.range_lo))
        else:
            st.bos_threshold = max(float(st.bos_threshold), float(st.range_hi))


    # def _emit_cts_established(self, idx: int, price: float, meta: Optional[dict] = None) -> None:
    #     self.events.append(
    #         StructureEvent(idx=idx, category="STRUCTURE", type="CTS_ESTABLISHED", price=price, meta=meta or {})
    #     )
    #     self.state.cts_event = "CTS_ESTABLISHED"  # written to df row via _write_df_row

    def _emit_cts_established(self, idx: int, price: float, meta: Optional[dict] = None) -> None:
        meta2 = dict(meta or {})
        # meta2.setdefault("cycle_id", int(self.state.cts_cycle_id))
        meta2["cycle_id"] = int(self.state.cts_cycle_id)
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
        self.events.append(
            StructureEvent(idx=idx, category="STRUCTURE", type="CTS_UPDATED", price=price, meta=meta2)
        )
        self.state.cts_event = "CTS_UPDATED"

    def _emit_cts_confirmed_once(self, idx: int, meta: Optional[dict] = None) -> None:
        st = self.state
        # confirm only once per CTS anchor idx
        cts_anchor = st.cts.idx if st.cts is not None else None
        if cts_anchor is not None and st.cts_confirmed_for_idx == cts_anchor:
            return
        meta2 = dict(meta or {})
        # meta2.setdefault("cycle_id", int(self.state.cts_cycle_id))
        meta2["cycle_id"] = int(self.state.cts_cycle_id)
        self.events.append(
            StructureEvent(idx=idx, category="STRUCTURE", type="CTS_CONFIRMED", price=None, meta=meta2)
        )
        st.cts_confirmed_for_idx = cts_anchor
        st.cts_event = "CTS_CONFIRMED"

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
        self.events.append(
            StructureEvent(idx=idx, category="STRUCTURE", type="BOS_CONFIRMED", price=price, meta=meta2)
        )
        self.state.bos_event = "BOS_CONFIRMED"
        self.state.bos_confirmed = Point(idx=idx, price=float(price))
        self.state.bos_threshold = float(price)

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
        Cycle k>1 BOS: pullback extreme between last pullback apply idx and this breakout apply idx.
        Returns (bos_idx, bos_price).
        """
        st = self.state
        pb_start = st.last_pullback_pat_apply_idx

        # If we somehow don't have a pullback anchor, fall back to cycle-1 rule
        if pb_start is None:
            return self._initial_bos_before_first_cts(breakout_apply_idx)

        s = int(pb_start)
        e = int(breakout_apply_idx)
        if e < s:
            s, e = e, s

        window = self.df.iloc[s : e + 1]
        if self.struct_direction == 1:
            series = window["l"].astype(float)
            # bos idx in positional coordinates
            rel = int(series.values.argmin())
            bos_idx = s + rel
            bos_price = float(series.min())
            return bos_idx, bos_price
        else:
            series = window["h"].astype(float)
            rel = int(series.values.argmax())
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
        c = float(self.df.iloc[i]["c"])
        h = float(self.df.iloc[i]["h"])
        l = float(self.df.iloc[i]["l"])

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
            return float(self.df.iloc[i]["l"]) < float(lo)
        return float(self.df.iloc[i]["h"]) > float(hi)

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


        # last breakout pat idx
        if "last_breakout_pat_apply_idx" not in out.columns:
            out["last_breakout_pat_apply_idx"] = -1

    def _write_df_row(self, i: int) -> None:
        st = self.state
        row = self.df.index[i]
        prev_row = self.df.index[i - 1] if i > 0 else row

        self.df.at[row, "market_state"] = st.state.value
        self.df.at[row, "range_active"] = int(st.range_active)
        self.df.at[row, "range_hi"] = float(st.range_hi) if st.range_hi is not None else float("nan")
        self.df.at[row, "range_lo"] = float(st.range_lo) if st.range_lo is not None else float("nan")
        self.df.at[row, "range_start_idx"] = int(st.range_start_idx) if st.range_start_idx is not None else -1
        self.df.at[row, "range_confirm_idx"] = int(st.range_confirm_idx) if st.range_confirm_idx is not None else -1

        if st.range_active and st.range_hi is not None and st.range_lo is not None:
            self.df.at[row, "breakout_th"] = self._range_breakout_threshold()
            self.df.at[row, "pullback_th"] = self._range_pullback_threshold()
        else:
            self.df.at[row, "breakout_th"] = float("nan")
            self.df.at[row, "pullback_th"] = float("nan")

        self.df.at[row, "cts_idx"] = int(st.cts.idx) if st.cts is not None else -1
        self.df.at[row, "cts_price"] = float(st.cts.price) if st.cts is not None else float("nan")
        self.df.at[row, "cts_event"] = st.cts_event
        self.df.at[row, "cts_cycle_id"] = int(st.cts_cycle_id)
        self.df.at[row, "cts_threshold"] = float(st.cts_threshold) if st.cts_threshold is not None else float("nan")
        self.df.at[row, "bos_threshold"] = float(st.bos_threshold) if st.bos_threshold is not None else float("nan")

        self.df.at[row, "bos_idx"] = int(st.bos_confirmed.idx) if st.bos_confirmed is not None else -1
        self.df.at[row, "bos_price"] = float(st.bos_confirmed.price) if st.bos_confirmed is not None else float("nan")
        self.df.at[row, "bos_event"] = st.bos_event

        self.df.at[row, "last_breakout_pat_apply_idx"] = (
            int(st.last_breakout_pat_apply_idx) if st.last_breakout_pat_apply_idx is not None else -1
        )

        # Debug: range body-break fraction on this candle (close-breaks only)
        # If candle CLOSE breaks above range_hi / below range_lo, compute the fraction of the
        # candle's real body that lies beyond the breached threshold.
        frac = float("nan")
        if self.df.at[prev_row, "range_hi"] is not None and self.df.at[prev_row, "range_lo"] is not None:
            o = float(self.df.at[row, COL_O])
            c = float(self.df.at[row, COL_C])
            body_low = min(o, c)
            body_high = max(o, c)
            body_len = body_high - body_low

            if body_len > 0:
                # Close-break above / below range
                if c > float(self.df.at[prev_row, "range_hi"]):
                    th = float(self.df.at[prev_row, "range_hi"])
                    above = max(0.0, body_high - max(th, body_low))
                    frac = above / body_len
                elif c < float(self.df.at[prev_row, "range_lo"]):
                    th = float(self.df.at[prev_row, "range_lo"])
                    below = max(0.0, min(th, body_high) - body_low)
                    frac = below / body_len

        self.df.at[row, "range_break_frac"] = frac

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
