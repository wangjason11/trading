from __future__ import annotations

from dataclasses import replace
from typing import Optional, Tuple, List

from engine_v2.common.types import PatternEvent, PatternStatus

class BreakoutPatterns:
    """
    Structure-only candle patterns used for breakout/pullback logic.
    All methods return PatternEvent (or None if not applicable).

    Direction is always +1 / -1.
    Confirmation is represented as the SAME pattern name with status=CONFIRMED.

    NOTE: This class expects candles_v2 feature columns:
      - body_len, candle_len
      - is_big_normal_as0 (big_normal)
      - is_big_normal_as2 (big_normal2)
      - is_big_maru_as0 (big_maru)
      - direction, candle_type, o/h/l/c
    """

    def __init__(self, df):
        self.df = df

    # ---------- helpers ----------
    # def body_check(self, candle, threshold: Optional[float], percent: float = 0.3, direction: int = 1) -> bool:
    #     if threshold is None:
    #         return True
    #     multiplier = 1 + percent if direction == 1 else 1 - percent
    #     return candle.body_len >= threshold * multiplier
    
    def body_check(self, candle, threshold: float, break_percent: float, direction: int) -> bool:
        """
        True iff at least `break_percent` of the candle's REAL BODY lies beyond `threshold`.
        - direction=+1: fraction of body above threshold
        - direction=-1: fraction of body below threshold
        """
        if threshold is None:
            return True

        o = float(candle.o)
        c = float(candle.c)

        body_low = min(o, c)
        body_high = max(o, c)
        body_len = body_high - body_low

        # Avoid division by zero (doji)
        if body_len <= 0:
            return False

        if direction == 1:
            # Body segment above threshold
            above = max(0.0, body_high - max(threshold, body_low))
            frac = above / body_len
            return frac >= break_percent
        else:
            # Body segment below threshold
            below = max(0.0, min(threshold, body_high) - body_low)
            frac = below / body_len
            return frac >= break_percent

    def check_break(self, candle, threshold: Optional[float], direction: int) -> bool:
        if threshold is None:
            return True
        return candle.c > threshold if direction == 1 else candle.c < threshold

    def confirmation_threshold(self, i: int, direction: int) -> float:
        c0 = self.df.iloc[i]
        c1 = self.df.iloc[i + 1]
        return max(c0.h, c1.h) if direction == 1 else min(c0.l, c1.l)

    def _price_confirmation(self, anchor_idx: int, direction: int, threshold: float) -> Tuple[bool, Optional[int]]:
        df = self.df
        for j in range(1, 5):
            k = anchor_idx + j
            if k >= len(df):
                break
            fwd = df.iloc[k]
            if fwd.direction != direction:
                continue
            if fwd.candle_type not in ["normal", "maru"]:
                continue
            if direction == 1 and fwd.c >= threshold:
                return True, k
            if direction == -1 and fwd.c <= threshold:
                return True, k
        return False, None

    def _confirm_if_needed(self, event: PatternEvent) -> PatternEvent:
        if event.status != PatternStatus.FAIL_NEEDS_CONFIRM or event.confirmation_threshold is None:
            return event

        ok, conf_idx = self._price_confirmation(
            anchor_idx=event.end_idx,
            direction=event.direction,
            threshold=event.confirmation_threshold,
        )
        if ok and conf_idx is not None:
            return replace(
                event,
                status=PatternStatus.CONFIRMED,
                confirmation_idx=conf_idx,
            )
        return event
    
    def _price_confirmation_1step(self, anchor_end_idx: int, direction: int, threshold: float) -> Tuple[bool, Optional[int]]:
        """
        Continuous-only confirmation:
        - Look ahead exactly 1 candle (k = anchor_end_idx + 1)
        - Candle must be normal/maru
        - Candle direction must match `direction`
        - Close must break beyond `threshold` in the direction
        """
        k = int(anchor_end_idx) + 1
        if k >= len(self.df):
            return False, None

        fwd = self.df.iloc[k]
        if int(fwd.direction) != int(direction):
            return False, None
        if str(fwd.candle_type) not in ["normal", "maru"]:
            return False, None

        c = float(fwd.c)
        if direction == 1 and c >= float(threshold):
            return True, k
        if direction == -1 and c <= float(threshold):
            return True, k
        return False, None


    # ---------- patterns ----------
    def continuous(
        self,
        idx: int,
        direction: int,
        break_threshold: Optional[float] = None,
        do_confirm: bool = True,
    ) -> Optional[PatternEvent]:
        if idx + 2 >= len(self.df):
            return None

        df = self.df
        c0, c1, c2 = df.iloc[idx], df.iloc[idx + 1], df.iloc[idx + 2]

        if not all(int(c.direction) == int(direction) for c in [c0, c1, c2]):
            return None

        # Keep: only require the first candle to respect break_threshold
        if not self.check_break(c0, break_threshold, direction):
            return None

        # --- your "3 valid pattern shapes" (keep as the gate) ---
        valid_pattern1 = (
            c0.candle_type in ["normal", "maru"]
            and c1.candle_type == "pinbar"
            and c2.candle_type == "maru"
        )

        valid_pattern2 = (
            c0.candle_type == "pinbar"
            and c2.candle_type == "maru"
        )

        valid_pattern3 = (
            c0.candle_type == "normal"
            and c1.candle_type == "normal"
            and c2.candle_type == "maru"
        )

        if not (valid_pattern1 or valid_pattern2 or valid_pattern3):
            return None

        # --- compute the 3-candle extreme reached by the pattern ---
        extreme = max(float(c0.h), float(c1.h), float(c2.h)) if direction == 1 else min(float(c0.l), float(c1.l), float(c2.l))

        # ------------------------------------------------------------
        # Variant 1 subconditions (3 conditions)
        # ------------------------------------------------------------
        v1_c0 = (c0.candle_type in ["normal", "maru"]) and (int(getattr(c0, "is_big_normal_as0", 0)) == 1)
        v1_c1 = (c1.candle_type == "pinbar")
        v1_c2 = (
            (c2.candle_type == "maru")
            and (int(getattr(c2, "is_big_normal_as2", 0)) == 1)
            and (float(c2.c) > max(float(c0.h), float(c1.h)) if direction == 1 else float(c2.c) < min(float(c0.l), float(c1.l)))
        )

        v1_sum = int(bool(v1_c0)) + int(bool(v1_c1)) + int(bool(v1_c2))
        v1_success = (v1_sum == 3)
        v1_need_confirm = (v1_sum == 2)

        # ------------------------------------------------------------
        # Variant 2 subconditions (3 conditions)
        # ------------------------------------------------------------
        close_to_ext = (
            abs(float(c1.c) - float(c0.h)) <= 0.00015
            if direction == 1
            else abs(float(c1.c) - float(c0.l)) <= 0.00015
        )
        v2_c0 = (c0.candle_type == "pinbar") and (int(getattr(c0, "is_big_normal_as0", 0)) == 1)
        v2_c1 = bool(close_to_ext)
        v2_c2 = (
            (c2.candle_type == "maru")
            and (int(getattr(c2, "is_big_normal_as2", 0)) == 1)
            and (float(c2.c) > max(float(c0.h), float(c1.h)) if direction == 1 else float(c2.c) < min(float(c0.l), float(c1.l)))
        )

        v2_sum = int(bool(v2_c0)) + int(bool(v2_c1)) + int(bool(v2_c2))
        v2_success = (v2_sum == 3)
        v2_need_confirm = (v2_sum == 2)

        # ------------------------------------------------------------
        # Variant 3 subconditions (3 conditions)
        # NOTE: you reference is_big_normal_as1 in your file; keep it.
        # ------------------------------------------------------------
        v3_c0 = (c0.candle_type == "normal") and (int(getattr(c0, "is_big_normal_as0", 0)) == 1)
        v3_c1 = (c1.candle_type == "normal") and (int(getattr(c1, "is_big_normal_as1", 0)) == 1)
        v3_c2 = (
            (c2.candle_type == "maru")
            and (int(getattr(c2, "is_big_normal_as2", 0)) == 1)
            and (float(c2.c) > max(float(c0.h), float(c1.h)) if direction == 1 else float(c2.c) < min(float(c0.l), float(c1.l)))
        )

        v3_sum = int(bool(v3_c0)) + int(bool(v3_c1)) + int(bool(v3_c2))
        v3_success = (v3_sum == 3)
        v3_need_confirm = (v3_sum == 2)

        # --- Choose which variant we are firing (stable priority) ---
        # If multiple could match (rare), we keep your existing implied ordering.
        variant = None
        if v1_success or v1_need_confirm:
            variant = "pattern1"
            success = v1_success
            need_confirm = v1_need_confirm
            conds = {"c0": v1_c0, "c1": v1_c1, "c2": v1_c2}
        elif v2_success or v2_need_confirm:
            variant = "pattern2"
            success = v2_success
            need_confirm = v2_need_confirm
            conds = {"c0": v2_c0, "c1": v2_c1, "c2": v2_c2}
        elif v3_success or v3_need_confirm:
            variant = "pattern3"
            success = v3_success
            need_confirm = v3_need_confirm
            conds = {"c0": v3_c0, "c1": v3_c1, "c2": v3_c2}
        else:
            return None

        # --- SUCCESS (no confirm needed) ---
        if success:
            return PatternEvent(
                name="continuous",
                direction=direction,
                start_idx=idx,
                end_idx=idx + 2,
                status=PatternStatus.SUCCESS,
                confirmation_threshold=None,
                confirmation_idx=None,
                break_threshold_used=break_threshold,
                debug={
                    "variant": variant,
                    "conds": {k: bool(v) for k, v in conds.items()},
                    "extreme": float(extreme),
                },
            )

        # --- FAIL_NEEDS_CONFIRM (exactly 2/3 conditions true) ---
        if need_confirm:
            ev = PatternEvent(
                name="continuous",
                direction=direction,
                start_idx=idx,
                end_idx=idx + 2,
                status=PatternStatus.FAIL_NEEDS_CONFIRM,
                confirmation_threshold=float(extreme),
                confirmation_idx=None,
                break_threshold_used=break_threshold,
                debug={
                    "variant": variant,
                    "conds": {k: bool(v) for k, v in conds.items()},
                    "extreme": float(extreme),
                },
            )

            if not do_confirm:
                return ev

            ok, conf_idx = self._price_confirmation_1step(
                anchor_end_idx=ev.end_idx,
                direction=ev.direction,
                threshold=float(ev.confirmation_threshold),
            )
            if ok and conf_idx is not None:
                return replace(ev, status=PatternStatus.CONFIRMED, confirmation_idx=int(conf_idx))

            return ev

        return None


    def double_maru(
        self,
        idx: int,
        direction: int,
        break_threshold: Optional[float] = None,
        do_confirm: bool = True,
    ) -> Optional[PatternEvent]:
        if idx + 1 >= len(self.df):
            return None

        df = self.df
        c0 = df.iloc[idx]
        c1 = df.iloc[idx + 1]

        if not (c0.candle_type == "maru" and c1.candle_type == "maru"):
            return None

        if not (c0.direction == direction and c1.direction == direction):
            return None

        if not self.check_break(c0, break_threshold, direction):
            return None

        cond1_valid = int(c0.is_big_normal_as0) == 1
        cond2_valid = (
            (c1.c > c0.c if direction == 1 else c1.c < c0.c)
            and c1.candle_len >= 0.7 * c0.candle_len
        )

        if cond1_valid and cond2_valid:
            return PatternEvent(
                name="double_maru",
                direction=direction,
                start_idx=idx,
                end_idx=idx + 1,
                status=PatternStatus.SUCCESS,
                confirmation_threshold=self.confirmation_threshold(idx, direction),
                break_threshold_used=break_threshold,
            )

        if cond1_valid ^ cond2_valid:
            ev = PatternEvent(
                name="double_maru",
                direction=direction,
                start_idx=idx,
                end_idx=idx + 1,
                status=PatternStatus.FAIL_NEEDS_CONFIRM,
                confirmation_threshold=self.confirmation_threshold(idx, direction),
                break_threshold_used=break_threshold,
                debug={"cond1_valid": cond1_valid, "cond2_valid": cond2_valid},
            )
            return self._confirm_if_needed(ev) if do_confirm else ev

        return None

    def one_maru_continuous(
        self,
        idx: int,
        direction: int,
        break_threshold: Optional[float] = None,
        break_percent: float = 0.3,
        small_body_tail: float = 0.5,
        do_confirm: bool = True,
    ) -> Optional[PatternEvent]:
        if idx + 1 >= len(self.df):
            return None

        df = self.df
        c0 = df.iloc[idx]
        c1 = df.iloc[idx + 1]

        # if idx == 100 and direction == 1:
        #     print(
        #         "[DEBUG PRECHECK idx=100 one_maru_continuous +1]",
        #         "time0=", getattr(c0, "time", None),
        #         "c0_type=", c0.candle_type,
        #         "c0_dir=", c0.direction,
        #         "c0_body_pct=", getattr(c0, "body_pct", None),
        #         "c0_is_big_maru_as0=", int(getattr(c0, "is_big_maru_as0", 0)),
        #         "c0_mid=", getattr(c0, "mid_price", None),
        #         "c1_dir=", c1.direction,
        #         "c1_low=", c1.l,
        #         "check_break=", self.check_break(c0, break_threshold, direction),
        #         "body_check=", self.body_check(c0, break_threshold, break_percent, direction),
        #         "cond2=", (c1.l > getattr(c0, "mid_price", float("nan"))),
        #         "confirm_thr=", self.confirmation_threshold(idx, direction),
        #     )

        if not c0.candle_type == "maru":
            return None

        if not (c0.direction == direction and c1.direction == direction):
            return None

        if not self.check_break(c0, break_threshold, direction):
            return None

        cond1_valid = (
            int(c0.is_big_maru_as0) == 1
            and self.body_check(c0, break_threshold, break_percent, direction)
        )

        if direction == 1:
            # c1_pos_check = c1.l > (c0.l + small_body_tail * c0.candle_len)
            c1_pos_check = c1.l > c0.mid_price
        else:
            # c1_pos_check = c1.h < (c0.l + small_body_tail * c0.candle_len)
            c1_pos_check = c1.h < c0.mid_price
        cond2_valid = c1_pos_check

        if cond1_valid and cond2_valid:
            return PatternEvent(
                name="one_maru_continuous",
                direction=direction,
                start_idx=idx,
                end_idx=idx + 1,
                status=PatternStatus.SUCCESS,
                confirmation_threshold=self.confirmation_threshold(idx, direction),
                break_threshold_used=break_threshold,
            )

        if cond1_valid ^ cond2_valid:
            ev = PatternEvent(
                name="one_maru_continuous",
                direction=direction,
                start_idx=idx,
                end_idx=idx + 1,
                status=PatternStatus.FAIL_NEEDS_CONFIRM,
                confirmation_threshold=self.confirmation_threshold(idx, direction),
                break_threshold_used=break_threshold,
                debug={"cond1_valid": cond1_valid, "cond2_valid": cond2_valid},
            )
            return self._confirm_if_needed(ev) if do_confirm else ev

        return None

    def one_maru_opposite(
        self,
        idx: int,
        direction: int,
        break_threshold: Optional[float] = None,
        break_percent: float = 0.3,
        small_body_tail: float = 0.5,
        small_body_size: float = 0.35,
        do_confirm: bool = True,
    ) -> Optional[PatternEvent]:
        if idx + 1 >= len(self.df):
            return None

        df = self.df
        c0 = df.iloc[idx]
        c1 = df.iloc[idx + 1]

        if not c0.candle_type == "maru":
            return None

        if not (c0.direction == direction and c1.direction != direction):
            return None

        if not self.check_break(c0, break_threshold, direction):
            return None

        # 2 ways for valid pattern:
        # 1) main: c0 big maru & break & c1 small body + c1 low higher than c0 mid price
        # 2) alt: c0 big maru & break & c1 is pinbar & beyond break threshold (only used when there is break threshold)
        # only the main method will be eligible for confirmation
        c1_len_check = c1.candle_len < small_body_size * c0.candle_len
        cond1_valid = (
            int(c0.is_big_maru_as0) == 1
            and c1_len_check
            and self.body_check(c0, break_threshold, break_percent, direction)
        )

        cond1_valid_alt = (
            int(c0.is_big_maru_as0) == 1
            and self.body_check(c0, break_threshold, break_percent, direction)
        )

        if direction == 1:
            c1_pos_check = c1.l > (c0.l + small_body_tail * c0.candle_len)
            c1_tail_check = False if break_threshold is None else c1.l > break_threshold
        else:
            c1_pos_check = c1.h < (c0.l + small_body_tail * c0.candle_len)
            c1_tail_check = False if break_threshold is None else c1.h < break_threshold

        cond2_valid = c1_pos_check
        cond2_valid_alt = c1_tail_check and c1.candle_type == "pinbar"

        if (cond1_valid and cond2_valid) or (cond1_valid_alt and cond2_valid_alt):
            return PatternEvent(
                name="one_maru_opposite",
                direction=direction,
                start_idx=idx,
                end_idx=idx + 1,
                status=PatternStatus.SUCCESS,
                confirmation_threshold=self.confirmation_threshold(idx, direction),
                break_threshold_used=break_threshold,
            )

        if cond1_valid ^ cond2_valid:
            ev = PatternEvent(
                name="one_maru_opposite",
                direction=direction,
                start_idx=idx,
                end_idx=idx + 1,
                status=PatternStatus.FAIL_NEEDS_CONFIRM,
                confirmation_threshold=self.confirmation_threshold(idx, direction),
                break_threshold_used=break_threshold,
                debug={"cond1_valid": cond1_valid, "cond2_valid": cond2_valid},
            )
            return self._confirm_if_needed(ev) if do_confirm else ev

        return None


    def detect_first_success(
        self,
        idx: int,
        direction: int,
        break_threshold: Optional[float] = None,
    ) -> Optional[PatternEvent]:

        # Pass A: compute 2-candle candidates once (NO confirmation)
        dm  = self.double_maru(idx, direction, break_threshold, do_confirm=False)
        omc = self.one_maru_continuous(idx, direction, break_threshold, do_confirm=False)
        omo = self.one_maru_opposite(idx, direction, break_threshold, do_confirm=False)

        # Pass A1: immediate SUCCESS from 2-candle patterns (priority order)
        for ev in (dm, omc, omo):
            if ev is not None and ev.status == PatternStatus.SUCCESS:
                return ev

        # Pass A2: 3-candle pattern (only after no 2-candle SUCCESS)
        cont = self.continuous(idx, direction, break_threshold, do_confirm=False)
        if cont is not None and cont.status == PatternStatus.SUCCESS:
            return cont

        # Pass B: confirmations (only if nothing succeeded)
        confirmed = []

        # NEW: continuous confirm (1-step)
        if cont is not None and cont.status == PatternStatus.FAIL_NEEDS_CONFIRM:
            ok, conf_idx = self._price_confirmation_1step(
                anchor_end_idx=cont.end_idx,
                direction=cont.direction,
                threshold=float(cont.confirmation_threshold),
            )
            if ok and conf_idx is not None:
                confirmed.append(replace(cont, status=PatternStatus.CONFIRMED, confirmation_idx=int(conf_idx)))

        # existing 2-candle confirmations...
        for ev in (dm, omc, omo):
            if ev is None:
                continue
            if ev.status != PatternStatus.FAIL_NEEDS_CONFIRM:
                continue

            ev2 = self._confirm_if_needed(ev)
            if ev2.status == PatternStatus.CONFIRMED and ev2.confirmation_idx is not None:
                confirmed.append(ev2)

        if confirmed:
            confirmed.sort(key=lambda e: e.confirmation_idx)
            return confirmed[0]

        return None


    def detect_best_for_anchor(
        self,
        idx: int,
        direction: int,
        break_threshold: Optional[float] = None,
    ) -> Optional[PatternEvent]:
        """
        New Week 5 behavior:
          Priority:
            1) continuous (SUCCESS only)
            2) one_maru_opposite (SUCCESS)
            3) one_maru_continuous (SUCCESS)
            4) double_maru (SUCCESS)
            5) confirmations (for 2-candle and 3-candle patterns), in the same priority order
        Notes:
          - Confirmation lookahead max is 4 candles AFTER end_idx, so latest confirmation is idx+5.
          - continuous has precedence even though it needs idx+2.
        """

        # 1) Highest priority: continuous
        cont = self.continuous(idx, direction, break_threshold, do_confirm=False)
        if cont is not None and cont.status == PatternStatus.SUCCESS:
            return cont

        # 2) Compute 2-candle candidates without confirmation
        omo = self.one_maru_opposite(idx, direction, break_threshold, do_confirm=False)
        omc = self.one_maru_continuous(idx, direction, break_threshold, do_confirm=False)
        dm  = self.double_maru(idx, direction, break_threshold, do_confirm=False)

        # 3) Immediate SUCCESS (priority order)
        for ev in (omo, omc, dm):
            if ev is not None and ev.status == PatternStatus.SUCCESS:
                return ev

        # 4) Confirmations LAST (priority order)
        confirmed: List[PatternEvent] = []

        # NEW: continuous confirmation (priority first in confirm stage)
        if cont is not None and cont.status == PatternStatus.FAIL_NEEDS_CONFIRM and cont.confirmation_threshold is not None:
            ok, conf_idx = self._price_confirmation_1step(
                anchor_end_idx=cont.end_idx,
                direction=cont.direction,
                threshold=float(cont.confirmation_threshold),
            )
            if ok and conf_idx is not None:
                confirmed.append(replace(cont, status=PatternStatus.CONFIRMED, confirmation_idx=int(conf_idx)))

        # existing 2-candle confirmations...
        for ev in (omo, omc, dm):
            if ev is None:
                continue
            if ev.status != PatternStatus.FAIL_NEEDS_CONFIRM:
                continue
            ev2 = self._confirm_if_needed(ev)
            if ev2.status == PatternStatus.CONFIRMED and ev2.confirmation_idx is not None:
                confirmed.append(ev2)

        if not confirmed:
            return None

        # If multiple confirmations exist, pick the earliest confirmation candle;
        # if tie, preserve priority by ordering in the same (omo, omc, dm) sequence.
        # We'll implement stable selection by sorting on confirmation_idx only
        # because list order already encodes priority.
        confirmed.sort(key=lambda e: e.confirmation_idx)
        return confirmed[0]
