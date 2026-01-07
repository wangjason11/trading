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
    def body_check(self, candle, threshold: Optional[float], percent: float = 0.5, direction: int = 1) -> bool:
        if threshold is None:
            return True
        multiplier = 1 + percent if direction == 1 else 1 - percent
        return candle.body_len >= threshold * multiplier

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

    # ---------- patterns ----------
    def continuous(self, idx: int, direction: int, break_threshold: Optional[float] = None) -> Optional[PatternEvent]:
        if idx + 2 >= len(self.df):
            return None

        df = self.df
        c0, c1, c2 = df.iloc[idx], df.iloc[idx + 1], df.iloc[idx + 2]

        if not all(c.direction == direction for c in [c0, c1, c2]):
            return None

        if not self.check_break(c0, break_threshold, direction):
            return None

        cond1_valid = (
            c0.candle_type in ["normal", "maru"]
            and c1.candle_type == "pinbar"
            and c2.candle_type == "maru"
            and int(c0.is_big_normal_as0) == 1
            and int(c2.is_big_normal_as2) == 1
            and (c2.c > max(c0.h, c1.h) if direction == 1 else c2.c < min(c0.l, c1.l))
        )

        close_to_high = abs(c1.c - c0.h) <= 0.00010 if direction == 1 else abs(c1.c - c0.l) <= 0.00010
        cond2_valid = (
            c0.candle_type == "pinbar"
            and close_to_high
            and c2.candle_type == "maru"
            and int(c0.is_big_normal_as0) == 1
            and int(c2.is_big_normal_as2) == 1
            and (c2.c > max(c0.h, c1.h) if direction == 1 else c2.c < min(c0.l, c1.l))
        )

        cond3_valid = (
            c0.candle_type == "normal"
            and c1.candle_type == "normal"
            and c2.candle_type == "maru"
            and int(c0.is_big_normal_as0) == 1
            and int(c1.is_big_normal_as1) == 1
            and int(c2.is_big_normal_as2) == 1
            and (c2.c > max(c0.h, c1.h) if direction == 1 else c2.c < min(c0.l, c1.l))
        )

        if not (cond1_valid or cond2_valid or cond3_valid):
            return None

        return PatternEvent(
            name="continuous",
            direction=direction,
            start_idx=idx,
            end_idx=idx + 2,
            status=PatternStatus.SUCCESS,
            confirmation_threshold=None,
            confirmation_idx=None,
            break_threshold_used=break_threshold,
            debug={"variant": "cond1_valid" if cond1_valid else "cond2_valid" if cond2_valid else "cond3_valid"},
        )

    def double_maru(self, idx: int, direction: int, break_threshold: Optional[float] = None) -> Optional[PatternEvent]:
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
            return self._confirm_if_needed(ev)

        return None

    def one_maru_continuous(
        self,
        idx: int,
        direction: int,
        break_threshold: Optional[float] = None,
        break_percent: float = 0.3,
        small_body_tail: float = 0.5,
    ) -> Optional[PatternEvent]:
        if idx + 1 >= len(self.df):
            return None

        df = self.df
        c0 = df.iloc[idx]
        c1 = df.iloc[idx + 1]

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
            return self._confirm_if_needed(ev)

        return None

    def one_maru_opposite(
        self,
        idx: int,
        direction: int,
        break_threshold: Optional[float] = None,
        break_percent: float = 0.3,
        small_body_tail: float = 0.5,
        small_body_size: float = 0.35,
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

        c1_len_check = c1.candle_len < small_body_size * c0.candle_len
        cond1_valid = (
            int(c0.is_big_maru_as0) == 1
            and c1_len_check
            and self.body_check(c0, break_threshold, break_percent, direction)
        )

        if direction == 1:
            c1_pos_check = c1.l > (c0.l + small_body_tail * c0.candle_len)
        else:
            c1_pos_check = c1.h < (c0.l + small_body_tail * c0.candle_len)
        cond2_valid = c1_pos_check

        if cond1_valid and cond2_valid:
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
            return self._confirm_if_needed(ev)

        return None

    def detect_first_success(
        self,
        idx: int,
        direction: int,
        break_threshold: Optional[float] = None,
    ) -> Optional[PatternEvent]:
        candidates: List[Optional[PatternEvent]] = [
            self.continuous(idx, direction, break_threshold),
            self.double_maru(idx, direction, break_threshold),
            self.one_maru_continuous(idx, direction, break_threshold),
            self.one_maru_opposite(idx, direction, break_threshold),
        ]
        for ev in candidates:
            if ev is None:
                continue
            if ev.status in (PatternStatus.SUCCESS, PatternStatus.CONFIRMED):
                return ev
        return None
