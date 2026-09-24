"""Isolation unit tests for the shared cycle-0 true-first-breakout routine.

`find_true_first_breakout` is the ONE routine both the unified probe and
MarketStructure's pre-CTS_0 scan call (DESIGN LOCKED 2026-06-07 +
scan-from-start pivot; see `memory/project_true_first_breakout_cycle0.md`).
A bug here corrupts everything downstream, so these tests are the
highest-priority isolation check (Verification process #4).

The detectors (continuous / double_maru / one_maru_*) are tested in
`test_structure_patterns*`; here we drive the routine with a FakeBP that
returns canned PatternEvents, so we isolate THIS routine's contributions:
  - condition 3: strict full-pattern new extreme over `[current_start, extreme_candle)`
  - condition 4: earliest apply/confirm idx, cycle-0 tie-break continuous>dm>omc>omo
  - est_idx == MS `_apply_idx` (confirmation_idx for CONFIRMED, else end_idx)
  - full-pattern extreme includes the confirm candle
  - mechanism-B: bos0_inner threshold is passed through to the detectors
  - SUCCESS/CONFIRMED only (FAIL_NEEDS_CONFIRM dropped)
  - confirm landing past the window (est_idx > upper_idx) is rejected
  - no qualifying candidate -> None
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytest

from engine_v2.common.types import PatternEvent, PatternStatus
from engine_v2.structure.true_first_breakout import (
    TrueFirstBreakout,
    find_true_first_breakout,
)


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------

_PAT_NAMES = ("continuous", "double_maru", "one_maru_continuous", "one_maru_opposite")


def _df_from_highs_lows(highs: List[float], lows: Optional[List[float]] = None) -> pd.DataFrame:
    """Build a 0-based RangeIndexed OHLC df. Only h/l matter for the
    routine's extreme computation; o/c are filled inertly."""
    n = len(highs)
    if lows is None:
        lows = [h - 1.0 for h in highs]
    return pd.DataFrame(
        {
            "o": [float(h) for h in highs],
            "h": [float(h) for h in highs],
            "l": [float(x) for x in lows],
            "c": [float(h) for h in highs],
        }
    )


def _pat(
    name: str,
    *,
    start_idx: int,
    end_idx: int,
    status: PatternStatus = PatternStatus.SUCCESS,
    confirmation_idx: Optional[int] = None,
    direction: int = 1,
) -> PatternEvent:
    return PatternEvent(
        name=name,
        direction=direction,
        start_idx=start_idx,
        end_idx=end_idx,
        status=status,
        confirmation_idx=confirmation_idx,
    )


class FakeBP:
    """Stand-in for BreakoutPatterns. Returns canned PatternEvents from a
    `(idx, pattern_name) -> PatternEvent` table and records every detector
    call (to assert the mechanism-B threshold is threaded)."""

    def __init__(self, df: pd.DataFrame, table: Dict[Tuple[int, str], PatternEvent]):
        self.df = df
        self.table = table
        self.calls: List[Tuple[str, int, int, Optional[float], bool]] = []

    def _lookup(self, name, idx, direction, break_threshold, do_confirm):
        self.calls.append((name, idx, direction, break_threshold, do_confirm))
        return self.table.get((idx, name))

    def continuous(self, idx, direction, break_threshold=None, do_confirm=True):
        return self._lookup("continuous", idx, direction, break_threshold, do_confirm)

    def double_maru(self, idx, direction, break_threshold=None, do_confirm=True):
        return self._lookup("double_maru", idx, direction, break_threshold, do_confirm)

    def one_maru_continuous(self, idx, direction, break_threshold=None, do_confirm=True):
        return self._lookup("one_maru_continuous", idx, direction, break_threshold, do_confirm)

    def one_maru_opposite(self, idx, direction, break_threshold=None, do_confirm=True):
        return self._lookup("one_maru_opposite", idx, direction, break_threshold, do_confirm)


# ---------------------------------------------------------------------------
# Condition 3 — strict full-pattern new extreme over [current_start, extreme_candle)
# ---------------------------------------------------------------------------

def test_strict_new_extreme_accepts_strictly_greater():
    # prior highs (idx 0..2): max = 13. Pattern at anchor 3, span [3,4],
    # extreme high 14 at idx4 > 13 strict -> accepted.
    df = _df_from_highs_lows([10, 13, 11, 12, 14])
    bp = FakeBP(df, {(3, "double_maru"): _pat("double_maru", start_idx=3, end_idx=4)})
    res = find_true_first_breakout(bp, current_start=0, upper_idx=4, direction=1, bos0_inner=0.0)
    assert res is not None
    assert res.est_idx == 4
    assert res.extreme_idx == 4
    assert res.extreme_price == 14.0
    assert res.pattern_anchor_idx == 3


def test_strict_new_extreme_rejects_a_tie():
    # Same as above but pattern extreme 13 ties the prior max 13 -> rejected
    # (ties do NOT count). No other candidate -> None.
    df = _df_from_highs_lows([10, 13, 11, 12, 13])
    bp = FakeBP(df, {(3, "double_maru"): _pat("double_maru", start_idx=3, end_idx=4)})
    res = find_true_first_breakout(bp, current_start=0, upper_idx=4, direction=1, bos0_inner=0.0)
    assert res is None


def test_strict_new_extreme_window_is_half_open_before_extreme_candle():
    # The extreme candle itself is excluded from the comparison window:
    # extreme at idx4=14; window [0,4) = highs[10,13,11,12]. The pattern's
    # own earlier candle (idx3=12) is INSIDE the window but below 14, so it
    # doesn't block. Confirms [current_start, extreme_idx) semantics.
    df = _df_from_highs_lows([10, 13, 11, 12, 14])
    bp = FakeBP(df, {(3, "continuous"): _pat("continuous", start_idx=3, end_idx=4)})
    res = find_true_first_breakout(bp, current_start=0, upper_idx=4, direction=1, bos0_inner=0.0)
    assert res is not None and res.extreme_idx == 4


def test_strict_new_extreme_downtrend_min():
    # direction=-1: strict new LOW. prior lows min = 0.40; pattern low 0.39
    # < 0.40 strict -> accepted.
    df = _df_from_highs_lows(
        highs=[1, 1, 1, 1, 1],
        lows=[0.50, 0.40, 0.45, 0.44, 0.39],
    )
    bp = FakeBP(df, {(3, "double_maru"): _pat("double_maru", start_idx=3, end_idx=4, direction=-1)})
    res = find_true_first_breakout(bp, current_start=0, upper_idx=4, direction=-1, bos0_inner=2.0)
    assert res is not None
    assert res.extreme_idx == 4
    assert res.extreme_price == pytest.approx(0.39)


def test_empty_prior_window_trivially_passes():
    # current_start == anchor and extreme at the very first candle of the
    # span -> window [current_start, extreme_idx) empty -> passes.
    df = _df_from_highs_lows([10, 9, 9])
    bp = FakeBP(df, {(0, "double_maru"): _pat("double_maru", start_idx=0, end_idx=1)})
    res = find_true_first_breakout(bp, current_start=0, upper_idx=2, direction=1, bos0_inner=0.0)
    assert res is not None and res.extreme_idx == 0


# ---------------------------------------------------------------------------
# Condition 4 — earliest apply/confirm idx + cycle-0 tie-break
# ---------------------------------------------------------------------------

def test_earliest_est_wins_across_anchors_not_first_anchor():
    # anchor 0 has a valid candidate with est=4; anchor 1 has est=2. The
    # GLOBAL earliest est (2, anchor 1) must win — not the first anchor.
    df = _df_from_highs_lows([10, 20, 11, 12, 21])
    table = {
        (0, "double_maru"): _pat("double_maru", start_idx=0, end_idx=4),  # est=4, ext idx4=21
        (1, "double_maru"): _pat("double_maru", start_idx=1, end_idx=2),  # est=2, ext idx1=20
    }
    bp = FakeBP(df, table)
    res = find_true_first_breakout(bp, current_start=0, upper_idx=4, direction=1, bos0_inner=0.0)
    assert res is not None
    assert res.est_idx == 2
    assert res.pattern_anchor_idx == 1


def test_tie_break_prefers_continuous_over_double_maru_at_equal_est():
    # Two candidates at the same est_idx=2: continuous (prio0) beats
    # double_maru (prio1) regardless of which anchor is scanned first.
    df = _df_from_highs_lows([10, 20, 21])
    table = {
        (0, "double_maru"): _pat("double_maru", start_idx=0, end_idx=2),
        (0, "continuous"): _pat("continuous", start_idx=0, end_idx=2),
    }
    bp = FakeBP(df, table)
    res = find_true_first_breakout(bp, current_start=0, upper_idx=2, direction=1, bos0_inner=0.0)
    assert res is not None
    assert res.pattern.name == "continuous"


def test_tie_break_full_priority_order():
    # All four fire at the same anchor/est. Order must be
    # continuous > double_maru > one_maru_continuous > one_maru_opposite.
    df = _df_from_highs_lows([10, 20, 21])
    base = dict(start_idx=0, end_idx=2)
    # Drop names one at a time; the survivor's winner must follow the order.
    order = ["continuous", "double_maru", "one_maru_continuous", "one_maru_opposite"]
    for drop_count in range(len(order)):
        present = order[drop_count:]
        table = {(0, nm): _pat(nm, **base) for nm in present}
        bp = FakeBP(df, table)
        res = find_true_first_breakout(bp, current_start=0, upper_idx=2, direction=1, bos0_inner=0.0)
        assert res is not None
        assert res.pattern.name == present[0]


# ---------------------------------------------------------------------------
# est_idx semantics (== MS _apply_idx) + confirmation extends the extreme
# ---------------------------------------------------------------------------

def test_confirmed_pattern_est_idx_is_confirmation_idx_and_extreme_includes_confirm_candle():
    # CONFIRMED: span extends to confirmation_idx=5. Highs rise so the
    # extreme lands ON the confirm candle (idx5). est_idx must be 5.
    df = _df_from_highs_lows([10, 11, 12, 13, 14, 20, 9])
    bp = FakeBP(
        df,
        {
            (2, "double_maru"): _pat(
                "double_maru", start_idx=2, end_idx=3,
                status=PatternStatus.CONFIRMED, confirmation_idx=5,
            )
        },
    )
    res = find_true_first_breakout(bp, current_start=0, upper_idx=6, direction=1, bos0_inner=0.0)
    assert res is not None
    assert res.est_idx == 5            # confirmation_idx, not end_idx
    assert res.extreme_idx == 5        # confirm candle included in the span
    assert res.extreme_price == 20.0


def test_success_pattern_est_idx_is_end_idx():
    df = _df_from_highs_lows([10, 11, 20])
    bp = FakeBP(df, {(0, "continuous"): _pat("continuous", start_idx=0, end_idx=2)})
    res = find_true_first_breakout(bp, current_start=0, upper_idx=2, direction=1, bos0_inner=0.0)
    assert res is not None and res.est_idx == 2


# ---------------------------------------------------------------------------
# Mechanism B — bos0_inner threaded to detectors
# ---------------------------------------------------------------------------

def test_bos0_inner_passed_as_break_threshold():
    df = _df_from_highs_lows([10, 11, 20])
    bp = FakeBP(df, {})
    find_true_first_breakout(bp, current_start=0, upper_idx=2, direction=1, bos0_inner=0.5)
    assert bp.calls, "detectors should have been called"
    # Every detector call must receive bos0_inner as break_threshold and do_confirm=True.
    assert all(c[3] == 0.5 and c[4] is True for c in bp.calls)
    # All four pattern detectors get consulted per anchor.
    assert {c[0] for c in bp.calls} == set(_PAT_NAMES)


# ---------------------------------------------------------------------------
# Rejection / empty paths
# ---------------------------------------------------------------------------

def test_no_candidates_returns_none():
    df = _df_from_highs_lows([10, 11, 12])
    bp = FakeBP(df, {})
    assert find_true_first_breakout(bp, 0, 2, 1, 0.0) is None


def test_fail_needs_confirm_is_dropped():
    df = _df_from_highs_lows([10, 11, 20])
    bp = FakeBP(
        df,
        {(0, "double_maru"): _pat(
            "double_maru", start_idx=0, end_idx=1, status=PatternStatus.FAIL_NEEDS_CONFIRM,
        )},
    )
    assert find_true_first_breakout(bp, 0, 2, 1, 0.0) is None


def test_confirmation_past_window_rejected():
    # est_idx (confirmation_idx=12) > upper_idx=10 -> rejected (edge-pending,
    # dormant in backtest).
    df = _df_from_highs_lows([float(x) for x in range(20)])
    bp = FakeBP(
        df,
        {(2, "double_maru"): _pat(
            "double_maru", start_idx=2, end_idx=3,
            status=PatternStatus.CONFIRMED, confirmation_idx=12,
        )},
    )
    assert find_true_first_breakout(bp, current_start=0, upper_idx=10, direction=1, bos0_inner=0.0) is None


def test_upper_idx_clamped_to_df_length():
    df = _df_from_highs_lows([10, 11, 20])
    bp = FakeBP(df, {(0, "continuous"): _pat("continuous", start_idx=0, end_idx=2)})
    # upper_idx beyond the df should not raise; the routine clamps to n-1.
    res = find_true_first_breakout(bp, current_start=0, upper_idx=999, direction=1, bos0_inner=0.0)
    assert res is not None and res.est_idx == 2


def test_lo_gt_hi_returns_none():
    df = _df_from_highs_lows([10, 11, 12])
    bp = FakeBP(df, {})
    assert find_true_first_breakout(bp, current_start=5, upper_idx=2, direction=1, bos0_inner=0.0) is None


# ---------------------------------------------------------------------------
# Early-stop correctness (must still find the global earliest est)
# ---------------------------------------------------------------------------

def test_early_stop_does_not_skip_a_later_anchor_with_smaller_est():
    # anchor 1 has a LARGE est (confirmation at 8); anchor 3 succeeds at
    # est=4. Earliest est=4 must win, and the scan must not stop before
    # reaching anchor 3.
    df = _df_from_highs_lows([10, 11, 12, 13, 30, 14, 15, 16, 25])
    table = {
        (1, "double_maru"): _pat(
            "double_maru", start_idx=1, end_idx=2,
            status=PatternStatus.CONFIRMED, confirmation_idx=8,  # est=8, ext idx8=25
        ),
        (3, "double_maru"): _pat("double_maru", start_idx=3, end_idx=4),  # est=4, ext idx4=30
    }
    bp = FakeBP(df, table)
    res = find_true_first_breakout(bp, current_start=0, upper_idx=8, direction=1, bos0_inner=0.0)
    assert res is not None
    assert res.est_idx == 4
    assert res.pattern_anchor_idx == 3
