"""A reversal applied inside a back-fill is terminal (2026-09-28; zones-audit "double reversal").

MARKET_STRUCTURE_SPEC "Reversal inside a back-fill": the pending reversal is applied by the per-candle step
(`_replay_step_no_patterns`), which also runs inside every FROZEN back-fill — `_step_anchor`'s winner back-fill, its
no-winner range back-fill, and `_post_apply_range_check`'s. Before the fix the step went on after it: the winner
applied at its later candle, the range finalized, the anchor was re-stepped — so the dead structure LEFT REVERSAL
and could reverse again. Two `STATE_CHANGED(to=reversal)` for one sid: the hand-off took the FIRST (`reversal_idx`,
the df mask's min) and every reader the LAST (`compute_reversal_idx_by_sid`, MAX). Fix (user decision, option A):
every back-fill ends the step at the terminal candle; `_set_state` asserts nothing leaves REVERSAL; a `_rewind_to`
rebuild asserts it never reaches one. (The df-level "later states must be reversal" check was deleted: `run()`
forward-stamps `market_state` before it, so it could never fire.)

Reference window (a shadow over every MS run — H1 main sids, both first-confluence probes' Phase 2, every sub build,
`review_scripts/reversal_shadow.py`): 12 runs, 5 reversals, 0 applied inside a back-fill, 0 leaves — byte-identical.
The fixtures: the winner back-fill (P1) from a random-tail search on the verified cycle-1 base; the range back-fill
(P2) and the reversal-winner back-fill are existing `test_ms_bounded_equals_truncated` fixtures at sd=-1, found by
the shadow over the suite.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from engine_v2.patterns.structure_patterns import BreakoutPatterns
from engine_v2.structure.market_structure import MarketState
from engine_v2.structure.structure_engine import _make_market_structure, compute_bounded_structure
from engine_v2.tests import test_ms_bounded_equals_truncated as _bet
from engine_v2.tests import test_ms_cts_update_no_regress as _noreg
from engine_v2.tests.test_unified_probe import (
    _R,
    _make_cycle0_reversal_data,
    _make_second_cts_moment_after_anchor_data,
    _prepare_df,
)
from engine_v2.zones.structure_lifecycle import compute_reversal_idx_by_sid


# ---------------------------------------------------------------------------------------------------------------
# P1 — the winner back-fill. sd=+1, start 0, NZD_USD H1.
#   0-10 = `_make_second_cts_moment_after_anchor_data()[:11]`: cycle 1 established at 10 (CTS anchor 9 .6147),
#        BOS_1 = l7 .6088, state BREAKOUT, no range.
#   11   small bear pinbar.
#   12   bear pinbar (long lower wick .60923) = c0 of the anchor-12 winner: `continuous(-1)` pullback (12 pinbar,
#        13-14 marus), FAIL_NEEDS_CONFIRM, confirmed at 17 (15 / 16 pinbars skipped) -> apply 17 = D. The back-fill
#        [12, 17) runs: 12 sd-zone proximity confirms the CTS and creates a range (RANGE);
#   14   bear maru closing .60391 < BOS .6088 -> REVERSAL_WATCH_START + the pending reversal
#        `one_maru_continuous(-1)` 14-15 (SUCCESS, apply 15);
#   15   the pending reversal applies -> STATE_CHANGED(range -> reversal) @15 = the terminal.
#   Before the fix: 16 back-filled in REVERSAL; the pullback applied at 17 (CTS_RECONFIRMED@17,
#   STATE_CHANGED(reversal -> pullback)@17); 17 close-broke again -> watch; 18-19 two bear marus -> a SECOND
#   reversal @19. Hand-off `reversal_idx` 15, `compute_reversal_idx_by_sid` 19.
#   20-21 bull fillers.
_P1_BOS, _P1_REVERSAL, _P1_WINNER_APPLY, _P1_SECOND = 0.6088, 15, 17, 19


def _p1_rows() -> list[dict]:
    rows = list(_make_second_cts_moment_after_anchor_data()[:11])
    rows += [
        _R(0.61380, 0.61382, 0.61358, 0.61364),   # 11
        _R(0.61364, 0.61374, 0.60923, 0.61328),   # 12
        _R(0.61328, 0.61340, 0.60945, 0.60951),   # 13
        _R(0.60951, 0.60955, 0.60372, 0.60391),   # 14
        _R(0.60391, 0.60411, 0.60340, 0.60365),   # 15
        _R(0.60365, 0.60386, 0.60363, 0.60364),   # 16
        _R(0.60364, 0.60382, 0.60315, 0.60333),   # 17
        _R(0.60333, 0.60340, 0.60050, 0.60060),   # 18
        _R(0.60060, 0.60065, 0.59700, 0.59710),   # 19
        _R(0.59710, 0.59760, 0.59700, 0.59750),   # 20
        _R(0.59750, 0.59800, 0.59740, 0.59790),   # 21
    ]
    return rows


def _p1_df(sd: int):
    """The P1 frame (sd=-1: the price mirror, candle shapes and types preserved), with the path's preconditions
    asserted on the detector: the anchor-12 pullback applies at 17, the pending reversal at 15 < 17."""
    rows = _p1_rows() if sd == 1 else _noreg._mirror(_p1_rows())
    df = _prepare_df(rows)
    bp = BreakoutPatterns(df)
    winner = bp.detect_best_for_anchor(12, -sd, None)
    assert winner is not None and winner.name == "continuous" and winner.confirmation_idx == _P1_WINNER_APPLY
    bos = _P1_BOS if sd == 1 else round(_noreg._MIRROR - _P1_BOS, 5)
    rev = bp.detect_best_for_anchor(14, -sd, bos)
    assert rev is not None and rev.end_idx == _P1_REVERSAL and rev.confirmation_idx is None
    return df


def _run(df, sd: int, **kw):
    with redirect_stdout(io.StringIO()):
        return compute_bounded_structure(df, 0, sd, **kw)


def _reversals(events):
    return [e for e in events if e.type == "STATE_CHANGED" and e.meta["to"] == "reversal"]


def _last_written_row(res) -> int:
    return int(res.df.index[res.df["structure_id"].astype(int) == 0].max())


def _assert_single_terminal(res, idx: int, from_state: str) -> None:
    """One reversal at `idx`, nothing emitted after it, nothing leaves it, hand-off == readers, no row past it."""
    rev = _reversals(res.events)
    assert [(e.idx, e.meta["from"]) for e in rev] == [(idx, from_state)]
    assert res.events[-1] is rev[0]                                   # emission order: the terminal is last
    assert max(int(e.idx) for e in res.events) == idx
    assert not [e for e in res.events if e.type == "STATE_CHANGED" and e.meta["from"] == "reversal"]
    assert res.reversal_idx == idx and compute_reversal_idx_by_sid(res.events) == {0: idx}
    assert _last_written_row(res) == idx


@pytest.mark.parametrize("sd", [1, -1])
def test_a_reversal_inside_a_winner_backfill_ends_the_structure(sd):
    res = _run(_p1_df(sd), sd)
    _assert_single_terminal(res, _P1_REVERSAL, "range")
    assert _reversals(res.events)[0].meta["pat"] == "one_maru_continuous"
    # the winner (the pullback at 17) never applies, and nothing reverses again
    assert not [e for e in res.events if e.type in ("CTS_RECONFIRMED", "CTS_CONFIRMED") and e.idx == _P1_WINNER_APPLY]
    assert _P1_SECOND not in {int(e.idx) for e in res.events}


# ---------------------------------------------------------------------------------------------------------------
# P2 — the no-winner range back-fill. `_make_l5_bos_inner_at_bound` run at sd=-1 (plain), 16 candles:
#   7  CTS_ESTABLISHED (one_maru_continuous(-1) 6-7), BOS = h6 .6062, BREAKOUT, no range.
#   8  no winner by D; candle 8 is a range candle (pinbar, confirm 12) -> range back-fill [8, 12):
#   9  bull maru closing .6084 > BOS -> watch + the pending reversal (double_maru(+1) 9-10, apply 10);
#   10 the pending reversal applies -> STATE_CHANGED(breakout -> reversal) @10 = the terminal.
#   Before the fix: 11 back-filled in REVERSAL, the range finalized (RANGE_STARTED@12 + STATE_CHANGED(reversal ->
#   range)@12, a moment after the terminal), candle 8 was re-stepped (rows 8-9 'range'), anchor 9 re-opened the
#   watch (duplicate WATCH_START / CANDIDATE @9) and reversed again @10 (+ RANGE_UPDATED@10).
def test_a_reversal_inside_a_range_backfill_does_not_finalize_the_range():
    res = _run(_prepare_df(_bet._make_l5_bos_inner_at_bound()), -1)
    _assert_single_terminal(res, 10, "breakout")
    types = [e.type for e in res.events]
    assert "RANGE_STARTED" not in types
    assert types.count("REVERSAL_WATCH_START") == 1 and types.count("REVERSAL_CANDIDATE") == 1
    assert list(res.df["market_state"].iloc[7:10]) == ["breakout", "breakout", "breakout"]


@pytest.mark.parametrize("bound", [12, 15])
def test_the_range_backfill_terminal_is_the_same_under_a_bound(bound):
    df = _prepare_df(_bet._make_l5_bos_inner_at_bound())
    unbounded, bounded = _run(df, -1), _run(df, -1, end_idx=bound)
    assert [(e.type, e.idx) for e in bounded.events] == [(e.type, e.idx) for e in unbounded.events]


# ---------------------------------------------------------------------------------------------------------------
# A reversal WINNER whose back-fill applies the pending reversal first. `_make_reversing_data` at sd=-1 (plain):
#   10 the pullback applied at 10 close-breaks the BOS .6134 -> the pending reversal (one_maru_continuous(+1) 10-11,
#      apply 11);
#   11 anchor 11: the winner is the reversal from 11 (frozen BOS, apply 12); its back-fill [11, 12) applies the
#      pending one at 11 = the terminal. Before the fix the winner still applied at 12 (no state change) and its
#      apply-row re-write emitted RANGE_UPDATED@12 and wrote row 12.
def test_a_pending_reversal_inside_a_reversal_winners_backfill_ends_the_step():
    res = _run(_prepare_df(_bet._make_reversing_data()), -1)
    _assert_single_terminal(res, 11, "pullback")


def test_a_reversal_winner_keeps_its_own_apply_row_rewrite():
    """The winner IS the reversal (anchor 9, apply 10): its apply candle's re-write stays (the one per-candle step
    of that candle), so RANGE_UPDATED@10 still follows the terminal event on the same candle — the reference
    window's 5 reversals all look like this. Pins the `kind != "reversal"` guard of the post-apply stop."""
    res = _run(_prepare_df(_make_cycle0_reversal_data()), 1)
    rev = _reversals(res.events)
    assert [e.idx for e in rev] == [10]
    assert [(e.type, e.idx) for e in res.events[res.events.index(rev[0]) + 1:]] == [("RANGE_UPDATED", 10)]
    assert res.reversal_idx == 10 and _last_written_row(res) == 10


# ---------------------------------------------------------------------------------------------------------------
# Wiring pins: the post-apply range back-fill (no real-data case found) and the two tripwires.
def _ms(sd: int = 1):
    with redirect_stdout(io.StringIO()):
        return _make_market_structure(_prepare_df(_p1_rows()), struct_direction=sd, start_idx=0)


def test_post_apply_range_backfill_stops_at_the_terminal(monkeypatch):
    ms = _ms()
    ms._init_output_arrays(len(ms.df))
    stepped, finalized = [], []

    def step(k, *, freeze_range=False):
        stepped.append((k, freeze_range))
        if k == 13:
            ms.state.state = MarketState.REVERSAL

    monkeypatch.setattr(ms, "_is_range_candle_given_confirm", lambda i: (True, i + 4))
    monkeypatch.setattr(ms, "_replay_step_no_patterns", step)
    monkeypatch.setattr(ms, "_finalize_range_candidate_offline", lambda i: finalized.append(i))
    ms._post_apply_range_check(12)
    assert stepped == [(12, True), (13, True)] and finalized == []


def test_step_anchor_ends_after_a_terminal_in_the_post_apply_backfill(monkeypatch):
    ms = _ms()
    ms._init_output_arrays(len(ms.df))
    stepped = []

    def apply(ev, apply_idx, kind):   # stands in for a breakout whose post-apply back-fill reversed
        ms.state.state = MarketState.REVERSAL

    monkeypatch.setattr(ms, "_best_bopb_pattern_at_anchor", lambda **kw: (object(), 14, "breakout"))
    monkeypatch.setattr(ms, "_replay_step_no_patterns", lambda k, *, freeze_range=False: stepped.append((k, freeze_range)))
    monkeypatch.setattr(ms, "_apply_pattern_at_apply_idx", apply)
    ms._step_anchor(12)
    assert stepped == [(12, True), (13, True)]   # the back-fill only — no apply-row re-write at 14


@pytest.mark.parametrize("to", [MarketState.BREAKOUT, MarketState.PULLBACK, MarketState.PULLBACK_RANGE, MarketState.RANGE])
def test_set_state_never_leaves_reversal(to):
    ms = _ms()
    ms.state.state = MarketState.REVERSAL
    with pytest.raises(AssertionError, match="left reversal"):
        ms._set_state(to, 5)


def test_set_state_reversal_to_reversal_is_silent():
    ms = _ms()
    ms.state.state = MarketState.REVERSAL
    n = len(ms.events)
    ms._set_state(MarketState.REVERSAL, 5)
    assert len(ms.events) == n and ms.state.state == MarketState.REVERSAL


def test_a_rewind_rebuild_never_reaches_a_reversal(monkeypatch):
    ms = _ms()
    ms._init_output_arrays(len(ms.df))

    def step(i):
        ms.state.state = MarketState.REVERSAL
        return i + 1

    monkeypatch.setattr(ms, "_step_anchor", step)
    with pytest.raises(AssertionError, match="rewind rebuild reached a reversal"):
        ms._rewind_to(3)
