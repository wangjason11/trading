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
The fixtures: the winner back-fill (P1) and the breakout's post-apply range back-fill (P3) from random-tail searches
on the verified cycle-1 base; the range back-fill (P2) and the reversal-winner back-fill are existing
`test_ms_bounded_equals_truncated` fixtures at sd=-1, found by the shadow over the suite; the outside-bar reversal
winner (landing review F1: the kept apply-candle step must not confirm the CTS / create a range after the terminal)
was constructed by the review.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from engine_v2.patterns.structure_patterns import BreakoutPatterns
from engine_v2.structure.event_fields import CTS_UPDATED_RAW_VIA
from engine_v2.structure.market_structure import MarketState, Point
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
#   STATE_CHANGED(reversal -> pullback)@17); 17 close-broke the BOS again -> a watch whose anchor had no reversal
#   pattern, resolved at once (BOS_THRESHOLD_UPDATED rv_anchor_failed -> l17 .60315); 18 close-broke .60315 -> watch +
#   `double_maru(-1)` 18-19 -> a SECOND reversal @19. Hand-off `reversal_idx` 15, `compute_reversal_idx_by_sid` 19.
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
# P3 — a breakout's post-apply range back-fill. Real data: a random-tail search on the P1 base (2026-09-28, landing
# review). sd=+1:
#   0-10 as P1: the one_maru_opposite(+1) breakout applies at 10 (cycle 1, BOS_1 .6088) and candle 10 is a range
#        candle (confirm 14) -> `_post_apply_range_check(10)` back-fills [10, 14):
#   11   bear maru: sd-zone proximity confirms CTS_1 and creates a range (RANGE);
#   12   bear maru closing .60646 < BOS .6088 -> watch + the pending reversal `one_maru_opposite(-1)` 12-13;
#   13   the pending reversal applies -> STATE_CHANGED(range -> reversal) @13 = the terminal.
#   Before the fix: the range finalized (RANGE_STARTED@14 (start 10) + STATE_CHANGED(reversal -> range)@14), row 10
#   was re-written, anchors 11.. re-ran (a pullback @12 with CTS_RECONFIRMED, a second watch @12) and it reversed
#   again @13.
def _p3_rows() -> list[dict]:
    rows = list(_make_second_cts_moment_after_anchor_data()[:11])
    rows += [
        _R(0.61380, 0.61402, 0.61145, 0.61161),   # 11
        _R(0.61161, 0.61182, 0.60644, 0.60646),   # 12
        _R(0.60646, 0.60792, 0.60627, 0.60790),   # 13
        _R(0.60790, 0.61364, 0.60775, 0.61362),   # 14
        _R(0.61362, 0.61387, 0.61139, 0.61167),   # 15
        _R(0.61167, 0.61185, 0.60764, 0.60791),   # 16
        _R(0.60791, 0.60801, 0.60443, 0.60770),   # 17
        _R(0.60770, 0.60792, 0.60654, 0.60655),   # 18
        _R(0.60655, 0.60665, 0.60143, 0.60632),   # 19
        _R(0.60632, 0.60641, 0.60450, 0.60459),   # 20
        _R(0.60459, 0.60701, 0.60449, 0.60484),   # 21
    ]
    return rows


@pytest.mark.parametrize("sd", [1, -1])
def test_a_reversal_inside_a_post_apply_range_backfill_does_not_finalize_the_range(sd):
    rows = _p3_rows() if sd == 1 else _noreg._mirror(_p3_rows())
    df = _prepare_df(rows)
    assert int(df["is_range"].iloc[10]) == 1 and int(df["is_range_confirm_idx"].iloc[10]) == 14   # precondition
    res = _run(df, sd)
    _assert_single_terminal(res, 13, "range")
    cyc1 = [e for e in res.events if e.idx >= 10]   # after cycle 1's establishment (the base's cycle 0 ends at 9)
    assert [(e.idx, e.meta["reason"]) for e in cyc1 if e.type == "RANGE_STARTED"] == [(11, "proximity_created_range")]
    assert not [e for e in cyc1 if e.type == "CTS_RECONFIRMED"]


# ---------------------------------------------------------------------------------------------------------------
# A reversal WINNER's own apply-candle step (landing review F1). sd=+1, the P1 base 0-10, then:
#   11   outside bar (h .6150 > CTS .6147, close .6062 < BOS .6088): the raw path moves the CTS to 11, so the
#        proximity gate `i > cts.idx` is shut on 11; the close-break makes the anchor-11 reversal the winner
#        (`one_maru_continuous(-1)` 11-12, apply 12) -> STATE_CHANGED(breakout -> reversal) @12 = the terminal.
#   12   the winner's apply-candle step: the first candle proximity may fire on.
#   Before the fix it did (CTS_CONFIRMED + RANGE_STARTED + STATE_CHANGED(reversal -> range) @12, then a second watch
#   and a second reversal @12); with only the `_set_state` tripwire it raised. The raw CTS update and proximity now
#   skip in REVERSAL, like the BOS barrier.
def _outside_bar_rows() -> list[dict]:
    rows = list(_make_second_cts_moment_after_anchor_data()[:11])
    rows += [
        _R(0.61440, 0.61500, 0.60600, 0.60620),   # 11
        _R(0.60620, 0.60640, 0.60000, 0.60020),   # 12
        _R(0.60020, 0.60060, 0.59980, 0.60040),   # 13
        _R(0.60040, 0.60080, 0.60010, 0.60060),   # 14
        _R(0.60060, 0.60090, 0.60030, 0.60070),   # 15
    ]
    return rows


@pytest.mark.parametrize("sd", [1, -1])
def test_a_reversal_winners_apply_step_confirms_nothing_after_the_terminal(sd):
    rows = _outside_bar_rows() if sd == 1 else _noreg._mirror(_outside_bar_rows())
    res = _run(_prepare_df(rows), sd)
    _assert_single_terminal(res, 12, "breakout")
    raw = [e for e in res.events if e.type == "CTS_UPDATED"]
    assert [(e.idx, e.meta["via"]) for e in raw][-1] == (11, CTS_UPDATED_RAW_VIA)   # precondition: the outside bar
    assert not [e for e in res.events if e.idx >= 10 and e.type in ("CTS_CONFIRMED", "RANGE_STARTED")]


@pytest.mark.parametrize("state, moves", [(MarketState.BREAKOUT, True), (MarketState.REVERSAL, False)])
def test_the_raw_cts_update_skips_in_reversal(state, moves):
    ms = _ms()
    st = ms.state
    st.cts, st.cts_phase, st.state = Point(idx=5, price=0.6000), "EST_OR_UPD", state   # candle 12's high .61374 is new
    n = len(ms.events)
    ms._maybe_update_cts_pre_confirm(12, via=CTS_UPDATED_RAW_VIA)
    assert (len(ms.events) > n) is moves and (st.cts.idx == 12) is moves


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
# Wiring pins: the post-apply range back-fill's stop and `_step_anchor`'s return after it (P3 above is the real-data
# case), and the two tripwires.
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
