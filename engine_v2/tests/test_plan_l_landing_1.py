"""PLAN_L landing 1 — the IC identification in the user's steps (`plans/PLAN_L_ic_identification_steps.md` §3; the
decisions PLAN_H §6 ICR-*): Step 1's per-candle checks (`check_ic_conditions` — every agreed-opposite candle of the
fib's window with IC conditions 2–4) and its answer (`find_ic_candidates`); the shared rule helpers (one function per
rule that is the same in Step 2's tiebreak rules and Step 3's ranking rules, each given its own input list — ICR-9);
the debug CSV's new rows + columns (ICR-4 / ICR-11); the drawn pick == MS's pick when a NON-candidate carries the
better pattern (PLAN_H D2 with teeth). The selection results are unchanged by landing 1.
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine_v2.patterns.imbalance import has_unfilled_imbalance
from engine_v2.tests.test_institutional_pattern import (
    D,
    U,
    _est,
    _far_gap,
    _fib_over,
    _frame,
    _select,
    _tb2_frame,
    _Tracker,
)
from engine_v2.debug.export_zones import IC_CANDIDATE_COLUMNS, export_ic_candidates
from engine_v2.features.candles_v2 import agreed_direction
from engine_v2.zones.cross_cycle_fib import ICReference
from engine_v2.patterns.institutional_pattern import IPResult
from engine_v2.zones.poi_zones import (
    IC_REF_PAST_REFERENCE as PAST,
    TIEBREAK_IC_CANDLE_TYPE,
    TIEBREAK_MOST_RECENT_BEFORE_IMBALANCE,
    POIConfig,
    apply_tiebreak_rules,
    check_ic_conditions,
    derive_ic_candidates,
    find_ic_candidates,
    find_ic_variant_candidates,
    immediate_unfilled_imbalance_first_c2_idx,
    most_recent_before_imbalance,
    poi_inner_for_fib,
    prefer_candle_type,
    select_ic,
)


def _checks(df, fib, ref=ICReference.NONE):
    agreed = agreed_direction(df["direction"].to_numpy(), df["vol_dir"].to_numpy())
    return check_ic_conditions(df, fib, POIConfig(), ic_reference=ref, agreed=agreed)


# ---------------------------------------------------------------------------
# Step 1 — the per-candle checks
# ---------------------------------------------------------------------------

def test_each_condition_is_reported_on_its_own():
    """`_tb2_frame` ICs 7 / 9 / 11 (highs 0.60093 / 0.60129 / 0.60165). Gaps first c2 8 and 10 → condition 3 (the
    immediate gap, PLAN_P): 7 / 9 yes, 11 no; a zone from just above 7's high → condition 4: 7 misses; a reference
    0.60150 → condition 2: 11 is past it. So 7 fails ONLY the touch, 9 passes all four, 11 fails the reference AND
    condition 3."""
    df = _tb2_frame()
    df.attrs["imbalances"] = [_far_gap(8), _far_gap(10)]
    h7, h9, h11 = (round(float(df.loc[i, "h"]), 5) for i in (7, 9, 11))
    assert (h7, h9, h11) == (0.60093, 0.60129, 0.60165)
    fib = _fib_over(df, 6, 12, h7 + 0.00001, 0.6100)
    checks = _checks(df, fib, ICReference(0.60150, "prev_cycle_cts"))
    got = {c.ic_idx: (c.immediate_imbalance_first_c2_idx, c.touches_zone, c.ic_ref_reason, c.is_ic_candidate)
           for c in checks}
    assert got == {7: (8, False, None, False), 9: (10, True, None, True), 11: (None, True, PAST, False)}
    # condition 3 IS `immediate_unfilled_imbalance_first_c2_idx` (set on EVERY check, unclassified too — PLAN_P); the
    # retired condition 3 is `has_unfilled_imbalance` over (idx, CTS anchor], fill horizon the anchor (information)
    for c in checks:
        assert c.immediate_imbalance_first_c2_idx == immediate_unfilled_imbalance_first_c2_idx(df, c.ic_idx, fib,
                                                                                               POIConfig())
        assert c.has_unfilled_imbalance_after is has_unfilled_imbalance(
            df, start_idx=c.ic_idx + 1, end_idx=12, check_to_idx=12, direction=1, fill_threshold=0.70,
            evaluated_at=None)
    assert [c.has_unfilled_imbalance_after for c in checks] == [True, True, False]
    assert find_ic_candidates(checks) == [9]
    assert all(c.pattern is None for c in checks)                                                  # unclassified


def test_the_checks_are_every_agreed_opposite_candle_of_the_window():
    """The row set = IC condition 1: a volume-only doji (7) is listed, a split candle (9) is not, the sd candles
    neither; the window is BOS … CTS inclusive."""
    df = _frame([("M", U, 10), ("Z", 0, 2, -3), ("M", U, 10), ("M", D, 3, 5), ("M", U, 10)])
    df.attrs["imbalances"] = [_far_gap(10)]
    assert [c.ic_idx for c in _checks(df, _fib_over(df, 6, 10, 0.5, 0.7))] == [7]
    assert [c.ic_idx for c in _checks(df, _fib_over(df, 8, 10, 0.5, 0.7))] == []     # 7 outside the window


# ---------------------------------------------------------------------------
# The shared rule helpers — one function, each rule set's own input list (ICR-9)
# ---------------------------------------------------------------------------

def test_the_candle_type_rule_with_and_without_pinbar():
    df = pd.DataFrame({"candle_type": ["maru", "normal", "pinbar", "maru"]}, index=[1, 2, 3, 4])
    assert prefer_candle_type([1, 2, 3, 4], df, include_pinbar=True) == [1, 4]
    assert prefer_candle_type([2, 3], df, include_pinbar=True) == [2]
    assert prefer_candle_type([3], df, include_pinbar=True) == [3]
    assert prefer_candle_type([1, 2], df, include_pinbar=False) == [1]
    assert prefer_candle_type([2], df, include_pinbar=False) == [2]
    with pytest.raises(ValueError):                      # the ranking rules exclude pinbars before this rule
        prefer_candle_type([2, 3], df, include_pinbar=False)


def test_most_recent_before_imbalance_on_two_input_lists():
    """Gaps first c2 10 and 12. [7, 9] → the reference after 7 is 10 → 9 (the gap's c1 — closest to the imbalance,
    ICR-1); [7] alone → 7; [9, 11] → the reference after 9 is 10 → 9; [11] → 12 → 11."""
    df = _tb2_frame()
    df.attrs["imbalances"] = [_far_gap(10), _far_gap(12)]
    fib = _fib_over(df, 6, 12, 0.5, 0.7)
    assert most_recent_before_imbalance([7, 9], df, fib, POIConfig()) == (9, 10)
    assert most_recent_before_imbalance([7], df, fib, POIConfig()) == (7, 10)
    assert most_recent_before_imbalance([9, 11], df, fib, POIConfig()) == (9, 10)
    assert most_recent_before_imbalance([11], df, fib, POIConfig()) == (11, 12)


def test_the_immediate_window_is_one_parameter():
    """`POIConfig.immediate_imbalance_candles` (= 2; renamed from `ip_immediate_imbalance_candles`): IC 7's gap at 10
    is IC + 3 → outside the default window, inside a window of 3."""
    df = _tb2_frame()
    df.attrs["imbalances"] = [_far_gap(10)]
    fib = _fib_over(df, 6, 12, 0.5, 0.7)
    assert POIConfig().immediate_imbalance_candles == 2
    assert immediate_unfilled_imbalance_first_c2_idx(df, 7, fib, POIConfig()) is None
    assert immediate_unfilled_imbalance_first_c2_idx(df, 7, fib, POIConfig(immediate_imbalance_candles=3)) == 10


# ---------------------------------------------------------------------------
# PLAN_H D2 with teeth — a NON-candidate with the better pattern is never picked, by either consumer
# ---------------------------------------------------------------------------

def _best_rank_frame():
    # ICs 7 (3A.i), 9 (1A.i), 11 (2A.i) — `test_best_rank_wins_over_three_candidates`
    return _frame([("M", U, 10), ("P", D, 6), ("M", U, 10), ("M", D, 6), ("M", U, 10), ("N", D, 6), ("M", U, 10)])


def test_a_non_candidate_with_the_better_pattern_is_never_picked():
    """One gap, first c2 8: only 7 has an unfilled imbalance after it. 9 (1A.i, the best pattern) and 11 fail ONLY
    condition 3 → the IC candidate 7 (3A.i) is the pick for the drawn record (`classify_all`, 9 classified) AND MS's
    read (only the IC candidates classified)."""
    df = _best_rank_frame()
    drawn = _select(df, 6, 12, [_far_gap(8)], classify_all=True)
    ms = _select(df, 6, 12, [_far_gap(8)])
    assert {c.ic_idx: (c.is_ic_candidate, c.pattern.ip_rank) for c in drawn.checks} == {
        7: (True, "3A.i"), 9: (False, "1A.i"), 11: (False, "2A.i")}
    assert [(c.ic_idx, c.pattern is None) for c in ms.checks] == [(7, False), (9, True), (11, True)]
    assert drawn.pick.ic_idx == ms.pick.ic_idx == 7


# ---------------------------------------------------------------------------
# The debug CSV — every agreed-opposite candle, with the two new columns (ICR-4 / ICR-11)
# ---------------------------------------------------------------------------

def test_the_csv_lists_a_candle_failing_condition_3_with_its_outcomes(tmp_path):
    df = _best_rank_frame()
    df.attrs["imbalances"] = [_far_gap(8)]
    recs = derive_ic_candidates(df, [_est(0, 0)], _Tracker([_fib_over(df, 6, 12, 0.5, 0.7)]), POIConfig(),
                                structure_path_id="H1.main")
    assert [(r.ic_idx, r.picked, r.ip_rank, r.has_unfilled_imbalance_after, r.is_ic_candidate) for r in recs] == [
        (7, True, "3A.i", True, True), (9, False, "1A.i", False, False), (11, False, "2A.i", False, False)]
    path = tmp_path / "x_ic_candidates.csv"
    export_ic_candidates(recs, path)
    out = pd.read_csv(path)
    cols = list(out.columns)
    assert cols == IC_CANDIDATE_COLUMNS
    k = cols.index("immediate_imbalance_first_c2_idx")
    assert cols[k + 1: k + 4] == ["has_unfilled_imbalance_after", "is_ic_candidate", "ic_zone_band"]   # PLAN_P
    assert out["has_unfilled_imbalance_after"].tolist() == [True, False, False]
    assert out["immediate_imbalance_first_c2_idx"].fillna(-1).tolist() == [8, -1, -1]    # condition 3 (PLAN_P)
    assert out["is_ic_candidate"].tolist() == [True, False, False]


def test_the_none_row_leaves_the_two_columns_empty():
    df = _best_rank_frame()
    df.attrs["imbalances"] = [_far_gap(8)]
    no_touch = _fib_over(df, 6, 12, 0.70, 0.71)                 # a zone above every candle → no IC candidate
    recs = derive_ic_candidates(df, [_est(0, 0)], _Tracker([no_touch]), POIConfig(), structure_path_id="H1.main")
    none = recs[-1]
    assert (none.ic_idx, none.reason, none.has_unfilled_imbalance_after, none.is_ic_candidate) == (
        None, "no_ic_candidate", None, None)
    assert [r.is_ic_candidate for r in recs[:-1]] == [False, False, False]


def test_a_cts_anchor_candle_is_checked_and_can_never_be_a_candidate():
    """The window is BOS … CTS INCLUSIVE: with the CTS anchor on IC 11 (an opposite candle), 11 is a check — its
    condition 3 window (IC + 1 … min(IC + 2, the anchor) — empty) finds nothing, nor does the retired one's `(11,
    11]`, so it fails (mutation L05: a window ending before the anchor). Gaps 8 / 10: 7 and 9 pass."""
    df = _tb2_frame()
    df.attrs["imbalances"] = [_far_gap(8), _far_gap(10)]
    checks = _checks(df, _fib_over(df, 6, 11, 0.5, 0.7))
    assert [(c.ic_idx, c.immediate_imbalance_first_c2_idx, c.has_unfilled_imbalance_after) for c in checks] == [
        (7, 8, True), (9, 10, True), (11, None, False)]
    assert find_ic_candidates(checks) == [7, 9]


def test_the_candle_type_runner_up_is_the_best_losing_type():
    """The tiebreak rules' candle type deciding (one maru, 5): the runner-up = the most recent IC of the BEST losing
    type — the normal 3, not the more recent pinbar 8 (mutation L20: the most recent loser)."""
    df = pd.DataFrame({"candle_type": ["normal", "maru", "pinbar"]}, index=[3, 5, 8])
    tied = [IPResult(ic_idx=3), IPResult(ic_idx=5), IPResult(ic_idx=8)]
    pick, tb = apply_tiebreak_rules(df, None, POIConfig(), tied)
    assert pick.ic_idx == 5 and tb == (TIEBREAK_IC_CANDLE_TYPE, 3, 3, None)


def test_the_v_path_finder_keeps_condition_3():
    """`find_ic_variant_candidates` (the unwired V30 / V60 / V90 path, kept — ICR-5): candle direction + an unfilled
    imbalance after; with one gap at 10, IC 11 has none (mutation L28: condition 3 dropped)."""
    df = _tb2_frame()
    df.attrs["imbalances"] = [_far_gap(10)]
    assert find_ic_variant_candidates(df, _fib_over(df, 6, 12, 0.5, 0.7), POIConfig()) == [7, 9]


def test_the_active_or_locked_guard_holds_for_both_consumers():
    """`select_ic` is the ONE holder of the guard since L1 (`check_ic_conditions` has none): a fib neither active nor
    locked → no checks, no pick (MS's `poi_inner_for_fib` reads an embedded tracker's representative fib, which can be
    one — the drawn path is shielded by the record gate); a LOCKED fib that is no longer active is still read (the
    record gate passes it) — in both classify modes. Mutations R08 / R10 (the landing review)."""
    import dataclasses

    df = _tb2_frame()
    df.attrs["imbalances"] = [_far_gap(10)]
    live = _fib_over(df, 6, 12, 0.5, 0.7)
    dead = dataclasses.replace(live, active=False, locked=False)
    locked = dataclasses.replace(live, active=False, locked=True)
    for classify_all in (False, True):
        sel = select_ic(df, dead, POIConfig(), ic_reference=ICReference.NONE, classify_all=classify_all)
        assert sel.checks == () and sel.pick is None
        assert select_ic(df, locked, POIConfig(), ic_reference=ICReference.NONE,
                         classify_all=classify_all).pick.ic_idx == 9

    class _NoRefTracker:
        def ic_reference(self, fib_state, df, *, evaluated_at):
            return ICReference.NONE

    assert poi_inner_for_fib(df, dead, 0.70, tracker=_NoRefTracker(), evaluated_at=12) is None
    assert poi_inner_for_fib(df, locked, 0.70, tracker=_NoRefTracker(), evaluated_at=12) == float(df.loc[9, "h"])


def test_the_most_recent_rule_runner_up_is_among_the_best_type_only():
    """Tiebreak rule 2 (the most recent before the imbalance; rule 3 until PLAN_P) deciding: its runner-up = the most
    recent OTHER IC of the candle type's survivors (the marus 7 / 9), never a more recent IC of a worse type still in
    play (the normal 11). The function on its own (no Step 1 here): reference gap 10 → 9 over 7, the count 2
    (mutation R02 — the landing review: the runner-up over the whole rule-1 input)."""
    df = _tb2_frame().copy()
    df.attrs["imbalances"] = [_far_gap(10)]
    df.loc[11, "candle_type"] = "normal"
    fib = _fib_over(df, 6, 12, 0.5, 0.7)
    tied = [IPResult(ic_idx=7), IPResult(ic_idx=9), IPResult(ic_idx=11)]
    pick, tb = apply_tiebreak_rules(df, fib, POIConfig(), tied)
    assert pick.ic_idx == 9 and tb == (TIEBREAK_MOST_RECENT_BEFORE_IMBALANCE, 7, 2, 10)

