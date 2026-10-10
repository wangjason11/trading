"""PLAN_K landing B — the IC reference condition APPLIED (PLAN_K §4 / §6 B; PLAN_H §6 RX1-F / RX1-N).

An IC candidate passes every IC condition together — the §3.1 ones, the touch, the IC reference (`is_ic_candidate`);
the IP is looked for among IC candidates only, in BOTH consumers (`select_ic`: the drawn POI and MS's read):
- the reference cuts the pick -> the next IC candidate is picked; the tiebreakers run over the SURVIVING tied ones;
- the before-the-break side cuts too;
- no IC candidate -> no POI, the none row `no_ic_candidate`; IC candidates without a pattern -> `no_pattern`;
- MS's in-flight read is filtered (a real MS run: the 302 / 167 J2 fixture's proximity confirmation disappears).
"""
from __future__ import annotations

import contextlib
import io

import pytest

import engine_v2.structure.structure_engine as se
from engine_v2.tests.test_institutional_pattern import _Tracker, _est, _far_gap, _fib_over, _select, _tb2_frame
from engine_v2.zones.cross_cycle_fib import ICReference
from engine_v2.zones.fib_tracker import FibTracker
from engine_v2.zones.poi_zones import (
    IC_REF_NOT_BEFORE_BREAK as NOT_BEFORE,
    IC_REF_PAST_REFERENCE as PAST,
    POIConfig,
    TIEBREAK_IMMEDIATE_UNFILLED_IMBALANCE,
    TIEBREAK_MOST_RECENT_BEFORE_IMBALANCE,
    ICCandleCheck,
    derive_ic_candidates,
)

_GAPS = (10, 12)        # the `_tb2_frame` gaps: 7 has none immediately after it; unreferenced pick 9 over 11 (2 in play)


def _tb2(ref):
    return _select(_tb2_frame(), 6, 12, [_far_gap(g) for g in _GAPS], ic_reference=ref, classify_all=True)


@pytest.mark.parametrize("unfilled_after, touches, reason, want", [
    (True, True, None, True), (True, True, PAST, False), (True, True, NOT_BEFORE, False), (True, False, None, False),
    (True, False, PAST, False), (False, True, None, False), (False, False, PAST, False)])
def test_an_ic_candidate_passes_every_condition(unfilled_after, touches, reason, want):
    """PLAN_L L1: the predicate is the check's property — IC conditions 2 (the reference), 3 (an unfilled imbalance
    after) and 4 (the touch) TOGETHER (condition 1, the agreed opposite direction, is the checks' row set)."""
    check = ICCandleCheck(ic_idx=7, has_unfilled_imbalance_after=unfilled_after, ic_ref_reason=reason,
                          touches_zone=touches)
    assert check.is_ic_candidate is want


def test_the_frame_ics_and_the_unreferenced_pick():
    """The fixture: ICs 7 / 9 / 11, all 1A.i maru, highs 0.60093 / 0.60129 / 0.60165."""
    df = _tb2_frame()
    assert [round(float(df.loc[i, "h"]), 5) for i in (7, 9, 11)] == [0.60093, 0.60129, 0.60165]
    sel = _tb2(ICReference.NONE)
    assert sel.pick.ic_idx == 9 and sel.tiebreak == (TIEBREAK_MOST_RECENT_BEFORE_IMBALANCE, 11, 2, 10)


def test_the_tiebreakers_run_over_the_surviving_tied_ics_only():
    """A reference at 0.60130 cuts 11 (high 0.60165, its immediate gap 12): over {7, 9} only 9 has an immediate
    unfilled gap → 9 by the immediate-unfilled-imbalance tiebreaker, runner-up 7, 2 tied (not 11 / 3)."""
    sel = _tb2(ICReference(0.60130, "prev_cycle_cts"))
    assert {c.ic_idx: c.ic_ref_reason for c in sel.checks} == {7: None, 9: None, 11: PAST}
    assert sel.pick.ic_idx == 9 and sel.tiebreak == (TIEBREAK_IMMEDIATE_UNFILLED_IMBALANCE, 7, 2, 10)


def test_the_reference_cutting_the_pick_moves_it_to_the_next_ic_candidate():
    sel = _tb2(ICReference(0.60100, "prev_cycle_cts"))                     # cuts 9 and 11
    assert {c.ic_idx: c.ic_ref_reason for c in sel.checks} == {7: None, 9: PAST, 11: PAST}
    assert sel.pick.ic_idx == 7 and sel.tiebreak is None


def test_the_before_the_break_side_cuts_too():
    sel = _tb2(ICReference(0.70, "cross_limit_anchor", 9, 10))              # every high under 0.70; break anchor 9
    assert {c.ic_idx: c.ic_ref_reason for c in sel.checks} == {7: None, 9: NOT_BEFORE, 11: NOT_BEFORE}
    assert sel.pick.ic_idx == 7


class _RefTracker(_Tracker):
    def __init__(self, fibs, ref):
        super().__init__(fibs)
        self.ref = ref

    def ic_reference(self, fib_state, df, *, evaluated_at):
        return self.ref


def _none_rows(ref):
    df = _tb2_frame()
    df.attrs["imbalances"] = [_far_gap(g) for g in _GAPS]
    recs = derive_ic_candidates(df, [_est(0, 0)], _RefTracker([_fib_over(df, 6, 12, 0.5, 0.7)], ref), POIConfig(),
                                structure_path_id="H1.main")
    return [(r.ic_idx, r.picked, r.ic_ref_reason, r.reason) for r in recs]


def test_no_ic_candidate_means_no_poi_and_the_none_row_says_so():
    """All three touch the zone and are patterned, and all three are cut: no IC candidate (PLAN_H RX1-N) — the none
    row never names the reference (the conditions filter together, RX1-F)."""
    assert _none_rows(ICReference(0.60090, "prev_cycle_cts")) == [
        (7, False, PAST, None), (9, False, PAST, None), (11, False, PAST, None), (None, False, None, "no_ic_candidate")]
    assert _none_rows(ICReference.NONE)[1][:2] == (9, True)


def test_ic_candidates_without_a_pattern_stay_no_pattern():
    """The M1 frame (`test_plan_h_landing_b`): 32 (2A.i, high 1.13) is cut by 1.128; 36 (high 1.125) is an IC
    candidate with no pattern -> `no_pattern`; 1.12 cuts both -> `no_ic_candidate`."""
    from engine_v2.tests._event_factory import make_cts_established
    from engine_v2.tests.test_plan_h_landing_b import _fib, _m1_frame
    from engine_v2.zones.fib_tracker import FibState
    df = _m1_frame()
    fs = FibState(structure_id=0, cycle_id=0, struct_direction=1, bos_idx=30, bos_price=1.0, cts_idx=40,
                  cts_price=1.35, active=True, locked=False, fib=_fib(1.35, 1.0))
    est = make_cts_established(cts_anchor_idx=40, confirmed_at=40, price=1.35, structure_id=0, cycle_id=0)

    def rows(ref):
        recs = derive_ic_candidates(df, [est], _RefTracker([fs], ref), POIConfig(), structure_path_id="H1.main")
        return [(r.ic_idx, r.touches_zone, r.ic_ref_reason, r.reason) for r in recs]

    assert rows(ICReference(1.128, "prev_cycle_cts")) == [
        (32, True, PAST, None), (36, True, None, "no_continuation"), (None, None, None, "no_pattern")]
    assert rows(ICReference(1.12, "prev_cycle_cts")) == [
        (32, True, PAST, None), (36, True, PAST, "no_continuation"), (None, None, None, "no_ic_candidate")]


# ---- MS's in-flight read is filtered too (a real MS run) ------------------------------------------------------------

def _ms_302_167(neutral):
    from engine_v2.tests import test_plan_j_j2_new_cycle_break as J
    reads = []
    real = se.poi_inner_for_fib

    def spy(df, fib, ft, **kw):
        out = real(df, fib, ft, **kw)
        if fib is not None and fib.cycle_id == 1 and kw["evaluated_at"] in (27, 28):
            reads.append((kw["evaluated_at"], out))
        return out

    se_patch = pytest.MonkeyPatch()
    se_patch.setattr(se, "poi_inner_for_fib", spy)
    try:
        ms, _ = J._run(J._ROWS_302_167, 1, neutral_ic_reference=neutral)
    finally:
        se_patch.undo()
    conf1 = [e for e in ms.events if e.type == "CTS_CONFIRMED" and e.meta["cycle_id"] == 1]
    return reads, [(int(e.idx), e.meta["confirmation_method"]) for e in conf1], ms


def test_ms_read_is_filtered_and_the_proximity_confirmation_follows():
    """302 / 167 (H1 buy): cycle 1's fib (BOS 9) references CTS_0's confirmed price 0.6122. Without the reference MS's
    read at 27 / 28 picks an IC with high 0.6166 (past it) and cycle 1 is CONFIRMED by proximity at 30; with it
    (production) that IC is no IC candidate, the read finds none and cycle 1 confirms later (40)."""
    reads_off, conf_off, _ = _ms_302_167(neutral=True)
    reads_on, conf_on, ms = _ms_302_167(neutral=False)
    assert reads_off and all(out == pytest.approx(0.6166) for _, out in reads_off)
    assert conf_off[0] == (30, "sd_zone_proximity")
    assert reads_on and all(out is None for _, out in reads_on)
    assert conf_on[0][0] == 40 and conf_on[0][1] != "sd_zone_proximity"
    fib = ms._fib_tracker.representative_fib(0, 1)
    ref = FibTracker.ic_reference(ms._fib_tracker, fib, ms.df, evaluated_at=40)
    assert (ref.price, ref.source) == (0.6122, "prev_cycle_cts")
