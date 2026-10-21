"""PLAN_K landing B — the IC reference condition APPLIED (PLAN_K §4 / §6 B; PLAN_H §6 RX1-F / RX1-N).

An IC candidate passes every IC condition together — the §3.1 ones, the touch, the IC reference (`is_ic_candidate`);
the IP is looked for among IC candidates only, in BOTH consumers (`select_ic`: the drawn POI and MS's read):
- the reference cuts the pick -> the next IC candidate is picked; the tiebreakers run over the SURVIVING tied ones;
- the before-the-break side cuts too;
- no IC candidate -> no POI, the none row `no_ic_candidate`; IC candidates without a pattern -> `no_pattern` (since
  PLAN_L L2 Step 2's `ip_outcome`, and the ranking rules pick among them);
- MS's in-flight read is filtered (a real MS run: the 302 / 167 J2 fixture's proximity confirmation disappears).
"""
from __future__ import annotations

import contextlib
import io

import pytest

import engine_v2.structure.structure_engine as se
from engine_v2.tests.test_institutional_pattern import (
    _Tracker, _bos_gap, _est, _far_gap, _fib_over, _select, _tb2_frame,
)
from engine_v2.zones.cross_cycle_fib import ICReference
from engine_v2.zones.fib_tracker import FibTracker
from engine_v2.zones.poi_zones import (
    IC_REF_NOT_BEFORE_BREAK as NOT_BEFORE,
    IC_REF_PAST_REFERENCE as PAST,
    POIConfig,
    TIEBREAK_MOST_RECENT_BEFORE_IMBALANCE,
    ICCandleCheck,
    derive_ic_candidates,
)

_GAPS = (10, 12)        # the `_tb2_frame` gaps: 7 has none immediately after it (no IC candidate — PLAN_P's
#                         condition 3); unreferenced pick 9 over 11 (2 in play)
_GAPS_7 = (8, 10, 12)   # + a gap immediately after 7: all three are IC candidates (PLAN_P)


def _tb2(ref, gaps=_GAPS):
    return _select(_tb2_frame(), 6, 12, [_far_gap(g) for g in gaps], ic_reference=ref, classify_all=True)


@pytest.mark.parametrize("immediate, unfilled_after, touches, reason, want", [
    (10, True, True, None, True), (10, True, True, PAST, False), (10, True, True, NOT_BEFORE, False),
    (10, True, False, None, False), (10, True, False, PAST, False),
    (None, True, True, None, False),          # the OLD condition 3 passes, the new one fails → not a candidate
    (None, False, True, None, False), (None, False, False, PAST, False)])
def test_an_ic_candidate_passes_every_condition(immediate, unfilled_after, touches, reason, want):
    """PLAN_L L1: the predicate is the check's property — IC conditions 2 (the reference), 3 (since PLAN_P the
    unfilled imbalance IMMEDIATELY after: `immediate_imbalance_first_c2_idx` not None) and 4 (the touch) TOGETHER
    (condition 1, the agreed opposite direction, is the checks' row set). `has_unfilled_imbalance_after` (the retired
    condition 3) is never read: the row with it True and no immediate gap is no candidate."""
    check = ICCandleCheck(ic_idx=7, has_unfilled_imbalance_after=unfilled_after, ic_ref_reason=reason,
                          touches_zone=touches, immediate_imbalance_first_c2_idx=immediate)
    assert check.is_ic_candidate is want


def test_the_frame_ics_and_the_unreferenced_pick():
    """The fixture: ICs 7 / 9 / 11, all 1A.i maru, highs 0.60093 / 0.60129 / 0.60165."""
    df = _tb2_frame()
    assert [round(float(df.loc[i, "h"]), 5) for i in (7, 9, 11)] == [0.60093, 0.60129, 0.60165]
    sel = _tb2(ICReference.NONE)
    assert sel.pick.ic_idx == 9 and sel.tiebreak == (TIEBREAK_MOST_RECENT_BEFORE_IMBALANCE, 11, 2, 10)


def test_the_tiebreakers_run_over_the_surviving_tied_ics_only():
    """Every IC has an immediate gap (8 / 10 / 12); a reference at 0.60130 cuts 11 (high 0.60165): the tiebreak rules
    run over {7, 9} only — the most recent before the imbalance after the oldest (7, gap 8) → 7, runner-up 9, **2**
    tied (over all three it would be 3)."""
    sel = _tb2(ICReference(0.60130, "prev_cycle_cts"), _GAPS_7)
    assert {c.ic_idx: c.ic_ref_reason for c in sel.checks} == {7: None, 9: None, 11: PAST}
    assert sel.pick.ic_idx == 7 and sel.tiebreak == (TIEBREAK_MOST_RECENT_BEFORE_IMBALANCE, 9, 2, 8)


def _best_rank(ref):
    """`test_plan_l_landing_1._best_rank_frame`: ICs 7 (3A.i, high 0.60132) / 9 (1A.i, 0.60171) / 11 (2A.i, 0.60219),
    each with its immediate gap (8 / 10 / 12) — the unreferenced pick is 9 (the best rank, no tie). (The `_tb2_frame`
    can no longer show a MOVE since PLAN_P: its tie goes to the oldest, 7, and a price cut there removes only newer
    ICs — the landing review, 2026-10-13.)"""
    from engine_v2.tests.test_plan_l_landing_1 import _best_rank_frame
    return _select(_best_rank_frame(), 6, 12, [_far_gap(g) for g in (8, 10, 12)], ic_reference=ref,
                   classify_all=True)


def test_the_reference_cutting_the_pick_moves_it_to_the_next_ic_candidate():
    assert _best_rank(ICReference.NONE).pick.ic_idx == 9
    sel = _best_rank(ICReference(0.60150, "prev_cycle_cts"))              # cuts 9 and 11
    assert {c.ic_idx: c.ic_ref_reason for c in sel.checks} == {7: None, 9: PAST, 11: PAST}
    assert (sel.pick.ic_idx, sel.pick.ip_rank, sel.tiebreak) == (7, "3A.i", None)


def test_the_before_the_break_side_cuts_too():
    sel = _best_rank(ICReference(0.70, "cross_limit_anchor", 9, 10))       # every high under 0.70; break anchor 9
    assert {c.ic_idx: c.ic_ref_reason for c in sel.checks} == {7: None, 9: NOT_BEFORE, 11: NOT_BEFORE}
    assert (sel.pick.ic_idx, sel.pick.ip_rank) == (7, "3A.i")


class _RefTracker(_Tracker):
    def __init__(self, fibs, ref):
        super().__init__(fibs)
        self.ref = ref

    def ic_reference(self, fib_state, df, *, evaluated_at):
        return self.ref


def _none_rows(ref):
    df = _tb2_frame()
    df.attrs["imbalances"] = [_bos_gap(6)] + [_far_gap(g) for g in _GAPS]      # 1B-neutral (PLAN_S step 3)
    recs = derive_ic_candidates(df, [_est(0, 0)], _RefTracker([_fib_over(df, 6, 12, 0.5, 0.7)], ref), POIConfig(),
                                structure_path_id="H1.main")
    return [(r.ic_idx, r.picked, r.ic_ref_reason, r.reason) for r in recs]


def test_no_ic_candidate_means_no_poi_and_the_none_row_says_so():
    """All three touch the 61.8–80 zone and are patterned, and all three are cut: no IC candidate (PLAN_H RX1-N; Step
    1A runs and finds none either) — the none row never names the reference (the conditions filter together, RX1-F)."""
    assert _none_rows(ICReference(0.60090, "prev_cycle_cts")) == [
        (7, False, PAST, None), (9, False, PAST, None), (11, False, PAST, None), (None, False, None, "no_ic_candidate")]
    assert _none_rows(ICReference.NONE)[1][:2] == (9, True)


def test_ic_candidates_without_a_pattern_are_no_pattern_and_ranked():
    """The M1 frame (`test_plan_h_landing_b`): 32 (2A.i, high 1.13) is cut by 1.128; 36 (high 1.125) is an IC
    candidate with no pattern -> `ip_outcome` `no_pattern` on every row and (PLAN_L L2) 36 is the ranked pick (the
    only IC candidate: no ranking rule) — no no-IC row; its row keeps the pattern diagnostics (`no_continuation`,
    ICR-15); 1.12 cuts both -> no IC candidate in 61.8–80 -> Step 1A (PLAN_P) re-checks the touch in 80–100 ([1.0,
    1.07]: neither low reaches it — every row's `touches_zone` is now THAT band's) -> `no_ic_candidate` (Step 2 never
    ran: `ip_outcome` empty), band 80-100 on every row."""
    from engine_v2.tests._event_factory import make_cts_established
    from engine_v2.tests.test_plan_h_landing_b import _fib, _m1_frame
    from engine_v2.zones.fib_tracker import FibState
    df = _m1_frame()
    fs = FibState(structure_id=0, cycle_id=0, struct_direction=1, bos_idx=30, bos_price=1.0, cts_idx=40,
                  cts_price=1.35, active=True, locked=False, fib=_fib(1.35, 1.0))
    est = make_cts_established(cts_anchor_idx=40, confirmed_at=40, price=1.35, structure_id=0, cycle_id=0)

    def rows(ref):
        recs = derive_ic_candidates(df, [est], _RefTracker([fs], ref), POIConfig(), structure_path_id="H1.main")
        return [(r.ic_idx, r.touches_zone, r.ic_ref_reason, r.reason, r.ip_outcome, r.picked, r.ranking_rule,
                 r.ic_zone_band) for r in recs]

    assert rows(ICReference(1.128, "prev_cycle_cts")) == [
        (32, True, PAST, None, "no_pattern", False, None, "61.8-80"),
        (36, True, None, "no_continuation", "no_pattern", True, None, "61.8-80")]
    assert rows(ICReference(1.12, "prev_cycle_cts")) == [
        (32, False, PAST, None, None, False, None, "80-100"), (36, False, PAST, "no_continuation", None, False, None,
                                                               "80-100"),
        (None, None, None, "no_ic_candidate", None, False, None, "80-100")]


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
