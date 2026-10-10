"""PLAN_K landing A's review (§11.2) — pins for the mutants the landing's own tests let survive (lens B: N04, N10,
N11, N17, N23 / N25, N24 / N28, N27), each validated to fail on its mutant and pass on HEAD.

- the watch's early stop never skips an anchor that can still confirm FIRST (B2);
- MS's read hands the reference builder its WHOLE frame, at the fed event's moment (B3; K-Q6);
- the sub `reversal_limit`: the reversing sub's most recent BOS_CONFIRMED BY MOMENT (B4), the reversal moment
  inclusive in both builders (B5 — the coded reading; whether a BOS can confirm on the reversal candle itself is open);
- the R6-2 tripwire compares the reversal break too (B6);
- a candidate missing the touch still carries its `ic_ref_reason` (B7 — the CSV's every row);
- the hover names a cycle only for the previous cycle's CTS (B8).
"""
from __future__ import annotations

from types import SimpleNamespace

from engine_v2.common.types import PatternStatus
from engine_v2.patterns.cross_limit_watch import find_cross_limit_break
from engine_v2.tests.test_plan_k_landing_a import P, _DM0, _DM1, _LOW, _detect, _frame, _k
from engine_v2.zones.cross_cycle_fib import ICReference


# ---- B2 (N04): the early stop ---------------------------------------------------------------------------------------

def test_the_early_stop_never_skips_an_anchor_that_can_still_confirm_first():
    """Anchor 1 (one_maru_opposite needing a confirmation) confirms at 5; anchor 3 (double_maru) SUCCEEDS at 4 — an
    anchor two candles before the running best's confirmation can still confirm earlier (a 2-candle SUCCESS applies
    at anchor + 1): the earliest confirmation is (3, 4)."""
    c1 = _k(1.095, 1.140, 1.090, 1.128, "maru", is_big_maru_as0=1)     # closes past P, body mostly past it
    c2 = _k(1.110, 1.112, 1.080, 1.085, "normal")                      # bearish, deep: omo needs a confirmation
    c5 = _k(1.135, 1.150, 1.130, 1.145, "maru")                        # closes >= max(h1, h2) = 1.140
    df = _frame(_LOW, c1, c2, _DM0, _DM1, c5, _LOW, _LOW)
    e1, e3 = _detect(df, 1), _detect(df, 3)
    assert (e1.name, e1.status, e1.confirmation_idx) == ("one_maru_opposite", PatternStatus.CONFIRMED, 5)
    assert (e3.name, e3.status, e3.end_idx) == ("double_maru", PatternStatus.SUCCESS, 4)
    assert find_cross_limit_break(df, P, 1, 0) == (3, 4)
    assert find_cross_limit_break(df, P, 1, 0, known_at=4) == (3, 4)


# ---- B8 (N17): the hover's cycle suffix -----------------------------------------------------------------------------

def test_the_hover_names_no_cycle_for_a_structure_reference():
    from engine_v2.charting._zone_render import fib_hover_ic_ref_text
    meta = {"ic_ref_price": 0.5736, "ic_ref_source": "prev_structure_bos", "ic_ref_break_pattern_anchor_idx": 897,
            "ic_ref_break_confirmed_idx": 902, "cross_fib_limit_source": "reversal_bos_frozen"}
    assert fib_hover_ic_ref_text(SimpleNamespace(meta=meta, cycle_id=1)) == (
        "IC ref 0.57360 (prev_structure_bos) · break 897")


# ---- B3 (N11): MS's read — the whole frame, at the moment (K-Q6) ----------------------------------------------------

def test_ms_read_builds_the_reference_as_of_the_moment_on_the_full_frame():
    import engine_v2.zones.poi_zones as poi_zones
    from engine_v2.tests.test_plan_h_landing_b import _fib, _m1_frame
    df = _m1_frame()
    fs = poi_zones.FibState(structure_id=0, cycle_id=1, struct_direction=1, bos_idx=30, bos_price=1.0, cts_idx=40,
                            cts_price=1.35, active=True, locked=False, fib=_fib(1.35, 1.0))
    calls = []

    class _T:
        def ic_reference(self, f, d, *, evaluated_at):
            calls.append((d is df, evaluated_at))
            return ICReference.NONE

    poi_zones.poi_inner_for_fib(df, fs, 0.7, tracker=_T(), evaluated_at=41)
    assert calls == [(True, 41)]


# ---- B4 / B5 (N23, N25, N24, N28): the reversal builders' "most recent BOS_CONFIRMED by the reversal moment" --------

def _rev_sub(events):
    from engine_v2.tests.test_plan_i_s2_cross_fib_limit import _geo_sub
    rev = _geo_sub(2, 1, 120, events, slice_begin=100)
    rev.natural_reversal_idx = 140
    return rev


def test_reversal_limit_takes_the_latest_bos_by_moment_of_several():
    import engine_v2.multitf.cross_fib_limit as cfl
    from engine_v2.tests.test_plan_i_s2_cross_fib_limit import _bos, _ev
    sc = _ev("STATE_CHANGED", 40, None, to="reversal", pattern_anchor_idx=37, bos_frozen=0.58156, structure_id=0)
    early = _bos("BOS_CONFIRMED", 10, 0.5850, moment=12)
    late = _bos("BOS_CONFIRMED", 20, 0.58198, moment=25)
    assert cfl.reversal_limit(_rev_sub([early, late, sc])).ref_anchor_price == 0.58198      # not the first
    assert cfl.reversal_limit(_rev_sub([late, early, sc])).ref_anchor_price == 0.58198      # by moment, not list order


def test_reversal_limit_counts_a_bos_confirmed_at_the_reversal_moment():
    """"by its reversal moment" is inclusive (`<=`, the code's and the docstring's reading) — sub and main."""
    import engine_v2.multitf.cross_fib_limit as cfl
    from engine_v2.tests._event_factory import make_bos_confirmed
    from engine_v2.tests.test_plan_i_s2_cross_fib_limit import _bos, _ev, _main_events
    from engine_v2.zones.structure_lifecycle import compute_main_cross_fib_limit
    sc = _ev("STATE_CHANGED", 40, None, to="reversal", pattern_anchor_idx=37, bos_frozen=0.58156, structure_id=0)
    late = _bos("BOS_CONFIRMED", 20, 0.58198, moment=25)
    at = _bos("BOS_CONFIRMED", 38, 0.5805, moment=40)
    assert cfl.reversal_limit(_rev_sub([late, at, sc])).ref_anchor_price == 0.5805
    evs = _main_events() + [make_bos_confirmed(bos_anchor_idx=899, confirmed_at=902, price=0.5720, structure_id=0,
                                               cycle_id=3, struct_direction=-1)]
    assert compute_main_cross_fib_limit(evs, 1).ref_anchor_price == 0.5720


# ---- B6 (N27): the R6-2 tripwire compares the reversal break --------------------------------------------------------

def test_the_tripwire_warns_on_a_mismatched_reversal_break():
    from engine_v2.multitf import cross_fib_limit as cfl
    from engine_v2.multitf.cross_fib_limit import NO_LIMIT
    from engine_v2.multitf.sub_structure_pool import StructureKey, SubStructurePool
    from engine_v2.tests.test_plan_i_s2_cross_fib_limit import PARENT, TF, _bos, _df400, _ev, _rec
    from engine_v2.zones.cross_cycle_fib import CrossFibLimit

    def run(brk):
        pool = SubStructurePool()
        x, _ = pool.get_or_create(StructureKey(PARENT, TF, 1, 50))
        x.geometry = (SimpleNamespace(events=[
            _bos("BOS_CONFIRMED", 280, 0.58, moment=290),
            _ev("STATE_CHANGED", 300, None, to="reversal", pattern_anchor_idx=297, bos_frozen=0.58, structure_id=0)],
            df=_df400()), 0)
        x.natural_reversal_idx = 300
        x.records.append(_rec(0, 100, trigger_type="first_confluence", seq=0, direction=1))
        x.start_idx = 100
        y, _ = pool.get_or_create(StructureKey(PARENT, TF, -1, 290))
        y.geometry = (SimpleNamespace(events=[], df=_df400()), 0)
        y.records.append(_rec(1, 320, trigger_type="first_counter", seq=1, trigger_idx=280))   # creating
        y.records.append(_rec(1, 300, trigger_type="reversal", seq=2, trigger_idx=300))        # started_by
        y.start_idx = 300
        pool.number_subs_by_start()
        x.cross_fib_limit = NO_LIMIT
        y.cross_fib_limit = CrossFibLimit(0.58, "reversal_bos_frozen", from_pool_idx=x.pool_idx, from_sub_id=x.sub_id,
                                          ref_anchor_price=0.58, break_pattern_anchor_idx=brk[0],
                                          break_confirmed_idx=brk[1])
        logs = []
        n = cfl.check_cross_fib_limits(pool, [], log=logs.append)
        assert logs[-1].startswith('[cross_fib_limit] tripwire: 1 subs re-derived'), logs
        return n

    assert run((297, 300)) == 0
    assert run((296, 300)) == 1           # the stored break anchor is not the reversing sub's realised pattern


# ---- B7 (N10): every classified candidate carries its reason, a touch miss included ---------------------------------

def test_a_candidate_missing_the_touch_still_carries_its_reason():
    """`*_ic_candidates.csv` reports `ic_ref_reason` on every row (PLAN_K §3.6; the census's cand -> pass counts
    include touch misses). The M1 frame with a fib whose zone neither candidate touches (cts 1.20)."""
    import engine_v2.zones.poi_zones as poi_zones
    from engine_v2.tests.test_plan_h_landing_b import _fib, _m1_frame
    from engine_v2.zones.poi_zones import IC_REF_PAST_REFERENCE as PAST, POIConfig
    df = _m1_frame()
    fs = poi_zones.FibState(structure_id=0, cycle_id=1, struct_direction=1, bos_idx=30, bos_price=1.0, cts_idx=40,
                            cts_price=1.2, active=True, locked=False, fib=_fib(1.2, 1.0))
    sel = poi_zones.select_ic(df, fs, POIConfig(), ic_reference=ICReference(1.128, "prev_cycle_cts"),
                                 classify_all=True)
    assert {c.ic_idx: (c.touches_zone, c.ic_ref_reason) for c in sel.checks} == {
        32: (False, PAST), 36: (False, None)}
