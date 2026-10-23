"""PLAN_M landing 1 (§7.1 / §7.2 / §7.4): the live-`P` plumbing — a cross-fib limit whose `P` MOVES.

No producer makes a limit move yet (landing 2: a counter sub's `parent_bos_threshold`), so every case here is a
synthetic moving limit; the engine's outputs are byte-identical to S_L2 (PLAN_M §8).

- `CrossFibLimit`: `price_at` (inclusive at a move's `known_at`), `moves_until`, `localized` (the moves shifted — the
  live source never has a break), the new asserts;
- `FibTracker`: a cycle's cross permission read LIVE until its CTS_ESTABLISHED, FIXED from then on (CL-1), undone
  by an MS snapshot restore; `_structure_reference_limit` takes the fixed answer; the cross-limit watch reads `P` as
  of the reading moment (CL-7); the export = `P` as of the structure's END (CL-6, decided (a) 2026-10-11) + the two
  keys when it moved; a moving limit read with no moment raises;
- the hover's `(moved N× from <start P>)`;
- the REAL collaborator: a real MarketStructure run (its embedded tracker) and the drawn tracker on the same moving
  limit, and the geometry's slice-local moves.

Fixtures: the flat frame of `test_plan_i_s3_cross_fib_limit_applied` (`_CYCLES`: CTS_0 1.15 at 20, confirmed 25;
CTS_1 1.35 at 40, confirmed 45; CTS_2 1.55 at 60 — each cycle with its own unfilled gap; sd −1 the exact mirror).
"""
from __future__ import annotations

import copy
import dataclasses
import math
from types import SimpleNamespace

import pytest

import engine_v2.structure.event_fields as ef
import engine_v2.structure.structure_engine as se
from engine_v2.charting._zone_render import fib_hover_limit_text
from engine_v2.pipeline.orchestrator import _run_downstream_pipeline
from engine_v2.structure.event_fields import CTS_UPDATED_RAW_VIA
from engine_v2.tests._cross_fib_limits import unreached
from engine_v2.tests.test_plan_i_s3_cross_fib_limit_applied import (
    _CYCLES, _conf, _df, _est, _ev, _gap, _kind, _m, _q,
)
from engine_v2.tests.test_plan_i_s5_ms_reads_the_fib import _rows, embedded  # noqa: F401  (fixture)
from engine_v2.tests.test_plan_k_landing_a import _probe, _tracker_with, _watch_frame
from engine_v2.tests.test_render_sub_projection import geometry, m15_df  # noqa: F401  (fixtures)
from engine_v2.tests.test_unified_probe import _prepare_df
from engine_v2.zones.cross_cycle_fib import CrossFibLimit, ICReference
from engine_v2.zones.fib_tracker import CrossFibLimitError, FibTracker, FibTrackerConfig


def _live(p0, moves, sd, anchor=None):
    """A `parent_bos_threshold` limit on the fixture's mirrored prices: `P` = `p0` at the start, then each move
    `(known_at, price)`; the anchor behind it (`ref_anchor_price`) = `anchor`, default `p0`."""
    mv = tuple((int(k), _m(p, sd)) for k, p in moves)
    return CrossFibLimit(mv[-1][1] if mv else _m(p0, sd), "parent_bos_threshold",
                         ref_anchor_price=_m(p0 if anchor is None else anchor, sd),
                         price_at_start=_m(p0, sd) if mv else None, price_moves=mv)


def _frame(sd):
    return _df(80, [_gap(g[0], sd, g[1], g[2]) for *_, g in _CYCLES])


def _run_cycles(lim, sd, upto=2, cycle2_at=60, df=None):
    """The three cycles of `_CYCLES` fed to a tracker holding `lim` (cycle 2 established at `cycle2_at`)."""
    t = FibTracker(FibTrackerConfig(fill_threshold=0.70), cross_fib_limits={0: lim})
    df = _frame(sd) if df is None else df
    for c, (bos, bos_p, cts, cts_p, conf, _g) in enumerate(_CYCLES[:upto + 1]):
        at = cycle2_at if c == 2 else cts
        _q(t.on_cts_established, _est(at, cts_p, c, sd), df, bos_idx=bos, bos_price=_m(bos_p, sd))
        if conf is not None and c < upto:
            _q(t.on_cts_confirmed, _conf(conf, cts_p, c, sd, cts))
    return t, df


# =====================================================================================================================
# 1. CrossFibLimit
# =====================================================================================================================

# The census case (2025-02-01 → 03-22, sub 6, a buy): `P` 0.57328 at the start, moved by the parent's threshold.
_SUB6 = CrossFibLimit(0.5763, "parent_bos_threshold", ref_anchor_price=0.57328, price_at_start=0.57328,
                      price_moves=((110, 0.575), (120, 0.57597), (130, 0.5763)))


def test_price_at_reads_the_last_move_known_by_the_moment_inclusive():
    p = _SUB6.price_at
    assert (p(None), p(0), p(109), p(110), p(119), p(120), p(129), p(130), p(10 ** 6)) == (
        0.5763, 0.57328, 0.57328, 0.575, 0.575, 0.57597, 0.57597, 0.5763, 0.5763)
    assert _SUB6.price == 0.5763                                    # the object's last P (up to the parent cycle's end)


def test_moves_until_counts_the_moves_known_by_the_moment():
    assert _SUB6.moves_until(None) == _SUB6.price_moves
    assert _SUB6.moves_until(109) == ()
    assert _SUB6.moves_until(110) == ((110, 0.575),)
    assert _SUB6.moves_until(129) == ((110, 0.575), (120, 0.57597))


def test_a_limit_that_never_moved_reads_its_one_price_at_every_moment():
    for lim in (CrossFibLimit(0.58105, "parent_bos_threshold", ref_anchor_price=0.58105),
                CrossFibLimit(0.57827, "prev_sub_bos_threshold", ref_anchor_price=0.57827),
                CrossFibLimit(None, "none")):
        assert {lim.price_at(t) for t in (None, 0, 500)} == {lim.price}
        assert lim.moves_until(None) == () and lim.moves_until(500) == ()


def test_localized_shifts_every_move_of_a_limit_without_a_break():
    """The live source never has a break: `localized` must still move the moves (the review found the early return)."""
    loc = _SUB6.localized(100)
    assert loc == dataclasses.replace(_SUB6, price_moves=((10, 0.575), (20, 0.57597), (30, 0.5763)))
    assert (loc.price, loc.price_at_start, loc.break_pattern_anchor_idx) == (0.5763, 0.57328, None)
    plain = CrossFibLimit(0.58105, "parent_bos_threshold", ref_anchor_price=0.58105)
    assert plain.localized(100) is plain                            # neither a break nor a move: itself


@pytest.mark.parametrize("kw, why", [
    (dict(source="prev_sub_bos_threshold"), "moves on a fixed source"),
    (dict(price_at_start=None), "without moves, or moves without it"),
    (dict(price_moves=()), "without moves, or moves without it"),
    (dict(price_moves=((110, 0.575), (110, 0.5763))), "not strictly increasing"),
    (dict(price_moves=((120, 0.575), (110, 0.5763))), "not strictly increasing"),
    (dict(price=0.575), "is not its last move"),
])
def test_the_new_asserts(kw, why):
    with pytest.raises(AssertionError, match=why):
        dataclasses.replace(_SUB6, **kw)


# =====================================================================================================================
# 2a. The pre-established phase reads `P` LIVE (the target is not established — never fixed)
# =====================================================================================================================

def _pre_est(lim, sd, updates):
    """`test_plan_i_s3`'s `_pre_established` with a given limit and CTS_THRESHOLD_UPDATED moments: cycle 0 (CTS 1.2)
    established at 20, confirmed at 25 → cycle 1 pre-established; the dip at 27 = the prospective BOS_1; the own gap
    29 (formed at 30). Returns the tracker and, per update, whether cycle 1's pre-established cross exists after it."""
    t = FibTracker(FibTrackerConfig(fill_threshold=0.70), cross_fib_limits={0: lim})
    df = _df(40, [_gap(15, sd, 1.10, 1.00), _gap(29, sd, 1.40, 1.30)])
    if sd == 1:
        df.at[27, "l"] = 1.45
    else:
        df.at[27, "h"] = _m(1.45, -1)
    _q(t.on_cts_established, _est(20, 1.2, 0, sd), df, bos_idx=10, bos_price=_m(0.9, sd))
    _q(t.on_cts_confirmed, _conf(25, 1.2, 0, sd, 20))
    seen = []
    for u in updates:
        _q(t.on_cts_threshold_updated, _ev("CTS_THRESHOLD_UPDATED", u, 1.52, 0, sd), df)
        latest = t._get_latest_cross(0, 1)
        seen.append(latest is not None and latest[1].active and latest[1].meta["cross_start_cycle"] == 0)
    return t, seen


@pytest.mark.parametrize("sd", [1, -1])
@pytest.mark.parametrize("move_at, want", [
    (None, [False, False]),         # P 1.1 fixed: CTS_0 1.2 is beyond -> M = 0, no cross
    (28, [True, True]),             # P -> 1.2 (CTS_0 AT P, not beyond) before the first update: the cross at 30
    (30, [True, True]),             # known AT the update's moment (inclusive)
    (31, [False, True]),            # after the first update: no cross at 30, the cross starts LATE at 33
    (34, [False, False]),           # after both
])
def test_a_pre_established_cross_reads_the_limit_as_of_each_update(sd, move_at, want):
    lim = _live(1.1, [] if move_at is None else [(move_at, 1.2)], sd)
    _t, seen = _pre_est(lim, sd, updates=(30, 33))
    assert seen == want


# =====================================================================================================================
# 2b. The permission is FIXED at the establishment
# =====================================================================================================================

@pytest.mark.parametrize("sd", [1, -1])
@pytest.mark.parametrize("move_at, want", [
    (58, "cross 0->2"),             # P 1.30 -> 1.35 before cycle 2's establishment: CTS_1 AT P -> M = 2, allowed
    (60, "cross 0->2"),             # AT the establishment moment (inclusive)
    (62, "single cross_fib_limit"),  # after it: M = 1 at 60 -> the single, fixed
])
def test_the_permission_is_read_at_the_establishment_and_fixed(sd, move_at, want):
    t, df = _run_cycles(_live(1.30, [(move_at, 1.35)], sd), sd)
    assert _kind(t, 1) == "cross 0->1"                              # P 1.30 at 40: CTS_1 1.35 beyond -> M = 1
    assert _kind(t, 2) == want
    assert t._cross_permission == {(0, 1): True, (0, 2): want.startswith("cross")}
    # a later CTS_UPDATED of cycle 2 (64, 1.57) — after every move — keeps the establishment's answer
    _q(t.on_cts_updated, _ev("CTS_UPDATED", 64, 1.57, 2, sd, via=CTS_UPDATED_RAW_VIA), df)
    assert _kind(t, 2) == want
    if want.startswith("single"):
        s = t._fibs[(0, 2)]
        assert (s.cts_idx, s.meta["via"], s.meta["activated_at"]) == (64, "cross_fib_limit", 60)   # extended
        assert t._get_latest_cross(0, 2) is None
    else:
        assert t._get_latest_cross(0, 2)[1].cts_idx == 64 and (0, 2) not in t._fibs
    # the exported M reads P as of the end (cap None: the last P 1.35 -> CTS_2 is the first beyond): M = 2 even on
    # the cycle established single — the consequence PLAN_M §7.2 states
    assert t._cross_fib_limit_meta(0, None)["cross_fib_limit_cycle"] == 2


@pytest.mark.parametrize("sd", [1, -1])
def test_a_cycle_established_as_a_cross_keeps_it(sd):
    """The fix's TRUE side. A move only ever goes further in production (CL-2; landing 2's producer asserts it), and
    then a fixed True never disagrees with the live answer — so this pin uses a NARROWING move (unreachable, but
    accepted by `CrossFibLimit`): P 1.30 → 1.35 at 58 (cycle 2 allowed at 60) → 1.20 at 63 (live at 64: CTS_1 1.35
    beyond → `M` = 1, refused). The cycle keeps its cross; the structure side keeps applying (landing review B, F2)."""
    lim = _live(1.30, [(58, 1.35), (63, 1.20)], sd)
    t, df = _run_cycles(lim, sd)
    assert _kind(t, 2) == "cross 0->2" and t._cross_permission[(0, 2)] is True
    assert not t._limit_reaches(0, 2, 64)                           # the live answer at the update: refused
    _q(t.on_cts_updated, _ev("CTS_UPDATED", 64, 1.57, 2, sd, via=CTS_UPDATED_RAW_VIA), df)
    assert _kind(t, 2) == "cross 0->2" and t._get_latest_cross(0, 2)[1].cts_idx == 64
    assert t._structure_reference_limit(0, 2, 64) is lim


# =====================================================================================================================
# 2c. The fix lives in the tracker's state: an MS snapshot restore undoes it; a second fix raises
# =====================================================================================================================

@pytest.mark.parametrize("sd", [1, -1])
def test_a_snapshot_restore_undoes_the_fix_and_a_second_fix_raises(sd):
    t, df = _run_cycles(_live(1.30, [(62, 1.35)], sd), sd, upto=1)
    _q(t.on_cts_confirmed, _conf(45, 1.35, 1, sd, 40))
    snap = copy.deepcopy(t.__dict__)                                # MS's `_snapshot_state` (the tracker's __dict__)
    est = lambda at: _q(t.on_cts_established, _est(at, 1.55, 2, sd), df, bos_idx=50, bos_price=_m(1.40, sd))  # noqa
    est(60)
    assert _kind(t, 2) == "single cross_fib_limit" and t._cross_permission[(0, 2)] is False
    t.__dict__.clear()                                              # MS's `_restore_state`
    t.__dict__.update(copy.deepcopy(snap))
    assert (0, 2) not in t._cross_permission
    est(63)                                                         # re-established after the move: the NEW answer
    assert _kind(t, 2) == "cross 0->2" and t._cross_permission[(0, 2)] is True
    with pytest.raises(CrossFibLimitError, match="sid=0 cycle=2: cross permission fixed twice"):
        est(64)


@pytest.mark.parametrize("sd", [1, -1])
def test_a_refusal_over_an_active_pre_established_cross_raises(sd):
    """Landing review A NOTE-8: a limit that moves NEARER (unreachable — `P` moves only further, CL-2) could refuse at
    the establishment a cycle whose pre-established cross is active; the tracker raises instead of leaving the cross
    active beside a fixed single. P 1.2 (CTS_0 AT P: allowed) → the cross at 30; P → 1.1 at 31 (CTS_0 beyond)."""
    t, seen = _pre_est(_live(1.2, [(31, 1.1)], sd), sd, updates=(30,))
    assert seen == [True]
    df = _df(40, [_gap(15, sd, 1.10, 1.00), _gap(29, sd, 1.40, 1.30)])
    with pytest.raises(CrossFibLimitError, match="sid=0 cycle=1: refused at its establishment while its pre-establ"):
        _q(t.on_cts_established, _est(36, 1.55, 1, sd), df, bos_idx=27, bos_price=_m(1.45, sd))


# =====================================================================================================================
# 2d. `_structure_reference_limit`: the fixed answer, else live at `at`
# =====================================================================================================================

@pytest.mark.parametrize("sd", [1, -1])
def test_the_structure_reference_takes_the_fixed_answer_else_the_live_one(sd):
    lim = _live(1.30, [(62, 1.35)], sd)
    t, _df_ = _run_cycles(lim, sd)                                  # cycle 2 established single at 60 (P 1.30)
    srl = t._structure_reference_limit
    assert [srl(0, 2, at) for at in (61, 62, 100, None)] == [None] * 4   # fixed: live would give `lim` from 62
    assert [srl(0, 1, at) for at in (41, None)] == [lim, lim]            # cycle 1 fixed allowed
    assert srl(0, 0, 25) is lim                                          # cycle 0 always, with a P
    # a pre-established target (never established): live at `at`; None = the object's last P
    pre = _live(1.1, [(31, 1.2)], sd)
    tp, _seen = _pre_est(pre, sd, updates=(30,))
    assert (0, 1) not in tp._cross_permission
    assert [tp._structure_reference_limit(0, 1, at) for at in (30, 31, None)] == [None, pre, pre]
    assert tp._structure_reference_limit(0, 0, 21) is pre               # P(21) = 1.1: CTS_0 beyond -> M = 0 >= 0
    # ... and `ic_reference` hands its reading moment on: a cross 0 -> 1 of the unfixed target — CTS_0 (PLAN_O: its
    # target's previous CTS, no bound while cycle 1 is not established) alone at 30 (no structure side), against the
    # structure side from 31 (`cross_limit_anchor` 1.1, further opposite)
    probe = _probe(1, sd, start=0)
    df = _df(40, [])
    assert tp.ic_reference(probe, df, evaluated_at=30) == ICReference(_m(1.2, sd), "prev_cycle_cts")
    assert tp.ic_reference(probe, df, evaluated_at=31) == ICReference(_m(1.1, sd), "cross_limit_anchor")


# =====================================================================================================================
# 2e. The watch reads `P` as of the reading moment (CL-7)
# =====================================================================================================================

@pytest.mark.parametrize("sd", [1, -1])
def test_a_break_of_an_earlier_p_stops_counting_once_p_moved_past_it(sd):
    """`_watch_frame(between=True)`: a double_maru at 12–13 closing at 1.100 / 1.115 — past P 1.09, short of 1.12.
    P 1.09 moved to 1.12 at 20: the break counts from its confirmation (13) until the move, never after."""
    df = _watch_frame(sd, between=True)
    t, _ = _tracker_with(_live(1.09, [(20, 1.12)], sd, anchor=1.05), sd, upto=0, df=df)
    ref = lambda at: t.ic_reference(t._fibs[(0, 0)], df, evaluated_at=at)  # noqa: E731
    near = ICReference(_m(1.05, sd), "cross_limit_anchor")
    assert ref(12) == near                                                   # not confirmed yet
    assert ref(13) == ref(19) == ICReference(_m(1.05, sd), "cross_limit_anchor", 12, 13)
    assert ref(20) == ref(50) == ref(None) == near                           # P 1.12: not broken


@pytest.mark.parametrize("sd", [1, -1])
def test_a_break_of_the_last_p_counts_on_both_sides_of_the_move(sd):
    df = _watch_frame(sd)                                                    # closes 1.135 / 1.155: past 1.12
    t, _ = _tracker_with(_live(1.09, [(20, 1.12)], sd, anchor=1.05), sd, upto=0, df=df)
    ref = lambda at: t.ic_reference(t._fibs[(0, 0)], df, evaluated_at=at)  # noqa: E731
    brk = ICReference(_m(1.05, sd), "cross_limit_anchor", 12, 13)
    assert ref(12) == ICReference(_m(1.05, sd), "cross_limit_anchor")
    assert ref(13) == ref(20) == ref(None) == brk


# =====================================================================================================================
# 2f. The export = `P` as of the structure's END (CL-6 (a)) + the two keys when it moved, in place
# =====================================================================================================================

def _exported(sd, cap):
    t, df = _run_cycles(_live(1.30, [(62, 1.35), (70, 1.36)], sd), sd)
    t._finalize_lifecycle_fields(lifecycle_cap=cap)
    _q(t.stamp_ic_references, df, cap)
    return t._fibs[(0, 2)].meta


@pytest.mark.parametrize("sd", [1, -1])
def test_the_export_reads_p_as_of_the_structures_end(sd):
    before = _exported(sd, 61)                                      # the sub ended before any move
    assert (before["cross_fib_limit_price"], before["cross_fib_limit_cycle"]) == (_m(1.30, sd), 1)
    assert "cross_fib_limit_price_at_start" not in before and "cross_fib_limit_move_count" not in before
    one = _exported(sd, 65)
    assert (one["cross_fib_limit_price"], one["cross_fib_limit_cycle"], one["cross_fib_limit_price_at_start"],
            one["cross_fib_limit_move_count"]) == (_m(1.35, sd), 2, _m(1.30, sd), 1)
    every = _exported(sd, None)                                     # an open sub: every move
    assert (every["cross_fib_limit_price"], every["cross_fib_limit_move_count"]) == (_m(1.36, sd), 2)
    assert list(one) == [
        "activated_at", "via",
        # cycle 2's reference: CTS_1 + the pattern that established cycle 2 (known at 60, before either cap)
        "ic_ref_price", "ic_ref_source", "ic_ref_break_pattern_anchor_idx", "ic_ref_break_confirmed_idx",
        "cross_fib_limit_price", "cross_fib_limit_source", "cross_fib_limit_cycle", "cross_fib_limit_from_sub_id",
        "cross_fib_limit_price_at_start", "cross_fib_limit_move_count"]
    assert list(before) == list(one)[:-2]


# =====================================================================================================================
# 2g. A moving limit needs the reading moment
# =====================================================================================================================

def test_a_moving_limit_read_outside_a_handler_raises_a_constant_one_does_not():
    t = FibTracker(FibTrackerConfig(fill_threshold=0.70), cross_fib_limits={0: _live(1.30, [(62, 1.35)], 1)})
    with pytest.raises(CrossFibLimitError, match="sid=0 cycle=1: a moving cross_fib_limit .* no moment"):
        t._cross_allowed(0, 1)
    t._cross_permission[(0, 1)] = True                              # a fixed answer needs no moment
    assert t._cross_allowed(0, 1) is True
    const = FibTracker(FibTrackerConfig(fill_threshold=0.70),
                       cross_fib_limits={0: CrossFibLimit(1.30, "parent_bos_threshold", ref_anchor_price=1.30)})
    assert const._cross_allowed(0, 1) is True                       # no anchor beyond yet: M None


@pytest.mark.parametrize("sd", [1, -1])
def test_m_at_a_moment(sd):
    t, _ = _run_cycles(_live(1.30, [(62, 1.35)], sd), sd)
    assert (t.cross_fib_limit_cycle(0, 61), t.cross_fib_limit_cycle(0, 62), t.cross_fib_limit_cycle(0)) == (1, 2, 2)


# =====================================================================================================================
# 2h. The hover
# =====================================================================================================================

def test_the_hover_qualifies_a_moved_price():
    meta = {"cross_fib_limit_price": 0.5763, "cross_fib_limit_source": "parent_bos_threshold",
            "cross_fib_limit_cycle": 4, "cross_fib_limit_from_sub_id": None,
            "cross_fib_limit_price_at_start": 0.57328, "cross_fib_limit_move_count": 7}
    assert fib_hover_limit_text(SimpleNamespace(meta=meta)) == (
        "limit_price=0.57630 (moved 7× from 0.57328) | parent_bos_threshold | cycle_limit=4")
    unmoved = {k: v for k, v in meta.items() if k not in ("cross_fib_limit_price_at_start",
                                                          "cross_fib_limit_move_count")}
    assert fib_hover_limit_text(SimpleNamespace(meta=unmoved)) == (
        "limit_price=0.57630 | parent_bos_threshold | cycle_limit=4")


# =====================================================================================================================
# 3. The REAL collaborator: a real MarketStructure run + the drawn tracker, on one moving limit
# =====================================================================================================================

def _real(sd, lim, embedded):  # noqa: F811
    """`test_plan_i_s5_ms_reads_the_fib._rows(sd)` (cycles 0–3; cycle 0 CTS 0.6122 at anchor 5, confirmed at 7;
    cycle 1 established at 10, CTS 0.6167) through a real MS run (its embedded tracker) and the downstream."""
    embedded.clear()
    res = _q(se.compute_bounded_structure, _prepare_df(_rows(sd)), 0, sd, cross_fib_limit=lim)
    ms_t, = embedded
    out = _q(_run_downstream_pipeline, res.df, res.events, sd, wvmi="off", cross_fib_limits={0: lim})
    return res, ms_t, out["fib_tracker"]


def _cycle_1(t):
    latest = t._get_latest_cross(0, 1)
    if latest is not None and latest[1].active:
        return "cross", latest[1]
    return "single", t._fibs[(0, 1)]


@pytest.mark.parametrize("sd", [1, -1])
def test_the_real_run_reads_the_move_at_cycle_1s_establishment(sd, embedded):  # noqa: F811
    """`P` 0.6121 at the start (cycle 0's CTS 0.6122 strictly beyond → `M` = 0: cycle 1 single); ONE move to exactly
    0.6122 (not beyond → `M` = 1: cycle 1 may cross). The anchor behind `P` 0.6050 (the structure-side IC reference).
    Its `known_at` from the run's own events: cycle 1's establishment moment (10)."""
    m = lambda p: p if sd == 1 else round(1.2200 - p, 4)  # noqa: E731  (`_mirror`'s pivot: 0.6122 <-> 0.6078)
    base, _ms, _ds = _real(sd, unreached(sd), embedded)
    conf0 = next(e for e in base.events if e.type == "CTS_CONFIRMED" and e.meta["cycle_id"] == 0)
    est1 = next(e for e in base.events if e.type == "CTS_ESTABLISHED" and e.meta["cycle_id"] == 1)
    assert (float(conf0.price), ef.event_moment(est1)) == (m(0.6122), 10)

    def lim(known_at):
        return CrossFibLimit(m(0.6122), "parent_bos_threshold", ref_anchor_price=m(0.6050),
                             price_at_start=m(0.6121), price_moves=((known_at, m(0.6122)),))

    # (a) the move known AT cycle 1's establishment: the cross 0 -> 1 — in BOTH trackers
    res, ms_t, ds_t = _real(sd, lim(10), embedded)
    for t in (ms_t, ds_t):
        kind, rec = _cycle_1(t)
        assert kind == "cross" and rec.meta["cross_start_cycle"] == 0 and t._cross_permission[(0, 1)] is True
    assert [id(e) for e in ms_t._handled] == [id(e) for e in ds_t._handled]
    assert ms_t.representative_fib(0, 1) is _cycle_1(ms_t)[1]
    assert ms_t.ic_reference(_cycle_1(ms_t)[1], res.df, evaluated_at=12).source == "cross_limit_anchor"
    rec = _cycle_1(ds_t)[1]
    assert (rec.meta["ic_ref_price"], rec.meta["ic_ref_source"]) == (m(0.6050), "cross_limit_anchor")
    assert (rec.meta["cross_fib_limit_price"], rec.meta["cross_fib_limit_cycle"],
            rec.meta["cross_fib_limit_price_at_start"], rec.meta["cross_fib_limit_move_count"]) == (
        m(0.6122), 1, m(0.6121), 1)

    # (b) the move one candle AFTER: cycle 1 established single (`via: "cross_fib_limit"`), fixed for its life —
    # its IC reference is CTS_0 only (the live answer at the end would add the structure side 0.6050 and win)
    res, ms_t, ds_t = _real(sd, lim(11), embedded)
    for t in (ms_t, ds_t):
        kind, rec = _cycle_1(t)
        assert kind == "single" and rec.meta["via"] == "cross_fib_limit" and t._cross_permission[(0, 1)] is False
        assert t._get_latest_cross(0, 1) is None
    assert [id(e) for e in ms_t._handled] == [id(e) for e in ds_t._handled]
    assert ms_t.representative_fib(0, 1) is _cycle_1(ms_t)[1]
    assert ms_t.ic_reference(_cycle_1(ms_t)[1], res.df, evaluated_at=12).source == "prev_cycle_cts"
    rec = _cycle_1(ds_t)[1]
    assert (rec.meta["ic_ref_price"], rec.meta["ic_ref_source"]) == (m(0.6122), "prev_cycle_cts")
    # the export as of the end (an open structure: every move) — `M` 1 at the last P although cycle 1 was refused
    # at its establishment (the consequence PLAN_M §7.2 states)
    assert (rec.meta["cross_fib_limit_price"], rec.meta["cross_fib_limit_cycle"],
            rec.meta["cross_fib_limit_move_count"]) == (m(0.6122), 1, 1)


def test_the_projection_hands_the_drawn_tracker_the_slice_local_moves(monkeypatch, geometry, m15_df):  # noqa: F811
    """`entity_df_mutation.render_sub_projection` (line 1300; landing review B, F1): the drawn tracker holds the moves
    shifted by `slice_begin` (55 − 50 = 5), and the export as of the cap counts a move known 2 candles before it
    (entity-absolute 83 = slice-local 78 ≤ the cap 80; unshifted it would read 83 > 80 and go unexported)."""
    import engine_v2.pipeline.orchestrator as orch
    from engine_v2.tests import test_render_sub_projection as RSP
    made = []

    class _Rec(FibTracker):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    monkeypatch.setattr(orch, "FibTracker", _Rec)
    pool, sub = geometry
    RSP._two_lens_records(pool, sub, m15_df)
    RSP._set_lifecycle(sub, RSP._START, RSP._END, "parent_end")
    _bounded, slice_begin = sub.geometry
    assert (slice_begin, RSP._END) == (5, 85)
    lim = CrossFibLimit(20.0, "parent_bos_threshold", ref_anchor_price=math.inf, price_at_start=10.0,
                        price_moves=((RSP._END - 2, 20.0),))           # never reached: the limit's keys only
    sub.cross_fib_limit = lim
    lens_dfs = RSP._lens_dfs(m15_df)
    _q(RSP._render, sub, m15_df, lens_dfs)
    t, = made
    assert t._limits == {0: dataclasses.replace(lim, price_moves=((RSP._END - 2 - 5, 20.0),))}
    fibs = [f for d in lens_dfs.values() for f in d.attrs.get("fib_states", [])]   # the mirrored (exported) records
    assert fibs and all(
        (f.meta["cross_fib_limit_price"], f.meta["cross_fib_limit_price_at_start"],
         f.meta["cross_fib_limit_move_count"]) == (20.0, 10.0, 1) for f in fibs)


def test_the_geometry_hands_its_tracker_the_slice_local_moves(embedded):  # noqa: F811
    """`entity_df_mutation._build_geometry` (line 1167): the pool's limit is entity-absolute; the geometry MS runs on
    the slice [start − 50, cap] — its embedded tracker holds every move shifted by `slice_begin` (60 − 50 = 10)."""
    from engine_v2.multitf import entity_df_mutation as edm
    from engine_v2.tests.test_render_sub_projection import _make_flat_prefix, _make_reversing_data
    df = _prepare_df(_make_flat_prefix(60) + _make_reversing_data())
    lim = CrossFibLimit(math.inf, "parent_bos_threshold", ref_anchor_price=math.inf, price_at_start=2.0,
                        price_moves=((100, 3.0), (105, math.inf)))           # never reached: a no-op limit
    embedded.clear()
    _q(edm._build_geometry, df, sd=1, start_abs=60, run_cap_abs=len(df) - 1, bos0_inner=None,
       timeframe="M15", cross_fib_limit=lim)
    t, = embedded
    assert t._limits == {0: dataclasses.replace(lim, price_moves=((90, 3.0), (95, math.inf)))}
