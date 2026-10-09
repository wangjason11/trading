"""PLAN_I S4 (R6-5: Q4, Q5, Q6, I2, I5, Q8): the H1 main on the ONE FibTracker logic — hand-built main streams through
the tracker + finalize (the S3 pin file's flat frame, its own mirror at the pivot 3.0, so sd = -1 is the exact
mirror). The reference window's main shows the same shapes (§9.5 c): (0,0) the new single, (1,0) collapsed, (1,1) the
pre-established cross clamped by sid 1's floor.

- Q6: main's cycle 0 gets the single (the h1 mode drew none for sid 0 cycle 0).
- I1 / Q1: sid 0 is started by start trading → no reference price → every cycle >= 1 a labelled single.
- I2 + R2-5: main gets the pre-established cross (within its limit); created before the structure's start (sid 1 is
  established retroactively, before sid 0's reversal applies), its lifecycle start is CLAMPED to the structure floor;
  the previous cycle's single, ended by it (Option A) before the floor, collapses.
"""
from __future__ import annotations

import contextlib
import io

import pytest

from engine_v2.structure.market_structure import StructureEvent
from engine_v2.tests._event_factory import make_cts_established
from engine_v2.tests.test_plan_i_s3_cross_fib_limit_applied import _df, _gap, _m
from engine_v2.tests._cross_fib_limits import limit
from engine_v2.zones.cross_cycle_fib import CrossFibLimit
from engine_v2.zones.fib_tracker import FibTracker, FibTrackerConfig


def _q(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _ev(etype, idx, price, sid, cyc, sd, **meta):
    return StructureEvent(idx=idx, category="STRUCTURE", type=etype, price=None if price is None else _m(price, sd),
                          meta=dict(structure_id=sid, cycle_id=cyc, struct_direction=sd, **meta))


def _est(moment, price, sid, cyc, sd):
    return make_cts_established(cts_anchor_idx=moment, confirmed_at=moment, price=_m(price, sd), structure_id=sid,
                                cycle_id=cyc, struct_direction=sd)


def _main_tracker(p1, sd):
    """Main's limits as `compute_main_cross_fib_limits` shapes them: sid 0 start trading (no P), sid 1 the reversal's
    `bos_frozen` (`p1`)."""
    return FibTracker(FibTrackerConfig(fill_threshold=0.70), cross_fib_limits={
        0: CrossFibLimit(None, "none", started_by="start_trading"),
        1: limit(_m(p1, sd), "reversal_bos_frozen", started_by="reversal")})


@pytest.mark.parametrize("sd", [1, -1])
def test_main_sid0_gets_the_cycle0_single_and_labelled_singles_after_it(sd):
    t = _main_tracker(9.0, sd)
    df = _df(80, [_gap(15, sd, 1.10, 1.00), _gap(35, sd, 1.30, 1.20)])
    c0 = _q(t.on_cts_established, _est(20, 1.15, 0, 0, sd), df, 10, _m(1.0, sd))
    assert c0 is t._fibs[(0, 0)] and c0.active and c0.meta == {"activated_at": 20}  # Q6
    _q(t.on_cts_confirmed, _ev("CTS_CONFIRMED", 25, 1.15, 0, 0, sd, cts_anchor_idx=20))
    c1 = _q(t.on_cts_established, _est(40, 1.35, 0, 1, sd), df, 30, _m(1.2, sd))
    assert c1 is t._fibs[(0, 1)] and c1.meta["via"] == "cross_fib_limit"          # sid 0: no P → no cross
    assert t._get_latest_cross(0, 1) is None


@pytest.mark.parametrize("sd", [1, -1])
def test_main_sid1_pre_established_cross_is_clamped_by_the_structure_floor(sd):
    """sid 0: cycle 0 (EST 20, confirmed 25), reversed at 90. sid 1 is established retroactively — cycle 0 EST 70,
    confirmed 75 — and a CTS_THRESHOLD_UPDATED at 80 (a dip at 77 = the prospective BOS_1, the own gap 78) creates
    sid 1 cycle 1's pre-established cross (within P); cycle 1 establishes at 100. Finalized with the events, sid 1's
    floor is the reversal 90: the cross starts at 90 (created 80), sid 1 cycle 0's single (activated 70, ended 80 by
    Option A) collapses; sid 0 cycle 0's single ends at the reversal."""
    t = _main_tracker(9.0, sd)
    df = _df(120, [_gap(15, sd, 1.10, 1.00), _gap(65, sd, 1.30, 1.20), _gap(78, sd, 1.40, 1.30)])
    if sd == 1:
        df.at[77, "l"] = 1.45
    else:
        df.at[77, "h"] = _m(1.45, -1)
    events = []

    def feed(e, *args):
        events.append(e)
        handler = {"CTS_ESTABLISHED": t.on_cts_established, "CTS_CONFIRMED": t.on_cts_confirmed,
                   "CTS_THRESHOLD_UPDATED": t.on_cts_threshold_updated}.get(e.type)
        if handler is not None:
            _q(handler, e, *args)

    feed(_est(20, 1.15, 0, 0, sd), df, 10, _m(1.0, sd))
    feed(_ev("CTS_CONFIRMED", 25, 1.15, 0, 0, sd, cts_anchor_idx=20))
    feed(_est(70, 1.35, 1, 0, sd), df, 60, _m(1.2, sd))
    feed(_ev("CTS_CONFIRMED", 75, 1.35, 1, 0, sd, cts_anchor_idx=70))
    feed(_ev("CTS_THRESHOLD_UPDATED", 80, 1.52, 1, 0, sd), df)
    assert t._get_latest_cross(1, 1)[1].meta["activated_at"] == 80               # I2: main's pre-established cross
    feed(_ev("STATE_CHANGED", 90, None, 0, 0, sd, to="reversal", bos_frozen=_m(1.0, sd)))
    feed(_est(100, 1.55, 1, 1, sd), df, 77, _m(1.45, sd))
    t.set_reversal_terminals({1: 90})
    t._finalize_lifecycle_fields(events=events)
    got = {k: (f.start_idx, f.end_idx, f.end_reason, f.status) for k, f in t._fibs.items()}
    assert got == {
        (0, 0): (20, 90, "reversal", "ended"),
        (1, 0): (None, 80, "new_cycle", "inactive"),                             # Option A end before the floor
        (1, 1, "cross", 0): (90, None, None, "active"),                          # R2-5: clamped to the floor 90
    }
