"""PLAN_I S5b (R9-3, §9.2 F19): a cycle's lifecycle end CAPS its fib's end.

The stream ends the structure's LAST cycle that has a fib (`_current_cycle`), set-if-absent: when the cycle after it
gets no fib, the earlier fib used to stay open until a later cycle's fib started (or the reversal) — it outlived its
own cycle, while its POI ended at the cycle table's end. At finalize the fib's end is now min(the stream terminal, the
cycle-table end `compute_cycle_lifecycle` — the end KL / POI take), with the table's reason when the table is STRICTLY
earlier; a tie keeps the stream's reason (`new_cycle` / `reversal`), Option A's earlier end stays, the collapse rule
applies to the capped end. 0 on the reference window (the S5b shadow: 0 / 27 fib cycles later than the table; 17
ties, 4 earlier on the stream — 3 Option A, 1 the sub floor's clamp — 6 without a stream terminal). A real instance: NZD_USD M15 from 2025-08-05 (`plans/plan_i_inputs/f19_detail.py M15
14715 333 20 1`) — cycle 0's locked single ends at 38 (08-05 13:15, cycle 1's establishment) instead of 203 (08-07
06:30, cycle 2's activation).

Fixture: the S3 pin file's flat frame (sd = -1 is the exact mirror of sd = +1). Cycle 0 (own gap 15) established at
20, confirmed at 25; cycle 1 established at 40 with NO gap in its own [30, 40] → no fib; confirmed at 45; cycle 2
(own gap 55) established at 60 → its single activates at 60 and the stream ends cycle 0 there.
"""
from __future__ import annotations

import pytest

from engine_v2.structure.market_structure import StructureEvent
from engine_v2.tests.test_plan_i_s3_cross_fib_limit_applied import (
    _CYCLES, _conf, _df, _est, _ev, _gap, _kind, _lim, _m, _pre_established, _q, _three_cycles, _tracker,
)


def _f19(p, sd, upto=2):
    """The F19 shape up to cycle `upto`; returns (tracker, the fed events)."""
    t = _tracker(_lim(p, sd))
    df = _df(80, [_gap(15, sd, 1.10, 1.00), _gap(55, sd, 1.50, 1.40)])
    events = []
    steps = [(20, 1.15, 0, 10, 1.00, 25), (40, 1.35, 1, 30, 1.20, 45), (60, 1.55, 2, 50, 1.40, None)]
    for moment, price, cyc, bos, bos_p, conf in steps[:upto + 1]:
        est = _est(moment, price, cyc, sd)
        events.append(est)
        _q(t.on_cts_established, est, df, bos_idx=bos, bos_price=_m(bos_p, sd))
        if conf is not None:
            ev = _conf(conf, price, cyc, sd, moment)
            events.append(ev)
            _q(t.on_cts_confirmed, ev)
    return t, events


def _life(t, key):
    f = t._fibs[key]
    return f.start_idx, f.end_idx, f.end_reason, f.status


@pytest.mark.parametrize("p, via", [(None, "cross_fib_limit"), (9.0, "cross_failed")])
@pytest.mark.parametrize("sd", [1, -1])
def test_f19_the_cycle_end_caps_a_fib_whose_next_cycle_got_no_fib(sd, p, via):
    """Without `P` (cycle 2 a past-limit single) and within it (cycle 1 dead → cycle 2's cross fails → its single):
    the stream ends cycle 0 at 60 (cycle 2's activation); the table caps it at 40 (cycle 1's establishment)."""
    t, events = _f19(p, sd)
    assert _kind(t, 1) is None and _kind(t, 2) == f"single {via}"
    assert t._terminal[(0, 0)] == (60, "new_cycle")                     # the stream: the LAST cycle with a fib
    _q(t._finalize_lifecycle_fields, events)
    assert t._terminal[(0, 0)] == (40, "next_cycle")
    assert _life(t, (0, 0)) == (20, 40, "next_cycle", "ended")
    assert t._fibs[(0, 0)].locked                                         # the lock is untouched
    assert _life(t, (0, 2)) == (60, None, None, "active")


@pytest.mark.parametrize("sd", [1, -1])
def test_a_tie_keeps_the_stream_reason(sd):
    """Adjacent cycles each with a fib: the stream end (`new_cycle` at the next activation) == the table end
    (`next_cycle`) → the stream's reason stays (a `<=` would rename every such end — 12 on the reference window)."""
    t = _three_cycles(None, sd)
    events = [_est(cts, cts_p, c, sd) for c, (_b, _bp, cts, cts_p, _cf, _g) in enumerate(_CYCLES)]
    _q(t._finalize_lifecycle_fields, events)
    assert _life(t, (0, 0)) == (20, 40, "new_cycle", "ended")
    assert _life(t, (0, 1)) == (40, 60, "new_cycle", "ended")


@pytest.mark.parametrize("sd", [1, -1])
def test_option_a_earlier_end_is_kept(sd):
    """The pre-established cross for cycle 1 (created at 30, before cycle 1's establishment at 35) ended cycle 0 at
    30: earlier than the table end (35) → kept (min, not a replacement)."""
    t = _pre_established(9.0, sd)
    df = _df(40, [_gap(15, sd, 1.10, 1.00), _gap(29, sd, 1.40, 1.30)])
    _q(t.on_cts_established, _est(35, 1.55, 1, sd), df, bos_idx=27, bos_price=_m(1.45, sd))
    _q(t._finalize_lifecycle_fields, [_est(20, 1.2, 0, sd), _est(35, 1.55, 1, sd)])
    assert _life(t, (0, 0)) == (20, 30, "new_cycle", "ended")


@pytest.mark.parametrize("sd", [1, -1])
def test_f19_through_the_reversal_terminal(sd):
    """No later cycle gets a fib: the reversal terminal (stamped on every cycle of the ended sid, set-if-absent) used to
    end cycle 0 at the reversal (70); the table caps it at cycle 1's establishment (40)."""
    t, events = _f19(None, sd, upto=1)
    assert (0, 0) not in t._terminal
    events.append(StructureEvent(idx=70, category="STRUCTURE", type="STATE_CHANGED", price=None,
                                 meta=dict(structure_id=0, to="reversal")))
    t.set_reversal_terminals({1: 70})
    assert t._terminal[(0, 0)] == (70, "reversal")
    _q(t._finalize_lifecycle_fields, events)
    assert _life(t, (0, 0)) == (20, 40, "next_cycle", "ended")


@pytest.mark.parametrize("sd", [1, -1])
def test_the_collapse_rule_applies_to_the_capped_end(sd):
    """A floor at 45 clamps cycle 0's start to 45 and cycle 1's table start to 45 → cycle 0's table end 45: the capped
    end makes the fib collapse (start None, `inactive`) — with the stream end (60) it read `ended` on [45, 60]."""
    t, events = _f19(None, sd)
    _q(t._finalize_lifecycle_fields, events, lifecycle_floor=45)
    assert _life(t, (0, 0)) == (None, 45, "next_cycle", "inactive")


def test_finalize_is_idempotent_with_the_cap():
    t, events = _f19(None, 1)
    _q(t._finalize_lifecycle_fields, events)
    first = {k: _life(t, k) for k in t._fibs}
    _q(t._finalize_lifecycle_fields, events)
    assert {k: _life(t, k) for k in t._fibs} == first
