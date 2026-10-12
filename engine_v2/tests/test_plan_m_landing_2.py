"""PLAN_M landing 2 (§7.3): a COUNTER sub's cross-fib limit = its first record's parent cycle's BOS threshold, read
live (`parent_bos_threshold`, CL-0 / CL-5).

- `parent_bos_limit`: `price_at_start` folds every step known by the record's start (inclusive), the later steps
  are the moves, steps at one M15 moment collapse (the last wins), `price` = the last, the anchor = the BOS price;
  each assert; the steps read for (S, C) only;
- `cross_fib_limit_for`: the counter test FIRST, for every trigger type, on the right sid's direction; a confluence
  sub keeps its rule;
- `ParentTables.bos_threshold_steps` (`build_parent_tables`): processing order, the injected LOH, an unmapped step
  stored (asserted only where it is read);
- the R6-2 tripwire compares `price_at_start` / `price_moves`;
- the reference window on its REAL H1 events and subs (S_L2's rows, embedded — a mutant tree has no `artifacts/`;
  the provenance test checks them against the committed save when it is there).

The parent's threshold moves only FURTHER: for a sell parent (its BOS a high) that is higher, which is "beyond" for
its counter (buy) sub. Every price below is a parent sid 0 SELL (−1) / sid 1 BUY (+1) unless stated.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from engine_v2.multitf import cross_fib_limit as cfl
from engine_v2.multitf.cross_fib_limit import NO_LIMIT
from engine_v2.multitf.sub_structure_pool import StructureKey, SubStructurePool
from engine_v2.structure import event_fields as ef
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.tests._event_factory import make_bos_confirmed, make_cts_established
from engine_v2.tests.test_parent_tables import _build
from engine_v2.tests.test_plan_i_s2_cross_fib_limit import (
    PARENT, TF, _bos, _df400, _ev, _geo_sub, _rec, _tables,
)
from engine_v2.zones.cross_cycle_fib import CrossFibLimit

BOS, UPD = "BOS_CONFIRMED", "BOS_THRESHOLD_UPDATED"

# (h1_known_at, m15_known_at, price, type) — LOH = 4h + 3. Parent (0, 1): the BOS 0.5700 at H1 100, moved at 104,
# 110, twice in H1 120 (one M15 moment: the second wins), 130; the cycle ends at 140. Decoys at distinct prices: the
# parent's cycle 0 and sid 1's cycles 0 / 1.
_STEPS = {
    (0, 1): ((100, 403, 0.5700, BOS), (104, 419, 0.5710, UPD), (110, 443, 0.5720, UPD), (120, 483, 0.5730, UPD),
             (120, 483, 0.5735, UPD), (130, 523, 0.5740, UPD)),
    (0, 0): ((50, 203, 0.5600, BOS), (60, 243, 0.5650, UPD)),
    (1, 0): ((80, 323, 0.5500, BOS),),
    (1, 1): ((105, 423, 0.5900, BOS), (115, 463, 0.5850, UPD)),
}
_END = {(0, 0): 100, (0, 1): 140, (1, 0): 105, (1, 1): None}


def _t(steps=None, end=None, parent_sd=None):
    return _tables({0: -1, 1: 1} if parent_sd is None else parent_sd, _STEPS if steps is None else steps,
                   _END if end is None else end)


# =====================================================================================================================
# 1. parent_bos_limit
# =====================================================================================================================

def test_the_start_folds_every_step_known_by_it_and_the_rest_are_the_moves():
    got = cfl.parent_bos_limit(_t(), 0, 1, 1, 419)                   # AT the first move's LOH: folded (inclusive)
    assert got == CrossFibLimit(0.5740, "parent_bos_threshold", ref_anchor_price=0.5700, price_at_start=0.5710,
                                price_moves=((443, 0.5720), (483, 0.5735), (523, 0.5740)))
    assert (got.from_pool_idx, got.break_pattern_anchor_idx, got.break_confirmed_idx) == (None, None, None)
    one_before = cfl.parent_bos_limit(_t(), 0, 1, 1, 418)
    assert (one_before.price_at_start, one_before.price_moves[0]) == (0.5700, (419, 0.5710))   # = the BOS price
    assert len(one_before.price_moves) == 4                          # the two steps of H1 120 counted once
    assert cfl.parent_bos_limit(_t(), 0, 1, 1, 403).price_at_start == 0.5700   # the BOS known AT the start


def test_a_start_after_every_step_is_a_constant_limit_at_the_last_threshold():
    got = cfl.parent_bos_limit(_t(), 0, 1, 1, 700)
    assert got == CrossFibLimit(0.5740, "parent_bos_threshold", ref_anchor_price=0.5700)
    assert (got.price_at_start, got.price_moves) == (None, ())


def test_steps_at_one_m15_moment_collapse_to_the_last():
    got = cfl.parent_bos_limit(_t(), 0, 1, 1, 450)
    assert got.price_moves == ((483, 0.5735), (523, 0.5740))        # not 0.5730
    assert got.price_at(483) == 0.5735


def test_two_hours_sharing_one_m15_moment_collapse_too():
    """The collapse keys on the M15 moment, not the H1 hour: an hour with no M15 candle maps (LOH) to the previous
    candle, so two H1 hours can share one moment (landing 2 review B, MINOR-3)."""
    steps = {(0, 1): ((100, 403, 0.5700, BOS), (104, 419, 0.5710, UPD), (105, 419, 0.5720, UPD))}
    assert cfl.parent_bos_limit(_t(steps, {(0, 1): None}), 0, 1, 1, 403).price_moves == ((419, 0.5720),)


def test_only_the_named_parent_cycle_is_read():
    assert cfl.parent_bos_limit(_t(), 0, 0, 1, 300) == CrossFibLimit(0.5650, "parent_bos_threshold",
                                                                     ref_anchor_price=0.5600)
    sell_under_buy = cfl.parent_bos_limit(_t(), 1, 1, -1, 430)     # sid 1 a BUY: its threshold moves LOWER
    assert sell_under_buy == CrossFibLimit(0.5850, "parent_bos_threshold", ref_anchor_price=0.5900,
                                           price_at_start=0.5900, price_moves=((463, 0.5850),))


@pytest.mark.parametrize("steps, s, end, why", [
    ({(0, 1): ()}, 500, None, "0 BOS_CONFIRMED"),
    ({(0, 1): ((100, 403, 0.57, BOS), (104, 419, 0.571, BOS))}, 500, None, "2 BOS_CONFIRMED"),
    ({(0, 1): ((99, 399, 0.569, UPD), (100, 403, 0.57, BOS))}, 500, None, "first step BOS_THRESHOLD_UPDATED"),
    ({(0, 1): ((100, None, 0.57, BOS),)}, 500, None, "before the M15 frame, or unpriced"),
    ({(0, 1): ((100, 403, None, BOS),)}, 500, None, "before the M15 frame, or unpriced"),
    ({(0, 1): ((100, 403, 0.57, BOS),)}, 402, None, "after the record start 402"),
    ({(0, 1): ((100, 403, 0.57, BOS), (139, 559, 0.571, UPD))}, 500, 139, "at / after the cycle's end 139"),
    ({(0, 1): ((100, 403, 0.57, BOS), (104, 419, 0.569, UPD))}, 500, None, "not strictly further"),
    ({(0, 1): ((100, 403, 0.57, BOS), (104, 419, 0.57, UPD))}, 500, None, "not strictly further"),
])
def test_each_assert(steps, s, end, why):
    with pytest.raises(AssertionError, match=why):
        cfl.parent_bos_limit(_t(steps, {(0, 1): end}), 0, 1, 1, s)


def test_a_step_one_hour_before_the_cycle_end_is_kept():
    steps = {(0, 1): ((100, 403, 0.57, BOS), (138, 555, 0.571, UPD))}
    assert cfl.parent_bos_limit(_t(steps, {(0, 1): 139}), 0, 1, 1, 500).price_moves == ((555, 0.571),)


# =====================================================================================================================
# 2. cross_fib_limit_for: the counter test FIRST, any trigger type, the right sid
# =====================================================================================================================

_COUNTER_LIM = CrossFibLimit(0.5740, "parent_bos_threshold", ref_anchor_price=0.5700, price_at_start=0.5710,
                             price_moves=((443, 0.5720), (483, 0.5735), (523, 0.5740)))


@pytest.mark.parametrize("tt", ["reversal", "first_counter", "subsequent_counter", "subsequent_confluence",
                                "first_confluence"])
def test_a_counter_sub_takes_the_parent_bos_threshold_whatever_its_trigger(tt):
    """A buy sub under sid 0 (a sell): counter — no reversing / previous sub is read (none handed in)."""
    got = cfl.cross_fib_limit_for(tt, (0, 1), 1, 419, h1_events=[], parent_tables=_t(), reversing_sub=None,
                                  prev_sub=None)
    assert got == _COUNTER_LIM


def test_a_confluence_sub_keeps_its_rule():
    kw = dict(h1_events=[], parent_tables=_t(), reversing_sub=None, prev_sub=None)
    assert cfl.cross_fib_limit_for("first_confluence", (0, 0), -1, 300, **kw) == NO_LIMIT
    for tt in ("first_counter", "subsequent_confluence", "subsequent_counter"):
        assert cfl.cross_fib_limit_for(tt, (0, 1), -1, 419, **kw) == NO_LIMIT         # no x−1
    with pytest.raises(AssertionError, match="a reversal needs its reversing sub"):
        cfl.cross_fib_limit_for("reversal", (0, 1), -1, 419, **kw)
    # a confluence reversal: the reversing sub's frozen BOS (the rule PLAN_M leaves to the confluence subs)
    x = _geo_sub(5, 1, 50, [_bos("BOS_CONFIRMED", 280, 0.58, moment=290),
                            _ev("STATE_CHANGED", 300, None, to="reversal", pattern_anchor_idx=297, bos_frozen=0.58,
                                structure_id=0)], n=400)
    x.natural_reversal_idx = 300
    got = cfl.cross_fib_limit_for("reversal", (0, 1), -1, 300, h1_events=[], parent_tables=_t(), reversing_sub=x,
                                  prev_sub=None)
    assert got == CrossFibLimit(0.58, "reversal_bos_frozen", from_pool_idx=5, ref_anchor_price=0.58,
                                break_pattern_anchor_idx=297, break_confirmed_idx=300)


def test_an_unknown_trigger_type_raises_on_a_counter_sub_too():
    with pytest.raises(AssertionError, match="unknown trigger type 'nonsense'"):
        cfl.cross_fib_limit_for("nonsense", (0, 1), 1, 419, h1_events=[], parent_tables=_t(), reversing_sub=None,
                                prev_sub=None)


def test_the_counter_test_reads_the_records_own_parent_sid():
    """sid 0 a SELL, sid 1 a BUY: a buy sub is counter under sid 0 and confluence under sid 1."""
    kw = dict(h1_events=[], parent_tables=_t(), reversing_sub=None, prev_sub=None)
    assert cfl.cross_fib_limit_for("subsequent_counter", (0, 1), 1, 419, **kw).source == "parent_bos_threshold"
    assert cfl.cross_fib_limit_for("subsequent_counter", (1, 1), 1, 430, **kw) == NO_LIMIT
    assert cfl.cross_fib_limit_for("subsequent_counter", (1, 1), -1, 430, **kw).source == "parent_bos_threshold"


# =====================================================================================================================
# 3. ParentTables.bos_threshold_steps
# =====================================================================================================================

def _est(sid, cyc, at, sd):
    return make_cts_established(cts_anchor_idx=at, confirmed_at=at, price=0.6, structure_id=sid, cycle_id=cyc,
                                struct_direction=sd)


def _bc(sid, cyc, at, price, sd):
    return make_bos_confirmed(bos_anchor_idx=at - 3, confirmed_at=at, price=price, structure_id=sid, cycle_id=cyc,
                              struct_direction=sd)


def _upd(sid, cyc, at, price, sd):
    return StructureEvent(idx=at, category="STRUCTURE", type=UPD, price=price,
                          meta={"structure_id": sid, "cycle_id": cyc, "struct_direction": sd})


def _h1_events():
    return [_est(0, 0, 20, -1), _bc(0, 0, 20, 0.570, -1), _upd(0, 0, 25, 0.571, -1), _upd(0, 0, 30, 0.572, -1),
            _est(0, 1, 40, -1), _bc(0, 1, 40, 0.575, -1), _upd(0, 1, 45, 0.576, -1)]


def test_the_steps_are_every_bos_and_threshold_event_per_cycle_in_processing_order():
    t = _build(_h1_events())
    assert t.bos_threshold_steps == {
        (0, 0): ((20, 83, 0.570, BOS), (25, 103, 0.571, UPD), (30, 123, 0.572, UPD)),
        (0, 1): ((40, 163, 0.575, BOS), (45, 183, 0.576, UPD)),
    }
    assert cfl.parent_bos_limit(t, 0, 0, 1, 103) == CrossFibLimit(
        0.572, "parent_bos_threshold", ref_anchor_price=0.570, price_at_start=0.571, price_moves=((123, 0.572),))


def test_an_unmapped_step_is_stored_and_raises_only_where_it_is_read():
    t = _build(_h1_events(), loh=lambda p, _h1, _m15: None if int(p) == 25 else 4 * int(p) + 3)
    assert t.bos_threshold_steps[(0, 0)][1] == (25, None, 0.571, UPD)
    assert cfl.parent_bos_limit(t, 0, 1, 1, 200).price == 0.576      # another cycle: unaffected
    with pytest.raises(AssertionError, match="before the M15 frame"):
        cfl.parent_bos_limit(t, 0, 0, 1, 200)


def test_a_table_built_without_the_stream_has_no_steps():
    from engine_v2.tests.test_lifecycle_sweep_unit import _tables as sweep_tables
    assert sweep_tables({(0, 0): (63, None)}, {0: +1}).bos_threshold_steps == {}


@pytest.mark.parametrize("lim, tail", [
    (_COUNTER_LIM, " moves=3"),
    (CrossFibLimit(0.5740, "parent_bos_threshold", ref_anchor_price=0.5700), ""),
])
def test_the_sweeps_limit_line_counts_the_moves(lim, tail):
    """`[sweep] cross_fib_limit …` ends with `moves=N` only for a moving limit (PLAN_M §7.3)."""
    from engine_v2.tests import test_lifecycle_sweep_unit as TS
    from engine_v2.multitf.sub_structure_pool import SubStructurePool as Pool
    pool = Pool()
    _res, logs = TS._run([TS._trig("confluence", 0, 0, "first_confluence", 63, +1)], pool=pool,
                         tables=TS._tables({(0, 0): (63, None)}, {0: +1}),
                         resolve=TS._resolver({("confluence", 63): TS._rs(100, 120)}),
                         geom=TS._geom(pool, {(+1, 100): None}), resolve_limit=lambda t, s, r: lim)
    line, = [x for x in logs if x.startswith("[sweep] cross_fib_limit")]
    assert line.endswith("(trigger first_confluence@63, record start 120)" + tail), line


# =====================================================================================================================
# 4. The R6-2 tripwire compares the moves
# =====================================================================================================================

def _counter_pool(stored):
    """One numbered sub: a BUY `subsequent_counter` under parent (0, 0) — sid 0 a SELL — started at 230."""
    pool = SubStructurePool()
    b, _ = pool.get_or_create(StructureKey(PARENT, TF, 1, 200))
    b.geometry = (SimpleNamespace(events=[], df=_df400()), 0)
    b.records.append(_rec(0, 230, trigger_type="subsequent_counter", direction=1))
    b.start_idx = 230
    pool.number_subs_by_start()
    b.cross_fib_limit = stored
    return pool


def test_the_tripwire_compares_the_start_price_and_the_moves():
    tables = _t()
    good = cfl.parent_bos_limit(tables, 0, 0, 1, 230)
    assert good == CrossFibLimit(0.5650, "parent_bos_threshold", ref_anchor_price=0.5600, price_at_start=0.5600,
                                 price_moves=((243, 0.5650),))
    logs = []
    assert cfl.check_cross_fib_limits(_counter_pool(good), [], tables, log=logs.append) == 0
    assert logs == ["[cross_fib_limit] tripwire: 1 subs re-derived at their start, 0 mismatches"]
    for bad in (CrossFibLimit(0.5650, "parent_bos_threshold", ref_anchor_price=0.5600, price_at_start=0.5610,
                              price_moves=((243, 0.5650),)),
                CrossFibLimit(0.5650, "parent_bos_threshold", ref_anchor_price=0.5600, price_at_start=0.5600,
                              price_moves=((242, 0.5650),)),
                CrossFibLimit(0.5650, "parent_bos_threshold", ref_anchor_price=0.5600)):
        logs = []
        assert cfl.check_cross_fib_limits(_counter_pool(bad), [], tables, log=logs.append) == 1
        assert logs[0].startswith("WARNING [cross_fib_limit] sub 0: stored P=")


def _two_record_pool(creating, started_by, stored):
    """One numbered BUY sub with two records: `creating` (seq 0) and `started_by` (seq 1, the earlier start) — each
    `(start_idx, trigger_type, parent_sid)`, parent cycle 0 (`_rec`); the sub starts at the started_by record."""
    pool = SubStructurePool()
    b, _ = pool.get_or_create(StructureKey(PARENT, TF, 1, 200))
    b.geometry = (SimpleNamespace(events=[], df=_df400()), 0)
    for seq, (start, tt, sid) in enumerate((creating, started_by)):
        r = _rec(0, start, trigger_type=tt, seq=seq, direction=1)
        r.parent_sid = sid
        b.records.append(r)
    b.start_idx = started_by[0]
    pool.number_subs_by_start()
    b.cross_fib_limit = stored
    return pool


def test_the_tripwire_re_derives_a_counter_sub_at_its_started_by_records_start():
    """R6-2 (a) shape (ii), PLAN_M §7.3: the creating record (start 260) folded the parent's move known at 243 into
    `price_at_start`; the sub starts at 230 (its started_by record), where that move is a MOVE — a mismatch."""
    tables = _t()
    stored = cfl.parent_bos_limit(tables, 0, 0, 1, 260)
    assert stored == CrossFibLimit(0.5650, "parent_bos_threshold", ref_anchor_price=0.5600)
    pool = _two_record_pool((260, "subsequent_counter", 0), (230, "subsequent_counter", 0), stored)
    logs = []
    assert cfl.check_cross_fib_limits(pool, [], tables, log=logs.append) == 1
    assert "re-derived at its start 230" in logs[0]


def test_the_tripwire_reads_the_started_by_records_parent_sid():
    """R6-2 (a) shape (i): created COUNTER under sid 0 (a sell) but started by a record under sid 1 (a buy) — the
    re-derivation is a CONFLUENCE `subsequent_confluence` with no x−1 (no P): a mismatch. Both starts fold every
    step of (0, 0), so only the parent sid tells them apart."""
    tables = _t()
    stored = cfl.parent_bos_limit(tables, 0, 0, 1, 300)
    pool = _two_record_pool((300, "subsequent_counter", 0), (280, "subsequent_confluence", 1), stored)
    logs = []
    assert cfl.check_cross_fib_limits(pool, [], tables, log=logs.append) == 1
    assert logs[0].startswith("WARNING [cross_fib_limit] sub 0: stored P=0.565 parent_bos_threshold")
    assert "P=None none" in logs[0]


def test_a_counter_reversal_sub_is_re_derived_without_its_reversing_sub():
    """A counter sub started by a reversal never reads the reversing sub — the tripwire looks for none (landing 2
    review A NOTE-5: with none in the pool it used to WARN "0 candidate reversing subs")."""
    tables = _t()
    stored = cfl.parent_bos_limit(tables, 0, 0, 1, 300)
    pool = _two_record_pool((320, "subsequent_counter", 0), (300, "reversal", 0), stored)
    logs = []
    assert cfl.check_cross_fib_limits(pool, [], tables, log=logs.append) == 0
    assert logs == ["[cross_fib_limit] tripwire: 1 subs re-derived at their start, 0 mismatches"]


# =====================================================================================================================
# 5. The reference window: S_L2's real H1 events and its 8 subs (PLAN_M §2 / §4)
# =====================================================================================================================

_SAVE = "artifacts/commits/week9-entries-decisions/20261010_155816_6d8cc67"
_PFX = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5_"

# The H1 `structure_events.csv` rows of S_L2 the limit rule reads (every CTS_ESTABLISHED / CTS_CONFIRMED /
# BOS_CONFIRMED / BOS_THRESHOLD_UPDATED — none on this window — and the reversal), `(idx, category, type, price, meta)`, copied by
# script; `test_the_embedded_rows_are_s_l2s` re-reads the save when it is present.
_H1_ROWS = [
    (115, 'STRUCTURE', 'CTS_ESTABLISHED', 0.56126, {'via': 'one_maru_continuous', 'pattern_anchor_idx': 114, 'pattern_name': 'one_maru_continuous', 'confirmed_at': 115, 'cts_frozen': None, 'cycle_id': 0, 'structure_id': 0, 'struct_direction': 1, 'cts_anchor_idx': 115}),  # noqa: E501
    (115, 'STRUCTURE', 'BOS_CONFIRMED', 0.55808, {'source': 'initial_prior_extreme', 'confirmed_at': 115, 'last_pullback_apply_idx': None, 'cycle_id': 0, 'structure_id': 0, 'struct_direction': 1, 'bos_anchor_idx': 96}),  # noqa: E501
    (439, 'STRUCTURE', 'CTS_CONFIRMED', 0.58246, {'via': 'one_maru_continuous', 'cycle_id': 0, 'structure_id': 0, 'struct_direction': 1, 'confirmed_at': 439, 'cts_anchor_idx': 430, 'confirmation_method': 'pullback'}),  # noqa: E501
    (652, 'STRUCTURE', 'CTS_ESTABLISHED', 0.58534, {'via': 'one_maru_opposite', 'pattern_anchor_idx': 651, 'pattern_name': 'one_maru_opposite', 'confirmed_at': 652, 'cts_frozen': 0.58448, 'cycle_id': 1, 'structure_id': 0, 'struct_direction': 1, 'cts_anchor_idx': 652}),  # noqa: E501
    (652, 'STRUCTURE', 'BOS_CONFIRMED', 0.5736, {'source': 'pullback_extreme', 'confirmed_at': 652, 'last_pullback_apply_idx': 439, 'cycle_id': 1, 'structure_id': 0, 'struct_direction': 1, 'bos_anchor_idx': 591}),  # noqa: E501
    (683, 'STRUCTURE', 'CTS_CONFIRMED', 0.58534, {'via': 'one_maru_continuous', 'cycle_id': 1, 'structure_id': 0, 'struct_direction': 1, 'confirmed_at': 683, 'cts_anchor_idx': 652, 'confirmation_method': 'pullback'}),  # noqa: E501
    (902, 'STATE', 'STATE_CHANGED', None, {'from': 'pullback_range', 'to': 'reversal', 'structure_id': 0, 'struct_direction': 1, 'reason': 'reversal_pattern', 'pat': 'continuous', 'pattern_anchor_idx': 897, 'bos_frozen': 0.5736, 'effective_idx': 902}),  # noqa: E501
    (703, 'STRUCTURE', 'CTS_ESTABLISHED', 0.58195, {'via': 'double_maru', 'pattern_anchor_idx': 702, 'pattern_name': 'double_maru', 'confirmed_at': 703, 'cts_frozen': None, 'cycle_id': 0, 'structure_id': 1, 'struct_direction': -1, 'cts_anchor_idx': 703}),  # noqa: E501
    (703, 'STRUCTURE', 'BOS_CONFIRMED', 0.58424, {'source': 'initial_prior_extreme', 'confirmed_at': 703, 'last_pullback_apply_idx': None, 'cycle_id': 0, 'structure_id': 1, 'struct_direction': -1, 'bos_anchor_idx': 689}),  # noqa: E501
    (727, 'STRUCTURE', 'CTS_CONFIRMED', 0.57902, {'via': 'one_maru_continuous', 'cycle_id': 0, 'structure_id': 1, 'struct_direction': -1, 'confirmed_at': 727, 'cts_anchor_idx': 710, 'confirmation_method': 'pullback'}),  # noqa: E501
    (748, 'STRUCTURE', 'CTS_ESTABLISHED', 0.57764, {'via': 'one_maru_opposite', 'pattern_anchor_idx': 747, 'pattern_name': 'one_maru_opposite', 'confirmed_at': 748, 'cts_frozen': 0.57856, 'cycle_id': 1, 'structure_id': 1, 'struct_direction': -1, 'cts_anchor_idx': 748}),  # noqa: E501
    (748, 'STRUCTURE', 'BOS_CONFIRMED', 0.58196, {'source': 'pullback_extreme', 'confirmed_at': 748, 'last_pullback_apply_idx': 727, 'cycle_id': 1, 'structure_id': 1, 'struct_direction': -1, 'bos_anchor_idx': 728}),  # noqa: E501
    (810, 'STRUCTURE', 'CTS_CONFIRMED', 0.574, {'via': 'one_maru_continuous', 'cycle_id': 1, 'structure_id': 1, 'struct_direction': -1, 'confirmed_at': 810, 'cts_anchor_idx': 761, 'confirmation_method': 'pullback'}),  # noqa: E501
    (902, 'STRUCTURE', 'CTS_ESTABLISHED', 0.57228, {'via': 'continuous', 'pattern_anchor_idx': 897, 'pattern_name': 'continuous', 'confirmed_at': 902, 'cts_frozen': 0.57376, 'cycle_id': 2, 'structure_id': 1, 'struct_direction': -1, 'cts_anchor_idx': 902}),  # noqa: E501
    (902, 'STRUCTURE', 'BOS_CONFIRMED', 0.58105, {'source': 'pullback_extreme', 'confirmed_at': 902, 'last_pullback_apply_idx': 810, 'cycle_id': 2, 'structure_id': 1, 'struct_direction': -1, 'bos_anchor_idx': 826}),  # noqa: E501
    (939, 'STRUCTURE', 'CTS_CONFIRMED', 0.57112, {'via': 'sd_zone_proximity', 'trigger_inner': 0.57866, 'zone_kind': 'POI', 'proximity_pips': 8, 'cycle_id': 2, 'structure_id': 1, 'struct_direction': -1, 'confirmed_at': 939, 'cts_anchor_idx': 905, 'confirmation_method': 'sd_zone_proximity'}),  # noqa: E501
]
# The 8 subs of both lenses' `*_subs.csv`: (sub_id, direction, start_idx, the first record's parent sid / cycle, its
# trigger type) — copied by the same script.
_SUBS = [
    (0, 1, 1020, 0, 0, 'first_confluence'),
    (1, -1, 1940, 0, 0, 'reversal'),
    (2, 1, 2470, 0, 0, 'reversal'),
    (3, -1, 2829, 0, 1, 'reversal'),
    (4, -1, 3621, 1, 2, 'first_confluence'),
    (5, 1, 3759, 1, 2, 'first_counter'),
    (6, -1, 3819, 1, 2, 'subsequent_confluence'),
    (7, 1, 4159, 1, 2, 'subsequent_counter'),
]


def _rows_to_events(rows):
    return [StructureEvent(idx=i, category=c, type=t, price=p, meta=dict(m)) for i, c, t, p, m in rows]


def test_the_reference_windows_counter_subs_take_the_parent_bos_and_the_rest_keep_their_rule():
    events = sorted(_rows_to_events(_H1_ROWS), key=ef.processing_order_key)
    tables = _build(events)                                          # LOH = 4h + 3, exact on this window
    assert tables.parent_sd == {0: 1, 1: -1}
    kw = dict(h1_events=events, parent_tables=tables, reversing_sub=None, prev_sub=None)
    counter = {}
    for sub_id, d, start, S, C, tt in _SUBS:
        if d != tables.parent_sd[S]:
            counter[sub_id] = cfl.cross_fib_limit_for(tt, (S, C), d, start, **kw)
        elif tt == "reversal":                                       # confluence: its own rule (needs the sub)
            with pytest.raises(AssertionError, match="a reversal needs its reversing sub"):
                cfl.cross_fib_limit_for(tt, (S, C), d, start, **kw)
        else:
            got = cfl.cross_fib_limit_for(tt, (S, C), d, start, **kw)
            assert got.source in ("none", "parent_cts_threshold"), (sub_id, got)
    assert counter == {                                              # PLAN_M §2 / §4; the replay's `[sweep]` lines
        1: CrossFibLimit(0.55808, "parent_bos_threshold", ref_anchor_price=0.55808),
        3: CrossFibLimit(0.5736, "parent_bos_threshold", ref_anchor_price=0.5736),
        5: CrossFibLimit(0.58105, "parent_bos_threshold", ref_anchor_price=0.58105),
        7: CrossFibLimit(0.58105, "parent_bos_threshold", ref_anchor_price=0.58105),
    }
    assert cfl.cross_fib_limit_for("first_confluence", (1, 2), -1, 3621, **kw) == CrossFibLimit(   # sub 4
        0.57376, "parent_cts_threshold", ref_anchor_price=0.574)      # sid 1 cycle 1's CTS_CONFIRMED (row 810)


def test_the_embedded_rows_are_s_l2s():
    root = Path(__file__).resolve().parents[2]
    save = root / _SAVE
    if not save.is_dir():
        pytest.skip("the S_L2 save is not in this tree (a mutant copy)")
    ev = pd.read_csv(save / (_PFX + "structure_events.csv"))
    rows = []
    for r in ev.itertuples():
        meta = ast.literal_eval(r.meta)
        if r.type in ("CTS_ESTABLISHED", "CTS_CONFIRMED", "BOS_CONFIRMED", "BOS_THRESHOLD_UPDATED") or (
                r.type == "STATE_CHANGED" and meta.get("to") == "reversal"):
            rows.append((int(r.idx), r.category, r.type, None if pd.isna(r.price) else float(r.price), meta))
    assert rows == _H1_ROWS
    subs = set()
    for lens in ("confluence", "counter"):
        df = pd.read_csv(save / (_PFX + f"M15_{lens}_subs.csv"))
        subs |= {(int(r.sub_id), int(r.direction), int(r.start_idx), int(r.first_record_parent_sid),
                  int(r.first_record_parent_cycle_id), r.first_record_trigger_type) for r in df.itertuples()}
    assert sorted(subs) == _SUBS
    assert os.path.basename(str(save)).endswith("6d8cc67")
