"""PLAN_I S2 — the cross-fib LIMIT fields, stamped but NOT applied (PLAN_I §9.3 D1 / D2 / D3, §9.4 S2).

- `beyond` (strict, both directions) and the `CrossFibLimit` value type (`zones/cross_cycle_fib.py`);
- main's `P` by its starting trigger (`structure_lifecycle.compute_main_started_by` /
  `compute_main_cross_fib_limits`) and `started_by` on the main SidRecord;
- MS's `cts_frozen` on every CTS_ESTABLISHED + the D3 WARNING (its carrier `winner_break_th`);
- the sub rule (`multitf/cross_fib_limit.py`): x−1 at the fire, the previous-sub read (by moment, inclusive, an
  ended x−1 read to its end, the fallbacks), the reversal / parent sources, the dispatch, the R6-2 tripwire;
- the sweep: `P` computed on a geometry MISS from the record's start, stored on the new sub, its `from_sub_id`
  remapped to the FINAL id;
- FibTracker: the latest CTS anchor, `M` (strict, sd ±1, final), the stamp at finalize, the F14 order tripwire;
- the fib hover's limit line (R6-7).
"""
from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace

import pandas as pd
import pytest

from engine_v2.charting._zone_render import fib_hover_customdata, fib_hover_limit_text
from engine_v2.multitf import cross_fib_limit as cfl
from engine_v2.multitf.cross_fib_limit import NO_LIMIT
from engine_v2.multitf.sid_records import build_sid_records_for_main
from engine_v2.multitf.sub_structure_pool import (
    LENS_CONFLUENCE as CONF,
    LENS_COUNTER as CTR,
    PooledStructure,
    StructureKey,
    SubStructurePool,
    TriggerRecord,
)
from engine_v2.structure.market_structure import MarketStructure, StructureEvent
from engine_v2.structure.structure_engine import compute_bounded_structure
from engine_v2.tests._cross_fib_limits import limit, unreached
from engine_v2.tests._event_factory import make_bos_confirmed, make_cts_established, make_cts_updated
from engine_v2.tests.test_imbalance_c3_knowability import _df as _c3_df, _gap
from engine_v2.tests.test_render_sub_projection import (  # noqa: F401 (pytest fixtures: geometry, m15_df)
    _START, _lens_dfs, _render, _set_lifecycle, _two_lens_records, geometry, m15_df,
)
from engine_v2.tests.test_unified_probe import _prepare_df
from engine_v2.zones.cross_cycle_fib import CROSS_FIB_LIMIT_SOURCES, CrossFibLimit, beyond
from engine_v2.zones.fib_tracker import FibEventOrderError, FibTracker, FibTrackerConfig
from engine_v2.zones.structure_lifecycle import compute_main_cross_fib_limits, compute_main_started_by
from engine_v2.tests._cross_fib_limits import NO_P

PARENT, TF = "H1.main", "M15"


def _q(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _ev(etype, idx, price=None, **meta):
    return StructureEvent(idx=idx, category="STRUCTURE", type=etype, price=price, meta=dict(meta))


def _est(moment, price, sid, cyc, sd=1, **extra):
    return make_cts_established(cts_anchor_idx=moment, confirmed_at=moment, price=price, structure_id=sid,
                                cycle_id=cyc, struct_direction=sd, meta=extra or None)


# =============================================================================
# beyond + CrossFibLimit
# =============================================================================

def test_beyond_is_strict_in_both_directions():
    assert beyond(0.60001, 0.6, 1) and not beyond(0.6, 0.6, 1) and not beyond(0.59999, 0.6, 1)
    assert beyond(0.59999, 0.6, -1) and not beyond(0.6, 0.6, -1) and not beyond(0.60001, 0.6, -1)
    with pytest.raises(AssertionError):
        beyond(0.61, 0.6, 0)


def test_cross_fib_limit_validates_its_source_and_price():
    assert CROSS_FIB_LIMIT_SOURCES == {
        "none", "reversal_bos_frozen", "parent_cts_threshold", "prev_sub_cts_threshold",
        "prev_sub_bos_threshold", "prev_sub_starting_price", "prev_sub_latest_cts", "prev_sub_no_cts",
        "parent_bos_threshold"}                                     # PLAN_M CL-5: a counter sub's live source
    CrossFibLimit(None, "none")
    CrossFibLimit(None, "prev_sub_no_cts", from_pool_idx=3)
    CrossFibLimit(0.5736, "reversal_bos_frozen", started_by="reversal", ref_anchor_price=0.5731,
                  break_pattern_anchor_idx=897, break_confirmed_idx=902)
    for bad in [(None, "reversal_bos_frozen"), (0.57, "none"), (0.57, "prev_sub_no_cts"), (0.57, "bos")]:
        with pytest.raises(AssertionError):
            CrossFibLimit(*bad)
    # PLAN_K A8: the anchor behind P iff P; the break iff a reversal source, both fields, anchor <= apply.
    for kw in [dict(price=0.57, source="parent_cts_threshold"),                                    # no anchor
               dict(price=None, source="none", ref_anchor_price=0.57),                            # anchor, no P
               dict(price=0.57, source="reversal_bos_frozen", ref_anchor_price=0.57),             # no break
               dict(price=0.57, source="parent_cts_threshold", ref_anchor_price=0.57,
                    break_pattern_anchor_idx=5, break_confirmed_idx=6),                           # break, not a reversal
               dict(price=0.57, source="reversal_bos_frozen", ref_anchor_price=0.57,
                    break_pattern_anchor_idx=5),                                                  # fields apart
               dict(price=0.57, source="reversal_bos_frozen", ref_anchor_price=0.57,
                    break_pattern_anchor_idx=7, break_confirmed_idx=6)]:                          # anchor after apply
        with pytest.raises(AssertionError):
            CrossFibLimit(**kw)


# =============================================================================
# Main: started_by + P (D2)
# =============================================================================

def _watch(idx, frozen, sid=0):
    return _ev("REVERSAL_WATCH_START", idx, frozen, structure_id=sid, bos_frozen=frozen, pattern_anchor_idx=idx,
               expires_idx=idx + 5)


def _main_events(*, watch_frozen=0.5736, rev_frozen=0.5736, second_reversal=False, open_watch=True):
    """sid 0 reverses at 902 out of the watch opened at 897; the cross-check reads THAT watch — the sid's LAST
    `REVERSAL_WATCH_START` at or before the reversal. Decoys (each one breaks a selector mutant): an earlier watch
    (880, another barrier), a later-sid watch (899, sid 1 — synthetic timing) and a watch after the reversal (960 —
    synthetic: MS stops at its reversal). sid 2 starts without a reversal of sid 1 — unreachable on main, it pins
    I1 (`started_by` attaches to the trigger) on a synthetic shape.

    PLAN_K A9: sid 0's most recent BOS_CONFIRMED by 902 is cycle 1's (moment 652, 0.5731 — the anchor, below its
    expanded threshold 0.5736); decoys: cycle 0's (115), sid 1's (905) and one after the reversal (950, synthetic)."""
    evs = [
        _est(115, 0.6, 0, 0, -1),
        make_bos_confirmed(bos_anchor_idx=96, confirmed_at=115, price=0.6100, structure_id=0, cycle_id=0,
                           struct_direction=-1),
        make_bos_confirmed(bos_anchor_idx=591, confirmed_at=652, price=0.5731, structure_id=0, cycle_id=1,
                           struct_direction=-1),
        make_bos_confirmed(bos_anchor_idx=900, confirmed_at=950, price=0.5600, structure_id=0, cycle_id=2,
                           struct_direction=-1),
        make_bos_confirmed(bos_anchor_idx=890, confirmed_at=905, price=0.5900, structure_id=1, cycle_id=0,
                           struct_direction=1),
        _watch(880, 0.5701),
        *([_watch(897, watch_frozen)] if open_watch else []),
        _ev("REVERSAL_CANDIDATE", 897, watch_frozen, structure_id=0, apply_idx=902, bos_frozen=watch_frozen),
        _watch(899, 0.5900, sid=1),
        _ev("STATE_CHANGED", 902, None, structure_id=0, to="reversal", pattern_anchor_idx=897,
            bos_frozen=rev_frozen),
        _est(903, 0.58, 1, 0, 1),
        _watch(960, 0.5800),
        _est(990, 0.59, 2, 0, -1),
    ]
    if second_reversal:
        evs.append(_ev("STATE_CHANGED", 950, None, structure_id=0, to="reversal", pattern_anchor_idx=945,
                       bos_frozen=0.5736))
    return evs


def test_main_started_by_attaches_to_the_starting_trigger_not_the_sid():
    """I1: sid 1 follows sid 0's reversal → `reversal`; sid 2 has no reversal of sid 1 → `start_trading`."""
    assert compute_main_started_by(_main_events()) == {0: "start_trading", 1: "reversal", 2: "start_trading"}


def test_main_limits_read_the_applied_reversals_bos_frozen():
    lim = compute_main_cross_fib_limits(_main_events())
    assert lim[0] == CrossFibLimit(None, "none", started_by="start_trading")
    assert lim[1] == CrossFibLimit(0.5736, "reversal_bos_frozen", started_by="reversal", ref_anchor_price=0.5731,
                                   break_pattern_anchor_idx=897, break_confirmed_idx=902)
    assert lim[2] == CrossFibLimit(None, "none", started_by="start_trading")


def test_main_limit_needs_a_bos_before_its_reversal():
    """PLAN_K A9: a reversal with no BOS_CONFIRMED of the reversing sid by its moment fails loudly (MS's reversal
    watch needs a BOS threshold — unreachable)."""
    evs = [e for e in _main_events() if not (e.type == "BOS_CONFIRMED" and e.meta["structure_id"] == 0
                                             and int(e.meta["confirmed_at"]) <= 902)]
    with pytest.raises(AssertionError, match="no BOS_CONFIRMED before it"):
        compute_main_cross_fib_limits(evs)


def test_main_limit_asserts_the_open_watch_agrees_and_one_reversal_per_sid():
    with pytest.raises(AssertionError, match="open watch"):
        compute_main_cross_fib_limits(_main_events(watch_frozen=0.5702))
    with pytest.raises(AssertionError, match="open watch"):
        compute_main_cross_fib_limits(_main_events(open_watch=False))     # the last watch <= 902 is 880's
    with pytest.raises(AssertionError, match="two reversals"):
        compute_main_cross_fib_limits(_main_events(second_reversal=True))
    with pytest.raises(AssertionError, match="no bos_frozen"):
        compute_main_cross_fib_limits(_main_events(rev_frozen=None))


def test_main_sid_records_carry_started_by():
    recs = build_sid_records_for_main(_main_events())
    assert [(r.sub_sid, r.meta["started_by"]) for r in recs] == [
        (0, "start_trading"), (1, "reversal"), (2, "start_trading")]


# =============================================================================
# MS: cts_frozen (D3)
# =============================================================================

def _ms_run(sd, *, force_break_th=None):
    """The in-watch-cycle fixture (cycle 0 established at 2, confirmed at 4, cycle 1 established at 7);
    `force_break_th` overwrites the carrier at the breakout's apply (the D3 WARNING's comparand)."""
    from engine_v2.tests.test_ms_new_cycle_ends_watch import _in_watch_cycle_rows, _run
    orig = MarketStructure._apply_pattern_at_apply_idx
    seen = []

    def spy(self, ev, apply_idx, kind):
        if kind == "breakout":
            if force_break_th is not None:
                self.state.winner_break_th = force_break_th(self.state)
            seen.append(self.state.winner_break_th)
        return orig(self, ev, apply_idx, kind)

    out = io.StringIO()
    with pytest.MonkeyPatch.context() as mp, contextlib.redirect_stdout(out):
        mp.setattr(MarketStructure, "_apply_pattern_at_apply_idx", spy)
        rows = _in_watch_cycle_rows()
        from engine_v2.tests import test_ms_cts_update_no_regress as _noreg
        from engine_v2.tests.test_unified_probe import _prepare_df
        from engine_v2.structure.structure_engine import compute_bounded_structure
        res = compute_bounded_structure(_prepare_df(rows if sd == 1 else _noreg._mirror(rows)), 0, sd, cross_fib_limit=NO_P)
    return res, seen, out.getvalue()


def _threshold_held_at(res, est):
    """c−1's `cts_threshold` as MS holds it when cycle c is established: its CTS_CONFIRMED price, moved by
    that cycle's CTS_THRESHOLD_UPDATED (F13) — read off the events, by moment, before the establishment."""
    c = int(est.meta["cycle_id"]) - 1
    thr = None
    for e in res.events:
        if e is est:
            break
        if e.meta.get("cycle_id") == c and e.type in ("CTS_CONFIRMED", "CTS_THRESHOLD_UPDATED"):
            thr = float(e.price)
    return thr


@pytest.mark.parametrize("sd", [1, -1])
def test_cts_frozen_is_on_every_establishment_none_on_cycle0(sd):
    res, _seen, _log = _ms_run(sd)
    est = [e for e in res.events if e.type == "CTS_ESTABLISHED"]
    assert [int(e.meta["cycle_id"]) for e in est] == [0, 1]
    assert all("cts_frozen" in e.meta for e in est)
    assert est[0].meta["cts_frozen"] is None
    held = _threshold_held_at(res, est[1])
    assert held is not None and est[1].meta["cts_frozen"] == held
    assert list(est[1].meta).index("cts_frozen") == list(est[1].meta).index("confirmed_at") + 1


@pytest.mark.parametrize("sd", [1, -1])
def test_cts_frozen_warning_fires_exactly_when_the_breakout_threshold_differs(sd):
    """D3: at a cycle >= 1 establishment, a `cts_frozen` != the winning breakout's detection threshold (or no
    threshold) is a WARNING, never an assert. Forced both ways through the carrier."""
    res, _seen, log = _ms_run(sd, force_break_th=lambda st: st.cts_threshold)
    assert "WARNING [ms] cts_frozen" not in log
    res, _seen, log = _ms_run(sd, force_break_th=lambda st: 123.0)
    warns = [l for l in log.splitlines() if l.startswith("WARNING [ms] cts_frozen")]
    assert len(warns) == 1 and "123.0" in warns[0] and "cycle=1" in warns[0]
    est1 = [e for e in res.events if e.type == "CTS_ESTABLISHED"][1]
    assert est1.meta["cts_frozen"] == _threshold_held_at(res, est1)       # the stamp is unaffected
    res, _seen, log = _ms_run(sd, force_break_th=lambda st: None)          # no detection threshold
    assert sum(l.startswith("WARNING [ms] cts_frozen") for l in log.splitlines()) == 1


# =============================================================================
# The sub rule (multitf/cross_fib_limit.py)
# =============================================================================

def _geo_sub(pool_idx, direction, starting_idx, events, *, slice_begin=0, start_idx=None, end_idx=None,
             highs=None, lows=None, n=60, records=()):
    df = pd.DataFrame({"h": highs or [0.60] * n, "l": lows or [0.55] * n})
    sub = PooledStructure(key=StructureKey(PARENT, TF, direction, starting_idx), pool_idx=pool_idx,
                          geometry=(SimpleNamespace(events=list(events), df=df), slice_begin),
                          start_idx=start_idx, end_idx=end_idx)
    sub.records.extend(records)
    return sub


def _rec(pool_idx, start_idx, *, zero_length=False, trigger_type="subsequent_confluence", seq=0,
         trigger_idx=None, direction=-1):
    r = TriggerRecord(lens=CONF, parent_sid=0, parent_cycle_id=0, trigger_sub_sid=0, sub_id=None,
                      pool_idx=pool_idx, trigger_type=trigger_type,
                      trigger_idx=start_idx if trigger_idx is None else trigger_idx,
                      probe_finalize_idx=start_idx, probe_finalize_condition="stub", parent_bos_anchor_idx=None,
                      probe_input_idx=None, starting_idx=0, direction=direction, sub_tf=TF,
                      relative_dir="confluence", parent_floor_idx=start_idx, start_idx=start_idx, seq=seq)
    if zero_length:
        r.trigger_end_idx = r.end_idx = start_idx
    return r


def _tables(parent_sd, steps=None, end_h1=None):
    """A `ParentTables` stand-in for the limit rule (PLAN_M §7.3): the parent sids' directions (a sub opposite its
    record's parent sid is COUNTER → `parent_bos_threshold`, checked FIRST) + the BOS-threshold steps it reads.
    TRAP (PLAN_M §9): `_rec` hard-codes parent (0, 0) for both directions — give sid 0 the direction that keeps
    the record a CONFLUENCE sub, or the test silently exercises the counter branch instead."""
    return SimpleNamespace(parent_sd=dict(parent_sd), bos_threshold_steps=dict(steps or {}), end_h1=dict(end_h1 or {}))


def test_prev_sub_at_fire_takes_the_latest_known_start_then_creation_index():
    a = _geo_sub(0, -1, 10, [], start_idx=100)
    b = _geo_sub(1, 1, 20, [], records=[_rec(1, 150), _rec(1, 90, zero_length=True)])   # known start 150
    c = _geo_sub(2, -1, 30, [], start_idx=150)
    d = _geo_sub(3, 1, 40, [], start_idx=300)
    e = _geo_sub(4, -1, 50, [], records=[_rec(4, 80, zero_length=True)])                  # no known start
    subs = [a, b, c, d, e]
    assert cfl.known_start(b) == 150 and cfl.known_start(e) is None
    assert cfl.prev_sub_at_fire(subs, 150) is c          # tie at 150 → the higher pool_idx (created later)
    assert cfl.prev_sub_at_fire(subs, 149) is a
    assert cfl.prev_sub_at_fire(subs, 300) is d          # inclusive
    assert cfl.prev_sub_at_fire(subs, 99) is None


def _cts(etype, idx, price, cyc, *, moment=None, anchor=None, sd=-1):
    m = dict(structure_id=0, cycle_id=cyc, struct_direction=sd)
    if etype == "CTS_ESTABLISHED":
        return make_cts_established(cts_anchor_idx=anchor if anchor is not None else idx,
                                    confirmed_at=moment if moment is not None else idx, price=price,
                                    structure_id=0, cycle_id=cyc, struct_direction=sd)
    if etype == "CTS_UPDATED":
        return make_cts_updated(cts_anchor_idx=anchor if anchor is not None else idx, via="continuous",
                                confirmed_at=moment if moment is not None else idx, price=price, meta=m)
    if etype == "CTS_CONFIRMED":
        m["cts_anchor_idx"] = anchor if anchor is not None else idx
    return StructureEvent(idx=idx, category="STRUCTURE", type=etype, price=price, meta=m)


# x−1 (sell, sd −1), slice_begin 100: entity-absolute moments = local + 100.
_SAME_DIR_EVENTS = [
    _cts("CTS_ESTABLISHED", 8, 0.580, 0, moment=10),      # moment 110
    _cts("CTS_UPDATED", 11, 0.579, 0, moment=12),         # moment 112
    _cts("CTS_CONFIRMED", 15, 0.579, 0, anchor=11),       # 115
    _cts("CTS_THRESHOLD_UPDATED", 18, 0.5785, 0),         # 118 (cycle 0's range sync)
    _cts("CTS_ESTABLISHED", 19, 0.577, 1, moment=20),     # 120 — cycle 1, never confirmed
    _cts("CTS_THRESHOLD_UPDATED", 22, 0.5900, 1),         # 122 — another cycle's: ignored (SYNTHETIC: MS range-syncs
                                                          #   only a confirmed cycle's threshold; pins the filter)
]


@pytest.mark.parametrize("s, want", [
    (118, (0.5785, "prev_sub_cts_threshold")),   # INCLUSIVE: the range sync ON x's start candle counts (R4-5)
    (117, (0.579, "prev_sub_cts_threshold")),    # the confirmed CTS price
    (130, (0.5785, "prev_sub_cts_threshold")),   # the most recent CONFIRMED cycle (0), not the established 1
    (114, (0.579, "prev_sub_latest_cts")),       # not confirmed yet → the latest anchor (R2-b): the update
    (111, (0.580, "prev_sub_latest_cts")),       # … before it: the establishment
])
def test_prev_sub_same_direction_reads_its_confirmed_threshold_by_moment(s, want):
    prev = _geo_sub(5, -1, 100, _SAME_DIR_EVENTS, slice_begin=100, start_idx=105)
    got = cfl.read_prev_sub(prev, s, -1, log=lambda m: None)
    assert (got.price, got.source, got.from_pool_idx) == want + (5,)


def test_prev_sub_same_direction_with_no_cts_has_no_price_and_warns():
    prev = _geo_sub(5, -1, 100, _SAME_DIR_EVENTS, slice_begin=100, start_idx=105)
    logs = []
    got = cfl.read_prev_sub(prev, 109, -1, log=logs.append)
    assert (got.price, got.source, got.from_pool_idx) == (None, "prev_sub_no_cts", 5)
    assert len(logs) == 1 and logs[0].startswith("WARNING [cross_fib_limit] x-1 pool#5")


def test_an_ended_prev_sub_is_read_as_of_its_end():
    """R4-6 / R5-3: x−1 ended at 116 → read until 116: the range sync at 118 is a POST-END event and never
    counts, whatever x's start."""
    prev = _geo_sub(5, -1, 100, _SAME_DIR_EVENTS, slice_begin=100, start_idx=105, end_idx=116)
    assert cfl.read_prev_sub(prev, 200, -1).price == 0.579
    prev_open = _geo_sub(5, -1, 100, _SAME_DIR_EVENTS, slice_begin=100, start_idx=105)
    assert cfl.read_prev_sub(prev_open, 200, -1).price == 0.5785


def _bos(etype, idx, price, *, moment=None):
    if etype == "BOS_CONFIRMED":
        return make_bos_confirmed(bos_anchor_idx=idx, confirmed_at=moment if moment is not None else idx,
                                  price=price, structure_id=0, cycle_id=0, struct_direction=-1)
    return StructureEvent(idx=idx, category="STRUCTURE", type=etype, price=price,
                          meta=dict(structure_id=0, cycle_id=0, struct_direction=-1))


def test_prev_sub_opposite_direction_reads_its_bos_threshold_by_moment_later_emitted_wins():
    evs = [
        _bos("BOS_CONFIRMED", 5, 0.585, moment=10),       # moment 10
        _bos("BOS_THRESHOLD_UPDATED", 14, 0.586),
        _bos("BOS_THRESHOLD_UPDATED", 14, 0.587),         # same moment, emitted later → wins
        _bos("BOS_THRESHOLD_UPDATED", 12, 0.584),         # emitted last, but an EARLIER moment
    ]
    prev = _geo_sub(2, -1, 0, evs, start_idx=1)
    assert cfl.read_prev_sub(prev, 20, 1) == CrossFibLimit(0.587, "prev_sub_bos_threshold", from_pool_idx=2,
                                                           ref_anchor_price=0.585)   # the BOS, not its threshold
    assert cfl.read_prev_sub(prev, 13, 1).price == 0.584
    assert cfl.read_prev_sub(prev, 11, 1).price == 0.585


@pytest.mark.parametrize("prev_dir, col_value", [(-1, 0.6123), (1, 0.5432)])
def test_prev_sub_opposite_without_a_bos_threshold_reads_its_starting_extreme(prev_dir, col_value):
    """R2-2: the high of a sell x−1's starting candle, the low of a buy x−1's (= its BOS_0)."""
    highs, lows = [0.60] * 60, [0.55] * 60
    highs[7], lows[7] = 0.6123, 0.5432                   # starting_idx 57 - slice_begin 50 = local 7
    prev = _geo_sub(1, prev_dir, 57, [], slice_begin=50, start_idx=60, highs=highs, lows=lows)
    got = cfl.read_prev_sub(prev, 70, -prev_dir)
    assert (got.price, got.source, got.ref_anchor_price) == (col_value, "prev_sub_starting_price", col_value)


def test_reversal_limit_is_the_reversing_subs_applied_bos_frozen():
    """+ PLAN_K A10: the reversing sub's most recent BOS_CONFIRMED by the reversal (moment 25, 0.58198 — not the
    later 45, synthetic) and the realised reversal pattern (37 → 40) ENTITY-ABSOLUTE (+ slice_begin 100)."""
    sc = _ev("STATE_CHANGED", 40, None, to="reversal", pattern_anchor_idx=37, bos_frozen=0.58156, structure_id=0)
    boses = [_bos("BOS_CONFIRMED", 20, 0.58198, moment=25), _bos("BOS_CONFIRMED", 44, 0.5810, moment=45)]
    rev = _geo_sub(2, 1, 120, [*boses, sc], slice_begin=100)
    rev.natural_reversal_idx = 140
    assert cfl.reversal_limit(rev) == CrossFibLimit(0.58156, "reversal_bos_frozen", from_pool_idx=2,
                                                    ref_anchor_price=0.58198, break_pattern_anchor_idx=137,
                                                    break_confirmed_idx=140)
    no_bos = _geo_sub(2, 1, 120, [sc], slice_begin=100)
    no_bos.natural_reversal_idx = 140
    with pytest.raises(AssertionError, match="no BOS_CONFIRMED before its reversal"):
        cfl.reversal_limit(no_bos)
    rev.natural_reversal_idx = 141
    with pytest.raises(AssertionError):
        cfl.reversal_limit(rev)
    two = _geo_sub(2, 1, 120, [sc, sc], slice_begin=100)
    two.natural_reversal_idx = 140
    with pytest.raises(AssertionError, match="2 reversals"):
        cfl.reversal_limit(two)


def _cts_conf(idx, price, sid, cyc, sd=-1, anchor=None):
    return _ev("CTS_CONFIRMED", idx, price, structure_id=sid, cycle_id=cyc, struct_direction=sd,
               cts_anchor_idx=idx - 1 if anchor is None else anchor)


def test_parent_cts_limit_reads_the_parents_cts_frozen():
    """+ PLAN_K A10: the anchor behind P = the parent cycle c−1's CTS_CONFIRMED price (0.574 — not its moved
    threshold 0.57376); decoys: the parent's cycle 0 and another sid's cycle 1."""
    h1 = [_est(902, 0.57, 1, 2, -1, cts_frozen=0.57376), _est(748, 0.58, 1, 1, -1, cts_frozen=None),
          _cts_conf(727, 0.57902, 1, 0, anchor=710), _cts_conf(810, 0.574, 1, 1, anchor=761),
          _cts_conf(500, 0.590, 0, 1, anchor=480)]
    assert cfl.parent_cts_limit(h1, 1, 2) == CrossFibLimit(0.57376, "parent_cts_threshold", ref_anchor_price=0.574)
    with pytest.raises(AssertionError, match="0 CTS_CONFIRMED"):
        cfl.parent_cts_limit([e for e in h1 if not (e.type == "CTS_CONFIRMED" and e.meta["cycle_id"] == 1
                                                    and e.meta["structure_id"] == 1)], 1, 2)
    with pytest.raises(AssertionError, match="cts_frozen None"):
        cfl.parent_cts_limit(h1, 1, 1)
    with pytest.raises(AssertionError, match="0 CTS_ESTABLISHED"):
        cfl.parent_cts_limit(h1, 1, 3)


def test_the_rule_dispatches_on_the_starting_trigger():
    h1 = [_est(902, 0.57, 1, 2, -1, cts_frozen=0.57376), _cts_conf(810, 0.574, 1, 1, anchor=761)]
    # sid 0 a buy, sid 1 a sell (PLAN_M): a sub OPPOSITE its parent sid is counter — checked first
    tables = _tables({0: 1, 1: -1}, {(1, 2): ((826, 3307, 0.58105, "BOS_CONFIRMED"),)}, {(1, 2): None})
    kw = dict(h1_events=h1, parent_tables=tables, reversing_sub=None, prev_sub=None)
    assert cfl.cross_fib_limit_for("first_confluence", (0, 0), 1, 500, **kw) == NO_LIMIT
    assert cfl.cross_fib_limit_for("first_confluence", (1, 2), -1, 3621, **kw).source == "parent_cts_threshold"
    # the x−1 rule now serves a CONFLUENCE `subsequent_confluence` only (the two counter types are counter subs by
    # construction — `uc1_trigger` / `subsequent_counter_pipeline` set the sub's direction to −parent's)
    assert cfl.cross_fib_limit_for("subsequent_confluence", (1, 2), -1, 3759, **kw) == NO_LIMIT   # no x−1
    for tt in ("first_counter", "subsequent_counter"):
        # PLAN_M: direction +1 under sid 1's −1 is COUNTER — the parent cycle's BOS threshold, x−1 never read
        assert cfl.cross_fib_limit_for(tt, (1, 2), 1, 3759, **kw) == CrossFibLimit(
            0.58105, "parent_bos_threshold", ref_anchor_price=0.58105)
    prev = _geo_sub(4, 1, 0, [_bos("BOS_CONFIRMED", 5, 0.58105, moment=10)], start_idx=1)
    got = cfl.cross_fib_limit_for("subsequent_confluence", (1, 2), -1, 20, h1_events=h1, parent_tables=tables,
                                  reversing_sub=None, prev_sub=prev)
    assert got == CrossFibLimit(0.58105, "prev_sub_bos_threshold", from_pool_idx=4, ref_anchor_price=0.58105)
    with pytest.raises(AssertionError, match="a reversal needs its reversing sub"):
        cfl.cross_fib_limit_for("reversal", (0, 0), 1, 20, **kw)
    with pytest.raises(AssertionError, match="unknown trigger type 'nonsense'"):
        cfl.cross_fib_limit_for("nonsense", (0, 0), 1, 20, **kw)


# =============================================================================
# The sweep: P at the fire (R6-2 (a)), stored on the new sub, from_sub_id remapped
# =============================================================================

def test_the_sweep_computes_p_on_a_miss_from_the_record_start_and_remaps_from_sub_id():
    """The S1 lag fixture (creation ≠ start order: pool#0 starts last): FC(0,0) fires 103, finalizes 600 →
    record start 600; first_counter(0,0) 203 → pool#1 (start 203); subsequent_counter(0,0) 303 → pool#2; a
    second subsequent_counter at 403 resolves to pool#2's key (a pool HIT → no P computed). pool#2's P names
    pool#0 as its supplier → after the numbering `from_sub_id` = pool#0's FINAL id 2."""
    from engine_v2.tests import test_lifecycle_sweep_unit as T
    pool = SubStructurePool()
    tables = T._tables({(0, 0): (100, None)}, {0: +1})
    resolve = T._resolver({
        (CONF, 103): T._rs(90, 600),
        (CTR, 203): T._rs(150, 203),
        (CTR, 303): T._rs(250, 303),
        (CTR, 403): T._rs(250, 403),
    })
    geom = T._geom(pool, {(+1, 90): None, (-1, 150): None, (-1, 250): None})
    calls = []

    def resolve_limit(t, record_start, reversing_sub):
        calls.append((t.trigger_type, t.trigger_idx, record_start, reversing_sub))
        if t.trigger_idx == 303:
            return limit(0.5, "prev_sub_bos_threshold", from_pool_idx=0)
        return NO_LIMIT

    _result, logs = T._run(
        [T._trig(CONF, 0, 0, "first_confluence", 103, +1), T._trig(CTR, 0, 0, "first_counter", 203, -1),
         T._trig(CTR, 0, 0, "subsequent_counter", 303, -1), T._trig(CTR, 0, 0, "subsequent_counter", 403, -1)],
        pool=pool, tables=tables, resolve=resolve, geom=geom, resolve_limit=resolve_limit,
    )
    assert calls == [("first_confluence", 103, 600, None), ("first_counter", 203, 203, None),
                     ("subsequent_counter", 303, 303, None)]                    # 403: a hit → not called
    fc, c1, c2 = T._sub(pool, +1, 90), T._sub(pool, -1, 150), T._sub(pool, -1, 250)
    assert (fc.sub_id, c1.sub_id, c2.sub_id) == (2, 0, 1)
    assert fc.cross_fib_limit is NO_LIMIT and c1.cross_fib_limit is NO_LIMIT
    assert c2.cross_fib_limit == limit(0.5, "prev_sub_bos_threshold", from_pool_idx=0, from_sub_id=2)
    assert any(l.startswith("[sweep] cross_fib_limit pool#2 P=0.5 source=prev_sub_bos_threshold from=pool#0")
               for l in logs)


def test_a_limit_from_a_sub_that_never_starts_has_no_from_sub_id_and_warns():
    from engine_v2.tests import test_lifecycle_sweep_unit as T
    pool = SubStructurePool()
    tables = T._tables({(1, 0): (100, None)}, {1: -1})
    resolve = T._resolver({(CONF, 200): T._rs(150, 200), (CONF, 300): T._rs(250, 400),
                           (CONF, 400): T._rs(350, 400)})
    geom = T._geom(pool, {(-1, 150): None, (-1, 250): None, (-1, 350): None})
    lim = {400: limit(0.5, "prev_sub_bos_threshold", from_pool_idx=1)}   # pool#1 never starts
    _result, logs = T._run(
        [T._trig(CONF, 1, 0, "first_confluence", 200, -1), T._trig(CONF, 1, 0, "first_confluence", 300, -1),
         T._trig(CONF, 1, 0, "subsequent_confluence", 400, -1)],
        pool=pool, tables=tables, resolve=resolve, geom=geom,
        resolve_limit=lambda t, s, r: lim.get(t.trigger_idx, NO_LIMIT),
    )
    c = T._sub(pool, -1, 350)
    assert c.cross_fib_limit.from_pool_idx == 1 and c.cross_fib_limit.from_sub_id is None
    assert any("cross-fib limit came from pool#1, which never started" in l for l in logs)


# =============================================================================
# The R6-2 tripwire
# =============================================================================

def _numbered_pool_for_tripwire(stored):
    """sub 0 (pool#0, sell, start 100, its BOS threshold 0.58105 by 150) → sub 1 (pool#1, buy, a
    subsequent_confluence starting 200 — x−1 = sub 0)."""
    pool = SubStructurePool()
    a, _ = pool.get_or_create(StructureKey(PARENT, TF, -1, 10))
    a.geometry = (SimpleNamespace(events=[_bos("BOS_CONFIRMED", 120, 0.58105, moment=150)],
                                  df=pd.DataFrame({"h": [0.6] * 300, "l": [0.5] * 300})), 0)
    a.records.append(_rec(0, 100, trigger_type="first_confluence"))
    a.start_idx = 100
    b, _ = pool.get_or_create(StructureKey(PARENT, TF, 1, 180))
    b.geometry = (SimpleNamespace(events=[], df=pd.DataFrame({"h": [0.6] * 300, "l": [0.5] * 300})), 0)
    b.records.append(_rec(1, 200, trigger_type="subsequent_confluence", seq=1, direction=1))
    b.start_idx = 200
    pool.number_subs_by_start()
    a.cross_fib_limit = NO_LIMIT
    b.cross_fib_limit = stored
    return pool


def test_the_r6_2_tripwire_re_derives_p_at_the_start_and_warns_on_a_mismatch():
    good = limit(0.58105, "prev_sub_bos_threshold", from_pool_idx=0, from_sub_id=0)
    logs = []
    assert cfl.check_cross_fib_limits(_numbered_pool_for_tripwire(good), [], _tables({0: 1}), log=logs.append) == 0
    assert logs == ["[cross_fib_limit] tripwire: 1 subs re-derived at their start, 0 mismatches"]
    for bad in (limit(0.58, "prev_sub_bos_threshold", from_pool_idx=0, from_sub_id=0),
                limit(0.58105, "prev_sub_cts_threshold", from_pool_idx=0, from_sub_id=0),
                limit(0.58105, "prev_sub_bos_threshold", from_pool_idx=0, from_sub_id=7),
                # PLAN_K A10: the tripwire compares the anchor behind P too
                limit(0.58105, "prev_sub_bos_threshold", from_pool_idx=0, from_sub_id=0, ref_anchor_price=0.581)):
        logs = []
        assert cfl.check_cross_fib_limits(_numbered_pool_for_tripwire(bad), [], _tables({0: 1}), log=logs.append) == 1
        assert logs[0].startswith("WARNING [cross_fib_limit] sub 1: stored P=")
        assert logs[1].endswith("1 mismatches")


# =============================================================================
# FibTracker: the latest anchor, M, the stamp, the order tripwire
# =============================================================================

def _tracker(limits=None):
    return FibTracker(FibTrackerConfig(fill_threshold=0.70), cross_fib_limits=limits)


# PLAN_I S3: a cross_cycle tracker consults its limit at a cycle >= 1 — the order tripwire tests stay within it.
_WITHIN = {0: unreached(1), 1: unreached(1)}


@pytest.mark.parametrize("sd", [1, -1])
def test_m_is_the_first_cycle_strictly_beyond_p(sd):
    def anchors(*prices):
        return {(0, c): (10 * c, p if sd == 1 else 2 - p, sd) for c, p in enumerate(prices)}

    def m(p, *prices):
        t = _tracker({0: limit(p if sd == 1 else 2 - p, "parent_cts_threshold")})
        t._latest_anchor = anchors(*prices)
        return t.cross_fib_limit_cycle(0)

    assert m(0.58, 0.59, 0.57, 0.60) == 0            # CTS_0 beyond → M = 0
    assert m(0.60, 0.59, 0.61, 0.62) == 1
    assert m(0.61, 0.59, 0.61, 0.62) == 2            # reaching P is not beyond (Q2)
    assert m(0.65, 0.59, 0.61, 0.62) is None         # not reached
    t = _tracker({0: NO_LIMIT})
    t._latest_anchor = anchors(0.59, 0.70)
    assert t.cross_fib_limit_cycle(0) is None        # no P → no M
    assert _tracker().cross_fib_limit_cycle(0) is None


def test_the_latest_anchor_follows_est_upd_and_the_confirmed_anchor_wins():
    t = _tracker()
    df = _c3_df(60, [_gap(15)])
    _q(t.on_cts_established, make_cts_established(cts_anchor_idx=20, confirmed_at=22, price=1.20,
                                                  structure_id=0, cycle_id=0, struct_direction=1),
       df, bos_idx=10, bos_price=1.0)
    assert t._latest_anchor[(0, 0)] == (20, 1.20, 1)
    _q(t.on_cts_updated, make_cts_updated(cts_anchor_idx=25, via="continuous", confirmed_at=27, price=1.25,
                                          meta=dict(structure_id=0, cycle_id=0, struct_direction=1)), df)
    assert t._latest_anchor[(0, 0)] == (25, 1.25, 1)
    # SYNTHETIC: a real CTS_CONFIRMED carries the current (latest updated) anchor; a different one here shows the
    # confirmed anchor is what is written (§4 N-f), not the last update kept.
    _q(t.on_cts_confirmed, _ev("CTS_CONFIRMED", 30, 1.24, structure_id=0, cycle_id=0, struct_direction=1,
                               cts_anchor_idx=24))
    assert t._latest_anchor[(0, 0)] == (24, 1.24, 1)


def test_every_fib_record_is_stamped_at_finalize_with_the_final_m():
    df = _c3_df(60, [_gap(15)])
    est = make_cts_established(cts_anchor_idx=20, confirmed_at=22, price=1.20, structure_id=0, cycle_id=0,
                               struct_direction=1)
    t = _tracker({0: limit(1.17, "prev_sub_bos_threshold", from_pool_idx=4, from_sub_id=5)})
    fib = _q(t.on_cts_established, est, df, bos_idx=10, bos_price=1.0)
    assert fib is not None and "cross_fib_limit_price" not in fib.meta        # stamped only at finalize
    _q(t._finalize_lifecycle_fields)
    t.stamp_ic_references(df)            # PLAN_K A11: the IC reference keys go BEFORE the limit keys (still last)
    (rec,) = t.get_all_fibs()
    keys = list(rec.meta)
    assert keys[-4:] == ["cross_fib_limit_price", "cross_fib_limit_source", "cross_fib_limit_cycle",
                         "cross_fib_limit_from_sub_id"]
    assert [rec.meta[k] for k in keys[-4:]] == [1.17, "prev_sub_bos_threshold", 0, 5]
    assert "started_by" not in rec.meta                                        # a sub's comes from the mirror
    main = _tracker({0: CrossFibLimit(None, "none", started_by="start_trading")})
    _q(main.on_cts_established, est, df, bos_idx=10, bos_price=1.0)
    _q(main._finalize_lifecycle_fields)
    main.stamp_ic_references(df)
    (mrec,) = main.get_all_fibs()
    assert {k: mrec.meta[k] for k in list(mrec.meta)[-5:]} == {
        "cross_fib_limit_price": None, "cross_fib_limit_source": "none", "cross_fib_limit_cycle": None,
        "cross_fib_limit_from_sub_id": None, "started_by": "start_trading"}
    bare = _tracker()
    _q(bare.on_cts_established, est, df, bos_idx=10, bos_price=1.0)
    _q(bare._finalize_lifecycle_fields)
    bare.stamp_ic_references(df)         # no limit entry: no structure reference, and NO raise (PLAN_K §9 C03)
    assert not any(k.startswith("cross_fib_limit") for k in bare.get_all_fibs()[0].meta)
    assert not any(k.startswith("ic_ref") for k in bare.get_all_fibs()[0].meta)   # cycle 0, no reference


def test_cycle_t_established_before_t_minus_1_confirmed_raises():
    """F14 (1), R6-4: the order tripwire RAISES a named RuntimeError (never swallowed by a geometry's
    `except (ValueError, IndexError)`)."""
    df = _c3_df(60, [_gap(15), _gap(35)])
    est1 = make_cts_established(cts_anchor_idx=40, confirmed_at=42, price=1.4, structure_id=0, cycle_id=1,
                                struct_direction=1)
    t = _tracker(_WITHIN)
    with pytest.raises(FibEventOrderError, match="before cycle 0's CTS_CONFIRMED"):
        _q(t.on_cts_established, est1, df, bos_idx=30, bos_price=1.2)
    assert issubclass(FibEventOrderError, RuntimeError) and not issubclass(FibEventOrderError, ValueError)
    t2 = _tracker(_WITHIN)
    _q(t2.on_cts_confirmed, _ev("CTS_CONFIRMED", 25, 1.2, structure_id=0, cycle_id=0, struct_direction=1,
                                cts_anchor_idx=20))
    _q(t2.on_cts_established, est1, df, bos_idx=30, bos_price=1.2)            # no raise
    t3 = _tracker(_WITHIN)
    _q(t3.on_cts_confirmed, _ev("CTS_CONFIRMED", 25, 1.2, structure_id=1, cycle_id=0, struct_direction=1,
                                cts_anchor_idx=20))
    with pytest.raises(FibEventOrderError):                                     # ANOTHER sid's cycle 0
        _q(t3.on_cts_established, est1, df, bos_idx=30, bos_price=1.2)


# =============================================================================
# The hover line (R6-7)
# =============================================================================

def _fib(**meta):
    return SimpleNamespace(meta=meta, start_idx=10, end_idx=None, end_reason=None, locked=False, status="active",
                           structure_id=0, cycle_id=0, struct_direction=1, bos_idx=1, bos_price=0.6, cts_idx=2,
                           cts_price=0.61)


def _lim(price, source, cycle=None, frm=None, **extra):
    return dict(cross_fib_limit_price=price, cross_fib_limit_source=source, cross_fib_limit_cycle=cycle,
                cross_fib_limit_from_sub_id=frm, **extra)


@pytest.mark.parametrize("meta, text", [
    (_lim(0.57827, "prev_sub_bos_threshold", 2, 6),
     "limit_price=0.57827 | prev_sub_bos_threshold (sub 6) | cycle_limit=2"),
    (_lim(0.5736, "reversal_bos_frozen", 2, started_by="reversal"),
     "limit_price=0.57360 | reversal_bos_frozen | cycle_limit=2"),
    (_lim(0.58105, "prev_sub_bos_threshold", None, 4),
     "limit_price=0.58105 | prev_sub_bos_threshold (sub 4) | cycle_limit=none"),
    (_lim(0.57926, "reversal_bos_frozen", 0, 1), "limit_price=0.57926 | reversal_bos_frozen (sub 1) | cycle_limit=0"),
    # no `P`: no cycle may cross — `limit_price=NA (<why>)` alone, no `cycle_limit` (the user's spec 2026-10-23)
    (_lim(None, "none", started_by="start_trading"), "limit_price=NA (start_trading)"),
    (_lim(None, "none", started_by="first_confluence", parent_cycle_id=0),
     "limit_price=NA (first_confluence on parent cycle 0)"),
    (_lim(None, "none", started_by="first_counter"), "limit_price=NA (no previous sub)"),
    (_lim(None, "prev_sub_no_cts", frm=3, started_by="subsequent_confluence"),
     "limit_price=NA (prev_sub_no_cts (sub 3))"),
])
def test_the_limit_hover_line(meta, text):
    assert fib_hover_limit_text(_fib(**meta)) == text


def test_the_limit_line_follows_the_ic_reference_and_is_absent_without_the_keys():
    with_lim = fib_hover_customdata(_fib(**_lim(0.57827, "prev_sub_bos_threshold", 2, 6)), None, 0.6, 0.6)[5]
    assert with_lim.split("<br>") == [
        "",
        "ic_ref: NA (no reference)",                           # PLAN_K A17: a stamped record with no reference
        "limit_price=0.57827 | prev_sub_bos_threshold (sub 6) | cycle_limit=2"]
    assert fib_hover_limit_text(_fib()) is None
    assert fib_hover_customdata(_fib(), None, 0.6, 0.6)[5] == ""



def test_main_limit_needs_a_watch_before_the_reversal():
    evs = [_est(115, 0.6, 0, 0, -1),
           _ev("STATE_CHANGED", 902, None, structure_id=0, to="reversal", bos_frozen=0.5736),
           _est(903, 0.58, 1, 0, 1)]
    with pytest.raises(AssertionError, match="open watch"):
        compute_main_cross_fib_limits(evs)


@pytest.mark.parametrize("sd, frozen", [(1, 0.5998), (-1, 0.6202)])
def test_main_limit_holds_when_a_later_anchors_reversal_winner_applies_first(sd, frozen):
    """S2 landing review (lens A M1): a later anchor's reversal WINNER applies at 8 before the watch's pending
    candidate (apply 9) — no REVERSAL_CANDIDATE sits at the realised reversal, yet the limit is the open watch's
    frozen barrier (MARKET_STRUCTURE_SPEC's later-winner case; `test_ms_reversal_on_expiry`)."""
    from engine_v2.tests import test_ms_reversal_on_expiry as RV
    res = _q(compute_bounded_structure, _prepare_df(RV._rows(RV._later_winner_rows, sd)), 0, sd, cross_fib_limit=NO_P)
    assert [int(e.idx) for e in res.events if e.type == "STATE_CHANGED" and e.meta.get("to") == "reversal"] == [8]
    assert [int(e.meta["apply_idx"]) for e in res.events if e.type == "REVERSAL_CANDIDATE"] == [9]
    lim = compute_main_cross_fib_limits(list(res.events) + [_est(20, 0.6, 1, 0, -sd)])
    (sc,) = [e for e in res.events if e.type == "STATE_CHANGED" and e.meta.get("to") == "reversal"]
    (cand,) = [e for e in res.events if e.type == "REVERSAL_CANDIDATE"]
    last_bos = max((e for e in res.events if e.type == "BOS_CONFIRMED" and int(e.meta["confirmed_at"]) <= 8),
                   key=lambda e: int(e.meta["confirmed_at"]))
    assert lim[1] == CrossFibLimit(frozen, "reversal_bos_frozen", started_by="reversal",
                                   ref_anchor_price=float(last_bos.price),
                                   break_pattern_anchor_idx=int(sc.meta["pattern_anchor_idx"]), break_confirmed_idx=8)
    # PLAN_K A13: the STATE_CHANGED's pattern is the later anchor's WINNER, not the pending candidate's
    assert int(sc.meta["pattern_anchor_idx"]) != int(cand.meta["pattern_anchor_idx"])


# --- S2 landing review, lens B: the tests that kill its 20 surviving mutants ---------------------------------

@pytest.mark.parametrize("sd", [1, -1])
def test_the_unforced_carrier_is_the_breakouts_detection_threshold(sd):
    """D3 / F12: unforced, the carrier at each breakout apply is the detection threshold (None for cycle 0's
    breakout from no range) and equals cycle 1's `cts_frozen` -> no WARNING."""
    res, seen, log = _ms_run(sd)
    est = [e for e in res.events if e.type == "CTS_ESTABLISHED"]
    assert seen == [None, est[1].meta["cts_frozen"]] and seen[1] is not None
    assert "WARNING [ms] cts_frozen" not in log


def test_prev_sub_same_direction_reads_the_most_recent_confirmed_cycle():
    """x−1 with TWO confirmed cycles (the realistic order pin): cycle 1's confirmation replaces cycle 0's
    threshold; each cycle's range sync moves its own."""
    evs = [
        _cts("CTS_ESTABLISHED", 8, 0.580, 0, moment=10),
        _cts("CTS_CONFIRMED", 15, 0.579, 0, anchor=8),
        _cts("CTS_THRESHOLD_UPDATED", 18, 0.5785, 0),
        _cts("CTS_ESTABLISHED", 19, 0.577, 1, moment=20),
        _cts("CTS_CONFIRMED", 25, 0.576, 1, anchor=19),
        _cts("CTS_THRESHOLD_UPDATED", 28, 0.5765, 1),
    ]
    prev = _geo_sub(5, -1, 100, evs, slice_begin=100, start_idx=105)
    got = [cfl.read_prev_sub(prev, s, -1, log=lambda m: None).price for s in (119, 124, 126, 130)]
    assert got == [0.5785, 0.5785, 0.576, 0.5765]


def test_a_prev_sub_ended_by_its_reversal_yields_the_frozen_barrier():
    """R5-3 (lens A m1): an opposite x−1 whose watch froze its BOS barrier at 16 and which reversed (ended) at 20 is
    read as of 20 — the frozen 0.586, whatever x's start; a later threshold event (25, synthetic: MS stops at its
    reversal) never counts."""
    evs = [
        _bos("BOS_CONFIRMED", 5, 0.585, moment=10),
        _bos("BOS_THRESHOLD_UPDATED", 14, 0.586),
        _ev("REVERSAL_WATCH_START", 16, 0.586, structure_id=0, bos_frozen=0.586),
        _ev("STATE_CHANGED", 20, None, structure_id=0, to="reversal", bos_frozen=0.586),
        _bos("BOS_THRESHOLD_UPDATED", 25, 0.590),
    ]
    prev = _geo_sub(4, -1, 0, evs, start_idx=1, end_idx=20)
    assert cfl.read_prev_sub(prev, 30, 1) == CrossFibLimit(0.586, "prev_sub_bos_threshold", from_pool_idx=4,
                                                           ref_anchor_price=0.585)


def test_known_start_is_the_earliest_live_record():
    b = _geo_sub(1, 1, 20, [], records=[_rec(1, 170), _rec(1, 150), _rec(1, 90, zero_length=True)])
    assert cfl.known_start(b) == 150


def test_first_confluence_reads_the_parent_cycle_not_the_sid():
    """sid 0 cycle 2 has a P; sid 1 cycle 0 has none; another sid's (S', c) is never read."""
    h1 = [_est(500, 0.57, 0, 2, -1, cts_frozen=0.5740), _est(902, 0.57, 1, 2, -1, cts_frozen=0.57376),
          _cts_conf(470, 0.5745, 0, 1), _cts_conf(810, 0.574, 1, 1, anchor=761)]
    kw = dict(h1_events=h1, parent_tables=_tables({0: -1, 1: -1}), reversing_sub=None, prev_sub=None)
    assert cfl.cross_fib_limit_for("first_confluence", (0, 2), -1, 600, **kw) == CrossFibLimit(
        0.5740, "parent_cts_threshold", ref_anchor_price=0.5745)
    assert cfl.cross_fib_limit_for("first_confluence", (1, 2), -1, 950, **kw) == CrossFibLimit(
        0.57376, "parent_cts_threshold", ref_anchor_price=0.574)
    assert cfl.cross_fib_limit_for("first_confluence", (1, 0), -1, 950, **kw) == NO_LIMIT


def test_the_sweep_hands_the_reversing_sub_to_the_limit_rule():
    from engine_v2.tests import test_lifecycle_sweep_unit as TS
    pool = SubStructurePool()
    calls = []

    def resolve_limit(t, s, r):
        calls.append((t.trigger_type, s, r))
        return NO_LIMIT

    TS._run([TS._trig(CONF, 0, 0, "first_confluence", 63, +1)], pool=pool,
            tables=TS._tables({(0, 0): (63, None)}, {0: +1}),
            resolve=TS._resolver({(CONF, 63): TS._rs(100, 120)}),
            geom=TS._geom(pool, {(+1, 100): 303, (-1, 280): None}),
            resolve_rev=TS._rev_resolver({(+1, 100): 280}), resolve_limit=resolve_limit)
    X = TS._sub(pool, +1, 100)
    assert [(tt, s) for tt, s, _r in calls] == [("first_confluence", 120), ("reversal", 303)]
    assert calls[0][2] is None and calls[1][2] is X


def _df400():
    return pd.DataFrame({"h": [0.6] * 400, "l": [0.5] * 400})


def test_the_tripwire_re_derives_from_the_started_by_record_at_the_start():
    """b (pool#0, buy) is CREATED by a first_confluence (fires 150, starts 260) but STARTED by a later
    subsequent_confluence (fires 190, starts 200). a (pool#1, sell, sub 0) supplies P; its BOS threshold moves at
    195 — after b's started_by trigger, before b's start: pool_idx != sub_id, started_by != creating record,
    start != trigger_idx."""
    pool = SubStructurePool()
    b, _ = pool.get_or_create(StructureKey(PARENT, TF, 1, 180))
    a, _ = pool.get_or_create(StructureKey(PARENT, TF, -1, 10))
    a.geometry = (SimpleNamespace(events=[_bos("BOS_CONFIRMED", 120, 0.58105, moment=150),
                                          _bos("BOS_THRESHOLD_UPDATED", 195, 0.5815)], df=_df400()), 0)
    a.records.append(_rec(1, 100, trigger_type="first_confluence", seq=1))
    a.start_idx = 100
    b.geometry = (SimpleNamespace(events=[], df=_df400()), 0)
    b.records.append(_rec(0, 260, trigger_type="first_confluence", seq=0, trigger_idx=150, direction=1))
    b.records.append(_rec(0, 200, trigger_type="subsequent_confluence", seq=2, trigger_idx=190, direction=1))
    b.start_idx = 200
    pool.number_subs_by_start()
    assert (a.sub_id, b.sub_id) == (0, 1)
    a.cross_fib_limit = NO_LIMIT
    b.cross_fib_limit = limit(0.5815, "prev_sub_bos_threshold", from_pool_idx=1, from_sub_id=0,
                              ref_anchor_price=0.58105)
    logs = []
    assert cfl.check_cross_fib_limits(pool, [], _tables({0: 1}), log=logs.append) == 0, logs
    assert logs == ["[cross_fib_limit] tripwire: 1 subs re-derived at their start, 0 mismatches"]


def test_the_tripwire_finds_the_one_opposite_live_reversing_sub():
    """Y (sell) is started by a reversal at R = 300 (created earlier by a first_counter that starts later). Three
    subs reverse at 300: X (buy, live at 300) is Y's reversing sub; Z (sell — Y's own direction) and W (buy, its
    record ended before 300) are not."""
    pool = SubStructurePool()

    def mk(d, k, recs, nat=None, sc_idx=None, frozen=0.58):
        s, _ = pool.get_or_create(StructureKey(PARENT, TF, d, k))
        evs = [] if sc_idx is None else [
            _bos("BOS_CONFIRMED", sc_idx - 20, frozen, moment=sc_idx - 10),
            _ev("STATE_CHANGED", sc_idx, None, to="reversal", pattern_anchor_idx=sc_idx - 3, bos_frozen=frozen,
                structure_id=0)]
        s.geometry = (SimpleNamespace(events=evs, df=_df400()), 0)
        s.natural_reversal_idx = nat
        s.records.extend(recs)
        s.start_idx = min(r.start_idx for r in recs)
        return s

    X = mk(1, 50, [_rec(0, 100, trigger_type="first_confluence", seq=0, direction=1)], nat=300, sc_idx=300)
    Z = mk(-1, 60, [_rec(1, 120, trigger_type="first_confluence", seq=1)], nat=300, sc_idx=300, frozen=0.61)
    w_rec = _rec(2, 130, trigger_type="first_confluence", seq=2, direction=1)
    w_rec.trigger_end_idx = w_rec.end_idx = 250
    W = mk(1, 70, [w_rec], nat=300, sc_idx=300, frozen=0.62)
    Y = mk(-1, 290, [_rec(3, 320, trigger_type="subsequent_confluence", seq=3, trigger_idx=280),
                     _rec(3, 300, trigger_type="reversal", seq=4, trigger_idx=300)])
    pool.number_subs_by_start()
    for s in (X, Z, W):
        s.cross_fib_limit = NO_LIMIT
    Y.cross_fib_limit = CrossFibLimit(0.58, "reversal_bos_frozen", from_pool_idx=X.pool_idx, from_sub_id=X.sub_id,
                                      ref_anchor_price=0.58, break_pattern_anchor_idx=297, break_confirmed_idx=300)
    logs = []
    assert cfl.check_cross_fib_limits(pool, [], _tables({0: -1}), log=logs.append) == 0, logs
    assert logs[-1] == "[cross_fib_limit] tripwire: 1 subs re-derived at their start, 0 mismatches"
    # Two equally qualified reversing subs: the warn-only tripwire says so instead of stopping (lens A m3).
    W.records[0].trigger_end_idx = W.records[0].end_idx = None
    logs = []
    assert cfl.check_cross_fib_limits(pool, [], _tables({0: -1}), log=logs.append) == 1
    assert "2 candidate reversing subs at 300" in logs[0]


def test_a_rendered_subs_fib_records_carry_its_limit(geometry, m15_df):
    pool, sub = geometry
    sub.cross_fib_limit = limit(0.5, "prev_sub_bos_threshold", from_pool_idx=0, from_sub_id=3)
    _two_lens_records(pool, sub, m15_df)
    _set_lifecycle(sub, _START, None, None)
    res = _render(sub, m15_df, _lens_dfs(m15_df))
    assert res.fib_states
    assert {(f.meta["cross_fib_limit_source"], f.meta["cross_fib_limit_from_sub_id"]) for f in res.fib_states} == {
        ("prev_sub_bos_threshold", 3)}


def test_run_pipeline_hands_main_its_cross_fib_limits(monkeypatch):
    import engine_v2.pipeline.orchestrator as orch
    import engine_v2.tests.test_render_sub_projection as RSP
    seen = {}
    real = orch._run_downstream_pipeline

    def spy(df, events, sd, **kw):
        seen.update(lim=kw.get("cross_fib_limits"), events=events)
        return real(df, events, sd, **kw)

    monkeypatch.setattr(orch, "_run_downstream_pipeline", spy)
    _q(orch.run_pipeline, RSP._make_raw_df(RSP._make_flat_prefix() + RSP._make_reversing_data()))
    assert seen["lim"] is not None and seen["lim"] == compute_main_cross_fib_limits(seen["events"])


def test_the_latest_anchor_keeps_a_sells_direction_through_every_handler():
    t = _tracker({0: limit(0.78, "prev_sub_bos_threshold", from_pool_idx=0, from_sub_id=0)})
    df = _c3_df(60, [_gap(15)])
    _q(t.on_cts_established, make_cts_established(cts_anchor_idx=20, confirmed_at=22, price=0.80, structure_id=0,
                                                  cycle_id=0, struct_direction=-1), df, bos_idx=10, bos_price=1.0)
    assert t._latest_anchor[(0, 0)] == (20, 0.80, -1) and t.cross_fib_limit_cycle(0) is None
    _q(t.on_cts_updated, make_cts_updated(cts_anchor_idx=25, via="continuous", confirmed_at=27, price=0.75,
                                          meta=dict(structure_id=0, cycle_id=0, struct_direction=-1)), df)
    assert t._latest_anchor[(0, 0)] == (25, 0.75, -1)
    _q(t.on_cts_confirmed, _ev("CTS_CONFIRMED", 30, 0.75, structure_id=0, cycle_id=0, struct_direction=-1,
                               cts_anchor_idx=25))
    assert t._latest_anchor[(0, 0)] == (25, 0.75, -1) and t.cross_fib_limit_cycle(0) == 0


def test_the_order_tripwire_reads_the_events_own_sid():
    df = _c3_df(60, [_gap(15), _gap(35)])
    t = _tracker(_WITHIN)
    _q(t.on_cts_confirmed, _ev("CTS_CONFIRMED", 25, 1.2, structure_id=1, cycle_id=0, struct_direction=1,
                               cts_anchor_idx=20))
    est1 = make_cts_established(cts_anchor_idx=40, confirmed_at=42, price=1.4, structure_id=1, cycle_id=1,
                                struct_direction=1)
    _q(t.on_cts_established, est1, df, bos_idx=30, bos_price=1.2)   # no raise
    assert (1, 1) in t._latest_anchor
    with pytest.raises(FibEventOrderError):
        _q(_tracker(_WITHIN).on_cts_established, est1, df, bos_idx=30, bos_price=1.2)


def test_the_driver_wires_the_fire_time_rule_and_the_tripwire(monkeypatch):
    """`_run_multi_tf_dual`'s closure reads x−1 at the RECORD START (not the trigger idx), and the R6-2 tripwire
    runs on the sweep's pool and the H1 events. The sweep stub calls the closure, then returns."""
    from engine_v2.multitf.lifecycle_sweep import SweepTrigger
    from engine_v2.pipeline.orchestrator import _run_multi_tf_dual

    class _Checked(Exception):
        pass

    got = {}

    def _sweep(_triggers, *, pool, resolve_cross_fib_limit, **_kw):
        x1, _ = pool.get_or_create(StructureKey("H1.main", "M15", -1, 40))
        x1.geometry = (SimpleNamespace(events=[_bos("BOS_CONFIRMED", 50, 0.58105, moment=55)], df=_df400()), 0)
        x1.start_idx = 60
        t = SweepTrigger(lens=CONF, parent_sid=0, parent_cycle_id=1, trigger_type="subsequent_confluence",
                         trigger_idx=50, direction=+1, trigger_event_idx=12, source=None)
        got["lim"] = resolve_cross_fib_limit(t, 70, None)
        t2 = SweepTrigger(lens=CONF, parent_sid=0, parent_cycle_id=1, trigger_type="subsequent_counter",
                          trigger_idx=150, direction=-1, trigger_event_idx=30, source=None)
        got["counter"] = resolve_cross_fib_limit(t2, 200, None)
        got["pool"] = pool
        return SimpleNamespace(unresolved=[], spawned=[])

    def _check(pool, h1_events, parent_tables, **_kw):
        got["checked"] = (pool, list(h1_events), parent_tables)
        raise _Checked

    for mod in ("first_confluence_pipeline", "subsequent_confluence_pipeline", "subsequent_counter_pipeline"):
        monkeypatch.setattr(f"engine_v2.multitf.{mod}.to_multi_tf_trigger",
                            lambda _v, _h1: SimpleNamespace(parent_sid=0, parent_cycle_id=1, lower_sd=+1))
    monkeypatch.setattr("engine_v2.multitf.uc1_trigger.detect_uc1_triggers", lambda *a, **k: [])
    monkeypatch.setattr("engine_v2.multitf.data_bridge.fetch_lower_tf_data",
                        lambda *a, **k: pd.DataFrame({"time": [0]}))
    monkeypatch.setattr("engine_v2.multitf.data_bridge.prepare_lower_tf_data", lambda df: df)
    tables = _tables({0: +1},                              # the trigger (+1) is a CONFLUENCE sub of sid 0
                     {(0, 0): ((10, 43, 0.59, "BOS_CONFIRMED"),), (0, 1): ((30, 123, 0.58, "BOS_CONFIRMED"),)},
                     {(0, 0): 30, (0, 1): None})
    monkeypatch.setattr("engine_v2.multitf.parent_tables.build_parent_tables", lambda *a, **k: tables)
    monkeypatch.setattr("engine_v2.multitf.lifecycle_sweep.run_lifecycle_sweep", _sweep)
    monkeypatch.setattr("engine_v2.multitf.cross_fib_limit.check_cross_fib_limits", _check)
    monkeypatch.setattr("engine_v2.multitf.entity_df_mutation._map_parent_idx_to_m15_hour_end",
                        lambda p, _h1, _m15: 4 * int(p) + 3)
    h1 = pd.DataFrame({"time": pd.date_range("2025-11-17", periods=40, freq="h", tz="UTC")})
    with pytest.raises(_Checked):
        _q(_run_multi_tf_dual, h1, [], [], [], {}, None, first_confluence_triggers=[],
           subsequent_confluence_triggers=[SimpleNamespace(parent_sid=0, parent_cycle_id=1, trigger_event_idx=20, input_idx=15)],
           subsequent_counter_triggers=[])
    assert got["lim"] == CrossFibLimit(0.58105, "prev_sub_bos_threshold", from_pool_idx=0, ref_anchor_price=0.58105)
    assert got["checked"][0] is got["pool"]
    assert got["checked"][2] is tables                      # PLAN_M: the same tables as the fire-time rule
    # PLAN_M: the closure hands the trigger's own parent CYCLE — a counter sub of (0, 1) reads (0, 1)'s threshold
    assert got["counter"] == CrossFibLimit(0.58, "parent_bos_threshold", ref_anchor_price=0.58)


def test_m_reads_only_its_own_sids_anchors():
    """A main tracker holds every sid; sid 1's M must not walk sid 0's cycles."""
    t = _tracker({0: CrossFibLimit(None, "none", started_by="start_trading"),
                  1: limit(0.58, "reversal_bos_frozen", started_by="reversal")})
    t._latest_anchor = {(0, 0): (5, 0.70, 1), (0, 1): (9, 0.72, 1), (0, 2): (12, 0.74, 1), (1, 0): (20, 0.59, -1),
                        (1, 1): (30, 0.585, -1)}
    assert (t.cross_fib_limit_cycle(0), t.cross_fib_limit_cycle(1)) == (None, None)   # sid 1 never beyond 0.58
    t._latest_anchor[(1, 2)] = (40, 0.575, -1)
    assert t.cross_fib_limit_cycle(1) == 2
