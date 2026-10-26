"""PLAN_R step 2 — the proximity overlay driven inside the lifecycle sweep (`multitf/proximity_overlay.py`): the sticky
proximity source sub (the hold), the frozen ended source, the strict most-recent cycle, the sd qualifying bound, the
usability clocks (the tick at LOH(h − 1), record-start gating, the POI from CTS_CONFIRMED), the opp_sd unlock (Q2 / Q3),
the no-M15 hour, and the zone view read as of each zone's own moment.

Synthetic bullish parent cycle (sid 0, cycle 0, sd = +1) on H1 with LOH(h) = 4h + 3 (M15 4h … 4h+3 = H1 hour h):
BOS inner 1.4000 / outer 1.3990, POI inner 1.4100 / outer 1.4090, CTS inner 1.4300 / outer 1.4310 (the step-1 test
fixtures, `test_proximity_reference_ladders`). The scan starts at the CTS_CONFIRMED candle 2. Sub zones are given as
`SubZone`s (the view a real sub's projections would yield) — the pool holds stub subs + records.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Optional

import pytest

from engine_v2.multitf.proximity_overlay import ProximityOverlay, SubZone, build_sub_zone_view
from engine_v2.multitf.sub_structure_pool import LENS_CONFLUENCE as CONF, LENS_COUNTER as CTR, StructureKey, SubStructurePool
from engine_v2.tests.test_first_trigger_migration import _rec
from engine_v2.tests.test_render_sub_projection import geometry, m15_df  # noqa: F401 — pytest fixtures
from engine_v2.tests.test_proximity_reference_ladders import _cycle, _df, _scan, _set

LOH = lambda h: 4 * int(h) + 3            # noqa: E731
KEY = (0, 0)


def _sub(pool: SubStructurePool, d: int, starting: int, *, start: int, zones: List[SubZone], overlay=None):
    sub, created = pool.get_or_create(StructureKey("H1.main", "M15", d, starting))
    assert created
    sub.start_idx = start
    sub.geometry = (SimpleNamespace(events=[], df=None), 0)
    return sub


def _record(pool, sub, lens, start, *, seq, end=None, ttype="first_counter"):
    return _rec(sub, pool=pool, lens=lens, S=0, C=0, tss=seq, trigger_type=ttype, trigger_idx=start,
                start_idx=start, trigger_end_idx=end, end_reason=("same_dir_replacement" if end else None), seq=seq,
                relative_dir="confluence" if sub.direction == 1 else "counter")


def _bos(c: int, inner: float, outer: float, at: int, *steps) -> SubZone:
    return SubZone(c, "BOS", inner, at, ((at, outer),) + tuple(steps))


def _poi(c: int, inner: float, outer: float, at: int) -> SubZone:
    return SubZone(c, "POI", inner, at, ((at, outer),))


class _Run:
    """The overlay over one synthetic cycle; `views[pool_idx]` = its zones; `events[idx]` = callables run when the
    sweep would process idx (a sub's end, a record's start). Ticks in sweep order."""

    def __init__(self, df, scan, pool, views: Dict[int, List[SubZone]], events=None, made=None):
        self.made_calls = []
        self.overlay = ProximityOverlay(
            scans={KEY: scan}, pool=pool, loh=LOH, project=lambda sub, cap: pytest.fail("no projection here"),
            make_triggers=lambda key, lst, h: (self.made_calls.append((key, h)) or (made or {}).get(h, [])),
            log=lambda *_: None,
        )
        self.overlay._views.update(views)
        self.scan = scan
        self.events = events or {}
        self.out = []

    def drive(self):
        for idx, _order, payload in sorted(self.overlay.initial_ticks()):
            for at in sorted(k for k in list(self.events) if k <= idx):
                self.events.pop(at)()
            fired, unresolved = self.overlay.on_tick(payload, idx)
            self.out.append((payload[1], fired, unresolved))
        return [(t.idx, t.direction, t.zone_kind, t.trigger_inner, t.meta["prox_ref_source"]) for t in self.scan.triggers]


def _ref(run: _Run, h: int, side: str):
    return next((e[2] if side == "sd" else e[3]) for e in run.overlay.ref_log if e[1] == h)


# ---------- usability clocks ----------

def test_a_zone_confirmed_in_hour_h_counts_from_h_plus_1():
    """A counter-sub BOS zone with moment M15 41 (H1 hour 10): the tick of candle 10 knows LOH(9) = 39 → not usable;
    the tick of candle 11 knows LOH(10) = 43 → the opp_sd reference."""
    pool = SubStructurePool()
    a = _sub(pool, -1, 20, start=23, zones=[])
    _record(pool, a, CTR, 23, seq=0)
    run = _Run(_df(16), _scan(_df(16), *_cycle()), pool, {a.pool_idx: [_bos(0, 1.4250, 1.4260, 41)]})
    run.drive()
    assert _ref(run, 10, "opp_sd").key == ("parent", "CTS")
    assert _ref(run, 11, "opp_sd").key == ("sub", a.pool_idx, 0, "BOS")


def test_a_sub_is_a_candidate_only_once_its_record_has_started():
    """Zone known at M15 9; the record starts at M15 51 = LOH(12) → the zone counts from candle 13."""
    pool = SubStructurePool()
    a = _sub(pool, -1, 5, start=51, zones=[])
    _record(pool, a, CTR, 51, seq=0)
    run = _Run(_df(16), _scan(_df(16), *_cycle()), pool, {a.pool_idx: [_bos(0, 1.4250, 1.4260, 9)]})
    run.drive()
    assert _ref(run, 12, "opp_sd").key == ("parent", "CTS")
    rung = _ref(run, 13, "opp_sd")
    assert rung.key == ("sub", a.pool_idx, 0, "BOS") and rung.usable_idx == 13


def test_a_zero_length_record_never_makes_a_candidate():
    pool = SubStructurePool()
    a = _sub(pool, -1, 5, start=9, zones=[])
    _record(pool, a, CTR, 11, seq=0, end=11)                  # zero-length: end <= start
    run = _Run(_df(16), _scan(_df(16), *_cycle()), pool, {a.pool_idx: [_bos(0, 1.4250, 1.4260, 9)]})
    run.drive()
    assert all(e[3].key == ("parent", "CTS") for e in run.overlay.ref_log)


# ---------- the rungs: latest source, strict most recent cycle, the sd bound ----------

def test_the_opp_sd_rungs_are_the_newest_counter_subs_most_recent_cycle():
    """Two counter subs; the later-started one (B) becomes the source; of B's cycles only the most recent with a usable
    BOS (c1) — its BOS and its POI (usable from c1's CTS_CONFIRMED)."""
    pool = SubStructurePool()
    a = _sub(pool, -1, 4, start=7, zones=[])
    b = _sub(pool, -1, 8, start=11, zones=[])
    _record(pool, a, CTR, 7, seq=0)
    _record(pool, b, CONF, 11, seq=1, ttype="reversal")
    views = {a.pool_idx: [_bos(0, 1.4260, 1.4270, 7)],
             b.pool_idx: [_bos(0, 1.4250, 1.4260, 9), _bos(1, 1.4240, 1.4245, 15), _poi(1, 1.4220, 1.4230, 21)]}
    run = _Run(_df(16), _scan(_df(16), *_cycle()), pool, views)
    run.drive()
    assert [r.key for r in run.scan.overlay["opp_sd"]] == [("sub", b.pool_idx, 1, "BOS"), ("sub", b.pool_idx, 1, "POI")]
    assert run.scan.refs["opp_sd"].inner == 1.4220


def test_the_sd_overlay_takes_only_zones_beyond_the_parent_poi_and_needs_one():
    """A same-direction sub with a zone inside the parent POI (1.4150 — never) and one beyond it (1.4050 — a rung). With
    no parent POI rung the sd side takes no sub zone at all."""
    pool = SubStructurePool()
    s = _sub(pool, +1, 4, start=7, zones=[])
    _record(pool, s, CONF, 7, seq=0, ttype="first_confluence")
    views = {s.pool_idx: [_bos(0, 1.4150, 1.4140, 7), _poi(0, 1.4050, 1.4040, 9)]}
    run = _Run(_df(16), _scan(_df(16), *_cycle()), pool, views)
    run.drive()
    assert [r.key for r in run.scan.overlay["sd"]] == [("sub", s.pool_idx, 0, "POI")]

    pool2 = SubStructurePool()
    s2 = _sub(pool2, +1, 4, start=7, zones=[])
    _record(pool2, s2, CONF, 7, seq=0, ttype="first_confluence")
    run2 = _Run(_df(16), _scan(_df(16), *_cycle(pois=())), pool2, {s2.pool_idx: views[s.pool_idx]})
    run2.drive()
    assert run2.scan.overlay["sd"] == []


def test_the_sd_source_moves_only_to_a_sub_with_a_usable_qualifying_zone():
    """S1 (qualifying 1.4050) is the sd source; S2, started later, has only a zone inside the parent POI → S1 holds."""
    pool = SubStructurePool()
    s1 = _sub(pool, +1, 4, start=7, zones=[])
    s2 = _sub(pool, +1, 8, start=19, zones=[])
    _record(pool, s1, CONF, 7, seq=0, ttype="first_confluence")
    _record(pool, s2, CONF, 19, seq=1, ttype="subsequent_confluence")
    views = {s1.pool_idx: [_bos(0, 1.4050, 1.4040, 7)], s2.pool_idx: [_bos(0, 1.4150, 1.4140, 19)]}
    run = _Run(_df(16), _scan(_df(16), *_cycle()), pool, views)
    run.drive()
    assert [r.key for r in run.scan.overlay["sd"]] == [("sub", s1.pool_idx, 0, "BOS")]


# ---------- the hold + the frozen source ----------

def test_the_hold_a_replaced_source_keeps_its_zones_until_the_newcomer_has_one_and_never_comes_back():
    """A (source) is replaced at M15 27 by B, whose first zone is known only at M15 37 (H1 9): through candle 9 the
    opp_sd side keeps A's zone; from candle 10 B's. A never comes back (even though it is older-but-still-valid)."""
    pool = SubStructurePool()
    a = _sub(pool, -1, 4, start=7, zones=[])
    b = _sub(pool, -1, 20, start=27, zones=[])
    _record(pool, a, CTR, 7, seq=0, end=27)
    _record(pool, b, CTR, 27, seq=1, ttype="subsequent_counter")
    views = {a.pool_idx: [_bos(0, 1.4250, 1.4260, 7)], b.pool_idx: [_bos(0, 1.4240, 1.4245, 37)]}
    run = _Run(_df(16), _scan(_df(16), *_cycle()), pool, views,
               events={27: lambda: setattr(a, "end_idx", 27)})
    run.drive()
    for h in range(8, 10):
        assert _ref(run, h, "opp_sd").key == ("sub", a.pool_idx, 0, "BOS"), h
    for h in range(10, 16):
        assert _ref(run, h, "opp_sd").key == ("sub", b.pool_idx, 0, "BOS"), h


def test_an_ended_source_is_frozen_its_post_end_cycle_and_outer_steps_are_never_read():
    """A ends at M15 23 (processed then). Its geometry still forms c1 (BOS known at M15 31) and expands c0's outer at
    M15 29 — the ladder keeps c0 and its pre-end outer: a close at 1.4262 (beyond the expanded outer 1.4265? no — beyond
    the frozen 1.4260 only) passes it."""
    pool = SubStructurePool()
    a = _sub(pool, -1, 4, start=7, zones=[])
    _record(pool, a, CTR, 7, seq=0, end=23)
    df = _df(16)
    _set(df, 10, h=1.4263, c=1.4262)
    views = {a.pool_idx: [_bos(0, 1.4250, 1.4260, 7, (29, 1.4265)), _bos(1, 1.4240, 1.4245, 31)]}
    run = _Run(df, _scan(df, *_cycle()), pool, views, events={23: lambda: setattr(a, "end_idx", 23)})
    run.drive()
    assert all(e[3].sub_cycle_id == 0 for e in run.overlay.ref_log if e[3] is not None and e[3].source == "sub")
    assert ("sub", a.pool_idx, 0, "BOS") in run.scan.passed       # frozen outer 1.4260 < close 1.4262


# ---------- the opp_sd unlock (Q2 = (b), Q3 = (ii)) and the permanent sd lock ----------

def _lock_then(newcomer: str):
    """Counter sub A's c0 BOS (inner 1.4250, outer 1.4260) is the opp_sd reference from the scan start. Candle 3 — sd
    expected, so no opp_sd test — closes 1.4270 beyond its outer: passed → from 4 the opp_sd side is LOCKED on the CTS.
    5 fires sd (the POI). Then a NEWER zone is known at M15 33 (usable from candle 9): B's c0 BOS ("sub") or A's own c1
    BOS ("cycle"), inner 1.4245. Candle 10's high 1.4242 reaches it (the CTS would need 1.4292)."""
    pool = SubStructurePool()
    a = _sub(pool, -1, 4, start=7, zones=[])
    _record(pool, a, CTR, 7, seq=0)
    views = {a.pool_idx: [_bos(0, 1.4250, 1.4260, 7)]}
    if newcomer == "sub":
        b = _sub(pool, -1, 20, start=27, zones=[])
        _record(pool, b, CTR, 27, seq=1, ttype="subsequent_counter")
        views[b.pool_idx] = [_bos(0, 1.4245, 1.4255, 33)]
    elif newcomer == "cycle":
        views[a.pool_idx] = views[a.pool_idx] + [_bos(1, 1.4245, 1.4255, 33)]
    df = _df(16)
    _set(df, 3, h=1.4275, c=1.4270)
    _set(df, 5, l=1.4105)
    _set(df, 10, h=1.4242)
    return _Run(df, _scan(df, *_cycle()), pool, views)


@pytest.mark.parametrize("newcomer", ["sub", "cycle"])
def test_an_opp_sd_lock_is_lifted_by_a_newer_zone(newcomer):
    """PLAN_R §1 Locking, Q2 = (b) / Q3 = (ii): a newer counter sub's usable zone, or the source's newer cycle's."""
    run = _lock_then(newcomer)
    trig = run.drive()
    assert _ref(run, 4, "opp_sd").key == ("parent", "CTS")
    assert _ref(run, 9, "opp_sd").inner == 1.4245
    assert trig == [(5, "sd", "POI", 1.41, "parent"), (10, "opp_sd", "BOS", 1.4245, "sub")]
    assert run.scan.locked["opp_sd"] is False


def test_without_a_newer_zone_the_opp_sd_lock_holds():
    run = _lock_then("none")
    trig = run.drive()
    assert run.scan.locked["opp_sd"] is True
    assert trig == [(5, "sd", "POI", 1.41, "parent")]


def test_the_sd_lock_never_reopens():
    """POI passed at 5 → the sd side locked on the BOS; a qualifying sd sub BOS zone (1.4060, beyond the parent POI)
    becomes usable at candle 9 and IS an sd rung — but the sd lock is permanent: 12's low 1.4055, within 8 pips of it,
    does not fire."""
    pool = SubStructurePool()
    s = _sub(pool, +1, 4, start=7, zones=[])
    _record(pool, s, CONF, 7, seq=0, ttype="first_confluence")
    df = _df(16)
    _set(df, 3, l=1.4105)
    _set(df, 5, c=1.4085, l=1.4085)
    _set(df, 7, h=1.4295)
    _set(df, 12, l=1.4055)                 # near the sub zone 1.4050, not the BOS
    run = _Run(df, _scan(df, *_cycle()), pool, {s.pool_idx: [_bos(0, 1.4060, 1.4050, 33)]})
    trig = run.drive()
    assert [r.key for r in run.scan.overlay["sd"]] == [("sub", s.pool_idx, 0, "BOS")]   # the rung IS there
    assert run.scan.locked == {"sd": True, "opp_sd": False}
    assert [t[0] for t in trig] == [3, 7]


# ---------- no M15 bar ----------

def test_an_h1_hour_without_an_m15_bar_steps_but_pushes_nothing():
    pool = SubStructurePool()
    df = _df(12)
    _set(df, 11, l=1.4105)                 # the last candle: sd on the POI
    scan = _scan(df, *_cycle())
    st = SimpleNamespace(trigger_type="first_counter")
    loh = lambda h: min(4 * int(h) + 3, 43)                       # noqa: E731 — H1 11 has no M15 bar (LOH 43 = LOH 10)
    ov = ProximityOverlay(scans={KEY: scan}, pool=pool, loh=loh, project=None,
                          make_triggers=lambda key, lst, h: [("var", st)], log=lambda *_: None)
    results = [ov.on_tick(payload, idx) for idx, _o, payload in sorted(ov.initial_ticks())]
    assert [t.idx for t in scan.triggers] == [11]
    assert results[-1][0] == [] and results[-1][1][0][1] == "no_m15_hour"
    assert ov.generated == [("var", st)]       # kept in the final subsequent_* lists (like a collapsed cycle's)


# ---------- the view ----------

def _ev(t: str, idx: int, c: int):
    return SimpleNamespace(type=t, idx=idx, meta={"cycle_id": c, "confirmed_at": idx, "cts_anchor_idx": idx - 1})


def _kl(c, inner, outer, side, steps):
    return SimpleNamespace(source_kind="BOS", side=side, meta={"cycle_id": c, "inner": inner, "bounds_steps": steps})


def _pz(c, top, bottom, side):
    return SimpleNamespace(top=top, bottom=bottom, side=side, meta={"cycle_id": c})


def test_the_view_reads_each_zone_as_of_its_own_moment_and_a_poi_needs_its_cts_confirmed():
    """Geometry (slice_begin 100): c0 BOS_CONFIRMED 10 + CTS_CONFIRMED 20; c1 BOS_CONFIRMED 30, no CTS_CONFIRMED. The
    projection capped at a moment returns that moment's zones; c1 has a POI in the natural-end projection but no
    CTS_CONFIRMED → never in the view. c0's BOS outer expands at slice 25 (entity 125). c0's POI pick moves AFTER
    its moment (1.418 at the cap, 1.419 at the natural end): the view reads it as of its moment."""
    sub = SimpleNamespace(pool_idx=3, geometry=(SimpleNamespace(events=[
        _ev("BOS_CONFIRMED", 10, 0), _ev("CTS_CONFIRMED", 20, 0), _ev("BOS_CONFIRMED", 30, 1)], df=None), 100))
    c0_steps = [{"event": "INIT", "base_idx": 4, "top": 1.42, "bottom": 1.41},
                {"event": "BOS_THRESHOLD_UPDATED", "expanded_idx": 25, "top": 1.42, "bottom": 1.405}]
    caps = []

    def project(s, cap):
        caps.append(cap)
        kl = [_kl(0, 1.42, 1.41, "buy", c0_steps if cap is None or cap >= 125 else c0_steps[:1])]
        if cap is None or cap >= 130:
            kl.append(_kl(1, 1.43, 1.425, "buy", [{"event": "INIT", "base_idx": 26, "top": 1.43, "bottom": 1.425}]))
        # the pick moves after its moment (only the natural end shows 1.419): the view must keep the as-of 1.418
        poi = [_pz(0, 1.418 if cap is not None else 1.419, 1.415, "buy")] if cap is None or cap >= 120 else []
        if cap is None:
            poi.append(_pz(1, 1.428, 1.426, "buy"))
        return {"kl_zones": kl, "poi_zones": poi}

    zones = build_sub_zone_view(sub, project, log=lambda *_: None)
    assert sorted(set(caps), key=lambda c: -1 if c is None else c) == [None, 110, 120, 130]
    assert zones == [SubZone(0, "BOS", 1.42, 110, ((110, 1.41), (125, 1.405))),
                     SubZone(0, "POI", 1.418, 120, ((120, 1.415),)),
                     SubZone(1, "BOS", 1.43, 130, ((130, 1.425),))]
    assert zones[0].outer_at(124) == 1.41 and zones[0].outer_at(125) == 1.405


def test_the_view_fails_loudly_when_an_inner_moved_after_its_moment():
    sub = SimpleNamespace(pool_idx=3, geometry=(SimpleNamespace(events=[_ev("BOS_CONFIRMED", 10, 0)], df=None), 0))
    steps = [{"event": "INIT", "base_idx": 4, "top": 1.42, "bottom": 1.41}]

    def project(s, cap):
        return {"kl_zones": [_kl(0, 1.42 if cap is not None else 1.421, 1.41, "buy", steps)], "poi_zones": []}

    with pytest.raises(AssertionError, match="inner moved"):
        build_sub_zone_view(sub, project, log=lambda *_: None)


# ---------- the predicted table (PLAN_R §5.4) on the real sweep + overlay, the reference window's real data ----------

def test_the_predicted_table_the_real_sweep_and_overlay_on_the_reference_windows_rows_and_zones():
    """PLAN_R §5.4, end to end on the REAL collaborators except the probes / geometry: the real `run_lifecycle_sweep`
    + the real `ProximityOverlay`, driving the real (0,1) / (1,2) parent scans (S_Q's zones on the real H1 rows,
    `_plan_r_reference_window`), the production `subsequent_*` detectors, and each sub's REAL zones as of their moments
    (the step-2 build's views, embedded). The probes / geometry are stubs returning the predicted starting candles,
    finalizes and natural reversals. Must reproduce, since step 3's proximity reference gate (PLAN_R §9.4): the 8 H1
    triggers (+820 / +872 on (0,1); the 954 relabel on (1,2) — 1045 / 1046 gone: the gate skips every sub_8 zone and
    the superseded sub_6 POI stays the opp_sd reference) and the step-3 shadow's pool table (`build/r6_gate_shadow.log`)
    — 9 subs, 14 records, the 4 collapsed-cycle unresolved."""
    from engine_v2.multitf.lifecycle_sweep import SweepTrigger, run_lifecycle_sweep
    from engine_v2.multitf.sub_structure_pool import resolve_lens
    from engine_v2.multitf.subsequent_confluence_trigger import detect_subsequent_confluence_triggers
    from engine_v2.multitf.subsequent_counter_trigger import detect_subsequent_counter_triggers
    from engine_v2.tests import _plan_r_reference_window as W
    from engine_v2.tests.test_lifecycle_sweep_unit import (
        _geom, _no_limit, _resolver, _rev_resolver, _rs, _synth_stub, _tables, _trig,
    )
    from engine_v2.tests.test_proximity_reference_ladders import PIP, _reference_window_inputs
    from engine_v2.zones.zone_proximity import build_cycle_proximity_scans

    df, events, kl, pz = _reference_window_inputs()
    scans = build_cycle_proximity_scans(df, events, kl, pz, PIP, "H1")
    loh = lambda h: min(4 * int(h) + 3, 4227)                    # noqa: E731 — H1 1057 has no M15 bar
    tables = _tables({(0, 0): (463, 2611), (0, 1): (2611, 3611), (1, 0): (3611, 3611), (1, 1): (3611, 3611),
                      (1, 2): (3611, None)}, {0: +1, 1: -1})
    parent_sd = {0: +1, 1: -1}

    def make(key, lst, h):
        one, out = {key: list(lst)}, []
        for uc, det in (("subsequent_confluence", detect_subsequent_confluence_triggers),
                        ("subsequent_counter", detect_subsequent_counter_triggers)):
            for v in det(events, one, df, parent_tf="H1"):
                if int(v.trigger_event_idx) == int(h):
                    d = parent_sd[key[0]] * (1 if uc == "subsequent_confluence" else -1)
                    out.append((v, SweepTrigger(lens=resolve_lens(uc), parent_sid=key[0], parent_cycle_id=key[1],
                                                trigger_type=uc, trigger_idx=loh(h), direction=d,
                                                trigger_event_idx=int(h), parent_input_idx=int(v.input_idx))))
        return out

    pool = SubStructurePool()
    ov = ProximityOverlay(scans=scans, pool=pool, loh=loh, project=None, make_triggers=make, log=lambda *_: None)
    ov.view = lambda sub: [SubZone(*z) for z in W.SUB_ZONES[(sub.direction, sub.starting_idx)]]
    triggers = [
        _trig(CONF, 0, 0, "first_confluence", 463, +1), _trig(CONF, 0, 1, "first_confluence", 2611, +1),
        _trig(CONF, 1, 0, "first_confluence", 2815, -1), _trig(CONF, 1, 1, "first_confluence", 2995, -1),
        _trig(CONF, 1, 2, "first_confluence", 3611, -1),
        _trig(CTR, 0, 1, "first_counter", 2995, -1), _trig(CTR, 1, 1, "first_counter", 3283, +1),
        _trig(CTR, 1, 2, "first_counter", 3743, +1),
        _trig(CONF, 1, 1, "subsequent_confluence", 3487, -1),      # a collapsed cycle: the batch var 3
    ]
    resolve = _resolver({
        (CONF, 463): _rs(454, 1020), (CONF, 2611): _rs(2365, 2608), (CTR, 2995): _rs(2639, 2995),
        (CONF, 3611): _rs(3304, 3491, cache_hit=True), (CTR, 3743): _rs(3621, 3743),
        (CONF, 3283): _rs(3230, 3283), (CTR, 3491): _rs(3304, 3491), (CONF, 3819): _rs(3760, 3819),
        (CTR, 4143): _rs(4027, 4143), (CONF, 4183): _rs(4179, 4183), (CTR, 4187): _rs(4183, 4187),
    })
    geom = _geom(pool, {(1, 454): 1940, (-1, 1797): 2470, (1, 2365): 2829, (1, 3230): 3589, (-1, 3760): 4200})
    rev = _rev_resolver({(1, 454): 1797, (-1, 1797): 2365, (1, 2365): 2639, (1, 3230): 3304, (-1, 3760): 4027})
    res = run_lifecycle_sweep(triggers, pool=pool, tables=tables, resolve_start=resolve, build_geometry=geom,
                              resolve_reversal=rev, resolve_cross_fib_limit=_no_limit,
                              synth_reversal_trigger=_synth_stub, log=lambda *_: None, proximity=ov)

    fired = {k: [(t.idx, t.direction, t.zone_kind, t.trigger_inner, t.meta["prox_ref_source"]) for t in sc.triggers]
             for k, sc in scans.items()}
    assert fired == {
        (0, 1): [(748, "sd", "POI", 0.57712, "parent"), (820, "opp_sd", "POI", 0.58096, "sub"),
                 (872, "sd", "BOS", 0.57494, "parent")],
        (1, 2): [(935, "sd", "POI", 0.57786, "parent"), (954, "opp_sd", "POI", 0.57306, "sub"),
                 (1035, "sd", "POI", 0.57786, "parent")],
    }
    # The (1,2) reference timeline (§9.4): from 1036 sub_8 (1,4027) is the opp_sd SOURCE but every one of its zones
    # fails the gate, so the superseded sub_6 (1,3621) c0 POI stays the reference to the scan end; the sd side moves out
    # after each pass — the parent POI (passed 1047) → 1048 sub_5 (−1,3304) c0 POI 0.57885 (58) → passed 1050 → 1051
    # the parent BOS 0.57996 (69, locked). No gate blocks any candle.
    by_key = {(s.direction, s.starting_idx): s.pool_idx for s in pool.all()}
    log12 = {e[1]: e for e in ov.ref_log if e[0] == (1, 2)}
    sub6_poi = ("sub", by_key[(1, 3621)], 0, "POI")
    assert all(log12[h][3].key == sub6_poi for h in range(947, 1058))
    assert ov._source[((1, 2), "opp_sd")] == by_key[(1, 4027)]
    assert [(h, log12[h][2].key, log12[h][4]) for h in (1047, 1048, 1050, 1051, 1057)] == [
        (1047, ("parent", "POI", 0), 48), (1048, ("sub", by_key[(-1, 3304)], 0, "POI"), 58),
        (1050, ("sub", by_key[(-1, 3304)], 0, "POI"), 58), (1051, ("parent", "BOS"), 69), (1057, ("parent", "BOS"), 69)]
    assert scans[(1, 2)].locked == {"sd": True, "opp_sd": False}
    assert all(e[6] is None for e in ov.ref_log)

    table = {(s.direction, s.starting_idx): (s.sub_id, s.start_idx, s.end_idx, s.end_reason,
                                             [(r.lens, (r.parent_sid, r.parent_cycle_id), r.trigger_sub_sid,
                                               r.trigger_type, r.trigger_idx, r.start_idx, r.end_idx, r.end_reason)
                                              for r in s.records])
             for s in pool.all()}
    assert table == {   # = the step-3 shadow run's [pool] table (PLAN_R §9.4; build/r6_gate_shadow.log)
        (1, 454): (0, 1020, 1940, "reversal", [(CONF, (0, 0), 0, "first_confluence", 463, 1020, 1940, "reversal")]),
        (-1, 1797): (1, 1940, 2470, "reversal", [(CONF, (0, 0), 1, "reversal", 1940, 1940, 2470, "reversal")]),
        (1, 2365): (2, 2470, 2829, "reversal", [(CONF, (0, 0), 2, "reversal", 2470, 2470, 2611, "parent_end"),
                                                (CONF, (0, 1), 0, "first_confluence", 2611, 2611, 2829, "reversal")]),
        (-1, 2639): (3, 2829, 3491, "same_dir_replacement",
                     [(CONF, (0, 1), 1, "reversal", 2829, 2829, 3589, "same_dir_replacement"),
                      (CTR, (0, 1), 0, "first_counter", 2995, 2995, 3491, "same_dir_replacement")]),
        (1, 3230): (4, 3283, 3589, "reversal",
                    [(CONF, (0, 1), 2, "subsequent_confluence", 3283, 3283, 3589, "reversal")]),
        (-1, 3304): (5, 3491, 3819, "same_dir_replacement",
                     [(CTR, (0, 1), 1, "subsequent_counter", 3491, 3491, 3611, "parent_end"),
                      (CONF, (0, 1), 3, "reversal", 3589, 3589, 3611, "parent_end"),
                      (CONF, (1, 2), 0, "first_confluence", 3611, 3611, 3819, "same_dir_replacement")]),
        (1, 3621): (6, 3743, 4143, "same_dir_replacement",
                    [(CTR, (1, 2), 0, "first_counter", 3743, 3743, 4143, "same_dir_replacement")]),
        (-1, 3760): (7, 3819, 4200, "reversal",
                     [(CONF, (1, 2), 1, "subsequent_confluence", 3819, 3819, 4200, "reversal")]),
        (1, 4027): (8, 4143, None, None,
                    [(CTR, (1, 2), 1, "subsequent_counter", 4143, 4143, None, None),
                     (CONF, (1, 2), 2, "reversal", 4200, 4200, None, None)]),
    }
    # Each replaced sub's pool-wide replacer (the chart's replacement break, 2026-10-24): sub_3's comes from its
    # COUNTER record (its confluence record runs to 3589); sub_5's from its (1,2) CONFLUENCE record — sub_7 is a
    # confluence-only sub, so the counter chart breaks sub_5 at a replacer it does not draw.
    repl = {(s.direction, s.starting_idx): (pool.get_by_id(s.ended_by_sub_id).direction,
                                            pool.get_by_id(s.ended_by_sub_id).starting_idx)
            for s in pool.all() if s.ended_by_sub_id is not None}
    assert repl == {(-1, 2639): (-1, 3304), (-1, 3304): (-1, 3760), (1, 3621): (1, 4027)}
    assert sorted((u.parent_sid, u.parent_cycle_id, u.trigger_type, u.trigger_idx, u.reason)
                  for u in res.unresolved) == [
        (1, 0, "first_confluence", 2815, "collapsed_parent_cycle"),
        (1, 1, "first_confluence", 2995, "collapsed_parent_cycle"),
        (1, 1, "first_counter", 3283, "collapsed_parent_cycle"),
        (1, 1, "subsequent_confluence", 3487, "collapsed_parent_cycle"),
    ]


# ---------- after the sweep: the final lists, the first-sd assert, view == projection ----------

def _zpt(idx, direction, kind, inner, *, source="parent", cycle=None, usable=None):
    from engine_v2.zones.zone_proximity import ZoneProximityTrigger
    return ZoneProximityTrigger(structure_id=0, cycle_id=1, direction=direction, idx=idx, trigger_inner=inner,
                                zone_kind=kind, proximity_pips=8, pip_size=0.0001, timeframe="H1",
                                meta={"prox_ref_gap_pips": 60, "narrow_prox_ref": False, "prox_ref_source": source,
                                      "prox_ref_sub_id": None, "prox_ref_cycle": cycle, "prox_ref_usable_idx": usable})


def _final_fixture():
    from engine_v2.zones.zone_proximity import ProximityRung
    pool = SubStructurePool()
    s = _sub(pool, -1, 2639, start=2829, zones=[])
    s.sub_id = 3
    rung = ProximityRung(key=("sub", s.pool_idx, 2, "POI"), side="opp_sd", zone_kind="POI", inner=0.58096,
                         source="sub", tie_rank=2, sub_pool_idx=s.pool_idx, sub_cycle_id=2, usable_idx=772)
    scan = SimpleNamespace(triggers=[_zpt(748, "sd", "POI", 0.57712),
                                     _zpt(820, "opp_sd", "POI", 0.58096, source="sub", cycle=2, usable=772)])
    ov = SimpleNamespace(scans={(0, 1): scan}, used={((0, 1), 820): rung}, ref_log=[], loh=lambda h: 4 * int(h) + 3)
    return pool, s, rung, ov


def test_the_final_lists_map_the_source_sub_id_and_keep_the_collapsed_batch_cycles():
    from engine_v2.pipeline.orchestrator import _final_proximity_triggers
    pool, _s, _rung, ov = _final_fixture()
    batch = {(0, 1): [_zpt(748, "sd", "POI", 0.57712)], (1, 1): [_zpt(820, "sd", "BOS", 0.58132)]}
    out = _final_proximity_triggers(batch, ov, pool)
    assert list(out) == [(0, 1), (1, 1)]
    assert out[(1, 1)] == batch[(1, 1)]                                    # not overlay-driven: the batch list
    assert [t.meta["prox_ref_sub_id"] for t in out[(0, 1)]] == [None, 3]   # the FINAL id, from the pool idx
    assert ov.scans[(0, 1)].triggers[1].meta["prox_ref_sub_id"] is None    # the scan's own objects untouched


def test_the_first_proximity_trigger_must_be_the_batchs():
    """The main WVMI gate read the batch's first sd per cycle; the sweep's must be the same trigger."""
    from engine_v2.pipeline.orchestrator import _final_proximity_triggers
    pool, _s, _rung, ov = _final_fixture()
    with pytest.raises(AssertionError, match="first proximity trigger"):
        _final_proximity_triggers({(0, 1): [_zpt(749, "sd", "POI", 0.57712)]}, ov, pool)
    with pytest.raises(AssertionError, match="first proximity trigger"):
        _final_proximity_triggers({}, ov, pool)            # the batch had none, the sweep has some


def test_a_sub_zone_a_trigger_used_must_be_in_its_subs_projection():
    from engine_v2.pipeline.orchestrator import _assert_proximity_zones_in_projection
    pool, s, _rung, ov = _final_fixture()
    poi = SimpleNamespace(top=0.58126, bottom=0.58096, side="sell", meta={"cycle_id": 2})
    good = SimpleNamespace(meta={"sub_id": 3, "slice_begin": 0}, kl_zones=[], poi_zones=[poi])
    _assert_proximity_zones_in_projection(ov, pool, [good])
    moved = SimpleNamespace(meta={"sub_id": 3, "slice_begin": 0}, kl_zones=[],
                            poi_zones=[SimpleNamespace(top=0.58126, bottom=0.58101, side="sell", meta={"cycle_id": 2})])
    with pytest.raises(AssertionError, match="read as of its moment"):
        _assert_proximity_zones_in_projection(ov, pool, [moved])


# ---------- the hover (both charts) ----------

def test_the_hover_type_line_names_a_sub_source_and_its_zone_confirmed_idx_is_its_usable_candle():
    from engine_v2.zones.zone_proximity import proximity_trigger_sub_usable_idx, proximity_trigger_type_label
    sub_t = _zpt(820, "opp_sd", "POI", 0.58096, source="sub", cycle=2, usable=772)
    sub_t.meta["prox_ref_sub_id"] = 3
    assert proximity_trigger_type_label(sub_t) == "opp_sd: sub_3 c2 POI"      # the inner: the zone_inner= line
    assert proximity_trigger_sub_usable_idx(sub_t) == 772
    par = _zpt(748, "sd", "POI", 0.57712)
    assert proximity_trigger_type_label(par) == "sd:POI" and proximity_trigger_sub_usable_idx(par) is None


def test_both_charts_branch_on_the_source_before_the_zone_kind(tmp_path, monkeypatch):
    """A sub POI trigger on a parent cycle that HAS a POI: the parent lookup (by sid, cycle, kind) would show that
    POI's floor — the silent mislabel PLAN_R §5.2 names. Both charts must show the sub zone's usable candle and name
    the source on the type line."""
    import contextlib
    import io
    import pandas as pd
    go = pytest.importorskip("plotly.graph_objects")
    from engine_v2.charting.export_m15_chart import _render_proximity_triggers_overlay
    from engine_v2.charting.export_plotly import export_chart_plotly
    from engine_v2.multitf.registry import StructureRegistry
    from engine_v2.tests.test_plan_h_landing_b import _pipeline
    from engine_v2.tests.test_proximity_trigger_hover import _marker_rows
    from engine_v2.tests.test_unified_probe import _make_ip_multicycle_data
    from engine_v2.zones.zone_proximity import ZoneProximityTrigger, poi_unclamped_activation_floor_idx

    res, out, _ = _pipeline(_make_ip_multicycle_data)
    poi = next(z for z in out["poi_zones"] if z.ic_idx == 14)
    sid, cyc = int(poi.meta["structure_id"]), int(poi.meta["cycle_id"])
    usable = poi_unclamped_activation_floor_idx(poi) + 3               # distinct from the parent POI's floor
    trig = ZoneProximityTrigger(structure_id=sid, cycle_id=cyc, direction="opp_sd", idx=poi.ic_idx + 1,
                                trigger_inner=0.61234, zone_kind="POI", proximity_pips=8, pip_size=0.0001,
                                timeframe="H1", meta={"prox_ref_source": "sub", "prox_ref_sub_id": 3,
                                                      "prox_ref_cycle": 2, "prox_ref_usable_idx": usable})
    df = res.df.copy()
    df.attrs.update(structure_events=res.events, kl_zones=out["kl_zones"], poi_zones=[poi],
                    zone_proximity_triggers={(sid, cyc): [trig]})
    reg = StructureRegistry()
    reg.register("H1", df=df, timeframe="H1", role="main")
    figs = []
    monkeypatch.setattr(go.Figure, "write_html", lambda self, *a, **k: figs.append(self))
    monkeypatch.setattr(go.Figure, "write_image", lambda self, *a, **k: None)
    cfg = {"structure": {"levels": False}, "struct_state": {"labels": False},
           "range_visual": {"rectangles": False}, "fib": {"lines": False},
           "imbalance": {"highlight": False}, "zones": {"KL": False, "wave_candles": False, "POI": False},
           "volume": {"bars": False, "ema_line": False, "spike_marker": False}}
    with contextlib.redirect_stdout(io.StringIO()):
        export_chart_plotly(title="t", registry=reg, path_id="H1", out_dir=tmp_path, cfg=cfg)
    fig, = figs
    row, = _marker_rows(fig)
    assert (row[3], row[5], row[7]) == ("opp_sd: sub_3 c2 POI", "0.61234", usable)     # type, zone_inner, conf

    m15 = pd.DataFrame({"time": pd.date_range(pd.to_datetime(df["time"].iloc[0], utc=True), periods=4 * len(df),
                                              freq="15min")})
    m15["o"] = m15["h"] = m15["l"] = m15["c"] = 0.6
    fig = go.Figure()
    with contextlib.redirect_stdout(io.StringIO()):
        _render_proximity_triggers_overlay(fig, m15, df, pd.Series(0.0, index=m15.index))
    row, = _marker_rows(fig)
    assert (row[5], row[7], row[9]) == ("opp_sd: sub_3 c2 POI", "0.61234", usable)


# ---------- the driver (`_run_multi_tf_dual`) wiring: the call sites (PLAN_R §5.2) ----------

class _Stop(Exception):
    pass


def _drive_ticks(proximity):
    """What the real sweep does with the overlay (no sub in these stubs: parent zones only)."""
    for idx, _order, payload in sorted(proximity.initial_ticks()):
        proximity.on_tick(payload, idx)


def _driver(monkeypatch, *, sweep, render=None, stamp=None, var3=(), var4=()):
    """`_run_multi_tf_dual` on the reference window's REAL (0,1) / (1,2) parent zones (so the overlay drives real
    scans) with every M15 collaborator stubbed. Cycles (0,1) / (1,2) live, (1,1) collapsed."""
    import pandas as pd
    import engine_v2.pipeline.orchestrator as orch
    from engine_v2.tests.test_proximity_reference_ladders import _reference_window_inputs

    df, events, kl, pz = _reference_window_inputs()
    df["time"] = pd.date_range("2025-11-17", periods=len(df), freq="h", tz="UTC")
    df.attrs["pair"] = "NZD_USD"
    sentinel = {"batch": True}
    df.attrs["zone_proximity_triggers"] = sentinel
    collapsed = {(1, 1)}
    tables = SimpleNamespace(has_cycle=lambda s, c: True, is_collapsed=lambda s, c: (s, c) in collapsed,
                             cts_moment={(0, 1): 652, (1, 1): 748, (1, 2): 902}, cycles=lambda: [(0, 1), (1, 2)])
    mt = lambda v, _h1: SimpleNamespace(parent_sid=v.parent_sid, parent_cycle_id=v.parent_cycle_id, lower_sd=+1)  # noqa: E731
    for mod in ("first_confluence_pipeline", "subsequent_confluence_pipeline", "subsequent_counter_pipeline"):
        monkeypatch.setattr(f"engine_v2.multitf.{mod}.to_multi_tf_trigger", mt)
    monkeypatch.setattr("engine_v2.multitf.uc1_trigger.detect_uc1_triggers", lambda *a, **k: [])
    monkeypatch.setattr("engine_v2.multitf.data_bridge.fetch_lower_tf_data",
                        lambda *a, **k: pd.DataFrame({"time": pd.date_range("2025-11-17", periods=4 * len(df),
                                                                            freq="15min", tz="UTC")}))
    monkeypatch.setattr("engine_v2.multitf.data_bridge.prepare_lower_tf_data", lambda d: d)
    monkeypatch.setattr("engine_v2.multitf.parent_tables.build_parent_tables", lambda *a, **k: tables)
    monkeypatch.setattr("engine_v2.multitf.entity_df_mutation._map_parent_idx_to_m15_hour_end",
                        lambda p, _h1, _m15: 4 * int(p) + 3)
    monkeypatch.setattr("engine_v2.multitf.cross_fib_limit.check_cross_fib_limits", lambda *a, **k: None)
    monkeypatch.setattr("engine_v2.multitf.lifecycle_sweep.run_lifecycle_sweep", sweep)
    if render is not None:
        monkeypatch.setattr("engine_v2.multitf.entity_df_mutation.render_sub_projection", render)
    monkeypatch.setattr(orch, "_stamp_sub_wvmi_trigger_meta", stamp or (lambda *a, **k: (_ for _ in ()).throw(_Stop())))
    meta = {}
    with pytest.raises(_Stop):
        orch._run_multi_tf_dual(
            df, events, kl, pz, meta, None, first_confluence_triggers=[],
            subsequent_confluence_triggers=list(var3), subsequent_counter_triggers=list(var4),
            main_zone_proximity_triggers={(0, 1): [_zpt(748, "sd", "POI", 0.57712)],
                                          (1, 1): [_zpt(820, "sd", "BOS", 0.58132)],
                                          (1, 2): [_zpt(935, "sd", "POI", 0.57786)]},
        )
    return df, meta, sentinel


def test_the_driver_feeds_the_sweep_only_the_collapsed_cycles_batch_subsequent_triggers(monkeypatch):
    """A driven cycle's subsequent_* come from the overlay — its batch var 3 / var 4 must NOT also reach the sweep
    (they would double); a collapsed cycle's batch ones still do (its unresolved rows)."""
    seen = []

    def sweep(triggers, *, proximity, **_kw):
        seen.extend((t.trigger_type, t.parent_sid, t.parent_cycle_id) for t in triggers)
        assert sorted(proximity.scans) == [(0, 1), (1, 2)]
        _drive_ticks(proximity)
        return SimpleNamespace(unresolved=[], spawned=[])

    v3 = [SimpleNamespace(parent_sid=1, parent_cycle_id=1, trigger_event_idx=871, input_idx=860),
          SimpleNamespace(parent_sid=1, parent_cycle_id=2, trigger_event_idx=954, input_idx=950)]
    _driver(monkeypatch, sweep=sweep, var3=v3)
    assert seen == [("subsequent_confluence", 1, 1)]


def test_the_driver_replaces_the_batch_lists_with_the_final_ones(monkeypatch):
    def sweep(triggers, *, proximity, **_kw):
        _drive_ticks(proximity)
        return SimpleNamespace(unresolved=[], spawned=[])

    collapsed_v3 = [SimpleNamespace(parent_sid=1, parent_cycle_id=1, trigger_event_idx=871, input_idx=860)]
    df, meta, sentinel = _driver(monkeypatch, sweep=sweep, var3=collapsed_v3)
    assert df.attrs["zone_proximity_triggers"] is not sentinel
    assert df.attrs["zone_proximity_triggers"] is meta["zone_proximity_triggers"]
    assert sorted(df.attrs["zone_proximity_triggers"]) == [(0, 1), (1, 1), (1, 2)]
    assert meta["proximity_overlay"].scans[(0, 1)].triggers[0].idx == 748


def test_the_driver_checks_every_used_sub_zone_against_its_projection(monkeypatch):
    """The view == projection assert is CALLED after the render: a used rung whose sub's projection lacks its inner
    stops the replay."""
    from engine_v2.multitf.sub_structure_pool import StructureKey
    from engine_v2.tests.test_render_sub_projection import _record, _set_lifecycle
    from engine_v2.tests.test_sub_wvmi_per_sub import _make_sub
    from engine_v2.zones.zone_proximity import ProximityRung

    def sweep(triggers, *, pool, proximity, **_kw):
        _drive_ticks(proximity)                   # parent zones only: the sub below is the projection-side stub
        sub, _ = pool.get_or_create(StructureKey("H1.main", "M15", -1, 2639))
        _record(pool, sub, None, CTR, start_idx=2829, seq=0, trigger_type="first_counter", parent_sid=0,
                parent_cycle_id=1)
        _set_lifecycle(sub, 2829, 3491, "same_dir_replacement")
        sub.sub_id = 0
        proximity.used[((0, 1), 820)] = ProximityRung(
            key=("sub", sub.pool_idx, 2, "POI"), side="opp_sd", zone_kind="POI", inner=0.58096, source="sub",
            tie_rank=2, sub_pool_idx=sub.pool_idx, sub_cycle_id=2, usable_idx=772)
        return SimpleNamespace(unresolved=[], spawned=[])

    def render(sub, _m15, *, lens_paths, lens_dfs, timeframe, ended_by_starting_idx=None):
        return _make_sub(sub.sub_id, start_idx=sub.start_idx, m15_end_idx=sub.end_idx, lenses=(CTR,), n=1)

    collapsed_v3 = [SimpleNamespace(parent_sid=1, parent_cycle_id=1, trigger_event_idx=871, input_idx=860)]
    with pytest.raises(AssertionError, match="read as of its moment"):
        _driver(monkeypatch, sweep=sweep, render=render, var3=collapsed_v3)


# ---------- the step-2 landing review (2026-10-20): the user's switch rule, the lock basis, the tick order ----------

def test_a_newer_sub_contributing_nothing_reachable_does_not_take_over_the_opp_sd_side():
    """The user, 2026-10-20 ("Contributed + reachable"): newer counter sub B's only zone is a BOS at 1.4320, beyond
    the parent CTS 1.4300 — never the reference. A (1.4250) holds; the side does not fall to the CTS. When B's POI
    1.4245 (inside the CTS) becomes usable (M15 37 → candle 10), B takes over."""
    pool = SubStructurePool()
    a = _sub(pool, -1, 4, start=7, zones=[])
    b = _sub(pool, -1, 12, start=15, zones=[])
    _record(pool, a, CTR, 7, seq=0)
    _record(pool, b, CTR, 15, seq=1, ttype="subsequent_counter")
    views = {a.pool_idx: [_bos(0, 1.4250, 1.4260, 7)],
             b.pool_idx: [_bos(0, 1.4320, 1.4330, 15), _poi(0, 1.4245, 1.4255, 37)]}
    run = _Run(_df(16), _scan(_df(16), *_cycle()), pool, views)
    run.drive()
    for h in range(5, 10):
        assert _ref(run, h, "opp_sd").key == ("sub", a.pool_idx, 0, "BOS"), h
    assert _ref(run, 10, "opp_sd").key == ("sub", b.pool_idx, 0, "POI")


def test_a_newer_sd_sub_whose_latest_cycle_contributes_nothing_does_not_take_over():
    """S1 holds a qualifying 1.4050. Newer S2 (record from M15 19): c0 1.4060 qualifies, but by the time S2's record
    starts its most recent usable cycle is c1 (1.4150, known at 15), inside the parent POI 1.41 → S2 contributes
    nothing → S1 holds (as built before the review, S2 took over on c0 and the sd overlay went empty)."""
    pool = SubStructurePool()
    s1 = _sub(pool, +1, 4, start=7, zones=[])
    s2 = _sub(pool, +1, 8, start=19, zones=[])
    _record(pool, s1, CONF, 7, seq=0, ttype="first_confluence")
    _record(pool, s2, CONF, 19, seq=1, ttype="subsequent_confluence")
    views = {s1.pool_idx: [_bos(0, 1.4050, 1.4040, 7)],
             s2.pool_idx: [_bos(0, 1.4060, 1.4050, 7), _bos(1, 1.4150, 1.4140, 15)]}
    run = _Run(_df(16), _scan(_df(16), *_cycle()), pool, views)
    run.drive()
    assert [r.key for r in run.scan.overlay["sd"]] == [("sub", s1.pool_idx, 0, "BOS")]


def test_an_opp_sd_lock_records_the_passed_zones_basis_not_the_refresh_of_its_tick():
    """A's c0 BOS (1.4250) is passed by candle 3's close; on the tick that locks (4) A's own NEWER cycle c1 is already
    usable — its BOS 1.4320 lies beyond the CTS, so the CTS is the pick and the side locks. The lock must record the
    PASSED zone's basis (A, c0): A's c1 is newer → the lock lifts; when c1's POI 1.4245 (inside the CTS) is usable
    (candle 9), it is the reference and candle 10 fires on it. (Recording the lock tick's own basis (A, c1) kept the
    side locked for good.)"""
    pool = SubStructurePool()
    a = _sub(pool, -1, 4, start=7, zones=[])
    _record(pool, a, CTR, 7, seq=0)
    views = {a.pool_idx: [_bos(0, 1.4250, 1.4260, 7), _bos(1, 1.4320, 1.4330, 15), _poi(1, 1.4245, 1.4255, 33)]}
    df = _df(16)
    _set(df, 3, h=1.4275, c=1.4270)
    _set(df, 5, l=1.4105)
    _set(df, 10, h=1.4242)
    run = _Run(df, _scan(df, *_cycle()), pool, views)
    trig = run.drive()
    assert _ref(run, 4, "opp_sd").key == ("parent", "CTS")
    assert trig == [(5, "sd", "POI", 1.41, "parent"), (10, "opp_sd", "POI", 1.4245, "sub")]


def test_a_zone_is_pass_tested_only_from_the_candle_it_is_usable_the_previous_close_runs_first():
    """PLAN_R §1 "From when (a)": the tick of candle 9 runs close(8) BEFORE the refresh that brings in A's zone
    (known M15 33 = usable from 9), so candle 8's close 1.4265 — beyond its outer 1.4260 — does not pass it; candle 10
    fires on it."""
    pool = SubStructurePool()
    a = _sub(pool, -1, 4, start=7, zones=[])
    _record(pool, a, CTR, 7, seq=0)
    df = _df(16)
    _set(df, 5, l=1.4105)
    _set(df, 8, h=1.4270, c=1.4265)
    _set(df, 10, h=1.4245)
    run = _Run(df, _scan(df, *_cycle()), pool, {a.pool_idx: [_bos(0, 1.4250, 1.4260, 33)]})
    trig = run.drive()
    assert all(h != 8 for _k, h, _keys in run.overlay.pass_log)
    assert trig == [(5, "sd", "POI", 1.41, "parent"), (10, "opp_sd", "BOS", 1.425, "sub")]


# ---------- the view projection on a REAL geometry ----------

def test_project_sub_as_of_is_the_subs_own_projection_capped_at_the_moment(geometry, m15_df):
    """`project_sub_as_of(sub, cap)` = `project_to_window` on the sub's own geometry, floorless, capped at the
    entity-absolute `cap` LOCALIZED to the slice, with the sub's cross-fib limit (the reversing fixture of
    `test_render_sub_projection`, slice_begin 5): a cycle's BOS zone exists capped at its BOS_CONFIRMED moment and not
    one candle before; the zones equal the direct call's."""
    import contextlib
    import io
    from engine_v2.multitf.pooled_structure_build import project_to_window
    from engine_v2.multitf.proximity_overlay import project_sub_as_of
    from engine_v2.structure import event_fields as ef

    _pool, sub = geometry
    bounded, sb = sub.geometry
    bos = [(int(e.meta["cycle_id"]), ef.event_moment(e) + sb) for e in bounded.events if e.type == "BOS_CONFIRMED"]
    assert bos
    with contextlib.redirect_stdout(io.StringIO()):
        for cyc, m in bos:
            at = project_sub_as_of(sub, m)
            before = project_sub_as_of(sub, m - 1)
            direct = project_to_window(bounded, floor=None, cap=m - sb, cap_reason="proximity_view", direction=1,
                                       timeframe="M15", cross_fib_limits={0: sub.cross_fib_limit.localized(sb)})
            has = lambda d: any(z.source_kind == "BOS" and int(z.meta["cycle_id"]) == cyc for z in d["kl_zones"])  # noqa: E731
            assert has(at) and not has(before), (cyc, m)
            assert repr(at["kl_zones"]) == repr(direct["kl_zones"]) and repr(at["poi_zones"]) == repr(direct["poi_zones"])
            assert repr(at["fib_states"]) == repr(direct["fib_states"])


# ---------- the trigger-making closure on the real lists ----------

def test_the_driver_makes_each_subsequent_trigger_once_at_its_own_candle(monkeypatch):
    """`_make_proximity_triggers` on the reference window's parent-only chains: (1,2)'s 954 opp_sd → one var 3 at
    LOH(954), 1035 sd → one var 4 at LOH(1035); (0,1)'s single 748 → none. Nothing re-emitted on a later candle."""
    fired = []

    def sweep(triggers, *, proximity, **_kw):
        for idx, _order, payload in sorted(proximity.initial_ticks()):
            got, _unres = proximity.on_tick(payload, idx)
            fired.extend((t.trigger_type, t.parent_sid, t.parent_cycle_id, t.trigger_idx, t.trigger_event_idx)
                         for t in got)
        return SimpleNamespace(unresolved=[], spawned=[])

    collapsed_v3 = [SimpleNamespace(parent_sid=1, parent_cycle_id=1, trigger_event_idx=871, input_idx=860)]
    _driver(monkeypatch, sweep=sweep, var3=collapsed_v3)
    assert fired == [("subsequent_confluence", 1, 2, 4 * 954 + 3, 954), ("subsequent_counter", 1, 2, 4 * 1035 + 3, 1035)]


def test_the_view_fails_loudly_when_a_bos_confirmed_cycle_has_no_bos_zone():
    """Every BOS_CONFIRMED derives its BOS zone; none in the capped projection = the zone's cycle and the event's
    disagree — never a silent skip (the sub would pass the switch test on its POI yet contribute no BOS)."""
    sub = SimpleNamespace(pool_idx=3, geometry=(SimpleNamespace(events=[_ev("BOS_CONFIRMED", 10, 0)], df=None), 0))

    def project(s, cap):
        return {"kl_zones": [], "poi_zones": []}

    with pytest.raises(AssertionError, match="0 BOS zones"):
        build_sub_zone_view(sub, project, log=lambda *_: None)


@pytest.mark.parametrize("raises", [False, True])
def test_the_view_re_prints_every_captured_line_the_post_replay_grep_flags_even_on_a_crash(raises):
    """The capture re-prints exactly the post-replay grep's lines (`collapsed` and `retry` included), also when the
    projection raises after printing them."""
    from engine_v2.multitf.proximity_overlay import _quiet_project
    logged = []

    def project(s, cap):
        print("[kl] built 3 zones")
        print("WARNING [chart] a collapsed thing")
        print("[x] RETRY 1/2 something")
        if raises:
            raise RuntimeError("boom")
        return {"ok": 1}

    sub = SimpleNamespace(pool_idx=7)
    if raises:
        with pytest.raises(RuntimeError):
            _quiet_project(project, sub, 33, logged.append)
    else:
        assert _quiet_project(project, sub, 33, logged.append) == {"ok": 1}
    assert logged == ["[prox_view] pool#7 cap=33: WARNING [chart] a collapsed thing",
                      "[prox_view] pool#7 cap=33: [x] RETRY 1/2 something"]


def test_the_projection_assert_covers_every_reference_rung_and_a_bos_zones_outer():
    """Not only the zones a trigger fired on: a sub rung that was ever a REFERENCE (here never fired on) must equal the
    sub's projection — its inner, and for a BOS zone its outer at every candle it was the reference (as of LOH(h - 1),
    frozen at the sub's end)."""
    from engine_v2.pipeline.orchestrator import _assert_proximity_zones_in_projection
    from engine_v2.zones.zone_proximity import ProximityRung
    pool = SubStructurePool()
    s = _sub(pool, -1, 2639, start=2829, zones=[])
    s.sub_id = 3
    rung = ProximityRung(key=("sub", s.pool_idx, 1, "BOS"), side="opp_sd", zone_kind="BOS", inner=0.58366,
                         outer_at=lambda i: 0.58424 if 4 * i + 3 < 2900 else 0.58440, source="sub", tie_rank=1,
                         sub_pool_idx=s.pool_idx, sub_cycle_id=1, usable_idx=708)
    ov = SimpleNamespace(used={}, loh=lambda h: 4 * int(h) + 3,
                         ref_log=[((0, 1), 708, None, rung, 65, False, None), ((0, 1), 730, None, rung, 65, False, None)])

    def res(steps, inner=0.58366):
        kl = SimpleNamespace(source_kind="BOS", side="sell", meta={"cycle_id": 1, "inner": inner, "bounds_steps": steps})
        return SimpleNamespace(meta={"sub_id": 3, "slice_begin": 2779}, kl_zones=[kl], poi_zones=[])

    init = {"event": "INIT", "base_idx": 40, "top": 0.58424, "bottom": 0.58366}
    grown = {"event": "BOS_THRESHOLD_UPDATED", "expanded_idx": 2900 - 2779, "top": 0.58440, "bottom": 0.58366}
    _assert_proximity_zones_in_projection(ov, pool, [res([init, grown])])
    with pytest.raises(AssertionError, match="read as of its moment"):
        _assert_proximity_zones_in_projection(ov, pool, [res([init, grown], inner=0.58370)])
    with pytest.raises(AssertionError, match="BOS outer"):
        _assert_proximity_zones_in_projection(ov, pool, [res([init])])     # the projection never expanded
