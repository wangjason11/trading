"""PLAN_T T2 — the cycle-0 carve-out on the main sid 0 and the sub its cycle-0 first_confluence creates only.

Pins: the gate itself (cycle 0 blocked / unblocked, cycle k > 0, the narrow-gap boundary `>=`), a real cycle 0 that
confirms by proximity only when unblocked, `compute_structure` blocking sid 0 and not sid 1 (a construction spy —
a Scenario-1 behaviour fixture is impractical), the predicate's truth table, the sweep's ONE evaluation handed to the
probe and to the geometry (None on a pool hit) + the creator tripwire, the pool builder's contract, and the probe
chain carrying the flag to Phase 2's MS. The `/compare` cannot kill the predicate / `structure_id` mutants on the
window (PLAN_T §3.4) — these pins must.
"""
from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace

import pytest

import engine_v2.structure.structure_engine as se
import engine_v2.structure.unified_probe as up
from engine_v2.multitf.lifecycle_sweep import cycle0_proximity_blocked_for
from engine_v2.multitf.sub_structure_pool import LENS_CONFLUENCE as CONF, LENS_COUNTER as CTR, SubStructurePool
from engine_v2.structure.market_structure import MarketStructure, Point
from engine_v2.tests import test_ms_bounded_equals_truncated as _bet
from engine_v2.tests._cross_fib_limits import NO_P
from engine_v2.tests.test_lifecycle_sweep_unit import _geom, _resolver, _rs, _run, _tables, _trig
from engine_v2.tests.test_main_reversal_probe import _make_reversing_data, _prepare_df as _prepare_main
from engine_v2.tests.test_unified_probe import _make_multicycle_data, _prepare_df, _ref_zone_uptrend


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


# ---------------------------------------------------------------------------
# The gate (`_maybe_confirm_cts_via_proximity`), in isolation
# ---------------------------------------------------------------------------

def _gate_ms(monkeypatch, *, blocked, cycle, gap_pips, min_gap_pips=30):
    """A direct-construction MS whose state is set for the gate: an established CTS at candle 5 (price 0.6000 +
    `gap_pips` pips) over a BOS threshold of 0.6000, a BOS inner, `cycle`; the proximity check stubbed to hit. Returns
    (ms, fired) — `fired` records each confirmation the gate lets through."""
    ms = MarketStructure(_prepare_df(_make_multicycle_data()), 1, min_gap_pips=min_gap_pips,
                         cycle0_proximity_blocked=blocked)
    st = ms.state
    st.cts = Point(idx=5, price=0.6000 + gap_pips * 0.0001)
    st.bos_threshold = 0.6000
    st.cts_phase = "EST_OR_UPD"
    st.cts_cycle_id = cycle
    st.bos_inner_for_cycle = 0.6005
    fired = []
    monkeypatch.setattr(ms, "_check_proximity_at_candle", lambda i: (0.6005, "BOS"))
    monkeypatch.setattr(ms, "_fire_cts_confirmation_via_proximity", lambda i, inner, kind: fired.append(i))
    return ms, fired


@pytest.mark.parametrize("blocked, cycle, fires", [
    (True, 0, False),     # the carve-out
    (False, 0, True),     # cycle 0 guarded by the narrow-gap gate alone
    (True, 1, True),      # a cycle k > 0 never reads the flag
    (False, 1, True),
])
def test_the_gate_reads_the_flag_on_cycle_0_only(monkeypatch, blocked, cycle, fires):
    ms, fired = _gate_ms(monkeypatch, blocked=blocked, cycle=cycle, gap_pips=40)
    ms._maybe_confirm_cts_via_proximity(6)
    assert (fired == [6]) is fires


@pytest.mark.parametrize("under, fires", [(0.0, True), (1e-12, False)])
def test_the_narrow_gap_boundary_is_inclusive(monkeypatch, under, fires):
    """|CTS − BOS threshold| >= min_gap: a gap of EXACTLY the floor confirms, a hair under does not (no existing pin
    held the `>=`, the review). The gap is built from the threshold itself (BOS threshold 0.0, CTS = the threshold)
    so float rounding cannot put it a hair above the floor. Cycle 0, unblocked — the gate T2 opens."""
    ms, fired = _gate_ms(monkeypatch, blocked=False, cycle=0, gap_pips=40)
    ms.state.bos_threshold = 0.0
    ms.state.cts = Point(idx=5, price=ms._min_gap_threshold - under)
    ms._maybe_confirm_cts_via_proximity(6)
    assert (fired == [6]) is fires


def test_direct_construction_keeps_the_pre_plan_t_rule():
    """MarketStructure's own default = the universal carve-out (direct construction, the unit tests); production
    builds go through `_make_market_structure`, where the flag is REQUIRED."""
    assert MarketStructure(_prepare_df(_make_multicycle_data()), 1).cycle0_proximity_blocked is True


# ---------------------------------------------------------------------------
# A real cycle 0: confirms by proximity only when unblocked
# ---------------------------------------------------------------------------

def _cycle0_confirmations(blocked):
    res = _quiet(se.compute_bounded_structure, _prepare_df(_bet._make_l5_bos_inner_at_bound()), 0, -1,
                 cross_fib_limit=NO_P, cycle0_proximity_blocked=blocked)
    return [(int(e.idx), e.meta.get("via")) for e in res.events
            if e.type == "CTS_CONFIRMED" and e.meta.get("cycle_id") == 0]


def test_a_real_cycle_0_confirms_by_proximity_only_when_unblocked():
    unblocked, blocked = _cycle0_confirmations(False), _cycle0_confirmations(True)
    assert unblocked and unblocked[0][1] == "sd_zone_proximity"
    assert all(via != "sd_zone_proximity" for _, via in blocked)


@pytest.mark.parametrize("sd, prox, pattern", [(1, 3, 4), (-1, 9, 10)])
def test_blocked_the_pattern_confirms_unblocked_proximity_then_reconfirmed(sd, prox, pattern):
    """`_make_l4_watch_at_bound` on M15 values, both sides: blocked — cycle 0 confirms by its pullback pattern, no
    CTS_RECONFIRMED; unblocked — proximity confirms a candle earlier and the pattern becomes CTS_RECONFIRMED (the
    shape subs 4, 7, 8 take on the reference window, PLAN_T §3.3)."""
    def run(blocked):
        res = _quiet(se.compute_bounded_structure, _prepare_df(_bet._make_l4_watch_at_bound()), 0, sd,
                     timeframe="M15", cross_fib_limit=NO_P, cycle0_proximity_blocked=blocked)
        return [(int(e.idx), e.type, e.meta.get("via") or e.meta.get("confirmation_method")) for e in res.events
                if e.type in ("CTS_CONFIRMED", "CTS_RECONFIRMED") and e.meta.get("cycle_id") == 0]

    blocked, unblocked = run(True), run(False)
    assert [(i, t) for i, t, _ in blocked] == [(pattern, "CTS_CONFIRMED")] and blocked[0][2] != "sd_zone_proximity"
    assert [(i, t) for i, t, _ in unblocked] == [(prox, "CTS_CONFIRMED"), (pattern, "CTS_RECONFIRMED")]
    assert unblocked[0][2] == "sd_zone_proximity"


def test_compute_bounded_structure_defaults_to_unblocked():
    """The rule for every sub but one: a bounded run without the flag = unblocked."""
    res = _quiet(se.compute_bounded_structure, _prepare_df(_bet._make_l5_bos_inner_at_bound()), 0, -1,
                 cross_fib_limit=NO_P)
    assert [(int(e.idx), e.meta.get("via")) for e in res.events
            if e.type == "CTS_CONFIRMED" and e.meta.get("cycle_id") == 0] == _cycle0_confirmations(False)


# ---------------------------------------------------------------------------
# The main: sid 0 blocked, a reversal's sid not
# ---------------------------------------------------------------------------

def test_compute_structure_blocks_sid_0_and_not_the_reversal(monkeypatch):
    seen = []
    real = se._make_market_structure

    def spy(df, struct_direction, **k):
        seen.append((int(k.get("structure_id", 0)), k["cycle0_proximity_blocked"]))
        return real(df, struct_direction, **k)

    monkeypatch.setattr(se, "_make_market_structure", spy)
    res = _quiet(se.compute_structure, _prepare_main(_make_reversing_data()))
    assert max(int(e.meta["structure_id"]) for e in res.events) >= 1, "the fixture reverses"
    assert seen[0] == (0, True) and all(not b for sid, b in seen[1:] if sid >= 1)
    assert {sid for sid, _ in seen} >= {0, 1}


def test_the_builder_requires_the_flag():
    with pytest.raises(TypeError, match="cycle0_proximity_blocked"):
        se._make_market_structure(_prepare_df(_make_multicycle_data()), 1, cross_fib_limit=NO_P)


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ttype, cycle, path, blocked", [
    ("first_confluence", (0, 0), "H1.main", True),
    ("first_confluence", (0, 1), "H1.main", False),
    ("first_confluence", (1, 0), "H1.main", False),
    ("subsequent_confluence", (0, 0), "H1.main", False),
    ("first_counter", (0, 0), "H1.main", False),
    ("reversal", (0, 0), "H1.main", False),
    ("first_confluence", (0, 0), "M15.confluence", False),
])
def test_the_predicate(ttype, cycle, path, blocked):
    assert cycle0_proximity_blocked_for(ttype, cycle, path) is blocked


# ---------------------------------------------------------------------------
# The sweep: one evaluation, handed to the probe and the geometry; the creator tripwire
# ---------------------------------------------------------------------------

def _spying(resolve, geom):
    seen = {"probe": [], "geometry": []}

    def resolve_start(t, hi, *, cycle0_proximity_blocked=False):
        seen["probe"].append((t.trigger_idx, cycle0_proximity_blocked))
        return resolve(t, hi)

    def build_geometry(key, bos0_inner, cross_fib_limit, *, cycle0_proximity_blocked=None):
        seen["geometry"].append((key.starting_idx, cycle0_proximity_blocked))
        return geom(key, bos0_inner, cross_fib_limit)

    return resolve_start, build_geometry, seen


def test_the_sweep_hands_one_evaluation_to_the_probe_and_the_geometry():
    pool = SubStructurePool()
    tables = _tables({(0, 0): (3, None), (0, 1): (43, None)}, {0: 1})
    triggers = [_trig(CONF, 0, 0, "first_confluence", 463, 1),   # the carve-out's sub: creates 454
                _trig(CONF, 0, 1, "first_confluence", 611, 1),   # another FC: lands on 454 (a pool hit)
                _trig(CONF, 0, 1, "subsequent_confluence", 640, 1)]   # creates 600
    resolve, geom, seen = _spying(_resolver({(CONF, 463): _rs(454, 470), (CONF, 611): _rs(454, 620),
                                             (CONF, 640): _rs(600, 640)}), _geom(pool, {}))
    _, logs = _run(triggers, pool=pool, tables=tables, resolve=resolve, geom=geom)
    assert seen["probe"] == [(463, True), (611, False), (640, False)]
    assert seen["geometry"] == [(454, True), (454, None), (600, False)]   # a pool hit: no flag
    assert not [line for line in logs if "did not create its sub" in line]


@pytest.mark.parametrize("cache_hit, pre_built", [(True, False), (False, True)])
def test_the_creator_tripwire(cache_hit, pre_built):
    """The premise "the main's earliest trigger creates the carve-out's sub with its own probe", checked: a
    probe-cache hit, or a start that lands on an existing sub, logs one WARNING."""
    pool = SubStructurePool()
    tables = _tables({(0, 0): (3, None)}, {0: 1})
    triggers = [_trig(CONF, 0, 0, "first_confluence", 463, 1)]
    mapping = {(CONF, 463): _rs(454, 470, cache_hit=cache_hit)}
    if pre_built:
        # an earlier trigger (stubbed) creates the same pool key (direction +1, start 454) first
        triggers = [_trig(CTR, 0, 0, "first_counter", 455, 1), _trig(CONF, 0, 0, "first_confluence", 463, 1)]
        mapping[(CTR, 455)] = _rs(454, 455)
    _, logs = _run(triggers, pool=pool, tables=tables, resolve=_resolver(mapping), geom=_geom(pool, {}))
    warns = [line for line in logs if "did not create its sub" in line]
    assert len(warns) == 1 and "(0, 0)@463" in warns[0]


# ---------------------------------------------------------------------------
# The pool builder's contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flag", [True, False])
def test_the_pool_builder_stores_and_guards_the_flag(monkeypatch, flag):
    import engine_v2.multitf.entity_df_mutation as edm
    handed = []

    def fake_geometry(m15_df, **k):
        handed.append(k["cycle0_proximity_blocked"])
        return (SimpleNamespace(reversal_idx=None, events=[]), 0)

    monkeypatch.setattr(edm, "_build_geometry", fake_geometry)
    import pandas as pd
    pool = SubStructurePool()
    m15 = pd.DataFrame({"time": range(40)})
    kw = dict(parent_path="H1.main", sd=1, start_abs=10, bos0_inner=None, timeframe="M15")
    with pytest.raises(AssertionError, match="no cycle-0 carve-out flag"):
        _quiet(edm.build_or_get_geometry, pool, m15, cross_fib_limit=NO_P, cycle0_proximity_blocked=None, **kw)
    sub, created = _quiet(edm.build_or_get_geometry, pool, m15, cross_fib_limit=NO_P, cycle0_proximity_blocked=flag,
                          **kw)
    assert created and sub.cycle0_proximity_blocked is flag and handed == [flag]
    with pytest.raises(AssertionError, match="carve-out flag handed in on a pool hit"):
        _quiet(edm.build_or_get_geometry, pool, m15, cross_fib_limit=None, cycle0_proximity_blocked=False, **kw)
    again, created = _quiet(edm.build_or_get_geometry, pool, m15, cross_fib_limit=None, cycle0_proximity_blocked=None,
                            **kw)
    assert again is sub and not created and handed == [flag]


def test_the_geometry_core_hands_the_flag_to_the_bounded_run(monkeypatch):
    import engine_v2.multitf.entity_df_mutation as edm
    import engine_v2.structure.structure_engine as se_mod
    seen = []
    real = se_mod.compute_bounded_structure

    def spy(*a, **k):
        seen.append(k["cycle0_proximity_blocked"])
        return real(*a, **k)

    monkeypatch.setattr(se_mod, "compute_bounded_structure", spy)
    df = _prepare_df(_make_multicycle_data())
    for flag in (True, False):
        _quiet(edm._build_geometry, df, sd=1, start_abs=0, run_cap_abs=len(df) - 1, bos0_inner=None,
               timeframe="M15", cross_fib_limit=NO_P, cycle0_proximity_blocked=flag)
    assert seen == [True, False]


# ---------------------------------------------------------------------------
# The probe chain: the flag reaches Phase 2's MS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flag", [True, False])
def test_phase2_ms_gets_the_flag(monkeypatch, flag):
    built = []
    real = up._make_market_structure

    def spy(*a, **k):
        built.append(k["cycle0_proximity_blocked"])
        return real(*a, **k)

    monkeypatch.setattr(up, "_make_market_structure", spy)
    _quiet(up.unified_probe, _prepare_df(_make_multicycle_data()), 0, 1, _ref_zone_uptrend(), 14, "M15",
           enable_phase2=True, cross_fib_limit=NO_P, cycle0_proximity_blocked=flag)
    assert built and set(built) == {flag}


@pytest.mark.parametrize("flag", [True, False])
def test_the_probe_cache_and_the_fc_resolver_pass_the_flag(monkeypatch, flag):
    import pandas as pd
    import engine_v2.multitf.data_bridge as db
    import engine_v2.multitf.entity_df_mutation as edm
    seen = {}

    def fake_probe(df, **kw):
        seen["probe"] = kw["cycle0_proximity_blocked"]
        return SimpleNamespace(starting_idx=5, status="finalized", finalize_condition="second_cts_reached",
                               iterations=1, bos0_inner=0.6, finalize_idx=9)

    monkeypatch.setattr(up, "unified_probe", fake_probe)
    monkeypatch.setattr(edm, "_build_first_confluence_ref_zone", lambda *a, **k: _ref_zone_uptrend())
    monkeypatch.setattr(db, "map_candle_to_lower_tf", lambda t, sd, m15: 3 if sd == -1 else 12)
    parent = pd.DataFrame({"time": pd.date_range("2025-11-17", periods=10, freq="h", tz="UTC")})
    trig = SimpleNamespace(use_case="first_confluence", parent_sid=0, parent_cycle_id=0, lower_sd=+1, lower_tf="M15",
                           meta={"parent_input_idx": 2, "parent_cts_anchor_idx": 5})
    _quiet(edm._resolve_trigger_m15_start, trig, parent, pd.DataFrame(), pool=SubStructurePool(), hi=40,
           m15_parent_cts_established_idx=0, cross_fib_limit=lambda: NO_P, cycle0_proximity_blocked=flag)
    assert seen == {"probe": flag}


def test_a_sibling_trigger_with_the_flag_asserts():
    import pandas as pd
    import engine_v2.multitf.entity_df_mutation as edm
    trig = SimpleNamespace(use_case="subsequent_confluence", parent_sid=0, parent_cycle_id=0, lower_sd=+1,
                           lower_tf="M15", meta={})
    with pytest.raises(AssertionError, match="only a first_confluence keeps the carve-out"):
        edm._resolve_trigger_m15_start(trig, pd.DataFrame(), pd.DataFrame(), pool=SubStructurePool(), hi=40,
                                       m15_parent_cts_established_idx=0, cycle0_proximity_blocked=True)
