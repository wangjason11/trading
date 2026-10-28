"""PLAN_T T1 — the first_confluence probe's Phase 2 MS reads what its sub's own MS reads.

Pins: the caller's timeframe values (proximity + narrow-gap; M15 and M5, so a hard-coded "M15" dies), the pair's
pip size, the handed cross-fib limit (REQUIRED with Phase 2; the probe cache calls the lazy limit on a MISS only),
the limit = THE formula for a first_confluence (none on parent cycle 0, the parent's frozen CTS threshold on c >= 1;
a counter-direction first_confluence asserts), and the probe/sub consistency tripwire (log only, one call per sub
CREATED). Under today's cycle-0 carve-out no BEHAVIOUR test can tell the probe's old H1 values from the caller's —
the replay's byte-identity is that argument (PLAN_T §2.3); these are construction pins.
"""
from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace

import pytest

import engine_v2.structure.unified_probe as up
from engine_v2.multitf.cross_fib_limit import NO_LIMIT, first_confluence_probe_limit
from engine_v2.multitf.entity_df_mutation import _probe_with_cache, check_probe_sub_consistency
from engine_v2.multitf.lifecycle_sweep import ResolvedStart
from engine_v2.multitf.sub_structure_pool import LENS_CONFLUENCE as CONF, SubStructurePool
from engine_v2.tests._event_factory import make_cts_established, make_event
from engine_v2.tests.test_lifecycle_sweep_unit import _geom, _resolver, _rs, _run, _tables, _trig
from engine_v2.tests.test_unified_probe import _make_multicycle_data, _prepare_df, _ref_zone_uptrend
from engine_v2.zones.cross_cycle_fib import CrossFibLimit
from engine_v2.zones.zone_proximity import (
    DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS,
    DEFAULT_PROXIMITY_PIPS,
)


def _spy_builds(monkeypatch):
    """Record every MS the probe's Phase 2 builds (+ the kwargs it was built with)."""
    built = []
    real = up._make_market_structure

    def spy(*a, **k):
        ms = real(*a, **k)
        built.append((ms, k))
        return ms

    monkeypatch.setattr(up, "_make_market_structure", spy)
    return built


def _probe(df, timeframe, limit):
    with contextlib.redirect_stdout(io.StringIO()):
        return up.unified_probe(df, 0, 1, _ref_zone_uptrend(), 14, timeframe, enable_phase2=True,
                                cross_fib_limit=limit)


# ---------------------------------------------------------------------------
# The probe's MS: timeframe values, pip size, the handed limit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("timeframe", ["M15", "M5", "H1"])
def test_phase2_ms_gets_the_callers_timeframe_values(monkeypatch, timeframe):
    built = _spy_builds(monkeypatch)
    _probe(_prepare_df(_make_multicycle_data()), timeframe, NO_LIMIT)
    assert built, "the fixture's probe runs Phase 2"
    for ms, kw in built:
        assert kw["timeframe"] == timeframe
        assert ms.proximity_pips == DEFAULT_PROXIMITY_PIPS[timeframe]
        assert ms.min_gap_pips == DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS[timeframe]


@pytest.mark.parametrize("pair, pip", [("NZD_USD", 0.0001), ("USD_JPY", 0.01)])
def test_phase2_ms_gets_the_pairs_pip_size(monkeypatch, pair, pip):
    built = _spy_builds(monkeypatch)
    df = _prepare_df(_make_multicycle_data())
    df.attrs["pair"] = pair
    _probe(df, "M15", NO_LIMIT)
    assert built and all(ms.pip_size == pip for ms, _ in built)


def test_phase2_ms_gets_the_handed_limit_object(monkeypatch):
    built = _spy_builds(monkeypatch)
    handed = CrossFibLimit(None, "none")
    _probe(_prepare_df(_make_multicycle_data()), "M15", handed)
    assert built and all(kw["cross_fib_limit"] is handed for _, kw in built)


def test_phase2_without_a_limit_raises_and_phase1_needs_none():
    df = _prepare_df(_make_multicycle_data())
    with contextlib.redirect_stdout(io.StringIO()):
        with pytest.raises(AssertionError, match="cross-fib limit"):
            up.unified_probe(df, 0, 1, _ref_zone_uptrend(), 14, "M15", enable_phase2=True)
        res = up.unified_probe(df, 0, 1, _ref_zone_uptrend(), 14, "M15")   # deterministic only: no MS, no limit
    assert res.status == "finalized"


# ---------------------------------------------------------------------------
# The probe cache: the lazy limit is called on a MISS only, and handed to the probe
# ---------------------------------------------------------------------------

def test_the_probe_cache_calls_the_limit_on_a_miss_only(monkeypatch):
    seen = []

    def fake_probe(df, **kw):
        seen.append(kw["cross_fib_limit"])
        return SimpleNamespace(starting_idx=5, status="finalized", finalize_condition="second_cts_reached",
                               iterations=1, bos0_inner=0.6, finalize_idx=9)

    monkeypatch.setattr(up, "unified_probe", fake_probe)
    handed = CrossFibLimit(None, "none")
    calls = []

    def limit():
        calls.append(1)
        return handed

    pool = SubStructurePool()
    kw = dict(m15_df=None, parent_path="H1.main", sub_tf="M15", direction=1, input_idx=3,
              reference_zone=_ref_zone_uptrend(), probe_end_idx=20, enable_phase2=True, timeframe="M15",
              label="t", cross_fib_limit=limit)
    with contextlib.redirect_stdout(io.StringIO()):
        first = _probe_with_cache(pool, **kw)
        second = _probe_with_cache(pool, **kw)          # same key → a cache hit: no probe, no limit
    assert first[-1] is False and second[-1] is True
    assert calls == [1] and seen == [handed]


# ---------------------------------------------------------------------------
# The limit: THE formula for a first_confluence, behind a direction assert
# ---------------------------------------------------------------------------

def _fc(S, C, d, idx=40):
    return _trig(CONF, S, C, "first_confluence", idx, d)


def test_the_first_confluence_limit_is_none_on_parent_cycle_0():
    tables = _tables({(0, 0): (3, None)}, {0: 1})
    assert first_confluence_probe_limit(_fc(0, 0, 1), h1_events=[], parent_tables=tables)() == NO_LIMIT


def test_the_first_confluence_limit_is_the_parent_cts_threshold_on_cycle_1():
    tables = _tables({(0, 0): (3, 43), (0, 1): (43, None)}, {0: 1})
    est = make_cts_established(cts_anchor_idx=10, confirmed_at=11, price=0.6120, cycle_id=1,
                               meta={"cts_frozen": 0.6100})
    conf = SimpleNamespace(type="CTS_CONFIRMED", price=0.6080, meta={"structure_id": 0, "cycle_id": 0})
    lim = first_confluence_probe_limit(_fc(0, 1, 1), h1_events=[est, conf], parent_tables=tables)()
    assert (lim.price, lim.source, lim.ref_anchor_price) == (0.6100, "parent_cts_threshold", 0.6080)


def test_the_first_confluence_limit_is_lazy():
    """The formula runs when the limit is CALLED (a probe-cache miss), not when it is built: on a parent cycle >= 1
    with no events, building does not raise and calling hits `parent_cts_limit`'s own assert (PLAN_T §2.2.2)."""
    tables = _tables({(0, 0): (3, 43), (0, 1): (43, None)}, {0: 1})
    limit = first_confluence_probe_limit(_fc(0, 1, 1), h1_events=[], parent_tables=tables)
    with pytest.raises(AssertionError, match="CTS_ESTABLISHED"):
        limit()


def test_a_counter_direction_first_confluence_asserts():
    tables = _tables({(0, 0): (3, None)}, {0: 1})
    with pytest.raises(AssertionError, match="is not the parent's"):
        first_confluence_probe_limit(_fc(0, 0, -1), h1_events=[], parent_tables=tables)


# ---------------------------------------------------------------------------
# The probe/sub consistency tripwire
# ---------------------------------------------------------------------------

def _sub(moments, slice_begin=100, conf0=None):
    """A sub whose geometry holds CTS_ESTABLISHED at the given (anchor, moment) pairs (slice-local) and, with
    `conf0`, cycle 0's CTS_CONFIRMED at that slice-local candle."""
    events = [make_cts_established(cts_anchor_idx=a, confirmed_at=m, cycle_id=i) for i, (a, m) in enumerate(moments)]
    if conf0 is not None:
        events.append(make_event("CTS_CONFIRMED", conf0, structure_id=0, cycle_id=0, struct_direction=1,
                                 cts_anchor_idx=4, confirmed_at=conf0))
    return SimpleNamespace(geometry=(SimpleNamespace(events=events), slice_begin), pool_idx=0,
                           key=("H1.main", "M15", 1, 100))


def _res(finalize, cond="second_cts_reached", cache_hit=False):
    return ResolvedStart(starting_idx=100, parent_bos_anchor_idx=None, bos0_inner=0.6, finalize_idx=finalize,
                         finalize_condition=cond, probe_input_idx=90, cache_hit=cache_hit)


FC = SimpleNamespace(trigger_type="first_confluence", trigger_idx=50)


@pytest.mark.parametrize("trigger, res, sub, warns", [
    (FC, _res(120), _sub([(4, 5), (19, 20)]), False),                                   # 20 + 100 == 120
    (FC, _res(121), _sub([(4, 5), (19, 20)]), True),                                    # off by one
    (FC, _res(120), _sub([(19, 20), (4, 5)]), False),                                   # anchor order, not list order
    (FC, _res(120), _sub([(4, 5)]), True),                                              # the sub has one CTS
    (FC, _res(121, cache_hit=True), _sub([(4, 5), (19, 20)]), False),                   # no probe ran
    (FC, _res(121, cond="no_retrace"), _sub([(4, 5), (19, 20)]), False),                # no cycle-0 confirmation
    (FC, _res(121, cond="no_retrace"), _sub([(4, 5)], conf0=10), True),                 # the sub confirmed at 110
    (FC, _res(110, cond="no_retrace"), _sub([(4, 5)], conf0=10), False),                # = the probe's
    (FC, _res(105, cond="no_retrace"), _sub([(4, 5)], conf0=10), False),                # sub after: not caught (§2.2.4)
    (FC, _res(121, cond="end_idx_reached"), _sub([(4, 5)], conf0=10), False),           # other conditions: unchecked
    (SimpleNamespace(trigger_type="first_counter", trigger_idx=50), _res(121), _sub([(4, 5), (19, 20)]), False),
])
def test_the_consistency_tripwire(trigger, res, sub, warns):
    logs = []
    check_probe_sub_consistency(trigger, res, sub, log=logs.append)
    assert bool(logs) is warns
    assert all(line.startswith("WARNING [sweep] probe/sub disagree on ") for line in logs)


def test_the_sweep_calls_the_check_once_per_sub_created(monkeypatch):
    import engine_v2.tests.test_lifecycle_sweep_unit as lsu
    pool = SubStructurePool()
    tables = _tables({(0, 0): (3, None), (0, 1): (43, None)}, {0: 1})
    triggers = [_trig(CONF, 0, 0, "first_confluence", 463, 1), _trig(CONF, 0, 1, "first_confluence", 611, 1)]
    resolve = _resolver({(CONF, 463): _rs(454, 470), (CONF, 611): _rs(454, 620)})   # both land on one key
    checked = []
    real = lsu.run_lifecycle_sweep

    def with_check(*a, **k):
        return real(*a, check_probe_sub_consistency=lambda t, r, s, log: checked.append((t.trigger_idx, s.pool_idx)),
                    **k)

    monkeypatch.setattr(lsu, "run_lifecycle_sweep", with_check)
    _run(triggers, pool=pool, tables=tables, resolve=resolve, geom=_geom(pool, {}))
    assert checked == [(463, 0)]                       # the second trigger's start hits the pool: no check


# ---------------------------------------------------------------------------
# The orchestrator's wiring: a first_confluence's resolver gets the lazy limit; the sweep gets the check
# ---------------------------------------------------------------------------

def test_the_orchestrator_hands_a_first_confluence_its_limit_and_the_sweep_its_check(monkeypatch):
    import pandas as pd
    from engine_v2.multitf.entity_df_mutation import check_probe_sub_consistency as the_check
    from engine_v2.pipeline.orchestrator import _run_multi_tf_dual

    class _Stop(Exception):
        pass

    seen, flags, built = [], [], []

    def _fake_resolve(trigger, _h1, _m15, *, pool, hi, m15_parent_cts_established_idx, parent_path,
                      cross_fib_limit, cycle0_proximity_blocked):
        seen.append(cross_fib_limit)
        flags.append(cycle0_proximity_blocked)

    def _fake_bog(pool, m15, **k):
        built.append(k["cycle0_proximity_blocked"])

    def _sweep(triggers, *, resolve_start, build_geometry, check_probe_sub_consistency, **_kw):
        assert check_probe_sub_consistency is the_check
        for i, t in enumerate(triggers):          # PLAN_T T2: the sweep's one evaluation, passed through as handed
            resolve_start(t, int(t.trigger_idx), cycle0_proximity_blocked=(i == 0))
        key = SimpleNamespace(parent_path="H1.main", sub_tf="M15", direction=1, starting_idx=7)
        for flag in (True, False, None):
            build_geometry(key, None, NO_LIMIT, cycle0_proximity_blocked=flag)
        raise _Stop

    monkeypatch.setattr("engine_v2.multitf.first_confluence_pipeline.to_multi_tf_trigger",
                        lambda v, _h1: SimpleNamespace(parent_sid=v.parent_sid, parent_cycle_id=v.parent_cycle_id,
                                                       lower_sd=+1))
    monkeypatch.setattr("engine_v2.multitf.uc1_trigger.detect_uc1_triggers", lambda *a, **k: [])
    monkeypatch.setattr("engine_v2.multitf.data_bridge.fetch_lower_tf_data", lambda *a, **k: pd.DataFrame({"time": [0]}))
    monkeypatch.setattr("engine_v2.multitf.data_bridge.prepare_lower_tf_data", lambda df, **_k: df)
    monkeypatch.setattr("engine_v2.multitf.parent_tables.build_parent_tables",
                        lambda *a, **k: SimpleNamespace(cts_moment={(0, 0): 5, (0, 1): 20}, floor=lambda s, c: 0,
                                                        parent_sd={0: 1}))
    monkeypatch.setattr("engine_v2.multitf.entity_df_mutation._map_parent_idx_to_m15_hour_end",
                        lambda p, _h1, _m15: 4 * int(p) + 3)
    monkeypatch.setattr("engine_v2.multitf.entity_df_mutation._resolve_trigger_m15_start", _fake_resolve)
    monkeypatch.setattr("engine_v2.multitf.lifecycle_sweep.run_lifecycle_sweep", _sweep)
    monkeypatch.setattr("engine_v2.multitf.entity_df_mutation.build_or_get_geometry", _fake_bog)

    h1 = pd.DataFrame({"time": pd.date_range("2025-11-17", periods=40, freq="h", tz="UTC")})
    v1 = SimpleNamespace(parent_sid=0, parent_cycle_id=0, trigger_event_idx=10, input_idx=5, status="finalized")
    v1b = SimpleNamespace(parent_sid=0, parent_cycle_id=1, trigger_event_idx=25, input_idx=21, status="finalized")
    est = make_cts_established(cts_anchor_idx=22, confirmed_at=23, price=0.6120, cycle_id=1, meta={"cts_frozen": 0.6100})
    conf = make_event("CTS_CONFIRMED", 18, price=0.6080, structure_id=0, cycle_id=0, struct_direction=1,
                      cts_anchor_idx=12, confirmed_at=18)
    with pytest.raises(_Stop):
        _run_multi_tf_dual(h1, [conf, est], [], [], {}, None, first_confluence_triggers=[v1, v1b],
                           subsequent_confluence_triggers=[], subsequent_counter_triggers=[])
    limit0, limit1 = seen
    assert callable(limit0) and limit0() == NO_LIMIT       # parent cycle 0: no limit, by THE formula
    lim = limit1()                                          # parent cycle 1: read from the events handed in
    assert (lim.price, lim.source, lim.ref_anchor_price) == (0.6100, "parent_cts_threshold", 0.6080)
    assert flags == [True, False] and built == [True, False, None]   # PLAN_T T2's flag, as the sweep handed it


def test_the_dispatcher_and_the_fc_resolver_pass_the_limit_to_the_probe_cache(monkeypatch):
    import pandas as pd
    import engine_v2.multitf.data_bridge as db
    import engine_v2.multitf.entity_df_mutation as edm

    seen = {}

    def spy_cache(pool, **kw):
        seen["limit"] = kw["cross_fib_limit"]
        seen["phase2"] = kw["enable_phase2"]
        return (7, 9, "second_cts_reached", 0.6, "finalized", False)

    monkeypatch.setattr(edm, "_probe_with_cache", spy_cache)
    monkeypatch.setattr(edm, "_build_first_confluence_ref_zone", lambda *a, **k: _ref_zone_uptrend())
    monkeypatch.setattr(db, "map_candle_to_lower_tf", lambda t, sd, m15: 3 if sd == -1 else 12)
    parent = pd.DataFrame({"time": pd.date_range("2025-11-17", periods=10, freq="h", tz="UTC")})
    trig = SimpleNamespace(use_case="first_confluence", parent_sid=0, parent_cycle_id=0, lower_sd=+1, lower_tf="M15",
                           meta={"parent_input_idx": 2, "parent_cts_anchor_idx": 5})

    def limit():
        return NO_LIMIT

    with contextlib.redirect_stdout(io.StringIO()):
        res = edm._resolve_trigger_m15_start(trig, parent, pd.DataFrame(), pool=None, hi=40,
                                             m15_parent_cts_established_idx=0, cross_fib_limit=limit)
    assert isinstance(res, ResolvedStart) and res.starting_idx == 7
    assert seen == {"limit": limit, "phase2": True}
