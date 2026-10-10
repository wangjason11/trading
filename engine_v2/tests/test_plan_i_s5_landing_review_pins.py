"""PLAN_I S5 landing review, lens B (mutation) — the pins for its 19 surviving gap mutants (72 mutants on the full
suite: 49 killed, 4 equivalent, 19 gaps; each pin below names the mutant(s) it kills, verified one mutant at a time).
Lens A's MAJOR (the threshold feed and the sub geometry's `P` unpinned) is the same pair: `test_ms_feeds_every_
threshold_update`, `test_the_geometry_hands_its_limit_to_the_embedded_tracker`. Record: PLAN_I §9.10, S5."""
from __future__ import annotations

import contextlib
import io

import pytest

import engine_v2.structure.structure_engine as se
from engine_v2.tests._cross_fib_limits import NO_P, NO_REF_TRACKER, unreached
from engine_v2.tests.test_plan_i_s5_ms_reads_the_fib import (
    _ev_conf, _ev_upd, _geometry, _quiet, _reversing, _rows, embedded,  # noqa: F401  (embedded = fixture)
)
from engine_v2.tests.test_unified_probe import _prepare_df
from engine_v2.zones.fib_tracker import FibEventOrderError, FibTracker


# --- ms_refresh_sid0: the read asks the tracker for THIS run's structure_id ---
def test_the_read_asks_for_this_runs_sid(monkeypatch):
    seen = []
    real = se.poi_inner_for_fib

    def spy(df, fib, ft, **kw):
        seen.append(fib)
        return real(df, fib, ft, **kw)

    monkeypatch.setattr(se, "poi_inner_for_fib", spy)
    ms = se._make_market_structure(_prepare_df(_rows(1)), 1, structure_id=3, cross_fib_limit=NO_P)
    _quiet(ms.run)
    fibs = [f for f in seen if f is not None]
    assert fibs and {f.structure_id for f in fibs} == {3}


# --- edm_geom_no_forward / edm_bog_no_forward: a sub geometry's MS gets the pool's P ---
@pytest.mark.parametrize("via", ["build_geometry", "build_or_get_geometry"])
def test_the_geometry_hands_its_limit_to_the_embedded_tracker(embedded, via):
    from engine_v2.multitf import entity_df_mutation as edm
    from engine_v2.multitf.sub_structure_pool import SubStructurePool
    df, inner = _reversing()
    lim = unreached(1)
    if via == "build_geometry":
        assert _geometry(df, inner, lim) is not None
    else:
        assert _quiet(edm.build_or_get_geometry, SubStructurePool(), df, parent_path="H1.main", sd=1, start_abs=0,
                      bos0_inner=inner, cross_fib_limit=lim, timeframe="M15") is not None
    t, = embedded
    assert t._limits == {0: lim}


# --- ms_refresh_skip_read_no_fib: a refresh with NO fib still calls the read (the stamp, F11) ---
def test_a_no_fib_refresh_still_reads_the_stamp(monkeypatch):
    from engine_v2.zones.poi_zones import ICSelectionError
    monkeypatch.setattr(FibTracker, "representative_fib", lambda self, sid, cycle: None)
    df = _prepare_df(_rows(1))
    del df.attrs["timeframe"]
    with pytest.raises(ICSelectionError) as exc:
        _quiet(se.compute_bounded_structure, df, 0, 1, cross_fib_limit=NO_P)
    assert isinstance(exc.value.__cause__, KeyError)


# --- ft_order_stamped_idx_only: an idx tie is broken by the TYPE (processing_order_key), not the feed order ---
def test_check_processing_order_breaks_an_idx_tie_by_type():
    t = FibTracker(cross_fib_limits={0: NO_P})
    t._handled[:] = [_ev_conf(23, 23), _ev_upd(23)]          # CTS_CONFIRMED < CTS_UPDATED: in order
    t.check_processing_order()
    t._handled[:] = [_ev_upd(23), _ev_conf(23, 23)]
    with pytest.raises(FibEventOrderError, match=r"#0: fed CTS_UPDATED"):
        t.check_processing_order()


# --- ft_quiet_forced_true: the DOWNSTREAM tracker (quiet=False, the default) still logs its decisions ---
def test_the_downstream_tracker_logs():
    from engine_v2.pipeline.orchestrator import _run_downstream_pipeline
    res = _quiet(se.compute_bounded_structure, _prepare_df(_rows(1)), 0, 1, cross_fib_limit=unreached(1))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        _run_downstream_pipeline(res.df, res.events, 1, wvmi="off", cross_fib_limits={0: unreached(1)})
    assert "[fib]" in out.getvalue()


# --- pz_fill_thr: the in-flight read runs the IP pick at the threshold it was handed (the tracker's) ---
def test_the_read_runs_the_pick_at_its_fill_threshold(monkeypatch):
    import engine_v2.zones.poi_zones as poi_zones
    from engine_v2.tests.test_plan_h_landing_b import _fib, _m1_frame
    got = []
    real = poi_zones.select_ic
    monkeypatch.setattr(poi_zones, "select_ic",
                        lambda df, fs, cfg, **kw: got.append(cfg.fill_threshold) or real(df, fs, cfg, **kw))
    fs = poi_zones.FibState(structure_id=0, cycle_id=1, struct_direction=1, bos_idx=30, bos_price=1.0, cts_idx=40,
                            cts_price=1.35, active=True, locked=False, fib=_fib(1.35, 1.0))
    poi_zones.poi_inner_for_fib(_m1_frame(), fs, 0.42, tracker=NO_REF_TRACKER, evaluated_at=40)
    assert got == [0.42]


# --- sl_one_rev_assert_removed: the PER-SID helper (compute_structure's call) asserts one reversal of sid N-1 ---
def test_the_per_sid_limit_asserts_one_reversal_of_the_previous_sid():
    from engine_v2.tests.test_plan_i_s2_cross_fib_limit import _main_events
    from engine_v2.zones.structure_lifecycle import compute_main_cross_fib_limit
    with pytest.raises(AssertionError, match="two reversals for main sid 0"):
        compute_main_cross_fib_limit(_main_events(second_reversal=True), 1)


# --- sl_map_assert_removed: the map's assert covers the LAST sid (no sid N+1 consults its reversals) ---
def test_the_map_asserts_one_reversal_of_the_last_sid():
    from engine_v2.tests.test_plan_i_s2_cross_fib_limit import _ev, _main_events
    from engine_v2.zones.structure_lifecycle import compute_main_cross_fib_limits
    evs = _main_events() + [_ev("STATE_CHANGED", 995, None, structure_id=2, to="reversal", bos_frozen=0.59),
                            _ev("STATE_CHANGED", 996, None, structure_id=2, to="reversal", bos_frozen=0.59)]
    with pytest.raises(AssertionError, match="two reversals for one main sid"):
        compute_main_cross_fib_limits(evs)


# --- sl_watch_lt: the open watch is the last one AT OR BEFORE the reversal (a watch on the reversal candle counts) ---
def test_a_watch_on_the_reversal_candle_is_the_open_watch():
    from engine_v2.tests.test_plan_i_s2_cross_fib_limit import _ev, _est, _watch
    from engine_v2.zones.structure_lifecycle import compute_main_cross_fib_limit
    from engine_v2.tests._event_factory import make_bos_confirmed
    evs = [_est(115, 0.6, 0, 0, -1),
           make_bos_confirmed(bos_anchor_idx=96, confirmed_at=115, price=0.61, structure_id=0, cycle_id=0,
                              struct_direction=-1),                      # PLAN_K A9: a reversal needs a BOS
           _watch(880, 0.5701), _watch(902, 0.5736),
           _ev("STATE_CHANGED", 902, None, structure_id=0, to="reversal", pattern_anchor_idx=900, bos_frozen=0.5736),
           _est(903, 0.58, 1, 0, 1)]
    assert compute_main_cross_fib_limit(evs, 1).price == 0.5736


# --- ms_drop_feed_thr / ft_handled_thr: MS feeds (and the tracker records) every CTS_THRESHOLD_UPDATED ---
_FIB_TYPES = ("CTS_ESTABLISHED", "CTS_UPDATED", "CTS_CONFIRMED", "CTS_THRESHOLD_UPDATED")


def test_ms_feeds_every_threshold_update(embedded):
    """`test_fib_never_established_cap`'s climb: ~14 range-sync CTS_THRESHOLD_UPDATEDs after CTS_0 CONFIRMED, the first
    pre-creating the cycle-1 cross (only a THR handler creates it) — the S5 agreement fixture has none."""
    from engine_v2.pipeline.orchestrator import _run_downstream_pipeline
    from engine_v2.tests.test_fib_never_established_cap import _rows as climb_rows
    from engine_v2.tests.test_plan_i_s5_ms_reads_the_fib import _sig
    lim = unreached(1)
    res = _quiet(se.compute_bounded_structure, _prepare_df(climb_rows()), 0, 1, timeframe="M15", cross_fib_limit=lim)
    ms_t, = embedded
    fed = [e for e in res.events if e.type in _FIB_TYPES]
    assert sum(e.type == "CTS_THRESHOLD_UPDATED" for e in fed) >= 10
    assert [id(e) for e in ms_t._handled] == [id(e) for e in fed]
    assert (0, 1, "cross", 0) in ms_t._fibs
    ds_t = _quiet(_run_downstream_pipeline, res.df, res.events, 1, wvmi="off", cross_fib_limits={0: lim})["fib_tracker"]
    assert sorted(map(_sig, ms_t.get_all_fibs())) == sorted(map(_sig, ds_t.get_all_fibs()))


# --- ft_rep_*: representative_fib = the ACTIVE latest cross, else the ACTIVE single, else None ---
def _fs(active, cross):
    from engine_v2.zones.fib_tracker import FibState
    return FibState(structure_id=0, cycle_id=1, struct_direction=1, bos_idx=1, bos_price=1.0, cts_idx=5,
                    cts_price=1.1, active=active, locked=False, fib=None, meta={"cross_cycle": True} if cross else {})


@pytest.mark.parametrize("cross, single, want", [
    (True, None, "cross"), (True, False, "cross"), (False, True, "single"), (None, True, "single"),
    (False, None, None), (None, False, None), (False, False, None), (None, None, None)])
def test_representative_fib_is_the_active_cross_else_the_active_single(cross, single, want):
    t = FibTracker(cross_fib_limits={0: NO_P}, quiet=True)
    recs = {}
    if cross is not None:
        t._cross_version[(0, 1)] = 0
        t._fibs[(0, 1, "cross", 0)] = recs["cross"] = _fs(cross, True)
    if single is not None:
        t._fibs[(0, 1)] = recs["single"] = _fs(single, False)
    assert t.representative_fib(0, 1) is (recs[want] if want else None)


# --- ms_refresh_drop_inner: the read's inner reaches the proximity check as the cycle's POI inner ---
def test_the_read_inner_reaches_the_proximity_check(monkeypatch):
    import engine_v2.structure.market_structure as msm
    seen = []
    real = msm._check_sd_proximity_at_candle

    def spy(*a, **k):
        seen.append(k.get("poi_inners"))
        return real(*a, **k)

    monkeypatch.setattr(msm, "_check_sd_proximity_at_candle", spy)
    monkeypatch.setattr(se, "poi_inner_for_fib", lambda df, fib, ft, **kw: None if fib is None else 0.4242)
    _quiet(se.compute_bounded_structure, _prepare_df(_rows(1)), 0, 1, cross_fib_limit=NO_P)
    assert [0.4242] in seen


# --- or_fill_thr: the downstream tracker is built with the one fill constant and the default levels ---
def test_the_downstream_tracker_uses_the_one_config():
    from engine_v2.features.fibonacci import DEFAULT_FIB_LEVELS
    from engine_v2.pipeline.orchestrator import _run_downstream_pipeline
    from engine_v2.zones.fib_tracker import FIB_FILL_THRESHOLD
    res = _quiet(se.compute_bounded_structure, _prepare_df(_rows(1)), 0, 1, cross_fib_limit=NO_P)
    t = _quiet(_run_downstream_pipeline, res.df, res.events, 1, wvmi="off", cross_fib_limits={0: NO_P})["fib_tracker"]
    assert (t.config.fill_threshold, t.config.fib_levels) == (FIB_FILL_THRESHOLD, list(DEFAULT_FIB_LEVELS))


# --- ms_refresh_full_df / ms_feed_upd_full_df: a bounded run hands the read and the tracker the TRUNCATED frame ---
def test_a_bounded_runs_read_and_feed_get_the_truncated_frame(monkeypatch):
    lens = []
    real_read = se.poi_inner_for_fib
    monkeypatch.setattr(se, "poi_inner_for_fib",
                        lambda d, f, t, **kw: lens.append(("read", len(d))) or real_read(d, f, t, **kw))
    for h in ("on_cts_established", "on_cts_updated", "on_cts_threshold_updated"):
        real_h = getattr(FibTracker, h)
        monkeypatch.setattr(FibTracker, h, lambda self, ev, df, *a, _h=h, _r=real_h: lens.append((_h, len(df))) or _r(self, ev, df, *a))
    df = _prepare_df(_rows(1))
    end = len(df) - 4
    _quiet(se.compute_bounded_structure, df, 0, 1, end_idx=end, cross_fib_limit=NO_P)
    kinds = {k for k, _ in lens}
    assert {"read", "on_cts_established", "on_cts_updated"} <= kinds
    assert {n for _, n in lens} == {end + 1}


@pytest.mark.parametrize("sd", [1, -1])
def test_a_reached_poi_inner_confirms_the_cts(monkeypatch, sd):
    """Behavioural twin of the above: an inner every candle reaches (+inf for a buy POI's top, -inf for a sell POI's
    bottom) wins over the BOS inner and confirms a cycle >= 1 CTS via proximity with `zone_kind` "POI"."""
    import math
    monkeypatch.setattr(se, "poi_inner_for_fib", lambda df, fib, ft, **kw: None if fib is None else sd * math.inf)
    res = _quiet(se.compute_bounded_structure, _prepare_df(_rows(sd)), 0, sd, cross_fib_limit=NO_P)
    assert any(e.type == "CTS_CONFIRMED" and e.meta.get("zone_kind") == "POI" for e in res.events)
