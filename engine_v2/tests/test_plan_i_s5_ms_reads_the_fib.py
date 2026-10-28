"""PLAN_I S5 (§9.3 D5, Q7): MarketStructure's in-flight POI read is the SAME fib decision as the drawn fib.

`structure_engine._make_market_structure` builds an EMBEDDED FibTracker (the downstream's logic and fill threshold,
quiet, the structure's REQUIRED cross-fib limit) and `poi_zones.poi_inner_for_fib`; MS feeds the tracker each CTS
event as it is emitted (CTS_ESTABLISHED after its cycle's BOS_CONFIRMED, every CTS_UPDATED, CTS_CONFIRMED,
CTS_THRESHOLD_UPDATED), reads `representative_fib(sid, cycle)` at every POI refresh, and calls
`check_processing_order()` at the end of every run — the order tripwire (R9-1: it RAISES on the F16 / F17 (i)
shapes; MS produces neither since PLAN_J J1 / J2 — the tripwire is a guard, its unit pinned here).

Pinned here: the agreement with the downstream tracker (the V5 sequence check, on a fixture with a cross, sd ±1);
the read takes the representative fib (the cross where the limit allows one, the single where it does not); the
tripwire on the REAL F16 candles (NZD_USD M15, 2025-03-24) and its unit behaviour; a tracker error inside a sub
geometry propagates; the probe's explicit "no P" and its independence from the read (F7); the wiring (quiet tracker,
one fill constant, the REQUIRED limit, main sid N's limit from the sids before it).
"""
from __future__ import annotations

import contextlib
import io
import math

import pytest

import engine_v2.structure.event_fields as ef
import engine_v2.structure.structure_engine as se
from engine_v2.pipeline.orchestrator import _run_downstream_pipeline
from engine_v2.structure.market_structure import EmbeddedFibTrackerError
from engine_v2.tests._cross_fib_limits import NO_P, limit, unreached
from engine_v2.tests._event_factory import make_cts_updated, make_event
from engine_v2.tests.test_ms_cts_update_no_regress import _mirror
from engine_v2.tests.test_unified_probe import _make_ip_multicycle_data, _prepare_df
from engine_v2.zones.cross_cycle_fib import CrossFibLimit
from engine_v2.common.types import IMBALANCE_FILL_THRESHOLD
from engine_v2.zones.fib_tracker import (
    CrossFibLimitError, FibEventOrderError, FibTracker, FibTrackerConfig,
)
from engine_v2.zones.poi_zones import POIConfig


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


@pytest.fixture
def embedded(monkeypatch):
    """Every FibTracker `structure_engine` builds (MS's embedded ones), in construction order."""
    made = []

    class _Rec(FibTracker):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    monkeypatch.setattr(se, "FibTracker", _Rec)
    return made


def _rows(sd):
    rows = _make_ip_multicycle_data()
    return rows if sd == 1 else _mirror(rows)


def _sig(f):
    return (f.structure_id, f.cycle_id, f.meta.get("version"), f.meta.get("cross_cycle"), f.bos_idx, f.cts_idx,
            f.active, f.locked)


# ---------------------------------------------------------------------------
# The agreement — the V5 sequence check on a fixture with a cross (§9.5 d's evidence, on fixtures)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sd", [1, -1])
@pytest.mark.parametrize("limit", ["unreached", "no_p"])
def test_ms_embedded_tracker_agrees_with_the_downstream(embedded, sd, limit):
    """MS's embedded tracker handles exactly the events the downstream tracker handles, in the same order, and ends
    on the same fib records (anchors / version / active / locked) — with a cross (unreached `P`: cycle 1 crosses
    back to cycle 0) and without (no `P`: every cycle a single)."""
    lim = unreached(sd) if limit == "unreached" else NO_P
    res = _quiet(se.compute_bounded_structure, _prepare_df(_rows(sd)), 0, sd, cross_fib_limit=lim)
    ms_t, = embedded
    out = _quiet(_run_downstream_pipeline, res.df, res.events, sd, wvmi="off", cross_fib_limits={0: lim})
    ds_t = out["fib_tracker"]
    assert len(ms_t._handled) == 13 and [id(e) for e in ms_t._handled] == [id(e) for e in ds_t._handled]
    assert sorted(map(_sig, ms_t.get_all_fibs())) == sorted(map(_sig, ds_t.get_all_fibs()))
    crosses = [k for k in ms_t._fibs if len(k) == 4]
    assert crosses == ([(0, 1, "cross", 0)] if limit == "unreached" else [])


@pytest.mark.parametrize("sd", [1, -1])
def test_the_read_takes_the_representative_fib(monkeypatch, embedded, sd):
    """At every refresh MS hands `poi_inner_for_fib` its tracker's representative fib of the CURRENT cycle: cycle 1's
    cross where the limit allows it, its single where there is no `P` (the retired resolver read the intra-cycle
    fib for every sub cycle — PLAN_H D5)."""
    seen = []
    real = se.poi_inner_for_fib

    def spy(df, fib, ft, **kw):
        seen.append(None if fib is None else (fib.cycle_id, fib.meta.get("cross_cycle") is not None, fib.bos_idx))
        assert ft == IMBALANCE_FILL_THRESHOLD
        return real(df, fib, ft, **kw)

    monkeypatch.setattr(se, "poi_inner_for_fib", spy)
    for lim, cross in ((unreached(sd), True), (NO_P, False)):
        seen.clear()
        res = _quiet(se.compute_bounded_structure, _prepare_df(_rows(sd)), 0, sd, cross_fib_limit=lim)
        c1 = [x for x in seen if x is not None and x[0] == 1]
        assert c1 and {x[1] for x in c1} == {cross}
        bos0 = [ef.bos_anchor_idx(e) for e in res.events if e.type == "BOS_CONFIRMED" and e.meta["cycle_id"] == 0]
        bos1 = [ef.bos_anchor_idx(e) for e in res.events if e.type == "BOS_CONFIRMED" and e.meta["cycle_id"] == 1]
        assert {x[2] for x in c1} == set(bos0 if cross else bos1)   # the cross spans from BOS_0


def test_no_tracker_means_the_bos_inner_only():
    """`MarketStructure(fib_tracker=None)` (direct construction) reads no POI inner — exactly as the retired
    `poi_inners_resolver=None`; the tracker and the read come together."""
    from engine_v2.structure.market_structure import MarketStructure
    df = _prepare_df(_rows(1))
    ms = MarketStructure(df, 1)
    _quiet(ms.run)
    assert ms.state.poi_inners_for_cycle == []
    with pytest.raises(AssertionError, match="come together"):
        MarketStructure(df, 1, fib_tracker=FibTracker(cross_fib_limits={0: NO_P}))


# ---------------------------------------------------------------------------
# The order tripwire (D5, R9-1)
# ---------------------------------------------------------------------------

# PLAN_I §9.2 F16 / §5 R9-1: NZD_USD M15, 2025-03-24 07:45 → 15:00 UTC (rows 5515-5544 of the session-4 REAL source;
# exact candles, no jitter), a bearish run started at row 11 (10:30). The range candidate 20 (12:45, confirmed 25)
# back-filled the raw CTS to 24 (13:45, low 0.57191); the re-stepped pullback anchored 22 applied at 23 (13:30) and
# confirmed the CTS with anchor 24 — emission UPD(24), CONF(23); processing order CONF(23), UPD(24): S5's end-of-run
# tripwire raised. Since PLAN_J J1 the range back-fill's look-ahead is undone: the re-steps read the state as of
# themselves, so the pullback at 13:30 confirms the CTS known at 13:30 (21, 13:00, 0.5723) and 24 is a
# post-confirmation candle.
_F16_ROWS = [
    (0.57314, 0.57334, 0.57304, 0.57331), (0.57333, 0.57376, 0.57332, 0.57362), (0.57364, 0.57401, 0.57354, 0.57394),
    (0.57393, 0.57398, 0.57334, 0.57364), (0.57366, 0.57374, 0.57324, 0.57332), (0.57335, 0.57364, 0.57322, 0.5736),
    (0.57358, 0.57385, 0.57346, 0.57378), (0.57381, 0.57401, 0.57366, 0.57397), (0.57396, 0.57433, 0.57394, 0.57417),
    (0.57418, 0.57455, 0.57402, 0.5745), (0.57452, 0.57472, 0.57447, 0.5747), (0.57471, 0.5748, 0.57442, 0.5745),
    (0.5745, 0.5745, 0.57418, 0.57422), (0.57425, 0.57426, 0.57402, 0.57413), (0.57414, 0.5742, 0.5739, 0.57395),
    (0.57397, 0.57398, 0.57367, 0.5738), (0.57378, 0.57389, 0.57334, 0.57334), (0.57334, 0.57352, 0.57292, 0.57292),
    (0.57292, 0.57298, 0.57262, 0.57278), (0.57278, 0.57279, 0.57236, 0.5725), (0.57249, 0.57256, 0.57234, 0.57256),
    (0.57257, 0.57258, 0.5723, 0.57254), (0.57254, 0.5728, 0.5724, 0.57278), (0.5728, 0.5731, 0.57268, 0.57303),
    (0.57301, 0.57301, 0.57191, 0.57192), (0.57192, 0.57259, 0.57182, 0.57248), (0.5725, 0.57362, 0.5724, 0.57321),
    (0.5732, 0.5732, 0.57254, 0.57265), (0.57266, 0.57274, 0.57224, 0.5725), (0.57252, 0.57254, 0.57206, 0.57227),
]


def _f16_ms():
    # the candles as pinned (PLAN_J J1), classified with the fixed 2.2-pip floor of that time: 30 rows hold no
    # rolling window (PLAN_U U2), so the floor is explicit
    df = _prepare_df([dict(o=o, h=h, l=l, c=c) for o, h, l, c in _F16_ROWS], body_floor_pips=2.2)
    return se._make_market_structure(df, struct_direction=-1, start_idx=11, structure_id=0, timeframe="M15",
                                     pip_size=se._pip_size_from_pair(df), cycle0_proximity_blocked=True, cross_fib_limit=NO_P)


def test_the_f16_candles_confirm_the_cts_known_at_the_confirmation():
    """PLAN_J J1 (was `test_the_f16_shape_raises_at_the_end_of_the_run`): the run completes (no order tripwire);
    the pullback applying at 23 (13:30) confirms CTS 21 (13:00, 0.5723); candle 24 (13:45, low 0.57191) is a
    post-confirmation candle — its low widens the range and moves the threshold AT 24, no raw CTS update; the range
    (candidate 20, confirmed 25) starts with the CTS as of 20 (19, 12:30)."""
    ms = _f16_ms()
    _quiet(ms.run)
    conf, = [e for e in ms.events if e.type == "CTS_CONFIRMED"]
    assert (int(conf.idx), ef.cts_anchor_idx(conf), conf.price, conf.meta["via"]) == (23, 21, 0.5723, "one_maru_continuous")
    assert not [e for e in ms.events if e.type == "CTS_UPDATED" and ef.stamped_idx(e) > 23]
    thr = [(int(e.idx), e.price) for e in ms.events if e.type == "CTS_THRESHOLD_UPDATED"]
    assert thr[:2] == [(24, 0.57191), (25, 0.57182)]
    rng, = [e for e in ms.events if e.type == "RANGE_STARTED"]
    assert (rng.meta["range_anchor_idx"], rng.meta["confirmed_at"], rng.meta["cts_anchor_idx"]) == (20, 25, 19)
    # The feed is complete and in emission order (landing review lens A M1 (b)) — CTS_THRESHOLD_UPDATEDs included —
    # and that order IS the processing order now.
    fed = [e for e in ms.events if e.type in _FED]
    assert any(e.type == "CTS_THRESHOLD_UPDATED" for e in fed)
    assert [id(e) for e in ms._fib_tracker._handled] == [id(e) for e in fed]
    assert [id(e) for e in sorted(fed, key=ef.processing_order_key)] == [id(e) for e in fed]
    assert "market_state" in ms.df.columns and (ms.df["market_state"].iloc[11:] != "").all()


_ATTR = {"structure_id": 0, "cycle_id": 0, "struct_direction": 1}


_FED = ("CTS_ESTABLISHED", "CTS_UPDATED", "CTS_CONFIRMED", "CTS_THRESHOLD_UPDATED")


def _ev_upd(idx):
    return make_cts_updated(cts_anchor_idx=idx, via=ef.CTS_UPDATED_RAW_VIA, price=1.0, meta=dict(_ATTR))


def _ev_conf(idx, anchor):
    return make_event("CTS_CONFIRMED", idx, price=1.0, cts_anchor_idx=anchor, confirmed_at=idx, **_ATTR)


def test_check_processing_order_unit():
    """Feed order == `processing_order_key` order passes (equal keys keep the feed order — a stable sort); the
    first event out of place is named."""
    t = FibTracker(cross_fib_limits={0: NO_P})
    upd24 = _ev_upd(24)
    conf23 = _ev_conf(23, 24)
    t._handled[:] = [_ev_upd(21), conf23, upd24]
    t.check_processing_order()
    t._handled[:] = [_ev_upd(21), _ev_upd(22), _ev_upd(22)]          # a tie keeps the feed order
    t.check_processing_order()
    t._handled[:] = [_ev_upd(21), upd24, conf23]
    with pytest.raises(FibEventOrderError, match=r"handled event #1: fed CTS_UPDATED \(cycle 0, idx 24"):
        t.check_processing_order()
    assert issubclass(FibEventOrderError, RuntimeError)


def test_every_public_handler_records_its_event():
    """All four handlers append their event, in feed order (lens A NIT: was two of the four)."""
    from engine_v2.tests._event_factory import make_cts_established
    t = FibTracker(cross_fib_limits={0: NO_P}, quiet=True)
    df = _prepare_df(_rows(1))
    est = make_cts_established(cts_anchor_idx=3, confirmed_at=4, price=1.0, **_ATTR)
    thr = make_event("CTS_THRESHOLD_UPDATED", 7, price=1.0, **_ATTR)
    evs = [est, _ev_upd(5), _ev_conf(6, 5), thr]
    t.on_cts_established(est, df, 1, 0.9)
    t.on_cts_updated(evs[1], df)
    t.on_cts_confirmed(evs[2])
    t.on_cts_threshold_updated(thr, df)
    assert [id(e) for e in t._handled] == [id(e) for e in evs]


# ---------------------------------------------------------------------------
# Errors inside the embedded tracker propagate (never a dropped sub)
# ---------------------------------------------------------------------------

def _geometry(df, inner, lim=NO_P):
    from engine_v2.multitf import entity_df_mutation as edm
    return _quiet(edm._build_geometry, df, sd=1, start_abs=0, run_cap_abs=len(df) - 1, bos0_inner=inner,
                  timeframe="M15", cycle0_proximity_blocked=False, cross_fib_limit=lim)


def _reversing():
    from engine_v2.structure.reference_zone import build_ad_hoc_bos0_reference_zone
    from engine_v2.tests.test_geometry_builder import _make_reversing_data
    df = _prepare_df(_make_reversing_data())
    return df, float(build_ad_hoc_bos0_reference_zone(df, 0, 1).inner)


def test_a_tracker_error_inside_a_sub_geometry_propagates(monkeypatch):
    """A non-RuntimeError inside the embedded tracker arrives as `EmbeddedFibTrackerError` (a RuntimeError) — the
    geometry's `except (ValueError, IndexError)` would otherwise drop the whole sub; mirrors
    `test_an_ip_error_inside_a_sub_geometry_propagates`."""
    df, inner = _reversing()
    assert _geometry(df, inner) is not None

    def boom(self, event, *a, **k):
        raise IndexError("boom")

    monkeypatch.setattr(FibTracker, "on_cts_updated", boom)
    with pytest.raises(EmbeddedFibTrackerError) as exc:
        _geometry(df, inner)
    assert isinstance(exc.value, RuntimeError) and isinstance(exc.value.__cause__, IndexError)


def test_a_tracker_runtime_error_passes_through_unchanged(monkeypatch):
    df, inner = _reversing()

    def boom(self, event, *a, **k):
        raise CrossFibLimitError("limit")

    monkeypatch.setattr(FibTracker, "on_cts_updated", boom)
    with pytest.raises(CrossFibLimitError):
        _geometry(df, inner)


# ---------------------------------------------------------------------------
# The probe: the limit it is handed (PLAN_T T1; F7's explicit "no P" until then), and — a BLOCKED probe, whose cycle 0
# cannot confirm by proximity (the main sid 0 cycle 0's first_confluence, PLAN_T T2) — its result does not depend on
# the read (F7). An unblocked probe's can, since PLAN_T T2.
# ---------------------------------------------------------------------------

def test_the_probe_passes_the_handed_limit_and_does_not_depend_on_the_read(monkeypatch):
    import engine_v2.structure.unified_probe as up
    from engine_v2.tests.test_unified_probe import _make_multicycle_data, _ref_zone_uptrend
    limits = []
    real_make = up._make_market_structure

    def spy_make(*a, **k):
        limits.append(k["cross_fib_limit"])
        return real_make(*a, **k)

    monkeypatch.setattr(up, "_make_market_structure", spy_make)

    handed = CrossFibLimit(None, "none")

    def probe():
        return _quiet(up.unified_probe, _prepare_df(_make_multicycle_data()), 0, 1, _ref_zone_uptrend(), 14, "H1",
                      enable_phase2=True, cross_fib_limit=handed, cycle0_proximity_blocked=True)

    base = probe()
    assert limits and all(lim is handed for lim in limits)   # the SAME object, not an equal one built in the probe
    reads = []

    def always_hit(df, fib, ft, **kw):           # an inner every candle's low reaches: proximity would confirm anything
        reads.append(1)
        return math.inf

    monkeypatch.setattr(se, "poi_inner_for_fib", always_hit)
    other = probe()
    assert reads, "the probe's Phase-2 MS reads the inner"
    assert (other.starting_idx, other.status, other.finalize_condition, other.bos0_inner, other.iterations) == (
        base.starting_idx, base.status, base.finalize_condition, base.bos0_inner, base.iterations)


# ---------------------------------------------------------------------------
# The wiring
# ---------------------------------------------------------------------------

def test_make_market_structure_builds_a_quiet_tracker_with_the_limit(embedded):
    df = _prepare_df(_rows(1))
    lim = limit(0.61, "parent_cts_threshold")
    ms = se._make_market_structure(df, 1, structure_id=3, cycle0_proximity_blocked=True, cross_fib_limit=lim)
    t, = embedded
    assert ms._fib_tracker is t and ms._poi_inner_for_fib is se.poi_inner_for_fib
    assert t._limits == {3: lim} and t._quiet
    assert (t.config.fill_threshold, t.config.fib_levels) == (IMBALANCE_FILL_THRESHOLD, [30.0, 50.0, 61.8, 80.0])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ms.run()
    assert t._handled and "[fib]" not in out.getvalue()


def test_the_limit_is_required_and_typed():
    df = _prepare_df(_rows(1))
    with pytest.raises(TypeError, match="cross_fib_limit"):
        se._make_market_structure(df, 1)
    with pytest.raises(TypeError, match="cross_fib_limit"):
        se.compute_bounded_structure(df, 0, 1)
    with pytest.raises(AssertionError, match="must be a CrossFibLimit"):
        se._make_market_structure(df, 1, cycle0_proximity_blocked=True, cross_fib_limit={"price": None})


def test_one_fill_threshold():
    """The embedded tracker, the downstream tracker and the drawn POI's IP pick share one value — the one constant
    `IMBALANCE_FILL_THRESHOLD`, 0.65 since PLAN_S step 1 (the user, 2026-10-20: 70 % → 65 %, the armed stroke only,
    everywhere) — and every fill-check function defaults to it (no literal default left to drift)."""
    import inspect

    from engine_v2.common.types import ImbalanceInstance
    from engine_v2.patterns.imbalance import has_unfilled_imbalance, iter_unfilled_imbalances
    from engine_v2.zones.cross_cycle_fib import resolve_cross_cycle_eligibility
    assert IMBALANCE_FILL_THRESHOLD == FibTrackerConfig().fill_threshold == POIConfig().fill_threshold == 0.65
    for fn in (ImbalanceInstance.is_filled, has_unfilled_imbalance, iter_unfilled_imbalances,
               resolve_cross_cycle_eligibility):
        assert inspect.signature(fn).parameters["fill_threshold"].default == 0.65, fn.__name__


def test_main_sid_n_gets_its_limit_from_the_sids_before_it(monkeypatch):
    """`compute_structure` hands each main MS run `compute_main_cross_fib_limit(events so far, sid)` — equal to the
    downstream's `compute_main_cross_fib_limits(all events)[sid]` (sid 0: start trading, no `P`; sid 1: sid 0's
    reversal `bos_frozen`)."""
    from engine_v2.tests.test_main_reversal_probe import _make_reversing_data
    from engine_v2.zones.structure_lifecycle import compute_main_cross_fib_limits
    got = {}
    real = se._make_market_structure

    def spy(*a, **k):
        got[int(k["structure_id"])] = k["cross_fib_limit"]
        return real(*a, **k)

    monkeypatch.setattr(se, "_make_market_structure", spy)
    res = _quiet(se.compute_structure, _prepare_df(_make_reversing_data()))
    final = compute_main_cross_fib_limits(res.events)
    assert sorted(got) == sorted(final) and len(got) >= 2
    assert got == final
    assert (got[0].price, got[0].source, got[0].started_by) == (None, "none", "start_trading")
    rev, = [e for e in res.events if e.type == "STATE_CHANGED" and e.meta.get("to") == "reversal"
            and e.meta["structure_id"] == 0]
    assert (got[1].price, got[1].source, got[1].started_by) == (rev.meta["bos_frozen"], "reversal_bos_frozen",
                                                                 "reversal")
