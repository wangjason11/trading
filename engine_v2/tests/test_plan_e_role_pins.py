"""Plan E E2b role pins (landing-review mutation lens): each site that E2b split
into a LOCATION (the anchor) and a TIME (then the anchor too, marked for an E3
stage) is pinned with an E4-shaped event (`idx` = the moment 12, anchor 9 — the
contract since Plan E E4a (CTS_ESTABLISHED) / E4b (BOS_CONFIRMED)) so a
location read switched to the moment — or a raw `ev.idx` read — fails.

The TIME pins state the value of the stage that switched them: the anchor
until the E3 stage named in the site's marker, the moment after it (E3a: the
FibTracker EST / update time halves and the MS in-flight fill horizon).
"""
from __future__ import annotations

import ast
import contextlib
import io
from pathlib import Path

import pandas as pd
import pytest

import engine_v2
import engine_v2.zones.poi_zones as poi_zones
from engine_v2.common.types import ImbalanceInstance
from engine_v2.tests._event_factory import make_cts_established
from engine_v2.tests.test_cross_cycle_fib_routine import _BOS, _CTS, _INST0, _INST1
from engine_v2.tests.test_cross_cycle_fib_routine import _df as _routine_df
from engine_v2.tests.test_imbalance_c3_knowability import _df, _ev, _gap, _quiet, _tracker
from engine_v2.zones.cross_cycle_fib import resolve_cross_cycle_eligibility
from engine_v2.zones.fib_tracker import select_fib_anchor_for_cycle


def _e4_est(anchor=20, moment=22, **kw):
    return make_cts_established(cts_anchor_idx=anchor, confirmed_at=moment, idx=moment,
                                price=1.2, **kw)


# --- FibTracker EST: location vs time -------------------------------------------

@pytest.mark.parametrize("mode", ["h1", "cross_cycle"])
def test_fib_tracker_est_reads_the_anchor_for_the_fib_and_the_moment_for_activated_at(mode):
    tracker = _tracker(mode)
    df = _df(40, [_gap(15)])
    # sid 0 cycle 1 (h1 simple flow activates at cycle >= 1); cross_cycle cycle 0.
    cyc = 1 if mode == "h1" else 0
    fib = _quiet(tracker.on_cts_established, _e4_est(structure_id=0, cycle_id=cyc), df,
                 bos_idx=10, bos_price=0.9)
    assert fib is not None and fib.active
    assert fib.cts_idx == 20                        # LOCATION: the CTS anchor, never the moment
    assert fib.meta["activated_at"] == 22           # TIME: the moment (Plan E E3a)


def test_fib_tracker_update_fill_horizon_and_stamp_are_the_moment():
    """Plan E E3a, update path (`_update_fib_cts`): the only gap (c2 15) fills
    at 31. A pattern-path update with anchor 30 / moment 32 asks the fill at
    32 → filled → deactivated, stamped 32. Asked at the anchor 30 it would stay
    active; stamped at the anchor it would read 30."""
    tracker = _tracker("h1")
    df = _df(40, [_gap(15)])
    df.at[31, "l"] = 0.99
    fib = _quiet(tracker.on_cts_established, _e4_est(anchor=20, moment=20, structure_id=0, cycle_id=1),
                 df, bos_idx=10, bos_price=0.9)
    assert fib.active
    upd = _quiet(tracker.on_cts_updated, _ev("CTS_UPDATED", 30, 1.3, 0, 1, via="continuous",
                                             confirmed_at=32), df)
    assert upd.cts_idx == 30                          # LOCATION
    assert not upd.active
    assert upd.meta["deactivated_at"] == 32           # TIME


@pytest.mark.parametrize("fixture, pair", [
    ("second_cts", (9, 10)),          # a lagging CTS_ESTABLISHED: anchor 9, apply 10
    ("lagging_update", (24, 25)),     # a lagging pattern-path CTS_UPDATED: anchor 24, apply 25
])
def test_ms_inflight_poi_refresh_fill_horizon_is_the_moment(monkeypatch, fixture, pair):
    """Plan E E3a, the MS mirror: `_refresh_poi_inners_for_cycle` hands the
    resolver the triggering event's moment as `fill_horizon_idx` (lock-step with
    FibTracker), never the CTS anchor; a raw update's moment IS its candle."""
    import engine_v2.structure.structure_engine as se
    from engine_v2.tests.test_imbalance_c3_knowability import _multicycle_with_lagging_pattern_update
    from engine_v2.tests.test_unified_probe import _make_second_cts_moment_after_anchor_data, _prepare_df
    calls = []
    real = se.compute_poi_inners_for_cycle

    def spy(df, bos_idx, bos_price, cts_idx, *a, **k):
        calls.append((int(cts_idx), k["fill_horizon_idx"]))
        return real(df, bos_idx, bos_price, cts_idx, *a, **k)

    monkeypatch.setattr(se, "compute_poi_inners_for_cycle", spy)
    rows = (_make_second_cts_moment_after_anchor_data() if fixture == "second_cts"
            else _multicycle_with_lagging_pattern_update())
    with contextlib.redirect_stdout(io.StringIO()):
        res = se.compute_bounded_structure(_prepare_df(rows), 0, +1)
    assert pair in calls
    assert all(h >= c for c, h in calls)
    raw = [e.idx for e in res.events if e.type == "CTS_UPDATED" and e.meta["via"] == "replay_raw"]
    assert all((r, r) in calls for r in raw)


# --- the shared routine / anchor selector: window vs horizon --------------------

def test_routine_own_window_end_and_fill_horizon_are_independent():
    """Window [30, 40] (location), horizon 38 (time): the own gap is filled only
    at 39, so asked at 38 it is still unfilled."""
    df = _routine_df(60, [_INST0, _INST1])
    df.at[39, "l"] = 1.22
    e = resolve_cross_cycle_eligibility(
        df=df, target_cycle=1, sd=1, own_window_end_idx=40, fill_horizon_idx=38, own_imb_start=30,
        anchor_idx=40, anchor_price=1.35, bos_by_cycle={0: _BOS[0]}, cts_by_cycle={0: _CTS[0]},
        dead_cycles=set(), fill_threshold=0.70, fill_as_of="snapshot",
        prior_cached_liveness={0: True}, evaluated_at=None, snapshot_horizon_idx=30)
    assert e.own_has is True


def test_anchor_selector_asks_cond3_at_the_snapshot_horizon():
    """Cycle 0's gap fills at 33: after BOS_1 (30), before CTS_1 (40). cond3 is
    asked at the snapshot horizon (BOS_1) → not yet filled → the cross forms."""
    df = _routine_df(60, [_INST0, _INST1])
    df.at[33, "l"] = 1.02
    c0 = {"bos_idx": 10, "bos_price": 1.0, "cts_idx": 20, "cts_price": 1.15, "has_unfilled": True}
    r = select_fib_anchor_for_cycle(df, 1, 1, 30, 1.2, 40, 1.35, c0, 0.70, struct_direction=1,
                                    evaluated_at=None, fill_horizon_idx=40, snapshot_horizon_idx=30)
    assert r[0] == 10 and r[-1] == "scenario_2_cross"


# --- POI sweep: cond1 is a location -----------------------------------------------

def _poi_history(anchor, moment=7, floor=None):
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    df = pd.DataFrame([
        {"time": t0 + pd.Timedelta(hours=i), "o": 0.6040, "h": 0.6045, "l": 0.6035, "c": 0.6040}
        for i in range(14)
    ])
    # IC at 6, inside the 61.8-80 band of (BOS 0.6000, CTS 0.6100).
    df.loc[6, ["o", "h", "l", "c"]] = [0.6034, 0.6035, 0.6025, 0.6026]
    imb = ImbalanceInstance(start_idx=8, end_idx=8, direction=1,     # formed at 9, after the IC
                            gap_top=0.6090, gap_bottom=0.6080, gap_size=0.0010)
    cts = make_cts_established(cts_anchor_idx=anchor, confirmed_at=moment, idx=moment,
                               price=0.6100, struct_direction=None)
    return poi_zones._compute_poi_activation_history(
        df, ic_idx=6, cts_established_idx=moment, sd=1, scan_end=12, fill_threshold=0.70,
        bos_price=0.6000, cts_events=[cts], fib_min_pct=61.8, fib_max_pct=80.0,
        variant_thresholds={"V30": 0.3, "V60": 0.6, "V90": 0.9},
        imbalances=[imb], fill_idx_cache={id(imb): (None, None)}, lifecycle_floor_idx=floor,
    )


def test_poi_sweep_cond1_reads_the_cts_anchor():
    """An IC (6) past the CTS anchor (5) but before the moment (7): cond1 ("the
    IC lies inside the fib", `cts_at_t >= ic`) reads the ANCHOR → never active.
    A moment read would activate it at 9."""
    assert _poi_history(anchor=5) == []


def test_poi_sweep_positive_control():
    """The same fixture with the anchor at the IC activates (so the pin above is not vacuous)."""
    hist = _poi_history(anchor=6)
    assert hist and hist[0]["idx"] == 9 and hist[0]["active"] is True


# --- source guard: the pinned sort call sites -----------------------------------

_PINNED = ("pipeline/orchestrator.py", "multitf/sub_wvmi.py", "zones/zone_proximity.py",
           "zones/poi_zones.py", "zones/wave_candles.py", "structure/reference_zone.py",
           "structure/unified_probe.py")


def test_no_pinned_module_sorts_events_on_a_raw_idx():
    """A sort / max key that reads `.idx` of an event would undo the E2 pins (the
    E4 flip would reorder). Keys must go through `ef.` (processing_order_key /
    stamped_idx / an anchor accessor)."""
    root = Path(engine_v2.__file__).parent
    offenders = []
    for rel in _PINNED:
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            name = getattr(n.func, "attr", None) or getattr(n.func, "id", None)
            if name not in ("sort", "sorted", "max", "min"):
                continue
            for kw in n.keywords:
                if kw.arg == "key" and isinstance(kw.value, ast.Lambda):
                    if any(isinstance(a, ast.Attribute) and a.attr == "idx" for a in ast.walk(kw.value.body)):
                        offenders.append(f"{rel}:{n.lineno}")
    assert offenders == []


# --- E2c: BOS readers outside the downstream pipeline ------------------------------
# (inside it, `test_e4_simulation` covers KL / fib / POI / wave candles / WVMI /
# zone proximity for a BOS flip)

def _e4_bos(anchor=20, moment=22, **kw):
    from engine_v2.tests._event_factory import make_bos_confirmed
    return make_bos_confirmed(bos_anchor_idx=anchor, confirmed_at=moment, idx=moment, **kw)


def test_first_confluence_input_is_the_bos_anchor():
    """B5 / R1: the FC probe input (the pool key) is the BOS ANCHOR; the trigger
    fires at the moment."""
    from engine_v2.multitf.first_confluence_trigger import detect_first_confluence_triggers
    t, = detect_first_confluence_triggers([_e4_bos(structure_id=0, cycle_id=1, price=0.61)])
    assert (t.input_idx, t.trigger_event_idx) == (20, 22)


def test_struct_start_is_the_moment_and_creation_idx_the_anchor():
    """B9 / B10 (Plan E E3f, user decision 2026-09-25): the lifecycle-start base
    is the first CTS_ESTABLISHED MOMENT (22, not BOS_0's anchor 20); the
    historical `creation_event_idx` stays the first ANCHOR (20)."""
    from engine_v2.multitf.sid_records import build_sid_records_for_main
    from engine_v2.zones.structure_lifecycle import compute_struct_start_by_sid
    evs = [_e4_est(anchor=21, moment=22, structure_id=0, cycle_id=0),
           _e4_bos(structure_id=0, cycle_id=0)]
    assert compute_struct_start_by_sid(evs, {}, None) == {0: 22}   # the moment (E3f)
    rec, = build_sid_records_for_main(evs)
    assert rec.creation_event_idx == 20                            # stays the anchor (E3f)


def test_structure_levels_are_timed_at_the_anchors():
    """L5 / B8: `structure_levels` place each CTS / BOS level at its ANCHOR candle
    (the E2c BOS-only variant replay caught a raw `ev.idx` read here)."""
    from engine_v2.structure.market_structure import MarketStructure
    from engine_v2.tests.test_unified_probe import _make_multicycle_data, _prepare_df
    ms = MarketStructure(_prepare_df(_make_multicycle_data()), 1)
    ms.events = [_e4_est(anchor=9, moment=12, structure_id=0, cycle_id=1),
                 _e4_bos(anchor=7, moment=12, structure_id=0, cycle_id=1, price=0.95)]
    t = pd.to_datetime(ms.df["time"], utc=True)
    cts, bos = ms._events_to_structure_levels()
    assert (cts.kind, cts.time) == ("CTS", t.iloc[9])
    assert (bos.kind, bos.time) == ("BOS", t.iloc[7])


def test_prev_bos_line_runs_anchor_to_anchor():
    """B6 / L9 (Q6): the line starts at sid 0's last BOS ANCHOR and ends at the
    ANCHOR of sid 1's first CTS known at/after the reversal (the filter: its moment, E3d)."""
    from engine_v2.pipeline.orchestrator import _prev_bos_lines
    from engine_v2.structure import event_fields as ef
    evs = [_e4_bos(anchor=7, moment=10, structure_id=0, cycle_id=1, price=0.95),
           _e4_est(anchor=16, moment=18, structure_id=1, cycle_id=0)]
    evs.sort(key=ef.processing_order_key)
    with contextlib.redirect_stdout(io.StringIO()):
        lines = _prev_bos_lines(evs, {1: 15})
    assert [(ln["start_idx"], ln["end_idx"]) for ln in lines] == [(7, 16)]


def test_prev_bos_line_filter_is_the_moment():
    """Plan E E3d: "the first CTS of sid 1 known at/after the reversal" is decided
    on the MOMENT. sid 1's CTS_0 is anchored at 13 but established at 16; the
    reversal is confirmed at 15 → it qualifies (on its anchor 13 it would not,
    and the line would end at the later update's anchor 20). END = its anchor 13."""
    from engine_v2.pipeline.orchestrator import _prev_bos_lines
    from engine_v2.structure import event_fields as ef
    from engine_v2.structure.market_structure import StructureEvent
    from engine_v2.tests._event_factory import make_bos_confirmed, make_cts_established
    upd = StructureEvent(idx=20, category="STRUCTURE", type="CTS_UPDATED", price=1.3,
                         meta={"structure_id": 1, "cycle_id": 0, "via": "replay_raw"})
    evs = [make_bos_confirmed(bos_anchor_idx=7, confirmed_at=10, structure_id=0, cycle_id=1, price=0.95),
           make_cts_established(cts_anchor_idx=13, confirmed_at=16, structure_id=1, cycle_id=0),
           upd]
    evs.sort(key=ef.processing_order_key)
    with contextlib.redirect_stdout(io.StringIO()):
        lines = _prev_bos_lines(evs, {1: 15})
    assert [(ln["start_idx"], ln["end_idx"]) for ln in lines] == [(7, 13)]


@pytest.mark.parametrize("mode", ["h1", "cross_cycle"])
def test_lagging_est_fib_lifecycle_is_timed_at_the_moment(mode):
    """PLAN_E §7 E3a unit: the 2nd CTS_ESTABLISHED has anchor 9 / moment 10.
    Through the downstream pipeline the cycle-1 fib starts (and is stamped
    `activated_at`) at 10, and in cross_cycle the cycle-0 fib's `new_cycle`
    terminal is 10 — the moment, not the anchor 9."""
    from engine_v2.structure.structure_engine import compute_bounded_structure
    from engine_v2.tests.test_e4_simulation import _run
    from engine_v2.tests.test_unified_probe import _make_second_cts_moment_after_anchor_data, _prepare_df
    with contextlib.redirect_stdout(io.StringIO()):
        res = compute_bounded_structure(_prepare_df(_make_second_cts_moment_after_anchor_data()), 0, +1)
    fibs = {(f.structure_id, f.cycle_id): f for f in _run(res.df, res.events, mode)["fib_states"]}
    c1 = fibs[(0, 1)]
    assert (c1.cts_idx, c1.start_idx, c1.meta["activated_at"]) == (9, 10, 10)
    if mode == "cross_cycle":
        assert (fibs[(0, 0)].end_idx, fibs[(0, 0)].end_reason) == (10, "new_cycle")


def _prev_bos_pairs(evs, rv):
    from engine_v2.pipeline.orchestrator import _prev_bos_lines
    from engine_v2.structure import event_fields as ef
    evs = sorted(evs, key=ef.processing_order_key)
    with contextlib.redirect_stdout(io.StringIO()):
        return [(ln["start_idx"], ln["end_idx"]) for ln in _prev_bos_lines(evs, {1: rv})]


def test_prev_bos_line_filter_is_inclusive_at_the_reversal():
    """E3d (landing review): "known AT/after the reversal" — a CTS_0 whose moment
    == the reversal qualifies."""
    from engine_v2.tests._event_factory import make_bos_confirmed, make_cts_established
    evs = [make_bos_confirmed(bos_anchor_idx=7, confirmed_at=10, structure_id=0, cycle_id=1, price=0.95),
           make_cts_established(cts_anchor_idx=13, confirmed_at=15, structure_id=1, cycle_id=0)]
    assert _prev_bos_pairs(evs, 15) == [(7, 13)]


def test_prev_bos_line_falls_through_to_a_raw_cts_update():
    """E3d (landing review): sid 1's CTS_0 is known BEFORE the reversal → the first
    qualifying CTS is a later raw CTS_UPDATED (moment == its idx) → END = its anchor."""
    from engine_v2.structure.event_fields import CTS_UPDATED_RAW_VIA
    from engine_v2.structure.market_structure import StructureEvent
    from engine_v2.tests._event_factory import make_bos_confirmed, make_cts_established
    upd = StructureEvent(idx=18, category="STRUCTURE", type="CTS_UPDATED", price=1.3,
                         meta={"structure_id": 1, "cycle_id": 0, "via": CTS_UPDATED_RAW_VIA})
    evs = [make_bos_confirmed(bos_anchor_idx=7, confirmed_at=10, structure_id=0, cycle_id=1, price=0.95),
           make_cts_established(cts_anchor_idx=12, confirmed_at=14, structure_id=1, cycle_id=0),
           upd]
    assert _prev_bos_pairs(evs, 15) == [(7, 18)]


def test_prev_bos_line_picks_the_earliest_moment_not_the_first_stamped():
    """E3d (landing review): a pattern-path CTS_UPDATED that REGRESSES the CTS
    (zones-audit latent bug (a)) is anchored at 15 (its processing key) but known
    at 20 (its idx since Plan E E4c), after a raw update at 17. Reversal 17: the earliest-known qualifying CTS is the raw
    update (END 17), not the first in processing order (END 15)."""
    from engine_v2.structure.event_fields import CTS_UPDATED_RAW_VIA
    from engine_v2.structure.market_structure import StructureEvent
    from engine_v2.tests._event_factory import make_bos_confirmed, make_cts_established
    raw = StructureEvent(idx=17, category="STRUCTURE", type="CTS_UPDATED", price=1.3,
                         meta={"structure_id": 1, "cycle_id": 0, "via": CTS_UPDATED_RAW_VIA})
    pat = StructureEvent(idx=20, category="STRUCTURE", type="CTS_UPDATED", price=1.25,
                         meta={"structure_id": 1, "cycle_id": 0, "via": "continuous", "confirmed_at": 20,
                               "cts_anchor_idx": 15})
    evs = [make_bos_confirmed(bos_anchor_idx=7, confirmed_at=10, structure_id=0, cycle_id=1, price=0.95),
           make_cts_established(cts_anchor_idx=13, confirmed_at=16, structure_id=1, cycle_id=0),
           raw, pat]
    assert _prev_bos_pairs(evs, 17) == [(7, 17)]


def test_poi_sweep_cond1_pre_window_reads_the_cts_anchor():
    """Plan E E4a, the PRE-WINDOW branch: the CTS_ESTABLISHED is known (moment 7)
    before first_active (the lifecycle floor 8), so it enters as prior state;
    its cond1 location is still the ANCHOR (5 < IC 6 -> never active). A read of
    `ev.idx` (the moment 7 since E4a) would activate at 9."""
    assert _poi_history(anchor=5, floor=8) == []


def test_poi_sweep_pre_window_positive_control():
    hist = _poi_history(anchor=6, floor=8)
    assert hist and hist[0]["idx"] == 9 and hist[0]["active"] is True


def test_h1_chart_unconfirmed_cts_marker_sits_at_the_cts_anchor(tmp_path, monkeypatch):
    """Plan E E4a: the H1 chart's "CTS (unconfirmed)" dot after a sid's last
    confirmed BOS is a LOCATION -- the CTS anchor (9), never the
    CTS_ESTABLISHED's `ev.idx` (its moment 10 since E4a)."""
    go = pytest.importorskip("plotly.graph_objects")
    from engine_v2.charting.export_plotly import export_chart_plotly
    from engine_v2.multitf.registry import StructureRegistry
    from engine_v2.structure.structure_engine import compute_bounded_structure
    from engine_v2.tests.test_unified_probe import (
        _make_second_cts_moment_after_anchor_data, _prepare_df)

    with contextlib.redirect_stdout(io.StringIO()):
        res = compute_bounded_structure(
            _prepare_df(_make_second_cts_moment_after_anchor_data()), 0, +1)
    # Stop the stream before CTS_CONFIRMED(0,1) @ 11: sid 0's last confirmed
    # point is BOS(0,1) @ anchor 7; CTS(0,1) is established (anchor 9,
    # moment 10) but unconfirmed.
    events = [e for e in res.events if e.idx <= 10]
    est = next(e for e in events if e.type == "CTS_ESTABLISHED" and e.meta["cycle_id"] == 1)
    assert (est.meta["cts_anchor_idx"], est.meta["confirmed_at"]) == (9, 10)
    df = res.df.copy()
    df.attrs["structure_events"] = events
    reg = StructureRegistry()
    reg.register("H1", df=df, timeframe="H1", role="main")
    figs = []
    monkeypatch.setattr(go.Figure, "write_html", lambda self, *a, **k: figs.append(self))
    monkeypatch.setattr(go.Figure, "write_image", lambda self, *a, **k: None)
    cfg = {"structure": {"levels": True}, "struct_state": {"labels": False},
           "range_visual": {"rectangles": False}, "fib": {"lines": False},
           "imbalance": {"highlight": False}, "zones": {"wave_candles": False},
           "volume": {"bars": False, "ema_line": False, "spike_marker": False}}
    with contextlib.redirect_stdout(io.StringIO()):
        export_chart_plotly(title="t", registry=reg, path_id="H1", out_dir=tmp_path, cfg=cfg)
    fig, = figs
    marks = [tr for tr in fig.data if (tr.name or "").startswith("CTS (unconfirmed)")]
    assert [int(c[0]) for tr in marks for c in tr.customdata] == [9]
    assert [x for tr in marks for x in tr.x] == [df.loc[9, "time"]]


def test_scenario3_exception_window_opens_after_the_cts0_anchor(monkeypatch):
    """Q10 test-only path, pinned against a raw read: scenario 3's BOS_0
    exception window opens at CTS_0's ANCHOR + 1 (5), not at its `ev.idx`
    (the moment 6 since Plan E E4a)."""
    import engine_v2.structure.structure_engine as se
    from engine_v2.tests.test_unified_probe import (
        _make_second_cts_moment_after_anchor_data, _prepare_df)
    df = _prepare_df(_make_second_cts_moment_after_anchor_data())
    probe_df = df.copy()
    probe_df["market_state"] = "pullback"
    probe_df["structure_id"] = 0
    est = make_cts_established(cts_anchor_idx=4, confirmed_at=6, price=1.2,
                               structure_id=0, cycle_id=0)

    class _MS:
        debug = False

        def run(self):
            return probe_df, [est], []

    lows = []
    monkeypatch.setattr(se, "_make_market_structure", lambda *a, **k: _MS())
    monkeypatch.setattr(se, "_get_bos0_zone_bounds", lambda *a, **k: (0.59, 0.60, "buy"))
    monkeypatch.setattr(se, "_find_closest_candle_to_outer",
                        lambda _df, lo, hi, *a: lows.append(lo))
    with contextlib.redirect_stdout(io.StringIO()):
        res = se.compute_structure_scenario_3(df, 0, +1, end_idx=15, run_continuation=False)
    assert res.status == "finalized"
    assert lows == [5]


# --- E4b landing review: the charts' BOS LOCATION reads (dots + PB→BOS lines) ---------
# A sid handoff built with LAGGING BOS events (anchor < moment == ev.idx since E4b):
# a BOS dot / a PB→BOS line end read from the raw `ev.idx` lands on the moment.

def _e4b_handoff_events():
    """sid 0 (sd +1): BOS(0,0) anchor 2 / moment 5, CTS_CONFIRMED(0,0) anchor 8 at 9,
    a pullback at 11, the reversal at 12. sid 1 (sd -1): BOS(1,0) anchor 13 /
    moment 16. Sid 0 ends on a CTS, so every chart draws its PB(11) → sid 1's
    first BOS line (the PB search's upper bound is that BOS's moment 16 > 11)."""
    from engine_v2.tests._event_factory import make_bos_confirmed, make_event
    return [
        make_bos_confirmed(bos_anchor_idx=2, confirmed_at=5, price=0.6010,
                           structure_id=0, cycle_id=0, struct_direction=1),
        make_event("CTS_CONFIRMED", 9, price=0.6120, cts_anchor_idx=8,
                   structure_id=0, cycle_id=0, struct_direction=1),
        make_event("STATE_CHANGED", 11, to="pullback", structure_id=0, struct_direction=1),
        make_event("STATE_CHANGED", 12, to="reversal", structure_id=0, struct_direction=1),
        make_bos_confirmed(bos_anchor_idx=13, confirmed_at=16, price=0.6130,
                           structure_id=1, cycle_id=0, struct_direction=-1),
    ]


def _e4b_ohlc(n, freq):
    return pd.DataFrame({
        "time": pd.date_range("2025-12-01", periods=n, freq=freq, tz="UTC"),
        "o": 0.6050, "h": [0.6100 + k * 1e-4 for k in range(n)],
        "l": [0.6000 - k * 1e-4 for k in range(n)], "c": 0.6050,
    })


def _e4b_sub_polylines():
    from types import SimpleNamespace
    pytest.importorskip("plotly")
    from engine_v2.charting.export_m15_chart import _build_sub_polylines
    lt = _e4b_ohlc(24, "15min")

    def lt_time(i):
        return lt["time"].iloc[int(i)] if 0 <= int(i) < len(lt) else None

    poly = _build_sub_polylines(
        SimpleNamespace(end_event_idx=None, starting_sd=1), _e4b_handoff_events(), lt,
        lt_time, lambda i: int(i) + 100, lambda i: True)
    return poly, lt


def test_m15_sub_polyline_bos_dots_sit_at_the_bos_anchor():
    """B1 (`_build_sub_polylines`): a sub's BOS dot is a LOCATION -- the BOS
    anchor (2, 13), never `ev.idx` (the moment 5, 16 since Plan E E4b)."""
    poly, lt = _e4b_sub_polylines()
    bos = sorted((p[4], p[0], p[1], p[7]) for pts in poly["points_by_sid"].values()
                 for p in pts if p[3] == "BOS")
    assert bos == [(0, 2, lt["time"].iloc[2], 102), (1, 13, lt["time"].iloc[13], 113)]


def test_m15_sub_polyline_pb_to_bos_line_ends_at_the_bos_anchor():
    """B2 (`_build_sub_polylines`): the cross-structure PB→BOS line runs from
    sid 0's pullback (11) to sid 1's first BOS ANCHOR (13), not its moment (16)."""
    poly, lt = _e4b_sub_polylines()
    (sid, pb_t, _, bos_t, bos_p, pb_idx, bos_idx), = poly["pb_to_bos_lines"]
    assert (sid, pb_idx, bos_idx, bos_p) == (0, 11, 13, 0.6130)
    assert (pb_t, bos_t) == (lt["time"].iloc[11], lt["time"].iloc[13])


def _e4b_h1_overlay_fig():
    go = pytest.importorskip("plotly.graph_objects")
    from engine_v2.charting.export_m15_chart import _render_h1_overlay
    h1 = _e4b_ohlc(24, "1h")
    h1.attrs["structure_events"] = _e4b_handoff_events()
    m15 = pd.DataFrame({"time": pd.date_range(h1["time"].iloc[0], periods=24 * 4,
                                              freq="15min", tz="UTC")})
    fig = go.Figure()
    _render_h1_overlay(fig, m15, h1, {}, {}, {"labels": False}, {"levels": True},
                       {"KL": False, "POI": False, "wave_candles": False})

    def m15_t(i):   # an H1 candle's M15 x: its 4th quarter (the default mapping)
        return h1["time"].iloc[i] + pd.Timedelta(minutes=45)
    return fig, m15_t


def test_h1_overlay_bos_dots_sit_at_the_bos_anchor():
    """B3 (`_render_h1_overlay`): the H1 overlay's BOS dots sit at the BOS
    anchors (2, 13), never at `ev.idx` (the moments 5, 16 since Plan E E4b)."""
    fig, m15_t = _e4b_h1_overlay_fig()
    tr, = [t for t in fig.data if t.name == "H1 BOS"]
    assert sorted(int(c[0]) for c in tr.customdata) == [2, 13]
    assert sorted(tr.x) == [m15_t(2), m15_t(13)]


def test_h1_overlay_pb_to_bos_line_ends_at_the_bos_anchor():
    """B4 (`_render_h1_overlay`): the H1 overlay's PB→BOS line ends at sid 1's
    first BOS ANCHOR (13), not its moment (16)."""
    fig, m15_t = _e4b_h1_overlay_fig()
    tr, = [t for t in fig.data if t.name == "H1 PB→BOS sid=0"]
    assert tuple(tr.x) == (m15_t(11), m15_t(13))
    assert tuple(tr.y)[1] == 0.6130


def _e4b_h1_chart_fig(tmp_path, monkeypatch):
    go = pytest.importorskip("plotly.graph_objects")
    from engine_v2.charting.export_plotly import export_chart_plotly
    from engine_v2.multitf.registry import StructureRegistry
    from engine_v2.structure.structure_engine import compute_bounded_structure
    from engine_v2.tests.test_unified_probe import _make_multicycle_data, _prepare_df

    with contextlib.redirect_stdout(io.StringIO()):
        res = compute_bounded_structure(_prepare_df(_make_multicycle_data()), 0, +1)
    df = res.df.copy()
    df.attrs["structure_events"] = _e4b_handoff_events()   # the chart reads events only
    reg = StructureRegistry()
    reg.register("H1", df=df, timeframe="H1", role="main")
    figs = []
    monkeypatch.setattr(go.Figure, "write_html", lambda self, *a, **k: figs.append(self))
    monkeypatch.setattr(go.Figure, "write_image", lambda self, *a, **k: None)
    cfg = {"structure": {"levels": True}, "struct_state": {"labels": False},
           "range_visual": {"rectangles": False}, "fib": {"lines": False},
           "imbalance": {"highlight": False}, "zones": {"wave_candles": False},
           "volume": {"bars": False, "ema_line": False, "spike_marker": False}}
    with contextlib.redirect_stdout(io.StringIO()):
        export_chart_plotly(title="t", registry=reg, path_id="H1", out_dir=tmp_path, cfg=cfg)
    fig, = figs
    return fig, df


def test_h1_chart_bos_dots_sit_at_the_bos_anchor(tmp_path, monkeypatch):
    """B5 (`export_plotly`, explicit since the E4b landing review): the H1
    chart's confirmed-BOS dots sit at the BOS anchors (2, 13), never at `ev.idx`
    (the moments 5, 16)."""
    fig, df = _e4b_h1_chart_fig(tmp_path, monkeypatch)
    trs = [t for t in fig.data if (t.name or "").startswith("BOS (confirmed)")]
    assert sorted(int(c[0]) for t in trs for c in t.customdata) == [2, 13]
    assert sorted(x for t in trs for x in t.x) == [df.loc[2, "time"], df.loc[13, "time"]]


def test_h1_chart_pb_to_next_bos_line_ends_at_the_bos_anchor(tmp_path, monkeypatch):
    """B6 (`export_plotly`): the H1 chart's PB→next-BOS line ends at sid 1's
    first BOS ANCHOR (13), not its moment (16)."""
    fig, df = _e4b_h1_chart_fig(tmp_path, monkeypatch)
    tr, = [t for t in fig.data if t.name == "PB→BOS sid=0→1"]
    assert tuple(tr.x) == (df.loc[11, "time"], df.loc[13, "time"])
    assert tuple(tr.y)[1] == 0.6130


# --- E4c landing review: pattern-path CTS_UPDATED LOCATION reads -----------------
# A LAGGING pattern-path CTS_UPDATED (anchor < moment == ev.idx since Plan E E4c):
# a location read of the raw `ev.idx` lands on the moment. Each pin is killed by
# the pre-review harness mutant named in its docstring (U#).

def _e4c_upd(anchor, moment, *, sid=0, cyc=0, price=0.6150, sd=1):
    from engine_v2.tests._event_factory import make_cts_updated
    ev = make_cts_updated(cts_anchor_idx=anchor, confirmed_at=moment, via="continuous",
                          price=price, meta={"structure_id": sid, "cycle_id": cyc,
                                             "struct_direction": sd})
    assert (ev.idx, ev.meta["cts_anchor_idx"]) == (moment, anchor)
    return ev


def _e4c_unconf_events():
    """sid 0 (sd +1): BOS(0,0) anchor 2 / moment 5, CTS_ESTABLISHED(0,0) anchor 6 /
    moment 7, a lagging pattern-path CTS_UPDATED(0,0) anchor 8 / moment 10. Sid 0's
    last confirmed point is its BOS, so every chart draws the unconfirmed-CTS dot
    at the latest CTS ANCHOR (8)."""
    from engine_v2.tests._event_factory import make_bos_confirmed
    return [
        make_bos_confirmed(bos_anchor_idx=2, confirmed_at=5, price=0.6010,
                           structure_id=0, cycle_id=0, struct_direction=1),
        make_cts_established(cts_anchor_idx=6, confirmed_at=7, price=0.6120,
                             structure_id=0, cycle_id=0, struct_direction=1),
        _e4c_upd(8, 10),
    ]


def _e4c_sub_polylines(events, owned_here):
    from types import SimpleNamespace
    pytest.importorskip("plotly")
    from engine_v2.charting.export_m15_chart import _build_sub_polylines
    lt = _e4b_ohlc(24, "15min")

    def lt_time(i):
        return lt["time"].iloc[int(i)] if 0 <= int(i) < len(lt) else None

    poly = _build_sub_polylines(SimpleNamespace(end_event_idx=None, starting_sd=1), events, lt,
                                lt_time, lambda i: int(i) + 100, owned_here)
    return poly, lt


def test_m15_sub_unconfirmed_cts_filter_reads_the_update_anchor():
    """U1 (`_build_sub_polylines`, the "CTS after the last BOS" filter): sid 0's
    last point is BOS(0,1) at anchor 9; the pattern-path CTS_UPDATED(0,0) is
    anchored at 8 (before it) but known at 10 (its `ev.idx`). The filter is a
    LOCATION test -> the update is not after the BOS, and CTS_ESTABLISHED(0,1)
    (anchor 14) lies outside the candles this sub owns (<= 12) -> no unconfirmed
    dot. Filtering on `ev.idx` 10 > 9 would draw a stale "CTS (updated)" dot."""
    from engine_v2.tests._event_factory import make_bos_confirmed, make_event
    events = _e4c_unconf_events() + [
        make_event("CTS_CONFIRMED", 11, price=0.6150, cts_anchor_idx=8,
                   structure_id=0, cycle_id=0, struct_direction=1),
        make_bos_confirmed(bos_anchor_idx=9, confirmed_at=15, price=0.6090,
                           structure_id=0, cycle_id=1, struct_direction=1),
        make_cts_established(cts_anchor_idx=14, confirmed_at=15, price=0.6170,
                             structure_id=0, cycle_id=1, struct_direction=1),
    ]
    poly, _ = _e4c_sub_polylines(events, lambda i: int(i) <= 12)
    assert [(p[0], p[3]) for p in poly["points_by_sid"][0]] == [(8, "CTS"), (2, "BOS"), (9, "BOS")]
    assert poly["extra_cts_pts"] == []


def test_m15_sub_unconfirmed_cts_dot_sits_at_the_update_anchor():
    """U2 (`_build_sub_polylines`): the unconfirmed-CTS dot after sid 0's last
    BOS sits at the lagging pattern-path CTS_UPDATED's ANCHOR (8), never its
    `ev.idx` (the moment 10 since Plan E E4c)."""
    poly, lt = _e4c_sub_polylines(_e4c_unconf_events(), lambda i: True)
    assert [(p[0], p[1], p[3], p[7]) for p in poly["extra_cts_pts"]] == [
        (8, lt["time"].iloc[8], "CTS (updated)", 108)]


def test_h1_overlay_unconfirmed_cts_dot_sits_at_the_update_anchor():
    """U3 (`_render_h1_overlay`): the H1 overlay's unconfirmed-CTS dot after
    sid 0's last BOS sits at the pattern-path CTS_UPDATED's ANCHOR (8), never
    its `ev.idx` (the moment 10)."""
    go = pytest.importorskip("plotly.graph_objects")
    from engine_v2.charting.export_m15_chart import _render_h1_overlay
    h1 = _e4b_ohlc(24, "1h")
    h1.attrs["structure_events"] = _e4c_unconf_events()
    m15 = pd.DataFrame({"time": pd.date_range(h1["time"].iloc[0], periods=24 * 4,
                                              freq="15min", tz="UTC")})
    fig = go.Figure()
    _render_h1_overlay(fig, m15, h1, {}, {}, {"labels": False}, {"levels": True},
                       {"KL": False, "POI": False, "wave_candles": False})
    tr, = [t for t in fig.data if t.name == "H1 CTS"]
    assert [int(c[0]) for c in tr.customdata] == [8]
    assert list(tr.x) == [h1["time"].iloc[8] + pd.Timedelta(minutes=45)]


def test_h1_chart_unconfirmed_cts_marker_sits_at_the_update_anchor(tmp_path, monkeypatch):
    """U4 (`export_plotly`): the H1 chart's "CTS (unconfirmed)" marker after
    sid 0's last BOS sits at the pattern-path CTS_UPDATED's ANCHOR (8), never
    its `ev.idx` (the moment 10 since Plan E E4c)."""
    go = pytest.importorskip("plotly.graph_objects")
    from engine_v2.charting.export_plotly import export_chart_plotly
    from engine_v2.multitf.registry import StructureRegistry
    from engine_v2.structure.structure_engine import compute_bounded_structure
    from engine_v2.tests.test_unified_probe import _make_multicycle_data, _prepare_df

    with contextlib.redirect_stdout(io.StringIO()):
        res = compute_bounded_structure(_prepare_df(_make_multicycle_data()), 0, +1)
    df = res.df.copy()
    df.attrs["structure_events"] = _e4c_unconf_events()   # the chart reads events only
    reg = StructureRegistry()
    reg.register("H1", df=df, timeframe="H1", role="main")
    figs = []
    monkeypatch.setattr(go.Figure, "write_html", lambda self, *a, **k: figs.append(self))
    monkeypatch.setattr(go.Figure, "write_image", lambda self, *a, **k: None)
    cfg = {"structure": {"levels": True}, "struct_state": {"labels": False},
           "range_visual": {"rectangles": False}, "fib": {"lines": False},
           "imbalance": {"highlight": False}, "zones": {"wave_candles": False},
           "volume": {"bars": False, "ema_line": False, "spike_marker": False}}
    with contextlib.redirect_stdout(io.StringIO()):
        export_chart_plotly(title="t", registry=reg, path_id="H1", out_dir=tmp_path, cfg=cfg)
    fig, = figs
    marks = [tr for tr in fig.data if (tr.name or "").startswith("CTS (unconfirmed)")]
    assert [int(c[0]) for tr in marks for c in tr.customdata] == [8]
    assert [x for tr in marks for x in tr.x] == [df.loc[8, "time"]]


def test_prev_bos_line_ends_at_a_pattern_update_anchor():
    """U5 (`_prev_bos_lines` END): sid 1's CTS_0 is known (14) before the
    reversal (17); the first CTS known at/after it is a lagging pattern-path
    CTS_UPDATED (anchor 16, moment 18 == its `ev.idx` since Plan E E4c). The
    line END is its ANCHOR 16, not its idx 18."""
    from engine_v2.tests._event_factory import make_bos_confirmed
    evs = [make_bos_confirmed(bos_anchor_idx=7, confirmed_at=10, structure_id=0, cycle_id=1, price=0.95),
           make_cts_established(cts_anchor_idx=13, confirmed_at=14, structure_id=1, cycle_id=0),
           _e4c_upd(16, 18, sid=1, cyc=0, price=1.3)]
    assert _prev_bos_pairs(evs, 17) == [(7, 16)]


def test_structure_levels_time_a_pattern_update_at_its_anchor():
    """U6 (`_events_to_structure_levels`): a pattern-path CTS_UPDATED's level is
    placed at its ANCHOR candle (9), not its `ev.idx` (the moment 12 since Plan E
    E4c); a raw-path one at its idx (13, anchor and moment at once)."""
    from engine_v2.structure.event_fields import CTS_UPDATED_RAW_VIA
    from engine_v2.structure.market_structure import MarketStructure
    from engine_v2.tests._event_factory import make_cts_updated
    from engine_v2.tests.test_unified_probe import _make_multicycle_data, _prepare_df
    ms = MarketStructure(_prepare_df(_make_multicycle_data()), 1)
    ms.events = [_e4c_upd(9, 12, price=1.2),
                 make_cts_updated(cts_anchor_idx=13, via=CTS_UPDATED_RAW_VIA, price=1.25,
                                  meta={"structure_id": 0, "cycle_id": 0})]
    t = pd.to_datetime(ms.df["time"], utc=True)
    assert [(lv.kind, lv.time, lv.meta["via"]) for lv in ms._events_to_structure_levels()] == [
        ("CTS", t.iloc[9], "continuous"), ("CTS", t.iloc[13], CTS_UPDATED_RAW_VIA)]


def _e4c_poi_history(cts_events, established_moment):
    """`_poi_history`'s frame (IC 6 inside the 61.8-80 band of BOS 0.6000 / CTS
    0.6100; an unfilled FVG formed at 9) with a caller-given CTS event list."""
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    df = pd.DataFrame([
        {"time": t0 + pd.Timedelta(hours=i), "o": 0.6040, "h": 0.6045, "l": 0.6035, "c": 0.6040}
        for i in range(14)
    ])
    df.loc[6, ["o", "h", "l", "c"]] = [0.6034, 0.6035, 0.6025, 0.6026]
    imb = ImbalanceInstance(start_idx=8, end_idx=8, direction=1,
                            gap_top=0.6090, gap_bottom=0.6080, gap_size=0.0010)
    return poi_zones._compute_poi_activation_history(
        df, ic_idx=6, cts_established_idx=established_moment, sd=1, scan_end=12,
        fill_threshold=0.70, bos_price=0.6000, cts_events=cts_events, fib_min_pct=61.8,
        fib_max_pct=80.0, variant_thresholds={"V30": 0.3, "V60": 0.6, "V90": 0.9},
        imbalances=[imb], fill_idx_cache={id(imb): (None, None)}, lifecycle_floor_idx=None,
    )


def test_poi_sweep_in_window_cond1_reads_a_pattern_update_anchor():
    """U10 (POI sweep, the IN-WINDOW CTS transition): CTS_ESTABLISHED anchor 3 /
    moment 4 (pre-window; first_active = the IC 6). A lagging pattern-path
    CTS_UPDATED anchored at 5 (before the IC) and known at 7 applies at 7 -- its
    cond1 location is the ANCHOR (5 < 6 -> still not met); a raw update at 11
    moves the CTS past the IC -> active at 11. Reading `ev.idx` (the moment 7)
    for cond1 would activate at 9, when the FVG forms."""
    from engine_v2.structure.event_fields import CTS_UPDATED_RAW_VIA
    from engine_v2.tests._event_factory import make_cts_updated
    evs = [make_cts_established(cts_anchor_idx=3, confirmed_at=4, price=0.6100, struct_direction=None),
           _e4c_upd(5, 7, price=0.6100),
           make_cts_updated(cts_anchor_idx=11, via=CTS_UPDATED_RAW_VIA, price=0.6100,
                            meta={"structure_id": 0, "cycle_id": 0})]
    hist = _e4c_poi_history(evs, established_moment=4)
    assert [(h["idx"], h["active"]) for h in hist] == [(11, True)]
