"""Plan E E2b role pins (landing-review mutation lens): each site that E2b split
into a LOCATION (the anchor) and a TIME (today the anchor too, marked for an E3
stage) is pinned with an E4-shaped event (`idx` = the moment 12, anchor 9) so a
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

@pytest.mark.illegal_event_contract
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
    from engine_v2.tests.test_unified_probe import _make_second_cts_moment_after_extreme_data, _prepare_df
    calls = []
    real = se.compute_poi_inners_for_cycle

    def spy(df, bos_idx, bos_price, cts_idx, *a, **k):
        calls.append((int(cts_idx), k["fill_horizon_idx"]))
        return real(df, bos_idx, bos_price, cts_idx, *a, **k)

    monkeypatch.setattr(se, "compute_poi_inners_for_cycle", spy)
    rows = (_make_second_cts_moment_after_extreme_data() if fixture == "second_cts"
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

def _poi_history(anchor, moment=7):
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
        imbalances=[imb], fill_idx_cache={id(imb): (None, None)}, lifecycle_floor_idx=None,
    )


@pytest.mark.illegal_event_contract
def test_poi_sweep_cond1_reads_the_cts_anchor():
    """An IC (6) past the CTS anchor (5) but before the moment (7): cond1 ("the
    IC lies inside the fib", `cts_at_t >= ic`) reads the ANCHOR → never active.
    A moment read would activate it at 9."""
    assert _poi_history(anchor=5) == []


@pytest.mark.illegal_event_contract
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


@pytest.mark.illegal_event_contract
def test_first_confluence_input_is_the_bos_anchor():
    """B5 / R1: the FC probe input (the pool key) is the BOS ANCHOR; the trigger
    fires at the moment."""
    from engine_v2.multitf.first_confluence_trigger import detect_first_confluence_triggers
    t, = detect_first_confluence_triggers([_e4_bos(structure_id=0, cycle_id=1, price=0.61)])
    assert (t.input_idx, t.trigger_event_idx) == (20, 22)


@pytest.mark.illegal_event_contract
def test_struct_start_and_creation_idx_are_pinned_to_the_stamped_idx():
    """B9 / B10: today's base = the first ANCHOR (BOS_0 20, not its moment 22);
    PLAN_E Q5 moves both to the moment in E3f."""
    from engine_v2.multitf.sid_records import build_sid_records_for_main
    from engine_v2.zones.structure_lifecycle import compute_struct_start_by_sid
    evs = [_e4_est(anchor=21, moment=22, structure_id=0, cycle_id=0),
           _e4_bos(structure_id=0, cycle_id=0)]
    assert compute_struct_start_by_sid(evs, {}, None) == {0: 20}   # Plan E E3f → 22
    rec, = build_sid_records_for_main(evs)
    assert rec.creation_event_idx == 20                            # Plan E E3f → 22


@pytest.mark.illegal_event_contract
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


@pytest.mark.illegal_event_contract
def test_prev_bos_line_runs_anchor_to_anchor():
    """B6 / L9 (Q6): the line starts at sid 0's last BOS ANCHOR and ends at the
    ANCHOR of sid 1's first CTS stamped at/after the reversal (the filter: E3d)."""
    from engine_v2.pipeline.orchestrator import _prev_bos_lines
    from engine_v2.structure import event_fields as ef
    evs = [_e4_bos(anchor=7, moment=10, structure_id=0, cycle_id=1, price=0.95),
           _e4_est(anchor=16, moment=18, structure_id=1, cycle_id=0)]
    evs.sort(key=ef.processing_order_key)
    with contextlib.redirect_stdout(io.StringIO()):
        lines = _prev_bos_lines(evs, {1: 15})
    assert [(ln["start_idx"], ln["end_idx"]) for ln in lines] == [(7, 16)]


@pytest.mark.parametrize("mode", ["h1", "cross_cycle"])
def test_lagging_est_fib_lifecycle_is_timed_at_the_moment(mode):
    """PLAN_E §7 E3a unit: the 2nd CTS_ESTABLISHED has anchor 9 / moment 10.
    Through the downstream pipeline the cycle-1 fib starts (and is stamped
    `activated_at`) at 10, and in cross_cycle the cycle-0 fib's `new_cycle`
    terminal is 10 — the moment, not the anchor 9."""
    from engine_v2.structure.structure_engine import compute_bounded_structure
    from engine_v2.tests.test_e4_simulation import _run
    from engine_v2.tests.test_unified_probe import _make_second_cts_moment_after_extreme_data, _prepare_df
    with contextlib.redirect_stdout(io.StringIO()):
        res = compute_bounded_structure(_prepare_df(_make_second_cts_moment_after_extreme_data()), 0, +1)
    fibs = {(f.structure_id, f.cycle_id): f for f in _run(res.df, res.events, mode)["fib_states"]}
    c1 = fibs[(0, 1)]
    assert (c1.cts_idx, c1.start_idx, c1.meta["activated_at"]) == (9, 10, 10)
    if mode == "cross_cycle":
        assert (fibs[(0, 0)].end_idx, fibs[(0, 0)].end_reason) == (10, "new_cycle")
