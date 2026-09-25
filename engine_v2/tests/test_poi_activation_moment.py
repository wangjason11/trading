"""Plan D (zones pass, 2026-09-22/23) — POI activation gates on the cycle's
CTS-established MOMENT, not on the CTS anchor (`CTS_ESTABLISHED.idx` until
Plan E E4a flipped that idx to the moment; the anchor is `meta["cts_anchor_idx"]`).

`poi_zones.derive_poi_zones` reads the cycle term of the activation floor as
`cts_established_idx = CTS_ESTABLISHED.meta["confirmed_at"]` (the moment, the
canonical cycle lifecycle-start — `structure_lifecycle.compute_cycle_lifecycle`)
and exports the same value as POI `meta["cts_established_idx"]`. Before Plan D
both held `CTS_ESTABLISHED.idx`, which precedes the moment whenever the breakout
pattern's extreme candle is not its apply candle — a POI could go live before
its cycle existed (look-ahead), overlap the previous cycle's POI, and activate
on a cycle collapsed at its moment. See `plans/PLAN_D_poi_activation_moment.md`
§5 and GLOSSARY "Naming Standard" (a past-participle `*_established_idx` names a
moment).

Fixture: the repo's live-MS candles whose cycle-1 CTS anchor (idx 9) precedes
its moment (confirmed_at 10) — `test_unified_probe._make_second_cts_moment_after_extreme_data`.
"""
from __future__ import annotations

import contextlib
import copy
import io

import pandas as pd
import pytest

import engine_v2.zones.poi_zones as poi_zones
from engine_v2.structure import event_fields as ef
from engine_v2.common.types import ImbalanceInstance
from engine_v2.pipeline.orchestrator import _run_downstream_pipeline
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.tests._event_factory import make_cts_established
from engine_v2.structure.structure_engine import compute_bounded_structure
from engine_v2.tests.test_unified_probe import (
    _make_multicycle_data,
    _make_second_cts_moment_after_extreme_data,
    _prepare_df,
)
from engine_v2.zones.poi_zones import POIConfig, derive_poi_zones
from engine_v2.zones.structure_lifecycle import (
    compute_cycle_lifecycle,
    compute_reversal_idx_by_sid,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _run(rows, *, mutate=None, lifecycle_floor=None, lifecycle_cap=None,
         cap_reason="parent_end"):
    """Live MS on the fixture, then the main downstream pipeline (h1 fibs)."""
    df = _prepare_df(rows)
    res = compute_bounded_structure(df, 0, +1)
    events = res.events
    if mutate is not None:
        events = copy.deepcopy(events)
        mutate(events)
        # A post-construction edit bypasses the autouse validator: re-check
        # the contract (the event's idx IS its moment, Plan E E4a / E4b).
        from engine_v2.tests.conftest import validate_event_contract
        for e in events:
            validate_event_contract(e)
    with contextlib.redirect_stdout(io.StringIO()):
        out = _run_downstream_pipeline(
            res.df, events, +1, fib_mode="h1", skip_wvmi=True,
            lifecycle_floor=lifecycle_floor, lifecycle_cap=lifecycle_cap,
            cap_reason=cap_reason,
        )
    return res, events, out


def _hist(zone):
    return [(e["idx"], e["active"]) for e in zone.meta["activation_history"]]


def _only_poi(out, sid, cyc):
    zones = [z for z in out["poi_zones"]
             if (z.meta["structure_id"], z.meta["cycle_id"]) == (sid, cyc)]
    assert len(zones) == 1, [(z.meta["structure_id"], z.meta["cycle_id"], z.ic_idx)
                             for z in out["poi_zones"]]
    return zones[0]


def _event(events, etype, sid, cyc):
    return next(e for e in events if e.type == etype
                and e.meta.get("structure_id") == sid and e.meta.get("cycle_id") == cyc)


def test_fixture_preconditions():
    """The cycle-1 CTS anchor (9) precedes its moment (10); one POI (0,1) at IC 7."""
    _, events, out = _run(_make_second_cts_moment_after_extreme_data())
    ev = _event(events, "CTS_ESTABLISHED", 0, 1)
    assert (ef.cts_anchor_idx(ev), ev.meta["confirmed_at"]) == (9, 10)
    assert [(z.meta["structure_id"], z.meta["cycle_id"], z.ic_idx)
            for z in out["poi_zones"]] == [(0, 1, 7)]


# ---------------------------------------------------------------------------
# (a) — the gate and the exported meta are the moment
# ---------------------------------------------------------------------------

def test_poi_first_activation_is_the_cycle_moment_not_the_anchor():
    _, _, out = _run(_make_second_cts_moment_after_extreme_data())
    z = _only_poi(out, 0, 1)
    assert [(e["idx"], e["active"], e["reason"]) for e in z.meta["activation_history"]] == [
        (10, True, "initial"), (12, False, "imbalance_filled"),
    ]
    assert z.meta["activation_history"][0]["versions"] == ["V30"]
    assert z.meta["confirmed_idx"] == 10
    assert (z.meta["end_idx"], z.meta["end_reason"]) == (14, "reversal")
    assert z.meta["status"] == "ended"
    # The exported cycle term is the moment (GLOSSARY "Naming Standard").
    assert z.meta["cts_established_idx"] == 10


# ---------------------------------------------------------------------------
# (b) — invariant: never before the cycle's lifecycle-start
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "maker", [_make_second_cts_moment_after_extreme_data, _make_multicycle_data],
)
def test_poi_never_activates_before_its_cycle_lifecycle_start(maker):
    _, events, out = _run(maker())
    life = compute_cycle_lifecycle(events, compute_reversal_idx_by_sid(events))
    checked = 0
    for z in out["poi_zones"]:
        history = z.meta["activation_history"]
        if not history:
            continue
        key = (z.meta["structure_id"], z.meta["cycle_id"])
        assert history[0]["idx"] >= life[key][0], (key, history[0]["idx"], life[key][0])
        checked += 1
    assert checked >= 1


# ---------------------------------------------------------------------------
# (c) — it reads confirmed_at, not "idx + 1"
# ---------------------------------------------------------------------------

def test_poi_first_activation_tracks_confirmed_at_exactly():
    def move_moment(events):
        # Keep the definitional identity: BOS and CTS_ESTABLISHED share the moment.
        # Each event's idx IS its moment (Plan E E4a / E4b): it moves with it (a
        # meta edit after construction bypasses the conftest validator; `_run`
        # re-validates).
        for etype in ("CTS_ESTABLISHED", "BOS_CONFIRMED"):
            ev = _event(events, etype, 0, 1)
            ev.meta["confirmed_at"] = 11
            ev.idx = 11

    _, _, out = _run(_make_second_cts_moment_after_extreme_data(), mutate=move_moment)
    z = _only_poi(out, 0, 1)
    assert _hist(z)[0] == (11, True)
    assert z.meta["cts_established_idx"] == 11


# ---------------------------------------------------------------------------
# (d) — anchor == moment: nothing moves
# ---------------------------------------------------------------------------

def test_poi_unchanged_when_anchor_equals_moment():
    _, events, out = _run(_make_multicycle_data())
    z = _only_poi(out, 0, 2)
    ev = _event(events, "CTS_ESTABLISHED", 0, 2)
    assert ef.cts_anchor_idx(ev) == ev.meta["confirmed_at"]
    assert z.ic_idx == 12
    # First activation only: with anchor == moment, first_active is identical
    # under both rules. The full history (incl. the c3-knowability deactivation
    # at 19) is pinned by test_imbalance_c3_knowability (Plan F).
    assert _hist(z)[0] == (15, True)
    assert (z.meta["end_idx"], z.meta["end_reason"]) == (20, "next_cycle")
    assert z.meta["cts_established_idx"] == ev.meta["confirmed_at"]


# ---------------------------------------------------------------------------
# (e) — the lifecycle floor vs the moment. floor <= anchor (9): the moment
#       decides (the IC-678 analogue); floor == moment (10): masked — the
#       activation is the same under both rules and the only delta is the
#       exported meta (the sub-3 analogue); floor > moment (11): the floor
#       decides. The exported cycle term is the moment in every case.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("floor, expected", [
    (9, [(10, True), (12, False)]),     # not masked: the old rule gave 9
    (10, [(10, True), (12, False)]),    # masked
    (11, [(11, True), (12, False)]),    # the floor decides
])
def test_poi_floor_vs_moment(floor, expected):
    _, _, out = _run(_make_second_cts_moment_after_extreme_data(), lifecycle_floor=floor)
    z = _only_poi(out, 0, 1)
    assert _hist(z) == expected
    assert z.meta["cts_established_idx"] == 10


# ---------------------------------------------------------------------------
# (f) — a cycle collapsed at its moment never activates (structure_lifecycle
#       collapsed-cycle contract; POI_ZONES_SPEC §4)
# ---------------------------------------------------------------------------

def test_poi_of_cycle_collapsed_at_its_moment_never_activates():
    _, _, out = _run(_make_second_cts_moment_after_extreme_data(), lifecycle_cap=10)
    z = _only_poi(out, 0, 1)
    assert z.meta["activation_history"] == []
    assert z.meta["confirmed_idx"] is None
    assert z.meta["status"] == "inactive"

    _, _, out = _run(_make_second_cts_moment_after_extreme_data(), lifecycle_cap=11)
    assert _hist(_only_poi(out, 0, 1)) == [(10, True)]


# ---------------------------------------------------------------------------
# (g) — the lookup itself has no silent fallback to ev.idx
# ---------------------------------------------------------------------------

def test_poi_lookup_reads_confirmed_at_without_fallback(monkeypatch):
    res, events, out = _run(_make_second_cts_moment_after_extreme_data())
    stripped = copy.deepcopy(events)
    del _event(stripped, "CTS_ESTABLISHED", 0, 1).meta["confirmed_at"]
    # Stub the upstream assert (compute_cycle_lifecycle) so the lookup's OWN
    # guard is what is tested. The tracker must be non-None: with None,
    # derive_poi_zones returns [] before any check.
    monkeypatch.setattr(poi_zones, "compute_cycle_lifecycle", lambda *a, **k: {})
    with pytest.raises(KeyError, match="confirmed_at"):
        derive_poi_zones(res.df, stripped, fib_tracker=out["fib_tracker"], config=POIConfig())


@pytest.mark.skipif(not __debug__, reason="asserts are stripped under python -O")
def test_derive_poi_zones_raises_on_cts_established_without_confirmed_at():
    """Without the stub the struct_start base (Plan E E3f: the first
    CTS_ESTABLISHED moment, read directly) raises first — loudly, as a KeyError."""
    res, events, out = _run(_make_second_cts_moment_after_extreme_data())
    stripped = copy.deepcopy(events)
    del _event(stripped, "CTS_ESTABLISHED", 0, 1).meta["confirmed_at"]
    with pytest.raises(KeyError, match="confirmed_at"):
        derive_poi_zones(res.df, stripped, fib_tracker=out["fib_tracker"], config=POIConfig())


# ---------------------------------------------------------------------------
# (h) — the pre-window loop applies the CTS state before the sweep (it is
#       load-bearing now that first_active sits past the CTS anchor)
# ---------------------------------------------------------------------------

def test_activation_applies_pre_window_cts_state():
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    df = pd.DataFrame([
        {"time": t0 + pd.Timedelta(hours=i), "o": 0.6040, "h": 0.6045, "l": 0.6035, "c": 0.6040}
        for i in range(10)
    ])
    # IC at 2, inside the 61.8-80 band of (BOS 0.6000, CTS 0.6100) = [0.6020, 0.60382].
    df.loc[2, ["o", "h", "l", "c"]] = [0.6034, 0.6035, 0.6025, 0.6026]
    imb = ImbalanceInstance(start_idx=3, end_idx=3, direction=1,
                            gap_top=0.6090, gap_bottom=0.6080, gap_size=0.0010)
    cts = make_cts_established(cts_anchor_idx=5, confirmed_at=6, price=0.6100, struct_direction=None)
    history = poi_zones._compute_poi_activation_history(
        df,
        ic_idx=2, cts_established_idx=6, sd=1, scan_end=9, fill_threshold=0.70,
        bos_price=0.6000, cts_events=[cts], fib_min_pct=61.8, fib_max_pct=80.0,
        variant_thresholds={"V30": 0.3, "V60": 0.6, "V90": 0.9},
        imbalances=[imb], fill_idx_cache={id(imb): (None, None)}, lifecycle_floor_idx=None,
    )
    # The CTS (anchor 5, moment 6) is applied BEFORE the sweep; the imbalance (formed at 4 —
    # its first c3, Plan F) enters at first_active; so the POI is active exactly
    # AT the moment, with variants computed from the pre-window CTS state.
    assert history and history[0]["idx"] == 6 and history[0]["active"]
    assert history[0]["versions"]


def test_activation_cts_update_applies_at_its_moment_not_its_anchor():
    """Plan E E3g-1 (§7.1 T1): a CTS event's transition time is its MOMENT. IC at 7
    lies past the EST anchor 5 (so cond1 fails until the CTS reaches it); a
    pattern-path CTS_UPDATED anchored at 8 is known only at 11; the imbalance
    forms at 9. Keyed on the anchor the POI would go active at 9; on the moment
    it goes active at 11."""
    from engine_v2.tests._event_factory import make_event
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    df = pd.DataFrame([
        {"time": t0 + pd.Timedelta(hours=i), "o": 0.6040, "h": 0.6045, "l": 0.6035, "c": 0.6040}
        for i in range(14)
    ])
    df.loc[7, ["o", "h", "l", "c"]] = [0.6034, 0.6035, 0.6025, 0.6026]
    imb = ImbalanceInstance(start_idx=8, end_idx=8, direction=1,
                            gap_top=0.6090, gap_bottom=0.6080, gap_size=0.0010)
    est = make_cts_established(cts_anchor_idx=5, confirmed_at=6, price=0.6100, struct_direction=None)
    upd = make_event("CTS_UPDATED", 8, price=0.6105, via="continuous", confirmed_at=11,
                     structure_id=0, cycle_id=0)
    history = poi_zones._compute_poi_activation_history(
        df,
        ic_idx=7, cts_established_idx=6, sd=1, scan_end=13, fill_threshold=0.70,
        bos_price=0.6000, cts_events=[est, upd], fib_min_pct=61.8, fib_max_pct=80.0,
        variant_thresholds={"V30": 0.3, "V60": 0.6, "V90": 0.9},
        imbalances=[imb], fill_idx_cache={id(imb): (None, None)}, lifecycle_floor_idx=None,
    )
    assert history and history[0]["idx"] == 11 and history[0]["active"]


def _e3g1_history(cts_events, *, ic_idx, floor, imb_start, n=16):
    from engine_v2.tests._event_factory import make_event  # noqa: F401
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    df = pd.DataFrame([
        {"time": t0 + pd.Timedelta(hours=i), "o": 0.6040, "h": 0.6045, "l": 0.6035, "c": 0.6040}
        for i in range(n)
    ])
    df.loc[ic_idx, ["o", "h", "l", "c"]] = [0.6034, 0.6035, 0.6025, 0.6026]
    imb = ImbalanceInstance(start_idx=imb_start, end_idx=imb_start, direction=1,
                            gap_top=0.6090, gap_bottom=0.6080, gap_size=0.0010)
    return poi_zones._compute_poi_activation_history(
        df,
        ic_idx=ic_idx, cts_established_idx=6, sd=1, scan_end=n - 1, fill_threshold=0.70,
        bos_price=0.6000, cts_events=cts_events, fib_min_pct=61.8, fib_max_pct=80.0,
        variant_thresholds={"V30": 0.3, "V60": 0.6, "V90": 0.9},
        imbalances=[imb], fill_idx_cache={id(imb): (None, None)}, lifecycle_floor_idx=floor,
    )


def test_activation_pre_window_split_is_the_moment():
    """Plan E E3g-1: the pre-window split keys on the MOMENT. Floor 9 → first_active 9;
    a pattern-path CTS_UPDATED anchored at 8 (< 9) is known at 11 (> 9) → it is an
    in-window transition at 11, not pre-window state (which would activate at 9)."""
    from engine_v2.tests._event_factory import make_event
    est = make_cts_established(cts_anchor_idx=5, confirmed_at=6, price=0.6100, struct_direction=None)
    upd = make_event("CTS_UPDATED", 8, price=0.6105, via="continuous", confirmed_at=11,
                     structure_id=0, cycle_id=0)
    history = _e3g1_history([est, upd], ic_idx=7, floor=9, imb_start=8)   # formed at 9
    assert history and history[0]["idx"] == 11


def test_activation_applies_cts_events_in_moment_order():
    """Plan E E3g-1: the sweep applies CTS events in MOMENT order. A pattern-path
    update that regresses the CTS (zones-audit latent (a): anchor 8, known at 12)
    lands after a raw update to 9 (known at 9); pre-window at first_active 13 the
    latest KNOWN CTS is anchor 8 < IC 9 → cond1 fails (in stamped order the raw
    9 would be applied last and the POI would activate at 13)."""
    from engine_v2.structure.event_fields import CTS_UPDATED_RAW_VIA
    from engine_v2.tests._event_factory import make_event
    est = make_cts_established(cts_anchor_idx=5, confirmed_at=6, price=0.6100, struct_direction=None)
    raw = make_event("CTS_UPDATED", 9, price=0.6110, via=CTS_UPDATED_RAW_VIA, structure_id=0, cycle_id=0)
    pat = make_event("CTS_UPDATED", 8, price=0.6105, via="continuous", confirmed_at=12,
                     structure_id=0, cycle_id=0)
    history = _e3g1_history([est, pat, raw], ic_idx=9, floor=13, imb_start=10)   # formed at 11
    assert history == []


# ---------------------------------------------------------------------------
# (i) — the fallback for a cycle with no CTS_ESTABLISHED is unchanged (Plan D
#       decision 3; the fib's CTS anchor — a known naming-standard exception)
# ---------------------------------------------------------------------------

def test_fallback_cycle_without_cts_established_keeps_fib_anchor():
    res, events, out = _run(_make_second_cts_moment_after_extreme_data())
    fib = next(f for f in out["fib_tracker"].get_fibs_for_charting()
               if (f.structure_id, f.cycle_id) == (0, 1))
    without = [e for e in events
               if not (e.type == "CTS_ESTABLISHED" and e.meta.get("structure_id") == 0
                       and e.meta.get("cycle_id") == 1)]
    with contextlib.redirect_stdout(io.StringIO()):
        zones = derive_poi_zones(res.df, without, fib_tracker=out["fib_tracker"], config=POIConfig())
    z = next(z for z in zones if (z.meta["structure_id"], z.meta["cycle_id"]) == (0, 1))
    assert z.meta["cts_established_idx"] == int(fib.cts_idx)
    assert z.meta["activation_history"] == []
    assert z.meta["status"] == "inactive"
