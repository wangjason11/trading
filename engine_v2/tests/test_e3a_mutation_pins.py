"""Plan E E3a mutation-review pins: every FibTracker TIME half on the H1-main
sid>=1 scenario paths + the update path is the handled event's MOMENT, never
the CTS anchor. Every event is LAGGING (moment = anchor + 2; legal contract:
`idx` = the anchor, `confirmed_at` = the moment), so an anchor read fails.

Grid (sid 1, sd +1; gaps 1.00-1.10 @15, 1.20-1.30 @35, 1.40-1.50 @55):
cycle 0 BOS 10 / CTS anchor 20 / moment 22; cycle 1 BOS 30 / 40 / 42;
cycle 2 BOS 50 / 60 / 62.
"""
from __future__ import annotations

import contextlib
import io

import pandas as pd
import pytest

from engine_v2.common.types import ImbalanceInstance
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.tests._event_factory import make_cts_established
from engine_v2.zones.fib_tracker import FibTracker, FibTrackerConfig

LAG = 2


def _df(n=80, insts=None, fills=()):
    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n, freq="h"),
        "o": [1.5] * n, "h": [1.501] * n, "l": [1.499] * n, "c": [1.5] * n,
        "direction": [0] * n, "is_imbalance": [0] * n,
    })
    df.attrs["imbalances"] = list(insts if insts is not None else [
        ImbalanceInstance(15, 15, 1, 1.10, 1.00, 0.10),
        ImbalanceInstance(35, 35, 1, 1.30, 1.20, 0.10),
        ImbalanceInstance(55, 55, 1, 1.50, 1.40, 0.10),
    ])
    for idx, low in fills:
        df.at[idx, "l"] = low
    return df


def _est(anchor, price, sid, cyc):
    return make_cts_established(cts_anchor_idx=anchor, confirmed_at=anchor + LAG, price=price,
                                structure_id=sid, cycle_id=cyc, struct_direction=1)


def _upd(anchor, price, sid, cyc):
    """A pattern-path CTS_UPDATED: anchor `anchor`, moment `anchor + LAG` (its idx
    since Plan E E4c)."""
    return StructureEvent(idx=anchor + LAG, category="STRUCTURE", type="CTS_UPDATED", price=price,
                          meta={"structure_id": sid, "cycle_id": cyc, "struct_direction": 1,
                                "via": "one_maru_continuous", "confirmed_at": anchor + LAG,
                                "cts_anchor_idx": anchor})


def _conf(idx, anchor, price, sid, cyc):
    return StructureEvent(idx=idx, category="STRUCTURE", type="CTS_CONFIRMED", price=price,
                          meta={"structure_id": sid, "cycle_id": cyc, "struct_direction": 1,
                                "cts_anchor_idx": anchor})


def _t(mode="h1"):
    return FibTracker(FibTrackerConfig(fill_threshold=0.70), fib_mode=mode)


def _q(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _c0(t, df, rv, **pb):
    return _q(t.on_cts_established, _est(20, 1.15, 1, 0), df, bos_idx=10, bos_price=1.0,
              reversal_confirmed_idx=rv, **pb)


def _c1(t, df, rv, bos_price=1.2, **pb):
    _q(t.on_cts_confirmed, _conf(25, 20, 1.15, 1, 0))
    return _q(t.on_cts_established, _est(40, 1.35, 1, 1), df, bos_idx=30, bos_price=bos_price,
              reversal_confirmed_idx=rv, **pb)


def _c2(t, df, rv, **pb):
    _q(t.on_cts_confirmed, _conf(45, 40, 1.35, 1, 1))
    return _q(t.on_cts_established, _est(60, 1.55, 1, 2), df, bos_idx=50, bos_price=1.4,
              reversal_confirmed_idx=rv, **pb)


# --- CTS_ESTABLISHED: the Scenario-1 comparison + every activation stamp ----------

@pytest.mark.parametrize("rv", [21, 22])   # anchor 20 < rv <= moment 22
def test_est_scenario1_is_decided_at_the_moment(rv):
    t = _t()
    fib = _c0(t, _df(), rv)
    assert t._scenario1[1] is True
    assert (fib.cts_idx, fib.meta["activated_at"]) == (20, 22)
    c1 = _c1(t, _df(), rv)                                   # Scenario 1 stays TRUE
    assert (c1.cts_idx, c1.meta["activated_at"], c1.meta["scenario1"]) == (40, 42, True)


def test_est_scenario1_revert_terminal_is_the_moment():
    t = _t()
    _c0(t, _df(), 21)
    _c1(t, _df(), 21, bos_price=1.2, prev_bos_outer=1.1, prev_sd=1)   # BOS_1 1.2 >= 1.1 → revert
    assert t._scenario1[1] is False
    assert t._terminal[(1, 0)] == (42, "scenario1_revert")


def test_est_cycle1_without_cycle0_data_stamps_the_moment():
    t = _t()
    fib = _q(t.on_cts_established, _est(40, 1.35, 1, 1), _df(), bos_idx=30, bos_price=1.2,
             reversal_confirmed_idx=100)
    assert (fib.cts_idx, fib.meta["activated_at"]) == (40, 42)


def test_est_scenario2_cross_and_scenario3_stamp_the_moment():
    t = _t()
    _c0(t, _df(), 100)
    cross = _c1(t, _df(), 100)
    assert cross.meta["scenario"] == 2 and cross.meta["activated_at"] == 42
    t3 = _t()
    df3 = _df(insts=[ImbalanceInstance(35, 35, 1, 1.30, 1.20, 0.10)])
    _c0(t3, df3, 100)
    s3 = _c1(t3, df3, 100)
    assert s3.meta["scenario"] == 3 and s3.meta["activated_at"] == 42


def test_est_scenario2_cond1_fill_horizon_is_the_moment():
    """Cycle-1 gap @35 fills at 41: after the anchor 40, by the moment 42 → cond1
    false → no cross, and the cycle-1 own check is filled too → nothing activates."""
    t = _t()
    df = _df(fills=[(41, 1.22)])
    _c0(t, df, 100)
    assert _c1(t, df, 100) is None
    assert t._get_latest_cross(1, 1) is None


def test_est_gate_fill_horizon_is_the_moment():
    """sid 0 simple flow: the only gap (@15) fills at 21 — after the anchor 20,
    by the moment 22 → no fib."""
    t = _t()
    df = _df(insts=[ImbalanceInstance(15, 15, 1, 1.10, 1.00, 0.10)], fills=[(21, 1.02)])
    ev = make_cts_established(cts_anchor_idx=20, confirmed_at=22, price=1.2, structure_id=0,
                              cycle_id=1, struct_direction=1)
    assert _q(t.on_cts_established, ev, df, bos_idx=10, bos_price=0.9) is None


def test_est_cycle2_single_and_cross_stamp_the_moment():
    t = _t()                                              # P_rev 1.30: cycle 1 clears → single
    pb = dict(prev_bos_outer=1.30, prev_sd=1)
    _c0(t, _df(), 100, **pb); _c1(t, _df(), 100, **pb)
    single = _c2(t, _df(), 100, **pb)
    assert single.meta.get("cross_cycle") is not True
    assert (single.cts_idx, single.meta["activated_at"]) == (60, 62)
    t2 = _t()                                             # P_rev 2.0: nothing clears → cross
    pb2 = dict(prev_bos_outer=2.0, prev_sd=1)
    _c0(t2, _df(), 100, **pb2); _c1(t2, _df(), 100, **pb2)
    cross = _c2(t2, _df(), 100, **pb2)
    assert cross.meta.get("cross_cycle") is True
    assert (cross.cts_idx, cross.meta["activated_at"]) == (60, 62)


def test_est_main_cross_peek_fill_horizon_is_the_moment():
    """§11b peek: the cycle-1 gap (@35, 1.45-1.50) fills at 61 — after the cycle-2
    anchor 60, by its moment 62 (cycle 0's gap died at 30; the cycle-2 own gap
    @55 sits lower, 1.20-1.30, and stays unfilled). Peeked at the moment no
    prior cycle is eligible → the plain single; a peek at the anchor would
    route into the cross machinery and come back a `cross_failed` single with
    cycle 1 recorded dead."""
    t = _t()
    pb = dict(prev_bos_outer=2.0, prev_sd=1)
    df = _df(insts=[ImbalanceInstance(15, 15, 1, 1.10, 1.00, 0.10),
                    ImbalanceInstance(35, 35, 1, 1.50, 1.45, 0.05),
                    ImbalanceInstance(55, 55, 1, 1.30, 1.20, 0.10)],
             fills=[(30, 1.02), (61, 1.46)])
    _c0(t, df, 100, **pb); _c1(t, df, 100, **pb)
    c2 = _c2(t, df, 100, **pb)
    assert (c2.bos_idx, c2.cts_idx, c2.meta) == (50, 60, {"activated_at": 62})
    assert not t._dead_cycles.get(1)


# --- CTS_UPDATED (pattern path, anchor A / moment A + 2) ------------------------

def test_update_cross_cycle_cycle0_first_activation_is_asked_at_the_moment():
    """cross_cycle cycle 0: the only gap (c2 21) lies past the EST anchor 20 → no
    fib at EST. The update (anchor 23, moment 25) covers it, but it fills at 24
    → asked at the moment: nothing activates."""
    t = _t("cross_cycle")
    df = _df(insts=[ImbalanceInstance(21, 21, 1, 1.10, 1.00, 0.10)], fills=[(24, 1.02)])
    assert _c0(t, df, 100) is None
    assert _q(t.on_cts_updated, _upd(23, 1.2, 1, 0), df) is None
    assert (1, 0) not in t._fibs


def test_update_cross_cycle_cycle1_own_fill_is_asked_at_the_moment():
    """cross_cycle cycle 1: the own gap (@35) fills at 46, between the update's
    anchor 45 and moment 47 → the cross deactivates, stamped 47."""
    t = _t("cross_cycle")
    df = _df(fills=[(46, 1.22)])
    _c0(t, df, 100); _c1(t, df, 100)
    _q(t.on_cts_updated, _upd(45, 1.40, 1, 1), df)
    cross = t._fibs[(1, 1, "cross", 0)]
    assert not cross.active and cross.meta["deactivated_at"] == 47


def test_update_h1_cycle2_cross_is_checked_at_the_moment():
    """h1 §11b cycle-2 cross: the own gap (@55) fills at 66, between the update's
    anchor 65 and moment 67 → the cross deactivates, stamped 67."""
    t = _t()
    pb = dict(prev_bos_outer=2.0, prev_sd=1)
    df = _df(fills=[(66, 1.42)])
    _c0(t, df, 100, **pb); _c1(t, df, 100, **pb); _c2(t, df, 100, **pb)
    _q(t.on_cts_updated, _upd(65, 1.60, 1, 2), df)
    cross = t._fibs[(1, 2, "cross", 0)]
    assert not cross.active and cross.meta["deactivated_at"] == 67


@pytest.mark.parametrize("rv", [24, 25])   # EST moment 22 < rv; update anchor 23 < rv <= moment 25
def test_update_scenario1_is_decided_at_the_moment(rv):
    t = _t()
    assert _c0(t, _df(), rv) is None and t._scenario1[1] is None
    fib = _q(t.on_cts_updated, _upd(23, 1.2, 1, 0), _df(), reversal_confirmed_idx=rv)
    assert t._scenario1[1] is True
    assert (fib.cts_idx, fib.meta["activated_at"]) == (23, 25)


def test_update_scenario1_true_late_activation_stamps_the_moment():
    """Scenario 1 TRUE at EST (rv 21) but the only gap (c2 22) lies past the EST
    anchor → no fib; the update (anchor 23, moment 25) activates it at 25."""
    t = _t()
    df = _df(insts=[ImbalanceInstance(22, 22, 1, 1.10, 1.00, 0.10)])
    assert _c0(t, df, 21) is None and t._scenario1[1] is True
    fib = _q(t.on_cts_updated, _upd(23, 1.2, 1, 0), df, reversal_confirmed_idx=21)
    assert (fib.cts_idx, fib.meta["activated_at"]) == (23, 25)


def test_update_fib_cts_reactivation_stamps_the_moment():
    """sid 0: gap @15 fills at 31 → the update (30 / 32) deactivates; a new gap
    (c2 33) → the update (35 / 37) reactivates, stamped 37."""
    t = _t()
    df = _df(insts=[ImbalanceInstance(15, 15, 1, 1.10, 1.00, 0.10),
                    ImbalanceInstance(33, 33, 1, 1.40, 1.30, 0.10)], fills=[(31, 0.99)])
    ev = make_cts_established(cts_anchor_idx=20, confirmed_at=22, price=1.2, structure_id=0,
                              cycle_id=1, struct_direction=1)
    assert _q(t.on_cts_established, ev, df, bos_idx=10, bos_price=0.9).active
    assert not _q(t.on_cts_updated, _upd(30, 1.3, 0, 1), df).active
    re = _q(t.on_cts_updated, _upd(35, 1.45, 0, 1), df)
    assert re.active and re.meta["reactivated_at"] == 37


def test_update_cycle1_main_cross_is_checked_at_the_moment():
    """h1 Scenario-2 cross: the cycle-1 gap (@35) fills at 46, between the
    update's anchor 45 and moment 47 → cond2 false → deactivated at 47, and the
    normal own check (same window, same moment) is filled too → no single."""
    t = _t()
    df = _df(fills=[(46, 1.22)])
    _c0(t, df, 100); _c1(t, df, 100)
    _q(t.on_cts_updated, _upd(45, 1.40, 1, 1), df)
    cross = t._fibs[(1, 1, "cross", 0)]
    assert not cross.active and cross.meta["deactivated_at"] == 47
    assert (1, 1) not in t._fibs


def test_update_cycle1_main_cross_reactivation_stamps_the_moment():
    t = _t()
    df = _df(insts=[ImbalanceInstance(15, 15, 1, 1.10, 1.00, 0.10),
                    ImbalanceInstance(35, 35, 1, 1.30, 1.20, 0.10),
                    ImbalanceInstance(48, 48, 1, 1.45, 1.40, 0.05)], fills=[(44, 1.22)])
    _c0(t, df, 100); _c1(t, df, 100)
    _q(t.on_cts_updated, _upd(45, 1.40, 1, 1), df)          # deactivated (filled at 44)
    assert not t._fibs[(1, 1, "cross", 0)].active
    _q(t.on_cts_updated, _upd(50, 1.50, 1, 1), df)          # new gap c2 48 → reactivated
    cross = t._fibs[(1, 1, "cross", 0)]
    assert cross.active and cross.meta["reactivated_at"] == 52


# --- the MS in-flight mirror ------------------------------------------------------

def _mirror(rows, k=1.2300):
    """Price-mirror an sd=+1 fixture into its sd=-1 twin (h <-> l)."""
    return [{"o": round(k - r["o"], 5), "h": round(k - r["l"], 5),
             "l": round(k - r["h"], 5), "c": round(k - r["c"], 5)} for r in rows]


def test_ms_inflight_poi_refresh_fill_horizon_is_the_moment_bearish(monkeypatch):
    """The sd=-1 twin of the role-pin test: the bearish raw-update branch hands
    its processing candle; the pattern tying the raw-updated CTS (extreme 24,
    apply 25) refreshes nothing (2026-09-27: strict new extreme on both paths)."""
    import engine_v2.structure.structure_engine as se
    from engine_v2.tests.test_imbalance_c3_knowability import _multicycle_with_tied_pattern_breakout
    from engine_v2.tests.test_unified_probe import _prepare_df
    calls = []
    real = se.compute_poi_inners_for_cycle

    def spy(df, bos_idx, bos_price, cts_idx, *a, **k):
        calls.append((int(cts_idx), k["fill_horizon_idx"]))
        return real(df, bos_idx, bos_price, cts_idx, *a, **k)

    monkeypatch.setattr(se, "compute_poi_inners_for_cycle", spy)
    with contextlib.redirect_stdout(io.StringIO()):
        res = se.compute_bounded_structure(
            _prepare_df(_mirror(_multicycle_with_tied_pattern_breakout())), 0, -1)
    raw = [e.idx for e in res.events if e.type == "CTS_UPDATED" and e.meta["via"] == "replay_raw"]
    assert raw                                     # the fixture exercises the bearish raw branch
    assert all((r, r) in calls for r in raw)
    assert int(res.df["last_breakout_pat_apply_idx"].iloc[25]) == 25   # precondition: the tying breakout WAS applied at 25
    assert (24, 24) in calls and not [h for _c, h in calls if h == 25]


def test_compute_poi_inners_uses_its_fill_horizon(monkeypatch):
    """`compute_poi_inners_for_cycle` forwards `fill_horizon_idx` (not `cts_idx`)
    as the Scenario-2 cond1 horizon: gap @40 fills at 41 → with the horizon 42
    cond1 is false → scenario_3."""
    import engine_v2.zones.poi_zones as poi_zones
    seen = []
    real = poi_zones.select_fib_anchor_for_cycle

    def spy(*a, **k):
        out = real(*a, **k)
        seen.append((k["fill_horizon_idx"], out[-1]))
        return out

    monkeypatch.setattr(poi_zones, "select_fib_anchor_for_cycle", spy)
    df = _df(60, [ImbalanceInstance(15, 15, 1, 1.10, 1.00, 0.10),
                  ImbalanceInstance(40, 40, 1, 1.30, 1.20, 0.10)], fills=[(41, 1.22)])
    c0 = {"bos_idx": 10, "bos_price": 1.0, "cts_idx": 20, "cts_price": 1.15,
          "has_unfilled": True, "scenario1": None}
    poi_zones.compute_poi_inners_for_cycle(df, 30, 1.2, 40, 1.35, 1, structure_id=1, cycle_id=1,
                                           c0_data=c0, fill_horizon_idx=42,
                                           snapshot_horizon_idx=42)
    assert seen == [(42, "scenario_3")]


# --- Plan E E3a′: the cycle-0 caches, cond3, `_c0_has_unfilled_now` ---------------
# (The update-path cond3 and cond1-c0 re-asks in `_update_cycle1_main` are
# equivalent to their anchor reverts on every reachable stream: a fill that makes
# the two horizons disagree already fails the same check at CTS_1 EST, so no cross
# reaches the update path.)

def test_e3ap_est_cycle0_cache_is_asked_at_the_cts0_moment():
    """The cycle-0 gap (@15) fills at 21 — after the CTS_0 anchor 20, by its
    moment 22 → the uncut cache says filled, and records its horizon 22."""
    t = _t()
    _c0(t, _df(fills=[(21, 1.02)]), 100)
    c0 = t._cross_cycle_data[1]["cycle0"]
    assert (c0["has_unfilled"], c0["fill_horizon_idx"]) == (False, 22)


def test_e3ap_update_cycle0_cache_is_asked_at_the_update_moment():
    """A CTS_0 update (anchor 23, moment 25): the gap fills at 24 → the
    re-snapshot says filled, horizon 25."""
    t = _t()
    df = _df(fills=[(24, 1.02)])
    _c0(t, df, 100)
    _q(t.on_cts_updated, _upd(23, 1.2, 1, 0), df, reversal_confirmed_idx=100)
    c0 = t._cross_cycle_data[1]["cycle0"]
    assert (c0["cts_idx"], c0["has_unfilled"], c0["fill_horizon_idx"]) == (23, False, 25)


def test_e3ap_scenario2_cond3_is_asked_at_the_bos1_moment():
    """cond3 "has BOS_1 filled cycle 0?" at BOS_1's MOMENT (== the CTS_1 EST
    moment 42), not its anchor 30: the cycle-0 gap fills at 35 → no cross
    (Scenario 3); asked at 30 it would still be unfilled → a Scenario-2 cross."""
    t = _t()
    df = _df(fills=[(35, 1.02)])
    _c0(t, df, 100)
    fib = _c1(t, df, 100)
    assert fib.meta["scenario"] == 3
    assert t._get_latest_cross(1, 1) is None


def test_e3ap_c0_has_unfilled_now_is_asked_at_the_update_moment():
    """Scenario 1 TRUE at EST (rv 21), the only gap (c2 21) outside [10, 20] → no
    fib. The update (anchor 23, moment 25) brings it in range, but it fills at
    24 → asked at the moment: no activation (at the anchor 23 it would activate)."""
    t = _t()
    df = _df(insts=[ImbalanceInstance(21, 21, 1, 1.10, 1.00, 0.10)], fills=[(24, 1.02)])
    assert _c0(t, df, 21) is None and t._scenario1[1] is True
    assert _q(t.on_cts_updated, _upd(23, 1.2, 1, 0), df, reversal_confirmed_idx=21) is None
    assert (1, 0) not in t._fibs


def test_e3ap_ms_cycle0_snapshot_is_asked_at_the_refresh_moment():
    """The MS mirror: CTS_0 anchor 20, refresh moment 22, the gap fills at 21 →
    filled (lock-step with FibTracker's cache)."""
    from types import SimpleNamespace
    from engine_v2.structure.market_structure import MarketStructure
    ms = MarketStructure.__new__(MarketStructure)
    ms.df = _df(fills=[(21, 1.02)])
    ms._fill_threshold = 0.70
    ms.state = SimpleNamespace(cts_cycle_id=0, cts=SimpleNamespace(idx=20, price=1.15),
                               bos=SimpleNamespace(idx=10, price=1.0),
                               struct_direction=1, cycle0_data=None)
    ms._update_cycle0_data(22)
    assert ms.state.cycle0_data["has_unfilled"] is False


def test_e3ap_compute_poi_inners_uses_its_snapshot_horizon(monkeypatch):
    """`compute_poi_inners_for_cycle` forwards `snapshot_horizon_idx` (not
    `bos_idx`) as cond3's horizon: the cycle-0 gap fills at 35 → asked at 42 →
    scenario_3 (at the BOS anchor 30 it would be a cross)."""
    import engine_v2.zones.poi_zones as poi_zones
    seen = []
    real = poi_zones.select_fib_anchor_for_cycle

    def spy(*a, **k):
        out = real(*a, **k)
        seen.append((k["snapshot_horizon_idx"], out[-1]))
        return out

    monkeypatch.setattr(poi_zones, "select_fib_anchor_for_cycle", spy)
    df = _df(60, [ImbalanceInstance(15, 15, 1, 1.10, 1.00, 0.10),
                  ImbalanceInstance(35, 35, 1, 1.30, 1.20, 0.10)], fills=[(35, 1.02)])
    c0 = {"bos_idx": 10, "bos_price": 1.0, "cts_idx": 20, "cts_price": 1.15,
          "has_unfilled": True, "scenario1": None}
    poi_zones.compute_poi_inners_for_cycle(df, 30, 1.2, 40, 1.35, 1, structure_id=1, cycle_id=1,
                                           c0_data=c0, fill_horizon_idx=42, snapshot_horizon_idx=42)
    assert seen == [(42, "scenario_3")]


def test_e3ap_ms_refresh_snapshot_horizon_is_the_cycle_established_moment(monkeypatch):
    """MS passes the current cycle's CTS_ESTABLISHED moment as the snapshot
    horizon: on the lagging-EST fixture (anchor 9, moment 10) every cycle-1
    refresh carries 10."""
    import engine_v2.structure.structure_engine as se
    from engine_v2.tests.test_unified_probe import _make_second_cts_moment_after_anchor_data, _prepare_df
    calls = []
    real = se.compute_poi_inners_for_cycle

    def spy(df, bos_idx, bos_price, cts_idx, cts_price, sd, sid, cycle_id, *a, **k):
        calls.append((cycle_id, k["snapshot_horizon_idx"]))
        return real(df, bos_idx, bos_price, cts_idx, cts_price, sd, sid, cycle_id, *a, **k)

    monkeypatch.setattr(se, "compute_poi_inners_for_cycle", spy)
    with contextlib.redirect_stdout(io.StringIO()):
        se.compute_bounded_structure(_prepare_df(_make_second_cts_moment_after_anchor_data()), 0, +1)
    assert (1, 10) in calls
    assert all(h == 10 for c, h in calls if c == 1)


def _raw_upd(i, price, sid, cyc):
    from engine_v2.structure.event_fields import CTS_UPDATED_RAW_VIA
    return StructureEvent(idx=i, category="STRUCTURE", type="CTS_UPDATED", price=price,
                          meta={"structure_id": sid, "cycle_id": cyc, "struct_direction": 1,
                                "via": CTS_UPDATED_RAW_VIA})


def _pat_upd(anchor, moment, price, sid, cyc):
    return StructureEvent(idx=moment, category="STRUCTURE", type="CTS_UPDATED", price=price,
                          meta={"structure_id": sid, "cycle_id": cyc, "struct_direction": 1,
                                "via": "one_maru_continuous", "confirmed_at": moment,
                                "cts_anchor_idx": anchor})


def test_e3ap_c0_now_on_an_equal_anchor_pattern_update():
    """(E3a′ landing review.) Scenario 1 undetermined at EST (22 < rv 24) and at
    the raw update (23); the pattern update restates anchor 23 at moment 25
    (>= rv) → Scenario 1 TRUE. The gap fills at 24 → asked at the moment: no
    activation (a read of an older horizon 23 would activate). The cycle-0 cache
    re-snapshots on the equal-anchor update too — horizon 25, filled — in
    lock-step with the MS mirror, which re-snapshots on every cycle-0 refresh.
    The equal-anchor update is (a synthetic stream: MS emits no such update since 2026-09-27; the reader's handling is defence in depth)."""
    t = _t()
    df = _df(fills=[(24, 1.02)])
    assert _c0(t, df, 24) is None and t._scenario1[1] is None
    assert _q(t.on_cts_updated, _raw_upd(23, 1.2, 1, 0), df, reversal_confirmed_idx=24) is None
    assert _q(t.on_cts_updated, _pat_upd(23, 25, 1.2, 1, 0), df, reversal_confirmed_idx=24) is None
    assert t._scenario1[1] is True and (1, 0) not in t._fibs
    c0 = t._cross_cycle_data[1]["cycle0"]
    assert (c0["cts_idx"], c0["fill_horizon_idx"], c0["has_unfilled"]) == (23, 25, False)


def test_e3ap_cycle0_cache_matches_the_ms_mirror_on_an_equal_anchor_update():
    """Parity pin (LANDMINES "Scenario 2 anchor agreement"): feed FibTracker and
    the MS `_update_cycle0_data` the same stream — EST (20 / 22), raw @23, pattern
    restating 23 at 25; the gap fills at 24 — both caches end (has_unfilled False)."""
    from types import SimpleNamespace
    from engine_v2.structure.market_structure import MarketStructure
    df = _df(fills=[(24, 1.02)])
    t = _t()
    _c0(t, df, 100)
    _q(t.on_cts_updated, _raw_upd(23, 1.2, 1, 0), df, reversal_confirmed_idx=100)
    _q(t.on_cts_updated, _pat_upd(23, 25, 1.2, 1, 0), df, reversal_confirmed_idx=100)
    ms = MarketStructure.__new__(MarketStructure)
    ms.df = df
    ms._fill_threshold = 0.70
    ms.state = SimpleNamespace(cts_cycle_id=0, cts=SimpleNamespace(idx=20, price=1.15),
                               bos=SimpleNamespace(idx=10, price=1.0),
                               struct_direction=1, cycle0_data=None)
    for cts, moment in ((20, 22), (23, 23), (23, 25)):
        ms.state.cts = SimpleNamespace(idx=cts, price=1.2)
        ms._update_cycle0_data(moment)
    assert t._cross_cycle_data[1]["cycle0"]["has_unfilled"] == ms.state.cycle0_data["has_unfilled"] is False


def test_e3ap_c0_now_asks_at_the_moment_not_the_cache_horizon():
    """`_c0_has_unfilled_now` asks at the update's moment even when the cache was
    not re-snapshotted — the one such stream is a pattern update whose anchor
    REGRESSES below the cached one (the latent "pattern-path CTS_UPDATED
    regressing st.cts" shape, zones-audit memory;(a synthetic stream: MS emits no such update since 2026-09-27; the reader's handling is defence in depth). EST 20 / 22 and raw @23 leave
    Scenario 1 undetermined (rv 26); the pattern (anchor 21, moment 27) makes it
    TRUE; the cache keeps horizon 23, the gap fills at 24 → asked at 27: no fib."""
    t = _t()
    df = _df(fills=[(24, 1.02)])
    assert _c0(t, df, 26) is None
    _q(t.on_cts_updated, _raw_upd(23, 1.2, 1, 0), df, reversal_confirmed_idx=26)
    assert _q(t.on_cts_updated, _pat_upd(21, 27, 1.2, 1, 0), df, reversal_confirmed_idx=26) is None
    assert t._scenario1[1] is True and (1, 0) not in t._fibs
    assert t._cross_cycle_data[1]["cycle0"]["fill_horizon_idx"] == 23
