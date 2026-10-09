"""Plan E E3a mutation-review pins: every FibTracker TIME half is the handled event's MOMENT, never the CTS
anchor. (Re-derived for the ONE FibTracker logic in PLAN_I S4 — the H1-main Scenario 1 / 2 / 3, revert, §11b-peek and
cycle-0-cache pins retired with their mechanisms; the MS in-flight ones stay until S5.) Every event is LAGGING (moment = anchor + 2; legal contract:
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
from engine_v2.tests._cross_fib_limits import NO_P

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


def _ip_ready(df):
    """`_df`'s frame through the real candle classifier (the columns + stamps the
    institutional pattern reads): the MS resolver runs `select_ip_ic` OUTSIDE its
    graceful `try` since PLAN_H landing B — an unready frame raises
    `IPSelectionError`."""
    from engine_v2.features.candle_classifier import apply_candle_classification
    out = apply_candle_classification(df).df
    out.attrs["imbalances"] = df.attrs["imbalances"]
    out.attrs["pair"] = "NZD_USD"
    return out


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


def _t(p=None):
    """`p` = sid 1's reference price (None: `unreached` — every cross the walk allows); sid 0 `unreached` too."""
    from engine_v2.tests._cross_fib_limits import limit, unreached
    from engine_v2.zones.cross_cycle_fib import CrossFibLimit
    lim1 = unreached(1) if p is None else limit(p, "reversal_bos_frozen")
    return FibTracker(FibTrackerConfig(fill_threshold=0.70), cross_fib_limits={0: unreached(1), 1: lim1})


def _q(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _c0(t, df):
    return _q(t.on_cts_established, _est(20, 1.15, 1, 0), df, bos_idx=10, bos_price=1.0)


def _c1(t, df, bos_price=1.2):
    _q(t.on_cts_confirmed, _conf(25, 20, 1.15, 1, 0))
    return _q(t.on_cts_established, _est(40, 1.35, 1, 1), df, bos_idx=30, bos_price=bos_price)


def _c2(t, df):
    _q(t.on_cts_confirmed, _conf(45, 40, 1.35, 1, 1))
    return _q(t.on_cts_established, _est(60, 1.55, 1, 2), df, bos_idx=50, bos_price=1.4)


# --- CTS_ESTABLISHED: every activation stamp and fill horizon -----------------------

def test_est_cycle1_own_window_fill_horizon_is_the_moment():
    """Re-derived (PLAN_I S4; was Scenario 2's cond1): the cycle-1 gap @35 fills at 41 — after the anchor 40, by the
    moment 42 → the own test fails at the moment → no cross and no single."""
    t = _t()
    df = _df(fills=[(41, 1.22)])
    _c0(t, df)
    assert _c1(t, df) is None
    assert t._get_latest_cross(1, 1) is None and (1, 1) not in t._fibs


def test_est_gate_fill_horizon_is_the_moment():
    """sid 0 cycle 1: the only gap (@15) fills at 21 — after the anchor 20,
    by the moment 22 → the own test fails at the moment → no fib."""
    t = _t()
    df = _df(insts=[ImbalanceInstance(15, 15, 1, 1.10, 1.00, 0.10)], fills=[(21, 1.02)])
    ev = make_cts_established(cts_anchor_idx=20, confirmed_at=22, price=1.2, structure_id=0,
                              cycle_id=1, struct_direction=1)
    # Cycle 0 confirmed first: MS establishes cycle 1 only after it (the order tripwire, PLAN_I S2 / F14).
    _q(t.on_cts_confirmed, _conf(6, 5, 0.95, 0, 0))
    assert _q(t.on_cts_established, ev, df, bos_idx=10, bos_price=0.9) is None


def test_est_cycle2_single_and_cross_stamp_the_moment():
    """Re-derived (PLAN_I S4; was the §11b ceiling on P_rev): P 1.30 — cycle 1's CTS 1.35 is beyond → M = 1 →
    cycle 2 past the limit → the single; P 2.0 — never beyond → the cross (0→2). Both stamped at the moment 62; the
    cross also ends cycle 1 there (`_obsolete_prev_cycle_all_fibs`)."""
    t = _t(1.30)
    _c0(t, _df()); _c1(t, _df())
    single = _c2(t, _df())
    assert single.meta.get("cross_cycle") is not True and single.meta["via"] == "cross_fib_limit"
    assert (single.cts_idx, single.meta["activated_at"]) == (60, 62)
    # cycle 1 is a CROSS: a later single never stamps it (the retired h1-only obsolete did) — finalize's pass-through
    # end (`next_cycle`) closes it, as main's (1,1) in PLAN_I §9.5 c.
    assert (1, 1) not in t._terminal
    t2 = _t(2.0)
    _c0(t2, _df()); _c1(t2, _df())
    cross = _c2(t2, _df())
    assert cross.meta.get("cross_cycle") is True and cross.meta["cross_start_cycle"] == 0
    assert (cross.cts_idx, cross.meta["activated_at"]) == (60, 62)
    assert t2._terminal[(1, 1)] == (62, "new_cycle")


def test_est_the_walk_asks_at_the_moment():
    """Re-derived (PLAN_I S4; was the §11b peek): cycle 1's gap (@35, 1.45-1.50) fills at 61 — after cycle 2's
    anchor 60, by its moment 62 (cycle 0's gap died at 30; cycle 2's own gap @55 sits lower and stays unfilled).
    Walked at the moment, cycle 1 is dead → no prior cycle eligible → the `cross_failed` single; at the anchor it
    would have crossed (1→2)."""
    t = _t()
    df = _df(insts=[ImbalanceInstance(15, 15, 1, 1.10, 1.00, 0.10),
                    ImbalanceInstance(35, 35, 1, 1.50, 1.45, 0.05),
                    ImbalanceInstance(55, 55, 1, 1.30, 1.20, 0.10)],
             fills=[(30, 1.02), (61, 1.46)])
    _c0(t, df); _c1(t, df)
    c2 = _c2(t, df)
    assert (c2.bos_idx, c2.cts_idx, c2.meta["via"], c2.meta["activated_at"]) == (50, 60, "cross_failed", 62)
    assert 1 in t._dead_cycles.get(1, set()) and t._get_latest_cross(1, 2) is None


# --- CTS_UPDATED (pattern path, anchor A / moment A + 2) ------------------------

def test_update_cross_cycle_cycle0_first_activation_is_asked_at_the_moment():
    """cross_cycle cycle 0: the only gap (c2 21) lies past the EST anchor 20 → no
    fib at EST. The update (anchor 23, moment 25) covers it, but it fills at 24
    → asked at the moment: nothing activates."""
    t = _t()
    df = _df(insts=[ImbalanceInstance(21, 21, 1, 1.10, 1.00, 0.10)], fills=[(24, 1.02)])
    assert _c0(t, df) is None
    assert _q(t.on_cts_updated, _upd(23, 1.2, 1, 0), df) is None
    assert (1, 0) not in t._fibs


def test_update_cross_cycle_cycle1_own_fill_is_asked_at_the_moment():
    """cross_cycle cycle 1: the own gap (@35) fills at 46, between the update's
    anchor 45 and moment 47 → the cross deactivates, stamped 47."""
    t = _t()
    df = _df(fills=[(46, 1.22)])
    _c0(t, df); _c1(t, df)
    _q(t.on_cts_updated, _upd(45, 1.40, 1, 1), df)
    cross = t._fibs[(1, 1, "cross", 0)]
    assert not cross.active and cross.meta["deactivated_at"] == 47


def test_update_fib_cts_reactivation_stamps_the_moment():
    """sid 0 without a reference price (cycle 1 a single — PLAN_I S4 re-pin): gap @15 fills at 31 → the update
    (30 / 32) deactivates; a new gap (c2 33) → the update (35 / 37) reactivates, stamped 37."""
    from engine_v2.zones.cross_cycle_fib import CrossFibLimit
    t = FibTracker(FibTrackerConfig(fill_threshold=0.70), cross_fib_limits={0: CrossFibLimit(None, "none")})
    df = _df(insts=[ImbalanceInstance(15, 15, 1, 1.10, 1.00, 0.10),
                    ImbalanceInstance(33, 33, 1, 1.40, 1.30, 0.10)], fills=[(31, 0.99)])
    ev = make_cts_established(cts_anchor_idx=20, confirmed_at=22, price=1.2, structure_id=0,
                              cycle_id=1, struct_direction=1)
    # Cycle 0 confirmed first: MS establishes cycle 1 only after it (the order tripwire, PLAN_I S2 / F14).
    _q(t.on_cts_confirmed, _conf(6, 5, 0.95, 0, 0))
    assert _q(t.on_cts_established, ev, df, bos_idx=10, bos_price=0.9).active
    assert not _q(t.on_cts_updated, _upd(30, 1.3, 0, 1), df).active
    re = _q(t.on_cts_updated, _upd(35, 1.45, 0, 1), df)
    assert re.active and re.meta["reactivated_at"] == 37



def test_update_cross_revival_stamps_the_moment():
    """Re-derived (PLAN_I S4 landing review m1; was `_update_cycle1_main`'s): the cycle-1 cross deactivates at the
    update 45 / 47 (its own gap @35 filled at 44); a new gap (c2 48) → the update 50 / 52 revives the SAME version in
    place (`_reactivate_cross_in_place`), stamped at the moment 52 — never the anchor 50."""
    t = _t()
    df = _df(insts=[ImbalanceInstance(15, 15, 1, 1.10, 1.00, 0.10),
                    ImbalanceInstance(35, 35, 1, 1.30, 1.20, 0.10),
                    ImbalanceInstance(48, 48, 1, 1.45, 1.40, 0.05)], fills=[(44, 1.22)])
    _c0(t, df); _c1(t, df)
    _q(t.on_cts_updated, _upd(45, 1.40, 1, 1), df)          # deactivated (filled at 44)
    assert not t._fibs[(1, 1, "cross", 0)].active
    _q(t.on_cts_updated, _upd(50, 1.50, 1, 1), df)          # new gap c2 48 → revived in place
    cross = t._fibs[(1, 1, "cross", 0)]
    assert cross.active and cross.meta["reactivated_at"] == 52 and t._cross_version[(1, 1)] == 0

# --- the MS in-flight mirror ------------------------------------------------------

def _mirror(rows, k=1.2300):
    """Price-mirror an sd=+1 fixture into its sd=-1 twin (h <-> l)."""
    return [{"o": round(k - r["o"], 5), "h": round(k - r["l"], 5),
             "l": round(k - r["h"], 5), "c": round(k - r["c"], 5)} for r in rows]


def test_ms_inflight_poi_refresh_fill_horizon_is_the_moment_bearish(monkeypatch):
    """The sd=-1 twin of the role-pin test, re-derived at the FEED (PLAN_I S5: MS's read is its embedded tracker's
    decision, asked at the fed event's moment): the bearish raw-update branch feeds its processing candle as the
    moment; the pattern tying the raw-updated CTS (extreme 24, apply 25) feeds nothing (2026-09-27: strict new
    extreme on both paths)."""
    import engine_v2.structure.structure_engine as se
    from engine_v2.structure import event_fields as _ef
    from engine_v2.tests.test_imbalance_c3_knowability import _multicycle_with_tied_pattern_breakout
    from engine_v2.tests.test_unified_probe import _prepare_df
    calls = []
    real = FibTracker.on_cts_updated

    def spy(self, ev, *a, **k):
        if self._quiet:                           # MS's embedded tracker
            calls.append((int(_ef.cts_anchor_idx(ev)), int(_ef.event_moment(ev))))
        return real(self, ev, *a, **k)

    monkeypatch.setattr(FibTracker, "on_cts_updated", spy)
    with contextlib.redirect_stdout(io.StringIO()):
        res = se.compute_bounded_structure(
            _prepare_df(_mirror(_multicycle_with_tied_pattern_breakout())), 0, -1, cross_fib_limit=NO_P)
    raw = [e.idx for e in res.events if e.type == "CTS_UPDATED" and e.meta["via"] == "replay_raw"]
    assert raw                                     # the fixture exercises the bearish raw branch
    assert all((r, r) in calls for r in raw)
    assert int(res.df["last_breakout_pat_apply_idx"].iloc[25]) == 25   # precondition: the tying breakout WAS applied at 25
    assert (24, 24) in calls and not [h for _c, h in calls if h == 25]


# PLAN_I S5 retired MS's resolver (`compute_poi_inners_for_cycle` → the Scenario-2 selector) and its cycle-0
# snapshot: `test_compute_poi_inners_uses_its_fill_horizon`, `test_e3ap_ms_cycle0_snapshot_is_asked_at_the_refresh_
# moment`, `test_e3ap_compute_poi_inners_uses_its_snapshot_horizon` and `test_e3ap_ms_refresh_snapshot_horizon_is_the_
# cycle_established_moment` went with them (the feed's moments: the role pins above and in test_plan_e_role_pins).


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


# --- hygiene 5b (2026-09-28): the create-on-fail single is gone; its invariant is asserted --------------
