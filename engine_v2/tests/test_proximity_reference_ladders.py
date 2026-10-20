"""PLAN_R step 1 — the proximity reference ladders and the narrow proximity-reference cap (`zones/zone_proximity.py`,
`CycleProximityScan`), on parent zones; the proximity overlay's entry point (`set_overlay`) where a rule only shows with
a sub zone on a side (the lock, the ties, the cap's narrowing).

Synthetic bullish cycle (sid 0, cycle 0, sd = +1) unless stated: BOS inner 1.4000 / outer 1.3990, POI inner 1.4100 /
outer 1.4090, CTS inner 1.4300 / outer 1.4310; thresholds BOS 1.3990, CTS 1.4310 (320 pips — Rules 2 / 3 off); every
default candle at 1.4200 (no zone within 8 pips). Proximity 8 pips: an sd hit on the POI needs a low <= 1.4108, on the
BOS <= 1.4008; an opp_sd hit on the CTS a high >= 1.4292. The scan starts at the CTS_CONFIRMED candle 2.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
import pytest

from engine_v2.structure.market_structure import StructureEvent
from engine_v2.tests import _plan_r_reference_window as W
from engine_v2.tests.test_zone_proximity import (
    _bos_confirmed,
    _bos_zone,
    _cts_zone,
    _make_df,
    _poi_zone,
)
from engine_v2.zones.zone_proximity import (
    DEFAULT_MIN_PROXIMITY_REFERENCE_GAP_PIPS,
    DEFAULT_PROXIMITY_PIPS,
    CycleProximityScan,
    ProximityRung,
    build_cycle_proximity_scans,
    check_zone_proximity,
    proximity_reference_gap_pips,
)

PIP = 0.0001
POI_KEY = ("parent", "POI", 0)
POI_B_KEY = ("parent", "POI", 1)
BOS_KEY = ("parent", "BOS")
CTS_KEY = ("parent", "CTS")


# ---------- helpers ----------

def _df(n: int = 30, mid: float = 1.4200) -> pd.DataFrame:
    """Neutral candles at `mid` (o = h = l = c). A narrow test puts `mid` BETWEEN its two 8-pip bands."""
    return _make_df(n, default_h=mid, default_l=mid)


def _set(df: pd.DataFrame, i: int, **ohlc: float) -> None:
    for k, v in ohlc.items():
        df.at[i, k] = v


def _cts_confirmed_at(idx: int, price: float, *, sid: int = 0, cycle_id: int = 0, sd: int = 1,
                      method: str = "pullback") -> StructureEvent:
    return StructureEvent(
        idx=idx, category="STRUCTURE", type="CTS_CONFIRMED", price=price,
        meta={"structure_id": sid, "cycle_id": cycle_id, "struct_direction": sd, "confirmed_at": idx,
              "confirmation_method": method},
    )


def _cycle(*, bos=(1.4000, 1.3990), cts=(1.4300, 1.4310), pois=((1.4100, 1.4090),),
           thresholds=(1.3990, 1.4310), method="pullback"):
    """(events, kl_zones, poi_zones) of the synthetic bullish cycle; `pois` = (inner, outer) pairs."""
    kl = [_bos_zone(0, 0, 1, *bos), _cts_zone(0, 0, 1, *cts)]
    pz = [_poi_zone(0, 0, 1, inner, outer, cts_established_idx=0) for inner, outer in pois]
    events = [_bos_confirmed(idx=0, price=thresholds[0]), _cts_confirmed_at(2, thresholds[1], method=method)]
    return events, kl, pz


def _scan(df: pd.DataFrame, events, kl, pz, **kw) -> CycleProximityScan:
    return build_cycle_proximity_scans(df, events, kl, pz, PIP, "H1", **kw)[(0, 0)]


def _drive(scan: CycleProximityScan, actions: Optional[Dict[int, Callable]] = None):
    """step(i) / close(i) over the window; `actions[i](scan)` runs before step(i) (where the sweep refreshes the
    overlay). Returns ([(idx, direction, zone_kind, inner)], {idx: [passed keys]})."""
    passes: Dict[int, list] = {}
    for i in range(scan.scan_start, scan.scan_end + 1):
        if actions and i in actions:
            actions[i](scan)
        scan.step(i)
        got = scan.close(i)
        if got:
            passes[i] = [r.key for r in got]
    return [(t.idx, t.direction, t.zone_kind, t.trigger_inner) for t in scan.triggers], passes


def _sub_rung(side: str, kind: str, inner: float, outer: float, *, pool_idx: int = 7, cycle: int = 0) -> ProximityRung:
    return ProximityRung(
        key=("sub", pool_idx, cycle, kind), side=side, zone_kind=kind, inner=inner,
        outer_at=lambda _i: outer, source="sub", tie_rank=1 if kind == "BOS" else 2,
        sub_pool_idx=pool_idx, sub_cycle_id=cycle,
    )


def _overlay(side: str, *rungs: ProximityRung) -> Callable:
    return lambda s: s.set_overlay(side, list(rungs))


# ---------- the constant ----------

def test_the_cap_constants_and_their_assert():
    assert DEFAULT_MIN_PROXIMITY_REFERENCE_GAP_PIPS == {"H1": 40, "M15": 25, "M5": 15}
    for tf, v in DEFAULT_MIN_PROXIMITY_REFERENCE_GAP_PIPS.items():
        assert v > DEFAULT_PROXIMITY_PIPS[tf]


# ---------- 1. passes ----------

def test_1_a_pass_moves_the_sd_reference_from_the_poi_to_the_bos():
    df = _df()
    _set(df, 5, l=1.4105)                 # sd on the POI
    _set(df, 7, l=1.4080, c=1.4085)       # close below the POI outer 1.4090 → passed from 8 (opp_sd expected here)
    _set(df, 10, h=1.4295)                # opp_sd on the CTS
    _set(df, 12, l=1.4105)                # near the passed POI: no trigger, the reference is the BOS
    _set(df, 15, l=1.4005)                # sd on the BOS
    scan = _scan(df, *_cycle())
    trig, passes = _drive(scan)
    assert trig == [(5, "sd", "POI", 1.41), (10, "opp_sd", "CTS", 1.43), (15, "sd", "BOS", 1.40)]
    assert passes == {7: [POI_KEY]}
    assert [t.meta for t in scan.triggers] == [
        {"prox_ref_gap_pips": 200, "narrow_prox_ref": False, "prox_ref_source": "parent"},
        {"prox_ref_gap_pips": 300, "narrow_prox_ref": False, "prox_ref_source": "parent"},
        {"prox_ref_gap_pips": 300, "narrow_prox_ref": False, "prox_ref_source": "parent"},
    ]


def test_1b_the_trigger_candles_own_close_passes_from_the_next_candle():
    """§1 "Effective: from the next candle; the trigger candle's own close counts": candle 5 dives to 1.4005 and
    closes 1.4085 — it fires on the POI (the reference at its start); the BOS is the reference only from 6."""
    df = _df()
    _set(df, 5, l=1.4005, c=1.4085)
    _set(df, 10, h=1.4295)
    _set(df, 12, l=1.4105)
    _set(df, 15, l=1.4005)
    scan = _scan(df, *_cycle())
    trig, passes = _drive(scan)
    assert trig == [(5, "sd", "POI", 1.41), (10, "opp_sd", "CTS", 1.43), (15, "sd", "BOS", 1.40)]
    assert passes == {5: [POI_KEY]}


# ---------- 2. a close AT the outer / a wick beyond it is no pass ----------

@pytest.mark.parametrize("low,close", [(1.4080, 1.4090), (1.4080, 1.4095)])
def test_2_a_close_at_the_outer_or_a_wick_beyond_it_is_not_a_pass(low, close):
    df = _df()
    _set(df, 5, l=1.4105)
    _set(df, 7, l=low, c=close)           # the close AT the outer 1.4090 / inside it, the wick beyond
    _set(df, 10, h=1.4295)
    _set(df, 12, l=1.4105)                # the POI is still the reference → sd fires
    scan = _scan(df, *_cycle())
    trig, passes = _drive(scan)
    assert passes == {}
    assert trig == [(5, "sd", "POI", 1.41), (10, "opp_sd", "CTS", 1.43), (12, "sd", "POI", 1.41)]


# ---------- 3. the lock ----------

def test_3_after_poi_to_bos_nothing_moves_the_sd_reference_again():
    """POI passed at 7 → the BOS from 8, LOCKED (the previous reference was passed and the next pick is the last
    rung). A qualifying sd overlay rung arriving at 11 (inner 1.4050, nearer than the BOS) never becomes the
    reference: 12's low 1.4055 is within 8 pips of it, but the sd side fires only on the BOS (15). Control: the same
    rung present BEFORE the pass (from 6) is the next pick at 8 → no lock, 12 fires on it."""
    df = _df()
    _set(df, 5, l=1.4105)
    _set(df, 7, l=1.4085, c=1.4085)
    _set(df, 10, h=1.4295)
    _set(df, 12, l=1.4055)
    _set(df, 15, l=1.4005)
    rung = _sub_rung("sd", "POI", 1.4050, 1.4040)
    scan = _scan(df, *_cycle())
    trig, _ = _drive(scan, {11: _overlay("sd", rung)})
    assert scan.locked == {"sd": True, "opp_sd": False}
    assert trig == [(5, "sd", "POI", 1.41), (10, "opp_sd", "CTS", 1.43), (15, "sd", "BOS", 1.40)]

    control = _scan(df, *_cycle())
    trig, _ = _drive(control, {6: _overlay("sd", rung)})
    assert control.locked == {"sd": False, "opp_sd": False}
    assert trig == [(5, "sd", "POI", 1.41), (10, "opp_sd", "CTS", 1.43), (12, "sd", "POI", 1.405)]


def test_3b_being_the_reference_by_default_never_locks():
    """The CTS is the opp_sd reference from the scan start (no overlay yet) — by default, not by a pass. A counter-sub
    zone added at 8 (inner 1.4250) becomes the reference: 10's high 1.4245 fires on it (the CTS would need 1.4292).
    Lock-by-default would keep the CTS (PLAN_R §1: it would kill the overlay)."""
    df = _df()
    _set(df, 5, l=1.4105)
    _set(df, 10, h=1.4245)
    scan = _scan(df, *_cycle())
    trig, _ = _drive(scan, {8: _overlay("opp_sd", _sub_rung("opp_sd", "BOS", 1.4250, 1.4260))})
    assert trig == [(5, "sd", "POI", 1.41), (10, "opp_sd", "BOS", 1.425)]
    assert scan.locked == {"sd": False, "opp_sd": False}
    assert scan.triggers[1].meta["prox_ref_source"] == "sub"


# ---------- 4. a POI at / beyond its BOS ----------

def test_4_a_poi_at_or_beyond_its_bos_is_never_used():
    df = _df()
    _set(df, 5, l=1.4005)
    scan = _scan(df, *_cycle(pois=((1.3980, 1.3970), (1.4000, 1.3995))))
    trig, _ = _drive(scan)
    assert [r.key for r in scan.ladder("sd")] == [BOS_KEY]
    assert trig[0] == (5, "sd", "BOS", 1.40)


# ---------- 5. the narrow proximity-reference cap ----------

def test_5a_the_counters_start_at_the_narrowing_earlier_triggers_do_not_count():
    """Wide (POI 1.41 vs CTS 1.43, 200) through 11: 5 sd, 10 opp_sd. A counter-sub zone at 12 (inner 1.4130) makes the
    gap 30 → narrow, counters 0 / 0: 15 sd and 18 opp_sd fire, 20's sd is capped. Candles at 1.4115 (between the
    bands <= 1.4108 and >= 1.4122)."""
    df = _df(mid=1.4115)
    _set(df, 5, l=1.4105)
    _set(df, 10, h=1.4295)
    _set(df, 15, l=1.4105)
    _set(df, 18, h=1.4125)
    _set(df, 20, l=1.4105)
    scan = _scan(df, *_cycle())
    trig, _ = _drive(scan, {12: _overlay("opp_sd", _sub_rung("opp_sd", "BOS", 1.4130, 1.4140))})
    assert [t[0] for t in trig] == [5, 10, 15, 18]
    assert scan.counters == {"sd": 1, "opp_sd": 1}
    assert [t.meta["narrow_prox_ref"] for t in scan.triggers] == [False, False, True, True]
    assert scan.triggers[2].meta["prox_ref_gap_pips"] == 30


def test_5b_an_outward_move_while_narrow_resets_only_that_side():
    """Narrow from the start: POI_a 1.4110 (outer 1.4105), POI_b 1.4100 (outer 1.4095), CTS 1.4140 → gap 30. 5 sd on
    POI_a, 8 opp_sd on the CTS (counters 1 / 1). 10's close 1.4102 passes POI_a (on a candle the cap blocks); from 11
    sd = POI_b — OUTWARD, gap 40 (still narrow, `<=`) → the sd counter resets: 12 fires on POI_b. The opp_sd counter
    stays 1: 14 is capped. Candles at 1.4125 (between <= 1.4118 and >= 1.4132)."""
    df = _df(mid=1.4125)
    _set(df, 5, l=1.4115)
    _set(df, 8, h=1.4135)
    _set(df, 10, l=1.4100, c=1.4102)
    _set(df, 12, l=1.4105)
    _set(df, 14, h=1.4135)
    scan = _scan(df, *_cycle(pois=((1.4110, 1.4105), (1.4100, 1.4095)), cts=(1.4140, 1.4150)))
    trig, passes = _drive(scan)
    assert trig == [(5, "sd", "POI", 1.411), (8, "opp_sd", "CTS", 1.414), (12, "sd", "POI", 1.41)]
    assert passes == {10: [POI_KEY]}
    assert scan.counters == {"sd": 1, "opp_sd": 1}
    assert scan.triggers[2].meta == {"prox_ref_gap_pips": 40, "narrow_prox_ref": True, "prox_ref_source": "parent"}


def test_5c_an_inward_move_while_narrow_does_not_reset():
    """Narrow from the start (POI 1.41, CTS 1.4130: 30). 5 sd, 8 opp_sd (counters 1 / 1). A nearer sd rung at 9 (inner
    1.4110) moves the sd reference INWARD: no reset — 10's low 1.4115 (within 8 pips of it) is capped. Candles at
    1.4120 (outside every band)."""
    df = _df(mid=1.4120)
    _set(df, 5, l=1.4105)
    _set(df, 8, h=1.4125)
    _set(df, 10, l=1.4115)
    scan = _scan(df, *_cycle(cts=(1.4130, 1.4140)))
    trig, _ = _drive(scan, {9: _overlay("sd", _sub_rung("sd", "POI", 1.4110, 1.4105))})
    assert [t[0] for t in trig] == [5, 8]
    assert scan.refs["sd"].inner == 1.4110 and scan.counters == {"sd": 1, "opp_sd": 1}


def test_5d_narrow_wide_narrow_restarts_the_counters():
    df = _df(mid=1.4115)
    _set(df, 5, l=1.4105)
    _set(df, 8, h=1.4125)
    _set(df, 10, l=1.4105)                # wide again (9): no cap
    _set(df, 13, h=1.4130)                # narrow again (11): fresh counters
    _set(df, 15, l=1.4105)
    _set(df, 17, h=1.4130)                # capped
    scan = _scan(df, *_cycle())
    trig, _ = _drive(scan, {
        3: _overlay("opp_sd", _sub_rung("opp_sd", "BOS", 1.4130, 1.4140)),     # gap 30: narrow
        9: _overlay("opp_sd"),                                                # back to the CTS: 200, wide
        11: _overlay("opp_sd", _sub_rung("opp_sd", "BOS", 1.4135, 1.4145, cycle=1)),   # 35: narrow
    })
    assert [t[0] for t in trig] == [5, 8, 10, 13, 15]
    assert [t.meta["narrow_prox_ref"] for t in scan.triggers] == [True, True, False, True, True]


# ---------- 6 / 7. the signed, half-up gap ----------

def test_6_a_crossed_gap_is_narrow():
    """An opp_sd rung 50 pips BELOW the sd reference (crossed): the signed gap is −50 → narrow (its absolute value, 50,
    would be wide)."""
    df = _df()
    scan = _scan(df, *_cycle())
    scan.set_overlay("opp_sd", [_sub_rung("opp_sd", "BOS", 1.4050, 1.4060)])
    scan.step(2)
    assert scan.ref_gap_pips == -50 and scan.narrow_prox_ref is True
    assert scan.counters == {"sd": 0, "opp_sd": 0}


def test_7_rounding_is_half_up_and_narrow_is_at_or_below():
    assert proximity_reference_gap_pips(1, 1.41, 1.41405, PIP) == 41
    assert proximity_reference_gap_pips(1, 1.41, 1.41404, PIP) == 40
    assert proximity_reference_gap_pips(-1, 1.41405, 1.41, PIP) == 41
    assert proximity_reference_gap_pips(1, 1.41, 1.40995, PIP) == -0       # −0.5 rounds half-up to 0
    # A half pip that float arithmetic lands just BELOW .5 — what the 1e-9 is for: (0.57405 − 0.57) / 0.0001 =
    # 40.49999999999998 → 41 (wide). Banker's `round` or a floor without the epsilon would give 40 (narrow).
    assert (0.57405 - 0.57) / PIP < 40.5
    assert proximity_reference_gap_pips(1, 0.57, 0.57405, PIP) == 41
    assert proximity_reference_gap_pips(-1, 0.57405, 0.57, PIP) == 41
    assert proximity_reference_gap_pips(1, 0.57001, 0.57026, PIP) == 3       # 2.4999999999997247
    for cts_inner, gap, narrow in ((1.41405, 41, False), (1.41404, 40, True), (1.4140, 40, True)):
        scan = _scan(_df(), *_cycle(cts=(cts_inner, cts_inner + 0.001)))
        scan.step(2)
        assert (scan.ref_gap_pips, scan.narrow_prox_ref) == (gap, narrow), cts_inner


# ---------- 8. both rule sets AND-ed ----------

def test_8_threshold_narrow_with_a_wide_reference_gap_rule_3_still_caps():
    """Thresholds 30 pips apart (Rule 3 on, a pullback CTS) while the references are 200 apart: 15's sd is capped by
    Rule 3 alone."""
    df = _df()
    _set(df, 5, l=1.4105)
    _set(df, 10, h=1.4295)
    _set(df, 15, l=1.4105)
    scan = _scan(df, *_cycle(thresholds=(1.4100, 1.4130)))
    trig, _ = _drive(scan)
    assert [t[0] for t in trig] == [5, 10]
    assert not any(t.meta["narrow_prox_ref"] for t in scan.triggers)


# ---------- 9. ties ----------

def test_9_tie_order_parent_first_then_a_subs_bos_before_its_poi():
    scan = _scan(_df(), *_cycle())
    scan.set_overlay("opp_sd", [_sub_rung("opp_sd", "POI", 1.4300, 1.4310)])      # == the CTS inner
    scan.step(2)
    assert scan.refs["opp_sd"].key == CTS_KEY
    scan.close(2)
    scan.set_overlay("opp_sd", [_sub_rung("opp_sd", "POI", 1.4250, 1.4260), _sub_rung("opp_sd", "BOS", 1.4250, 1.4255)])
    scan.step(3)
    assert scan.refs["opp_sd"].key == ("sub", 7, 0, "BOS")


# ---------- 10. the real collaborator: S_Q's parent ladders ----------

def _reference_window_inputs():
    events, kl, pz = [], [], []
    for (sid, cyc), c in W.CYCLES.items():
        sd = c["sd"]
        kl += [_bos_zone(sid, cyc, sd, c["bos"]["inner"], c["bos"]["outer"]),
               _cts_zone(sid, cyc, sd, c["cts"]["inner"], c["cts"]["outer"])]
        p = c["poi"]
        pz.append(_poi_zone(sid, cyc, sd, p["top"], p["bottom"], cts_established_idx=p["cts_established_idx"],
                            ic_idx=p["ic_idx"], pattern_confirmed_idx=p["pattern_confirmed_idx"], end_idx=p["end_idx"]))
        bi, bp = c["bos_confirmed"]
        events.append(_bos_confirmed(idx=bi, price=bp, sid=sid, cycle_id=cyc, sd=sd))
        ci, cp, method = c["cts_confirmed"]
        events.append(_cts_confirmed_at(ci, cp, sid=sid, cycle_id=cyc, sd=sd, method=method))
        if c["reversal"] is not None:
            events.append(StructureEvent(idx=c["reversal"], category="STATE", type="STATE_CHANGED", price=None,
                                         meta={"structure_id": sid, "to": "reversal"}))
    return W.h1_frame(), events, kl, pz


def test_10_the_reference_window_parent_ladders_reproduce_s_q_and_the_passes_stay_invisible():
    """S_Q's (0,1) / (1,2) parent zones on the real H1 rows: today's triggers (748; 935 / 954 / 1035), the POI passed
    at 754 (close 0.57657 < outer 0.57676) and 1047 (close 0.57852 > outer 0.5785). The passes change no trigger
    because of ALTERNATION, not price (PLAN_R §4): after each cycle's last sd the opp_sd never fires — (0,1)'s highest
    high 749–901 is 0.58105, short of the 0.58324 the CTS needs; (1,2)'s lowest low 1036–1057 is 0.57598, short of
    0.57362 — so the sd reference is never consulted again, though 106 lows in 755–901 come within 8 pips of (0,1)'s
    passed POI."""
    df, events, kl, pz = _reference_window_inputs()
    scans = build_cycle_proximity_scans(df, events, kl, pz, PIP, "H1")
    assert sorted(scans) == [(0, 1), (1, 2)]
    assert (scans[(0, 1)].scan_start, scans[(0, 1)].scan_end) == (683, 901)
    assert (scans[(1, 2)].scan_start, scans[(1, 2)].scan_end) == (935, 1057)
    t01, p01 = _drive(scans[(0, 1)])
    t12, p12 = _drive(scans[(1, 2)])
    assert t01 == [(748, "sd", "POI", 0.57712)]
    assert p01 == {754: [POI_KEY]}
    assert t12 == [(935, "sd", "POI", 0.57786), (954, "opp_sd", "CTS", 0.57282), (1035, "sd", "POI", 0.57786)]
    assert p12 == {1047: [POI_KEY]}
    for key in ((0, 1), (1, 2)):
        assert scans[key].refs["sd"].key == BOS_KEY and scans[key].locked["sd"]
        assert scans[key].expected_dir == "opp_sd"
        assert not scans[key].narrow_prox_ref
    assert df.loc[749:901, "h"].max() == 0.58105 < round(0.58404 - 8 * PIP, 5)
    assert df.loc[1036:1057, "l"].min() == 0.57598 > round(0.57282 + 8 * PIP, 5)
    assert int((df.loc[755:901, "l"] <= 0.57712 + 8 * PIP).sum()) == 106
    # the batch scan IS these objects driven over their windows
    batch = check_zone_proximity(df, events, kl, pz, PIP, "H1")
    assert list(batch) == [(0, 1), (1, 2)]          # the cycles in CTS_CONFIRMED order (the H1 marker trace's order)
    assert {k: [(t.idx, t.direction, t.zone_kind, t.trigger_inner) for t in v] for k, v in batch.items()} == {
        (0, 1): t01, (1, 2): t12}


def test_the_embedded_rows_match_the_save():
    """Provenance of `_plan_r_reference_window` (skipped where the save is absent, e.g. a mutant tree)."""
    root = Path(__file__).resolve().parents[2]
    final = root / W.SAVE / "NZD_USD_H1_2025-11-15_2026-01-20_final.csv"
    if not final.exists():
        pytest.skip("S_Q save not present")
    sq = pd.read_csv(final)
    df = W.h1_frame()
    for col in "ohlc":
        assert (sq.loc[W.FIRST_IDX:W.LAST_IDX, col].round(5).values
                == df.loc[W.FIRST_IDX:W.LAST_IDX, col].round(5).values).all(), col


# ---------- 11. both references move on one candle ----------

def test_11_one_side_outward_one_inward_on_one_candle_resets_only_the_outward_side():
    """Narrow: POI_a 1.4110 (outer 1.4105), POI_b 1.4100, the opp_sd rung X 1.4140 (gap 30). 5 sd on POI_a, 8 opp_sd on X
    (1 / 1). Candle 11 sees BOTH moves: POI_a passed at 10 → sd = POI_b (outward), and the opp_sd rung Y 1.4135
    replaces X (inward); gap 35, still narrow. Only sd resets: 12 fires, 14's opp_sd is capped. Judged on the net gap
    (30 → 35, wider) both would reset and 14 would fire. Candles at 1.4125."""
    df = _df(mid=1.4125)
    _set(df, 5, l=1.4115)
    _set(df, 8, h=1.4135)
    _set(df, 10, l=1.4100, c=1.4102)
    _set(df, 12, l=1.4105)
    _set(df, 14, h=1.4130)
    scan = _scan(df, *_cycle(pois=((1.4110, 1.4105), (1.4100, 1.4095))))
    trig, passes = _drive(scan, {
        3: _overlay("opp_sd", _sub_rung("opp_sd", "BOS", 1.4140, 1.4150)),
        11: _overlay("opp_sd", _sub_rung("opp_sd", "BOS", 1.4135, 1.4145, cycle=1)),
    })
    assert passes == {10: [POI_KEY]}
    assert trig == [(5, "sd", "POI", 1.411), (8, "opp_sd", "BOS", 1.414), (12, "sd", "POI", 1.41)]
    assert scan.counters == {"sd": 1, "opp_sd": 1}
    assert scan.triggers[2].meta["prox_ref_gap_pips"] == 35


# ---------- 12. the last rung is never passed ----------

def test_12_a_close_beyond_the_bos_or_cts_outer_leaves_it_the_reference():
    df = _df()
    _set(df, 5, l=1.4005)
    _set(df, 7, l=1.3980, c=1.3985)       # beyond the BOS outer 1.3990
    _set(df, 10, h=1.4295)
    _set(df, 12, h=1.4330, c=1.4320)      # beyond the CTS outer 1.4310
    _set(df, 15, l=1.4005)
    _set(df, 18, h=1.4295)
    scan = _scan(df, *_cycle(pois=()))
    trig, passes = _drive(scan)
    assert passes == {}
    assert trig == [(5, "sd", "BOS", 1.40), (10, "opp_sd", "CTS", 1.43), (15, "sd", "BOS", 1.40),
                    (18, "opp_sd", "CTS", 1.43)]
    assert scan.locked == {"sd": False, "opp_sd": False}


# ---------- 13. a pass on a blocked candle counts ----------

@pytest.mark.parametrize("label,thresholds,method", [
    ("rule 3", (1.4100, 1.4130), "pullback"),
    ("rule 2", (1.4100, 1.4130), "sd_zone_proximity"),
    ("the cap", (1.3990, 1.4310), "pullback"),
])
def test_13_a_pass_on_a_blocked_candle_still_counts(label, thresholds, method):
    """POI_a 1.4110 (outer 1.4105), POI_b 1.4100. The candle 10 closes 1.4102 (beyond POI_a's outer) while a gate blocks
    the sd test — Rule 3 (5 sd and 8 opp_sd fired while the threshold gap is narrow), Rule 2 (narrow, not a pullback
    CTS: nothing fires) or the cap (an opp_sd rung at 1.4140: narrow 30, counters 1 / 1). POI_a is passed all the same:
    from 11 the sd reference is POI_b. Candles at 1.4125."""
    df = _df(mid=1.4125)
    _set(df, 5, l=1.4115)
    _set(df, 8, h=1.4295 if label != "the cap" else 1.4135)
    _set(df, 10, l=1.4100, c=1.4102)
    scan = _scan(df, *_cycle(pois=((1.4110, 1.4105), (1.4100, 1.4095)), thresholds=thresholds, method=method))
    actions = {3: _overlay("opp_sd", _sub_rung("opp_sd", "BOS", 1.4140, 1.4150))} if label == "the cap" else None
    for i in range(2, 12):
        if actions and i in actions:
            actions[i](scan)
        t = scan.step(i)
        if i == 10:
            assert t is None and scan.gate is not None, label
        scan.close(i)
    assert POI_KEY in scan.passed
    assert scan.refs["sd"].key == POI_B_KEY


# ---------- 14. narrow from the scan start ----------

def test_14_a_cycle_narrow_from_the_scan_start_starts_its_counters_at_0():
    """(1,0)-like: POI 1.41 vs CTS 1.4130, 30 pips at the scan start → counters 0 / 0 at once: one sd, one opp_sd, then
    capped. Candles at 1.4115."""
    df = _df(mid=1.4115)
    _set(df, 5, l=1.4105)
    _set(df, 8, h=1.4125)
    _set(df, 10, l=1.4105)
    scan = _scan(df, *_cycle(cts=(1.4130, 1.4140)))
    scan.step(2)
    assert scan.narrow_prox_ref and scan.counters == {"sd": 0, "opp_sd": 0}
    scan.close(2)
    for i in range(3, scan.scan_end + 1):
        scan.step(i)
        scan.close(i)
    assert [t.idx for t in scan.triggers] == [5, 8]


# ---------- the per-candle protocol ----------

def test_step_then_close_in_order_once_per_candle():
    scan = _scan(_df(), *_cycle())
    with pytest.raises(AssertionError):
        scan.close(2)                     # close before step
    scan.step(2)
    with pytest.raises(AssertionError):
        scan.step(3)                      # step before the previous close
    scan.close(2)
    with pytest.raises(AssertionError):
        scan.close(2)                     # twice
    with pytest.raises(AssertionError):
        scan.step(2)                      # backwards


# ---------- the outer as of the close ----------

def test_a_kl_rungs_outer_is_its_init_outer_then_each_expansion_from_its_moment():
    """PLAN_R §1 "Which outer": outers expand (an H1 CTS zone), inners never move. The INIT step's index is the base
    LOCATION (not read); an expansion counts from its moment `expanded_idx` (inclusive: as of that candle's close).
    A fixture zone without `bounds_steps` keeps `meta["outer"]`."""
    from engine_v2.zones.zone_proximity import _kl_outer_at
    cts = _cts_zone(0, 0, 1, inner=1.4300, outer=1.4310)
    cts.meta["bounds_steps"] = [
        {"event": "INIT", "base_idx": 40, "top": 1.4310, "bottom": 1.4300},
        {"event": "CTS_THRESHOLD_UPDATED", "expanded_idx": 13, "top": 1.4330, "bottom": 1.4300},
    ]
    f = _kl_outer_at(cts)                          # a sell zone: the outer is its top
    assert (f(2), f(12), f(13), f(30)) == (1.4310, 1.4310, 1.4330, 1.4330)
    bos = _bos_zone(0, 0, 1, inner=1.4000, outer=1.3990)
    bos.meta["bounds_steps"] = [{"event": "INIT", "base_idx": 0, "top": 1.4000, "bottom": 1.3990}]
    assert _kl_outer_at(bos)(99) == 1.3990         # a buy zone: its bottom
    assert _kl_outer_at(_bos_zone(0, 0, 1, inner=1.4000, outer=1.3985))(5) == 1.3985


# ---------- the landing review's test gaps (2026-10-20): paths step 2 drives ----------
# Each pins a mutant that survived the full suite at d26208e (the opp_sd side's and an overlay rung's pass test, an
# opp_sd outward reset while narrow, the sd = −1 cap, the outer read at the pass site, the gap-None path) or was
# killed only incidentally ("Rule 2 is not part of the cap"). The reviewer's sketches, kept as written.

def _bear_cycle(pois, cts=(1.4160, 1.4150), bos=(1.4300, 1.4310), thresholds=(1.4310, 1.4150), method="pullback"):
    kl = [_bos_zone(0, 0, -1, *bos), _cts_zone(0, 0, -1, *cts)]
    # sd=-1: a POI's inner is its BOTTOM, its outer its TOP
    pz = [_poi_zone(0, 0, -1, outer, inner, cts_established_idx=0) for inner, outer in pois]
    events = [_bos_confirmed(idx=0, price=thresholds[0], sd=-1), _cts_confirmed_at(2, thresholds[1], sd=-1, method=method)]
    return events, kl, pz


def test_review_1_bearish_outward_move_while_narrow_resets_that_side():
    """Mirror of 5b on sd=-1: POI_a 1.4190 (outer 1.4195), POI_b 1.4200, CTS 1.4160 -> gap 30. 5 sd, 8 opp_sd; 10's
    close 1.4198 passes POI_a on a capped candle; from 11 sd = POI_b (UP = outward for a bearish cycle), gap 40 ->
    the sd counter resets: 12 fires on POI_b; 14's opp_sd stays capped."""
    df = _df(mid=1.4175)
    _set(df, 5, h=1.4185)
    _set(df, 8, l=1.4165)
    _set(df, 10, h=1.4200, c=1.4198)
    _set(df, 12, h=1.4195)
    _set(df, 14, l=1.4165)
    scan = _scan(df, *_bear_cycle(pois=((1.4190, 1.4195), (1.4200, 1.4205))))
    trig, passes = _drive(scan)
    assert passes == {10: [("parent", "POI", 0)]}
    assert trig == [(5, "sd", "POI", 1.419), (8, "opp_sd", "CTS", 1.416), (12, "sd", "POI", 1.42)]
    assert scan.triggers[2].meta["prox_ref_gap_pips"] == 40


def test_review_2_an_opp_sd_pass_moves_the_reference_to_the_cts_and_locks():
    """An opp_sd overlay rung X 1.4250 (outer 1.4260) is passed by 3's close 1.4265 (sd expected): from 4 the opp_sd
    reference is the CTS, locked. 8's high 1.4245 (within 8 pips of X) does not fire; 10 fires on the CTS."""
    df = _df()
    _set(df, 3, h=1.4270, c=1.4265)
    _set(df, 5, l=1.4105)
    _set(df, 8, h=1.4245)
    _set(df, 10, h=1.4295)
    x = _sub_rung("opp_sd", "BOS", 1.4250, 1.4260)
    scan = _scan(df, *_cycle())
    trig, passes = _drive(scan, {2: _overlay("opp_sd", x)})
    assert passes == {3: [x.key]}
    assert trig == [(5, "sd", "POI", 1.41), (10, "opp_sd", "CTS", 1.43)]
    assert scan.locked == {"sd": False, "opp_sd": True}


def test_review_3_an_opp_sd_outward_move_while_narrow_resets_the_opp_sd_counter():
    """Wide to 5 (sd fires); X 1.4130 / Y 1.4135 opp_sd rungs from 6 -> narrow 30, counters 0/0. 8 opp_sd on X, 10 sd
    (1/1). 11 (opp_sd expected, capped) closes 1.4142 > X's outer 1.4140: from 12 opp_sd = Y, OUTWARD (gap 35) ->
    the opp_sd counter resets: 13 fires on Y."""
    df = _df(mid=1.4115)
    _set(df, 5, l=1.4105)
    _set(df, 8, h=1.4125)
    _set(df, 10, l=1.4105)
    _set(df, 11, h=1.4150, c=1.4142)
    _set(df, 13, h=1.4130)
    x = _sub_rung("opp_sd", "BOS", 1.4130, 1.4140)
    y = _sub_rung("opp_sd", "POI", 1.4135, 1.4145)
    scan = _scan(df, *_cycle())
    trig, passes = _drive(scan, {6: _overlay("opp_sd", x, y)})
    assert passes == {11: [x.key]}
    assert [t[:3] for t in trig] == [(5, "sd", "POI"), (8, "opp_sd", "BOS"), (10, "sd", "POI"), (13, "opp_sd", "POI")]


def test_review_4_the_pass_reads_the_outer_as_of_the_close_including_an_expansion_on_that_candle():
    """An sd overlay rung whose outer expands 1.4090 -> 1.4080 at moment 7: 7's close 1.4085 is beyond the OLD outer
    but not beyond the outer as of its close -> no pass; 12 still fires on the rung."""
    df = _df()
    _set(df, 5, l=1.4105)
    _set(df, 7, l=1.4082, c=1.4085)
    _set(df, 10, h=1.4295)
    _set(df, 12, l=1.4105)
    r = ProximityRung(key=("sub", 7, 0, "BOS"), side="sd", zone_kind="BOS", inner=1.4100,
                      outer_at=lambda i: 1.4090 if i < 7 else 1.4080, source="sub", tie_rank=1)
    scan = _scan(df, *_cycle(pois=()))
    trig, passes = _drive(scan, {2: _overlay("sd", r)})
    assert passes == {}
    assert trig == [(5, "sd", "BOS", 1.41), (10, "opp_sd", "CTS", 1.43), (12, "sd", "BOS", 1.41)]


def test_review_5_a_cycle_without_a_cts_zone_is_wide_until_an_opp_sd_reference_exists():
    """No CTS zone: the gap is None -> wide (no counters). 5 sd on the BOS (meta None / False). An opp_sd overlay rung
    at 8 makes the gap 30 -> narrow with FRESH counters: 10 opp_sd, 12 sd both fire."""
    df = _df(mid=1.4115)
    _set(df, 5, l=1.4105)
    _set(df, 10, h=1.4125)
    _set(df, 12, l=1.4105)
    events = [_bos_confirmed(idx=0, price=1.3990), _cts_confirmed_at(2, 1.4310)]
    kl = [_bos_zone(0, 0, 1, 1.4100, 1.4090)]
    scan = build_cycle_proximity_scans(df, events, kl, [], PIP, "H1")[(0, 0)]
    trig, _ = _drive(scan, {8: _overlay("opp_sd", _sub_rung("opp_sd", "BOS", 1.4130, 1.4140))})
    assert scan.triggers[0].meta == {"prox_ref_gap_pips": None, "narrow_prox_ref": False, "prox_ref_source": "parent"}
    assert [t[0] for t in trig] == [5, 10, 12]


def test_review_6_rule_2_is_not_part_of_the_cap():
    """Thresholds wide (320) but the CTS was confirmed by sd-proximity; the references are 30 apart (narrow by the cap):
    Rule 2 does not apply -> one sd + one opp_sd fire (5, 8), then the cap."""
    df = _df(mid=1.4115)
    _set(df, 5, l=1.4105)
    _set(df, 8, h=1.4125)
    _set(df, 10, l=1.4105)
    scan = _scan(df, *_cycle(cts=(1.4130, 1.4140), method="sd_zone_proximity"))
    trig, _ = _drive(scan)
    assert [t[0] for t in trig] == [5, 8]


def test_review_7_an_overlay_rung_is_pass_tested():
    """Test 1 with the POI as an sd overlay rung: 7's close passes it, from 8 the sd reference is the BOS (locked)."""
    df = _df()
    _set(df, 5, l=1.4105)
    _set(df, 7, l=1.4080, c=1.4085)
    _set(df, 10, h=1.4295)
    _set(df, 12, l=1.4105)
    _set(df, 15, l=1.4005)
    r = _sub_rung("sd", "POI", 1.4100, 1.4090)
    scan = _scan(df, *_cycle(pois=()))
    trig, passes = _drive(scan, {2: _overlay("sd", r)})
    assert passes == {7: [r.key]}
    assert trig == [(5, "sd", "POI", 1.41), (10, "opp_sd", "CTS", 1.43), (15, "sd", "BOS", 1.40)]
    assert scan.locked["sd"]
