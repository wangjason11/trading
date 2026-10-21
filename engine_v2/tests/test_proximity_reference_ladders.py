"""PLAN_R steps 1 + 3 — the proximity reference ladders and the proximity reference gate (`zones/zone_proximity.py`,
`CycleProximityScan`; step 3 replaced steps 1–2's narrow proximity-reference cap), on parent zones; the proximity
overlay's entry point (`set_overlay`) where a rule only shows with a sub zone on a side (the lock, the ties, the gate's
updates, a superseded reference).

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

def test_the_gate_minimum_constants_and_their_assert():
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
        {"prox_ref_gap_pips": 200, "narrow_prox_ref": False, "prox_ref_source": "parent", "prox_ref_sub_id": None, "prox_ref_cycle": None, "prox_ref_usable_idx": None},
        {"prox_ref_gap_pips": 300, "narrow_prox_ref": False, "prox_ref_source": "parent", "prox_ref_sub_id": None, "prox_ref_cycle": None, "prox_ref_usable_idx": None},
        {"prox_ref_gap_pips": 300, "narrow_prox_ref": False, "prox_ref_source": "parent", "prox_ref_sub_id": None, "prox_ref_cycle": None, "prox_ref_usable_idx": None},
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


# ---------- 5. the proximity reference gate (PLAN_R §9) ----------

@pytest.mark.parametrize("inner,gap,taken", [(1.4130, 30, False), (1.4140, 40, False), (1.4141, 41, True)])
def test_5a_an_inward_update_that_would_be_too_narrow_is_skipped(inner, gap, taken):
    """POI 1.41 vs CTS 1.43 (200): 5 sd, 10 opp_sd. A counter-sub zone at 12, `gap` pips from the sd reference: at or
    below 40 (`<=`) the update is SKIPPED — the opp_sd reference stays the CTS, 18's high 1.4135 does not fire and 20's
    1.4295 fires on the CTS; at 41 it is taken and 18 fires on it. No counters: alternation continues either way.
    Candles at 1.4115 (outside every band)."""
    df = _df(mid=1.4115)
    _set(df, 5, l=1.4105)
    _set(df, 10, h=1.4295)
    _set(df, 15, l=1.4105)
    _set(df, 18, h=1.4135)
    _set(df, 20, h=1.4295)
    scan = _scan(df, *_cycle())
    rung = _sub_rung("opp_sd", "BOS", inner, inner + 0.0010)
    trig, _ = _drive(scan, {12: _overlay("opp_sd", rung)})
    assert proximity_reference_gap_pips(1, 1.41, inner, PIP) == gap
    assert [r.key for r in scan.ladder("opp_sd")] == [CTS_KEY, rung.key]          # on the ladder, skipped by the gate
    if taken:
        assert trig[2:] == [(15, "sd", "POI", 1.41), (18, "opp_sd", "BOS", inner)]
        assert scan.refs["opp_sd"].key == rung.key and scan.triggers[3].meta["prox_ref_gap_pips"] == 41
    else:
        assert trig[2:] == [(15, "sd", "POI", 1.41), (20, "opp_sd", "CTS", 1.43)]
        assert scan.refs["opp_sd"].key == CTS_KEY
    assert not any(t.meta["narrow_prox_ref"] for t in scan.triggers)


def test_5b_the_closest_candidate_that_passes_the_gate_is_taken():
    """Point 2 (round 15: "Move to the closest candidate that passes the gate"): two opp_sd rungs inside the CTS, X
    1.4130 (30, too narrow) and Y 1.4150 (50) → the reference is Y — not X, and not "stay on the CTS". 9's high 1.4145
    fires on Y."""
    df = _df(mid=1.4115)
    _set(df, 5, l=1.4105)
    _set(df, 9, h=1.4145)
    x = _sub_rung("opp_sd", "BOS", 1.4130, 1.4140)
    y = _sub_rung("opp_sd", "POI", 1.4150, 1.4160)
    scan = _scan(df, *_cycle())
    trig, _ = _drive(scan, {6: _overlay("opp_sd", x, y)})
    assert trig == [(5, "sd", "POI", 1.41), (9, "opp_sd", "POI", 1.415)]
    assert scan.refs["opp_sd"].key == y.key and scan.ref_gap_pips == 50


def test_5c_a_superseded_reference_whose_every_inward_rung_is_too_narrow_stays_never_the_cts():
    """Point 1 (round 15, 1036's case: "Keep sub_6's POI as the reference, or fall back to the parent CTS? Keep it"):
    sub A's rung 1.4150 (50) is the opp_sd reference from 3; at 9 sub B replaces A on the ladder with rungs 1.4130 /
    1.4135 (30 / 35, both too narrow) → A's rung is SUPERSEDED and stays the reference (not the CTS): 12's high 1.4145
    fires on it. It stays pass-testable: 13's close 1.4165 (sd expected) is beyond its outer 1.4160 → passed (B's
    taller rungs, outers 1.4175 / 1.4180, are not); from 14 the side moves to the closest rung passing the gate —
    B's still fail → the CTS, LOCKED (a pass to the last rung)."""
    df = _df(mid=1.4115)
    _set(df, 5, l=1.4105)
    _set(df, 12, h=1.4145)
    _set(df, 13, h=1.4170, l=1.4120, c=1.4165)
    a = _sub_rung("opp_sd", "POI", 1.4150, 1.4160, pool_idx=3)
    b1 = _sub_rung("opp_sd", "BOS", 1.4130, 1.4175, pool_idx=5)
    b2 = _sub_rung("opp_sd", "POI", 1.4135, 1.4180, pool_idx=5)
    scan = _scan(df, *_cycle())
    refs, passes = {}, {}
    acts = {3: _overlay("opp_sd", a), 9: _overlay("opp_sd", b1, b2)}
    for i in range(scan.scan_start, scan.scan_end + 1):
        if i in acts:
            acts[i](scan)
        scan.step(i)
        refs[i] = scan.refs["opp_sd"].key
        got = scan.close(i)
        if got:
            passes[i] = [r.key for r in got]
    assert [r.key for r in scan.ladder("opp_sd")] == [CTS_KEY, b1.key, b2.key]     # A is gone from the ladder
    assert {i: refs[i] for i in (3, 9, 12, 13, 14)} == {3: a.key, 9: a.key, 12: a.key, 13: a.key, 14: CTS_KEY}
    assert [(t.idx, t.direction, t.trigger_inner) for t in scan.triggers] == [(5, "sd", 1.41), (12, "opp_sd", 1.415)]
    assert passes == {13: [a.key]}
    assert scan.locked == {"sd": False, "opp_sd": True}


def test_5d_a_superseded_reference_moves_when_its_ladders_closest_rung_is_outward():
    """§9.2 item 4: an outward update is never blocked. A's rung 1.4150 is the reference; at 9 sub B replaces it with a
    rung 1.4170 — OUTWARD of A (farther from the sd side) → the reference moves to B's rung: 12's high 1.4165 fires
    on it, while 1.4145 (A's band) at 10 did not."""
    df = _df(mid=1.4115)
    _set(df, 5, l=1.4105)
    _set(df, 10, h=1.4145)
    _set(df, 12, h=1.4165)
    a = _sub_rung("opp_sd", "POI", 1.4150, 1.4160, pool_idx=3)
    b = _sub_rung("opp_sd", "BOS", 1.4170, 1.4180, pool_idx=5)
    scan = _scan(df, *_cycle())
    trig, _ = _drive(scan, {3: _overlay("opp_sd", a), 9: _overlay("opp_sd", b)})
    assert trig == [(5, "sd", "POI", 1.41), (12, "opp_sd", "BOS", 1.417)]
    assert scan.refs["opp_sd"].key == b.key


def test_5e_the_parent_poi_is_gated_too_the_sd_side_can_start_at_the_bos():
    """Point 4: each side starts on its last rung; the POI is an inward update judged by the gate. POI 1.41 vs CTS
    1.4140 = 40 (too narrow) → the sd side stays on the BOS 1.40 (140): 5's low 1.4105 (the POI's band) does not fire,
    8's 1.4005 fires on the BOS. With the CTS at 1.4141 (41) the POI is taken and 5 fires on it."""
    for cts_inner, first in ((1.4140, (8, "sd", "BOS", 1.40)), (1.4141, (5, "sd", "POI", 1.41))):
        df = _df(mid=1.4120)
        _set(df, 5, l=1.4105)
        _set(df, 8, l=1.4005)
        scan = _scan(df, *_cycle(cts=(cts_inner, cts_inner + 0.0010)))
        trig, _ = _drive(scan)
        assert trig[0] == first, cts_inner


def test_5f_alternation_is_unlimited_on_a_wide_cycle():
    """No counters (the user: "we always keep reference zones where we allow for unlimited alternations"): POI 1.41,
    an opp_sd rung 1.4145 (45, taken at 3) → six alternating triggers."""
    df = _df(mid=1.4125)
    for i in (5, 11, 17):
        _set(df, i, l=1.4105)
    for i in (8, 14, 20):
        _set(df, i, h=1.4140)
    scan = _scan(df, *_cycle())
    trig, _ = _drive(scan, {3: _overlay("opp_sd", _sub_rung("opp_sd", "BOS", 1.4145, 1.4155))})
    assert [(t[0], t[1]) for t in trig] == [(5, "sd"), (8, "opp_sd"), (11, "sd"), (14, "opp_sd"), (17, "sd"),
                                            (20, "opp_sd")]
    assert {t.meta["prox_ref_gap_pips"] for t in scan.triggers} == {45}


def test_5g_the_invariant_no_narrow_gap_outside_the_exception():
    """The gate admits no update that leaves the gap <= the minimum, so outside the exception a narrow gap at a
    candle's start is a bug: asserted (an update forced around the gate here)."""
    scan = _scan(_df(), *_cycle())
    scan.step(2)
    scan.close(2)
    scan._update_references = lambda first: scan.refs.update(opp_sd=_sub_rung("opp_sd", "BOS", 1.4130, 1.4140))
    with pytest.raises(AssertionError, match="outside the narrow parent-reference exception"):
        scan.step(3)


# ---------- 6 / 7. the signed, half-up gap ----------

def test_6_a_crossed_update_is_too_narrow():
    """An opp_sd rung 50 pips BELOW the sd reference (crossed): the signed gap would be −50 → too narrow (its absolute
    value, 50, would pass) → skipped; the opp_sd reference stays the CTS (200)."""
    assert proximity_reference_gap_pips(1, 1.4100, 1.4050, PIP) == -50
    df = _df()
    scan = _scan(df, *_cycle())
    scan.set_overlay("opp_sd", [_sub_rung("opp_sd", "BOS", 1.4050, 1.4060)])
    scan.step(2)
    assert scan.refs["opp_sd"].key == CTS_KEY
    assert scan.ref_gap_pips == 200 and scan.narrow_prox_ref is False


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
    # The gate on these values (POI 1.41, BOS 1.40): POI–CTS 41 → the POI is taken; 40 (`<=`) → skipped, the sd side
    # stays on the BOS (140).
    for cts_inner, gap, sd_key in ((1.41405, 41, POI_KEY), (1.41404, 140, BOS_KEY), (1.4140, 140, BOS_KEY)):
        scan = _scan(_df(), *_cycle(cts=(cts_inner, cts_inner + 0.001)))
        scan.step(2)
        assert (scan.ref_gap_pips, scan.refs["sd"].key, scan.narrow_prox_ref) == (gap, sd_key, False), cts_inner


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

def test_11_outward_moves_first_then_inward_judged_against_the_updated_other_side():
    """Point 3 (round 15): POI_a 1.4110 (outer 1.4105), POI_b 1.4100; the opp_sd reference the CTS. Candle 11 sees BOTH:
    POI_a passed at 10 → the sd side is FORCED out to POI_b, and an opp_sd rung Y 1.4145 arrives — 35 from POI_a (too
    narrow) but 45 from POI_b. Outward first → Y is judged against POI_b and taken on candle 11 itself: 11's high
    1.4140 fires on it. Judged against the pre-candle sd reference it would be skipped at 11 (and taken a candle
    later — so the trigger is ON the candle of both moves; mutant 3-03 first survived a draft firing at 12). Candles
    at 1.4125."""
    df = _df(mid=1.4125)
    _set(df, 5, l=1.4115)
    _set(df, 10, l=1.4100, c=1.4102)
    _set(df, 11, h=1.4140)
    y = _sub_rung("opp_sd", "BOS", 1.4145, 1.4155)
    scan = _scan(df, *_cycle(pois=((1.4110, 1.4105), (1.4100, 1.4095))))
    trig, passes = _drive(scan, {11: _overlay("opp_sd", y)})
    assert passes == {10: [POI_KEY]}
    assert proximity_reference_gap_pips(1, 1.4110, 1.4145, PIP) == 35
    assert trig == [(5, "sd", "POI", 1.411), (11, "opp_sd", "BOS", 1.4145)]
    assert (scan.refs["sd"].key, scan.refs["opp_sd"].key, scan.triggers[1].meta["prox_ref_gap_pips"]) == (
        POI_B_KEY, y.key, 45)


def test_11b_both_sides_offered_inward_rungs_on_one_candle_sd_is_judged_first_the_stated_default():
    """PLAN_R §9.2's DEFAULT for a case the user has not decided (absent on the reference window): BOS 1.40, CTS 1.43,
    no POI; the first sd fires on the BOS at 3 (an sd sub zone is a candidate only after it, §9.2 item 7). At 6 the sd
    side is offered S 1.4150 (150 from the CTS) and the opp_sd side O 1.4180 (180 from the BOS, 30 from S). sd first:
    S is taken, then O is judged against S → too narrow → the opp_sd side stays on the CTS, so 6's high 1.4175 (opp_sd
    expected; O's band >= 1.4172) does not fire. (opp_sd first would take O and fire on it at 6.) Candles at 1.4165 —
    BETWEEN the two rungs (outside S's band <= 1.4158 and O's >= 1.4172), so no close passes either: at 1.4200 the first
    close would pass O and both orders would converge a candle later (the first draft passed mutant 3-13 so)."""
    df = _df(mid=1.4165)
    _set(df, 3, l=1.4005)
    _set(df, 6, h=1.4175)
    s = _sub_rung("sd", "BOS", 1.4150, 1.4140)
    o = _sub_rung("opp_sd", "BOS", 1.4180, 1.4190, pool_idx=8)
    scan = _scan(df, *_cycle(pois=()))
    trig, _ = _drive(scan, {6: lambda sc: (sc.set_overlay("sd", [s]), sc.set_overlay("opp_sd", [o]))})
    assert (scan.refs["sd"].key, scan.refs["opp_sd"].key) == (s.key, CTS_KEY)
    assert trig == [(3, "sd", "BOS", 1.40)]


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
])
def test_13_a_pass_on_a_blocked_candle_still_counts(label, thresholds, method):
    """POI_a 1.4110 (outer 1.4105), POI_b 1.4100. The candle 10 closes 1.4102 (beyond POI_a's outer) while a gate blocks
    the sd test — Rule 3 (5 sd and 8 opp_sd fired while the threshold gap is narrow) or Rule 2 (narrow, not a pullback
    CTS: nothing fires). POI_a is passed all the same: from 11 the sd reference is POI_b. Candles at 1.4125. (The
    exception's gate: test 14b.)"""
    df = _df(mid=1.4125)
    _set(df, 5, l=1.4115)
    _set(df, 8, h=1.4295)
    _set(df, 10, l=1.4100, c=1.4102)
    scan = _scan(df, *_cycle(pois=((1.4110, 1.4105), (1.4100, 1.4095)), thresholds=thresholds, method=method))
    for i in range(2, 12):
        t = scan.step(i)
        if i == 10:
            assert t is None and scan.gate is not None, label
        scan.close(i)
    assert POI_KEY in scan.passed
    assert scan.refs["sd"].key == POI_B_KEY


# ---------- 14. the narrow parent-reference exception (PLAN_R §9.1) ----------

def test_14_the_narrow_parent_reference_exception_fixes_the_references_and_allows_one_sd_one_opp_sd():
    """(1,0)-like: BOS 1.4100 vs CTS 1.4130 — 30 pips, at or below 40 from the CTS_CONFIRMED (static) → the references
    stay on the BOS / CTS (a POI 1.4115 between them and an opp_sd rung 1.4120 are never used), one sd (5) and one
    opp_sd (8) fire, then the sd at 10 is blocked by the exception (`narrow_parent_refs`) — the thresholds are 320
    apart, so Rules 2 / 3 play no part. Candles at 1.4115: inside the POI's and the opp_sd rung's 8-pip bands (every
    candle would fire on them), outside the BOS's (<= 1.4108) and the CTS's (>= 1.4122)."""
    df = _df(mid=1.4115)
    _set(df, 5, l=1.4105)
    _set(df, 8, h=1.4126)
    _set(df, 10, l=1.4105)
    scan = _scan(df, *_cycle(bos=(1.4100, 1.4090), cts=(1.4130, 1.4140), pois=((1.4115, 1.4112),)))
    assert scan.narrow_parent_refs
    gates = {}
    acts = {3: _overlay("opp_sd", _sub_rung("opp_sd", "BOS", 1.4120, 1.4125))}
    for i in range(scan.scan_start, scan.scan_end + 1):
        if i in acts:
            acts[i](scan)
        scan.step(i)
        gates[i] = scan.gate
        assert (scan.refs["sd"].key, scan.refs["opp_sd"].key) == (BOS_KEY, CTS_KEY), i
        scan.close(i)
    assert [(t.idx, t.direction, t.zone_kind) for t in scan.triggers] == [(5, "sd", "BOS"), (8, "opp_sd", "CTS")]
    assert gates[10] == "narrow_parent_refs"
    assert all(t.meta["prox_ref_gap_pips"] == 30 and t.meta["narrow_prox_ref"] for t in scan.triggers)


def test_14b_a_pass_on_a_candle_the_exception_blocks_still_counts():
    """The exception keeps the references on the BOS / CTS but the pass test runs on every candle (§1): POI 1.4115
    (outer 1.4112) inside a 35-pip BOS–CTS; 10 (sd expected, blocked by the exception after 5 / 8 fired) closes 1.4110
    → the POI is passed; the sd reference is still the BOS."""
    df = _df(mid=1.4122)
    _set(df, 5, l=1.4105)
    _set(df, 8, h=1.4130)
    _set(df, 10, l=1.4105, c=1.4110)
    scan = _scan(df, *_cycle(bos=(1.4100, 1.4090), cts=(1.4135, 1.4145), pois=((1.4115, 1.4112),)))
    for i in range(2, 12):
        t = scan.step(i)
        if i == 10:
            assert t is None and scan.gate == "narrow_parent_refs"
        scan.close(i)
    assert POI_KEY in scan.passed and scan.refs["sd"].key == BOS_KEY


def test_14c_the_exception_reads_the_inner_gap_not_the_threshold_gap():
    """Inner gap 50 (BOS 1.4100, CTS 1.4150: not the exception) while the THRESHOLD gap is 30 (Rule 3 on): Rule 3 caps
    (5, 8, then 10 blocked by `rule3`); and the reverse — inner gap 30 with thresholds 320 apart — is the exception
    (test 14). The two measures are separate (point 5: "this isn't Rules 1–3, which use the wider threshold gap")."""
    df = _df(mid=1.4125)
    _set(df, 5, l=1.4105)
    _set(df, 8, h=1.4145)
    _set(df, 10, l=1.4105)
    scan = _scan(df, *_cycle(bos=(1.4100, 1.4090), cts=(1.4150, 1.4160), pois=(), thresholds=(1.4100, 1.4130)))
    assert not scan.narrow_parent_refs
    gates = {}
    for i in range(2, 12):
        scan.step(i)
        gates[i] = scan.gate
        scan.close(i)
    assert [t.idx for t in scan.triggers] == [5, 8] and gates[10] == "rule3"


# ---------- 15–19. the step-3 landing review's cases (2026-10-20) ----------

def _walk(scan, acts, upto):
    """step / close to `upto`; {i: (sd ref key, opp_sd ref key)} as of each candle's start."""
    refs = {}
    for i in range(scan.scan_start, upto + 1):
        if i in acts:
            acts[i](scan)
        scan.step(i)
        refs[i] = tuple(scan.refs[s].key if scan.refs[s] is not None else None for s in ("sd", "opp_sd"))
        scan.close(i)
    return refs


def test_15_an_sd_sub_zone_is_no_candidate_before_the_cycles_first_sd_trigger():
    """§9.2 item 7 (the rules reviewer's blocker): POI 1.41 vs CTS 1.4140 = 40 → the gate keeps it out (point 4), the
    sd side starts on the BOS. An sd sub zone S 1.4050 (beyond the POI, 90 from the CTS) would pass the gate — but
    before the cycle's first sd trigger it is no candidate, so the first sd fires where the batch scan (no overlay)
    fires: 5's low 1.4055 (S's band) does not, 8's 1.4005 fires on the BOS — the trigger the main WVMI gate reads,
    asserted by `orchestrator._final_proximity_triggers`. From 9 S is a candidate and becomes the sd reference."""
    from types import SimpleNamespace

    from engine_v2.pipeline.orchestrator import _final_proximity_triggers
    df = _df(mid=1.4120)
    _set(df, 5, l=1.4055)
    _set(df, 8, l=1.4005)
    batch = _scan(df, *_cycle(cts=(1.4140, 1.4150)))
    btrig, _ = _drive(batch)
    sweep = _scan(df, *_cycle(cts=(1.4140, 1.4150)))
    s = _sub_rung("sd", "BOS", 1.4050, 1.4040, pool_idx=4)
    refs = _walk(sweep, {2: _overlay("sd", s)}, sweep.scan_end)
    assert btrig[0] == (8, "sd", "BOS", 1.40) and sweep.triggers[0].idx == 8 and sweep.triggers[0].zone_kind == "BOS"
    assert sweep.triggers[0].trigger_inner == 1.40 and sweep.triggers[0].meta["prox_ref_source"] == "parent"
    assert refs[8][0] == BOS_KEY and refs[9][0] == s.key
    overlay = SimpleNamespace(scans={(0, 0): sweep}, used={})
    _final_proximity_triggers({(0, 0): batch.triggers}, overlay, SimpleNamespace(get_by_pool_idx=lambda i: None))


def test_15b_the_first_candles_sd_first_order_keeps_the_first_sd_on_the_batchs_rung():
    """§9.2: the scan's first candle judges the sd side first, against the CTS — like the batch scan, which has no
    opp_sd overlay. POI 1.41 (50 from the CTS 1.4150) and an opp_sd sub rung O 1.4120 (20 from the POI) at the first
    candle: sd first → the POI is taken, O (20) is skipped → 5 fires on the POI, as in the batch. (opp_sd first would
    take O — 120 from the BOS — and keep the POI out: the first sd would leave the batch's rung.)"""
    df = _df(mid=1.4114)
    _set(df, 5, l=1.4105)
    o = _sub_rung("opp_sd", "BOS", 1.4120, 1.4130)
    batch = _scan(df, *_cycle(cts=(1.4150, 1.4160)))
    btrig, _ = _drive(batch)
    sweep = _scan(df, *_cycle(cts=(1.4150, 1.4160)))
    strig, _ = _drive(sweep, {2: _overlay("opp_sd", o)})
    assert btrig[0] == strig[0] == (5, "sd", "POI", 1.41)
    assert sweep.refs["opp_sd"].key == CTS_KEY


@pytest.mark.parametrize("with_x", [True, False])
def test_16_a_side_without_a_last_rung_may_be_left_without_a_reference(with_x):
    """A cycle without a CTS zone (no opp_sd last rung — the floor): its opp_sd reference Y 1.4145 is passed by 14's
    close 1.4158 and no other rung passes the gate (X 1.4130 is 30 from the BOS; or there is none) → the side has NO
    reference from 15 (step 2's outcome; was an assert — both reviewers' crash). A later rung Z 1.4150 (50) is taken."""
    df = _df(mid=1.4115)
    _set(df, 5, l=1.4105)
    _set(df, 12, h=1.4140)
    _set(df, 14, h=1.4160, c=1.4158)
    events = [_bos_confirmed(idx=0, price=1.3990), _cts_confirmed_at(2, 1.4310)]
    scan = build_cycle_proximity_scans(df, events, [_bos_zone(0, 0, 1, 1.4100, 1.4090)], [], PIP, "H1")[(0, 0)]
    x = _sub_rung("opp_sd", "BOS", 1.4130, 1.4170)          # tall: 14's close 1.4158 does not pass it
    y = _sub_rung("opp_sd", "POI", 1.4145, 1.4155)
    z = _sub_rung("opp_sd", "POI", 1.4150, 1.4175, pool_idx=9)
    first = [x, y] if with_x else [y]
    refs = _walk(scan, {11: _overlay("opp_sd", *first), 18: _overlay("opp_sd", *first, z)}, 19)
    assert refs[12][1] == y.key and refs[15][1] is None and refs[17][1] is None and refs[19][1] == z.key
    assert [(t.idx, t.direction) for t in scan.triggers] == [(5, "sd"), (12, "opp_sd")]
    assert not scan.locked["opp_sd"]


def test_17_a_superseded_references_outward_move_comes_first_point_3():
    """The rules reviewer's case: a SUPERSEDED opp_sd reference whose ladder's closest rung is OUTWARD moves in the
    outward phase (point 3: "Apply outward moves first"), so the sd side's inward candidate is judged against it.
    POI passed at 3 → the sd side to S 1.4050; opp_sd A 1.4110 from 4. At 8 sub B replaces A with B 1.4130 (outward)
    and the sd side is offered S2 1.4080 — 30 from A, 50 from B → taken at 8 (judged against the moved opp_sd side) and
    fired on at 8 (l 1.4085). Judged against A it would wait a candle."""
    df = _df(mid=1.4095)
    _set(df, 2, l=1.4105)
    _set(df, 3, l=1.4085, c=1.4085)
    _set(df, 6, h=1.4105)
    _set(df, 8, l=1.4085)
    s = _sub_rung("sd", "BOS", 1.4050, 1.4040, pool_idx=2)
    s2 = _sub_rung("sd", "POI", 1.4080, 1.4075, pool_idx=2, cycle=1)
    a = _sub_rung("opp_sd", "POI", 1.4110, 1.4120, pool_idx=3)
    b = _sub_rung("opp_sd", "BOS", 1.4130, 1.4140, pool_idx=5)
    scan = _scan(df, *_cycle())
    refs = _walk(scan, {2: _overlay("sd", s), 4: _overlay("opp_sd", a),
                        8: lambda sc: (sc.set_overlay("opp_sd", [b]), sc.set_overlay("sd", [s, s2]))}, 9)
    assert refs[7] == (s.key, a.key) and refs[8] == (s2.key, b.key)
    assert [(t.idx, t.direction, t.trigger_inner) for t in scan.triggers] == [
        (2, "sd", 1.41), (6, "opp_sd", 1.411), (8, "sd", 1.408)]


def test_18_a_forced_tie_with_the_last_rung_goes_to_the_parent_and_locks():
    """Ties (§1, the tests reviewer's R02): an sd sub rung AT the BOS inner; the POI (fired on at 5) is passed at 7 →
    the forced pick, among equal inners, is the parent BOS (parent first) → the side LOCKS."""
    df = _df()
    _set(df, 5, l=1.4105)
    _set(df, 7, l=1.4080, c=1.4085)
    scan = _scan(df, *_cycle())
    tie = _sub_rung("sd", "BOS", 1.4000, 1.3990)
    refs = _walk(scan, {2: _overlay("sd", tie)}, 9)
    assert refs[8][0] == BOS_KEY and scan.locked["sd"]


def test_18b_a_superseded_sd_reference_stays_pass_tested():
    """5c's sd mirror (the tests reviewer's R07): after the first sd on the POI (3), sd sub A 1.4250 (outer 1.4240) is
    the reference from 4; at 9 sub B replaces it (1.4265 / 1.4270: 35 / 30 from the CTS, too narrow; tall, outers
    1.4230 / 1.4225) → A superseded, kept; 12 fires on A and closes 1.4235 beyond A's outer → passed → 13 the closest
    rung passing the gate, the parent POI. Candles at 1.4275 (outside every band, beyond no outer)."""
    df = _df(mid=1.4275)
    _set(df, 3, l=1.4105)
    _set(df, 6, h=1.4295)
    _set(df, 12, l=1.4230, c=1.4235)
    a = _sub_rung("sd", "POI", 1.4250, 1.4240, pool_idx=3)
    b1 = _sub_rung("sd", "BOS", 1.4265, 1.4230, pool_idx=5)
    b2 = _sub_rung("sd", "POI", 1.4270, 1.4225, pool_idx=5)
    scan = _scan(df, *_cycle())
    refs = _walk(scan, {4: _overlay("sd", a), 9: _overlay("sd", b1, b2)}, 13)
    assert (refs[4][0], refs[9][0], refs[11][0], refs[13][0]) == (a.key, a.key, a.key, POI_KEY)
    assert [(t.idx, t.direction, t.trigger_inner) for t in scan.triggers] == [
        (3, "sd", 1.41), (6, "opp_sd", 1.43), (12, "sd", 1.425)]
    assert a.key in scan.passed and not scan.locked["sd"]


def test_19_edge_choices_pinned():
    """§9.2's edge choices (the tests reviewer's R04 / R19 / R26; none on the reference window): (a) a side with no
    reference yet (no CTS zone) takes the CLOSEST rung passing the gate (point 2): X 1.4145 (45) over Y 1.4160 (60);
    (b) a superseded reference moves to a rung AT its inner (the gap is unchanged, so the update keeps it above the
    minimum); (c) a superseded reference whose ladder's closest rung is the CTS moves there but does NOT lock (only a
    pass locks)."""
    df = _df(mid=1.4115)
    events = [_bos_confirmed(idx=0, price=1.3990), _cts_confirmed_at(2, 1.4310)]
    scan = build_cycle_proximity_scans(df, events, [_bos_zone(0, 0, 1, 1.4100, 1.4090)], [], PIP, "H1")[(0, 0)]
    x = _sub_rung("opp_sd", "BOS", 1.4145, 1.4155)
    y = _sub_rung("opp_sd", "POI", 1.4160, 1.4170)
    assert _walk(scan, {8: _overlay("opp_sd", x, y)}, 9)[8][1] == x.key
    a = _sub_rung("opp_sd", "POI", 1.4150, 1.4160, pool_idx=3)
    for b, want in ((_sub_rung("opp_sd", "BOS", 1.4150, 1.4170, pool_idx=5), "b"),
                    (_sub_rung("opp_sd", "BOS", 1.4350, 1.4360, pool_idx=5), CTS_KEY)):
        scan = _scan(_df(mid=1.4115), *_cycle())
        refs = _walk(scan, {3: _overlay("opp_sd", a), 9: _overlay("opp_sd", b)}, 10)
        assert refs[8][1] == a.key and refs[9][1] == (b.key if want == "b" else want)
        assert scan.locked["opp_sd"] is False


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
# opp_sd outward move, the sd = −1 side, the outer read at the pass site, the gap-None path) or was killed only
# incidentally ("Rule 2 is not part of the cap"). The reviewer's sketches; review_1 / 3 / 5 / 6 re-written for step 3's
# gate (PLAN_R §9) on the same paths.

def _bear_cycle(pois, cts=(1.4160, 1.4150), bos=(1.4300, 1.4310), thresholds=(1.4310, 1.4150), method="pullback"):
    kl = [_bos_zone(0, 0, -1, *bos), _cts_zone(0, 0, -1, *cts)]
    # sd=-1: a POI's inner is its BOTTOM, its outer its TOP
    pz = [_poi_zone(0, 0, -1, outer, inner, cts_established_idx=0) for inner, outer in pois]
    events = [_bos_confirmed(idx=0, price=thresholds[0], sd=-1), _cts_confirmed_at(2, thresholds[1], sd=-1, method=method)]
    return events, kl, pz


def test_review_1_bearish_outward_first_then_inward_against_the_updated_other_side():
    """Test 11's mirror on sd = −1: POI_a 1.4210 (outer 1.4215), POI_b 1.4220 (outer 1.4225), CTS 1.4160 (POI_a 50).
    5 sd on POI_a; 10's close 1.4218 passes POI_a (opp_sd expected); at 11 the sd side is forced UP (outward for a
    bearish cycle) to POI_b, and an opp_sd rung Y 1.4175 arrives — 35 from POI_a, 45 from POI_b → taken: 11's low
    1.4180 fires on it ON the candle of both moves (the tests reviewer: a hit at 12 let mutant 3-03 survive, as in
    test 11's first draft). Candles at 1.4195."""
    df = _df(mid=1.4195)
    _set(df, 5, h=1.4205)
    _set(df, 10, h=1.4220, c=1.4218)
    _set(df, 11, l=1.4180)
    y = _sub_rung("opp_sd", "BOS", 1.4175, 1.4165)
    scan = _scan(df, *_bear_cycle(pois=((1.4210, 1.4215), (1.4220, 1.4225))))
    trig, passes = _drive(scan, {11: _overlay("opp_sd", y)})
    assert passes == {10: [("parent", "POI", 0)]}
    assert proximity_reference_gap_pips(-1, 1.4210, 1.4175, PIP) == 35
    assert trig == [(5, "sd", "POI", 1.421), (11, "opp_sd", "BOS", 1.4175)]
    assert scan.triggers[1].meta["prox_ref_gap_pips"] == 45


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


def test_review_3_a_passed_opp_sd_reference_moves_outward_to_the_next_rung_not_the_cts():
    """X 1.4150 / Y 1.4160 opp_sd rungs from 6 (50 / 60 from the POI) → X. 5 sd, 8 opp_sd on X; 9 (sd expected) closes
    1.4157 > X's outer 1.4155 → passed; from 10 the opp_sd side is forced OUTWARD to the closest rung passing the gate,
    Y (not the CTS, no lock): 11 sd, 13 opp_sd on Y."""
    df = _df(mid=1.4125)
    _set(df, 5, l=1.4105)
    _set(df, 8, h=1.4145)
    _set(df, 9, h=1.4158, l=1.4140, c=1.4157)
    _set(df, 11, l=1.4105)
    _set(df, 13, h=1.4155)
    x = _sub_rung("opp_sd", "BOS", 1.4150, 1.4155)
    y = _sub_rung("opp_sd", "POI", 1.4160, 1.4170)
    scan = _scan(df, *_cycle())
    trig, passes = _drive(scan, {6: _overlay("opp_sd", x, y)})
    assert passes == {9: [x.key]}
    assert [t[:4] for t in trig] == [(5, "sd", "POI", 1.41), (8, "opp_sd", "BOS", 1.415), (11, "sd", "POI", 1.41),
                                     (13, "opp_sd", "POI", 1.416)]
    assert scan.refs["opp_sd"].key == y.key and scan.locked == {"sd": False, "opp_sd": False}


def test_review_4_the_pass_reads_the_outer_as_of_the_close_including_an_expansion_on_that_candle():
    """An sd overlay rung whose outer expands 1.4090 -> 1.4080 at moment 7: 7's close 1.4085 is beyond the OLD outer
    but not beyond the outer as of its close -> no pass; 12 still fires on the rung. (The first sd fires on the BOS at
    3 — an sd sub zone is a candidate only after it, PLAN_R §9.2 item 7 — so the rung is the reference from 4.)"""
    df = _df()
    _set(df, 3, l=1.4005)
    _set(df, 7, l=1.4082, c=1.4085)
    _set(df, 10, h=1.4295)
    _set(df, 12, l=1.4105)
    r = ProximityRung(key=("sub", 7, 0, "BOS"), side="sd", zone_kind="BOS", inner=1.4100,
                      outer_at=lambda i: 1.4090 if i < 7 else 1.4080, source="sub", tie_rank=1)
    scan = _scan(df, *_cycle(pois=()))
    trig, passes = _drive(scan, {2: _overlay("sd", r)})
    assert passes == {}
    assert trig == [(3, "sd", "BOS", 1.40), (10, "opp_sd", "CTS", 1.43), (12, "sd", "BOS", 1.41)]


def test_review_5_a_cycle_without_a_cts_zone_takes_its_first_opp_sd_reference_through_the_gate():
    """No CTS zone: no opp_sd last rung, the gap is None, no exception. 5 sd on the BOS (meta None / False). An opp_sd
    overlay rung X at 8, 30 from the BOS → too narrow: the side stays without a reference (10's high 1.4125 does not
    fire); at 11 Y 1.4145 (45) joins → taken: 12's high 1.4140 fires on it."""
    df = _df(mid=1.4115)
    _set(df, 5, l=1.4105)
    _set(df, 10, h=1.4125)
    _set(df, 12, h=1.4140)
    events = [_bos_confirmed(idx=0, price=1.3990), _cts_confirmed_at(2, 1.4310)]
    kl = [_bos_zone(0, 0, 1, 1.4100, 1.4090)]
    scan = build_cycle_proximity_scans(df, events, kl, [], PIP, "H1")[(0, 0)]
    x = _sub_rung("opp_sd", "BOS", 1.4130, 1.4140)
    y = _sub_rung("opp_sd", "POI", 1.4145, 1.4155)
    trig, _ = _drive(scan, {8: _overlay("opp_sd", x), 11: _overlay("opp_sd", x, y)})
    assert not scan.narrow_parent_refs
    assert scan.triggers[0].meta == {"prox_ref_gap_pips": None, "narrow_prox_ref": False, "prox_ref_source": "parent", "prox_ref_sub_id": None, "prox_ref_cycle": None, "prox_ref_usable_idx": None}
    assert trig == [(5, "sd", "BOS", 1.41), (12, "opp_sd", "POI", 1.4145)]
    assert scan.triggers[1].meta["prox_ref_gap_pips"] == 45


def test_review_6_rule_2_is_not_part_of_the_exception():
    """Thresholds wide (320) but the CTS was confirmed by sd-proximity; the parent BOS–CTS inners 30 apart (the
    exception): Rule 2 does not apply → one sd + one opp_sd fire (5, 8), then the exception blocks 10."""
    df = _df(mid=1.4115)
    _set(df, 5, l=1.4105)
    _set(df, 8, h=1.4125)
    _set(df, 10, l=1.4105)
    scan = _scan(df, *_cycle(bos=(1.4100, 1.4090), cts=(1.4130, 1.4140), pois=(), method="sd_zone_proximity"))
    gates = {}
    for i in range(2, 12):
        scan.step(i)
        gates[i] = scan.gate
        scan.close(i)
    assert [t.idx for t in scan.triggers] == [5, 8] and gates[10] == "narrow_parent_refs"


def test_review_7_an_overlay_rung_is_pass_tested():
    """Test 1 with the POI as an sd overlay rung (the reference from 4, after the first sd on the BOS at 3 — PLAN_R
    §9.2 item 7): 7's close passes it, from 8 the sd reference is the BOS (locked) — 12's low 1.4105, within 8 pips of
    the passed rung, does not fire; 15 fires on the BOS."""
    df = _df()
    _set(df, 3, l=1.4005)
    _set(df, 7, l=1.4080, c=1.4085)
    _set(df, 10, h=1.4295)
    _set(df, 12, l=1.4105)
    _set(df, 15, l=1.4005)
    r = _sub_rung("sd", "POI", 1.4100, 1.4090)
    scan = _scan(df, *_cycle(pois=()))
    trig, passes = _drive(scan, {2: _overlay("sd", r)})
    assert passes == {7: [r.key]}
    assert trig == [(3, "sd", "BOS", 1.40), (10, "opp_sd", "CTS", 1.43), (15, "sd", "BOS", 1.40)]
    assert scan.locked["sd"]
