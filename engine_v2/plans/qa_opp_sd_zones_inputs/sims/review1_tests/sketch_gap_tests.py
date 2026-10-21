"""Reviewer's sketch tests for the surviving mutants (scratch, not for the repo). Adopted as
`tests/test_proximity_reference_ladders.py::test_review_1..7`. Renamed from `test_gap_sketches.py` at PLAN_R step 3
(2026-10-20) so pytest no longer collects it: four sketches pin steps 1-2's narrow proximity-reference cap, which the
proximity reference gate replaced (the step-2 suite count 1820 included these 7)."""
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.tests.test_zone_proximity import _bos_confirmed, _bos_zone, _cts_zone, _poi_zone
from engine_v2.tests.test_proximity_reference_ladders import (
    _cts_confirmed_at, _cycle, _df, _drive, _overlay, _scan, _set, _sub_rung, PIP, BOS_KEY, CTS_KEY,
)
from engine_v2.zones.zone_proximity import ProximityRung, build_cycle_proximity_scans


def _bear_cycle(pois, cts=(1.4160, 1.4150), bos=(1.4300, 1.4310), thresholds=(1.4310, 1.4150), method="pullback"):
    kl = [_bos_zone(0, 0, -1, *bos), _cts_zone(0, 0, -1, *cts)]
    # sd=-1: a POI's inner is its BOTTOM, its outer its TOP
    pz = [_poi_zone(0, 0, -1, outer, inner, cts_established_idx=0) for inner, outer in pois]
    events = [_bos_confirmed(idx=0, price=thresholds[0], sd=-1), _cts_confirmed_at(2, thresholds[1], sd=-1, method=method)]
    return events, kl, pz


def test_T1_bearish_outward_move_while_narrow_resets_that_side():
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


def test_T2_an_opp_sd_pass_moves_the_reference_to_the_cts_and_locks():
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


def test_T3_an_opp_sd_outward_move_while_narrow_resets_the_opp_sd_counter():
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


def test_T4_the_pass_reads_the_outer_as_of_the_close_including_an_expansion_on_that_candle():
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


def test_T5_a_cycle_without_a_cts_zone_is_wide_until_an_opp_sd_reference_exists():
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


def test_T6_rule_2_is_not_part_of_the_cap():
    """Thresholds wide (320) but the CTS was confirmed by sd-proximity; the references are 30 apart (narrow by the cap):
    Rule 2 does not apply -> one sd + one opp_sd fire (5, 8), then the cap."""
    df = _df(mid=1.4115)
    _set(df, 5, l=1.4105)
    _set(df, 8, h=1.4125)
    _set(df, 10, l=1.4105)
    scan = _scan(df, *_cycle(cts=(1.4130, 1.4140), method="sd_zone_proximity"))
    trig, _ = _drive(scan)
    assert [t[0] for t in trig] == [5, 8]


def test_T9_an_overlay_rung_is_pass_tested():
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
