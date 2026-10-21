"""Unit tests for zones/zone_proximity.py — zone proximity trigger detection."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import pytest

from engine_v2.common.types import KLZone
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.tests._event_factory import make_bos_confirmed
from engine_v2.zones.poi_zones import POIZone
from engine_v2.zones.zone_proximity import (
    DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS,
    DEFAULT_PROXIMITY_PIPS,
    ZoneProximityTrigger,
    check_zone_proximity,
    poi_unclamped_activation_floor_idx,
)


# ---------- Helpers ----------

def _make_df(n: int, default_h: float = 1.50, default_l: float = 1.40) -> pd.DataFrame:
    """Build a minimal df with neutral OHLC; tests override specific candles."""
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n, freq="h"),
        "o": [(default_h + default_l) / 2] * n,
        "h": [default_h] * n,
        "l": [default_l] * n,
        "c": [(default_h + default_l) / 2] * n,
    })


def _cts_confirmed(idx: int, sid: int = 0, cycle_id: int = 0, sd: int = 1,
                   confirmation_method: str = "pullback") -> StructureEvent:
    return StructureEvent(
        idx=idx, category="STRUCTURE", type="CTS_CONFIRMED",
        price=1.50,
        meta={"structure_id": sid, "cycle_id": cycle_id,
              "struct_direction": sd, "confirmed_at": idx,
              "confirmation_method": confirmation_method},
    )


def _bos_confirmed(idx: int, price: float, sid: int = 0, cycle_id: int = 0,
                   sd: int = 1, confirmed_at: Optional[int] = None) -> StructureEvent:
    return make_bos_confirmed(
        bos_anchor_idx=idx, price=price, structure_id=sid, cycle_id=cycle_id,
        struct_direction=sd, confirmed_at=confirmed_at if confirmed_at is not None else idx,
    )


def _cts_thresh_updated(idx: int, price: float, prev: float, sid: int = 0,
                        cycle_id: int = 0) -> StructureEvent:
    return StructureEvent(
        idx=idx, category="STRUCTURE", type="CTS_THRESHOLD_UPDATED",
        price=price,
        meta={"structure_id": sid, "cycle_id": cycle_id, "prev": prev},
    )


def _bos_thresh_updated(idx: int, price: float, prev: float, sid: int = 0,
                        cycle_id: int = 0) -> StructureEvent:
    return StructureEvent(
        idx=idx, category="STRUCTURE", type="BOS_THRESHOLD_UPDATED",
        price=price,
        meta={"structure_id": sid, "cycle_id": cycle_id, "prev": prev},
    )


def _bos_zone(sid: int, cycle_id: int, sd: int, inner: float, outer: float) -> KLZone:
    """Build a BOS KL zone. For sd=+1 inner is the TOP (zone sits above outer)."""
    side = "buy" if sd == 1 else "sell"
    if sd == 1:
        top, bottom = inner, outer
    else:
        top, bottom = outer, inner
    return KLZone(
        start_time=pd.Timestamp("2026-01-01", tz="UTC"),
        end_time=None, side=side, top=top, bottom=bottom,
        source_kind="BOS",
        source_time=pd.Timestamp("2026-01-01", tz="UTC"),
        source_price=outer,
        meta={
            "structure_id": sid, "cycle_id": cycle_id,
            "struct_direction": sd, "inner": inner, "outer": outer,
        },
    )


def _cts_zone(sid: int, cycle_id: int, sd: int, inner: float, outer: float) -> KLZone:
    """Build a CTS KL zone. For sd=+1 the CTS zone is sell-side (inner is the BOTTOM)."""
    side = "sell" if sd == 1 else "buy"
    if sd == 1:
        top, bottom = outer, inner
    else:
        top, bottom = inner, outer
    return KLZone(
        start_time=pd.Timestamp("2026-01-01", tz="UTC"),
        end_time=None, side=side, top=top, bottom=bottom,
        source_kind="CTS",
        source_time=pd.Timestamp("2026-01-01", tz="UTC"),
        source_price=outer,
        meta={
            "structure_id": sid, "cycle_id": cycle_id,
            "struct_direction": sd, "inner": inner, "outer": outer,
        },
    )


def _poi_zone(sid: int, cycle_id: int, sd: int, top: float, bottom: float, *,
              cts_established_idx: int, ic_idx: Optional[int] = None,
              pattern_confirmed_idx: Optional[int] = None, end_idx=None,
              activation_history=None) -> POIZone:
    """Build a POI zone. The proximity scan reads it like a KL zone (PLAN_Q): at
    every candle of the scan, asserting its unclamped activation floor —
    max(`cts_established_idx`, the IC[, `pattern_confirmed_idx`]); the IC
    defaults to the candle before the establishment — is at or before the scan
    start, and NEVER reading its `activation_history` (default: one activate at
    `cts_established_idx`; pass it explicitly to model a POI that flaps, or `[]`
    for a collapsed cycle's) nor the scalar `confirmed_idx` (the last activate,
    stamped like the real meta)."""
    side = "buy" if sd == 1 else "sell"
    if activation_history is None:
        activation_history = [{"idx": cts_established_idx, "active": True, "reason": "initial"}]
    activates = [ev["idx"] for ev in activation_history if ev["active"]]
    return POIZone(
        start_time=pd.Timestamp("2026-01-01", tz="UTC"),
        end_time=None, side=side, top=top, bottom=bottom,
        ic_idx=cts_established_idx - 1 if ic_idx is None else ic_idx,
        meta={
            "structure_id": sid, "cycle_id": cycle_id,
            "confirmed_idx": activates[-1] if activates else None, "end_idx": end_idx,
            "ip_rank": "1A.i" if pattern_confirmed_idx is not None else None,
            "pattern_confirmed_idx": pattern_confirmed_idx,
            "cts_established_idx": cts_established_idx,
            "activation_history": activation_history,
        },
    )


# The proximity reference gate OUT of the picture (PLAN_R §9): no rounded gap is ever <= this — no update is skipped
# and no cycle is the narrow parent-reference exception. For the tests that isolate Rules 2 / 3 (the BOS–CTS threshold
# gap) on zones whose inners sit 20 pips apart — the exception under the default minimum (<= 40), so the two rule sets
# would both bind.
_NO_REF_CAP = -10 ** 6


# ---------- Tests ----------

def test_first_sd_trigger_fires_on_buy_zone():
    """sd=+1: BOS inner=1.40, H1 threshold 8 pips (0.0008). Candle low 1.4005 → triggers."""
    df = _make_df(20, default_h=1.50, default_l=1.49)
    df.at[5, "l"] = 1.4005  # 5 pips above inner 1.40, within 8p H1 threshold
    bos = _bos_zone(sid=0, cycle_id=0, sd=1, inner=1.40, outer=1.39)

    triggers = check_zone_proximity(
        df=df, sorted_events=[_cts_confirmed(2)],
        kl_zones=[bos], poi_zones=[],
        pip_size=0.0001, timeframe="H1",
    )
    assert (0, 0) in triggers
    assert len(triggers[(0, 0)]) == 1
    assert triggers[(0, 0)][0].direction == "sd"
    assert triggers[(0, 0)][0].idx == 5
    assert triggers[(0, 0)][0].zone_kind == "BOS"


def test_alternation_sd_then_opp_sd_then_sd():
    """V movement: sd → opp_sd → sd within one cycle."""
    df = _make_df(30, default_h=1.50, default_l=1.49)
    # sd trigger at idx 5: low 1.4005 (within 8p of BOS inner 1.40)
    df.at[5, "l"] = 1.4005
    # opp_sd trigger at idx 10: high 1.5995 (within 8p of CTS inner 1.60)
    df.at[10, "h"] = 1.5995
    # second sd trigger at idx 15: low 1.4004
    df.at[15, "l"] = 1.4004

    bos = _bos_zone(sid=0, cycle_id=0, sd=1, inner=1.40, outer=1.39)
    cts = _cts_zone(sid=0, cycle_id=0, sd=1, inner=1.60, outer=1.61)

    triggers = check_zone_proximity(
        df=df, sorted_events=[_cts_confirmed(2)],
        kl_zones=[bos, cts], poi_zones=[],
        pip_size=0.0001, timeframe="H1",
    )
    assert len(triggers[(0, 0)]) == 3
    assert triggers[(0, 0)][0].direction == "sd" and triggers[(0, 0)][0].idx == 5
    assert triggers[(0, 0)][1].direction == "opp_sd" and triggers[(0, 0)][1].idx == 10
    assert triggers[(0, 0)][2].direction == "sd" and triggers[(0, 0)][2].idx == 15


def test_no_alternation_without_opp_sd_zone():
    """If no CTS zone exists, only the first sd trigger fires; no further triggers."""
    df = _make_df(30, default_h=1.50, default_l=1.49)
    df.at[5, "l"] = 1.401
    df.at[10, "h"] = 1.60   # would have been opp_sd if CTS zone existed
    df.at[15, "l"] = 1.40   # second sd attempt — but state is "expecting opp_sd" forever

    bos = _bos_zone(sid=0, cycle_id=0, sd=1, inner=1.40, outer=1.39)

    triggers = check_zone_proximity(
        df=df, sorted_events=[_cts_confirmed(2)],
        kl_zones=[bos], poi_zones=[],
        pip_size=0.0001, timeframe="H1",
    )
    assert len(triggers[(0, 0)]) == 1
    assert triggers[(0, 0)][0].direction == "sd"


def test_poi_included_in_sd_inners_nearer_than_bos():
    """POI zone closer to current price than BOS should win as trigger_inner."""
    df = _make_df(30, default_h=1.50, default_l=1.49)
    # BOS inner 1.40, POI top 1.45 (closer to current price for sd=+1).
    # Candle low 1.4505 → within 8 pips of POI inner (1.45) but not BOS (1.40 + 0.0008 = 1.4008 too low).
    df.at[5, "l"] = 1.4505
    bos = _bos_zone(sid=0, cycle_id=0, sd=1, inner=1.40, outer=1.39)
    poi = _poi_zone(sid=0, cycle_id=0, sd=1, top=1.45, bottom=1.44, cts_established_idx=2)

    triggers = check_zone_proximity(
        df=df, sorted_events=[_cts_confirmed(2)],
        kl_zones=[bos], poi_zones=[poi],
        pip_size=0.0001, timeframe="H1",
    )
    assert triggers[(0, 0)][0].zone_kind == "POI"
    assert triggers[(0, 0)][0].trigger_inner == 1.45


def test_poi_counts_from_the_scan_start_candle_with_its_floor_at_it():
    """PLAN_Q: a POI counts from the scan start (2) itself, like the BOS zone —
    its unclamped activation floor AT the scan start passes the assert (the
    pattern confirmed at 2: the `<=` boundary) and its EMPTY activation history
    is not read. Candle 2 sits 5 pips from the POI inner (within H1's 8); the BOS
    inner is far. Replaces `test_poi_not_active_yet_excluded`, whose candle sat
    10 pips away — outside 8 since the 2026-05-28 tuning, so it passed whatever
    the gate did."""
    df = _make_df(30, default_h=1.50, default_l=1.49)
    df.at[2, "l"] = 1.4505
    bos = _bos_zone(sid=0, cycle_id=0, sd=1, inner=1.40, outer=1.39)
    poi = _poi_zone(sid=0, cycle_id=0, sd=1, top=1.45, bottom=1.44,
                    cts_established_idx=2, ic_idx=1, pattern_confirmed_idx=2,
                    activation_history=[])

    triggers = check_zone_proximity(
        df=df, sorted_events=[_cts_confirmed(2)],
        kl_zones=[bos], poi_zones=[poi],
        pip_size=0.0001, timeframe="H1",
    )
    assert [(t.idx, t.zone_kind) for t in triggers[(0, 0)]] == [(2, "POI")]


def test_poi_floor_after_the_scan_start_raises():
    """PLAN_Q §3: a POI's unclamped activation floor is provably at or before its
    cycle's scan start (the IC is at or before the fib's final CTS anchor, which
    precedes the CTS_CONFIRMED moment; the pattern confirms by the anchor + 1). A POI
    past it could not yet be active where the location read counts it — it fails
    loudly instead of being read early (or gated by a rule nobody decided)."""
    df = _make_df(30, default_h=1.50, default_l=1.49)
    bos = _bos_zone(sid=0, cycle_id=0, sd=1, inner=1.40, outer=1.39)
    poi = _poi_zone(sid=0, cycle_id=0, sd=1, top=1.45, bottom=1.44,
                    cts_established_idx=2, ic_idx=1, pattern_confirmed_idx=3)
    with pytest.raises(AssertionError, match="after the scan start 2"):
        check_zone_proximity(
            df=df, sorted_events=[_cts_confirmed(2)],
            kl_zones=[bos], poi_zones=[poi],
            pip_size=0.0001, timeframe="H1",
        )


def test_poi_fires_in_first_active_stretch_not_last_activate():
    """Regression (sid1 cyc2 / idx 926): a POI that flaps active->inactive->
    active has its scalar `confirmed_idx` collapsed to the LAST activate (20).
    The scan must not gate on that scalar — since PLAN_Q it reads neither the
    scalar nor the history (the POI counts from the scan start, 2) — so candle 6
    triggers."""
    df = _make_df(30, default_h=1.50, default_l=1.49)
    df.at[6, "l"] = 1.4505  # within 8p of POI inner 1.45; far from BOS inner 1.40
    bos = _bos_zone(sid=0, cycle_id=0, sd=1, inner=1.40, outer=1.39)
    # Active [5,9], inactive [10,19], active [20,...]. Scalar confirmed_idx=20.
    poi = _poi_zone(
        sid=0, cycle_id=0, sd=1, top=1.45, bottom=1.44, cts_established_idx=2,
        activation_history=[
            {"idx": 5, "active": True}, {"idx": 10, "active": False},
            {"idx": 20, "active": True},
        ],
    )
    assert poi.meta["confirmed_idx"] == 20
    triggers = check_zone_proximity(
        df=df, sorted_events=[_cts_confirmed(2)],
        kl_zones=[bos], poi_zones=[poi],
        pip_size=0.0001, timeframe="H1",
    )
    assert (0, 0) in triggers
    assert triggers[(0, 0)][0].direction == "sd"
    assert triggers[(0, 0)][0].idx == 6
    assert triggers[(0, 0)][0].zone_kind == "POI"


def test_poi_counts_during_inactive_stretch():
    """PLAN_Q (the user, 2026-10-14: "I would like POI to be treated exactly like
    a KL zone"): a candle inside the POI's INACTIVE stretch (its imbalance filled,
    never re-met) still triggers off it — the activation history is not read.
    Replaces `test_poi_suppressed_during_inactive_stretch` (the old gate)."""
    df = _make_df(30, default_h=1.50, default_l=1.49)
    df.at[12, "l"] = 1.4504  # 4 pips from POI inner 1.45; BOS too far
    bos = _bos_zone(sid=0, cycle_id=0, sd=1, inner=1.40, outer=1.39)
    # Active [5,9] then inactive from 10 onward (no reactivation).
    poi = _poi_zone(
        sid=0, cycle_id=0, sd=1, top=1.45, bottom=1.44, cts_established_idx=2,
        activation_history=[
            {"idx": 5, "active": True}, {"idx": 10, "active": False},
        ],
    )
    triggers = check_zone_proximity(
        df=df, sorted_events=[_cts_confirmed(2)],
        kl_zones=[bos], poi_zones=[poi],
        pip_size=0.0001, timeframe="H1",
    )
    assert [(t.idx, t.zone_kind, t.trigger_inner) for t in triggers[(0, 0)]] == [(12, "POI", 1.45)]


def test_poi_on_collapsed_cycle_counts_like_its_bos_zone():
    """PLAN_Q, S_O's H1 structure 1 cycle 1 (collapsed: lifecycle [902, 902)):
    its POI — IC 732, zone [0.58068, 0.58118], `inactive`, EMPTY activation
    history — counts exactly like the cycle's (equally inactive) BOS zone, inner
    0.58132. The 820 candle (high 0.58064) is within 8 pips of both; the POI's
    inner is the nearer one for a sell structure, so the trigger reads sd:POI at
    0.58068 (the old gate ignored the POI: sd:BOS at 0.58132)."""
    df = _make_df(30, default_h=0.5790, default_l=0.5780)
    df.at[10, "h"] = 0.58064
    bos = _bos_zone(sid=1, cycle_id=1, sd=-1, inner=0.58132, outer=0.58196)
    poi = _poi_zone(sid=1, cycle_id=1, sd=-1, top=0.58118, bottom=0.58068,
                    cts_established_idx=2, ic_idx=1, pattern_confirmed_idx=2,
                    end_idx=40, activation_history=[])
    assert poi.meta["confirmed_idx"] is None

    triggers = check_zone_proximity(
        df=df, sorted_events=[_cts_confirmed(2, sid=1, cycle_id=1, sd=-1)],
        kl_zones=[bos], poi_zones=[poi],
        pip_size=0.0001, timeframe="H1",
    )
    assert [(t.idx, t.zone_kind, t.trigger_inner) for t in triggers[(1, 1)]] == [(10, "POI", 0.58068)]


def test_poi_unclamped_activation_floor_idx_is_the_floor_without_the_lifecycle_term():
    """PLAN_Q: max(the cycle's CTS-established moment, the IC[, the pattern's
    confirmation]) — H1 cycle (0,1)'s POI: established 652, IC 607, pattern
    confirmed 610 → 652; a ranked IC has no pattern term; a late pattern binds;
    an IC past the establishment binds (S_O's cycle (0,0) POI: established 115,
    IC 180 — the fib's CTS extends until its confirmation). The activation
    history (here empty) is never read."""
    ranked = _poi_zone(sid=0, cycle_id=1, sd=1, top=0.57712, bottom=0.57676,
                       cts_established_idx=652, ic_idx=607, activation_history=[])
    pattern = _poi_zone(sid=0, cycle_id=1, sd=1, top=0.57712, bottom=0.57676,
                        cts_established_idx=652, ic_idx=607, pattern_confirmed_idx=610)
    late = _poi_zone(sid=0, cycle_id=1, sd=1, top=0.57712, bottom=0.57676,
                     cts_established_idx=652, ic_idx=607, pattern_confirmed_idx=700)
    ic_past_est = _poi_zone(sid=0, cycle_id=0, sd=1, top=0.56781, bottom=0.56684,
                            cts_established_idx=115, ic_idx=180)
    assert poi_unclamped_activation_floor_idx(ranked) == 652
    assert poi_unclamped_activation_floor_idx(pattern) == 652
    assert poi_unclamped_activation_floor_idx(late) == 700
    assert poi_unclamped_activation_floor_idx(ic_past_est) == 180


@pytest.mark.parametrize("end_idx, raises", [(10, True), (19, True), (20, False), (None, False)])
def test_poi_end_inside_the_scan_raises(end_idx, raises):
    """PLAN_Q §3: a POI's end is not read (like a KL zone's) because it is always
    past the scan's last candle (19 here: 20 candles, no next BOS) — on real
    data exactly the next candle, e.g. the S_P (0,1) POI ends at 902, its scan
    at 901 — so one inside the scan, the last candle included, fails loudly."""
    df = _make_df(20, default_h=1.50, default_l=1.49)
    bos = _bos_zone(sid=0, cycle_id=0, sd=1, inner=1.40, outer=1.39)
    poi = _poi_zone(sid=0, cycle_id=0, sd=1, top=1.45, bottom=1.44,
                    cts_established_idx=2, end_idx=end_idx)
    kwargs = dict(df=df, sorted_events=[_cts_confirmed(2)], kl_zones=[bos], poi_zones=[poi],
                  pip_size=0.0001, timeframe="H1")
    if raises:
        with pytest.raises(AssertionError, match="inside the scan"):
            check_zone_proximity(**kwargs)
    else:
        assert check_zone_proximity(**kwargs) == {}


def test_threshold_default_lookup_by_timeframe():
    """H1 default is 8 pips; M15 is 5 pips."""
    df = _make_df(20, default_h=1.50, default_l=1.49)
    df.at[5, "l"] = 1.4006  # 6 pips from BOS inner 1.40 — within H1 (8p) but outside M15 (5p)
    bos = _bos_zone(sid=0, cycle_id=0, sd=1, inner=1.40, outer=1.39)

    h1_triggers = check_zone_proximity(
        df=df, sorted_events=[_cts_confirmed(2)],
        kl_zones=[bos], poi_zones=[],
        pip_size=0.0001, timeframe="H1",
    )
    m15_triggers = check_zone_proximity(
        df=df, sorted_events=[_cts_confirmed(2)],
        kl_zones=[bos], poi_zones=[],
        pip_size=0.0001, timeframe="M15",
    )
    assert (0, 0) in h1_triggers
    assert (0, 0) not in m15_triggers


def test_proximity_pips_override_takes_precedence():
    """Explicit proximity_pips overrides the timeframe default."""
    df = _make_df(20, default_h=1.50, default_l=1.49)
    df.at[5, "l"] = 1.4030  # 30 pips from BOS inner — outside H1 default but inside override
    bos = _bos_zone(sid=0, cycle_id=0, sd=1, inner=1.40, outer=1.39)

    triggers = check_zone_proximity(
        df=df, sorted_events=[_cts_confirmed(2)],
        kl_zones=[bos], poi_zones=[],
        pip_size=0.0001, timeframe="H1", proximity_pips=50,
    )
    assert triggers[(0, 0)][0].proximity_pips == 50


def test_scan_window_bounded_by_next_bos():
    """Triggers after next BOS_CONFIRMED.confirmed_at are excluded."""
    df = _make_df(30, default_h=1.50, default_l=1.49)
    df.at[15, "l"] = 1.401  # would trigger if scan reached it

    bos = _bos_zone(sid=0, cycle_id=0, sd=1, inner=1.40, outer=1.39)
    next_bos_event = make_bos_confirmed(
        bos_anchor_idx=12, confirmed_at=12, price=1.50, structure_id=0, cycle_id=1,
    )

    triggers = check_zone_proximity(
        df=df, sorted_events=[_cts_confirmed(2), next_bos_event],
        kl_zones=[bos], poi_zones=[],
        pip_size=0.0001, timeframe="H1",
    )
    # scan_end = 12 - 1 = 11, so idx 15 not reached
    assert (0, 0) not in triggers


def test_scan_starts_at_cts_confirmed_candle_itself():
    """Scan starts AT the CTS_CONFIRMED candle (no +1 offset)."""
    df = _make_df(20, default_h=1.50, default_l=1.49)
    df.at[2, "l"] = 1.4005  # the CTS_CONFIRMED candle itself

    bos = _bos_zone(sid=0, cycle_id=0, sd=1, inner=1.40, outer=1.39)

    triggers = check_zone_proximity(
        df=df, sorted_events=[_cts_confirmed(2)],
        kl_zones=[bos], poi_zones=[],
        pip_size=0.0001, timeframe="H1",
    )
    assert (0, 0) in triggers
    assert triggers[(0, 0)][0].idx == 2  # CTS_CONFIRMED candle itself triggered


def test_no_trigger_when_bos_zone_missing():
    df = _make_df(20)
    df.at[5, "l"] = 1.401
    triggers = check_zone_proximity(
        df=df, sorted_events=[_cts_confirmed(2)],
        kl_zones=[], poi_zones=[],
        pip_size=0.0001, timeframe="H1",
    )
    assert triggers == {}


def test_default_proximity_pips_constants():
    assert DEFAULT_PROXIMITY_PIPS["H1"] == 8
    assert DEFAULT_PROXIMITY_PIPS["M15"] == 5
    assert DEFAULT_PROXIMITY_PIPS["M5"] == 3


def test_default_min_gap_pips_constants():
    assert DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS["H1"] == 50
    assert DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS["M15"] == 30
    assert DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS["M5"] == 15


def test_narrow_cycle_caps_at_one_sd_and_one_opp_sd():
    """Narrow gap + pullback CTS_confirmed: at most 1 sd + 1 opp_sd allowed
    per cycle (Rule 3). A third candle that would otherwise trigger sd is
    rejected."""
    # Tight price band sits between BOS inner 1.4000 + 8p and CTS inner
    # 1.4020 - 8p — so default candles don't trigger; only the explicit
    # overrides do.
    df = _make_df(30, default_h=1.4010, default_l=1.4010)
    df.at[5, "l"] = 1.4005
    df.at[10, "h"] = 1.4015
    df.at[15, "l"] = 1.4004

    bos = _bos_zone(sid=0, cycle_id=0, sd=1, inner=1.40, outer=1.39)
    cts = _cts_zone(sid=0, cycle_id=0, sd=1, inner=1.4020, outer=1.4030)

    # Narrow gap: CTS=1.4020, BOS=1.40 → 20 pips < 50p H1 threshold.
    # Event timeline: BOS_CONFIRMED price=1.40 → CTS_CONFIRMED price=1.4020.
    events = [
        _bos_confirmed(idx=0, price=1.40, sid=0, cycle_id=0, sd=1),
        _cts_confirmed(2, sid=0, cycle_id=0, sd=1),  # price=1.50 in helper but we override below
    ]
    # _cts_confirmed default price is 1.50; override to CTS=1.4020 to get a 20p gap.
    events[1] = StructureEvent(
        idx=2, category="STRUCTURE", type="CTS_CONFIRMED", price=1.4020,
        meta={"structure_id": 0, "cycle_id": 0, "struct_direction": 1,
              "confirmed_at": 2, "confirmation_method": "pullback"},
    )

    triggers = check_zone_proximity(
        df=df, sorted_events=events,
        kl_zones=[bos, cts], poi_zones=[],
        pip_size=0.0001, timeframe="H1",
    )
    # Only the first sd + first opp_sd should fire; the third sd attempt
    # is capped by Rule 3.
    assert len(triggers[(0, 0)]) == 2
    assert triggers[(0, 0)][0].direction == "sd" and triggers[(0, 0)][0].idx == 5
    assert triggers[(0, 0)][1].direction == "opp_sd" and triggers[(0, 0)][1].idx == 10


def test_narrow_cycle_without_pullback_cts_no_triggers():
    """Narrow gap + CTS_CONFIRMED via sd_zone_proximity (NOT pullback):
    Rule 2 blocks the scan entirely — no triggers."""
    df = _make_df(20, default_h=1.4010, default_l=1.4010)
    df.at[5, "l"] = 1.4005
    df.at[10, "h"] = 1.4015

    bos = _bos_zone(sid=0, cycle_id=0, sd=1, inner=1.40, outer=1.39)
    cts = _cts_zone(sid=0, cycle_id=0, sd=1, inner=1.4020, outer=1.4030)

    events = [
        _bos_confirmed(idx=0, price=1.40, sid=0, cycle_id=0, sd=1),
        StructureEvent(
            idx=2, category="STRUCTURE", type="CTS_CONFIRMED", price=1.4020,
            meta={"structure_id": 0, "cycle_id": 0, "struct_direction": 1,
                  "confirmed_at": 2, "confirmation_method": "sd_zone_proximity"},
        ),
    ]
    triggers = check_zone_proximity(
        df=df, sorted_events=events,
        kl_zones=[bos, cts], poi_zones=[],
        pip_size=0.0001, timeframe="H1",
    )
    # Rule 2: narrow cycle requires pullback CTS_confirmed → no triggers.
    assert (0, 0) not in triggers


def test_wide_cycle_unlimited_alternation():
    """Wide gap (≥ 50p H1): Rules 2/3 don't apply; default alternation."""
    # Mid-band default — no default candle reaches either zone's threshold.
    df = _make_df(30, default_h=1.4500, default_l=1.4500)
    df.at[5, "l"] = 1.4005
    df.at[10, "h"] = 1.4995  # opp_sd within 8p of CTS inner 1.50
    df.at[15, "l"] = 1.4004

    bos = _bos_zone(sid=0, cycle_id=0, sd=1, inner=1.40, outer=1.39)
    cts = _cts_zone(sid=0, cycle_id=0, sd=1, inner=1.50, outer=1.51)

    # Wide gap: CTS=1.50, BOS=1.40 → 1000 pips >> 50p
    events = [
        _bos_confirmed(idx=0, price=1.40, sid=0, cycle_id=0, sd=1),
        _cts_confirmed(2, sid=0, cycle_id=0, sd=1),  # price=1.50 default
    ]
    triggers = check_zone_proximity(
        df=df, sorted_events=events,
        kl_zones=[bos, cts], poi_zones=[],
        pip_size=0.0001, timeframe="H1",
    )
    # All 3 triggers fire — no caps in wide mode.
    assert len(triggers[(0, 0)]) == 3


def test_mid_cycle_crossing_lifts_caps():
    """Cycle starts narrow, crosses to wide mid-scan via a
    CTS_THRESHOLD_UPDATED event: caps lift after crossing, alternation
    continues seamlessly."""
    # Tight band — high under CTS zone inner 1.4020 (minus 8p), low above
    # BOS inner 1.4000 (plus 8p). Default candles don't trigger.
    df = _make_df(40, default_h=1.4010, default_l=1.4010)
    df.at[5, "l"] = 1.4005   # sd, idx 5 (narrow)
    df.at[10, "h"] = 1.4015  # opp_sd, idx 10 (narrow)
    # Post-crossing triggers — should fire (caps don't apply):
    df.at[20, "l"] = 1.4004  # sd, idx 20 (wide)
    df.at[25, "h"] = 1.4055  # opp_sd, idx 25 (wide; near new CTS inner 1.4060)
    df.at[30, "l"] = 1.4003  # sd, idx 30 (wide)

    bos = _bos_zone(sid=0, cycle_id=0, sd=1, inner=1.40, outer=1.39)
    # CTS zone object inner=1.4020 — the opp_sd proximity reference. Both
    # pre-crossing (idx 10 high 1.4015 within 8p of 1.4020) and post-crossing
    # (idx 25 high 1.4055 ≥ 1.4020 - 8p trivially) opp_sd candidates reach it.
    cts = _cts_zone(sid=0, cycle_id=0, sd=1, inner=1.4020, outer=1.4030)

    # Event timeline: narrow at scan_start (gap=20p), then a
    # CTS_THRESHOLD_UPDATED at idx 13 widens cts_threshold from 1.4020 to
    # 1.4060 (gap → 60p, wide). After the crossing, caps lift.
    events = [
        _bos_confirmed(idx=0, price=1.40, sid=0, cycle_id=0, sd=1),
        StructureEvent(
            idx=2, category="STRUCTURE", type="CTS_CONFIRMED", price=1.4020,
            meta={"structure_id": 0, "cycle_id": 0, "struct_direction": 1,
                  "confirmed_at": 2, "confirmation_method": "pullback"},
        ),
        _cts_thresh_updated(idx=13, price=1.4060, prev=1.4020, sid=0, cycle_id=0),
    ]
    triggers = check_zone_proximity(
        df=df, sorted_events=events,
        kl_zones=[bos, cts], poi_zones=[],
        pip_size=0.0001, timeframe="H1",
        min_proximity_reference_gap_pips=_NO_REF_CAP,
    )
    # Expect 5 triggers total: 1 sd narrow (idx 5) + 1 opp_sd narrow
    # (idx 10), then wide-mode alternation continues at idx 20/25/30
    # (sd, opp_sd, sd).
    assert len(triggers[(0, 0)]) == 5
    assert [t.direction for t in triggers[(0, 0)]] == [
        "sd", "opp_sd", "sd", "opp_sd", "sd"
    ]
    assert [t.idx for t in triggers[(0, 0)]] == [5, 10, 20, 25, 30]
    # Both rule sets AND-ed (PLAN_R §9): the BOS–CTS inners sit 20 pips apart
    # (inners never move) — the narrow parent-reference exception under the
    # default minimum (40), so after Rules 2 / 3 lift the exception still allows
    # one sd + one opp_sd in the whole cycle.
    with_cap = check_zone_proximity(
        df=df, sorted_events=events,
        kl_zones=[bos, cts], poi_zones=[],
        pip_size=0.0001, timeframe="H1",
    )
    assert [t.idx for t in with_cap[(0, 0)]] == [5, 10]
    assert all(t.meta["prox_ref_gap_pips"] == 20 and t.meta["narrow_prox_ref"] for t in with_cap[(0, 0)])


def test_missing_bos_confirmed_falls_back_to_default():
    """When BOS_CONFIRMED event is absent (events incomplete), the
    narrow-mode check can't evaluate gap → falls back to default
    alternation. Preserves prior behavior for callers that don't pass
    threshold-defining events."""
    df = _make_df(30, default_h=1.4010, default_l=1.4010)
    df.at[5, "l"] = 1.4005
    df.at[10, "h"] = 1.4015
    df.at[15, "l"] = 1.4004

    bos = _bos_zone(sid=0, cycle_id=0, sd=1, inner=1.40, outer=1.39)
    cts = _cts_zone(sid=0, cycle_id=0, sd=1, inner=1.4020, outer=1.4030)

    # Only CTS_CONFIRMED — no BOS_CONFIRMED. Gap is undefined → default mode.
    events = [_cts_confirmed(2, sid=0, cycle_id=0, sd=1)]
    triggers = check_zone_proximity(
        df=df, sorted_events=events,
        kl_zones=[bos, cts], poi_zones=[],
        pip_size=0.0001, timeframe="H1",
        min_proximity_reference_gap_pips=_NO_REF_CAP,
    )
    # Default mode → all 3 triggers fire.
    assert len(triggers[(0, 0)]) == 3


def test_bos_threshold_update_alone_can_cross_into_wide():
    """A BOS_THRESHOLD_UPDATED event mid-cycle can also widen gap past
    min_gap_pips (BOS_n monotonically extends away from CTS via probe).
    Verifies the gap-source treats BOS and CTS updates symmetrically."""
    df = _make_df(40, default_h=1.4010, default_l=1.4010)
    df.at[5, "l"] = 1.4005   # sd narrow
    df.at[10, "h"] = 1.4015  # opp_sd narrow
    # After BOS extends down at idx 13 → gap widens → caps lift
    df.at[20, "l"] = 1.4004  # sd wide

    bos = _bos_zone(sid=0, cycle_id=0, sd=1, inner=1.40, outer=1.39)
    cts = _cts_zone(sid=0, cycle_id=0, sd=1, inner=1.4020, outer=1.4030)

    events = [
        _bos_confirmed(idx=0, price=1.40, sid=0, cycle_id=0, sd=1),
        StructureEvent(
            idx=2, category="STRUCTURE", type="CTS_CONFIRMED", price=1.4020,
            meta={"structure_id": 0, "cycle_id": 0, "struct_direction": 1,
                  "confirmed_at": 2, "confirmation_method": "pullback"},
        ),
        # BOS extends from 1.40 to 1.395 → gap = |1.4020 - 1.395| = 70p (wide).
        _bos_thresh_updated(idx=13, price=1.395, prev=1.40, sid=0, cycle_id=0),
    ]
    triggers = check_zone_proximity(
        df=df, sorted_events=events,
        kl_zones=[bos, cts], poi_zones=[],
        pip_size=0.0001, timeframe="H1",
        min_proximity_reference_gap_pips=_NO_REF_CAP,
    )
    assert len(triggers[(0, 0)]) == 3
    assert [t.direction for t in triggers[(0, 0)]] == ["sd", "opp_sd", "sd"]
    assert [t.idx for t in triggers[(0, 0)]] == [5, 10, 20]


def test_sell_direction_trigger():
    """sd=-1 (bearish): inverse logic — sd zone is sell-side, opp_sd is buy-side (CTS)."""
    df = _make_df(30, default_h=1.50, default_l=1.49)
    # BOS inner 1.55 (top of sell zone); approach from below — high 1.5495 within 8p
    df.at[5, "h"] = 1.5495
    bos = _bos_zone(sid=0, cycle_id=0, sd=-1, inner=1.55, outer=1.56)

    triggers = check_zone_proximity(
        df=df, sorted_events=[_cts_confirmed(2, sd=-1)],
        kl_zones=[bos], poi_zones=[],
        pip_size=0.0001, timeframe="H1",
    )
    assert (0, 0) in triggers
    assert triggers[(0, 0)][0].direction == "sd"
    assert triggers[(0, 0)][0].idx == 5


def test_threshold_timeline_is_ordered_by_moment():
    """Plan E E3g-2: the threshold timeline sorts on (moment, type). A BOS_CONFIRMED
    anchored at 5 is known at 12 → after a threshold event at 10 (on the stamped
    idx it would sort first)."""
    from engine_v2.zones.zone_proximity import _build_cycle_threshold_timeline
    bos = make_bos_confirmed(bos_anchor_idx=5, confirmed_at=12, structure_id=0, cycle_id=1, price=0.60)
    thr = StructureEvent(idx=10, category="STRUCTURE", type="CTS_THRESHOLD_UPDATED", price=0.61,
                         meta={"structure_id": 0, "cycle_id": 1})
    tl = _build_cycle_threshold_timeline([bos, thr], 0, 1)
    assert [e.type for e in tl] == ["CTS_THRESHOLD_UPDATED", "BOS_CONFIRMED"]


def test_initial_pass_applies_events_known_at_scan_start():
    """(E3f/E3g landing review.) The scan_start candle sees the thresholds known AT
    scan_start (inclusive initial pass): an sd touch ON the CTS_CONFIRMED candle of
    a narrow cycle already counts against the Rule-3 cap, so a later sd touch is capped."""
    df = _make_df(30, default_h=1.4010, default_l=1.4010)
    df.at[2, "l"] = 1.4005    # sd touch on scan_start itself
    df.at[10, "h"] = 1.4015   # opp_sd
    df.at[15, "l"] = 1.4004   # second sd -> capped only if 2 was narrow
    bos = _bos_zone(sid=0, cycle_id=0, sd=1, inner=1.40, outer=1.39)
    cts = _cts_zone(sid=0, cycle_id=0, sd=1, inner=1.4020, outer=1.4030)
    events = [
        _bos_confirmed(idx=0, price=1.40, sid=0, cycle_id=0, sd=1, confirmed_at=2),
        StructureEvent(idx=2, category="STRUCTURE", type="CTS_CONFIRMED", price=1.4020,
                       meta={"structure_id": 0, "cycle_id": 0, "struct_direction": 1,
                             "confirmed_at": 2, "confirmation_method": "pullback"}),
    ]
    tr = check_zone_proximity(df=df, sorted_events=events, kl_zones=[bos, cts],
                              poi_zones=[], pip_size=0.0001, timeframe="H1")[(0, 0)]
    assert [(t.direction, t.idx) for t in tr] == [("sd", 2), ("opp_sd", 10)]


# --- E4b landing review: the scan window's next-BOS bound is a TIME -------------------

def test_scan_window_ends_before_the_next_bos_moment_not_its_anchor():
    """(Plan E E4b landing review, mutation lens.) The scan window ends at the next
    cycle's BOS_CONFIRMED MOMENT - 1 (`confirmed_at` 16 -> 15), not at its anchor
    - 1 (12 -> 11): a touch at 14 -- after the next BOS's anchor but before that
    BOS is known -- still fires."""
    df = _make_df(30, default_h=1.50, default_l=1.49)
    df.at[14, "l"] = 1.4005   # 5 pips above the BOS inner 1.40
    bos = _bos_zone(sid=0, cycle_id=0, sd=1, inner=1.40, outer=1.39)
    next_bos = make_bos_confirmed(bos_anchor_idx=12, confirmed_at=16, price=1.50,
                                  structure_id=0, cycle_id=1)
    tr = check_zone_proximity(df=df, sorted_events=[_cts_confirmed(2), next_bos],
                              kl_zones=[bos], poi_zones=[], pip_size=0.0001, timeframe="H1")
    assert [(t.direction, t.idx) for t in tr[(0, 0)]] == [("sd", 14)]
