"""Unit tests for zones/zone_proximity.py — zone proximity trigger detection."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import pytest

from engine_v2.common.types import KLZone
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.zones.poi_zones import POIZone
from engine_v2.zones.zone_proximity import (
    DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS,
    DEFAULT_PROXIMITY_PIPS,
    ZoneProximityTrigger,
    check_zone_proximity,
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
    return StructureEvent(
        idx=idx, category="STRUCTURE", type="BOS_CONFIRMED",
        price=price,
        meta={"structure_id": sid, "cycle_id": cycle_id,
              "struct_direction": sd,
              "confirmed_at": confirmed_at if confirmed_at is not None else idx},
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


def _poi_zone(sid: int, cycle_id: int, sd: int, top: float, bottom: float,
              confirmed_idx: int, end_idx=None, activation_history=None) -> POIZone:
    """Build a POI zone. The proximity gate reads `activation_history` (not
    the scalar `confirmed_idx`); a single-span POI defaults to one activate
    at `confirmed_idx`. Pass `activation_history` explicitly to model a POI
    that flaps active/inactive within its cycle."""
    side = "buy" if sd == 1 else "sell"
    if activation_history is None:
        activation_history = [{"idx": confirmed_idx, "active": True, "reason": "initial"}]
    return POIZone(
        start_time=pd.Timestamp("2026-01-01", tz="UTC"),
        end_time=None, side=side, top=top, bottom=bottom,
        ic_idx=confirmed_idx - 1,
        meta={
            "structure_id": sid, "cycle_id": cycle_id,
            "confirmed_idx": confirmed_idx, "end_idx": end_idx,
            "versions": ["V60"],
            "activation_history": activation_history,
        },
    )


# ---------- Tests ----------

def test_first_sd_trigger_fires_on_buy_zone():
    """sd=+1: BOS inner=1.40, H1 threshold 9 pips (0.0009). Candle low 1.4005 → triggers."""
    df = _make_df(20, default_h=1.50, default_l=1.49)
    df.at[5, "l"] = 1.4005  # 5 pips above inner 1.40, within 9p H1 threshold
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
    # sd trigger at idx 5: low 1.4005 (within 9p of BOS inner 1.40)
    df.at[5, "l"] = 1.4005
    # opp_sd trigger at idx 10: high 1.5995 (within 9p of CTS inner 1.60)
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


def test_poi_included_in_sd_inners_when_active():
    """POI zone closer to current price than BOS should win as trigger_inner."""
    df = _make_df(30, default_h=1.50, default_l=1.49)
    # BOS inner 1.40, POI top 1.45 (closer to current price for sd=+1).
    # Candle low 1.4505 → within 9 pips of POI inner (1.45) but not BOS (1.40 + 0.0009 = 1.4009 too low).
    df.at[5, "l"] = 1.4505
    bos = _bos_zone(sid=0, cycle_id=0, sd=1, inner=1.40, outer=1.39)
    poi = _poi_zone(sid=0, cycle_id=0, sd=1, top=1.45, bottom=1.44, confirmed_idx=3)

    triggers = check_zone_proximity(
        df=df, sorted_events=[_cts_confirmed(2)],
        kl_zones=[bos], poi_zones=[poi],
        pip_size=0.0001, timeframe="H1",
    )
    assert triggers[(0, 0)][0].zone_kind == "POI"
    assert triggers[(0, 0)][0].trigger_inner == 1.45


def test_poi_not_active_yet_excluded():
    """POI confirmed_idx > current candle → POI not included in inners."""
    df = _make_df(30, default_h=1.50, default_l=1.49)
    df.at[5, "l"] = 1.451  # close to POI inner if active
    bos = _bos_zone(sid=0, cycle_id=0, sd=1, inner=1.40, outer=1.39)
    poi = _poi_zone(sid=0, cycle_id=0, sd=1, top=1.45, bottom=1.44, confirmed_idx=10)

    triggers = check_zone_proximity(
        df=df, sorted_events=[_cts_confirmed(2)],
        kl_zones=[bos], poi_zones=[poi],
        pip_size=0.0001, timeframe="H1",
    )
    # Should NOT trigger off POI; should only consider BOS (1.40 + 0.002 = 1.402, low 1.451 too high)
    # → no trigger
    assert (0, 0) not in triggers


def test_poi_fires_in_first_active_stretch_not_last_activate():
    """Regression (sid1 cyc2 / idx 926): a POI that flaps active->inactive->
    active has its scalar `confirmed_idx` collapsed to the LAST activate. The
    gate must use the per-candle activation history, so a candle inside the
    FIRST active stretch still triggers — even though it is far before the
    scalar confirmed_idx."""
    df = _make_df(30, default_h=1.50, default_l=1.49)
    df.at[6, "l"] = 1.4505  # within 9p of POI inner 1.45; far from BOS inner 1.40
    bos = _bos_zone(sid=0, cycle_id=0, sd=1, inner=1.40, outer=1.39)
    # Active [5,9], inactive [10,19], active [20,...]. Scalar confirmed_idx=20.
    poi = _poi_zone(
        sid=0, cycle_id=0, sd=1, top=1.45, bottom=1.44, confirmed_idx=20,
        activation_history=[
            {"idx": 5, "active": True}, {"idx": 10, "active": False},
            {"idx": 20, "active": True},
        ],
    )
    triggers = check_zone_proximity(
        df=df, sorted_events=[_cts_confirmed(2)],
        kl_zones=[bos], poi_zones=[poi],
        pip_size=0.0001, timeframe="H1",
    )
    # Old code (scalar confirmed_idx=20) would exclude the POI at idx 6 and
    # fire nothing; the per-candle gate fires sd:POI at 6.
    assert (0, 0) in triggers
    assert triggers[(0, 0)][0].direction == "sd"
    assert triggers[(0, 0)][0].idx == 6
    assert triggers[(0, 0)][0].zone_kind == "POI"


def test_poi_suppressed_during_inactive_stretch():
    """A candle inside an INACTIVE stretch must not trigger off the POI, even
    though the scalar confirmed_idx (first/only activate) precedes it."""
    df = _make_df(30, default_h=1.50, default_l=1.49)
    df.at[12, "l"] = 1.4504  # would hit POI inner 1.45 IF active; BOS too far
    bos = _bos_zone(sid=0, cycle_id=0, sd=1, inner=1.40, outer=1.39)
    # Active [5,9] then inactive from 10 onward (no reactivation).
    poi = _poi_zone(
        sid=0, cycle_id=0, sd=1, top=1.45, bottom=1.44, confirmed_idx=5,
        activation_history=[
            {"idx": 5, "active": True}, {"idx": 10, "active": False},
        ],
    )
    triggers = check_zone_proximity(
        df=df, sorted_events=[_cts_confirmed(2)],
        kl_zones=[bos], poi_zones=[poi],
        pip_size=0.0001, timeframe="H1",
    )
    assert (0, 0) not in triggers


def test_threshold_default_lookup_by_timeframe():
    """H1 default is 9 pips; M15 is 6 pips."""
    df = _make_df(20, default_h=1.50, default_l=1.49)
    df.at[5, "l"] = 1.4008  # 8 pips from BOS inner 1.40 — within H1 (9p) but outside M15 (6p)
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
    next_bos_event = StructureEvent(
        idx=12, category="STRUCTURE", type="BOS_CONFIRMED",
        price=1.50,
        meta={"structure_id": 0, "cycle_id": 1, "confirmed_at": 12,
              "struct_direction": 1},
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
    assert DEFAULT_PROXIMITY_PIPS["H1"] == 9
    assert DEFAULT_PROXIMITY_PIPS["M15"] == 6
    assert DEFAULT_PROXIMITY_PIPS["M5"] == 3


def test_default_min_gap_pips_constants():
    assert DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS["H1"] == 50
    assert DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS["M15"] == 30
    assert DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS["M5"] == 15


def test_narrow_cycle_caps_at_one_sd_and_one_opp_sd():
    """Narrow gap + pullback CTS_confirmed: at most 1 sd + 1 opp_sd allowed
    per cycle (Rule 3). A third candle that would otherwise trigger sd is
    rejected."""
    # Tight price band sits between BOS inner 1.4000 + 9p and CTS inner
    # 1.4020 - 9p — so default candles don't trigger; only the explicit
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
    df.at[10, "h"] = 1.4995  # opp_sd within 9p of CTS inner 1.50
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
    # Tight band — high under CTS zone inner 1.4020 (minus 9p), low above
    # BOS inner 1.4000 (plus 9p). Default candles don't trigger.
    df = _make_df(40, default_h=1.4010, default_l=1.4010)
    df.at[5, "l"] = 1.4005   # sd, idx 5 (narrow)
    df.at[10, "h"] = 1.4015  # opp_sd, idx 10 (narrow)
    # Post-crossing triggers — should fire (caps don't apply):
    df.at[20, "l"] = 1.4004  # sd, idx 20 (wide)
    df.at[25, "h"] = 1.4055  # opp_sd, idx 25 (wide; near new CTS inner 1.4060)
    df.at[30, "l"] = 1.4003  # sd, idx 30 (wide)

    bos = _bos_zone(sid=0, cycle_id=0, sd=1, inner=1.40, outer=1.39)
    # CTS zone object inner=1.4020 — used as opp_sd trigger level by
    # _try_trigger_at_candle. Both pre-crossing (idx 10 high 1.4015 within
    # 9p of 1.4020) and post-crossing (idx 25 high 1.4055 ≥ 1.4020 - 9p
    # trivially) opp_sd candidates reach it.
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
    )
    # Expect 5 triggers total: 1 sd narrow (idx 5) + 1 opp_sd narrow
    # (idx 10), then wide-mode alternation continues at idx 20/25/30
    # (sd, opp_sd, sd).
    assert len(triggers[(0, 0)]) == 5
    assert [t.direction for t in triggers[(0, 0)]] == [
        "sd", "opp_sd", "sd", "opp_sd", "sd"
    ]
    assert [t.idx for t in triggers[(0, 0)]] == [5, 10, 20, 25, 30]


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
    )
    assert len(triggers[(0, 0)]) == 3
    assert [t.direction for t in triggers[(0, 0)]] == ["sd", "opp_sd", "sd"]
    assert [t.idx for t in triggers[(0, 0)]] == [5, 10, 20]


def test_sell_direction_trigger():
    """sd=-1 (bearish): inverse logic — sd zone is sell-side, opp_sd is buy-side (CTS)."""
    df = _make_df(30, default_h=1.50, default_l=1.49)
    # BOS inner 1.55 (top of sell zone); approach from below — high 1.5495 within 9p
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
