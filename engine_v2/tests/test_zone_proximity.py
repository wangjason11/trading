"""Unit tests for zones/zone_proximity.py — zone proximity trigger detection."""
from __future__ import annotations

import pandas as pd
import pytest

from engine_v2.common.types import KLZone
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.zones.poi_zones import POIZone
from engine_v2.zones.zone_proximity import (
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


def _cts_confirmed(idx: int, sid: int = 0, cycle_id: int = 0, sd: int = 1) -> StructureEvent:
    return StructureEvent(
        idx=idx, category="STRUCTURE", type="CTS_CONFIRMED",
        price=1.50,
        meta={"structure_id": sid, "cycle_id": cycle_id,
              "struct_direction": sd, "confirmed_at": idx},
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
              confirmed_idx: int, end_idx=None) -> POIZone:
    side = "buy" if sd == 1 else "sell"
    return POIZone(
        start_time=pd.Timestamp("2026-01-01", tz="UTC"),
        end_time=None, side=side, top=top, bottom=bottom,
        ic_idx=confirmed_idx - 1,
        meta={
            "structure_id": sid, "cycle_id": cycle_id,
            "confirmed_idx": confirmed_idx, "end_idx": end_idx,
            "versions": ["V60"],
        },
    )


# ---------- Tests ----------

def test_first_sd_trigger_fires_on_buy_zone():
    """sd=+1: BOS inner=1.40, threshold 20 pips (0.0020). Candle low 1.401 → triggers."""
    df = _make_df(20, default_h=1.50, default_l=1.49)
    df.at[5, "l"] = 1.401  # within 20 pips of inner 1.40
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
    # sd trigger at idx 5: low 1.401 (within 20p of BOS inner 1.40)
    df.at[5, "l"] = 1.401
    # opp_sd trigger at idx 10: high 1.598 (within 20p of CTS inner 1.60)
    df.at[10, "h"] = 1.598
    # second sd trigger at idx 15: low 1.402
    df.at[15, "l"] = 1.402

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
    # Candle low 1.451 → within 20 pips of POI inner (1.45) but not BOS (1.40+0.0020=1.402 too low).
    df.at[5, "l"] = 1.451
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


def test_threshold_default_lookup_by_timeframe():
    """M15 default is 10 pips; H1 is 20 pips."""
    df = _make_df(20, default_h=1.50, default_l=1.49)
    df.at[5, "l"] = 1.4015  # 15 pips from BOS inner 1.40 — within H1 (20p) but outside M15 (10p)
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
    df.at[2, "l"] = 1.401  # the CTS_CONFIRMED candle itself

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
    assert DEFAULT_PROXIMITY_PIPS["H1"] == 20
    assert DEFAULT_PROXIMITY_PIPS["M15"] == 10
    assert DEFAULT_PROXIMITY_PIPS["M5"] == 5


def test_sell_direction_trigger():
    """sd=-1 (bearish): inverse logic — sd zone is sell-side, opp_sd is buy-side (CTS)."""
    df = _make_df(30, default_h=1.50, default_l=1.49)
    # BOS inner 1.55 (top of sell zone); approach from below — high 1.5485 within 20p
    df.at[5, "h"] = 1.5485
    bos = _bos_zone(sid=0, cycle_id=0, sd=-1, inner=1.55, outer=1.56)

    triggers = check_zone_proximity(
        df=df, sorted_events=[_cts_confirmed(2, sd=-1)],
        kl_zones=[bos], poi_zones=[],
        pip_size=0.0001, timeframe="H1",
    )
    assert (0, 0) in triggers
    assert triggers[(0, 0)][0].direction == "sd"
    assert triggers[(0, 0)][0].idx == 5
