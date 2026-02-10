"""Unit tests for engine_v2.zones.wave_candles."""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from engine_v2.common.types import KLZone
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.zones.wave_candles import (
    WaveCandleResult,
    _is_qualified,
    _touches_zone,
    _closes_within_zone,
    _close_distance_to_outer,
    _is_compound_first_wave,
    _find_cts_established,
    _find_cts_updated_events,
    _find_cts_confirmed,
    _find_last_pullback_idx,
    _find_prior_cts_anchor,
    identify_wave_candles,
)


# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------

def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Create a minimal OHLC DataFrame with candle features from a list of row dicts."""
    defaults = {
        "time": "2024-01-01",
        "o": 1.0,
        "h": 1.1,
        "l": 0.9,
        "c": 1.05,
        "volume": 100,
        "direction": 1,
        "candle_type": "normal",
        "pinbar_dir": 0,
        "is_big_normal_as0": 0,
        "is_big_maru_as0": 0,
    }
    data = []
    for i, row in enumerate(rows):
        r = {**defaults, **row}
        r.setdefault("time", f"2024-01-{i+1:02d}")
        data.append(r)
    return pd.DataFrame(data)


def _make_zone(side: str, top: float, bottom: float, source_kind: str = "BOS",
               meta: dict | None = None) -> KLZone:
    m = {
        "structure_id": 0,
        "cycle_id": 1,
        "anchor_idx": 5,
        "base_pattern": "base",
        "outer": bottom if side == "buy" else top,
        **(meta or {}),
    }
    return KLZone(
        start_time=pd.Timestamp("2024-01-01", tz="UTC"),
        end_time=None,
        side=side,
        top=top,
        bottom=bottom,
        source_kind=source_kind,
        source_time=pd.Timestamp("2024-01-05", tz="UTC"),
        source_price=1.0,
        meta=m,
    )


def _make_event(etype: str, idx: int, price: float = None, meta: dict | None = None) -> StructureEvent:
    return StructureEvent(
        idx=idx,
        category="STRUCTURE" if etype != "STATE_CHANGED" else "STATE",
        type=etype,
        price=price,
        meta=meta or {},
    )


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestIsQualified:
    def test_matching_direction(self):
        df = _make_df([{"direction": 1}])
        assert _is_qualified(df, 0, 1) is True

    def test_non_matching_direction(self):
        df = _make_df([{"direction": -1}])
        assert _is_qualified(df, 0, 1) is False

    def test_out_of_bounds(self):
        df = _make_df([{"direction": 1}])
        assert _is_qualified(df, 99, 1) is False


class TestTouchesZone:
    def test_buy_zone_low_touches(self):
        df = _make_df([{"l": 0.95}])
        zone = _make_zone("buy", top=1.0, bottom=0.9)
        assert _touches_zone(df, 0, zone) is True

    def test_buy_zone_low_does_not_touch(self):
        df = _make_df([{"l": 1.05}])
        zone = _make_zone("buy", top=1.0, bottom=0.9)
        assert _touches_zone(df, 0, zone) is False

    def test_sell_zone_high_touches(self):
        df = _make_df([{"h": 1.05}])
        zone = _make_zone("sell", top=1.1, bottom=1.0)
        assert _touches_zone(df, 0, zone) is True

    def test_sell_zone_high_does_not_touch(self):
        df = _make_df([{"h": 0.95}])
        zone = _make_zone("sell", top=1.1, bottom=1.0)
        assert _touches_zone(df, 0, zone) is False


class TestClosesWithinZone:
    def test_within(self):
        df = _make_df([{"c": 0.95}])
        zone = _make_zone("buy", top=1.0, bottom=0.9)
        assert _closes_within_zone(df, 0, zone) is True

    def test_below(self):
        df = _make_df([{"c": 0.85}])
        zone = _make_zone("buy", top=1.0, bottom=0.9)
        assert _closes_within_zone(df, 0, zone) is False

    def test_above(self):
        df = _make_df([{"c": 1.05}])
        zone = _make_zone("buy", top=1.0, bottom=0.9)
        assert _closes_within_zone(df, 0, zone) is False

    def test_on_boundary(self):
        df = _make_df([{"c": 1.0}])
        zone = _make_zone("buy", top=1.0, bottom=0.9)
        assert _closes_within_zone(df, 0, zone) is True


class TestCloseDistanceToOuter:
    def test_buy_zone(self):
        df = _make_df([{"c": 0.95}])
        zone = _make_zone("buy", top=1.0, bottom=0.9, meta={"outer": 0.9})
        assert _close_distance_to_outer(df, 0, zone) == pytest.approx(0.05)

    def test_no_outer_meta(self):
        df = _make_df([{"c": 0.95}])
        zone = _make_zone("buy", top=1.0, bottom=0.9, meta={"outer": None})
        # When outer is None, replaced via meta override
        # _make_zone sets outer = bottom for buy, so let's explicitly set it None
        z = KLZone(
            start_time=pd.Timestamp("2024-01-01", tz="UTC"),
            end_time=None, side="buy", top=1.0, bottom=0.9,
            source_kind="BOS", source_time=pd.Timestamp("2024-01-05", tz="UTC"),
            source_price=1.0, meta={"structure_id": 0, "cycle_id": 1, "anchor_idx": 5},
        )
        assert _close_distance_to_outer(df, 0, z) == float("inf")


class TestIsCompoundFirstWave:
    def test_big_normal_maru(self):
        df = _make_df([{"is_big_normal_as0": 1, "candle_type": "maru"}])
        assert _is_compound_first_wave(df, 0, 1) is True

    def test_big_normal_normal(self):
        df = _make_df([{"is_big_normal_as0": 1, "candle_type": "normal"}])
        assert _is_compound_first_wave(df, 0, 1) is True

    def test_big_maru_pinbar_matching_dir(self):
        df = _make_df([{"is_big_maru_as0": 1, "candle_type": "pinbar", "pinbar_dir": 1}])
        assert _is_compound_first_wave(df, 0, 1) is True

    def test_big_maru_pinbar_wrong_dir(self):
        df = _make_df([{"is_big_maru_as0": 1, "candle_type": "pinbar", "pinbar_dir": -1}])
        assert _is_compound_first_wave(df, 0, 1) is False

    def test_not_big(self):
        df = _make_df([{"is_big_normal_as0": 0, "candle_type": "normal"}])
        assert _is_compound_first_wave(df, 0, 1) is False

    def test_out_of_bounds(self):
        df = _make_df([{"is_big_normal_as0": 1, "candle_type": "maru"}])
        assert _is_compound_first_wave(df, 99, 1) is False


# ---------------------------------------------------------------------------
# Event query helper tests
# ---------------------------------------------------------------------------

class TestEventHelpers:
    def test_find_cts_established(self):
        events = [
            _make_event("CTS_ESTABLISHED", 10, 1.0, {"structure_id": 0, "cycle_id": 1}),
            _make_event("CTS_ESTABLISHED", 20, 1.1, {"structure_id": 0, "cycle_id": 2}),
        ]
        ev = _find_cts_established(events, 0, 1)
        assert ev is not None
        assert ev.idx == 10

    def test_find_cts_established_not_found(self):
        events = [
            _make_event("CTS_ESTABLISHED", 10, 1.0, {"structure_id": 0, "cycle_id": 1}),
        ]
        assert _find_cts_established(events, 0, 2) is None

    def test_find_cts_updated_events(self):
        events = [
            _make_event("CTS_UPDATED", 12, 1.01, {"structure_id": 0, "cycle_id": 1}),
            _make_event("CTS_UPDATED", 15, 1.02, {"structure_id": 0, "cycle_id": 1}),
            _make_event("CTS_UPDATED", 11, 1.03, {"structure_id": 0, "cycle_id": 2}),
        ]
        result = _find_cts_updated_events(events, 0, 1)
        assert len(result) == 2
        assert result[0].idx == 12
        assert result[1].idx == 15

    def test_find_cts_confirmed(self):
        events = [
            _make_event("CTS_CONFIRMED", 18, meta={"structure_id": 0, "cycle_id": 1, "confirmed_at": 18, "cts_anchor_idx": 10}),
        ]
        ev = _find_cts_confirmed(events, 0, 1)
        assert ev is not None
        assert ev.idx == 18

    def test_find_last_pullback_idx(self):
        events = [
            _make_event("STATE_CHANGED", 5, meta={"structure_id": 0, "to": "pullback"}),
            _make_event("STATE_CHANGED", 8, meta={"structure_id": 0, "to": "pullback"}),
            _make_event("STATE_CHANGED", 12, meta={"structure_id": 0, "to": "breakout"}),
        ]
        assert _find_last_pullback_idx(events, 0, 10) == 8
        assert _find_last_pullback_idx(events, 0, 5) is None

    def test_find_prior_cts_anchor(self):
        events = [
            _make_event("CTS_CONFIRMED", 18, meta={"structure_id": 0, "cycle_id": 0, "cts_anchor_idx": 10}),
        ]
        assert _find_prior_cts_anchor(events, 0, 1) == 10
        assert _find_prior_cts_anchor(events, 0, 0) is None


# ---------------------------------------------------------------------------
# BOS wave candle tests
# ---------------------------------------------------------------------------

class TestBOSCycle0:
    def test_bib_backward_fallback_uses_5_candle_lookback(self):
        """BOS cycle 0 + BIB: backward fallback looks back 5 candles (not event-based)."""
        rows = []
        for i in range(20):
            rows.append({
                "direction": 1, "c": 1.05, "l": 1.03, "h": 1.1,
                "candle_type": "normal", "is_big_normal_as0": 0,
            })
        # Bearish candle at idx=5 (anchor-5) close to outer — within 5-candle lookback
        rows[5] = {
            "direction": -1, "c": 0.91, "l": 0.88, "h": 1.0,
            "candle_type": "normal", "is_big_normal_as0": 0,
        }
        # Big bullish candle at idx=12 for first breakout
        rows[12] = {
            "direction": 1, "c": 1.1, "l": 0.98, "h": 1.12,
            "candle_type": "maru", "is_big_normal_as0": 1,
        }
        df = _make_df(rows)
        zone = _make_zone("buy", top=1.0, bottom=0.9, source_kind="BOS", meta={
            "cycle_id": 0, "anchor_idx": 10, "base_pattern": "base inside bar",
            "outer": 0.9,
        })
        result = identify_wave_candles(
            anchor_idx=10, anchor_type="BOS", zone=zone, events=[],
            structure_id=0, struct_direction=1, df=df,
        )
        assert result is not None
        assert result.cycle_id == 0
        assert result.source_kind == "BOS"
        assert result.last_wave_candle_idx == 5
        assert result.first_wave_candle_idx == 12

    def test_bib_candle_beyond_5_not_found(self):
        """BOS cycle 0 + BIB: candle at anchor-6 is outside 5-candle lookback."""
        rows = []
        for i in range(20):
            rows.append({
                "direction": 1, "c": 1.05, "l": 1.03, "h": 1.1,
                "candle_type": "normal", "is_big_normal_as0": 0,
            })
        # Bearish candle at idx=4 (anchor-6) — outside 5-candle lookback from anchor=10
        rows[4] = {
            "direction": -1, "c": 0.91, "l": 0.88, "h": 1.0,
            "candle_type": "normal", "is_big_normal_as0": 0,
        }
        df = _make_df(rows)
        zone = _make_zone("buy", top=1.0, bottom=0.9, source_kind="BOS", meta={
            "cycle_id": 0, "anchor_idx": 10, "base_pattern": "base inside bar",
            "outer": 0.9,
        })
        result = identify_wave_candles(
            anchor_idx=10, anchor_type="BOS", zone=zone, events=[],
            structure_id=0, struct_direction=1, df=df,
        )
        assert result is not None
        assert result.last_wave_candle_idx is None


class TestBOSNonBIB:
    def test_window_search_finds_correct_candle(self):
        """Non-BIB BOS: window search finds qualified candle touching zone closest to outer."""
        rows = []
        for i in range(15):
            rows.append({
                "direction": 1, "c": 1.05, "l": 1.0, "h": 1.1,
                "candle_type": "normal", "is_big_normal_as0": 0,
            })
        # Place a bearish candle at idx=4 that touches the buy zone and is close to outer
        rows[4] = {
            "direction": -1, "c": 0.91, "l": 0.89, "h": 1.0,
            "candle_type": "normal", "is_big_normal_as0": 0,
        }
        # Place another bearish candle at idx=6 further from outer
        rows[6] = {
            "direction": -1, "c": 0.95, "l": 0.93, "h": 1.02,
            "candle_type": "normal", "is_big_normal_as0": 0,
        }
        # Put a big bullish candle at idx=8 for first wave
        rows[8] = {
            "direction": 1, "c": 1.1, "l": 0.98, "h": 1.12,
            "candle_type": "maru", "is_big_normal_as0": 1,
        }

        df = _make_df(rows)
        zone = _make_zone("buy", top=1.0, bottom=0.9, source_kind="BOS", meta={
            "cycle_id": 1, "anchor_idx": 5, "base_pattern": "no base",
            "outer": 0.9,
        })

        # CTS_CONFIRMED for upper bound of first wave search
        events = [
            _make_event("CTS_CONFIRMED", 12, meta={"structure_id": 0, "cycle_id": 1, "confirmed_at": 12}),
        ]

        result = identify_wave_candles(
            anchor_idx=5, anchor_type="BOS", zone=zone, events=events,
            structure_id=0, struct_direction=1, df=df,
        )

        assert result is not None
        assert result.source_kind == "BOS"
        # idx=4 is closer to outer(0.9): close 0.91 vs idx=6 close 0.95
        assert result.last_wave_candle_idx == 4
        assert result.first_wave_candle_idx == 8


class TestBOSBIB:
    def test_forward_search_accepted(self):
        """BIB BOS: forward search candidate closer to outer than anchor's close."""
        rows = []
        for i in range(20):
            rows.append({
                "direction": 1, "c": 1.05, "l": 1.0, "h": 1.1,
                "candle_type": "normal", "is_big_normal_as0": 0,
            })
        # Anchor at idx=5 (BOS) — close is 1.05 (dist to outer=0.9 is 0.15)
        rows[5] = {
            "direction": 1, "c": 1.05, "l": 0.95, "h": 1.1,
            "candle_type": "normal", "is_big_normal_as0": 0,
        }
        # Forward candidate at idx=7 — bearish, close 0.92 (dist to outer 0.02 < 0.15)
        rows[7] = {
            "direction": -1, "c": 0.92, "l": 0.88, "h": 1.0,
            "candle_type": "normal", "is_big_normal_as0": 0,
        }

        df = _make_df(rows)
        zone = _make_zone("buy", top=1.0, bottom=0.9, source_kind="BOS", meta={
            "cycle_id": 1, "anchor_idx": 5, "base_pattern": "base inside bar",
            "outer": 0.9,
        })

        events = [
            _make_event("CTS_ESTABLISHED", 10, 1.08, {
                "structure_id": 0, "cycle_id": 1, "anchor_idx": 9,
            }),
        ]

        result = identify_wave_candles(
            anchor_idx=5, anchor_type="BOS", zone=zone, events=events,
            structure_id=0, struct_direction=1, df=df,
        )

        assert result is not None
        assert result.last_wave_candle_idx == 7

    def test_forward_rejected_falls_to_backward(self):
        """BIB BOS: forward candidate NOT closer → backward fallback."""
        rows = []
        for i in range(20):
            rows.append({
                "direction": 1, "c": 1.05, "l": 1.0, "h": 1.1,
                "candle_type": "normal", "is_big_normal_as0": 0,
            })
        # Anchor at idx=8 — close is 0.92 (dist to outer=0.9 is 0.02)
        rows[8] = {
            "direction": 1, "c": 0.92, "l": 0.88, "h": 1.0,
            "candle_type": "normal", "is_big_normal_as0": 0,
        }
        # Forward candidate at idx=10 — bearish but close 0.95 (dist 0.05 > 0.02)
        rows[10] = {
            "direction": -1, "c": 0.95, "l": 0.93, "h": 1.01,
            "candle_type": "normal", "is_big_normal_as0": 0,
        }
        # Backward candidate at idx=6 — bearish, close 0.91 (dist 0.01 < 0.02)
        rows[6] = {
            "direction": -1, "c": 0.91, "l": 0.89, "h": 1.0,
            "candle_type": "normal", "is_big_normal_as0": 0,
        }

        df = _make_df(rows)
        zone = _make_zone("buy", top=1.0, bottom=0.9, source_kind="BOS", meta={
            "cycle_id": 1, "anchor_idx": 8, "base_pattern": "base inside bar",
            "outer": 0.9,
        })

        events = [
            _make_event("CTS_ESTABLISHED", 12, 1.08, {
                "structure_id": 0, "cycle_id": 1, "anchor_idx": 11,
            }),
            _make_event("STATE_CHANGED", 5, meta={
                "structure_id": 0, "to": "pullback",
            }),
        ]

        result = identify_wave_candles(
            anchor_idx=8, anchor_type="BOS", zone=zone, events=events,
            structure_id=0, struct_direction=1, df=df,
        )

        assert result is not None
        assert result.last_wave_candle_idx == 6


# ---------------------------------------------------------------------------
# CTS wave candle tests
# ---------------------------------------------------------------------------

class TestCTSNonBIB:
    def test_window_search(self):
        """Non-BIB CTS: window search finds qualified candle touching zone closest to outer."""
        rows = []
        for i in range(20):
            rows.append({
                "direction": -1, "c": 0.95, "l": 0.9, "h": 1.0,
                "candle_type": "normal", "is_big_normal_as0": 0,
            })
        # Sell zone: last breakout is bullish (+1), first pullback is bearish (-1)
        # Place a bullish candle at idx=9 touching sell zone
        rows[9] = {
            "direction": 1, "c": 1.05, "l": 0.98, "h": 1.08,
            "candle_type": "normal", "is_big_normal_as0": 0,
        }
        # Place another bullish candle at idx=11 further from outer
        rows[11] = {
            "direction": 1, "c": 1.02, "l": 0.99, "h": 1.05,
            "candle_type": "normal", "is_big_normal_as0": 0,
        }
        # Big bearish candle at idx=13 for first pullback
        rows[13] = {
            "direction": -1, "c": 0.93, "l": 0.9, "h": 1.0,
            "candle_type": "maru", "is_big_normal_as0": 1,
        }

        df = _make_df(rows)
        zone = _make_zone("sell", top=1.1, bottom=1.0, source_kind="CTS", meta={
            "cycle_id": 1, "anchor_idx": 10, "base_pattern": "no base",
            "outer": 1.1,
        })

        events = [
            _make_event("CTS_CONFIRMED", 15, meta={
                "structure_id": 0, "cycle_id": 1, "confirmed_at": 15,
            }),
        ]

        result = identify_wave_candles(
            anchor_idx=10, anchor_type="CTS", zone=zone, events=events,
            structure_id=0, struct_direction=1, df=df,
        )

        assert result is not None
        assert result.source_kind == "CTS"
        # idx=9 is closer to outer(1.1): close 1.05 vs idx=11 close 1.02
        assert result.last_wave_candle_idx == 9
        assert result.first_wave_candle_idx == 13


class TestCTSBIB:
    def test_step1_direct_match(self):
        """CTS BIB Step 1: event candle directly qualifies."""
        rows = []
        for i in range(20):
            rows.append({
                "direction": -1, "c": 0.95, "l": 0.9, "h": 1.0,
                "candle_type": "normal", "is_big_normal_as0": 0,
            })
        # CTS_ESTABLISHED at idx=10, candle is bullish, wick enters sell zone, close within
        rows[10] = {
            "direction": 1, "c": 1.05, "l": 0.98, "h": 1.12,
            "candle_type": "normal", "is_big_normal_as0": 0,
        }

        df = _make_df(rows)
        zone = _make_zone("sell", top=1.1, bottom=1.0, source_kind="CTS", meta={
            "cycle_id": 1, "anchor_idx": 10, "base_pattern": "base inside bar",
            "outer": 1.1,
        })

        events = [
            _make_event("CTS_ESTABLISHED", 10, 1.08, {
                "structure_id": 0, "cycle_id": 1,
            }),
            _make_event("CTS_CONFIRMED", 15, meta={
                "structure_id": 0, "cycle_id": 1, "confirmed_at": 15,
            }),
        ]

        result = identify_wave_candles(
            anchor_idx=10, anchor_type="CTS", zone=zone, events=events,
            structure_id=0, struct_direction=1, df=df,
        )

        assert result is not None
        assert result.last_wave_candle_idx == 10

    def test_step1_scan_between_updates(self):
        """CTS BIB Step 1: event candle doesn't qualify, scan between updates."""
        rows = []
        for i in range(25):
            rows.append({
                "direction": -1, "c": 0.95, "l": 0.9, "h": 0.99,
                "candle_type": "normal", "is_big_normal_as0": 0,
            })
        # CTS_ESTABLISHED at idx=10 — bearish, doesn't qualify (need bullish for sell zone)
        rows[10] = {
            "direction": -1, "c": 0.95, "l": 0.9, "h": 1.0,
            "candle_type": "normal", "is_big_normal_as0": 0,
        }
        # Candle at idx=12 — bullish, closes within sell zone
        rows[12] = {
            "direction": 1, "c": 1.05, "l": 0.98, "h": 1.08,
            "candle_type": "normal", "is_big_normal_as0": 0,
        }

        df = _make_df(rows)
        zone = _make_zone("sell", top=1.1, bottom=1.0, source_kind="CTS", meta={
            "cycle_id": 1, "anchor_idx": 10, "base_pattern": "base inside bar",
            "outer": 1.1,
        })

        events = [
            _make_event("CTS_ESTABLISHED", 10, 1.08, {
                "structure_id": 0, "cycle_id": 1,
            }),
            _make_event("CTS_UPDATED", 14, 1.09, {
                "structure_id": 0, "cycle_id": 1,
            }),
            _make_event("CTS_CONFIRMED", 18, meta={
                "structure_id": 0, "cycle_id": 1, "confirmed_at": 18,
            }),
        ]

        result = identify_wave_candles(
            anchor_idx=10, anchor_type="CTS", zone=zone, events=events,
            structure_id=0, struct_direction=1, df=df,
        )

        assert result is not None
        assert result.last_wave_candle_idx == 12

    def test_step2_fallback(self):
        """CTS BIB Step 2 fallback: no match in event walk, scan around anchor."""
        rows = []
        for i in range(25):
            rows.append({
                "direction": -1, "c": 0.95, "l": 0.9, "h": 0.99,
                "candle_type": "normal", "is_big_normal_as0": 0,
            })
        # No candles qualify in the event walk
        # But candle at idx=8 (before anchor) is bullish and closes within zone
        rows[8] = {
            "direction": 1, "c": 1.05, "l": 0.98, "h": 1.08,
            "candle_type": "normal", "is_big_normal_as0": 0,
        }

        df = _make_df(rows)
        zone = _make_zone("sell", top=1.1, bottom=1.0, source_kind="CTS", meta={
            "cycle_id": 1, "anchor_idx": 10, "base_pattern": "base inside bar",
            "outer": 1.1,
        })

        events = [
            _make_event("CTS_ESTABLISHED", 10, 1.08, {
                "structure_id": 0, "cycle_id": 1,
            }),
            _make_event("CTS_CONFIRMED", 15, meta={
                "structure_id": 0, "cycle_id": 1, "confirmed_at": 15,
            }),
        ]

        result = identify_wave_candles(
            anchor_idx=10, anchor_type="CTS", zone=zone, events=events,
            structure_id=0, struct_direction=1, df=df,
        )

        assert result is not None
        assert result.last_wave_candle_idx == 8


# ---------------------------------------------------------------------------
# First wave candle compound condition
# ---------------------------------------------------------------------------

class TestFirstWaveCandle:
    def test_first_wave_found(self):
        """First wave candle is the first compound candle after last_wave_candle."""
        rows = []
        for i in range(20):
            rows.append({
                "direction": -1, "c": 0.95, "l": 0.9, "h": 1.0,
                "candle_type": "normal", "is_big_normal_as0": 0,
            })
        # Last pullback at idx=4
        rows[4] = {
            "direction": -1, "c": 0.91, "l": 0.88, "h": 0.99,
            "candle_type": "normal", "is_big_normal_as0": 0,
        }
        # Small bullish at idx=5 (not compound)
        rows[5] = {
            "direction": 1, "c": 1.01, "l": 0.98, "h": 1.02,
            "candle_type": "normal", "is_big_normal_as0": 0,
        }
        # Big bullish maru at idx=7 (compound first wave)
        rows[7] = {
            "direction": 1, "c": 1.08, "l": 0.97, "h": 1.1,
            "candle_type": "maru", "is_big_normal_as0": 1,
        }

        df = _make_df(rows)
        zone = _make_zone("buy", top=1.0, bottom=0.9, source_kind="BOS", meta={
            "cycle_id": 1, "anchor_idx": 5, "base_pattern": "no base",
            "outer": 0.9,
        })

        events = [
            _make_event("CTS_CONFIRMED", 15, meta={
                "structure_id": 0, "cycle_id": 1, "confirmed_at": 15,
            }),
        ]

        result = identify_wave_candles(
            anchor_idx=5, anchor_type="BOS", zone=zone, events=events,
            structure_id=0, struct_direction=1, df=df,
        )

        assert result is not None
        # idx=4 is the only bearish candle touching zone in window [0,10]
        assert result.last_wave_candle_idx == 4
        assert result.first_wave_candle_idx == 7


class TestEmptyResults:
    def test_no_qualified_candles(self):
        """When no candles match the required direction, return None for last/first."""
        rows = []
        for i in range(20):
            # All bullish — no bearish pullback candles for buy zone
            rows.append({
                "direction": 1, "c": 1.05, "l": 1.0, "h": 1.1,
                "candle_type": "normal", "is_big_normal_as0": 0,
            })

        df = _make_df(rows)
        zone = _make_zone("buy", top=1.0, bottom=0.9, source_kind="BOS", meta={
            "cycle_id": 1, "anchor_idx": 5, "base_pattern": "no base",
            "outer": 0.9,
        })

        events = []

        result = identify_wave_candles(
            anchor_idx=5, anchor_type="BOS", zone=zone, events=events,
            structure_id=0, struct_direction=1, df=df,
        )

        assert result is not None
        assert result.last_wave_candle_idx is None
        assert result.first_wave_candle_idx is None
