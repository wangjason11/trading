"""Unit tests for engine_v2.zones.wvmi."""
from __future__ import annotations

import pandas as pd
import pytest

from engine_v2.common.types import KLZone, WVMIRecord
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.zones.wave_candles import WaveCandleResult
from engine_v2.zones.wvmi import (
    WVMITracker,
    _compute_last_wave_weight,
    _find_temporary_lp,
    _find_wave_candle,
    _find_bos_zone,
    _assign_direction_labels,
)


# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------

def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Create a minimal OHLC DataFrame with candle features."""
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
        "body_pct": 0.50,
        "vol_dir": 0,
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


def _make_wc(sid: int, cycle_id: int, source_kind: str, zone_side: str,
             last_idx: int | None, first_idx: int | None) -> WaveCandleResult:
    return WaveCandleResult(
        structure_id=sid,
        cycle_id=cycle_id,
        source_kind=source_kind,
        zone_side=zone_side,
        last_wave_candle_idx=last_idx,
        first_wave_candle_idx=first_idx,
    )


# ---------------------------------------------------------------------------
# _compute_last_wave_weight tests
# ---------------------------------------------------------------------------

class TestComputeLastWaveWeight:
    def test_big_normal_maru_returns_1(self):
        df = _make_df([{"is_big_normal_as0": 1, "candle_type": "maru"}])
        assert _compute_last_wave_weight(df, 0, 1) == 1.0

    def test_big_normal_normal_returns_1(self):
        df = _make_df([{"is_big_normal_as0": 1, "candle_type": "normal"}])
        assert _compute_last_wave_weight(df, 0, 1) == 1.0

    def test_big_maru_pinbar_opposite_dir_returns_1(self):
        """pinbar_dir != wave_dir -> weight 1.0 (last candle rule)."""
        df = _make_df([{"is_big_maru_as0": 1, "candle_type": "pinbar", "pinbar_dir": -1}])
        assert _compute_last_wave_weight(df, 0, 1) == 1.0

    def test_big_maru_pinbar_same_dir_not_1(self):
        """pinbar_dir == wave_dir -> NOT weight 1.0 for last candle."""
        df = _make_df([{"is_big_maru_as0": 1, "candle_type": "pinbar", "pinbar_dir": 1, "body_pct": 0.50}])
        assert _compute_last_wave_weight(df, 0, 1) == 0.7

    def test_body_pct_lte_10_returns_05(self):
        df = _make_df([{"body_pct": 0.08}])
        assert _compute_last_wave_weight(df, 0, 1) == 0.5

    def test_body_pct_exactly_10_returns_05(self):
        df = _make_df([{"body_pct": 0.10}])
        assert _compute_last_wave_weight(df, 0, 1) == 0.5

    def test_body_pct_11_returns_07(self):
        df = _make_df([{"body_pct": 0.11}])
        assert _compute_last_wave_weight(df, 0, 1) == 0.7

    def test_default_returns_07(self):
        df = _make_df([{"body_pct": 0.50}])
        assert _compute_last_wave_weight(df, 0, 1) == 0.7

    def test_out_of_bounds_returns_07(self):
        df = _make_df([{}])
        assert _compute_last_wave_weight(df, 99, 1) == 0.7


# ---------------------------------------------------------------------------
# _find_temporary_lp tests
# ---------------------------------------------------------------------------

class TestFindTemporaryLp:
    def test_finds_closest_to_outer(self):
        """Among qualified candles, pick closest close to outer."""
        rows = [
            {"direction": -1, "vol_dir": -1, "c": 0.95},  # idx=0, dist=0.05
            {"direction": -1, "vol_dir": -1, "c": 0.92},  # idx=1, dist=0.02 (closer)
            {"direction": -1, "vol_dir": -1, "c": 0.98},  # idx=2, dist=0.08
        ]
        df = _make_df(rows)
        # FP at some external idx; set FP direction in df at a separate row
        # FP_idx is before search range; we just need its direction
        fp_rows = [{"direction": -1, "vol_dir": -1, "c": 1.0}] + rows
        fp_df = _make_df(fp_rows)
        zone = _make_zone("buy", top=1.0, bottom=0.9, meta={"outer": 0.9})
        result = _find_temporary_lp(fp_df, 0, zone, 1, 3)
        assert result == 2  # idx=2 in fp_df has c=0.92, closest to 0.9

    def test_no_qualified_returns_none(self):
        """All candles wrong direction -> None."""
        rows = [
            {"direction": 1, "vol_dir": 1, "c": 0.95},
            {"direction": 1, "vol_dir": 1, "c": 0.92},
        ]
        df = _make_df(rows)
        # FP at idx=0 with direction=-1 (not present in df)
        # Create FP row + search rows
        fp_rows = [{"direction": -1, "vol_dir": -1, "c": 1.0}] + rows
        fp_df = _make_df(fp_rows)
        zone = _make_zone("buy", top=1.0, bottom=0.9, meta={"outer": 0.9})
        result = _find_temporary_lp(fp_df, 0, zone, 1, 2)
        assert result is None

    def test_vol_dir_zero_qualifies(self):
        """Candle with vol_dir=0 qualifies regardless."""
        rows = [
            {"direction": -1, "vol_dir": 0, "c": 0.92},
        ]
        fp_rows = [{"direction": -1, "vol_dir": -1, "c": 1.0}] + rows
        fp_df = _make_df(fp_rows)
        zone = _make_zone("buy", top=1.0, bottom=0.9, meta={"outer": 0.9})
        result = _find_temporary_lp(fp_df, 0, zone, 1, 1)
        assert result == 1


# ---------------------------------------------------------------------------
# Lookup helper tests
# ---------------------------------------------------------------------------

class TestLookupHelpers:
    def test_find_wave_candle_found(self):
        wcs = [_make_wc(0, 1, "BOS", "buy", 5, 7), _make_wc(0, 1, "CTS", "sell", 10, 12)]
        assert _find_wave_candle(wcs, 0, 1, "CTS").last_wave_candle_idx == 10

    def test_find_wave_candle_not_found(self):
        wcs = [_make_wc(0, 1, "BOS", "buy", 5, 7)]
        assert _find_wave_candle(wcs, 0, 2, "BOS") is None

    def test_find_bos_zone_found(self):
        zones = [
            _make_zone("buy", 1.0, 0.9, "BOS", {"structure_id": 0, "cycle_id": 1}),
            _make_zone("sell", 1.2, 1.1, "CTS", {"structure_id": 0, "cycle_id": 1}),
        ]
        z = _find_bos_zone(zones, 0, 1)
        assert z is not None
        assert z.side == "buy"

    def test_find_bos_zone_not_found(self):
        zones = [_make_zone("sell", 1.2, 1.1, "CTS", {"structure_id": 0, "cycle_id": 1})]
        assert _find_bos_zone(zones, 0, 1) is None


class TestAssignDirectionLabels:
    def test_buy_zone(self):
        buy, sell = _assign_direction_labels("buy", 1.5, 0.8)
        assert buy == 1.5
        assert sell == 0.8

    def test_sell_zone(self):
        buy, sell = _assign_direction_labels("sell", 1.5, 0.8)
        assert buy == 0.8
        assert sell == 1.5

    def test_none_values(self):
        buy, sell = _assign_direction_labels("buy", 1.5, None)
        assert buy == 1.5
        assert sell is None


# ---------------------------------------------------------------------------
# WVMITracker.on_cts_confirmed tests
# ---------------------------------------------------------------------------

class TestOnCtsConfirmed:
    def _setup_basic(self):
        """Create a basic scenario: BOS zone + wave candles + CTS_CONFIRMED event."""
        rows = []
        for i in range(20):
            rows.append({
                "direction": 1, "c": 1.05, "volume": 100,
                "candle_type": "normal", "is_big_normal_as0": 0,
                "body_pct": 0.50, "vol_dir": 0,
            })
        # FB at idx=3 (first breakout from BOS wave candle): vol=200
        rows[3] = {**rows[3], "direction": 1, "volume": 200, "is_big_normal_as0": 1, "candle_type": "maru"}
        # LB at idx=8 (last breakout from CTS wave candle): vol=300
        rows[8] = {**rows[8], "direction": 1, "volume": 300, "is_big_normal_as0": 1, "candle_type": "normal"}
        # FP at idx=10 (first pullback from CTS wave candle): vol=150
        rows[10] = {**rows[10], "direction": -1, "volume": 150, "vol_dir": -1}
        # Potential temp LP at idx=12: vol=120, direction=-1
        rows[12] = {**rows[12], "direction": -1, "volume": 120, "c": 0.92, "vol_dir": -1}

        df = _make_df(rows)

        zone = _make_zone("buy", 1.0, 0.9, "BOS", {
            "structure_id": 0, "cycle_id": 1, "anchor_idx": 5, "outer": 0.9,
        })

        wave_candles = [
            _make_wc(0, 1, "BOS", "buy", last_idx=2, first_idx=3),   # FB=3
            _make_wc(0, 1, "CTS", "sell", last_idx=8, first_idx=10),  # LB=8, FP=10
        ]

        ev = _make_event("CTS_CONFIRMED", 14, meta={"structure_id": 0, "cycle_id": 1})
        return df, zone, wave_candles, ev

    def test_creates_record_with_locked_breakout(self):
        df, zone, wcs, ev = self._setup_basic()
        tracker = WVMITracker()
        rec = tracker.on_cts_confirmed(ev, df, wcs, [zone])

        assert rec is not None
        assert rec.bos_structure_id == 0
        assert rec.bos_cycle_id == 1
        assert rec.zone_side == "buy"
        assert rec.source == "main"
        assert rec.fb_idx == 3
        assert rec.lb_idx == 8
        assert rec.fp_idx == 10
        assert rec.fb_volume == 200
        assert rec.lb_volume == 300
        assert rec.fp_volume == 150
        assert rec.status == "created"
        assert rec.lp_locked is False

    def test_breakout_momentum_calculation(self):
        df, zone, wcs, ev = self._setup_basic()
        tracker = WVMITracker()
        rec = tracker.on_cts_confirmed(ev, df, wcs, [zone])

        # LB weight: is_big_normal_as0=1, candle_type=normal -> 1.0
        assert rec.lb_weight == 1.0
        # breakout_momentum = (300 * 1.0) / 200 = 1.5
        assert rec.breakout_momentum == pytest.approx(1.5)

    def test_buy_sell_labels_buy_zone(self):
        df, zone, wcs, ev = self._setup_basic()
        tracker = WVMITracker()
        rec = tracker.on_cts_confirmed(ev, df, wcs, [zone])

        # Buy zone: buy_momentum = breakout, sell_momentum = pullback
        assert rec.buy_momentum == rec.breakout_momentum
        assert rec.sell_momentum == rec.pullback_momentum

    def test_returns_none_if_no_bos_zone(self):
        df, zone, wcs, ev = self._setup_basic()
        tracker = WVMITracker()
        # Pass empty zones
        rec = tracker.on_cts_confirmed(ev, df, wcs, [])
        assert rec is None

    def test_returns_none_if_missing_wave_candle(self):
        df, zone, wcs, ev = self._setup_basic()
        tracker = WVMITracker()
        # Only BOS wave candle, no CTS
        rec = tracker.on_cts_confirmed(ev, df, [wcs[0]], [zone])
        assert rec is None

    def test_returns_none_if_zero_fb_volume(self):
        df, zone, wcs, ev = self._setup_basic()
        df.loc[3, "volume"] = 0  # FB volume = 0
        tracker = WVMITracker()
        rec = tracker.on_cts_confirmed(ev, df, wcs, [zone])
        assert rec is None

    def test_returns_none_if_zero_fp_volume(self):
        df, zone, wcs, ev = self._setup_basic()
        df.loc[10, "volume"] = 0  # FP volume = 0
        tracker = WVMITracker()
        rec = tracker.on_cts_confirmed(ev, df, wcs, [zone])
        assert rec is None

    def test_no_temp_lp_still_creates_record(self):
        """When no qualified temp LP exists, record has lp_idx=None and pullback=None."""
        rows = []
        for i in range(20):
            rows.append({
                "direction": 1, "c": 1.05, "volume": 100,
                "candle_type": "normal", "is_big_normal_as0": 0,
                "body_pct": 0.50, "vol_dir": 1,
            })
        rows[3] = {**rows[3], "direction": 1, "volume": 200, "is_big_normal_as0": 1, "candle_type": "maru"}
        rows[8] = {**rows[8], "direction": 1, "volume": 300, "is_big_normal_as0": 1, "candle_type": "normal"}
        rows[10] = {**rows[10], "direction": -1, "volume": 150, "vol_dir": -1}
        # All candles after FP are bullish (wrong direction) -> no temp LP
        df = _make_df(rows)

        zone = _make_zone("buy", 1.0, 0.9, "BOS", {
            "structure_id": 0, "cycle_id": 1, "anchor_idx": 5, "outer": 0.9,
        })
        wcs = [
            _make_wc(0, 1, "BOS", "buy", last_idx=2, first_idx=3),
            _make_wc(0, 1, "CTS", "sell", last_idx=8, first_idx=10),
        ]
        ev = _make_event("CTS_CONFIRMED", 14, meta={"structure_id": 0, "cycle_id": 1})

        tracker = WVMITracker()
        rec = tracker.on_cts_confirmed(ev, df, wcs, [zone])

        assert rec is not None
        assert rec.lp_idx is None
        assert rec.pullback_momentum is None
        assert rec.breakout_momentum is not None


# ---------------------------------------------------------------------------
# WVMITracker.update_temporary_lp tests
# ---------------------------------------------------------------------------

class TestUpdateTemporaryLp:
    def test_shifts_lp_to_closer_candle(self):
        """When a closer qualified candle appears in extended df, LP shifts."""
        # Phase 1: create df with 15 rows (idx 0-14), initial LP at idx=12
        rows = []
        for i in range(15):
            rows.append({
                "direction": 1, "c": 1.05, "volume": 100,
                "candle_type": "normal", "is_big_normal_as0": 0,
                "body_pct": 0.50, "vol_dir": 0,
            })
        rows[3] = {**rows[3], "direction": 1, "volume": 200, "is_big_normal_as0": 1, "candle_type": "maru"}
        rows[8] = {**rows[8], "direction": 1, "volume": 300, "is_big_normal_as0": 1, "candle_type": "normal"}
        rows[10] = {**rows[10], "direction": -1, "volume": 150, "vol_dir": -1}
        rows[12] = {**rows[12], "direction": -1, "volume": 120, "c": 0.95, "vol_dir": -1}

        df_initial = _make_df(rows)

        zone = _make_zone("buy", 1.0, 0.9, "BOS", {
            "structure_id": 0, "cycle_id": 1, "anchor_idx": 5, "outer": 0.9,
        })
        wcs = [
            _make_wc(0, 1, "BOS", "buy", last_idx=2, first_idx=3),
            _make_wc(0, 1, "CTS", "sell", last_idx=8, first_idx=10),
        ]
        ev = _make_event("CTS_CONFIRMED", 14, meta={"structure_id": 0, "cycle_id": 1})

        tracker = WVMITracker()
        tracker.on_cts_confirmed(ev, df_initial, wcs, [zone])
        assert tracker.get_records()[0].lp_idx == 12

        # Phase 2: extend df with closer candidate at idx=17
        rows_ext = rows + [
            {"direction": 1, "c": 1.05, "volume": 100, "candle_type": "normal",
             "is_big_normal_as0": 0, "body_pct": 0.50, "vol_dir": 0},
            {"direction": 1, "c": 1.05, "volume": 100, "candle_type": "normal",
             "is_big_normal_as0": 0, "body_pct": 0.50, "vol_dir": 0},
            {"direction": -1, "c": 0.91, "volume": 110, "candle_type": "normal",
             "is_big_normal_as0": 0, "body_pct": 0.50, "vol_dir": -1},
        ]
        df_extended = _make_df(rows_ext)

        updated = tracker.update_temporary_lp(df_extended, [zone])
        assert len(updated) == 1
        rec = updated[0]
        assert rec.lp_idx == 17
        assert rec.status == "updated"

    def test_locked_record_not_updated(self):
        """Locked records should not be updated."""
        rows = []
        for i in range(20):
            rows.append({
                "direction": 1, "c": 1.05, "volume": 100,
                "candle_type": "normal", "is_big_normal_as0": 0,
                "body_pct": 0.50, "vol_dir": 0,
            })
        rows[3] = {**rows[3], "direction": 1, "volume": 200, "is_big_normal_as0": 1, "candle_type": "maru"}
        rows[8] = {**rows[8], "direction": 1, "volume": 300, "is_big_normal_as0": 1, "candle_type": "normal"}
        rows[10] = {**rows[10], "direction": -1, "volume": 150, "vol_dir": -1}
        rows[12] = {**rows[12], "direction": -1, "volume": 120, "c": 0.92, "vol_dir": -1}

        df = _make_df(rows)
        zone = _make_zone("buy", 1.0, 0.9, "BOS", {
            "structure_id": 0, "cycle_id": 1, "anchor_idx": 5, "outer": 0.9,
        })
        wcs = [
            _make_wc(0, 1, "BOS", "buy", last_idx=2, first_idx=3),
            _make_wc(0, 1, "CTS", "sell", last_idx=8, first_idx=10),
        ]

        tracker = WVMITracker()
        cts_ev = _make_event("CTS_CONFIRMED", 14, meta={"structure_id": 0, "cycle_id": 1})
        tracker.on_cts_confirmed(cts_ev, df, wcs, [zone])

        # Lock via BOS_n+1
        bos_wc_2 = _make_wc(0, 2, "BOS", "buy", last_idx=15, first_idx=16)
        bos_ev = _make_event("BOS_CONFIRMED", 16, meta={"structure_id": 0, "cycle_id": 2})
        tracker.on_bos_confirmed(bos_ev, df, wcs + [bos_wc_2])

        rec_before = tracker.get_records()[0]
        assert rec_before.lp_locked is True
        lp_before = rec_before.lp_idx

        # Update should not change locked record
        updated = tracker.update_temporary_lp(df, [zone])
        assert len(updated) == 0
        rec_after = tracker.get_records()[0]
        assert rec_after.lp_idx == lp_before


# ---------------------------------------------------------------------------
# WVMITracker.on_bos_confirmed tests
# ---------------------------------------------------------------------------

class TestOnBosConfirmed:
    def test_locks_previous_cycle(self):
        """BOS_n+1 locks WVMI for cycle_n."""
        rows = []
        for i in range(25):
            rows.append({
                "direction": 1, "c": 1.05, "volume": 100,
                "candle_type": "normal", "is_big_normal_as0": 0,
                "body_pct": 0.50, "vol_dir": 0,
            })
        rows[3] = {**rows[3], "direction": 1, "volume": 200, "is_big_normal_as0": 1, "candle_type": "maru"}
        rows[8] = {**rows[8], "direction": 1, "volume": 300, "is_big_normal_as0": 1, "candle_type": "normal"}
        rows[10] = {**rows[10], "direction": -1, "volume": 150, "vol_dir": -1}
        rows[12] = {**rows[12], "direction": -1, "volume": 120, "c": 0.92, "vol_dir": -1}
        # BOS_2 LP wave candle at idx=18
        rows[18] = {**rows[18], "direction": -1, "volume": 180, "c": 0.93, "vol_dir": -1}

        df = _make_df(rows)
        zone = _make_zone("buy", 1.0, 0.9, "BOS", {
            "structure_id": 0, "cycle_id": 1, "anchor_idx": 5, "outer": 0.9,
        })
        wcs = [
            _make_wc(0, 1, "BOS", "buy", last_idx=2, first_idx=3),
            _make_wc(0, 1, "CTS", "sell", last_idx=8, first_idx=10),
            _make_wc(0, 2, "BOS", "buy", last_idx=18, first_idx=19),  # BOS_2 wave candle
        ]

        tracker = WVMITracker()
        cts_ev = _make_event("CTS_CONFIRMED", 14, meta={"structure_id": 0, "cycle_id": 1})
        tracker.on_cts_confirmed(cts_ev, df, wcs, [zone])

        # BOS_n+1 (cycle=2) locks cycle=1
        bos_ev = _make_event("BOS_CONFIRMED", 20, meta={"structure_id": 0, "cycle_id": 2})
        locked = tracker.on_bos_confirmed(bos_ev, df, wcs)

        assert len(locked) == 1
        rec = locked[0]
        assert rec.lp_locked is True
        assert rec.status == "locked"
        assert rec.locked_by_cycle_id == 2
        # LP should be from BOS_2's last wave candle
        assert rec.lp_idx == 18
        assert rec.lp_volume == 180

    def test_bos_cycle_0_does_not_lock_negative(self):
        """BOS with cycle_id=0 should not try to lock cycle_id=-1."""
        tracker = WVMITracker()
        bos_ev = _make_event("BOS_CONFIRMED", 5, meta={"structure_id": 0, "cycle_id": 0})
        df = _make_df([{}])
        locked = tracker.on_bos_confirmed(bos_ev, df, [])
        assert len(locked) == 0

    def test_locks_with_existing_temp_lp_when_no_wave_candle(self):
        """When BOS_n+1 has no wave candle, lock with existing temp LP."""
        rows = []
        for i in range(20):
            rows.append({
                "direction": 1, "c": 1.05, "volume": 100,
                "candle_type": "normal", "is_big_normal_as0": 0,
                "body_pct": 0.50, "vol_dir": 0,
            })
        rows[3] = {**rows[3], "direction": 1, "volume": 200, "is_big_normal_as0": 1, "candle_type": "maru"}
        rows[8] = {**rows[8], "direction": 1, "volume": 300, "is_big_normal_as0": 1, "candle_type": "normal"}
        rows[10] = {**rows[10], "direction": -1, "volume": 150, "vol_dir": -1}
        rows[12] = {**rows[12], "direction": -1, "volume": 120, "c": 0.92, "vol_dir": -1}

        df = _make_df(rows)
        zone = _make_zone("buy", 1.0, 0.9, "BOS", {
            "structure_id": 0, "cycle_id": 1, "anchor_idx": 5, "outer": 0.9,
        })
        wcs = [
            _make_wc(0, 1, "BOS", "buy", last_idx=2, first_idx=3),
            _make_wc(0, 1, "CTS", "sell", last_idx=8, first_idx=10),
            # BOS_2 has no last wave candle
            _make_wc(0, 2, "BOS", "buy", last_idx=None, first_idx=None),
        ]

        tracker = WVMITracker()
        cts_ev = _make_event("CTS_CONFIRMED", 14, meta={"structure_id": 0, "cycle_id": 1})
        tracker.on_cts_confirmed(cts_ev, df, wcs, [zone])

        initial_lp = tracker.get_records()[0].lp_idx

        bos_ev = _make_event("BOS_CONFIRMED", 16, meta={"structure_id": 0, "cycle_id": 2})
        locked = tracker.on_bos_confirmed(bos_ev, df, wcs)

        assert len(locked) == 1
        rec = locked[0]
        assert rec.lp_locked is True
        # LP unchanged — locked with temp LP
        assert rec.lp_idx == initial_lp


# ---------------------------------------------------------------------------
# Scenario 3 stubs
# ---------------------------------------------------------------------------

class TestScenario3Stubs:
    def test_add_and_discard_scenario3(self):
        tracker = WVMITracker()
        record = WVMIRecord(
            bos_structure_id=0, bos_cycle_id=1,
            zone_side="buy", source="scenario3",
        )
        tracker.add_scenario3_record(0, 1, record)
        assert len(tracker.get_records()) == 1

        tracker.discard_scenario3(0, 1)
        assert len(tracker.get_records()) == 0

    def test_discard_nonexistent_is_noop(self):
        tracker = WVMITracker()
        tracker.discard_scenario3(0, 1)  # Should not raise
        assert len(tracker.get_records()) == 0


# ---------------------------------------------------------------------------
# Sell zone tests (direction reversal)
# ---------------------------------------------------------------------------

class TestSellZone:
    def test_sell_zone_labels(self):
        """Sell zone: buy_momentum=pullback, sell_momentum=breakout."""
        rows = []
        for i in range(20):
            rows.append({
                "direction": -1, "c": 0.95, "volume": 100,
                "candle_type": "normal", "is_big_normal_as0": 0,
                "body_pct": 0.50, "vol_dir": 0,
            })
        # For sell zone: BOS wave candle first_idx is FB (bearish breakout)
        rows[3] = {**rows[3], "direction": -1, "volume": 200, "is_big_normal_as0": 1, "candle_type": "maru"}
        # CTS last_wave_candle_idx is LB (bearish)
        rows[8] = {**rows[8], "direction": -1, "volume": 300, "is_big_normal_as0": 1, "candle_type": "normal"}
        # CTS first_wave_candle_idx is FP (bullish pullback)
        rows[10] = {**rows[10], "direction": 1, "volume": 150, "vol_dir": 1}
        # Temp LP (bullish)
        rows[12] = {**rows[12], "direction": 1, "volume": 120, "c": 1.08, "vol_dir": 1}

        df = _make_df(rows)
        zone = _make_zone("sell", 1.1, 1.0, "BOS", {
            "structure_id": 0, "cycle_id": 1, "anchor_idx": 5, "outer": 1.1,
        })
        wcs = [
            _make_wc(0, 1, "BOS", "sell", last_idx=2, first_idx=3),
            _make_wc(0, 1, "CTS", "buy", last_idx=8, first_idx=10),
        ]
        ev = _make_event("CTS_CONFIRMED", 14, meta={"structure_id": 0, "cycle_id": 1})

        tracker = WVMITracker()
        rec = tracker.on_cts_confirmed(ev, df, wcs, [zone])

        assert rec is not None
        # Sell zone: buy_momentum = pullback, sell_momentum = breakout
        assert rec.sell_momentum == rec.breakout_momentum
        assert rec.buy_momentum == rec.pullback_momentum
