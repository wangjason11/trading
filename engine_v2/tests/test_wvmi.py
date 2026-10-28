"""Unit tests for engine_v2.zones.wvmi."""
from __future__ import annotations

import pandas as pd
import pytest

from engine_v2.common.types import KLZone, WVMIRecord
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.tests._event_factory import make_event
from engine_v2.zones.wave_candles import WaveCandleResult
from engine_v2.zones.wvmi import (
    WVMITracker,
    _compute_last_wave_weight,
    _temporary_lp_sequence,
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
    meta = dict(meta or {})
    if etype in ("CTS_ESTABLISHED", "BOS_CONFIRMED"):
        # the `idx` argument is the anchor (`make_event`); the moment defaults to it (lag 0).
        meta.setdefault("confirmed_at", idx)
    return make_event(
        etype, idx, price=price,
        category="STRUCTURE" if etype != "STATE_CHANGED" else "STATE",
        **meta,
    )


def _cts_confirmed(idx: int = 14, *, pattern_anchor_idx: int = 12, method: str = "pullback",
                   sid: int = 0, cycle_id: int = 1) -> StructureEvent:
    """A CTS_CONFIRMED as MS emits it: `confirmation_method`, and on a pullback the pattern's first candle
    (`pattern_anchor_idx`, read by the first temporary LP). The default pattern [12, 14] holds the setups' LP 12."""
    meta = {"structure_id": sid, "cycle_id": cycle_id, "confirmation_method": method}
    if method == "pullback":
        meta["pattern_anchor_idx"] = pattern_anchor_idx
    return _make_event("CTS_CONFIRMED", idx, meta=meta)


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
# _temporary_lp_sequence tests (the furthest-close rule, 2026-10-26)
# ---------------------------------------------------------------------------

_N = (1, 0, 1.05)   # a bullish candle: never a buy zone's (bearish) pullback LP


def _q(close: float, vol_dir: int = -1) -> tuple:
    """A qualified bearish pullback candle (a buy zone's): direction -1, vol_dir -1 or 0."""
    return (-1, vol_dir, close)


def _seq(spec: list, fp: int, ev: StructureEvent, search_end: int, pb_wave_dir: int = -1) -> list:
    """`_temporary_lp_sequence` on one row per candle `(direction, vol_dir, close)`."""
    df = _make_df([{"direction": d, "vol_dir": v, "c": c} for d, v, c in spec])
    return _temporary_lp_sequence(df, fp, pb_wave_dir, ev, search_end)


class TestTemporaryLpSequence:
    def test_pattern_route_takes_the_furthest_pattern_close_then_each_further_close(self):
        """A buy zone (pullback = lower closes). The pattern [3, 5]: of its qualified candles 3 (.93) and 4 (.95) the
        furthest is 3, as of the CTS moment 5 — not the last one, and not 1 (.90, further but BEFORE the pattern's
        first candle). Then 6 (.93) ties 3 and does not move it; 7 (.92) and 9 (.91) do; 8 (.915 > .91) comes before
        9 and is further than 7, so it is one too."""
        spec = [_q(1.00), _q(0.90), _N, _q(0.93), _q(0.95), _N, _q(0.93), _q(0.92), _q(0.915), _q(0.91)]
        ev = _cts_confirmed(5, pattern_anchor_idx=3)
        assert _seq(spec, 0, ev, 9) == [(5, 3), (7, 7), (8, 8), (9, 9)]

    def test_never_fp_even_when_the_pattern_starts_on_it(self):
        """The pattern [1, 3] starts ON FP 1, which closes furthest (.90): the first LP is the next pattern candle 2."""
        spec = [_N, _q(0.90), _q(0.95), _N]
        assert _seq(spec, 1, _cts_confirmed(3, pattern_anchor_idx=1), 3) == [(3, 2)]

    def test_the_fallback_waits_for_a_close_beyond_fp_not_beyond_the_confirmation_candle(self):
        """FP 0 (.99) is the pattern's only qualified candle; the confirmation candle 1 closes away (1.05). 2 closes
        beyond 1.05 but not beyond FP's .99, so it is not the first LP; 3 (.98) is; 4 ties; 5 (.97) moves it."""
        spec = [_q(0.99), _N, _q(1.02), _q(0.98), _q(0.98), _q(0.97)]
        assert _seq(spec, 0, _cts_confirmed(1, pattern_anchor_idx=0), 5) == [(3, 3), (5, 5)]

    def test_the_proximity_route_takes_the_furthest_prior_close_beyond_the_proximity_candle(self):
        """p = 5 (.98, itself qualified). Prior candidates in [FP + 1, p - 1] closing beyond .98: 1 (.97), 2 (.94),
        4 (.96) -> 2, as of 5; FP 0 (.90) is never one. Then 6 ties, 7 (.93) moves it."""
        spec = [_q(0.90), _q(0.97), _q(0.94), _N, _q(0.96), _q(0.98), _q(0.94), _q(0.93)]
        ev = _cts_confirmed(5, method="sd_zone_proximity")
        assert _seq(spec, 0, ev, 7) == [(5, 2), (7, 7)]

    def test_the_proximity_candle_itself_when_no_prior_close_is_beyond_it(self):
        """1 (.99) is qualified but not beyond p's .97: p 3 is the first LP."""
        spec = [_q(1.00), _q(0.99), _N, _q(0.97)]
        assert _seq(spec, 0, _cts_confirmed(3, method="sd_zone_proximity"), 3) == [(3, 3)]

    def test_a_proximity_candle_on_fp_is_never_the_lp(self):
        """p = FP 1: step 2 is skipped; step 3 waits for a close beyond p's .97 — 2 (.98) is not, 3 (.96) is."""
        spec = [_N, _q(0.97), _q(0.98), _q(0.96)]
        assert _seq(spec, 1, _cts_confirmed(1, method="sd_zone_proximity"), 3) == [(3, 3)]

    def test_an_unqualified_proximity_candle_waits_for_a_close_beyond_its_own(self):
        """p 1 is bullish (1.05): the first LP is the first later qualified candle closing beyond 1.05 — 2 (1.03),
        not one beyond FP's 1.00."""
        spec = [_q(1.00), _N, _q(1.03), _q(0.99)]
        assert _seq(spec, 0, _cts_confirmed(1, method="sd_zone_proximity"), 3) == [(2, 2), (3, 3)]

    def test_no_candidate_after_the_search_end(self):
        """The search ends at the cycle's end - 1: 4 (.96) is an LP only when the end is past it, and a pattern
        candidate past the end is none either (search end 0: no LP at all)."""
        spec = [_q(1.00), _q(0.98), _N, _q(0.97), _q(0.96)]
        ev = _cts_confirmed(2, pattern_anchor_idx=1)
        assert _seq(spec, 0, ev, 3) == [(2, 1), (3, 3)]
        assert _seq(spec, 0, ev, 4) == [(2, 1), (3, 3), (4, 4)]
        assert _seq(spec, 0, ev, 0) == []

    def test_a_sell_zone_pullback_moves_on_higher_closes(self):
        """A sell zone (pullback = bullish, higher closes further): 2 (1.04) beats 1 (1.02) in the pattern; 3 (1.03)
        is not further; 4 (1.05) is."""
        spec = [(1, 1, 1.00), (1, 1, 1.02), (1, 0, 1.04), (1, 1, 1.03), (1, 1, 1.05)]
        assert _seq(spec, 0, _cts_confirmed(2, pattern_anchor_idx=1), 4, pb_wave_dir=1) == [(2, 2), (4, 4)]

    def test_only_qualified_candles(self):
        """Qualified = the pullback direction AND the volume direction agrees or is 0: 2 (.80, vol_dir +1) is not a
        candidate, 3 (.85, vol_dir 0) is; 4 is bullish."""
        spec = [_q(1.00), _q(0.95), (-1, 1, 0.80), _q(0.85, vol_dir=0), (1, -1, 0.70)]
        assert _seq(spec, 0, _cts_confirmed(1, pattern_anchor_idx=1), 4) == [(1, 1), (3, 3)]

    def test_a_volume_only_doji_is_a_candidate(self):
        """PLAN_V 8c (2026-10-28): a volume-only doji in the pullback direction (2: direction 0, vol_dir -1, .90) is
        a qualified candle — a temporary LP; one with vol_dir +1 (3) is not, nor is a split candle (4). Until 8c the
        sequence stopped at 1 (on the window: sub 3's M15 2907 / 2913 / 3252, PLAN_V §3.1)."""
        spec = [_q(1.00), _q(0.95), (0, -1, 0.90), (0, 1, 0.80), (-1, 1, 0.70)]
        assert _seq(spec, 0, _cts_confirmed(1, pattern_anchor_idx=1), 4) == [(1, 1), (2, 2)]

    def test_no_qualified_candle_no_lp(self):
        spec = [_q(1.00), _N, _N, (1, 1, 0.95)]
        assert _seq(spec, 0, _cts_confirmed(2, pattern_anchor_idx=1), 3) == []


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

        ev = _cts_confirmed()
        return df, zone, wave_candles, ev

    def test_creates_record_with_locked_breakout(self):
        df, zone, wcs, ev = self._setup_basic()
        tracker = WVMITracker()
        rec = tracker.on_cts_confirmed(ev, df, wcs, [zone])

        assert rec is not None
        assert rec.bos_structure_id == 0
        assert rec.bos_cycle_id == 1
        assert rec.zone_side == "buy"
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
        ev = _cts_confirmed()

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
    def test_shifts_lp_to_a_further_close(self):
        """When a qualified candle closing further (.91 < .95) appears in the extended df, the LP shifts to it."""
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
        ev = _cts_confirmed()

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

        updated = tracker.update_temporary_lp(df_extended)
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
        cts_ev = _cts_confirmed()
        tracker.on_cts_confirmed(cts_ev, df, wcs, [zone])

        # Lock via BOS_n+1
        bos_wc_2 = _make_wc(0, 2, "BOS", "buy", last_idx=15, first_idx=16)
        bos_ev = _make_event("BOS_CONFIRMED", 16, meta={"structure_id": 0, "cycle_id": 2})
        tracker.on_bos_confirmed(bos_ev, df, wcs + [bos_wc_2])

        rec_before = tracker.get_records()[0]
        assert rec_before.lp_locked is True
        lp_before = rec_before.lp_idx

        # Update should not change locked record
        updated = tracker.update_temporary_lp(df)
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
        cts_ev = _cts_confirmed()
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

    def _lock(self, capsys, bos2_last):
        """`test_locks_previous_cycle`'s setup (temp LP 12) on a named tracker, BOS_2's last wave candle `bos2_last`
        (12, 18 or None): the lock's printed lines."""
        rows = [{"direction": 1, "c": 1.05, "volume": 100, "vol_dir": 0} for _ in range(25)]
        rows[3] = {**rows[3], "volume": 200, "is_big_normal_as0": 1, "candle_type": "maru"}
        rows[8] = {**rows[8], "volume": 300, "is_big_normal_as0": 1}
        rows[10] = {**rows[10], "direction": -1, "volume": 150, "vol_dir": -1}
        rows[12] = {**rows[12], "direction": -1, "volume": 120, "c": 0.92, "vol_dir": -1}
        rows[18] = {**rows[18], "direction": -1, "volume": 180, "c": 0.93, "vol_dir": -1}
        df = _make_df(rows)
        zone = _make_zone("buy", 1.0, 0.9, "BOS", {"structure_id": 0, "cycle_id": 1, "anchor_idx": 5, "outer": 0.9})
        wcs = [_make_wc(0, 1, "BOS", "buy", last_idx=2, first_idx=3), _make_wc(0, 1, "CTS", "sell", last_idx=8, first_idx=10),
               _make_wc(0, 2, "BOS", "buy", last_idx=bos2_last, first_idx=19)]
        tracker = WVMITracker(structure_path_id="H1.main")
        tracker.on_cts_confirmed(_cts_confirmed(), df, wcs, [zone])
        capsys.readouterr()
        (rec,) = tracker.on_bos_confirmed(_make_event("BOS_CONFIRMED", 20, meta={"structure_id": 0, "cycle_id": 2}), df, wcs)
        return rec, capsys.readouterr().out

    def test_a_lock_lp_other_than_the_last_temporary_lp_prints_a_warning(self, capsys):
        """The locked LP (BOS_2's last wave candle 18) is not the last temporary LP 12: one WARNING line, the LP swapped."""
        rec, out = self._lock(capsys, 18)
        assert rec.lp_idx == 18
        assert [ln for ln in out.splitlines() if "WARNING" in ln] == [
            "[wvmi] WARNING H1.main sid=0 cycle=1: the locked LP 18 (BOS cycle=2's last wave candle) "
            "!= the last temporary LP 12"]

    @pytest.mark.parametrize("bos2_last", [12, None], ids=["same_lp", "no_lock_lp"])
    def test_no_warning_when_the_lock_keeps_the_last_temporary_lp(self, capsys, bos2_last):
        """The lock's LP IS the last temporary LP, or BOS_2 has no last wave candle (the temp LP is kept): no warning."""
        rec, out = self._lock(capsys, bos2_last)
        assert rec.lp_idx == 12 and rec.lp_locked
        assert "WARNING" not in out

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
        cts_ev = _cts_confirmed()
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
# structure_path_id attribution (Part 4 §8.7)
# ---------------------------------------------------------------------------

class TestStructurePathIdAttribution:
    def test_records_carry_path_id_from_tracker(self):
        tracker = WVMITracker(structure_path_id="H1.main")
        # Empty tracker — verify field is plumbed through to new records when
        # they would be emitted. Direct construction smoke test:
        record = WVMIRecord(
            bos_structure_id=0, bos_cycle_id=1,
            zone_side="buy",
            structure_path_id="H1.main >> M15.confluence",
        )
        assert record.structure_path_id == "H1.main >> M15.confluence"

    def test_default_path_id_is_none(self):
        tracker = WVMITracker()
        # No explicit path_id → tracker carries None internally; records
        # emitted by it will have structure_path_id=None.
        assert tracker._structure_path_id is None


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
        ev = _cts_confirmed()

        tracker = WVMITracker()
        rec = tracker.on_cts_confirmed(ev, df, wcs, [zone])

        assert rec is not None
        # Sell zone: buy_momentum = pullback, sell_momentum = breakout
        assert rec.sell_momentum == rec.breakout_momentum
        assert rec.buy_momentum == rec.pullback_momentum
