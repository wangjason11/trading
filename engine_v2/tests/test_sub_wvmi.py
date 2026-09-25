"""Unit tests for engine_v2.multitf.sub_wvmi (Part 4 §8.3 / §8.4)."""
from __future__ import annotations

import pandas as pd
import pytest

from engine_v2.common.types import KLZone
from engine_v2.multitf.sub_wvmi import (
    ParentTrigger,
    compute_parent_driven_sub_wvmi,
)
from engine_v2.multitf.types import LowerTFResult, MultiTFTrigger
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.tests._event_factory import make_event
from engine_v2.zones.wave_candles import WaveCandleResult


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _make_df(rows: list[dict]) -> pd.DataFrame:
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
               sid: int = 0, cycle_id: int = 1) -> KLZone:
    outer = bottom if side == "buy" else top
    return KLZone(
        start_time=pd.Timestamp("2024-01-01", tz="UTC"),
        end_time=None,
        side=side,
        top=top,
        bottom=bottom,
        source_kind=source_kind,
        source_time=pd.Timestamp("2024-01-05", tz="UTC"),
        source_price=1.0,
        meta={
            "structure_id": sid,
            "cycle_id": cycle_id,
            "anchor_idx": 5,
            "base_pattern": "base",
            "outer": outer,
        },
    )


def _make_event(etype: str, idx: int, sid: int = 0, cycle_id: int = 1,
                price: float | None = None) -> StructureEvent:
    meta = {"structure_id": sid, "cycle_id": cycle_id}
    if etype in ("CTS_ESTABLISHED", "BOS_CONFIRMED"):
        # the `idx` argument is the anchor (`make_event`); the moment = it (lag 0).
        meta["confirmed_at"] = idx
    return make_event(etype, idx, price=price, **meta)


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


def _make_trigger(use_case: str, parent_sid: int = 0, parent_cycle_id: int = 1,
                  parent_sd: int = 1) -> MultiTFTrigger:
    lower_sd = -parent_sd if use_case == "first_counter" else parent_sd
    return MultiTFTrigger(
        parent_tf="H1",
        parent_sid=parent_sid,
        parent_cycle_id=parent_cycle_id,
        parent_sd=parent_sd,
        use_case=use_case,
        lower_tf="M15",
        lower_sd=lower_sd,
    )


def _make_basic_buy_result(use_case: str = "first_counter") -> LowerTFResult:
    """Build a minimal LowerTFResult with one CTS_CONFIRMED + one BOS_CONFIRMED.

    Cycle 1 has CTS_CONFIRMED at idx=14 with valid wave candles (FB=3, LB=8,
    FP=10) and an LP candidate at idx=12. Cycle 2 BOS_CONFIRMED at idx=20
    with its own LP wave candle at idx=18.
    """
    rows = []
    for i in range(25):
        rows.append({"direction": 1, "c": 1.05, "volume": 100, "vol_dir": 0})
    rows[3] = {**rows[3], "direction": 1, "volume": 200,
               "is_big_normal_as0": 1, "candle_type": "maru"}
    rows[8] = {**rows[8], "direction": 1, "volume": 300,
               "is_big_normal_as0": 1, "candle_type": "normal"}
    rows[10] = {**rows[10], "direction": -1, "volume": 150, "vol_dir": -1}
    rows[12] = {**rows[12], "direction": -1, "volume": 120,
                "c": 0.92, "vol_dir": -1}
    rows[18] = {**rows[18], "direction": -1, "volume": 180,
                "c": 0.93, "vol_dir": -1}
    df = _make_df(rows)

    zone = _make_zone("buy", 1.0, 0.9, "BOS", sid=0, cycle_id=1)
    wcs = [
        _make_wc(0, 1, "BOS", "buy", last_idx=2, first_idx=3),
        _make_wc(0, 1, "CTS", "sell", last_idx=8, first_idx=10),
        _make_wc(0, 2, "BOS", "buy", last_idx=18, first_idx=19),
    ]
    events = [
        _make_event("CTS_CONFIRMED", 14, sid=0, cycle_id=1),
        _make_event("BOS_CONFIRMED", 20, sid=0, cycle_id=2),
    ]

    return LowerTFResult(
        trigger=_make_trigger(use_case),
        df=df,
        events=events,
        kl_zones=[zone],
        wave_candles=wcs,
        fib_states=[],
        poi_zones=[],
        wvmi_records=[],
        prev_bos_lines=[],
        status="finalized",
        meta={},
    )


# ---------------------------------------------------------------------------
# compute_parent_driven_sub_wvmi tests
# ---------------------------------------------------------------------------

class TestComputeParentDrivenSubWvmi:
    def test_creates_record_for_cts_confirmed(self):
        """CTS_CONFIRMED in sub events produces a WVMI record."""
        result = _make_basic_buy_result()
        parent = ParentTrigger(
            idx=140, event_type="ZONE_PROXIMITY_TRIGGER",
            parent_path_id="H1.main",
        )
        records = compute_parent_driven_sub_wvmi(
            result, "H1.main >> M15.counter", parent,
        )
        assert len(records) == 1
        rec = records[0]
        assert rec.bos_structure_id == 0
        assert rec.bos_cycle_id == 1
        assert rec.structure_path_id == "H1.main >> M15.counter"

    def test_locks_record_at_next_bos(self):
        """BOS_CONFIRMED for cycle n+1 locks cycle n's record."""
        result = _make_basic_buy_result()
        parent = ParentTrigger(
            idx=140, event_type="ZONE_PROXIMITY_TRIGGER",
            parent_path_id="H1.main",
        )
        records = compute_parent_driven_sub_wvmi(
            result, "H1.main >> M15.counter", parent,
        )
        assert len(records) == 1
        rec = records[0]
        assert rec.lp_locked is True
        assert rec.status == "locked"
        assert rec.locked_by_cycle_id == 2

    def test_records_carry_section_8_7_attribution(self):
        """Records carry triggered_by_event_idx, _type, parent_path_id meta."""
        result = _make_basic_buy_result()
        parent = ParentTrigger(
            idx=137, event_type="SUBSEQUENT_CONFLUENCE_TRIGGER",
            parent_path_id="H1.main",
        )
        records = compute_parent_driven_sub_wvmi(
            result, "H1.main >> M15.counter", parent,
        )
        assert records
        rec = records[0]
        assert rec.meta["triggered_by_event_idx"] == 137
        assert rec.meta["triggered_by_event_type"] == "SUBSEQUENT_CONFLUENCE_TRIGGER"
        assert rec.meta["parent_path_id"] == "H1.main"

    def test_no_cts_confirmed_returns_empty(self):
        """Sub with only BOS_CONFIRMED events produces no records."""
        result = _make_basic_buy_result()
        # Filter to only BOS events
        result.events = [
            ev for ev in result.events if ev.type == "BOS_CONFIRMED"
        ]
        parent = ParentTrigger(
            idx=140, event_type="ZONE_PROXIMITY_TRIGGER",
            parent_path_id="H1.main",
        )
        records = compute_parent_driven_sub_wvmi(
            result, "H1.main >> M15.counter", parent,
        )
        assert records == []

    def test_path_id_set_per_call(self):
        """structure_path_id on records reflects the sub_path_id passed in."""
        result = _make_basic_buy_result(use_case="first_confluence")
        parent = ParentTrigger(
            idx=140, event_type="ZONE_PROXIMITY_TRIGGER",
            parent_path_id="H1.main",
        )
        records = compute_parent_driven_sub_wvmi(
            result, "H1.main >> M15.confluence", parent,
        )
        assert records
        assert records[0].structure_path_id == "H1.main >> M15.confluence"

    def test_unlocked_record_when_no_next_bos(self):
        """No BOS_n+1 leaves the record with temp LP and status != locked."""
        result = _make_basic_buy_result()
        # Drop the BOS_CONFIRMED so cycle 1 stays open
        result.events = [
            ev for ev in result.events if ev.type != "BOS_CONFIRMED"
        ]
        parent = ParentTrigger(
            idx=140, event_type="ZONE_PROXIMITY_TRIGGER",
            parent_path_id="H1.main",
        )
        records = compute_parent_driven_sub_wvmi(
            result, "H1.main >> M15.counter", parent,
        )
        assert len(records) == 1
        assert records[0].lp_locked is False
