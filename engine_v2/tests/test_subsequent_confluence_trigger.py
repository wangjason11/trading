"""Unit tests for subsequent_confluence (var 3) trigger detection."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from engine_v2.multitf.subsequent_confluence_trigger import (
    detect_subsequent_confluence_triggers,
)
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.zones.zone_proximity import ZoneProximityTrigger


def _ev(idx: int, type_: str, sid: int, cycle: int, sd: int = 1, **extra) -> StructureEvent:
    meta: Dict[str, Any] = {
        "structure_id": sid,
        "cycle_id": cycle,
        "struct_direction": sd,
    }
    meta.update(extra)
    return StructureEvent(idx=idx, category="STRUCTURE", type=type_, meta=meta)


def _trig(sid: int, cycle: int, direction: str, idx: int,
          zone_kind: str = "BOS") -> ZoneProximityTrigger:
    return ZoneProximityTrigger(
        structure_id=sid, cycle_id=cycle, direction=direction,
        idx=idx, trigger_inner=0.6000, zone_kind=zone_kind,
        proximity_pips=20, pip_size=0.0001, timeframe="H1",
    )


def _df_with_lows(low_by_idx: Dict[int, float], n: int = 100) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "time": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=i),
            "o": 0.6010, "h": 0.6020,
            "l": low_by_idx.get(i, 0.6000),
            "c": 0.6010, "volume": 100,
        })
    df = pd.DataFrame(rows)
    df.attrs["pair"] = "NZD_USD"
    return df


def test_no_triggers_when_no_proximity_alternation():
    events = [_ev(20, "CTS_CONFIRMED", sid=0, cycle=1, sd=1)]
    triggers: Dict[Tuple[int, int], List[ZoneProximityTrigger]] = {
        (0, 1): [_trig(0, 1, "sd", 30)],   # only an sd trigger
    }
    out = detect_subsequent_confluence_triggers(events, triggers, _df_with_lows({}))
    assert out == []


def test_emits_one_per_opp_sd_after_sd():
    events = [_ev(20, "CTS_CONFIRMED", sid=0, cycle=1, sd=1)]
    triggers = {
        (0, 1): [
            _trig(0, 1, "sd", 30),
            _trig(0, 1, "opp_sd", 50, zone_kind="CTS"),
        ],
    }
    out = detect_subsequent_confluence_triggers(events, triggers, _df_with_lows({}))
    assert len(out) == 1
    t = out[0]
    assert t.parent_sid == 0
    assert t.parent_cycle_id == 1
    assert t.parent_sd == 1
    assert t.end_idx == 50
    assert t.trigger_event_idx == 50
    assert t.meta["prior_sd_trigger_idx"] == 30
    assert t.meta["prior_sd_zone_kind"] == "BOS"
    assert t.meta["cts_prox_zone_kind"] == "CTS"


def test_input_idx_is_lowest_low_for_bullish_parent():
    events = [_ev(20, "CTS_CONFIRMED", sid=0, cycle=1, sd=1)]
    triggers = {
        (0, 1): [
            _trig(0, 1, "sd", 30),
            _trig(0, 1, "opp_sd", 50, zone_kind="CTS"),
        ],
    }
    df = _df_with_lows({30: 0.6005, 35: 0.5990, 42: 0.5985, 50: 0.6000})
    out = detect_subsequent_confluence_triggers(events, triggers, df)
    assert out[0].input_idx == 42  # lowest low in [30, 50]


def test_input_idx_is_highest_high_for_bearish_parent():
    events = [_ev(20, "CTS_CONFIRMED", sid=0, cycle=1, sd=-1)]
    triggers = {
        (0, 1): [
            _trig(0, 1, "sd", 30),
            _trig(0, 1, "opp_sd", 50, zone_kind="CTS"),
        ],
    }
    rows = []
    for i in range(60):
        h = 0.6100 if i == 38 else 0.6050
        rows.append({
            "time": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=i),
            "o": 0.6010, "h": h, "l": 0.6000,
            "c": 0.6010, "volume": 100,
        })
    df = pd.DataFrame(rows)
    out = detect_subsequent_confluence_triggers(events, triggers, df)
    assert out[0].parent_sd == -1
    assert out[0].input_idx == 38


def test_multiple_sequential_var3_per_cycle():
    events = [_ev(20, "CTS_CONFIRMED", sid=0, cycle=1, sd=1)]
    triggers = {
        (0, 1): [
            _trig(0, 1, "sd", 30),
            _trig(0, 1, "opp_sd", 40, zone_kind="CTS"),
            _trig(0, 1, "sd", 55),
            _trig(0, 1, "opp_sd", 70, zone_kind="CTS"),
        ],
    }
    out = detect_subsequent_confluence_triggers(events, triggers, _df_with_lows({}))
    assert len(out) == 2
    assert [t.end_idx for t in out] == [40, 70]
    assert out[0].meta["prior_sd_trigger_idx"] == 30
    assert out[1].meta["prior_sd_trigger_idx"] == 55


def test_skips_cycle_without_struct_direction():
    events: List[StructureEvent] = []  # no CTS_CONFIRMED → no sd lookup
    triggers = {
        (0, 1): [
            _trig(0, 1, "sd", 30),
            _trig(0, 1, "opp_sd", 50, zone_kind="CTS"),
        ],
    }
    out = detect_subsequent_confluence_triggers(events, triggers, _df_with_lows({}))
    assert out == []
