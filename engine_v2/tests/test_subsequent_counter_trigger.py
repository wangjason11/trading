"""Unit tests for subsequent_counter (var 4) trigger detection."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from engine_v2.multitf.subsequent_counter_trigger import (
    detect_subsequent_counter_triggers,
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


def _df_with_highs(high_by_idx: Dict[int, float], n: int = 100) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "time": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=i),
            "o": 0.6010,
            "h": high_by_idx.get(i, 0.6020),
            "l": 0.6000,
            "c": 0.6010, "volume": 100,
        })
    df = pd.DataFrame(rows)
    df.attrs["pair"] = "NZD_USD"
    return df


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


def test_no_triggers_when_only_var2_fires():
    """A cycle with only the first sd trigger has no var 4 (need index >= 2)."""
    events = [_ev(20, "CTS_CONFIRMED", sid=0, cycle=1, sd=1)]
    triggers: Dict[Tuple[int, int], List[ZoneProximityTrigger]] = {
        (0, 1): [_trig(0, 1, "sd", 30)],   # only var 2
    }
    out = detect_subsequent_counter_triggers(events, triggers, _df_with_highs({}))
    assert out == []


def test_no_triggers_when_only_var2_and_var3_fire():
    """[sd, opp_sd] has no var 4 — need a third sd trigger to form Λ/V."""
    events = [_ev(20, "CTS_CONFIRMED", sid=0, cycle=1, sd=1)]
    triggers = {
        (0, 1): [
            _trig(0, 1, "sd", 30),
            _trig(0, 1, "opp_sd", 50, zone_kind="CTS"),
        ],
    }
    out = detect_subsequent_counter_triggers(events, triggers, _df_with_highs({}))
    assert out == []


def test_emits_one_per_sd_at_index_2_or_higher():
    """[sd, opp_sd, sd] forms one Λ; the trailing sd is the var 4."""
    events = [_ev(20, "CTS_CONFIRMED", sid=0, cycle=1, sd=1)]
    triggers = {
        (0, 1): [
            _trig(0, 1, "sd", 30),
            _trig(0, 1, "opp_sd", 50, zone_kind="CTS"),
            _trig(0, 1, "sd", 70),
        ],
    }
    out = detect_subsequent_counter_triggers(events, triggers, _df_with_highs({}))
    assert len(out) == 1
    t = out[0]
    assert t.parent_sid == 0
    assert t.parent_cycle_id == 1
    assert t.parent_sd == 1
    assert t.end_idx == 70
    assert t.trigger_event_idx == 70
    assert t.meta["prior_sd_trigger_idx"] == 30
    assert t.meta["prior_cts_prox_idx"] == 50
    assert t.meta["prior_sd_zone_kind"] == "BOS"
    assert t.meta["this_sd_zone_kind"] == "BOS"
    assert t.meta["sequence_index_in_cycle"] == 2


def test_input_idx_is_highest_high_for_bullish_parent():
    """Bullish parent: Λ apex toward CTS = highest high in window."""
    events = [_ev(20, "CTS_CONFIRMED", sid=0, cycle=1, sd=1)]
    triggers = {
        (0, 1): [
            _trig(0, 1, "sd", 30),
            _trig(0, 1, "opp_sd", 50, zone_kind="CTS"),
            _trig(0, 1, "sd", 70),
        ],
    }
    df = _df_with_highs({30: 0.6030, 45: 0.6080, 55: 0.6090, 70: 0.6020})
    out = detect_subsequent_counter_triggers(events, triggers, df)
    assert out[0].input_idx == 55  # highest high in [30, 70]


def test_input_idx_is_lowest_low_for_bearish_parent():
    """Bearish parent: V trough toward CTS = lowest low in window."""
    events = [_ev(20, "CTS_CONFIRMED", sid=0, cycle=1, sd=-1)]
    triggers = {
        (0, 1): [
            _trig(0, 1, "sd", 30),
            _trig(0, 1, "opp_sd", 50, zone_kind="CTS"),
            _trig(0, 1, "sd", 70),
        ],
    }
    df = _df_with_lows({30: 0.6010, 45: 0.5970, 55: 0.5950, 70: 0.6005})
    out = detect_subsequent_counter_triggers(events, triggers, df)
    assert out[0].parent_sd == -1
    assert out[0].input_idx == 55


def test_multiple_sequential_var4_per_cycle():
    """[sd, opp_sd, sd, opp_sd, sd] yields two var 4 triggers (indices 2 and 4)."""
    events = [_ev(20, "CTS_CONFIRMED", sid=0, cycle=1, sd=1)]
    triggers = {
        (0, 1): [
            _trig(0, 1, "sd", 30),
            _trig(0, 1, "opp_sd", 40, zone_kind="CTS"),
            _trig(0, 1, "sd", 55),
            _trig(0, 1, "opp_sd", 70, zone_kind="CTS"),
            _trig(0, 1, "sd", 85),
        ],
    }
    out = detect_subsequent_counter_triggers(events, triggers, _df_with_highs({}))
    assert len(out) == 2
    assert [t.end_idx for t in out] == [55, 85]
    assert out[0].meta["prior_sd_trigger_idx"] == 30
    assert out[0].meta["prior_cts_prox_idx"] == 40
    assert out[1].meta["prior_sd_trigger_idx"] == 55
    assert out[1].meta["prior_cts_prox_idx"] == 70
    assert out[0].meta["sequence_index_in_cycle"] == 2
    assert out[1].meta["sequence_index_in_cycle"] == 4


def test_var4_window_includes_endpoints():
    """`input_idx` may equal prior_sd_idx or this_sd_idx if extreme is at the edge."""
    events = [_ev(20, "CTS_CONFIRMED", sid=0, cycle=1, sd=1)]
    triggers = {
        (0, 1): [
            _trig(0, 1, "sd", 30),
            _trig(0, 1, "opp_sd", 50, zone_kind="CTS"),
            _trig(0, 1, "sd", 70),
        ],
    }
    # Highest high lands exactly on the prior sd-trigger candle (idx 30).
    df = _df_with_highs({30: 0.6090, 50: 0.6020, 70: 0.6020})
    out = detect_subsequent_counter_triggers(events, triggers, df)
    assert out[0].input_idx == 30


def test_lifecycle_end_uses_next_cycle_bos_confirmed_at():
    events = [
        _ev(20, "CTS_CONFIRMED",  sid=0, cycle=1, sd=1),
        _ev(80, "BOS_CONFIRMED",  sid=0, cycle=2, sd=1, confirmed_at=85),
    ]
    triggers = {
        (0, 1): [
            _trig(0, 1, "sd", 30),
            _trig(0, 1, "opp_sd", 50, zone_kind="CTS"),
            _trig(0, 1, "sd", 70),
        ],
    }
    out = detect_subsequent_counter_triggers(events, triggers, _df_with_highs({}))
    assert out[0].lifecycle_end_idx == 85


def test_lifecycle_end_uses_reversal_when_no_next_bos():
    events = [
        _ev(20, "CTS_CONFIRMED",      sid=0, cycle=1, sd=1),
        _ev(75, "REVERSAL_CANDIDATE", sid=0, cycle=1, sd=1, apply_idx=78),
    ]
    triggers = {
        (0, 1): [
            _trig(0, 1, "sd", 30),
            _trig(0, 1, "opp_sd", 50, zone_kind="CTS"),
            _trig(0, 1, "sd", 70),
        ],
    }
    out = detect_subsequent_counter_triggers(events, triggers, _df_with_highs({}))
    assert out[0].lifecycle_end_idx == 78


def test_skips_cycle_without_struct_direction():
    """No CTS_CONFIRMED for a cycle → no parent_sd known → skip."""
    events: List[StructureEvent] = []
    triggers = {
        (0, 1): [
            _trig(0, 1, "sd", 30),
            _trig(0, 1, "opp_sd", 50, zone_kind="CTS"),
            _trig(0, 1, "sd", 70),
        ],
    }
    out = detect_subsequent_counter_triggers(events, triggers, _df_with_highs({}))
    assert out == []


def test_results_sorted_by_trigger_event_idx():
    """Cross-cycle triggers come out time-ordered."""
    events = [
        _ev(20, "CTS_CONFIRMED", sid=0, cycle=1, sd=1),
        _ev(50, "CTS_CONFIRMED", sid=0, cycle=2, sd=1),
    ]
    triggers = {
        (0, 2): [
            _trig(0, 2, "sd", 60),
            _trig(0, 2, "opp_sd", 75, zone_kind="CTS"),
            _trig(0, 2, "sd", 90),
        ],
        (0, 1): [
            _trig(0, 1, "sd", 25),
            _trig(0, 1, "opp_sd", 35, zone_kind="CTS"),
            _trig(0, 1, "sd", 45),
        ],
    }
    out = detect_subsequent_counter_triggers(events, triggers, _df_with_highs({}))
    assert [t.trigger_event_idx for t in out] == [45, 90]
