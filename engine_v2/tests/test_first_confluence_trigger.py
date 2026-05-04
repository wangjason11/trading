"""Unit tests for first_confluence (var 1) trigger detection (Part 4 Step 3a)."""
from __future__ import annotations

from typing import Any, Dict

from engine_v2.multitf.first_confluence_trigger import (
    detect_first_confluence_triggers,
)
from engine_v2.structure.market_structure import StructureEvent


def _ev(idx: int, type_: str, sid: int, cycle: int, sd: int = 1,
        price: float = 0.0, **extra) -> StructureEvent:
    meta: Dict[str, Any] = {
        "structure_id": sid,
        "cycle_id": cycle,
        "struct_direction": sd,
    }
    meta.update(extra)
    return StructureEvent(idx=idx, category="STRUCTURE", type=type_,
                          price=price, meta=meta)


def test_no_bos_no_triggers():
    events = [_ev(5, "CTS_ESTABLISHED", sid=0, cycle=0, sd=1)]
    assert detect_first_confluence_triggers(events) == []


def test_single_bos_with_matching_cts_finalized():
    events = [
        _ev(5,  "CTS_ESTABLISHED", sid=0, cycle=0, sd=1),
        _ev(20, "BOS_CONFIRMED",   sid=0, cycle=1, sd=1,
            price=0.610, confirmed_at=22),
        _ev(35, "CTS_ESTABLISHED", sid=0, cycle=1, sd=1),
        _ev(40, "CTS_CONFIRMED",   sid=0, cycle=1, sd=1,
            confirmation_method="pullback"),
    ]
    triggers = detect_first_confluence_triggers(events)
    assert len(triggers) == 1
    t = triggers[0]
    assert t.parent_sid == 0
    assert t.parent_cycle_id == 1
    assert t.parent_sd == 1
    assert t.input_idx == 20             # BOS extreme
    assert t.trigger_event_idx == 22     # confirmed_at
    assert t.end_idx == 40               # CTS_CONFIRMED idx
    assert t.lifecycle_end_idx is None   # no next BOS, no reversal
    assert t.status == "finalized"
    assert t.meta["cts_confirmation_method"] == "pullback"


def test_lifecycle_end_uses_next_cycle_bos_confirmed_at():
    events = [
        _ev(20, "BOS_CONFIRMED", sid=0, cycle=1, sd=1, confirmed_at=22),
        _ev(40, "CTS_CONFIRMED", sid=0, cycle=1, sd=1),
        # BOS for next cycle: ev.idx = extreme (60), confirmed_at = 65
        _ev(60, "BOS_CONFIRMED", sid=0, cycle=2, sd=1, confirmed_at=65),
        _ev(80, "CTS_CONFIRMED", sid=0, cycle=2, sd=1),
    ]
    triggers = detect_first_confluence_triggers(events)
    assert len(triggers) == 2
    # First trigger ends at next BOS confirmed_at (65), not the extreme idx (60)
    assert triggers[0].lifecycle_end_idx == 65
    # Second trigger has no successor — None
    assert triggers[1].lifecycle_end_idx is None


def test_lifecycle_end_uses_reversal_when_no_next_bos():
    events = [
        _ev(20, "BOS_CONFIRMED",      sid=0, cycle=1, sd=1, confirmed_at=22),
        _ev(40, "CTS_CONFIRMED",      sid=0, cycle=1, sd=1),
        _ev(70, "REVERSAL_CANDIDATE", sid=0, cycle=1, sd=1, apply_idx=72),
    ]
    triggers = detect_first_confluence_triggers(events)
    assert len(triggers) == 1
    assert triggers[0].lifecycle_end_idx == 72


def test_lifecycle_end_takes_min_of_next_bos_and_reversal():
    events = [
        _ev(20, "BOS_CONFIRMED",      sid=0, cycle=1, sd=1, confirmed_at=22),
        _ev(40, "CTS_CONFIRMED",      sid=0, cycle=1, sd=1),
        _ev(60, "BOS_CONFIRMED",      sid=0, cycle=2, sd=1, confirmed_at=85),
        _ev(70, "REVERSAL_CANDIDATE", sid=0, cycle=2, sd=1, apply_idx=72),
    ]
    triggers = detect_first_confluence_triggers(events)
    assert triggers[0].lifecycle_end_idx == 72  # reversal first
    assert triggers[1].lifecycle_end_idx == 72


def test_bos_without_cts_is_pending():
    events = [
        _ev(20, "BOS_CONFIRMED", sid=0, cycle=1, sd=-1,
            price=0.590, confirmed_at=20),
    ]
    triggers = detect_first_confluence_triggers(events)
    assert len(triggers) == 1
    assert triggers[0].status == "pending"
    assert triggers[0].end_idx is None


def test_multiple_bos_emit_in_trigger_event_order():
    events = [
        _ev(50, "BOS_CONFIRMED", sid=0, cycle=2, sd=1, confirmed_at=52),
        _ev(60, "CTS_CONFIRMED", sid=0, cycle=2, sd=1),
        _ev(20, "BOS_CONFIRMED", sid=0, cycle=1, sd=1, confirmed_at=22),
        _ev(40, "CTS_CONFIRMED", sid=0, cycle=1, sd=1),
    ]
    triggers = detect_first_confluence_triggers(events)
    assert [t.trigger_event_idx for t in triggers] == [22, 52]
    assert [t.parent_cycle_id for t in triggers] == [1, 2]


def test_multiple_sids_each_paired_independently():
    events = [
        _ev(20, "BOS_CONFIRMED", sid=0, cycle=1, sd=1, confirmed_at=20),
        _ev(30, "CTS_CONFIRMED", sid=0, cycle=1, sd=1),
        _ev(50, "BOS_CONFIRMED", sid=1, cycle=1, sd=-1, confirmed_at=50),
        _ev(70, "CTS_CONFIRMED", sid=1, cycle=1, sd=-1),
    ]
    triggers = detect_first_confluence_triggers(events)
    assert len(triggers) == 2
    assert triggers[0].parent_sd == 1
    assert triggers[0].end_idx == 30
    assert triggers[1].parent_sd == -1
    assert triggers[1].end_idx == 70


def test_skips_zero_sd():
    events = [
        _ev(20, "BOS_CONFIRMED", sid=0, cycle=1, sd=0, confirmed_at=20),
    ]
    assert detect_first_confluence_triggers(events) == []
