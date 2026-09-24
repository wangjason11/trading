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
    # CTS_CONFIRMED carries cts_anchor_idx (the CTS extreme) distinct from
    # idx (the confirmation candle). var-1 probe end_idx reads cts_anchor_idx.
    # Default it to idx for tests that don't exercise the distinction; tests
    # that do pass an explicit (earlier) cts_anchor_idx via **extra.
    if type_ == "CTS_CONFIRMED":
        meta["cts_anchor_idx"] = idx
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
        # CTS extreme at 35; confirmed at candle 40 (the CTS_CONFIRMED idx).
        _ev(40, "CTS_CONFIRMED",   sid=0, cycle=1, sd=1,
            cts_anchor_idx=35, confirmation_method="pullback"),
    ]
    triggers = detect_first_confluence_triggers(events)
    assert len(triggers) == 1
    t = triggers[0]
    assert t.parent_sid == 0
    assert t.parent_cycle_id == 1
    assert t.parent_sd == 1
    assert t.input_idx == 20             # BOS extreme
    assert t.trigger_event_idx == 22     # confirmed_at
    assert t.probe_end_idx == 35               # CTS extreme (cts_anchor_idx), NOT the
                                         # confirmation candle (CTS_CONFIRMED.idx==40)
    assert t.status == "finalized"
    assert t.meta["cts_confirmation_method"] == "pullback"


def test_end_idx_is_cts_extreme_not_confirmation_candle():
    """Probe end_idx must be the CTS extreme (cts_anchor_idx), which is
    earlier than the confirmation candle (CTS_CONFIRMED.idx). The confirmation
    candle only gates *when* the value is known, not the value itself (§4.3.2).
    """
    events = [
        _ev(20, "BOS_CONFIRMED", sid=0, cycle=1, sd=1, confirmed_at=22),
        # CTS extreme at 38, but it isn't CONFIRMED until candle 45.
        _ev(45, "CTS_CONFIRMED", sid=0, cycle=1, sd=1, cts_anchor_idx=38),
    ]
    triggers = detect_first_confluence_triggers(events)
    assert len(triggers) == 1
    assert triggers[0].probe_end_idx == 38          # the CTS extreme
    assert triggers[0].status == "finalized"  # resolved once CTS_CONFIRMED fired


def test_bos_without_cts_is_pending():
    events = [
        _ev(20, "BOS_CONFIRMED", sid=0, cycle=1, sd=-1,
            price=0.590, confirmed_at=20),
    ]
    triggers = detect_first_confluence_triggers(events)
    assert len(triggers) == 1
    assert triggers[0].status == "pending"
    assert triggers[0].probe_end_idx is None


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
        _ev(30, "CTS_CONFIRMED", sid=0, cycle=1, sd=1, cts_anchor_idx=28),
        _ev(50, "BOS_CONFIRMED", sid=1, cycle=1, sd=-1, confirmed_at=50),
        _ev(70, "CTS_CONFIRMED", sid=1, cycle=1, sd=-1, cts_anchor_idx=66),
    ]
    triggers = detect_first_confluence_triggers(events)
    assert len(triggers) == 2
    assert triggers[0].parent_sd == 1
    assert triggers[0].probe_end_idx == 28     # CTS extreme, not confirmation candle (30)
    assert triggers[1].parent_sd == -1
    assert triggers[1].probe_end_idx == 66     # CTS extreme, not confirmation candle (70)


def test_skips_zero_sd():
    events = [
        _ev(20, "BOS_CONFIRMED", sid=0, cycle=1, sd=0, confirmed_at=20),
    ]
    assert detect_first_confluence_triggers(events) == []
