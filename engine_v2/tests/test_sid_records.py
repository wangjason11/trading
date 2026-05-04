"""Unit tests for per-sid record building (Part 4 Step 3a)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from engine_v2.multitf.sid_records import (
    build_sid_records_for_main,
    build_sid_records_for_subordinate,
)
from engine_v2.multitf.types import LowerTFResult, MultiTFTrigger
from engine_v2.structure.market_structure import StructureEvent


def _ev(idx: int, type_: str, sid: int, sd: int = 1, **extra) -> StructureEvent:
    meta: Dict[str, Any] = {"structure_id": sid, "struct_direction": sd}
    meta.update(extra)
    return StructureEvent(idx=idx, category="STRUCTURE", type=type_, meta=meta)


def test_main_single_sid_no_reversal():
    events = [
        _ev(10, "CTS_ESTABLISHED", sid=0, sd=1),
        _ev(15, "BOS_CONFIRMED", sid=0, sd=1, cycle_id=1, confirmed_at=15),
        _ev(20, "CTS_CONFIRMED", sid=0, sd=1, cycle_id=1),
    ]
    out = build_sid_records_for_main(events)
    assert len(out) == 1
    rec = out[0]
    assert rec.sid == 0
    assert rec.starting_sd == 1
    assert rec.creation_event_idx == 10
    assert rec.end_event_idx is None
    assert rec.end_reason is None
    assert rec.parent_sid is None
    assert rec.parent_cycle_id is None


def test_main_multiple_sids_with_reversal():
    events = [
        _ev(10, "CTS_ESTABLISHED", sid=0, sd=1),
        _ev(20, "BOS_CONFIRMED", sid=0, sd=1, confirmed_at=20),
        _ev(30, "REVERSAL_CANDIDATE", sid=0, sd=1, apply_idx=35),
        _ev(36, "CTS_ESTABLISHED", sid=1, sd=-1),
        _ev(45, "BOS_CONFIRMED", sid=1, sd=-1, confirmed_at=45),
    ]
    out = build_sid_records_for_main(events)
    assert [r.sid for r in out] == [0, 1]

    sid0 = out[0]
    assert sid0.starting_sd == 1
    assert sid0.creation_event_idx == 10
    assert sid0.end_event_idx == 35
    assert sid0.end_reason == "reversal"

    sid1 = out[1]
    assert sid1.starting_sd == -1
    assert sid1.creation_event_idx == 36
    assert sid1.end_event_idx is None
    assert sid1.end_reason is None


def test_main_skips_events_without_structure_id():
    events = [
        _ev(5, "CTS_ESTABLISHED", sid=0, sd=1),
        StructureEvent(idx=6, category="RANGE", type="RANGE_STARTED", meta={}),
        _ev(10, "BOS_CONFIRMED", sid=0, sd=1, confirmed_at=10),
    ]
    out = build_sid_records_for_main(events)
    assert len(out) == 1
    assert out[0].sid == 0
    assert out[0].creation_event_idx == 5


def _trigger(parent_sid: int, parent_cycle: int, lower_sd: int) -> MultiTFTrigger:
    return MultiTFTrigger(
        parent_tf="H1",
        parent_sid=parent_sid,
        parent_cycle_id=parent_cycle,
        parent_sd=-lower_sd,
        use_case="first_counter",
        lower_tf="M15",
        lower_sd=lower_sd,
        start_time=pd.Timestamp("2026-01-01", tz="UTC"),
        start_price=0.6000,
        lifecycle_end_idx=None,
    )


def _ltf_result(parent_sid: int, parent_cycle: int, lower_sd: int,
                m15_start: int, m15_end: int) -> LowerTFResult:
    trig = _trigger(parent_sid, parent_cycle, lower_sd)
    return LowerTFResult(
        trigger=trig,
        df=pd.DataFrame(),
        events=[],
        kl_zones=[],
        wave_candles=[],
        fib_states=[],
        poi_zones=[],
        wvmi_records=[],
        prev_bos_lines=[],
        status="finalized",
        meta={
            "m15_start_idx": m15_start,
            "m15_end_idx": m15_end,
            "validated_h1_start": 100,
            "slice_begin": m15_start - 50,
            "use_case": trig.use_case,
        },
    )


def test_subordinate_records_one_per_result():
    results = [
        _ltf_result(parent_sid=0, parent_cycle=1, lower_sd=-1, m15_start=200, m15_end=400),
        _ltf_result(parent_sid=0, parent_cycle=2, lower_sd=-1, m15_start=500, m15_end=700),
        _ltf_result(parent_sid=1, parent_cycle=0, lower_sd=1, m15_start=900, m15_end=1100),
    ]
    out = build_sid_records_for_subordinate(results)
    assert [r.sid for r in out] == [0, 1, 2]
    assert [r.parent_sid for r in out] == [0, 0, 1]
    assert [r.parent_cycle_id for r in out] == [1, 2, 0]
    assert [r.starting_sd for r in out] == [-1, -1, 1]
    assert [r.creation_event_idx for r in out] == [200, 500, 900]
    assert [r.end_event_idx for r in out] == [400, 700, 1100]
    assert all(r.end_reason == "lifecycle_end" for r in out)


def test_subordinate_handles_missing_meta_indices():
    trig = _trigger(parent_sid=0, parent_cycle=1, lower_sd=-1)
    result = LowerTFResult(
        trigger=trig,
        df=pd.DataFrame(),
        events=[],
        kl_zones=[],
        wave_candles=[],
        fib_states=[],
        poi_zones=[],
        wvmi_records=[],
        prev_bos_lines=[],
        status="finalized",
        meta={},
    )
    out = build_sid_records_for_subordinate([result])
    assert len(out) == 1
    assert out[0].creation_event_idx is None
    assert out[0].end_event_idx is None
    assert out[0].end_reason is None
