"""Unit tests for per-sid record building (Part 4 Step 3a; Plan C §2.5).

Main-entity records are unchanged by Plan C (`sub_sid = structure_id`,
`sub_id = None`). Subordinate records are ONE ROW PER UNIQUE SUB, built from
the per-sub projection `LowerTFResult` (§6.1) whose `meta` carries the
lifecycle + provenance fields; identity is `sub_id`, `sub_sid` is None and the
parent attribution lives on the record table, not on the `SidRecord`.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import pandas as pd

from engine_v2.multitf.sid_records import (
    build_sid_records_for_main,
    build_sid_records_for_subordinate,
)
from engine_v2.multitf.types import LowerTFResult, MultiTFTrigger, SidRecord
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.tests._event_factory import make_event


def _ev(idx: int, type_: str, sid: int, sd: int = 1, **extra) -> StructureEvent:
    meta: Dict[str, Any] = {"structure_id": sid, "struct_direction": sd}
    # CTS_ESTABLISHED / BOS_CONFIRMED: the `idx` argument is the anchor
    # (`make_event`); the moment defaults to it (lag 0) unless a test passes
    # `confirmed_at` (the event's own idx is the moment, Plan E E4a / E4b).
    if type_ in ("CTS_ESTABLISHED", "BOS_CONFIRMED"):
        meta["confirmed_at"] = idx
    meta.update(extra)
    return make_event(type_, idx, **meta)


# --- main (unchanged by Plan C) -----------------------------------------------

def test_main_single_sid_no_reversal():
    events = [
        _ev(10, "CTS_ESTABLISHED", sid=0, sd=1),
        _ev(15, "BOS_CONFIRMED", sid=0, sd=1, cycle_id=1, confirmed_at=15),
        _ev(20, "CTS_CONFIRMED", sid=0, sd=1, cycle_id=1),
    ]
    out = build_sid_records_for_main(events)
    assert len(out) == 1
    rec = out[0]
    assert rec.sub_sid == 0
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
    assert [r.sub_sid for r in out] == [0, 1]

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
    assert out[0].sub_sid == 0
    assert out[0].creation_event_idx == 5


def test_main_sid_record_shape_unchanged_under_plan_c():
    """Plan C §2.5: `SidRecord` stays ONE class in two roles. A main row is
    constructed exactly as before (`sub_sid = structure_id`) and the new
    sub-only fields default to their empty values."""
    rec = SidRecord(
        sub_sid=0, starting_sd=1, creation_event_idx=10,
        end_event_idx=None, end_reason=None,
    )
    assert rec.sub_sid == 0
    assert rec.sub_id is None
    assert rec.start_idx is None
    assert rec.lenses == ()
    assert rec.relative_dir_segments == ()
    assert rec.parent_sid is None and rec.parent_cycle_id is None

    # The builder's main output carries the same defaults.
    out = build_sid_records_for_main([_ev(10, "CTS_ESTABLISHED", sid=0, sd=1)])
    assert out[0].sub_id is None
    assert out[0].start_idx is None
    assert out[0].lenses == ()


# --- subordinate (Plan C §2.5: one row per unique sub) --------------------------

def _trigger(parent_sid: int, parent_cycle: int, lower_sd: int,
             use_case: str) -> MultiTFTrigger:
    """The projection's `LowerTFResult.trigger` = the sub's FIRST live record's
    `source_trigger` (§6.1) — a `MultiTFTrigger`. Its `use_case` / parent
    fields are informational only; `lower_sd` == the sub's direction."""
    kwargs: Dict[str, Any] = dict(
        parent_tf="H1",
        parent_sid=parent_sid,
        parent_cycle_id=parent_cycle,
        parent_sd=lower_sd if use_case.endswith("confluence") else -lower_sd,
        use_case=use_case,
        lower_tf="M15",
        lower_sd=lower_sd,
    )
    return MultiTFTrigger(**kwargs)


def _projection_result(
    *,
    sub_id: int,
    direction: int,
    starting_idx: int,
    start_idx: int,
    end_idx: Optional[int],
    end_reason: Optional[str],
    natural_reversal_idx: Optional[int],
    lenses: Sequence[str],
    relative_dir_segments: Sequence[Tuple[int, str]],
    first_record: Dict[str, Any],
    n_records: int,
    m15_edge: int = 4400,
) -> LowerTFResult:
    """One per-sub projection result with the §6.1 meta shape."""
    trig = _trigger(
        first_record["parent_sid"], first_record["parent_cycle_id"], direction,
        # A reversal-born first record's source trigger is synthesised from the
        # spawning record's trigger (§4.3) — use a named use_case for the stub.
        {"reversal": "first_confluence"}.get(
            first_record["trigger_type"], first_record["trigger_type"],
        ),
    )
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
            "sub_id": sub_id,
            "m15_start_idx": starting_idx,           # = the pool key's starting_idx (historical)
            "start_idx": start_idx,                  # sub lifecycle start (real-time)
            "end_idx": end_idx,                      # sub lifecycle end; None while open
            # §6.1 df-slice key: `end_idx or m15_edge`. NOT a lifecycle value —
            # the SidRecord must read `end_idx`, never this.
            "m15_end_idx": end_idx if end_idx is not None else m15_edge,
            "end_reason": end_reason,
            "natural_reversal_idx": natural_reversal_idx,
            "lenses": tuple(lenses),
            "relative_dir_segments": tuple(relative_dir_segments),
            "n_records": n_records,
            "first_record": dict(first_record),
            "slice_begin": starting_idx - 50,
        },
    )


def _predicted_table_projections():
    """Three subs from the Plan C predicted table
    (`reference_pool_redesign_groundtruth.md`), `sub_id` in creation order
    (454->0, 1797->1, 2365->2, 2639->3, 3304->4, 3621->5, 3760->6, 4027->7).

    sub 2 = 2365/+1: records conf(0,0) reversal [2470,2611] parent_end and
      conf(0,1) FC [2611,2829] reversal -> sub window [2470,2829] reversal
      (same-candle handover at 2611: max_start=2611, 2611>2611 false -> continuous);
      nat_rev 2829; first live record (min start_idx) = the reversal-born one
      -> parent_bos_anchor_idx None (reversal-born).
    sub 3 = 2639/-1: conf(0,1) reversal [2829,3611] parent_end + ctr(0,1)
      first_counter [2843,3611] parent_end -> sub [2829,3611] parent_end,
      lenses both; first record = the confluence reversal at 2829.
    sub 7 = 4027/+1: ctr(1,2) subsequent_counter [4083,None] + conf(1,2)
      reversal-born [4200,None] -> sub [4083,None] open, lenses both,
      first record = subsequent_counter, trigger_idx 4083 = LOH(1020) = 4*1020+3,
      parent_bos_anchor_idx None (FC-only since PLAN_E E5·4).
    """
    return [
        _projection_result(
            sub_id=2, direction=1, starting_idx=2365, start_idx=2470,
            end_idx=2829, end_reason="reversal", natural_reversal_idx=2829,
            lenses=("confluence",),
            relative_dir_segments=((2470, "confluence"),),
            first_record={
                "lens": "confluence", "parent_sid": 0, "parent_cycle_id": 0,
                "trigger_type": "reversal", "trigger_idx": 2470, "start_idx": 2470,
            },
            n_records=2,
        ),
        _projection_result(
            sub_id=3, direction=-1, starting_idx=2639, start_idx=2829,
            end_idx=3611, end_reason="parent_end", natural_reversal_idx=None,
            lenses=("confluence", "counter"),
            relative_dir_segments=((2829, "counter"),),
            first_record={
                "lens": "confluence", "parent_sid": 0, "parent_cycle_id": 1,
                "trigger_type": "reversal", "trigger_idx": 2829, "start_idx": 2829,
            },
            n_records=2,
        ),
        _projection_result(
            sub_id=7, direction=1, starting_idx=4027, start_idx=4083,
            end_idx=None, end_reason=None, natural_reversal_idx=None,
            lenses=("counter", "confluence"),
            relative_dir_segments=((4083, "counter"),),
            first_record={
                "lens": "counter", "parent_sid": 1, "parent_cycle_id": 2,
                "trigger_type": "subsequent_counter", "trigger_idx": 4083,
                "start_idx": 4083,
            },
            n_records=2,
        ),
    ]


def test_subordinate_one_row_per_unique_sub_with_sub_id_identity():
    """§2.5: one `SidRecord` per projection result (= per unique sub), keyed by
    `sub_id`; `sub_sid` is None; parent attribution is NOT on the row."""
    out = build_sid_records_for_subordinate(_predicted_table_projections())
    assert len(out) == 3
    assert [r.sub_id for r in out] == [2, 3, 7]
    assert all(r.sub_sid is None for r in out)
    assert all(r.parent_sid is None for r in out)
    assert all(r.parent_cycle_id is None for r in out)
    assert [r.starting_sd for r in out] == [1, -1, 1]


def test_subordinate_lifecycle_fields_from_projection_meta():
    """creation_event_idx = starting_idx (historical anchor); start_idx /
    end_event_idx = the sub's real-time lifecycle (`end_idx`, None while open);
    end_reason passes through (None for the open sub — no 'lifecycle_end'
    fallback, that vocabulary is retired)."""
    out = build_sid_records_for_subordinate(_predicted_table_projections())
    assert [r.creation_event_idx for r in out] == [2365, 2639, 4027]
    assert [r.start_idx for r in out] == [2470, 2829, 4083]
    assert [r.end_event_idx for r in out] == [2829, 3611, None]
    assert [r.end_reason for r in out] == ["reversal", "parent_end", None]
    for r in out:
        assert r.end_reason in {"reversal", "same_dir_replacement", "parent_end", None}
        assert r.end_reason != "lifecycle_end"


def test_subordinate_open_sub_reads_end_idx_not_m15_end_idx():
    """The projection meta also carries `m15_end_idx = end_idx or edge` (the df
    slice bound, §6.1). An OPEN sub has `end_idx=None` and must surface as
    `end_event_idx=None` / `end_reason=None` — not as the data edge."""
    res = _projection_result(
        sub_id=7, direction=1, starting_idx=4027, start_idx=4083,
        end_idx=None, end_reason=None, natural_reversal_idx=None,
        lenses=("counter",), relative_dir_segments=((4083, "counter"),),
        first_record={
            "lens": "counter", "parent_sid": 1, "parent_cycle_id": 2,
            "trigger_type": "subsequent_counter", "trigger_idx": 4083,
            "start_idx": 4083,
        },
        n_records=1, m15_edge=4400,
    )
    assert res.meta["m15_end_idx"] == 4400          # fixture sanity: edge is present
    out = build_sid_records_for_subordinate([res])
    assert len(out) == 1
    assert out[0].end_event_idx is None
    assert out[0].end_reason is None
    assert out[0].start_idx == 4083


def test_subordinate_lenses_and_segments_are_tuples():
    out = build_sid_records_for_subordinate(_predicted_table_projections())
    for r in out:
        assert isinstance(r.lenses, tuple)
        assert isinstance(r.relative_dir_segments, tuple)
        assert all(isinstance(seg, tuple) and len(seg) == 2
                   for seg in r.relative_dir_segments)
    assert set(out[0].lenses) == {"confluence"}
    assert set(out[1].lenses) == {"confluence", "counter"}
    assert set(out[2].lenses) == {"counter", "confluence"}
    assert out[0].relative_dir_segments == ((2470, "confluence"),)
    assert out[1].relative_dir_segments == ((2829, "counter"),)
    assert out[2].relative_dir_segments == ((4083, "counter"),)


def test_subordinate_meta_carries_provenance():
    """meta = {natural_reversal_idx, n_records, first_record, slice_begin}
    (§2.5). `first_record` is the sub's first live record `(lens, parent_sid,
    parent_cycle_id, trigger_type, trigger_idx, start_idx)`. (The unread
    `validated_parent_start` was deleted in Plan E E1b; the per-record value is
    `TriggerRecord.parent_bos_anchor_idx`, exported in `_triggers.csv`.)"""
    out = build_sid_records_for_subordinate(_predicted_table_projections())
    for r in out:
        for k in ("natural_reversal_idx", "n_records", "first_record",
                  "slice_begin"):
            assert k in r.meta, f"meta missing {k!r}"

    assert [r.meta["natural_reversal_idx"] for r in out] == [2829, None, None]
    assert [r.meta["n_records"] for r in out] == [2, 2, 2]
    assert [r.meta["slice_begin"] for r in out] == [2315, 2589, 3977]   # starting_idx - 50

    assert out[0].meta["first_record"] == {
        "lens": "confluence", "parent_sid": 0, "parent_cycle_id": 0,
        "trigger_type": "reversal", "trigger_idx": 2470, "start_idx": 2470,
    }
    assert out[2].meta["first_record"] == {
        "lens": "counter", "parent_sid": 1, "parent_cycle_id": 2,
        "trigger_type": "subsequent_counter", "trigger_idx": 4083,
        "start_idx": 4083,
    }


def test_subordinate_preserves_input_order():
    """Rows come out in the order the projections were given (the caller
    passes them in `sub_id` / creation order)."""
    projs = _predicted_table_projections()
    out = build_sid_records_for_subordinate(list(reversed(projs)))
    assert [r.sub_id for r in out] == [7, 3, 2]
