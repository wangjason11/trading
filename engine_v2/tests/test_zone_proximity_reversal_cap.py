"""The zone-proximity scan ends at the REALISED reversal (2026-09-28; zones-audit "Still open": the scan cap).

`check_zone_proximity` capped every cycle of a sid at `apply_idx - 1` of the sid's LAST `REVERSAL_CANDIDATE` — a
SCHEDULED apply. A candidate is emitted at scheduling and survives rewinds; a watch expiry (run before the pending
apply in the per-candle step) discards it; a step-anchor reversal under the frozen threshold can realise EARLIER than
the candidate's apply. So a discarded LAST candidate capped every later cycle below its own scan start (no sd /
opp_sd triggers, no WVMI gate record, no var2/var3/var4 triggers or M15 subs), and cut the cycle it fell in short.
Now (user decision A) the cap is `structure_lifecycle.compute_reversal_idx_by_sid` — the `STATE_CHANGED(to=reversal)`
candle, the map KL / POI / fib / the orchestrator read (MAIN B, `test_orchestrator_reversal_source.py`). The H1-main
`SidRecord.end_event_idx` moved with it (`test_sid_records.py`). Reference window: 1 H1 candidate, realised at its
apply (902) — every scan window unchanged (measured).
"""
from __future__ import annotations

import contextlib
import copy
import io

import pytest

from engine_v2.multitf.sid_records import build_sid_records_for_main
from engine_v2.pipeline.orchestrator import _run_downstream_pipeline
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.structure.structure_engine import compute_bounded_structure
from engine_v2.tests.test_ms_stop_after_cts import _make_double_rewind_data
from engine_v2.tests.test_unified_probe import _prepare_df
from engine_v2.tests.test_zone_proximity import _bos_zone, _cts_confirmed, _make_df
from engine_v2.zones.zone_proximity import check_zone_proximity


def _run(end_idx, mutate=None):
    """The real MS on `_make_double_rewind_data` (sd +1) bounded at `end_idx`, then the H1 downstream pipeline."""
    with contextlib.redirect_stdout(io.StringIO()):
        res = compute_bounded_structure(_prepare_df(_make_double_rewind_data()), 0, 1, end_idx=end_idx)
        events = copy.deepcopy(res.events)
        if mutate:
            mutate(events)
        out = _run_downstream_pipeline(res.df, events, 1, fib_mode="h1", lifecycle_cap=end_idx)
    return events, out


def _cands(events):
    return [(e.idx, e.meta["apply_idx"]) for e in events if e.type == "REVERSAL_CANDIDATE"]


def _reversals(events):
    return [e.idx for e in events if e.type == "STATE_CHANGED" and e.meta.get("to") == "reversal"]


def _triggers(out):
    return {k: [(t.idx, t.direction) for t in v] for k, v in out["zone_proximity_triggers"].items()}


_CYCLE0 = [(4, "sd"), (6, "opp_sd"), (7, "sd"), (8, "opp_sd"), (9, "sd"), (10, "opp_sd"), (11, "sd")]


def test_a_discarded_last_candidate_no_longer_skips_the_later_cycle():
    """Bounded at 16: candle 4 close-breaks BOS_0 and schedules a reversal (candidate apply 9); at 9 the watch expiry
    runs first and discards it; cycle 1 is re-established at 12 and CTS-confirmed at 14; nothing ever reverses.
    The old cap (9 - 1 = 8) cut cycle 0 to [4, 8] and skipped cycle 1 ([14, 15] with 14 > 8): no trigger, no WVMI
    gate record for cycle 1. Now cycle 0 scans to its next BOS (12 - 1) and cycle 1 to the data end."""
    events, out = _run(16)
    assert _cands(events) == [(4, 9)] and _reversals(events) == []                 # a dead LAST candidate
    assert [(e.meta["cycle_id"], e.idx) for e in events if e.type == "CTS_CONFIRMED"] == [(0, 4), (1, 14)]
    assert _triggers(out) == {(0, 0): _CYCLE0, (0, 1): [(14, "sd")]}                # was (0,0)[:4], (0,1) absent
    assert [(r.bos_structure_id, r.bos_cycle_id) for r in out["wvmi_records"]] == [(0, 0), (0, 1)]  # was [(0, 0)]
    (rec,) = build_sid_records_for_main(events)
    assert (rec.end_event_idx, rec.end_reason) == (None, None)                     # was (9, "reversal")


def test_a_discarded_candidate_inside_a_cycle_no_longer_cuts_it_short():
    """Bounded at 13 (cycle 1 not yet CTS-confirmed): cycle 0 scans to its next BOS (12 - 1), not to 9 - 1."""
    events, out = _run(13)
    assert _cands(events) == [(4, 9)] and _reversals(events) == []
    assert _triggers(out) == {(0, 0): _CYCLE0}


def test_a_realised_reversal_still_caps_the_scan():
    """Positive control, the full fixture: the J2 candidate realises at its apply 17; the last cycle scans to 16.
    With that candidate's apply moved to 19 (a step-anchor reversal realising EARLIER than the candidate) the cap
    stays 16 — on this fixture no candle in 17..18 approaches a zone, so the unit test below pins the overshoot."""
    for mutate in (None, _apply_17_to_19):
        events, out = _run(None, mutate)
        assert _reversals(events) == [17]
        assert _triggers(out) == {(0, 0): _CYCLE0, (0, 1): [(14, "sd")]}
        (rec,) = build_sid_records_for_main(events)
        assert (rec.end_event_idx, rec.end_reason) == (17, "reversal")


def _apply_17_to_19(events):
    for e in events:
        if e.type == "REVERSAL_CANDIDATE" and e.meta["apply_idx"] == 17:
            e.meta["apply_idx"] = 19


# --- unit level (the `test_zone_proximity` helpers: sd +1, BOS zone inner 1.40, a 5-pip wick = an sd trigger) ------

def _candidate(anchor, apply_idx, sid=0):
    """A candidate the MS could emit: its apply lies inside the watch (`expires_idx` = anchor + range_max_k 5;
    `_schedule_reversal_from_anchor` drops a later apply)."""
    assert anchor < apply_idx <= anchor + 5
    return StructureEvent(idx=anchor, category="STRUCTURE", type="REVERSAL_CANDIDATE", price=1.39,
                          meta={"structure_id": sid, "apply_idx": apply_idx, "pattern_anchor_idx": anchor,
                                "expires_idx": anchor + 5})


def _reversal(idx, sid=0):
    return StructureEvent(idx=idx, category="STATE", type="STATE_CHANGED", price=None,
                          meta={"structure_id": sid, "from": "pullback", "to": "reversal"})


def _scan(events, wick_at):
    df = _make_df(20, default_h=1.50, default_l=1.49)
    df.at[wick_at, "l"] = 1.4005
    kl = [_bos_zone(sid=0, cycle_id=0, sd=1, inner=1.40, outer=1.39)]
    out = check_zone_proximity(df=df, sorted_events=events, kl_zones=kl, poi_zones=[], pip_size=0.0001,
                               timeframe="H1")
    return [(k, t.idx, t.direction) for k, v in out.items() for t in v]


@pytest.mark.parametrize("wick_at, expected", [(9, [((0, 0), 9, "sd")]), (10, []), (11, [])])
def test_the_scan_stops_before_the_realised_reversal_not_the_candidates_apply(wick_at, expected):
    """CTS confirmed at 5; the reversal realises at 10, the candidate's scheduled apply is 12. The old cap
    (12 - 1 = 11) let the scan run past the reversal: a wick at 10 or 11 fired on a dead structure."""
    events = [_cts_confirmed(5), _candidate(7, 12), _reversal(10)]
    assert _scan(events, wick_at) == expected


def test_a_dead_candidate_caps_nothing():
    """A candidate with no realised reversal (its watch expired) caps no cycle: the wick at 15 fires (old: none)."""
    events = [_cts_confirmed(5), _candidate(6, 9)]
    assert _scan(events, 15) == [((0, 0), 15, "sd")]
