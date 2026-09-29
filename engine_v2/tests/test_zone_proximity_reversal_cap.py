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


# `_make_double_rewind_data` since 2026-09-29 (MARKET_STRUCTURE_SPEC "A new cycle ends an open watch"): cycle 1 is
# established at 8 inside the watch opened at 4 and ends it (its candidate, apply 9, dropped); CTS confirmations
# 4 / 9 / 14, cycles 0 / 1 / 2 at 2 / 8 / 12. Until then the expiry at 9 discarded the candidate and cycle 1 came at 12.
_CYCLE0 = [(4, "sd"), (6, "opp_sd")]
_CYCLE1 = [(9, "sd"), (10, "opp_sd")]


def test_a_discarded_last_candidate_no_longer_skips_the_later_cycle():
    """Bounded at 16: candle 4 close-breaks BOS_0 and schedules a reversal (candidate apply 9); cycle 1 (established
    at 8 inside that watch) ends it and drops the candidate; cycle 2 at 12, CTS-confirmed at 14; nothing ever
    reverses. The old cap (9 - 1 = 8) skipped cycles 1 ([9, 11]) and 2 ([14, 15]), both past 8: no trigger, no WVMI
    gate record. Now each cycle scans to its next BOS - 1 and cycle 2 to the data end."""
    events, out = _run(16)
    assert _cands(events) == [(4, 9)] and _reversals(events) == []                 # a dead LAST candidate
    assert [(e.meta["cycle_id"], e.idx) for e in events if e.type == "CTS_CONFIRMED"] == [(0, 4), (1, 9), (2, 14)]
    assert _triggers(out) == {(0, 0): _CYCLE0, (0, 1): _CYCLE1, (0, 2): [(14, "sd")]}   # old: (0,1), (0,2) absent
    assert [(r.bos_structure_id, r.bos_cycle_id) for r in out["wvmi_records"]] == [(0, 0), (0, 1), (0, 2)]
    (rec,) = build_sid_records_for_main(events)
    assert (rec.end_event_idx, rec.end_reason) == (None, None)                     # was (9, "reversal")


def test_a_dead_candidate_at_the_last_cycles_confirmation_no_longer_skips_it():
    """Bounded at 13 (cycle 2 not yet CTS-confirmed): the dead candidate's apply 9 is the LAST cycle's own CTS
    confirmation; the old cap (8) skipped that cycle. Until 2026-09-29 the apply fell inside cycle 0 and the old cap cut
    it short — on this fixture that mode is gone (the new cycle at 8 ends the watch); `test_a_dead_candidate_caps_nothing`
    pins it at unit level."""
    events, out = _run(13)
    assert _cands(events) == [(4, 9)] and _reversals(events) == []
    assert _triggers(out) == {(0, 0): _CYCLE0, (0, 1): _CYCLE1}


def test_a_realised_reversal_still_caps_the_scan():
    """Positive control, the full fixture: the J2 candidate realises at its apply 17; the last cycle scans to 16.
    With that candidate's apply moved to 19 (a step-anchor reversal realising EARLIER than the candidate) the cap
    stays 16 — on this fixture no candle in 17..18 approaches a zone, so the unit test below pins the overshoot."""
    for mutate in (None, _apply_17_to_19):
        events, out = _run(None, mutate)
        assert _reversals(events) == [17]
        assert _triggers(out) == {(0, 0): _CYCLE0, (0, 1): _CYCLE1, (0, 2): [(14, "sd")]}
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
