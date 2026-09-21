"""Plan C section 3 -- the static parent tables (`multitf/parent_tables.py`).

Written TESTS-FIRST against `plans/PLAN_C_lifecycle_rewrite.md` section 3 and the
session API contract (`PLAN_C_API_CONTRACT.md`):

    rev_by_sid[S]      = compute_reversal_idx_by_sid(events)      # STATE_CHANGED->reversal, NOT REVERSAL_CANDIDATE
    struct_start[S]    = compute_struct_start_by_sid(events, rev_by_sid, None)
    cts_moment[(S,C)]  = CTS_ESTABLISHED.meta["confirmed_at"]      # the MOMENT (== BOS.confirmed_at); NOT .idx; LAST-seen
    parent_sd[S]       = CTS_ESTABLISHED.meta["struct_direction"]
    floor_h1[(S,C)]    = max(struct_start[S], cts_moment[(S,C)])   # the CLAMPED cycle start
    end_h1[(S,C)]      = floor_h1[(S,C+1)] if (S,C+1) exists else rev_by_sid.get(S) else None
    floor_m15 / end_m15 = LOH(...)                                 # every map must succeed (assert)
    degenerate[(S,C)]  = end_m15 is not None and floor_m15 >= end_m15

Asserts (raise, do not degrade): BOS_CONFIRMED(S,C).confirmed_at == CTS_ESTABLISHED(S,C).confirmed_at
(on the MOMENT, never on CTS_ESTABLISHED.idx -- the extreme); a BOS_CONFIRMED without a matching
CTS_ESTABLISHED raises; an LOH map returning None raises.

Pure logic: `h1_df=None, m15_df=None` with an injected `loh=lambda p, h1, m15: 4*p + 3`
(the exact last-of-hour identity on the reference window, groundtruth memory).
The reference-window fixture reproduces the H1 rows of the Plan B save
(`artifacts/commits/week8-volmom-multitf/20260920_104606_189c127/*_structure_events.csv`).
"""
from __future__ import annotations

import dataclasses
import re
from typing import Any, Dict, List, Optional

import pytest

from engine_v2.multitf.parent_tables import ParentTables, build_parent_tables
from engine_v2.structure.market_structure import StructureEvent


# --- fixtures ----------------------------------------------------------------

def _ev(idx: int, type_: str, sid: int, cycle: Optional[int] = None, sd: int = 1,
        **extra: Any) -> StructureEvent:
    meta: Dict[str, Any] = {"structure_id": sid, "struct_direction": sd}
    if cycle is not None:
        meta["cycle_id"] = cycle
    meta.update(extra)
    category = "STATE" if type_ == "STATE_CHANGED" else "STRUCTURE"
    return StructureEvent(idx=idx, category=category, type=type_, meta=meta)


def _cts_est(idx: int, sid: int, cycle: int, confirmed_at: int, sd: int = 1) -> StructureEvent:
    # market_structure._emit_cts_established stamps confirmed_at = the apply candle;
    # `idx` is the CTS EXTREME (historical) and may precede confirmed_at.
    return _ev(idx, "CTS_ESTABLISHED", sid, cycle, sd, confirmed_at=confirmed_at,
               anchor_idx=idx - 1, pattern_name="one_maru_continuous")


def _bos(idx: int, sid: int, cycle: int, confirmed_at: int, sd: int = 1) -> StructureEvent:
    # BOS_CONFIRMED.idx = the BOS extreme; confirmed_at = the same apply candle as the CTS_EST.
    return _ev(idx, "BOS_CONFIRMED", sid, cycle, sd, confirmed_at=confirmed_at,
               source="pullback_extreme")


def _reversal(idx: int, sid: int, sd: int = 1) -> StructureEvent:
    # STATE_CHANGED to=reversal is what compute_reversal_idx_by_sid reads.
    return _ev(idx, "STATE_CHANGED", sid, None, sd,
               **{"from": "pullback_range", "to": "reversal", "reason": "reversal_pattern"})


def _rev_candidate(idx: int, sid: int, apply_idx: int, sd: int = 1) -> StructureEvent:
    # A prediction, not a lifecycle fact: MUST be ignored by the tables.
    return _ev(idx, "REVERSAL_CANDIDATE", sid, None, sd, anchor_idx=idx, apply_idx=apply_idx,
               expires_idx=apply_idx)


def _sorted(events: List[StructureEvent]) -> List[StructureEvent]:
    # The orchestrator's `sorted_events` order (orchestrator.py:147).
    return sorted(events, key=lambda e: (e.idx, e.type))


def _loh(parent_idx: int, h1_df: Any, m15_df: Any) -> Optional[int]:
    # LOH(h) = 4h + 3 -- exact on the reference window (last M15 candle of the H1 hour).
    return 4 * int(parent_idx) + 3


def _build(events: List[StructureEvent], *, loh=_loh, log=None) -> ParentTables:
    return build_parent_tables(_sorted(events), None, None, loh=loh,
                               log=log if log is not None else (lambda s: None))


def _reference_window_events() -> List[StructureEvent]:
    """The H1 facts of the NZD_USD 2025-11-15 -> 2026-01-20 window (groundtruth memory;
    verified row-by-row against the Plan B save's H1 `_structure_events.csv`):

        CTS_ESTABLISHED (S,C).idx / confirmed_at : (0,0)=115/115 (0,1)=652/652 (1,0)=703/703
                                                   (1,1)=748/748 (1,2)=902/902
        BOS_CONFIRMED   (S,C).idx / confirmed_at : (0,0)=96/115  (0,1)=591/652 (1,0)=689/703
                                                   (1,1)=728/748 (1,2)=826/902
        STATE_CHANGED->reversal: sid 0 @ 902.  REVERSAL_CANDIDATE: idx 897, apply 902.
        sid 0 sd=+1, sid 1 sd=-1; sid 1 never reverses.
    """
    return [
        _bos(96, 0, 0, 115), _cts_est(115, 0, 0, 115),
        _bos(591, 0, 1, 652), _cts_est(652, 0, 1, 652),
        _rev_candidate(897, 0, 902), _reversal(902, 0),
        _bos(689, 1, 0, 703, sd=-1), _cts_est(703, 1, 0, 703, sd=-1),
        _bos(728, 1, 1, 748, sd=-1), _cts_est(748, 1, 1, 748, sd=-1),
        _bos(826, 1, 2, 902, sd=-1), _cts_est(902, 1, 2, 902, sd=-1),
    ]


# --- the moment, not the extreme ---------------------------------------------

def test_cts_moment_is_confirmed_at_not_the_extreme():
    # CTS_ESTABLISHED idx 10 (extreme) / confirmed_at 11 (moment); BOS idx 5 / confirmed_at 11.
    t = _build([_bos(5, 0, 0, 11), _cts_est(10, 0, 0, 11)])
    assert t.cts_moment == {(0, 0): 11}           # the MOMENT, not .idx == 10
    # struct_start[0] = min event idx = 5 (the BOS extreme); floor_h1 = max(5, 11) = 11 -- uses 11, not 10
    assert t.struct_start == {0: 5}
    assert t.floor_h1[(0, 0)] == 11
    assert t.floor_m15[(0, 0)] == 4 * 11 + 3 == 47
    assert t.floor(0, 0) == 47
    # single open cycle: no next cycle, no reversal -> end None; never degenerate
    assert t.end_h1[(0, 0)] is None
    assert t.end_m15[(0, 0)] is None
    assert t.end(0, 0) is None
    assert t.degenerate[(0, 0)] is False
    assert t.is_degenerate(0, 0) is False
    assert t.rev_by_sid == {}
    assert t.parent_sd == {0: 1}


def test_cts_moment_is_last_seen_per_cycle():
    # Two CTS_ESTABLISHED for the same (S,C) (a rewind re-emits): LAST-seen wins (plan section 3).
    # PLAN-AMBIGUITY: under a rewind that also re-emits BOS_CONFIRMED for the same (S,C) the plan does
    # not say which BOS the identity assert compares against (each, or the last-seen); this fixture is
    # deliberately CTS-only so it pins only the last-seen rule for cts_moment.
    t = _build([_cts_est(10, 0, 0, 11), _cts_est(12, 0, 0, 13)])
    assert t.cts_moment == {(0, 0): 13}
    # struct_start = min idx = 10; floor = max(10, 13) = 13
    assert t.floor_h1[(0, 0)] == 13


# --- floor = max(struct_start, moment) with the reversal handoff --------------

def test_reversal_handoff_produces_zero_inverted_and_real_cycles():
    # sid 0 (+1): BOS 5/10, CTS (0,0) 10/10, reverses @40.
    # sid 1 (-1): CTS (1,0) 20/20, CTS (1,1) 30/30, reverses @35 (retroactive twice: established and
    #             reversed before sid 0's reversal candle -- the only way an INVERTED cycle can arise
    #             under CLAMPED ends, see PLAN-AMBIGUITY below).
    # sid 2 (+1): CTS (2,0) 45/45, open.
    events = [
        _bos(5, 0, 0, 10), _cts_est(10, 0, 0, 10), _reversal(40, 0),
        _cts_est(20, 1, 0, 20, sd=-1), _cts_est(30, 1, 1, 30, sd=-1), _reversal(35, 1, sd=-1),
        _cts_est(45, 2, 0, 45),
    ]
    t = _build(events)
    assert t.rev_by_sid == {0: 40, 1: 35}
    # struct_start: sid 0 = min idx 5; sid 1 = rev[0] = 40 (handoff overrides min idx 20);
    #               sid 2 = rev[1] = 35 (handoff overrides min idx 45)
    assert t.struct_start == {0: 5, 1: 40, 2: 35}
    # floor_h1 = max(struct_start, moment): (0,0)=max(5,10)=10; (1,0)=max(40,20)=40;
    #            (1,1)=max(40,30)=40; (2,0)=max(35,45)=45
    assert t.floor_h1 == {(0, 0): 10, (1, 0): 40, (1, 1): 40, (2, 0): 45}
    # end_h1: (0,0): no (0,1) -> rev[0]=40; (1,0): floor_h1[(1,1)]=40; (1,1): no (1,2) -> rev[1]=35;
    #         (2,0): no (2,1), sid 2 never reverses -> None
    assert t.end_h1 == {(0, 0): 40, (1, 0): 40, (1, 1): 35, (2, 0): None}
    # LOH = 4p+3: floors 43/163/163/183; ends 163/163/143/None
    assert t.floor_m15 == {(0, 0): 43, (1, 0): 163, (1, 1): 163, (2, 0): 183}
    assert t.end_m15 == {(0, 0): 163, (1, 0): 163, (1, 1): 143, (2, 0): None}
    # degenerate = end_m15 is not None and floor_m15 >= end_m15:
    #   (0,0) 43 >= 163 F (real); (1,0) 163 >= 163 T (ZERO); (1,1) 163 >= 143 T (INVERTED);
    #   (2,0) end None -> F (real, open) -- the real cycle after the two degenerate ones
    assert t.degenerate == {(0, 0): False, (1, 0): True, (1, 1): True, (2, 0): False}
    # PLAN-AMBIGUITY: with end_h1 = the NEXT cycle's CLAMPED start, an inverted (floor > end) cycle can
    # only come from the reversal-end branch (end = rev_by_sid[S] < floor); the groundtruth memory's
    # "inverted (1,0): floor 3611 vs end 2995" refers to the RAW next-CTS end that Plan C no longer uses
    # ((1,0) is ZERO-length under Plan C). Both shapes are pinned here.


def test_end_is_the_next_cycles_clamped_start_not_its_raw_moment():
    # The reference (1,0) case: sid 0 reverses @902 so struct_start[1] = 902; (1,1)'s raw moment is 748.
    events = [
        _bos(96, 0, 0, 115), _cts_est(115, 0, 0, 115), _reversal(902, 0),
        _cts_est(703, 1, 0, 703, sd=-1), _cts_est(748, 1, 1, 748, sd=-1),
    ]
    t = _build(events)
    assert t.cts_moment[(1, 1)] == 748                   # the raw moment is kept in cts_moment ...
    assert t.floor_h1[(1, 1)] == max(902, 748) == 902    # ... but its clamped start is 902
    assert t.end_h1[(1, 0)] == 902                       # end = clamped next start (NOT the raw 748)
    assert t.end_h1[(1, 0)] != 748
    assert t.end_m15[(1, 0)] == 4 * 902 + 3 == 3611
    # (1,0) floor = max(902, 703) = 902 == end 902 -> zero-length -> degenerate
    assert t.floor_h1[(1, 0)] == 902
    assert t.degenerate[(1, 0)] is True
    # (1,1) is the open last cycle of sid 1 (no (1,2), sid 1 never reverses)
    assert t.end_h1[(1, 1)] is None
    assert t.degenerate[(1, 1)] is False


# --- end fallbacks -------------------------------------------------------------

def test_end_falls_back_to_state_changed_reversal_not_reversal_candidate_apply_idx():
    # Last cycle of sid 0: no (0,1). REVERSAL_CANDIDATE idx 37 apply_idx 42 (a prediction) vs the
    # STATE_CHANGED->reversal at 40 (the fact). end_h1 must be 40 -- neither 42 nor 37.
    events = [
        _cts_est(10, 0, 0, 10), _rev_candidate(37, 0, 42), _reversal(40, 0),
    ]
    t = _build(events)
    assert t.rev_by_sid == {0: 40}
    assert t.end_h1[(0, 0)] == 40
    assert t.end_h1[(0, 0)] not in (37, 42)
    assert t.end_m15[(0, 0)] == 4 * 40 + 3 == 163
    # floor = max(10, 10) = 10 -> 43 < 163 -> real
    assert t.degenerate[(0, 0)] is False
    # The candidate event must not create a cycle of its own.
    assert set(t.cycles()) == {(0, 0)}


def test_open_last_cycle_has_no_end_and_is_never_degenerate():
    # No reversal, no next cycle -> end None on both TFs; degenerate requires end_m15 is not None.
    t = _build([_cts_est(10, 0, 0, 10), _cts_est(30, 0, 1, 30)])
    assert t.end_h1[(0, 0)] == 30 and t.end_m15[(0, 0)] == 123    # (0,0) closed by (0,1): 4*30+3
    assert t.end_h1[(0, 1)] is None and t.end_m15[(0, 1)] is None  # (0,1) open
    assert t.degenerate[(0, 1)] is False
    # A reversal of a DIFFERENT sid does not close it either.
    t2 = _build([_cts_est(10, 0, 0, 10), _reversal(40, 0),
                 _cts_est(50, 1, 0, 50, sd=-1)])
    assert t2.end_h1[(1, 0)] is None
    assert t2.degenerate[(1, 0)] is False


# --- degenerate is evaluated on the M15 values --------------------------------

def test_degenerate_is_evaluated_on_m15_values():
    # floor_h1 10 < end_h1 11 on H1, but an LOH map that collapses both hours onto the same M15 candle
    # gives floor_m15 == end_m15 -> degenerate (the rule is `floor_m15 >= end_m15`, not on H1).
    def collapsing_loh(p: int, h1: Any, m15: Any) -> Optional[int]:
        return (int(p) // 2) * 8 + 7          # 10 -> 47, 11 -> 47

    t = _build([_cts_est(10, 0, 0, 10), _cts_est(11, 0, 1, 11)], loh=collapsing_loh)
    assert t.floor_h1[(0, 0)] == 10 and t.end_h1[(0, 0)] == 11     # not degenerate on H1 ...
    assert t.floor_m15[(0, 0)] == 47 and t.end_m15[(0, 0)] == 47   # ... but equal on M15
    assert t.degenerate[(0, 0)] is True
    assert t.is_degenerate(0, 0) is True
    # and with the monotone map the same events are a real cycle: 43 < 47
    t2 = _build([_cts_est(10, 0, 0, 10), _cts_est(11, 0, 1, 11)])
    assert (t2.floor_m15[(0, 0)], t2.end_m15[(0, 0)]) == (43, 47)
    assert t2.degenerate[(0, 0)] is False


# --- asserts (raise, do not degrade) -----------------------------------------

def test_bos_confirmed_at_mismatching_cts_confirmed_at_raises():
    # BOS_CONFIRMED(0,0).confirmed_at 12 != CTS_ESTABLISHED(0,0).confirmed_at 11 -> identity broken.
    with pytest.raises(AssertionError):
        _build([_bos(5, 0, 0, 12), _cts_est(10, 0, 0, 11)])


def test_bos_cts_identity_is_on_the_moment_never_on_the_extreme():
    # NEGATIVE CONTROL: CTS_ESTABLISHED.idx 10 != confirmed_at 11, but BOS.confirmed_at 11 ==
    # CTS.confirmed_at 11 -> must NOT raise (the assert is on the moment, never on .idx). 3 such pairs
    # exist in the saved M15 streams (1223/1224, 2828/2829 x2).
    t = _build([_bos(5, 0, 0, 11), _cts_est(10, 0, 0, 11)])
    assert t.cts_moment[(0, 0)] == 11
    # Real M15 pair: CTS_ESTABLISHED idx 1223 / confirmed_at 1224 (sub 0 cycle 2 in the Plan B save).
    t2 = _build([_cts_est(1020, 0, 1, 1020), _bos(1100, 0, 2, 1224), _cts_est(1223, 0, 2, 1224)])
    assert t2.cts_moment[(0, 2)] == 1224
    assert t2.floor_h1[(0, 2)] == 1224          # struct_start 1020 < 1224
    assert t2.end_h1[(0, 1)] == 1224            # (0,1) ends at (0,2)'s clamped start on the MOMENT, not 1223


def test_bos_confirmed_without_matching_cts_established_raises():
    # Every cycle that has a BOS_CONFIRMED has a CTS_ESTABLISHED (they are emitted together).
    with pytest.raises(AssertionError):
        _build([_bos(5, 0, 0, 11)])
    # Also when OTHER cycles are fine: (0,1) has a BOS but no CTS_ESTABLISHED.
    with pytest.raises(AssertionError):
        _build([_bos(5, 0, 0, 11), _cts_est(10, 0, 0, 11), _bos(20, 0, 1, 30)])


def test_loh_returning_none_raises():
    events = [_cts_est(10, 0, 0, 10), _reversal(40, 0)]
    # None for every map (the floor is the first map attempted).
    with pytest.raises(AssertionError):
        _build(events, loh=lambda p, h1, m15: None)
    # None only for the END map (floor 10 maps, end 40 does not) -> still raises.
    with pytest.raises(AssertionError):
        _build(events, loh=lambda p, h1, m15: (4 * p + 3) if p < 40 else None)
    # Sanity: the same events with a total map build fine.
    t = _build(events)
    assert (t.floor_m15[(0, 0)], t.end_m15[(0, 0)]) == (43, 163)


# --- logging -------------------------------------------------------------------

def test_log_lines_one_per_cycle_and_one_warning_per_degenerate_cycle():
    lines: List[str] = []
    _build(_reference_window_events(), log=lines.append)
    per_cycle = [s for s in lines if s.startswith("[parent_tables]")
                 and "degenerate parent cycle" not in s]
    warnings = [s for s in lines if "WARNING [parent_tables] degenerate parent cycle" in s]
    assert len(per_cycle) == 5                    # five parent cycles: (0,0) (0,1) (1,0) (1,1) (1,2)
    assert len(warnings) == 2                     # degenerate = {(1,0), (1,1)}
    # each warning names its cycle and carries floor=.. end=.. (both 3611 on this window)
    assert any(re.search(r"\(1,\s*0\)", s) for s in warnings)
    assert any(re.search(r"\(1,\s*1\)", s) for s in warnings)
    for w in warnings:
        assert "floor=3611" in w and "end=3611" in w
    # every cycle line names its cycle
    for (s, c) in [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]:
        assert any(re.search(r"\(%d,\s*%d\)" % (s, c), ln) for ln in per_cycle), (s, c)


def test_no_warning_when_nothing_is_degenerate():
    lines: List[str] = []
    _build([_cts_est(10, 0, 0, 10), _cts_est(30, 0, 1, 30)], log=lines.append)
    assert len([s for s in lines if s.startswith("[parent_tables]")
                and "degenerate parent cycle" not in s]) == 2
    assert not [s for s in lines if "WARNING" in s]


# --- the reference window ------------------------------------------------------

def test_reference_window_tables():
    """Plan C section 3: floors 463/2611/3611/3611/3611, ends 2611/3611/3611/3611/None,
    degenerate {(1,0), (1,1)} -- from synthetic events reproducing the H1 facts."""
    t = _build(_reference_window_events())
    assert t.rev_by_sid == {0: 902}
    # struct_start: sid 0 = min idx 96 (the (0,0) BOS extreme); sid 1 = rev[0] = 902 (handoff; min idx 689)
    assert t.struct_start == {0: 96, 1: 902}
    assert t.cts_moment == {(0, 0): 115, (0, 1): 652, (1, 0): 703, (1, 1): 748, (1, 2): 902}
    assert t.parent_sd == {0: 1, 1: -1}
    # floor_h1 = max(struct_start, moment): (0,0)=max(96,115)=115; (0,1)=max(96,652)=652;
    #            (1,0)=max(902,703)=902; (1,1)=max(902,748)=902; (1,2)=max(902,902)=902
    assert t.floor_h1 == {(0, 0): 115, (0, 1): 652, (1, 0): 902, (1, 1): 902, (1, 2): 902}
    # end_h1: (0,0)=floor(0,1)=652; (0,1)=no (0,2) -> rev[0]=902; (1,0)=floor(1,1)=902;
    #         (1,1)=floor(1,2)=902; (1,2)=no (1,3), sid 1 never reverses -> None
    assert t.end_h1 == {(0, 0): 652, (0, 1): 902, (1, 0): 902, (1, 1): 902, (1, 2): None}
    # LOH = 4p+3: 115->463, 652->2611, 902->3611
    assert t.floor_m15 == {(0, 0): 463, (0, 1): 2611, (1, 0): 3611, (1, 1): 3611, (1, 2): 3611}
    assert t.end_m15 == {(0, 0): 2611, (0, 1): 3611, (1, 0): 3611, (1, 1): 3611, (1, 2): None}
    # degenerate: (0,0) 463>=2611 F; (0,1) 2611>=3611 F; (1,0) 3611>=3611 T; (1,1) 3611>=3611 T; (1,2) open F
    assert t.degenerate == {(0, 0): False, (0, 1): False, (1, 0): True, (1, 1): True, (1, 2): False}
    assert {k for k, v in t.degenerate.items() if v} == {(1, 0), (1, 1)}
    # accessors
    assert t.cycles() == [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]
    assert [t.floor(s, c) for (s, c) in t.cycles()] == [463, 2611, 3611, 3611, 3611]
    assert [t.end(s, c) for (s, c) in t.cycles()] == [2611, 3611, 3611, 3611, None]
    assert [t.is_degenerate(s, c) for (s, c) in t.cycles()] == [False, False, True, True, False]
    # a cycle with no CTS_ESTABLISHED -> KeyError (callers assert first)
    with pytest.raises(KeyError):
        t.floor(1, 3)
    with pytest.raises(KeyError):
        t.end(2, 0)
    # ParentTables is a frozen dataclass
    assert dataclasses.is_dataclass(t)
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.rev_by_sid = {}           # type: ignore[misc]


def test_reference_window_is_insensitive_to_event_order():
    # The saved CSV is in EMISSION order (CTS 115 before BOS 96); the orchestrator sorts by (idx, type).
    # Reversing the sorted stream must not change any table (only last-seen is order-dependent, and no
    # (S,C) repeats here).
    evs = _reference_window_events()
    a = build_parent_tables(_sorted(evs), None, None, loh=_loh, log=lambda s: None)
    b = build_parent_tables(list(reversed(_sorted(evs))), None, None, loh=_loh, log=lambda s: None)
    assert a.floor_m15 == b.floor_m15 and a.end_m15 == b.end_m15 and a.degenerate == b.degenerate
