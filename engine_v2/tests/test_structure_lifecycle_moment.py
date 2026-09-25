"""Plan C section 3 -- the shared lifecycle leaf moves the cycle start to the MOMENT.

`zones/structure_lifecycle.compute_cycle_lifecycle` today uses `CTS_ESTABLISHED.ev.idx`
(the CTS EXTREME, a historical anchor) as the cycle-start term. Plan C section 3 (decided
2026-09-19; PART4 section 17.6) changes the canonical rule for main / sub / parent alike:

    cycle start = max(CTS_ESTABLISHED.meta["confirmed_at"], struct_start, floor)
    cycle end   = min(next cycle's CLAMPED start, reversal_idx_by_sid[sid], cap)   # unchanged rule,
                                                                                 # now on the moment

`compute_struct_start_by_sid` (min event idx, reversal handoff, floor) and
`compute_reversal_idx_by_sid` are UNCHANGED.

Written TESTS-FIRST. Tests (a) and (c) and the M15 fixture FAIL at the base (start/end == the
extreme); (b), (e), (f) pin behaviour the plan says must SURVIVE (H1 byte-identical on the reference
window because extreme == moment on all five cycles).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from engine_v2.structure.market_structure import StructureEvent
from engine_v2.tests._event_factory import make_event
from engine_v2.zones.structure_lifecycle import (
    compute_cycle_lifecycle,
    compute_reversal_idx_by_sid,
    compute_struct_start_by_sid,
)


# --- fixtures ----------------------------------------------------------------

def _ev(idx: int, type_: str, sid: int, cycle: Optional[int] = None, sd: int = 1,
        **extra: Any) -> StructureEvent:
    meta: Dict[str, Any] = {"structure_id": sid, "struct_direction": sd}
    if cycle is not None:
        meta["cycle_id"] = cycle
    meta.update(extra)
    category = "STATE" if type_ == "STATE_CHANGED" else "STRUCTURE"
    if type_ in ("CTS_ESTABLISHED", "BOS_CONFIRMED") and "confirmed_at" not in meta:
        # Contract-illegal on purpose (test (d); `illegal_event_contract`).
        return StructureEvent(idx=idx, category=category, type=type_, meta=meta)
    # CTS_ESTABLISHED / BOS_CONFIRMED: idx is the anchor (tests/_event_factory.py).
    return make_event(type_, idx, category=category, **meta)


def _cts_est(idx: int, sid: int, cycle: int, confirmed_at: Optional[int], sd: int = 1) -> StructureEvent:
    # idx = the CTS extreme (historical); confirmed_at = the apply candle (the MOMENT), as stamped by
    # market_structure._emit_cts_established. `confirmed_at=None` OMITS the key (test (d)).
    extra: Dict[str, Any] = {"pattern_anchor_idx": idx - 1, "pattern_name": "one_maru_continuous"}
    if confirmed_at is not None:
        extra["confirmed_at"] = confirmed_at
    return _ev(idx, "CTS_ESTABLISHED", sid, cycle, sd, **extra)


def _bos(idx: int, sid: int, cycle: int, confirmed_at: int, sd: int = 1) -> StructureEvent:
    return _ev(idx, "BOS_CONFIRMED", sid, cycle, sd, confirmed_at=confirmed_at,
               source="pullback_extreme")


def _reversal(idx: int, sid: int, sd: int = 1) -> StructureEvent:
    return _ev(idx, "STATE_CHANGED", sid, None, sd,
               **{"from": "pullback_range", "to": "reversal", "reason": "reversal_pattern"})


def _rev_candidate(idx: int, sid: int, apply_idx: int, sd: int = 1) -> StructureEvent:
    return _ev(idx, "REVERSAL_CANDIDATE", sid, None, sd, pattern_anchor_idx=idx, apply_idx=apply_idx,
               expires_idx=apply_idx)


def _sorted(events: List[StructureEvent]) -> List[StructureEvent]:
    return sorted(events, key=lambda e: (e.idx, e.type))


def _life(events: List[StructureEvent], **kw):
    evs = _sorted(events)
    return compute_cycle_lifecycle(evs, compute_reversal_idx_by_sid(evs), **kw)


def _h1_reference_events() -> List[StructureEvent]:
    """H1 facts of the reference window (groundtruth memory; Plan B save rows):
    CTS_EST idx/confirmed_at (0,0)=115/115 (0,1)=652/652 (1,0)=703/703 (1,1)=748/748 (1,2)=902/902;
    BOS idx (0,0)=96 (0,1)=591 (1,0)=689 (1,1)=728 (1,2)=826 with confirmed_at = the CTS moment;
    STATE_CHANGED->reversal sid 0 @902; REVERSAL_CANDIDATE idx 897 apply 902."""
    return [
        _bos(96, 0, 0, 115), _cts_est(115, 0, 0, 115),
        _bos(591, 0, 1, 652), _cts_est(652, 0, 1, 652),
        _rev_candidate(897, 0, 902), _reversal(902, 0),
        _bos(689, 1, 0, 703, sd=-1), _cts_est(703, 1, 0, 703, sd=-1),
        _bos(728, 1, 1, 748, sd=-1), _cts_est(748, 1, 1, 748, sd=-1),
        _bos(826, 1, 2, 902, sd=-1), _cts_est(902, 1, 2, 902, sd=-1),
    ]


# --- (a) the moment, not the extreme -------------------------------------------

def test_a_cycle_start_is_the_moment_not_the_extreme():
    # CTS_ESTABLISHED idx 10 / confirmed_at 11; BOS idx 5 / confirmed_at 11.
    # struct_start[0] = min event idx = 5; start = max(moment 11, 5) = 11   (base: max(idx 10, 5) = 10)
    life = _life([_bos(5, 0, 0, 11), _cts_est(10, 0, 0, 11)])
    assert life[(0, 0)] == (11, None, None)
    # The same with no BOS: struct_start = 10 (the CTS extreme itself); start = max(11, 10) = 11
    life2 = _life([_cts_est(10, 0, 0, 11)])
    assert life2[(0, 0)] == (11, None, None)


def test_a_floor_still_applies_on_top_of_the_moment():
    # floor 20 > moment 11 -> start = max(11, max(5, 20)) = 20 (the floor term is unchanged)
    life = _life([_bos(5, 0, 0, 11), _cts_est(10, 0, 0, 11)], lifecycle_floor=20)
    assert life[(0, 0)] == (20, None, None)
    # floor 11 == moment 11 -> 11; floor 10 (== the extreme) does NOT pull the start back to 10
    assert _life([_cts_est(10, 0, 0, 11)], lifecycle_floor=11)[(0, 0)][0] == 11
    assert _life([_cts_est(10, 0, 0, 11)], lifecycle_floor=10)[(0, 0)][0] == 11


# --- (b) extreme == moment -> unchanged (SURVIVES the base) ---------------------

def test_b_cycle_start_unchanged_when_extreme_equals_moment():
    # SURVIVES: idx 10 / confirmed_at 10 -> start 10 under both rules.
    life = _life([_bos(5, 0, 0, 10), _cts_est(10, 0, 0, 10)])
    assert life[(0, 0)] == (10, None, None)


# --- (c) the next cycle's end uses the CLAMPED start on the MOMENT ---------------

def test_c_next_cycle_end_uses_the_clamped_start_on_the_moment():
    # sid 0 reverses @40 -> struct_start[1] = 40 (reversal handoff).
    # sid 1: CTS (1,0) idx 15 / conf 16; (1,1) idx 25 / conf 26; (1,2) idx 50 / conf 51.
    events = [
        _cts_est(10, 0, 0, 10), _reversal(40, 0),
        _cts_est(15, 1, 0, 16, sd=-1), _cts_est(25, 1, 1, 26, sd=-1), _cts_est(50, 1, 2, 51, sd=-1),
    ]
    life = _life(events)
    # starts: (1,0) = max(16, 40) = 40; (1,1) = max(26, 40) = 40; (1,2) = max(51, 40) = 51 (base: 50)
    assert life[(1, 0)][0] == 40
    assert life[(1, 1)][0] == 40
    assert life[(1, 2)][0] == 51
    # ends: (1,0) = (1,1)'s CLAMPED start 40 -- not the raw moment 26, not the raw extreme 25
    #       (collapsed 40..40; the caller renders it inert)
    assert life[(1, 0)] == (40, 40, "next_cycle")
    #       (1,1) = (1,2)'s clamped start = max(51, 40) = 51 on the MOMENT (base: 50)
    assert life[(1, 1)] == (40, 51, "next_cycle")
    #       (1,2) open (no (1,3); sid 1 never reverses)
    assert life[(1, 2)] == (51, None, None)
    # sid 0's only cycle ends by its reversal
    assert life[(0, 0)] == (10, 40, "reversal")


def test_c_reversal_end_still_wins_ties_and_cap_is_unchanged():
    # Reversal @30 and next cycle whose clamped start is also 30 -> "reversal" wins the tie (set first).
    events = [_cts_est(10, 0, 0, 10), _cts_est(29, 0, 1, 30), _reversal(30, 0)]
    life = _life(events)
    assert life[(0, 0)] == (10, 30, "reversal")
    assert life[(0, 1)] == (30, 30, "reversal")
    # A cap strictly earlier than both overrides with cap_reason.
    life_cap = _life(events, lifecycle_cap=20, cap_reason="parent_end")
    assert life_cap[(0, 0)] == (10, 20, "parent_end")


# --- (d) a CTS_ESTABLISHED without confirmed_at ---------------------------------

@pytest.mark.illegal_event_contract
def test_d_cts_established_without_confirmed_at_raises():
    # PLAN-AMBIGUITY: the plan says the cycle start IS meta["confirmed_at"] and is silent on a missing key.
    # market_structure._emit_cts_established ALWAYS stamps confirmed_at (the only live emitter), and
    # section 17.7 says graceful-degradation paths become asserts -- so a silent fallback to `.idx`
    # (the extreme) is NOT justified: the leaf must raise (KeyError from the literal meta["confirmed_at"]
    # or an explicit AssertionError). The base returns start == idx == 10 here, so this fails at the base.
    with pytest.raises((KeyError, AssertionError)):
        _life([_cts_est(10, 0, 0, None)])


# --- (e) compute_struct_start_by_sid / compute_reversal_idx_by_sid unchanged (SURVIVE) --

def test_e_compute_struct_start_by_sid_is_the_first_cts_established_moment():
    evs = _sorted(_h1_reference_events())
    rev = compute_reversal_idx_by_sid(evs)
    # STATE_CHANGED->reversal only: {0: 902}; the REVERSAL_CANDIDATE (idx 897, apply 902) is ignored.
    assert rev == {0: 902}
    # base = the first CTS_ESTABLISHED MOMENT (Plan E E3f / Q5): sid 0 = 115 (NOT the BOS_0
    # anchor 96); sid 1 = 703 before the reversal handoff -> rev[0] = 902
    assert compute_struct_start_by_sid(evs, rev, None) == {0: 115, 1: 902}
    # floor raises every sid: max(115, 1000) = 1000; max(902, 1000) = 1000
    assert compute_struct_start_by_sid(evs, rev, 1000) == {0: 1000, 1: 1000}
    # floor below both: no-op
    assert compute_struct_start_by_sid(evs, rev, 50) == {0: 115, 1: 902}
    # a CTS_EST idx 10 / confirmed_at 11 alone gives its moment 11
    assert compute_struct_start_by_sid([_cts_est(10, 0, 0, 11)], {}, None) == {0: 11}
    # a sid that never established (data edge) keeps its first stamped idx
    st = StructureEvent(idx=7, category="STATE", type="STATE_CHANGED", price=None,
                        meta={"structure_id": 3})
    assert compute_struct_start_by_sid([st], {}, None) == {3: 7}


# --- (f) the H1 reference window is byte-identical under the moment rule (SURVIVES) ------

def test_f_h1_reference_window_is_byte_identical_under_the_moment_rule():
    life = _life(_h1_reference_events())
    # starts = max(moment, struct_start): (0,0)=max(115,96)=115; (0,1)=max(652,96)=652;
    #          (1,0)=max(703,902)=902; (1,1)=max(748,902)=902; (1,2)=max(902,902)=902
    # ends: (0,0)=next (0,1) 652 < reversal 902 -> 652 next_cycle; (0,1)=reversal 902;
    #       (1,0)=next 902; (1,1)=next 902; (1,2)=None
    assert life == {
        (0, 0): (115, 652, "next_cycle"),
        (0, 1): (652, 902, "reversal"),
        (1, 0): (902, 902, "next_cycle"),
        (1, 1): (902, 902, "next_cycle"),
        (1, 2): (902, None, None),
    }


# --- the three M15 sub cycles that shift +1 (the plan's stated consequence) --------------

def test_m15_sub_cycle_with_extreme_before_moment_shifts_plus_one():
    # Sub 0 of the reference window (FC(0,0), +1, starting 454; Plan B save M15 confluence stream):
    #   CTS_EST idx/confirmed_at: c0 458/458, c1 1020/1020, c2 1223/1224 (!), c3 1721/1721, c4 1793/1793
    #   own reversal @1940. Lifecycle floor = the sub's start_idx 1020 (entity-absolute for readability).
    events = [
        _cts_est(458, 0, 0, 458), _cts_est(1020, 0, 1, 1020), _cts_est(1223, 0, 2, 1224),
        _cts_est(1721, 0, 3, 1721), _cts_est(1793, 0, 4, 1793), _reversal(1940, 0),
    ]
    life = _life(events, lifecycle_floor=1020)
    # struct_start = max(min idx 458, floor 1020) = 1020
    # starts: c0 = max(458, 1020) = 1020; c1 = max(1020, 1020) = 1020; c2 = max(1224, 1020) = 1224 (base 1223);
    #         c3 = 1721; c4 = 1793
    # ends:   c0 = c1 start 1020 (collapsed); c1 = c2 start 1224 (base 1223); c2 = 1721; c3 = 1793;
    #         c4 = reversal 1940
    assert life[(0, 0)] == (1020, 1020, "next_cycle")
    assert life[(0, 1)] == (1020, 1224, "next_cycle")
    assert life[(0, 2)] == (1224, 1721, "next_cycle")
    assert life[(0, 3)] == (1721, 1793, "next_cycle")
    assert life[(0, 4)] == (1793, 1940, "reversal")


def test_m15_2828_2829_pair_shifts_only_when_no_floor_masks_it():
    # Sub 2639/-1 (both lenses): c0 2649/2649, c1 2828/2829 (!), c2 2992/2992, c3 3589/3589; no own reversal
    # in the saved bounded stream. Without a floor the moment rule is visible (c0 end / c1 start 2829,
    # base 2828); with the sub's floor 2829 (its Plan C start_idx) both rules agree -- the floor masks it.
    events = [
        _cts_est(2649, 0, 0, 2649, sd=-1), _cts_est(2828, 0, 1, 2829, sd=-1),
        _cts_est(2992, 0, 2, 2992, sd=-1), _cts_est(3589, 0, 3, 3589, sd=-1),
    ]
    life = _life(events)
    assert life[(0, 0)] == (2649, 2829, "next_cycle")     # base: end 2828
    assert life[(0, 1)] == (2829, 2992, "next_cycle")     # base: start 2828
    life_floored = _life(events, lifecycle_floor=2829)
    assert life_floored[(0, 0)] == (2829, 2829, "next_cycle")   # collapsed under both rules
    assert life_floored[(0, 1)] == (2829, 2992, "next_cycle")   # identical under both rules
