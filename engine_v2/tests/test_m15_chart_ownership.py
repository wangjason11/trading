"""M15 chart ownership + identity helpers under the pool (Plan C §6.2 / PART4 §17.9).

- Identity everywhere = `sub_id` for subs (`sub_sid` stays main's identity).
- Ownership = the LIFECYCLE window, per direction:
  `owner_by_idx_dir[(candle, direction)]` over `[start_idx, end_idx or edge]`
  — NOT the structural anchor (`creation_event_idx` = `starting_idx`).
- Opposite-direction subs may both own a candle; same-direction overlap is
  resolved by the later `start_idx` (later-live wins).

Pure helpers — no figure is built. Importing `export_m15_chart` needs plotly.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from engine_v2.charting.export_m15_chart import (
    _compute_forming_by_idx_dir,
    _compute_owner_by_idx_dir,
    _sid_parent,
    _sid_record_identity,
    _sub_identity,
)
from engine_v2.multitf.types import SidRecord


# --- fixtures ------------------------------------------------------------------

def _sub(
    sub_id: int, direction: int, starting_idx: int,
    start_idx: Optional[int], end_idx: Optional[int],
    end_reason: Optional[str] = None,
    first_record: Optional[Dict[str, Any]] = None,
    lenses: Tuple[str, ...] = ("confluence",),
) -> SidRecord:
    """A §2.5 sub-level SidRecord: `sub_sid=None`, `sub_id` set, parent fields
    None, `creation_event_idx = starting_idx`, `end_event_idx = end_idx`."""
    return SidRecord(
        sub_sid=None,
        starting_sd=direction,
        creation_event_idx=starting_idx,
        end_event_idx=end_idx,
        end_reason=end_reason,
        parent_sid=None,
        parent_cycle_id=None,
        sub_id=sub_id,
        start_idx=start_idx,
        lenses=lenses,
        meta={
            "natural_reversal_idx": None,
            "n_records": 1,
            "first_record": first_record or {},
            "slice_begin": max(0, starting_idx - 50),
            "validated_parent_start": None,
        },
    )


def _main(sub_sid: int, direction: int, creation_idx: int,
          end_idx: Optional[int]) -> SidRecord:
    """A main-entity SidRecord (unchanged by Plan C)."""
    return SidRecord(
        sub_sid=sub_sid, starting_sd=direction,
        creation_event_idx=creation_idx, end_event_idx=end_idx,
        end_reason="reversal" if end_idx is not None else None,
    )


# Predicted-table subs (reference_pool_redesign_groundtruth.md), sub_id in
# creation order: 3304/-1 -> 4, 3760/-1 -> 6, 4027/+1 -> 7.
#   sub 4: conf (1,2) tss0 [3621, 3819] same_dir_replacement (by sub 6)
#   sub 6: conf (1,2) tss1 [3819, 4200] reversal
#   sub 7: ctr (1,2) tss1 [4083, None]  open -> runs to the edge
_EDGE = 4400
SUB_3304 = _sub(4, -1, 3304, 3621, 3819, "same_dir_replacement")
SUB_3760 = _sub(6, -1, 3760, 3819, 4200, "reversal")
SUB_4027 = _sub(7, +1, 4027, 4083, None, None, lenses=("counter", "confluence"))


# --- ownership -----------------------------------------------------------------

def test_owner_keyed_by_candle_and_direction_over_lifecycle_window():
    """Keys are `(candle, direction)`; the range is `[start_idx, end_idx]`
    inclusive on both ends — sub 6 owns exactly 4200 - 3819 + 1 = 382 candles."""
    owner = _compute_owner_by_idx_dir([SUB_3760], _EDGE)
    assert set(owner.keys()) == {(c, -1) for c in range(3819, 4200 + 1)}
    assert len(owner) == 382
    assert owner[(3819, -1)] == 6
    assert owner[(4200, -1)] == 6
    assert all(v == 6 for v in owner.values())
    # Direction is part of the key: nothing is owned on the +1 side.
    assert (4000, +1) not in owner


def test_ownership_uses_lifecycle_start_not_the_anchor():
    """§6.2: the lifecycle, NOT the anchor. Sub 6's anchor (starting_idx =
    creation_event_idx) is 3760 but its record only exists from 3819 — candles
    3760..3818 are NOT owned (pre-`start_idx` dots stay hidden)."""
    owner = _compute_owner_by_idx_dir([SUB_3760], _EDGE)
    assert SUB_3760.creation_event_idx == 3760          # fixture sanity
    for c in range(3760, 3819):
        assert (c, -1) not in owner
    assert owner[(3819, -1)] == 6


def test_open_sub_runs_to_edge_idx():
    """`end_idx=None` (open) -> the range closes at `edge_idx`, inclusive:
    4083..4400 = 318 candles."""
    owner = _compute_owner_by_idx_dir([SUB_4027], _EDGE)
    assert owner[(4083, +1)] == 7
    assert owner[(_EDGE, +1)] == 7
    assert (_EDGE + 1, +1) not in owner
    assert (4082, +1) not in owner
    assert len(owner) == _EDGE - 4083 + 1 == 318


def test_opposite_direction_overlap_both_own_their_candles():
    """The predicted table's intended overlap: `4027/+1` on confluence from
    4083 while `3760/-1` runs to 4200. Over [4083, 4200] BOTH own their
    candle under their own direction key; neither displaces the other."""
    owner = _compute_owner_by_idx_dir([SUB_3760, SUB_4027], _EDGE)
    for c in range(4083, 4200 + 1):
        assert owner[(c, +1)] == 7
        assert owner[(c, -1)] == 6
    # Outside the overlap each side is single-owner.
    assert owner[(4000, -1)] == 6 and (4000, +1) not in owner
    assert owner[(4300, +1)] == 7 and (4300, -1) not in owner
    # Totals: sub 6 = 382 candles, sub 7 = 318 candles, disjoint keys.
    assert len(owner) == 382 + 318


def test_same_direction_later_start_wins_on_overlapping_candle():
    """Same-lens handover 3304 -> 3760 at 3819: both closed ranges contain
    3819; the later `start_idx` (sub 6) wins that candle. 3818 stays with
    sub 4, 3820 with sub 6."""
    owner = _compute_owner_by_idx_dir([SUB_3304, SUB_3760], _EDGE)
    assert owner[(3818, -1)] == 4
    assert owner[(3819, -1)] == 6
    assert owner[(3820, -1)] == 6
    assert owner[(3621, -1)] == 4
    # Union of [3621,3819] and [3819,4200] = [3621,4200] = 580 keys.
    assert len(owner) == 4200 - 3621 + 1 == 580


def test_later_start_wins_regardless_of_sub_id_order():
    """'Later start wins' is ordered by `(start_idx, identity)`, NOT by
    `sub_id`: a LOWER sub_id with the LATER start still wins the overlap.
    sub 9 [100,200] vs sub 3 [150,300] (both +1) -> 150..200 belong to sub 3."""
    lo_id_later_start = _sub(3, +1, 140, 150, 300, "reversal")
    hi_id_earlier_start = _sub(9, +1, 90, 100, 200, "same_dir_replacement")
    owner = _compute_owner_by_idx_dir([lo_id_later_start, hi_id_earlier_start], 500)
    for c in range(150, 200 + 1):
        assert owner[(c, +1)] == 3
    for c in range(100, 150):
        assert owner[(c, +1)] == 9
    for c in range(201, 300 + 1):
        assert owner[(c, +1)] == 3
    # Input order must not matter.
    owner2 = _compute_owner_by_idx_dir([hi_id_earlier_start, lo_id_later_start], 500)
    assert owner2 == owner


def test_record_with_start_idx_none_is_skipped():
    """A sub with no live record has `start_idx=None` (logged, not rendered,
    §4.5) — it can bound no range and owns nothing, even though its anchor
    (creation_event_idx) is set."""
    ghost = _sub(8, -1, 2803, None, None)
    assert ghost.creation_event_idx == 2803
    assert _compute_owner_by_idx_dir([ghost], _EDGE) == {}
    # And it does not disturb a real record next to it.
    owner = _compute_owner_by_idx_dir([ghost, SUB_3760], _EDGE)
    assert len(owner) == 382
    assert all(v == 6 for v in owner.values())


def test_owner_values_are_sub_identities():
    owner = _compute_owner_by_idx_dir([SUB_3304, SUB_3760, SUB_4027], _EDGE)
    assert set(owner.values()) == {4, 6, 7}
    assert all(isinstance(v, int) for v in owner.values())
    assert owner[(3700, -1)] == _sid_record_identity(SUB_3304)


# --- identity helpers ------------------------------------------------------------

def test_sub_identity_reads_sub_id_only():
    """`_sub_identity(meta)` = `meta["sub_id"]` (an int); None when absent. The
    legacy `sub_sid` / parent keys are IGNORED (they are informational after
    the rename, never identity)."""
    assert _sub_identity({"sub_id": 6, "parent_sid": 1, "parent_cycle_id": 2}) == 6
    assert _sub_identity({"sub_id": 0}) == 0
    assert _sub_identity({}) is None
    assert _sub_identity({"sub_sid": 3, "parent_sid": 1, "parent_cycle_id": 2}) is None
    # sub_id wins even when a stale sub_sid is present.
    assert _sub_identity({"sub_id": 7, "sub_sid": 1}) == 7


def test_sid_record_identity_is_sub_id_for_subs_and_sub_sid_for_main():
    assert _sid_record_identity(SUB_3760) == 6
    assert _sid_record_identity(SUB_4027) == 7
    main0 = _main(0, +1, 96, 902)
    main1 = _main(1, -1, 902, None)
    assert _sid_record_identity(main0) == 0
    assert _sid_record_identity(main1) == 1
    # Sub rows carry no sub_sid; main rows carry no sub_id.
    assert SUB_3760.sub_sid is None and main0.sub_id is None


def test_sid_parent_main_uses_parent_fields():
    """Main rows have no parent (both None) — `_sid_parent` returns the row's
    own `(parent_sid, parent_cycle_id)`; explicitly attributed rows return
    theirs, taking precedence over any `first_record` in meta."""
    assert _sid_parent(_main(0, +1, 96, 902)) == (None, None)
    attributed = SidRecord(
        sub_sid=0, starting_sd=-1, creation_event_idx=2639,
        end_event_idx=3611, end_reason="parent_end",
        parent_sid=0, parent_cycle_id=1,
        meta={"first_record": {"parent_sid": 9, "parent_cycle_id": 9}},
    )
    assert _sid_parent(attributed) == (0, 1)


def test_sid_parent_sub_uses_first_record():
    """Sub rows carry parent attribution ONLY via `meta["first_record"]`
    (§2.5): sub 7's first record is `subsequent_counter` in H1 (1,2)."""
    rec = _sub(
        7, +1, 4027, 4083, None, None,
        first_record={
            "lens": "counter", "parent_sid": 1, "parent_cycle_id": 2,
            "trigger_type": "subsequent_counter", "trigger_idx": 4083,
            "start_idx": 4083,
        },
        lenses=("counter", "confluence"),
    )
    assert rec.parent_sid is None and rec.parent_cycle_id is None
    assert _sid_parent(rec) == (1, 2)


# --- forming layer (chart review 2026-09-20, option 2) --------------------------

SUB_2639 = _sub(3, -1, 2639, 2829, 3611, "parent_end", lenses=("confluence", "counter"))


def test_forming_span_is_anchor_to_start_exclusive():
    """The FORMING layer covers `[creation_event_idx (anchor), start_idx)` —
    sub 6: 3760..3818 (59 candles), never 3819 (that is live)."""
    forming = _compute_forming_by_idx_dir([SUB_3760])
    assert set(forming.keys()) == {(c, -1) for c in range(3760, 3819)}
    assert all(v == 6 for v in forming.values())
    assert (3819, -1) not in forming


def test_forming_later_anchor_wins_and_rows_without_start_skipped():
    a = _sub(1, -1, 100, 300, 400)          # forming 100..299
    b = _sub(2, -1, 200, 350, 500)          # forming 200..349 — later anchor wins the overlap
    ghost = _sub(9, -1, 50, None, None)     # no live record: no forming span either
    forming = _compute_forming_by_idx_dir([a, b, ghost])
    assert forming[(150, -1)] == 1 and forming[(250, -1)] == 2 and forming[(320, -1)] == 2
    assert (50, -1) not in forming and (99, -1) not in forming
    assert (350, -1) not in forming


def test_forming_draws_under_a_live_same_direction_sub():
    """Chart rule (2026-09-20): the live map decides among LIVE subs, the forming
    map among FORMING subs — independently. Sub 4 (`3304/-1`) forms under live
    sub 3 (`2639/-1`, live [2829, 3611]) over [3304, 3611]: sub 3 is the live
    owner there AND sub 4 is the forming owner there, so sub 3's live dots and
    sub 4's forming (dimmed) dots are both drawn."""
    live = _compute_owner_by_idx_dir([SUB_2639, SUB_3304], _EDGE)
    forming = _compute_forming_by_idx_dir([SUB_2639, SUB_3304])
    assert live[(3304, -1)] == 3 and live[(3611, -1)] == 3
    assert forming[(3304, -1)] == 4 and forming[(3620, -1)] == 4
    assert (3621, -1) not in forming and live[(3621, -1)] == 4
    assert forming[(2700, -1)] == 3                 # sub 3's own forming span 2639..2828
    assert (2638, -1) not in forming and (2638, -1) not in live
