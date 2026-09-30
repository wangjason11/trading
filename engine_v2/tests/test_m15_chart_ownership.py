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
    _group_flag_runs,
    _prior_line_segments,
    _recency_key,
    _replacement_break_point,
    _sid_parent,
    _sid_record_identity,
    _split_polyline_by_wave,
    _sub_identity,
    _wave_touches_window,
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


# --- wave rule (chart review 2026-09-21) ----------------------------------------
# A WAVE (segment between two consecutive drawn points, over the EXTREME candles)
# is solid iff any part of its candle span lies inside the lifecycle window
# `[start_idx, end_idx]`; only waves wholly outside it are forming (sub charts)
# or hidden (H1 overlay). Reference-window facts: sub 0 (454/+1) live [1020, 1940]


def test_wave_crossing_start_idx_is_live_whole():
    """A wave covering the window's first candle counts live as a whole
    (sub 0's BOS c1@917 -> CTS c1@1020, start_idx 1020)."""
    assert _wave_touches_window(917, 1020, 1020, 1940)
    assert _wave_touches_window(1020, 917, 1020, 1940)      # order-insensitive


def test_wave_entirely_before_start_is_not_live():
    """Waves ending before the window are not live, even though the BOS at 917
    was CONFIRMED at 1020 (the rule reads the extreme candles, not the
    confirmation candles)."""
    assert not _wave_touches_window(454, 784, 1020, 1940)
    assert not _wave_touches_window(784, 917, 1020, 1940)
    assert not _wave_touches_window(1019, 1019, 1020, 1940)


def test_wave_window_end_is_inclusive_and_open_end_never_bounds():
    assert _wave_touches_window(1900, 2000, 1020, 1940)      # starts inside, runs past the end
    assert _wave_touches_window(1940, 2000, 1020, 1940)      # starts ON the end candle
    assert not _wave_touches_window(1941, 2000, 1020, 1940)  # wholly after the end
    assert _wave_touches_window(4222, 4300, 4083, None)      # open sub: nothing bounds the right side


def test_structure_without_start_idx_has_no_live_wave():
    assert not _wave_touches_window(100, 200, None, None)
    assert _split_polyline_by_wave([100, 200, 300], None, None) == [(False, 0, 2)]


def test_split_polyline_never_live_prefix_then_live_suffix():
    """Points 454, 784, 917, 1020, 1107 with window [1020, 1940] -> two runs
    sharing point 917: not-live 454->784->917 (2 waves), live 917->1020->1107
    (the wave crossing the window start counts live as a whole)."""
    runs = _split_polyline_by_wave([454, 784, 917, 1020, 1107], 1020, 1940)
    assert runs == [(False, 0, 2), (True, 2, 4)]


def test_split_all_live_when_the_only_pre_start_wave_crosses_start():
    """Sub 4 (3304/-1, live [3621, 3819]): BOS c0@3304 -> CTS c0@3621 -> ext 3819
    — no forming run at all (its whole pre-start span sits inside one wave that
    goes live)."""
    assert _split_polyline_by_wave([3304, 3621, 3819], 3621, 3819) == [(True, 0, 2)]


def test_split_h1_sid1_hides_the_retroactive_waves_only():
    """H1 sid 1 (live from 902): 689->710->728->761->826 wholly before 902 ->
    one hidden run; 826->905->edge -> one drawn run. Point 826 belongs to both."""
    runs = _split_polyline_by_wave([689, 710, 728, 761, 826, 905, 1057], 902, None)
    assert runs == [(False, 0, 4), (True, 4, 6)]


def test_split_degenerate_inputs():
    assert _split_polyline_by_wave([], 10, None) == []
    assert _split_polyline_by_wave([5], 10, None) == []
    assert _split_polyline_by_wave([5, 20], 10, None) == [(True, 0, 1)]
    assert _split_polyline_by_wave([5, 9], 10, None) == [(False, 0, 1)]


def test_split_alternates_when_a_wave_lies_past_a_closed_end():
    """A wave wholly after `end_idx` is not live either (defensive: points past
    the window end are normally clipped by the projection)."""
    runs = _split_polyline_by_wave([90, 110, 150, 200], 100, 120)
    assert runs == [(True, 0, 2), (False, 2, 3)]


# --- recent-vs-prior rule (chart review 2026-09-22) -----------------------------
# Where two structures draw lines over the same candles the MOST RECENT one
# (higher `_recency_key`) stays solid and the prior one's WHOLE segment is
# dotted; a segment nothing overlaps is solid even if the structure was never
# live. Overlap needs >1 shared candle and ignores direction. Reference-window
# confluence segments (measured from the rendered chart):
#   sub 0 ... 1761->1794, 1794->1940(ext)     sub 1  1797->1816 ... 2365->2470(ext)
#   sub 2  2365->2557 ... 2609->2829(ext)     sub 3  2639->2736 ... 3304->3611
#   sub 4  3304->3621, 3621->3818(ext)        sub 6  3760->4000, 4000->4200(ext)
#   sub 7  4027->4086 ...

def _segs(*spans):
    return [((i,), lo, hi) for i, (lo, hi) in enumerate(spans)]


def test_prior_marks_the_older_structures_segment_only():
    """sub 0's extension 1794->1940 overlaps sub 1's first waves: sub 1 is newer,
    so sub 0's segment is PRIOR and every segment of sub 1 stays solid."""
    prior = _prior_line_segments({
        0: ((0, 0, 0), _segs((1761, 1794), (1794, 1940))),
        1: ((0, 0, 1), _segs((1797, 1816), (1816, 1837), (1837, 1898), (1898, 1934), (1934, 1945))),
    })
    assert prior == {(0, (1,))}


def test_single_shared_candle_is_not_an_overlap():
    """sub 1's 2270->2365 meets sub 2's 2365->2557 at exactly one candle — the
    join is not an overlap, both stay solid (chart review 2026-09-22)."""
    prior = _prior_line_segments({
        1: ((0, 0, 1), _segs((2270, 2365))),
        2: ((0, 0, 2), _segs((2365, 2557))),
    })
    assert prior == set()


def test_partial_overlap_dots_the_whole_segment():
    """sub 4's extension 3621->3818 overlaps sub 6's 3760->4000 only over
    [3760, 3818]; the WHOLE 3621->3818 segment is dotted, sub 6 stays solid."""
    prior = _prior_line_segments({
        4: ((1, 2, 4), _segs((3304, 3621), (3621, 3818))),
        6: ((1, 2, 6), _segs((3760, 4000), (4000, 4200))),
    })
    assert prior == {(4, (1,))}


def test_rule_is_direction_agnostic_and_chains():
    """sub 6 (-1) and sub 7 (+1) are both live and opposite-direction at
    [4027, 4200]; the newer sub 7 still wins, so sub 6's extension is prior.
    With three structures the two older ones are prior and only the newest is
    solid."""
    prior = _prior_line_segments({
        4: ((1, 2, 4), _segs((3621, 3900))),
        6: ((1, 2, 6), _segs((3760, 4000), (4000, 4200))),
        7: ((1, 2, 7), _segs((4027, 4086))),
    })
    assert prior == {(4, (0,)), (6, (1,))}


def test_structure_with_no_overlap_is_solid_even_if_never_live():
    """Sub 0's 454->784->917 prefix (never live in real time) has nothing over
    the same candles, so nothing is prior."""
    prior = _prior_line_segments({0: ((0, 0, 0), _segs((454, 784), (784, 917)))})
    assert prior == set()


def test_recency_key_is_the_hierarchical_tuple():
    sub = _sub(4, -1, 3304, 3621, 3819, first_record={"parent_sid": 1, "parent_cycle_id": 2})
    assert _recency_key(sub) == (1, 2, 4)
    main = _main(3, +1, 10, None)
    assert _recency_key(main) == (-1, -1, 3)       # main rows: no parent -> sorts first
    a = _sub(2, +1, 100, 200, 300, first_record={"parent_sid": 0, "parent_cycle_id": 1})
    b = _sub(3, -1, 150, 250, 350, first_record={"parent_sid": 0, "parent_cycle_id": 1})
    assert _recency_key(b) > _recency_key(a)


def test_group_flag_runs_alternates_and_shares_boundary_points():
    assert _group_flag_runs([]) == []
    assert _group_flag_runs([True]) == [(True, 0, 1)]
    assert _group_flag_runs([False, False, True, True]) == [(False, 0, 2), (True, 2, 4)]
    assert _group_flag_runs([True, False, True]) == [(True, 0, 1), (False, 1, 2), (True, 2, 3)]


# --- replacement break point (chart review 2026-09-22b) -------------------------
# A sub ended by `same_dir_replacement` runs its final segment through the last
# structural swing before the REPLACING structure's anchor, instead of straight
# to the handover candle. Reference window: sub `3304/-1` (conf) breaks at 3760
# (bounded by sub `3760/-1`'s anchor) and NOT at the literal high 3806; sub
# `3621/+1` (counter) breaks at 4000 (bounded by sub `4027/+1`'s anchor 4027).

import pandas as pd  # noqa: E402

from engine_v2.common.types import COL_H, COL_L  # noqa: E402


class _Rec:
    """Minimal TriggerRecord stand-in (only the fields the helper reads)."""

    def __init__(self, end_reason, ended_by_sub_id, end_idx):
        self.end_reason = end_reason
        self.ended_by_sub_id = ended_by_sub_id
        self.end_idx = end_idx


def _df(highs, lows):
    return pd.DataFrame({COL_H: highs, COL_L: lows})


# 11 candles. The high rises to a peak at 5 (the structural swing, = the
# replacing anchor) and overshoots marginally at 8 (a later, higher high).
_HIGHS = [1.0, 1.1, 1.2, 1.3, 1.4, 1.50, 1.45, 1.46, 1.52, 1.48, 1.47]
_LOWS = [0.9, 0.8, 0.7, 0.6, 0.55, 0.50, 0.60, 0.58, 0.45, 0.62, 0.61]
_DF = _df(_HIGHS, _LOWS)
_REPL = [_Rec("same_dir_replacement", 9, 10)]


def test_break_is_bounded_by_the_replacing_anchor_not_the_literal_extreme():
    """A `-1` sub pulls UP: the extreme is searched only to the replacement's
    anchor (5), so the break is the structural swing 5 @ 1.50 — not the higher
    high at 8 (1.52). This is the 3760-vs-3806 case."""
    sub = _sub(4, -1, 0, 1, 10, "same_dir_replacement")
    assert _replacement_break_point(sub, _REPL, {9: 5}, _DF, 1, 10) == (5, 1.50)
    # unbounded (anchor past the end) would have picked the overshoot at 8
    assert _replacement_break_point(sub, _REPL, {9: 99}, _DF, 1, 10) == (8, 1.52)


def test_plus_one_sub_breaks_at_the_lowest_low():
    """A `+1` sub pulls DOWN: lowest low in (1, anchor]. With anchor 5 that is
    idx 5 @ 0.50 (idx 8's 0.45 is past the anchor)."""
    sub = _sub(5, +1, 0, 1, 10, "same_dir_replacement")
    assert _replacement_break_point(sub, _REPL, {9: 5}, _DF, 1, 10) == (5, 0.50)


def test_no_break_for_other_end_reasons():
    """Reversal / parent_end / open subs already end at a structural point —
    measured on the reference window, so the rule is replacement-only."""
    for reason in ("reversal", "parent_end", None):
        sub = _sub(4, -1, 0, 1, 10, reason)
        recs = [_Rec(reason, 9, 10)]
        assert _replacement_break_point(sub, recs, {9: 5}, _DF, 1, 10) is None


def test_no_break_when_the_extreme_is_an_endpoint_or_the_range_is_empty():
    sub = _sub(4, -1, 0, 1, 10, "same_dir_replacement")
    # anchor == last point -> empty range
    assert _replacement_break_point(sub, _REPL, {9: 1}, _DF, 1, 10) is None
    # the extreme lands ON the extension end -> no degenerate zero-length piece
    assert _replacement_break_point(sub, _REPL, {9: 5}, _DF, 4, 5) is None
    # unknown replacing sub / missing anchor
    assert _replacement_break_point(sub, _REPL, {}, _DF, 1, 10) is None
    assert _replacement_break_point(sub, [], {9: 5}, _DF, 1, 10) is None


def test_break_uses_the_record_that_ended_this_sub():
    """A sub with several records: the one whose `end_idx` is the sub's end
    carries `ended_by_sub_id`."""
    sub = _sub(4, -1, 0, 1, 10, "same_dir_replacement")
    recs = [_Rec("parent_end", None, 4), _Rec("same_dir_replacement", 9, 10)]
    assert _replacement_break_point(sub, recs, {9: 5}, _DF, 1, 10) == (5, 1.50)


def test_h1_overlay_window_starts_at_the_first_anchor_not_the_moment():
    """User decision 2026-09-25 (Plan E E3f landing review): the sub charts' H1
    overlay window opens at the sid's first structural ANCHOR (a location), so a
    lagging CTS_0's defining leg (BOS_0 anchor 96 → CTS_0 anchor 110, CTS_0 known
    at 115) is still drawn; a sid whose predecessor reversed opens at that
    reversal (the handoff), so its retroactive legs stay hidden."""
    from engine_v2.charting.export_m15_chart import (
        _h1_overlay_window_start_by_sid,
        _wave_touches_window,
    )
    from engine_v2.tests._event_factory import make_bos_confirmed, make_cts_established
    evs = [make_bos_confirmed(bos_anchor_idx=96, confirmed_at=115, structure_id=0, cycle_id=0),
           make_cts_established(cts_anchor_idx=110, confirmed_at=115, structure_id=0, cycle_id=0),
           make_bos_confirmed(bos_anchor_idx=689, confirmed_at=703, structure_id=1, cycle_id=0),
           make_cts_established(cts_anchor_idx=703, confirmed_at=703, structure_id=1, cycle_id=0)]
    start = _h1_overlay_window_start_by_sid(evs, {0: 902})
    assert start == {0: 96, 1: 902}
    assert _wave_touches_window(96, 110, start[0], 902)          # the defining leg is drawn
    assert not _wave_touches_window(96, 110, 115, 902)           # (keyed on the moment it would not be)
    assert not _wave_touches_window(689, 703, start[1], None)    # sid 1's retroactive leg stays hidden
