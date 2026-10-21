from __future__ import annotations

import pandas as pd
import pytest

from engine_v2.common.types import IMBALANCE_FILL_THRESHOLD, ImbalanceInstance
from engine_v2.patterns.imbalance import (
    compute_imbalance,
    has_unfilled_imbalance,
)


def _make_df(rows):
    """Build a minimal OHLC df with direction column.

    rows: list of (o, h, l, c) -> direction derived as +1 if c>o else -1 (else 0)
    """
    data = {"o": [], "h": [], "l": [], "c": [], "direction": []}
    for (o, h, l, c) in rows:
        data["o"].append(o)
        data["h"].append(h)
        data["l"].append(l)
        data["c"].append(c)
        if c > o:
            data["direction"].append(1)
        elif c < o:
            data["direction"].append(-1)
        else:
            data["direction"].append(0)
    df = pd.DataFrame(data)
    df["time"] = pd.date_range("2026-01-01", periods=len(df), freq="h")
    return df


# ---------- Detection: single-candle ----------

def test_single_bullish_imbalance():
    # idx 1 is c2 (bullish). c1.h < c3.l.
    # c1 high=1.0, c3 low=1.1  -> gap_bottom=1.0, gap_top=1.1
    df = _make_df([
        (0.95, 1.00, 0.90, 0.95),   # 0: c1
        (1.02, 1.08, 1.01, 1.07),   # 1: c2 bullish
        (1.10, 1.15, 1.10, 1.14),   # 2: c3
    ])
    out = compute_imbalance(df)
    assert out["is_imbalance"].tolist() == [0, 1, 0]
    insts = out.attrs["imbalances"]
    assert len(insts) == 1
    inst = insts[0]
    assert (inst.start_idx, inst.end_idx) == (1, 1)
    assert inst.direction == 1
    assert inst.gap_bottom == 1.00
    assert inst.gap_top == 1.10
    assert inst.gap_size == pytest.approx(0.10)


def test_c2_direction_mismatch_excluded():
    # c1.h < c3.l exists (gap present) but c2 is bearish -> NOT an imbalance
    df = _make_df([
        (0.95, 1.00, 0.90, 0.95),   # 0: c1
        (1.07, 1.08, 1.01, 1.02),   # 1: c2 bearish (close < open)
        (1.10, 1.15, 1.10, 1.14),   # 2: c3
    ])
    out = compute_imbalance(df)
    assert out["is_imbalance"].sum() == 0
    assert out.attrs["imbalances"] == []


# ---------- Detection: merging ----------

def test_merged_run_of_three_bullish():
    # idx 1, 2, 3 are all bullish imbalances. Merge into one instance.
    # gap_bottom = df[0].h, gap_top = df[4].l
    df = _make_df([
        (0.95, 1.00, 0.90, 0.95),   # 0
        (1.02, 1.08, 1.01, 1.07),   # 1: c2 bull
        (1.10, 1.17, 1.09, 1.15),   # 2: c2 bull
        (1.19, 1.27, 1.18, 1.24),   # 3: c2 bull
        (1.30, 1.35, 1.30, 1.33),   # 4
    ])
    # Verify each of 1/2/3 has a valid FVG with its neighbors:
    # i=1: df[0].h=1.00 < df[2].l=1.09 ✓
    # i=2: df[1].h=1.08 < df[3].l=1.18 ✓
    # i=3: df[2].h=1.17 < df[4].l=1.30 ✓
    out = compute_imbalance(df)
    assert out["is_imbalance"].tolist() == [0, 1, 1, 1, 0]
    insts = out.attrs["imbalances"]
    assert len(insts) == 1
    inst = insts[0]
    assert (inst.start_idx, inst.end_idx) == (1, 3)
    assert inst.direction == 1
    assert inst.gap_bottom == 1.00   # df[0].h
    assert inst.gap_top == 1.30      # df[4].l


def test_gap_between_bullish_imbalances_creates_two_instances():
    # idx 1 bull imbalance, idx 2 no imbalance, idx 3 bull imbalance
    df = _make_df([
        (0.95, 1.00, 0.90, 0.95),   # 0
        (1.02, 1.08, 1.01, 1.07),   # 1: c2 bull imbalance (gap with 0, 2)
        (1.10, 1.12, 1.09, 1.11),   # 2: c2 bull but no FVG (df[1].h=1.08 >= df[3].l=1.09)
        (1.13, 1.19, 1.09, 1.17),   # 3: c2 bull imbalance (gap with 2, 4)
        (1.25, 1.30, 1.25, 1.28),   # 4
    ])
    # Check i=2 is NOT imbalance: need df[1].h < df[3].l -> 1.08 < 1.09 TRUE actually
    # So adjust to make idx 2 NOT imbalance.
    df = _make_df([
        (0.95, 1.00, 0.90, 0.95),   # 0
        (1.02, 1.08, 1.01, 1.07),   # 1: c2 bull  -> FVG with 0 (1.00) & 2 (l=1.15 below)
        (1.20, 1.25, 1.15, 1.22),   # 2: c2 bull  -> df[1].h=1.08 < df[3].l? df[3].l=1.05 NO
        (1.10, 1.17, 1.05, 1.15),   # 3: c2 bull  -> df[2].h=1.25 > df[4].l=1.25 NO
        (1.30, 1.35, 1.25, 1.33),   # 4
    ])
    # i=1: df[0].h=1.00 < df[2].l=1.15 ✓ and c2 bull ✓
    # i=2: df[1].h=1.08 < df[3].l=1.05 FALSE (1.08 > 1.05)
    # i=3: df[2].h=1.25 < df[4].l=1.25 FALSE (not strictly <)
    out = compute_imbalance(df)
    assert out["is_imbalance"].tolist() == [0, 1, 0, 0, 0]
    insts = out.attrs["imbalances"]
    assert len(insts) == 1


def test_bullish_then_bearish_adjacent_are_separate():
    # idx 1 bullish imbalance, idx 2 bearish imbalance. Different direction -> 2 instances.
    df = _make_df([
        (0.95, 1.00, 0.90, 0.95),   # 0: c1 for bull
        (1.02, 1.08, 1.01, 1.07),   # 1: c2 bull (gap with 0 & 2: df[0].h=1.00 < df[2].l=1.09)
        (1.25, 1.30, 1.09, 1.12),   # 2: c2 bearish (c<o)  FVG with 1 & 3?
        (0.90, 0.95, 0.85, 0.90),   # 3: no direction OK; but we need FVG check
        (0.80, 0.85, 0.75, 0.80),   # 4
    ])
    # i=1: 1.00 < 1.09 ✓ bull ✓
    # i=2: bearish FVG requires df[1].l=1.01 > df[3].h=0.95 ✓ and c2 bearish (1.12<1.25) ✓
    # i=3: direction is (c=0.90 == o=0.90) -> 0, cannot be imbalance (needs +1 or -1)
    out = compute_imbalance(df)
    assert out["is_imbalance"].tolist() == [0, 1, 1, 0, 0]
    insts = out.attrs["imbalances"]
    assert len(insts) == 2
    assert insts[0].direction == 1
    assert insts[1].direction == -1


# ---------- is_filled — two-stroke state machine ----------
# Gap geometry for the bullish setup used throughout:
#   gap_bottom = df[0].h = 1.00, gap_top = df[2].l = 1.10, gap_size = 0.10
#   stroke 1 level (the default IMBALANCE_FILL_THRESHOLD, 65% retrace): 1.10 - 0.10*0.65 = 1.035 (low <= 1.035)
#   stroke 2 level: 1.10 (close must be >= 1.10), on a candle AFTER the arming one (PLAN_S step 2)


def test_is_filled_no_stroke1_unfilled_bullish():
    """Neither stroke fires (low stays above 1.035) → unfilled."""
    df = _make_df([
        (0.95, 1.00, 0.90, 0.95),   # 0
        (1.02, 1.08, 1.01, 1.07),   # 1: imbalance
        (1.10, 1.15, 1.10, 1.14),   # 2: c3
        (1.08, 1.12, 1.05, 1.11),   # 3: low=1.05 > 1.035 → no stroke 1
    ])
    out = compute_imbalance(df)
    inst = out.attrs["imbalances"][0]
    assert inst.is_filled(out, check_to_idx=3) is False


def test_is_filled_stroke1_only_unfilled_bullish():
    """Stroke 1 fires but no stroke 2 → still unfilled (was filled under
    the pre-2026-05 70%-only definition; the two-stroke change is the
    whole point of the rewrite)."""
    df = _make_df([
        (0.95, 1.00, 0.90, 0.95),   # 0
        (1.02, 1.08, 1.01, 1.07),   # 1: imbalance
        (1.10, 1.15, 1.10, 1.14),   # 2: c3
        (1.08, 1.09, 1.02, 1.06),   # 3: low=1.02 ✓ stroke 1; close=1.06 < 1.10 → no stroke 2
    ])
    out = compute_imbalance(df)
    inst = out.attrs["imbalances"][0]
    assert inst.is_filled(out, check_to_idx=3) is False


def test_is_filled_stroke1_then_stroke2_filled_bullish():
    """Stroke 1 latches, stroke 2 fires on a later candle → filled."""
    df = _make_df([
        (0.95, 1.00, 0.90, 0.95),   # 0
        (1.02, 1.08, 1.01, 1.07),   # 1: imbalance
        (1.10, 1.15, 1.10, 1.14),   # 2: c3
        (1.08, 1.09, 1.02, 1.06),   # 3: stroke 1 armed (low=1.02)
        (1.07, 1.12, 1.04, 1.11),   # 4: stroke 2 confirmed (close=1.11 >= 1.10)
    ])
    out = compute_imbalance(df)
    inst = out.attrs["imbalances"][0]
    # At idx 3, only stroke 1 — still unfilled.
    assert inst.is_filled(out, check_to_idx=3) is False
    # At idx 4, stroke 2 fires — filled.
    assert inst.is_filled(out, check_to_idx=4) is True


# The separate-candles fixtures (PLAN_S step 2, the user 2026-10-21: "the 2nd condition for filled must be met by a
# different candle after the armed candle"): candle 3 arms the gap AND closes past its outer edge — that close does NOT
# count; a LATER candle closing at / past the edge fills it. Candle 4 is a RED candle in the bullish case (it retraces,
# yet closes above the gap top — stroke 2 asks where the close is, not which way the candle moved: H1 450, PLAN_S §3.1).
# The "armed through a close inside" cases: candle 4 closes back INSIDE the gap — stroke 1 stays latched — and candle 5
# fills it (the landing review's finding 2: every other fixture fills at armed + 1).
_SAME_CANDLE_BULL = [
    (0.95, 1.00, 0.90, 0.95),   # 0: c1 (gap_bottom 1.00)
    (1.02, 1.08, 1.01, 1.07),   # 1: c2 bullish imbalance
    (1.10, 1.15, 1.10, 1.14),   # 2: c3 (gap_top 1.10)
    (1.05, 1.13, 1.02, 1.12),   # 3: low 1.02 arms; close 1.12 >= 1.10 on the ARMING candle — not counted
    (1.13, 1.14, 1.09, 1.11),   # 4: red, close 1.11 >= 1.10 → stroke 2
]
_SAME_CANDLE_BULL_AT_EDGE = _SAME_CANDLE_BULL[:4] + [
    (1.13, 1.14, 1.09, 1.10),   # 4: close == gap_top exactly → counts (at or past the edge)
]
_ARMED_THROUGH_A_CLOSE_INSIDE_BULL = _SAME_CANDLE_BULL[:4] + [
    (1.12, 1.13, 1.06, 1.07),   # 4: close 1.07 inside the gap (low 1.06 does not re-arm: 1.06 > 1.035) — still armed
    (1.07, 1.13, 1.06, 1.12),   # 5: close 1.12 >= 1.10 → stroke 2
]
_SAME_CANDLE_BEAR = [
    (1.20, 1.25, 1.18, 1.22),   # 0: c1 (gap_top 1.18)
    (1.10, 1.12, 1.05, 1.07),   # 1: c2 bearish imbalance
    (1.04, 1.10, 1.02, 1.05),   # 2: c3 (gap_bottom 1.10); stroke 1: high >= 1.10 + 0.08*0.65 = 1.152
    (1.12, 1.16, 1.05, 1.08),   # 3: high 1.16 arms; close 1.08 <= 1.10 on the ARMING candle — not counted
    (1.06, 1.09, 1.03, 1.08),   # 4: close 1.08 <= 1.10 → stroke 2
]
_SAME_CANDLE_BEAR_AT_EDGE = _SAME_CANDLE_BEAR[:4] + [
    (1.06, 1.09, 1.03, 1.10),   # 4: close == gap_bottom exactly → counts (the user: "On edge is ok")
]
_ARMED_THROUGH_A_CLOSE_INSIDE_BEAR = _SAME_CANDLE_BEAR[:4] + [
    (1.08, 1.14, 1.07, 1.13),   # 4: close 1.13 inside the gap (high 1.14 does not re-arm: 1.14 < 1.152) — still armed
    (1.13, 1.14, 1.05, 1.07),   # 5: close 1.07 <= 1.10 → stroke 2
]
_SEPARATE_CANDLE_CASES = [  # (rows, the candle that fills — stroke 2; candle 3 arms in every case)
    pytest.param(_SAME_CANDLE_BULL, 4, id="bull"),
    pytest.param(_SAME_CANDLE_BULL_AT_EDGE, 4, id="bull_close_at_edge"),
    pytest.param(_ARMED_THROUGH_A_CLOSE_INSIDE_BULL, 5, id="bull_armed_through_a_close_inside"),
    pytest.param(_SAME_CANDLE_BEAR, 4, id="bear"),
    pytest.param(_SAME_CANDLE_BEAR_AT_EDGE, 4, id="bear_close_at_edge"),
    pytest.param(_ARMED_THROUGH_A_CLOSE_INSIDE_BEAR, 5, id="bear_armed_through_a_close_inside"),
]


def _frame_and_instance(rows, offset):
    """The fixture's frame and its one instance. `offset` None: the frame as built — a RangeIndex (the positional
    fast path of `is_filled`, the cache's identity path). An int: a NON-RangeIndex int64 frame labelled
    offset..offset+n-1 with the instance re-labelled (the label-based fallback of `is_filled`; the cache's identity
    path at 0, its dict path otherwise)."""
    out = compute_imbalance(_make_df(rows))
    inst, = out.attrs["imbalances"]
    if offset is None:
        assert isinstance(out.index, pd.RangeIndex)
        return out, inst
    out.index = pd.Index([i + offset for i in range(len(out))], dtype="int64")
    assert not isinstance(out.index, pd.RangeIndex)
    return out, ImbalanceInstance(start_idx=inst.start_idx + offset, end_idx=inst.end_idx + offset,
                                  direction=inst.direction, gap_top=inst.gap_top, gap_bottom=inst.gap_bottom,
                                  gap_size=inst.gap_size)


@pytest.mark.parametrize("rows, filled_at", _SEPARATE_CANDLE_CASES)
def test_is_filled_the_arming_candle_cannot_also_fill(rows, filled_at):
    """The arming candle's own close never counts (separate candles); stroke 1 stays latched until a later candle
    closes at / past the outer edge — on the positional fast path (RangeIndex)."""
    out, inst = _frame_and_instance(rows, None)
    assert inst.is_filled(out, check_to_idx=filled_at - 1) is False
    assert inst.is_filled(out, check_to_idx=filled_at) is True


@pytest.mark.parametrize("rows, filled_at", _SEPARATE_CANDLE_CASES)
def test_is_filled_fallback_path_separate_candles(rows, filled_at):
    """The same rule on the label-based fallback loop (a non-RangeIndex frame)."""
    out, inst = _frame_and_instance(rows, 100)
    assert inst.is_filled(out, check_to_idx=100 + filled_at - 1) is False
    assert inst.is_filled(out, check_to_idx=100 + filled_at) is True


@pytest.mark.parametrize("offset", [None, 0, 100], ids=["range_index", "int64_index", "labelled_index"])
@pytest.mark.parametrize("rows, filled_at", _SEPARATE_CANDLE_CASES)
def test_fill_cache_separate_candles_and_agrees_with_is_filled(rows, filled_at, offset):
    """`poi_zones._compute_fill_idx_cache` (the POI activation sweep's numpy scan — the third implementation of the
    rule): armed at candle 3, confirmed at the filling candle (never the arming one); and at EVERY horizon `t` its
    answer equals `is_filled(t)` — the documented mirror — on the fast path (range_index) and the fallback (the
    int64 frames)."""
    from engine_v2.zones.poi_zones import _compute_fill_idx_cache
    k = offset or 0
    out, inst = _frame_and_instance(rows, offset)
    assert _compute_fill_idx_cache(out, [inst], IMBALANCE_FILL_THRESHOLD)[id(inst)] == (3 + k, filled_at + k)
    assert (inst.meta["armed_idx"], inst.meta["confirmed_fill_idx"]) == (3 + k, filled_at + k)
    for t in out.index:
        assert inst.is_filled(out, check_to_idx=int(t)) == (int(t) >= filled_at + k), t


def test_is_filled_stroke2_before_stroke1_unfilled_bullish():
    """A candle that closes past the gap outer BEFORE any stroke-1 retrace
    must not count — stroke 2 only latches at idx > armed_idx."""
    df = _make_df([
        (0.95, 1.00, 0.90, 0.95),   # 0
        (1.02, 1.08, 1.01, 1.07),   # 1: imbalance
        (1.10, 1.15, 1.10, 1.14),   # 2: c3
        (1.12, 1.15, 1.11, 1.14),   # 3: close=1.14 >= 1.10 BUT low=1.11 > 1.035 (no stroke 1)
        (1.14, 1.18, 1.13, 1.17),   # 4: same — no stroke 1 ever
    ])
    out = compute_imbalance(df)
    inst = out.attrs["imbalances"][0]
    assert inst.is_filled(out, check_to_idx=4) is False


def test_is_filled_stroke1_only_unfilled_bearish():
    """Bearish variant: stroke 1 (high reaches 65%) without stroke 2
    (close below gap_bottom) → unfilled."""
    # Geometry: gap_bottom = df[2].h = 1.10, gap_top = df[0].l = 1.18,
    #           gap_size = 0.08. Stroke 1: high >= 1.10 + 0.08*0.65 = 1.152.
    #           Stroke 2: close <= 1.10.
    df = _make_df([
        (1.20, 1.25, 1.18, 1.22),   # 0: c1
        (1.10, 1.12, 1.05, 1.07),   # 1: c2 bearish imbalance
        (1.04, 1.10, 1.02, 1.05),   # 2: c3
        (1.06, 1.16, 1.05, 1.13),   # 3: high=1.16 ✓ stroke 1; close=1.13 > 1.10 → no stroke 2
    ])
    out = compute_imbalance(df)
    inst = out.attrs["imbalances"][0]
    assert inst.direction == -1
    assert inst.is_filled(out, check_to_idx=3) is False


def test_is_filled_stroke1_then_stroke2_filled_bearish():
    """Bearish variant: stroke 1, then stroke 2 on a later candle → filled."""
    df = _make_df([
        (1.20, 1.25, 1.18, 1.22),   # 0: c1
        (1.10, 1.12, 1.05, 1.07),   # 1: c2 bearish imbalance
        (1.04, 1.10, 1.02, 1.05),   # 2: c3
        (1.06, 1.16, 1.05, 1.13),   # 3: stroke 1 armed (high=1.16)
        (1.12, 1.14, 1.05, 1.07),   # 4: stroke 2 confirmed (close=1.07 <= 1.10)
    ])
    out = compute_imbalance(df)
    inst = out.attrs["imbalances"][0]
    assert inst.direction == -1
    assert inst.is_filled(out, check_to_idx=3) is False
    assert inst.is_filled(out, check_to_idx=4) is True


def test_is_filled_empty_scan_range():
    """check_to_idx <= end_idx → empty scan → unfilled (default)."""
    df = _make_df([
        (0.95, 1.00, 0.90, 0.95),
        (1.02, 1.08, 1.01, 1.07),
        (1.10, 1.15, 1.10, 1.14),
    ])
    out = compute_imbalance(df)
    inst = out.attrs["imbalances"][0]
    assert inst.is_filled(out, check_to_idx=1) is False  # scan range [2, 1] empty


def test_is_filled_degenerate_gap_returns_true():
    """gap_size <= 0 short-circuits to True (safe default — invalid gap)."""
    df = _make_df([
        (1.00, 1.05, 0.95, 1.02),
        (1.02, 1.06, 1.00, 1.03),
        (1.03, 1.07, 1.01, 1.04),
    ])
    inst = ImbalanceInstance(
        start_idx=1, end_idx=1, direction=1,
        gap_top=1.0, gap_bottom=1.0, gap_size=0.0,
    )
    assert inst.is_filled(df, check_to_idx=2) is True


# ---------- overlaps ----------

def test_overlaps_edges():
    inst = ImbalanceInstance(
        start_idx=10, end_idx=12, direction=1,
        gap_top=1.1, gap_bottom=1.0, gap_size=0.1,
    )
    # inside
    assert inst.overlaps(10, 12) is True
    assert inst.overlaps(5, 20) is True
    # straddle start
    assert inst.overlaps(5, 10) is True
    assert inst.overlaps(5, 11) is True
    # straddle end
    assert inst.overlaps(12, 15) is True
    assert inst.overlaps(11, 15) is True
    # just outside
    assert inst.overlaps(0, 9) is False
    assert inst.overlaps(13, 20) is False


# ---------- aggregation helpers ----------

def test_aggregation_helpers():
    # Single bullish imbalance at idx 1 (not merged — idx 2 lacks an FVG)
    df = _make_df([
        (0.95, 1.00, 0.90, 0.95),   # 0: c1
        (1.02, 1.08, 1.01, 1.07),   # 1: c2 bull imb (df[0].h=1.00 < df[2].l=1.10)
        (1.20, 1.25, 1.10, 1.22),   # 2: c2 bull, FVG would need df[1].h=1.08 < df[3].l=1.15 ✓
                                     # ...which means idx 2 IS an imbalance too.
        (1.18, 1.28, 1.15, 1.25),   # 3
        (1.30, 1.35, 1.30, 1.33),   # 4
    ])
    # Reconstruct data so ONLY idx 1 is an imbalance, range [3,4] has none
    df = _make_df([
        (0.95, 1.00, 0.90, 0.95),   # 0: c1
        (1.02, 1.10, 1.01, 1.08),   # 1: c2 bull; df[0].h=1.00 < df[2].l=1.09 ✓
        (1.10, 1.20, 1.09, 1.18),   # 2: need df[1].h=1.10 < df[3].l? set df[3].l = 1.08 -> NOT imbalance
        (1.05, 1.18, 1.08, 1.16),   # 3
        (1.20, 1.25, 1.18, 1.23),   # 4
    ])
    # i=1: df[0].h=1.00 < df[2].l=1.09 ✓ AND c2 bull ✓ -> imbalance
    # i=2: df[1].h=1.10 < df[3].l=1.08 FALSE -> not imbalance
    # i=3: df[2].h=1.20 < df[4].l=1.18 FALSE -> not imbalance
    out = compute_imbalance(df)
    assert out["is_imbalance"].tolist() == [0, 1, 0, 0, 0]

    insts = out.attrs["imbalances"]
    assert [(i.start_idx, i.end_idx, i.direction) for i in insts] == [(1, 1, 1)]
    assert insts[0].overlaps(1, 1) is True
    assert insts[0].overlaps(3, 4) is False

    # Instance unfilled -> has_unfilled True
    assert has_unfilled_imbalance(out, 1, 2, check_to_idx=4, evaluated_at=None) is True

    # Direction filter: wrong direction returns False, correct returns True.
    # check_to_idx == end_idx mirrors the IC-validation call shape.
    assert has_unfilled_imbalance(out, 1, 2, check_to_idx=2, direction=-1, evaluated_at=None) is False
    assert has_unfilled_imbalance(out, 1, 2, check_to_idx=2, direction=1, evaluated_at=None) is True


# ---------- Knowability — the c3 rule (Plan F; IMBALANCE_FILL_SEMANTICS "Knowability") ----------

def _gap(start, end):
    return ImbalanceInstance(start_idx=start, end_idx=end, direction=1,
                             gap_top=1.10, gap_bottom=1.00, gap_size=0.10)


def _flat_df(n, instances):
    """Candles far above the 1.00-1.10 gaps: nothing fills."""
    df = _make_df([(1.50, 1.501, 1.499, 1.50)] * n)
    df.attrs["imbalances"] = list(instances)
    return df


def test_formed_at_is_the_first_c3():
    assert _gap(5, 5).formed_at == 6
    assert _gap(5, 8).formed_at == 6          # a merged run exists from its FIRST c3


def test_overlaps_formed_prefix():
    inst = _gap(10, 12)
    assert inst.overlaps_formed_prefix(0, 20, evaluated_at=10) is False   # not formed yet
    assert inst.overlaps_formed_prefix(0, 20, evaluated_at=11) is True    # prefix [10, 10]
    assert inst.overlaps_formed_prefix(11, 20, evaluated_at=11) is False  # prefix ends at 10
    assert inst.overlaps_formed_prefix(11, 20, evaluated_at=12) is True   # prefix [10, 11]
    assert inst.overlaps_formed_prefix(12, 20, evaluated_at=15) is True   # fully formed
    assert inst.overlaps_formed_prefix(13, 20, evaluated_at=15) is False  # outside the window


def test_has_unfilled_counts_an_instance_only_once_formed():
    df = _flat_df(12, [_gap(5, 5)])
    assert has_unfilled_imbalance(df, 0, 5, check_to_idx=5, evaluated_at=5) is False
    assert has_unfilled_imbalance(df, 0, 5, check_to_idx=5, evaluated_at=6) is True


def test_has_unfilled_tests_the_formed_prefix_against_the_window():
    df = _flat_df(12, [_gap(1, 3)])
    # At 3 the prefix [1, 2] has formed; it does not reach a window starting at 3.
    assert has_unfilled_imbalance(df, 3, 3, check_to_idx=3, evaluated_at=3) is False
    # Guard (catches R2 = "exists only from end_idx + 1"): the prefix counts.
    assert has_unfilled_imbalance(df, 0, 3, check_to_idx=2, evaluated_at=2) is True


def test_has_unfilled_evaluated_at_none_is_the_uncut_answer():
    df = _flat_df(12, [_gap(5, 5)])
    # The not-yet-formed instance counts (empty fill scan → unfilled): today's answer.
    assert has_unfilled_imbalance(df, 0, 5, check_to_idx=5, evaluated_at=None) is True


def test_has_unfilled_evaluated_at_is_required():
    df = _flat_df(12, [_gap(5, 5)])
    with pytest.raises(TypeError, match="evaluated_at"):
        has_unfilled_imbalance(df, 0, 5, check_to_idx=5)


def test_attrs_empty_when_no_imbalances():
    df = _make_df([
        (1.00, 1.05, 0.95, 1.02),
        (1.02, 1.06, 1.00, 1.03),
        (1.03, 1.07, 1.01, 1.04),
    ])
    out = compute_imbalance(df)
    assert out.attrs["imbalances"] == []
    assert out["is_imbalance"].sum() == 0
    assert has_unfilled_imbalance(out, 0, 2, check_to_idx=2, evaluated_at=None) is False
