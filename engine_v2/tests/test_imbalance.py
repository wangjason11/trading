from __future__ import annotations

import pandas as pd
import pytest

from engine_v2.common.types import ImbalanceInstance
from engine_v2.patterns.imbalance import (
    compute_imbalance,
    has_imbalance_in_range,
    has_unfilled_imbalance,
    get_unfilled_imbalances,
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


# ---------- is_filled ----------

def test_is_filled_bullish_below_threshold():
    # gap_bottom=1.00, gap_top=1.10, size=0.10, threshold=0.70
    # fill_level = 1.10 - 0.10*0.70 = 1.03
    # A candle with low=1.02 (<=1.03) fills it.
    df = _make_df([
        (0.95, 1.00, 0.90, 0.95),   # 0
        (1.02, 1.08, 1.01, 1.07),   # 1: imbalance
        (1.10, 1.15, 1.10, 1.14),   # 2: c3
        (1.08, 1.09, 1.02, 1.06),   # 3: fills (low=1.02 <= 1.03)
    ])
    out = compute_imbalance(df)
    inst = out.attrs["imbalances"][0]
    assert inst.is_filled(out, check_to_idx=3) is True


def test_is_filled_bullish_above_threshold():
    df = _make_df([
        (0.95, 1.00, 0.90, 0.95),   # 0
        (1.02, 1.08, 1.01, 1.07),   # 1: imbalance
        (1.10, 1.15, 1.10, 1.14),   # 2: c3
        (1.08, 1.12, 1.05, 1.11),   # 3: low=1.05 > 1.03 -> not filled
    ])
    out = compute_imbalance(df)
    inst = out.attrs["imbalances"][0]
    assert inst.is_filled(out, check_to_idx=3) is False


def test_is_filled_empty_scan_range():
    # check_to_idx <= end_idx -> empty scan -> not filled
    df = _make_df([
        (0.95, 1.00, 0.90, 0.95),
        (1.02, 1.08, 1.01, 1.07),
        (1.10, 1.15, 1.10, 1.14),
    ])
    out = compute_imbalance(df)
    inst = out.attrs["imbalances"][0]
    assert inst.is_filled(out, check_to_idx=1) is False  # scan range [2, 1] empty


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

    assert has_imbalance_in_range(out, 1, 1) is True
    assert has_imbalance_in_range(out, 3, 4) is False

    # Instance unfilled -> has_unfilled True
    assert has_unfilled_imbalance(out, 1, 2, check_to_idx=4) is True

    # Direction filter: wrong direction returns False, correct returns True.
    # check_to_idx == end_idx mirrors the IC-validation call shape.
    assert has_unfilled_imbalance(out, 1, 2, check_to_idx=2, direction=-1) is False
    assert has_unfilled_imbalance(out, 1, 2, check_to_idx=2, direction=1) is True

    unfilled = get_unfilled_imbalances(out, 1, 4, check_to_idx=4)
    assert len(unfilled) == 1
    assert unfilled[0].direction == 1


def test_attrs_empty_when_no_imbalances():
    df = _make_df([
        (1.00, 1.05, 0.95, 1.02),
        (1.02, 1.06, 1.00, 1.03),
        (1.03, 1.07, 1.01, 1.04),
    ])
    out = compute_imbalance(df)
    assert out.attrs["imbalances"] == []
    assert out["is_imbalance"].sum() == 0
    assert has_imbalance_in_range(out, 0, 2) is False
    assert has_unfilled_imbalance(out, 0, 2, check_to_idx=2) is False
