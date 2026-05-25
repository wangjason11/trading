"""Unit tests for compute_bounded_structure (Part 4 §5 bounded single-structure primitive).

The primitive runs exactly ONE directional structure (structure_id=0) over
[start_idx, end_idx] and stops at its first internal reversal, reporting that
reversal idx. It must NOT roll past the reversal into structure_id>=1 the way
compute_structure_from_start does. These tests pin that contract and anchor
parity against the first segment of compute_structure_from_start.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine_v2.structure.structure_engine import (
    BoundedStructureResult,
    compute_bounded_structure,
    compute_structure_from_start,
)
from engine_v2.features.candle_classifier import apply_candle_classification
from engine_v2.patterns.pattern_engine import detect_patterns
from engine_v2.patterns.imbalance import compute_imbalance


# ---------------------------------------------------------------------------
# Helpers (mirror test_scenario3.py — self-contained fixtures)
# ---------------------------------------------------------------------------

def _make_raw_df(ohlc_rows: list[dict], pair: str = "NZD_USD") -> pd.DataFrame:
    base_time = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    records = []
    for i, row in enumerate(ohlc_rows):
        records.append({
            "time": base_time + pd.Timedelta(hours=i),
            "o": row["o"], "h": row["h"], "l": row["l"], "c": row["c"],
            "volume": row.get("volume", 100),
        })
    df = pd.DataFrame(records)
    df.attrs["pair"] = pair
    return df


def _prepare_df(ohlc_rows: list[dict], pair: str = "NZD_USD") -> pd.DataFrame:
    raw = _make_raw_df(ohlc_rows, pair=pair)
    c_res = apply_candle_classification(raw)
    p_res = detect_patterns(c_res.df)
    df = compute_imbalance(p_res.df)
    df.attrs["pair"] = pair
    return df


def _make_uptrend_data(n: int = 80, base: float = 0.6000,
                       step: float = 0.0020, noise: float = 0.0005) -> list[dict]:
    rows = []
    rng = np.random.RandomState(42)
    price = base
    for _ in range(n):
        body = step + rng.uniform(0, noise)
        o = price
        c = price + body
        h = c + rng.uniform(0.0001, 0.0005)
        l = o - rng.uniform(0.0001, 0.0005)
        rows.append({"o": round(o, 5), "h": round(h, 5),
                     "l": round(l, 5), "c": round(c, 5)})
        price = c
    return rows


def _seg(rows, price, n, direction, step, noise, rng):
    for _ in range(n):
        body = step + rng.uniform(0, noise)
        o = price
        c = price + direction * body
        if direction > 0:
            h = c + rng.uniform(0.0001, 0.0004)
            l = o - rng.uniform(0.0001, 0.0004)
        else:
            h = o + rng.uniform(0.0001, 0.0004)
            l = c - rng.uniform(0.0001, 0.0004)
        rows.append({"o": round(o, 5), "h": round(h, 5),
                     "l": round(l, 5), "c": round(c, 5)})
        price = c
    return price


def _make_reversing_data() -> list[dict]:
    """Up impulses with pullbacks (so CTS confirms → BOS forms), then a strong
    down leg that close-breaks the BOS and drives a reversal.

    Empirically validated to reverse at idx 52 on a 61-candle series: a single
    MarketStructure run stops there, while compute_structure_from_start rolls
    on into structure_id=1. That contrast is the whole point of the primitive.
    """
    rng = np.random.RandomState(7)
    rows: list[dict] = []
    price = 0.6000
    for _ in range(3):
        price = _seg(rows, price, 6, +1, 0.0020, 0.0004, rng)   # impulse up
        price = _seg(rows, price, 3, -1, 0.0008, 0.0003, rng)   # shallow pullback
    price = _seg(rows, price, 6, +1, 0.0020, 0.0004, rng)       # impulse up → BOS
    price = _seg(rows, price, 20, -1, 0.0022, 0.0004, rng)      # reversal down
    price = _seg(rows, price, 8, -1, 0.0006, 0.0003, rng)       # tail
    return rows


def _max_structure_id(events) -> int:
    ids = [int(ev.meta.get("structure_id", 0)) for ev in events
           if ev.meta.get("structure_id") is not None]
    return max(ids) if ids else 0


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

class TestBoundedStructureContract:
    def test_returns_bounded_structure_result(self):
        df = _prepare_df(_make_uptrend_data(n=80))
        res = compute_bounded_structure(df, start_idx=0, struct_direction=1)
        assert isinstance(res, BoundedStructureResult)
        assert res.struct_direction == 1
        assert res.start_idx == 0
        assert isinstance(res.events, list)
        assert isinstance(res.levels, list)

    def test_no_reversal_returns_none(self):
        """A pure uptrend never reverses → reversal_idx is None."""
        df = _prepare_df(_make_uptrend_data(n=80))
        res = compute_bounded_structure(df, start_idx=0, struct_direction=1)
        assert res.reversal_idx is None
        rev_rows = ((res.df["market_state"].astype(str).str.lower() == "reversal")
                    & (res.df["structure_id"].astype(int) == 0))
        assert not rev_rows.any()


# ---------------------------------------------------------------------------
# Single-structure guarantee + reversal reporting (require a reversing fixture)
# ---------------------------------------------------------------------------

class TestBoundedStructureReversal:
    def test_single_structure_stops_at_reversal(self):
        """Primitive reverses, reports the idx, and stays at structure_id=0."""
        df = _prepare_df(_make_reversing_data())
        bounded = compute_bounded_structure(df, start_idx=0, struct_direction=1)

        assert bounded.reversal_idx is not None
        assert _max_structure_id(bounded.events) == 0   # never rolls past the reversal

    def test_reversal_idx_is_first_sid0_reversal_row(self):
        """reversal_idx is the first structure_id=0 reversal-marked candle."""
        df = _prepare_df(_make_reversing_data())
        bounded = compute_bounded_structure(df, start_idx=0, struct_direction=1)
        ridx = bounded.reversal_idx
        assert ridx is not None

        row = bounded.df.loc[ridx]
        assert str(row["market_state"]).lower() == "reversal"
        assert int(row["structure_id"]) == 0

        rev_mask = ((bounded.df["market_state"].astype(str).str.lower() == "reversal")
                    & (bounded.df["structure_id"].astype(int) == 0))
        assert ridx == int(bounded.df.loc[rev_mask].index.min())
        assert bounded.start_idx <= ridx

    def test_rolls_past_only_in_multi_structure(self):
        """Contrast: compute_structure_from_start rolls into structure_id>=1."""
        df = _prepare_df(_make_reversing_data())
        bounded = compute_bounded_structure(df, start_idx=0, struct_direction=1)
        multi = compute_structure_from_start(df, 0, 1)

        assert _max_structure_id(bounded.events) == 0
        assert _max_structure_id(multi.events) >= 1

    def test_end_idx_caps_before_reversal(self):
        """end_idx below the reversal apply idx → bounded out (reversal_idx None)."""
        df = _prepare_df(_make_reversing_data())
        ridx = compute_bounded_structure(df, 0, 1).reversal_idx
        assert ridx is not None and ridx > 10

        cap = ridx - 5
        bounded = compute_bounded_structure(df, start_idx=0, struct_direction=1, end_idx=cap)
        assert bounded.reversal_idx is None
        assert all(int(ev.idx) <= cap for ev in bounded.events)


# ---------------------------------------------------------------------------
# Parity with compute_structure_from_start's first segment
# ---------------------------------------------------------------------------

class TestBoundedStructureParity:
    def test_first_segment_events_match(self):
        """structure_id=0 events match compute_structure_from_start (identical MS run).

        Events are append-only in both paths, so the multi-structure run's
        structure_id=0 events are exactly its first segment — which must equal
        the bounded single-structure run. (The df itself diverges because the
        multi run overwrites rows for structure_id>=1; events do not.)
        """
        df = _prepare_df(_make_reversing_data())

        bounded = compute_bounded_structure(df, start_idx=0, struct_direction=1)
        multi = compute_structure_from_start(df, 0, 1)

        def sid0(events):
            return [(ev.type, int(ev.idx)) for ev in events
                    if int(ev.meta.get("structure_id", 0)) == 0]

        assert sid0(bounded.events) == sid0(multi.events)
