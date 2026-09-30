"""Unit tests for compute_bounded_structure (Part 4 §5 bounded single-structure primitive).

The primitive runs exactly ONE directional structure (structure_id=0) over
[start_idx, end_idx] and stops at its first internal reversal, reporting that
reversal idx. It must NOT roll past the reversal into structure_id>=1. These
tests pin that contract and the first segment's exact event sequence (frozen
2026-09-30 from the then-equal multi-structure `compute_structure_from_start`,
deleted that day as dead code — no production caller).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine_v2.structure.structure_engine import (
    BoundedStructureResult,
    compute_bounded_structure,
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
    MarketStructure run stops there (the multi-structure driver would roll on
    into structure_id=1). That contrast is the whole point of the primitive.
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
# The first segment's exact events (frozen; was: parity with the deleted
# compute_structure_from_start's structure_id=0 events — equal on 2026-09-30)
# ---------------------------------------------------------------------------

_FIRST_SEGMENT_EVENTS = [
    ('CTS_ESTABLISHED', 2),
    ('BOS_CONFIRMED', 2),
    ('STATE_CHANGED', 2),
    ('CTS_UPDATED', 3),
    ('CTS_UPDATED', 4),
    ('CTS_UPDATED', 5),
    ('CTS_UPDATED', 10),
    ('CTS_UPDATED', 11),
    ('CTS_UPDATED', 12),
    ('CTS_UPDATED', 13),
    ('CTS_UPDATED', 14),
    ('CTS_UPDATED', 15),
    ('RANGE_STARTED', 18),
    ('STATE_CHANGED', 18),
    ('RANGE_UPDATED', 16),
    ('RANGE_UPDATED', 17),
    ('CTS_UPDATED', 19),
    ('RANGE_RESET', 20),
    ('CTS_UPDATED', 20),
    ('STATE_CHANGED', 20),
    ('CTS_UPDATED', 21),
    ('CTS_UPDATED', 22),
    ('CTS_UPDATED', 23),
    ('CTS_UPDATED', 28),
    ('CTS_UPDATED', 29),
    ('CTS_UPDATED', 30),
    ('CTS_UPDATED', 31),
    ('CTS_UPDATED', 32),
    ('RANGE_STARTED', 34),
    ('CTS_CONFIRMED', 34),
    ('STATE_CHANGED', 34),
    ('RANGE_UPDATED', 35),
    ('RANGE_UPDATED', 36),
    ('RANGE_UPDATED', 37),
    ('RANGE_UPDATED', 38),
    ('RANGE_UPDATED', 39),
    ('RANGE_UPDATED', 40),
    ('RANGE_UPDATED', 41),
    ('RANGE_UPDATED', 42),
    ('RANGE_UPDATED', 43),
    ('RANGE_UPDATED', 44),
    ('RANGE_UPDATED', 45),
    ('RANGE_UPDATED', 46),
    ('RANGE_UPDATED', 47),
    ('RANGE_UPDATED', 48),
    ('RANGE_UPDATED', 49),
    ('RANGE_UPDATED', 50),
    ('REVERSAL_WATCH_START', 51),
    ('REVERSAL_CANDIDATE', 51),
    ('STATE_CHANGED', 52),
    ('RANGE_UPDATED', 52),
]


class TestBoundedStructureParity:
    def test_first_segment_events_match(self):
        """structure_id=0 events == the frozen first segment (51 events, reversal at 52).

        Frozen from `compute_structure_from_start(df, 0, 1)`'s structure_id=0 events,
        which equalled the bounded run's (same MS run; events are append-only) when
        that function was deleted as dead code (2026-09-30).
        """
        df = _prepare_df(_make_reversing_data())
        bounded = compute_bounded_structure(df, start_idx=0, struct_direction=1)

        got = [(ev.type, int(ev.idx)) for ev in bounded.events
               if int(ev.meta.get("structure_id", 0)) == 0]
        assert len(_FIRST_SEGMENT_EVENTS) == 51
        assert got == _FIRST_SEGMENT_EVENTS
        assert bounded.reversal_idx == 52
