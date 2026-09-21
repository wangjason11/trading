"""Tests for the surviving sub-geometry builder (Plan C §5.4, PART4 §17.8).

`entity_df_mutation.build_or_get_geometry` (today's `_build_or_get_sub_geometry`,
renamed) is the SINGLE owner of pool lookup + MS run + `natural_reversal_idx` +
`bos0_inner`. Plan C pins three things about it:

  * run cap = the DATA EDGE (`run_cap_abs = len(m15) - 1`, always) — geometry is
    computed once per unique sub and projected per lifecycle later;
  * ordering on a miss: MS runs FIRST, the pool entry is created only on
    success (a failed build returns None and consumes NO `sub_id` — rev 1's
    `get_or_create`-first order left a geometry-less entry that made every
    later trigger to that key a false hit);
  * the return is `(sub, created)` (or None), with `sub.geometry ==
    (bounded, slice_begin)` where `bounded.events` / `bounded.df` are
    SLICE-LOCAL and `slice_begin = max(0, start - 50)`.

`_build_geometry` is the pool-free core (slice + 50-lookback + `reset_index` +
`compute_imbalance` + `is_range` re-derivation + ONE `compute_bounded_structure`).

This is the Layer-1 port of `test_sub_chain.py::TestBuildOneSid` (the chain and
`build_one_sid` are deleted by Plan C §7). Fixtures are copied from that file
(15-minute candles; the reversing series is the one validated in
`test_bounded_structure` to reverse at idx 52 of 61).

The new names are reached through the module object (`edm.build_or_get_geometry`)
so each test fails on its own missing attribute at the Plan-B base instead of
the whole file erroring at collection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine_v2.multitf import entity_df_mutation as edm
from engine_v2.multitf.sub_structure_pool import StructureKey, SubStructurePool
from engine_v2.structure import structure_engine
from engine_v2.structure.structure_engine import (
    BoundedStructureResult,
    build_ad_hoc_bos0_reference_zone,
)
from engine_v2.features.candle_classifier import apply_candle_classification
from engine_v2.patterns.pattern_engine import detect_patterns
from engine_v2.patterns.imbalance import compute_imbalance


# ---------------------------------------------------------------------------
# Fixtures (copied from test_sub_chain.py — self-contained, 15-minute candles)
# ---------------------------------------------------------------------------

def _make_raw_df(ohlc_rows: list[dict], pair: str = "NZD_USD") -> pd.DataFrame:
    base_time = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    records = []
    for i, row in enumerate(ohlc_rows):
        records.append({
            "time": base_time + pd.Timedelta(minutes=15 * i),
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
    """Up impulses with pullbacks, then a strong down leg → reversal at idx 52
    of 61 (validated in test_bounded_structure)."""
    rng = np.random.RandomState(7)
    rows: list[dict] = []
    price = 0.6000
    for _ in range(3):
        price = _seg(rows, price, 6, +1, 0.0020, 0.0004, rng)
        price = _seg(rows, price, 3, -1, 0.0008, 0.0003, rng)
    price = _seg(rows, price, 6, +1, 0.0020, 0.0004, rng)
    price = _seg(rows, price, 20, -1, 0.0022, 0.0004, rng)
    price = _seg(rows, price, 8, -1, 0.0006, 0.0003, rng)
    return rows


_PREFIX_N = 55


def _make_flat_prefix(n: int = _PREFIX_N, base: float = 0.6000) -> list[dict]:
    """`n` sideways noise candles around `base` (no structure), so the reversing
    series can be placed at entity-absolute `start = n` with a non-zero
    `slice_begin = n - 50`."""
    rng = np.random.RandomState(3)
    rows = []
    for _ in range(n):
        o = base + rng.uniform(-0.0003, 0.0003)
        c = base + rng.uniform(-0.0003, 0.0003)
        h = max(o, c) + rng.uniform(0.0001, 0.0003)
        l = min(o, c) - rng.uniform(0.0001, 0.0003)
        rows.append({"o": round(o, 5), "h": round(h, 5),
                     "l": round(l, 5), "c": round(c, 5)})
    return rows


_PARENT = "H1.main"
_TF = "M15"


@pytest.fixture(scope="module")
def reversing_df() -> pd.DataFrame:
    return _prepare_df(_make_reversing_data())          # 61 candles


@pytest.fixture(scope="module")
def uptrend_df() -> pd.DataFrame:
    return _prepare_df(_make_uptrend_data(n=80))        # 80 candles, never reverses


@pytest.fixture(scope="module")
def shifted_reversing_df() -> pd.DataFrame:
    """55 flat candles + the 61-candle reversing series = 116 candles; the
    reversing structure starts at entity-absolute 55."""
    return _prepare_df(_make_flat_prefix() + _make_reversing_data())


def _build(pool, df, *, sd=1, start_abs=0, bos0_inner=None):
    return edm.build_or_get_geometry(
        pool, df, parent_path=_PARENT, sd=sd, start_abs=start_abs,
        bos0_inner=bos0_inner, timeframe=_TF,
    )


def _patch_compute_bounded(monkeypatch, fn) -> None:
    """Patch `compute_bounded_structure` wherever the builder may resolve it:
    the defining module (covers a lazy `from … import` inside the builder, as
    today) and `entity_df_mutation` itself if it holds a module-level binding."""
    monkeypatch.setattr(structure_engine, "compute_bounded_structure", fn)
    if hasattr(edm, "compute_bounded_structure"):
        monkeypatch.setattr(edm, "compute_bounded_structure", fn)


# ---------------------------------------------------------------------------
# (1) miss → build + pool entry, (sub, True)
# ---------------------------------------------------------------------------

def test_miss_builds_and_creates_pool_entry(reversing_df):
    pool = SubStructurePool()
    # A realistic first-probe inner: the ad-hoc BOS_0 reference at (start 0, +1).
    inner = float(build_ad_hoc_bos0_reference_zone(reversing_df, 0, 1).inner)
    out = _build(pool, reversing_df, sd=1, start_abs=0, bos0_inner=inner)
    assert out is not None
    sub, created = out
    assert created is True
    assert sub.key == StructureKey(_PARENT, _TF, 1, 0)
    assert pool.get(sub.key) is sub
    assert pool.all() == [sub] and sub.sub_id == 0

    # geometry = (bounded, slice_begin); slice_begin = max(0, 0 - 50) = 0.
    bounded, slice_begin = sub.geometry
    assert isinstance(bounded, BoundedStructureResult)
    assert slice_begin == 0
    assert bounded.struct_direction == 1
    assert bounded.start_idx == 0                         # start_abs - slice_begin

    # natural_reversal_idx is ENTITY-ABSOLUTE (= slice-local + slice_begin);
    # the fixture reverses mid-series (> 10, validated in test_bounded_structure).
    assert bounded.reversal_idx is not None
    assert sub.natural_reversal_idx == int(bounded.reversal_idx) + slice_begin
    assert sub.natural_reversal_idx > 10

    # bos0_inner from the first probe is stored on the sub (§2.2).
    assert sub.bos0_inner == pytest.approx(inner)

    # The slice carries its own imbalances (re-computed after reset_index —
    # LANDMINE "M15 slice … must re-run compute_imbalance").
    assert "imbalances" in bounded.df.attrs


def test_miss_with_no_inner_stores_none(reversing_df):
    pool = SubStructurePool()
    out = _build(pool, reversing_df, sd=1, start_abs=0, bos0_inner=None)
    assert out is not None
    sub, created = out
    assert created is True
    assert sub.bos0_inner is None
    assert sub.natural_reversal_idx is not None and sub.natural_reversal_idx > 10


# ---------------------------------------------------------------------------
# (2) hit → (same sub, False), no second MS run
# ---------------------------------------------------------------------------

def test_hit_returns_same_sub_without_second_ms_run(reversing_df, monkeypatch):
    real = structure_engine.compute_bounded_structure
    calls = []

    def counting(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    _patch_compute_bounded(monkeypatch, counting)

    pool = SubStructurePool()
    first = _build(pool, reversing_df, sd=1, start_abs=0)
    assert first is not None and first[1] is True
    assert len(calls) == 1

    second = _build(pool, reversing_df, sd=1, start_abs=0)
    assert second is not None
    sub2, created2 = second
    assert sub2 is first[0]
    assert created2 is False
    assert len(calls) == 1                                # NO second MS run
    assert pool.all() == [first[0]]                       # no new sub_id
    assert sub2.geometry is first[0].geometry             # the shared object

    # A different key (same direction, start 5) is a miss: a second run, a
    # second sub_id — the pool dedups on the full key, not on the frame.
    third = _build(pool, reversing_df, sd=1, start_abs=5)
    assert third is not None
    assert third[1] is True and third[0].sub_id == 1
    assert third[0] is not first[0]
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# (3) run cap = data edge
# ---------------------------------------------------------------------------

def test_run_cap_is_the_data_edge(uptrend_df, reversing_df):
    """`run_cap_abs = len(m15) - 1` always: the MS bound, slice-local, is the last
    row of the slice, `len(slice) - 1` == `(len(df) - 1) - slice_begin`."""
    # Uptrend n=80, start 60: slice = iloc[10:80] (70 rows); bound = 69.
    pool = SubStructurePool()
    out = _build(pool, uptrend_df, sd=1, start_abs=60)
    assert out is not None
    bounded, slice_begin = out[0].geometry
    assert slice_begin == 10                              # max(0, 60 - 50)
    assert len(bounded.df) == len(uptrend_df) - slice_begin      # 80 - 10 = 70
    assert bounded.end_idx == len(bounded.df) - 1                # 69
    assert bounded.end_idx == (len(uptrend_df) - 1) - slice_begin
    assert out[0].natural_reversal_idx is None            # uptrend never reverses

    # Reversing 61, start 0: slice is the whole frame; bound = 60 even though
    # the run stops at its reversal (52) — the cap is a COMPUTE bound only.
    pool2 = SubStructurePool()
    out2 = _build(pool2, reversing_df, sd=1, start_abs=0)
    assert out2 is not None
    b2, sb2 = out2[0].geometry
    assert sb2 == 0
    assert b2.end_idx == len(reversing_df) - 1            # 60
    assert out2[0].natural_reversal_idx is not None
    assert out2[0].natural_reversal_idx < b2.end_idx


# ---------------------------------------------------------------------------
# (4) start past end of data → None, no entry, no sub_id consumed
# ---------------------------------------------------------------------------

def test_start_past_end_of_data_returns_none_and_no_entry(uptrend_df):
    pool = SubStructurePool()
    n = len(uptrend_df)                                   # 80
    assert _build(pool, uptrend_df, sd=1, start_abs=n) is None        # start == n
    assert _build(pool, uptrend_df, sd=1, start_abs=n + 5) is None    # past n
    assert _build(pool, uptrend_df, sd=1, start_abs=n - 1) is None    # start == run cap (79)
    assert pool.all() == []                               # NO pool entry
    assert pool.get(StructureKey(_PARENT, _TF, 1, n)) is None
    # No sub_id was consumed: the next successful build is sub_id 0.
    ok = _build(pool, uptrend_df, sd=1, start_abs=0)
    assert ok is not None and ok[0].sub_id == 0 and ok[1] is True


# ---------------------------------------------------------------------------
# (5) < 5 candles after start → None
# ---------------------------------------------------------------------------

def test_short_tail_returns_none(uptrend_df):
    pool = SubStructurePool()
    n = len(uptrend_df)                                   # 80
    # start 76 → candles [76..79] = 4 after start → None (guard is `< 5`).
    assert _build(pool, uptrend_df, sd=1, start_abs=n - 4) is None
    # start 77 → 3 → None.
    assert _build(pool, uptrend_df, sd=1, start_abs=n - 3) is None
    assert pool.all() == []
    # Boundary: start 75 → [75..79] = 5 candles → builds (slice_begin 25).
    ok = _build(pool, uptrend_df, sd=1, start_abs=n - 5)
    assert ok is not None and ok[1] is True
    assert ok[0].geometry[1] == n - 5 - 50                # 25
    assert ok[0].sub_id == 0                              # the failed ones consumed nothing


# ---------------------------------------------------------------------------
# (6) MS ValueError → None, no entry
# ---------------------------------------------------------------------------

def test_ms_value_error_returns_none_and_no_entry(reversing_df, monkeypatch):
    def boom(*args, **kwargs):
        raise ValueError("synthetic MS failure")

    _patch_compute_bounded(monkeypatch, boom)
    pool = SubStructurePool()
    assert _build(pool, reversing_df, sd=1, start_abs=0) is None
    assert pool.all() == []                               # MS ran first; no entry
    assert pool.get(StructureKey(_PARENT, _TF, 1, 0)) is None

    # Recovery: with MS restored, the same key is a MISS (no false hit on a
    # geometry-less entry) and gets sub_id 0.
    monkeypatch.undo()
    ok = _build(pool, reversing_df, sd=1, start_abs=0)
    assert ok is not None and ok[1] is True and ok[0].sub_id == 0
    assert ok[0].geometry is not None


# ---------------------------------------------------------------------------
# (7) slice_begin = max(0, start - 50); events are slice-local
# ---------------------------------------------------------------------------

def test_slice_begin_and_slice_local_events(uptrend_df, shifted_reversing_df):
    # Uptrend n=80, start 60 → slice_begin = 60 - 50 = 10.
    pool = SubStructurePool()
    out = _build(pool, uptrend_df, sd=1, start_abs=60)
    assert out is not None
    bounded, slice_begin = out[0].geometry
    assert slice_begin == max(0, 60 - 50) == 10
    assert bounded.start_idx == 60 - slice_begin          # 50, slice-local
    assert bounded.events, "expected MS events on the uptrend"
    for ev in bounded.events:
        # Slice-local: inside the slice frame …
        assert 0 <= int(ev.idx) < len(bounded.df)
        # … and, shifted by slice_begin, inside the entity frame.
        assert 0 <= int(ev.idx) + slice_begin <= len(uptrend_df) - 1
    # The structure's own events sit at/after its start once shifted (the
    # 50-candle lookback is context, not structure) — so the shift is real.
    assert any(int(ev.idx) + slice_begin >= 60 for ev in bounded.events)

    # Shifted reversing: 55 flat + 61 reversing = 116; start 55 → slice_begin 5.
    # The reversing series reverses at its local idx 52 → entity-absolute
    # 55 + 52 = 107 → slice-local 107 - 5 = 102.
    pool2 = SubStructurePool()
    out2 = _build(pool2, shifted_reversing_df, sd=1, start_abs=_PREFIX_N)
    assert out2 is not None
    b2, sb2 = out2[0].geometry
    assert sb2 == _PREFIX_N - 50                          # 5
    assert b2.reversal_idx is not None
    assert out2[0].natural_reversal_idx == int(b2.reversal_idx) + sb2
    assert out2[0].natural_reversal_idx == _PREFIX_N + 52  # 107
    assert b2.end_idx == (len(shifted_reversing_df) - 1) - sb2   # 115 - 5 = 110


# ---------------------------------------------------------------------------
# (8) _build_geometry — the pool-free core
# ---------------------------------------------------------------------------

def test_build_geometry_core_matches_pool_builder(reversing_df, uptrend_df):
    inner = float(build_ad_hoc_bos0_reference_zone(reversing_df, 0, 1).inner)
    core = edm._build_geometry(
        reversing_df, sd=1, start_abs=0, run_cap_abs=len(reversing_df) - 1,
        bos0_inner=inner, timeframe=_TF,
    )
    assert core is not None
    bounded, slice_begin = core
    assert isinstance(bounded, BoundedStructureResult)
    assert slice_begin == 0

    pool = SubStructurePool()
    via_pool = _build(pool, reversing_df, sd=1, start_abs=0, bos0_inner=inner)
    assert via_pool is not None
    pb, psb = via_pool[0].geometry
    assert (slice_begin, bounded.end_idx, bounded.reversal_idx, bounded.start_idx) == (
        psb, pb.end_idx, pb.reversal_idx, pb.start_idx,
    )
    assert [(ev.type, int(ev.idx)) for ev in bounded.events] == [
        (ev.type, int(ev.idx)) for ev in pb.events
    ]
    assert len(bounded.df) == len(pb.df)

    # Same values with a non-zero slice_begin (uptrend, start 60 → 10, cap 69).
    core2 = edm._build_geometry(
        uptrend_df, sd=1, start_abs=60, run_cap_abs=len(uptrend_df) - 1,
        bos0_inner=None, timeframe=_TF,
    )
    assert core2 is not None
    b2, sb2 = core2
    assert sb2 == 10 and b2.end_idx == 69 and b2.reversal_idx is None

    # The core has the same guards and creates nothing (no pool to create in).
    assert edm._build_geometry(
        uptrend_df, sd=1, start_abs=len(uptrend_df), run_cap_abs=len(uptrend_df) - 1,
        bos0_inner=None, timeframe=_TF,
    ) is None
    assert edm._build_geometry(
        uptrend_df, sd=1, start_abs=len(uptrend_df) - 4, run_cap_abs=len(uptrend_df) - 1,
        bos0_inner=None, timeframe=_TF,
    ) is None
