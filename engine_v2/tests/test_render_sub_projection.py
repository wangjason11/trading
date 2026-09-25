"""Pins for `entity_df_mutation.render_sub_projection` +
`mirror_lower_tf_result_to_entity_df` (Plan C §6.1 / PART4 §17.9).

The rules under test (every expected value below is derived from one of them,
never read off the code's output):

  R1  ONE projection per unique sub, over the sub's REAL-TIME window
      `[sub.start_idx, sub.end_idx]` — `floor_local = start_idx - slice_begin`,
      `cap_local = end_idx - slice_begin` (None while open), `cap_reason =
      sub.end_reason` — mirrored into EVERY lens df in `sub.lenses()` and into
      no other lens df (§6.1 / §17.9 "one projection per sub, mirrored per lens").
  R2  `LowerTFResult.trigger` = the FIRST LIVE record's `source_trigger`,
      `first = min(sub.live_records(), key=(start_idx, seq))` (§6.1). A
      zero-length record participates in nothing (§2.1) — not in `first`, not
      in `lenses()`.
  R3  `result.meta` carries `{sub_id, m15_start_idx (= starting_idx), start_idx,
      end_idx (None when open), m15_end_idx (= end_idx, or the geometry EDGE when
      open), end_reason, natural_reversal_idx, slice_begin, lenses,
      relative_dir_segments, n_records, first_record{lens, parent_sid,
      parent_cycle_id, trigger_type, trigger_idx, start_idx}, timeframe, use_case,
      started_by, parent_tf, parent_sid, parent_cycle_id}` (§6.1; GLOSSARY
      "projection").
  R4  Attribution stamped on EVERY mirrored event / KL zone / POI / fib / wave /
      prev-BOS line: `structure_path_id` (THIS lens's path), `timeframe`,
      `parent_tf`, `sub_id` (identity) + informational `parent_sid` /
      `parent_cycle_id` / `use_case` / `started_by` from the first record (§6.1
      / §17.9).
  R5  Events are clipped by KNOWABLE-AT at the cap (every event at its moment:
      `ev.idx` — a `CTS_ESTABLISHED` / `BOS_CONFIRMED`'s idx IS its moment
      since Plan E E4a / E4b; a pattern-path `CTS_UPDATED` is known at
      `confirmed_at` since E3b, none lags in this geometry; `<= cap` inclusive)
      and DEEP-COPIED — geometry objects are shared, so mutating a mirrored
      event's meta must not touch `bounded.events` (§6.1, §12 landmine).
  R6  KL zones inherit the window: `confirmed_idx >= sub.start_idx` (the floor
      clamps a zone's first-active up to the structure lifecycle-start, B1) and
      `end_idx <= sub.end_idx` with `end_reason == sub.end_reason` when the cap
      is the earliest end term (`end = min(next_cycle, reversal, cap)`, B2).
  R7  `_STRUCTURE_COLS` are painted into the lens df ONLY over the sub's LIVE
      rows `[start_idx, end]` — rows before `start_idx` (the 50-candle lookback
      and the pre-start anchor rows) are untouched, which is what lets
      start_idx-ordered mirroring implement "later-live wins" (§17.9; cold
      review 2026-09-20). An OPEN sub paints to the frame edge.
  R8  A sub with no live record has no lens and is not renderable (§4.5) —
      `render_sub_projection` raises `AssertionError`.

Fixture: the real reversing M15 series behind a 55-candle flat prefix (copied
from `test_geometry_builder.py`), built through `build_or_get_geometry` with a
pool so the geometry is the production shape (slice-local `bounded.events` /
`bounded.df` + `slice_begin = 5`, natural reversal at entity-absolute 107, run
cap = the data edge 115). Records are constructed by hand and the sub's
lifecycle fields are set directly — the projection is under test, not the
sweep.
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import pytest

from engine_v2.multitf import entity_df_mutation as edm
from engine_v2.structure import event_fields as ef
from engine_v2.multitf.sub_structure_pool import (
    LENS_CONFLUENCE,
    LENS_COUNTER,
    PooledStructure,
    SubStructurePool,
    TriggerRecord,
)
from engine_v2.multitf.types import LowerTFResult, MultiTFTrigger
from engine_v2.features.candle_classifier import apply_candle_classification
from engine_v2.patterns.pattern_engine import detect_patterns
from engine_v2.patterns.imbalance import compute_imbalance


# ---------------------------------------------------------------------------
# Fixture builders (copied from test_geometry_builder.py — self-contained,
# 15-minute candles; the reversing series reverses at idx 52 of 61)
# ---------------------------------------------------------------------------

def _make_raw_df(ohlc_rows: list, pair: str = "NZD_USD") -> pd.DataFrame:
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


def _prepare_df(ohlc_rows: list, pair: str = "NZD_USD") -> pd.DataFrame:
    raw = _make_raw_df(ohlc_rows, pair=pair)
    c_res = apply_candle_classification(raw)
    p_res = detect_patterns(c_res.df)
    df = compute_imbalance(p_res.df)
    df.attrs["pair"] = pair
    return df


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


def _make_reversing_data() -> list:
    """Up impulses with pullbacks, then a strong down leg -> reversal at idx 52
    of 61 (validated in test_bounded_structure)."""
    rng = np.random.RandomState(7)
    rows: list = []
    price = 0.6000
    for _ in range(3):
        price = _seg(rows, price, 6, +1, 0.0020, 0.0004, rng)
        price = _seg(rows, price, 3, -1, 0.0008, 0.0003, rng)
    price = _seg(rows, price, 6, +1, 0.0020, 0.0004, rng)
    price = _seg(rows, price, 20, -1, 0.0022, 0.0004, rng)
    price = _seg(rows, price, 8, -1, 0.0006, 0.0003, rng)
    return rows


_PREFIX_N = 55


def _make_flat_prefix(n: int = _PREFIX_N, base: float = 0.6000) -> list:
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


# ---------------------------------------------------------------------------
# Constants + helpers
# ---------------------------------------------------------------------------

_PARENT = "H1.main"
_TF = "M15"
_LENS_PATHS = {
    LENS_CONFLUENCE: "H1.main >> M15.confluence",
    LENS_COUNTER: "H1.main >> M15.counter",
}
# The structural anchor of the fixture sub (= where the reversing series starts).
_STARTING_IDX = _PREFIX_N                       # 55
# "A few candles after starting_idx" — the sub's real-time start (R1).
_START = _STARTING_IDX + 5                      # 60
# A cap strictly inside the pre-reversal region (the natural reversal is at 107,
# so the cap — not the reversal — is the earliest end term, R6) with MS events
# on both sides of it (R5 needs a real clip).
_END = 85

# Sentinels pre-filled into the lens dfs' structure columns (R7).
_SENTINEL_SID = -777
_SENTINEL_STATE = "__untouched__"

_ATTRIBUTION_KEYS = (
    "structure_path_id", "timeframe", "parent_tf", "sub_id",
    "parent_sid", "parent_cycle_id", "use_case", "started_by",
)


@pytest.fixture(scope="module")
def m15_df() -> pd.DataFrame:
    """55 flat candles + the 61-candle reversing series = 116 candles; the
    reversing structure starts at entity-absolute 55."""
    return _prepare_df(_make_flat_prefix() + _make_reversing_data())


@pytest.fixture
def geometry(m15_df) -> Tuple[SubStructurePool, PooledStructure]:
    """A FRESH pool + unique sub per test (each test writes the sub's records
    and lifecycle fields in place)."""
    pool = SubStructurePool()
    out = edm.build_or_get_geometry(
        pool, m15_df, parent_path=_PARENT, sd=1, start_abs=_STARTING_IDX,
        bos0_inner=None, timeframe=_TF,
    )
    assert out is not None, "geometry build must succeed on the reversing fixture"
    sub, created = out
    assert created is True
    bounded, slice_begin = sub.geometry
    # Fixture preconditions (proven in test_geometry_builder): slice_begin =
    # max(0, 55 - 50) = 5; run cap = the data edge; the natural reversal is
    # entity-absolute 55 + 52 = 107 and lies strictly inside (start, edge).
    assert slice_begin == _STARTING_IDX - 50
    assert slice_begin + len(bounded.df) - 1 == len(m15_df) - 1
    assert sub.natural_reversal_idx is not None
    assert _START < _END < sub.natural_reversal_idx < len(m15_df) - 1
    return pool, sub


def _trigger(m15_df: pd.DataFrame, use_case: str, parent_sid: int,
             parent_cycle_id: int) -> MultiTFTrigger:
    """A real `MultiTFTrigger` (the shape the four detectors emit)."""
    return MultiTFTrigger(
        parent_tf="H1",
        parent_sid=parent_sid,
        parent_cycle_id=parent_cycle_id,
        parent_sd=1,
        use_case=use_case,
        lower_tf=_TF,
        lower_sd=1,
        meta={},
    )


def _record(
    pool: SubStructurePool,
    sub: PooledStructure,
    m15_df: pd.DataFrame,
    lens: str,
    *,
    start_idx: int,
    seq: int,
    trigger_type: str,
    parent_sid: int,
    parent_cycle_id: int,
    trigger_idx: int = None,
    validated_parent_idx: int = None,
    trigger_end_idx: int = None,
) -> TriggerRecord:
    """A §2.1 `TriggerRecord` appended to `sub.records` the way the sweep does.

    The three historical terms default to `start_idx` so the §2.1 rule
    `start_idx == max(probe_finalize_idx, trigger_idx, parent_floor_idx)` holds
    by construction. `trigger_end_idx` (when given) is applied the way phase 3
    writes it (`end_idx = max(trigger_end_idx, start_idx)`); a value
    `<= start_idx` makes the record ZERO-LENGTH (§2.1).
    """
    trigger_idx = start_idx if trigger_idx is None else trigger_idx
    rec = TriggerRecord(
        lens=lens,
        parent_sid=parent_sid,
        parent_cycle_id=parent_cycle_id,
        trigger_sub_sid=pool.next_trigger_sub_sid(lens, parent_sid, parent_cycle_id),
        sub_id=sub.sub_id,
        trigger_type=trigger_type,
        trigger_idx=trigger_idx,
        probe_finalize_idx=start_idx,
        probe_finalize_condition="phase1_bounded",
        validated_parent_idx=validated_parent_idx,
        starting_idx=sub.starting_idx,
        direction=sub.direction,
        sub_tf=sub.sub_tf,
        relative_dir="confluence",          # direction (+1) == parent_sd (+1)
        parent_floor_idx=start_idx,
        start_idx=start_idx,
        source_trigger=_trigger(m15_df, trigger_type, parent_sid, parent_cycle_id),
        seq=seq,
    )
    assert rec.start_idx == max(rec.probe_finalize_idx, rec.trigger_idx, rec.parent_floor_idx)
    if trigger_end_idx is not None:
        rec.trigger_end_idx = trigger_end_idx
        rec.end_idx = max(trigger_end_idx, start_idx)
        rec.end_reason = "parent_end"
    sub.records.append(rec)
    return rec


def _set_lifecycle(sub: PooledStructure, start_idx: int, end_idx, end_reason) -> None:
    """The sub's real-time window, as the sweep would have set it (§4.4/§4.5)."""
    sub.start_idx = start_idx
    sub.end_idx = end_idx
    sub.end_reason = end_reason
    sub.relative_dir_segments = [(start_idx, "confluence")]


def _lens_dfs(m15_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Two lens dfs = `df.copy()` each, with two `_STRUCTURE_COLS` pre-filled
    with sentinels on EVERY row (R7). The base frame carries no MS columns."""
    assert "structure_id" not in m15_df.columns and "market_state" not in m15_df.columns
    out = {}
    for lens in (LENS_CONFLUENCE, LENS_COUNTER):
        d = m15_df.copy()
        d["structure_id"] = _SENTINEL_SID
        d["market_state"] = _SENTINEL_STATE
        out[lens] = d
    return out


def _render(sub: PooledStructure, m15_df: pd.DataFrame,
            lens_dfs: Dict[str, pd.DataFrame]) -> LowerTFResult:
    return edm.render_sub_projection(
        sub, m15_df, lens_paths=_LENS_PATHS, lens_dfs=lens_dfs, timeframe=_TF,
    )


def _knowable_at(ev) -> int:
    """§17.9 / R5, written out independently of `knowable_at_idx`: a
    `BOS_CONFIRMED` is known at `meta["confirmed_at"]`; every other event at
    `ev.idx`. (Since Plan E E4a / E4b a `CTS_ESTABLISHED` / `BOS_CONFIRMED`'s idx
    IS its moment, so the BOS branch equals `ev.idx`; the anchor-vs-moment
    discrimination lives in the straddle test's anchor 55 / moment 57.)"""
    if ev.type == "BOS_CONFIRMED" and ev.meta.get("confirmed_at") is not None:
        return int(ev.meta["confirmed_at"])
    return int(ev.idx)


def _bos0(bounded):
    """The geometry's first BOS_CONFIRMED (slice-local)."""
    return next(ev for ev in bounded.events if ev.type == "BOS_CONFIRMED")


def _element_metas(lens_df: pd.DataFrame) -> Iterable[Tuple[str, dict]]:
    """Every mirrored element's attribution-bearing meta, by kind."""
    for ev in lens_df.attrs.get("events", []):
        yield "events", ev.meta
    for z in lens_df.attrs.get("kl_zones", []):
        yield "kl_zones", z.meta
    for z in lens_df.attrs.get("poi_zones", []):
        yield "poi_zones", z.meta
    for f in lens_df.attrs.get("fib_states", []):
        yield "fib_states", f.meta
    for w in lens_df.attrs.get("wave_candles", []):
        yield "wave_candles", w.meta
    for ln in lens_df.attrs.get("prev_bos_lines", []):
        yield "prev_bos_lines", ln["meta"]


def _expected_attribution(lens: str, sub: PooledStructure, first: TriggerRecord) -> dict:
    """R4: this lens's path + identity `sub_id` + the first record's informational
    fields (`use_case` / `started_by` = its trigger_type; `parent_tf` = its
    trigger's parent TF)."""
    return {
        "structure_path_id": _LENS_PATHS[lens],
        "timeframe": _TF,
        "parent_tf": first.source_trigger.parent_tf,
        "sub_id": sub.sub_id,
        "parent_sid": first.parent_sid,
        "parent_cycle_id": first.parent_cycle_id,
        "use_case": first.trigger_type,
        "started_by": first.trigger_type,
    }


def _two_lens_records(pool, sub, m15_df):
    """Four records on the sub, in CREATION order:
      Z — counter lens, start 58, zero-length (trigger_end 58 <= start) — the
          EARLIEST start, but participates in nothing (§2.1);
      A — counter lens, start 63, live (seq 1);
      B — confluence lens, start 60, live (seq 2) — created after A but starts
          FIRST among the live records -> the first live record (R2);
      D — counter lens, start 70, live (seq 3) — created LAST.
    So `first` (B) is neither the first-created nor the last-created live
    record. Distinct parent scopes / trigger types / validated_parent_idx so
    the attribution reveals which record was chosen."""
    z = _record(pool, sub, m15_df, LENS_COUNTER, start_idx=_START - 2, seq=0,
                trigger_type="first_counter", parent_sid=2, parent_cycle_id=0,
                trigger_end_idx=_START - 2)
    a = _record(pool, sub, m15_df, LENS_COUNTER, start_idx=_START + 3, seq=1,
                trigger_type="subsequent_counter", parent_sid=3, parent_cycle_id=2,
                validated_parent_idx=40)
    b = _record(pool, sub, m15_df, LENS_CONFLUENCE, start_idx=_START, seq=2,
                trigger_type="first_confluence", parent_sid=3, parent_cycle_id=1,
                trigger_idx=_START - 3, validated_parent_idx=12)
    d = _record(pool, sub, m15_df, LENS_COUNTER, start_idx=_START + 10, seq=3,
                trigger_type="subsequent_counter", parent_sid=4, parent_cycle_id=0,
                validated_parent_idx=77)
    assert z.is_zero_length
    assert not a.is_zero_length and not b.is_zero_length and not d.is_zero_length
    assert sub.live_records() == [a, b, d]                 # creation order; B in the middle
    assert sub.lenses() == {LENS_CONFLUENCE, LENS_COUNTER}
    return z, a, b


# ---------------------------------------------------------------------------
# R2 — LowerTFResult.trigger = the first LIVE record's source trigger
# ---------------------------------------------------------------------------

def test_trigger_is_first_live_record_by_start_then_seq(geometry, m15_df):
    pool, sub = geometry
    z, a, b = _two_lens_records(pool, sub, m15_df)
    _set_lifecycle(sub, _START, _END, "parent_end")

    res = _render(sub, m15_df, _lens_dfs(m15_df))

    # first = min(live_records, key=(start_idx, seq)): B starts at 60 < A's 63
    # < D's 70; Z (start 58) is zero-length and excluded despite the earliest
    # start and the lowest seq. Creation order (Z, A, B, D) is NOT the rule —
    # B is neither the first- nor the last-created live record.
    assert res.trigger is b.source_trigger
    assert res.trigger is not a.source_trigger and res.trigger is not z.source_trigger
    assert res.trigger is not sub.live_records()[-1].source_trigger
    assert res.meta["first_record"] == {
        "lens": LENS_CONFLUENCE,
        "parent_sid": b.parent_sid,             # 3
        "parent_cycle_id": b.parent_cycle_id,   # 1
        "trigger_type": "first_confluence",
        "trigger_idx": b.trigger_idx,           # 57 (historical; != start_idx)
        "start_idx": b.start_idx,               # 60
    }
    assert res.meta["use_case"] == res.meta["started_by"] == "first_confluence"
    assert res.meta["parent_sid"] == 3 and res.meta["parent_cycle_id"] == 1


def test_first_live_record_tie_breaks_on_seq(geometry, m15_df):
    pool, sub = geometry
    # Two live records with the SAME start_idx, appended in reverse seq order
    # (list order [seq 7, seq 3]) — the (start_idx, seq) key picks seq 3 — plus
    # a later-starting record appended LAST so the winner is neither the
    # first- nor the last-created live record.
    late = _record(pool, sub, m15_df, LENS_COUNTER, start_idx=_START, seq=7,
                   trigger_type="first_counter", parent_sid=5, parent_cycle_id=0,
                   validated_parent_idx=50)
    early = _record(pool, sub, m15_df, LENS_CONFLUENCE, start_idx=_START, seq=3,
                    trigger_type="subsequent_confluence", parent_sid=4, parent_cycle_id=2,
                    validated_parent_idx=21)
    tail = _record(pool, sub, m15_df, LENS_CONFLUENCE, start_idx=_START + 8, seq=9,
                   trigger_type="subsequent_confluence", parent_sid=4, parent_cycle_id=3,
                   validated_parent_idx=99)
    assert sub.live_records() == [late, early, tail]
    _set_lifecycle(sub, _START, _END, "parent_end")

    res = _render(sub, m15_df, _lens_dfs(m15_df))

    assert res.trigger is early.source_trigger
    assert res.meta["first_record"]["lens"] == LENS_CONFLUENCE
    assert res.meta["first_record"]["trigger_type"] == "subsequent_confluence"
    assert (res.meta["parent_sid"], res.meta["parent_cycle_id"]) == (4, 2)
    assert res.trigger is not late.source_trigger and res.trigger is not tail.source_trigger


# ---------------------------------------------------------------------------
# R3 — meta contract (capped sub)
# ---------------------------------------------------------------------------

def test_meta_contract_capped_sub(geometry, m15_df):
    pool, sub = geometry
    z, a, b = _two_lens_records(pool, sub, m15_df)
    _set_lifecycle(sub, _START, _END, "parent_end")
    bounded, slice_begin = sub.geometry

    res = _render(sub, m15_df, _lens_dfs(m15_df))

    required = {
        "sub_id", "m15_start_idx", "start_idx", "end_idx", "m15_end_idx",
        "end_reason", "natural_reversal_idx", "slice_begin", "lenses",
        "relative_dir_segments", "n_records", "first_record",
        "timeframe", "use_case", "started_by",
        "parent_tf", "parent_sid", "parent_cycle_id",
    }
    assert required <= set(res.meta), sorted(required - set(res.meta))

    m = res.meta
    assert m["sub_id"] == sub.sub_id == 0                 # first sub in a fresh pool
    assert m["m15_start_idx"] == sub.starting_idx == _STARTING_IDX   # the ANCHOR (historical)
    assert m["start_idx"] == _START                       # the real-time start
    assert m["end_idx"] == _END                           # capped: the real end
    assert m["m15_end_idx"] == _END                       # = end_idx when capped
    assert m["end_reason"] == "parent_end"
    assert m["natural_reversal_idx"] == sub.natural_reversal_idx == _STARTING_IDX + 52
    assert m["slice_begin"] == slice_begin == _STARTING_IDX - 50     # 5
    assert isinstance(m["lenses"], tuple) and set(m["lenses"]) == {LENS_CONFLUENCE, LENS_COUNTER}
    assert tuple(m["relative_dir_segments"]) == ((_START, "confluence"),)
    # n_records is a LOGGING count (the `_subs.csv` row); a zero-length record
    # is "logged only" (§2.1), so it counts: Z + A + B + D = 4.
    assert m["n_records"] == len(sub.records) == 4
    assert m["timeframe"] == _TF
    assert m["parent_tf"] == b.source_trigger.parent_tf == "H1"
    assert res.status == "finalized"
    # The result df is the slice up to and including the cap row:
    # rows [0, cap_local] -> cap_local + 1 = (85 - 5) + 1 = 81 rows.
    assert len(res.df) == (_END - slice_begin) + 1


# ---------------------------------------------------------------------------
# R1 / R4 — mirrored into EVERY lens in sub.lenses(), per-lens path, full
# attribution on every element kind
# ---------------------------------------------------------------------------

def test_mirrored_into_both_lenses_with_per_lens_attribution(geometry, m15_df):
    pool, sub = geometry
    z, a, b = _two_lens_records(pool, sub, m15_df)
    _set_lifecycle(sub, _START, _END, "parent_end")
    lens_dfs = _lens_dfs(m15_df)

    res = _render(sub, m15_df, lens_dfs)

    for lens in (LENS_CONFLUENCE, LENS_COUNTER):
        ldf = lens_dfs[lens]
        # Every list is mirrored; the projection's own lists are the source of
        # truth for the counts (one projection, mirrored as-is per lens).
        assert len(ldf.attrs["events"]) == len(res.events) > 0
        assert len(ldf.attrs["kl_zones"]) == len(res.kl_zones) > 0
        assert len(ldf.attrs["fib_states"]) == len(res.fib_states) > 0
        assert len(ldf.attrs["wave_candles"]) == len(res.wave_candles) > 0
        assert len(ldf.attrs["poi_zones"]) == len(res.poi_zones)
        assert len(ldf.attrs["prev_bos_lines"]) == len(res.prev_bos_lines)
        expected = _expected_attribution(lens, sub, b)
        seen_kinds = set()
        for kind, meta in _element_metas(ldf):
            seen_kinds.add(kind)
            got = {k: meta.get(k) for k in _ATTRIBUTION_KEYS}
            assert got == expected, (lens, kind, got)
        assert {"events", "kl_zones", "fib_states", "wave_candles"} <= seen_kinds

    # The two lens dfs hold DISTINCT objects (the mirror deep-copies per lens):
    # if they shared the geometry's event objects, the second mirror would
    # overwrite the first lens's `structure_path_id` with its own.
    conf_ev, ctr_ev = lens_dfs[LENS_CONFLUENCE].attrs["events"], lens_dfs[LENS_COUNTER].attrs["events"]
    assert all(c is not k for c, k in zip(conf_ev, ctr_ev))
    assert {e.meta["structure_path_id"] for e in conf_ev} == {_LENS_PATHS[LENS_CONFLUENCE]}
    assert {e.meta["structure_path_id"] for e in ctr_ev} == {_LENS_PATHS[LENS_COUNTER]}
    conf_kl, ctr_kl = lens_dfs[LENS_CONFLUENCE].attrs["kl_zones"], lens_dfs[LENS_COUNTER].attrs["kl_zones"]
    assert all(c is not k for c, k in zip(conf_kl, ctr_kl))
    assert {z.meta["structure_path_id"] for z in conf_kl} == {_LENS_PATHS[LENS_CONFLUENCE]}
    assert {z.meta["structure_path_id"] for z in ctr_kl} == {_LENS_PATHS[LENS_COUNTER]}


def test_lens_subset_only_confluence_leaves_counter_df_untouched(geometry, m15_df):
    pool, sub = geometry
    # A zero-length COUNTER record (participates in nothing -> no counter
    # lens) + one live CONFLUENCE record -> sub.lenses() == {"confluence"}.
    z = _record(pool, sub, m15_df, LENS_COUNTER, start_idx=_START - 2, seq=0,
                trigger_type="first_counter", parent_sid=2, parent_cycle_id=0,
                trigger_end_idx=_START - 2)
    b = _record(pool, sub, m15_df, LENS_CONFLUENCE, start_idx=_START, seq=1,
                trigger_type="first_confluence", parent_sid=3, parent_cycle_id=1)
    assert sub.lenses() == {LENS_CONFLUENCE}
    _set_lifecycle(sub, _START, _END, "parent_end")
    lens_dfs = _lens_dfs(m15_df)
    ctr_attrs_before = set(lens_dfs[LENS_COUNTER].attrs)

    res = _render(sub, m15_df, lens_dfs)

    assert res.meta["lenses"] == (LENS_CONFLUENCE,)
    assert res.trigger is b.source_trigger
    # Confluence lens: mirrored + attributed with ITS path.
    conf = lens_dfs[LENS_CONFLUENCE]
    assert len(conf.attrs["events"]) == len(res.events) > 0
    expected = _expected_attribution(LENS_CONFLUENCE, sub, b)
    for kind, meta in _element_metas(conf):
        assert {k: meta.get(k) for k in _ATTRIBUTION_KEYS} == expected, kind
    # Counter lens: NOT in sub.lenses() -> nothing mirrored, nothing painted.
    ctr = lens_dfs[LENS_COUNTER]
    for key in ("events", "kl_zones", "poi_zones", "fib_states", "wave_candles",
                "wvmi", "prev_bos_lines"):
        assert key not in ctr.attrs or ctr.attrs[key] == [], key
    assert set(ctr.attrs) == ctr_attrs_before
    assert (ctr["structure_id"] == _SENTINEL_SID).all()
    assert (ctr["market_state"] == _SENTINEL_STATE).all()


# ---------------------------------------------------------------------------
# R5 — events clipped by knowable-at at the cap (inclusive), deep-copied
# ---------------------------------------------------------------------------

def test_events_clipped_by_knowable_at_inclusive_at_cap(geometry, m15_df):
    pool, sub = geometry
    _two_lens_records(pool, sub, m15_df)
    _set_lifecycle(sub, _START, _END, "parent_end")
    bounded, slice_begin = sub.geometry
    cap_local = _END - slice_begin

    # Derived expectation: the geometry's events whose knowable-at <= cap_local,
    # shifted to entity-absolute. Preconditions make the clip REAL: at least
    # one event sits exactly AT the cap (pins the inclusive `<=`) and at least
    # one lies past it (the reversal tail).
    expected = sorted(
        (ev.type, int(ev.idx) + slice_begin, _knowable_at(ev) + slice_begin)
        for ev in bounded.events if _knowable_at(ev) <= cap_local
    )
    assert any(_knowable_at(ev) == cap_local for ev in bounded.events)
    assert any(_knowable_at(ev) > cap_local for ev in bounded.events)
    assert 0 < len(expected) < len(bounded.events)

    lens_dfs = _lens_dfs(m15_df)
    res = _render(sub, m15_df, lens_dfs)

    # The projection's events are slice-local and already clipped …
    assert sorted((ev.type, int(ev.idx), _knowable_at(ev)) for ev in res.events) == sorted(
        (t, i - slice_begin, k - slice_begin) for (t, i, k) in expected
    )
    # … and each lens holds the same set translated to entity-absolute (idx AND
    # the idx-bearing meta such as `confirmed_at` shift by slice_begin).
    for lens in (LENS_CONFLUENCE, LENS_COUNTER):
        got = sorted(
            (ev.type, int(ev.idx), _knowable_at(ev)) for ev in lens_dfs[lens].attrs["events"]
        )
        assert got == expected, lens
        assert max(k for (_, _, k) in got) == _END          # the event AT the cap is in
        assert all(k <= sub.end_idx for (_, _, k) in got)


def test_bos_straddling_the_cap_is_clipped_by_its_moment_not_its_anchor(geometry, m15_df):
    """The BOS_0 of this geometry has its EXTREME (anchor) at 55 and is CONFIRMED
    at 57 (entity-absolute; its `ev.idx` since Plan E E4b). A cap of 56 lies
    between them: the anchor (55) is inside the window but the break was not
    knowable until 57 -> the event must be excluded. A cap of 57 includes it.
    (start_idx = the anchor here — the only way to put a cap between the two.)"""
    pool, sub = geometry
    bounded, slice_begin = sub.geometry
    bos = _bos0(bounded)
    bos_anchor_abs = ef.bos_anchor_idx(bos) + slice_begin
    bos_known_abs = int(bos.meta["confirmed_at"]) + slice_begin
    assert bos_anchor_abs == _STARTING_IDX and bos_known_abs == _STARTING_IDX + 2   # 55 / 57

    _record(pool, sub, m15_df, LENS_CONFLUENCE, start_idx=_STARTING_IDX, seq=0,
            trigger_type="first_confluence", parent_sid=3, parent_cycle_id=1)

    _set_lifecycle(sub, _STARTING_IDX, bos_known_abs - 1, "parent_end")      # cap 56
    res_out = _render(sub, m15_df, _lens_dfs(m15_df))
    assert all(ev.type != "BOS_CONFIRMED" for ev in res_out.events)
    assert all(_knowable_at(ev) + slice_begin <= bos_known_abs - 1 for ev in res_out.events)

    _set_lifecycle(sub, _STARTING_IDX, bos_known_abs, "parent_end")          # cap 57
    lens_dfs = _lens_dfs(m15_df)
    res_in = _render(sub, m15_df, lens_dfs)
    kinds_in = [ev.type for ev in res_in.events]
    assert "BOS_CONFIRMED" in kinds_in
    mirrored_bos = [ev for ev in lens_dfs[LENS_CONFLUENCE].attrs["events"] if ev.type == "BOS_CONFIRMED"]
    assert len(mirrored_bos) == 1
    # Entity-absolute after the mirror: anchor 55, confirmed_at = idx 57 (the
    # mirror shifts idx and both meta keys by the same offset).
    assert ef.bos_anchor_idx(mirrored_bos[0]) == bos_anchor_abs
    assert int(mirrored_bos[0].idx) == int(mirrored_bos[0].meta["confirmed_at"]) == bos_known_abs


def test_mirrored_events_are_deep_copies_of_the_shared_geometry(geometry, m15_df):
    pool, sub = geometry
    _two_lens_records(pool, sub, m15_df)
    _set_lifecycle(sub, _START, _END, "parent_end")
    bounded, slice_begin = sub.geometry
    geometry_ids = {id(ev) for ev in bounded.events}
    geometry_meta_keys_before = [set(ev.meta) for ev in bounded.events]
    lens_dfs = _lens_dfs(m15_df)

    res = _render(sub, m15_df, lens_dfs)

    # (a) The projection hands out COPIES (clip deep-copies): no result event is
    #     a geometry object.
    assert res.events and all(id(ev) not in geometry_ids for ev in res.events)
    # (b) Mutating a mirrored event's meta must not reach the geometry.
    for lens in (LENS_CONFLUENCE, LENS_COUNTER):
        for ev in lens_dfs[lens].attrs["events"]:
            assert id(ev) not in geometry_ids
            ev.meta["__mutated_by_test__"] = lens
    for ev, keys_before in zip(bounded.events, geometry_meta_keys_before):
        assert "__mutated_by_test__" not in ev.meta
        # nor did the mirror's attribution stamp leak onto the shared objects
        assert "sub_id" not in ev.meta and "structure_path_id" not in ev.meta
        assert set(ev.meta) == keys_before
    # (c) And the geometry's idx values are still slice-local (nothing shifted
    #     them in place): every geometry event idx < len(bounded.df).
    assert all(0 <= int(ev.idx) < len(bounded.df) for ev in bounded.events)


# ---------------------------------------------------------------------------
# R6 — KL zones inherit the sub's window (floor clamp + cap end/reason)
# ---------------------------------------------------------------------------

def test_kl_zones_take_the_floor_and_the_cap(geometry, m15_df):
    pool, sub = geometry
    _two_lens_records(pool, sub, m15_df)
    _set_lifecycle(sub, _START, _END, "parent_end")
    bounded, slice_begin = sub.geometry
    # Precondition: the BOS_0 zone's RAW first-active (its confirmed_at, 57)
    # precedes the sub's start (60) — so the floor clamp is what decides
    # `confirmed_idx` (B1: `max(raw_confirmed, struct_start)` with
    # `struct_start` floored at `lifecycle_floor = start_idx`).
    raw_confirmed_abs = int(_bos0(bounded).meta["confirmed_at"]) + slice_begin
    assert raw_confirmed_abs < _START
    # Precondition: the cap (85) precedes the natural reversal (107), so the
    # cap is the earliest end term (B2: `end = min(next_cycle, reversal, cap)`).
    assert _END < sub.natural_reversal_idx

    lens_dfs = _lens_dfs(m15_df)
    res = _render(sub, m15_df, lens_dfs)

    assert res.kl_zones, "the fixture derives a BOS zone"
    for lens in (LENS_CONFLUENCE, LENS_COUNTER):
        zones = lens_dfs[lens].attrs["kl_zones"]
        assert len(zones) == len(res.kl_zones)
        for z in zones:
            assert z.meta["confirmed_idx"] >= sub.start_idx
            assert z.meta["end_idx"] <= sub.end_idx
            assert z.meta["end_reason"] == sub.end_reason == "parent_end"
        bos_zones = [z for z in zones if z.source_kind == "BOS"]
        assert bos_zones
        # The BOS_0 zone: clamped first-active == the floor (raw 57 -> 60);
        # end == the cap (85), reason == the sub's.
        z0 = min(bos_zones, key=lambda z: z.meta["anchor_idx"])
        assert z0.meta["confirmed_idx"] == _START
        assert z0.meta["end_idx"] == _END
        # The zone's structural anchor is the BOS extreme (55) — historical,
        # NOT floored (a floor is never applied to a historical field, §1).
        assert z0.meta["anchor_idx"] == ef.bos_anchor_idx(_bos0(bounded)) + slice_begin == _STARTING_IDX
        # Fib lifecycle scalars follow the same window (FIB_LIFECYCLE_SPEC §15).
        for f in lens_dfs[lens].attrs["fib_states"]:
            assert f.start_idx >= sub.start_idx
            assert f.end_idx is not None and f.end_idx <= sub.end_idx
        # Everything mirrored is entity-absolute inside the frame.
        n = len(m15_df)
        for ev in lens_dfs[lens].attrs["events"]:
            assert 0 <= int(ev.idx) < n
        for w in lens_dfs[lens].attrs["wave_candles"]:
            for v in (w.first_wave_candle_idx, w.last_wave_candle_idx):
                assert v is None or 0 <= int(v) < n


def _mirrored_pois_first_activations(sub, m15_df):
    """Render the sub and return (res, {lens: [(ic_idx, first activation idx,
    meta cts_established_idx)]}) — all entity-absolute after the mirror."""
    lens_dfs = _lens_dfs(m15_df)
    res = _render(sub, m15_df, lens_dfs)
    out = {}
    for lens in (LENS_CONFLUENCE, LENS_COUNTER):
        zones = lens_dfs[lens].attrs["poi_zones"]
        assert len(zones) == len(res.poi_zones) >= 1
        out[lens] = [(int(z.ic_idx), z.meta["activation_history"][0]["idx"],
                      z.meta["cts_established_idx"])
                     for z in zones if z.meta["activation_history"]]
        assert out[lens], "at least one mirrored POI activates"
    return res, out


def _cycle_cts_established(sub, cycle_id):
    """The geometry's (slice-local) CTS_ESTABLISHED for `cycle_id`."""
    bounded, _ = sub.geometry
    return next(ev for ev in bounded.events if ev.type == "CTS_ESTABLISHED"
                and ev.meta.get("cycle_id") == cycle_id)


def test_poi_first_activation_at_or_after_sub_start(geometry, m15_df):
    """The POI twin of the R6 KL pin (Plan D §5) in an OPEN window — under the
    R6 cap (85) the only fib is active-unlocked with an end, so poi_zones' fib
    gate yields zero POIs and a pin there would pass vacuously. Here the floor
    does NOT bind (IC 63 and the raw activation 75 are both after start 60), so
    `first >= start_idx` holds by construction: this test pins non-vacuity and
    the mirror's rebase of `cts_established_idx` (the cycle's moment,
    slice-local -> entity-absolute via `_ZONE_META_IDX_KEYS`). The floor
    plumbing is pinned by the floor-decides test below; the moment rule by
    tests/test_poi_activation_moment.py and the moved-moment test below."""
    pool, sub = geometry
    _two_lens_records(pool, sub, m15_df)
    _set_lifecycle(sub, _START, None, None)
    _, slice_begin = sub.geometry
    moment_abs = int(_cycle_cts_established(sub, 0).meta["confirmed_at"]) + slice_begin
    _, firsts = _mirrored_pois_first_activations(sub, m15_df)
    for lens, rows in firsts.items():
        for _ic, first, _cse in rows:
            assert first >= sub.start_idx
        assert rows == [(63, 75, moment_abs)], (lens, rows)


def test_poi_floor_decides_first_activation_through_the_projection(geometry, m15_df):
    """Start the sub AFTER the raw activation (75): the sub's start_idx (78)
    reaches the POI activation floor through render_sub_projection. (The sub's
    start_idx is set directly, as `_set_lifecycle` does elsewhere; the records
    are not re-derived — only the projection's floor plumbing is under test.)"""
    pool, sub = geometry
    _two_lens_records(pool, sub, m15_df)
    _set_lifecycle(sub, 78, None, None)
    _, firsts = _mirrored_pois_first_activations(sub, m15_df)
    for lens, rows in firsts.items():
        assert [(ic, first) for ic, first, _ in rows] == [(63, 78)], (lens, rows)


def test_poi_cycle_term_is_the_moment_through_the_projection(geometry, m15_df):
    """Plan D through projection + mirror (the replay's sub-3 case: a meta-only
    re-value). Move the geometry's cycle-0 moment 2 candles past its anchor
    (CTS_ESTABLISHED and BOS_CONFIRMED together — the definitional identity;
    both idx move with it, Plan E E4a / E4b): the mirrored
    `cts_established_idx` follows the MOMENT, not the anchor; the first
    activation is unchanged (IC 63 still decides)."""
    pool, sub = geometry
    _two_lens_records(pool, sub, m15_df)
    _set_lifecycle(sub, _START, None, None)
    bounded, slice_begin = sub.geometry
    est = _cycle_cts_established(sub, 0)
    bos = next(ev for ev in bounded.events if ev.type == "BOS_CONFIRMED"
               and ev.meta.get("cycle_id") == 0)
    assert est.meta["cts_anchor_idx"] == est.meta["confirmed_at"] == bos.meta["confirmed_at"]
    moved = int(est.meta["confirmed_at"]) + 2
    # A meta edit after construction bypasses the conftest validator: keep the
    # contract by hand (the idx IS the moment, Plan E E4a / E4b).
    est.meta["confirmed_at"] = moved
    est.idx = moved
    bos.meta["confirmed_at"] = moved
    bos.idx = moved
    from engine_v2.tests.conftest import validate_event_contract
    validate_event_contract(est)
    validate_event_contract(bos)
    _, firsts = _mirrored_pois_first_activations(sub, m15_df)
    for lens, rows in firsts.items():
        # Old rule (pre-Plan D): the CTS anchor + slice_begin (2 less).
        assert rows == [(63, 75, moved + slice_begin)], (lens, rows)


# ---------------------------------------------------------------------------
# R7 — _STRUCTURE_COLS painted only over the live rows [start_idx, end]
# ---------------------------------------------------------------------------

def test_structure_cols_painted_only_over_live_rows(geometry, m15_df):
    pool, sub = geometry
    _two_lens_records(pool, sub, m15_df)
    _set_lifecycle(sub, _START, _END, "parent_end")
    bounded, slice_begin = sub.geometry
    lens_dfs = _lens_dfs(m15_df)

    _render(sub, m15_df, lens_dfs)

    n = len(m15_df)
    start_local, cap_local = _START - slice_begin, _END - slice_begin
    for lens in (LENS_CONFLUENCE, LENS_COUNTER):
        ldf = lens_dfs[lens]
        for col, sentinel in (("structure_id", _SENTINEL_SID), ("market_state", _SENTINEL_STATE)):
            assert col in bounded.df.columns, col
            got = ldf[col]
            # Rows BEFORE start_idx: untouched — this covers the 50-candle
            # lookback [5, 54] AND the anchor rows [55, 59] that the geometry
            # painted but the sub was not live for.
            assert (got.iloc[:_START] == sentinel).all(), (lens, col, "pre-start rows painted")
            # Rows [start_idx, end_idx]: the slice's values for the same candles.
            expected_live = bounded.df[col].iloc[start_local:cap_local + 1].tolist()
            assert got.iloc[_START:_END + 1].tolist() == expected_live, (lens, col)
            assert not (got.iloc[_START:_END + 1] == sentinel).any()
            # Rows AFTER the cap: untouched.
            assert (got.iloc[_END + 1:] == sentinel).all(), (lens, col, "post-cap rows painted")
        assert len(ldf) == n                                      # no rows added


def test_later_live_wins_over_an_earlier_sub_when_mirrored_in_start_order(geometry, m15_df):
    """§17.9: subs are mirrored in start_idx order and the mirror paints only
    from `start_idx`, so a LATER-starting sub's lookback / pre-start rows never
    overwrite an EARLIER sub's live rows. Two subs sharing this geometry (only
    the lifecycle differs) into one lens df: sub X live [60, 85], sub Y live
    [80, edge]. Rows [60, 79] must still be X's, rows [80, ...] Y's."""
    pool, sub_x = geometry
    _record(pool, sub_x, m15_df, LENS_CONFLUENCE, start_idx=_START, seq=0,
            trigger_type="first_confluence", parent_sid=3, parent_cycle_id=1)
    _set_lifecycle(sub_x, _START, _END, "same_dir_replacement")
    sub_y = PooledStructure(key=sub_x.key, sub_id=sub_x.sub_id + 1, geometry=sub_x.geometry,
                            natural_reversal_idx=sub_x.natural_reversal_idx)
    y_start = _END - 5                                           # 80
    _record(pool, sub_y, m15_df, LENS_CONFLUENCE, start_idx=y_start, seq=1,
            trigger_type="subsequent_confluence", parent_sid=3, parent_cycle_id=2)
    _set_lifecycle(sub_y, y_start, None, None)
    lens_dfs = _lens_dfs(m15_df)
    conf = lens_dfs[LENS_CONFLUENCE]

    _render(sub_x, m15_df, lens_dfs)
    x_rows = conf["structure_id"].iloc[_START:y_start].tolist()
    assert not any(v == _SENTINEL_SID for v in x_rows)
    _render(sub_y, m15_df, lens_dfs)                             # later-live, mirrored second

    # X's rows before Y's start are intact (Y painted nothing before 80) …
    assert conf["structure_id"].iloc[_START:y_start].tolist() == x_rows
    assert (conf["structure_id"].iloc[:_START] == _SENTINEL_SID).all()
    # … and Y owns [80, edge] (an open sub paints to the frame edge).
    assert not (conf["structure_id"].iloc[y_start:] == _SENTINEL_SID).any()
    # Attribution tells the two apart: events from both subs, each with its own sub_id.
    assert {e.meta["sub_id"] for e in conf.attrs["events"]} == {sub_x.sub_id, sub_y.sub_id}


# ---------------------------------------------------------------------------
# R1 / R3 / R6 / R7 — an OPEN sub (end_idx None)
# ---------------------------------------------------------------------------

def test_open_sub_paints_to_edge_and_meta_end_is_none(geometry, m15_df):
    pool, sub = geometry
    z, a, b = _two_lens_records(pool, sub, m15_df)
    _set_lifecycle(sub, _START, None, None)
    bounded, slice_begin = sub.geometry
    n = len(m15_df)
    edge = slice_begin + len(bounded.df) - 1                     # the geometry edge …
    assert edge == n - 1                                         # … = the data edge (run cap, §5.4)
    lens_dfs = _lens_dfs(m15_df)

    res = _render(sub, m15_df, lens_dfs)

    # Meta: end_idx None while open; m15_end_idx = the edge.
    assert res.meta["end_idx"] is None
    assert res.meta["m15_end_idx"] == edge
    assert res.meta["end_reason"] is None
    assert res.meta["start_idx"] == _START
    # No cap -> no clip: every geometry event is projected (deep-copied) and
    # mirrored; the result df is the whole slice.
    assert len(res.events) == len(bounded.events)
    assert len(res.df) == len(bounded.df)
    for lens in (LENS_CONFLUENCE, LENS_COUNTER):
        ldf = lens_dfs[lens]
        assert len(ldf.attrs["events"]) == len(bounded.events)
        # Paint: [start_idx, edge] painted, rows before start_idx untouched.
        sid = ldf["structure_id"]
        assert (sid.iloc[:_START] == _SENTINEL_SID).all()
        assert not (sid.iloc[_START:] == _SENTINEL_SID).any()
        assert sid.iloc[_START:].tolist() == bounded.df["structure_id"].iloc[_START - slice_begin:].tolist()
        # KL: with no cap the earliest end term is the structure's own reversal
        # (STATE_CHANGED -> reversal at the natural reversal idx), tagged
        # "reversal"; the floor clamp still applies.
        zones = ldf.attrs["kl_zones"]
        assert zones
        for zn in zones:
            assert zn.meta["confirmed_idx"] >= _START
            assert zn.meta["end_idx"] == sub.natural_reversal_idx
            assert zn.meta["end_reason"] == "reversal"
        # Attribution on the open sub is the same contract.
        expected = _expected_attribution(lens, sub, b)
        for kind, meta in _element_metas(ldf):
            assert {k: meta.get(k) for k in _ATTRIBUTION_KEYS} == expected, kind


# ---------------------------------------------------------------------------
# R8 — a sub with no live record is not renderable
# ---------------------------------------------------------------------------

def test_sub_with_no_live_record_raises_and_touches_nothing(geometry, m15_df):
    pool, sub = geometry
    # Only a zero-length record: the sweep never sets `start_idx` for such a
    # sub (§4.4 phase 2 requires a non-zero-length record) and §4.5 skips it
    # ("no live record: logged, NOT rendered — such a sub has no lens").
    _record(pool, sub, m15_df, LENS_CONFLUENCE, start_idx=_START, seq=0,
            trigger_type="first_confluence", parent_sid=3, parent_cycle_id=1,
            trigger_end_idx=_START)
    assert sub.live_records() == [] and sub.lenses() == set()
    assert sub.start_idx is None
    lens_dfs = _lens_dfs(m15_df)

    with pytest.raises(AssertionError):
        _render(sub, m15_df, lens_dfs)

    for ldf in lens_dfs.values():
        assert "events" not in ldf.attrs and "kl_zones" not in ldf.attrs
        assert (ldf["structure_id"] == _SENTINEL_SID).all()
