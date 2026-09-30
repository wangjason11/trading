"""Tests for the reversal-handoff resolver (Plan C §4.3 step 2 "reversal",
§5.3; PART4 §17.8) — `entity_df_mutation._resolve_reversal_start`.

What is pinned:

  * the handoff probe runs on the reversing sub's SLICE-LOCAL geometry
    (`bounded.events` / `bounded.df`, bound = `bounded.reversal_idx`) and EVERY
    returned idx is shifted by `slice_begin` (Plan C §4.3 step 2);
  * the reference is the reversing structure's (sid=0) most recent CTS, read
    from the sub's OWN events with `kl_zones=[]` (§5.1 note / §17.8);
  * the probe cache key is ENTITY-ABSOLUTE — `(parent_path, "M15",
    probe_direction, input_abs = anchor_idx + slice_begin)` (§5.3; the
    pool is shared across subs, so a slice-local key would collide) — and the
    entry stores the probe's OWN bound `probe_end_idx = R` and the reference
    inner it ran against;
  * a second call of the same key is a HIT: `unified_probe` is skipped and the
    cached values are returned verbatim, regardless of the bound (§17.8 (a));
  * `pool=None` runs without caching;
  * the failure branches (no CTS reference / input >= reversal / pending) return
    a `ProbeFailure` — never a `ResolvedStart`, never a cache write;
  * `R` must be the sub's natural reversal (asserted against the geometry).

Fixture: the reversing 15-minute series validated in `test_bounded_structure`
(reverses at its local idx 52 of 61; builders copied from
`test_geometry_builder.py`), prepended with `_PREFIX_N = 70` flat noise candles
so the structure starts at entity-absolute 70 and the geometry's `slice_begin =
70 - 50 = 20` is NON-ZERO — the only way a missing shift is observable. The
reversal probe is Phase-1 only (`enable_phase2=False`), bounded at the
reversal candle; per the `ProbeResult` docstring both finalized Phase-1
conditions (`no_retrace`, `end_idx_reached`) finalize AT `probe_end_idx`, so the
successor's `finalize_idx == R` by rule (§4.3: "Successor start_idx =
max(finalize=R, trigger=R, floor) = R").

Expected values are DERIVED: the reference zone and the slice-local probe are
recomputed here from the same primitives on the same slice-local geometry, then
shifted — the resolver must agree with that independent computation, not with
its own printout.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from engine_v2.structure import event_fields as ef
from engine_v2.multitf import entity_df_mutation as edm
from engine_v2.multitf.lifecycle_sweep import ProbeFailure, ResolvedStart
from engine_v2.multitf.sub_structure_pool import (
    PooledStructure,
    ProbeCacheEntry,
    StructureKey,
    SubStructurePool,
)
from engine_v2.structure import unified_probe as up_mod
from engine_v2.structure.reference_zone import build_reference_zone_from_cts_event
from engine_v2.structure.unified_probe import ProbeResult
from engine_v2.features.candle_classifier import apply_candle_classification
from engine_v2.patterns.pattern_engine import detect_patterns
from engine_v2.patterns.imbalance import compute_imbalance


# ---------------------------------------------------------------------------
# Fixtures (copied from test_geometry_builder.py — 15-minute candles)
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


# >= 60 so that slice_begin = start_abs - 50 is comfortably non-zero (20).
_PREFIX_N = 70
_REV_LOCAL_IN_SERIES = 52          # the reversing series' own reversal idx


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
_SD = 1                             # the reversing sub's direction (+1)
_PROBE_DIR = -_SD                   # the successor probes the flipped direction


@pytest.fixture(scope="module")
def shifted_df() -> pd.DataFrame:
    """70 flat + 61 reversing = 131 candles; the +1 structure starts at 70."""
    return _prepare_df(_make_flat_prefix() + _make_reversing_data())


@pytest.fixture
def built(shifted_df):
    """A FRESH pool + the real reversing sub built through the pool's own
    geometry builder (so `sub.geometry`, `natural_reversal_idx` and
    `slice_begin` are real). Function-scoped: the probe cache is per test."""
    pool = SubStructurePool()
    out = edm.build_or_get_geometry(
        pool, shifted_df, parent_path=_PARENT, sd=_SD, start_abs=_PREFIX_N,
        bos0_inner=None, timeframe=_TF,
    )
    assert out is not None, "fixture must build"
    sub, created = out
    assert created
    bounded, slice_begin = sub.geometry
    # Fixture sanity (derived, not observed):
    #   slice_begin = max(0, start - 50) = 70 - 50 = 20  (> 0: the shift is real)
    assert slice_begin == _PREFIX_N - 50 == 20
    #   the series reverses at its local 52 → entity-absolute 70 + 52 = 122
    assert sub.natural_reversal_idx is not None, "fixture must still reverse"
    assert sub.natural_reversal_idx == _PREFIX_N + _REV_LOCAL_IN_SERIES
    #   geometry stores the SLICE-LOCAL reversal: 122 - 20 = 102
    assert int(bounded.reversal_idx) == sub.natural_reversal_idx - slice_begin
    return pool, sub, shifted_df


def _expected_reference(sub):
    """The reference the rule prescribes: the reversing structure's (sid=0) most
    recent CTS, read from the sub's OWN slice-local events with `kl_zones=[]`,
    no idx window (Plan C §4.3 step 2 / §5.1 note)."""
    bounded, _ = sub.geometry
    rz = build_reference_zone_from_cts_event(
        bounded.events, [], bounded.df, sid=0,
        probe_direction=_PROBE_DIR, idx_window=None,
    )
    assert rz is not None, "the reversing fixture has CTS events"
    return rz


def _expected_local_probe(sub, R):
    """The slice-local probe the resolver must run: Phase-1 only, input = the
    reference's `anchor_idx` (local), bound = `R - slice_begin`."""
    bounded, slice_begin = sub.geometry
    rz = _expected_reference(sub)
    return rz, up_mod.unified_probe(
        bounded.df,
        input_idx=int(rz.anchor_idx),
        direction=_PROBE_DIR,
        reference_zone=rz,
        probe_end_idx=int(R) - int(slice_begin),
        timeframe=_TF,
        enable_phase2=False,
    )


def _resolve(pool, sub, R, df, probe_direction=_PROBE_DIR):
    return edm._resolve_reversal_start(
        pool, sub, R, probe_direction, df, timeframe=_TF, parent_path=_PARENT,
    )


class _CountingProbe:
    """Wrap the REAL `unified_probe` so call counts are exact but behaviour is
    unchanged. `_resolve_reversal_start` imports the function INSIDE its body
    (`from engine_v2.structure.unified_probe import unified_probe`), so the
    module attribute is the binding to patch."""

    def __init__(self):
        self.real = up_mod.unified_probe
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.real(*args, **kwargs)


# ---------------------------------------------------------------------------
# (1) happy path — slice-local probe, every returned idx shifted by slice_begin
# ---------------------------------------------------------------------------

def test_happy_path_shifts_every_idx_by_slice_begin(built):
    pool, sub, df = built
    bounded, slice_begin = sub.geometry
    R = sub.natural_reversal_idx                                  # 122

    # Independent expectation: reference from the sub's own events, probe
    # slice-local, then shift.
    rz, local = _expected_local_probe(sub, R)
    assert local.status == "finalized"
    expected_input_abs = int(rz.anchor_idx) + slice_begin

    res = _resolve(pool, sub, R, df)

    assert isinstance(res, ResolvedStart)
    # finalize == R: a Phase-1 probe bounded at R finalizes AT its bound
    # (ProbeResult: `no_retrace` / `end_idx_reached` → probe_end_idx), and
    # the resolver's bound is the reversal candle → local R-20, shifted → R.
    assert res.finalize_condition in {"no_retrace", "end_idx_reached"}
    assert res.finalize_idx == R
    assert res.finalize_idx == int(local.finalize_idx) + slice_begin
    # starting_idx = the slice-local probe's anchor + slice_begin.
    assert res.starting_idx == int(local.starting_idx) + slice_begin
    # probe_input_idx = the reference's source event, shifted.
    assert res.probe_input_idx == expected_input_abs
    # The shift is real (slice_begin = 20 > 0): a slice-local return would be
    # 20 candles too early on every field.
    assert res.starting_idx != int(local.starting_idx)
    assert res.finalize_idx != int(local.finalize_idx)
    assert res.probe_input_idx != int(rz.anchor_idx)
    # Entity-absolute: inside the entity frame and never before the slice.
    for v in (res.starting_idx, res.finalize_idx, res.probe_input_idx):
        assert slice_begin <= v < len(df)
    # A Phase-1 probe's anchor is at/after its input (resets move forward).
    assert res.probe_input_idx <= res.starting_idx <= res.finalize_idx
    # Reversal-born: no parent probe seeded it (§2.1 parent_bos_anchor_idx None).
    assert res.parent_bos_anchor_idx is None
    # First probe of this key → it RAN (no hit).
    assert res.cache_hit is False
    # bos0_inner is the probe's own (iter-1 == the reference inner).
    assert res.bos0_inner == pytest.approx(local.bos0_inner)


def test_reference_is_the_sub_own_most_recent_cts(built):
    """The reference the resolver ran against is derivable from the sub's OWN
    events alone (sid=0, kl_zones=[]): its inner is what the cache stores as
    `ref_inner`, and its `anchor_idx` (+ slice_begin) is the input."""
    pool, sub, df = built
    bounded, slice_begin = sub.geometry
    R = sub.natural_reversal_idx
    rz = _expected_reference(sub)
    # The winning event is the most recent {CONF/UPD/EST} CTS of sid 0 — its
    # CTS anchor (`ef.cts_anchor_idx`: the meta key on CONF/EST, idx on UPD) is
    # the source idx.
    cts = [e for e in bounded.events
           if e.type in ("CTS_CONFIRMED", "CTS_UPDATED", "CTS_ESTABLISHED")
           and e.meta.get("structure_id") == 0]
    assert cts
    winner = max(cts, key=lambda e: (int(e.idx), {"CTS_CONFIRMED": 2, "CTS_UPDATED": 1,
                                                  "CTS_ESTABLISHED": 0}[e.type]))
    extreme = ef.cts_anchor_idx(winner)
    assert int(rz.anchor_idx) == extreme
    # Probe direction -1 → the reference sits ABOVE the body: inner < outer.
    assert rz.inner < rz.outer and rz.side == "sell"

    res = _resolve(pool, sub, R, df)
    assert isinstance(res, ResolvedStart)
    assert res.probe_input_idx == extreme + slice_begin
    hit = pool.get_cached_probe(_PARENT, _TF, _PROBE_DIR, extreme + slice_begin)
    assert hit is not None
    assert hit.ref_inner == pytest.approx(float(rz.inner))


# ---------------------------------------------------------------------------
# (2) cache entry under the ENTITY-ABSOLUTE key
# ---------------------------------------------------------------------------

def test_cache_entry_is_written_under_entity_absolute_key(built):
    pool, sub, df = built
    bounded, slice_begin = sub.geometry
    R = sub.natural_reversal_idx
    rz, local = _expected_local_probe(sub, R)
    input_local = int(rz.anchor_idx)
    input_abs = input_local + slice_begin

    assert pool.get_cached_probe(_PARENT, _TF, _PROBE_DIR, input_abs) is None
    res = _resolve(pool, sub, R, df)
    assert isinstance(res, ResolvedStart)

    # Key = (parent_path, "M15", probe_direction, input_abs) — §5.3.
    entry = pool.get_cached_probe(_PARENT, _TF, _PROBE_DIR, input_abs)
    assert isinstance(entry, ProbeCacheEntry)
    # The entry stores the probe's OWN bound (R, entity-absolute) …
    assert entry.probe_end_idx == R
    # … the reference inner it ran against (the §5.3 tripwire comparand) …
    assert entry.ref_inner == pytest.approx(float(rz.inner))
    # … and the shifted result, byte-equal to what was returned.
    assert entry.starting_idx == res.starting_idx == int(local.starting_idx) + slice_begin
    assert entry.finalize_idx == res.finalize_idx == R
    assert entry.finalize_condition == res.finalize_condition == local.finalize_condition
    assert entry.bos0_inner == pytest.approx(res.bos0_inner)

    # NOT under the slice-local input (slice_begin = 20 separates them), not
    # under the sub's own direction, not under another sub_tf.
    assert pool.get_cached_probe(_PARENT, _TF, _PROBE_DIR, input_local) is None
    assert pool.get_cached_probe(_PARENT, _TF, _SD, input_abs) is None
    assert pool.get_cached_probe(_PARENT, "M5", _PROBE_DIR, input_abs) is None
    assert pool.get_cached_probe("H1.other", _TF, _PROBE_DIR, input_abs) is None


# ---------------------------------------------------------------------------
# (3) second call = cache hit: unified_probe skipped, identical values
# ---------------------------------------------------------------------------

def test_second_call_is_a_cache_hit_and_skips_unified_probe(built):
    pool, sub, df = built
    R = sub.natural_reversal_idx
    counting = _CountingProbe()
    with patch.object(up_mod, "unified_probe", new=counting):
        first = _resolve(pool, sub, R, df)
        assert isinstance(first, ResolvedStart)
        assert counting.calls == 1                   # the probe RAN once
        assert first.cache_hit is False

        second = _resolve(pool, sub, R, df)
        assert counting.calls == 1                   # UNCHANGED: hit skips the probe
    assert isinstance(second, ResolvedStart)
    assert second.cache_hit is True
    # Identical starting / finalize (the cached entry is the truth).
    assert second.starting_idx == first.starting_idx
    assert second.finalize_idx == first.finalize_idx == R
    assert second.finalize_condition == first.finalize_condition
    assert second.probe_input_idx == first.probe_input_idx
    assert second.bos0_inner == pytest.approx(first.bos0_inner)
    assert second.parent_bos_anchor_idx is None


@pytest.mark.parametrize("bound_delta", [0, -7], ids=["same_bound", "different_bound_APPROX"])
def test_seeded_cache_entry_is_returned_verbatim(built, bound_delta):
    """§17.8 hit rule: the first probe to finalize for a key is the truth for
    every later probe of that key, (a) regardless of its bound. Seed the cache
    under the entity-absolute key with sentinel values; the resolver must
    return them and never call `unified_probe`."""
    pool, sub, df = built
    bounded, slice_begin = sub.geometry
    R = sub.natural_reversal_idx
    rz = _expected_reference(sub)
    input_abs = int(rz.anchor_idx) + slice_begin

    sentinel = ProbeCacheEntry(
        starting_idx=999, finalize_idx=998, finalize_condition="end_idx_reached",
        bos0_inner=0.123456, probe_end_idx=R + bound_delta, ref_inner=float(rz.inner),
    )
    pool.record_probe(_PARENT, _TF, _PROBE_DIR, input_abs, sentinel)

    boom = MagicMock(side_effect=AssertionError("unified_probe must not run on a hit"))
    with patch.object(up_mod, "unified_probe", new=boom):
        res = _resolve(pool, sub, R, df)
    assert boom.call_count == 0
    assert isinstance(res, ResolvedStart)
    assert res.cache_hit is True
    assert res.starting_idx == 999 and res.finalize_idx == 998
    assert res.finalize_condition == "end_idx_reached"
    assert res.bos0_inner == pytest.approx(0.123456)
    assert res.probe_input_idx == input_abs
    assert res.parent_bos_anchor_idx is None
    # The entry is untouched (first write wins; a hit writes nothing).
    assert pool.get_cached_probe(_PARENT, _TF, _PROBE_DIR, input_abs) == sentinel


def test_seeded_entry_under_slice_local_key_is_not_a_hit(built):
    """The converse of the key rule: an entry under the SLICE-LOCAL input is
    invisible — the resolver probes (miss) and writes the absolute key."""
    pool, sub, df = built
    bounded, slice_begin = sub.geometry
    R = sub.natural_reversal_idx
    rz = _expected_reference(sub)
    input_local = int(rz.anchor_idx)
    decoy = ProbeCacheEntry(
        starting_idx=999, finalize_idx=998, finalize_condition="end_idx_reached",
        bos0_inner=0.5, probe_end_idx=R, ref_inner=float(rz.inner),
    )
    pool.record_probe(_PARENT, _TF, _PROBE_DIR, input_local, decoy)

    counting = _CountingProbe()
    with patch.object(up_mod, "unified_probe", new=counting):
        res = _resolve(pool, sub, R, df)
    assert counting.calls == 1                       # a MISS: the probe ran
    assert isinstance(res, ResolvedStart)
    assert res.cache_hit is False
    assert res.starting_idx != 999 and res.finalize_idx == R
    assert pool.get_cached_probe(_PARENT, _TF, _PROBE_DIR, input_local + slice_begin) is not None


# ---------------------------------------------------------------------------
# (4) wrong R → AssertionError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("delta", [+1, -1, +20])
def test_wrong_reversal_idx_is_an_assertion(built, delta):
    """`R` must be the sub's natural reversal: `bounded.reversal_idx ==
    R - slice_begin` is asserted before anything runs (the bound and the cache
    key would otherwise be silently wrong)."""
    pool, sub, df = built
    R = sub.natural_reversal_idx
    with pytest.raises(AssertionError):
        _resolve(pool, sub, R + delta, df)
    # Nothing was probed or cached (the assert precedes the reference read).
    rz = _expected_reference(sub)
    _, slice_begin = sub.geometry
    assert pool.get_cached_probe(_PARENT, _TF, _PROBE_DIR,
                                 int(rz.anchor_idx) + slice_begin) is None


def test_sub_without_geometry_is_an_assertion(built):
    pool, sub, df = built
    bare = PooledStructure(key=sub.key, sub_id=sub.sub_id, geometry=None,
                           natural_reversal_idx=sub.natural_reversal_idx)
    with pytest.raises(AssertionError):
        _resolve(pool, bare, sub.natural_reversal_idx, df)


# ---------------------------------------------------------------------------
# (5) pool=None → runs, no caching
# ---------------------------------------------------------------------------

def test_pool_none_runs_without_caching(built):
    pool, sub, df = built
    R = sub.natural_reversal_idx
    counting = _CountingProbe()
    with patch.object(up_mod, "unified_probe", new=counting):
        a = _resolve(None, sub, R, df)
        b = _resolve(None, sub, R, df)
    # No cache → the probe runs EVERY time (2 calls), never a hit.
    assert counting.calls == 2
    assert isinstance(a, ResolvedStart) and isinstance(b, ResolvedStart)
    assert a.cache_hit is False and b.cache_hit is False
    assert a == b                                    # deterministic re-probe

    # Same values as the pooled run (the pool only adds the cache).
    pooled = _resolve(pool, sub, R, df)
    assert isinstance(pooled, ResolvedStart)
    assert (a.starting_idx, a.finalize_idx, a.finalize_condition, a.probe_input_idx) == (
        pooled.starting_idx, pooled.finalize_idx, pooled.finalize_condition,
        pooled.probe_input_idx,
    )
    assert a.finalize_idx == R


# ---------------------------------------------------------------------------
# (6) failure branches → ProbeFailure
# ---------------------------------------------------------------------------

def test_no_cts_event_is_a_probe_failure(built):
    """A sub whose geometry has NO CTS event of any type → no reference →
    `ProbeFailure` (no successor), no probe run, no cache write."""
    pool, sub, df = built
    bounded, slice_begin = sub.geometry
    R = sub.natural_reversal_idx
    fake = PooledStructure(
        key=sub.key, sub_id=sub.sub_id,
        geometry=(SimpleNamespace(events=[], df=bounded.df,
                                  reversal_idx=R - slice_begin), slice_begin),
        natural_reversal_idx=R,
    )
    boom = MagicMock(side_effect=AssertionError("unified_probe must not run"))
    with patch.object(up_mod, "unified_probe", new=boom):
        res = _resolve(pool, fake, R, df)
    assert boom.call_count == 0
    assert isinstance(res, ProbeFailure)
    assert not isinstance(res, ResolvedStart)
    assert "no CTS" in res.detail
    assert res.probe_input_idx is None               # no input was ever derived
    assert pool.get_cached_probe(_PARENT, _TF, _PROBE_DIR, R) is None


def test_non_cts_events_only_is_a_probe_failure(built):
    """Same branch with events present but none of the three CTS types (the
    primitive filters on type, so BOS/STATE/RANGE events do not count)."""
    pool, sub, df = built
    bounded, slice_begin = sub.geometry
    R = sub.natural_reversal_idx
    non_cts = [e for e in bounded.events
               if e.type not in ("CTS_CONFIRMED", "CTS_UPDATED", "CTS_ESTABLISHED")]
    assert non_cts
    fake = PooledStructure(
        key=sub.key, sub_id=sub.sub_id,
        geometry=(SimpleNamespace(events=non_cts, df=bounded.df,
                                  reversal_idx=R - slice_begin), slice_begin),
        natural_reversal_idx=R,
    )
    res = _resolve(pool, fake, R, df)
    assert isinstance(res, ProbeFailure)
    assert res.probe_input_idx is None


def test_input_at_or_past_reversal_is_a_probe_failure(built):
    """Degenerate window: the reference's source idx >= the reversal (`>=`, so
    EQUAL is already a failure) → `ProbeFailure` carrying the ENTITY-ABSOLUTE
    input; no probe run, no cache write."""
    pool, sub, df = built
    bounded, slice_begin = sub.geometry
    rz = _expected_reference(sub)
    input_local = int(rz.anchor_idx)
    # Spoof the geometry's reversal onto the input candle: R' = input_abs.
    fake_rev_local = input_local
    R_prime = fake_rev_local + slice_begin
    fake = PooledStructure(
        key=sub.key, sub_id=sub.sub_id,
        geometry=(SimpleNamespace(events=bounded.events, df=bounded.df,
                                  reversal_idx=fake_rev_local), slice_begin),
        natural_reversal_idx=R_prime,
    )
    boom = MagicMock(side_effect=AssertionError("unified_probe must not run"))
    with patch.object(up_mod, "unified_probe", new=boom):
        res = _resolve(pool, fake, R_prime, df)
    assert boom.call_count == 0
    assert isinstance(res, ProbeFailure)
    assert res.probe_input_idx == input_local + slice_begin      # entity-absolute
    assert str(input_local + slice_begin) in res.detail and str(R_prime) in res.detail
    assert pool.get_cached_probe(_PARENT, _TF, _PROBE_DIR, input_local + slice_begin) is None


def test_pending_probe_is_a_probe_failure_and_writes_no_cache(built):
    """`status == "pending"` → `ProbeFailure("reversal probe pending",
    input_abs)`; the cache must NOT be written (a pending probe is not "the
    first probe to finalize" for its key)."""
    pool, sub, df = built
    bounded, slice_begin = sub.geometry
    R = sub.natural_reversal_idx
    rz = _expected_reference(sub)
    input_abs = int(rz.anchor_idx) + slice_begin

    def pending(df_, input_idx, direction, reference_zone, probe_end_idx, timeframe, **kw):
        # The resolver must call the probe SLICE-LOCAL: on bounded.df with the
        # local input and the local bound, in the successor's direction.
        assert df_ is bounded.df
        assert input_idx == int(rz.anchor_idx)
        assert probe_end_idx == R - slice_begin
        assert direction == _PROBE_DIR
        assert timeframe == _TF
        assert kw.get("enable_phase2", False) is False        # Phase 1 only
        # … against EXACTLY the prescribed reference: the sub's own most recent
        # CTS, derived for the SUCCESSOR's direction (outer/inner/side all
        # keyed off `probe_direction`; on this fixture a wrong-direction
        # derivation shares the inner but not the outer/side).
        assert reference_zone == rz
        return ProbeResult(
            starting_idx=int(input_idx), status="pending", iterations=1,
            original_ref_zone=reference_zone, finalize_condition="max_iterations",
            bos0_inner=float(reference_zone.inner), finalize_idx=None,
        )

    with patch.object(up_mod, "unified_probe", new=pending):
        res = _resolve(pool, sub, R, df)
    assert isinstance(res, ProbeFailure)
    assert "pending" in res.detail
    assert res.probe_input_idx == input_abs
    assert pool.get_cached_probe(_PARENT, _TF, _PROBE_DIR, input_abs) is None
