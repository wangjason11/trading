"""Guard: every candle-index meta key on a mirrored sub event / zone / fib /
wave-candle result is shifted slice-local -> entity-absolute by the mirror
(Plan E §4.3, IN R5; Post-E·2).

`entity_df_mutation._shift_meta_indices` translates only the keys listed in
`_EVENT_META_IDX_KEYS` / `_ZONE_META_IDX_KEYS` / `_FIB_META_IDX_KEYS` /
`_WAVE_CANDLE_META_IDX_KEYS`, and only values that are a Python `int`. A new or
renamed index key that is missing from its list is exported slice-local on
every M15 row without any error — so this test fails the moment an emitter
starts writing an unlisted `*_idx` / `*_at` key.

There is NO slice-local allow-list any more: Post-E·2 (2026-09-26, PLAN_E §9.3)
moved the last slice-local keys (the coordinate-hygiene families F1–F8) into
the shift lists. A new index key is added to its shift list, never exempted.

Fixture: the production-shaped sub from `test_render_sub_projection.py` (the
reversing M15 series; it emits CTS_ESTABLISHED, BOS_CONFIRMED, CTS_CONFIRMED,
REVERSAL_WATCH_START, REVERSAL_CANDIDATE, RANGE_* and a KL / POI zone, a fib and
wave-candle results), mirrored into both lens dfs by `render_sub_projection`.
Keys the fixture never produces are covered by static scans: every event-meta
key `market_structure` writes (Plan E E2a, closing the E1 gap) and every
`meta=` key `kl_zones_v1` / `poi_zones` / `fib_tracker` / `wave_candles` write
(Post-E·2). The VALUES are pinned by pairing each mirrored element with its
slice-local source: every listed int key is exactly `+ slice_begin`, every
other key unchanged. The Post-E·2 landing review added the last section: the
lists' own validity + the inverse check, `_shift_meta_indices`' contract, every
mirror shift site on a synthetic result, a broad emitter scan and a value-type
classification.
"""
from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest

from engine_v2.structure import event_fields as ef
from engine_v2.structure import market_structure
from engine_v2.zones import fib_tracker, kl_zones_v1, poi_zones, wave_candles

from engine_v2.multitf import entity_df_mutation as edm
from engine_v2.multitf.types import LowerTFResult
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.tests.test_render_sub_projection import (  # noqa: F401 (fixtures)
    _START,
    _lens_dfs,
    _render,
    _set_lifecycle,
    _two_lens_records,
    geometry,
    m15_df,
)

# Index-valued event-meta keys that carry neither the `_idx` nor the `_at` suffix.
EXTRA_EVENT_IDX_KEYS = frozenset({"pb_start"})

# Nested lists the mirror shifts with their own loops: list key -> item key.
NESTED_IDX_LISTS = {"bounds_steps": "start_idx", "activation_history": "idx"}


def _is_idx_key(key: str) -> bool:
    return key.endswith("_idx") or key.endswith("_at")


def _render_both_lenses(geometry, m15_df):
    """(slice-local source result, the two mirrored lens dfs)."""
    pool, sub = geometry
    _two_lens_records(pool, sub, m15_df)
    _set_lifecycle(sub, _START, None, None)
    lens_dfs = _lens_dfs(m15_df)
    res = _render(sub, m15_df, lens_dfs)
    return res, lens_dfs


def _mirrored(geometry, m15_df):
    _, lens_dfs = _render_both_lenses(geometry, m15_df)
    events = [ev for d in lens_dfs.values() for ev in d.attrs.get("events", [])]
    zones = [z for d in lens_dfs.values()
             for z in d.attrs.get("kl_zones", []) + d.attrs.get("poi_zones", [])]
    return events, zones


def _mirrored_attr(geometry, m15_df, attr):
    _, lens_dfs = _render_both_lenses(geometry, m15_df)
    return [x for d in lens_dfs.values() for x in d.attrs.get(attr, [])]


def test_fixture_emits_the_event_types_that_carry_index_keys(geometry, m15_df):
    """Precondition: the guard below is only as good as the element mix it sees."""
    _, lens_dfs = _render_both_lenses(geometry, m15_df)
    for d in lens_dfs.values():
        types = {ev.type for ev in d.attrs.get("events", [])}
        assert {"CTS_ESTABLISHED", "BOS_CONFIRMED", "CTS_CONFIRMED",
                "REVERSAL_WATCH_START", "REVERSAL_CANDIDATE", "RANGE_STARTED"} <= types
        assert d.attrs.get("kl_zones") and d.attrs.get("poi_zones"), "a KL and a POI zone"
        assert d.attrs.get("fib_states"), "… at least one fib"
        assert d.attrs.get("wave_candles"), "… and one wave-candle result"


def test_every_event_meta_index_key_is_shifted(geometry, m15_df):
    events, _ = _mirrored(geometry, m15_df)
    unlisted = sorted({
        (ev.type, k) for ev in events for k in ev.meta
        if (_is_idx_key(k) or k in EXTRA_EVENT_IDX_KEYS)
        and k not in edm._EVENT_META_IDX_KEYS
    })
    assert unlisted == [], f"event meta index keys not in _EVENT_META_IDX_KEYS: {unlisted}"


def test_shifted_event_meta_values_are_int_or_none(geometry, m15_df):
    """`_shift_meta_indices` shifts only a Python `int`: a numpy int or float
    under a listed key would be exported slice-local silently."""
    events, _ = _mirrored(geometry, m15_df)
    bad = sorted({
        (ev.type, k, type(ev.meta[k]).__name__) for ev in events for k in ev.meta
        if k in edm._EVENT_META_IDX_KEYS
        and not (ev.meta[k] is None or type(ev.meta[k]) is int)
    })
    assert bad == []


def test_pattern_anchor_idx_is_listed_and_emitted(geometry, m15_df):
    """Plan E E1: the pattern-realm key is on all three event types and shifted."""
    assert "pattern_anchor_idx" in edm._EVENT_META_IDX_KEYS
    events, _ = _mirrored(geometry, m15_df)
    for t in ("CTS_ESTABLISHED", "REVERSAL_WATCH_START", "REVERSAL_CANDIDATE"):
        evs = [ev for ev in events if ev.type == t]
        assert evs
        for ev in evs:
            assert type(ev.meta["pattern_anchor_idx"]) is int
            assert "anchor_idx" not in ev.meta


def test_every_zone_meta_index_key_is_shifted(geometry, m15_df):
    _, zones = _mirrored(geometry, m15_df)
    unlisted = sorted({
        k for z in zones for k in z.meta
        if _is_idx_key(k) and k not in edm._ZONE_META_IDX_KEYS
    })
    assert unlisted == [], f"zone meta index keys not in _ZONE_META_IDX_KEYS: {unlisted}"
    bad = sorted({
        (k, type(z.meta[k]).__name__) for z in zones for k in z.meta
        if k in edm._ZONE_META_IDX_KEYS
        and not (z.meta[k] is None or type(z.meta[k]) is int)
    })
    assert bad == []


def test_every_fib_meta_index_key_is_shifted(geometry, m15_df):
    fibs = _mirrored_attr(geometry, m15_df, "fib_states")
    unlisted = sorted({k for f in fibs for k in f.meta
                       if _is_idx_key(k) and k not in edm._FIB_META_IDX_KEYS})
    assert unlisted == [], f"fib meta index keys not in _FIB_META_IDX_KEYS: {unlisted}"
    bad = sorted({
        (k, type(f.meta[k]).__name__) for f in fibs for k in f.meta
        if k in edm._FIB_META_IDX_KEYS and not (f.meta[k] is None or type(f.meta[k]) is int)
    })
    assert bad == []


def test_every_wave_candle_meta_index_key_is_shifted(geometry, m15_df):
    wcs = _mirrored_attr(geometry, m15_df, "wave_candles")
    unlisted = sorted({k for w in wcs for k in w.meta
                       if _is_idx_key(k) and k not in edm._WAVE_CANDLE_META_IDX_KEYS})
    assert unlisted == [], unlisted


def test_pattern_anchor_idx_values_obey_the_pattern_bound(geometry, m15_df):
    """The VALUE, not just the key (a mutation writing the apply candle under the
    key survived the key-only pins). ARCHITECTURE "`ev.idx` convention" bound:
    `pattern_anchor_idx <= cts_anchor_idx <= confirmed_at <= pattern_anchor_idx + 5`
    (and `ev.idx == confirmed_at` since Plan E E4a), and
    `pattern_anchor_idx < confirmed_at` strictly — every breakout pattern spans at
    least two candles, so its first candle is never its apply candle. Both
    REVERSAL events are stamped AT the close-break candle, which is the key."""
    events, _ = _mirrored(geometry, m15_df)
    est = [ev for ev in events if ev.type == "CTS_ESTABLISHED"]
    assert est
    for ev in est:
        pa, conf = ev.meta["pattern_anchor_idx"], ev.meta["confirmed_at"]
        assert pa <= ef.cts_anchor_idx(ev) <= conf <= pa + 5
        assert ev.idx == conf
        assert pa < conf
    for t in ("REVERSAL_WATCH_START", "REVERSAL_CANDIDATE"):
        for ev in (e for e in events if e.type == t):
            assert ev.meta["pattern_anchor_idx"] == ev.idx


def test_mirrored_index_values_are_source_plus_slice_begin(geometry, m15_df):
    """Post-E·2 VALUE pin, per shift site: pair every mirrored element (each lens
    df holds only this sub, in emission order) with its slice-local source in the
    rendered result. A listed key holding an int is exactly `+ slice_begin`; the
    nested `bounds_steps[*].start_idx` / `activation_history[*].idx` likewise;
    every other key (attribution aside) is unchanged. The last assert names the
    family keys the fixture produces, so a site that stops shifting one fails."""
    res, lens_dfs = _render_both_lenses(geometry, m15_df)
    sb = int(res.meta["slice_begin"])
    assert sb > 0
    attribution = set(edm._sub_attribution(res, "x"))
    sites = (
        ("events", res.events, edm._EVENT_META_IDX_KEYS),
        ("kl_zones", res.kl_zones, edm._ZONE_META_IDX_KEYS),
        ("poi_zones", res.poi_zones, edm._ZONE_META_IDX_KEYS),
        ("fib_states", res.fib_states, edm._FIB_META_IDX_KEYS),
        ("wave_candles", res.wave_candles, edm._WAVE_CANDLE_META_IDX_KEYS),
    )
    shifted = Counter()
    for d in lens_dfs.values():
        for attr, src, keys in sites:
            mir = d.attrs.get(attr, [])
            assert len(mir) == len(src), attr
            for s, m in zip(src, mir):
                for k, v in s.meta.items():
                    if k in attribution:
                        continue
                    if k in keys:
                        assert v is None or type(v) is int, (attr, k, type(v))
                    if k in keys and v is not None:
                        assert m.meta[k] == v + sb, (attr, k, v, m.meta[k])
                        shifted[(attr, k)] += 1
                    elif k in NESTED_IDX_LISTS and v:
                        ik = NESTED_IDX_LISTS[k]
                        assert [x[ik] for x in m.meta[k]] == [x[ik] + sb for x in v], (attr, k)
                        shifted[(attr, k)] += 1
                    else:
                        assert m.meta[k] == v, (attr, k, v, m.meta[k])
    assert {
        ("events", "effective_idx"), ("events", "start_idx"), ("events", "confirm_idx"),
        ("events", "cts_idx"), ("events", "pullback_apply_idx"), ("events", "expires_idx"),
        ("kl_zones", "anchor_idx"), ("kl_zones", "bounds_steps"),
        ("poi_zones", "bos_idx"), ("poi_zones", "cts_idx"),
        ("fib_states", "activated_at"), ("fib_states", "locked_at"),
        ("wave_candles", "anchor_idx"),
    } <= set(shifted), sorted(shifted)


# --- static coverage (keys the fixture never produces) --------------------------

def _ms_emitted_event_meta_keys():
    """Every string key `market_structure` writes into an event meta: a dict
    literal passed as `meta=`, or a `meta[...]` / `meta2[...]` assignment."""
    tree = ast.parse(Path(market_structure.__file__).read_text(encoding="utf-8"))
    keys = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            for kw in n.keywords:
                if kw.arg == "meta" and isinstance(kw.value, ast.Dict):
                    keys |= {k.value for k in kw.value.keys if isinstance(k, ast.Constant)}
        if isinstance(n, ast.Assign):
            for tg in n.targets:
                if (isinstance(tg, ast.Subscript) and isinstance(tg.value, ast.Name)
                        and tg.value.id in ("meta", "meta2") and isinstance(tg.slice, ast.Constant)):
                    keys.add(tg.slice.value)
    return keys


def test_every_ms_emitted_event_meta_index_key_is_shifted():
    """Static: covers keys the fixture never produces (e.g. RANGE_STARTED
    `proximity_apply_idx`, emitted only on the proximity-created-range path; an
    int `pb_start`)."""
    keys = _ms_emitted_event_meta_keys()
    assert {"confirmed_at", "cts_anchor_idx", "bos_anchor_idx", "pattern_anchor_idx",
            "proximity_apply_idx", "pb_start"} <= keys
    unlisted = sorted(
        k for k in keys
        if (_is_idx_key(k) or k in EXTRA_EVENT_IDX_KEYS)
        and k not in edm._EVENT_META_IDX_KEYS
    )
    assert unlisted == [], unlisted


def _meta_kw_keys(module):
    """Every string key in a dict literal passed as `meta=` in `module` (the
    element constructors and the `replace(…, meta={**old, …})` updates). Only
    `meta=` literals: `poi_zones`' `inst.meta[...]` writes are ImbalanceInstance
    meta (the fill cache), which the mirror never copies."""
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    keys = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            for kw in n.keywords:
                if kw.arg == "meta" and isinstance(kw.value, ast.Dict):
                    keys |= {k.value for k in kw.value.keys if isinstance(k, ast.Constant)}
    return keys


@pytest.mark.parametrize("module, list_name, must_see", [
    (kl_zones_v1, "_ZONE_META_IDX_KEYS", {"anchor_idx", "expanded_last_idx", "pb_reconfirm_idx"}),
    (poi_zones, "_ZONE_META_IDX_KEYS", {"bos_idx", "cts_idx", "cts_established_idx"}),
    (fib_tracker, "_FIB_META_IDX_KEYS",
     {"activated_at", "reactivated_at", "locked_at", "deactivated_at", "cycle1_bos_idx"}),
    (wave_candles, "_WAVE_CANDLE_META_IDX_KEYS", {"anchor_idx"}),
], ids=["kl_zones_v1", "poi_zones", "fib_tracker", "wave_candles"])
def test_every_element_emitter_meta_index_key_is_shifted(module, list_name, must_see):
    """Static, Post-E·2: KL `expanded_last_idx` / `pb_reconfirm_idx` and fib
    `reactivated_at` / `cycle1_bos_idx` are never produced by the fixture."""
    keys = _meta_kw_keys(module)
    assert must_see <= keys, sorted(must_see - keys)
    listed = getattr(edm, list_name)
    unlisted = sorted(k for k in keys if _is_idx_key(k) and k not in listed)
    assert unlisted == [], f"{module.__name__} meta index keys not in {list_name}: {unlisted}"


def test_anchor_keys_are_int_and_idx_is_the_contract_index(geometry, m15_df):
    """Plan E E2a: CTS_ESTABLISHED `cts_anchor_idx` / BOS_CONFIRMED
    `bos_anchor_idx` and `confirmed_at` are Python ints, and `ev.idx` equals the
    index the contract names — after the mirror's shift too (all are shifted by
    the same offset): the moment `confirmed_at` on both types (Plan E E4a
    CTS_ESTABLISHED, E4b BOS_CONFIRMED)."""
    events, _ = _mirrored(geometry, m15_df)
    for t, key, idx_key in (("CTS_ESTABLISHED", "cts_anchor_idx", "confirmed_at"),
                            ("BOS_CONFIRMED", "bos_anchor_idx", "confirmed_at")):
        evs = [ev for ev in events if ev.type == t]
        assert evs, t
        for ev in evs:
            assert type(ev.meta[key]) is int and type(ev.meta["confirmed_at"]) is int, (t, ev.meta)
            assert ev.meta[idx_key] == ev.idx, (t, ev.idx, ev.meta)


# --- Post-E·2 landing review (2026-09-26): the review lenses' pins ---------------
# The value pin above is self-referential (it reads the same lists the mirror
# uses), covers META only, and only what the fixture emits; the static scans
# above read only `meta=` literals and suffixed names. The tests below close
# those gaps: the lists' own validity (index-like, never a known non-index, and
# every entry emitted by its element kind), `_shift_meta_indices`' contract,
# every shift site of the mirror on a synthetic result (incl. the dataclass
# fields the fixture-based pin cannot see), a broad emitter scan, and a
# value-type classification of every int meta value.

# Int-valued meta keys that are NOT candle indices (the fixture's + the §9.3
# census's non-indices; the attribution ints included). None may ever be
# listed; the H1 `triggered_by_event_idx` is in parent coords (never shifted).
NON_INDEX_INT_META_KEYS = frozenset({
    "structure_id", "cycle_id", "struct_direction",
    "parent_sid", "parent_cycle_id", "sub_id",
    "version", "scenario", "cross_start_cycle", "proximity_pips",
})
NEVER_LISTED = NON_INDEX_INT_META_KEYS | {"triggered_by_event_idx"}

_ALL_LISTS = ("_EVENT_META_IDX_KEYS", "_ZONE_META_IDX_KEYS",
              "_FIB_META_IDX_KEYS", "_WAVE_CANDLE_META_IDX_KEYS")


def test_shift_lists_hold_only_index_like_keys():
    """A non-index key added to a list (the conformance lens: `cycle_id` in the
    event list survived every other test — it would shift `cycle_id` on every
    M15 event) fails here."""
    for name in _ALL_LISTS:
        keys = getattr(edm, name)
        assert len(set(keys)) == len(keys), (name, "duplicate entry: shifted twice")
        bad = sorted(k for k in keys if not (_is_idx_key(k) or k in EXTRA_EVENT_IDX_KEYS)
                     or k in NEVER_LISTED)
        assert bad == [], (name, bad)


def test_every_listed_key_is_emitted_by_its_element_kind(geometry, m15_df):
    """Inverse guard: each list holds only keys its element kind carries (the
    `meta=` literal scans + the fixture). Kills a misfile like `pb_reconfirm_idx`
    in the EVENT list (the pre-Post-E·2 state) and dead entries (five deleted in
    the Post-E·2 landing review)."""
    res, _ = _render_both_lenses(geometry, m15_df)
    fixture = {
        "_EVENT_META_IDX_KEYS": {k for e in res.events for k in e.meta},
        "_ZONE_META_IDX_KEYS": {k for z in res.kl_zones + res.poi_zones for k in z.meta},
        "_FIB_META_IDX_KEYS": {k for f in res.fib_states for k in f.meta},
        "_WAVE_CANDLE_META_IDX_KEYS": {k for w in res.wave_candles for k in w.meta},
    }
    static = {
        "_EVENT_META_IDX_KEYS": _ms_emitted_event_meta_keys(),
        "_ZONE_META_IDX_KEYS": _meta_kw_keys(kl_zones_v1) | _meta_kw_keys(poi_zones),
        "_FIB_META_IDX_KEYS": _meta_kw_keys(fib_tracker),
        "_WAVE_CANDLE_META_IDX_KEYS": _meta_kw_keys(wave_candles),
    }
    for name in _ALL_LISTS:
        stale = sorted(set(getattr(edm, name)) - static[name] - fixture[name])
        assert stale == [], (name, stale)


def test_shift_meta_indices_contract():
    """A Python int under a listed key (0 included) gets `+ offset`; None / float
    / str / unlisted keys pass through; the input dict is NOT mutated (the same
    result is mirrored into two lens dfs — an in-place shift double-shifts the
    second). numpy ints / bools are deliberately not pinned (conftest documents
    numpy as unshifted; the fixture guards assert neither occurs under a key)."""
    src = {"a_idx": 0, "b_idx": None, "c_idx": 3, "d_idx": 1.5, "e_idx": "x", "free_idx": 9}
    before = deepcopy(src)
    out = edm._shift_meta_indices(src, ("a_idx", "b_idx", "c_idx", "d_idx", "e_idx", "missing_idx"), 10)
    assert out == {"a_idx": 10, "b_idx": None, "c_idx": 13, "d_idx": 1.5, "e_idx": "x", "free_idx": 9}
    assert "missing_idx" not in out
    assert out is not src and src == before


# Minimal stand-ins: the mirror uses only attribute access, deepcopy and
# dataclasses.replace, so it never checks the concrete element classes.
@dataclass
class _Zone:
    meta: dict


@dataclass
class _Poi:
    ic_idx: int
    meta: dict


@dataclass
class _Fib:
    bos_idx: int
    cts_idx: int
    start_idx: Optional[int]
    end_idx: Optional[int]
    meta: dict
    cts_history: tuple = ()


@dataclass
class _Wave:
    first_wave_candle_idx: Optional[int]
    last_wave_candle_idx: Optional[int]
    meta: dict


@dataclass
class _Wvmi:
    fb_idx: Optional[int]
    lb_idx: Optional[int]
    fp_idx: Optional[int]
    lp_idx: Optional[int]
    meta: dict


_V = 7      # every slice-local index value
_SB = 40    # slice_begin


def _all(keys) -> dict:
    return {k: _V for k in keys}


def _synthetic_result() -> LowerTFResult:
    return LowerTFResult(
        trigger=None,
        df=pd.DataFrame(),                       # no structure-column rows to paint
        events=[StructureEvent(idx=_V, category="STRUCTURE", type="CTS_ESTABLISHED",
                               meta={**_all(edm._EVENT_META_IDX_KEYS), "version": _V})],
        kl_zones=[_Zone(meta={**_all(edm._ZONE_META_IDX_KEYS), "version": _V,
                              "bounds_steps": [{"start_idx": _V, "top": 1.0}],
                              "activation_history": [{"idx": _V, "active": True}]})],
        poi_zones=[_Poi(ic_idx=_V, meta={**_all(edm._ZONE_META_IDX_KEYS), "version": _V,
                                         "activation_history": [{"idx": _V, "active": True}]})],
        fib_states=[_Fib(bos_idx=_V, cts_idx=_V + 1, start_idx=_V + 2, end_idx=_V + 3,
                         meta={**_all(edm._FIB_META_IDX_KEYS), "version": _V},
                         cts_history=((_V, 0.5), (_V + 1, 0.6))),
                    _Fib(bos_idx=_V, cts_idx=_V, start_idx=None, end_idx=None, meta={})],
        wave_candles=[_Wave(first_wave_candle_idx=_V, last_wave_candle_idx=_V + 1,
                            meta={**_all(edm._WAVE_CANDLE_META_IDX_KEYS), "version": _V}),
                      _Wave(first_wave_candle_idx=None, last_wave_candle_idx=None, meta={})],
        wvmi_records=[_Wvmi(fb_idx=_V, lb_idx=_V + 1, fp_idx=None, lp_idx=_V + 2,
                            meta={"triggered_by_event_idx": _V})],   # parent (H1) coords
        prev_bos_lines=[{"start_idx": _V, "end_idx": _V + 1, "price": 1.0}, "opaque"],
        status="finalized",
        meta={"slice_begin": _SB, "sub_id": 3, "timeframe": "M15", "parent_tf": "H1"},
    )


def _check_mirrored(d: pd.DataFrame) -> None:
    s = _V + _SB
    (ev,) = d.attrs["events"]
    assert ev.idx == s
    assert all(ev.meta[k] == s for k in edm._EVENT_META_IDX_KEYS)
    assert ev.meta["version"] == _V
    (kl,) = d.attrs["kl_zones"]
    assert all(kl.meta[k] == s for k in edm._ZONE_META_IDX_KEYS) and kl.meta["version"] == _V
    assert kl.meta["bounds_steps"] == [{"start_idx": s, "top": 1.0}]
    assert kl.meta["activation_history"] == [{"idx": s, "active": True}]
    (poi,) = d.attrs["poi_zones"]
    assert poi.ic_idx == s
    assert all(poi.meta[k] == s for k in edm._ZONE_META_IDX_KEYS) and poi.meta["version"] == _V
    assert poi.meta["activation_history"] == [{"idx": s, "active": True}]
    f0, f1 = d.attrs["fib_states"]
    assert (f0.bos_idx, f0.cts_idx, f0.start_idx, f0.end_idx) == (s, s + 1, s + 2, s + 3)
    assert f0.cts_history == ((s, 0.5), (s + 1, 0.6))
    assert all(f0.meta[k] == s for k in edm._FIB_META_IDX_KEYS) and f0.meta["version"] == _V
    assert (f1.start_idx, f1.end_idx, f1.cts_history) == (None, None, ())
    w0, w1 = d.attrs["wave_candles"]
    assert (w0.first_wave_candle_idx, w0.last_wave_candle_idx) == (s, s + 1)
    assert all(w0.meta[k] == s for k in edm._WAVE_CANDLE_META_IDX_KEYS) and w0.meta["version"] == _V
    assert (w1.first_wave_candle_idx, w1.last_wave_candle_idx) == (None, None)
    ln, opaque = d.attrs["prev_bos_lines"]
    assert (ln["start_idx"], ln["end_idx"], ln["price"]) == (s, s + 1, 1.0)
    assert ln["meta"]["sub_id"] == 3 and opaque == "opaque"


def test_mirror_shifts_every_site_exactly_once():
    """Every shift site of `mirror_lower_tf_result_to_entity_df` on a synthetic
    result with one element of each kind: the event idx, POI `ic_idx`, fib
    `bos_idx` / `cts_idx` / `start_idx` / `end_idx` / `cts_history`, wave-candle
    `first_/last_wave_candle_idx`, WVMI `fb/lb/fp/lp_idx`, `prev_bos_lines`
    `start_idx` / `end_idx`, each list at its own site, the nested lists, an
    unlisted int key untouched — mirrored into TWO lens dfs (each shifted once;
    the source untouched). The mutation lens found the fib fields (the M15 fib
    CSV's `bos_idx` / `cts_idx` / `end_idx` columns + the fib drawing), the
    wave-candle fields and `prev_bos_lines` (M15 chart positions) unpinned."""
    res = _synthetic_result()
    src = deepcopy(res)
    dfs = [pd.DataFrame(index=range(100)) for _ in range(2)]
    for i, d in enumerate(dfs):
        edm.mirror_lower_tf_result_to_entity_df(d, res, structure_path_id=f"p{i}")
    for d in dfs:
        _check_mirrored(d)
        (w,) = d.attrs["wvmi"]
        assert (w.fb_idx, w.lb_idx, w.fp_idx, w.lp_idx) == (_V + _SB, _V + 1 + _SB, None, _V + 2 + _SB)
        assert w.meta["triggered_by_event_idx"] == _V          # parent coords: never shifted
    assert res.events[0].idx == src.events[0].idx and res.events[0].meta == src.events[0].meta
    for a in ("kl_zones", "poi_zones", "fib_states", "wave_candles"):
        assert [x.meta for x in getattr(res, a)] == [x.meta for x in getattr(src, a)], a
    assert res.fib_states[0].bos_idx == _V and res.fib_states[0].cts_history == src.fib_states[0].cts_history
    assert res.prev_bos_lines == src.prev_bos_lines


def test_persist_facade_wvmi_shifts_once_and_keeps_parent_coords():
    res = _synthetic_result()
    d = pd.DataFrame(index=range(100))
    edm.persist_facade_wvmi_to_entity_df(d, res, structure_path_id="p")
    (w,) = d.attrs["wvmi"]
    assert (w.fb_idx, w.lb_idx, w.fp_idx, w.lp_idx) == (_V + _SB, _V + 1 + _SB, None, _V + 2 + _SB)
    assert w.meta["triggered_by_event_idx"] == _V and res.wvmi_records[0].fb_idx == _V


def _all_index_like_keys(module) -> set:
    """Every `*_idx` / `*_at` string `module` uses as (a) a key of ANY dict
    literal (incl. `**{...}` splats), (b) a constant subscript-assignment target
    on ANY expression (`self.events[-1].meta["k"] = …`), (c) `.setdefault("k", …)`,
    (d) a keyword of `.update(k=…)` / `dict(k=…)`. Broader than `_meta_kw_keys`."""
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    keys = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Dict):
            keys |= {k.value for k in n.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        elif isinstance(n, (ast.Assign, ast.AugAssign)):
            for tg in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                if isinstance(tg, ast.Subscript) and isinstance(tg.slice, ast.Constant):
                    keys.add(tg.slice.value)
        elif isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute) and f.attr == "setdefault" and n.args \
                    and isinstance(n.args[0], ast.Constant):
                keys.add(n.args[0].value)
            if (isinstance(f, ast.Attribute) and f.attr == "update") or \
                    (isinstance(f, ast.Name) and f.id == "dict"):
                keys |= {kw.arg for kw in n.keywords if kw.arg}
    return {k for k in keys if isinstance(k, str) and _is_idx_key(k)}


# Index-suffixed dict keys in the emitter modules that provably never reach a
# mirrored element's meta (measured at 574de2a). A new entry needs a reason.
_NOT_MIRRORED = {
    "market_structure": {
        "range_start_idx", "range_confirm_idx",   # st.jump_seed_state (rewind seed)
        "bos_idx",                                # st.cycle0_data cache
        "last_breakout_pat_apply_idx",            # a df column (out[...] = …)
    },
    "poi_zones": {"armed_idx", "confirmed_fill_idx"},  # ImbalanceInstance.meta (fill cache)
    "fib_tracker": {"bos_idx", "cts_idx", "fill_horizon_idx"},  # _cross_cycle_data cache
    "kl_zones_v1": {"start_idx"},                 # bounds_steps items (the nested loop)
    "wave_candles": set(),
}


@pytest.mark.parametrize("module, list_name", [
    (market_structure, "_EVENT_META_IDX_KEYS"),
    (kl_zones_v1, "_ZONE_META_IDX_KEYS"),
    (poi_zones, "_ZONE_META_IDX_KEYS"),
    (fib_tracker, "_FIB_META_IDX_KEYS"),
    (wave_candles, "_WAVE_CANDLE_META_IDX_KEYS"),
], ids=["market_structure", "kl_zones_v1", "poi_zones", "fib_tracker", "wave_candles"])
def test_emitter_modules_have_no_unlisted_index_keys(module, list_name):
    """Closes the literal scans' blind spots (the mutation lens: a new index key
    via a `**{...}` splat on the fib reactivation / KL expansion / MS proximity-
    range paths, or `self.events[-1].meta["k"] = …`, survived — all live sub
    paths the fixture does not run)."""
    short = module.__name__.rsplit(".", 1)[-1]
    listed = set(getattr(edm, list_name))
    unlisted = sorted(_all_index_like_keys(module) - listed - _NOT_MIRRORED[short])
    assert unlisted == [], f"{short}: index keys neither in {list_name} nor _NOT_MIRRORED: {unlisted}"


def test_every_int_meta_value_is_listed_or_known_non_index(geometry, m15_df):
    """The name guards only see `*_idx` / `*_at` (+ `pb_start`): a new UNSUFFIXED
    index key (the mutation lens: `"watch_from": int(i)` on REVERSAL_WATCH_START)
    survived. Classify by VALUE TYPE: every Python-int meta value (top level and
    inside the nested lists) on a fixture element is under a shift-listed key, the
    nested item key, or a known non-index key."""
    res, _ = _render_both_lenses(geometry, m15_df)
    sites = (
        ("events", res.events, edm._EVENT_META_IDX_KEYS),
        ("kl_zones", res.kl_zones, edm._ZONE_META_IDX_KEYS),
        ("poi_zones", res.poi_zones, edm._ZONE_META_IDX_KEYS),
        ("fib_states", res.fib_states, edm._FIB_META_IDX_KEYS),
        ("wave_candles", res.wave_candles, edm._WAVE_CANDLE_META_IDX_KEYS),
    )
    bad = set()
    for attr, elems, listed in sites:
        for e in elems:
            for k, v in e.meta.items():
                if type(v) is int and k not in listed and k not in NON_INDEX_INT_META_KEYS:
                    bad.add((attr, getattr(e, "type", ""), k))
                if k in NESTED_IDX_LISTS and v:
                    for item in v:
                        for ik, iv in item.items():
                            if type(iv) is int and ik != NESTED_IDX_LISTS[k]:
                                bad.add((attr, k, ik))
    assert bad == set(), sorted(bad)
