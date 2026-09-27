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
other key unchanged.
"""
from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pytest

from engine_v2.structure import event_fields as ef
from engine_v2.structure import market_structure
from engine_v2.zones import fib_tracker, kl_zones_v1, poi_zones, wave_candles

from engine_v2.multitf import entity_df_mutation as edm
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
