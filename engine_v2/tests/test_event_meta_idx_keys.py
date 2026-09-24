"""Guard: every candle-index meta key on a mirrored sub event / zone is either
shifted slice-local -> entity-absolute by the mirror, or explicitly known to be
slice-local (Plan E §4.3, IN R5).

`entity_df_mutation._shift_meta_indices` translates only the keys listed in
`_EVENT_META_IDX_KEYS` / `_ZONE_META_IDX_KEYS`, and only values that are a
Python `int`. A new or renamed index key that is missing from the list is
exported slice-local on every M15 row without any error — so this test fails
the moment an emitter starts writing an unlisted `*_idx` / `*_at` key.

The allow-lists below are the slice-local keys that exist TODAY (the export
hygiene item in the zones-timing audit memory; none is read cross-frame). A key
moves from an allow-list into the shift list, never the other way round.

Fixture: the production-shaped sub from `test_render_sub_projection.py` (the
reversing M15 series; it emits CTS_ESTABLISHED, BOS_CONFIRMED, CTS_CONFIRMED,
REVERSAL_WATCH_START, REVERSAL_CANDIDATE, RANGE_* and a KL / POI zone), mirrored
into both lens dfs by `render_sub_projection`.
"""
from __future__ import annotations

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

# Event-meta index keys that stay slice-local on a mirrored sub event.
KNOWN_SLICE_LOCAL = frozenset({
    "confirm_idx",          # RANGE_STARTED
    "cts_idx",              # RANGE_STARTED (pairs with cts_price)
    "pullback_apply_idx",   # RANGE_STARTED
    "start_idx",            # RANGE_STARTED
    "expires_idx",          # REVERSAL_WATCH_START / REVERSAL_CANDIDATE
    "effective_idx",        # STATE_CHANGED
    "pb_start",             # BOS_CONFIRMED
})

# Index-valued event-meta keys that carry neither the `_idx` nor the `_at` suffix.
EXTRA_EVENT_IDX_KEYS = frozenset({"pb_start"})

# Zone-meta index keys that stay slice-local on a mirrored sub zone.
KNOWN_SLICE_LOCAL_ZONE = frozenset({
    "bos_idx",              # KL zone
    "cts_idx",              # KL zone
    "source_event_idx",     # KL zone (write-only; Plan E E4b-pre deletes it)
})


def _is_idx_key(key: str) -> bool:
    return key.endswith("_idx") or key.endswith("_at")


def _mirrored(geometry, m15_df):
    pool, sub = geometry
    _two_lens_records(pool, sub, m15_df)
    _set_lifecycle(sub, _START, None, None)
    lens_dfs = _lens_dfs(m15_df)
    _render(sub, m15_df, lens_dfs)
    events = [ev for d in lens_dfs.values() for ev in d.attrs.get("events", [])]
    zones = [z for d in lens_dfs.values()
             for z in d.attrs.get("kl_zones", []) + d.attrs.get("poi_zones", [])]
    return events, zones


def test_fixture_emits_the_event_types_that_carry_index_keys(geometry, m15_df):
    """Precondition: the guard below is only as good as the event mix it sees."""
    events, zones = _mirrored(geometry, m15_df)
    types = {ev.type for ev in events}
    assert {"CTS_ESTABLISHED", "BOS_CONFIRMED", "CTS_CONFIRMED",
            "REVERSAL_WATCH_START", "REVERSAL_CANDIDATE", "RANGE_STARTED"} <= types
    assert zones, "the fixture must mirror at least one zone"


def test_every_event_meta_index_key_is_shifted_or_known_slice_local(geometry, m15_df):
    events, _ = _mirrored(geometry, m15_df)
    unlisted = sorted({
        (ev.type, k) for ev in events for k in ev.meta
        if (_is_idx_key(k) or k in EXTRA_EVENT_IDX_KEYS)
        and k not in edm._EVENT_META_IDX_KEYS and k not in KNOWN_SLICE_LOCAL
    })
    assert unlisted == [], (
        f"event meta index keys neither in _EVENT_META_IDX_KEYS nor KNOWN_SLICE_LOCAL: {unlisted}"
    )


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


def test_every_zone_meta_index_key_is_shifted_or_known_slice_local(geometry, m15_df):
    _, zones = _mirrored(geometry, m15_df)
    unlisted = sorted({
        k for z in zones for k in z.meta
        if _is_idx_key(k) and k not in edm._ZONE_META_IDX_KEYS
        and k not in KNOWN_SLICE_LOCAL_ZONE
    })
    assert unlisted == [], (
        f"zone meta index keys neither in _ZONE_META_IDX_KEYS nor KNOWN_SLICE_LOCAL_ZONE: {unlisted}"
    )
    bad = sorted({
        (k, type(z.meta[k]).__name__) for z in zones for k in z.meta
        if k in edm._ZONE_META_IDX_KEYS
        and not (z.meta[k] is None or type(z.meta[k]) is int)
    })
    assert bad == []


def test_pattern_anchor_idx_values_obey_the_pattern_bound(geometry, m15_df):
    """The VALUE, not just the key (a mutation writing the apply candle under the
    key survived the key-only pins). ARCHITECTURE "`ev.idx` convention" bound:
    `pattern_anchor_idx <= ev.idx <= confirmed_at <= pattern_anchor_idx + 5`, and
    `pattern_anchor_idx < confirmed_at` strictly — every breakout pattern spans at
    least two candles, so its first candle is never its apply candle. Both
    REVERSAL events are stamped AT the close-break candle, which is the key."""
    events, _ = _mirrored(geometry, m15_df)
    est = [ev for ev in events if ev.type == "CTS_ESTABLISHED"]
    assert est
    for ev in est:
        pa, conf = ev.meta["pattern_anchor_idx"], ev.meta["confirmed_at"]
        assert pa <= ev.idx <= conf <= pa + 5
        assert pa < conf
    for t in ("REVERSAL_WATCH_START", "REVERSAL_CANDIDATE"):
        for ev in (e for e in events if e.type == t):
            assert ev.meta["pattern_anchor_idx"] == ev.idx
