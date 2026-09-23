"""Stage 2 equivalence tests (PART4_REFACTOR_SPEC.md §17.8–§17.9; Plan C §5.4).

De-risks the pool's dedup-reuse guarantee: running a sub structure to its
NATURAL end and projecting/clipping it to a window is byte-equivalent to a
window-bounded build. Two claims:

  1. MS `end_idx`-causality — events knowable-at <= B are identical whether
     `compute_bounded_structure` ran to B or to its natural reversal R > B.
  2. `project_to_window(natural-end run, [floor, cap])` == a window-bounded
     `compute_bounded_structure(end_idx=cap)` + downstream, for events + KL zones
     + POI zones.

Plan C §5.4: `entity_df_mutation.build_or_get_geometry` (the renamed
`_build_or_get_sub_geometry`) is the ONE surviving geometry builder — run cap =
the data edge (`len(m15) - 1`), pool entry created only AFTER a successful MS
run, returns `(sub, created)` with `sub.geometry = (bounded, slice_begin)`.
`pooled_structure_build.build_structure_geometry` (never had a live caller) is
deleted. With `start=0` the slice is the whole df, so `slice_begin == 0` and the
Claim-2 equivalence assertions hold unchanged.

Uses the `_make_reversing_data` fixture (reverses mid-series) so there is a real
tail past the window to prove causality against.
"""
from __future__ import annotations

import copy

import pytest

import engine_v2.multitf.pooled_structure_build as pooled_structure_build
from engine_v2.structure.structure_engine import compute_bounded_structure
from engine_v2.pipeline.orchestrator import _run_downstream_pipeline
from engine_v2.multitf.entity_df_mutation import build_or_get_geometry
from engine_v2.multitf.pooled_structure_build import (
    clip_events_to_window,
    project_to_window,
)
from engine_v2.multitf.sub_structure_pool import (
    StructureKey,
    SubStructurePool,
    knowable_at_idx,
)
from engine_v2.tests.test_bounded_structure import (
    _make_reversing_data,
    _prepare_df,
)

_SD = 1
_START = 0
_PARENT_PATH = "H1.main"


# --- signatures for order-insensitive comparison ------------------------------

def _ev_sig(events):
    return sorted(
        (ev.type, int(ev.idx), int(ev.meta.get("confirmed_at", ev.idx)))
        for ev in events
    )


def _kl_sig(zones):
    return sorted(
        (
            z.source_kind, z.side, round(float(z.top), 5), round(float(z.bottom), 5),
            z.meta.get("structure_id"), z.meta.get("cycle_id"),
            z.meta.get("confirmed_idx"), z.meta.get("end_idx"), z.meta.get("status"),
        )
        for z in zones
    )


def _poi_sig(zones):
    return sorted(
        (
            round(float(z.top), 5), round(float(z.bottom), 5), int(z.ic_idx),
            z.meta.get("structure_id"), z.meta.get("cycle_id"),
            z.meta.get("end_idx"), z.meta.get("status"),
            # Activation + its cycle term (Plan D): the dedup-equivalence must
            # cover when a POI goes live, not only its geometry and end.
            z.meta.get("confirmed_idx"), z.meta.get("cts_established_idx"),
            tuple(
                (e["idx"], e["active"], e.get("reason"), tuple(e.get("versions", ())))
                for e in z.meta.get("activation_history") or []
            ),
            tuple(z.meta.get("current_versions") or ()),
        )
        for z in zones
    )


@pytest.fixture(scope="module")
def reversing_df():
    return _prepare_df(_make_reversing_data())


@pytest.fixture(scope="module")
def natural_reversal_idx(reversing_df):
    r = compute_bounded_structure(reversing_df, _START, _SD).reversal_idx
    assert r is not None and r > 10, "fixture must reverse mid-series"
    return int(r)


def _build_geometry(df):
    """The surviving builder (§5.4), on a fresh pool. Returns `(sub, created)`.

    Signature pinned by the API contract:
    `build_or_get_geometry(pool, m15_df, *, parent_path, sd, start_abs,
    bos0_inner, timeframe="M15") -> Optional[Tuple[PooledStructure, bool]]`.
    """
    pool = SubStructurePool()
    built = build_or_get_geometry(
        pool, df,
        parent_path=_PARENT_PATH, sd=_SD, start_abs=_START,
        bos0_inner=None, timeframe="M15",
    )
    assert built is not None, "geometry build must succeed on the reversing fixture"
    return pool, built


# --- Claim 1: MS end_idx-causality -------------------------------------------

def test_events_knowable_at_are_causal_in_end_idx(reversing_df, natural_reversal_idx):
    """Events knowable-at <= B are identical whether the run ended at B or at its
    natural reversal R > B. This is the core render-path claim (§17.9)."""
    R = natural_reversal_idx
    cap = R - 6                       # safely inside the pre-reversal region
    full = compute_bounded_structure(reversing_df, _START, _SD)          # to R
    capped = compute_bounded_structure(reversing_df, _START, _SD, end_idx=cap)

    full_clipped = clip_events_to_window(full.events, cap)
    assert _ev_sig(full_clipped) == _ev_sig(capped.events)
    # And the tail really exists (otherwise the test proves nothing).
    assert len(full.events) > len(capped.events)


def test_capped_run_emits_nothing_past_cap(reversing_df, natural_reversal_idx):
    cap = natural_reversal_idx - 6
    capped = compute_bounded_structure(reversing_df, _START, _SD, end_idx=cap)
    for ev in capped.events:
        assert knowable_at_idx(ev.type, ev.idx, ev.meta.get("confirmed_at")) <= cap
        # Plan A: a bounded run stamps NO event past its bound — `ev.idx` included
        # (for BOS_CONFIRMED the extreme, which precedes `confirmed_at`). MS asserts
        # this itself post-run; stated here so the guarantee is explicit.
        assert int(ev.idx) <= cap


def _cols_equal(a, b):
    """Elementwise equality over two column slices, treating NaN == NaN."""
    import pandas as pd
    fa, fb = list(a), list(b)
    if len(fa) != len(fb):
        return False
    return all(
        x == y or (pd.isna(x) and pd.isna(y)) for x, y in zip(fa, fb)
    )


def test_structure_columns_causal_over_window(reversing_df, natural_reversal_idx):
    """df 'current-truth' columns match over the whole window [0, cap] between a
    run-to-R and a run-to-cap. On this fixture the match is exact even at the
    boundary; in general the last few candles before `cap` could differ due to
    in-flight confirmations — the knowable-at event clip (§17.9) handles that on
    the render path, and the df-column mirror is itself window-clipped."""
    R = natural_reversal_idx
    cap = R - 6
    full = compute_bounded_structure(reversing_df, _START, _SD)
    capped = compute_bounded_structure(reversing_df, _START, _SD, end_idx=cap)
    # Real MS output column names (no bare `cycle_id` / `cts_phase`).
    cols = ["structure_id", "cts_cycle_id", "market_state", "range_lo", "range_hi"]
    for col in cols:
        assert _cols_equal(
            full.df[col].iloc[0:cap + 1], capped.df[col].iloc[0:cap + 1],
        ), f"column {col} diverged within [0, cap]"


# --- Claim 2: project_to_window == window-bounded build ----------------------

def _bounded_build(df, cap, floor, cap_reason):
    """A Phase-1-style window-bounded build (MS bounded to cap + downstream with
    floor/cap) — the reference the pool must match."""
    ref = compute_bounded_structure(df, _START, _SD, end_idx=cap)
    ref.df.attrs["imbalances"] = df.attrs.get("imbalances", [])
    down = _run_downstream_pipeline(
        ref.df, ref.events, _SD,
        source_kinds=["BOS"], fib_mode="cross_cycle",
        skip_wvmi=True, timeframe="M15",
        lifecycle_floor=floor, lifecycle_cap=cap, cap_reason=cap_reason,
    )
    down["events"] = ref.events
    return down


def test_build_or_get_geometry_creates_pool_entry_with_natural_end(
    reversing_df, natural_reversal_idx,
):
    """§5.4: the builder is the single owner of get_or_create + MS run +
    `natural_reversal_idx`. On a miss it returns `(sub, created=True)` with
    `sub.geometry = (bounded, slice_begin)`; `start=0` → `slice_begin=0` (the
    slice is the whole df) so the entity-absolute natural reversal equals the
    slice-local one (R + 0). Run cap = the data edge, so the reversal is found."""
    R = natural_reversal_idx
    pool, (sub, created) = _build_geometry(reversing_df)
    assert created is True
    assert sub.key == StructureKey(_PARENT_PATH, "M15", _SD, _START)
    assert pool.get(sub.key) is sub
    assert pool.get_by_id(sub.sub_id) is sub

    bounded, slice_begin = sub.geometry
    assert slice_begin == 0
    assert bounded.reversal_idx == R                       # slice-local == absolute here
    assert sub.natural_reversal_idx == R + slice_begin     # entity-absolute
    assert sub.bos0_inner is None                          # the probe's inner, none given

    # A second call for the same key is a pool HIT: same object, created=False,
    # no new sub_id consumed.
    again = build_or_get_geometry(
        pool, reversing_df,
        parent_path=_PARENT_PATH, sd=_SD, start_abs=_START,
        bos0_inner=None, timeframe="M15",
    )
    assert again is not None
    sub2, created2 = again
    assert sub2 is sub and created2 is False
    assert len(pool.all()) == 1


def test_project_to_window_matches_bounded_build(reversing_df, natural_reversal_idx):
    """The dedup-reuse guarantee: a natural-end run projected to [floor, cap]
    reproduces the window-bounded build's events + KL zones + POI zones."""
    R = natural_reversal_idx
    cap = R - 6
    floor = _START
    cap_reason = "parent_end"

    # Pooled: build to natural end ONCE (run cap = data edge), then project.
    _pool, (sub, _created) = _build_geometry(reversing_df)
    bounded, slice_begin = sub.geometry
    assert slice_begin == 0                    # start 0 → whole-df slice
    assert bounded.reversal_idx == R           # natural end captured
    proj = project_to_window(
        bounded, floor=floor, cap=cap, cap_reason=cap_reason, direction=_SD,
    )

    # Reference: window-bounded build.
    ref = _bounded_build(reversing_df, cap, floor, cap_reason)

    assert _ev_sig(proj["events"]) == _ev_sig(ref["events"])
    assert _kl_sig(proj["kl_zones"]) == _kl_sig(ref["kl_zones"])
    assert _poi_sig(proj["poi_zones"]) == _poi_sig(ref["poi_zones"])


def test_project_open_window_keeps_full_run(reversing_df, natural_reversal_idx):
    """cap=None (open lifecycle to the edge) keeps every event — no clip."""
    _pool, (sub, _created) = _build_geometry(reversing_df)
    bounded, _slice_begin = sub.geometry
    proj = project_to_window(
        bounded, floor=_START, cap=None, cap_reason=None, direction=_SD,
    )
    assert _ev_sig(proj["events"]) == _ev_sig(bounded.events)


# --- Shared-geometry safety (Plan C §6.1 / §12) --------------------------------

def _assert_clip_returns_deep_copies(source_events, clipped):
    assert len(clipped) > 0
    for c in clipped:
        assert all(c is not s for s in source_events), \
            "clip_events_to_window handed out a SHARED event object"
    # Mutating a returned event's meta must not reach the source geometry.
    before = copy.deepcopy([dict(s.meta) for s in source_events])
    clipped[0].meta["__plan_c_probe__"] = "mutated"
    clipped[0].meta["structure_id"] = 999
    after = [dict(s.meta) for s in source_events]
    assert after == before, "mutating a clipped event's meta mutated the source event"


def test_clip_events_to_window_returns_deep_copies(reversing_df, natural_reversal_idx):
    """§6.1: `clip_events_to_window` must DEEPCOPY before returning — geometry
    objects are shared across every lens / record that reads them, and the
    mirror stamps attribution into `ev.meta`."""
    full = compute_bounded_structure(reversing_df, _START, _SD)
    cap = natural_reversal_idx - 6
    clipped = clip_events_to_window(full.events, cap)
    _assert_clip_returns_deep_copies(full.events, clipped)


def test_clip_events_open_window_returns_deep_copies(reversing_df):
    """The `cap=None` branch (no clip) must deep-copy too — it hands out the
    whole shared stream otherwise."""
    full = compute_bounded_structure(reversing_df, _START, _SD)
    clipped = clip_events_to_window(full.events, None)
    assert len(clipped) == len(full.events)
    _assert_clip_returns_deep_copies(full.events, clipped)


def test_build_structure_geometry_is_deleted():
    """§5.4 / §7: the dead twin never had a live caller; the surviving builder
    is `entity_df_mutation.build_or_get_geometry`."""
    assert not hasattr(pooled_structure_build, "build_structure_geometry")
