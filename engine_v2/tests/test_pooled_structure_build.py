"""Stage 2 equivalence tests (PART4_REFACTOR_SPEC.md §17.5–§17.11).

De-risks the switchover BEFORE it happens: proves that running a sub structure
to its NATURAL end and projecting/clipping it to a window is byte-equivalent to
today's window-bounded build. Two claims:

  1. MS `end_idx`-causality — events knowable-at <= B are identical whether
     `compute_bounded_structure` ran to B or to its natural reversal R > B.
  2. `project_to_window(natural-end run, [floor, cap])` == a window-bounded
     `compute_bounded_structure(end_idx=cap)` + downstream, for events + KL zones
     + POI zones.

Uses the `_make_reversing_data` fixture (reverses mid-series) so there is a real
tail past the window to prove causality against. Dead code — no live caller yet.
"""
from __future__ import annotations

import pytest

from engine_v2.structure.structure_engine import compute_bounded_structure
from engine_v2.pipeline.orchestrator import _run_downstream_pipeline
from engine_v2.multitf.pooled_structure_build import (
    build_structure_geometry,
    clip_events_to_window,
    project_to_window,
)
from engine_v2.multitf.sub_structure_pool import knowable_at_idx
from engine_v2.tests.test_bounded_structure import (
    _make_reversing_data,
    _prepare_df,
)

_SD = 1
_START = 0


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


# --- Claim 1: MS end_idx-causality -------------------------------------------

def test_events_knowable_at_are_causal_in_end_idx(reversing_df, natural_reversal_idx):
    """Events knowable-at <= B are identical whether the run ended at B or at its
    natural reversal R > B. This is the core render-path claim (§17.11)."""
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
    in-flight confirmations — the knowable-at event clip (§17.11) handles that on
    the render path, and the Stage-3 df-column mirror is itself window-clipped."""
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


def test_project_to_window_matches_bounded_build(reversing_df, natural_reversal_idx):
    """The dedup-reuse guarantee: a natural-end run projected to [floor, cap]
    reproduces the window-bounded build's events + KL zones + POI zones."""
    R = natural_reversal_idx
    cap = R - 6
    floor = _START
    cap_reason = "lifecycle_end"

    # Pooled: build to natural end ONCE, then project to the window.
    geom = build_structure_geometry(
        reversing_df, starting_idx=_START, direction=_SD, run_cap=None,
    )
    assert geom.reversal_idx == R          # natural end captured
    proj = project_to_window(
        geom, floor=floor, cap=cap, cap_reason=cap_reason, direction=_SD,
    )

    # Reference: window-bounded build.
    ref = _bounded_build(reversing_df, cap, floor, cap_reason)

    assert _ev_sig(proj["events"]) == _ev_sig(ref["events"])
    assert _kl_sig(proj["kl_zones"]) == _kl_sig(ref["kl_zones"])
    assert _poi_sig(proj["poi_zones"]) == _poi_sig(ref["poi_zones"])


def test_project_open_window_keeps_full_run(reversing_df, natural_reversal_idx):
    """cap=None (open lifecycle to the edge) keeps every event — no clip."""
    geom = build_structure_geometry(
        reversing_df, starting_idx=_START, direction=_SD, run_cap=None,
    )
    proj = project_to_window(
        geom, floor=_START, cap=None, cap_reason=None, direction=_SD,
    )
    assert _ev_sig(proj["events"]) == _ev_sig(geom.events)
