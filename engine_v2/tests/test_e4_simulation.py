"""E4 simulation (PLAN_E §6.4): the downstream pipeline must not depend on what
`ev.idx` holds on CTS_ESTABLISHED / BOS_CONFIRMED.

`_run_downstream_pipeline` runs twice on lagging fixtures — once on the real MS
events (`ev.idx` = the anchor) and once on clones with `idx := confirmed_at`
(the E4 shape) — and every output must be identical except the declared raw
readers: `sorted_events` (it holds the events themselves), the `fib_tracker`
object (its repr is an address), and — on a BOS flip — the write-only KL meta
`source_event_idx` (PLAN_E §6.3; Plan E E4b-pre deletes it).

Fixtures: `_make_second_cts_moment_after_extreme_data` (EST (9, 10); BOS (0, 2),
(7, 10)) and `_make_multicycle_data` (4 lagging BOS, no lagging EST — PLAN_E §6.4
said 3; measured 2026-09-24), in both fib modes. E2b made the EST half pass; E2c the BOS half. Plan E E4 turns this into a
regression pin of the real emitter.
"""
from __future__ import annotations

import contextlib
import copy
import io

import pytest

from engine_v2.pipeline.orchestrator import _run_downstream_pipeline
from engine_v2.structure.structure_engine import compute_bounded_structure
from engine_v2.tests.test_unified_probe import (
    _make_multicycle_data,
    _make_second_cts_moment_after_extreme_data,
    _prepare_df,
)

pytestmark = pytest.mark.illegal_event_contract  # the clones are E4-shaped

_FLIP_TYPES = {
    "est": ("CTS_ESTABLISHED",),
    "bos": ("BOS_CONFIRMED",),
    "both": ("CTS_ESTABLISHED", "BOS_CONFIRMED"),
}
_EXCLUDED = ("sorted_events", "fib_tracker")


def _lags(events, types):
    return [(e.type, e.idx, e.meta["confirmed_at"]) for e in events
            if e.type in types and e.idx != e.meta["confirmed_at"]]


def _strip_raw(out, flip):
    res = {k: repr(v) for k, v in out.items() if k not in _EXCLUDED}
    if "BOS_CONFIRMED" in _FLIP_TYPES[flip]:
        zones = copy.deepcopy(out["kl_zones"])
        for z in zones:
            z.meta.pop("source_event_idx", None)
        res["kl_zones"] = repr(zones)
    return res


def _run(df, events, mode):
    with contextlib.redirect_stdout(io.StringIO()):
        return _run_downstream_pipeline(df, events, +1, fib_mode=mode, skip_wvmi=False)


_E2C = pytest.mark.xfail(strict=True, reason="BOS readers migrate in Plan E E2c")
_SECOND = _make_second_cts_moment_after_extreme_data
_MULTI = _make_multicycle_data


@pytest.mark.parametrize("mode", ["h1", "cross_cycle"])
@pytest.mark.parametrize("flip,maker", [
    ("est", _SECOND),                       # the only fixture with a lagging EST (9 → 10)
    pytest.param("bos", _SECOND, marks=_E2C),
    pytest.param("bos", _MULTI, marks=_E2C),  # 4 lagging BOS
    pytest.param("both", _SECOND, marks=_E2C),
    pytest.param("both", _MULTI, marks=_E2C),
])
def test_downstream_outputs_do_not_depend_on_the_flip(flip, maker, mode):
    res = compute_bounded_structure(_prepare_df(maker()), 0, +1)
    types = _FLIP_TYPES[flip]
    assert _lags(res.events, types), "the fixture must carry a lagging event of the flipped type"
    flipped = copy.deepcopy(res.events)
    for e in flipped:
        if e.type in types:
            e.idx = int(e.meta["confirmed_at"])
    a = _strip_raw(_run(res.df, res.events, mode), flip)
    b = _strip_raw(_run(res.df, flipped, mode), flip)
    assert a.keys() == b.keys()
    assert [k for k in a if a[k] != b[k]] == []
