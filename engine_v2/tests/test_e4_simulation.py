"""E4 regression pins (PLAN_E §6.4 → §8): the real emitter stamps each flipped
event at its MOMENT, and the downstream pipeline does not depend on what
`ev.idx` holds on CTS_ESTABLISHED / BOS_CONFIRMED.

1. The emitter: since Plan E E4a every CTS_ESTABLISHED of a real MS run carries
   `idx == meta["confirmed_at"]`, and the lagging one keeps its anchor in
   `meta["cts_anchor_idx"]` (9, known at 10).
2. The swap: `_run_downstream_pipeline` runs twice on lagging fixtures — once on
   the real MS events and once on clones whose swapped type holds the OTHER
   role in `ev.idx`: a CTS_ESTABLISHED back at its anchor (the pre-E4a shape), a
   BOS_CONFIRMED at its moment (the E4b shape, until E4b lands) — and every
   output must be identical except the declared raw readers: `sorted_events` (it
   holds the events themselves), the `fib_tracker` object (its repr is an
   address), and — on a BOS swap — the write-only KL meta `source_event_idx`
   (PLAN_E §6.3; Plan E E4b-pre deletes it).

Fixtures: `_make_second_cts_moment_after_extreme_data` (EST (9, 10); BOS (0, 2),
(7, 10)) and `_make_multicycle_data` (4 lagging BOS, no lagging EST — PLAN_E §6.4
said 3; measured 2026-09-24), in both fib modes. History: E2b made the EST half
of the (then forward) simulation pass, E2c the BOS half; E4a turned the EST half
into the pin above.
"""
from __future__ import annotations

import contextlib
import copy
import io

import pytest

from engine_v2.pipeline.orchestrator import _run_downstream_pipeline
from engine_v2.structure import event_fields as ef
from engine_v2.structure.structure_engine import compute_bounded_structure
from engine_v2.tests.test_unified_probe import (
    _make_multicycle_data,
    _make_second_cts_moment_after_extreme_data,
    _prepare_df,
)

_SWAP_TYPES = {
    "est": ("CTS_ESTABLISHED",),
    "bos": ("BOS_CONFIRMED",),
    "both": ("CTS_ESTABLISHED", "BOS_CONFIRMED"),
}
# The index the swapped clone holds: the role the contract does NOT name.
_OTHER_ROLE = {
    "CTS_ESTABLISHED": ef.cts_anchor_idx,              # pre-E4a shape
    "BOS_CONFIRMED": lambda e: e.meta["confirmed_at"],  # E4b shape
}
_EXCLUDED = ("sorted_events", "fib_tracker")


def _swappable(events, types):
    return [(e.type, e.idx, _OTHER_ROLE[e.type](e)) for e in events
            if e.type in types and e.idx != _OTHER_ROLE[e.type](e)]


def _strip_raw(out, swap):
    res = {k: repr(v) for k, v in out.items() if k not in _EXCLUDED}
    if "BOS_CONFIRMED" in _SWAP_TYPES[swap]:
        zones = copy.deepcopy(out["kl_zones"])
        for z in zones:
            z.meta.pop("source_event_idx", None)
        res["kl_zones"] = repr(zones)
    return res


def _run(df, events, mode):
    with contextlib.redirect_stdout(io.StringIO()):
        return _run_downstream_pipeline(df, events, +1, fib_mode=mode, skip_wvmi=False)


_SECOND = _make_second_cts_moment_after_extreme_data
_MULTI = _make_multicycle_data


@pytest.mark.parametrize("maker", [_SECOND, _MULTI])
def test_real_emitter_stamps_cts_established_at_its_moment(maker):
    """Plan E E4a: `ev.idx` of every CTS_ESTABLISHED IS its moment."""
    res = compute_bounded_structure(_prepare_df(maker()), 0, +1)
    ests = [e for e in res.events if e.type == "CTS_ESTABLISHED"]
    assert ests
    assert all(e.idx == e.meta["confirmed_at"] for e in ests)
    if maker is _SECOND:  # the lagging one: anchor 9, moment 10
        assert [(ef.cts_anchor_idx(e), e.idx) for e in ests if ef.cts_anchor_idx(e) != e.idx] == [(9, 10)]


@pytest.mark.skipif(not __debug__, reason="the emitter's check is an assert (stripped under -O)")
def test_emit_cts_established_asserts_idx_is_the_moment():
    """The emitter refuses an `idx` other than `meta["confirmed_at"]` (Plan E E4a)
    — before any event is built."""
    from engine_v2.structure.market_structure import MarketStructure
    ms = MarketStructure(_prepare_df(_SECOND()), 1)
    with pytest.raises(AssertionError, match="ev.idx is the moment"):
        ms._emit_cts_established(9, 1.0, cts_anchor_idx=9, meta={"confirmed_at": 10})
    assert ms.events == []
    ms._emit_cts_established(10, 1.0, cts_anchor_idx=9, meta={"confirmed_at": 10})
    ev = ms.events[-1]
    assert (ev.idx, ev.meta["cts_anchor_idx"], ev.meta["confirmed_at"]) == (10, 9, 10)
    assert ms.state.cts_established_moment_idx == 10


@pytest.mark.illegal_event_contract  # the clones hold the other role
@pytest.mark.parametrize("mode", ["h1", "cross_cycle"])
@pytest.mark.parametrize("swap,maker", [
    ("est", _SECOND),    # the only fixture with a lagging EST (9 → 10)
    ("bos", _SECOND),
    ("bos", _MULTI),     # 4 lagging BOS
    ("both", _SECOND),
    ("both", _MULTI),
])
def test_downstream_outputs_do_not_depend_on_the_idx_role(swap, maker, mode):
    res = compute_bounded_structure(_prepare_df(maker()), 0, +1)
    types = _SWAP_TYPES[swap]
    assert {t for t, _, _ in _swappable(res.events, types)} >= (
        set(types) if maker is _SECOND else {"BOS_CONFIRMED"}
    ), "the fixture must carry a lagging event of every swapped type"
    swapped = copy.deepcopy(res.events)
    for e in swapped:
        if e.type in types:
            e.idx = int(_OTHER_ROLE[e.type](e))
    a = _strip_raw(_run(res.df, res.events, mode), swap)
    b = _strip_raw(_run(res.df, swapped, mode), swap)
    assert a.keys() == b.keys()
    assert [k for k in a if a[k] != b[k]] == []
