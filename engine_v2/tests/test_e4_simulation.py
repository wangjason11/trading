"""E4 regression pins (PLAN_E §6.4 → §8): the real emitter stamps each flipped
event at its MOMENT, and the downstream pipeline does not depend on what
`ev.idx` holds on CTS_ESTABLISHED / BOS_CONFIRMED / pattern-path CTS_UPDATED
(E4c, the last two tests).

1. The emitter: since Plan E E4a (CTS_ESTABLISHED) / E4b (BOS_CONFIRMED) every
   such event of a real MS run carries `idx == meta["confirmed_at"]`, and the
   lagging ones keep their anchor in `meta["cts_anchor_idx"]` /
   `meta["bos_anchor_idx"]` (EST 9 known at 10; BOS 0/2, 7/10, 12/15, 17/20).
2. The swap: `_run_downstream_pipeline` runs twice on lagging fixtures — once on
   the real MS events and once on clones whose swapped type is back at its
   ANCHOR in `ev.idx` (the pre-E4 shape) — and every
   output must be identical except the declared raw readers: `sorted_events` (it
   holds the events themselves) and the `fib_tracker` object (its repr is an
   address). (The write-only KL meta `source_event_idx`, the third one on a BOS
   swap, was deleted in Plan E E4b-pre — the KL output is now compared whole.)

Fixtures: `_make_second_cts_moment_after_anchor_data` (EST (9, 10); BOS (0, 2),
(7, 10)) and `_make_multicycle_data` (4 lagging BOS, no lagging EST — PLAN_E §6.4
said 3; measured 2026-09-24), in both fib modes. History: E2b made the EST half
of the (then forward) simulation pass, E2c the BOS half; E4a / E4b turned each
half into the pins above.
"""
from __future__ import annotations

import contextlib
import copy
import io
from types import SimpleNamespace

import pytest

from engine_v2.pipeline.orchestrator import _run_downstream_pipeline
from engine_v2.structure import event_fields as ef
from engine_v2.structure.structure_engine import compute_bounded_structure
from engine_v2.tests.test_unified_probe import (
    _make_multicycle_data,
    _make_second_cts_moment_after_anchor_data,
    _prepare_df,
)

_SWAP_TYPES = {
    "est": ("CTS_ESTABLISHED",),
    "bos": ("BOS_CONFIRMED",),
    "both": ("CTS_ESTABLISHED", "BOS_CONFIRMED"),
}
# The index the swapped clone holds: the ANCHOR (the pre-E4 shape) — the
# contract names the moment on both types since Plan E E4a / E4b.
_OTHER_ROLE = {
    "CTS_ESTABLISHED": ef.cts_anchor_idx,
    "BOS_CONFIRMED": ef.bos_anchor_idx,
}
_EXCLUDED = ("sorted_events", "fib_tracker")


def _swappable(events, types):
    return [(e.type, e.idx, _OTHER_ROLE[e.type](e)) for e in events
            if e.type in types and e.idx != _OTHER_ROLE[e.type](e)]


def _strip_raw(out):
    return {k: repr(v) for k, v in out.items() if k not in _EXCLUDED}


def _run(df, events, mode):
    with contextlib.redirect_stdout(io.StringIO()):
        return _run_downstream_pipeline(df, events, +1, fib_mode=mode, skip_wvmi=False)


_SECOND = _make_second_cts_moment_after_anchor_data
_MULTI = _make_multicycle_data


@pytest.mark.parametrize("etype,maker,lagging", [
    ("CTS_ESTABLISHED", _SECOND, [(9, 10)]),
    ("CTS_ESTABLISHED", _MULTI, []),
    ("BOS_CONFIRMED", _SECOND, [(0, 2), (7, 10)]),
    ("BOS_CONFIRMED", _MULTI, [(0, 2), (7, 10), (12, 15), (17, 20)]),
])
def test_real_emitter_stamps_the_moment(etype, maker, lagging):
    """Plan E E4a / E4b: `ev.idx` of every CTS_ESTABLISHED / BOS_CONFIRMED IS its
    moment; the lagging ones keep their (anchor, moment) apart."""
    res = compute_bounded_structure(_prepare_df(maker()), 0, +1)
    evs = [e for e in res.events if e.type == etype]
    assert evs
    assert all(e.idx == e.meta["confirmed_at"] for e in evs)
    anchor = _OTHER_ROLE[etype]
    assert [(anchor(e), e.idx) for e in evs if anchor(e) != e.idx] == lagging


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


@pytest.mark.skipif(not __debug__, reason="the emitter's check is an assert (stripped under -O)")
def test_emit_bos_confirmed_asserts_idx_is_the_moment():
    """The BOS emitter refuses an `idx` other than `meta["confirmed_at"]` (Plan E
    E4b) — before any event is built or `st.bos` moves."""
    from engine_v2.structure.market_structure import MarketStructure
    ms = MarketStructure(_prepare_df(_SECOND()), 1)
    with pytest.raises(AssertionError, match="ev.idx is the moment"):
        ms._emit_bos_confirmed(7, 0.95, bos_anchor_idx=7, meta={"confirmed_at": 10})
    assert ms.events == [] and ms.state.bos is None
    ms._emit_bos_confirmed(10, 0.95, bos_anchor_idx=7, meta={"confirmed_at": 10})
    ev = ms.events[-1]
    assert (ev.idx, ev.meta["bos_anchor_idx"], ev.meta["confirmed_at"]) == (10, 7, 10)
    assert ms.state.bos.idx == 7


@pytest.mark.illegal_event_contract  # the clones hold the anchor (the pre-E4 shape)
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
    a = _strip_raw(_run(res.df, res.events, mode))
    b = _strip_raw(_run(res.df, swapped, mode))
    assert a.keys() == b.keys()
    assert [k for k in a if a[k] != b[k]] == []


# --- E4c landing review: the pattern-path CTS_UPDATED half ------------------------

@pytest.mark.illegal_event_contract  # else the validator pre-empts the emitter's own assert
@pytest.mark.skipif(not __debug__, reason="the emitter's check is an assert (stripped under -O)")
def test_emit_cts_updated_asserts_the_pattern_path_idx_is_the_moment():
    """The pattern-path CTS_UPDATED emitter refuses an `idx` other than
    `meta["confirmed_at"]`, or an anchor past it (Plan E E4c) -- before any event
    is built. The raw path carries neither key."""
    from engine_v2.structure.event_fields import CTS_UPDATED_RAW_VIA
    from engine_v2.structure.market_structure import MarketStructure
    ms = MarketStructure(_prepare_df(_SECOND()), 1)
    with pytest.raises(AssertionError, match="ev.idx is the moment"):
        ms._emit_cts_updated(9, 1.0, meta={"via": "continuous", "confirmed_at": 10, "cts_anchor_idx": 9})
    with pytest.raises(AssertionError, match="ev.idx is the moment"):
        ms._emit_cts_updated(10, 1.0, meta={"via": "continuous", "confirmed_at": 10, "cts_anchor_idx": 11})
    assert ms.events == []
    ms._emit_cts_updated(10, 1.0, meta={"via": "continuous", "confirmed_at": 10, "cts_anchor_idx": 9})
    ms._emit_cts_updated(11, 1.1, meta={"via": CTS_UPDATED_RAW_VIA})
    pat, raw = ms.events
    assert (pat.idx, pat.meta["cts_anchor_idx"], pat.meta["confirmed_at"]) == (10, 9, 10)
    assert raw.idx == 11 and "confirmed_at" not in raw.meta and "cts_anchor_idx" not in raw.meta


@pytest.mark.illegal_event_contract  # the clones hold the anchor (the pre-E4c shape)
@pytest.mark.parametrize("mode", ["h1", "cross_cycle"])
def test_downstream_outputs_do_not_depend_on_the_pattern_update_idx_role(mode):
    """The E4a / E4b swap for pattern-path CTS_UPDATED (Plan E E4c): a lagging
    pattern-path update (anchor 24, moment 25) is cloned back to its ANCHOR in
    `ev.idx`; every downstream output is identical (the raw readers excepted,
    `_EXCLUDED`). MS no longer emits a lagging one (2026-09-27: the fixture's
    one_maru_opposite at 25 ties the raw-updated extreme 24 and emits nothing), so
    the pre-fix event is INJECTED — the downstream readers stay pinned to the roles
    (defence in depth: a future event source may lag again)."""
    from engine_v2.tests.test_imbalance_c3_knowability import _multicycle_with_tied_pattern_breakout
    with contextlib.redirect_stdout(io.StringIO()):
        res = compute_bounded_structure(_prepare_df(_multicycle_with_tied_pattern_breakout()), 0, +1)
    assert int(res.df["last_breakout_pat_apply_idx"].iloc[25]) == 25   # precondition: the tying breakout WAS applied at 25
    raw24 = next(e for e in res.events if e.type == "CTS_UPDATED" and e.idx == 24)
    assert not [e for e in res.events if e.type == "CTS_UPDATED" and e.idx == 25]   # the tie: not emitted
    lagging = copy.deepcopy(raw24)
    lagging.idx = 25
    lagging.meta = {**raw24.meta, "via": "one_maru_opposite", "confirmed_at": 25, "cts_anchor_idx": 24}
    events = list(res.events)
    events.insert(events.index(raw24) + 1, lagging)     # where the pre-fix MS emitted it (pattern apply 25)
    upd = [e for e in events if e.type == "CTS_UPDATED"]
    assert [(e.idx, ef.cts_anchor_idx(e)) for e in upd if e.idx != ef.cts_anchor_idx(e)] == [(25, 24)]
    res = SimpleNamespace(df=res.df, events=events)
    swapped = copy.deepcopy(res.events)
    for e in swapped:
        if e.type == "CTS_UPDATED":
            e.idx = ef.cts_anchor_idx(e)
    a = _strip_raw(_run(res.df, res.events, mode))
    b = _strip_raw(_run(res.df, swapped, mode))
    assert a.keys() == b.keys()
    assert [k for k in a if a[k] != b[k]] == []
