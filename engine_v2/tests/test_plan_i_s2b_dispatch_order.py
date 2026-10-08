"""PLAN_I S2b (§9.2 F17, R8-2): the downstream dispatch RAISES on a CTS_ESTABLISHED handled before its own cycle's
BOS_CONFIRMED — it used to drop it silently (no fib for the cycle; the tracker's order tripwire never reached).

`processing_order_key` puts BOS_CONFIRMED at its BOS anchor and CTS_ESTABLISHED at its CTS anchor, so a cycle whose BOS
anchor lies AFTER its CTS anchor reaches the dispatch establishment-first. 0 / 34 on the reference window; the
session-4 random-tail search produced it from real MS runs (synthetic candles) in two ways — the F14 (1) shape (a cycle
established on the previous cycle's CTS candle, its BOS the proximity candle that confirmed that cycle) and a breakout
whose deepest pullback follows its extreme.
"""
from __future__ import annotations

import contextlib
import copy
import io

import pytest

import engine_v2.structure.event_fields as ef
from engine_v2.pipeline.orchestrator import _run_downstream_pipeline
from engine_v2.structure.structure_engine import _make_market_structure, _pip_size_from_pair
from engine_v2.tests._event_factory import make_bos_confirmed, make_cts_established
from engine_v2.tests.test_orchestrator_reversal_source import _ms
from engine_v2.tests.test_unified_probe import _prepare_df
from engine_v2.zones.fib_tracker import FibEventOrderError


def _downstream(df, events, sd):
    with contextlib.redirect_stdout(io.StringIO()):
        return _run_downstream_pipeline(df, events, sd, fib_mode="h1")


def _with_sid1_cycle0(bos_anchor_idx, cts_anchor_idx):
    """The double-rewind run (sid 0, up) + a sid-1 (down) cycle 0 established at 13 with the given anchors — the
    only change between the cases is where the BOS anchor sits relative to the CTS anchor."""
    res = _ms(None)
    sid1 = dict(structure_id=1, cycle_id=0, struct_direction=-1)
    events = copy.deepcopy(res.events) + [
        make_bos_confirmed(bos_anchor_idx=bos_anchor_idx, confirmed_at=13, price=0.60720, **sid1),
        make_cts_established(cts_anchor_idx=cts_anchor_idx, confirmed_at=13, price=0.60380, **sid1),
    ]
    return res.df, events


def test_an_establishment_before_its_bos_raises_instead_of_being_dropped():
    df, events = _with_sid1_cycle0(bos_anchor_idx=12, cts_anchor_idx=11)   # BOS anchor AFTER the CTS anchor
    order = [e.type for e in sorted(events, key=ef.processing_order_key) if e.meta.get("structure_id") == 1]
    assert order == ["CTS_ESTABLISHED", "BOS_CONFIRMED"]
    with pytest.raises(FibEventOrderError, match=r"sid=1 cycle=0 CTS_ESTABLISHED \(anchor 11, moment 13\) "
                                                 r"handled before its cycle's BOS_CONFIRMED"):
        _downstream(df, events, 1)


def test_it_is_a_runtime_error_so_a_sub_geometry_except_never_swallows_it():
    assert issubclass(FibEventOrderError, RuntimeError)
    assert not issubclass(FibEventOrderError, (ValueError, IndexError))


@pytest.mark.parametrize("bos_anchor_idx", [10, 11])   # before the CTS anchor, and the tie
def test_a_bos_at_or_before_the_cts_anchor_is_handled_first(bos_anchor_idx):
    """The tie sorts BOS first ("BOS_CONFIRMED" < "CTS_ESTABLISHED"): the establishment reaches the tracker."""
    df, events = _with_sid1_cycle0(bos_anchor_idx=bos_anchor_idx, cts_anchor_idx=11)
    order = [e.type for e in sorted(events, key=ef.processing_order_key) if e.meta.get("structure_id") == 1]
    assert order == ["BOS_CONFIRMED", "CTS_ESTABLISHED"]
    _downstream(df, events, 1)


# The session-4 search's WAVES seed 413 trial 1074 (synthetic M15 candles, a bearish run; the generator's zone touch
# makes candle 47's 164-pip upper wick): cycle 1's CTS raw-updated to 46 inside cycle 2's breakout window 44 -> 49,
# cycle 1 confirmed by sd-zone proximity at 47, cycle 2 established at 49 ON cycle 1's CTS candle 46 with its BOS at
# the proximity candle 47 (0.5785). Before S2b the dispatch dropped cycle 2's establishment: no cycle-2 fib record.
_F14_1_ROWS = [
    (0.60015, 0.6006, 0.59737, 0.59838), (0.59825, 0.5984, 0.59574, 0.59591), (0.59572, 0.59709, 0.59487, 0.59681),
    (0.59672, 0.59747, 0.59461, 0.59568), (0.59561, 0.59582, 0.59521, 0.59534), (0.59521, 0.5954, 0.59364, 0.59367),
    (0.59383, 0.59431, 0.59178, 0.59246), (0.5923, 0.5932, 0.58901, 0.58935), (0.58921, 0.5901, 0.58751, 0.58972),
    (0.58954, 0.5945, 0.58897, 0.58917), (0.58931, 0.59039, 0.58569, 0.58998), (0.59005, 0.59094, 0.58674, 0.58764),
    (0.58747, 0.5896, 0.58649, 0.5869), (0.58674, 0.58931, 0.58582, 0.58862), (0.58869, 0.58917, 0.58524, 0.58643),
    (0.58643, 0.58968, 0.5857, 0.58855), (0.58839, 0.59076, 0.58762, 0.58791), (0.58805, 0.59001, 0.58737, 0.58937),
    (0.58928, 0.59195, 0.58908, 0.59193), (0.5918, 0.59558, 0.59096, 0.59146), (0.5914, 0.59149, 0.5854, 0.58549),
    (0.58549, 0.58576, 0.58054, 0.58512), (0.58511, 0.5852, 0.57972, 0.57984), (0.57987, 0.58003, 0.57711, 0.57717),
    (0.57737, 0.57785, 0.57416, 0.57683), (0.57664, 0.58188, 0.57651, 0.58174), (0.58162, 0.58221, 0.57867, 0.57903),
    (0.57899, 0.57909, 0.57409, 0.57874), (0.57884, 0.5798, 0.57677, 0.57955), (0.5797, 0.5834, 0.5791, 0.5819),
    (0.58175, 0.58179, 0.57813, 0.5783), (0.57844, 0.57844, 0.57361, 0.57364), (0.57351, 0.57365, 0.57253, 0.57302),
    (0.57318, 0.57327, 0.57024, 0.57039), (0.57058, 0.57104, 0.57, 0.57013), (0.57008, 0.57035, 0.56917, 0.56962),
    (0.56971, 0.57006, 0.56911, 0.56935), (0.56954, 0.56977, 0.56902, 0.56923), (0.56921, 0.57246, 0.56835, 0.5719),
    (0.57203, 0.5767, 0.57191, 0.57668), (0.57669, 0.57771, 0.57377, 0.57399), (0.574, 0.57488, 0.57222, 0.57279),
    (0.57271, 0.57286, 0.56714, 0.56725), (0.56714, 0.56773, 0.56461, 0.5655), (0.56554, 0.56564, 0.5632, 0.56338),
    (0.56353, 0.56361, 0.55877, 0.55897), (0.55908, 0.56285, 0.55791, 0.56188), (0.56169, 0.5785, 0.56133, 0.56209),
    (0.56205, 0.56246, 0.56115, 0.56145), (0.56143, 0.56147, 0.55799, 0.55809),
]


def _f14_1_run():
    rows = [dict(o=o, h=h, l=l, c=c) for o, h, l, c in _F14_1_ROWS]
    with contextlib.redirect_stdout(io.StringIO()):
        df = _prepare_df(rows)
        ms = _make_market_structure(df, struct_direction=-1, start_idx=0, structure_id=0, timeframe="M15",
                                    pip_size=_pip_size_from_pair(df))
        df2, _events, _levels = ms.run()
    return ms, df2


def test_the_f14_1_shape_from_a_real_ms_run_now_stops_the_downstream():
    ms, df2 = _f14_1_run()
    by = {(e.type, e.meta.get("cycle_id")): e for e in ms.events
          if e.type in ("CTS_ESTABLISHED", "BOS_CONFIRMED", "CTS_CONFIRMED")}
    est2, bos2, conf1 = by[("CTS_ESTABLISHED", 2)], by[("BOS_CONFIRMED", 2)], by[("CTS_CONFIRMED", 1)]
    # The shape (all measured on this run): cycle 2 on cycle 1's CTS candle; its BOS = the proximity candle.
    assert (ef.cts_anchor_idx(est2), int(est2.idx), est2.meta["pattern_anchor_idx"]) == (46, 49, 44)
    assert (int(conf1.idx), ef.cts_anchor_idx(conf1), conf1.meta["via"]) == (47, 46, "sd_zone_proximity")
    assert (ef.bos_anchor_idx(bos2), bos2.price) == (47, 0.5785)
    # MS emits cycle 1's confirmation BEFORE cycle 2's establishment; the processing order inverts them AND puts the
    # establishment before its own BOS.
    emitted = [(e.type, e.meta.get("cycle_id")) for e in ms.events]
    assert emitted.index(("CTS_CONFIRMED", 1)) < emitted.index(("CTS_ESTABLISHED", 2))
    processed = [(e.type, e.meta.get("cycle_id")) for e in sorted(ms.events, key=ef.processing_order_key)]
    assert processed.index(("CTS_ESTABLISHED", 2)) < processed.index(("BOS_CONFIRMED", 2))
    assert processed.index(("CTS_ESTABLISHED", 2)) < processed.index(("CTS_CONFIRMED", 1))
    for mode in ("h1", "cross_cycle"):
        with pytest.raises(FibEventOrderError, match=r"sid=0 cycle=2 CTS_ESTABLISHED \(anchor 46, moment 49\)"):
            with contextlib.redirect_stdout(io.StringIO()):
                _run_downstream_pipeline(df2, ms.events, -1, fib_mode=mode)
