"""PLAN_J J3 (PLAN_I R9-2 (ii), §9.2 F17 (ii)): the cycle >= 1 BOS window ends at the new cycle's CTS anchor.

The window used to end at the breakout's APPLY candle: a breakout whose extreme is not its apply candle could take a
deeper pullback candle AFTER its extreme as the BOS — the BOS anchor then sorts after the CTS anchor
(`processing_order_key`), and the downstream dispatch raises `FibEventOrderError` (PLAN_I S2b). 0 on the reference
window; the session-4 random-tail corpus had it in 7 trials (WAVES-B 4, WAVES 3) — 0 after J3.

The real MS run: WAVES seed 301 trial 572 (synthetic H1 candles, sd -1; the generator's zone-touch made candle 44's
250-pip upper wick). Cycle 0 confirmed by the pullback applying at 7; a `one_maru_continuous` anchored at 40 applies at
45 — cycle 1 ESTABLISHED, its CTS anchor 43 (low 0.57645). The BOS window [7, 45] took candle 44's high 0.60401 (after
the CTS); [7, 43] takes candle 39's 0.59943.
"""
from __future__ import annotations

import contextlib
import io

import engine_v2.structure.event_fields as ef
from engine_v2.pipeline.orchestrator import _run_downstream_pipeline
from engine_v2.structure.structure_engine import _make_market_structure, _pip_size_from_pair
from engine_v2.tests._cross_fib_limits import NO_P, main_limits
from engine_v2.tests.test_unified_probe import _prepare_df

_ROWS_301_572 = [
    (0.6, 0.6002, 0.5978, 0.598), (0.598, 0.5982, 0.5958, 0.596), (0.596, 0.5962, 0.5938, 0.594),
    (0.594, 0.5942, 0.5918, 0.592), (0.592, 0.5922, 0.5898, 0.59), (0.59, 0.5902, 0.5878, 0.588),
    (0.588, 0.5897, 0.5878, 0.5895), (0.5895, 0.5912, 0.5893, 0.591), (0.591, 0.5911, 0.5883, 0.5885),
    (0.58836, 0.58991, 0.58824, 0.58987), (0.58982, 0.59059, 0.5865, 0.5904), (0.59054, 0.59125, 0.59039, 0.59111),
    (0.59123, 0.59458, 0.59027, 0.59364), (0.59367, 0.59384, 0.59061, 0.59063), (0.59057, 0.59161, 0.58779, 0.58803),
    (0.5881, 0.59135, 0.5877, 0.58783), (0.58769, 0.58818, 0.58734, 0.58754), (0.58744, 0.58813, 0.58339, 0.5845),
    (0.58432, 0.58747, 0.584, 0.58495), (0.58506, 0.58888, 0.58444, 0.58469), (0.58451, 0.58485, 0.58432, 0.5847),
    (0.5848, 0.59043, 0.58479, 0.59036), (0.59052, 0.59146, 0.58892, 0.59128), (0.59125, 0.59206, 0.58911, 0.58937),
    (0.58919, 0.59244, 0.58837, 0.58848), (0.58832, 0.5895, 0.58547, 0.58666), (0.5867, 0.58691, 0.58601, 0.58647),
    (0.58662, 0.59046, 0.58646, 0.5872), (0.58711, 0.5875, 0.5868, 0.58723), (0.58732, 0.59074, 0.58678, 0.58979),
    (0.58989, 0.59151, 0.58972, 0.59139), (0.5915, 0.59508, 0.59145, 0.59498), (0.59483, 0.59804, 0.59405, 0.59446),
    (0.5946, 0.59669, 0.59429, 0.59581), (0.59572, 0.59622, 0.59561, 0.59601), (0.59619, 0.59622, 0.59094, 0.59105),
    (0.59109, 0.59128, 0.58942, 0.58953), (0.5895, 0.58977, 0.58742, 0.58927), (0.58947, 0.58963, 0.58363, 0.58367),
    (0.58352, 0.59943, 0.58326, 0.58392), (0.58408, 0.58417, 0.58233, 0.58252), (0.58246, 0.58288, 0.57881, 0.57976),
    (0.57985, 0.58248, 0.57924, 0.57961), (0.57975, 0.58013, 0.57645, 0.57916), (0.57904, 0.60401, 0.57879, 0.57981),
    (0.57978, 0.58018, 0.5771, 0.57748), (0.57762, 0.57793, 0.57706, 0.57749), (0.57762, 0.57802, 0.57678, 0.57711),
    (0.57692, 0.57696, 0.57255, 0.57265), (0.57278, 0.57286, 0.56796, 0.56816), (0.56809, 0.56844, 0.56748, 0.56794),
    (0.56795, 0.56837, 0.56769, 0.56818), (0.56837, 0.57164, 0.56834, 0.57161), (0.57159, 0.57353, 0.57114, 0.57248),
    (0.57263, 0.57572, 0.57171, 0.57549), (0.57554, 0.57574, 0.5729, 0.57298), (0.57301, 0.57457, 0.57213, 0.5723),
    (0.57242, 0.57265, 0.57165, 0.57209), (0.57197, 0.57543, 0.57179, 0.57524), (0.57538, 0.57901, 0.57495, 0.57596),
    (0.57603, 0.57624, 0.57558, 0.57569), (0.57572, 0.57576, 0.57311, 0.57323), (0.57336, 0.57384, 0.57292, 0.57347),
    (0.57365, 0.57375, 0.56838, 0.56852), (0.56865, 0.56894, 0.56308, 0.56806), (0.56812, 0.56851, 0.56785, 0.568),
]


def _run():
    df = _prepare_df([dict(o=o, h=h, l=l, c=c) for o, h, l, c in _ROWS_301_572])
    ms = _make_market_structure(df, cross_fib_limit=NO_P, struct_direction=-1, start_idx=0, structure_id=0,
                                timeframe="H1", pip_size=_pip_size_from_pair(df))
    with contextlib.redirect_stdout(io.StringIO()):
        ms.run()
    return ms


def test_the_bos_of_301_572_is_before_its_cts():
    ms = _run()
    est = {e.meta["cycle_id"]: e for e in ms.events if e.type == "CTS_ESTABLISHED"}
    bos = {e.meta["cycle_id"]: e for e in ms.events if e.type == "BOS_CONFIRMED"}
    assert (int(est[1].idx), ef.cts_anchor_idx(est[1]), est[1].price) == (45, 43, 0.57645)
    assert (int(bos[1].idx), ef.bos_anchor_idx(bos[1]), bos[1].price) == (45, 39, 0.59943)   # was 44, 0.60401
    for c in est:
        assert ef.bos_anchor_idx(bos[c]) <= ef.cts_anchor_idx(est[c])
    assert bos[1].meta["last_pullback_apply_idx"] == 7                                       # the window's start


def test_the_downstream_dispatch_takes_301_572():
    """S2b's raise (an establishment handled before its BOS) no longer fires: the BOS sorts first."""
    ms = _run()
    order = [(e.type, e.meta["cycle_id"]) for e in sorted(ms.events, key=ef.processing_order_key)
             if e.type in ("BOS_CONFIRMED", "CTS_ESTABLISHED")]
    assert order.index(("BOS_CONFIRMED", 1)) < order.index(("CTS_ESTABLISHED", 1))
    with contextlib.redirect_stdout(io.StringIO()):
        _run_downstream_pipeline(ms.df, ms.events, -1, cross_fib_limits=main_limits(ms.events))
