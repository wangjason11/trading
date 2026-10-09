"""PLAN_J J2 (PLAN_I R9-2 (i), §9.2 F17 (i); round 1: J2b): a breakout establishing cycle T >= 1 must break cycle T-1's
`cts_threshold` as it stands at its APPLY — else it is VOID: nothing applied, its back-fill undone, the anchors it
skipped scanned (`MarketStructure._apply_pattern_at_apply_idx` returns True; `_step_anchor` restores the snapshot it
took before the back-fill, steps the anchor candle and continues at anchor + 1).

The winner is picked at its anchor against the detection threshold of that moment (`winner_break_th`: the range's
breakout bound, None when no range is active — then anything passes); its back-fill can confirm the cycle (a proximity
hit) and set the threshold. 0 on the reference window (every establishment detects against the threshold it lands
on, PLAN_I §9.2 F12); the session-4 corpus had 112 such establishments in 72,000 trials, every one detected with no
range (PLAN_J §3).

Real MS runs on synthetic WAVES candles (H1, sd +1; `random_tail_search.py wrows SEED T`):
- 302 / 167: cycle 1 (CTS 28, 0.62883) CONFIRMED by proximity at 30; a `one_maru_continuous` anchored at 27 (no range)
  applies at 32 — it used to establish cycle 2 there (candle 32 closed 0.62951 above 0.62883, but the pattern's first
  candle 27 closed 0.62812 below it). VOID; the scan finds the breakout anchored at 32 -> cycle 2 at 36 (CTS 0.63453).
- 410 / 230: cycle 1 CONFIRMED by proximity at 24 (CTS 23, 0.62592); a `continuous` anchored at 24 — picked before
  candle 24's own step confirmed cycle 1, with no range active — applied at 28 and established cycle 2 there. VOID; the scan finds a `one_maru_continuous` anchored at 25 that applies at 26 — EARLIER
  than the void winner: today's winner pre-empted it.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import io
import sys

import numpy as np
import pytest

import engine_v2.structure.event_fields as ef
from engine_v2.pipeline.orchestrator import _run_downstream_pipeline
from engine_v2.structure.structure_engine import _make_market_structure, _pip_size_from_pair
from engine_v2.tests._cross_fib_limits import NO_P, main_limits
from engine_v2.tests.test_unified_probe import _make_multicycle_data, _prepare_df

_ROWS_302_167 = [
    (0.6, 0.6022, 0.5998, 0.602), (0.602, 0.6042, 0.6018, 0.604), (0.604, 0.6062, 0.6038, 0.606),
    (0.606, 0.6082, 0.6058, 0.608), (0.608, 0.6102, 0.6078, 0.61), (0.61, 0.6122, 0.6098, 0.612),
    (0.612, 0.6122, 0.6103, 0.6105), (0.6105, 0.6107, 0.6088, 0.609), (0.609, 0.6117, 0.6089, 0.6115),
    (0.61155, 0.61186, 0.60666, 0.61053), (0.61044, 0.61096, 0.61005, 0.61079), (0.61088, 0.61142, 0.61044, 0.61125),
    (0.6111, 0.61504, 0.6107, 0.61142), (0.61151, 0.61224, 0.61113, 0.61179), (0.61175, 0.61401, 0.61158, 0.61228),
    (0.61243, 0.61344, 0.60921, 0.61298), (0.61299, 0.6174, 0.61269, 0.61377), (0.61365, 0.61406, 0.61279, 0.61321),
    (0.61308, 0.61606, 0.61248, 0.61507), (0.61515, 0.6156, 0.61468, 0.61485), (0.61476, 0.61521, 0.61393, 0.61425),
    (0.61418, 0.61599, 0.61301, 0.61557), (0.61568, 0.6166, 0.61384, 0.61461), (0.61478, 0.61844, 0.6145, 0.61725),
    (0.61744, 0.61795, 0.61708, 0.61764), (0.61776, 0.62129, 0.61707, 0.62061), (0.62079, 0.62494, 0.62064, 0.62482),
    (0.62484, 0.62831, 0.6247, 0.62812), (0.62823, 0.62883, 0.62608, 0.62847), (0.62856, 0.6288, 0.62657, 0.62757),
    (0.62773, 0.62784, 0.61289, 0.62733), (0.6272, 0.6274, 0.62345, 0.62655), (0.62663, 0.63016, 0.62613, 0.62951),
    (0.62956, 0.63035, 0.62938, 0.62991), (0.62987, 0.63048, 0.62799, 0.62888), (0.62903, 0.6296, 0.62746, 0.62935),
    (0.62921, 0.63453, 0.62914, 0.63446), (0.63432, 0.63753, 0.63407, 0.6349), (0.63504, 0.63529, 0.63177, 0.6327),
    (0.63261, 0.63297, 0.63191, 0.63239), (0.6324, 0.63286, 0.62979, 0.63022), (0.63011, 0.63032, 0.62949, 0.62993),
    (0.62977, 0.63062, 0.62625, 0.62681), (0.62686, 0.62926, 0.62588, 0.62813), (0.62798, 0.63062, 0.6275, 0.6301),
    (0.63018, 0.63318, 0.6299, 0.63275), (0.6329, 0.63668, 0.63287, 0.63653), (0.63669, 0.63682, 0.63606, 0.63639),
    (0.63656, 0.63693, 0.63574, 0.63597), (0.6358, 0.6378, 0.63548, 0.63713), (0.6373, 0.63756, 0.63436, 0.63681),
    (0.63666, 0.63673, 0.62666, 0.63292), (0.633, 0.6348, 0.63285, 0.63461), (0.63447, 0.63625, 0.63429, 0.63619),
    (0.63632, 0.63681, 0.63557, 0.63598), (0.63581, 0.63867, 0.63564, 0.63865), (0.63872, 0.64355, 0.63858, 0.64345),
    (0.64352, 0.64719, 0.64256, 0.64642), (0.64655, 0.64995, 0.64544, 0.64949), (0.64955, 0.65027, 0.64769, 0.6501),
    (0.64998, 0.65091, 0.64954, 0.6505), (0.65038, 0.65144, 0.65024, 0.65096), (0.65113, 0.65299, 0.65107, 0.65288),
    (0.65283, 0.65309, 0.65231, 0.65253), (0.65254, 0.65372, 0.64981, 0.65001), (0.65015, 0.6504, 0.64951, 0.64963),
    (0.64976, 0.65003, 0.64951, 0.64987), (0.64996, 0.65276, 0.64925, 0.65165), (0.65167, 0.65231, 0.65148, 0.65196),
    (0.65191, 0.65476, 0.65098, 0.65398), (0.6541, 0.65603, 0.65306, 0.65532), (0.65523, 0.65561, 0.65488, 0.65546),
    (0.65531, 0.65604, 0.6551, 0.65556), (0.65563, 0.65581, 0.65499, 0.65536), (0.65548, 0.65583, 0.65381, 0.65501),
    (0.65481, 0.65513, 0.65453, 0.65499), (0.65489, 0.65679, 0.65452, 0.65591), (0.65586, 0.65596, 0.65037, 0.6505),
    (0.65053, 0.65299, 0.65046, 0.6529), (0.65295, 0.65342, 0.65249, 0.6528),
]

_ROWS_410_230 = [
    (0.6, 0.6022, 0.5998, 0.602), (0.602, 0.6042, 0.6018, 0.604), (0.604, 0.6062, 0.6038, 0.606),
    (0.606, 0.6082, 0.6058, 0.608), (0.608, 0.6102, 0.6078, 0.61), (0.61, 0.6122, 0.6098, 0.612),
    (0.612, 0.6122, 0.6103, 0.6105), (0.6105, 0.6107, 0.6088, 0.609), (0.609, 0.6117, 0.6089, 0.6115),
    (0.61145, 0.6123, 0.61113, 0.612), (0.61215, 0.61281, 0.60889, 0.60986), (0.61004, 0.61016, 0.60551, 0.6098),
    (0.60988, 0.61174, 0.60556, 0.6095), (0.60937, 0.60948, 0.6061, 0.60616), (0.60611, 0.60671, 0.60219, 0.60633),
    (0.60619, 0.60696, 0.60585, 0.60657), (0.60643, 0.60669, 0.60573, 0.60616), (0.60619, 0.61024, 0.60612, 0.61007),
    (0.61003, 0.61016, 0.60929, 0.60948), (0.60953, 0.61507, 0.60953, 0.61498), (0.615, 0.61861, 0.61409, 0.61771),
    (0.61786, 0.62163, 0.61776, 0.62153), (0.6215, 0.62343, 0.6206, 0.62286), (0.62294, 0.62592, 0.62233, 0.62251),
    (0.6226, 0.62349, 0.60408, 0.62327), (0.62316, 0.62861, 0.62308, 0.62854), (0.62861, 0.63389, 0.62845, 0.63378),
    (0.63394, 0.63421, 0.63347, 0.63409), (0.63406, 0.63586, 0.63375, 0.63509), (0.63518, 0.63688, 0.63469, 0.63658),
    (0.63656, 0.64141, 0.63636, 0.63733), (0.63747, 0.64197, 0.63736, 0.64191), (0.64183, 0.64261, 0.63891, 0.6401),
    (0.64024, 0.64074, 0.63577, 0.64056), (0.64065, 0.64155, 0.63861, 0.63894), (0.63905, 0.63976, 0.63875, 0.63938),
    (0.63951, 0.64326, 0.63927, 0.64233), (0.64217, 0.64287, 0.63813, 0.6425), (0.64241, 0.64284, 0.63953, 0.64207),
    (0.64202, 0.64218, 0.63845, 0.63857), (0.63877, 0.6391, 0.63824, 0.6385), (0.63848, 0.63911, 0.63831, 0.63881),
    (0.6389, 0.63897, 0.63701, 0.63713), (0.63719, 0.63812, 0.63462, 0.63787), (0.6378, 0.63795, 0.6355, 0.63568),
    (0.6358, 0.63618, 0.63254, 0.63526), (0.63541, 0.63948, 0.63491, 0.63515), (0.63499, 0.63583, 0.63114, 0.63555),
    (0.63546, 0.6364, 0.63353, 0.6346), (0.63463, 0.63822, 0.63456, 0.63806), (0.63818, 0.63904, 0.63645, 0.6388),
    (0.63861, 0.6403, 0.6384, 0.64004), (0.64013, 0.64174, 0.63954, 0.63983), (0.63985, 0.64027, 0.63741, 0.6393),
    (0.63924, 0.64248, 0.63835, 0.64132), (0.64113, 0.64412, 0.64097, 0.64403), (0.64416, 0.64832, 0.64397, 0.64455),
    (0.64445, 0.64493, 0.64416, 0.64478), (0.64486, 0.64654, 0.64465, 0.64587), (0.64593, 0.64633, 0.6455, 0.64614),
    (0.6463, 0.64723, 0.64382, 0.64704), (0.64701, 0.64762, 0.64318, 0.64726), (0.64744, 0.64772, 0.64704, 0.64755),
    (0.64751, 0.64998, 0.64744, 0.6499), (0.64984, 0.65022, 0.64973, 0.65001), (0.64984, 0.65021, 0.64559, 0.64936),
    (0.64918, 0.65217, 0.6489, 0.65098), (0.65115, 0.6527, 0.65027, 0.65049), (0.6506, 0.65108, 0.6464, 0.64995),
    (0.64977, 0.65085, 0.6494, 0.65064), (0.65071, 0.65108, 0.65043, 0.65092), (0.65106, 0.65153, 0.64741, 0.64855),
    (0.64852, 0.64884, 0.64793, 0.64806), (0.64804, 0.64933, 0.64605, 0.64884),
]


def _run(rows, sd, tf="H1", hook=None):
    df = _prepare_df([dict(o=o, h=h, l=l, c=c) for o, h, l, c in rows])
    ms = _make_market_structure(df, cross_fib_limit=NO_P, struct_direction=sd, start_idx=0, structure_id=0,
                                timeframe=tf, pip_size=_pip_size_from_pair(df))
    if hook is not None:
        hook(ms)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ms.run()
    return ms, out.getvalue()


def _est(ms):
    return {e.meta["cycle_id"]: e for e in ms.events if e.type == "CTS_ESTABLISHED"}


def _bos(ms):
    return {e.meta["cycle_id"]: e for e in ms.events if e.type == "BOS_CONFIRMED"}


def test_302_167_the_void_breakout_and_the_one_the_scan_finds():
    ms, log = _run(_ROWS_302_167, 1)
    assert "[ms] breakout VOID: one_maru_continuous anchor 27 apply 32 does not break cts_threshold 0.62883 " \
           "(detected against None)" in log
    est, bos = _est(ms), _bos(ms)
    assert all(int(e.idx) != 32 for e in est.values())                       # nothing established at the void apply
    assert (int(est[2].idx), ef.cts_anchor_idx(est[2]), est[2].price, est[2].meta["pattern_anchor_idx"]) == \
        (36, 36, 0.63453, 32)
    assert est[2].meta["cts_frozen"] == 0.62883                              # it broke the threshold it lands on
    assert (ef.bos_anchor_idx(bos[2]), bos[2].price) == (30, 0.61289)
    assert "WARNING [ms] cts_frozen" not in log


def test_410_230_the_scan_finds_a_breakout_the_void_winner_had_pre_empted():
    ms, log = _run(_ROWS_410_230, 1)
    assert "[ms] breakout VOID: continuous" in log and "apply 28" in log
    est = _est(ms)
    assert (int(est[2].idx), ef.cts_anchor_idx(est[2]), est[2].price, est[2].meta["pattern_anchor_idx"],
            est[2].meta["via"]) == (26, 26, 0.63389, 25, "one_maru_continuous")
    conf1 = [e for e in ms.events if e.type == "CTS_CONFIRMED" and e.meta["cycle_id"] == 1][0]
    assert (int(conf1.idx), ef.cts_anchor_idx(conf1), conf1.meta["confirmation_method"]) == (24, 23, "sd_zone_proximity")


@pytest.mark.parametrize("rows", [_ROWS_302_167, _ROWS_410_230], ids=["302_167", "410_230"])
def test_the_runs_are_whole_and_the_downstream_takes_them(rows):
    """No order tripwire (MS completes), every cycle's BOS sorts before its establishment, the dispatch runs."""
    ms, _ = _run(rows, 1)
    est, bos = _est(ms), _bos(ms)
    for c in est:
        assert ef.bos_anchor_idx(bos[c]) <= ef.cts_anchor_idx(est[c])
    with contextlib.redirect_stdout(io.StringIO()):
        _run_downstream_pipeline(ms.df, ms.events, 1, cross_fib_limits=main_limits(ms.events))


def test_snapshot_restore_round_trip():
    """`_snapshot_state` / `_restore_state` (PLAN_J): the state (in place — aliases stay valid), the events (truncated,
    the kept ones the same objects), the embedded tracker (in place, its handled events kept by identity) and the
    output rows come back exactly."""
    df = _prepare_df(_make_multicycle_data())
    ms = _make_market_structure(df, cross_fib_limit=NO_P, struct_direction=1, start_idx=0, structure_id=0,
                                timeframe="M15", pip_size=_pip_size_from_pair(df))
    with contextlib.redirect_stdout(io.StringIO()):
        ms.run()
    st, trk = ms.state, ms._fib_tracker
    before_state = copy.deepcopy(st.__dict__)
    before_events = list(ms.events)
    before_handled = list(trk._handled)
    before_fibs = copy.deepcopy(trk._fibs)
    before_rows = {c: a[3:9].copy() for c, a in ms._out.items()}
    snap = ms._snapshot_state(3, 9)
    st.cts_threshold = -1.0
    st.cts_phase = "EST_OR_UPD"
    ms.events.append(object())
    trk._handled.append(object())
    trk._fibs.clear()
    for a in ms._out.values():
        a[3:9] = a[0]
    ms._restore_state(snap)
    assert ms.state is st and ms._fib_tracker is trk
    assert st.__dict__ == before_state
    assert len(ms.events) == len(before_events) and all(a is b for a, b in zip(ms.events, before_events))
    assert len(trk._handled) == len(before_handled) and all(a is b for a, b in zip(trk._handled, before_handled))
    assert trk._fibs == before_fibs
    for c, a in before_rows.items():
        got = ms._out[c][3:9]
        if a.dtype.kind == "f":
            assert np.array_equal(got, a, equal_nan=True), c
        else:
            assert list(got) == list(a), c


def _retest_calls(ms):
    """Record every `detect_best_for_anchor` call made by `_apply_pattern_at_apply_idx` (the J2 re-test) and the
    `cts_phase` it was made in."""
    calls, orig = [], ms._bp.detect_best_for_anchor

    def spy(idx, direction, break_threshold=None):
        if sys._getframe(1).f_code.co_name == "_apply_pattern_at_apply_idx":
            calls.append((int(idx), break_threshold, ms.state.cts_phase))
        return orig(idx, direction, break_threshold)
    ms._bp.detect_best_for_anchor = spy
    return calls


def test_only_an_establishment_is_retested():
    """410 / 230 also has a CONTINUATION breakout (anchored at 63, applying at 64) detected against the range bound
    0.64832 while cycle 2 is unconfirmed (`cts_threshold` None — it is set at the confirmation). It updates the CTS;
    it is not re-tested. The one re-test is the void establishment (anchored at 24), against cycle 1's 0.62592."""
    box = {}
    ms, _ = _run(_ROWS_410_230, 1, hook=lambda m: box.setdefault("calls", _retest_calls(m)))
    calls = box["calls"]
    assert calls == [(24, 0.62592, "CONFIRMED")]


@pytest.mark.parametrize("change", ["name", "apply"])
def test_a_retest_that_finds_another_pattern_or_apply_is_void(change):
    """302 / 167 with the re-test made to return the winner itself, but renamed / applying a candle later: still VOID
    (the same pattern must qualify at the same apply) — cycle 2 still comes at 36 from the pattern anchored at 32."""
    def hook(ms):
        orig = ms._bp.detect_best_for_anchor

        def fake(idx, direction, break_threshold=None):
            if sys._getframe(1).f_code.co_name == "_apply_pattern_at_apply_idx":
                won = orig(idx, direction, None)                           # the winner (detected with no range)
                assert won is not None and int(ms._apply_idx(won)) == 32
                if change == "name":
                    return dataclasses.replace(won, name="double_maru")
                return dataclasses.replace(won, confirmation_idx=int(won.confirmation_idx) + 1)
            return orig(idx, direction, break_threshold)
        ms._bp.detect_best_for_anchor = fake
    ms, log = _run(_ROWS_302_167, 1, hook=hook)
    assert "[ms] breakout VOID: one_maru_continuous anchor 27 apply 32" in log
    assert int(_est(ms)[2].idx) == 36


def test_the_anchors_a_void_winner_skipped_see_the_state_as_of_themselves():
    """302 / 167: the void winner's back-fill [27, 32) confirmed cycle 1 by proximity at 30. Undone: the anchors stepped
    after the void that precede 30 see cycle 1 UNCONFIRMED (no `cts_confirmed_idx`) — not the back-fill's state."""
    seen = []

    def hook(ms):
        orig = ms._step_anchor

        def step(i):
            seen.append((int(i), ms.state.cts_confirmed_idx, ms.state.cts_phase))
            return orig(i)
        ms._step_anchor = step
    _run(_ROWS_302_167, 1, hook=hook)
    after_void = [x for x in seen if 27 < x[0] < 30]
    assert after_void and after_void[0][0] == 28
    assert all(c is None and phase == "EST_OR_UPD" for _, c, phase in after_void)
