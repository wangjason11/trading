"""A WVMI record's temporary LP never passes its cycle's lifecycle end (2026-09-28).

The temp LP (the qualifying pullback candle whose close is closest to the BOS zone's outer bound) was searched from
FP + 1 to the DATA end. A record locks only at the next cycle's BOS, so a cycle that ENDED otherwise (a reversal on
main) kept searching into whatever superseded it: on the reference window H1 (0,1), ended by the sid-0 reversal at
902, took candle 988 (a sid-1 candle) — pullback_momentum 0.904 instead of 0.759 (LP 896). Now the main tracker gets
each cycle's end from `compute_cycle_lifecycle` (the table KL / POI read; half-open `[start, end)`) and searches to
`end - 1`; an open cycle still searches to the data end. Sub sweeps pass no ends: a sub projection's frame already
stops at the sub's end (measured 5/5) — the one-candle end-inclusive difference belongs to the deferred WVMI plan.
"""
from __future__ import annotations

import contextlib
import copy
import io

import pandas as pd

from engine_v2.pipeline.orchestrator import _run_downstream_pipeline
from engine_v2.structure.structure_engine import compute_bounded_structure
from engine_v2.tests.test_ms_stop_after_cts import _R, _make_double_rewind_data
from engine_v2.tests.test_unified_probe import _prepare_df
from engine_v2.zones.structure_lifecycle import compute_cycle_lifecycle, compute_reversal_idx_by_sid
from engine_v2.zones.wvmi import WVMITracker


def _post_reversal_lure():
    """`_make_double_rewind_data` (sd +1; sid 0 reverses at 17) with candles 18-19 replaced AFTER the reversal:
    18 closes up at .5960, 19 closes DOWN at .5920 (direction -1, vol_dir -1 = the FP's) — 0.0002 from the cycle-1
    BOS zone's outer .5918, nearer than the in-cycle LP candle 14 (close .5900, 0.0018)."""
    rows = list(_make_double_rewind_data())
    rows[18] = _R(0.58800, 0.59650, 0.58780, 0.59600)
    rows[19] = _R(0.59600, 0.59620, 0.59150, 0.59200)
    return rows


def _run(rows):
    with contextlib.redirect_stdout(io.StringIO()):
        res = compute_bounded_structure(_prepare_df(rows), 0, 1, end_idx=None)
        out = _run_downstream_pipeline(res.df, copy.deepcopy(res.events), 1, fib_mode="h1")
    return res, out


def _sig(events):
    return [(e.idx, e.type, repr(sorted(e.meta.items()))) for e in events]


def test_an_ended_cycle_takes_no_lp_past_its_end():
    base, _ = _run(_make_double_rewind_data())
    res, out = _run(_post_reversal_lure())
    assert _sig(res.events) == _sig(base.events)             # the lure is after the reversal: MS unchanged
    life = compute_cycle_lifecycle(res.events, compute_reversal_idx_by_sid(res.events))
    assert life[(0, 1)][1:] == (17, "reversal")
    recs = {(r.bos_structure_id, r.bos_cycle_id): r for r in out["wvmi_records"]}
    r = recs[(0, 1)]
    assert (r.fp_idx, r.lp_idx, r.lp_locked) == (13, 14, False)   # unbounded: 19 (the lure), momentum 0.7
    assert r.pullback_momentum == 1.0 and r.sell_momentum == 1.0
    # the bound applies at creation too: an update-only bound would move 19 -> 14 and mark it "updated"
    assert r.status == "created"
    assert (recs[(0, 0)].lp_idx, recs[(0, 0)].lp_locked) == (9, True)   # a locked record keeps the official LP


def test_the_search_end_is_the_last_live_candle_or_the_data_end():
    df = pd.DataFrame({"c": [1.0] * 20})
    t = WVMITracker(cycle_end_by_key={(0, 1): 17, (0, 2): 40})
    assert t._lp_search_end((0, 1), df) == 16          # half-open [start, end): end - 1
    assert t._lp_search_end((0, 2), df) == 19          # an end past the data: the data end
    assert t._lp_search_end((0, 3), df) == 19          # an open cycle: the data end
    assert WVMITracker()._lp_search_end((0, 1), df) == 19   # no ends (the sub sweep): the data end
