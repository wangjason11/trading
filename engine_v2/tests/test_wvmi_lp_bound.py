"""A WVMI record's temporary LP never passes its cycle's lifecycle end (2026-09-28).

The temp LP (the qualifying pullback candle whose close is closest to the BOS zone's outer bound) was searched from
FP + 1 to the DATA end. A record locks only at the next cycle's BOS, so a cycle that ENDED otherwise (a reversal on
main) kept searching into whatever superseded it: on the reference window H1 (0,1), ended by the sid-0 reversal at
902, took candle 988 (a sid-1 candle) — pullback_momentum 0.904 instead of 0.759 (LP 896). Now the main tracker gets
each cycle's end from `compute_cycle_lifecycle` (the table KL / POI read; half-open `[start, end)`) and searches to
`end - 1`; an open cycle still searches to the data end. Since Plan G (Q4) a sub's WVMI is computed inside its
projection by the same helper, so a sub record gets the same `end - 1` bound (before it, the sub sweep passed no ends
and searched to its frame end, the sub's end candle included).
"""
from __future__ import annotations

import contextlib
import copy
import io

import pandas as pd

import engine_v2.pipeline.orchestrator as orch
from engine_v2.multitf.pooled_structure_build import project_to_window
from engine_v2.pipeline.orchestrator import _run_downstream_pipeline
from engine_v2.structure.structure_engine import compute_bounded_structure
from engine_v2.tests.test_ms_stop_after_cts import _R, _make_double_rewind_data
from engine_v2.tests.test_unified_probe import _prepare_df
from engine_v2.tests import test_wvmi as tw   # module import: importing its Test class would re-collect it here
from engine_v2.zones.structure_lifecycle import compute_cycle_lifecycle, compute_reversal_idx_by_sid
from engine_v2.zones.wvmi import WVMITracker


def _post_reversal_lure():
    """`_make_double_rewind_data` (sd +1; sid 0 reverses at 17; cycles at 2 / 8 / 12) with candles 18-19 replaced
    AFTER the reversal: 18 closes up at .5960, 19 closes DOWN at .5920 (direction -1, vol_dir -1 = the FP's) — 0.0002
    from the cycle-2 BOS zone's outer .5918, nearer than the in-cycle LP candle 14 (close .5900, 0.0018)."""
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
    assert life[(0, 2)][1:] == (17, "reversal")
    recs = {(r.bos_structure_id, r.bos_cycle_id): r for r in out["wvmi_records"]}
    r = recs[(0, 2)]
    assert (r.fp_idx, r.lp_idx, r.lp_locked) == (13, 14, False)   # unbounded: 19 (the lure), momentum 0.7
    assert r.pullback_momentum == 1.0 and r.sell_momentum == 1.0
    # the bound applies at creation too: an update-only bound would move 19 -> 14 and mark it "updated"
    assert r.status == "created"
    assert (recs[(0, 1)].lp_idx, recs[(0, 1)].lp_locked) == (9, True)   # a locked record keeps the official LP


def test_the_search_end_is_the_last_live_candle_or_the_data_end():
    df = pd.DataFrame({"c": [1.0] * 20})
    t = WVMITracker(cycle_end_by_key={(0, 1): 17, (0, 2): 40})
    assert t._lp_search_end((0, 1), df) == 16          # half-open [start, end): end - 1
    assert t._lp_search_end((0, 2), df) == 19          # an end past the data: the data end
    assert t._lp_search_end((0, 3), df) == 19          # an open cycle: the data end
    assert WVMITracker()._lp_search_end((0, 1), df) == 19   # no ends given: the data end


# --- landing-review pins (2026-09-28, mutation lens): the orchestrator wiring, an empty bound, a growing frame -------

def _late_candidate(events, apply_idx):
    """The J2 candidate (anchor 14, realised at 17) with a LATER scheduled apply — a step-anchor reversal realising
    before its watch candidate's apply; realisable only up to anchor + range_max_k = 19."""
    assert apply_idx <= 14 + 5
    events = copy.deepcopy(events)
    for e in events:
        if e.type == "REVERSAL_CANDIDATE" and e.meta["apply_idx"] == 17:
            e.meta["apply_idx"] = apply_idx
    return events


def _captured_ends(monkeypatch, res, events, cap=None):
    seen = []

    class Capture(WVMITracker):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            seen.append(dict(self._cycle_end_by_key))

    monkeypatch.setattr(orch, "WVMITracker", Capture)
    with contextlib.redirect_stdout(io.StringIO()):
        orch._run_downstream_pipeline(res.df, copy.deepcopy(events), 1, fib_mode="h1", lifecycle_cap=cap)
    assert len(seen) == 1
    return seen[0]


def test_the_main_tracker_gets_every_cycle_end_from_the_lifecycle_table(monkeypatch):
    """The ends the orchestrator hands the tracker ARE `compute_cycle_lifecycle`'s: the next-cycle start for (0,0) and
    (0,1), the REALISED reversal for (0,2) (not the candidate's apply 19), the cap when one applies — exclusive values
    (the tracker subtracts 1). The end-to-end test above accepts any bound 14..18, so this pins the wiring."""
    with contextlib.redirect_stdout(io.StringIO()):
        res = compute_bounded_structure(_prepare_df(_post_reversal_lure()), 0, 1, end_idx=None)
    events = _late_candidate(res.events, 19)
    assert _captured_ends(monkeypatch, res, events) == {(0, 0): 8, (0, 1): 12, (0, 2): 17}
    assert _captured_ends(monkeypatch, res, events, cap=15) == {(0, 0): 8, (0, 1): 12, (0, 2): 15}


def test_a_cycle_ending_before_any_pullback_candidate_gets_no_lp():
    """`test_wvmi`'s basic setup (FP 10, the qualifying candle 12): an end of 12 leaves nothing to search — the record
    is still created, with no LP and no pullback momentum (no fallback to the data end); an end of 13 finds 12."""
    df, zone, wcs, ev = tw.TestOnCtsConfirmed()._setup_basic()
    rec = WVMITracker(cycle_end_by_key={(0, 1): 12}).on_cts_confirmed(ev, df, wcs, [zone])
    assert (rec.lp_idx, rec.pullback_momentum, rec.sell_momentum, rec.status) == (None, None, None, "created")
    assert WVMITracker(cycle_end_by_key={(0, 1): 13}).on_cts_confirmed(ev, df, wcs, [zone]).lp_idx == 12


def _growing_frames():
    base = {"direction": 1, "c": 1.05, "volume": 100, "candle_type": "normal", "is_big_normal_as0": 0,
            "body_pct": 0.50, "vol_dir": 0}
    rows = [dict(base) for _ in range(15)]
    rows[3] = {**base, "volume": 200, "is_big_normal_as0": 1, "candle_type": "maru"}
    rows[8] = {**base, "volume": 300, "is_big_normal_as0": 1}
    rows[10] = {**base, "direction": -1, "volume": 150, "vol_dir": -1}
    rows[12] = {**base, "direction": -1, "volume": 120, "c": 0.95, "vol_dir": -1}
    grown = rows + [dict(base), dict(base), {**base, "direction": -1, "c": 0.91, "volume": 110, "vol_dir": -1}]
    return tw._make_df(rows), tw._make_df(grown)


def test_an_update_on_a_grown_frame_respects_the_bound():
    """A live-style caller: the frame grows by 3 candles after creation, the new candle 17 is nearer the outer. Past
    the cycle (end 17) it is ignored — no update, LP stays 12; inside it (end 18) it wins and the labels move too."""
    df0, df1 = _growing_frames()
    zone = tw._make_zone("buy", 1.0, 0.9, "BOS", {"structure_id": 0, "cycle_id": 1, "anchor_idx": 5, "outer": 0.9})
    wcs = [tw._make_wc(0, 1, "BOS", "buy", last_idx=2, first_idx=3), tw._make_wc(0, 1, "CTS", "sell", last_idx=8, first_idx=10)]
    ev = tw._make_event("CTS_CONFIRMED", 14, meta={"structure_id": 0, "cycle_id": 1})
    t = WVMITracker(cycle_end_by_key={(0, 1): 17})
    t.on_cts_confirmed(ev, df0, wcs, [zone])
    assert t.update_temporary_lp(df1, [zone]) == [] and t.get_records()[0].lp_idx == 12
    t = WVMITracker(cycle_end_by_key={(0, 1): 18})
    t.on_cts_confirmed(ev, df0, wcs, [zone])
    (rec,) = t.update_temporary_lp(df1, [zone])
    assert (rec.lp_idx, rec.status, rec.sell_momentum) == (17, "updated", round(110 * 0.7 / 150, 2))


def _project(res, floor=None, cap=None, reason=None):
    with contextlib.redirect_stdout(io.StringIO()):
        return project_to_window(res, floor=floor, cap=cap, cap_reason=reason, direction=1, fib_mode="h1")


# --- Plan G Q4 (2026-09-30): a sub projection's temp LP stops at the cycle's end - 1 too ------------------------

def test_an_open_projection_bounds_the_temp_lp_by_the_cycle_end():
    """Q4 end to end: the lure (a nearer qualifying candle 19 after the reversal at 17) is inside an OPEN projection's
    frame; the lifecycle table ends (0,2) at 17, so its temp LP stays 14 (pullback 1.0). The no-ends sub sweep before
    Plan G took 19 (0.7). (A capped projection cannot show it: under cap 17 the frame already stops at 17.)"""
    res, _ = _run(_post_reversal_lure())
    recs = {(r.bos_structure_id, r.bos_cycle_id): r for r in _project(res, floor=5)["wvmi_records"]}
    r = recs[(0, 2)]
    assert (r.fp_idx, r.lp_idx, r.lp_locked, r.pullback_momentum) == (13, 14, False, 1.0)


def test_the_projection_passes_every_cycle_end(monkeypatch):
    """Q4 wiring (the rewritten `test_the_sub_sweep_passes_no_ends` — a conscious change): the ends the projection
    hands the tracker ARE `compute_cycle_lifecycle`'s on the clipped events with the sub's floor / cap / reason."""
    seen = []

    class Capture(WVMITracker):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            seen.append(dict(self._cycle_end_by_key))

    res, _ = _run(_post_reversal_lure())
    monkeypatch.setattr(orch, "WVMITracker", Capture)
    down = _project(res, floor=5, cap=15, reason="parent_end")
    table = compute_cycle_lifecycle(down["events"], compute_reversal_idx_by_sid(down["events"]), 5, 15, "parent_end")
    assert seen == [{(0, 0): 8, (0, 1): 12, (0, 2): 15}]
    assert seen[0] == {k: e for k, (_s, e, _r) in table.items() if e is not None}
