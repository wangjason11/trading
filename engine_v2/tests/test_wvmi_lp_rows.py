"""The per-LP WVMI rows + `*_wvmi_lps.csv` ("WVMI back on the chart" step 4 — the user's rounds 3 / 8 / 10 / 11,
2026-10-26; `plans/qa_chart_edits_inputs/user_spec_20261024_wvmi_on_chart.md`; WVMI_SPEC "Per-LP rows").

Each record keeps every LP it has had (`WVMIRecord.lp_rows`, built by `WVMITracker._set_lp_rows` on every create /
re-scan / lock): each LP but the last is `temporary` (replaced by a further close at the next LP's as-of candle);
the last is `locked` (cycle N+1's BOS confirmed at the lock), or — the locked LP another candle — `temporary`
(replaced by the locked LP) + a `locked` row as of the lock; without a lock `open` (no cycle end) or `unlocked`
(the cycle's end + why). `export_wvmi_lps` writes one row per dot: collapsed records none; `relative_dir` = the sub's
at the LP candle (`NA (before start_idx)` before it is live). The sub mirror's shift of the three row indices is pinned
in `test_event_meta_idx_keys.py::test_mirror_shifts_every_site_exactly_once`.

The tracker setup (a buy zone: the pullback closes LOWER): FP 10 (close .99, vol 150); the pullback pattern [12, 14]
(CTS confirmed at 14) holds 12 (.95, vol 120) and 13 (.96 — not further); then 16 (.93, vol 90), 18 (.94 — not
further; BOS_2's last wave candle in the swap case, vol 180), 19 (.91, vol 60). So the temporary LPs are 12 as of 14,
16, 19 — every weight 0.7 (body 50 %, not big), pullback momentum = vol x 0.7 / 150.
"""
from __future__ import annotations

import contextlib
import io

import pandas as pd
import pytest

from engine_v2.common.types import WVMILPRow, WVMIRecord
from engine_v2.debug.export_wvmi import export_wvmi_lps
from engine_v2.multitf.types import SidRecord
from engine_v2.pipeline.orchestrator import _compute_wvmi_records
from engine_v2.structure import event_fields as ef
from engine_v2.tests import test_wvmi as tw   # module import: importing its Test classes would re-collect them
from engine_v2.zones.wvmi import WVMITracker

_FURTHER = "replaced by a further close"


def _pm(vol: float) -> float:
    return vol * 0.7 / 150


def _setup(*, bos2_last=19):
    rows = [{"direction": 1, "c": 1.05, "volume": 100, "vol_dir": 0} for _ in range(25)]
    rows[3] = {**rows[3], "volume": 200, "is_big_normal_as0": 1, "candle_type": "maru"}     # FB
    rows[8] = {**rows[8], "volume": 300, "is_big_normal_as0": 1}                           # LB
    rows[10] = {**rows[10], "direction": -1, "volume": 150, "vol_dir": -1, "c": 0.99}      # FP
    for i, c, v in ((12, 0.95, 120), (13, 0.96, 130), (16, 0.93, 90), (18, 0.94, 180), (19, 0.91, 60)):
        rows[i] = {**rows[i], "direction": -1, "volume": v, "c": c, "vol_dir": -1}
    df = tw._make_df(rows)
    zone = tw._make_zone("buy", 1.0, 0.9, "BOS", {"structure_id": 0, "cycle_id": 1, "anchor_idx": 5, "outer": 0.9})
    wcs = [tw._make_wc(0, 1, "BOS", "buy", last_idx=2, first_idx=3),
           tw._make_wc(0, 1, "CTS", "sell", last_idx=8, first_idx=10),
           tw._make_wc(0, 2, "BOS", "buy", last_idx=bos2_last, first_idx=20)]
    return df, wcs, [zone]


def _q(fn):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


def _rows(rec):
    return [(r.lp_idx, r.as_of_idx, r.lp_status, r.lp_status_reason, r.lp_status_idx, r.pullback_momentum)
            for r in rec.lp_rows]


def _check_last_is_the_record(rec):
    """The last row is the record's LP and pullback momentum (one formula, two places)."""
    assert rec.lp_rows[-1].lp_idx == rec.lp_idx
    assert rec.lp_rows[-1].pullback_momentum == pytest.approx(rec.pullback_momentum)


def _create(df, wcs, zones, *, ends=None, reasons=None):
    tracker = WVMITracker(structure_path_id="x", cycle_end_by_key=ends, cycle_end_reason_by_key=reasons)
    rec = _q(lambda: tracker.on_cts_confirmed(tw._cts_confirmed(), df, wcs, zones))
    return tracker, rec


def _lock(tracker, df, wcs):
    bos2 = tw._make_event("BOS_CONFIRMED", 21, meta={"structure_id": 0, "cycle_id": 2})
    (rec,) = _q(lambda: tracker.on_bos_confirmed(bos2, df, wcs))
    return rec


# --- the tracker's rows -------------------------------------------------------------------------------------------

def test_an_open_record_temporary_rows_then_open():
    """No cycle end (the data end): each LP but the last replaced by a further close at the NEXT LP's as-of candle;
    the first LP 12 is as of the CTS moment 14; the last is open with no status candle."""
    df, wcs, zones = _setup()
    _, rec = _create(df, wcs, zones)
    assert _rows(rec) == [
        (12, 14, "temporary", _FURTHER, 16, pytest.approx(_pm(120))),
        (16, 16, "temporary", _FURTHER, 19, pytest.approx(_pm(90))),
        (19, 19, "open", "still moving at the data end", None, pytest.approx(_pm(60))),
    ]
    _check_last_is_the_record(rec)


@pytest.mark.parametrize("reason, text", [
    ("reversal", "cycle ended by a reversal"),
    ("same_dir_replacement", "sub replaced"),
    ("parent_end", "sub ended"),
    ("proximity_view", "cycle ended (proximity_view)"),     # any other reason prints raw
])
def test_an_unlocked_record_ends_at_its_cycle_end_with_why(reason, text):
    df, wcs, zones = _setup()
    _, rec = _create(df, wcs, zones, ends={(0, 1): 21}, reasons={(0, 1): reason})
    assert _rows(rec)[-1] == (19, 19, "unlocked", text, 21, pytest.approx(_pm(60)))
    assert [r.lp_status for r in rec.lp_rows] == ["temporary", "temporary", "unlocked"]


def test_the_rows_stop_at_the_cycles_end_minus_one():
    """End 19: the search stops at 18, so 19 is no LP — the rows end on 16, unlocked at 19."""
    df, wcs, zones = _setup()
    _, rec = _create(df, wcs, zones, ends={(0, 1): 19}, reasons={(0, 1): "reversal"})
    assert _rows(rec) == [
        (12, 14, "temporary", _FURTHER, 16, pytest.approx(_pm(120))),
        (16, 16, "unlocked", "cycle ended by a reversal", 19, pytest.approx(_pm(90))),
    ]
    _check_last_is_the_record(rec)


def test_a_lock_on_the_last_temporary_lp_makes_it_the_locked_row():
    """BOS_2 (moment 21) locks on its last wave candle 19 = the last temporary LP: that row becomes `locked` (as of
    its own candle; the lock is the status candle) — no extra row."""
    df, wcs, zones = _setup(bos2_last=19)
    tracker, _ = _create(df, wcs, zones, ends={(0, 1): 21}, reasons={(0, 1): "next_cycle"})
    rec = _lock(tracker, df, wcs)
    assert _rows(rec) == [
        (12, 14, "temporary", _FURTHER, 16, pytest.approx(_pm(120))),
        (16, 16, "temporary", _FURTHER, 19, pytest.approx(_pm(90))),
        (19, 19, "locked", "cycle 2's BOS confirmed", 21, pytest.approx(_pm(60))),
    ]
    _check_last_is_the_record(rec)


def test_the_lock_is_dated_by_the_bos_moment_not_the_cycle_end():
    """`cycle N+1's BOS confirmed at <idx>` = the BOS_{n+1} moment (21). The cycle's lifecycle end equals it on a
    visible record, but a COLLAPSED record's end is the next cycle's CLAMPED start (raised to the floor), past the
    moment — here 23 (the rows stay on the record; only the CSV drops them)."""
    df, wcs, zones = _setup(bos2_last=19)
    tracker, _ = _create(df, wcs, zones, ends={(0, 1): 23}, reasons={(0, 1): "next_cycle"})
    rec = _lock(tracker, df, wcs)
    assert _rows(rec)[-1][:5] == (19, 19, "locked", "cycle 2's BOS confirmed", 21)


def test_a_lock_on_another_candle_replaces_the_last_temporary_lp():
    """BOS_2's last wave candle 18 is not the last temporary LP 19 (the warning case): 19 stays `temporary`, replaced
    by the locked LP at the lock 21, and the locked LP 18 gets its own row as of the lock."""
    df, wcs, zones = _setup(bos2_last=18)
    tracker, _ = _create(df, wcs, zones, ends={(0, 1): 21}, reasons={(0, 1): "next_cycle"})
    rec = _lock(tracker, df, wcs)
    assert _rows(rec) == [
        (12, 14, "temporary", _FURTHER, 16, pytest.approx(_pm(120))),
        (16, 16, "temporary", _FURTHER, 19, pytest.approx(_pm(90))),
        (19, 19, "temporary", "replaced by the locked LP", 21, pytest.approx(_pm(60))),
        (18, 21, "locked", "cycle 2's BOS confirmed", 21, pytest.approx(_pm(180))),
    ]
    _check_last_is_the_record(rec)


@pytest.mark.parametrize("bos2_last", [None, 30], ids=["no_lock_lp", "lock_lp_past_the_frame"])
def test_a_lock_that_keeps_the_temporary_lp_locks_the_last_row(bos2_last):
    df, wcs, zones = _setup(bos2_last=bos2_last)
    tracker, _ = _create(df, wcs, zones, ends={(0, 1): 21}, reasons={(0, 1): "next_cycle"})
    rec = _lock(tracker, df, wcs)
    assert [r[:5] for r in _rows(rec)] == [(12, 14, "temporary", _FURTHER, 16), (16, 16, "temporary", _FURTHER, 19),
                                          (19, 19, "locked", "cycle 2's BOS confirmed", 21)]
    _check_last_is_the_record(rec)


def test_a_lock_before_any_lp_is_the_only_row():
    """End 12: no candidate up to 11 (the pattern's 12 is past the bound), so no temporary LP; the lock on 18 is the
    one row, as of the lock."""
    df, wcs, zones = _setup(bos2_last=18)
    tracker, rec = _create(df, wcs, zones, ends={(0, 1): 12}, reasons={(0, 1): "next_cycle"})
    assert rec.lp_idx is None and rec.lp_rows == []                # no LP, no rows
    rec = _lock(tracker, df, wcs)
    assert _rows(rec) == [(18, 21, "locked", "cycle 2's BOS confirmed", 21, pytest.approx(_pm(180)))]
    _check_last_is_the_record(rec)


def test_no_lp_and_no_lock_lp_no_rows():
    df, wcs, zones = _setup(bos2_last=None)
    tracker, _ = _create(df, wcs, zones, ends={(0, 1): 12}, reasons={(0, 1): "next_cycle"})
    rec = _lock(tracker, df, wcs)
    assert rec.lp_idx is None and rec.lp_rows == []


def test_the_rescan_rebuilds_the_rows():
    """Created on the candles up to 16 (LPs 12, 16 — open), re-scanned on the full frame: 16 is replaced by 19."""
    df, wcs, zones = _setup()
    tracker, rec = _create(df.iloc[:17], wcs, zones)
    assert [r[:5] for r in _rows(rec)] == [(12, 14, "temporary", _FURTHER, 16),
                                          (16, 16, "open", "still moving at the data end", None)]
    _q(lambda: tracker.update_temporary_lp(df))
    assert [r[:5] for r in _rows(rec)] == [(12, 14, "temporary", _FURTHER, 16), (16, 16, "temporary", _FURTHER, 19),
                                          (19, 19, "open", "still moving at the data end", None)]
    _check_last_is_the_record(rec)


def test_a_lock_after_a_rescan_reads_the_rescans_sequence():
    """The lock rebuilds the rows from the LATEST sequence: created up to 16, re-scanned on the full frame (LP 19),
    then BOS_2 locks on 19 — the last temporary LP, so no swap."""
    df, wcs, zones = _setup(bos2_last=19)
    tracker, _ = _create(df.iloc[:17], wcs, zones)
    _q(lambda: tracker.update_temporary_lp(df))
    rec = _lock(tracker, df, wcs)
    assert [r[:5] for r in _rows(rec)] == [(12, 14, "temporary", _FURTHER, 16), (16, 16, "temporary", _FURTHER, 19),
                                          (19, 19, "locked", "cycle 2's BOS confirmed", 21)]


def test_the_tracker_loop_reads_the_lifecycle_end_and_its_reason():
    """`_compute_wvmi_records` hands the tracker the lifecycle table's end AND end_reason: a sub capped at 21 by a
    replacement ends its record `unlocked (sub replaced at 21)`."""
    df, wcs, zones = _setup()
    events = [tw._make_event("CTS_ESTABLISHED", 9, meta={"structure_id": 0, "cycle_id": 1}), tw._cts_confirmed()]
    (rec,) = _q(lambda: _compute_wvmi_records(
        df, events, sorted(events, key=ef.processing_order_key), wcs, zones, structure_path_id="x",
        lifecycle_floor=None, lifecycle_cap=21, cap_reason="same_dir_replacement"))
    assert [r[:5] for r in _rows(rec)][-1] == (19, 19, "unlocked", "sub replaced", 21)
    assert rec.cycle_collapsed is False


# --- the CSV --------------------------------------------------------------------------------------------------------

_SUB_PATH = "H1.main >> M15.counter"


def _row(lp, as_of, status="temporary", reason=_FURTHER, status_idx=None, pm=0.5):
    return WVMILPRow(lp_idx=lp, as_of_idx=as_of, lp_status=status, lp_status_reason=reason, lp_status_idx=status_idx,
                     pullback_momentum=pm)


def _sub_rec(cycle, rows, *, collapsed=False, sub_id=5):
    return WVMIRecord(bos_structure_id=0, bos_cycle_id=cycle, zone_side="sell", structure_path_id=_SUB_PATH,
                      breakout_momentum=1.25, cycle_collapsed=collapsed, lp_rows=rows,
                      meta={"parent_sid": 1, "parent_cycle_id": 2, "sub_id": sub_id})


def _sid(sub_id=5, start=100, segs=((100, "counter"), (105, "confluence"))):
    return SidRecord(sub_sid=None, starting_sd=-1, creation_event_idx=90, end_event_idx=None,
                     end_reason=None, sub_id=sub_id, start_idx=start, relative_dir_segments=segs)


def _export(tmp_path, records, sids=()):
    path = tmp_path / "lps.csv"
    export_wvmi_lps(records, path, sid_records=sids)
    return path, pd.read_csv(path)


def test_the_columns_in_the_users_order(tmp_path):
    _, out = _export(tmp_path, [])
    assert list(out.columns) == [
        "structure_path_id", "parent_sid", "parent_cycle_id", "sub_id", "bos_structure_id", "bos_cycle_id",
        "relative_dir", "zone_side", "lp_idx", "as_of_idx", "lp_status", "lp_status_reason", "lp_status_idx",
        "breakout_momentum", "pullback_momentum"]
    assert out.empty


def test_one_row_per_dot_collapsed_records_none(tmp_path):
    """Record order, then LP order; the collapsed record (cycle 1) has rows on the record but none in the CSV."""
    recs = [_sub_rec(0, [_row(101, 102, status_idx=103), _row(103, 103, "open", "still moving at the data end")]),
            _sub_rec(1, [_row(104, 104, "unlocked", "sub replaced", 110)], collapsed=True),
            _sub_rec(2, [_row(106, 107, "locked", "cycle 3's BOS confirmed", 108, pm=0.75)])]
    _, out = _export(tmp_path, recs, [_sid()])
    assert list(zip(out.bos_cycle_id, out.lp_idx, out.as_of_idx, out.lp_status)) == [
        (0, 101, 102, "temporary"), (0, 103, 103, "open"), (2, 106, 107, "locked")]
    assert list(out.pullback_momentum) == [0.5, 0.5, 0.75]
    assert set(out.breakout_momentum) == {1.25}
    assert list(out.lp_status_reason) == [_FURTHER, "still moving at the data end", "cycle 3's BOS confirmed"]


def test_relative_dir_is_the_subs_at_the_lp_candle(tmp_path):
    """The step function at the LP candle (not its as-of candle): 99 is before the sub's start 100 -> `NA (before
    start_idx)` (the user, round 11); 104 (as of 105) -> counter; 105 -> confluence."""
    recs = [_sub_rec(0, [_row(99, 101, status_idx=104), _row(104, 105, status_idx=105),
                         _row(105, 105, "open", "still moving at the data end")])]
    _, out = _export(tmp_path, recs, [_sid(sub_id=9, start=50, segs=((50, "confluence"),)), _sid()])
    assert list(out.relative_dir) == ["NA (before start_idx)", "counter", "confluence"]
    assert list(out.sub_id) == [5, 5, 5]


def test_an_h1_row_has_no_relative_dir_and_the_ints_stay_ints(tmp_path):
    rec = WVMIRecord(bos_structure_id=1, bos_cycle_id=2, zone_side="buy", structure_path_id="H1.main",
                     breakout_momentum=0.9, lp_rows=[_row(935, 935, status_idx=940), _row(940, 940, "open",
                                                                                         "still moving at the data end")],
                     meta={"triggered_by_event_idx": 935})
    path, out = _export(tmp_path, [rec])
    assert out.relative_dir.isna().all() and out.sub_id.isna().all() and out.parent_sid.isna().all()
    lines = path.read_text().splitlines()
    assert lines[1] == f"H1.main,,,,1,2,,buy,935,935,temporary,{_FURTHER},940,0.9,0.5"
    assert lines[2] == "H1.main,,,,1,2,,buy,940,940,open,still moving at the data end,,0.9,0.5"


def test_a_sub_row_without_its_sid_record_fails_loudly(tmp_path):
    with pytest.raises(KeyError):
        export_wvmi_lps([_sub_rec(0, [_row(101, 101)])], tmp_path / "x.csv", sid_records=[_sid(sub_id=6)])
