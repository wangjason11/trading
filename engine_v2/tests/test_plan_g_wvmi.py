"""Plan G — WVMI on the unique sub (`plans/PLAN_G_wvmi_unique_sub.md`, 2026-09-30).

G1: ONE tracker helper (`orchestrator._compute_wvmi_records`) for the main (gated on its first sd zone-proximity
trigger) and every sub projection (`wvmi="none"`, ungated, inside `project_to_window`); the tracker frame is
`df.iloc[:cap + 1]` under a cap; the temp LP stops at the cycle's `end - 1` (Q4); a lock LP outside the frame falls
back to the temp LP (Q10). G2: `WVMIRecord.cycle_collapsed` from the same `compute_cycle_lifecycle` table (main AND
subs). G3: the mirror persists one copy per lens (the FIELD `structure_path_id` = the lens path); the exporter writes
`cycle_collapsed` after `lp_locked` and `triggered_by_event_idx` as nullable `Int64`. The per-lens trigger post-pass
(G4) is pinned in `test_sub_wvmi_per_sub.py`.

Replaces `test_sub_wvmi.py` (the deleted `multitf/sub_wvmi.py`): its create / lock / unlocked / no-CTS cases are
ported to the helper below. Every expected value was measured with the Plan G code (the checklist's values,
`plans/plan_g_inputs/cold_review_20260930.md`, re-measured on the capped frame).
"""
from __future__ import annotations

import contextlib
import copy
import csv
import importlib.util
import io
from types import SimpleNamespace

import pandas as pd
import pytest

import engine_v2.pipeline.orchestrator as orch
from engine_v2.charting import _zone_render
from engine_v2.common.types import WVMIRecord
from engine_v2.debug.export_wvmi import export_wvmi
from engine_v2.multitf import entity_df_mutation as edm
from engine_v2.multitf.pooled_structure_build import project_to_window
from engine_v2.pipeline.orchestrator import _compute_wvmi_records, _run_downstream_pipeline
from engine_v2.structure.structure_engine import compute_bounded_structure
from engine_v2.tests import test_wvmi as tw   # module import: importing its Test classes would re-collect them
from engine_v2.tests.test_ms_stop_after_cts import _make_double_rewind_data
from engine_v2.tests.test_unified_probe import _prepare_df
from engine_v2.zones.structure_lifecycle import compute_cycle_lifecycle, compute_reversal_idx_by_sid
from engine_v2.zones.wvmi import WVMITracker

_SUB = "H1.main >> M15.confluence"


def _q(fn):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


def _vals(records):
    return [((r.bos_structure_id, r.bos_cycle_id), (r.fb_idx, r.lb_idx, r.fp_idx, r.lp_idx), r.status,
             r.locked_by_cycle_id) for r in records]


# --- the ported `test_sub_wvmi` cases: the helper, ungated --------------------------------------------------------

def _basic(n=25, *, bos2=True, fp=10, bos2_last=18):
    """`test_sub_wvmi`'s fixture + the CTS_ESTABLISHED events the lifecycle table needs: cycle 1 established at 9,
    confirmed at 14 (FB 3 vol 200, LB 8 vol 300 weight 1.0, FP `fp` vol 150; temp-LP candidate 12 close .92 vol 120);
    cycle 2 established + BOS_CONFIRMED at 17, its last wave candle `bos2_last` (18: close .93 vol 180)."""
    rows = [{"direction": 1, "c": 1.05, "volume": 100, "vol_dir": 0} for _ in range(n)]
    rows[3] = {**rows[3], "volume": 200, "is_big_normal_as0": 1, "candle_type": "maru"}
    rows[8] = {**rows[8], "volume": 300, "is_big_normal_as0": 1, "candle_type": "normal"}
    rows[fp] = {**rows[fp], "direction": -1, "volume": 150, "vol_dir": -1}
    rows[12] = {**rows[12], "direction": -1, "volume": 120, "c": 0.92, "vol_dir": -1}
    if 18 < n:
        rows[18] = {**rows[18], "direction": -1, "volume": 180, "c": 0.93, "vol_dir": -1}
    df = tw._make_df(rows)
    zone = tw._make_zone("buy", 1.0, 0.9, "BOS", {"structure_id": 0, "cycle_id": 1, "anchor_idx": 5, "outer": 0.9})
    wcs = [tw._make_wc(0, 1, "BOS", "buy", last_idx=2, first_idx=3),
           tw._make_wc(0, 1, "CTS", "sell", last_idx=8, first_idx=fp),
           tw._make_wc(0, 2, "BOS", "buy", last_idx=bos2_last, first_idx=19)]
    m1, m2 = {"structure_id": 0, "cycle_id": 1}, {"structure_id": 0, "cycle_id": 2}
    events = [tw._make_event("CTS_ESTABLISHED", 9, meta=m1), tw._make_event("CTS_CONFIRMED", 14, meta=m1)]
    if bos2:
        events += [tw._make_event("CTS_ESTABLISHED", 17, meta=m2), tw._make_event("BOS_CONFIRMED", 17, meta=m2)]
    return df, events, wcs, [zone]


def _helper(df, events, wcs, zones, *, gate=None, floor=None, cap=None, reason="lifecycle_end"):
    from engine_v2.structure import event_fields as ef
    return _q(lambda: _compute_wvmi_records(
        df, events, sorted(events, key=ef.processing_order_key), wcs, zones, gate=gate, structure_path_id=_SUB,
        lifecycle_floor=floor, lifecycle_cap=cap, cap_reason=reason))


def test_ungated_creates_and_locks_every_confirmed_cycle():
    """No gate: the CTS_CONFIRMED makes a record with meta `{}` and the tracker's path; BOS_2 locks it on its last
    wave candle 18 (180 * 0.7 / 150). Breakout = 300 * 1.0 / 200."""
    (rec,) = _helper(*_basic())
    assert rec.meta == {} and rec.structure_path_id == _SUB
    assert _vals([rec]) == [((0, 1), (3, 8, 10, 18), "locked", 2)]
    assert (rec.breakout_momentum, rec.lp_volume, rec.pullback_momentum) == (1.5, 180.0, 180 * 0.7 / 150)
    assert rec.cycle_collapsed is False


def test_no_cts_confirmed_no_record():
    df, events, wcs, zones = _basic()
    assert _helper(df, [e for e in events if e.type != "CTS_CONFIRMED"], wcs, zones) == []


def test_unlocked_open_cycle_keeps_its_temp_lp_and_is_not_collapsed():
    """No BOS_2 / CTS_ESTABLISHED 2: cycle 1 is OPEN (end None) — temp LP 12 (120 * 0.7 / 150), not locked, and NOT
    collapsed (an open cycle has no empty window; the G2e mutant flags it)."""
    df, events, wcs, zones = _basic(bos2=False)
    life = compute_cycle_lifecycle(events, compute_reversal_idx_by_sid(events))
    assert life == {(0, 1): (9, None, None)}
    (rec,) = _helper(df, events, wcs, zones)
    assert _vals([rec]) == [((0, 1), (3, 8, 10, 12), "created", None)]
    assert rec.pullback_momentum == 120 * 0.7 / 150 and rec.cycle_collapsed is False


def test_the_gate_keeps_only_gated_cycles_and_writes_their_meta_first():
    df, events, wcs, zones = _basic()
    assert _helper(df, events, wcs, zones, gate={(0, 2): {"x": 1}}) == []
    meta = {"triggered_by_event_idx": 13, "triggered_by_event_type": "ZONE_PROXIMITY_TRIGGER"}
    (rec,) = _helper(df, events, wcs, zones, gate={(0, 1): meta})
    assert rec.meta == meta and rec.meta is not meta


# --- G1: the frame under a cap, the capped-record assert, Q10 -------------------------------------------------------

def test_a_capped_tracker_never_reads_past_the_cap():
    """G1 frame: under cap 15 the tracker frame is `df.iloc[:16]` — an FP at 16 (past the cap; reachable when a CTS
    confirmed at/before the cap has its first pullback after it) cannot be read, so no record. The natural-end frame
    (the `bounded.df` a projection passes) would make one from a candle past the sub's end."""
    df, events, wcs, zones = _basic(fp=16)
    events = [e for e in events if e.meta["cycle_id"] == 1]
    assert _helper(df, events, wcs, zones, cap=15) == []
    (rec,) = _helper(df, events, wcs, zones)                   # the same inputs uncapped: FP 16 is read
    assert rec.fp_idx == 16
    # the cap candle itself is inside the frame (`iloc[:cap + 1]`): an FP ON it is read; the LP search [16, 14]
    # (the cycle ends at the cap 15) is empty
    df, events, wcs, zones = _basic(fp=15)
    (rec,) = _helper(df, [e for e in events if e.meta["cycle_id"] == 1], wcs, zones, cap=15)
    assert (rec.fp_idx, rec.lp_idx, rec.pullback_momentum) == (15, None, None)


@pytest.mark.parametrize("cap", [20, None], ids=["capped", "uncapped"])
def test_a_record_without_a_lifecycle_row_fails_loudly(cap):
    """Unreachable from MS output (a record needs its cycle's CTS wave candle, so its CTS_ESTABLISHED; the clip keeps
    it with its CTS_CONFIRMED) — the helper asserts, capped or not, instead of exporting a record with no bound / no
    flag (landing review NIT 1: the uncapped path used to raise a bare KeyError). Under a cap a row always has an end
    (the table's cap term), so no separate "unbounded" assert exists."""
    df, events, wcs, zones = _basic(bos2=False)
    events = [e for e in events if e.type != "CTS_ESTABLISHED"]
    with pytest.raises(AssertionError, match="no lifecycle row"):
        _helper(df, events, wcs, zones, cap=cap)


def test_a_lock_lp_outside_the_frame_falls_back_to_the_temp_lp():
    """Q10: BOS_2 (at 17 <= cap 19) locks cycle 1, but its last wave candle 21 is past the cap — the record locks
    with its temp LP 12 (bounded by cycle 1's end 17), never an LP the frame cannot read (before Plan G: idx 21 with
    volume / pullback None)."""
    df, events, wcs, zones = _basic(bos2_last=21)
    (rec,) = _helper(df, events, wcs, zones, cap=19, reason="parent_end")
    assert _vals([rec]) == [((0, 1), (3, 8, 10, 12), "locked", 2)]
    assert (rec.lp_volume, rec.pullback_momentum, rec.lp_locked) == (120.0, 120 * 0.7 / 150, True)


def test_a_lock_lp_on_the_cap_candle_is_used():
    """Q10's boundary (landing review N24): BOS_2's last wave candle ON the cap candle 19 is INSIDE the frame
    (`iloc[:cap + 1]`) — the lock takes it (100 * 0.7 / 150), no fallback."""
    df, events, wcs, zones = _basic(bos2_last=19)
    (rec,) = _helper(df, events, wcs, zones, cap=19, reason="parent_end")
    assert _vals([rec]) == [((0, 1), (3, 8, 10, 19), "locked", 2)]
    assert (rec.lp_volume, rec.lp_weight, rec.pullback_momentum) == (100.0, 0.7, 100 * 0.7 / 150)


def test_the_tracker_falls_back_on_a_lock_lp_past_the_data_edge():
    """Q10 on the tracker itself (the main's case: a data edge instead of a cap)."""
    df, events, wcs, zones = _basic(n=20, bos2_last=21)
    t = WVMITracker()
    t.on_cts_confirmed(events[1], df, wcs, zones)
    (rec,) = t.on_bos_confirmed(events[3], df, wcs)
    assert (rec.lp_idx, rec.lp_volume, rec.pullback_momentum, rec.status) == (12, 120.0, 120 * 0.7 / 150, "locked")


# --- G1 on real MS output: the sub projection, the main, the gate, the mode -----------------------------------------

@pytest.fixture(scope="module")
def dr():
    """`_make_double_rewind_data` (sd +1): cycles established at 2 / 8 / 12, CTS_CONFIRMED at 4 / 9 / 14, sid 0
    reverses at 17; lifecycle {(0,0): [2, 8), (0,1): [8, 12), (0,2): [12, 17)}."""
    return _q(lambda: compute_bounded_structure(_prepare_df(_make_double_rewind_data()), 0, 1))


def _project(res, floor=None, cap=None, reason=None):
    return _q(lambda: project_to_window(res, floor=floor, cap=cap, cap_reason=reason, direction=1, fib_mode="h1"))


_DR_VALUES = [((0, 0), (1, 1, 3, 4), "locked", 1), ((0, 1), (5, 7, 9, 9), "locked", 2),
              ((0, 2), (10, 11, 13, 14), "created", None)]


def test_a_sub_projection_computes_its_own_ungated_wvmi(dr, monkeypatch):
    """G1: `project_to_window` passes `wvmi="none"` — every CTS_CONFIRMED gets a record, meta `{}`, the call's path;
    zone proximity is never run for a sub (stubbed to raise). The main on the same data with proximity stubbed to
    `{}` gets NONE (its gate): the default mode is the main's gate, not "none" (G1a), and a projection that passed
    the main's mode would get none either (G1b)."""
    def _boom(**_kw):
        raise AssertionError("check_zone_proximity ran for a sub")
    monkeypatch.setattr(orch, "check_zone_proximity", _boom)
    down = _project(dr)
    assert _vals(down["wvmi_records"]) == _DR_VALUES
    assert {r.structure_path_id for r in down["wvmi_records"]} == {"H1.main >> M15"}
    assert [r.meta for r in down["wvmi_records"]] == [{}, {}, {}]
    assert down["zone_proximity_triggers"] == {}
    monkeypatch.setattr(orch, "check_zone_proximity", lambda **_kw: {})
    main = _q(lambda: _run_downstream_pipeline(dr.df, copy.deepcopy(dr.events), 1, fib_mode="h1"))
    assert main["wvmi_records"] == []


def test_the_main_keeps_its_gate_values_and_meta_key_order(dr):
    out = _q(lambda: _run_downstream_pipeline(dr.df, copy.deepcopy(dr.events), 1, fib_mode="h1"))
    assert _vals(out["wvmi_records"]) == _DR_VALUES
    assert [list(r.meta.items()) for r in out["wvmi_records"]] == [
        [("triggered_by_event_idx", t), ("triggered_by_event_type", "ZONE_PROXIMITY_TRIGGER"),
         ("structure_path_id", "H1.main"), ("trigger_inner", inner), ("proximity_pips", 8)]
        for t, inner in ((4, 0.602), (9, 0.599), (14, 0.601))]
    assert {r.structure_path_id for r in out["wvmi_records"]} == {"H1.main"}


def test_the_main_gate_reads_only_the_first_trigger_of_a_cycle(dr, monkeypatch):
    """A cycle whose FIRST zone-proximity trigger is opp_sd gets no main record, even with an sd trigger after it
    (the M1 mutant "any sd trigger")."""
    trig = lambda i, d: SimpleNamespace(idx=i, direction=d, trigger_inner=0.6, proximity_pips=8)   # noqa: E731
    monkeypatch.setattr(orch, "check_zone_proximity", lambda **_kw: {
        (0, 0): [trig(4, "opp_sd"), trig(6, "sd")], (0, 1): [trig(9, "sd")], (0, 2): [trig(14, "sd")]})
    out = _q(lambda: _run_downstream_pipeline(dr.df, copy.deepcopy(dr.events), 1, fib_mode="h1"))
    assert [(r.bos_structure_id, r.bos_cycle_id) for r in out["wvmi_records"]] == [(0, 1), (0, 2)]


def test_the_main_gate_meta_is_the_first_triggers_even_with_a_later_sd(dr, monkeypatch):
    """(landing review N6) sd @4 (inner .61, 7 pips), opp_sd @5, sd @6 (.55, 3): the record's attribution is the
    FIRST trigger's — UC1 copies `triggered_by_event_idx` into the first_counter trigger (its sweep `trigger_idx`)."""
    trig = lambda i, d, inner, pips: SimpleNamespace(idx=i, direction=d, trigger_inner=inner,  # noqa: E731
                                                     proximity_pips=pips)
    monkeypatch.setattr(orch, "check_zone_proximity", lambda **_kw: {
        (0, 0): [trig(4, "sd", 0.61, 7), trig(5, "opp_sd", 0.5, 5), trig(6, "sd", 0.55, 3)],
        (0, 1): [trig(9, "sd", 0.6, 8)], (0, 2): [trig(14, "sd", 0.6, 8)]})
    out = _q(lambda: _run_downstream_pipeline(dr.df, copy.deepcopy(dr.events), 1, fib_mode="h1"))
    rec = out["wvmi_records"][0]
    assert (rec.bos_structure_id, rec.bos_cycle_id) == (0, 0)
    assert (rec.meta["triggered_by_event_idx"], rec.meta["trigger_inner"], rec.meta["proximity_pips"]) == (4, 0.61, 7)


def test_record_order_follows_the_processing_order_not_the_events_list(dr):
    """(landing review N7) the helper walks `sorted_events` (`ef.processing_order_key`): handed the RAW events
    reversed, the records still come in cycle order with the same values."""
    from engine_v2.structure import event_fields as ef
    down = _project(dr)
    events = list(down["events"])
    recs = _q(lambda: _compute_wvmi_records(
        dr.df, list(reversed(events)), sorted(events, key=ef.processing_order_key), down["wave_candles"],
        down["kl_zones"], gate=None, structure_path_id="x", lifecycle_floor=None, lifecycle_cap=None,
        cap_reason="lifecycle_end"))
    assert [(r.bos_cycle_id, r.lp_idx, r.status) for r in recs] == [(0, 4, "locked"), (1, 9, "locked"),
                                                                     (2, 14, "created")]


@pytest.mark.parametrize("mode", ["", "None", "gated", "first_sd", True])
def test_an_unknown_wvmi_mode_raises(dr, mode):
    with pytest.raises(ValueError, match="wvmi="):
        _q(lambda: _run_downstream_pipeline(dr.df, copy.deepcopy(dr.events), 1, wvmi=mode))


def test_off_computes_nothing(dr, monkeypatch):
    monkeypatch.setattr(orch, "check_zone_proximity", lambda **_kw: (_ for _ in ()).throw(AssertionError))
    out = _q(lambda: _run_downstream_pipeline(dr.df, copy.deepcopy(dr.events), 1, wvmi="off"))
    assert out["wvmi_records"] == [] and out["zone_proximity_triggers"] == {}


# --- G2: cycle_collapsed ------------------------------------------------------------------------------------------

def _flags(records):
    return [r.cycle_collapsed for r in records]


@pytest.mark.parametrize("floor, expected", [(None, [False] * 3), (9, [True, False, False]),
                                             (13, [True, True, False])])
def test_sub_records_flag_their_collapsed_cycles_like_the_kl_zones(dr, floor, expected):
    """A floor past a cycle's end empties its window (`start >= end`): floor 9 collapses (0,0) = [2, 8) -> [9, 8);
    floor 13 also (0,1) = [8, 12) -> [13, 12); (0,2) = [13, 17) stays live. The flags equal the KL rendering's
    `collapsed_cycles` (G2: `start >= end` on the same table)."""
    down = _project(dr, floor=floor)
    assert _vals(down["wvmi_records"]) == _DR_VALUES      # values unchanged: the flag is attribution only
    assert _flags(down["wvmi_records"]) == expected
    assert {(r.bos_structure_id, r.bos_cycle_id) for r in down["wvmi_records"] if r.cycle_collapsed} == \
        _zone_render.collapsed_cycles(down["kl_zones"])


@pytest.mark.parametrize("floor_abs, expected", [(90, False), (107, True), (108, True)])
def test_collapse_is_start_at_or_past_the_end(floor_abs, expected):
    """The render fixture's one cycle (0,0) ends at the natural reversal 107 (abs; established 57): floor 90 -> [90,
    107) live; floor 107 -> [107, 107) empty (the `>` mutant misses it); floor 108 -> [108, 107) (the `==` mutant
    misses it)."""
    from engine_v2.tests import test_render_sub_projection as trs
    from engine_v2.multitf.sub_structure_pool import SubStructurePool
    m15 = trs._prepare_df(trs._make_flat_prefix() + trs._make_reversing_data())
    sub, _ = _q(lambda: edm.build_or_get_geometry(SubStructurePool(), m15, parent_path=trs._PARENT, sd=1,
                                                  start_abs=trs._STARTING_IDX, bos0_inner=None, timeframe=trs._TF))
    bounded, sb = sub.geometry
    down = _q(lambda: project_to_window(bounded, floor=floor_abs - sb, cap=None, cap_reason=None, direction=1))
    (rec,) = down["wvmi_records"]
    assert (rec.bos_structure_id, rec.bos_cycle_id, rec.cycle_collapsed) == (0, 0, expected)
    assert ({(0, 0)} if expected else set()) == _zone_render.collapsed_cycles(down["kl_zones"])


@pytest.mark.parametrize("floor, expected", [(None, [False] * 3), (13, [True, True, False])])
def test_the_main_flags_its_collapsed_cycles_too(dr, floor, expected):
    out = _q(lambda: _run_downstream_pipeline(dr.df, copy.deepcopy(dr.events), 1, fib_mode="h1",
                                              lifecycle_floor=floor))
    assert _vals(out["wvmi_records"]) == _DR_VALUES and _flags(out["wvmi_records"]) == expected


def test_a_record_is_not_collapsed_by_default():
    assert WVMIRecord(bos_structure_id=0, bos_cycle_id=0, zone_side="buy").cycle_collapsed is False


# --- G3: one copy per lens, no second persister -----------------------------------------------------------------------

def test_render_hands_over_the_projection_records_in_order(monkeypatch):
    """(landing review N18 / N17) `render_sub_projection` keeps the projection's record order on the result AND on
    the lens copies (the real render fixtures hold 0 or 1 record, so no order check on them could fail)."""
    from engine_v2.multitf import pooled_structure_build as psb
    from engine_v2.multitf.sub_structure_pool import LENS_CONFLUENCE, SubStructurePool
    from engine_v2.tests import test_render_sub_projection as trs
    m15 = trs._prepare_df(trs._make_flat_prefix() + trs._make_reversing_data())
    pool = SubStructurePool()
    sub, _ = _q(lambda: edm.build_or_get_geometry(pool, m15, parent_path=trs._PARENT, sd=1,
                                                  start_abs=trs._STARTING_IDX, bos0_inner=None, timeframe=trs._TF))
    trs._record(pool, sub, m15, LENS_CONFLUENCE, start_idx=trs._START, seq=0, trigger_type="first_confluence",
                parent_sid=3, parent_cycle_id=1)
    trs._set_lifecycle(sub, trs._START, None, None)
    real = psb.project_to_window

    def _three(*a, **k):
        down = real(*a, **k)
        down["wvmi_records"] = [WVMIRecord(bos_structure_id=0, bos_cycle_id=c, zone_side="buy", fb_idx=c)
                                for c in (0, 1, 2)]
        return down

    monkeypatch.setattr(psb, "project_to_window", _three)
    lens_dfs = trs._lens_dfs(m15)
    res = _q(lambda: edm.render_sub_projection(sub, m15, lens_paths=trs._LENS_PATHS, lens_dfs=lens_dfs,
                                               timeframe=trs._TF))
    assert [w.bos_cycle_id for w in res.wvmi_records] == [0, 1, 2]
    assert [w.bos_cycle_id for w in lens_dfs[LENS_CONFLUENCE].attrs["wvmi"]] == [0, 1, 2]


def test_the_sub_wvmi_module_and_the_persister_are_gone():
    assert importlib.util.find_spec("engine_v2.multitf.sub_wvmi") is None
    assert not hasattr(edm, "persist_facade_wvmi_to_entity_df")
    assert not hasattr(orch, "_assign_sub_wvmi_per_sub")


# --- the exporter -----------------------------------------------------------------------------------------------------

def test_the_exporter_writes_the_flag_and_the_trigger_idx_as_int(tmp_path):
    """`cycle_collapsed` right after `lp_locked`, read from the FIELD; `triggered_by_event_idx` as nullable `Int64`:
    a column mixing ints and None would print `710.0` (measured) — `710` and an empty cell instead; `meta` is
    `str()` of the record's meta, the None-trigger row included."""
    tbe = lambda i, t: {"triggered_by_event_idx": i, "triggered_by_event_type": t,   # noqa: E731
                        "parent_path_id": "H1.main", "sub_id": 3}
    recs = [WVMIRecord(bos_structure_id=0, bos_cycle_id=0, zone_side="buy", structure_path_id=_SUB, lp_locked=True,
                       status="locked", locked_by_cycle_id=1, meta=tbe(710, "ZONE_PROXIMITY_TRIGGER")),
            WVMIRecord(bos_structure_id=0, bos_cycle_id=1, zone_side="buy", structure_path_id=_SUB,
                       cycle_collapsed=True, meta=tbe(None, None))]
    path = tmp_path / "w.csv"
    export_wvmi(recs, path)
    with open(path, newline="", encoding="utf-8") as fh:
        header, *rows = list(csv.reader(fh))
    assert header[header.index("lp_locked") + 1] == "cycle_collapsed"
    got = [dict(zip(header, r)) for r in rows]
    assert [(g["lp_locked"], g["cycle_collapsed"]) for g in got] == [("True", "False"), ("False", "True")]
    assert [g["triggered_by_event_idx"] for g in got] == ["710", ""]
    assert [g["triggered_by_event_type"] for g in got] == ["ZONE_PROXIMITY_TRIGGER", ""]
    assert [g["meta"] for g in got] == [str(r.meta) for r in recs]
    # the path column is the record FIELD (these metas carry no path — N20); parent_path_id the meta's
    assert [g["structure_path_id"] for g in got] == [_SUB, _SUB]
    assert [g["parent_path_id"] for g in got] == ["H1.main", "H1.main"]
    empty = tmp_path / "e.csv"
    export_wvmi([], empty)
    assert "cycle_collapsed" in empty.read_text(encoding="utf-8").splitlines()[0]


def test_the_exporter_leaves_a_main_rows_parent_path_empty(tmp_path):
    """(landing review N21) a MAIN record's meta has no `parent_path_id`: its column is empty, a sub row's is the
    meta's (no default)."""
    main = WVMIRecord(bos_structure_id=0, bos_cycle_id=1, zone_side="buy", structure_path_id="H1.main",
                      meta={"triggered_by_event_idx": 4, "triggered_by_event_type": "ZONE_PROXIMITY_TRIGGER",
                            "structure_path_id": "H1.main", "trigger_inner": 0.6, "proximity_pips": 8})
    sub = WVMIRecord(bos_structure_id=0, bos_cycle_id=0, zone_side="buy", structure_path_id=_SUB,
                     meta={"triggered_by_event_idx": None, "triggered_by_event_type": None,
                           "parent_path_id": "H1.main", "structure_path_id": _SUB, "sub_id": 3})
    path = tmp_path / "w.csv"
    export_wvmi([main, sub], path)
    with open(path, newline="", encoding="utf-8") as fh:
        header, *rows = list(csv.reader(fh))
    got = [dict(zip(header, r)) for r in rows]
    assert [(g["structure_path_id"], g["parent_path_id"]) for g in got] == [("H1.main", ""), (_SUB, "H1.main")]


def test_the_exporter_refuses_an_unstamped_record(tmp_path):
    """(landing review MINOR 1, rule 3) the trigger keys are read strictly: a record without them — an unstamped sub
    copy — raises instead of exporting as "no trigger in the window"."""
    rec = WVMIRecord(bos_structure_id=0, bos_cycle_id=0, zone_side="buy", structure_path_id=_SUB, meta={"sub_id": 3})
    with pytest.raises(KeyError, match="triggered_by_event_idx"):
        export_wvmi([rec], tmp_path / "w.csv")
    # each key on its own (the fold-in's F2 mutant: a `.get` on the type alone survived the no-key record)
    rec.meta = {"triggered_by_event_idx": 5, "sub_id": 3}
    with pytest.raises(KeyError, match="triggered_by_event_type"):
        export_wvmi([rec], tmp_path / "w.csv")
