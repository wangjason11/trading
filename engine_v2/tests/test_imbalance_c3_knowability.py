"""Plan F (zones pass, 2026-09-23/24) — an FVG counts only once it has FORMED.

`compute_imbalance` flags c2 (the middle candle); the gap exists only once a c3
closes. An `ImbalanceInstance` is visible at a moment K iff its first c3 has
closed (`formed_at = start_idx + 1 <= K`), and only its formed prefix
`[start_idx, min(end_idx, K-1)]` is tested (rule R1 — IMBALANCE_FILL_SEMANTICS
"Knowability"). The cut is keyed on the MOMENT the question is asked
(`evaluated_at`), never on the fill horizon `check_to_idx`.

Consumers covered here through their existing entry points: the POI activation
sweep, FibTracker (CTS_ESTABLISHED / raw + pattern CTS_UPDATED /
CTS_THRESHOLD_UPDATED), and the unchanged MS in-flight POI resolver (the
accepted M1 divergence). See `plans/PLAN_F_imbalance_c3_knowability.md` §5.

Fixture convention (as in test_cross_cycle_fib.py): flat candles o=c=1.5,
h=1.501, l=1.499 far above the bullish gaps (1.0-1.3), so nothing fills unless
a test sets a low.
"""
from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace

import pandas as pd
import pytest

import engine_v2.zones.poi_zones as poi_zones
from engine_v2.common.types import ImbalanceInstance
from engine_v2.structure import event_fields as ef
from engine_v2.structure.event_fields import CTS_UPDATED_RAW_VIA
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.tests._event_factory import make_cts_established, make_cts_updated
from engine_v2.tests.test_poi_activation_moment import _only_poi, _run
from engine_v2.tests.test_unified_probe import _make_multicycle_data
from engine_v2.zones.fib_tracker import FibTracker, FibTrackerConfig, select_fib_anchor_for_cycle


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _df(n, instances):
    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n, freq="15min"),
        "o": [1.5] * n, "h": [1.501] * n, "l": [1.499] * n, "c": [1.5] * n,
        "direction": [0] * n, "is_imbalance": [0] * n,
    })
    df.attrs["imbalances"] = list(instances)
    return df


def _gap(start, end=None, top=1.10, bottom=1.00):
    """A bullish instance (default gap 1.00-1.10; stroke-1 level 1.03)."""
    return ImbalanceInstance(start, start if end is None else end, 1, top, bottom, top - bottom)


def _ev(etype, idx, price, sid, cyc, *, confirmed_at=None, via=CTS_UPDATED_RAW_VIA,
        cts_anchor_idx=None):
    if etype == "CTS_ESTABLISHED":
        return make_cts_established(
            cts_anchor_idx=idx, confirmed_at=idx if confirmed_at is None else confirmed_at,
            price=price, structure_id=sid, cycle_id=cyc,
        )
    meta = {"structure_id": sid, "cycle_id": cyc, "struct_direction": 1}
    if etype == "CTS_UPDATED" and via != CTS_UPDATED_RAW_VIA:
        # Pattern path: the `idx` ARGUMENT is the anchor; the event stamps its
        # moment, the apply candle (`confirmed_at`, Plan E E3·0; the idx, E4c).
        return make_cts_updated(cts_anchor_idx=idx, via=via,
                                confirmed_at=idx if confirmed_at is None else confirmed_at,
                                price=price, meta=meta)
    if etype == "CTS_UPDATED":
        meta["via"] = via
    if cts_anchor_idx is not None:
        meta["cts_anchor_idx"] = cts_anchor_idx
    return StructureEvent(idx=idx, category="STRUCTURE", type=etype, price=price, meta=meta)


def _tracker(mode):
    return FibTracker(FibTrackerConfig(fill_threshold=0.70), fib_mode=mode)


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _sweep(imbalances, *, first_active=6, scan_end=12):
    """The POI sweep on the Plan D (h) geometry: IC 2 inside the 61.8-80 band
    of (BOS 0.6000, CTS 0.6100); CTS applied before the window."""
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    df = pd.DataFrame([
        {"time": t0 + pd.Timedelta(hours=i), "o": 0.6040, "h": 0.6045, "l": 0.6035, "c": 0.6040}
        for i in range(scan_end + 3)
    ])
    df.loc[2, ["o", "h", "l", "c"]] = [0.6034, 0.6035, 0.6025, 0.6026]
    cts = make_cts_established(cts_anchor_idx=5, confirmed_at=first_active, price=0.6100,
                               struct_direction=None)
    return poi_zones._compute_poi_activation_history(
        df, ic_idx=2, cts_established_idx=first_active, sd=1, scan_end=scan_end,
        fill_threshold=0.70, bos_price=0.6000, cts_events=[cts],
        fib_min_pct=61.8, fib_max_pct=80.0,
        variant_thresholds={"V30": 0.3, "V60": 0.6, "V90": 0.9},
        imbalances=imbalances,
        fill_idx_cache={id(i): (None, None) for i in imbalances},
        lifecycle_floor_idx=None,
    )


def _sweep_gap(start, end=None):
    return ImbalanceInstance(start, start if end is None else end, 1, 0.6090, 0.6080, 0.0010)


# ===========================================================================
# BEHAVIOUR TESTS — each fails on the pre-Plan-F code
# ===========================================================================

def test_sweep_enters_an_imbalance_at_its_first_c3_not_its_c2():
    """c2 == first_active: the gap does not exist until first_active + 1."""
    history = _sweep([_sweep_gap(6)], first_active=6)
    assert [(e["idx"], e["active"]) for e in history] == [(7, True)]


def test_sweep_enters_a_merged_run_at_its_first_c3_not_its_last():
    """R1 not R2: a merged run (c2s 8-10) exists from 9, although it keeps
    growing until 11 (catches R2 = enter at end_idx + 1)."""
    history = _sweep([_sweep_gap(8, 10)], first_active=6)
    assert [(e["idx"], e["active"]) for e in history] == [(9, True)]


def test_multicycle_poi_full_history():
    """`_make_multicycle_data` POI (0,2) IC 12 (supersedes Plan D's first-only pin).

    Instance (14,14) commit-fills at 19; the run (19,22) forms only at 20 — past
    this POI's scan end 19 (its cycle ends at 20) — so the POI deactivates at 19.
    Pre-Plan-F the run entered at its c2 = 19, the same candle, masking the fill."""
    _, _, out = _run(_make_multicycle_data())
    z = _only_poi(out, 0, 2)
    assert [(e["idx"], e["active"], e["reason"]) for e in z.meta["activation_history"]] == [
        (15, True, "initial"), (19, False, "imbalance_filled"),
    ]


def test_cross_cycle_cycle0_lag0_est_gap_at_the_moment_no_fib():
    """Lag-0 CTS_ESTABLISHED (anchor == moment == 20) whose only sd gap has
    c2 == 20: nothing has formed at 20 → no fib."""
    tracker = _tracker("cross_cycle")
    df = _df(40, [_gap(20)])
    res = _quiet(tracker.on_cts_established, _ev("CTS_ESTABLISHED", 20, 1.2, 0, 0), df,
                 bos_idx=10, bos_price=0.9)
    assert res is None
    assert (0, 0) not in tracker._fibs


def test_h1_sid0_cycle1_lag0_est_gap_at_the_moment_no_fib():
    tracker = _tracker("h1")
    df = _df(60, [_gap(40)])
    res = _quiet(tracker.on_cts_established, _ev("CTS_ESTABLISHED", 40, 1.35, 0, 1), df,
                 bos_idx=30, bos_price=1.2)
    assert res is None
    assert (0, 1) not in tracker._fibs


def test_raw_update_gap_at_the_candle_activates_one_update_later():
    """cross_cycle cycle-0 first activation on a raw CTS_UPDATED (fib_tracker
    `_handle_cross_cycle_cts_updated`): at 25 the gap (c2 25) has not formed; at
    the next raw update (26) it has."""
    tracker = _tracker("cross_cycle")
    df = _df(40, [_gap(25)])
    assert _quiet(tracker.on_cts_established, _ev("CTS_ESTABLISHED", 20, 1.2, 0, 0), df,
                  bos_idx=10, bos_price=0.9) is None
    assert _quiet(tracker.on_cts_updated, _ev("CTS_UPDATED", 25, 1.25, 0, 0), df) is None
    assert (0, 0) not in tracker._fibs
    upd = _quiet(tracker.on_cts_updated, _ev("CTS_UPDATED", 26, 1.26, 0, 0), df)
    assert upd is not None and upd.active
    assert upd.meta["activated_at"] == 26
    assert upd.meta["activated_on"] == "update"


def test_update_fib_cts_deactivates_until_the_new_gap_forms():
    """`_update_fib_cts`: the old gap (15) commit-fills at 22; a raw update at
    25 whose only unfilled gap has c2 == 25 → no FORMED unfilled imbalance →
    deactivated at 25 (reason all_imbalances_filled); the next raw update (26)
    reactivates it."""
    tracker = _tracker("cross_cycle")
    df = _df(40, [_gap(15), _gap(25)])
    df.at[22, "l"] = 1.02                       # stroke 1 (<= 1.03); close 1.5 >= 1.10 = stroke 2
    est = _quiet(tracker.on_cts_established, _ev("CTS_ESTABLISHED", 20, 1.2, 0, 0), df,
                 bos_idx=10, bos_price=0.9)
    assert est is not None and est.active
    upd = _quiet(tracker.on_cts_updated, _ev("CTS_UPDATED", 25, 1.25, 0, 0), df)
    assert upd.active is False
    assert upd.meta["deactivated_at"] == 25
    assert upd.meta["reason"] == "all_imbalances_filled"
    upd = _quiet(tracker.on_cts_updated, _ev("CTS_UPDATED", 26, 1.26, 0, 0), df)
    assert upd.active is True
    assert upd.meta["reactivated_at"] == 26


def _pre_established_cycle1(own_gap_start):
    """cross_cycle: cycle 0 (gap 15, unfilled) established at 20 and confirmed
    at 25 → cycle 1 pre_established; a dip at 27 makes 27 the prospective BOS_1,
    so a CTS_THRESHOLD_UPDATED at 30 asks for an own gap in [27, 30]."""
    tracker = _tracker("cross_cycle")
    df = _df(40, [_gap(15), _gap(own_gap_start, top=1.40, bottom=1.30)])
    df.at[27, "l"] = 1.45
    _quiet(tracker.on_cts_established, _ev("CTS_ESTABLISHED", 20, 1.2, 0, 0), df,
           bos_idx=10, bos_price=0.9)
    _quiet(tracker.on_cts_confirmed, _ev("CTS_CONFIRMED", 25, 1.2, 0, 0, cts_anchor_idx=20))
    assert tracker._m15_phase[(0, 1)] == "pre_established"
    _quiet(tracker.on_cts_threshold_updated, _ev("CTS_THRESHOLD_UPDATED", 30, 1.52, 0, 0), df)
    return tracker


def test_threshold_update_own_gap_at_the_candle_creates_no_cross():
    """The counter sub-5 shape: the only own gap has c2 == the threshold-update
    candle, so it has not formed → no pre-established cross."""
    tracker = _pre_established_cycle1(own_gap_start=30)
    assert tracker._get_latest_cross(0, 1) is None


def test_h1_cycle0_scenario1_est_gap_at_the_moment_no_activation():
    """sid >= 1, h1: Scenario 1 TRUE at a lag-0 cycle-0 EST whose only gap has
    c2 == CTS_0 → no cycle-0 activation at the EST."""
    tracker = _tracker("h1")
    df = _df(40, [_gap(20)])
    res = _quiet(tracker.on_cts_established, _ev("CTS_ESTABLISHED", 20, 1.2, 1, 0), df,
                 bos_idx=10, bos_price=0.9, reversal_confirmed_idx=15)
    assert res is None
    assert (1, 0) not in tracker._fibs


def test_h1_cycle0_scenario1_on_update_reads_the_cache_at_the_moment():
    """The cycle-0 cache re-snapshot on a raw update writes the UNCUT value, but
    the Scenario-1 activation at that update asks at its moment: the gap (c2 25)
    has not formed at 25 → no activation."""
    tracker = _tracker("h1")
    df = _df(40, [_gap(25)])
    _quiet(tracker.on_cts_established, _ev("CTS_ESTABLISHED", 20, 1.2, 1, 0), df,
           bos_idx=10, bos_price=0.9, reversal_confirmed_idx=25)
    res = _quiet(tracker.on_cts_updated, _ev("CTS_UPDATED", 25, 1.25, 1, 0), df, 25)
    assert res is None
    assert (1, 0) not in tracker._fibs
    assert tracker._cross_cycle_data[1]["cycle0"]["has_unfilled"] is True   # uncut cache


# ===========================================================================
# GUARD PINS — pass on the pre-Plan-F code; each catches a named wrong variant
# ===========================================================================

def test_guard_lag1_est_gap_at_the_anchor_still_activates():
    """Anchor 20, moment 21, gap c2 20: formed at 21 == the moment → counted.
    Catches a cut keyed on check_to_idx (the anchor)."""
    tracker = _tracker("cross_cycle")
    df = _df(40, [_gap(20)])
    res = _quiet(tracker.on_cts_established,
                 _ev("CTS_ESTABLISHED", 20, 1.2, 0, 0, confirmed_at=21), df,
                 bos_idx=10, bos_price=0.9)
    assert res is not None and res.active


def test_pattern_path_update_is_cut_at_its_apply_not_its_anchor():
    """Plan E E3·0: a pattern-path CTS_UPDATED records its moment (the apply
    candle, `meta["confirmed_at"]`), so the knowability cut applies there.
    Anchor 25, apply 26. Gap c2 25 → formed at 26 == the apply → counted (a cut
    at the anchor 25 would drop it). The activation stamp is the moment
    too (Plan E E3a)."""
    tracker = _tracker("cross_cycle")
    df = _df(40, [_gap(25)])
    _quiet(tracker.on_cts_established, _ev("CTS_ESTABLISHED", 20, 1.2, 0, 0), df,
           bos_idx=10, bos_price=0.9)
    upd = _quiet(tracker.on_cts_updated,
                 _ev("CTS_UPDATED", 25, 1.25, 0, 0, via="one_maru_continuous", confirmed_at=26), df)
    assert upd is not None and upd.active
    assert upd.meta["activated_at"] == 26   # the moment (Plan E E3a), not the anchor 25


def test_pattern_path_update_gap_at_its_apply_waits_one_update():
    """Plan E E3·0: anchor = apply = 25; the only gap has c2 25 → formed at 26,
    after the moment → not counted (uncut — before E3·0 — it activated here).
    The next pattern update (anchor = apply = 26) counts it. Landing review:
    only this shape changes the answer between cut and uncut."""
    tracker = _tracker("cross_cycle")
    df = _df(40, [_gap(25)])
    _quiet(tracker.on_cts_established, _ev("CTS_ESTABLISHED", 20, 1.2, 0, 0), df,
           bos_idx=10, bos_price=0.9)
    upd = _quiet(tracker.on_cts_updated,
                 _ev("CTS_UPDATED", 25, 1.25, 0, 0, via="one_maru_continuous", confirmed_at=25), df)
    assert upd is None or not upd.active
    upd = _quiet(tracker.on_cts_updated,
                 _ev("CTS_UPDATED", 26, 1.26, 0, 0, via="one_maru_continuous", confirmed_at=26), df)
    assert upd is not None and upd.active
    assert upd.meta["activated_at"] == 26


def test_guard_cycle0_cache_stays_uncut():
    """The cycle-0 liveness cache is Scenario-2 cond2, used at CTS_1 EST when
    every gap in [BOS_0, CTS_0] has formed → it stays uncut. Catches a cut cache."""
    tracker = _tracker("h1")
    df = _df(40, [_gap(20)])
    _quiet(tracker.on_cts_established, _ev("CTS_ESTABLISHED", 20, 1.2, 1, 0), df,
           bos_idx=10, bos_price=0.9, reversal_confirmed_idx=15)
    assert tracker._cross_cycle_data[1]["cycle0"]["has_unfilled"] is True


def test_guard_threshold_update_own_gap_formed_at_the_candle_creates_the_cross():
    """Own gap c2 29 → formed at 30 == the threshold-update candle → counted:
    the pre-established cross (0→1) is created (positive coverage)."""
    tracker = _pre_established_cycle1(own_gap_start=29)
    latest = tracker._get_latest_cross(0, 1)
    assert latest is not None and latest[1].active
    assert latest[1].meta["cross_start_cycle"] == 0


def test_guard_routine_evaluated_at_none_is_uncut():
    """`evaluated_at=None` = no knowability cut (today's answer) on the routine
    the unchanged MS in-flight resolver uses."""
    from engine_v2.zones.cross_cycle_fib import resolve_cross_cycle_eligibility
    df = _df(40, [_gap(30)])
    elig = resolve_cross_cycle_eligibility(
        df=df, target_cycle=1, sd=1, own_window_end_idx=30, fill_horizon_idx=30, own_imb_start=27,
        anchor_idx=30, anchor_price=1.5, bos_by_cycle={}, cts_by_cycle={},
        dead_cycles=set(), fill_threshold=0.70, fill_as_of="current", evaluated_at=None,
        snapshot_horizon_idx=None,
    )
    assert elig.own_has is True                 # the not-yet-formed gap still counts, uncut


# ===========================================================================
# THE ACCEPTED DIVERGENCE (Plan F §2, cold review M1) — the MS half passes on
# the pre-Plan-F code, the FibTracker half is behaviour
# ===========================================================================

def test_m1_ms_inflight_keeps_the_inner_fibtracker_creates_no_fib():
    """The unchanged MS in-flight resolver still counts a gap formed one candle
    after the refresh (its only reader is gated i > cts.idx), while FibTracker,
    cutting at the event, never creates the fib. Flips when the "re-ask when the
    gap forms" follow-up lands."""
    df = _df(60, [_gap(40, top=1.30, bottom=1.20)])
    # IC candidate at 32: bearish, inside the 61.8-80% band of (BOS 1.00 → CTS 1.35).
    df.loc[32, ["o", "h", "l", "c", "direction"]] = [1.12, 1.13, 1.08, 1.09, -1]
    inners = poi_zones.compute_poi_inners_for_cycle(
        df, 30, 1.00, 40, 1.35, 1, structure_id=0, cycle_id=1, fill_horizon_idx=40, snapshot_horizon_idx=40,
    )
    assert inners == [1.13]
    tracker = _tracker("h1")
    _quiet(tracker.on_cts_established, _ev("CTS_ESTABLISHED", 40, 1.35, 0, 1), df,
           bos_idx=30, bos_price=1.00)
    assert (0, 1) not in tracker._fibs


# ===========================================================================
# event_moment — the moment of a CTS / BOS event (structure/event_fields.py;
# moved from market_structure and extended in Plan E E2a)
# ===========================================================================

def test_event_moment_per_type():
    assert ef.event_moment(_ev("CTS_ESTABLISHED", 9, 1.2, 0, 1, confirmed_at=10)) == 10
    assert ef.event_moment(_ev("CTS_UPDATED", 25, 1.2, 0, 0)) == 25
    # pattern path: the apply candle (Plan E E3·0), not the anchor idx
    assert ef.event_moment(_ev("CTS_UPDATED", 25, 1.2, 0, 0, via="continuous", confirmed_at=27)) == 27
    assert ef.event_moment(_ev("CTS_THRESHOLD_UPDATED", 30, 1.2, 0, 0)) == 30
    # Plan E E3g-2: BOS_THRESHOLD_UPDATED is stamped at its processing candle too
    assert ef.event_moment(_ev("BOS_THRESHOLD_UPDATED", 31, 1.1, 0, 0)) == 31
    # Plan E E2a extends it (reversing Plan F's "any other CTS type raises"):
    # a CTS_CONFIRMED's idx IS its confirmation candle.
    assert ef.event_moment(_ev("CTS_CONFIRMED", 25, 1.2, 0, 0)) == 25
    with pytest.raises(ValueError, match="RANGE_STARTED"):
        ef.event_moment(_ev("RANGE_STARTED", 25, 1.2, 0, 0))


def test_event_moment_reads_the_contract_keys_directly():
    ev = _ev("CTS_ESTABLISHED", 9, 1.2, 0, 1)
    del ev.meta["confirmed_at"]
    with pytest.raises(KeyError):
        ef.event_moment(ev)
    ev = _ev("CTS_UPDATED", 9, 1.2, 0, 1)
    del ev.meta["via"]
    with pytest.raises(KeyError):
        ef.event_moment(ev)


def _multicycle_with_tied_pattern_breakout():
    """`_make_multicycle_data` (0-23, cycle 3 established at 20, pre-confirm) +
    24 big bull maru (the new CTS extreme; a raw update at 24) + 25 small bear
    normal → `one_maru_opposite(+1)` applied at 25, whose pattern extreme is 24
    — the candle the raw path already took in the back-fill: a TIE, so since
    2026-09-27 (pattern path = the raw path's strict new-extreme rule) it emits
    NO CTS_UPDATED and does not refresh (it was a pattern-path CTS_UPDATED with
    anchor 24 / moment 25 before, hand-verified 2026-09-24) + 26 a filler."""
    from engine_v2.tests.test_unified_probe import _R
    rows = _make_multicycle_data()
    c = rows[-1]["c"]                                              # .6299
    rows.append(_R(c, c + 0.0052, c - 0.0002, c + 0.0050))         # 24
    c += 0.0050
    rows.append(_R(c - 0.0001, c + 0.0001, c - 0.0009, c - 0.0007))  # 25
    c -= 0.0007
    rows.append(_R(c, c + 0.0002, c - 0.0006, c - 0.0004))         # 26
    return rows


def test_ms_emits_raw_updates_with_the_raw_via_and_patterns_with_their_apply():
    """A live MS run: every raw-path CTS_UPDATED carries CTS_UPDATED_RAW_VIA and
    sits on its processing candle; every other via is a breakout-pattern name and
    records its apply candle as `meta["confirmed_at"]` (Plan E E3·0) and as its
    `idx` (E4c) — the moment `event_moment` returns, never before the anchor
    `meta["cts_anchor_idx"]`. Since 2026-09-27 the two are EQUAL on every
    pattern-path update MS emits: every span candle before the apply candle was
    raw-processed in the back-fill, so only the apply candle can be a strict new
    extreme — the fixture's one_maru_opposite at 25 (extreme 24, the raw-updated
    CTS) is a tie and emits nothing."""
    from engine_v2.structure.structure_engine import compute_bounded_structure
    from engine_v2.tests.test_unified_probe import _prepare_df
    with contextlib.redirect_stdout(io.StringIO()):
        res = compute_bounded_structure(_prepare_df(_multicycle_with_tied_pattern_breakout()), 0, +1)
    updates = [e for e in res.events if e.type == "CTS_UPDATED"]
    raw = [e for e in updates if e.meta["via"] == CTS_UPDATED_RAW_VIA]
    patterns = [e for e in updates if e.meta["via"] != CTS_UPDATED_RAW_VIA]
    assert raw and patterns
    assert all("confirmed_at" not in e.meta and "cts_anchor_idx" not in e.meta for e in raw)
    for e in patterns:
        assert ef.event_moment(e) == e.meta["confirmed_at"] == e.idx >= e.meta["cts_anchor_idx"]
    lagging = [(ef.cts_anchor_idx(e), e.idx, e.meta["via"]) for e in patterns
               if ef.cts_anchor_idx(e) != e.idx]
    assert lagging == []
    assert int(res.df["last_breakout_pat_apply_idx"].iloc[25]) == 25   # precondition: the tying breakout WAS applied at 25
    assert 25 not in [e.idx for e in updates] and 24 in [e.idx for e in raw]   # the tie: raw 24 stands alone


# ===========================================================================
# Landing cold review (2026-09-24): pins for the remaining cut / uncut sites
# ===========================================================================

def test_h1_cycle0_scenario1_true_activation_on_update_waits_for_the_gap():
    """Scenario 1 already TRUE, no cycle-0 fib yet: the activation-on-update read
    (`_c0_has_unfilled_now`) asks at the update's moment — gap c2 25 has not formed
    at 25, has at 26."""
    tracker = _tracker("h1")
    df = _df(40, [_gap(25)])
    _quiet(tracker.on_cts_established, _ev("CTS_ESTABLISHED", 20, 1.2, 1, 0), df,
           bos_idx=10, bos_price=0.9, reversal_confirmed_idx=15)
    assert _quiet(tracker.on_cts_updated, _ev("CTS_UPDATED", 25, 1.25, 1, 0), df, 15) is None
    upd = _quiet(tracker.on_cts_updated, _ev("CTS_UPDATED", 26, 1.26, 1, 0), df, 15)
    assert upd.meta["activated_at"] == 26


def _drive_h1_cycle1(confirmed_at, extra_gaps=()):
    """sid 1, h1: cycle 0 (gap 15; Scenario 1 FALSE, rv 100) → CTS_1 EST at 40."""
    tracker = _tracker("h1")
    df = _df(80, [_gap(15), *extra_gaps])
    _quiet(tracker.on_cts_established, _ev("CTS_ESTABLISHED", 20, 1.15, 1, 0), df,
           bos_idx=10, bos_price=1.0, reversal_confirmed_idx=100)
    _quiet(tracker.on_cts_confirmed, _ev("CTS_CONFIRMED", 25, 1.15, 1, 0, cts_anchor_idx=20))
    _quiet(tracker.on_cts_established,
           _ev("CTS_ESTABLISHED", 40, 1.35, 1, 1, confirmed_at=confirmed_at), df,
           bos_idx=30, bos_price=1.2, reversal_confirmed_idx=100)
    return tracker, df


@pytest.mark.parametrize("confirmed_at, crosses", [(40, False), (41, True)])
def test_h1_scenario2_cond1_asks_at_the_cts1_moment(confirmed_at, crosses):
    """Scenario-2 cond1 (`select_fib_anchor_for_cycle` from FibTracker): cycle 1's
    only own gap has c2 == CTS_1 anchor 40 — not formed at a lag-0 moment (no
    cross), formed at a lag-1 moment 41 (cross)."""
    tracker, _ = _drive_h1_cycle1(confirmed_at, [_gap(40, top=1.30, bottom=1.20)])
    assert (tracker._get_latest_cross(1, 1) is not None) is crosses


def test_h1_cycle1_cross_update_cuts_cond2():
    """`_update_cycle1_main`: cycle 1's own gap (35) fills at 42; the update at 45
    whose only other gap has c2 == 45 → cond2 False (not formed) → the cross
    deactivates at 45 and create-on-fail finds nothing formed either."""
    tracker, df = _drive_h1_cycle1(40, [_gap(35, top=1.30, bottom=1.20),
                                        _gap(45, top=1.45, bottom=1.40)])
    df.at[42, "l"] = 1.22
    assert tracker._get_latest_cross(1, 1)[1].active
    _quiet(tracker.on_cts_updated, _ev("CTS_UPDATED", 45, 1.46, 1, 1), df, 100)
    cross = tracker._get_latest_cross(1, 1)[1]
    assert cross.active is False and cross.meta["deactivated_at"] == 45
    assert (1, 1) not in tracker._fibs


def _c0():
    return {"bos_idx": 10, "bos_price": 1.0, "cts_idx": 20, "cts_price": 1.15,
            "has_unfilled": True, "scenario1": None}


@pytest.mark.parametrize("evaluated_at, label", [
    (None, "scenario_2_cross"), (40, "scenario_3"), (41, "scenario_2_cross"),
])
def test_select_fib_anchor_cond1_at_the_moment(evaluated_at, label):
    df = _df(60, [_gap(15), _gap(40, top=1.30, bottom=1.20)])
    out = select_fib_anchor_for_cycle(df, 1, 1, 30, 1.2, 40, 1.35, _c0(), 0.70,
                                      struct_direction=1, evaluated_at=evaluated_at,
                                      fill_horizon_idx=40, snapshot_horizon_idx=30)
    assert out[-1] == label


def test_guard_ms_inflight_select_call_is_uncut(monkeypatch):
    """The MS in-flight resolver passes evaluated_at=None (Plan F §2 decision)."""
    seen = []
    real = poi_zones.select_fib_anchor_for_cycle

    def spy(*a, **k):
        seen.append(k["evaluated_at"])
        return real(*a, **k)

    monkeypatch.setattr(poi_zones, "select_fib_anchor_for_cycle", spy)
    df = _df(60, [_gap(15), _gap(40, top=1.30, bottom=1.20)])
    poi_zones.compute_poi_inners_for_cycle(df, 30, 1.2, 40, 1.35, 1,
                                           structure_id=1, cycle_id=1, c0_data=_c0(),
                                           fill_horizon_idx=40, snapshot_horizon_idx=40)
    assert seen == [None]


def test_guard_ms_cycle0_snapshot_is_uncut():
    """`MarketStructure._update_cycle0_data` (Scenario-2 cond2 mirror) stays uncut:
    a gap with c2 == CTS_0 counts (it has formed by any later cycle-1 read)."""
    from engine_v2.structure.market_structure import MarketStructure
    ms = MarketStructure.__new__(MarketStructure)
    ms.df = _df(40, [_gap(20)])
    ms._fill_threshold = 0.70
    ms.state = SimpleNamespace(cts_cycle_id=0, cts=SimpleNamespace(idx=20, price=1.2),
                               bos=SimpleNamespace(idx=10, price=0.9),
                               struct_direction=1, cycle0_data=None)
    ms._update_cycle0_data(20)
    assert ms.state.cycle0_data["has_unfilled"] is True


def test_event_scope_is_restored_after_every_handler():
    tracker = _tracker("cross_cycle")
    df = _df(40, [_gap(15)])
    _quiet(tracker.on_cts_established, _ev("CTS_ESTABLISHED", 20, 1.2, 0, 0), df,
           bos_idx=10, bos_price=0.9)
    _quiet(tracker.on_cts_updated, _ev("CTS_UPDATED", 25, 1.25, 0, 0), df)
    _quiet(tracker.on_cts_threshold_updated, _ev("CTS_THRESHOLD_UPDATED", 26, 1.2, 0, 0), df)
    assert tracker._evaluated_at is None and tracker._in_event is False
    with tracker._evaluating(_ev("CTS_UPDATED", 27, 1.3, 0, 0)):
        with pytest.raises(AssertionError, match="must not nest"):
            with tracker._evaluating(_ev("CTS_UPDATED", 28, 1.3, 0, 0)):
                pass


def test_a_malformed_event_does_not_wedge_the_tracker():
    """event_moment raising (missing `via`) must not leave the scope flag set."""
    tracker = _tracker("h1")
    df = _df(40, [_gap(25)])
    bad = _ev("CTS_UPDATED", 25, 1.25, 1, 0)
    del bad.meta["via"]
    with pytest.raises(KeyError):
        _quiet(tracker.on_cts_updated, bad, df, 15)
    _quiet(tracker.on_cts_updated, _ev("CTS_UPDATED", 26, 1.26, 1, 0), df, 15)
    assert tracker._evaluated_at is None and tracker._in_event is False
