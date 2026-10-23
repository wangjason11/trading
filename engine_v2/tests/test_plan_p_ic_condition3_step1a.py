"""PLAN_P — IC condition 3 = the IMMEDIATE unfilled imbalance + Step 1A, the 80–100 fallback zone (the user's changes 1
+ 3, 2026-10-13; `plans/PLAN_P_ic_condition3_step1a.md`, POI_ZONES_SPEC §3).

Pinned (each with a fixture where the old and the new rule DIFFER): Step 1A runs when Step 1 finds no IC candidate,
re-evaluating ONLY condition 4 on 80–100 (conditions 1–3 kept), and never after Steps 2–3 (an all-pinbar answer stays
no IC); `ic_zone_band` on the selection, the POI meta, every CSV row and the two hovers; the POI's activation touch in
the IC's band; the activation's own imbalance condition stays ANY gap after the IC (decoupled from condition 3 — the
user, 2026-10-13); MS's in-flight read takes a Step 1A pick (one selector, PLAN_H D2); the band → zone-parameter map.
Condition 3 itself and the shortened rule sets are pinned where the old rules were (`test_institutional_pattern`,
`test_plan_k_landing_b`, `test_plan_l_landing_1` / `_2`).
"""
from __future__ import annotations

import contextlib
import dataclasses
import io

import pandas as pd
import pytest

import engine_v2.zones.poi_zones as poi_zones
from engine_v2.charting._zone_render import fib_hover_no_ic_text, poi_ic_hover_text
from engine_v2.common.types import ImbalanceInstance
from engine_v2.debug.export_zones import export_ic_candidates
from engine_v2.tests._cross_fib_limits import NO_REF_TRACKER
from engine_v2.tests._event_factory import make_cts_established
from engine_v2.tests.test_fib_chart_render import _ip
from engine_v2.tests.test_institutional_pattern import (
    D,
    NO_1B_META,
    U,
    _bos_gap,
    _far_gap,
    _fib_over,
    _frame,
    _select,
    _tb2_frame,
    _Tracker,
)
from engine_v2.zones.cross_cycle_fib import ICReference
from engine_v2.zones.poi_zones import (
    ALL_IC_CANDIDATES_PINBAR,
    IC_REF_PAST_REFERENCE as PAST,
    IC_ZONE_FALLBACK,
    IC_ZONE_PRIMARY,
    NO_IC_CANDIDATE,
    TIEBREAK_MOST_RECENT_BEFORE_IMBALANCE as MOST_RECENT,
    POIConfig,
    derive_ic_candidates,
    derive_poi_zones,
    fib_zone_bounds,
    ic_touches_zone,
    ic_zone_config,
    select_ic,
    select_ic_records,
)

S = ("N", U, 2)            # a small sd spacer (a 1-pip body: a pinbar; agreed sd — never a check)
_GAPS = (8, 10, 12)        # the `_tb2_frame` ICs' immediate gaps (7 / 9 / 11)
# A 61.8–80 zone ABOVE every `_tb2_frame` IC (highs 0.60093 … 0.60165): its 80–100 zone [0.59980, 0.6020] holds them
_ABOVE = (0.6020, 0.6040)


def _zone(fib, band):
    return fib_zone_bounds(fib.fib.anchor_high, fib.fib.anchor_low, 1, ic_zone_config(POIConfig(), band))


def _touches(df, i, zone):
    return ic_touches_zone(float(df.loc[i, "h"]), float(df.loc[i, "l"]), *zone)


# ---------------------------------------------------------------------------
# Step 1A — when it runs, what it re-evaluates
# ---------------------------------------------------------------------------

def test_step_1a_picks_from_the_80_100_zone_when_step_1_finds_none():
    """No IC touches 61.8–80 → no IC candidate in Step 1 → Step 1A: the same checks, condition 4 on 80–100 — all three
    touch it and keep their immediate gaps → the usual Steps 2 / 3 (three 1A.i → the tiebreak rules → the oldest, 7),
    band 80-100. The control: the same frame with a 61.8–80 zone holding them → Step 1 finds them, band 61.8-80."""
    df = _tb2_frame()
    fib = _fib_over(df, 6, 12, *_ABOVE)
    assert not any(_touches(df, i, _zone(fib, IC_ZONE_PRIMARY)) for i in (7, 9, 11))
    sel = _select(df, 6, 12, [_far_gap(g) for g in _GAPS], zone=_ABOVE, classify_all=True)
    assert sel.ic_zone_band == IC_ZONE_FALLBACK
    assert [(c.ic_idx, c.touches_zone, c.is_ic_candidate) for c in sel.checks] == [
        (7, True, True), (9, True, True), (11, True, True)]
    assert sel.pick.ic_idx == 7 and sel.tiebreak == (MOST_RECENT, 11, 3, 8)
    assert (sel.ip_outcome, sel.pick.ip_rank) == ("pattern", "1A.i")        # Steps 2 / 3 run as usual after 1A
    control = _select(_tb2_frame(), 6, 12, [_far_gap(g) for g in _GAPS], zone=(0.5, 0.7))
    assert control.ic_zone_band == IC_ZONE_PRIMARY and control.pick.ic_idx == 7


def _pinbars_then_a_maru():
    """7 / 9 pinbars (no pattern) touching a 61.8–80 zone that 11 (a maru) misses — 11 touches only its 80–100 zone;
    each with its immediate gap (8 / 10 / 12)."""
    df = _frame([("M", U, 10), ("P", D, 6), S, ("P", D, 6), S, ("M", D, 6), S])
    lo = (float(df.loc[11, "h"]) + min(float(df.loc[7, "h"]), float(df.loc[9, "h"]))) / 2
    fib = _fib_over(df, 6, 12, lo, lo + 0.0002)
    assert [str(df.loc[i, "candle_type"]) for i in (7, 9, 11)] == ["pinbar", "pinbar", "maru"]
    assert [_touches(df, i, _zone(fib, IC_ZONE_PRIMARY)) for i in (7, 9, 11)] == [True, True, False]
    assert _touches(df, 11, _zone(fib, IC_ZONE_FALLBACK))
    df.attrs["imbalances"] = [_bos_gap(6)] + [_far_gap(g) for g in _GAPS]      # 1B-neutral (PLAN_S step 3)
    return df, fib


def test_step_1a_never_runs_after_an_all_pinbar_step_3():
    """The user: Step 1A "would only ever come into play immediate after Step1" — Step 1 HAS candidates (the pinbars
    7 / 9), Step 3 excludes both → no IC (`all_ic_candidates_pinbar`, band 61.8-80), although the maru 11 with its
    immediate gap touches the 80–100 zone (a Step 1A run would pick it)."""
    df, fib = _pinbars_then_a_maru()
    sel = select_ic(df, fib, POIConfig(), ic_reference=ICReference.NONE, classify_all=True)
    assert [(c.ic_idx, c.touches_zone, c.is_ic_candidate) for c in sel.checks] == [
        (7, True, True), (9, True, True), (11, False, False)]
    assert (sel.pick, sel.no_ic_reason, sel.ic_zone_band) == (None, ALL_IC_CANDIDATES_PINBAR, IC_ZONE_PRIMARY)


def test_step_1a_keeps_conditions_2_and_3():
    """Step 1A re-evaluates ONLY condition 4: every IC touches the 80–100 zone, yet with no immediate gap (condition 3)
    — or with a reference every IC is past (condition 2) — there is no IC candidate: `no_ic_candidate`, band 80-100
    (Step 1A ran and found none)."""
    no_gap = _select(_tb2_frame(), 6, 12, [], zone=_ABOVE)
    assert [(c.touches_zone, c.immediate_imbalance_first_c2_idx) for c in no_gap.checks] == [(True, None)] * 3
    assert (no_gap.pick, no_gap.no_ic_reason, no_gap.ic_zone_band) == (None, NO_IC_CANDIDATE, IC_ZONE_FALLBACK)
    cut = _select(_tb2_frame(), 6, 12, [_far_gap(g) for g in _GAPS], zone=_ABOVE,
                  ic_reference=ICReference(0.60050, "prev_cycle_cts"))
    assert [(c.touches_zone, c.ic_ref_reason) for c in cut.checks] == [(True, PAST)] * 3
    assert (cut.pick, cut.no_ic_reason, cut.ic_zone_band) == (None, NO_IC_CANDIDATE, IC_ZONE_FALLBACK)


def test_no_checks_keep_the_primary_band():
    """No agreed-opposite candle in the window, or a fib neither active nor locked: no checks → Step 1A does not run
    (nothing to re-evaluate) → band 61.8-80."""
    df = _frame([("M", U, 10), ("M", U, 10), ("M", U, 10)])
    df.attrs["imbalances"] = []
    fib = _fib_over(df, 6, 8, 0.5, 0.7)
    sel = select_ic(df, fib, POIConfig(), ic_reference=ICReference.NONE)
    assert (sel.checks, sel.no_ic_reason, sel.ic_zone_band) == ((), NO_IC_CANDIDATE, IC_ZONE_PRIMARY)
    dead = dataclasses.replace(fib, active=False, locked=False)
    assert select_ic(df, dead, POIConfig(), ic_reference=ICReference.NONE).ic_zone_band == IC_ZONE_PRIMARY


def test_the_band_maps_to_its_zone_parameters():
    """`ic_zone_config` — the one place a band becomes zone parameters: the primary band returns the config itself;
    the fallback = (`ic_fallback_fib_min`, `ic_fallback_fib_max`) = (80, 100), everything else unchanged — the
    80–100 zone of a buy fib is [the BOS price, the 80 % price]."""
    cfg = POIConfig(fill_threshold=0.42)
    assert ic_zone_config(cfg, IC_ZONE_PRIMARY) is cfg
    fb = ic_zone_config(cfg, IC_ZONE_FALLBACK)
    assert (fb.ic_fib_min, fb.ic_fib_max, fb.fill_threshold) == (80.0, 100.0, 0.42)
    assert (POIConfig().ic_fallback_fib_min, POIConfig().ic_fallback_fib_max) == (80.0, 100.0)
    assert (IC_ZONE_PRIMARY, IC_ZONE_FALLBACK) == ("61.8-80", "80-100")
    with pytest.raises(AssertionError):
        ic_zone_config(cfg, "61.8-100")
    fib = _fib_over(_tb2_frame(), 6, 12, *_ABOVE)
    lo, hi = _zone(fib, IC_ZONE_FALLBACK)
    assert lo == pytest.approx(fib.bos_price) and hi == pytest.approx(_ABOVE[0])


# ---------------------------------------------------------------------------
# The consumers: the drawn POI (meta, activation), the CSV, the hovers, MS's read
# ---------------------------------------------------------------------------

def _drawn_step_1a():
    df = _tb2_frame()
    df.attrs["imbalances"] = [_bos_gap(6)] + [_far_gap(g) for g in _GAPS]      # 1B-neutral (PLAN_S step 3)
    fib = _fib_over(df, 6, 12, *_ABOVE)
    est = [make_cts_established(cts_anchor_idx=12, confirmed_at=12, price=fib.cts_price, structure_id=0,
                                cycle_id=0)]
    tracker = _Tracker([fib])
    recs = select_ic_records(df, est, tracker, POIConfig())
    with contextlib.redirect_stdout(io.StringIO()):
        pois = derive_poi_zones(df, est, fib_tracker=tracker, config=POIConfig(), ic_records=recs)
    rows = derive_ic_candidates(df, est, tracker, POIConfig(), structure_path_id="H1.main", ic_records=recs)
    return df, fib, pois, rows


def test_a_step_1a_poi_carries_its_band_and_activates_in_it():
    """The drawn POI of a Step 1A pick: meta `ic_zone_band` 80-100, and its activation re-checks the touch in 80–100
    (the user: "POI activation check should always reference the zone parameters the IC came from") — active at the
    CTS-established moment 12; on the 61.8–80 zone the IC would never touch (never active)."""
    df, fib, (poi,), _ = _drawn_step_1a()
    assert (poi.ic_idx, poi.meta["ic_zone_band"]) == (7, IC_ZONE_FALLBACK)
    assert [(e["idx"], e["active"], e["reason"]) for e in poi.meta["activation_history"]] == [(12, True, "initial")]
    assert not _touches(df, 7, _zone(fib, IC_ZONE_PRIMARY))


def test_the_csv_carries_the_band_on_every_row(tmp_path):
    _, _, _, rows = _drawn_step_1a()
    assert [(r.ic_idx, r.picked, r.ic_zone_band) for r in rows] == [
        (7, True, IC_ZONE_FALLBACK), (9, False, IC_ZONE_FALLBACK), (11, False, IC_ZONE_FALLBACK)]
    path = tmp_path / "ic.csv"
    export_ic_candidates(rows, path)
    assert pd.read_csv(path, dtype=str)["ic_zone_band"].tolist() == ["80-100"] * 3
    no_ic = derive_ic_candidates(*_no_ic_inputs(), POIConfig(), structure_path_id="H1.main")
    assert [(r.ic_idx, r.reason, r.ic_zone_band) for r in no_ic][-1] == (None, NO_IC_CANDIDATE, IC_ZONE_FALLBACK)


def _no_ic_inputs():
    df = _tb2_frame()
    df.attrs["imbalances"] = []
    fib = _fib_over(df, 6, 12, *_ABOVE)
    est = [make_cts_established(cts_anchor_idx=12, confirmed_at=12, price=fib.cts_price, structure_id=0,
                                cycle_id=0)]
    return df, est, _Tracker([fib])


def test_the_poi_hover_names_the_fallback_band_only():
    """The POI hover's extra line `ic_zone=80-100 (fallback)` — after the pattern line (or a ranked IC's line); the
    61.8–80 band adds nothing."""
    _, _, (poi,), _ = _drawn_step_1a()
    assert poi_ic_hover_text(poi.meta, 7) == ("ic_reason=IP 1A.i · legs 1-1-1 (6 | 7 | 8)<br>ic_tiebreak=over 11: most recent "
                                              "before imbalance 8<br>ic_zone=80-100 (fallback)")
    primary = {**poi.meta, "ic_zone_band": IC_ZONE_PRIMARY}
    assert "fallback" not in poi_ic_hover_text(primary, 7)
    ranked = {"ip_outcome": "no_pattern", "ip_rank": None, "ip_leg_candle_counts": None, "pattern_anchor_idx": None,
              "opp_leg_first_idx": None, "pattern_end_idx": None, "pattern_confirmed_idx": None, "ip_tiebreak": None,
              "ranking_rule": None, "ranking_runner_up_ic_idx": None, "ranking_imbalance_first_c2_idx": None,
              "ic_zone_band": IC_ZONE_FALLBACK, **NO_1B_META}
    assert poi_ic_hover_text(ranked, 1934) == "ic_reason=No pattern · only IC candidate<br>ic_zone=80-100 (fallback)"
    with pytest.raises(AssertionError, match="ic_zone_band"):            # a POI without a known band fails loudly
        poi_ic_hover_text({**ranked, "ic_zone_band": None}, 1934)


@pytest.mark.parametrize("row, line", [
    # a pick's band is on its POI hover (`ic_zone=`) — the fib hover keeps an IC line only when it found no IC
    # (the user's fib-hover spec 2026-10-23)
    (_ip(ic=4131, rank="3B.iii", picked=True, band=IC_ZONE_FALLBACK), None),
    (_ip(ic=1934, picked=True, outcome="no_pattern", ranking_rule="exclude_pinbar", band=IC_ZONE_FALLBACK), None),
    (_ip(reason="no_ic_candidate", band=IC_ZONE_FALLBACK),
     "ic_idx=NA (no_ic_candidate) | ic_zone=80-100 (fallback)"),     # Step 1A ran and found none
    (_ip(reason="no_ic_candidate"), "ic_idx=NA (no_ic_candidate)"),
    (_ip(ic=4131, rank="3B.iii", picked=True), None),
    (None, "ic_idx=NA (not evaluated: no POI record)"),
])
def test_the_fib_no_ic_line_names_the_fallback_band_only(row, line):
    assert fib_hover_no_ic_text(row) == line


def test_the_fib_no_ic_line_rejects_a_row_without_a_known_band():
    with pytest.raises(AssertionError, match="ic_zone_band"):
        fib_hover_no_ic_text(_ip(ic=4131, rank="3B.iii", picked=True, band=None))


def test_ms_reads_a_step_1a_ics_inner():
    """One selector (PLAN_H D2): MS's in-flight read (`poi_inner_for_fib`) takes the Step 1A pick — IC 7's high."""
    df = _tb2_frame()
    df.attrs["imbalances"] = [_bos_gap(6)] + [_far_gap(g) for g in _GAPS]      # 1B-neutral (PLAN_S step 3)
    fib = _fib_over(df, 6, 12, *_ABOVE)
    assert poi_zones.poi_inner_for_fib(df, fib, 0.70, tracker=NO_REF_TRACKER, evaluated_at=12) == float(
        df.loc[7, "h"])


# ---------------------------------------------------------------------------
# The POI activation's own imbalance condition — ANY unfilled gap after the IC (decoupled from condition 3)
# ---------------------------------------------------------------------------

def _sweep(fills):
    """IC 4 inside the 61.8–80 zone of (BOS 0.6000, CTS 0.6100); CTS_ESTABLISHED anchor 5 / moment 6. Two sd gaps:
    the IC's IMMEDIATE one (c2 5, formed 6) and a later one (c2 8, formed 9); `fills` = each one's commit-fill idx
    (None = never)."""
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    df = pd.DataFrame([{"time": t0 + pd.Timedelta(hours=i), "o": 0.6040, "h": 0.6045, "l": 0.6035, "c": 0.6040}
                       for i in range(16)])
    df.loc[4, ["o", "h", "l", "c"]] = [0.6035, 0.6035, 0.6025, 0.6025]
    imbs = [ImbalanceInstance(start_idx=s, end_idx=s, direction=1, gap_top=0.6090, gap_bottom=0.6080,
                              gap_size=0.0010) for s in (5, 8)][:len(fills)]
    est = make_cts_established(cts_anchor_idx=5, confirmed_at=6, price=0.6100, struct_direction=None)
    hist = poi_zones._compute_poi_activation_history(
        df, ic_idx=4, cts_established_idx=6, sd=1, scan_end=15, bos_price=0.6000, cts_events=[est],
        config=POIConfig(), imbalances=imbs, fill_idx_cache={id(m): (None, f) for m, f in zip(imbs, fills)},
        lifecycle_floor_idx=None, pattern_confirmed_idx=None)
    return [(e["idx"], e["active"], e["reason"]) for e in hist]


def test_the_activation_keeps_any_unfilled_gap_after_the_ic():
    """The user (2026-10-13): "keeping a POI zone active centers more around that we currently still have unfilled
    imbalances in our current cycle" — the IC's immediate gap commit-filled at 10 while a later gap is unfilled: the
    POI STAYS active; with only the immediate gap it deactivates at 10."""
    assert _sweep([10, None]) == [(6, True, "initial")]
    assert _sweep([10]) == [(6, True, "initial"), (10, False, "imbalance_filled")]


def test_a_step_1a_poi_is_inactive_while_its_ic_touches_only_the_as_of_61_8_80_zone():
    """The band in the activation (decision 2): an IC (0.6025–0.6035) inside the as-of 61.8–80 zone of (BOS 0.6000,
    CTS 0.6100) = [0.6020, 0.60382] — NOT its 80–100 zone [0.6000, 0.6020] — is a Step 1A POI's non-touch until a raw
    CTS update at 11 (0.6200) moves the 80–100 zone up to [0.6000, 0.6040]: active only from 11. The 61.8–80 band
    (a primary POI) is active from 9 and leaves at 11 (the 61.8–80 zone moved above the IC). This is where 80–100
    and 61.8–100 differ — on the pick they cannot (a check touching 61.8–80 already failed condition 2 or 3)."""
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    df = pd.DataFrame([{"time": t0 + pd.Timedelta(hours=i), "o": 0.6040, "h": 0.6045, "l": 0.6035, "c": 0.6040}
                       for i in range(16)])
    df.loc[4, ["o", "h", "l", "c"]] = [0.6035, 0.6035, 0.6025, 0.6025]
    imb = ImbalanceInstance(start_idx=8, end_idx=8, direction=1, gap_top=0.6090, gap_bottom=0.6080, gap_size=0.0010)
    from engine_v2.structure.event_fields import CTS_UPDATED_RAW_VIA
    from engine_v2.tests._event_factory import make_event
    est = make_cts_established(cts_anchor_idx=5, confirmed_at=6, price=0.6100, struct_direction=None)
    upd = make_event("CTS_UPDATED", 11, price=0.6200, via=CTS_UPDATED_RAW_VIA, structure_id=0, cycle_id=0)

    def hist(band):
        h = poi_zones._compute_poi_activation_history(
            df, ic_idx=4, cts_established_idx=6, sd=1, scan_end=15, bos_price=0.6000, cts_events=[est, upd],
            config=ic_zone_config(POIConfig(), band), imbalances=[imb], fill_idx_cache={id(imb): (None, None)},
            lifecycle_floor_idx=None, pattern_confirmed_idx=None)
        return [(e["idx"], e["active"], e["reason"]) for e in h]

    assert hist(IC_ZONE_FALLBACK) == [(11, True, "initial")]
    assert hist(IC_ZONE_PRIMARY) == [(9, True, "initial"), (11, False, "ic_not_touching_fib_zone")]
