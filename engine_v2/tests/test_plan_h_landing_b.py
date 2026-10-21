"""PLAN_H landing B (Week 9 Part 1 item 1, 2026-10-06): the institutional-pattern pick is SWITCHED ON — the drawn
POI (one per fib record, `derive_poi_zones`) and MS's in-flight read (`poi_inner_for_fib` since PLAN_I S5; was
`compute_poi_inners_for_cycle`) use the same `select_ic` (D2) — with the rule-3 migration of the POI meta (`versions` / `current_versions` -> `ip_rank` + the
legs / tiebreak / `pattern_confirmed_idx`).

Pinned here (PLAN_H §8.6 "B", the census of §8.4a): the activation's cond5 = the touch of the AS-OF 61.8-80 zone
(reason `ic_not_touching_fib_zone`, R3; inclusive edges); the `pattern_confirmed_idx` floor (R1); the resolver
returns the PICK's inner where the V30 / V60 / V90 selection disagrees; its fail-loud scopes (PR1: an unstamped
frame raises; an IP error inside a sub geometry is not swallowed by its `except (ValueError, IndexError)`); D2 on
the real collaborator; the POI meta key order; the ONE hover builder (HV) and the H1 fib hover's "IP: none" (C1;
"IC: none" since PLAN_L L2); the exporter's `ip_rank` column. PLAN_L L2 (the ranking rules) rewrote the no-pattern
pins: a record without a pattern draws its RANKED IC, MS's read returns the ranked inner, all pinbars -> none. The landing review (PLAN_H §8.5c) added: the sell-side touch, the resolver's
graceful anchor scope (retired with it, PLAN_I S5), the candidate rows read from the handed records, the asymmetric leg order, and the RENDERED
hovers at all three POI sites (`_rendered`: what the template shows, not only what `customdata` holds).
"""
from __future__ import annotations

import contextlib
import io
import re
import traceback

import pandas as pd
import pytest

from engine_v2.tests._cross_fib_limits import main_limits

import engine_v2.zones.poi_zones as poi_zones
from engine_v2.charting._zone_render import poi_ic_hover_text
from engine_v2.common.types import ImbalanceInstance
from engine_v2.structure.event_fields import CTS_UPDATED_RAW_VIA
from engine_v2.structure.structure_engine import compute_bounded_structure
from engine_v2.tests._event_factory import make_cts_established, make_event
from engine_v2.tests.test_imbalance_c3_knowability import _df, _gap, _ip_ready
from engine_v2.tests.test_institutional_pattern import NO_1B_META
from engine_v2.tests.test_unified_probe import (
    _make_ip_moment_after_anchor_data,
    _make_ip_multicycle_data,
    _prepare_df,
)
from engine_v2.zones.poi_zones import ICSelectionError, POIConfig, fib_zone_bounds
from engine_v2.tests._cross_fib_limits import NO_P, NO_REF_TRACKER
from engine_v2.zones.cross_cycle_fib import ICReference


# ---------------------------------------------------------------------------
# the activation sweep: cond5 = the as-of touch; the R1 floor
# ---------------------------------------------------------------------------

def _sweep(*, ic=(0.6035, 0.6025), cts_events=None, pattern_confirmed_idx=4, n=16, imb_start=8):
    """IC 4 inside the 61.8-80 zone of (BOS 0.6000, CTS 0.6100) = [0.6020, 0.60382]; CTS_ESTABLISHED anchor 5 /
    moment 6; one sd FVG formed at `imb_start + 1` (never filled)."""
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    df = pd.DataFrame([
        {"time": t0 + pd.Timedelta(hours=i), "o": 0.6040, "h": 0.6045, "l": 0.6035, "c": 0.6040}
        for i in range(n)
    ])
    df.loc[4, ["o", "h", "l", "c"]] = [ic[0], ic[0], ic[1], ic[1]]
    imb = ImbalanceInstance(start_idx=imb_start, end_idx=imb_start, direction=1,
                            gap_top=0.6090, gap_bottom=0.6080, gap_size=0.0010)
    if cts_events is None:
        cts_events = [make_cts_established(cts_anchor_idx=5, confirmed_at=6, price=0.6100, struct_direction=None)]
    return poi_zones._compute_poi_activation_history(
        df, ic_idx=4, cts_established_idx=6, sd=1, scan_end=n - 1, bos_price=0.6000, cts_events=cts_events,
        config=POIConfig(), imbalances=[imb], fill_idx_cache={id(imb): (None, None)}, lifecycle_floor_idx=None,
        pattern_confirmed_idx=pattern_confirmed_idx,
    )


def _hist(history):
    return [(e["idx"], e["active"], e["reason"]) for e in history]


def test_sweep_deactivates_when_the_ic_stops_touching_the_as_of_zone():
    """A raw CTS update at 11 to .6200 moves the zone to [.6040, .61764] — above the IC (high .6035): the POI
    deactivates at 11 with R3's reason; before it, the touch of the first fib held (active at 9, the FVG)."""
    est = make_cts_established(cts_anchor_idx=5, confirmed_at=6, price=0.6100, struct_direction=None)
    upd = make_event("CTS_UPDATED", 11, price=0.6200, via=CTS_UPDATED_RAW_VIA, structure_id=0, cycle_id=0)
    assert _hist(_sweep(cts_events=[est, upd])) == [
        (9, True, "initial"), (11, False, "ic_not_touching_fib_zone"),
    ]


def test_sweep_activates_only_once_the_as_of_zone_reaches_the_ic():
    """The T1 window case (conf sub 3 cyc 2 / sub 7 cyc 2 on the reference window): the IC (.6060-.6050) misses
    the first fib's zone [.6020, .60382] and touches only after the raw update at 12 (.6150 -> [.60430, .60573])."""
    est = make_cts_established(cts_anchor_idx=5, confirmed_at=6, price=0.6100, struct_direction=None)
    upd = make_event("CTS_UPDATED", 12, price=0.6150, via=CTS_UPDATED_RAW_VIA, structure_id=0, cycle_id=0)
    assert _hist(_sweep(ic=(0.6060, 0.6050), cts_events=[est, upd])) == [(12, True, "initial")]
    assert _sweep(ic=(0.6060, 0.6050)) == []


@pytest.mark.parametrize("edge", ["high_at_zone_lo", "low_at_zone_hi"])
def test_sweep_touch_is_inclusive_at_both_edges(edge):
    lo, hi = fib_zone_bounds(0.6100, 0.6000, 1, POIConfig())
    ic = (lo, lo - 0.0005) if edge == "high_at_zone_lo" else (hi + 0.0005, hi)
    assert _hist(_sweep(ic=ic)) == [(9, True, "initial")]
    miss = (lo - 1e-9, lo - 0.0005) if edge == "high_at_zone_lo" else (hi + 0.0005, hi + 1e-9)
    assert _sweep(ic=miss) == []


@pytest.mark.parametrize("ic, expected", [
    ((0.6075, 0.6065), [(9, True, "initial")]),       # inside [.60618, .6080]
    ((0.6095, 0.6085), []),                           # above the zone (toward the BOS): no touch
])
def test_sweep_sell_side_touch(ic, expected):
    """The sd = -1 branch of the as-of fib (BOS .6100 above, CTS .6000 below -> zone [.60618, .6080]); a sell FVG
    formed at 9 (landing review A01: swapped sell anchors made every sell POI never active, unseen by the buy-only
    tests)."""
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    df = pd.DataFrame([{"time": t0 + pd.Timedelta(hours=i), "o": 0.6060, "h": 0.6065, "l": 0.6055, "c": 0.6060}
                       for i in range(16)])
    df.loc[4, ["o", "h", "l", "c"]] = [ic[1], ic[0], ic[1], ic[0]]
    imb = ImbalanceInstance(start_idx=8, end_idx=8, direction=-1, gap_top=0.6020, gap_bottom=0.6010, gap_size=0.0010)
    est = make_cts_established(cts_anchor_idx=5, confirmed_at=6, price=0.6000, struct_direction=None)
    hist = poi_zones._compute_poi_activation_history(
        df, ic_idx=4, cts_established_idx=6, sd=-1, scan_end=15, bos_price=0.6100, cts_events=[est],
        config=POIConfig(), imbalances=[imb], fill_idx_cache={id(imb): (None, None)}, lifecycle_floor_idx=None,
        pattern_confirmed_idx=4)
    assert _hist(hist) == expected


@pytest.mark.parametrize("pattern_confirmed_idx, expected", [
    (4, [(9, True, "initial")]),       # the IC itself: never binds
    (9, [(9, True, "initial")]),       # == the FVG's formation: masked
    (11, [(11, True, "initial")]),     # R1 decides: the continuation leg is known complete only at 11
    (16, []),                          # past the frame's last candle (15): never active
])
def test_sweep_first_active_takes_the_pattern_confirmation_moment(pattern_confirmed_idx, expected):
    assert _hist(_sweep(pattern_confirmed_idx=pattern_confirmed_idx)) == expected


# ---------------------------------------------------------------------------
# the MS resolver: the pick, fail loudly
# ---------------------------------------------------------------------------

def _m1_frame(*, extra_v_pick=True):
    """`test_imbalance_c3_knowability`'s M1 frame (IC 32 `2A.i`, inner 1.13) + a LATER bearish candle 36 fully
    inside the zone, with no pattern (its continuation would be true dojis only) — the V30 / V60 / V90 pick. Since
    PLAN_P each candidate has its IMMEDIATE unfilled gap (33 / 37 — IC condition 3; far below every later low, never
    filled). Since PLAN_S step 3 a never-filled gap AT the BOS anchor (30) keeps the frame neutral to Step 1B (every
    candidate has an unfilled prior gap — `test_institutional_pattern._bos_gap`)."""
    df = _df(60, [_gap(30, top=1.05, bottom=1.04), _gap(33, top=1.05, bottom=1.04), _gap(37, top=1.05, bottom=1.04),
                  _gap(40, top=1.30, bottom=1.20)])
    for i, base in zip(range(25, 30), (1.00, 1.02, 1.04, 1.06, 1.08)):
        df.loc[i, ["o", "h", "l", "c"]] = [base, base + 0.021, base - 0.001, base + 0.02]
    df.loc[31, ["o", "h", "l", "c"]] = [1.10, 1.121, 1.099, 1.12]
    df.loc[32, ["o", "h", "l", "c"]] = [1.12, 1.13, 1.08, 1.09]
    df.loc[33, ["o", "h", "l", "c"]] = [1.09, 1.201, 1.089, 1.20]
    if extra_v_pick:
        df.loc[36, ["o", "h", "l", "c"]] = [1.12, 1.125, 1.09, 1.10]
    return _ip_ready(df)


def _resolve(df):
    """MS's in-flight read on the M1 frame's fib (BOS 30 → CTS 40, sd +1) — since PLAN_I S5 `poi_inner_for_fib`
    on the embedded tracker's representative fib (was the retired `compute_poi_inners_for_cycle`)."""
    fs = poi_zones.FibState(structure_id=0, cycle_id=1, struct_direction=1, bos_idx=30, bos_price=1.0, cts_idx=40,
                            cts_price=1.35, active=True, locked=False, fib=_fib(1.35, 1.0))
    return poi_zones.poi_inner_for_fib(df, fs, 0.70, tracker=NO_REF_TRACKER, evaluated_at=40)


def test_resolver_returns_the_pick_where_the_variants_disagree():
    df = _m1_frame()
    fs = poi_zones.FibState(structure_id=0, cycle_id=1, struct_direction=1, bos_idx=30, bos_price=1.0, cts_idx=40,
                            cts_price=1.35, active=True, locked=False, fib=_fib(1.35, 1.0))
    cands = poi_zones.find_ic_variant_candidates(df, fs, POIConfig())
    assert cands == [32, 36]
    assert poi_zones.select_ic_variants(cands, df, fs, POIConfig()) == {36: ["V30", "V60", "V90"]}   # V: 36
    pick = poi_zones.select_ic(df, fs, POIConfig(), ic_reference=ICReference.NONE).pick
    assert pick is not None and (pick.ic_idx, pick.ip_rank) == (32, "2A.i")                         # IP: 32
    assert _resolve(df) == 1.13            # the pick's inner (IC 32's high), not 36's 1.125


def _fib(high, low):
    from engine_v2.features.fibonacci import DEFAULT_FIB_LEVELS, create_fib_retracement
    return create_fib_retracement(anchor_high=high, anchor_low=low, direction=1, levels=DEFAULT_FIB_LEVELS)


def _no_pattern_frame(*, pinbar_32=False, pinbar_36=False):
    """The M1 frame with a small continuation (33 not big_maru): neither IC candidate (32, 36 — both normal, each
    with its immediate gap) has a pattern; `pinbar_*` shrinks a candidate's body (same range) to a pinbar."""
    df = _m1_frame()
    df.loc[33, ["o", "h", "l", "c"]] = [1.09, 1.10, 1.089, 1.095]       # a small continuation: not big_maru
    if pinbar_32:
        df.loc[32, ["o", "h", "l", "c"]] = [1.095, 1.13, 1.08, 1.09]
    if pinbar_36:
        df.loc[36, ["o", "h", "l", "c"]] = [1.101, 1.125, 1.09, 1.098]
    return _ip_ready(df)


def test_resolver_returns_the_ranked_ic_when_no_candidate_has_a_pattern():
    """PLAN_L L2 (ICR-12 — one selector): no pattern -> MS reads the RANKED IC's inner. 32 and 36 both normal ->
    (PLAN_P) ranking rule 3: the reference = the earliest unfilled gap after the oldest in play (32) = its own
    immediate gap 33 -> the oldest, 32 (high 1.13; the user 2026-10-13: the last rule picks the oldest); a pinbar 32
    -> rule 1 excludes it -> 36 (high 1.125). (Was None until L2: no pattern -> no POI.)"""
    assert _resolve(_no_pattern_frame()) == 1.13
    assert _resolve(_no_pattern_frame(pinbar_32=True)) == 1.125       # ranking rule 1 excludes 32 -> 36
    fs = poi_zones.FibState(structure_id=0, cycle_id=1, struct_direction=1, bos_idx=30, bos_price=1.0, cts_idx=40,
                            cts_price=1.35, active=True, locked=False, fib=_fib(1.35, 1.0))
    sel = poi_zones.select_ic(_no_pattern_frame(), fs, POIConfig(), ic_reference=ICReference.NONE)
    assert (sel.pick.ic_idx, sel.ranking) == (32, ("most_recent_before_imbalance", 36, 2, 33))
    sel = poi_zones.select_ic(_no_pattern_frame(pinbar_36=True), fs, POIConfig(), ic_reference=ICReference.NONE)
    assert (sel.pick.ic_idx, sel.ranking) == (32, ("exclude_pinbar", None, 2, None))


def test_resolver_returns_nothing_when_every_ic_candidate_is_a_pinbar():
    """No pattern and every IC candidate a pinbar -> no IC (`all_ic_candidates_pinbar`): the BOS inner only."""
    df = _no_pattern_frame(pinbar_32=True, pinbar_36=True)
    assert [str(df.loc[i, "candle_type"]) for i in (32, 36)] == ["pinbar", "pinbar"]
    assert _resolve(df) is None


def test_resolver_on_an_unstamped_frame_raises():
    """PR1: a frame without the candle stamp is an error, never D7's legitimate "no IC" — and since PLAN_I S5
    (§9.2 F11) also on a refresh with NO fib (the read takes the stamp first on every call)."""
    df = _m1_frame()
    del df.attrs["timeframe"]
    with pytest.raises(ICSelectionError) as exc:
        _resolve(df)
    assert isinstance(exc.value.__cause__, KeyError) and exc.value.__cause__.args == ("timeframe",)
    with pytest.raises(ICSelectionError) as exc:
        poi_zones.poi_inner_for_fib(df, None, 0.70, tracker=NO_REF_TRACKER, evaluated_at=0)
    assert isinstance(exc.value.__cause__, KeyError) and exc.value.__cause__.args == ("timeframe",)


def test_resolver_reraises_any_ip_error_past_a_narrow_except(monkeypatch):
    """A ValueError inside the IP arrives as `ICSelectionError` (a RuntimeError) — not a ValueError / IndexError
    that a caller's narrow `except` would read as a legitimate failure."""
    def boom(*a, **k):
        raise ValueError("boom")

    monkeypatch.setattr(poi_zones, "select_ic", boom)
    with pytest.raises(ICSelectionError) as exc:
        _resolve(_m1_frame())
    assert not isinstance(exc.value, (ValueError, IndexError))
    assert isinstance(exc.value.__cause__, ValueError)


# (`test_resolver_anchor_selection_error_stays_graceful` retired, PLAN_I S5: the read selects no anchors — the
# fib is the embedded tracker's — so no graceful scope is left; a tracker error propagates,
# `test_plan_i_s5_ms_reads_the_fib.py`.)


def test_an_ip_error_inside_a_sub_geometry_propagates(monkeypatch):
    """The SECOND swallowing scope (PLAN_H §8.5 step 3): `_build_geometry` wraps the sub's MS run in `except
    (ValueError, IndexError): return None` — the whole sub would vanish (`geometry_failed`). An IP error inside
    the MS resolver must reach the caller instead."""
    from engine_v2.multitf import entity_df_mutation as edm
    from engine_v2.structure.reference_zone import build_ad_hoc_bos0_reference_zone
    from engine_v2.tests.test_geometry_builder import _make_reversing_data

    df = _prepare_df(_make_reversing_data())
    inner = float(build_ad_hoc_bos0_reference_zone(df, 0, 1).inner)
    calls = []
    real = poi_zones.select_ic
    monkeypatch.setattr(poi_zones, "select_ic", lambda *a, **k: calls.append(1) or real(*a, **k))
    with contextlib.redirect_stdout(io.StringIO()):
        assert edm._build_geometry(df, sd=1, start_abs=0, run_cap_abs=len(df) - 1, bos0_inner=inner,
                                   timeframe="M15", cross_fib_limit=NO_P) is not None
    assert calls, "the fixture's MS run reads the resolver"

    def boom(*a, **k):
        raise IndexError("boom")

    monkeypatch.setattr(poi_zones, "select_ic", boom)
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(ICSelectionError):
        edm._build_geometry(df, sd=1, start_abs=0, run_cap_abs=len(df) - 1, bos0_inner=inner, timeframe="M15", cross_fib_limit=NO_P)


def test_ms_on_an_unstamped_frame_raises():
    """Every MS run (main, sub geometry, probe Phase 2) reads the resolver: an unstamped frame fails the run."""
    df = _prepare_df(_make_ip_moment_after_anchor_data())
    del df.attrs["timeframe"]
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(ICSelectionError):
        compute_bounded_structure(df, 0, +1, cross_fib_limit=NO_P)


def test_probe_phase2_on_an_unstamped_frame_raises(monkeypatch):
    """PLAN_H §8.5 step 3's pin, through the probe itself (landing review C2): `unified_probe`'s Phase-2 MS
    (`_make_market_structure` on `df_probe`) reads the resolver, and an unstamped frame fails the probe — no
    scope between them swallows `ICSelectionError`."""
    from engine_v2.structure.unified_probe import unified_probe
    from engine_v2.tests.test_unified_probe import _make_multicycle_data, _ref_zone_uptrend
    import engine_v2.structure.structure_engine as se

    calls = []
    real = se.poi_inner_for_fib
    monkeypatch.setattr(se, "poi_inner_for_fib", lambda *a, **k: calls.append(1) or real(*a, **k))
    df = _prepare_df(_make_multicycle_data())
    with contextlib.redirect_stdout(io.StringIO()):
        unified_probe(df, 0, 1, _ref_zone_uptrend(), 14, "H1", enable_phase2=True)
    assert calls, "the fixture's Phase-2 MS reads the resolver"

    df = _prepare_df(_make_multicycle_data())
    del df.attrs["timeframe"]
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(ICSelectionError) as exc:
        unified_probe(df, 0, 1, _ref_zone_uptrend(), 14, "H1", enable_phase2=True)
    assert "_run_phase2" in [f.name for f in traceback.extract_tb(exc.tb)]


# ---------------------------------------------------------------------------
# D2 on the real collaborator: the drawn POI == the resolver's pick on the final fib
# ---------------------------------------------------------------------------

def _sell(maker):
    """The sd = -1 mirror of a fixture (`test_ms_cts_update_no_regress._mirror`: prices reflected)."""
    from engine_v2.tests.test_ms_cts_update_no_regress import _mirror

    def rows():
        return _mirror(maker())
    rows.__name__ = f"{maker.__name__}_sell"
    return rows


def _pipeline(maker, *, spy=None, sd=+1):
    from engine_v2.pipeline.orchestrator import _run_downstream_pipeline
    import engine_v2.structure.structure_engine as se
    df = _prepare_df(maker())
    real = se.poi_inner_for_fib
    calls = []

    def _spy(df, fib, fill_threshold, **kw):
        out = real(df, fib, fill_threshold, **kw)
        if fib is not None:
            calls.append((int(fib.structure_id), int(fib.cycle_id), int(fib.bos_idx), int(fib.cts_idx), out))
        return out

    if spy is not None:
        spy.setattr(se, "poi_inner_for_fib", _spy)
    with contextlib.redirect_stdout(io.StringIO()):
        res = compute_bounded_structure(df, 0, sd, cross_fib_limit=NO_P)
        out = _run_downstream_pipeline(res.df, res.events, sd, wvmi="off", cross_fib_limits=main_limits(res.events))
    return res, out, calls


@pytest.mark.parametrize("maker, key, sd", [(_make_ip_moment_after_anchor_data, (0, 1), +1),
                                            (_make_ip_multicycle_data, (0, 2), +1),
                                            (_sell(_make_ip_moment_after_anchor_data), (0, 1), -1)])
def test_drawn_poi_is_the_ms_resolvers_pick_on_the_final_fib(monkeypatch, maker, key, sd):
    res, out, calls = _pipeline(maker, spy=monkeypatch, sd=sd)
    poi, = [z for z in out["poi_zones"] if (z.meta["structure_id"], z.meta["cycle_id"]) == key]
    assert (poi.ic_idx, poi.side) == (9 if key == (0, 1) else 14, "buy" if sd == 1 else "sell")
    final = [c for c in calls if (c[0], c[1]) == key and (c[2], c[3]) == (poi.meta["bos_anchor_idx"],
                                                                         poi.meta["cts_anchor_idx"])]
    assert final, "MS read the resolver on the final fib's anchors"
    # the inner: a buy POI's top, a sell POI's bottom (the IC's high / low)
    assert final[-1][4] == (float(poi.top) if sd == 1 else float(poi.bottom))


def test_poi_meta_carries_the_pick_in_place_of_the_variants():
    _, out, _ = _pipeline(_make_ip_moment_after_anchor_data)
    poi, = out["poi_zones"]
    assert list(poi.meta) == [
        "structure_id", "struct_direction", "cycle_id", "confirmed_idx", "end_idx", "end_reason",
        "ip_outcome", "ic_zone_band", "ip_rank", "ip_leg_candle_counts", "pattern_anchor_idx", "opp_leg_first_idx",
        "pattern_end_idx", "pattern_confirmed_idx", "ip_tiebreak", "ip_tb_runner_up_ic_idx", "ip_tied_count",
        "ip_tb_imbalance_first_c2_idx", "ranking_rule", "ranking_runner_up_ic_idx", "ranking_in_play_count",
        "ranking_imbalance_first_c2_idx",
        "prior_imbalance_outcome", "prior_imbalance_first_c2_idx", "prior_imbalance_passer_count", "ic_candidate_count",
        "prior_imbalance_outranked_ic_idx", "prior_imbalance_outranked_ip_rank",            # PLAN_S step 3
        "status", "activation_history", "bos_anchor_idx", "cts_anchor_idx",
        "cts_established_idx",
    ]
    assert poi.meta["ip_outcome"] == "pattern"                                       # PLAN_L L2
    assert poi.meta["ic_zone_band"] == "61.8-80"                                     # PLAN_P: Step 1 found it
    assert (poi.meta["ranking_rule"], poi.meta["ranking_runner_up_ic_idx"], poi.meta["ranking_in_play_count"],
            poi.meta["ranking_imbalance_first_c2_idx"]) == (None, None, None, None)
    assert (poi.ic_idx, poi.meta["ip_rank"], poi.meta["ip_leg_candle_counts"]) == (9, "1A.iii", "1-1-3")
    assert (poi.meta["pattern_anchor_idx"], poi.meta["opp_leg_first_idx"], poi.meta["pattern_end_idx"],
            poi.meta["pattern_confirmed_idx"]) == (8, 9, 12, 12)
    assert all(type(poi.meta[k]) is int for k in ("pattern_anchor_idx", "opp_leg_first_idx", "pattern_end_idx",
                                                  "pattern_confirmed_idx"))
    assert (poi.meta["ip_tiebreak"], poi.meta["ip_tb_runner_up_ic_idx"], poi.meta["ip_tied_count"],
            poi.meta["ip_tb_imbalance_first_c2_idx"]) == (None, None, None, None)
    assert all(set(e) == {"idx", "active", "reason"} for e in poi.meta["activation_history"])


def test_a_record_without_a_pattern_draws_its_ranked_ic():
    """PLAN_L L2 (was `test_a_record_without_a_pattern_draws_no_poi`): cycles 1 and 3 have no pattern -> the ranking
    rules; each has exactly ONE IC candidate, a maru (7 / 19) -> drawn with `ranking_rule` None (ICR-10 — "only IC
    candidate"), their pattern meta all None; cycle 0 has no IC candidate -> no POI, the no-IC row."""
    _, out, _ = _pipeline(_make_ip_multicycle_data)
    got = [(z.meta["cycle_id"], z.ic_idx, z.meta["ip_outcome"], z.meta["ip_rank"], z.meta["ranking_rule"])
           for z in out["poi_zones"]]
    assert got == [(1, 7, "no_pattern", None, None), (2, 14, "pattern", "1A.iii", None),
                   (3, 19, "no_pattern", None, None)]
    for z in out["poi_zones"]:
        if z.meta["ip_outcome"] == "no_pattern":
            assert [z.meta[k] for k in ("ip_leg_candle_counts", "pattern_anchor_idx", "opp_leg_first_idx",
                                        "pattern_end_idx", "pattern_confirmed_idx")] == [None] * 5
    none_rows = [(r.cycle_id, r.reason, r.ip_outcome) for r in out["ic_candidates"] if r.ic_idx is None]
    # cycle 0: main's single since PLAN_I S4 (Q6), no candidates — Step 2 never ran
    assert none_rows == [(0, "no_ic_candidate", None)]


def test_ic_candidates_read_the_records_they_are_handed():
    """ONE selection per pipeline run (§8.5a deviation 2): `derive_ic_candidates` builds its rows from the
    `ic_records` it is handed, never re-selects (landing review D01)."""
    res, out, _ = _pipeline(_make_ip_moment_after_anchor_data)
    assert out["ic_candidates"]
    assert poi_zones.derive_ic_candidates(res.df, res.events, out["fib_tracker"], POIConfig(),
                                          structure_path_id="H1.main", ic_records=[]) == []


# ---------------------------------------------------------------------------
# HV: the one hover builder; C1: the H1 fib hover's "IP: none" ("IC: none" since PLAN_L L2)
# ---------------------------------------------------------------------------

_CUSTOMDATA_REF = re.compile(r"%\{customdata\[(\d+)\](?::[^}]*)?\}")


def _rendered(tr, k=0):
    """A trace's hover as Plotly shows it: every `%{customdata[i]}` substituted with row `k`'s cell (Plotly's hover
    template has no `%%` escape — PLAN_H §8.9 F1·1). Pins what the TEMPLATE displays, not only what `customdata`
    holds (landing review P01 / M01 / M09)."""
    row = tr.customdata[k]
    return _CUSTOMDATA_REF.sub(lambda m: str(row[int(m.group(1))]), tr.hovertemplate)


def _meta(rank, counts, anchor, opp_first, end, tb=None, runner=None, c2=None, band="61.8-80"):
    """A pattern IC's POI meta (the hover's keys; `ip_outcome` "pattern", no ranking — PLAN_L L2; the IC's zone band,
    PLAN_P)."""
    return {"ip_outcome": "pattern", "ip_rank": rank, "ip_leg_candle_counts": counts, "pattern_anchor_idx": anchor,
            "opp_leg_first_idx": opp_first, "pattern_end_idx": end, "ip_tiebreak": tb,
            "ip_tb_runner_up_ic_idx": runner, "ip_tb_imbalance_first_c2_idx": c2, "ranking_rule": None,
            "ic_zone_band": band, **NO_1B_META}


def test_hover_single_legs_print_one_index_each():
    assert poi_ic_hover_text(_meta("1A.i", "1-1-1", 2703, 2704, 2705), 2704) == "IP 1A.i · legs 1-1-1 (2703 | 2704 | 2705)"


def test_hover_asymmetric_legs_keep_their_order():
    """initial | opp | continuation, never reordered (landing review H01: every other case is symmetric)."""
    assert poi_ic_hover_text(_meta("1A.iii", "1-1-3", 8, 9, 12), 9) == "IP 1A.iii · legs 1-1-3 (8 | 9 | 10–12)"


def test_hover_multi_legs_print_ranges():
    assert poi_ic_hover_text(_meta("3C.iv", "3-2-3", 683, 686, 690), 687) == (
        "IP 3C.iv · legs 3-2-3 (683–685 | 686–687 | 688–690)")


def test_hover_tiebreak_lines():
    assert poi_ic_hover_text(_meta("1A.iv", "2-1-2", 4081, 4083, 4085, "most_recent_before_imbalance", 4136, 4132),
                             4083) == ("IP 1A.iv · legs 2-1-2 (4081–4082 | 4083 | 4084–4085)"
                                       "<br>Tiebreak over 4136: most recent before imbalance 4132")
    assert poi_ic_hover_text(_meta("2A.i", "1-1-1", 9, 10, 11, "ic_candle_type", 14), 10) == (
        "IP 2A.i · legs 1-1-1 (9 | 10 | 11)<br>Tiebreak over 14: IC candle type")
    with pytest.raises(AssertionError, match="unknown ip_tiebreak 'TB1'"):     # the pre-TBU-3 values are gone
        poi_ic_hover_text(_meta("2A.i", "1-1-1", 9, 10, 11, "TB1", 14), 10)
    # PLAN_P: the immediate-unfilled-imbalance tiebreaker became IC condition 3 — its value is gone too
    with pytest.raises(AssertionError, match="unknown ip_tiebreak 'immediate_unfilled_imbalance'"):
        poi_ic_hover_text(_meta("1A.iv", "3-1-3", 1657, 1660, 1663, "immediate_unfilled_imbalance", 1447, 1662), 1660)


def test_hover_rejects_legs_that_do_not_match_their_indices():
    """A mirror that shifted the IC but not a leg index (or vice versa) is caught at render."""
    with pytest.raises(AssertionError):
        poi_ic_hover_text(_meta("1A.i", "1-1-1", 2703, 2704, 2705), 2704 + 50)
    with pytest.raises(AssertionError):
        poi_ic_hover_text(_meta("1A.i", "1-1-1", 2703, 2704, 2705, "TB3"), 2704)


def test_h1_chart_hovers_show_the_pattern_the_ranking_and_ic_none(tmp_path, monkeypatch):
    """The H1 chart: the POI hover lines carry the builder's text — the pattern's (cycle 2) and, since PLAN_L L2, a
    ranked IC's `No pattern · only IC candidate` (cycles 1 / 3, no `IC <idx>` — ICR-15); the fib hovers: the
    pattern pick `IP <rank> · IC <idx>`, a ranked pick `No pattern · IC <idx> · only IC candidate` (the index kept),
    the record without an IC candidate `IC: none (no_ic_candidate)` (was `IP: none (no_pattern)` on cycle 1)."""
    go = pytest.importorskip("plotly.graph_objects")
    from engine_v2.charting.export_plotly import export_chart_plotly
    from engine_v2.multitf.registry import StructureRegistry

    res, out, _ = _pipeline(_make_ip_multicycle_data)
    df = res.df.copy()
    df.attrs.update(structure_events=res.events, fib_states=out["fib_states"], poi_zones=out["poi_zones"],
                    ic_candidates=out["ic_candidates"], kl_zones=out["kl_zones"])
    reg = StructureRegistry()
    reg.register("H1", df=df, timeframe="H1", role="main")
    figs = []
    monkeypatch.setattr(go.Figure, "write_html", lambda self, *a, **k: figs.append(self))
    monkeypatch.setattr(go.Figure, "write_image", lambda self, *a, **k: None)
    cfg = {"structure": {"levels": True}, "struct_state": {"labels": False},
           "range_visual": {"rectangles": False}, "fib": {"lines": True},
           "imbalance": {"highlight": False}, "zones": {"wave_candles": False, "POI": True},
           "volume": {"bars": False, "ema_line": False, "spike_marker": False}}
    with contextlib.redirect_stdout(io.StringIO()):
        export_chart_plotly(title="t", registry=reg, path_id="H1", out_dir=tmp_path, cfg=cfg)
    fig, = figs
    by_ic = {z.ic_idx: z for z in out["poi_zones"]}
    assert sorted(by_ic) == [7, 14, 19]
    poi_tr = [tr for tr in fig.data if (tr.name or "").startswith("POI zone")]
    assert len(poi_tr) == 2 * len(by_ic) and {tr.customdata[0][3] for tr in poi_tr} == set(by_ic)
    for tr in poi_tr:
        z = by_ic[tr.customdata[0][3]]
        ip = poi_ic_hover_text(z.meta, z.ic_idx)
        assert all(c[6] == ip for c in tr.customdata)
        assert f"<br>{ip}<br>top=" in _rendered(tr)                               # the template shows it
        assert "versions=" not in tr.hovertemplate
    assert poi_ic_hover_text(by_ic[7].meta, 7) == "No pattern · only IC candidate"
    assert poi_ic_hover_text(by_ic[14].meta, 14).startswith("IP 1A.iii · legs ")
    fib_tr = [tr for tr in fig.data if (tr.name or "").startswith("fib:")]
    fib_status = {tr.name: tr.customdata[0][8] for tr in fib_tr}
    # the IC line, then the IC reference, then S2's limit line (main's limits since PLAN_I S4: sid 0 started by
    # start trading)
    lines = fib_status["fib:sid0:c1"].split("<br>")
    assert lines[2:] == ["No pattern · IC 7 · only IC candidate", lines[3], "limit: none (start_trading)"]
    assert lines[3].startswith("IC ref ") and "(prev_cycle_cts, cycle 0)" in lines[3]   # PLAN_K A17
    assert fib_status["fib:sid0:c2"].split("<br>")[2] == "IP 1A.iii · IC 14"
    assert fib_status["fib:sid0:c0"].split("<br>")[2] == "IC: none (no_ic_candidate)"
    assert not any("IP: none" in v or "IP: not evaluated" in v for v in fib_status.values())
    # F1·1: a literal `%` (Plotly has no `%%` escape)
    assert all("<br>61.8%=" in _rendered(tr) and "<br>80%=" in _rendered(tr) and "%%" not in tr.hovertemplate
               for tr in fib_tr)


def test_poi_meta_copies_the_selections_tiebreak():
    """`derive_poi_zones` copies the record's tiebreak tuple into the POI meta (rule, runner-up IC, tied count,
    the imbalance's first c2) — the reference window's tied POIs (conf 1660 / 1875 / 4083) go through it."""
    import dataclasses
    res, out, _ = _pipeline(_make_ip_moment_after_anchor_data)
    recs = poi_zones.select_ic_records(res.df, res.events, out["fib_tracker"], POIConfig())
    assert [r.selection.pick.ic_idx for r in recs if r.selection.pick is not None] == [9]
    recs = [dataclasses.replace(r, selection=dataclasses.replace(
        r.selection, tiebreak=("most_recent_before_imbalance", 7, 2, 11)))
            if r.selection.pick is not None else r for r in recs]     # a tiebreak needs a pattern pick (PLAN_L L2)
    with contextlib.redirect_stdout(io.StringIO()):
        poi, = poi_zones.derive_poi_zones(res.df, res.events, fib_tracker=out["fib_tracker"], config=POIConfig(),
                                          ic_records=recs)
    assert (poi.meta["ip_tiebreak"], poi.meta["ip_tb_runner_up_ic_idx"], poi.meta["ip_tied_count"],
            poi.meta["ip_tb_imbalance_first_c2_idx"]) == ("most_recent_before_imbalance", 7, 2, 11)


def test_drawn_poi_activation_and_meta_take_the_picks_confirmation_moment():
    """The R1 floor through `derive_poi_zones` (the call site + the meta): on the fixture the pattern is confirmed at
    12, before the moment 13, so R1 never binds there — a pick confirmed at 14 must move the first activation to 14
    and be exported as such (not the IC, not `pattern_end_idx` 12)."""
    import dataclasses
    res, out, _ = _pipeline(_make_ip_moment_after_anchor_data)
    recs = poi_zones.select_ic_records(res.df, res.events, out["fib_tracker"], POIConfig())
    pick = next(r.selection.pick for r in recs if r.selection.pick is not None)
    assert (pick.ic_idx, pick.pattern_end_idx, pick.pattern_confirmed_idx) == (9, 12, 12)
    late = dataclasses.replace(pick, pattern_confirmed_idx=14)
    recs = [dataclasses.replace(r, selection=dataclasses.replace(r.selection, pick=late))
            if r.selection.pick is not None else r for r in recs]
    with contextlib.redirect_stdout(io.StringIO()):
        poi, = poi_zones.derive_poi_zones(res.df, res.events, fib_tracker=out["fib_tracker"], config=POIConfig(),
                                          ic_records=recs)
    assert _hist(poi.meta["activation_history"]) == [(14, True, "initial"), (15, False, "imbalance_filled")]
    assert (poi.meta["pattern_confirmed_idx"], poi.meta["pattern_end_idx"]) == (14, 12)


def test_m15_lens_sub_poi_hover_is_rendered(tmp_path, monkeypatch):
    """The third POI hover site: a lens chart RENDERED with a sub POI (landing review M01 / M02 — it replaces a
    source pin that passed with the whole sub-POI block skipped)."""
    from engine_v2.tests import test_fib_chart_render as F
    from engine_v2.zones.poi_zones import POIZone
    meta = {"sub_id": 5, "structure_id": 0, "cycle_id": 1, "struct_direction": 1, "status": "active",
            "confirmed_idx": 22, "end_idx": None, "activation_history": [{"idx": 22, "active": True, "reason": "initial"}],
            **_meta("1A.iii", "1-1-3", 17, 18, 21)}
    poi = POIZone(start_time=F._ohlc(48, "15min").loc[18, "time"], end_time=None, side="buy", top=0.6010,
                  bottom=0.6000, ic_idx=18, meta=meta)
    fig = F._render_m15(monkeypatch, tmp_path, {"zones": {"KL": False, "POI": True, "wave_candles": False}},
                        poi_zones=[poi])
    poi_tr = [tr for tr in fig.data if tr.name == "M15 POI zone"]
    ip = poi_ic_hover_text(meta, 18)
    assert ip == "IP 1A.iii · legs 1-1-3 (17 | 18 | 19–21)"
    assert len(poi_tr) == 2 and all(f"<br>{ip}<br>top=" in _rendered(tr) for tr in poi_tr)
    assert all("versions" not in tr.hovertemplate for tr in poi_tr)


def test_h1_overlay_poi_hover_uses_the_builder():
    """The M15 charts' H1 overlay (`export_m15_chart._render_h1_overlay`) — one of the three POI hover sites."""
    go = pytest.importorskip("plotly.graph_objects")
    from engine_v2.charting.export_m15_chart import _render_h1_overlay
    res, out, _ = _pipeline(_make_ip_moment_after_anchor_data)
    h1 = res.df.copy()
    h1.attrs.update(structure_events=res.events, poi_zones=out["poi_zones"], kl_zones=out["kl_zones"])
    m15 = pd.DataFrame({"time": pd.date_range(h1["time"].iloc[0], periods=len(h1) * 4, freq="15min", tz="UTC")})
    fig = go.Figure()
    _render_h1_overlay(fig, m15, h1, {}, {}, {"labels": False}, {"levels": False},
                       {"KL": False, "POI": True, "wave_candles": False})
    poi, = out["poi_zones"]
    trs = [t for t in fig.data if t.name == "H1 POI zone (overlay)"]
    assert len(trs) == 2
    ip = poi_ic_hover_text(poi.meta, poi.ic_idx)
    assert all(c[6] == ip for t in trs for c in t.customdata)
    assert all(f"<br>{ip}<br>top=" in _rendered(t) for t in trs)
    assert all("versions=" not in t.hovertemplate for t in trs)


# ---------------------------------------------------------------------------
# the exporter (the save-format boundary)
# ---------------------------------------------------------------------------

def test_poi_export_columns(tmp_path):
    from engine_v2.debug.export_zones import export_poi_zones
    path = tmp_path / "poi.csv"
    export_poi_zones([], path)
    assert pd.read_csv(path).columns.tolist() == [
        "ic_idx", "start_time", "end_time", "side", "top", "bottom", "structure_id", "cycle_id", "parent_sid",
        "parent_cycle_id", "confirmed_idx", "ip_rank", "status", "meta",
    ]
    _, out, _ = _pipeline(_make_ip_moment_after_anchor_data)
    export_poi_zones(out["poi_zones"], path)
    assert pd.read_csv(path)["ip_rank"].tolist() == ["1A.iii"]
