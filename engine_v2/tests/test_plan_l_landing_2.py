"""PLAN_L landing 2 — Step 3, the ranking rules (`plans/PLAN_L_ic_identification_steps.md` §4; the decisions PLAN_H §6
ICR-*): when Step 2 finds no institutional pattern, the IC is ranked over the UNFILTERED IC candidates — (1) exclude
pinbars, (2) the immediate unfilled imbalance (a filter), (3) maru > normal, (4) the most recent before the earliest
unfilled imbalance after the oldest candidate STILL IN PLAY — a rule set of its own (ICR-S), sharing one helper per
identical rule with the tiebreak rules, each given its own input list (ICR-9). Pinned: each rule deciding and passing
through; ICR-1 / ICR-8 / ICR-10; all pinbars -> no IC; Step 3 never runs with a pattern; MS's inner from a ranked
pick (one selector, ICR-12); the activation floor without the pattern term (ICR-2); the POI meta (`ip_outcome`, the
`ranking_*` keys, the pattern meta nulled), its mirror shift; the CSV columns; the POI hover's ranked line.

The fixtures are real classified frames (`test_institutional_pattern._frame`: 6 prefix marus, then the specs from
index 6; `S` = a small sd spacer — a 1-pip body, a pinbar, never a candidate), every candidate touching the zone.
"""
from __future__ import annotations

import contextlib
import dataclasses
import io

import pandas as pd
import pytest

import engine_v2.zones.poi_zones as poi_zones
from engine_v2.charting._zone_render import poi_ic_hover_text
from engine_v2.debug.export_zones import IC_CANDIDATE_COLUMNS, export_ic_candidates
from engine_v2.multitf import entity_df_mutation as edm
from engine_v2.tests.test_institutional_pattern import D, U, _est, _far_gap, _fib_over, _frame, _select, _Tracker
from engine_v2.zones.cross_cycle_fib import ICReference
from engine_v2.zones.poi_zones import (
    ALL_IC_CANDIDATES_PINBAR,
    IP_OUTCOME_NO_PATTERN,
    IP_OUTCOME_PATTERN,
    NO_IC_CANDIDATE,
    RANKING_EXCLUDE_PINBAR,
    RANKING_IC_CANDLE_TYPE,
    RANKING_IMMEDIATE_UNFILLED_IMBALANCE,
    RANKING_MOST_RECENT_BEFORE_IMBALANCE,
    ICSelection,
    POIConfig,
    apply_ranking_rules,
    apply_tiebreak_rules,
    derive_ic_candidates,
    derive_poi_zones,
)

S = ("N", U, 2)      # a small sd spacer (body 1 pip < the H1 floor: a pinbar; agreed sd — never a candidate)

# name: (specs, CTS anchor, the far (never filled) sd gaps' first c2) — BOS anchor 6 for every fixture
FIX = {
    # 7 pinbar, 9 maru, 11 pinbar with the only immediate gap (12) — the H1 sid 0 cyc 0 shape (177 / 180 / 183)
    "pinbar_with_the_gap": ([("M", U, 10), ("P", D, 6), S, ("M", D, 6), S, ("P", D, 6), S], 12, [12]),
    # 7 pinbar (gap 8 right after it), marus 9 / 13 / 17 with no immediate gap; gaps 16, 20 — the sub 0 cyc 0 shape
    "oldest_in_play": ([("M", U, 10), ("P", D, 6), S, ("M", D, 6), S, S, S, ("M", D, 6), S, S, S, ("M", D, 6),
                        S, S, S], 20, [8, 16, 20]),
    # 7 normal with an immediate gap (8), 9 maru without one (12 = IC + 3)
    "one_immediate": ([("M", U, 10), ("N", D, 6), S, ("M", D, 6), S, S, S], 12, [8, 12]),
    # 7 normal with the only immediate gap (8); marus 9 / 11 without (14 = 11 + 3)
    "one_immediate_of_three": ([("M", U, 10), ("N", D, 6), S, ("M", D, 6), S, ("M", D, 6), S, S, S], 14, [8, 14]),
    # 7 normal with the only immediate gap (8), 9 maru, 11 a PINBAR (excluded by rule 1)
    "runner_up_not_a_pinbar": ([("M", U, 10), ("N", D, 6), S, ("M", D, 6), S, ("P", D, 6), S, S, S], 14, [8, 14]),
    # 7 maru (gap 8), 9 normal (gap 10), 11 normal without one — rule 2 leaves 7 / 9
    "runner_up_still_in_play": ([("M", U, 10), ("M", D, 6), S, ("N", D, 6), S, ("N", D, 6), S, S, S], 14,
                                [8, 10, 14]),
    # 7 normal (gap 8) and 13 normal (gap 14) pass rule 2; 9 maru fails it
    "two_immediate": ([("M", U, 10), ("N", D, 6), S, ("M", D, 6), S, S, S, ("N", D, 6), S, S], 15, [8, 14]),
    # 7 normal, 9 maru, no immediate gap (12 = IC + 3 / + 5)
    "no_immediate": ([("M", U, 10), ("N", D, 6), S, ("M", D, 6), S, S, S], 12, [12]),
    # marus 7 and 8 share the gap 9 (8 its c1) — ICR-1
    "neighbours": ([("M", U, 10), ("M", D, 6), ("M", D, 6), S, S], 10, [9]),
    # one candidate, a maru — ICR-10
    "single": ([("M", U, 10), ("M", D, 6), S, S], 9, [9]),
    # 7 and 9 pinbars
    "all_pinbar": ([("M", U, 10), ("P", D, 6), S, ("P", D, 6), S], 10, [10]),
    # 7 a pinbar WITH a pattern (3A.i: the big maru 8 continues it); 9 a maru without one
    "pattern_pinbar": ([("M", U, 10), ("P", D, 6), ("M", U, 10), ("M", D, 6), S, S], 11, [11]),
    # normals 7 / 11 around the maru 9, no immediate gap (14)
    "two_normals": ([("M", U, 10), ("N", D, 6), S, ("M", D, 6), S, ("N", D, 6), S, S, S], 14, [14]),
    # 7 normal, marus 11 / 15, 19 normal; no immediate gap; gaps 10 / 14 / 18 / 22 (each IC + 3)
    "rule_4_after_rule_3": ([("M", U, 10), ("N", D, 6), S, S, S, ("M", D, 6), S, S, S, ("M", D, 6), S, S, S,
                             ("N", D, 6), S, S, S], 22, [10, 14, 18, 22]),
    # 7 an IC candidate (gap 9 after it); 10 a check failing condition 3 (the CTS anchor: no gap after it)
    "with_a_non_candidate": ([("M", U, 10), ("M", D, 6), S, S, ("M", D, 6)], 10, [9]),
}


def _sel(name, *, classify_all=False):
    specs, cts, gaps = FIX[name]
    df = _frame(specs)
    return _select(df, 6, cts, [_far_gap(g) for g in gaps], classify_all=classify_all), df


def _outcome(sel):
    return (None if sel.pick is None else sel.pick.ic_idx, sel.ip_outcome, sel.ranking, sel.no_ic_reason)


# ---------------------------------------------------------------------------
# Each ranking rule: the decider and the pass-through (select_ic — both consumers' classify modes)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("classify_all", [False, True])
@pytest.mark.parametrize("name, expected", [
    # rule 1 decides: 7 / 11 excluded -> 9; rules 2–4 never run -> no runner-up (ICR-14); 3 entered rule 1
    ("pinbar_with_the_gap", (9, "no_pattern", (RANKING_EXCLUDE_PINBAR, None, 3, None), None)),
    # rule 2 decides: only 7 has an immediate gap -> 7 over 9 (the maru rule 3 would take); 2 entered; its gap 8
    ("one_immediate", (7, "no_pattern", (RANKING_IMMEDIATE_UNFILLED_IMBALANCE, 9, 2, 8), None)),
    # ... its runner-up = the MOST RECENT candidate in play failing it (11, not 9); 3 entered rule 2
    ("one_immediate_of_three", (7, "no_pattern", (RANKING_IMMEDIATE_UNFILLED_IMBALANCE, 11, 3, 8), None)),
    # ... the runner-up is a candidate IN PLAY (9) — never the more recent pinbar 11 that rule 1 excluded
    ("runner_up_not_a_pinbar", (7, "no_pattern", (RANKING_IMMEDIATE_UNFILLED_IMBALANCE, 9, 2, 8), None)),
    # rule 3's runner-up = the most recent normal STILL IN PLAY (9) — not 11, which failed rule 2
    ("runner_up_still_in_play", (7, "no_pattern", (RANKING_IC_CANDLE_TYPE, 9, 2, None), None)),
    # rule 2 with two passing: only 7 / 13 go on (9 the maru out) -> rule 3 ties (both normal) -> rule 4: the
    # reference after the oldest in play (7) = 8 -> 7; runner-up 13; 2 entered rule 4
    ("two_immediate", (7, "no_pattern", (RANKING_MOST_RECENT_BEFORE_IMBALANCE, 13, 2, 8), None)),
    # rule 2 with none passing: all go on -> rule 3 decides: 9 the maru over the most recent normal 7
    ("no_immediate", (9, "no_pattern", (RANKING_IC_CANDLE_TYPE, 7, 2, None), None)),
    # ICR-1: both pass rule 2 and are marus -> rule 4: the reference after 7 = the shared gap 9 -> 8, closer to it
    ("neighbours", (8, "no_pattern", (RANKING_MOST_RECENT_BEFORE_IMBALANCE, 7, 2, 9), None)),
    # ICR-10: one IC candidate, not a pinbar -> nothing ranked
    ("single", (7, "no_pattern", None, None)),
    # every IC candidate a pinbar -> no IC
    ("all_pinbar", (None, "no_pattern", None, ALL_IC_CANDIDATES_PINBAR)),
    # rule 3's runner-up = the MOST RECENT normal (11, not 7)
    ("two_normals", (9, "no_pattern", (RANKING_IC_CANDLE_TYPE, 11, 3, None), None)),
    # rule 4 runs over rule 3's survivors (11 / 15): the reference after 11 = 14 -> 11 — not after the older normal 7
    # (gap 10 -> 7); its runner-up is a survivor (15), not the more recent normal 19; 2 entered rule 4
    ("rule_4_after_rule_3", (11, "no_pattern", (RANKING_MOST_RECENT_BEFORE_IMBALANCE, 15, 2, 14), None)),
    # the ranking rules see only the IC candidates: 10 (no gap after it) is no candidate -> 7 alone, nothing ranked
    ("with_a_non_candidate", (7, "no_pattern", None, None)),
])
def test_the_ranking_rules_decide_in_order(name, expected, classify_all):
    sel, _ = _sel(name, classify_all=classify_all)
    assert all(c.pattern.rank is None for c in sel.ic_candidates)            # no pattern: Step 3
    assert sel.tiebreak is None
    assert _outcome(sel) == expected


def test_rule_4_takes_its_reference_after_the_oldest_candidate_still_in_play():
    """ICR-8 (the sub 0 cyc 0 shape): after rules 1–3 the marus 9 / 13 / 17 are in play (7 the pinbar excluded, no
    immediate gap) -> the earliest unfilled gap after 9 is 16 -> 13. From the oldest Step 1 candidate (7) the
    reference would be 8 — before every candidate in play, no pick."""
    sel, df = _sel("oldest_in_play")
    assert [(c.ic_idx, str(df.loc[c.ic_idx, "candle_type"]), c.immediate_imbalance_first_c2_idx)
            for c in sel.ic_candidates] == [(7, "pinbar", 8), (9, "maru", None), (13, "maru", None), (17, "maru", None)]
    assert _outcome(sel) == (13, "no_pattern", (RANKING_MOST_RECENT_BEFORE_IMBALANCE, 17, 3, 16), None)


def test_pinbars_are_excluded_before_the_immediate_imbalance():
    """ICR-S (the H1 sid 0 cyc 0 shape — IC candidates 177 pinbar, 180 maru, 183 pinbar with the only immediate gap
    -> 180): the ranking rules exclude the pinbar 11 first -> 9; the TIEBREAK rules' order (the immediate imbalance
    first) over the same list would take 11 — the two rule sets are separate wholes."""
    sel, df = _sel("pinbar_with_the_gap")
    assert [c.immediate_imbalance_first_c2_idx for c in sel.ic_candidates] == [None, None, 12]
    assert sel.pick.ic_idx == 9
    fib = _fib_over(df, 6, 12, 0.5, 0.7)
    pick, tb = apply_tiebreak_rules(df, fib, POIConfig(), [c.pattern for c in sel.ic_candidates],
                                    {c.ic_idx: c.immediate_imbalance_first_c2_idx for c in sel.ic_candidates})
    assert (pick.ic_idx, tb[0]) == (11, "immediate_unfilled_imbalance")


def test_step_3_never_runs_when_a_pattern_exists(monkeypatch):
    """7 (a pinbar) has a pattern (3A.i), 9 (a maru) none: the pattern decides — the ranking rules (which would
    exclude 7 and take 9) are not even called."""
    calls = []
    real = poi_zones.apply_ranking_rules
    monkeypatch.setattr(poi_zones, "apply_ranking_rules", lambda *a, **k: calls.append(1) or real(*a, **k))
    sel, _ = _sel("pattern_pinbar")
    assert [(c.ic_idx, c.pattern.ip_rank) for c in sel.ic_candidates] == [(7, "3A.i"), (9, None)]
    assert _outcome(sel) == (7, "pattern", None, None) and calls == []
    _sel("single")
    assert calls == [1]                                                      # ... and is called without one


def test_no_ic_candidate_and_the_early_return_report_no_ic_candidate():
    """No IC candidate -> Step 2 never runs (`ip_outcome` None); a fib neither active nor locked returns early with
    the same reason (PLAN_L A6)."""
    specs, cts, gaps = FIX["single"]
    df = _frame(specs)
    no_gap = _select(df, 6, cts, [])                                         # condition 3 fails for 7
    assert [c.is_ic_candidate for c in no_gap.checks] == [False]
    assert _outcome(no_gap) == (None, None, None, NO_IC_CANDIDATE)
    df.attrs["imbalances"] = [_far_gap(g) for g in gaps]
    dead = dataclasses.replace(_fib_over(df, 6, cts, 0.5, 0.7), active=False, locked=False)
    sel = poi_zones.select_ic(df, dead, POIConfig(), ic_reference=ICReference.NONE)
    assert sel.checks == () and _outcome(sel) == (None, None, None, NO_IC_CANDIDATE)


def test_the_selection_rejects_inconsistent_fields():
    """`ICSelection` asserts its fields' consistency (a pick ⇔ no reason; a tiebreak only with a pattern; a ranking
    only on a ranked pick; no IC candidate ⇔ Step 2 never ran; all pinbars ⇒ no pattern; a pattern pick has a rank, a
    ranked pick none — MS's read never passes `derive_poi_zones`' assert, so the object is its only guard)."""
    sel, _ = _sel("single")
    pick = sel.pick
    ok = dict(checks=(), pick=pick, tiebreak=None, ip_outcome="no_pattern", ranking=None, no_ic_reason=None)
    ICSelection(**ok)
    for bad in (dict(no_ic_reason=NO_IC_CANDIDATE), dict(pick=None), dict(tiebreak=("ic_candle_type", 1, 2, None)),
                dict(ip_outcome="pattern", ranking=(RANKING_EXCLUDE_PINBAR, None, 2, None)),
                dict(ip_outcome=None), dict(ip_outcome="nonsense"),
                dict(pick=None, no_ic_reason=ALL_IC_CANDIDATES_PINBAR, ranking=(RANKING_EXCLUDE_PINBAR, None, 2, None)),
                dict(pick=None, no_ic_reason="no_pattern"),
                dict(pick=None, ip_outcome="pattern", no_ic_reason=ALL_IC_CANDIDATES_PINBAR),   # all pinbars: Step 2 ran, no pattern
                dict(pick=dataclasses.replace(pick, rank="1A", shape="i")),                   # a "ranked" pick with a rank
                dict(ip_outcome="pattern")):                                                  # a "pattern" pick without one
        with pytest.raises(AssertionError):
            ICSelection(**{**ok, **bad})


# ---------------------------------------------------------------------------
# ICR-9: one helper per shared rule, each rule set's OWN input list
# ---------------------------------------------------------------------------

def test_the_shared_helpers_receive_each_rule_sets_own_list(monkeypatch):
    """The immediate-imbalance helper: ranking rule 2 receives what ranking rule 1 left of ALL Step 1 candidates
    (the pinbar 7 dropped); the candle-type helper runs WITHOUT pinbar there (ranking rule 3)."""
    seen = []
    real_f, real_t = poi_zones.filter_immediate_unfilled_imbalance, poi_zones.prefer_candle_type
    monkeypatch.setattr(poi_zones, "filter_immediate_unfilled_imbalance",
                        lambda in_play, imm: seen.append(("imm", list(in_play))) or real_f(in_play, imm))
    monkeypatch.setattr(poi_zones, "prefer_candle_type",
                        lambda in_play, df, *, include_pinbar: seen.append(("type", list(in_play), include_pinbar))
                        or real_t(in_play, df, include_pinbar=include_pinbar))
    sel, _ = _sel("oldest_in_play")
    assert sel.pick.ic_idx == 13
    assert seen == [("imm", [9, 13, 17]), ("type", [9, 13, 17], False)]


# ---------------------------------------------------------------------------
# The consumers: MS's read (ICR-12), the drawn POI (meta, floor), the CSV, the mirror, the hover
# ---------------------------------------------------------------------------

def test_ms_reads_the_ranked_ics_inner():
    """`poi_inner_for_fib` (MS's in-flight read) returns a ranked pick's inner like a pattern pick's — the IC's high
    on a buy; all pinbars -> None (the BOS inner only)."""
    from engine_v2.tests._cross_fib_limits import NO_REF_TRACKER
    for name, ic in (("oldest_in_play", 13), ("pinbar_with_the_gap", 9), ("all_pinbar", None)):
        specs, cts, gaps = FIX[name]
        df = _frame(specs)
        df.attrs["imbalances"] = [_far_gap(g) for g in gaps]
        fib = _fib_over(df, 6, cts, 0.5, 0.7)
        inner = poi_zones.poi_inner_for_fib(df, fib, 0.70, tracker=NO_REF_TRACKER, evaluated_at=cts)
        assert inner == (None if ic is None else float(df.loc[ic, "h"]))


def _drawn(name, *, mutate=None):
    """The drawn POIs of a fixture's single record — its CTS_ESTABLISHED at the CTS anchor (anchor = moment), at the
    fib's own CTS price (the as-of zone = the fib's 61.8–80 zone: every candidate touches it)."""
    from engine_v2.tests._event_factory import make_cts_established
    specs, cts, gaps = FIX[name]
    df = _frame(specs)
    df.attrs["imbalances"] = [_far_gap(g) for g in gaps]
    fib = _fib_over(df, 6, cts, 0.5, 0.7)
    est = [make_cts_established(cts_anchor_idx=cts, confirmed_at=cts, price=fib.cts_price, structure_id=0,
                                cycle_id=0)]
    tracker = _Tracker([fib])
    recs = poi_zones.select_ic_records(df, est, tracker, POIConfig())
    if mutate is not None:
        recs = [mutate(r) for r in recs]
    with contextlib.redirect_stdout(io.StringIO()):
        return derive_poi_zones(df, est, fib_tracker=tracker, config=POIConfig(), ic_records=recs), df


def test_a_ranked_poi_carries_no_pattern_and_its_ranking():
    """PLAN_L A1 / B-M3: the ranked pick 9's check carries its first failing variant's counts ('1-1-1') — the POI
    meta nulls EVERY pattern field explicitly; `ip_outcome` + the `ranking_*` keys; every value an `int` (B-m9: the
    mirror shifts only `isinstance(v, int)`)."""
    (poi,), _ = _drawn("pinbar_with_the_gap")
    sel, _ = _sel("pinbar_with_the_gap")
    assert sel.pick.ip_leg_candle_counts == "1-1-1"                         # what must NOT reach the POI
    m = poi.meta
    assert (poi.ic_idx, m["ip_outcome"]) == (9, "no_pattern")
    assert [m[k] for k in ("ip_rank", "ip_leg_candle_counts", "pattern_anchor_idx", "opp_leg_first_idx",
                           "pattern_end_idx", "pattern_confirmed_idx", "ip_tiebreak", "ip_tb_runner_up_ic_idx",
                           "ip_tied_count", "ip_tb_imbalance_first_c2_idx")] == [None] * 10
    assert (m["ranking_rule"], m["ranking_runner_up_ic_idx"], m["ranking_in_play_count"],
            m["ranking_imbalance_first_c2_idx"]) == ("exclude_pinbar", None, 3, None)
    (poi,), _ = _drawn("oldest_in_play")
    m = poi.meta
    assert (poi.ic_idx, m["ranking_rule"], m["ranking_runner_up_ic_idx"], m["ranking_in_play_count"],
            m["ranking_imbalance_first_c2_idx"]) == (13, "most_recent_before_imbalance", 17, 3, 16)
    assert all(type(m[k]) is int for k in ("ranking_runner_up_ic_idx", "ranking_in_play_count",
                                           "ranking_imbalance_first_c2_idx"))


def test_a_ranked_poi_activates_without_the_pattern_term():
    """ICR-2: the floor = max(the CTS-established moment, the IC, the structure's start) — no pattern term. The sweep
    takes `pattern_confirmed_idx=None`; through `derive_poi_zones` a ranked POI's history is the sweep's with None."""
    from engine_v2.tests.test_plan_h_landing_b import _hist, _sweep
    assert _hist(_sweep(pattern_confirmed_idx=None)) == [(9, True, "initial")]        # max(moment 6, IC 4)
    assert _hist(_sweep(pattern_confirmed_idx=11)) == [(11, True, "initial")]         # a pattern's term binds
    (poi,), df = _drawn("oldest_in_play")
    assert poi.meta["pattern_confirmed_idx"] is None
    # first_active = max(the moment 20, the IC 13); the gap 16 formed at 17 -> live from 20
    assert [(e["idx"], e["active"]) for e in poi.meta["activation_history"]] == [(20, True)]
    with pytest.raises(TypeError):                                           # keyword-required: no default (A7)
        poi_zones._compute_poi_activation_history(
            df, ic_idx=13, cts_established_idx=12, sd=1, scan_end=20, bos_price=0.6, cts_events=[],
            config=POIConfig(), imbalances=[], fill_idx_cache={})


def test_the_pattern_moment_matches_the_outcome():
    """The ⇔ assert in `derive_poi_zones`: a ranked pick with a pattern moment, or a pattern pick without one, fails
    loudly (a pick's kind and its floor term cannot disagree)."""
    def ranked_with_moment(r):
        return dataclasses.replace(r, selection=dataclasses.replace(
            r.selection, pick=dataclasses.replace(r.selection.pick, pattern_confirmed_idx=14)))
    with pytest.raises(AssertionError, match="pattern_confirmed_idx 14 vs ip_outcome 'no_pattern'"):
        _drawn("single", mutate=ranked_with_moment)

    def pattern_without_moment(r):
        return dataclasses.replace(r, selection=dataclasses.replace(
            r.selection, pick=dataclasses.replace(r.selection.pick, pattern_confirmed_idx=None)))
    with pytest.raises(AssertionError, match="pattern_confirmed_idx None vs ip_outcome 'pattern'"):
        _drawn("pattern_pinbar", mutate=pattern_without_moment)


def test_the_mirror_shifts_the_two_ranking_indices():
    """A sub's ranked POI and its CSV row: the mirror shifts `ranking_runner_up_ic_idx` /
    `ranking_imbalance_first_c2_idx` (meta keys + record fields), never `ranking_in_play_count`."""
    from engine_v2.tests.test_event_meta_idx_keys import _Poi, _synthetic_result
    meta = {"ip_outcome": "no_pattern", "ranking_rule": "most_recent_before_imbalance",
            "ranking_runner_up_ic_idx": 17, "ranking_in_play_count": 3, "ranking_imbalance_first_c2_idx": 16,
            "activation_history": []}
    row = dataclasses.replace(
        derive_ic_candidates(*_ic_inputs("oldest_in_play"), structure_path_id="M15")[2], meta={})
    assert (row.ic_idx, row.picked, row.ranking_runner_up_ic_idx, row.ranking_imbalance_first_c2_idx) == (13, True, 17, 16)
    res = dataclasses.replace(_synthetic_result(), poi_zones=[_Poi(ic_idx=13, meta=meta)], ic_candidates=[row])
    d = pd.DataFrame(index=range(100))
    edm.mirror_lower_tf_result_to_entity_df(d, res, structure_path_id="p")
    (poi,), (r,) = d.attrs["poi_zones"], d.attrs["ic_candidates"]
    sb = res.meta["slice_begin"]
    assert (poi.meta["ranking_runner_up_ic_idx"], poi.meta["ranking_imbalance_first_c2_idx"],
            poi.meta["ranking_in_play_count"]) == (17 + sb, 16 + sb, 3)
    assert (r.ranking_runner_up_ic_idx, r.ranking_imbalance_first_c2_idx, r.ranking_in_play_count) == (
        17 + sb, 16 + sb, 3)


def _ic_inputs(name):
    specs, cts, gaps = FIX[name]
    df = _frame(specs)
    df.attrs["imbalances"] = [_far_gap(g) for g in gaps]
    return df, [_est(0, 0)], _Tracker([_fib_over(df, 6, cts, 0.5, 0.7)]), POIConfig()


def test_the_csv_carries_the_outcome_on_every_row_and_the_ranking_on_the_pick(tmp_path):
    """`ip_outcome` (record-level) on every row; the `ranking_*` columns on the picked row only; the ranked pick
    keeps its pattern DIAGNOSTICS (ICR-15); all pinbars -> the no-IC row `all_ic_candidates_pinbar`."""
    i = IC_CANDIDATE_COLUMNS.index("is_ic_candidate")
    assert IC_CANDIDATE_COLUMNS[i + 1:i + 3] == ["ip_outcome", "ip_rank"]
    assert IC_CANDIDATE_COLUMNS[-4:] == ["ranking_rule", "ranking_runner_up_ic_idx", "ranking_in_play_count",
                                         "ranking_imbalance_first_c2_idx"]
    recs = derive_ic_candidates(*_ic_inputs("oldest_in_play"), structure_path_id="H1.main")
    path = tmp_path / "ic.csv"
    export_ic_candidates(recs, path)
    got = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(got.columns) == IC_CANDIDATE_COLUMNS
    assert got[["ic_idx", "ip_outcome", "picked", "ranking_rule", "ranking_runner_up_ic_idx", "ranking_in_play_count",
                "ranking_imbalance_first_c2_idx", "reason"]].values.tolist() == [
        ["7", "no_pattern", "False", "", "", "", "", "continuation_not_big_maru"],
        ["9", "no_pattern", "False", "", "", "", "", "continuation_not_big_maru"],
        ["13", "no_pattern", "True", "most_recent_before_imbalance", "17", "3", "16", "continuation_not_big_maru"],
        ["17", "no_pattern", "False", "", "", "", "", "continuation_not_big_maru"],
    ]
    assert got.loc[2, "ip_leg_candle_counts"] == "3-1-3" and got.loc[2, "legs"] != ""    # the diagnostics kept
    recs = derive_ic_candidates(*_ic_inputs("all_pinbar"), structure_path_id="H1.main")
    assert [(r.ic_idx, r.ip_outcome, r.picked, r.reason) for r in recs] == [
        (7, "no_pattern", False, "continuation_not_big_maru"), (9, "no_pattern", False, "continuation_not_big_maru"),
        (None, "no_pattern", False, "all_ic_candidates_pinbar")]
    recs = derive_ic_candidates(*_ic_inputs("pattern_pinbar"), structure_path_id="H1.main")
    assert {r.ip_outcome for r in recs} == {"pattern"} and all(r.ranking_rule is None for r in recs)


def _ranked_meta(rule, runner_up=None, gap=None):
    return {"ip_outcome": "no_pattern", "ip_rank": None, "ip_leg_candle_counts": None, "pattern_anchor_idx": None,
            "opp_leg_first_idx": None, "pattern_end_idx": None, "pattern_confirmed_idx": None, "ip_tiebreak": None,
            "ranking_rule": rule, "ranking_runner_up_ic_idx": runner_up, "ranking_imbalance_first_c2_idx": gap}


@pytest.mark.parametrize("meta, text", [
    (_ranked_meta(None), "No pattern · only IC candidate"),
    (_ranked_meta("exclude_pinbar"), "No pattern · ranking rule exclude_pinbar"),
    (_ranked_meta("immediate_unfilled_imbalance", 2570, 2577),
     "No pattern · ranking rule immediate_unfilled_imbalance over 2570 (gap 2577)"),
    (_ranked_meta("ic_candle_type", 7), "No pattern · ranking rule ic_candle_type over 7"),
    (_ranked_meta("most_recent_before_imbalance", 17, 16),
     "No pattern · ranking rule most_recent_before_imbalance over 17 (gap 16)"),
])
def test_the_poi_hover_names_the_ranking_without_the_ic_index(meta, text):
    """ICR-7 / ICR-15: the POI hover's ranked line — `No pattern` in the place of `IP <rank>`, the deciding rule + its
    runner-up / gap when it has them; NO `IC <idx>` (the hover's own `ic_idx=` line names the IC)."""
    assert poi_ic_hover_text(meta, 13) == text


@pytest.mark.parametrize("meta", [
    {**_ranked_meta("exclude_pinbar"), "ip_leg_candle_counts": "1-1-1"},    # a ranked POI with pattern meta
    _ranked_meta("ic_candle_type"),                                         # the rule's runner-up missing
    _ranked_meta("exclude_pinbar", 7),                                      # a runner-up rule 1 never computes
    _ranked_meta(None, 7),                                                  # a value without a rule
    _ranked_meta("TB1", 7),                                                 # not a ranking rule
    {**_ranked_meta(None), "ip_outcome": None},                             # a POI without its outcome
])
def test_the_poi_hover_rejects_an_inconsistent_ranking(meta):
    with pytest.raises(AssertionError):
        poi_ic_hover_text(meta, 13)


def test_a_pattern_poi_hover_rejects_a_ranking_rule():
    from engine_v2.tests.test_plan_h_landing_b import _meta
    poi_ic_hover_text(_meta("1A.i", "1-1-1", 2703, 2704, 2705), 2704)
    with pytest.raises(AssertionError, match="carries ranking_rule"):
        poi_ic_hover_text({**_meta("1A.i", "1-1-1", 2703, 2704, 2705), "ranking_rule": "exclude_pinbar"}, 2704)


def test_the_ranking_constants_are_the_decided_words():
    """PLAN_H §6 ICR-3: the ranking rules 2–4 take the tiebreak rules' descriptive words; the outcome / reasons."""
    assert (RANKING_EXCLUDE_PINBAR, RANKING_IMMEDIATE_UNFILLED_IMBALANCE, RANKING_IC_CANDLE_TYPE,
            RANKING_MOST_RECENT_BEFORE_IMBALANCE) == ("exclude_pinbar", "immediate_unfilled_imbalance",
                                                      "ic_candle_type", "most_recent_before_imbalance")
    assert (IP_OUTCOME_PATTERN, IP_OUTCOME_NO_PATTERN, NO_IC_CANDIDATE, ALL_IC_CANDIDATES_PINBAR) == (
        "pattern", "no_pattern", "no_ic_candidate", "all_ic_candidates_pinbar")


def test_apply_ranking_rules_reads_the_checks_it_is_handed():
    """A9: the immediate gap comes from the CHECKS (nothing re-derived) — a check stripped of its gap changes rule 2."""
    sel, df = _sel("one_immediate")
    fib = _fib_over(df, 6, 12, 0.5, 0.7)
    cands = sel.ic_candidates
    assert apply_ranking_rules(df, fib, POIConfig(), cands)[0].ic_idx == 7
    stripped = tuple(dataclasses.replace(c, immediate_imbalance_first_c2_idx=None) for c in cands)
    pick, ranking = apply_ranking_rules(df, fib, POIConfig(), stripped)
    assert (pick.ic_idx, ranking[0]) == (9, "ic_candle_type")
