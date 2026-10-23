"""PLAN_S step 3 — IC Step 1B, the prior-imbalance preference (`plans/PLAN_S_fill_rule_and_ic_step1b.md` §4; the
user's write-up + Q&A rounds 2–7, `plans/qa_ic_revisit_inputs/qa_findings.md`).

After Step 1 (or Step 1A), each IC candidate's closest prior imbalance = the sd instance whose c2 run starts at / after
the fib's BOS anchor and ends before the candidate, the one ending latest; the candidate PASSES when that gap is not
two-stroke-filled through the candle right BEFORE it (no prior gap -> fails). The passers go through Steps 2–3; no IC
from them -> Steps 2–3 on ALL the candidates; no passer -> all the candidates. 1B never adds or removes an IC; it
outranks the institutional pattern. Pinned here: the gap's bounds, the fill horizon IC − 1 and the config's threshold,
the flow and its outcome on every path (both selection modes — the drawn record's `classify_all` and MS's read), the
`ICSelection` invariants, the CSV / POI meta / mirror, the hovers.

The flow fixtures are `test_plan_l_landing_2.FIX`'s frames with `neutral_1b=False` — their never-filled immediate gaps
are also the next candidate's prior gap, while the oldest candidate has none (with `neutral_1b`, the default, a gap at
the BOS anchor makes every candidate pass: the pick WITHOUT 1B).
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
from engine_v2.debug.export_zones import IC_CANDIDATE_COLUMNS, export_ic_candidates
from engine_v2.multitf import entity_df_mutation as edm
from engine_v2.tests._event_factory import make_cts_established
from engine_v2.tests.test_institutional_pattern import (
    D,
    U,
    _est,
    _far_gap,
    _fib_over,
    _frame,
    _select,
    _tb2_frame,
    _Tracker,
)
from engine_v2.tests.test_plan_l_landing_2 import FIX, S
from engine_v2.zones.poi_zones import (
    PRIOR_IMBALANCE_FALLBACK_NO_PASSER,
    PRIOR_IMBALANCE_FALLBACK_PASSERS_NO_IC,
    PRIOR_IMBALANCE_PREFERRED,
    ICCandleCheck,
    POIConfig,
    check_prior_imbalance,
    closest_prior_imbalance,
    derive_ic_candidates,
    derive_poi_zones,
    prior_imbalance_narrowed,
    select_ic_records,
)

PREFERRED, NO_IC_FROM_PASSERS, NO_PASSER = (PRIOR_IMBALANCE_PREFERRED, PRIOR_IMBALANCE_FALLBACK_PASSERS_NO_IC,
                                            PRIOR_IMBALANCE_FALLBACK_NO_PASSER)
_ABOVE = (0.602, 0.604)     # `test_plan_p_ic_condition3_step1a._ABOVE`: no candle touches 61.8–80 -> Step 1A


def test_the_outcome_names_are_the_users():
    """Round 6 (the user chose the `prior_imbalance_*` names): the three outcome values."""
    assert (PREFERRED, NO_IC_FROM_PASSERS, NO_PASSER) == ("preferred", "fallback_passers_no_ic", "fallback_no_passer")


# ---------------------------------------------------------------------------
# The gap: `closest_prior_imbalance`
# ---------------------------------------------------------------------------

def _inst(start, end=None, direction=1):
    end = start if end is None else end
    return ImbalanceInstance(start_idx=start, end_idx=end, direction=direction, gap_top=0.5000, gap_bottom=0.4990,
                             gap_size=0.0010)


@pytest.mark.parametrize("ic, expected", [
    (13, (12, 12)),   # a gap whose c3 IS the IC counts (round 2 Q4: "A gap formed by IC counts")
    (12, (8, 9)),     # 12..12 does not end before 12; the −sd gap 10 is ignored; a merged run = one gap
    (11, (8, 9)),     # the −sd gap 10 ends later but is the IC's direction, not the structure's
    (8, (6, 6)),      # 5..7 ends later but straddles the BOS anchor 6 — never counts; 6..6 starts AT the anchor
    (7, (6, 6)),
    (6, None),        # nothing in the window ends before the BOS anchor itself
])
def test_the_closest_prior_gap_is_the_latest_ending_sd_run_from_the_bos_anchor(ic, expected):
    df = _frame([("M", U, 10)] + [S] * 10)
    df.attrs["imbalances"] = [_inst(5, 7), _inst(6), _inst(8, 9), _inst(10, direction=-1), _inst(12)]
    fib = _fib_over(df, 6, 16, 0.5, 0.7)                                   # a buy fib, BOS anchor 6
    inst = closest_prior_imbalance(df, ic, fib)
    assert (None if inst is None else (inst.start_idx, inst.end_idx)) == expected


# ---------------------------------------------------------------------------
# The test: `check_prior_imbalance` — through IC − 1, the config's threshold
# ---------------------------------------------------------------------------

# A buy gap at c2 7, [0.6000, 0.6010]: candle 8's low 0.6003 arms it at 0.65 (level 0.60035) but not at 0.90 (0.6001);
# candle 9's close 0.6012 >= the top commits it -> committed AT 9.
_RAW = [("raw", 0.6000, 0.6015, 0.5995, 0.6012),      # 6 the BOS anchor
        ("raw", 0.6012, 0.6030, 0.6011, 0.6028),      # 7 the gap's c2
        ("raw", 0.6020, 0.6022, 0.6003, 0.6005),      # 8 arms (0.65)
        ("raw", 0.6005, 0.6014, 0.6004, 0.6012),      # 9 commits
        ("raw", 0.6012, 0.6013, 0.6006, 0.6008),      # 10
        ("raw", 0.6008, 0.6010, 0.6002, 0.6004)]      # 11
_GAP = ImbalanceInstance(start_idx=7, end_idx=7, direction=1, gap_top=0.6010, gap_bottom=0.6000, gap_size=0.0010)


def _check(ic):
    return ICCandleCheck(ic_idx=ic, has_unfilled_imbalance_after=False, ic_ref_reason=None, touches_zone=True,
                         immediate_imbalance_first_c2_idx=None)


def _prior(ic, *, gap=_GAP, thr=None):
    df = _frame(_RAW)
    df.attrs["imbalances"] = [gap]
    cfg = POIConfig() if thr is None else POIConfig(fill_threshold=thr)
    c = check_prior_imbalance(df, _check(ic), _fib_over(df, 6, 11, 0.5, 0.7), cfg)
    return c.prior_imbalance_first_c2_idx, c.prior_imbalance_unfilled


def test_the_fixture_fills_the_gap_exactly_at_9():
    df = _frame(_RAW)
    assert not _GAP.is_filled(df, 8) and _GAP.is_filled(df, 9)            # armed 8, committed AT 9


@pytest.mark.parametrize("ic, expected", [
    (7, (None, False)),   # the gap ends AT the candidate: no prior gap -> fails (the author's default, round 3)
    (8, (7, True)),       # its c3 is the IC: the scan (7, 7] is empty -> unfilled
    (9, (7, True)),       # armed by 8, committed BY the IC's own close -> PASSES: the IC never disqualifies itself
    (10, (7, False)),     # committed at 9 = IC − 1 -> fails (a horizon of IC − 2 would pass it)
    (11, (7, False)),     # stays filled
])
def test_the_gap_is_judged_through_the_candle_before_the_ic(ic, expected):
    """Round 5 (verbatim): "the balance needs to be unfilled up to the candle right before the IC"."""
    assert _prior(ic) == expected


def test_a_merged_gap_is_named_by_its_first_c2():
    """A merged run (c2s 6..7, the fill scanned from 8) is ONE gap, named by its FIRST gap candle (like every
    `*_imbalance_first_c2_idx`; sub 3 cycle 1's 2810..2812 -> 2810 on the window); filled at 9 as before."""
    run = dataclasses.replace(_GAP, start_idx=6, end_idx=7)
    assert _prior(9, gap=run) == (6, True)
    assert _prior(10, gap=run) == (6, False)


def test_the_threshold_is_the_configs():
    """The fill reads `config.fill_threshold` (the drawn POI's `POIConfig`, MS's tracker value): at 0.90 candle 8
    (low 0.6003 > the 0.6001 level) never arms the gap -> unfilled through 9; at the default it fails."""
    assert _prior(10, thr=0.90) == (7, True)
    assert _prior(10) == (7, False)


def test_a_degenerate_closest_gap_counts_as_filled():
    """A zero-size merged gap: `is_filled`'s safe default (filled) — even with nothing to scan (0 on the window)."""
    flat = dataclasses.replace(_GAP, gap_top=0.6000, gap_size=0.0)
    assert _prior(8, gap=flat) == (7, False)


# ---------------------------------------------------------------------------
# The flow through `select_ic` — both selection modes
# ---------------------------------------------------------------------------

def _sel(name, *, neutral, classify_all=True, zone=(0.5, 0.7), specs=None):
    specs_, cts, gaps = FIX[name] if specs is None else specs
    df = _frame(specs_)
    return _select(df, 6, cts, [_far_gap(g) for g in gaps], zone=zone, classify_all=classify_all,
                   neutral_1b=neutral), df


def _summary(sel):
    out = sel.prior_imbalance_outranked
    return (None if sel.pick is None else int(sel.pick.ic_idx), sel.ip_outcome, sel.ranking,
            sel.prior_imbalance_outcome, sel.prior_imbalance_passer_count, sel.ic_candidate_count,
            None if out is None else (int(out.ic_idx), out.ip_rank))


# 7 a maru (immediate gap 8), 9 a PINBAR (gap 10; its prior gap 8): the lone passer is a no-pattern pinbar
_FALLBACK = ([("M", U, 10), ("M", D, 6), S, ("P", D, 6), S], 10, [8, 10])
# 7 a maru (gap 8, no prior gap), 9 a maru carrying 2B.iii (gap 10, prior gap 8 — the gaps' c2s are sd candles): 1B
# narrows 2 -> 1, and Step 2 over ALL the candidates picks the same passer — nothing outranked (sub 8 cross 0→2's shape
# on the window; found by search, PLAN_S step 3)
_PASSER_PATTERN = ([("M", U, 10), ("M", D, 6), ("N", U, 6), ("M", D, 6), ("M", U, 10), S, S], 12, [8, 10])


@pytest.mark.parametrize("classify_all", [True, False])
@pytest.mark.parametrize("name, specs, without_1b, with_1b", [
    # 7 (maru) has no prior gap, 9 (normal) has 8: today the candle type takes 7 — with 1B the passer 9, alone in
    # Step 3 (no ranking rule), the preference narrowed 2 -> 1
    ("runner_up_still_in_play", None, (7, "no_pattern", ("ic_candle_type", 9, 2, None), PREFERRED, 2, 2, None),
     (9, "no_pattern", None, PREFERRED, 1, 2, None)),
    # 1B OUTRANKS the institutional pattern (round 2 Q5): 7 carries 3A.i but has no prior gap; 9 passes -> the
    # ranked 9; the outranked pattern = Step 2's pick over all the candidates (7, 3A.i — round 7)
    ("pattern_pinbar", None, (7, "pattern", None, PREFERRED, 2, 2, None),
     (9, "no_pattern", None, PREFERRED, 1, 2, (7, "3A.i"))),
    # narrowed, but the pattern pick IS Step 2's pick over all the candidates: nothing outranked
    ("passer_pattern", _PASSER_PATTERN, (9, "pattern", None, PREFERRED, 2, 2, None),
     (9, "pattern", None, PREFERRED, 1, 2, None)),
    # Steps 2–3 run over the PASSERS: ranking rule 1 decides over 2 (9, 11), not 3 — the same IC
    ("pinbar_with_the_gap", None, (9, "no_pattern", ("exclude_pinbar", None, 3, None), PREFERRED, 3, 3, None),
     (9, "no_pattern", ("exclude_pinbar", None, 2, None), PREFERRED, 2, 3, None)),
    # the lone passer 9 is a no-pattern pinbar -> no IC from the passers -> Steps 2–3 on ALL: today's 7, its
    # ranking over both candidates (the fallback's fields, never the passers' run)
    ("fallback", _FALLBACK, (7, "no_pattern", ("exclude_pinbar", None, 2, None), PREFERRED, 2, 2, None),
     (7, "no_pattern", ("exclude_pinbar", None, 2, None), NO_IC_FROM_PASSERS, 1, 2, None)),
    # every candidate a pinbar: no IC either way (1B never adds one)
    ("all_pinbar", None, (None, "no_pattern", None, NO_IC_FROM_PASSERS, 2, 2, None),
     (None, "no_pattern", None, NO_IC_FROM_PASSERS, 1, 2, None)),
    # no passer -> Steps 2–3 on all the candidates = today's pick
    ("single", None, (7, "no_pattern", None, PREFERRED, 1, 1, None), (7, "no_pattern", None, NO_PASSER, 0, 1, None)),
    ("neighbours", None, (8, "no_pattern", ("most_recent_before_imbalance", 7, 2, 9), PREFERRED, 2, 2, None),
     (8, "no_pattern", ("most_recent_before_imbalance", 7, 2, 9), NO_PASSER, 0, 2, None)),
    # no IC candidate: 1B never runs (no outcome, no counts)
    ("no_immediate", None, (None, None, None, None, None, None, None), (None, None, None, None, None, None, None)),
])
def test_the_flow_on_every_path(name, specs, without_1b, with_1b, classify_all):
    neutral, _ = _sel(name, neutral=True, classify_all=classify_all, specs=specs)
    assert _summary(neutral) == without_1b
    sel, _ = _sel(name, neutral=False, classify_all=classify_all, specs=specs)
    assert _summary(sel) == with_1b


def test_1b_runs_after_step_1a():
    """Step 1A's candidates (the 80–100 zone; all three 1A.i, tied) go through 1B the same way (the write-up: "Step 1B
    would be applied after 1A in the same way"): 7 has no prior gap -> the tie is broken among the passers 9 / 11 ->
    9; the outranked pattern = Step 2's pick over all three (7)."""
    gaps = [8, 10, 12]
    neutral = _select(_tb2_frame(), 6, 12, [_far_gap(g) for g in gaps], zone=_ABOVE, classify_all=True)
    assert (neutral.ic_zone_band, neutral.pick.ic_idx, neutral.pick.ip_rank) == ("80-100", 7, "1A.i")
    sel = _select(_tb2_frame(), 6, 12, [_far_gap(g) for g in gaps], zone=_ABOVE, classify_all=True,
                  neutral_1b=False)
    assert sel.ic_zone_band == "80-100"
    assert _summary(sel) == (9, "pattern", None, PREFERRED, 2, 3, (7, "1A.i"))
    assert sel.tiebreak == ("most_recent_before_imbalance", 11, 2, 10)      # the tie among the PASSERS (9, 11)


@pytest.mark.parametrize("name", sorted(FIX) + ["fallback", "passer_pattern"])
def test_1b_never_adds_or_removes_an_ic(name):
    """The plan's consequence (`qa_findings.md` round 2): an IC with 1B ⇔ an IC without it — 1B only swaps it."""
    specs = {"fallback": _FALLBACK, "passer_pattern": _PASSER_PATTERN}.get(name)
    without, _ = _sel(name, neutral=True, specs=specs)
    with_1b, _ = _sel(name, neutral=False, specs=specs)
    assert (without.pick is None) == (with_1b.pick is None)
    assert without.no_ic_reason == with_1b.no_ic_reason


def test_1b_is_evaluated_on_every_check_only_with_classify_all():
    """`classify_all` (the drawn record -> the CSV) tests every check; MS's read (without it) only its IC candidates —
    11 (no immediate gap: no candidate) keeps None there; the pick is the same."""
    full, _ = _sel("runner_up_still_in_play", neutral=False, classify_all=True)
    ms, _ = _sel("runner_up_still_in_play", neutral=False, classify_all=False)
    by = {c.ic_idx: (c.prior_imbalance_first_c2_idx, c.prior_imbalance_unfilled) for c in full.checks}
    assert by == {7: (None, False), 9: (8, True), 11: (10, True)}
    by = {c.ic_idx: (c.prior_imbalance_first_c2_idx, c.prior_imbalance_unfilled) for c in ms.checks}
    assert by == {7: (None, False), 9: (8, True), 11: (None, None)}
    assert full.pick.ic_idx == ms.pick.ic_idx == 9


def test_ms_reads_the_preferred_ics_inner():
    """MS's in-flight read (`poi_inner_for_fib`, the same selector) returns the 1B pick's inner — 9's high, not 7's."""
    from engine_v2.tests._cross_fib_limits import NO_REF_TRACKER
    specs, cts, gaps = FIX["runner_up_still_in_play"]
    df = _frame(specs)
    df.attrs["imbalances"] = [_far_gap(g) for g in gaps]
    fib = _fib_over(df, 6, cts, 0.5, 0.7)
    inner = poi_zones.poi_inner_for_fib(df, fib, POIConfig().fill_threshold, tracker=NO_REF_TRACKER, evaluated_at=cts)
    assert inner == float(df.loc[9, "h"]) != float(df.loc[7, "h"])


# ---------------------------------------------------------------------------
# `ICSelection`'s invariants
# ---------------------------------------------------------------------------

def test_the_selection_rejects_an_inconsistent_preference():
    sel, _ = _sel("runner_up_still_in_play", neutral=False)            # 7 fails, 9 passes; pick 9, preferred
    dataclasses.replace(sel)                                            # the real one is consistent
    seven = next(c for c in sel.checks if c.ic_idx == 7).pattern        # a candidate that fails 1B (no pattern)
    nine = sel.pick
    unevaluated = tuple(dataclasses.replace(c, prior_imbalance_unfilled=None) if c.ic_idx == 9 else c
                        for c in sel.checks)
    no_gap = tuple(dataclasses.replace(c, prior_imbalance_first_c2_idx=None) if c.ic_idx == 9 else c
                   for c in sel.checks)
    with_a_pattern, _ = _sel("pattern_pinbar", neutral=False)
    pattern_7 = with_a_pattern.prior_imbalance_outranked
    for bad in (dict(prior_imbalance_outcome=None),                     # 1B runs whenever there are IC candidates
                dict(prior_imbalance_outcome="nonsense"),
                dict(pick=seven),                                       # "preferred" -> the pick passes
                dict(prior_imbalance_outcome=NO_PASSER),                # a passer exists
                dict(prior_imbalance_outcome=NO_IC_FROM_PASSERS),       # ... and the fallback's IC is never a passer
                dict(checks=unevaluated),                               # every candidate tested
                dict(checks=no_gap),                                    # an unfilled prior gap exists
                dict(checks=()),                                        # the pick is an IC candidate (lens B)
                dict(prior_imbalance_outranked=nine),                   # outranked: a ranked non-passer only
                dict(prior_imbalance_outranked=pattern_7, prior_imbalance_outcome=NO_IC_FROM_PASSERS, pick=seven)):
        with pytest.raises(AssertionError):
            dataclasses.replace(sel, **bad)
    # The pick is an IC candidate (plan review lens B) — on its own: 7 is the only candidate (it fails 1B, no passer:
    # no other 1B assert involved), 10 a checked non-candidate (no immediate gap) with a classified, rank-less pattern
    lone, _ = _sel("with_a_non_candidate", neutral=False)
    assert (lone.pick.ic_idx, lone.prior_imbalance_outcome) == (7, NO_PASSER)
    ten = next(c for c in lone.checks if c.ic_idx == 10)
    assert not ten.is_ic_candidate and ten.pattern.rank is None
    with pytest.raises(AssertionError, match="is no IC candidate"):
        dataclasses.replace(lone, pick=ten.pattern)
    # Each new 1B assert ALONE (landing review lens 2): "single" without the neutral gap = fallback_no_passer, pick 7
    single, _ = _sel("single", neutral=False)
    assert (single.pick.ic_idx, single.prior_imbalance_outcome) == (7, NO_PASSER)
    untested = tuple(dataclasses.replace(c, prior_imbalance_unfilled=None) for c in single.checks)
    with pytest.raises(AssertionError, match="without its Step 1B test"):        # every candidate tested
        dataclasses.replace(single, checks=untested)
    with pytest.raises(AssertionError, match=r", \[\]\)$"):                     # "passers gave no IC" needs a passer
        dataclasses.replace(single, prior_imbalance_outcome=NO_IC_FROM_PASSERS)
    gap_at_9 = tuple(dataclasses.replace(c, prior_imbalance_first_c2_idx=9) if c.ic_idx == 9 else c for c in sel.checks)
    with pytest.raises(AssertionError, match="prior_imbalance_first_c2_idx=9, prior_imbalance_unfilled=True"):
        dataclasses.replace(sel, checks=gap_at_9)                                   # a passer's gap ends before it
    no_ic, _ = _sel("no_immediate", neutral=False)
    with pytest.raises(AssertionError):
        dataclasses.replace(no_ic, prior_imbalance_outcome=NO_PASSER)  # no IC candidate -> no outcome
    assert (no_ic.prior_imbalance_passer_count, no_ic.ic_candidate_count) == (None, None)


# ---------------------------------------------------------------------------
# The consumers: the CSV, the POI meta + the mirror, the hovers
# ---------------------------------------------------------------------------

def _inputs(specs, *, zone=(0.5, 0.7)):
    specs_, cts, gaps = specs
    df = _frame(specs_)
    df.attrs["imbalances"] = [_far_gap(g) for g in gaps]                 # NOT neutral: 1B decides
    fib = _fib_over(df, 6, cts, *zone)
    est = [make_cts_established(cts_anchor_idx=cts, confirmed_at=cts, price=fib.cts_price, structure_id=0,
                                cycle_id=0)]
    return df, est, _Tracker([fib])


def _drawn(specs):
    df, est, tracker = _inputs(specs)
    recs = select_ic_records(df, est, tracker, POIConfig())
    with contextlib.redirect_stdout(io.StringIO()):
        (poi,) = derive_poi_zones(df, est, fib_tracker=tracker, config=POIConfig(), ic_records=recs)
    rows = derive_ic_candidates(df, est, tracker, POIConfig(), structure_path_id="H1.main", ic_records=recs)
    return poi, rows


def test_the_csv_carries_the_test_per_check_and_the_outcome_per_record(tmp_path):
    """Step 1B's columns in step order (after Step 1A's band, before Step 2's outcome); the test on every check row
    (11 is no candidate but tested), the outcome + counts on every row of the record — the no-IC row included."""
    i = IC_CANDIDATE_COLUMNS.index("ic_zone_band")
    assert IC_CANDIDATE_COLUMNS[i + 1:i + 7] == ["prior_imbalance_first_c2_idx", "prior_imbalance_unfilled",
                                                 "prior_imbalance_outcome", "prior_imbalance_passer_count",
                                                 "ic_candidate_count", "ip_outcome"]
    _, rows = _drawn(FIX["runner_up_still_in_play"])
    path = tmp_path / "ic.csv"
    export_ic_candidates(rows, path)
    got = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert got[["ic_idx", "picked", "prior_imbalance_first_c2_idx", "prior_imbalance_unfilled",
                "prior_imbalance_outcome", "prior_imbalance_passer_count", "ic_candidate_count"]].values.tolist() == [
        ["7", "False", "", "False", "preferred", "1", "2"],
        ["9", "True", "8", "True", "preferred", "1", "2"],
        ["11", "False", "10", "True", "preferred", "1", "2"],
    ]
    df, est, tracker = _inputs(FIX["all_pinbar"])
    rows = derive_ic_candidates(df, est, tracker, POIConfig(), structure_path_id="H1.main")
    assert [(r.ic_idx, r.prior_imbalance_first_c2_idx, r.prior_imbalance_unfilled, r.prior_imbalance_outcome,
             r.prior_imbalance_passer_count, r.ic_candidate_count, r.reason) for r in rows] == [
        (7, None, False, NO_IC_FROM_PASSERS, 1, 2, "continuation_not_big_maru"),
        (9, 8, True, NO_IC_FROM_PASSERS, 1, 2, "continuation_not_big_maru"),
        (None, None, None, NO_IC_FROM_PASSERS, 1, 2, "all_ic_candidates_pinbar")]


def test_a_filled_prior_gap_fails_end_to_end(tmp_path):
    """A candidate whose closest prior gap is FILLED before it (landing review lens 2: every other fixture's prior gap
    is unfilled or absent): `two_immediate` with its gap 8 two-stroke-filled AT 10 — 7 loses IC condition 3 (its
    immediate gap filled before the CTS anchor), 13 (immediate gap 14) is the only candidate, its prior gap 8 filled
    -> it fails 1B (fallback_no_passer, still the IC); the CSV row reads c2 8, unfilled False — 9 (no candidate,
    tested with `classify_all`) reads the same gap UNFILLED through 8."""
    from engine_v2.tests.test_institutional_pattern import _filled_gap
    specs, cts, _ = FIX["two_immediate"]
    df = _frame(specs)
    df.attrs["imbalances"] = [_filled_gap(df, 8, 10), _far_gap(14)]
    fib = _fib_over(df, 6, cts, 0.5, 0.7)
    est = [make_cts_established(cts_anchor_idx=cts, confirmed_at=cts, price=fib.cts_price, structure_id=0, cycle_id=0)]
    rows = derive_ic_candidates(df, est, _Tracker([fib]), POIConfig(), structure_path_id="H1.main")
    path = tmp_path / "ic.csv"
    export_ic_candidates(rows, path)
    got = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert got[["ic_idx", "is_ic_candidate", "picked", "prior_imbalance_first_c2_idx", "prior_imbalance_unfilled",
                "prior_imbalance_outcome", "prior_imbalance_passer_count", "ic_candidate_count"]].values.tolist() == [
        ["7", "False", "False", "", "False", "fallback_no_passer", "0", "1"],
        ["9", "False", "False", "8", "True", "fallback_no_passer", "0", "1"],
        ["13", "True", "True", "8", "False", "fallback_no_passer", "0", "1"],
    ]


def test_the_poi_meta_carries_the_preference_and_the_outranked_pattern():
    poi, _ = _drawn(FIX["pattern_pinbar"])
    m = poi.meta
    assert (poi.ic_idx, m["ip_outcome"], m["prior_imbalance_outcome"], m["prior_imbalance_first_c2_idx"],
            m["prior_imbalance_passer_count"], m["ic_candidate_count"], m["prior_imbalance_outranked_ic_idx"],
            m["prior_imbalance_outranked_ip_rank"]) == (9, "no_pattern", PREFERRED, 8, 1, 2, 7, "3A.i")
    assert all(type(m[k]) is int for k in ("prior_imbalance_first_c2_idx", "prior_imbalance_passer_count",
                                           "ic_candidate_count", "prior_imbalance_outranked_ic_idx"))
    poi, _ = _drawn(_FALLBACK)                                          # the IC is the fallback's 7: no prior gap
    assert (poi.ic_idx, poi.meta["prior_imbalance_outcome"], poi.meta["prior_imbalance_first_c2_idx"],
            poi.meta["prior_imbalance_outranked_ic_idx"]) == (7, NO_IC_FROM_PASSERS, None, None)


def test_the_mirror_shifts_the_two_indices_and_never_the_counts():
    """A sub's POI and its CSV row: `prior_imbalance_first_c2_idx` (meta + record field) and
    `prior_imbalance_outranked_ic_idx` shift by the slice begin; the counts never."""
    from engine_v2.tests.test_event_meta_idx_keys import _Poi, _synthetic_result
    poi, rows = _drawn(FIX["pattern_pinbar"])
    row = dataclasses.replace(next(r for r in rows if r.picked), meta={})
    res = dataclasses.replace(_synthetic_result(), poi_zones=[_Poi(ic_idx=9, meta=dict(poi.meta))], ic_candidates=[row])
    d = pd.DataFrame(index=range(100))
    edm.mirror_lower_tf_result_to_entity_df(d, res, structure_path_id="p")
    (p,), (r,) = d.attrs["poi_zones"], d.attrs["ic_candidates"]
    sb = res.meta["slice_begin"]
    assert (p.meta["prior_imbalance_first_c2_idx"], p.meta["prior_imbalance_outranked_ic_idx"]) == (8 + sb, 7 + sb)
    assert (p.meta["prior_imbalance_passer_count"], p.meta["ic_candidate_count"]) == (1, 2)
    assert (r.prior_imbalance_first_c2_idx, r.prior_imbalance_passer_count, r.ic_candidate_count) == (8 + sb, 1, 2)


@pytest.mark.parametrize("specs, text", [
    # narrowed, a pattern outranked (sub 3 cycle 1's shape on the window — round 7)
    (FIX["pattern_pinbar"], "ic_reason=No pattern · only preferred IC candidate<br>ic_prior_imbalance=8 unfilled · preferred 1 of 2 "
                            "IC candidates (over pattern 3A.i at 7)"),
    # narrowed, nothing outranked (sub 0 cycle 2's shape)
    (FIX["runner_up_still_in_play"], "ic_reason=No pattern · only preferred IC candidate<br>ic_prior_imbalance=8 unfilled · "
                                     "preferred 1 of 2 IC candidates"),
    # a pattern pick from narrowed passers, nothing outranked (sub 8 cross 0→2's shape)
    (_PASSER_PATTERN, "ic_reason=IP 2B.iii · legs 1-1-3 (8 | 9 | 10–12)<br>ic_prior_imbalance=8 unfilled · preferred 1 of 2 IC "
                      "candidates"),
    # narrowed over a ranked list: the ranking line keeps its rule
    (FIX["pinbar_with_the_gap"], "ic_reason=No pattern · ranking rule exclude_pinbar<br>ic_prior_imbalance=8 unfilled · preferred 2 "
                                 "of 3 IC candidates"),
    # passers < candidates but the fallback gave the IC -> no 1B line (the user: "Nothing when … it fell back")
    (_FALLBACK, "ic_reason=No pattern · ranking rule exclude_pinbar"),
    # no passer -> no line; the lone candidate stays "only IC candidate"
    (FIX["single"], "ic_reason=No pattern · only IC candidate"),
])
def test_the_poi_hover_shows_the_preference_only_when_it_narrowed(specs, text):
    poi, _ = _drawn(specs)
    assert poi_ic_hover_text(poi.meta, poi.ic_idx) == text


def test_a_pattern_poi_hover_carries_the_line_before_the_band():
    """A pattern pick from narrowed passers (Step 1A's fixture): the IP line, then 1B's, then the band's."""
    df = _tb2_frame()
    df.attrs["imbalances"] = [_far_gap(g) for g in (8, 10, 12)]
    fib = _fib_over(df, 6, 12, *_ABOVE)
    est = [make_cts_established(cts_anchor_idx=12, confirmed_at=12, price=fib.cts_price, structure_id=0, cycle_id=0)]
    tracker = _Tracker([fib])
    recs = select_ic_records(df, est, tracker, POIConfig())
    with contextlib.redirect_stdout(io.StringIO()):
        (poi,) = derive_poi_zones(df, est, fib_tracker=tracker, config=POIConfig(), ic_records=recs)
    assert poi_ic_hover_text(poi.meta, poi.ic_idx) == (
        "ic_reason=IP 1A.i · legs 1-1-1 (8 | 9 | 10)<br>ic_tiebreak=over 11: most recent before imbalance 10<br>ic_prior_imbalance=8 "
        "unfilled · preferred 2 of 3 IC candidates (over pattern 1A.i at 7)<br>ic_zone=80-100 (fallback)")


def test_the_poi_hover_rejects_an_inconsistent_preference():
    poi, _ = _drawn(FIX["pattern_pinbar"])
    for bad in (dict(prior_imbalance_outranked_ip_rank=None),           # an IC without its rank
                dict(prior_imbalance_first_c2_idx=None),                # a preferred IC without its gap
                dict(prior_imbalance_passer_count=2)):                  # an outranked pattern without narrowing
        with pytest.raises(AssertionError):
            poi_ic_hover_text({**poi.meta, **bad}, poi.ic_idx)


@pytest.mark.parametrize("specs, ic, reason", [
    (FIX["runner_up_still_in_play"], 9, "No pattern · only preferred IC candidate"),
    (FIX["single"], 7, "No pattern · only IC candidate"),
])
def test_the_poi_hover_alone_names_the_preferred_ic(specs, ic, reason):
    """Since the user's fib-hover spec (2026-10-23) a pick's IC is on its POI hover only (the fib hover keeps an IC line
    only when it found none) — the POI hover names the preference (one narrowing test, `_poi_narrowed`)."""
    poi, rows = _drawn(specs)
    (row,) = [r for r in rows if r.picked]
    assert row.ic_idx == ic == poi.ic_idx and fib_hover_no_ic_text(row) is None
    assert poi_ic_hover_text(poi.meta, poi.ic_idx).split("<br>")[0] == f"ic_reason={reason}"


@pytest.mark.parametrize("outcome, passers, candidates, narrowed", [
    (PREFERRED, 1, 2, True), (PREFERRED, 2, 2, False), (NO_IC_FROM_PASSERS, 1, 2, False), (NO_PASSER, 0, 2, False),
    (None, None, None, False),
])
def test_narrowed_means_preferred_and_fewer_passers(outcome, passers, candidates, narrowed):
    assert prior_imbalance_narrowed(outcome, passers, candidates) is narrowed
