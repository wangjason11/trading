"""The KL / POI zone hover — the user's spec 2026-10-23
(`plans/qa_chart_edits_inputs/user_spec_20261023_zone_hover.md`).

KL:  TF / KL Zone / side / identity / base_pattern / base_idx / confirmed_idx / lifecycle / top / bottom /
     zone_size / struct_direction
POI: TF / POI Zone / side / identity / ic_idx / confirmed_idx / the labelled IC lines / lifecycle / top / bottom /
     zone_size / struct_direction
identity = `sid | cycle_id` (main: the H1 chart + the M15 charts' H1 overlay) or `sub_id | cycle_id` + the dot
hover's `first record:` line (sub). `lifecycle` = the zone's OWN active window + its end reason.
The fixtures are zones of the reference window (S_dot_hover).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from engine_v2.charting._zone_render import (
    HOVER_NEVER_ACTIVE,
    kl_zone_customdata,
    kl_zone_hovertemplate,
    main_zone_id_line,
    poi_zone_customdata,
    poi_zone_hovertemplate,
    zone_lifecycle_line,
    zone_size_text,
)
from engine_v2.charting.export_m15_chart import _sub_zone_id_lines
from engine_v2.tests.test_plan_h_landing_b import _meta
from engine_v2.tests.test_plan_l_landing_2 import _ranked_meta
from engine_v2.tests.test_structure_dot_hover import _render

PIP = 0.0001


def _act(*idx_active):
    return [{"idx": i, "active": a, "reason": "x"} for i, a in idx_active]


def _kl_sid0_c0_bos():
    """H1 sid 0 cycle 0's BOS KL zone: active from 115, ended by the next cycle at 652."""
    return SimpleNamespace(side="buy", meta={
        "structure_id": 0, "cycle_id": 0, "struct_direction": 1, "base_pattern": "base inside bar", "base_idx": 96,
        "confirmed_idx": 115, "activation_history": _act((115, True)), "end_idx": 652, "end_reason": "next_cycle"})


def _poi_sid1_c2():
    """H1 sid 1 cycle 2's POI (IC 859, ranked): on 902, off 952, on 954 (`confirmed_idx` = the most recent), open."""
    return SimpleNamespace(side="sell", ic_idx=859, meta={
        "structure_id": 1, "cycle_id": 2, "struct_direction": -1, "confirmed_idx": 954,
        "activation_history": _act((902, True), (952, False), (954, True)), "end_idx": None, "end_reason": None,
        **_ranked_meta("exclude_pinbar")})


def _poi_sub8_c2():
    """Sub 8 cycle 2's POI (IC 4131): a pattern IC Step 1B preferred, from the 80–100 fallback zone; open."""
    return SimpleNamespace(side="buy", ic_idx=4131, meta={
        "structure_id": 0, "cycle_id": 2, "struct_direction": 1, "confirmed_idx": 4179,
        "activation_history": _act((4179, True)), "end_idx": None, "end_reason": None,
        **_meta("3B.iii", "1-1-3", 4130, 4131, 4134, band="80-100"),
        "prior_imbalance_outcome": "preferred", "prior_imbalance_first_c2_idx": 4118,
        "prior_imbalance_passer_count": 1, "ic_candidate_count": 2})


# --- the shared lines -----------------------------------------------------------

def test_lifecycle_is_the_zones_own_window_with_its_end_reason():
    assert zone_lifecycle_line(_kl_sid0_c0_bos().meta) == "lifecycle=[115, 652] | end_reason=next_cycle"
    # a POI starts at its FIRST activation, not at its confirmed_idx (the most recent one)
    assert zone_lifecycle_line(_poi_sid1_c2().meta) == "lifecycle=[902, open] | end_reason=open"


def test_lifecycle_of_a_zone_that_never_activated():
    meta = {"activation_history": [], "end_idx": 1940, "end_reason": "next_cycle"}
    assert zone_lifecycle_line(meta) == f"lifecycle=[{HOVER_NEVER_ACTIVE}, 1940] | end_reason=next_cycle"


def test_lifecycle_rejects_an_end_without_its_reason():
    with pytest.raises(AssertionError, match="end_reason"):
        zone_lifecycle_line({"activation_history": _act((5, True)), "end_idx": 9, "end_reason": None})


def test_zone_size_in_pips_one_decimal():
    assert zone_size_text(0.55910, 0.55808, PIP) == "10.2 pips"
    assert zone_size_text(0.55808, 0.55910, PIP) == "10.2 pips"          # the order of the bounds does not matter
    assert zone_size_text(0.57462, 0.57422, PIP) == "4.0 pips"
    assert zone_size_text(150.25, 150.10, 0.01) == "15.0 pips"           # a JPY pair's pip


# --- main ------------------------------------------------------------------------

def test_main_kl_hover_renders_the_spec_order():
    z = _kl_sid0_c0_bos()
    row = kl_zone_customdata(z, main_zone_id_line(z.meta), 0.55910, 0.55808, PIP)
    assert _render(kl_zone_hovertemplate("1H"), row).split("<br>") == [
        "TF=1H",
        "KL Zone",
        "side=buy",
        "sid=0 | cycle_id=0",
        "base_pattern=base inside bar",
        "base_idx=96",
        "confirmed_idx=115",
        "lifecycle=[115, 652] | end_reason=next_cycle",
        "top=0.55910",
        "bottom=0.55808",
        "zone_size=10.2 pips",
        "struct_direction=1",
    ]


def test_kl_zone_has_one_active_stretch_from_its_confirmed_idx():
    z = _kl_sid0_c0_bos()
    z.meta["activation_history"] = _act((120, True))
    with pytest.raises(AssertionError, match="KL zone confirmed 115"):
        kl_zone_customdata(z, "id", 0.55910, 0.55808, PIP)


def test_main_poi_hover_renders_the_spec_order():
    z = _poi_sid1_c2()
    row = poi_zone_customdata(z, main_zone_id_line(z.meta), 0.57850, 0.57786, PIP)
    assert _render(poi_zone_hovertemplate("1H"), row).split("<br>") == [
        "TF=1H",
        "<b>POI Zone</b>",
        "side=sell",
        "sid=1 | cycle_id=2",
        "ic_idx=859",
        "confirmed_idx=954",
        "ic_reason=No pattern · ranking rule exclude_pinbar",
        "lifecycle=[902, open] | end_reason=open",
        "top=0.57850",
        "bottom=0.57786",
        "zone_size=6.4 pips",
        "struct_direction=-1",
    ]


def test_poi_that_never_activated_says_so_twice():
    z = _poi_sid1_c2()
    z.meta.update(confirmed_idx=None, activation_history=[], end_idx=1940, end_reason="next_cycle")
    lines = _render(poi_zone_hovertemplate("1H"), poi_zone_customdata(z, "id", 0.57850, 0.57786, PIP)).split("<br>")
    assert lines[5] == f"confirmed_idx={HOVER_NEVER_ACTIVE}"
    assert lines[7] == f"lifecycle=[{HOVER_NEVER_ACTIVE}, 1940] | end_reason=next_cycle"


# --- sub -------------------------------------------------------------------------

def test_sub_poi_hover_renders_the_spec_order_with_every_ic_line_labelled():
    z = _poi_sub8_c2()
    first = "first record: lens=counter | parent_sid=1 | parent_cycle_id=2"
    row = poi_zone_customdata(z, _sub_zone_id_lines(8, 2, first), 0.57462, 0.57422, PIP)
    assert _render(poi_zone_hovertemplate("15M"), row).split("<br>") == [
        "TF=15M",
        "<b>POI Zone</b>",
        "side=buy",
        "sub_id=8 | cycle_id=2",
        "first record: lens=counter | parent_sid=1 | parent_cycle_id=2",
        "ic_idx=4131",
        "confirmed_idx=4179",
        "ic_reason=IP 3B.iii · legs 1-1-3 (4130 | 4131 | 4132–4134)",
        "ic_prior_imbalance=4118 unfilled · preferred 1 of 2 IC candidates",
        "ic_zone=80-100 (fallback)",
        "lifecycle=[4179, open] | end_reason=open",
        "top=0.57462",
        "bottom=0.57422",
        "zone_size=4.0 pips",
        "struct_direction=1",
    ]


def test_sub_kl_hover_carries_the_sub_identity_and_the_first_record():
    z = _kl_sid0_c0_bos()
    first = "first record: lens=confluence | parent_sid=0 | parent_cycle_id=0"
    row = kl_zone_customdata(z, _sub_zone_id_lines(0, 0, first), 0.55910, 0.55808, PIP)
    lines = _render(kl_zone_hovertemplate("15M"), row).split("<br>")
    assert lines[:5] == ["TF=15M", "KL Zone", "side=buy", "sub_id=0 | cycle_id=0", first]
    assert len(lines) == 13
