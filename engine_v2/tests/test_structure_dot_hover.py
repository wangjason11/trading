"""The BOS / CTS dot hover — the user's spec 2026-10-22
(`plans/qa_chart_edits_inputs/user_spec_20261022_dot_hover.md`).

Main (the H1 chart + the M15 charts' H1 overlay, one template):
    TF / idx / kind / price / `sid | cycle_id` / `trigger | trigger_idx` /
    `struct_window=[start, end] | end_reason` / struct_direction
Sub (the M15 lens charts):
    TF / idx / idx_1H / kind / price / `sub_id | cycle_id` / `first record: lens | parent_sid |
    parent_cycle_id` / `fr trigger | fr trigger_idx` / `lens records:` + one line per record /
    `phase | layer` / `sub_window=[start, end] | end_reason` / relative_dir / struct_direction
"""
from __future__ import annotations

import re
from types import SimpleNamespace

import plotly.graph_objects as go
import pytest

from engine_v2.charting._zone_render import (
    MAIN_DOT_HOVERTEMPLATE,
    main_dot_customdata,
    main_dot_structure_lines,
)
from engine_v2.charting.export_m15_chart import (
    RELATIVE_DIR_BEFORE_START,
    _render_m15_dots_layer,
    _sub_dot_hover_lines,
    _sub_relative_dir_at,
)
from engine_v2.multitf.types import SidRecord
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.tests._event_factory import make_bos_confirmed, make_cts_established

_FIELD = re.compile(r"%\{customdata\[(\d+)\](?::([^}]*))?\}")


def _render(template: str, row: list) -> str:
    """The hover Plotly shows: the template filled with one customdata row."""
    def rep(m):
        v = row[int(m.group(1))]
        return format(v, m.group(2)) if m.group(2) else str(v)
    return _FIELD.sub(rep, template).replace("<extra></extra>", "")


def _reversal(sid: int, idx: int) -> StructureEvent:
    return StructureEvent(idx=idx, category="STATE", type="STATE_CHANGED", price=None,
                          meta={"structure_id": sid, "to": "reversal"})


def _main_events():
    """sid 0 (start_trading): BOS_0 anchored 96, established at 115, reversed at 902;
    sid 1: established at 905, reversed at 950; sid 2: established at 960, open."""
    return [
        make_cts_established(cts_anchor_idx=110, confirmed_at=115, structure_id=0),
        make_bos_confirmed(bos_anchor_idx=96, confirmed_at=115, structure_id=0),
        _reversal(0, 902),
        make_cts_established(cts_anchor_idx=903, confirmed_at=905, structure_id=1, struct_direction=-1),
        _reversal(1, 950),
        make_cts_established(cts_anchor_idx=955, confirmed_at=960, structure_id=2),
    ]


# --- main ---------------------------------------------------------------------

def test_main_lines_start_trading_reversal_and_open():
    lines = main_dot_structure_lines(_main_events())
    assert lines == {
        0: ("trigger=start_trading | trigger_idx=NA", "struct_window=[115, 902] | end_reason=reversal"),
        1: ("trigger=reversal | trigger_idx=902", "struct_window=[902, 950] | end_reason=reversal"),
        2: ("trigger=reversal | trigger_idx=950", "struct_window=[950, open] | end_reason=open"),
    }


def test_main_dot_hover_renders_the_spec_order():
    lines = main_dot_structure_lines(_main_events())
    p = (905, "t", 0.57112, "CTS", 1, 2, -1)          # (idx, time, price, kind, sid, cycle, sd)
    row = main_dot_customdata(p, lines)
    assert _render(MAIN_DOT_HOVERTEMPLATE, row).split("<br>") == [
        "TF=1H",
        "idx=905",
        "kind=CTS",
        "price=0.57112",
        "sid=1 | cycle_id=2",
        "trigger=reversal | trigger_idx=902",
        "struct_window=[902, 950] | end_reason=reversal",
        "struct_direction=-1",
    ]


# --- sub ----------------------------------------------------------------------

def _rec(**kw) -> SimpleNamespace:
    base = dict(lens="confluence", parent_sid=0, parent_cycle_id=1, trigger_type="reversal",
                trigger_idx=3589, start_idx=3589, is_zero_length=False, seq=0)
    base.update(kw)
    return SimpleNamespace(**base)


def _sub(start_idx, end_idx, end_reason, first_record) -> SidRecord:
    return SidRecord(
        sub_sid=None, starting_sd=-1, creation_event_idx=3304, end_event_idx=end_idx,
        end_reason=end_reason, sub_id=5, start_idx=start_idx, lenses=("confluence", "counter"),
        meta={"first_record": first_record},
    )


_FR_COUNTER = {"lens": "counter", "parent_sid": 0, "parent_cycle_id": 1,
               "trigger_type": "subsequent_counter", "trigger_idx": 3491, "start_idx": 3491}


def test_sub_lines_first_record_on_the_other_lens_two_records():
    """Confluence sub 5 on S_S3: the first record is its COUNTER record (hence the lens);
    this lens's two records go one per line."""
    recs = [_rec(), _rec(parent_sid=1, parent_cycle_id=2, trigger_type="first_confluence",
                         trigger_idx=3611, start_idx=3611, seq=1)]
    first, fr_trigger, window, records = _sub_dot_hover_lines(
        _sub(3491, 3819, "same_dir_replacement", _FR_COUNTER), recs)
    assert first == "first record: lens=counter | parent_sid=0 | parent_cycle_id=1"
    assert fr_trigger == "fr trigger=subsequent_counter | fr trigger_idx=3491"
    assert window == "sub_window=[3491, 3819] | end_reason=same_dir_replacement"
    assert records.split("<br>") == [
        "lens records:",
        "confluence(0,1) | trigger=reversal | trigger_idx=3589 | start_idx=3589",
        "confluence(1,2) | trigger=first_confluence | trigger_idx=3611 | start_idx=3611",
    ]


def test_sub_lines_open_window_zero_length_and_no_records():
    sub = _sub(4143, None, None, _FR_COUNTER)
    _, _, window, records = _sub_dot_hover_lines(sub, [_rec(is_zero_length=True)])
    assert window == "sub_window=[4143, open] | end_reason=open"
    assert records.split("<br>")[1].endswith(" | start_idx=3589 | zero_length")
    assert _sub_dot_hover_lines(sub, [])[3] == "lens records: none"


def test_sub_lines_need_the_first_record():
    with pytest.raises(KeyError):
        _sub_dot_hover_lines(_sub(3491, 3819, "reversal", {}), [_rec()])


def test_relative_dir_na_before_start_then_the_step_function():
    segs = ((3491, "counter"), (3589, "confluence"))
    assert RELATIVE_DIR_BEFORE_START == "NA (before start_idx)"
    assert _sub_relative_dir_at(3304, segs, 3491, 5) == RELATIVE_DIR_BEFORE_START
    assert _sub_relative_dir_at(3490, segs, 3491, 5) == RELATIVE_DIR_BEFORE_START
    assert _sub_relative_dir_at(3491, segs, 3491, 5) == "counter"
    assert _sub_relative_dir_at(3588, segs, 3491, 5) == "counter"
    assert _sub_relative_dir_at(3589, segs, 3491, 5) == "confluence"
    assert _sub_relative_dir_at(3600, segs, None, 5) == RELATIVE_DIR_BEFORE_START   # never started


def test_relative_dir_live_without_a_segment_fails_loudly():
    with pytest.raises(AssertionError, match="no relative_dir"):
        _sub_relative_dir_at(3500, (), 3491, 5)


def test_sub_dot_hover_renders_the_spec_order():
    recs = [_rec(), _rec(parent_sid=1, parent_cycle_id=2, trigger_type="first_confluence",
                         trigger_idx=3611, start_idx=3611, seq=1)]
    first, fr_trigger, window, records = _sub_dot_hover_lines(
        _sub(3491, 3819, "same_dir_replacement", _FR_COUNTER), recs)
    fig = go.Figure()
    t = "2026-01-06T08:00:00+00:00"
    pts = [(3304, t, 0.58105, "BOS", 0, 0, -1, 3304)]   # (idx, time, price, kind, sid, cycle, sd, full idx)
    _render_m15_dots_layer(
        fig, pts, "BOS", first, fr_trigger, 5, {t: ("h1t", 826)}, window, records,
        lambda idx: _sub_relative_dir_at(idx, ((3491, "counter"),), 3491, 5),
        layer="recent", phase_at=lambda idx: idx >= 3491,
    )
    tr = fig.data[0]
    assert _render(tr.hovertemplate, list(tr.customdata[0])).split("<br>") == [
        "TF=15M",
        "idx=3304",
        "idx_1H=826",
        "kind=BOS",
        "price=0.58105",
        "sub_id=5 | cycle_id=0",
        "first record: lens=counter | parent_sid=0 | parent_cycle_id=1",
        "fr trigger=subsequent_counter | fr trigger_idx=3491",
        "lens records:",
        "confluence(0,1) | trigger=reversal | trigger_idx=3589 | start_idx=3589",
        "confluence(1,2) | trigger=first_confluence | trigger_idx=3611 | start_idx=3611",
        "phase=forming | layer=recent",
        "sub_window=[3491, 3819] | end_reason=same_dir_replacement",
        "relative_dir=NA (before start_idx)",
        "struct_direction=-1",
    ]
