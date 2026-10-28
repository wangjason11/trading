"""PLAN_W 9c-1 (SESSION ORDER item 9c, the user 2026-10-28: "Export + chart, toggle off"; `plans/PLAN_W_part1_last_items.md`
§3.6).

A sub's CTS KL zones — derived inside every sub projection since the multi-TF foundation, RETURNED BOS-only until 9c-1 —
reach the lens frame (`attrs["kl_zones"]`, so the lens `*_kl_zones.csv`). The M15 charts draw them exactly like the BOS
zones, but every trace and shape a CTS zone adds is in the group `sub_cts_zones`, hidden on load; the control bar's
`Sub CTS zones` button (M15 charts only) shows / hides the group.
"""
from __future__ import annotations

import collections
import dataclasses
import json

import pandas as pd
import pytest

from engine_v2.tests import test_fib_chart_render as F
from engine_v2.tests.test_plan_n_naming_audit import _two_step_zone
from engine_v2.tests.test_plan_v_registry_styles import _render_m15

GROUP = "sub_cts_zones"


def _zones():
    """Sub 5's BOS zone (two steps, `test_plan_n_naming_audit`) and a CTS zone of its next cycle (the same geometry,
    the opposite side)."""
    tz = list(pd.to_datetime(F._ohlc(48, "15min")["time"], utc=True))
    bos = _two_step_zone(tz, 14, 20, 24, end=40, sub_id=5)
    cts = dataclasses.replace(bos, source_kind="CTS", side="sell", meta={**bos.meta, "cycle_id": 2})
    return bos, cts


def _js(x):
    return json.dumps(x.to_plotly_json(), sort_keys=True, default=str)


def test_the_m15_chart_hides_exactly_the_sub_cts_zones(tmp_path, monkeypatch):
    """Rendered with and without the CTS zone: every trace / shape the CTS zone ADDS — its hover lines, its stepped
    outline, its fills, its confirm line — is in `sub_cts_zones` and hidden (`visible=False`); everything else is
    the same as without it, the BOS zone untouched (no group, shown)."""
    bos, cts = _zones()
    base = _render_m15(monkeypatch, tmp_path, m15_attrs={"kl_zones": [bos]}, zones={"KL": True})
    both = _render_m15(monkeypatch, tmp_path, m15_attrs={"kl_zones": [bos, cts]}, zones={"KL": True})
    tagged = [t for t in both.data if t.legendgroup == GROUP]
    shapes = [s for s in both.layout.shapes if s.legendgroup == GROUP]
    assert len(tagged) == len(both.data) - len(base.data) > 0
    assert len(shapes) == len(both.layout.shapes) - len(base.layout.shapes) > 0
    assert all(t.visible is False for t in tagged) and all(s.visible is False for s in shapes)
    assert collections.Counter(t.name for t in tagged) == {"M15 KL zone": 4, "M15 KL outline sub5 c2": 1}
    assert any(s.type == "line" for s in shapes) and any(s.type == "rect" for s in shapes)     # the confirm line, fills
    # the rest = the chart without the CTS zone (multisets of the full JSON)
    assert collections.Counter(_js(t) for t in both.data if t.legendgroup != GROUP) == \
        collections.Counter(_js(t) for t in base.data)
    assert collections.Counter(_js(s) for s in both.layout.shapes if s.legendgroup != GROUP) == \
        collections.Counter(_js(s) for s in base.layout.shapes)


def test_the_control_bar_carries_the_sub_cts_button_on_the_m15_charts_only():
    """`_chart_controls_cfg(..., sub_cts_group=)` → `subCtsGroup` (None on the H1 chart); the script builds the
    `Sub CTS zones: hidden` button only when the group is set, and flips it like the parent overlay."""
    from engine_v2.charting.export_plotly import _chart_controls_cfg
    from pathlib import Path
    dfx = F._ohlc(3, "15min")
    m15 = _chart_controls_cfg(dfx, tf="15M", idx_1h=["", 5, 6], parent_tf="1H", parent_group="parent_overlay",
                              sub_cts_group=GROUP)
    h1 = _chart_controls_cfg(dfx, tf="1H")
    assert (m15["subCtsGroup"], h1["subCtsGroup"]) == (GROUP, None)
    js = (Path(__file__).parents[1] / "charting" / "chart_controls.js").read_text(encoding="utf-8")
    assert "CFG.subCtsGroup" in js and "Sub CTS zones" in js and "fxsubcts" in js


def test_the_m15_html_carries_the_group(tmp_path, monkeypatch):
    """The exported M15 chart's control config names the group (the button's target)."""
    bos, cts = _zones()
    _render_m15(monkeypatch, tmp_path, m15_attrs={"kl_zones": [bos, cts]}, zones={"KL": True})
    (html,) = list(tmp_path.glob("*.html"))
    assert f'"subCtsGroup": "{GROUP}"' in html.read_text(encoding="utf-8")
