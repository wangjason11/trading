"""The chart control bar (item 5's final part, 2026-10-27 — CHARTING_SPEC §14; the design rounds, the prototype and
the plan: `plans/qa_chart_edits_inputs/user_spec_20261027_parent_toggle_and_zoom.md`).

Pinned here: the parent-overlay group on the M15 charts (`export_m15_chart._tag_parent_overlay` — exactly what
`_render_h1_overlay` drew; the proximity triggers and the H1 WVMI LP circles, drawn after it, stay out — the user's
"All except both"), the bar's config (`export_plotly._chart_controls_cfg` — the candle table from the DRAWN frame, the
`chart.controls` knobs) and its injection into both charts' HTML. The script's behaviour in a browser (the button, - / +,
the wheel, Box zoom | Pan, the idx box) is checked headless on the replay's charts (`proto_headless_check.py`).
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import pandas as pd
import pytest

from engine_v2.charting.style_registry import STYLE
from engine_v2.tests import test_fib_chart_render as F
from engine_v2.tests import test_wvmi_lp_marks as W

GROUP = "parent_overlay"
VOLUME = {"bars": True, "ema_line": False, "spike_marker": False, "wvmi_lps": True}


def _capture(monkeypatch):
    go = pytest.importorskip("plotly.graph_objects")
    figs = []
    monkeypatch.setattr(go.Figure, "write_html", lambda self, *a, **k: figs.append(self))
    monkeypatch.setattr(go.Figure, "write_image", lambda self, *a, **k: None)
    return figs


def _render_m15(monkeypatch, tmp_path, *, h1_attrs=None, zones=None, volume=None, idx_range=None):
    """An M15 lens chart with shapes AND traces before the H1 overlay (the candles, the sub fibs) and after it (the
    proximity triggers, the sub WVMI dots + the H1 WVMI circles, the chart border). Returns (fig, the HTML text = the appended scripts only)."""
    from engine_v2.charting.export_m15_chart import export_m15_chart_plotly
    from engine_v2.multitf.registry import StructureRegistry
    figs = _capture(monkeypatch)
    h1, m15 = W._ohlcv(12, "1h"), W._ohlcv(48, "15min")
    h1.attrs.update(wvmi=list(W._H1_WVMI), **(h1_attrs or {}))
    m15.attrs.update(sids=W._sids(), triggers=[], events=[], poi_zones=[], wave_candles=[], wvmi=list(W._M15_WVMI),
                     prev_bos_lines=[], kl_zones=[], fib_states=list(F._M15_FIBS.values()),
                     ic_candidates=list(F._M15_IP))
    reg = StructureRegistry()
    reg.register("H1.main", df=h1, timeframe="H1", role="main")
    reg.register("H1.main >> M15.confluence", df=m15, timeframe="M15", role="subordinate",
                 starting_alignment="confluence")
    cfg = {"structure": {"levels": False}, "zones": zones or {"KL": False, "POI": False, "wave_candles": False},
           "imbalance": {"highlight": False}, "fib": {"lines": True}, "volume": volume or VOLUME}
    with contextlib.redirect_stdout(io.StringIO()):
        paths = export_m15_chart_plotly(title="t", registry=reg, path_id="H1.main >> M15.confluence",
                                        out_dir=tmp_path, cfg=cfg, idx_range=idx_range)
    fig, = figs
    return fig, paths.html_path.read_text(encoding="utf-8")


def _render_h1(monkeypatch, tmp_path, *, volume=None, idx_range=None):
    from engine_v2.charting.export_plotly import export_chart_plotly
    from engine_v2.multitf.registry import StructureRegistry
    figs = _capture(monkeypatch)
    reg = StructureRegistry()
    reg.register("H1.main", df=W._ohlcv(12, "1h"), timeframe="H1", role="main")
    cfg = {"structure": {"levels": False}, "struct_state": {"labels": False}, "range_visual": {"rectangles": False},
           "fib": {"lines": False}, "imbalance": {"highlight": False},
           "zones": {"wave_candles": False, "POI": False, "KL": False}, "volume": volume or VOLUME}
    with contextlib.redirect_stdout(io.StringIO()):
        paths = export_chart_plotly(title="t", registry=reg, path_id="H1.main", out_dir=tmp_path, cfg=cfg,
                                    idx_range=idx_range)
    fig, = figs
    return fig, paths.html_path.read_text(encoding="utf-8")


def _cfg_in(html):
    """The control bar's config as injected (the one `var CFG = {...};`)."""
    assert html.count("var CFG = ") == 1 and html.count("(function initChartControls()") == 1
    return json.JSONDecoder().raw_decode(html, html.index("var CFG = ") + len("var CFG = "))[0]


def _tagged(items):
    return [it for it in items if it.legendgroup == GROUP]


# ---------------------------------------------------------------------------
# the parent-overlay group
# ---------------------------------------------------------------------------

def test_the_group_is_exactly_what_the_h1_overlay_draws(tmp_path, monkeypatch):
    """A stub overlay between the sub elements (before) and the proximity triggers / the WVMI marks / the border
    (after): exactly its two traces and two shapes join the group; the sub fibs keep their own group."""
    go = pytest.importorskip("plotly.graph_objects")
    from engine_v2.charting import export_m15_chart as M

    def overlay(fig, *a, **k):
        fig.add_trace(go.Scatter(x=[], y=[], name="stub overlay 1"))
        fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, name="stub overlay shape 1")
        fig.add_trace(go.Scatter(x=[], y=[], name="stub overlay 2"))
        fig.add_shape(type="rect", x0=0, x1=1, y0=0, y1=1, name="stub overlay shape 2")

    monkeypatch.setattr(M, "_render_h1_overlay", overlay)
    monkeypatch.setattr(M, "_render_proximity_triggers_overlay",
                        lambda fig, *a, **k: fig.add_trace(go.Scatter(x=[], y=[], name="zone_proximity:trigger")))
    fig, _ = _render_m15(monkeypatch, tmp_path)
    names = [tr.name for tr in fig.data]
    assert [tr.name for tr in _tagged(fig.data)] == ["stub overlay 1", "stub overlay 2"]
    assert [s.name for s in _tagged(fig.layout.shapes)] == ["stub overlay shape 1", "stub overlay shape 2"]
    # the fixture really draws on both sides of the overlay
    before = names[:names.index("stub overlay 1")]
    after = names[names.index("stub overlay 2") + 1:]
    assert "OHLC" in before and "M15 fib" in before
    assert {"zone_proximity:trigger", "M15 WVMI LP sell", "H1 WVMI LP buy", "H1 WVMI LP sell"} <= set(after)
    shape_names = [s.name for s in fig.layout.shapes]
    first = shape_names.index("stub overlay shape 1")
    assert first > 0 and all(s.legendgroup == "m15_fib" for s in fig.layout.shapes[:first])
    assert len(fig.layout.shapes) == first + 2 + 4                       # + the chart border (4 lines)


def test_the_real_overlay_tags_its_elements_and_not_the_wvmi_circles(tmp_path, monkeypatch):
    """One pin on the REAL `_render_h1_overlay`: an H1 Prev BOS line and an H1 POI zone (its fill, outline and confirm
    line shapes + two hover traces) join the group; the H1 WVMI circles and every sub element stay out."""
    from engine_v2.tests.test_plan_h_landing_b import _meta
    from engine_v2.zones.poi_zones import POIZone
    h1_times = W._ohlcv(12, "1h")["time"]
    meta = {"structure_id": 0, "cycle_id": 1, "struct_direction": 1, "status": "active", "confirmed_idx": 4,
            "end_idx": None, "end_reason": None, "activation_history": [{"idx": 4, "active": True, "reason": "initial"}],
            **_meta("1A.iii", "1-1-3", 2, 3, 6)}
    poi = POIZone(start_time=h1_times[3], end_time=None, side="buy", top=0.6010, bottom=0.6000, ic_idx=3, meta=meta)
    prev_bos = {"start_idx": 2, "end_idx": 6, "price": 0.6005, "prev_structure_id": 0}
    fig, _ = _render_m15(monkeypatch, tmp_path, h1_attrs={"poi_zones": [poi], "prev_bos_lines": [prev_bos]},
                         zones={"KL": False, "POI": True, "wave_candles": False})
    assert sorted(tr.name for tr in _tagged(fig.data)) == [
        "H1 POI zone (overlay)", "H1 POI zone (overlay)", "H1 Prev BOS (sid=0)"]
    assert not [tr for tr in fig.data if "WVMI LP" in (tr.name or "") and tr.legendgroup == GROUP]
    shapes = fig.layout.shapes
    n_fib = sum(s.legendgroup == "m15_fib" for s in shapes)
    assert n_fib and len(_tagged(shapes)) == len(shapes) - n_fib - 4 >= 3  # every other shape but the border


def test_the_h1_chart_has_no_parent_group(tmp_path, monkeypatch):
    fig, _ = _render_h1(monkeypatch, tmp_path)
    assert not _tagged(fig.data) and not _tagged(fig.layout.shapes)


# ---------------------------------------------------------------------------
# the config + the injection
# ---------------------------------------------------------------------------

def _frame(times, index):
    return pd.DataFrame({"time": times, "l": [0.5, 0.6, 0.7], "h": [0.51, 0.61, 0.71]}, index=index)


def test_cfg_candle_table_and_knobs():
    from engine_v2.charting.export_plotly import _chart_controls_cfg
    t = pd.date_range("2026-01-05 21:00", periods=3, freq="1h", tz="UTC")
    dfx = _frame(t, [3, 4, 7])                                            # an idx gap stays a gap
    st = STYLE["chart.controls"]
    knobs = {"half": st["idx_half_window"], "pricePad": st["price_pad"], "minCandles": st["min_candles"],
             "wheelStep": st["wheel_step"], "markColor": st["idx_mark_color"]}
    assert _chart_controls_cfg(dfx, tf="1H") == {
        "tf": "1H", "parentTf": None, "parentGroup": None, "subCtsGroup": None, **knobs,      # (PLAN_W 9c-1)
        "candles": [[3, "2026-01-05 21:00:00", 0.5, 0.51], [4, "2026-01-05 22:00:00", 0.6, 0.61],
                    [7, "2026-01-05 23:00:00", 0.7, 0.71]]}
    cfg = _chart_controls_cfg(dfx, tf="15M", idx_1h=["", 5, 6], parent_tf="1H", parent_group=GROUP)
    assert (cfg["tf"], cfg["parentTf"], cfg["parentGroup"]) == ("15M", "1H", GROUP)
    assert [row[4] for row in cfg["candles"]] == [None, 5, 6]               # "" = unmapped
    assert knobs == {"half": 48, "pricePad": 0.06, "minCandles": 10, "wheelStep": 1.2,
                     "markColor": "rgba(30,120,255,0.18)"}                  # the values the user looked at


@pytest.mark.parametrize("times", [
    pd.date_range("2026-01-05 21:00", periods=3, freq="1h"),                                   # naive = UTC
    pd.date_range("2026-01-05 16:00", periods=3, freq="1h", tz="America/New_York"),          # -> the UTC clock
])
def test_cfg_time_is_the_utc_wall_clock(times):
    from engine_v2.charting.export_plotly import _chart_controls_cfg
    got = [row[1] for row in _chart_controls_cfg(_frame(times, [0, 1, 2]), tf="1H")["candles"]]
    assert got == ["2026-01-05 21:00:00", "2026-01-05 22:00:00", "2026-01-05 23:00:00"]


def test_cfg_refuses_an_unordered_frame_or_a_misaligned_idx_1h():
    from engine_v2.charting.export_plotly import _chart_controls_cfg
    t = pd.date_range("2026-01-05", periods=3, freq="1h", tz="UTC")
    with pytest.raises(AssertionError, match="time order"):
        _chart_controls_cfg(_frame(t[::-1], [0, 1, 2]), tf="1H")
    with pytest.raises(AssertionError, match="align"):
        _chart_controls_cfg(_frame(t, [0, 1, 2]), tf="15M", idx_1h=[0, 0])


def test_inject_appends_the_script_with_its_cfg(tmp_path):
    from engine_v2.charting.export_plotly import _inject_chart_controls_js
    p = tmp_path / "c.html"
    p.write_text("<html>chart</html>", encoding="utf-8")
    cfg = {"tf": "1H", "parentTf": None, "parentGroup": None, "candles": [[0, "2026-01-05 00:00:00", 0.5, 0.6]],
           "half": 48, "pricePad": 0.06, "minCandles": 10, "wheelStep": 1.2, "markColor": "x"}
    _inject_chart_controls_js(p, cfg)
    html = p.read_text(encoding="utf-8")
    assert html.startswith("<html>chart</html>\n<script>\n") and html.rstrip().endswith("</script>")
    assert "__CFG__" not in html and _cfg_in(html) == cfg
    js = Path(__file__).resolve().parents[1] / "charting" / "chart_controls.js"
    assert js.read_bytes().isascii()                                     # appended as utf-8 to a utf-8 page


# ---------------------------------------------------------------------------
# both exporters
# ---------------------------------------------------------------------------

def test_m15_chart_carries_the_bar_with_its_parent_and_the_hover_idx_1h(tmp_path, monkeypatch):
    _, html = _render_m15(monkeypatch, tmp_path)
    cfg = _cfg_in(html)
    assert (cfg["tf"], cfg["parentTf"], cfg["parentGroup"]) == ("15M", "1H", GROUP)
    m15 = W._ohlcv(48, "15min")
    assert [row[:4] for row in cfg["candles"]] == [
        [i, t.strftime("%Y-%m-%d %H:%M:%S"), lo, hi] for i, t, lo, hi in zip(m15.index, m15["time"], m15["l"], m15["h"])]
    assert [row[4] for row in cfg["candles"]] == [i // 4 for i in range(48)]   # H1 hour h = M15 4h .. 4h + 3
    assert html.index("initVolumeAutoscale") < html.index("initChartControls")


def test_h1_chart_carries_the_bar_without_a_parent(tmp_path, monkeypatch):
    cfg = _cfg_in(_render_h1(monkeypatch, tmp_path)[1])
    assert (cfg["tf"], cfg["parentTf"], cfg["parentGroup"]) == ("1H", None, None)
    assert [row[0] for row in cfg["candles"]] == list(range(12)) and all(len(row) == 4 for row in cfg["candles"])


@pytest.mark.parametrize("render", ["h1", "m15"])
def test_the_bar_needs_no_volume_panel_and_reads_the_drawn_slice(render, tmp_path, monkeypatch):
    no_volume = {"bars": False, "ema_line": False, "spike_marker": False}
    if render == "h1":
        html = _render_h1(monkeypatch, tmp_path, volume=no_volume, idx_range=(4, 9))[1]
    else:
        html = _render_m15(monkeypatch, tmp_path, volume=no_volume, idx_range=(16, 31))[1]
    assert "initVolumeAutoscale" not in html
    lo, hi = (4, 9) if render == "h1" else (16, 31)
    assert [row[0] for row in _cfg_in(html)["candles"]] == list(range(lo, hi + 1))
