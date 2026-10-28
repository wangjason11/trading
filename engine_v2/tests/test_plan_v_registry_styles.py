"""PLAN_V 8a (SESSION ORDER item 8a, the user's scope 2026-10-28; `plans/PLAN_V_part1_minor_items.md` §1).

The chart styles the charting audit (2026-09-21) found hardcoded now read the style registry: the sub-native zone
outline (`zone.m15.outline`, audit #7), the hover-label boxes (`hoverlabel.m15_structure`, `hoverlabel.zone_proximity`,
#8), the zone hover hitboxes (`zone.kl.hover_line`, `zone.poi.hover_line`, #9) and their wave-candle / fib twins
(`wave_candle.hover_line`, `fib.hover_line`). The registry holds the very values the literals had (the landing is
byte-equal), so a revert to a literal survives the replay by construction: each pin PATCHES the key to a sentinel and
shows the drawn element follows it. Also §1.3: `run_replay` refuses a non-H1 main timeframe.

PLAN_V 8c (§3, the user 2026-10-28: "Draw it in its vol_dir colour"): a wave candle on a volume-only doji (direction
0 — a qualified candle since 8c) is drawn in its `vol_dir`'s colour; every renderer skipped it before. None exists on
the window, so the pins turn the fixture's wave candles into volume-only dojis and show the same verticals draw.
"""
from __future__ import annotations

import contextlib
import dataclasses
import io

import pandas as pd
import pytest

from engine_v2.charting.style_registry import STYLE
from engine_v2.tests import test_fib_chart_render as F

HITBOX = {"line": {"width": 7.5, "color": "rgba(1,2,3,0)"}}
OUTLINE = {"rgb": "1, 2, 3", "width": 1.25}
LABEL = {"bgcolor": "plum", "font_color": "teal"}


def _is_hitbox(line) -> bool:
    return line.width == 7.5 and line.color == "rgba(1,2,3,0)"


def _render_m15(monkeypatch, tmp_path, *, m15_attrs=None, h1_attrs=None, zones=None):
    """An M15 lens chart over `test_fib_chart_render`'s frames (subs 5 [12, 40] and 6 [30, 40]); `m15_attrs` /
    `h1_attrs` override the empty element lists."""
    go = pytest.importorskip("plotly.graph_objects")
    from engine_v2.charting.export_m15_chart import export_m15_chart_plotly
    from engine_v2.multitf.registry import StructureRegistry
    h1, m15 = F._ohlc(12, "1h"), F._ohlc(48, "15min")
    h1.attrs.update(structure_events=[], **(h1_attrs or {}))
    m15.attrs.update(sids=[F._sub_rec(5, 12), F._sub_rec(6, 30)], triggers=[], events=[], poi_zones=[],
                     wave_candles=[], wvmi=[], prev_bos_lines=[], kl_zones=[], fib_states=[], ic_candidates=[])
    m15.attrs.update(m15_attrs or {})
    reg = StructureRegistry()
    reg.register("H1.main", df=h1, timeframe="H1", role="main")
    reg.register("H1.main >> M15.confluence", df=m15, timeframe="M15", role="subordinate",
                 starting_alignment="confluence")
    figs = []
    monkeypatch.setattr(go.Figure, "write_html", lambda self, *a, **k: figs.append(self))
    monkeypatch.setattr(go.Figure, "write_image", lambda self, *a, **k: None)
    cfg = {"structure": {"levels": False}, "zones": {"KL": False, "POI": False, "wave_candles": False, **(zones or {})},
           "imbalance": {"highlight": False}, "fib": {"lines": True},
           "volume": {"bars": False, "ema_line": False, "spike_marker": False}}
    with contextlib.redirect_stdout(io.StringIO()):
        export_m15_chart_plotly(title="t", registry=reg, path_id="H1.main >> M15.confluence", out_dir=tmp_path, cfg=cfg)
    fig, = figs
    return fig


# ---------------------------------------------------------------------------
# #7 the sub-native outline + #9 the zone hitboxes (the sub's zones and the H1 overlay's)
# ---------------------------------------------------------------------------

def _kl_zones():
    from engine_v2.tests.test_plan_n_naming_audit import _two_step_zone
    tz = list(pd.to_datetime(F._ohlc(48, "15min")["time"], utc=True))
    h1tz = list(pd.to_datetime(F._ohlc(12, "1h")["time"], utc=True))
    return {"m15_attrs": {"kl_zones": [_two_step_zone(tz, 14, 20, 24, end=40, sub_id=5)]},
            "h1_attrs": {"kl_zones": [_two_step_zone(h1tz, 2, 5, 7)]}, "zones": {"KL": True}}


def _poi_zones():
    from engine_v2.tests.test_plan_h_landing_b import _meta
    from engine_v2.zones.poi_zones import POIZone
    act = {"status": "active", "end_idx": None, "end_reason": None, "struct_direction": 1, "cycle_id": 1,
           "structure_id": 0}
    sub = POIZone(start_time=F._ohlc(48, "15min").loc[18, "time"], end_time=None, side="buy", top=0.6010,
                  bottom=0.6000, ic_idx=18,
                  meta={**act, "sub_id": 5, "confirmed_idx": 22,
                        "activation_history": [{"idx": 22, "active": True, "reason": "initial"}],
                        **_meta("1A.iii", "1-1-3", 17, 18, 21)})
    h1 = POIZone(start_time=F._ohlc(12, "1h").loc[3, "time"], end_time=None, side="buy", top=0.6010, bottom=0.6000,
                 ic_idx=3, meta={**act, "confirmed_idx": 4,
                                 "activation_history": [{"idx": 4, "active": True, "reason": "initial"}],
                                 **_meta("1A.iii", "1-1-3", 2, 3, 6)})
    return {"m15_attrs": {"poi_zones": [sub]}, "h1_attrs": {"poi_zones": [h1]}, "zones": {"POI": True}}


def test_the_sub_kl_zone_outline_and_both_kl_hitboxes_read_the_registry(tmp_path, monkeypatch):
    def outline(fig):
        (tr,) = [t for t in fig.data if (t.name or "").startswith("M15 KL outline sub5")]
        return tr.line

    fig = _render_m15(monkeypatch, tmp_path, **_kl_zones())
    assert outline(fig).color.startswith("rgba(0, 0, 0, ") and outline(fig).width == 0.5      # the registry's values
    monkeypatch.setitem(STYLE, "zone.m15.outline", OUTLINE)
    monkeypatch.setitem(STYLE, "zone.kl.hover_line", HITBOX)
    fig = _render_m15(monkeypatch, tmp_path, **_kl_zones())
    assert outline(fig).color.startswith("rgba(1, 2, 3, ") and outline(fig).width == 1.25
    for name in ("M15 KL zone", "H1 KL zone (overlay)"):
        lines = [t.line for t in fig.data if t.name == name]
        assert len(lines) == 4 and all(_is_hitbox(ln) for ln in lines), name


def test_the_sub_poi_zone_outline_and_both_poi_hitboxes_read_the_registry(tmp_path, monkeypatch):
    def outlines(fig, rgb):
        return [s.line for s in fig.layout.shapes
                if s.type == "rect" and (s.line.color or "").startswith(f"rgba({rgb}, ")]

    fig = _render_m15(monkeypatch, tmp_path, **_poi_zones())
    assert [ln.width for ln in outlines(fig, "0, 0, 0")] == [0.5]
    monkeypatch.setitem(STYLE, "zone.m15.outline", OUTLINE)
    monkeypatch.setitem(STYLE, "zone.poi.hover_line", HITBOX)
    fig = _render_m15(monkeypatch, tmp_path, **_poi_zones())
    assert [ln.width for ln in outlines(fig, "1, 2, 3")] == [1.25] and not outlines(fig, "0, 0, 0")
    for name in ("M15 POI zone", "H1 POI zone (overlay)"):
        lines = [t.line for t in fig.data if t.name == name]
        assert len(lines) == 2 and all(_is_hitbox(ln) for ln in lines), name


# ---------------------------------------------------------------------------
# #8 the hover-label boxes
# ---------------------------------------------------------------------------

def test_the_m15_dot_hover_label_reads_the_registry(monkeypatch):
    go = pytest.importorskip("plotly.graph_objects")
    from engine_v2.charting.export_m15_chart import _render_m15_dots_layer
    from engine_v2.tests.test_structure_dot_hover import _FR_COUNTER, _rec, _sub
    from engine_v2.charting.export_m15_chart import _sub_dot_hover_lines
    first, fr_trigger, window, records = _sub_dot_hover_lines(_sub(3491, 3819, "same_dir_replacement", _FR_COUNTER),
                                                              [_rec()])
    t = "2026-01-06T08:00:00+00:00"

    def label():
        fig = go.Figure()
        _render_m15_dots_layer(fig, [(3304, t, 0.58105, "BOS", 0, 0, -1, 3304)], "BOS", first, fr_trigger, 5,
                               {t: ("h1t", 826)}, window, records, lambda idx: "NA", layer="recent",
                               phase_at=lambda idx: True)
        return fig.data[0].hoverlabel

    assert (label().bgcolor, label().font.color) == ("royalblue", "white")
    monkeypatch.setitem(STYLE, "hoverlabel.m15_structure", LABEL)
    assert (label().bgcolor, label().font.color) == ("plum", "teal")


def test_the_m15_prev_bos_hover_label_reads_the_registry(tmp_path, monkeypatch):
    lines = {"m15_attrs": {"prev_bos_lines": [{"start_idx": 14, "end_idx": 20, "price": 0.6005,
                                               "meta": {"sub_id": 5}}]}}

    def label():
        fig = _render_m15(monkeypatch, tmp_path, **lines)
        (tr,) = [t for t in fig.data if (t.name or "").startswith("M15 Prev BOS")]
        return tr.hoverlabel

    assert (label().bgcolor, label().font.color) == ("royalblue", "white")
    monkeypatch.setitem(STYLE, "hoverlabel.m15_structure", LABEL)
    assert (label().bgcolor, label().font.color) == ("plum", "teal")


def test_the_m15_proximity_trigger_hover_label_reads_the_registry(monkeypatch):
    go = pytest.importorskip("plotly.graph_objects")
    from engine_v2.charting.export_m15_chart import _render_proximity_triggers_overlay
    from engine_v2.structure.market_structure import StructureEvent
    from engine_v2.tests.test_proximity_trigger_hover import _poi_trigger
    from engine_v2.zones.poi_zones import POIZone
    h1 = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC"),
                       "o": 1.4600, "h": 1.4700, "l": 1.4505, "c": 1.4650})
    m15 = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=32, freq="15min", tz="UTC"),
                        "o": 1.4600, "h": 1.4700, "l": 1.4550, "c": 1.4650})
    poi = POIZone(start_time=h1.loc[3, "time"], end_time=None, side="buy", top=1.4500, bottom=1.4490, ic_idx=3,
                  meta={"structure_id": 0, "cycle_id": 0, "cts_established_idx": 2, "pattern_confirmed_idx": None,
                        "end_idx": None, "activation_history": [], "confirmed_idx": None})
    cts = StructureEvent(idx=4, category="STRUCTURE", type="CTS_CONFIRMED", price=1.47,
                         meta={"structure_id": 0, "cycle_id": 0, "struct_direction": 1, "confirmed_at": 4,
                               "cts_anchor_idx": 4})
    h1.attrs.update(structure_events=[cts], kl_zones=[], poi_zones=[poi],
                    zone_proximity_triggers={(0, 0): [_poi_trigger(poi, 5, sd=+1)]})

    def label():
        fig = go.Figure()
        with contextlib.redirect_stdout(io.StringIO()):
            _render_proximity_triggers_overlay(fig, m15, h1, (m15["h"] - m15["l"]) * 0.15)
        (tr,) = [t for t in fig.data if t.name == "zone_proximity:trigger"]
        return tr.hoverlabel

    assert (label().bgcolor, label().font.color) == ("black", "white")
    monkeypatch.setitem(STYLE, "hoverlabel.zone_proximity", LABEL)
    assert (label().bgcolor, label().font.color) == ("plum", "teal")


# ---------------------------------------------------------------------------
# the fib + wave-candle hitboxes (#9's twins) — the M15 fib on the lens chart; the H1 chart and the M15 charts' H1
# overlay on a real pipeline's H1 output
# ---------------------------------------------------------------------------

def test_the_m15_fib_hitbox_reads_the_registry(tmp_path, monkeypatch):
    m15_fibs = {"m15_attrs": {"fib_states": list(F._M15_FIBS.values()), "ic_candidates": list(F._M15_IP)}}
    fig = _render_m15(monkeypatch, tmp_path, **m15_fibs)
    assert {(t.line.width, t.line.color) for t in fig.data if t.name == "M15 fib"} == {(6, "rgba(0,0,0,0)")}
    monkeypatch.setitem(STYLE, "fib.hover_line", HITBOX)
    fig = _render_m15(monkeypatch, tmp_path, **m15_fibs)
    lines = [t.line for t in fig.data if t.name == "M15 fib"]
    assert len(lines) == 4 and all(_is_hitbox(ln) for ln in lines)          # no collapsed KL here: sub 5's c0 draws too


@pytest.fixture(scope="module")
def h1_out():
    from engine_v2.tests.test_plan_h_landing_b import _pipeline
    from engine_v2.tests.test_unified_probe import _make_ip_multicycle_data
    res, out, _ = _pipeline(_make_ip_multicycle_data)
    assert out["wave_candles"], "the fixture needs wave candles"
    return res, out


def _wave_hitboxes(fig):
    return [t.line for t in fig.data
            if t.type == "scatter" and t.mode == "lines" and "Wave Candle" in (t.hovertemplate or "")]


def test_the_h1_charts_fib_and_wave_candle_hitboxes_read_the_registry(tmp_path, monkeypatch, h1_out):
    go = pytest.importorskip("plotly.graph_objects")
    from engine_v2.charting.export_plotly import export_chart_plotly
    from engine_v2.multitf.registry import StructureRegistry
    res, out = h1_out

    def render():
        df = res.df.copy()
        df.attrs.update(structure_events=res.events, fib_states=out["fib_states"], poi_zones=out["poi_zones"],
                        ic_candidates=out["ic_candidates"], kl_zones=list(out["kl_zones"]),
                        wave_candles=out["wave_candles"])
        reg = StructureRegistry()
        reg.register("H1", df=df, timeframe="H1", role="main")
        figs = []
        monkeypatch.setattr(go.Figure, "write_html", lambda self, *a, **k: figs.append(self))
        monkeypatch.setattr(go.Figure, "write_image", lambda self, *a, **k: None)
        cfg = {"structure": {"levels": True}, "struct_state": {"labels": False}, "range_visual": {"rectangles": False},
               "fib": {"lines": True}, "imbalance": {"highlight": False},
               "zones": {"wave_candles": True, "POI": False, "KL": False},
               "volume": {"bars": False, "ema_line": False, "spike_marker": False}}
        with contextlib.redirect_stdout(io.StringIO()):
            export_chart_plotly(title="t", registry=reg, path_id="H1", out_dir=tmp_path, cfg=cfg)
        fig, = figs
        return [t.line for t in fig.data if (t.name or "").startswith("fib:")], _wave_hitboxes(fig)

    fibs, waves = render()
    assert fibs and waves
    assert {(ln.width, ln.color) for ln in fibs} == {(6, "rgba(0,0,0,0)")}
    assert {(ln.width, ln.color) for ln in waves} == {(8, "rgba(0,0,0,0)")}
    monkeypatch.setitem(STYLE, "fib.hover_line", HITBOX)
    monkeypatch.setitem(STYLE, "wave_candle.hover_line", HITBOX)
    fibs, waves = render()
    assert fibs and all(_is_hitbox(ln) for ln in fibs)
    assert waves and all(_is_hitbox(ln) for ln in waves)


def test_the_h1_overlays_wave_candle_hitbox_reads_the_registry(monkeypatch, h1_out):
    go = pytest.importorskip("plotly.graph_objects")
    from engine_v2.charting.export_m15_chart import _render_h1_overlay
    res, out = h1_out
    h1 = res.df.copy()
    h1.attrs.update(structure_events=res.events, kl_zones=out["kl_zones"], wave_candles=out["wave_candles"])
    m15 = pd.DataFrame({"time": pd.date_range(h1["time"].iloc[0], periods=len(h1) * 4, freq="15min", tz="UTC")})
    m15["l"], m15["h"] = float(h1["l"].min()), float(h1["h"].max())

    def render():
        fig = go.Figure()
        with contextlib.redirect_stdout(io.StringIO()):
            _render_h1_overlay(fig, m15, h1, {}, {}, {"labels": False}, {"levels": False},
                               {"KL": False, "POI": False, "wave_candles": True})
        return _wave_hitboxes(fig)

    waves = render()
    assert waves and {(ln.width, ln.color) for ln in waves} == {(8, "rgba(0,0,0,0)")}
    monkeypatch.setitem(STYLE, "wave_candle.hover_line", HITBOX)
    waves = render()
    assert waves and all(_is_hitbox(ln) for ln in waves)


def test_every_wave_candle_hitbox_site_reads_the_registry():
    """The M15 sub's wave-candle site has no render fixture (the per-candle lifecycle gate needs a sub's KL zones +
    CTS events); like `test_wave_candles.test_every_wave_candle_hover_site_uses_the_label_helper`, the sites are
    counted in the source — the H1 chart's one and the H1 overlay's one are rendered above."""
    from pathlib import Path
    import engine_v2.charting as charting
    root = Path(charting.__file__).parent
    counts = {name: (root / name).read_text(encoding="utf-8").count('_style("wave_candle.hover_line")')
              for name in ("export_plotly.py", "export_m15_chart.py")}
    assert counts == {"export_plotly.py": 1, "export_m15_chart.py": 2}


# ---------------------------------------------------------------------------
# §1.3 the replay refuses a non-H1 main timeframe
# ---------------------------------------------------------------------------

def test_run_replay_refuses_a_non_h1_main_timeframe(monkeypatch):
    """Without the assert an M15-main config reaches the history fetch (and then `run_pipeline`, which classifies the
    frame "H1" and reads the "H1" floor seed) — the stub records that call."""
    import engine_v2.run_replay as rr
    fetched = []

    def _fetch(**kw):
        fetched.append(kw["timeframe"])
        raise RuntimeError("the fetch ran")

    monkeypatch.setattr(rr, "CONFIG", dataclasses.replace(rr.CONFIG, timeframe="M15"))
    monkeypatch.setattr(rr, "fetch_history_with_auto_extend", _fetch)
    with pytest.raises(AssertionError, match="H1-main"):
        rr.main()
    assert fetched == []


# ---------------------------------------------------------------------------
# PLAN_V 8c: a wave candle on a volume-only doji draws in its vol_dir colour
# ---------------------------------------------------------------------------

def _as_volume_only_dojis(df, waves):
    """Every wave candle's candle -> a volume-only doji in its own direction (direction 0, vol_dir = the old direction)."""
    out = df.copy()
    out.attrs = dict(df.attrs)
    idxs = {int(i) for w in waves for i in (w.last_wave_candle_idx, w.first_wave_candle_idx)
            if i is not None and int(i) in out.index and int(out.loc[int(i), "direction"]) != 0}
    for i in idxs:
        out.loc[i, "vol_dir"] = int(out.loc[i, "direction"])
        out.loc[i, "direction"] = 0
    return out, idxs


def _wave_verticals(fig, keys):
    colours = {f"rgba({STYLE[k]['line']['color_rgb']}, {float(STYLE[k].get('opacity', 0.8))})" for k in keys}
    return sorted((str(s.x0), s.line.color) for s in fig.layout.shapes if s.type == "line" and s.line.color in colours)


def test_the_h1_chart_draws_a_volume_only_doji_wave_candle_in_its_vol_dir_colour(tmp_path, monkeypatch, h1_out):
    go = pytest.importorskip("plotly.graph_objects")
    from engine_v2.charting.export_plotly import export_chart_plotly
    from engine_v2.multitf.registry import StructureRegistry
    res, out = h1_out

    def render(df):
        df.attrs.update(structure_events=res.events, fib_states=[], poi_zones=[], ic_candidates=[],
                        kl_zones=list(out["kl_zones"]), wave_candles=out["wave_candles"])
        reg = StructureRegistry()
        reg.register("H1", df=df, timeframe="H1", role="main")
        figs = []
        monkeypatch.setattr(go.Figure, "write_html", lambda self, *a, **k: figs.append(self))
        monkeypatch.setattr(go.Figure, "write_image", lambda self, *a, **k: None)
        cfg = {"structure": {"levels": False}, "struct_state": {"labels": False}, "range_visual": {"rectangles": False},
               "fib": {"lines": False}, "imbalance": {"highlight": False},
               "zones": {"wave_candles": True, "POI": False, "KL": False},
               "volume": {"bars": False, "ema_line": False, "spike_marker": False}}
        with contextlib.redirect_stdout(io.StringIO()):
            export_chart_plotly(title="t", registry=reg, path_id="H1", out_dir=tmp_path, cfg=cfg)
        fig, = figs
        return _wave_verticals(fig, ("wave_candle.bullish", "wave_candle.bearish"))

    plain = render(res.df.copy())
    dojis, idxs = _as_volume_only_dojis(res.df, out["wave_candles"])
    assert plain and idxs
    assert render(dojis) == plain


def test_the_h1_overlay_draws_a_volume_only_doji_wave_candle_in_its_vol_dir_colour(monkeypatch, h1_out):
    go = pytest.importorskip("plotly.graph_objects")
    from engine_v2.charting.export_m15_chart import _render_h1_overlay
    res, out = h1_out
    m15 = pd.DataFrame({"time": pd.date_range(res.df["time"].iloc[0], periods=len(res.df) * 4, freq="15min",
                                              tz="UTC")})
    m15["l"], m15["h"] = float(res.df["l"].min()), float(res.df["h"].max())

    def render(h1):
        h1.attrs.update(structure_events=res.events, kl_zones=out["kl_zones"], wave_candles=out["wave_candles"])
        fig = go.Figure()
        with contextlib.redirect_stdout(io.StringIO()):
            _render_h1_overlay(fig, m15, h1, {}, {}, {"labels": False}, {"levels": False},
                               {"KL": False, "POI": False, "wave_candles": True})
        return _wave_verticals(fig, ("wave_candle.h1_overlay.bullish", "wave_candle.h1_overlay.bearish"))

    plain = render(res.df.copy())
    dojis, idxs = _as_volume_only_dojis(res.df, out["wave_candles"])
    assert plain and idxs
    assert render(dojis) == plain


def test_every_wave_candle_renderer_colours_a_direction_0_candle_by_its_vol_dir():
    """The M15 sub site has no render fixture (as above): counted in the source — the other two are rendered."""
    from pathlib import Path
    import engine_v2.charting as charting
    root = Path(charting.__file__).parent
    m15 = (root / "export_m15_chart.py").read_text(encoding="utf-8")
    h1 = (root / "export_plotly.py").read_text(encoding="utf-8")
    assert m15.count('candle_dir = int(lt_df.iloc[idx]["vol_dir"]) if "vol_dir" in lt_df.columns else 0') == 1
    assert m15.count('candle_dir = int(h1_df.loc[idx, "vol_dir"]) if "vol_dir" in h1_df.columns else 0') == 1
    assert h1.count('candle_dir = int(dfx.loc[idx, "vol_dir"]) if "vol_dir" in dfx.columns else 0') == 1
