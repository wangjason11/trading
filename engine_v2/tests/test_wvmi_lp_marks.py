"""The WVMI LP marks above the volume bars ("WVMI back on the chart" step 5, 2026-10-26; the user's design rounds 2-3,
9, 11 in `plans/qa_chart_edits_inputs/user_spec_20261024_wvmi_on_chart.md`; CHARTING_SPEC "WVMI LP marks").

One mark per per-LP row (`WVMIRecord.lp_rows`) of every NON-collapsed record: a sub's = a small dot on the M15 lens
chart at its LP candle; an H1 record's = a circle — on the H1 chart at its LP candle, on the M15 charts at the LAST M15
candle of its LP's hour (`_map_parent_idx_to_m15_hour_end`). Fixed rows on the marks' own axis `y3` over the volume
panel: sell just above the bars, buy one step higher; colour by zone side. Hover = the user's accepted drafts:
`TF` / `<Sub|Main> WVMI` / identity / `lp_idx[ | as_of_idx] | lp_status=<status> (<reason>[ at <idx>])` /
`zone_side | breakout_momentum | pullback_momentum` (3 decimals). The volume axis keeps `volume.headroom` above the
tallest bar (the static layout and the zoom JS).
"""
from __future__ import annotations

import contextlib
import io

import pandas as pd
import pytest

from engine_v2.charting._zone_render import (
    wvmi_lp_hovertext,
    wvmi_lp_idx_fields,
    wvmi_lp_marks,
    wvmi_lp_status_text,
)
from engine_v2.charting.style_registry import STYLE
from engine_v2.common.types import WVMILPRow, WVMIRecord

HEADROOM = STYLE["volume.headroom"]["factor"]
ROW_Y = STYLE["wvmi.lp_row"]


def _row(lp, as_of, status, reason, status_idx, pm):
    return WVMILPRow(lp_idx=lp, as_of_idx=as_of, lp_status=status, lp_status_reason=reason,
                     lp_status_idx=status_idx, pullback_momentum=pm)


def _rec(sid, cyc, side, bm, rows, *, sub_id=None, collapsed=False):
    meta = {} if sub_id is None else {"sub_id": sub_id, "parent_sid": 0, "parent_cycle_id": 1}
    return WVMIRecord(bos_structure_id=sid, bos_cycle_id=cyc, zone_side=side, breakout_momentum=bm,
                      lp_idx=rows[-1].lp_idx if rows else None, cycle_collapsed=collapsed,
                      lp_rows=list(rows), meta=meta)


# ---------------------------------------------------------------------------
# The pure helpers (`_zone_render`)
# ---------------------------------------------------------------------------

def test_marks_are_every_row_of_every_non_collapsed_record_in_order():
    a = _rec(0, 1, "buy", 1.0, [_row(3, 3, "temporary", "replaced by a further close", 5, 0.5),
                                _row(5, 5, "open", "still moving at the data end", None, 0.4)])
    hidden = _rec(0, 2, "buy", 1.0, [_row(8, 8, "open", "still moving at the data end", None, 0.4)], collapsed=True)
    b = _rec(1, 0, "sell", 2.0, [_row(9, 10, "locked", "cycle 1's BOS confirmed", 11, 0.3)])
    assert [(r.bos_cycle_id, row.lp_idx) for r, row in wvmi_lp_marks([a, hidden, b])] == [(1, 3), (1, 5), (0, 9)]
    assert wvmi_lp_marks([hidden]) == []


@pytest.mark.parametrize("row, text", [
    (_row(3112, 3112, "temporary", "replaced by a further close", 3118, 0.6),
     "temporary (replaced by a further close at 3118)"),
    (_row(590, 590, "locked", "cycle 1's BOS confirmed", 652, 0.5), "locked (cycle 1's BOS confirmed at 652)"),
    (_row(901, 901, "unlocked", "cycle ended by a reversal", 902, 0.5), "unlocked (cycle ended by a reversal at 902)"),
    (_row(1055, 1055, "open", "still moving at the data end", None, 0.1), "open (still moving at the data end)"),
])
def test_status_text_carries_its_why_and_when(row, text):
    assert wvmi_lp_status_text(row) == text


def test_as_of_joins_the_lp_line_only_when_it_differs():
    same = _row(4129, 4129, "locked", "x", 4157, 1.0)
    later = _row(4102, 4103, "temporary", "x", 4129, 1.0)
    assert wvmi_lp_idx_fields(same) == ["lp_idx=4129"]
    assert wvmi_lp_idx_fields(later) == ["lp_idx=4102", "as_of_idx=4103"]
    assert wvmi_lp_idx_fields(later, suffix="_1H") == ["lp_idx_1H=4102", "as_of_idx_1H=4103"]


def test_hover_is_the_accepted_draft():
    rec = _rec(0, 2, "sell", 1.28735, [], sub_id=3)
    row = _row(3112, 3112, "temporary", "replaced by a further close", 3118, 0.65849)
    assert wvmi_lp_hovertext("15M", "Sub", "sub_id=3 | cycle_id=2 | relative_dir=counter",
                             wvmi_lp_idx_fields(row), rec, row) == (
        "TF=15M<br><b>Sub WVMI</b><br>sub_id=3 | cycle_id=2 | relative_dir=counter<br>"
        "lp_idx=3112 | lp_status=temporary (replaced by a further close at 3118)<br>"
        "zone_side=sell | breakout_momentum=1.287 | pullback_momentum=0.658")


def test_hover_momentum_none_is_na_with_why():
    rec = _rec(0, 1, "buy", None, [])
    row = _row(5, 5, "open", "still moving at the data end", None, None)
    assert wvmi_lp_hovertext("1H", "Main", "sid=0 | cycle_id=1", wvmi_lp_idx_fields(row), rec, row).endswith(
        "zone_side=buy | breakout_momentum=NA (not computed) | pullback_momentum=NA (not computed)")


def test_the_rows_put_buy_one_step_above_sell():
    assert 1 / HEADROOM < ROW_Y["sell"] < ROW_Y["buy"] <= 1.0      # both rows above the tallest bar


# ---------------------------------------------------------------------------
# The two charts
# ---------------------------------------------------------------------------

def _ohlcv(n, freq):
    t = pd.date_range("2026-01-05", periods=n, freq=freq, tz="UTC")
    c = [0.600 + 0.0002 * (i % 7) for i in range(n)]
    return pd.DataFrame({"time": t, "o": c, "h": [x + 0.0003 for x in c], "l": [x - 0.0003 for x in c], "c": c,
                         "volume": [100.0 + 10 * (i % 5) for i in range(n)], "mid_price": c, "body_pct": 0.0,
                         "candle_type": "normal", "body_len": 0.0, "candle_len": 0.0006, "direction": 0})


# The H1 records: (0, 1) buy — LP 3 then 5, unlocked; (1, 2) sell — LP 10, open; (1, 0) collapsed (hidden).
_H1_WVMI = [
    _rec(0, 1, "buy", 1.102145, [_row(3, 3, "temporary", "replaced by a further close", 5, 0.759),
                                 _row(5, 5, "unlocked", "cycle ended by a reversal", 9, 0.477863)]),
    _rec(1, 0, "sell", 0.39, [_row(7, 7, "locked", "cycle 1's BOS confirmed", 8, 0.25)], collapsed=True),
    _rec(1, 2, "sell", 0.21386, [_row(10, 10, "open", "still moving at the data end", None, 0.081912)]),
]

# The lens's sub records: sub 5 c1 sell — LP 15 as of 16 (the LP before sub 5's start 16, its as-of candle on it; and
# on the candle of the H1 buy circle of H1 hour 3), then 18 locked; sub 6 c0 buy — LP 30 unlocked; sub 6 c1 collapsed.
_M15_WVMI = [
    _rec(0, 1, "sell", 1.28821, [_row(15, 16, "temporary", "replaced by a further close", 18, 5.050633),
                                 _row(18, 18, "locked", "cycle 2's BOS confirmed", 22, 0.65849)], sub_id=5),
    _rec(0, 0, "buy", 0.9, [_row(30, 30, "unlocked", "sub replaced", 35, 1.0191)], sub_id=6),
    _rec(0, 1, "buy", 0.9, [_row(40, 40, "open", "still moving at the data end", None, 1.0)], sub_id=6,
         collapsed=True),
]


def _sids():
    from engine_v2.tests.test_sub_tables_export import _sub
    fr = {"lens": "confluence", "parent_sid": 0, "parent_cycle_id": 1, "trigger_type": "first_confluence",
          "trigger_idx": 16, "start_idx": 16}
    return [_sub(5, -1, 10, 16, 40, "parent_end", None, ("confluence",), ((16, "confluence"),), 1, fr),
            _sub(6, 1, 20, 26, 40, "parent_end", None, ("confluence",), ((26, "confluence"), (28, "counter")), 1,
                 {**fr, "trigger_idx": 26, "start_idx": 26})]


def _figs(monkeypatch):
    go = pytest.importorskip("plotly.graph_objects")
    figs = []
    monkeypatch.setattr(go.Figure, "write_html", lambda self, *a, **k: figs.append(self))
    monkeypatch.setattr(go.Figure, "write_image", lambda self, *a, **k: None)
    return figs


def _render_m15(monkeypatch, tmp_path, volume=None, h1_wvmi=None, idx_range=None):
    from engine_v2.charting.export_m15_chart import export_m15_chart_plotly
    from engine_v2.multitf.registry import StructureRegistry
    figs = _figs(monkeypatch)
    h1, m15 = _ohlcv(12, "1h"), _ohlcv(48, "15min")
    h1.attrs.update(wvmi=list(_H1_WVMI if h1_wvmi is None else h1_wvmi))
    m15.attrs.update(sids=_sids(), triggers=[], events=[], poi_zones=[], wave_candles=[], wvmi=list(_M15_WVMI),
                     prev_bos_lines=[], kl_zones=[], fib_states=[], ic_candidates=[])
    reg = StructureRegistry()
    reg.register("H1.main", df=h1, timeframe="H1", role="main")
    reg.register("H1.main >> M15.confluence", df=m15, timeframe="M15", role="subordinate",
                 starting_alignment="confluence")
    cfg = {"structure": {"levels": False}, "zones": {"KL": False, "POI": False, "wave_candles": False},
           "imbalance": {"highlight": False}, "fib": {"lines": False},
           "volume": {"bars": True, "ema_line": False, "spike_marker": False, **(volume or {})}}
    with contextlib.redirect_stdout(io.StringIO()):
        export_m15_chart_plotly(title="t", registry=reg, path_id="H1.main >> M15.confluence",
                                out_dir=tmp_path, cfg=cfg, idx_range=idx_range)
    fig, = figs
    return fig, h1, m15


def _render_h1(monkeypatch, tmp_path, volume=None, h1_wvmi=None, idx_range=None):
    from engine_v2.charting.export_plotly import export_chart_plotly
    from engine_v2.multitf.registry import StructureRegistry
    figs = _figs(monkeypatch)
    h1 = _ohlcv(12, "1h")
    h1.attrs.update(wvmi=list(_H1_WVMI if h1_wvmi is None else h1_wvmi))
    reg = StructureRegistry()
    reg.register("H1.main", df=h1, timeframe="H1", role="main")
    cfg = {"structure": {"levels": False}, "struct_state": {"labels": False}, "range_visual": {"rectangles": False},
           "fib": {"lines": False}, "imbalance": {"highlight": False},
           "zones": {"wave_candles": False, "POI": False, "KL": False},
           "volume": {"bars": True, "ema_line": False, "spike_marker": False, **(volume or {})}}
    with contextlib.redirect_stdout(io.StringIO()):
        export_chart_plotly(title="t", registry=reg, path_id="H1.main", out_dir=tmp_path, cfg=cfg,
                            idx_range=idx_range)
    fig, = figs
    return fig, h1


def _traces(fig, prefix):
    return {tr.name: tr for tr in fig.data if (tr.name or "").startswith(prefix)}


def _ts(df, idxs):
    return [pd.Timestamp(df.loc[i, "time"]) for i in idxs]


def _check_marks(tr, df, idxs, side, kind):
    assert [pd.Timestamp(x) for x in tr.x] == _ts(df, idxs)
    assert list(tr.y) == [ROW_Y[side]] * len(idxs)
    assert tr.yaxis == "y3" and tr.mode == "markers" and tr.hovertemplate == "%{customdata}<extra></extra>"
    want = STYLE[f"wvmi.lp_dot.{kind}.{side}"]["marker"]
    assert (tr.marker.symbol, tr.marker.size, tr.marker.color) == (want["symbol"], want["size"], want["color"])


def test_h1_chart_draws_a_circle_per_row_at_its_lp_candle(tmp_path, monkeypatch):
    fig, h1 = _render_h1(monkeypatch, tmp_path)
    tr = _traces(fig, "WVMI LP")
    assert sorted(tr) == ["WVMI LP buy", "WVMI LP sell"]          # the collapsed (1, 0) hidden
    _check_marks(tr["WVMI LP buy"], h1, [3, 5], "buy", "main")
    _check_marks(tr["WVMI LP sell"], h1, [10], "sell", "main")
    assert list(tr["WVMI LP buy"].customdata) == [
        "TF=1H<br><b>Main WVMI</b><br>sid=0 | cycle_id=1<br>"
        "lp_idx=3 | lp_status=temporary (replaced by a further close at 5)<br>"
        "zone_side=buy | breakout_momentum=1.102 | pullback_momentum=0.759",
        "TF=1H<br><b>Main WVMI</b><br>sid=0 | cycle_id=1<br>"
        "lp_idx=5 | lp_status=unlocked (cycle ended by a reversal at 9)<br>"
        "zone_side=buy | breakout_momentum=1.102 | pullback_momentum=0.478",
    ]
    assert list(tr["WVMI LP sell"].customdata) == [
        "TF=1H<br><b>Main WVMI</b><br>sid=1 | cycle_id=2<br>"
        "lp_idx=10 | lp_status=open (still moving at the data end)<br>"
        "zone_side=sell | breakout_momentum=0.214 | pullback_momentum=0.082"]


def test_m15_chart_draws_the_lens_sub_dots_and_the_h1_circles_at_the_hours_last_candle(tmp_path, monkeypatch):
    fig, h1, m15 = _render_m15(monkeypatch, tmp_path)
    sub, main = _traces(fig, "M15 WVMI LP"), _traces(fig, "H1 WVMI LP")
    assert sorted(sub) == ["M15 WVMI LP buy", "M15 WVMI LP sell"]       # sub 6 c1 (collapsed) hidden
    _check_marks(sub["M15 WVMI LP sell"], m15, [15, 18], "sell", "sub")
    _check_marks(sub["M15 WVMI LP buy"], m15, [30], "buy", "sub")
    assert list(sub["M15 WVMI LP sell"].customdata) == [
        "TF=15M<br><b>Sub WVMI</b><br>sub_id=5 | cycle_id=1 | relative_dir=NA (before start_idx)<br>"   # at the LP
        "lp_idx=15 | as_of_idx=16 | lp_status=temporary (replaced by a further close at 18)<br>"
        "zone_side=sell | breakout_momentum=1.288 | pullback_momentum=5.051",
        "TF=15M<br><b>Sub WVMI</b><br>sub_id=5 | cycle_id=1 | relative_dir=confluence<br>"
        "lp_idx=18 | lp_status=locked (cycle 2's BOS confirmed at 22)<br>"
        "zone_side=sell | breakout_momentum=1.288 | pullback_momentum=0.658",
    ]
    assert list(sub["M15 WVMI LP buy"].customdata) == [       # relative_dir at the LP candle: sub 6's counter (28 →)
        "TF=15M<br><b>Sub WVMI</b><br>sub_id=6 | cycle_id=0 | relative_dir=counter<br>"
        "lp_idx=30 | lp_status=unlocked (sub replaced at 35)<br>"
        "zone_side=buy | breakout_momentum=0.900 | pullback_momentum=1.019"]

    # H1 hour h = M15 candles 4h .. 4h + 3: the circle at the hour's LAST one (H1 3 → 15, 5 → 23, 10 → 43)
    assert sorted(main) == ["H1 WVMI LP buy", "H1 WVMI LP sell"]
    _check_marks(main["H1 WVMI LP buy"], m15, [15, 23], "buy", "main")
    _check_marks(main["H1 WVMI LP sell"], m15, [43], "sell", "main")
    assert list(main["H1 WVMI LP buy"].customdata) == [
        "TF=1H>>15M<br><b>Main WVMI</b><br>sid=0 | cycle_id=1<br>"
        "lp_idx_1H=3 | lp_idx=15 | lp_status=temporary (replaced by a further close at 5)<br>"
        "zone_side=buy | breakout_momentum=1.102 | pullback_momentum=0.759",
        "TF=1H>>15M<br><b>Main WVMI</b><br>sid=0 | cycle_id=1<br>"
        "lp_idx_1H=5 | lp_idx=23 | lp_status=unlocked (cycle ended by a reversal at 9)<br>"
        "zone_side=buy | breakout_momentum=1.102 | pullback_momentum=0.478",
    ]
    # the circles are drawn after (over) the sub dots
    names = [tr.name for tr in fig.data]
    assert max(names.index(n) for n in sub) < min(names.index(n) for n in main)


def test_m15_chart_h1_as_of_carries_the_1h_suffix(tmp_path, monkeypatch):
    late = [_rec(0, 1, "buy", 1.0, [_row(3, 4, "temporary", "replaced by a further close", 6, 0.5)])]
    fig, _, _ = _render_m15(monkeypatch, tmp_path, h1_wvmi=late)
    cd, = _traces(fig, "H1 WVMI LP")["H1 WVMI LP buy"].customdata
    assert "<br>lp_idx_1H=3 | as_of_idx_1H=4 | lp_idx=15 | lp_status=" in cd


@pytest.mark.parametrize("render", ["h1", "m15"])
def test_marks_axis_and_volume_headroom(render, tmp_path, monkeypatch):
    fig, df = ((_render_h1(monkeypatch, tmp_path)) if render == "h1"
               else _render_m15(monkeypatch, tmp_path)[::2])
    lay = fig.layout
    assert lay.yaxis3.overlaying == "y2" and tuple(lay.yaxis3.range) == (0, 1)
    assert lay.yaxis3.fixedrange is True and lay.yaxis3.visible is False and lay.yaxis3.showgrid is False
    assert tuple(lay.yaxis2.range) == pytest.approx((0, df["volume"].max() * HEADROOM))


@pytest.mark.parametrize("render", ["h1", "m15"])
@pytest.mark.parametrize("volume", [{"wvmi_lps": False}, {"bars": False}])
def test_no_marks_without_the_toggle_or_the_volume_panel(render, volume, tmp_path, monkeypatch):
    fig = (_render_h1 if render == "h1" else _render_m15)(monkeypatch, tmp_path, volume)[0]
    assert not [tr for tr in fig.data if "WVMI LP" in (tr.name or "")]
    assert "yaxis3" not in fig.layout.to_plotly_json()


def test_volume_zoom_js_uses_the_headroom(tmp_path):
    from engine_v2.charting.export_m15_chart import _inject_volume_autoscale_js
    p = tmp_path / "c.html"
    p.write_text("")
    _inject_volume_autoscale_js(p)
    assert f"'yaxis2.range': [0, maxVol * {float(HEADROOM)!r}]" in p.read_text()


def test_h1_chart_volume_zoom_js_uses_the_headroom(tmp_path, monkeypatch):
    go = pytest.importorskip("plotly.graph_objects")
    monkeypatch.setattr(go.Figure, "write_html", lambda self, path, *a, **k: open(path, "w").close())
    monkeypatch.setattr(go.Figure, "write_image", lambda self, *a, **k: None)
    from engine_v2.charting.export_plotly import export_chart_plotly
    from engine_v2.multitf.registry import StructureRegistry
    h1 = _ohlcv(12, "1h")
    reg = StructureRegistry()
    reg.register("H1.main", df=h1, timeframe="H1", role="main")
    cfg = {"structure": {"levels": False}, "struct_state": {"labels": False}, "range_visual": {"rectangles": False},
           "fib": {"lines": False}, "imbalance": {"highlight": False},
           "zones": {"wave_candles": False, "POI": False, "KL": False},
           "volume": {"bars": True, "ema_line": False, "spike_marker": False}}
    with contextlib.redirect_stdout(io.StringIO()):
        paths = export_chart_plotly(title="t", registry=reg, path_id="H1.main", out_dir=tmp_path, cfg=cfg)
    js = paths.html_path.read_text()
    assert f"'yaxis2.range': [0, maxVol * {float(HEADROOM)!r}]" in js and "__HEADROOM__" not in js


def test_h1_chart_marks_the_lp_candle_not_the_as_of_one(tmp_path, monkeypatch):
    late = [_rec(0, 1, "buy", 1.0, [_row(3, 4, "temporary", "replaced by a further close", 6, 0.5)])]
    fig, h1 = _render_h1(monkeypatch, tmp_path, h1_wvmi=late)
    tr = _traces(fig, "WVMI LP")["WVMI LP buy"]
    _check_marks(tr, h1, [3], "buy", "main")
    assert "<br>lp_idx=3 | as_of_idx=4 | lp_status=" in tr.customdata[0]


def test_a_zoomed_export_draws_only_its_own_candles_marks(tmp_path, monkeypatch):
    fig, h1 = _render_h1(monkeypatch, tmp_path, idx_range=(4, 11))
    _check_marks(_traces(fig, "WVMI LP")["WVMI LP buy"], h1, [5], "buy", "main")        # LP 3 is outside
    fig, _, m15 = _render_m15(monkeypatch, tmp_path, idx_range=(16, 47))
    tr = _traces(fig, "")
    _check_marks(tr["M15 WVMI LP sell"], m15, [18], "sell", "sub")                     # LP 15 is outside
    _check_marks(tr["H1 WVMI LP buy"], m15, [23], "buy", "main")                        # H1 3 -> 15 is outside


def test_an_unknown_zone_side_fails_loudly():
    go = pytest.importorskip("plotly.graph_objects")
    from engine_v2.charting.export_plotly import _add_wvmi_lp_traces
    with pytest.raises(AssertionError, match="zone_side"):
        _add_wvmi_lp_traces(go.Figure(), [(pd.Timestamp("2026-01-05", tz="UTC"), "neutral", "h")],
                            kind="sub", name_prefix="x")


@pytest.mark.parametrize("volume, rng", [([], None), ([0.0, 0.0], None), ([10.0, 40.0], [0, 40.0 * HEADROOM])])
def test_volume_axis_range(volume, rng):
    from engine_v2.charting.export_plotly import _volume_axis_range
    assert _volume_axis_range(pd.Series(volume, dtype=float)) == rng
