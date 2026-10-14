"""PLAN_Q — the zone-proximity trigger markers' POI hover value, rendered (both copies: the H1 chart and the M15
charts' H1 overlay). A POI marker's `zone_confirmed_idx` (customdata[8]) = `poi_unclamped_activation_floor_idx` —
the POI's activation floor without the structure lifecycle-start term — not its CTS-established moment (on the
reference window they coincide on every POI, so the landing replay cannot tell them apart: the plan review's
finding 3) and not the activation-history stretch it showed before PLAN_Q."""
from __future__ import annotations

import contextlib
import dataclasses
import io

import pandas as pd
import pytest

from engine_v2.structure.market_structure import StructureEvent
from engine_v2.zones.poi_zones import POIZone
from engine_v2.zones.zone_proximity import ZoneProximityTrigger, poi_unclamped_activation_floor_idx


def _poi_trigger(poi: POIZone, idx: int, sd: int) -> ZoneProximityTrigger:
    return ZoneProximityTrigger(
        structure_id=int(poi.meta["structure_id"]), cycle_id=int(poi.meta["cycle_id"]), direction="sd", idx=idx,
        trigger_inner=float(poi.top) if sd == 1 else float(poi.bottom), zone_kind="POI",
        proximity_pips=8, pip_size=0.0001, timeframe="H1",
    )


def _marker_rows(fig):
    trs = [tr for tr in fig.data if tr.name == "zone_proximity:trigger"]
    assert len(trs) == 1
    return [list(r) for r in trs[0].customdata]


def test_h1_chart_poi_marker_shows_the_unclamped_activation_floor(tmp_path, monkeypatch):
    go = pytest.importorskip("plotly.graph_objects")
    from engine_v2.charting.export_plotly import export_chart_plotly
    from engine_v2.multitf.registry import StructureRegistry
    from engine_v2.tests.test_plan_h_landing_b import _pipeline
    from engine_v2.tests.test_unified_probe import _make_ip_multicycle_data

    res, out, _ = _pipeline(_make_ip_multicycle_data)
    poi0 = next(z for z in out["poi_zones"] if z.ic_idx == 14)
    est = int(poi0.meta["cts_established_idx"])
    late = max(est, poi0.ic_idx) + 2                       # a floor that is neither the establishment nor the IC
    poi = dataclasses.replace(poi0, meta={**poi0.meta, "pattern_confirmed_idx": late})
    assert poi_unclamped_activation_floor_idx(poi) == late != est

    df = res.df.copy()
    df.attrs.update(structure_events=res.events, kl_zones=out["kl_zones"], poi_zones=[poi],
                    zone_proximity_triggers={(int(poi.meta["structure_id"]), int(poi.meta["cycle_id"])):
                                             [_poi_trigger(poi, poi.ic_idx + 1, sd=+1)]})
    reg = StructureRegistry()
    reg.register("H1", df=df, timeframe="H1", role="main")
    figs = []
    monkeypatch.setattr(go.Figure, "write_html", lambda self, *a, **k: figs.append(self))
    monkeypatch.setattr(go.Figure, "write_image", lambda self, *a, **k: None)
    cfg = {"structure": {"levels": False}, "struct_state": {"labels": False},
           "range_visual": {"rectangles": False}, "fib": {"lines": False},
           "imbalance": {"highlight": False}, "zones": {"KL": False, "wave_candles": False, "POI": False},
           "volume": {"bars": False, "ema_line": False, "spike_marker": False}}
    with contextlib.redirect_stdout(io.StringIO()):
        export_chart_plotly(title="t", registry=reg, path_id="H1", out_dir=tmp_path, cfg=cfg)
    fig, = figs
    row, = _marker_rows(fig)
    assert (row[1], row[5], row[7], row[8]) == (poi.ic_idx + 1, "sd:POI", "POI", late)


def test_m15_overlay_poi_marker_shows_the_unclamped_activation_floor():
    go = pytest.importorskip("plotly.graph_objects")
    from engine_v2.charting.export_m15_chart import _render_proximity_triggers_overlay

    h1 = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC"),
                       "o": 1.4600, "h": 1.4700, "l": 1.4505, "c": 1.4650})
    m15 = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=32, freq="15min", tz="UTC")})
    m15["o"], m15["h"], m15["c"] = 1.4600, 1.4700, 1.4650
    m15["l"] = [1.4505 if i % 4 == 2 else 1.4550 for i in range(32)]      # each hour's low on its 3rd quarter
    # IC 3 past the establishment 2 (the fib's CTS extends until its confirmation): the floor is the IC.
    poi = POIZone(start_time=h1.loc[3, "time"], end_time=None, side="buy", top=1.4500, bottom=1.4490, ic_idx=3,
                  meta={"structure_id": 0, "cycle_id": 0, "cts_established_idx": 2, "pattern_confirmed_idx": None,
                        "end_idx": None, "activation_history": [], "confirmed_idx": None})
    assert poi_unclamped_activation_floor_idx(poi) == 3
    cts = StructureEvent(idx=4, category="STRUCTURE", type="CTS_CONFIRMED", price=1.47,
                         meta={"structure_id": 0, "cycle_id": 0, "struct_direction": 1, "confirmed_at": 4})
    h1.attrs.update(structure_events=[cts], kl_zones=[], poi_zones=[poi],
                    zone_proximity_triggers={(0, 0): [_poi_trigger(poi, 5, sd=+1)]})
    fig = go.Figure()
    with contextlib.redirect_stdout(io.StringIO()):
        _render_proximity_triggers_overlay(fig, m15, h1, (m15["h"] - m15["l"]) * 0.15)
    row, = _marker_rows(fig)
    assert (row[0], row[1], row[5], row[8]) == (5 * 4 + 2, 5, "sd:POI", 3)
