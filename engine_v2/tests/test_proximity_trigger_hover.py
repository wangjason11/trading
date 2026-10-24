"""PLAN_Q — the zone-proximity trigger markers' POI hover value, rendered (both copies: the H1 chart and the M15
charts' H1 overlay). A POI marker's `zone_confirmed_idx` (customdata[7] on H1, [9] on M15 since the user's marker spec
2026-10-24; [8] before) = `poi_unclamped_activation_floor_idx` —
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
    assert (row[0], row[3], row[7]) == (poi.ic_idx + 1, "sd:POI", late)
    # the user's marker spec (2026-10-23 / 24): the line order; no WVMI / subsequent lists in the fixture → no sub trigger
    tr, = [t for t in fig.data if t.name == "zone_proximity:trigger"]
    assert [ln.split("=")[0] for ln in _rendered(tr).split("<br>")] == [
        "TF", "idx", "sid", "type", "trigger_price", "zone_inner", "distance_pips", "zone_confirmed_idx", "sub_trigger",
        "struct_direction"]
    assert row[8] == "NA (no sub trigger)"


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
    assert (row[0], row[1], row[5], row[9]) == (5 * 4 + 2, 5, "sd:POI", 3)
    # the M15 copy: the H1 wick (the hour's low 1.4505, a buy's sd trigger approaches from above), no struct_direction
    assert (row[6], row[7], row[8]) == ("1.45050", "1.45000", 5.0)
    tr, = [t for t in fig.data if t.name == "zone_proximity:trigger"]
    assert [ln.split("=")[0] for ln in _rendered(tr).split("<br>")] == [
        "TF", "idx", "idx_1H", "parent_sid", "parent_direction", "type", "trigger_price", "zone_inner",
        "distance_pips", "zone_confirmed_idx", "sub_trigger"]


def _rendered(tr):
    from engine_v2.tests.test_plan_h_landing_b import _rendered as render
    return render(tr)


# ---------- the marker spec's helpers (the user, 2026-10-23 / 24) ----------

def _zpt(idx, direction, inner, sid=0, cyc=1):
    return ZoneProximityTrigger(structure_id=sid, cycle_id=cyc, direction=direction, idx=idx, trigger_inner=inner,
                                zone_kind="POI", proximity_pips=8, pip_size=0.0001, timeframe="H1")


@pytest.mark.parametrize("sd, direction, price, dist", [
    (1, "sd", 0.57764, 5.2),          # H1 748 (a buy's sd: from above, the low 0.57764 over the inner 0.57712)
    (1, "opp_sd", 0.57835, -12.3),    # a buy's opp_sd: from below, the high — past the inner = negative
    (-1, "sd", 0.57835, -12.3),       # a sell's sd: from below, the high
    (-1, "opp_sd", 0.57764, 5.2),     # a sell's opp_sd: from above, the low
])
def test_the_trigger_price_is_the_wick_and_the_distance_is_signed(sd, direction, price, dist):
    from engine_v2.zones.zone_proximity import proximity_trigger_price_and_distance
    assert proximity_trigger_price_and_distance(_zpt(748, direction, 0.57712), sd, 0.57835, 0.57764) == (price, dist)


def test_the_sub_trigger_text_flags_a_collapsed_parent_cycle_and_names_none():
    from engine_v2.charting._zone_render import proximity_sub_trigger_text
    types = {(0, 1, 748): "first_counter", (1, 1, 820): "first_counter"}
    assert proximity_sub_trigger_text(_zpt(748, "sd", 0.5), types, {(1, 1)}) == "first_counter"
    assert proximity_sub_trigger_text(_zpt(820, "sd", 0.5, sid=1), types, {(1, 1)}) == (
        "first_counter (not built: collapsed parent cycle)")
    assert proximity_sub_trigger_text(_zpt(749, "opp_sd", 0.5), types, set()) == "NA (no sub trigger)"


def test_the_sub_trigger_types_read_the_detectors_lists_and_uc1():
    """`subsequent_*` from the H1 frame's final lists (by `trigger_event_idx`); `first_counter` = UC1 over the main WVMI
    records (its `trigger_event_idx` = the record's first sd trigger), silenced."""
    from types import SimpleNamespace
    from engine_v2.charting._zone_render import proximity_sub_trigger_types
    h1 = pd.DataFrame({"h": [1.0] * 12}, index=range(12))
    conf = StructureEvent(idx=6, category="STRUCTURE", type="CTS_CONFIRMED", price=1.0,
                          meta={"structure_id": 0, "cycle_id": 1, "struct_direction": 1, "confirmed_at": 6,
                                "cts_anchor_idx": 5})
    wvmi = SimpleNamespace(bos_structure_id=0, bos_cycle_id=1, meta={"triggered_by_event_idx": 7})
    h1.attrs.update(structure_events=[conf], wvmi=[wvmi], kl_zones=[],
                    subsequent_confluence_triggers=[SimpleNamespace(parent_sid=0, parent_cycle_id=1, trigger_event_idx=9)],
                    subsequent_counter_triggers=[SimpleNamespace(parent_sid=0, parent_cycle_id=1, trigger_event_idx=11)])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        got = proximity_sub_trigger_types(h1)
    assert got == {(0, 1, 7): "first_counter", (0, 1, 9): "subsequent_confluence", (0, 1, 11): "subsequent_counter"}
    assert out.getvalue() == ""
