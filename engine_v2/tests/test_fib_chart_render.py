"""The fib layer on both charts (PLAN_H §8.9 F1, the user 2026-10-06).

ONE draw gate (`_zone_render.fib_record_drawn`: not collapsed, not disappeared,
locked OR active — ended or not), ONE hover status cell (`fib_hover_status_text`:
the status, `life=[start, end] reason`, the cycle's IP outcome from
`ip_rows_by_fib_record`) and ONE box drawer (`export_plotly._add_fib_box`), used
by the H1 chart and the M15 lens charts. The M15 charts draw each sub's fib
records (they were CSV-only) in one legend group, "M15 fib".
"""
from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace

import pandas as pd
import pytest

from engine_v2.charting._zone_render import (
    fib_hover_status_text,
    fib_is_bright,
    fib_record_drawn,
    ip_rows_by_fib_record,
)
from engine_v2.charting.style_registry import STYLE
from engine_v2.features.fibonacci import FibRetracement
from engine_v2.zones.fib_tracker import FibState
from engine_v2.zones.poi_zones import IPCandidateRecord


def _fib(sid=0, cyc=1, *, active=True, locked=False, status="active", start=None, end=None,
         end_reason=None, bos=10, cts=20, sub_id=None):
    return FibState(
        structure_id=sid, cycle_id=cyc, struct_direction=1,
        bos_idx=bos, bos_price=0.60, cts_idx=cts, cts_price=0.61,
        active=active, locked=locked,
        fib=FibRetracement(anchor_high=0.61, anchor_low=0.60, direction=1, levels=()),
        meta={} if sub_id is None else {"sub_id": sub_id},
        start_idx=start, end_idx=end, end_reason=end_reason, status=status,
    )


def _ip(sid=0, cyc=1, *, bos=10, cts=20, ic=None, rank=None, reason=None, picked=False, sub_id=None):
    return IPCandidateRecord(
        structure_id=sid, cycle_id=cyc, struct_direction=1, bos_anchor_idx=bos, cts_anchor_idx=cts,
        ic_idx=ic, candle_type=None, agreed_dir=None, touches_zone=None, ip_rank=rank, reason=reason,
        pattern_anchor_idx=None, opp_leg_first_idx=None, pattern_end_idx=None, pattern_confirmed_idx=None,
        ip_leg_candle_counts=None, picked=picked, meta={} if sub_id is None else {"sub_id": sub_id},
    )


def _collapsed_kl(sid, cyc, sub_id=None):
    """A BOS KL zone of a COLLAPSED cycle (`is_collapsed_cycle_zone`: inactive, clamped confirmed >= end)."""
    meta = {"status": "inactive", "end_idx": 12, "confirmed_idx": 12, "structure_id": sid, "cycle_id": cyc}
    if sub_id is not None:
        meta["sub_id"] = sub_id
    return SimpleNamespace(source_kind="BOS", meta=meta)


# ---------------------------------------------------------------------------
# The draw gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rec, drawn", [
    (_fib(locked=True, status="ended", start=5, end=9, end_reason="new_cycle"), True),       # locked, ended
    (_fib(status="active", start=5), True),                                                    # the live version
    (_fib(status="ended", start=5, end=9, end_reason="parent_end"), True),                    # ended never-locked (F1)
    (_fib(status="ended", start=5, end=9, end_reason="new_cycle"), True),                     # ... any end reason
    (_fib(active=False, status="active", start=5), False),                                    # condition-inactive
    (_fib(active=False, status="ended", start=5, end=9, end_reason="parent_end"), False),     # ... also when ended
    (_fib(locked=True, status="disappeared"), False),                                         # scenario1_revert
])
def test_draw_gate(rec, drawn):
    assert fib_record_drawn(rec, set()) is drawn


def test_draw_gate_hides_a_collapsed_cycle_even_when_locked():
    rec = _fib(sid=1, cyc=1, locked=True, status="inactive", end=902, end_reason="new_cycle")
    assert fib_record_drawn(rec, set()) is True
    assert fib_record_drawn(rec, {(1, 1)}) is False
    assert fib_record_drawn(rec, {(1, 2), (0, 1)}) is True     # keyed on (structure_id, cycle_id)


@pytest.mark.parametrize("rec, bright", [
    (_fib(status="active"), True),
    (_fib(status="active", locked=True), False),
    (_fib(status="ended", end=9, end_reason="parent_end"), False),
    (_fib(status="active", active=False), False),
])
def test_bright_only_for_the_live_version(rec, bright):
    assert fib_is_bright(rec) is bright


# ---------------------------------------------------------------------------
# The IP outcome lookup + the hover status cell
# ---------------------------------------------------------------------------

def test_ip_rows_keep_only_the_pick_and_the_none_row_keyed_per_sub():
    rows = [
        _ip(ic=14, rank="2A.i", picked=False, sub_id=5),                    # a losing candidate: skipped
        _ip(ic=18, rank="1A.iv", picked=True, sub_id=5),
        _ip(reason="no_pattern", sub_id=6),                                 # same (sid, cyc, bos, cts), other sub
    ]
    got = ip_rows_by_fib_record(rows, by_sub=True)
    assert got == {(5, 0, 1, 10, 20): rows[1], (6, 0, 1, 10, 20): rows[2]}
    with pytest.raises(AssertionError, match="two IP outcome rows"):
        ip_rows_by_fib_record(rows, by_sub=False)                            # without sub_id the keys collide
    with pytest.raises(KeyError):
        ip_rows_by_fib_record([_ip(reason="no_pattern")], by_sub=True)      # an unattributed row fails loudly
    assert ip_rows_by_fib_record([_ip(reason="no_pattern", cyc=2)], by_sub=False) == {
        (0, 2, 10, 20): _ip(reason="no_pattern", cyc=2)}


@pytest.mark.parametrize("rec, row, text", [
    (_fib(locked=True, status="ended", start=652, end=902, end_reason="reversal"),
     _ip(ic=607, rank="3A.iii", picked=True),
     "ended (locked)<br>life=[652, 902] reversal<br>IP 3A.iii · IC 607"),
    (_fib(locked=True, status="ended", start=1020, end=1169, end_reason="new_cycle"),
     _ip(reason="no_pattern"),
     "ended (locked)<br>life=[1020, 1169] new_cycle<br>IP: none (no_pattern)"),
    (_fib(status="ended", start=3589, end=3611, end_reason="parent_end"), None,
     "ended<br>life=[3589, 3611] parent_end<br>IP: not evaluated (no POI record)"),
    (_fib(status="active", start=4177), _ip(ic=4083, rank="1A.iv", picked=True),
     "active<br>life=[4177, open]<br>IP 1A.iv · IC 4083"),
    (_fib(locked=True, status="inactive", end=934, end_reason="new_cycle"), _ip(reason="no_pattern"),
     "inactive (locked)<br>life=[–, 934] new_cycle<br>IP: none (no_pattern)"),
])
def test_hover_status_cell(rec, row, text):
    assert fib_hover_status_text(rec, row) == text


# ---------------------------------------------------------------------------
# The H1 chart: the shared gate at its call site (a collapsed cycle is hidden)
# ---------------------------------------------------------------------------

def _render_h1(df, monkeypatch, tmp_path):
    go = pytest.importorskip("plotly.graph_objects")
    from engine_v2.charting.export_plotly import export_chart_plotly
    from engine_v2.multitf.registry import StructureRegistry
    reg = StructureRegistry()
    reg.register("H1", df=df, timeframe="H1", role="main")
    figs = []
    monkeypatch.setattr(go.Figure, "write_html", lambda self, *a, **k: figs.append(self))
    monkeypatch.setattr(go.Figure, "write_image", lambda self, *a, **k: None)
    cfg = {"structure": {"levels": True}, "struct_state": {"labels": False},
           "range_visual": {"rectangles": False}, "fib": {"lines": True},
           "imbalance": {"highlight": False}, "zones": {"wave_candles": False, "POI": True, "KL": False},
           "volume": {"bars": False, "ema_line": False, "spike_marker": False}}
    with contextlib.redirect_stdout(io.StringIO()):
        export_chart_plotly(title="t", registry=reg, path_id="H1", out_dir=tmp_path, cfg=cfg)
    fig, = figs
    return fig


def test_h1_chart_hides_a_collapsed_cycles_fib_and_hovers_the_lifecycle_and_ip(tmp_path, monkeypatch):
    from engine_v2.tests.test_plan_h_landing_b import _pipeline
    from engine_v2.tests.test_unified_probe import _make_ip_multicycle_data
    res, out, _ = _pipeline(_make_ip_multicycle_data)
    df = res.df.copy()
    df.attrs.update(structure_events=res.events, fib_states=out["fib_states"], poi_zones=out["poi_zones"],
                    ip_candidates=out["ip_candidates"], kl_zones=list(out["kl_zones"]))
    rows = ip_rows_by_fib_record(out["ip_candidates"], by_sub=False)
    recs = {(f.structure_id, f.cycle_id): f for f in out["fib_states"]}
    assert set(recs) == {(0, 1), (0, 2), (0, 3)}      # c1 / c2 locked + ended, c3 live (unlocked, open)

    fig = _render_h1(df, monkeypatch, tmp_path)
    fib_tr = {tr.name: tr for tr in fig.data if (tr.name or "").startswith("fib:")}
    assert sorted(fib_tr) == ["fib:sid0:c1", "fib:sid0:c2", "fib:sid0:c3"]
    for (sid, cyc), rec in recs.items():
        cell = fib_tr[f"fib:sid{sid}:c{cyc}"].customdata[0][8]
        assert cell == fib_hover_status_text(rec, rows.get((sid, cyc, rec.bos_idx, rec.cts_idx)))
        assert "<br>life=[" in cell
    n_shapes = len(fig.layout.shapes)

    df.attrs["kl_zones"] = list(out["kl_zones"]) + [_collapsed_kl(0, 1)]       # (0, 1) collapsed
    fig = _render_h1(df, monkeypatch, tmp_path)
    assert sorted(tr.name for tr in fig.data if (tr.name or "").startswith("fib:")) == ["fib:sid0:c2", "fib:sid0:c3"]
    assert len(fig.layout.shapes) == n_shapes - 5


# ---------------------------------------------------------------------------
# The M15 lens chart: each sub's fibs, one legend group
# ---------------------------------------------------------------------------

def _sub_rec(sub_id, start_idx):
    from engine_v2.tests.test_sub_tables_export import _sub
    return _sub(sub_id, 1, 2, start_idx, 40, "parent_end", None, ("confluence",),
                ((start_idx, "confluence"),), 1,
                {"lens": "confluence", "parent_sid": 0, "parent_cycle_id": 1,
                 "trigger_type": "first_confluence", "trigger_idx": start_idx, "start_idx": start_idx})


def _ohlc(n, freq):
    t = pd.date_range("2026-01-05", periods=n, freq=freq, tz="UTC")
    c = [0.600 + 0.0002 * (i % 7) for i in range(n)]
    return pd.DataFrame({"time": t, "o": c, "h": [x + 0.0003 for x in c], "l": [x - 0.0003 for x in c], "c": c})


# sub 5: (0,0) collapsed; (0,1) locked + picked; (0,2) ended never-locked by parent_end (no IP row) and its
# dead superseded version; sub 6: (0,1) the live version with the SAME (sid, cyc, anchors) as sub 5's, IP none.
_M15_FIBS = {
    "s5c0": _fib(0, 0, locked=True, status="inactive", end=12, end_reason="new_cycle", bos=2, cts=8, sub_id=5),
    "s5c1": _fib(0, 1, locked=True, status="ended", start=12, end=25, end_reason="new_cycle", sub_id=5),
    "s5c2": _fib(0, 2, status="ended", start=25, end=40, end_reason="parent_end", bos=22, cts=35, sub_id=5),
    "s5c2_dead": _fib(0, 2, active=False, status="ended", start=25, end=40, end_reason="parent_end",
                      bos=22, cts=30, sub_id=5),
    "s6c1": _fib(0, 1, status="active", start=30, sub_id=6),
}
_M15_IP = [
    _ip(ic=18, rank="1A.iv", picked=True, sub_id=5),
    _ip(ic=14, rank="2A.i", picked=False, sub_id=5),
    _ip(reason="no_pattern", sub_id=6),
]


def _render_m15(monkeypatch, tmp_path, cfg_extra=None, poi_zones=(), triggers=()):
    go = pytest.importorskip("plotly.graph_objects")
    from engine_v2.charting.export_m15_chart import export_m15_chart_plotly
    from engine_v2.multitf.registry import StructureRegistry
    h1, m15 = _ohlc(12, "1h"), _ohlc(48, "15min")
    m15.attrs.update(sids=[_sub_rec(5, 12), _sub_rec(6, 30)], triggers=list(triggers), events=[],
                     poi_zones=list(poi_zones),
                     wave_candles=[], wvmi=[], prev_bos_lines=[], kl_zones=[_collapsed_kl(0, 0, sub_id=5)],
                     fib_states=list(_M15_FIBS.values()), ip_candidates=list(_M15_IP))
    reg = StructureRegistry()
    reg.register("H1.main", df=h1, timeframe="H1", role="main")
    reg.register("H1.main >> M15.confluence", df=m15, timeframe="M15", role="subordinate",
                 starting_alignment="confluence")
    figs = []
    monkeypatch.setattr(go.Figure, "write_html", lambda self, *a, **k: figs.append(self))
    monkeypatch.setattr(go.Figure, "write_image", lambda self, *a, **k: None)
    cfg = {"structure": {"levels": False}, "zones": {"KL": False, "POI": False, "wave_candles": False},
           "imbalance": {"highlight": False}, "volume": {"bars": False, "ema_line": False, "spike_marker": False}}
    cfg.update(cfg_extra or {})
    with contextlib.redirect_stdout(io.StringIO()):
        export_m15_chart_plotly(title="t", registry=reg, path_id="H1.main >> M15.confluence",
                                out_dir=tmp_path, cfg=cfg)
    fig, = figs
    return fig


def test_m15_chart_draws_each_subs_fibs_in_one_legend_group(tmp_path, monkeypatch):
    fig = _render_m15(monkeypatch, tmp_path)
    fib_tr = [tr for tr in fig.data if tr.name == "M15 fib"]
    got = sorted((c[0], c[5], c[6], c[8], c[12]) for c in (tr.customdata[0] for tr in fib_tr))
    rows = ip_rows_by_fib_record(_M15_IP, by_sub=True)

    def _cell(key, sub):
        rec = _M15_FIBS[key]
        return fib_hover_status_text(rec, rows.get((sub, 0, rec.cycle_id, rec.bos_idx, rec.cts_idx)))

    assert got == [
        (5, 1, 10, 20, _cell("s5c1", 5)),
        (5, 2, 22, 35, _cell("s5c2", 5)),
        (6, 1, 10, 20, _cell("s6c1", 6)),
    ]
    assert [g[4].rsplit("<br>", 1)[1] for g in got] == [
        "IP 1A.iv · IC 18", "IP: not evaluated (no POI record)", "IP: none (no_pattern)"]
    assert all(tr.legendgroup == "m15_fib" and tr.showlegend is False for tr in fib_tr)
    assert all("sub_id=%{customdata[0]}" in tr.hovertemplate for tr in fib_tr)

    shapes = [s for s in fig.layout.shapes if s.legendgroup == "m15_fib"]
    assert len(shapes) == 5 * len(fib_tr) == len(fig.layout.shapes)
    legend = [s for s in shapes if s.showlegend]
    assert len(legend) == 1 and legend[0].name == "M15 fib" and legend[0].type == "rect"
    bright = STYLE["fib.anchor_line.active"]["line"]["color"]
    assert sum(s.line.color == bright for s in shapes if s.type == "line") == 4      # only sub 6's live version

    # The box and the hover cells (landing review P03 / P04 / M03 / M04 / M05). Every fixture fib is a buy over
    # .61 / .60: 61.8 % = .60382, 80 % = .60200; both subs sit under parent (0, 1).
    from engine_v2.tests.test_plan_h_landing_b import _rendered
    for tr in fib_tr:
        c = tr.customdata[0]
        assert (c[1], c[2], c[10], c[11]) == (0, 1, "0.60382", "0.60200")
        assert "<br>61.8%=0.60382<br>80%=0.60200<br>" in _rendered(tr) and "%%" not in tr.hovertemplate
    rects = [s for s in shapes if s.type == "rect"]
    assert {(round(s.y0, 5), round(s.y1, 5)) for s in rects} == {(0.60382, 0.602)}
    zone_bright = STYLE["fib.zone_rect.active"]["line"]["color"]
    assert [s.line.color for s in rects].count(zone_bright) == 1                     # the live version's box only


def test_m15_fib_lines_toggle_off_draws_nothing(tmp_path, monkeypatch):
    fig = _render_m15(monkeypatch, tmp_path, {"fib": {"lines": False}})
    assert not [tr for tr in fig.data if tr.name == "M15 fib"]
    assert not [s for s in fig.layout.shapes if s.legendgroup == "m15_fib"]


def test_m15_chart_skips_a_never_started_subs_record(tmp_path, monkeypatch):
    """PLAN_I R4-4: a never-started sub's zero-length record carries `sub_id` None —
    the chart groups records by `sub_id` and must skip it (drawn nowhere), not
    `int(None)`. The figure equals the one without that record."""
    from engine_v2.multitf.sub_structure_pool import TriggerRecord
    orphan = TriggerRecord(
        lens="confluence", parent_sid=0, parent_cycle_id=0, trigger_sub_sid=0,
        sub_id=None, pool_idx=3, trigger_type="subsequent_confluence", trigger_idx=20,
        probe_finalize_idx=20, probe_finalize_condition="no_retrace",
        parent_bos_anchor_idx=None, probe_input_idx=18, starting_idx=18, direction=1, sub_tf="M15",
        relative_dir="confluence", parent_floor_idx=20, start_idx=20,
        trigger_end_idx=20, end_idx=20, end_reason="parent_end", seq=0,
    )
    assert orphan.is_zero_length
    base = _render_m15(monkeypatch, tmp_path)
    got = _render_m15(monkeypatch, tmp_path, triggers=[orphan])
    assert got.to_json() == base.to_json()
