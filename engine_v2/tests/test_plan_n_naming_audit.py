"""PLAN_N (2026-10-12): the Naming-Standard audit of the never-audited index keys — the pins P1–P7 (PLAN_N §5).

- N1 — a KL `bounds_steps` entry carries ONE index key per kind: `base_idx` on INIT (the base pattern's first candle,
  a location), `expanded_idx` on an expansion (the threshold update's moment); every renderer and the M15 mirror
  read it through `kl_zones_v1.bounds_step_idx_key` / `bounds_step_draw_idx`.
- N2 / N5 — `RANGE_STARTED` carries `range_anchor_idx` (MS's own range start on EVERY path: the label candle; the CTS
  anchor on a pullback- / proximity-created range) and `confirmed_at` (== `ev.idx`); the H1 range box starts at the
  anchor, and a range's start never sorts before the previous range's reset.
- N3 — `STATE_CHANGED.meta["effective_idx"] == ev.idx` on every row, and cannot be overridden.
- N4 — "collapsed" everywhere for the one test (`ParentTables.collapsed`, reason `collapsed_parent_cycle`).

Each pin is built so its named mutant changes the outcome (a start ≠ its confirm, an anchor ≠ the moment, a base ≠
the expansion, two steps with different bounds).
"""
from __future__ import annotations

import ast
import contextlib
import io
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd
import pytest

import engine_v2
from engine_v2.charting._zone_render import build_stepped_outline_xy
from engine_v2.common.types import KLZone
from engine_v2.structure import market_structure
from engine_v2.structure.market_structure import MarketState, StructureEvent
from engine_v2.structure.structure_engine import compute_bounded_structure
from engine_v2.tests import test_ms_cts_update_no_regress as _noreg
from engine_v2.tests._cross_fib_limits import NO_P
from engine_v2.tests.test_meta_key_renames import _bos1, _kl_run
from engine_v2.tests.test_ms_stop_after_cts import _make_double_rewind_data
from engine_v2.tests.test_unified_probe import _make_uptrend_data, _prepare_df
from engine_v2.zones import kl_zones_v1
from engine_v2.zones.kl_zones_v1 import bounds_step_draw_idx, bounds_step_idx_key


def _real_run(sd):
    rows = _make_double_rewind_data()
    with redirect_stdout(io.StringIO()):
        return compute_bounded_structure(_prepare_df(rows if sd == 1 else _noreg._mirror(rows)), 0, sd,
                                         cross_fib_limit=NO_P)


# ---------------------------------------------------------------------------------------------------------------
# N3 — effective_idx (P1, P2)
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sd", [1, -1])
def test_p1_effective_idx_is_the_state_change_idx_on_every_row(sd):
    """The fixture's label range (anchor 9, confirmed 13) makes a `range_confirmed` STATE_CHANGED at 13 whose range
    start differs from its idx — the old override wrote 9 there."""
    res = _real_run(sd)
    sc = [e for e in res.events if e.type == "STATE_CHANGED"]
    assert sc and all(e.meta["effective_idx"] == int(e.idx) for e in sc)
    (rc,) = [e for e in sc if e.meta["reason"] == "range_confirmed"]
    (rs,) = [e for e in res.events if e.type == "RANGE_STARTED" and int(e.idx) == int(rc.idx)]
    assert (int(rc.idx), rs.meta["range_anchor_idx"]) == (13, 9)       # the forbidden input: start != confirm


def test_p2_effective_idx_cannot_be_overridden():
    from engine_v2.tests.test_ms_reversal_terminal import _ms
    ms = _ms()
    with pytest.raises(AssertionError, match="effective_idx"):
        ms._set_state(MarketState.BREAKOUT, 5, meta={"reason": "x", "effective_idx": 3})


# ---------------------------------------------------------------------------------------------------------------
# N2 / N5 — RANGE_STARTED (P3)
# ---------------------------------------------------------------------------------------------------------------

def _range_started_emit_keys() -> list:
    tree = ast.parse(Path(market_structure.__file__).read_text(encoding="utf-8"))
    out = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        typ = next((k.value for k in n.keywords if k.arg == "type"), None)
        if isinstance(typ, ast.Constant) and typ.value == "RANGE_STARTED":
            meta = next(k.value for k in n.keywords if k.arg == "meta")
            out.append({k.value for k in meta.keys if isinstance(k, ast.Constant)})
    return out


def test_p3_every_range_started_emit_literal_carries_the_new_keys():
    keys = _range_started_emit_keys()
    assert len(keys) == 3            # range label, pullback_created_range, proximity_created_range
    for k in keys:
        assert {"range_anchor_idx", "confirmed_at"} <= k and not {"start_idx", "confirm_idx"} & k, sorted(k)


@pytest.mark.parametrize("sd", [1, -1])
def test_p3_label_and_pullback_ranges_carry_ms_own_range_start(sd):
    """Label path: the range candle (9) < the moment (13). Pullback path: the CTS anchor (2, 12) — MS's
    `st.range_start_idx` — not the pullback apply candle (4, 14)."""
    res = _real_run(sd)
    rs = [(int(e.idx), e.meta.get("reason"), e.meta["range_anchor_idx"], e.meta["confirmed_at"], e.meta["cts_anchor_idx"])
          for e in res.events if e.type == "RANGE_STARTED"]
    assert rs == [(4, "pullback_created_range", 2, 4, 2), (13, None, 9, 13, 8), (14, "pullback_created_range", 12, 14, 12)]


@pytest.mark.parametrize("sd", [1, -1])
def test_p3_a_proximity_range_carries_the_cts_anchor(sd, monkeypatch):
    import engine_v2.structure.structure_engine as se
    from engine_v2.tests.test_ms_reversal_terminal import _p3_rows, _run
    monkeypatch.setattr(se, "poi_inner_for_fib", lambda df, fib, ft, **kw: None)
    rows = _p3_rows() if sd == 1 else _noreg._mirror(_p3_rows())
    res = _run(_prepare_df(rows), sd)
    (e,) = [e for e in res.events if e.type == "RANGE_STARTED" and e.meta.get("reason") == "proximity_created_range"]
    assert int(e.idx) == 12 and e.meta["confirmed_at"] == 12
    assert e.meta["range_anchor_idx"] == e.meta["cts_anchor_idx"] != 12


# ---------------------------------------------------------------------------------------------------------------
# N2 / N5 — the H1 range boxes and hover lines (P4) — through the chart's real call sites
# ---------------------------------------------------------------------------------------------------------------

def _render_h1(df, monkeypatch, tmp_path, *, ranges=False, kl=False, idx_range=None):
    go = pytest.importorskip("plotly.graph_objects")
    from engine_v2.charting.export_plotly import export_chart_plotly
    from engine_v2.multitf.registry import StructureRegistry
    reg = StructureRegistry()
    reg.register("H1", df=df, timeframe="H1", role="main")
    figs = []
    monkeypatch.setattr(go.Figure, "write_html", lambda self, *a, **k: figs.append(self))
    monkeypatch.setattr(go.Figure, "write_image", lambda self, *a, **k: None)
    cfg = {"structure": {"levels": False}, "struct_state": {"labels": False}, "range_visual": {"rectangles": ranges},
           "fib": {"lines": False}, "imbalance": {"highlight": False},
           "zones": {"wave_candles": False, "POI": False, "KL": kl},
           "volume": {"bars": False, "ema_line": False, "spike_marker": False}}
    with contextlib.redirect_stdout(io.StringIO()):
        export_chart_plotly(title="t", registry=reg, path_id="H1", out_dir=tmp_path, cfg=cfg, idx_range=idx_range)
    fig, = figs
    return fig


def _ts(x):
    """A naive-UTC Timestamp (Plotly hands figure x values back without their tz)."""
    x = pd.Timestamp(x)
    return x.tz_convert("UTC").tz_localize(None) if x.tzinfo is not None else x


def _rng(i, typ, sid, **m):
    return StructureEvent(idx=i, category="RANGE", type=typ, meta={"structure_id": sid, "struct_direction": 1, **m})


def _range_events():
    """Three independent structures (the builders keep one open range per structure):
    sid 0 — a label range anchored at 5, confirmed at 7, its back-filled RANGE_UPDATEDs at 6 and 7 emitted AFTER it
            (the S_M2 H1 shape: STARTED @117 anchor 115, UPDATED @116);
    sid 1 — a pullback-created range at 15 anchored at the CTS anchor 13;
    sid 2 — a range 19–24 (reset at 24), then a pullback-created range at 25 anchored at 22 < 24 (the anchor-lag
            case the reset guard exists for)."""
    return [
        _rng(7, "RANGE_STARTED", 0, range_anchor_idx=5, confirmed_at=7, hi=1.0, lo=0.9),
        _rng(6, "RANGE_UPDATED", 0, hi=1.0, lo=0.85),
        _rng(7, "RANGE_UPDATED", 0, hi=1.02, lo=0.85),
        _rng(15, "RANGE_STARTED", 1, range_anchor_idx=13, confirmed_at=15, reason="pullback_created_range",
             hi=2.0, lo=1.9),
        _rng(21, "RANGE_STARTED", 2, range_anchor_idx=19, confirmed_at=21, hi=3.0, lo=2.9),
        _rng(24, "RANGE_RESET", 2),
        _rng(25, "RANGE_STARTED", 2, range_anchor_idx=22, confirmed_at=25, reason="pullback_created_range",
             hi=3.2, lo=3.1),
    ]


def test_p4_the_range_boxes_start_at_the_anchor_in_guarded_order(tmp_path, monkeypatch):
    df = _prepare_df(_make_uptrend_data(32))
    df.attrs["structure_events"] = _range_events()
    fig = _render_h1(df, monkeypatch, tmp_path, ranges=True)
    t = [_ts(x) for x in df["time"]]
    end = t[-1]
    boxes = sorted((s.y0, s.y1, _ts(s.x0), _ts(s.x1)) for s in fig.layout.shapes if s.type == "rect")
    assert boxes == sorted([
        # (a) the update at 6 splits the label range's box (a sort by ev.idx would drop it: one box 5–7)
        (0.9, 1.0, t[5], t[6]), (0.85, 1.0, t[6], t[7]), (0.85, 1.02, t[7], end),
        # (b) the pullback range's box starts at its anchor 13, not at its idx 15
        (1.9, 2.0, t[13], end),
        # (c) the reset at 24 closes the OLD range; the new one opens after it, drawn from its anchor 22
        (2.9, 3.0, t[19], t[24]), (3.1, 3.2, t[22], end),
    ])
    # The hover lines (the second builder): each point's `range_start_idx=` (customdata[1]) is the open range's anchor.
    starts = {tr.name: sorted({tuple(c)[1] for c in tr.customdata}) for tr in fig.data
              if (tr.name or "").startswith("range:top:")}
    assert starts == {"range:top:sid0": [5], "range:top:sid1": [13], "range:top:sid2": [19, 22]}
    # ... and replays in the SAME order as the boxes, snapshotting at the replay candle (the landing review's surviving
    # LR1: the hover builder back on a plain idx sort): sid 0's line starts at its anchor 5, where its box starts (the
    # PLAN_N hover follow-up, 2026-10-12; before it, at its first back-filled update 6).
    first_x = {tr.name: _ts(min(tr.x)) for tr in fig.data if (tr.name or "").startswith("range:top:")}
    assert first_x["range:top:sid0"] == t[5]


def test_p4_each_range_hover_line_runs_where_its_box_runs_across_structures(tmp_path, monkeypatch):
    """(The PLAN_N hover follow-up — the landing review's finding 1, its probe.) sid 0: a label range anchored at 1,
    confirmed at 3, reset at 12; sid 1: a pullback range at 14 anchored at 8 — replayed at 8, BEFORE sid 0's reset.
    Snapshotted at each event's idx, the snapshot at 14 froze sid 0 as still open (its line came back from 14 on) and
    sid 1's line began at 12; snapshotted at the replay candle, each line runs where its box runs: sid 0 from its anchor
    1 to 11 (its box closes at the reset 12), sid 1 from its anchor 8."""
    df = _prepare_df(_make_uptrend_data(32))
    df.attrs["structure_events"] = [
        _rng(3, "RANGE_STARTED", 0, range_anchor_idx=1, confirmed_at=3, hi=1.0, lo=0.9),
        _rng(12, "RANGE_RESET", 0),
        _rng(14, "RANGE_STARTED", 1, range_anchor_idx=8, confirmed_at=14, reason="pullback_created_range",
             hi=2.0, lo=1.9),
    ]
    fig = _render_h1(df, monkeypatch, tmp_path, ranges=True)
    t = [_ts(x) for x in df["time"]]
    span = {tr.name: (_ts(min(tr.x)), _ts(max(tr.x)), len(tr.x)) for tr in fig.data
            if (tr.name or "").startswith("range:top:")}
    assert span == {"range:top:sid0": (t[1], t[11], 11), "range:top:sid1": (t[8], t[-1], len(t) - 8)}


def test_p4_a_label_ranges_hover_keeps_its_back_filled_update_on_the_confirm_candle(tmp_path, monkeypatch):
    """(The follow-up's second symptom.) A label range anchored at 5, confirmed at 7, its back-filled update at 6
    (emitted after it, hi 1.0 -> 1.05), none at 7. Snapshotted at its idx 7, the STARTED froze the anchor-time state
    (hi 1.0, last update 5) on 7 and every candle forward-filled from it (on S_N: sid 1, H1 716-723, under a box drawn
    at the update's top); at its replay candle 5 the line starts with the box and keeps the update from 6 on."""
    df = _prepare_df(_make_uptrend_data(12))
    df.attrs["structure_events"] = [
        _rng(7, "RANGE_STARTED", 0, range_anchor_idx=5, confirmed_at=7, hi=1.0, lo=0.9),
        _rng(6, "RANGE_UPDATED", 0, hi=1.05, lo=0.9),
    ]
    fig = _render_h1(df, monkeypatch, tmp_path, ranges=True)
    top, = [tr for tr in fig.data if tr.name == "range:top:sid0"]
    # customdata = (idx, range_start_idx, range_hi, range_lo, last_update_idx, structure_id)
    assert [tuple(c)[:5] for c in top.customdata] == [(5, 5, 1.0, 0.9, 5)] + [(i, 5, 1.05, 0.9, 6) for i in range(6, 12)]
    assert list(top.y) == [1.0] + [1.05] * 6


def test_p4_a_sliced_charts_hover_line_starts_at_its_first_candle_like_the_clamped_box(tmp_path, monkeypatch):
    """A range open before an `idx_range` slice (8-15) shows from the slice's first candle, like its box (clamped
    there); the pass used to snapshot visible candles only, so the line began at the first event inside the slice."""
    df = _prepare_df(_make_uptrend_data(20))
    df.attrs["structure_events"] = [
        _rng(4, "RANGE_STARTED", 0, range_anchor_idx=2, confirmed_at=4, hi=1.0, lo=0.9),
        _rng(12, "RANGE_UPDATED", 0, hi=1.1, lo=0.9),
    ]
    fig = _render_h1(df, monkeypatch, tmp_path, ranges=True, idx_range=(8, 15))
    t = [_ts(x) for x in df["time"]]
    boxes = sorted((s.y0, s.y1, _ts(s.x0), _ts(s.x1)) for s in fig.layout.shapes if s.type == "rect")
    assert boxes == [(0.9, 1.0, t[8], t[12]), (0.9, 1.1, t[12], t[15])]
    top, = [tr for tr in fig.data if tr.name == "range:top:sid0"]
    assert [_ts(x) for x in top.x] == t[8:16]
    assert list(top.y) == [1.0] * 4 + [1.1] * 4


def test_p4_the_reset_guard_is_per_structure_and_only_for_an_earlier_reset():
    """`_range_events_chart_order` directly (its two conditions; the landing review's surviving LR2 / LR3):
    a reset of ANOTHER structure (sid 5 at 32, emitted first) never delays a label range (sid 4, anchor 30, confirmed
    32) — its back-filled update at 31 must still follow it; a same-structure reset stamped AFTER a STARTED's moment
    (a reset at 12 before a STARTED at 10, anchor 8 — outside the domain) is ignored."""
    from engine_v2.charting.export_plotly import _range_events_chart_order
    evs = [
        _rng(28, "RANGE_STARTED", 5, range_anchor_idx=27, confirmed_at=28, hi=1.0, lo=0.9),
        _rng(32, "RANGE_RESET", 5),
        _rng(32, "RANGE_STARTED", 4, range_anchor_idx=30, confirmed_at=32, hi=2.0, lo=1.9),
        _rng(31, "RANGE_UPDATED", 4, hi=2.0, lo=1.8),
        _rng(32, "RANGE_UPDATED", 4, hi=2.1, lo=1.8),
    ]
    sid4 = [(e.type, int(e.idx)) for e in _range_events_chart_order(evs) if e.meta["structure_id"] == 4]
    assert sid4 == [("RANGE_STARTED", 32), ("RANGE_UPDATED", 31), ("RANGE_UPDATED", 32)]
    late = [_rng(12, "RANGE_RESET", 6), _rng(10, "RANGE_STARTED", 6, range_anchor_idx=8, confirmed_at=10, hi=1, lo=0)]
    assert [e.type for e in _range_events_chart_order(late)] == ["RANGE_STARTED", "RANGE_RESET"]


# ---------------------------------------------------------------------------------------------------------------
# N1 — KL steps (P5 real run, P6 every renderer call site)
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sd", [1, -1])
def test_p5_kl_steps_carry_one_key_per_kind(sd):
    res, zones = _kl_run(sd)
    (thr,) = [e for e in res.events if e.type == "BOS_THRESHOLD_UPDATED"]
    z = _bos1(zones)
    init, exp = z.meta["bounds_steps"]
    assert set(init) == {"base_idx", "top", "bottom", "event"} and init["event"] == "INIT"
    assert set(exp) == {"expanded_idx", "top", "bottom", "event", "price"}
    assert init["base_idx"] == z.meta["base_idx"] == 4
    assert exp["expanded_idx"] == int(thr.idx) == 9
    # the forbidden inputs: the base precedes the zone's moment and its anchor is another candle than the expansion
    assert z.meta["confirmed_idx"] != 4 and z.meta["anchor_idx"] != 9
    assert not [s for zz in zones for s in zz.meta["bounds_steps"] if "start_idx" in s]
    assert [bounds_step_idx_key(s) for s in z.meta["bounds_steps"]] == ["base_idx", "expanded_idx"]


_INIT = {"base_idx": None, "top": 0.70, "bottom": 0.60, "event": "INIT"}
_EXP = {"expanded_idx": None, "top": 0.70, "bottom": 0.55, "event": "BOS_THRESHOLD_UPDATED", "price": 0.55}


def _two_step_zone(times, base, exp, conf, *, end=None, **meta):
    """A buy zone: INIT [0.60, 0.70] from `base`, expanded DOWN to [0.55, 0.70] at `exp`; confirmed at `conf`
    inside the expansion step — so the confirm line's y-range tells the two steps apart."""
    steps = [{**_INIT, "base_idx": base}, {**_EXP, "expanded_idx": exp}]
    return KLZone(start_time=times[base], end_time=None if end is None else times[end], side="buy", top=0.70,
                  bottom=0.55, source_kind="BOS", source_time=times[conf], source_price=0.55,
                  meta={"structure_id": 0, "struct_direction": 1, "cycle_id": 1, "confirmed_idx": conf,
                        "anchor_idx": base, "base_idx": base, "base_pattern": "base", "bounds_steps": steps,
                        "expanded": True, "end_idx": end, "end_reason": None,
                        "activation_history": [{"idx": conf, "active": True, "reason": "confirmed"}],
                        "status": "active", **meta})


def _step_traces(fig, name):
    """(first x, last x, y) of each hover trace `name`, in draw order (per step: top, bottom)."""
    return [(_ts(min(tr.x)), _ts(max(tr.x)), tr.y[0]) for tr in fig.data if tr.name == name]


def _confirm_lines(fig):
    return [(_ts(s.x0), s.y0, s.y1) for s in fig.layout.shapes if s.type == "line" and s.x0 == s.x1]


def test_p6_the_stepped_outline_reads_each_steps_own_key():
    t = {i: pd.Timestamp("2026-01-05", tz="UTC") + pd.Timedelta(hours=i) for i in range(30)}
    steps = [{**_INIT, "base_idx": 5}, {**_EXP, "expanded_idx": 12}]
    xs, ys = build_stepped_outline_xy(steps, lambda i: t.get(i), t[5], t[29], fallback_top=0.70, fallback_bottom=0.55)
    assert list(zip(xs, ys)) == [
        (t[5], 0.70), (t[12], 0.70), (t[29], 0.70),          # top edge (the same top: no jump)
        (t[29], 0.55), (t[12], 0.55), (t[12], 0.60),         # right edge, bottom R->L with the jump at 12
        (t[5], 0.60), (t[5], 0.70)]                          # left edge, closed


def test_p6_the_h1_kl_renderer_reads_each_steps_own_key(tmp_path, monkeypatch):
    df = _prepare_df(_make_uptrend_data(32))
    tz = list(pd.to_datetime(df["time"], utc=True))
    t = [_ts(x) for x in tz]
    df.attrs["structure_events"] = []
    df.attrs["kl_zones"] = [_two_step_zone(tz, 5, 12, 15)]
    fig = _render_h1(df, monkeypatch, tmp_path, kl=True)
    assert _step_traces(fig, "KL zone") == [
        (t[5], t[11], 0.70), (t[5], t[11], 0.60),               # INIT ends before the expansion candle
        (t[12], t[-1], 0.70), (t[12], t[-1], 0.55)]
    assert _confirm_lines(fig) == [(t[15], 0.55, 0.70)]          # inside the EXPANSION step
    (outline,) = [tr for tr in fig.data if (tr.name or "").startswith("KL outline")]
    assert _ts(outline.x[0]) == t[5] and t[12] in [_ts(x) for x in outline.x]


def _render_m15(monkeypatch, tmp_path, *, h1_kl=(), m15_kl=()):
    go = pytest.importorskip("plotly.graph_objects")
    from engine_v2.charting.export_m15_chart import export_m15_chart_plotly
    from engine_v2.multitf.registry import StructureRegistry
    from engine_v2.tests.test_fib_chart_render import _ohlc, _sub_rec
    h1, m15 = _ohlc(12, "1h"), _ohlc(48, "15min")
    h1.attrs.update(kl_zones=list(h1_kl), structure_events=[])
    m15.attrs.update(sids=[_sub_rec(5, 12)], triggers=[], events=[], poi_zones=[], wave_candles=[], wvmi=[],
                     prev_bos_lines=[], kl_zones=list(m15_kl), fib_states=[], ic_candidates=[])
    reg = StructureRegistry()
    reg.register("H1.main", df=h1, timeframe="H1", role="main")
    reg.register("H1.main >> M15.confluence", df=m15, timeframe="M15", role="subordinate",
                 starting_alignment="confluence")
    figs = []
    monkeypatch.setattr(go.Figure, "write_html", lambda self, *a, **k: figs.append(self))
    monkeypatch.setattr(go.Figure, "write_image", lambda self, *a, **k: None)
    cfg = {"structure": {"levels": False}, "zones": {"KL": True, "POI": False, "wave_candles": False},
           "imbalance": {"highlight": False}, "fib": {"lines": False},
           "volume": {"bars": False, "ema_line": False, "spike_marker": False}}
    with contextlib.redirect_stdout(io.StringIO()):
        export_m15_chart_plotly(title="t", registry=reg, path_id="H1.main >> M15.confluence", out_dir=tmp_path, cfg=cfg)
    fig, = figs
    return fig, h1, m15


def test_p6_the_m15_sub_kl_renderer_reads_each_steps_own_key(tmp_path, monkeypatch):
    from engine_v2.tests.test_fib_chart_render import _ohlc
    tz = list(pd.to_datetime(_ohlc(48, "15min")["time"], utc=True))
    t = [_ts(x) for x in tz]
    fig, _, _ = _render_m15(monkeypatch, tmp_path, m15_kl=[_two_step_zone(tz, 14, 20, 24, end=40, sub_id=5)])
    tr = _step_traces(fig, "M15 KL zone")
    assert tr == [(t[14], t[19], 0.70), (t[14], t[19], 0.60), (t[20], t[40], 0.70), (t[20], t[40], 0.55)]
    assert (t[24], 0.55, 0.70) in _confirm_lines(fig)


def test_p6_the_m15_h1_overlay_kl_renderer_reads_each_steps_own_key(tmp_path, monkeypatch):
    from engine_v2.tests.test_fib_chart_render import _ohlc
    h1tz = list(pd.to_datetime(_ohlc(12, "1h")["time"], utc=True))
    h1t = [_ts(x) for x in h1tz]
    fig, _, m15 = _render_m15(monkeypatch, tmp_path, h1_kl=[_two_step_zone(h1tz, 2, 5, 7)])
    m15t = [_ts(x) for x in m15["time"]]
    q = pd.Timedelta(minutes=45)                                  # an H1 idx maps to its hour's 4th M15 candle
    tr = _step_traces(fig, "H1 KL zone (overlay)")
    assert [(a, b, y) for a, b, y in tr] == [
        (h1t[2] + q, h1t[5] + q - pd.Timedelta(minutes=15), 0.70), (h1t[2] + q, h1t[5] + q - pd.Timedelta(minutes=15), 0.60),
        (h1t[5] + q, m15t[-1], 0.70), (h1t[5] + q, m15t[-1], 0.55)]
    assert (h1t[7] + q, 0.55, 0.70) in _confirm_lines(fig)


# The mirror's shift of both step kinds is pinned by `test_event_meta_idx_keys` (`_synthetic_result` carries an INIT
# and an expansion step; `_check_mirrored` requires both shifted) and its real-render value pin.


# ---------------------------------------------------------------------------------------------------------------
# N4 + the retired keys (P7, static)
# ---------------------------------------------------------------------------------------------------------------

def _production_trees():
    root = Path(engine_v2.__file__).parent
    for path in root.rglob("*.py"):
        if {"legacy_2025", "tests", "plans"} & set(path.parts):
            continue
        yield path, ast.parse(path.read_text(encoding="utf-8"))


def test_p7_no_production_code_says_degenerate_for_the_collapsed_test():
    hits = []
    for path, tree in _production_trees():
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and "degenerate_parent_cycle" in n.value:
                hits.append((path.name, n.lineno, n.value[:40]))
            if isinstance(n, ast.Attribute) and n.attr in ("degenerate", "is_degenerate"):
                hits.append((path.name, n.lineno, n.attr))
            if isinstance(n, ast.keyword) and n.arg == "degenerate":
                hits.append((path.name, getattr(n, "lineno", 0), "degenerate="))
    assert hits == []


def test_p7_no_kl_step_literal_carries_start_idx():
    tree = ast.parse(Path(kl_zones_v1.__file__).read_text(encoding="utf-8"))
    keys = {k.value for n in ast.walk(tree) if isinstance(n, ast.Dict) for k in n.keys if isinstance(k, ast.Constant)}
    assert "start_idx" not in keys and {"base_idx", "expanded_idx"} <= keys
