"""M15 chart with H1 overlay — separate chart file alongside the H1 chart.

Renders the full M15 dataset as the base candle layer, with all M15 structures
(from all UC1 triggers) and all H1 structures/zones overlaid.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from engine_v2.common.types import COL_C, COL_H, COL_L, COL_O, COL_TIME, COL_V, PatternStatus
from engine_v2.charting.style_registry import STYLE
from engine_v2.charting.export_plotly import (
    _rgba_from_rgb,
    _zone_style,
    _style,
    _opacity_tier,
    _deep_merge,
    _get_reversal_confirmed_by_sid,
    ChartExportPaths,
)


# ---------------------------------------------------------------------------
# M15 chart config defaults
# ---------------------------------------------------------------------------
M15_CHART_DEFAULTS = {
    "show_ohlc": True,
    "candle_types": {},
    "patterns": {
        "continuous": True,
        "double_maru": True,
        "one_maru_continuous": True,
        "one_maru_opposite": True,
    },
    "struct_state": {"labels": False},
    "range_visual": {"rectangles": False},
    "structure": {"levels": True},
    "zones": {"KL": True, "POI": True, "wave_candles": True, "num_structures": 99},
    "fib": {"lines": False},
    "imbalance": {"highlight": True},
    "volume": {"bars": True, "ema_line": True, "spike_marker": True},
    "range_candle_marker": False,
}


# ---------------------------------------------------------------------------
# H1→M15 time mapping helpers
# ---------------------------------------------------------------------------

def _build_h1_to_m15_map(
    h1_times: pd.Series,
    m15_times: pd.Series,
) -> dict:
    """Map each H1 candle time to the 4th (last) M15 candle in that H1 hour.

    Returns {h1_time → m15_time}.
    Fallback: h1_time + 45 min if no M15 candle in that hour.
    """
    m15_by_hour: dict = {}
    for t in m15_times:
        hour_key = t.floor("h")
        m15_by_hour.setdefault(hour_key, []).append(t)

    mapping = {}
    for h1t in h1_times:
        h1_hour = h1t.floor("h") if hasattr(h1t, "floor") else pd.Timestamp(h1t, tz="UTC").floor("h")
        candidates = m15_by_hour.get(h1_hour, [])
        if candidates:
            mapping[h1t] = max(candidates)  # Last M15 in that hour
        else:
            mapping[h1t] = h1t + timedelta(minutes=45)
    return mapping


def _build_m15_to_h1_map(
    h1_df: pd.DataFrame,
    m15_times: pd.Series,
) -> dict:
    """Map each M15 candle time to its parent H1 candle's (time, idx).

    Returns {m15_time → (h1_time, h1_idx)}.
    Uses m15_time.floor('h') to find the parent H1 hour.
    """
    h1_times = pd.to_datetime(h1_df[COL_TIME], utc=True)
    h1_by_hour: dict = {}
    for idx, t in zip(h1_df.index, h1_times):
        hour_key = t.floor("h")
        h1_by_hour[hour_key] = (t, int(idx))

    mapping = {}
    for m15t in m15_times:
        m15_hour = m15t.floor("h") if hasattr(m15t, "floor") else pd.Timestamp(m15t, tz="UTC").floor("h")
        if m15_hour in h1_by_hour:
            mapping[m15t] = h1_by_hour[m15_hour]
        else:
            mapping[m15t] = (None, None)
    return mapping


# ---------------------------------------------------------------------------
# M15 opacity tier helpers
# ---------------------------------------------------------------------------

def _m15_opacity_tier_for_zone(
    zone,
    most_recent_parent_sid: int,
    recent_cycle_ids: set,
) -> float:
    """Compute 3-tier opacity for an M15 zone based on parent H1 identifiers."""
    active = bool(zone.meta.get("active", False)) and (zone.end_time is None)
    if active:
        return _opacity_tier("active")
    parent_sid = zone.meta.get("parent_sid")
    parent_cycle = zone.meta.get("parent_cycle_id")
    if parent_sid == most_recent_parent_sid and parent_cycle in recent_cycle_ids:
        return _opacity_tier("recent_inactive")
    return _opacity_tier("prior_inactive")


def _m15_opacity_tier_for_events(
    parent_sid: int,
    parent_cycle_id: int,
    most_recent_parent_sid: int,
    recent_cycle_ids: set,
    is_active_sid: bool,
) -> float:
    """Compute 3-tier opacity for M15 structure elements."""
    if is_active_sid:
        return _opacity_tier("active")
    if parent_sid == most_recent_parent_sid and parent_cycle_id in recent_cycle_ids:
        return _opacity_tier("recent_inactive")
    return _opacity_tier("prior_inactive")


def _compute_m15_tier_context(lower_tf_results: list) -> tuple:
    """Compute most_recent_parent_sid and recent_cycle_ids across all triggers.

    Returns (most_recent_parent_sid, recent_cycle_ids).
    """
    if not lower_tf_results:
        return (0, set())
    all_parent_sids = [lt.trigger.parent_sid for lt in lower_tf_results]
    most_recent = max(all_parent_sids)
    cycles_for_recent = sorted(
        [lt.trigger.parent_cycle_id for lt in lower_tf_results
         if lt.trigger.parent_sid == most_recent],
        reverse=True,
    )
    recent_cycle_ids = set(cycles_for_recent[:2])
    return (most_recent, recent_cycle_ids)


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export_m15_chart_plotly(
    m15_df: pd.DataFrame,
    h1_df: pd.DataFrame,
    lower_tf_results: list,
    *,
    title: str,
    out_dir: str | Path = "artifacts/charts",
    basename: str = "chart_m15",
    max_points: Optional[int] = None,
    idx_range: Optional[tuple[int, int]] = None,
    cfg: Optional[dict] = None,
) -> ChartExportPaths:
    """Export an interactive M15 chart with H1 overlay elements."""

    cfg = _deep_merge(M15_CHART_DEFAULTS, cfg or {})
    pat_cfg = cfg.get("patterns", {}) or {}
    struct_cfg = cfg.get("structure", {}) or {}
    zone_cfg = cfg.get("zones", {}) or {}
    imbalance_cfg = cfg.get("imbalance", {}) or {}
    volume_cfg = cfg.get("volume", {}) or {}
    state_cfg = cfg.get("struct_state", {}) or {}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfx = m15_df.copy()

    # Optional index-range slicing
    if idx_range is not None:
        i0, i1 = idx_range
        if i0 > i1:
            i0, i1 = i1, i0
        i0 = max(int(i0), int(dfx.index.min()))
        i1 = min(int(i1), int(dfx.index.max()))
        dfx = dfx.loc[i0:i1].copy()

    if max_points is not None and len(dfx) > max_points:
        dfx = dfx.iloc[-max_points:].copy()

    dfx[COL_TIME] = pd.to_datetime(dfx[COL_TIME], utc=True)

    # Build time mapping helpers
    m15_times = dfx[COL_TIME]
    h1_times = pd.to_datetime(h1_df[COL_TIME], utc=True)
    h1_to_m15 = _build_h1_to_m15_map(h1_times, m15_times)
    m15_to_h1 = _build_m15_to_h1_map(h1_df, m15_times)

    # M15 opacity tier context
    m15_most_recent_psid, m15_recent_cycles = _compute_m15_tier_context(lower_tf_results)

    volume_enabled = volume_cfg.get("bars", False) and COL_V in dfx.columns
    fig = go.Figure()

    wick_offset = (dfx[COL_H] - dfx[COL_L]) * 0.15

    # ---------------------------------------------------------------
    # Helper columns
    # ---------------------------------------------------------------
    candle_idx = dfx.index.to_numpy()

    range_break_frac = (
        dfx["range_break_frac"].astype(float)
        if "range_break_frac" in dfx.columns
        else pd.Series([float("nan")] * len(dfx), index=dfx.index)
    )

    def _col_or_default(name, default):
        return dfx[name] if name in dfx.columns else pd.Series([default] * len(dfx), index=dfx.index)

    is_big_normal_as0 = _col_or_default("is_big_normal_as0", False)
    is_big_maru_as0 = _col_or_default("is_big_maru_as0", False)
    big_ratio_as0 = _col_or_default("big_ratio_as0", 0.0).astype(float)
    vol_spike_ratio = _col_or_default("vol_spike_ratio", float("nan")).astype(float)

    # Build idx_1H and time_1H columns for hover
    idx_1h_col = []
    time_1h_col = []
    for t in dfx[COL_TIME]:
        h1_info = m15_to_h1.get(t, (None, None))
        idx_1h_col.append(h1_info[1] if h1_info[1] is not None else "")
        time_1h_col.append(str(h1_info[0]) if h1_info[0] is not None else "")

    customdata = list(zip(
        candle_idx,                       # 0
        dfx["mid_price"].astype(float) if "mid_price" in dfx.columns
        else ((dfx[COL_H] + dfx[COL_L]) / 2).astype(float),  # 1
        dfx["body_pct"].astype(float) if "body_pct" in dfx.columns
        else pd.Series([0.0] * len(dfx), index=dfx.index),    # 2
        dfx["candle_type"].astype(str) if "candle_type" in dfx.columns
        else pd.Series([""] * len(dfx), index=dfx.index),     # 3
        dfx["body_len"].astype(float) if "body_len" in dfx.columns
        else pd.Series([0.0] * len(dfx), index=dfx.index),    # 4
        dfx["candle_len"].astype(float) if "candle_len" in dfx.columns
        else pd.Series([0.0] * len(dfx), index=dfx.index),    # 5
        range_break_frac,                 # 6
        is_big_normal_as0,                # 7
        is_big_maru_as0,                  # 8
        big_ratio_as0,                    # 9
        vol_spike_ratio,                  # 10
        dfx[COL_TIME].astype(str),        # 11
        idx_1h_col,                       # 12
        time_1h_col,                      # 13
    ))

    candle_hover = (
        "TF=15M<br>"
        "idx=%{customdata[0]}<br>"
        "time=%{customdata[11]}<br>"
        "idx_1H=%{customdata[12]}<br>"
        "time_1H=%{customdata[13]}<br>"
        "O=%{open}<br>"
        "H=%{high}<br>"
        "L=%{low}<br>"
        "C=%{close}<br>"
        "candle_type=%{customdata[3]}<br>"
        "body_len=%{customdata[4]:.5f}<br>"
        "candle_len=%{customdata[5]:.5f}<br>"
        "body_pct=%{customdata[2]:.2%}<br>"
        "mid_price=%{customdata[1]:.5f}<br>"
        "big_normal=%{customdata[7]}  big_maru=%{customdata[8]}  big_ratio=%{customdata[9]:.2f}<br>"
        "range_break_frac=%{customdata[6]:.2%}<br>"
        "vol_spike_ratio=%{customdata[10]:.2f}"
        "<extra></extra>"
    )

    # ===================================================================
    # Phase A: M15 base layer
    # ===================================================================

    # --- Candlesticks (with imbalance highlighting) ---
    imbalance_highlight = imbalance_cfg.get("highlight", False) and "is_imbalance" in dfx.columns

    if imbalance_highlight:
        is_imb = dfx["is_imbalance"] == 1
        direction = dfx["direction"]

        regular_mask = ~is_imb
        if regular_mask.any():
            regular_df = dfx[regular_mask]
            regular_cd = [customdata[i] for i in range(len(dfx)) if regular_mask.iloc[i]]
            fig.add_trace(go.Candlestick(
                x=regular_df[COL_TIME], open=regular_df[COL_O],
                high=regular_df[COL_H], low=regular_df[COL_L], close=regular_df[COL_C],
                name="OHLC", showlegend=False,
                customdata=regular_cd, hovertemplate=candle_hover,
            ))

        for imb_dir, imb_label, style_key in [
            (1, "BULLISH IMBALANCE", "imbalance.bullish"),
            (-1, "BEARISH IMBALANCE", "imbalance.bearish"),
        ]:
            mask = is_imb & (direction == imb_dir)
            if mask.any():
                sub_df = dfx[mask]
                sub_cd = [customdata[i] for i in range(len(dfx)) if mask.iloc[i]]
                color = _style(style_key).get("rgba", "rgba(128,128,128,0.8)")
                fig.add_trace(go.Candlestick(
                    x=sub_df[COL_TIME], open=sub_df[COL_O],
                    high=sub_df[COL_H], low=sub_df[COL_L], close=sub_df[COL_C],
                    name=f"imbalance:{imb_label.lower().split()[0]}",
                    showlegend=False,
                    increasing=dict(line=dict(color=color), fillcolor=color),
                    decreasing=dict(line=dict(color=color), fillcolor=color),
                    customdata=sub_cd,
                    hovertemplate=f"<b>{imb_label}</b><br>" + candle_hover,
                ))
    else:
        fig.add_trace(go.Candlestick(
            x=dfx[COL_TIME], open=dfx[COL_O],
            high=dfx[COL_H], low=dfx[COL_L], close=dfx[COL_C],
            name="OHLC", showlegend=False,
            customdata=customdata, hovertemplate=candle_hover,
        ))

    # --- Structure pattern markers (triangles) ---
    if "pat" in dfx.columns and "pat_dir" in dfx.columns and "pat_status" in dfx.columns:
        enabled_names = {k for k, v in pat_cfg.items() if v is True}
        sub = dfx[
            (dfx["pat"] != "")
            & (dfx["pat"].isin(enabled_names))
            & (dfx["pat_status"].isin([PatternStatus.SUCCESS.value, PatternStatus.CONFIRMED.value]))
        ]
        if not sub.empty:
            for status in [PatternStatus.SUCCESS.value, PatternStatus.CONFIRMED.value]:
                for d, pos_col, off_sign, style_sfx in [
                    (1, COL_H, 1, "up"),
                    (-1, COL_L, -1, "down"),
                ]:
                    pts = sub[(sub["pat_status"] == status) & (sub["pat_dir"] == d)]
                    if pts.empty:
                        continue
                    skey = f"structure.{'success' if status == PatternStatus.SUCCESS.value else 'confirmed'}.{style_sfx}"
                    fig.add_trace(go.Scatter(
                        x=pts[COL_TIME],
                        y=pts[pos_col] + off_sign * wick_offset.loc[pts.index],
                        mode="markers", name=f"struct:{status}:{d:+d}",
                        customdata=pts[["pat", "pat_dir", "pat_status",
                                        "pat_start_idx", "pat_end_idx", "pat_confirm_idx"]].values,
                        hovertemplate=(
                            "TF=15M<br>"
                            "pat=%{customdata[0]}<br>"
                            "dir=%{customdata[1]}<br>"
                            "status=%{customdata[2]}<br>"
                            "start=%{customdata[3]} end=%{customdata[4]} conf=%{customdata[5]}"
                            "<extra></extra>"
                        ),
                        **_style(skey),
                    ))

    # --- Volume spike markers ---
    if volume_cfg.get("spike_marker", False) and "is_vol_spike" in dfx.columns:
        spike_candles = dfx[dfx["is_vol_spike"] == True]
        if len(spike_candles) > 0:
            spike_y = (
                spike_candles["mid_price"] if "mid_price" in spike_candles.columns
                else (spike_candles[COL_H] + spike_candles[COL_L]) / 2.0
            )
            fig.add_trace(go.Scatter(
                x=spike_candles[COL_TIME], y=spike_y,
                mode="markers", name="volume:spike",
                marker=_style("volume.spike_marker").get("marker", {}),
                hoverinfo="skip",
            ))

    # --- Volume bars + EMA ---
    if volume_enabled:
        volume = dfx[COL_V].astype(float)
        vol_dir = _col_or_default("vol_dir", 0).astype(int)
        vol_ema20 = _col_or_default("vol_ema20", float("nan")).astype(float)

        colors = []
        for v_dir in vol_dir:
            if v_dir == 1:
                colors.append(_style("volume.bar.up").get("color", "rgba(0,180,0,0.7)"))
            elif v_dir == -1:
                colors.append(_style("volume.bar.down").get("color", "rgba(220,0,0,0.7)"))
            else:
                colors.append(_style("volume.bar.neutral").get("color", "rgba(128,128,128,0.7)"))

        fig.add_trace(go.Bar(
            x=dfx[COL_TIME], y=volume, name="Volume",
            marker_color=colors, showlegend=False, yaxis="y2",
            hovertemplate="TF=15M<br>Volume: %{y:,.0f}<extra></extra>",
        ))

        if volume_cfg.get("ema_line", False):
            ema_style = _style("volume.ema_line").get("line", {"width": 1.5, "color": "rgba(0,100,255,0.8)"})
            fig.add_trace(go.Scatter(
                x=dfx[COL_TIME], y=vol_ema20,
                mode="lines", name="Vol EMA(20)", line=ema_style,
                showlegend=False, yaxis="y2",
                hovertemplate="TF=15M<br>EMA(20): %{y:,.0f}<extra></extra>",
            ))

    # ===================================================================
    # Phase B: M15 structure elements (from each LowerTFResult)
    # ===================================================================
    time_by_idx_m15 = {int(i): t for i, t in zip(dfx.index.to_numpy(), dfx[COL_TIME])}
    idx_set_m15 = set(map(int, dfx.index.to_numpy()))

    for lt in lower_tf_results:
        lt_df = lt.df
        if lt_df.empty:
            continue

        slice_begin = int(lt.meta.get("slice_begin", 0))
        p_sid = lt.trigger.parent_sid
        p_cycle = lt.trigger.parent_cycle_id
        lt_times = pd.to_datetime(lt_df[COL_TIME], utc=True)

        # Build time mapping: slice idx → M15 chart time
        def _lt_time(slice_idx: int):
            """Get M15 chart time from lifecycle-slice index."""
            if 0 <= slice_idx < len(lt_times):
                return lt_times.iloc[slice_idx]
            return None

        def _lt_full_idx(slice_idx: int) -> int:
            """Get full M15 idx from slice idx."""
            return slice_idx + slice_begin

        # Determine if this trigger's structure is the "most recent active"
        # across all lower_tf_results
        last_m15_events = sorted(lt.events, key=lambda e: e.idx)
        last_m15_sid = max((int(e.meta.get("structure_id", 0)) for e in lt.events), default=0) if lt.events else 0
        is_active_trigger = (p_sid == m15_most_recent_psid and p_cycle in m15_recent_cycles)

        # --- Structure swing lines ---
        if struct_cfg.get("levels", False):
            cts_events = [e for e in lt.events if e.type == "CTS_CONFIRMED"]
            bos_events = [e for e in lt.events if e.type == "BOS_CONFIRMED"]
            all_sids_lt = set()
            for e in cts_events + bos_events:
                all_sids_lt.add(int(e.meta.get("structure_id", 0)))
            most_recent_lt_sid = max(all_sids_lt) if all_sids_lt else 0

            # Build confirmed points per sid
            points_by_sid = defaultdict(list)
            for ev in cts_events:
                p_idx = int(ev.meta.get("cts_anchor_idx", ev.idx))
                t = _lt_time(p_idx)
                if t is None:
                    continue
                price = float(ev.price) if ev.price is not None else 0.0
                if price == 0.0 and "cts_price" in lt_df.columns and p_idx < len(lt_df):
                    price = float(lt_df.iloc[p_idx]["cts_price"]) if not pd.isna(lt_df.iloc[p_idx].get("cts_price", float("nan"))) else 0.0
                sid = int(ev.meta.get("structure_id", 0))
                cycle = int(ev.meta.get("cycle_id", 0))
                sd = int(ev.meta.get("struct_direction", 0))
                full_idx = _lt_full_idx(p_idx)
                points_by_sid[sid].append((p_idx, t, price, "CTS", sid, cycle, sd, full_idx))

            for ev in bos_events:
                t = _lt_time(ev.idx)
                if t is None:
                    continue
                price = float(ev.price) if ev.price is not None else 0.0
                sid = int(ev.meta.get("structure_id", 0))
                cycle = int(ev.meta.get("cycle_id", 0))
                sd = int(ev.meta.get("struct_direction", 0))
                full_idx = _lt_full_idx(ev.idx)
                points_by_sid[sid].append((ev.idx, t, price, "BOS", sid, cycle, sd, full_idx))

            # Unconfirmed CTS + PB dots
            cts_unconf = [e for e in lt.events if e.type in ("CTS_ESTABLISHED", "CTS_UPDATED")]
            pb_events = [e for e in lt.events if e.type == "STATE_CHANGED" and e.meta.get("to") == "pullback"]

            extra_cts_pts = []
            extra_pb_pts = []
            pb_to_bos_lines = []

            for sid in sorted(all_sids_lt):
                sid_pts = sorted(points_by_sid.get(sid, []), key=lambda x: x[0])
                if not sid_pts:
                    continue
                last_pt = sid_pts[-1]
                last_kind = last_pt[3]
                last_slice_idx = last_pt[0]
                sd_for_sid = last_pt[6]

                if last_kind == "BOS":
                    cts_after = [e for e in cts_unconf
                                 if int(e.meta.get("structure_id", -1)) == sid
                                 and int(e.idx) > last_slice_idx]
                    if cts_after:
                        latest = max(cts_after, key=lambda e: int(e.idx))
                        t = _lt_time(latest.idx)
                        if t is not None:
                            price = float(latest.price) if latest.price is not None else 0.0
                            cycle = int(latest.meta.get("cycle_id", 0))
                            full_idx = _lt_full_idx(latest.idx)
                            kind_label = latest.type.replace("CTS_", "").lower()
                            points_by_sid[sid].append((latest.idx, t, price, "CTS", sid, cycle, sd_for_sid, full_idx))
                            extra_cts_pts.append((latest.idx, t, price, f"CTS ({kind_label})", sid, cycle, sd_for_sid, full_idx))

                elif last_kind == "CTS" and sid != most_recent_lt_sid:
                    next_sid = sid + 1
                    next_bos_evs = sorted(
                        [e for e in bos_events if int(e.meta.get("structure_id", -1)) == next_sid],
                        key=lambda e: int(e.idx),
                    )
                    next_bos_idx = int(next_bos_evs[0].idx) if next_bos_evs else None

                    pb_after = [e for e in pb_events
                                if int(e.meta.get("structure_id", -1)) == sid
                                and int(e.idx) > last_slice_idx
                                and (next_bos_idx is None or int(e.idx) < next_bos_idx)]
                    if pb_after:
                        latest_pb = max(pb_after, key=lambda e: int(e.idx))
                        t = _lt_time(latest_pb.idx)
                        if t is not None:
                            if sd_for_sid == 1:
                                pb_price = float(lt_df.iloc[latest_pb.idx][COL_L])
                            else:
                                pb_price = float(lt_df.iloc[latest_pb.idx][COL_H])
                            full_idx = _lt_full_idx(latest_pb.idx)
                            points_by_sid[sid].append((latest_pb.idx, t, pb_price, "PB", sid, 0, sd_for_sid, full_idx))
                            extra_pb_pts.append((latest_pb.idx, t, pb_price, "PB", sid, 0, sd_for_sid, full_idx))
                            if next_bos_evs:
                                fb = next_bos_evs[0]
                                fb_t = _lt_time(fb.idx)
                                fb_price = float(fb.price) if fb.price is not None else 0.0
                                if fb_t is not None:
                                    pb_to_bos_lines.append((sid, t, pb_price, fb_t, fb_price))

            # Draw swing lines per sid
            for sid in sorted(points_by_sid.keys()):
                sid_pts = sorted(points_by_sid[sid], key=lambda x: x[0])
                x_line = [p[1] for p in sid_pts]
                y_line = [p[2] for p in sid_pts]

                if sid == most_recent_lt_sid:
                    # Extend to last candle in lifecycle
                    end_time = lt_times.iloc[-1]
                    end_price = float(lt_df[COL_C].iloc[-1])
                    x_line.append(end_time)
                    y_line.append(end_price)

                line_style = _style("structure.swing_line").copy()
                base_opacity = float(line_style.get("opacity", 0.9))
                op_mult = _m15_opacity_tier_for_events(
                    p_sid, p_cycle, m15_most_recent_psid, m15_recent_cycles,
                    is_active_trigger and sid == most_recent_lt_sid,
                )
                if "line" in line_style:
                    line_style["line"] = dict(line_style["line"])
                else:
                    line_style["line"] = {}
                line_style["opacity"] = base_opacity * op_mult

                fig.add_trace(go.Scatter(
                    x=x_line, y=y_line, mode="lines",
                    name=f"M15 swing h1s{p_sid}c{p_cycle}_m15s{sid}",
                    hoverinfo="skip", line_shape="linear",
                    showlegend=False, **line_style,
                ))

            # Cross-structure PB→BOS lines
            for _pb_sid, pb_t, pb_p, bos_t, bos_p in pb_to_bos_lines:
                line_style = _style("structure.swing_line").copy()
                base_opacity = float(line_style.get("opacity", 0.9))
                op_mult = _m15_opacity_tier_for_events(
                    p_sid, p_cycle, m15_most_recent_psid, m15_recent_cycles, False,
                )
                if "line" in line_style:
                    line_style["line"] = dict(line_style["line"])
                else:
                    line_style["line"] = {}
                line_style["opacity"] = base_opacity * op_mult
                fig.add_trace(go.Scatter(
                    x=[pb_t, bos_t], y=[pb_p, bos_p], mode="lines",
                    name=f"M15 PB→BOS h1s{p_sid}c{p_cycle}",
                    hoverinfo="skip", line_shape="linear", showlegend=False, **line_style,
                ))

            # --- CTS confirmed dots ---
            all_cts_pts = []
            for sid_pts in points_by_sid.values():
                for p in sid_pts:
                    if p[3] == "CTS":
                        all_cts_pts.append(p)
            if all_cts_pts:
                _render_m15_dots(fig, all_cts_pts, "CTS", p_sid, p_cycle,
                                 m15_most_recent_psid, m15_recent_cycles, is_active_trigger,
                                 most_recent_lt_sid, m15_to_h1)

            # --- BOS confirmed dots ---
            all_bos_pts = []
            for sid_pts in points_by_sid.values():
                for p in sid_pts:
                    if p[3] == "BOS":
                        all_bos_pts.append(p)
            if all_bos_pts:
                _render_m15_dots(fig, all_bos_pts, "BOS", p_sid, p_cycle,
                                 m15_most_recent_psid, m15_recent_cycles, is_active_trigger,
                                 most_recent_lt_sid, m15_to_h1)

            # --- Unconfirmed CTS dots ---
            if extra_cts_pts:
                _render_m15_dots(fig, extra_cts_pts, "CTS (unconf)", p_sid, p_cycle,
                                 m15_most_recent_psid, m15_recent_cycles, is_active_trigger,
                                 most_recent_lt_sid, m15_to_h1)

            # --- PB dots ---
            if extra_pb_pts:
                _render_m15_dots(fig, extra_pb_pts, "PB", p_sid, p_cycle,
                                 m15_most_recent_psid, m15_recent_cycles, is_active_trigger,
                                 most_recent_lt_sid, m15_to_h1)

        # --- KL zone rectangles + hover ---
        if zone_cfg.get("KL", False) and lt.kl_zones:
            t_last_m15 = dfx[COL_TIME].iloc[-1]

            for zone in lt.kl_zones:
                side = str(zone.side)
                stz = _zone_style(side)
                op_mult = _m15_opacity_tier_for_zone(zone, m15_most_recent_psid, m15_recent_cycles)

                base_fill_op = float(stz.get("fill_opacity_active", 0.18))
                base_line_op = float(stz.get("confirm_opacity_active", 0.9))
                fill_op = base_fill_op * op_mult
                line_op = base_line_op * op_mult
                rgb = str(stz.get("rgb", "0,180,0" if side == "buy" else "220,0,0"))
                confirm_w = int(stz.get("confirm_line_width", 2))

                fillcolor = _rgba_from_rgb(rgb, fill_op)
                linecolor = _rgba_from_rgb(rgb, line_op)

                x0 = pd.to_datetime(zone.start_time, utc=True)
                x1 = pd.to_datetime(zone.end_time, utc=True) if zone.end_time else t_last_m15

                y0 = float(min(zone.top, zone.bottom))
                y1 = float(max(zone.top, zone.bottom))

                conf_idx = int(zone.meta.get("confirmed_idx", -1))
                conf_time = None
                # conf_idx is slice-relative — convert to time from lt_df
                if 0 <= conf_idx < len(lt_df):
                    conf_time = pd.to_datetime(lt_df.iloc[conf_idx][COL_TIME], utc=True)

                steps = list((zone.meta or {}).get("bounds_steps", []))
                if not steps:
                    steps = [{"start_idx": int(zone.meta.get("base_idx", 0)),
                              "top": y1, "bottom": y0, "event": "FALLBACK"}]
                steps = sorted(steps, key=lambda s: int(s.get("start_idx", -1)))

                for k, s in enumerate(steps):
                    seg_start_slice = int(s.get("start_idx", -1))
                    seg_x0 = _lt_time(seg_start_slice)
                    if seg_x0 is None:
                        continue
                    if seg_x0 < x0:
                        seg_x0 = x0

                    if k + 1 < len(steps):
                        nxt = _lt_time(int(steps[k + 1].get("start_idx", -1))) or x1
                        seg_x1 = nxt - pd.Timedelta(microseconds=1)
                    else:
                        seg_x1 = x1

                    seg_top = float(s.get("top", y1))
                    seg_bot = float(s.get("bottom", y0))
                    sy0 = min(seg_bot, seg_top)
                    sy1 = max(seg_bot, seg_top)

                    fig.add_shape(type="rect", xref="x", yref="y",
                                  x0=seg_x0, x1=seg_x1, y0=sy0, y1=sy1,
                                  fillcolor=fillcolor, line=dict(width=0), layer="below")

                    if conf_time is not None and seg_x0 <= conf_time <= seg_x1:
                        fig.add_shape(type="line", xref="x", yref="y",
                                      x0=conf_time, x1=conf_time, y0=sy0, y1=sy1,
                                      line=dict(color=linecolor, width=confirm_w), layer="below")

                    # Hover
                    seg_times = dfx[COL_TIME][(dfx[COL_TIME] >= seg_x0) & (dfx[COL_TIME] <= seg_x1)]
                    if len(seg_times) == 0:
                        seg_times = pd.Series([seg_x0, seg_x1])

                    structure_id = int(zone.meta.get("structure_id", -1))
                    cycle_id = int(zone.meta.get("cycle_id", 0))
                    hover_cd = [[
                        side, structure_id,
                        int(zone.meta.get("struct_direction", 0)),
                        str(zone.meta.get("base_pattern", "")),
                        int(zone.meta.get("base_idx", -1)),
                        conf_idx, cycle_id,
                        sy1, sy0, p_sid, p_cycle,
                    ]] * len(seg_times)

                    kl_hover_line = {"width": 6, "color": "rgba(0,0,0,0)"}
                    for yval in (sy1, sy0):
                        fig.add_trace(go.Scatter(
                            x=seg_times, y=[yval] * len(seg_times),
                            mode="lines", name="M15 KL zone", showlegend=False,
                            line=kl_hover_line, line_shape="hv",
                            hovertemplate=(
                                "TF=15M<br>"
                                "KL Zone<br>"
                                "side=%{customdata[0]}<br>"
                                "sid=%{customdata[1]} | parent_sid=%{customdata[9]}<br>"
                                "cycle_id=%{customdata[6]} | parent_cycle_id=%{customdata[10]}<br>"
                                "struct_direction=%{customdata[2]}<br>"
                                "base_pattern=%{customdata[3]}<br>"
                                "base_idx=%{customdata[4]}<br>"
                                "confirmed_idx=%{customdata[5]}<br>"
                                "top=%{customdata[7]:.5f}<br>"
                                "bottom=%{customdata[8]:.5f}"
                                "<extra></extra>"
                            ),
                            customdata=hover_cd,
                        ))

        # --- Wave candle verticals + WVMI hover ---
        if zone_cfg.get("wave_candles", True) and lt.wave_candles:
            wvmi_by_idx = {}
            for rec in lt.wvmi_records:
                for _iv, _role in [(rec.fb_idx, "FB"), (rec.lb_idx, "LB"),
                                   (rec.fp_idx, "FP"), (rec.lp_idx, "LP")]:
                    if _iv is not None:
                        wvmi_by_idx[_iv] = (rec, _role)

            wc_y_min = float(dfx[COL_L].min())
            wc_y_max = float(dfx[COL_H].max())

            zone_lookup_lt = {}
            for z in lt.kl_zones:
                zk = (int(z.meta.get("structure_id", 0)), int(z.meta.get("cycle_id", 0)), str(z.source_kind))
                zone_lookup_lt[zk] = z

            for wc in lt.wave_candles:
                zk = (wc.structure_id, wc.cycle_id, wc.source_kind)
                parent_zone = zone_lookup_lt.get(zk)
                if parent_zone is not None:
                    op_mult = _m15_opacity_tier_for_zone(parent_zone, m15_most_recent_psid, m15_recent_cycles)
                else:
                    op_mult = _m15_opacity_tier_for_events(
                        p_sid, p_cycle, m15_most_recent_psid, m15_recent_cycles, is_active_trigger,
                    )

                for idx in (wc.last_wave_candle_idx, wc.first_wave_candle_idx):
                    if idx is None or idx >= len(lt_df):
                        continue
                    candle_dir = int(lt_df.iloc[idx]["direction"])
                    if candle_dir == 0:
                        continue

                    style_key = "wave_candle.bullish" if candle_dir == 1 else "wave_candle.bearish"
                    wc_style = _style(style_key)
                    line_info = wc_style.get("line", {})
                    color_rgb = line_info.get("color_rgb", "128,128,128")
                    base_opacity = float(wc_style.get("opacity", 0.8))
                    line_width = int(line_info.get("width", 1))

                    final_color = f"rgba({color_rgb}, {base_opacity * op_mult})"
                    wc_time = _lt_time(idx)
                    if wc_time is None:
                        continue

                    fig.add_shape(type="line", xref="x", yref="paper",
                                  x0=wc_time, x1=wc_time, y0=0, y1=1,
                                  line=dict(color=final_color, width=line_width), layer="below")

                    # WVMI hover
                    vol = float(lt_df.iloc[idx]["volume"])
                    full_idx = _lt_full_idx(idx)
                    h1_info = m15_to_h1.get(wc_time, (None, None))

                    wvmi_entry = wvmi_by_idx.get(idx)
                    if wvmi_entry:
                        rec, role = wvmi_entry
                        if role == "LB":
                            weighted_vol = vol * rec.lb_weight
                        elif role == "LP":
                            weighted_vol = vol * rec.lp_weight
                        else:
                            weighted_vol = vol
                    else:
                        role = None
                        weighted_vol = vol

                    hover_lines = [
                        "TF=15M",
                        "<b>Wave Candle</b>",
                        f"idx={full_idx}  idx_1H={h1_info[1]}",
                        f"BOS zone: sid={wc.structure_id} cycle={wc.cycle_id}",
                        f"parent_sid={p_sid} parent_cycle={p_cycle}",
                        f"Volume: {vol:.0f}",
                        f"Weighted vol: {weighted_vol:.0f}",
                    ]

                    if wvmi_entry and role in ("LB", "LP"):
                        rec, role = wvmi_entry
                        hover_lines.append("---")
                        mom_label = "Buy" if candle_dir == 1 else "Sell"
                        if role == "LB":
                            mom_val = rec.breakout_momentum
                            hover_lines.append(f"{mom_label} momentum: {mom_val:.4f}" if mom_val is not None else f"{mom_label} momentum: N/A")
                            hover_lines.append(f"FB wt vol: {rec.fb_volume:.0f}" if rec.fb_volume is not None else "FB wt vol: N/A")
                            hover_lines.append(f"LB wt vol: {weighted_vol:.0f}")
                        else:
                            mom_val = rec.pullback_momentum
                            hover_lines.append(f"{mom_label} momentum: {mom_val:.4f}" if mom_val is not None else f"{mom_label} momentum: N/A")
                            hover_lines.append(f"FP wt vol: {rec.fp_volume:.0f}" if rec.fp_volume is not None else "FP wt vol: N/A")
                            hover_lines.append(f"LP wt vol: {weighted_vol:.0f}")

                    _n_pts = 12
                    _y_pts = [wc_y_min + i * (wc_y_max - wc_y_min) / (_n_pts - 1) for i in range(_n_pts)]
                    fig.add_trace(go.Scatter(
                        x=[wc_time] * _n_pts, y=_y_pts,
                        mode="lines", showlegend=False,
                        line=dict(width=8, color="rgba(0,0,0,0)"),
                        hovertemplate="<br>".join(hover_lines) + "<extra></extra>",
                    ))

        # --- POI zone rectangles + hover ---
        if zone_cfg.get("POI", False) and lt.poi_zones:
            t_last_m15 = dfx[COL_TIME].iloc[-1]

            for poi in lt.poi_zones:
                if poi.meta.get("status") == "disappeared":
                    continue
                side = str(poi.side)
                stz = STYLE.get(f"zone.poi.{side}", {})
                zone_status = poi.meta.get("status", "active")

                op_mult = _m15_opacity_tier_for_zone(poi, m15_most_recent_psid, m15_recent_cycles)

                base_fill_op = float(stz.get("fill_opacity_active", 0.35))
                base_line_op = float(stz.get("confirm_opacity_active", 0.85))
                rgb = str(stz.get("rgb", "255, 215, 0"))
                confirm_rgb = str(stz.get("confirm_line_rgb", "139, 69, 19"))
                confirm_w = int(stz.get("confirm_line_width", 2))

                fill_op = base_fill_op * op_mult
                line_op = base_line_op * op_mult
                fillcolor = _rgba_from_rgb(rgb, fill_op)
                linecolor = _rgba_from_rgb(confirm_rgb, line_op)

                x0 = pd.to_datetime(poi.start_time, utc=True)
                x1 = pd.to_datetime(poi.end_time, utc=True) if poi.end_time else t_last_m15

                y0 = float(min(poi.top, poi.bottom))
                y1 = float(max(poi.top, poi.bottom))

                conf_idx = int(poi.meta.get("confirmed_idx", poi.ic_idx))
                conf_time = None
                if 0 <= conf_idx < len(lt_df):
                    conf_time = pd.to_datetime(lt_df.iloc[conf_idx][COL_TIME], utc=True)

                fig.add_shape(type="rect", xref="x", yref="y",
                              x0=x0, x1=x1, y0=y0, y1=y1,
                              fillcolor=fillcolor, line=dict(width=0), layer="below")

                if conf_time is not None:
                    fig.add_shape(type="line", xref="x", yref="y",
                                  x0=conf_time, x1=conf_time, y0=y0, y1=y1,
                                  line=dict(color=linecolor, width=confirm_w), layer="below")

                seg_times = dfx[COL_TIME][(dfx[COL_TIME] >= x0) & (dfx[COL_TIME] <= x1)]
                if len(seg_times) == 0:
                    seg_times = pd.Series([x0, x1])

                versions = poi.meta.get("versions", [])
                versions_str = ", ".join(versions) if versions else "none"
                poi_cd = [[
                    side, poi.meta.get("structure_id", -1), poi.meta.get("struct_direction", 0),
                    poi.ic_idx, conf_idx, poi.meta.get("cycle_id", 0),
                    versions_str, y1, y0, zone_status, p_sid, p_cycle,
                ]] * len(seg_times)

                poi_hover_line = {"width": 6, "color": "rgba(0,0,0,0)"}
                for yval in (y1, y0):
                    fig.add_trace(go.Scatter(
                        x=seg_times, y=[yval] * len(seg_times),
                        mode="lines", name="M15 POI zone", showlegend=False,
                        line=poi_hover_line, line_shape="hv",
                        hovertemplate=(
                            "TF=15M<br>"
                            "<b>POI Zone</b><br>"
                            "side=%{customdata[0]}<br>"
                            "sid=%{customdata[1]} | parent_sid=%{customdata[10]}<br>"
                            "cycle_id=%{customdata[5]} | parent_cycle_id=%{customdata[11]}<br>"
                            "struct_direction=%{customdata[2]}<br>"
                            "ic_idx=%{customdata[3]}<br>"
                            "confirmed_idx=%{customdata[4]}<br>"
                            "versions=%{customdata[6]}<br>"
                            "top=%{customdata[7]:.5f}<br>"
                            "bottom=%{customdata[8]:.5f}<br>"
                            "status=%{customdata[9]}"
                            "<extra></extra>"
                        ),
                        customdata=poi_cd,
                    ))

        # --- Prev BOS lines ---
        for line_info in lt.prev_bos_lines:
            start_slice = line_info["start_idx"]
            end_slice = line_info["end_idx"]
            price = line_info["price"]
            t0 = _lt_time(start_slice)
            t1 = _lt_time(end_slice)
            if t0 is not None and t1 is not None:
                op_mult = _m15_opacity_tier_for_events(
                    p_sid, p_cycle, m15_most_recent_psid, m15_recent_cycles, False,
                )
                line_s = _style("prev_bos_line").get("line", {"width": 2, "color": "black"})
                line_s = dict(line_s)
                # Apply opacity
                fig.add_trace(go.Scatter(
                    x=[t0, t1], y=[price, price], mode="lines", line=line_s,
                    name=f"M15 Prev BOS h1s{p_sid}c{p_cycle}", showlegend=False,
                    hovertemplate=(
                        f"TF=15M<br>"
                        f"Prev BOS Line<br>"
                        f"Price: {price:.5f}<br>"
                        f"parent_sid={p_sid} parent_cycle={p_cycle}<br>"
                        f"<extra></extra>"
                    ),
                ))

    # ===================================================================
    # Phase C: H1 overlay
    # ===================================================================
    _render_h1_overlay(fig, dfx, h1_df, h1_to_m15, m15_to_h1,
                       state_cfg, struct_cfg, zone_cfg)

    # ===================================================================
    # Phase D: Layout + Output
    # ===================================================================
    fig.update_layout(
        title=title,
        xaxis_title="Time (UTC)" if not volume_enabled else None,
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        legend_title="Overlays",
        height=800,
    )

    if volume_enabled:
        fig.update_layout(
            xaxis=dict(anchor="y2", side="bottom", title_text="Time (UTC)", showline=False),
            yaxis=dict(domain=[0.15, 1.0], showline=False),
            yaxis2=dict(domain=[0, 0.15], showgrid=False, zeroline=False,
                        showticklabels=True, side="right", tickformat=",", showline=False),
        )
        border_color = "rgba(0,0,0,0.4)"
        for x0, x1, y0, y1 in [(0, 1, 1, 1), (0, 1, 0, 0), (0, 0, 0, 1), (1, 1, 0, 1)]:
            fig.add_shape(type="line", xref="paper", yref="paper",
                          x0=x0, x1=x1, y0=y0, y1=y1,
                          line=dict(color=border_color, width=1))

    fig.update_layout(**_style("chart.layout"))
    fig.update_xaxes(**_style("chart.axis"))
    fig.update_yaxes(**_style("chart.axis"))

    # Dynamic gap removal
    t = pd.to_datetime(dfx[COL_TIME]).sort_values().reset_index(drop=True)
    dt = t.diff()
    expected = dt[dt.notna()].median()
    if pd.isna(expected) or expected <= pd.Timedelta(0):
        expected = pd.Timedelta(minutes=15)

    gap_mask = dt > (expected * 2.0)
    missing = []
    t_values = t.to_list()
    for i in range(1, len(t_values)):
        if bool(gap_mask.iloc[i]):
            start = t_values[i - 1] + expected
            end = t_values[i] - expected
            if start <= end:
                missing.extend(pd.date_range(start, end, freq=expected).to_pydatetime())

    present = set(t_values)
    missing = [x for x in missing if x not in present]
    if missing:
        fig.update_xaxes(
            rangebreaks=[dict(values=missing, dvalue=int(expected / pd.Timedelta(milliseconds=1)))]
        )

    html_path = out_dir / f"{basename}.html"
    png_path = out_dir / f"{basename}.png"

    print(f"[m15_chart] traces: {len(fig.data)}, shapes: {len(fig.layout.shapes) if fig.layout.shapes else 0}")

    fig.write_html(str(html_path), include_plotlyjs="cdn")

    # Volume auto-rescale JS
    if volume_enabled:
        _inject_volume_autoscale_js(html_path)

    fig.write_image(str(png_path), scale=2)

    return ChartExportPaths(html_path=html_path, png_path=png_path)


# ---------------------------------------------------------------------------
# Helper: Render M15 structure dots (CTS/BOS/PB)
# ---------------------------------------------------------------------------

def _render_m15_dots(
    fig, pts, kind_label, p_sid, p_cycle,
    most_recent_psid, recent_cycles, is_active_trigger,
    most_recent_lt_sid, m15_to_h1,
):
    """Render M15 structure dots with TF=15M hover."""
    style_key = "structure.cts" if "CTS" in kind_label else "structure.bos"
    style = _style(style_key).copy()

    cd = []
    for p in pts:
        # p = (slice_idx, time, price, kind, sid, cycle, sd, full_idx)
        h1_info = m15_to_h1.get(p[1], (None, None))
        cd.append([
            p[7],  # full M15 idx
            p[3],  # kind
            p[2],  # price
            p[4],  # m15 sid
            p[5],  # m15 cycle_id
            p[6],  # sd
            p_sid,  # parent_sid
            p_cycle,  # parent_cycle_id
            h1_info[1] if h1_info[1] is not None else "",  # idx_1H
            str(h1_info[0]) if h1_info[0] is not None else "",  # time_1H
        ])

    fig.add_trace(go.Scatter(
        x=[p[1] for p in pts],
        y=[p[2] for p in pts],
        mode="markers",
        name=f"M15 {kind_label} h1s{p_sid}c{p_cycle}",
        showlegend=False,
        customdata=cd,
        hovertemplate=(
            "TF=15M<br>"
            "idx=%{customdata[0]}<br>"
            "idx_1H=%{customdata[8]}<br>"
            "kind=%{customdata[1]}<br>"
            "price=%{customdata[2]:.5f}<br>"
            "sid=%{customdata[3]} | parent_sid=%{customdata[6]}<br>"
            "cycle_id=%{customdata[4]} | parent_cycle_id=%{customdata[7]}<br>"
            "struct_direction=%{customdata[5]}"
            "<extra></extra>"
        ),
        **style,
    ))


# ---------------------------------------------------------------------------
# Phase C: H1 overlay rendering
# ---------------------------------------------------------------------------

def _render_h1_overlay(fig, dfx, h1_df, h1_to_m15, m15_to_h1, state_cfg, struct_cfg, zone_cfg):
    """Render all H1 elements as overlays on the M15 chart."""
    h1_times = pd.to_datetime(h1_df[COL_TIME], utc=True)
    structure_events = h1_df.attrs.get("structure_events", [])
    m15_t_last = dfx[COL_TIME].iloc[-1]
    m15_t_first = dfx[COL_TIME].iloc[0]

    def _h1_to_m15_time(h1_time):
        """Map H1 time to M15 4th candle, clamped to M15 range."""
        t = h1_to_m15.get(h1_time, h1_time + timedelta(minutes=45))
        if t < m15_t_first:
            return m15_t_first
        if t > m15_t_last:
            return m15_t_last
        return t

    def _h1_idx_to_m15_time(h1_idx):
        """Map H1 index to M15 4th candle time."""
        if h1_idx not in h1_df.index:
            return None
        h1_time = pd.to_datetime(h1_df.loc[h1_idx, COL_TIME], utc=True)
        return _h1_to_m15_time(h1_time)

    # Find most recent H1 sid for opacity
    all_h1_sids = set()
    for ev in structure_events:
        sid = ev.meta.get("structure_id")
        if sid is not None:
            all_h1_sids.add(int(sid))
    most_recent_h1_sid = max(all_h1_sids) if all_h1_sids else 0

    rev_confirmed_by_sid = _get_reversal_confirmed_by_sid(structure_events)

    # --- H1 breakout labels ("bo") ---
    if state_cfg.get("labels", False) or True:  # Always show H1 labels on M15 chart
        bo_events = [
            ev for ev in structure_events
            if ev.type == "STATE_CHANGED" and ev.meta.get("to") == "breakout"
            and ev.idx in h1_df.index
        ]
        if bo_events:
            x_vals, y_vals, cd = [], [], []
            for ev in bo_events:
                m15_t = _h1_idx_to_m15_time(ev.idx)
                if m15_t is None:
                    continue
                sd = int(ev.meta.get("struct_direction", 1))
                sid = int(ev.meta.get("structure_id", 0))
                row = h1_df.loc[ev.idx]
                wo = (float(row[COL_H]) - float(row[COL_L])) * 0.15
                y = float(row[COL_H]) + wo * 2 if sd == 1 else float(row[COL_L]) - wo * 2
                x_vals.append(m15_t)
                y_vals.append(y)
                cd.append((ev.idx, "breakout", sid, sd))

            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals, mode="text", text=["bo"] * len(x_vals),
                name="H1 bo", textposition="middle center", showlegend=False,
                hovertemplate=(
                    "TF=1H<br>idx=%{customdata[0]}<br>state=%{customdata[1]}<br>"
                    "sid=%{customdata[2]}<br>struct_direction=%{customdata[3]}<extra></extra>"
                ),
                customdata=cd,
            ))

        # --- H1 state labels (pb/pr/rv) ---
        label_map = {"pullback": "pb", "pullback_range": "pr", "reversal": "rv"}
        state_events = [
            ev for ev in structure_events
            if ev.type == "STATE_CHANGED" and ev.meta.get("to") in label_map
            and ev.idx in h1_df.index
        ]
        if state_events:
            x_vals, y_vals, text_vals, cd = [], [], [], []
            for ev in state_events:
                m15_t = _h1_idx_to_m15_time(ev.idx)
                if m15_t is None:
                    continue
                sd = int(ev.meta.get("struct_direction", 1))
                sid = int(ev.meta.get("structure_id", 0))
                row = h1_df.loc[ev.idx]
                wo = (float(row[COL_H]) - float(row[COL_L])) * 0.15
                y = float(row[COL_L]) - wo * 2 if sd == 1 else float(row[COL_H]) + wo * 2
                x_vals.append(m15_t)
                y_vals.append(y)
                text_vals.append(label_map[ev.meta["to"]])
                cd.append((ev.idx, ev.meta["to"], sid, sd))

            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals, mode="text", text=text_vals,
                name="H1 state labels", textposition="middle center", showlegend=False,
                hovertemplate=(
                    "TF=1H<br>idx=%{customdata[0]}<br>state=%{customdata[1]}<br>"
                    "sid=%{customdata[2]}<br>struct_direction=%{customdata[3]}<extra></extra>"
                ),
                customdata=cd,
            ))

    # --- H1 structure swing lines (dashed) ---
    if struct_cfg.get("levels", False):
        cts_events = [ev for ev in structure_events if ev.type == "CTS_CONFIRMED"]
        bos_events = [ev for ev in structure_events if ev.type == "BOS_CONFIRMED"]

        points_by_sid = defaultdict(list)
        for ev in cts_events:
            p_idx = int(ev.meta.get("cts_anchor_idx", ev.idx))
            m15_t = _h1_idx_to_m15_time(p_idx)
            if m15_t is None:
                continue
            price = float(ev.price) if ev.price is not None else 0.0
            if price == 0.0 and p_idx in h1_df.index and "cts_price" in h1_df.columns:
                price = float(h1_df.loc[p_idx, "cts_price"])
            sid = int(ev.meta.get("structure_id", 0))
            cycle = int(ev.meta.get("cycle_id", 0))
            sd = int(ev.meta.get("struct_direction", 0))
            points_by_sid[sid].append((p_idx, m15_t, price, "CTS", sid, cycle, sd))

        for ev in bos_events:
            m15_t = _h1_idx_to_m15_time(ev.idx)
            if m15_t is None:
                continue
            price = float(ev.price) if ev.price is not None else 0.0
            sid = int(ev.meta.get("structure_id", 0))
            cycle = int(ev.meta.get("cycle_id", 0))
            sd = int(ev.meta.get("struct_direction", 0))
            points_by_sid[sid].append((ev.idx, m15_t, price, "BOS", sid, cycle, sd))

        # Unconfirmed CTS + PB for H1
        cts_unconf = [ev for ev in structure_events if ev.type in ("CTS_ESTABLISHED", "CTS_UPDATED")]
        pb_state = [ev for ev in structure_events if ev.type == "STATE_CHANGED" and ev.meta.get("to") == "pullback"]
        pb_to_bos_lines = []

        for sid in sorted(all_h1_sids):
            sid_pts = sorted(points_by_sid.get(sid, []), key=lambda x: x[0])
            if not sid_pts:
                continue
            last_pt = sid_pts[-1]
            last_kind = last_pt[3]
            last_idx = last_pt[0]
            sd_for_sid = last_pt[6]

            if last_kind == "BOS":
                cts_after = [e for e in cts_unconf if int(e.meta.get("structure_id", -1)) == sid and int(e.idx) > last_idx]
                if cts_after:
                    latest = max(cts_after, key=lambda e: int(e.idx))
                    m15_t = _h1_idx_to_m15_time(latest.idx)
                    if m15_t is not None:
                        price = float(latest.price) if latest.price is not None else 0.0
                        cycle = int(latest.meta.get("cycle_id", 0))
                        points_by_sid[sid].append((latest.idx, m15_t, price, "CTS", sid, cycle, sd_for_sid))

            elif last_kind == "CTS" and sid != most_recent_h1_sid:
                next_sid = sid + 1
                next_bos = sorted(
                    [e for e in bos_events if int(e.meta.get("structure_id", -1)) == next_sid],
                    key=lambda e: int(e.idx),
                )
                next_bos_idx = int(next_bos[0].idx) if next_bos else None
                pb_after = [e for e in pb_state
                            if int(e.meta.get("structure_id", -1)) == sid
                            and int(e.idx) > last_idx
                            and (next_bos_idx is None or int(e.idx) < next_bos_idx)]
                if pb_after:
                    latest_pb = max(pb_after, key=lambda e: int(e.idx))
                    m15_t = _h1_idx_to_m15_time(latest_pb.idx)
                    if m15_t is not None:
                        if sd_for_sid == 1:
                            pb_price = float(h1_df.loc[latest_pb.idx, COL_L])
                        else:
                            pb_price = float(h1_df.loc[latest_pb.idx, COL_H])
                        points_by_sid[sid].append((latest_pb.idx, m15_t, pb_price, "PB", sid, 0, sd_for_sid))
                        if next_bos:
                            fb = next_bos[0]
                            fb_t = _h1_idx_to_m15_time(fb.idx)
                            fb_price = float(fb.price) if fb.price is not None else 0.0
                            if fb_t is not None:
                                pb_to_bos_lines.append((sid, m15_t, pb_price, fb_t, fb_price))

        # Draw H1 swing lines (dashed)
        for sid in sorted(points_by_sid.keys()):
            sid_pts = sorted(points_by_sid[sid], key=lambda x: x[0])
            x_line = [p[1] for p in sid_pts]
            y_line = [p[2] for p in sid_pts]

            if sid == most_recent_h1_sid:
                end_t = _h1_idx_to_m15_time(int(h1_df.index[-1]))
                if end_t is not None:
                    end_price = float(h1_df[COL_C].iloc[-1])
                    x_line.append(end_t)
                    y_line.append(end_price)

            line_style = _style("structure.h1_overlay.swing_line").copy()
            base_opacity = float(line_style.get("opacity", 0.9))
            op_mult = _opacity_tier("active") if sid == most_recent_h1_sid else _opacity_tier("recent_inactive")
            if "line" in line_style:
                line_style["line"] = dict(line_style["line"])
            else:
                line_style["line"] = {}
            line_style["opacity"] = base_opacity * op_mult

            fig.add_trace(go.Scatter(
                x=x_line, y=y_line, mode="lines",
                name=f"H1 swing sid={sid}", hoverinfo="skip",
                line_shape="linear", showlegend=False, **line_style,
            ))

        # Cross-structure PB→BOS lines (dashed)
        for _pb_sid, pb_t, pb_p, bos_t, bos_p in pb_to_bos_lines:
            line_style = _style("structure.h1_overlay.swing_line").copy()
            base_opacity = float(line_style.get("opacity", 0.9))
            op_mult = _opacity_tier("active") if _pb_sid == most_recent_h1_sid else _opacity_tier("recent_inactive")
            if "line" in line_style:
                line_style["line"] = dict(line_style["line"])
            else:
                line_style["line"] = {}
            line_style["opacity"] = base_opacity * op_mult
            fig.add_trace(go.Scatter(
                x=[pb_t, bos_t], y=[pb_p, bos_p], mode="lines",
                name=f"H1 PB→BOS sid={_pb_sid}", hoverinfo="skip",
                line_shape="linear", showlegend=False, **line_style,
            ))

        # H1 CTS/BOS dots
        all_pts = []
        for sid_pts in points_by_sid.values():
            all_pts.extend(sid_pts)
        for kind_filter, style_key in [("CTS", "structure.cts"), ("BOS", "structure.bos"), ("PB", "structure.bos")]:
            pts = [p for p in all_pts if p[3] == kind_filter or (kind_filter == "CTS" and p[3].startswith("CTS"))]
            if not pts:
                continue
            style = _style(style_key).copy()
            cd = [[p[0], p[3], p[2], p[4], p[5], p[6]] for p in pts]
            fig.add_trace(go.Scatter(
                x=[p[1] for p in pts], y=[p[2] for p in pts],
                mode="markers", name=f"H1 {kind_filter}",
                showlegend=False, customdata=cd,
                hovertemplate=(
                    "TF=1H<br>idx=%{customdata[0]}<br>kind=%{customdata[1]}<br>"
                    "price=%{customdata[2]:.5f}<br>sid=%{customdata[3]}<br>"
                    "cycle_id=%{customdata[4]}<br>struct_direction=%{customdata[5]}<extra></extra>"
                ),
                **style,
            ))

    # --- H1 KL zones (transparent, dashed border) ---
    h1_kl_zones = h1_df.attrs.get("kl_zones", [])
    if zone_cfg.get("KL", False) and h1_kl_zones:
        by_struct = {}
        for z in h1_kl_zones:
            sid = int(z.meta.get("structure_id", 0))
            by_struct.setdefault(sid, []).append(z)

        all_kl_sids = sorted(by_struct.keys(), reverse=True)
        num_structures = int(zone_cfg.get("num_structures", 99))
        selected_kl_sids = set(all_kl_sids[:num_structures])
        most_recent_kl_sid = all_kl_sids[0] if all_kl_sids else 0

        for z in h1_kl_zones:
            zone_sid = int(z.meta.get("structure_id", 0))
            if zone_sid not in selected_kl_sids:
                continue

            side = str(z.side)
            stz = STYLE.get(f"zone.h1_overlay.kl.{side}", {})
            active = bool(z.meta.get("active", False)) and (z.end_time is None)

            if active:
                op_mult = _opacity_tier("active")
            elif zone_sid == most_recent_kl_sid:
                op_mult = _opacity_tier("recent_inactive")
            else:
                op_mult = _opacity_tier("prior_inactive")

            rgb = str(stz.get("rgb", "0,180,0" if side == "buy" else "220,0,0"))
            fill_op = 0.0  # Always transparent fill for H1 overlay
            border_op = float(stz.get("border_opacity_active", 0.9)) * op_mult
            confirm_op = float(stz.get("confirm_opacity_active", 0.9)) * op_mult
            border_w = int(stz.get("border_line_width", 2))
            confirm_w = int(stz.get("confirm_line_width", 2))
            line_dash = stz.get("line_dash", "dash")

            border_color = _rgba_from_rgb(rgb, border_op)
            confirm_color = _rgba_from_rgb(rgb, confirm_op)

            x0 = _h1_to_m15_time(pd.to_datetime(z.start_time, utc=True))
            x1 = _h1_to_m15_time(pd.to_datetime(z.end_time, utc=True)) if z.end_time else m15_t_last

            conf_idx = int(z.meta.get("confirmed_idx", -1))
            conf_time = _h1_idx_to_m15_time(conf_idx) if conf_idx >= 0 else None

            steps = list((z.meta or {}).get("bounds_steps", []))
            if not steps:
                steps = [{"start_idx": int(z.meta.get("base_idx", 0)),
                          "top": float(max(z.top, z.bottom)),
                          "bottom": float(min(z.top, z.bottom)),
                          "event": "FALLBACK"}]
            steps = sorted(steps, key=lambda s: int(s.get("start_idx", -1)))

            for k, s in enumerate(steps):
                seg_x0 = _h1_idx_to_m15_time(int(s.get("start_idx", -1)))
                if seg_x0 is None:
                    continue
                if seg_x0 < x0:
                    seg_x0 = x0

                if k + 1 < len(steps):
                    nxt = _h1_idx_to_m15_time(int(steps[k + 1].get("start_idx", -1))) or x1
                    seg_x1 = nxt - pd.Timedelta(microseconds=1)
                else:
                    seg_x1 = x1

                seg_top = float(s.get("top", z.top))
                seg_bot = float(s.get("bottom", z.bottom))
                sy0 = min(seg_bot, seg_top)
                sy1 = max(seg_bot, seg_top)

                # Rectangle with transparent fill and dashed border
                fig.add_shape(type="rect", xref="x", yref="y",
                              x0=seg_x0, x1=seg_x1, y0=sy0, y1=sy1,
                              fillcolor=f"rgba({rgb}, 0.0)",
                              line=dict(width=border_w, dash=line_dash, color=border_color),
                              layer="below")

                # Confirm line (solid, not dashed)
                if conf_time is not None and seg_x0 <= conf_time <= seg_x1:
                    fig.add_shape(type="line", xref="x", yref="y",
                                  x0=conf_time, x1=conf_time, y0=sy0, y1=sy1,
                                  line=dict(color=confirm_color, width=confirm_w), layer="below")

                # Hover
                seg_times = dfx[COL_TIME][(dfx[COL_TIME] >= seg_x0) & (dfx[COL_TIME] <= seg_x1)]
                if len(seg_times) == 0:
                    seg_times = pd.Series([seg_x0, seg_x1])

                hover_cd = [[
                    side, int(z.meta.get("structure_id", -1)),
                    int(z.meta.get("struct_direction", 0)),
                    str(z.meta.get("base_pattern", "")),
                    int(z.meta.get("base_idx", -1)),
                    conf_idx, int(z.meta.get("cycle_id", 0)),
                    sy1, sy0,
                ]] * len(seg_times)

                h1_hover_line = {"width": 6, "color": "rgba(0,0,0,0)"}
                for yval in (sy1, sy0):
                    fig.add_trace(go.Scatter(
                        x=seg_times, y=[yval] * len(seg_times),
                        mode="lines", name="H1 KL zone (overlay)", showlegend=False,
                        line=h1_hover_line, line_shape="hv",
                        hovertemplate=(
                            "TF=1H<br>"
                            "KL Zone<br>"
                            "side=%{customdata[0]}<br>"
                            "structure_id=%{customdata[1]}<br>"
                            "struct_direction=%{customdata[2]}<br>"
                            "base_pattern=%{customdata[3]}<br>"
                            "base_idx=%{customdata[4]}<br>"
                            "confirmed_idx=%{customdata[5]}<br>"
                            "cycle_id=%{customdata[6]}<br>"
                            "top=%{customdata[7]:.5f}<br>"
                            "bottom=%{customdata[8]:.5f}"
                            "<extra></extra>"
                        ),
                        customdata=hover_cd,
                    ))

    # --- H1 Wave candle verticals (dashed) ---
    h1_wave_candles = h1_df.attrs.get("wave_candles", [])
    h1_wvmi = h1_df.attrs.get("wvmi", [])
    if zone_cfg.get("wave_candles", True) and h1_wave_candles:
        wvmi_by_idx_h1 = {}
        for rec in h1_wvmi:
            for _iv, _role in [(rec.fb_idx, "FB"), (rec.lb_idx, "LB"),
                               (rec.fp_idx, "FP"), (rec.lp_idx, "LP")]:
                if _iv is not None:
                    wvmi_by_idx_h1[_iv] = (rec, _role)

        wc_y_min = float(dfx[COL_L].min())
        wc_y_max = float(dfx[COL_H].max())

        h1_zone_lookup = {}
        for z in h1_kl_zones:
            zk = (int(z.meta.get("structure_id", 0)), int(z.meta.get("cycle_id", 0)), str(z.source_kind))
            h1_zone_lookup[zk] = z

        # Determine most recent H1 sid for wave candle opacity
        wc_h1_sids = sorted(set(wc.structure_id for wc in h1_wave_candles), reverse=True)
        most_recent_wc_sid = wc_h1_sids[0] if wc_h1_sids else 0

        for wc in h1_wave_candles:
            zk = (wc.structure_id, wc.cycle_id, wc.source_kind)
            parent_zone = h1_zone_lookup.get(zk)

            if parent_zone is not None:
                z_active = bool(parent_zone.meta.get("active", False)) and (parent_zone.end_time is None)
                zone_sid = int(parent_zone.meta.get("structure_id", 0))
            else:
                z_active = False
                zone_sid = wc.structure_id

            if z_active:
                op_mult = _opacity_tier("active")
            elif zone_sid == most_recent_wc_sid:
                op_mult = _opacity_tier("recent_inactive")
            else:
                op_mult = _opacity_tier("prior_inactive")

            for idx in (wc.last_wave_candle_idx, wc.first_wave_candle_idx):
                if idx is None or idx not in h1_df.index:
                    continue
                candle_dir = int(h1_df.loc[idx, "direction"])
                if candle_dir == 0:
                    continue

                style_key = "wave_candle.h1_overlay.bullish" if candle_dir == 1 else "wave_candle.h1_overlay.bearish"
                wc_style = _style(style_key)
                line_info = wc_style.get("line", {})
                color_rgb = line_info.get("color_rgb", "128,128,128")
                base_opacity = float(wc_style.get("opacity", 0.8))
                line_width = int(line_info.get("width", 1))
                dash = line_info.get("dash", "dash")

                final_color = f"rgba({color_rgb}, {base_opacity * op_mult})"
                wc_time = _h1_idx_to_m15_time(idx)
                if wc_time is None:
                    continue

                fig.add_shape(type="line", xref="x", yref="paper",
                              x0=wc_time, x1=wc_time, y0=0, y1=1,
                              line=dict(color=final_color, width=line_width, dash=dash),
                              layer="below")

                # WVMI hover
                vol = float(h1_df.loc[idx, "volume"])
                wvmi_entry = wvmi_by_idx_h1.get(idx)
                if wvmi_entry:
                    rec, role = wvmi_entry
                    if role == "LB":
                        weighted_vol = vol * rec.lb_weight
                    elif role == "LP":
                        weighted_vol = vol * rec.lp_weight
                    else:
                        weighted_vol = vol
                else:
                    role = None
                    weighted_vol = vol

                hover_lines = [
                    "TF=1H",
                    "<b>Wave Candle</b>",
                    f"idx={idx}",
                    f"BOS zone: sid={wc.structure_id} cycle={wc.cycle_id}",
                    f"Volume: {vol:.0f}",
                    f"Weighted vol: {weighted_vol:.0f}",
                ]
                if wvmi_entry and role in ("LB", "LP"):
                    rec, role = wvmi_entry
                    hover_lines.append("---")
                    mom_label = "Buy" if candle_dir == 1 else "Sell"
                    if role == "LB":
                        mv = rec.breakout_momentum
                        hover_lines.append(f"{mom_label} momentum: {mv:.4f}" if mv is not None else f"{mom_label} momentum: N/A")
                        hover_lines.append(f"FB wt vol: {rec.fb_volume:.0f}" if rec.fb_volume is not None else "FB wt vol: N/A")
                        hover_lines.append(f"LB wt vol: {weighted_vol:.0f}")
                    else:
                        mv = rec.pullback_momentum
                        hover_lines.append(f"{mom_label} momentum: {mv:.4f}" if mv is not None else f"{mom_label} momentum: N/A")
                        hover_lines.append(f"FP wt vol: {rec.fp_volume:.0f}" if rec.fp_volume is not None else "FP wt vol: N/A")
                        hover_lines.append(f"LP wt vol: {weighted_vol:.0f}")

                _n_pts = 12
                _y_pts = [wc_y_min + i * (wc_y_max - wc_y_min) / (_n_pts - 1) for i in range(_n_pts)]
                fig.add_trace(go.Scatter(
                    x=[wc_time] * _n_pts, y=_y_pts,
                    mode="lines", showlegend=False,
                    line=dict(width=8, color="rgba(0,0,0,0)"),
                    hovertemplate="<br>".join(hover_lines) + "<extra></extra>",
                ))

    # --- H1 POI zones (transparent, dashed border) ---
    h1_poi_zones = h1_df.attrs.get("poi_zones", [])
    if zone_cfg.get("POI", False) and h1_poi_zones:
        all_poi_sids = set(int(z.meta.get("structure_id", 0)) for z in h1_poi_zones)
        most_recent_poi_sid = max(all_poi_sids) if all_poi_sids else 0

        for poi in h1_poi_zones:
            if poi.meta.get("status") == "disappeared":
                continue

            side = str(poi.side)
            stz = STYLE.get(f"zone.h1_overlay.poi.{side}", {})
            zone_status = poi.meta.get("status", "active")
            zone_sid = int(poi.meta.get("structure_id", 0))

            if zone_status == "active":
                op_mult = _opacity_tier("active")
            elif zone_sid == most_recent_poi_sid:
                op_mult = _opacity_tier("recent_inactive")
            else:
                op_mult = _opacity_tier("prior_inactive")

            rgb = str(stz.get("rgb", "255, 215, 0"))
            border_op = float(stz.get("border_opacity_active", 0.9)) * op_mult
            confirm_rgb = str(stz.get("confirm_line_rgb", "101, 67, 33"))
            confirm_op = float(stz.get("confirm_opacity_active", 0.9)) * op_mult
            border_w = int(stz.get("border_line_width", 2))
            confirm_w = int(stz.get("confirm_line_width", 2))
            line_dash = stz.get("line_dash", "dash")

            border_color = _rgba_from_rgb(rgb, border_op)
            confirm_color = _rgba_from_rgb(confirm_rgb, confirm_op)

            x0 = _h1_to_m15_time(pd.to_datetime(poi.start_time, utc=True))
            x1 = _h1_to_m15_time(pd.to_datetime(poi.end_time, utc=True)) if poi.end_time else m15_t_last

            y0 = float(min(poi.top, poi.bottom))
            y1 = float(max(poi.top, poi.bottom))

            conf_idx = int(poi.meta.get("confirmed_idx", poi.ic_idx))
            conf_time = _h1_idx_to_m15_time(conf_idx) if conf_idx >= 0 else None

            fig.add_shape(type="rect", xref="x", yref="y",
                          x0=x0, x1=x1, y0=y0, y1=y1,
                          fillcolor=f"rgba({rgb}, 0.0)",
                          line=dict(width=border_w, dash=line_dash, color=border_color),
                          layer="below")

            if conf_time is not None:
                fig.add_shape(type="line", xref="x", yref="y",
                              x0=conf_time, x1=conf_time, y0=y0, y1=y1,
                              line=dict(color=confirm_color, width=confirm_w), layer="below")

            seg_times = dfx[COL_TIME][(dfx[COL_TIME] >= x0) & (dfx[COL_TIME] <= x1)]
            if len(seg_times) == 0:
                seg_times = pd.Series([x0, x1])

            versions = poi.meta.get("versions", [])
            versions_str = ", ".join(versions) if versions else "none"
            poi_cd = [[
                side, poi.meta.get("structure_id", -1), poi.meta.get("struct_direction", 0),
                poi.ic_idx, conf_idx, poi.meta.get("cycle_id", 0),
                versions_str, y1, y0, zone_status,
            ]] * len(seg_times)

            h1_hover_line = {"width": 6, "color": "rgba(0,0,0,0)"}
            for yval in (y1, y0):
                fig.add_trace(go.Scatter(
                    x=seg_times, y=[yval] * len(seg_times),
                    mode="lines", name="H1 POI zone (overlay)", showlegend=False,
                    line=h1_hover_line, line_shape="hv",
                    hovertemplate=(
                        "TF=1H<br>"
                        "<b>POI Zone</b><br>"
                        "side=%{customdata[0]}<br>"
                        "structure_id=%{customdata[1]}<br>"
                        "struct_direction=%{customdata[2]}<br>"
                        "ic_idx=%{customdata[3]}<br>"
                        "confirmed_idx=%{customdata[4]}<br>"
                        "cycle_id=%{customdata[5]}<br>"
                        "versions=%{customdata[6]}<br>"
                        "top=%{customdata[7]:.5f}<br>"
                        "bottom=%{customdata[8]:.5f}<br>"
                        "status=%{customdata[9]}"
                        "<extra></extra>"
                    ),
                    customdata=poi_cd,
                ))

    # --- H1 Prev BOS lines (dashed) ---
    h1_prev_bos = h1_df.attrs.get("prev_bos_lines", [])
    for line_info in h1_prev_bos:
        start_idx = line_info["start_idx"]
        end_idx = line_info["end_idx"]
        price = line_info["price"]
        t0 = _h1_idx_to_m15_time(start_idx)
        t1 = _h1_idx_to_m15_time(end_idx)
        if t0 is not None and t1 is not None:
            fig.add_trace(go.Scatter(
                x=[t0, t1], y=[price, price], mode="lines",
                line=_style("prev_bos_line.h1_overlay").get("line", {"width": 2, "color": "black", "dash": "dash"}),
                name=f"H1 Prev BOS (sid={line_info.get('prev_structure_id', '?')})",
                showlegend=False,
                hovertemplate=(
                    f"TF=1H<br>"
                    f"Prev BOS Line<br>"
                    f"Price: {price:.5f}<br>"
                    f"From idx: {start_idx}<br>"
                    f"To idx: {end_idx}<br>"
                    f"<extra></extra>"
                ),
            ))


# ---------------------------------------------------------------------------
# JS injection for volume auto-rescale
# ---------------------------------------------------------------------------

def _inject_volume_autoscale_js(html_path: Path):
    js = """
<script>
(function initVolumeAutoscale() {
    var gd = document.querySelector('.plotly-graph-div');
    if (!gd || !gd._fullData) {
        setTimeout(initVolumeAutoscale, 200);
        return;
    }
    var volumeTraceIdx = -1;
    for (var i = 0; i < gd._fullData.length; i++) {
        if (gd._fullData[i].type === 'bar' && gd._fullData[i].yaxis === 'y2') {
            volumeTraceIdx = i;
            break;
        }
    }
    if (volumeTraceIdx === -1) return;
    var volumeX = gd._fullData[volumeTraceIdx].x;
    var volumeY = gd._fullData[volumeTraceIdx].y;
    function rescaleVolumeY() {
        var xRange = gd.layout.xaxis.range;
        if (!xRange) return;
        var x0 = new Date(xRange[0]).getTime();
        var x1 = new Date(xRange[1]).getTime();
        var maxVol = 0;
        for (var i = 0; i < volumeX.length; i++) {
            var t = new Date(volumeX[i]).getTime();
            if (t >= x0 && t <= x1) {
                if (volumeY[i] > maxVol) maxVol = volumeY[i];
            }
        }
        if (maxVol > 0) {
            Plotly.relayout(gd, {'yaxis2.range': [0, maxVol * 1.1]});
        }
    }
    gd.on('plotly_relayout', function(eventData) {
        if (eventData['xaxis.range[0]'] !== undefined ||
            eventData['xaxis.range'] !== undefined ||
            eventData['xaxis.autorange'] !== undefined) {
            setTimeout(rescaleVolumeY, 100);
        }
    });
    rescaleVolumeY();
})();
</script>
"""
    with open(html_path, "a") as f:
        f.write(js)
