from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from engine_v2.common.types import COL_C, COL_H, COL_L, COL_O, COL_TIME
from engine_v2.charting.style_registry import STYLE
from engine_v2.common.types import PatternStatus


def _rgba_from_rgb(rgb: str, opacity: float) -> str:
    return f"rgba({rgb}, {opacity})"


def _get_reversal_confirmed_by_sid(structure_events: list) -> dict:
    """
    Get reversal confirmed idx per structure_id from STATE_CHANGED events.
    Returns dict: {structure_id: last_reversal_idx}

    This is more reliable than df columns because market_state gets overwritten
    by subsequent structures, but events are preserved.
    """
    rev_by_sid = {}
    for ev in structure_events:
        if getattr(ev, "type", None) != "STATE_CHANGED":
            continue
        if ev.meta.get("to") != "reversal":
            continue
        sid = ev.meta.get("structure_id")
        if sid is None:
            continue
        sid = int(sid)
        idx = int(ev.idx)
        # Keep the MAX idx for each structure_id (reversal confirmed = last reversal candle)
        if sid not in rev_by_sid or idx > rev_by_sid[sid]:
            rev_by_sid[sid] = idx
    return rev_by_sid

def _zone_style(side: str) -> dict:
    # side is "buy" or "sell"
    key = f"zone.kl.{side}"
    return STYLE.get(key, {})

def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _style(key: str) -> dict:
    return STYLE.get(key, {})


@dataclass(frozen=True)
class ChartExportPaths:
    html_path: Path
    png_path: Path


def export_chart_plotly(
    df: pd.DataFrame,
    *,
    title: str,
    structure_levels: Optional[list] = None,
    out_dir: str | Path = "artifacts/charts",
    basename: str = "chart",
    max_points: Optional[int] = None,
    cfg: Optional[dict] = None,
    idx_range: Optional[tuple[int, int]] = None,
) -> ChartExportPaths:
    """
    Export an interactive HTML + PNG candlestick chart with basic overlays.

    Expected columns (minimum):
      - time, o, h, l, c

    Optional overlays (enabled via cfg and if columns/data present):
      - candle types (e.g., candle_type markers)
      - patterns (pinbar_dir, engulfing, star, etc.)
      - structure levels (BOS/CTS lines)  [can be disabled by default]
      - zones (rectangles)               [future]

    Notes
    -----
    - OHLC is always rendered.
    - Everything else should be toggleable via cfg.
    """

    # ---------- Chart defaults (safe fallback) ----------
    CHART_DEFAULTS = {
        "show_ohlc": True,  # always True; kept for symmetry
        "candle_types": {
            "pinbar": False,
            "maru": False,
            "normal": False,
            "big_maru": False,
            "big_normal": False,
        },
        "patterns": {
            "engulfing": False,
            "star": False,
            # Week 4+ patterns:
            "continuous": False,
            "double_maru": False,
            "one_maru_continuous": False,
            "one_maru_opposite": False,
        },
        "struct_state": {
            "labels": True,
        },
        "range_visual": {
            "rectangles": True,
        },
        "structure": {
            "levels": False,  # placeholder/noisy -> OFF by default
            "labels": False,  # text labels are very noisy; OFF by default
        },
        "zones": {
            "KL": False,
            "OB": False,
        },
    }


    cfg = _deep_merge(CHART_DEFAULTS, cfg or {})
    candle_cfg = cfg.get("candle_types", {}) or {} 
    pat_cfg = cfg.get("patterns", {}) or {}
    state_cfg = cfg.get("struct_state", {}) or {}
    range_vis_cfg = cfg.get("range_visual", {}) or {}
    struct_cfg = cfg.get("structure", {}) or {}
    zone_cfg = cfg.get("zones", {}) or {}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfx = df.copy()

    # Optional index-range slicing (for zoomed debug exports)
    if idx_range is not None:
        i0, i1 = idx_range
        if i0 > i1:
            i0, i1 = i1, i0
        # clamp to available index range
        i0 = max(int(i0), int(dfx.index.min()))
        i1 = min(int(i1), int(dfx.index.max()))
        dfx = dfx.loc[i0:i1].copy()

    if max_points is not None and len(dfx) > max_points:
        dfx = dfx.iloc[-max_points:].copy()

    # Ensure datetime
    dfx[COL_TIME] = pd.to_datetime(dfx[COL_TIME], utc=True)

    fig = go.Figure()

    # Offset to keep markers away from candle wicks
    wick_offset = (dfx[COL_H] - dfx[COL_L]) * 0.15

    # Candles
    candle_idx = dfx.index.to_numpy()

    range_break_frac = (
        dfx["range_break_frac"].astype(float)
        if "range_break_frac" in dfx.columns
        else pd.Series([float("nan")] * len(dfx), index=dfx.index)
    )

    def _col_or_default(name, default):
        return dfx[name] if name in dfx.columns else pd.Series([default] * len(dfx), index=dfx.index)

    cycle_stage = _col_or_default("cycle_stage", "")
    cts_phase_debug = _col_or_default("cts_phase_debug", "")
    cts_cycle_id = _col_or_default("cts_cycle_id", 0).astype(int)

    bos_th = _col_or_default("bos_threshold", float("nan")).astype(float)
    cts_th = _col_or_default("cts_threshold", float("nan")).astype(float)

    rv_watch = _col_or_default("reversal_watch_active", 0)
    rv_bos_frozen = _col_or_default("reversal_bos_th_frozen", float("nan")).astype(float)
    rv_extreme = _col_or_default("reversal_watch_extreme", float("nan")).astype(float)

    customdata = list(zip(
        candle_idx,
        dfx["mid_price"].astype(float),
        dfx["body_pct"].astype(float),
        dfx["candle_type"].astype(str),
        dfx["body_len"].astype(float),
        dfx["candle_len"].astype(float),
        range_break_frac,
        cts_cycle_id,
        cts_phase_debug.astype(str),
        cycle_stage.astype(str),
        cts_th,
        bos_th,
        rv_watch,
        rv_bos_frozen,
        rv_extreme,
    ))


    fig.add_trace(
        go.Candlestick(
            x=dfx[COL_TIME],
            open=dfx[COL_O],
            high=dfx[COL_H],
            low=dfx[COL_L],
            close=dfx[COL_C],
            name="OHLC",
            showlegend=False,
            customdata=customdata,
            hovertemplate=(
                "idx=%{customdata[0]}<br>"
                "time=%{x}<br>"
                "O=%{open}<br>"
                "H=%{high}<br>"
                "L=%{low}<br>"
                "C=%{close}<br>"
                "candle_type=%{customdata[3]}<br>"
                "body_len=%{customdata[4]:.5f}<br>"
                "candle_len=%{customdata[5]:.5f}<br>"
                "body_pct=%{customdata[2]:.2%}<br>"
                "mid_price=%{customdata[1]:.5f}<br>"
                "range_break_frac=%{customdata[6]:.2%}<br>"
                "cts_cycle_id=%{customdata[7]}<br>"
                "cts_phase=%{customdata[8]}<br>"
                "cycle_stage=%{customdata[9]}<br>"
                "cts_th=%{customdata[10]:.5f}<br>"
                "bos_th=%{customdata[11]:.5f}<br>"
                "rv_watch=%{customdata[12]}<br>"
                "rv_bos_frozen=%{customdata[13]:.5f}<br>"
                "rv_extreme=%{customdata[14]:.5f}"
                "<extra></extra>"
            ),
        )
    )

    # Candle types
    if "candle_type" in dfx.columns:
        for ctype, enabled in candle_cfg.items():
            if not enabled:
                continue
            sub = dfx[dfx["candle_type"] == ctype]
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub[COL_TIME],
                    y=sub[COL_H],  # or choose a consistent anchor per type
                    mode="markers",
                    name=f"candle:{ctype}",
                    **_style(f"candle.{ctype}")  # defaults to {}
                )
            )

    # Engulfing markers
    if pat_cfg.get("engulfing", False) and "engulfing" in dfx.columns:
        bull = dfx[dfx["engulfing"] == 1]
        bear = dfx[dfx["engulfing"] == -1]
        if not bull.empty:
            fig.add_trace(
                go.Scatter(
                    x=bull[COL_TIME],
                    y=bull[COL_L],
                    mode="markers",
                    name="engulfing +1",
                    **_style("pattern.engulfing.up"),
                )
            )
        if not bear.empty:
            fig.add_trace(
                go.Scatter(
                    x=bear[COL_TIME],
                    y=bear[COL_H],
                    mode="markers",
                    name="engulfing -1",
                    **_style("pattern.engulfing.down"),
                )
            )


    # Star markers
    if pat_cfg.get("star", False) and "star" in dfx.columns:
        bull = dfx[dfx["star"] == 1]
        bear = dfx[dfx["star"] == -1]
        if not bull.empty:
            fig.add_trace(
                go.Scatter(
                    x=bull[COL_TIME],
                    y=bull[COL_L],
                    mode="markers",
                    name="star +1",
                    **_style("pattern.star.up"),
                )
            )
        if not bear.empty:
            fig.add_trace(
                go.Scatter(
                    x=bear[COL_TIME],
                    y=bear[COL_H],
                    mode="markers",
                    name="star -1",
                    **_style("pattern.star.down"),
                )
            )


    # -------------------------------------------------
    # Week 4 Structure-pattern markers (triangles)
    # -------------------------------------------------
    if "pat" in dfx.columns and "pat_dir" in dfx.columns and "pat_status" in dfx.columns:
        enabled_names = {k for k, v in pat_cfg.items() if v is True}

        sub = dfx[
            (dfx["pat"] != "")
            & (dfx["pat"].isin(enabled_names))
            & (dfx["pat_status"].isin([PatternStatus.SUCCESS.value, PatternStatus.CONFIRMED.value]))
        ]

        if not sub.empty:
            for status in [PatternStatus.SUCCESS.value, PatternStatus.CONFIRMED.value]:
                up = sub[(sub["pat_status"] == status) & (sub["pat_dir"] == 1)]
                dn = sub[(sub["pat_status"] == status) & (sub["pat_dir"] == -1)]

                # +1 => above high
                if not up.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=up[COL_TIME],
                            y=up[COL_H] + wick_offset.loc[up.index],
                            mode="markers",
                            name=f"struct:{status}:+1",
                            customdata=up[
                                ["pat", "pat_dir", "pat_status", "pat_start_idx", "pat_end_idx", "pat_confirm_idx"]
                            ].values,
                            hovertemplate=(
                                "pat=%{customdata[0]}<br>"
                                "dir=%{customdata[1]}<br>"
                                "status=%{customdata[2]}<br>"
                                "start=%{customdata[3]} end=%{customdata[4]} conf=%{customdata[5]}"
                                "<extra></extra>"
                            ),
                            **_style("structure.success.up" if status == PatternStatus.SUCCESS.value else "structure.confirmed.up"),
                        )
                    )

                # -1 => below low
                if not dn.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=dn[COL_TIME],
                            y=dn[COL_L] - wick_offset.loc[dn.index],
                            mode="markers",
                            name=f"struct:{status}:-1",
                            customdata=dn[
                                ["pat", "pat_dir", "pat_status", "pat_start_idx", "pat_end_idx", "pat_confirm_idx"]
                            ].values,
                            hovertemplate=(
                                "pat=%{customdata[0]}<br>"
                                "dir=%{customdata[1]}<br>"
                                "status=%{customdata[2]}<br>"
                                "start=%{customdata[3]} end=%{customdata[4]} conf=%{customdata[5]}"
                                "<extra></extra>"
                            ),
                            **_style("structure.success.down" if status == PatternStatus.SUCCESS.value else "structure.confirmed.down"),
                        )
                    )

    # -------------------------------------------------
    # Week 4 Day 3: Range markers (orange dots, mid-candle)
    # -------------------------------------------------
    if "is_range" in dfx.columns and dfx["is_range"].any():
        range_candles = dfx[dfx["is_range"] == 1].copy()

        # place in middle of the candle (mid_price if present, otherwise (H+L)/2)
        if "mid_price" in dfx.columns:
            y_mid = range_candles["mid_price"]
        else:
            y_mid = (range_candles[COL_H] + range_candles[COL_L]) / 2.0

        # optional debug fields if present
        for col in ["is_range_confirm_idx", "is_range_lag"]:
            if col not in range_candles.columns:
                range_candles[col] = -1

        fig.add_trace(
            go.Scatter(
                x=range_candles[COL_TIME],
                y=y_mid,
                mode="markers",
                name="range:candle",
                customdata=list(zip(
                    range_candles.index.to_numpy(),
                    range_candles["is_range_confirm_idx"].astype(int),
                    range_candles["is_range_lag"].astype(int),
                    range_candles[COL_L].astype(float),
                    range_candles[COL_H].astype(float),
                )),
                hovertemplate=(
                    "idx=%{customdata[0]}<br>"
                    "confirm_idx=%{customdata[1]}<br>"
                    "lag=%{customdata[2]}<br>"
                    "low=%{customdata[3]}<br>"
                    "high=%{customdata[4]}"
                    "<extra></extra>"
                ),
                **_style("range.candle"),
            )
        )

    # -------------------------------------------------
    # Week 5 Phase 1A: Breakout apply labels ("bo") from STATE_CHANGED events
    # Use STATE_CHANGED events (survives structure overwrites)
    # Position by struct_direction: +1 -> above candle, -1 -> below candle
    # (opposite of pb/pr/rv which are counter-trend; bo is with-trend)
    # -------------------------------------------------
    if state_cfg.get("labels", False):
        structure_events = dfx.attrs.get("structure_events", [])
        bo_events = [
            ev for ev in structure_events
            if getattr(ev, "type", None) == "STATE_CHANGED"
            and ev.meta.get("to") == "breakout"
            and ev.idx in dfx.index
        ]

        if bo_events:
            x_vals = []
            y_vals = []
            customdata = []

            for ev in bo_events:
                idx = ev.idx
                sd = int(ev.meta.get("struct_direction", 1))
                sid = int(ev.meta.get("structure_id", 0))

                row = dfx.loc[idx]
                wo = float(wick_offset.loc[idx]) if idx in wick_offset.index else 0.0

                # Position by struct_direction (opposite of pb/pr/rv):
                # +1 (bullish): bo is with-trend move UP, so label ABOVE candle
                # -1 (bearish): bo is with-trend move DOWN, so label BELOW candle
                if sd == 1:
                    y = float(row[COL_H]) + wo * 2.0
                else:
                    y = float(row[COL_L]) - wo * 2.0

                x_vals.append(row[COL_TIME])
                y_vals.append(y)
                customdata.append((idx, "breakout", sid, sd))

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="text",
                    text=["bo"] * len(bo_events),
                    name="bo apply",
                    textposition="middle center",
                    showlegend=False,
                    hovertemplate=(
                        "idx=%{customdata[0]}<br>"
                        "state=%{customdata[1]}<br>"
                        "sid=%{customdata[2]}<br>"
                        "struct_direction=%{customdata[3]}"
                        "<extra></extra>"
                    ),
                    customdata=customdata,
                )
            )

    # -------------------------------------------------
    # Week 5 Phase 1A: Market state change labels (pb/pr/rv)
    # Use STATE_CHANGED events (survives structure overwrites)
    # Position by struct_direction: +1 -> below candle, -1 -> above candle
    # -------------------------------------------------
    if state_cfg.get("labels", False):
        label_map = {
            "pullback": "pb",
            "pullback_range": "pr",
            "reversal": "rv",
        }

        # Get STATE_CHANGED events
        structure_events = dfx.attrs.get("structure_events", [])
        state_events = [
            ev for ev in structure_events
            if getattr(ev, "type", None) == "STATE_CHANGED"
            and ev.meta.get("to") in label_map
            and ev.idx in dfx.index
        ]

        if state_events:
            # Build lists for plotting
            x_vals = []
            y_vals = []
            text_vals = []
            customdata = []

            for ev in state_events:
                idx = ev.idx
                to_state = ev.meta.get("to", "")
                sd = int(ev.meta.get("struct_direction", 1))
                sid = int(ev.meta.get("structure_id", 0))

                # Get candle data
                row = dfx.loc[idx]
                wo = float(wick_offset.loc[idx]) if idx in wick_offset.index else 0.0

                # Position by struct_direction:
                # +1 (bullish): pb/pr/rv are counter-trend moves DOWN, so label BELOW candle
                # -1 (bearish): pb/pr/rv are counter-trend moves UP, so label ABOVE candle
                if sd == 1:
                    y = float(row[COL_L]) - wo * 2.0
                else:
                    y = float(row[COL_H]) + wo * 2.0

                x_vals.append(row[COL_TIME])
                y_vals.append(y)
                text_vals.append(label_map.get(to_state, to_state))
                customdata.append((idx, to_state, sid, sd))

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="text",
                    text=text_vals,
                    name="state labels",
                    textposition="middle center",
                    showlegend=False,
                    hovertemplate=(
                        "idx=%{customdata[0]}<br>"
                        "state=%{customdata[1]}<br>"
                        "sid=%{customdata[2]}<br>"
                        "struct_direction=%{customdata[3]}"
                        "<extra></extra>"
                    ),
                    customdata=customdata,
                )
            )

    # -------------------------------------------------
    # Week 5 Phase 1B: Active range rectangles (event-based, multi-structure support)
    # -------------------------------------------------
    # Uses RANGE_STARTED, RANGE_UPDATED, RANGE_RESET events to build segments per structure_id.
    # Allows overlapping rectangles from multiple structures with opacity differentiation.
    if range_vis_cfg.get("rectangles", False):
        time_by_idx = {int(i): t for i, t in zip(dfx.index.to_numpy(), dfx[COL_TIME])}
        idx_set = set(map(int, dfx.index.to_numpy()))
        last_visible_idx = max(idx_set) if idx_set else 0

        structure_events = dfx.attrs.get("structure_events", [])

        # Filter range events
        range_started = [ev for ev in structure_events if getattr(ev, "type", None) == "RANGE_STARTED"]
        range_updated = [ev for ev in structure_events if getattr(ev, "type", None) == "RANGE_UPDATED"]
        range_reset = [ev for ev in structure_events if getattr(ev, "type", None) == "RANGE_RESET"]

        # Find most recent structure_id for opacity
        all_sids = set()
        for ev in range_started + range_updated + range_reset:
            sid = ev.meta.get("structure_id")
            if sid is not None:
                all_sids.add(int(sid))
        most_recent_sid = max(all_sids) if all_sids else 0

        # Collect all range events, sort by (idx, type priority)
        # Priority: RANGE_RESET=0, RANGE_STARTED=1, RANGE_UPDATED=2
        # RANGE_RESET must come before RANGE_STARTED at same idx so closing previous range
        # happens before opening new range (e.g., RANGE_RESET at 748, RANGE_STARTED with start_idx=748)
        # For RANGE_STARTED, use start_idx (from meta) for sorting so it's processed
        # before any RANGE_UPDATED events that fall within its [start_idx, confirm_idx] window
        type_priority = {"RANGE_RESET": 0, "RANGE_STARTED": 1, "RANGE_UPDATED": 2}
        all_range_events = []
        for ev in range_started + range_updated + range_reset:
            all_range_events.append(ev)

        def _range_event_sort_key(e):
            if e.type == "RANGE_STARTED":
                # Use start_idx so RANGE_STARTED is processed before RANGE_UPDATED
                # events that occur between start_idx and confirm_idx
                return (e.meta.get("start_idx", e.idx), type_priority.get(e.type, 99))
            return (e.idx, type_priority.get(e.type, 99))

        all_range_events.sort(key=_range_event_sort_key)

        # Build segments per structure_id
        # Segment: (start_idx, end_idx, hi, lo, structure_id, struct_direction)
        range_segments = []

        # Track open ranges per structure_id
        # open_ranges[sid] = {"start_idx": int, "hi": float, "lo": float, "struct_direction": int}
        open_ranges = {}

        for ev in all_range_events:
            sid = int(ev.meta.get("structure_id", 0))
            sd = int(ev.meta.get("struct_direction", 1))
            idx = int(ev.idx)

            if ev.type == "RANGE_STARTED":
                # Close any existing open range for this sid first
                if sid in open_ranges:
                    seg = open_ranges[sid]
                    range_segments.append((
                        seg["start_idx"], idx, seg["hi"], seg["lo"], sid, seg["struct_direction"]
                    ))
                # Start new range
                open_ranges[sid] = {
                    "start_idx": ev.meta.get("start_idx", idx),
                    "hi": float(ev.meta.get("hi", 0)),
                    "lo": float(ev.meta.get("lo", 0)),
                    "struct_direction": sd,
                }

            elif ev.type == "RANGE_UPDATED":
                if sid in open_ranges:
                    seg = open_ranges[sid]
                    # Close current segment at this idx
                    range_segments.append((
                        seg["start_idx"], idx, seg["hi"], seg["lo"], sid, seg["struct_direction"]
                    ))
                    # Start new segment with updated bounds
                    open_ranges[sid] = {
                        "start_idx": idx,
                        "hi": float(ev.meta.get("hi", seg["hi"])),
                        "lo": float(ev.meta.get("lo", seg["lo"])),
                        "struct_direction": sd,
                    }

            elif ev.type == "RANGE_RESET":
                if sid in open_ranges:
                    seg = open_ranges[sid]
                    # Close segment at this idx
                    range_segments.append((
                        seg["start_idx"], idx, seg["hi"], seg["lo"], sid, seg["struct_direction"]
                    ))
                    del open_ranges[sid]

        # Close any ranges still open - at reversal confirmed if exists, else end of window
        # Use events to find reversal confirmed idx per structure_id
        # (df columns get overwritten by subsequent structures, events are preserved)
        rev_confirmed_by_sid = _get_reversal_confirmed_by_sid(structure_events)

        for sid, seg in open_ranges.items():
            # Use reversal confirmed idx if exists, else end of visible window
            end_idx = rev_confirmed_by_sid.get(sid, last_visible_idx)
            range_segments.append((
                seg["start_idx"], end_idx, seg["hi"], seg["lo"], sid, seg["struct_direction"]
            ))

        # Get base style
        range_rect_style = _style("range.rect")

        # Render segments
        shapes_added = 0
        MAX_SHAPES = 500

        for start_idx, end_idx, hi, lo, sid, sd in range_segments:
            if shapes_added >= MAX_SHAPES:
                break

            x0 = time_by_idx.get(int(start_idx))
            x1 = time_by_idx.get(int(end_idx))

            if x0 is None or x1 is None:
                # Clamp to visible window
                if int(start_idx) < min(idx_set):
                    x0 = time_by_idx.get(min(idx_set))
                if int(end_idx) > max(idx_set):
                    x1 = time_by_idx.get(max(idx_set))
                if x0 is None or x1 is None:
                    continue

            y0 = min(lo, hi)
            y1 = max(lo, hi)

            # Opacity: most recent structure_id = full, prior = 50%
            if sid == most_recent_sid:
                opacity_mult = 1.0
            else:
                opacity_mult = 0.5

            # Apply opacity multiplier
            style_copy = dict(range_rect_style)
            base_opacity = float(style_copy.get("opacity", 0.15))
            style_copy["opacity"] = base_opacity * opacity_mult

            fig.add_shape(
                type="rect",
                xref="x",
                yref="y",
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
                **style_copy,
            )
            shapes_added += 1

    # -------------------------------------------------
    # Range hover lines (event-based, multi-structure support)
    # Shapes (rectangles) don't support hover, so we add
    # thin line traces that carry hover data per structure_id.
    # -------------------------------------------------
    if range_vis_cfg.get("rectangles", False):
        structure_events = dfx.attrs.get("structure_events", [])
        time_by_idx = {int(i): t for i, t in zip(dfx.index.to_numpy(), dfx[COL_TIME])}
        idx_set = set(map(int, dfx.index.to_numpy()))
        last_visible_idx = max(idx_set) if idx_set else 0

        # Build range spans per structure_id: list of (start_idx, end_idx, hi_values, lo_values, sid, sd)
        # where hi_values and lo_values are lists of (idx, value) for step changes
        range_started = [ev for ev in structure_events if getattr(ev, "type", None) == "RANGE_STARTED"]
        range_updated = [ev for ev in structure_events if getattr(ev, "type", None) == "RANGE_UPDATED"]
        range_reset = [ev for ev in structure_events if getattr(ev, "type", None) == "RANGE_RESET"]

        # Find most recent structure_id
        all_sids = set()
        for ev in range_started + range_updated + range_reset:
            sid = ev.meta.get("structure_id")
            if sid is not None:
                all_sids.add(int(sid))
        most_recent_sid = max(all_sids) if all_sids else 0

        type_priority = {"RANGE_RESET": 0, "RANGE_STARTED": 1, "RANGE_UPDATED": 2}
        all_range_events = range_started + range_updated + range_reset

        def _range_event_sort_key_hover(e):
            if e.type == "RANGE_STARTED":
                return (e.meta.get("start_idx", e.idx), type_priority.get(e.type, 99))
            return (e.idx, type_priority.get(e.type, 99))

        all_range_events.sort(key=_range_event_sort_key_hover)

        # Build reversal confirmed idx per structure_id (ranges end at reversal)
        # Use events instead of df columns (df columns get overwritten by subsequent structures)
        rev_confirmed_by_sid_hover = _get_reversal_confirmed_by_sid(structure_events)

        # Build hover data per structure_id
        # hover_data[sid] = {"points": [(idx, time, hi, lo, last_update_idx)...]}
        hover_data = {}
        open_ranges = {}

        for ev in all_range_events:
            sid = int(ev.meta.get("structure_id", 0))
            sd = int(ev.meta.get("struct_direction", 1))
            idx = int(ev.idx)

            if ev.type == "RANGE_STARTED":
                start_idx = ev.meta.get("start_idx", idx)
                hi = float(ev.meta.get("hi", 0))
                lo = float(ev.meta.get("lo", 0))
                open_ranges[sid] = {
                    "start_idx": start_idx,
                    "hi": hi,
                    "lo": lo,
                    "last_update_idx": start_idx,
                    "struct_direction": sd,
                }
                if sid not in hover_data:
                    hover_data[sid] = {"points_hi": [], "points_lo": [], "struct_direction": sd}

            elif ev.type == "RANGE_UPDATED":
                if sid in open_ranges:
                    open_ranges[sid]["hi"] = float(ev.meta.get("hi", open_ranges[sid]["hi"]))
                    open_ranges[sid]["lo"] = float(ev.meta.get("lo", open_ranges[sid]["lo"]))
                    open_ranges[sid]["last_update_idx"] = idx

            elif ev.type == "RANGE_RESET":
                if sid in open_ranges:
                    del open_ranges[sid]

        # Generate hover line points by iterating through visible candles
        # For each active range at each candle, add a point
        open_ranges = {}  # Reset for second pass

        for ev in all_range_events:
            sid = int(ev.meta.get("structure_id", 0))
            sd = int(ev.meta.get("struct_direction", 1))
            idx = int(ev.idx)

            if ev.type == "RANGE_STARTED":
                start_idx = ev.meta.get("start_idx", idx)
                hi = float(ev.meta.get("hi", 0))
                lo = float(ev.meta.get("lo", 0))
                open_ranges[sid] = {
                    "start_idx": start_idx,
                    "end_idx": None,
                    "hi": hi,
                    "lo": lo,
                    "last_update_idx": start_idx,
                    "struct_direction": sd,
                }
                if sid not in hover_data:
                    hover_data[sid] = {"points_hi": [], "points_lo": [], "struct_direction": sd}

            elif ev.type == "RANGE_UPDATED":
                if sid in open_ranges:
                    open_ranges[sid]["hi"] = float(ev.meta.get("hi", open_ranges[sid]["hi"]))
                    open_ranges[sid]["lo"] = float(ev.meta.get("lo", open_ranges[sid]["lo"]))
                    open_ranges[sid]["last_update_idx"] = idx

            elif ev.type == "RANGE_RESET":
                if sid in open_ranges:
                    open_ranges[sid]["end_idx"] = idx
                    # Finalize this range
                    del open_ranges[sid]

        # For simplicity, build hover lines per structure_id by iterating through
        # visible candles and checking which ranges are active at each candle
        # Reset and rebuild properly
        open_ranges = {}
        range_by_candle = {}  # {idx: {sid: {hi, lo, start_idx, last_update_idx, sd}}}

        for ev in all_range_events:
            sid = int(ev.meta.get("structure_id", 0))
            sd = int(ev.meta.get("struct_direction", 1))
            idx = int(ev.idx)

            if ev.type == "RANGE_STARTED":
                start_idx = int(ev.meta.get("start_idx", idx))
                open_ranges[sid] = {
                    "start_idx": start_idx,
                    "hi": float(ev.meta.get("hi", 0)),
                    "lo": float(ev.meta.get("lo", 0)),
                    "last_update_idx": start_idx,
                    "struct_direction": sd,
                }
            elif ev.type == "RANGE_UPDATED":
                if sid in open_ranges:
                    open_ranges[sid]["hi"] = float(ev.meta.get("hi", open_ranges[sid]["hi"]))
                    open_ranges[sid]["lo"] = float(ev.meta.get("lo", open_ranges[sid]["lo"]))
                    open_ranges[sid]["last_update_idx"] = idx
            elif ev.type == "RANGE_RESET":
                if sid in open_ranges:
                    del open_ranges[sid]

            # Snapshot active ranges at this idx
            if idx in idx_set:
                range_by_candle[idx] = {s: dict(r) for s, r in open_ranges.items()}

        # Also handle final state for candles after last event
        for idx in sorted(idx_set):
            if idx not in range_by_candle:
                # Use last known state
                prev_idxs = [i for i in range_by_candle.keys() if i < idx]
                if prev_idxs:
                    range_by_candle[idx] = range_by_candle[max(prev_idxs)]
                else:
                    range_by_candle[idx] = {}

        # Build hover traces per structure_id
        for sid in all_sids:
            times_hi = []
            vals_hi = []
            customdata_hi = []
            times_lo = []
            vals_lo = []
            customdata_lo = []

            # Get reversal confirmed idx for this structure_id (ranges end there)
            rev_end_idx = rev_confirmed_by_sid_hover.get(sid, last_visible_idx + 1)

            for idx in sorted(idx_set):
                # Skip candles after reversal confirmed for this structure
                if idx > rev_end_idx:
                    continue
                if idx in range_by_candle and sid in range_by_candle[idx]:
                    rng = range_by_candle[idx][sid]
                    t = time_by_idx.get(idx)
                    if t is not None:
                        times_hi.append(t)
                        vals_hi.append(rng["hi"])
                        customdata_hi.append((idx, rng["start_idx"], rng["hi"], rng["lo"], rng["last_update_idx"], sid))
                        times_lo.append(t)
                        vals_lo.append(rng["lo"])
                        customdata_lo.append((idx, rng["start_idx"], rng["hi"], rng["lo"], rng["last_update_idx"], sid))

            if not times_hi:
                continue

            # Opacity for hover lines
            opacity_mult = 1.0 if sid == most_recent_sid else 0.5

            # Top line (range_hi)
            fig.add_trace(
                go.Scatter(
                    x=times_hi,
                    y=vals_hi,
                    mode="lines",
                    name=f"range:top:sid{sid}",
                    showlegend=False,
                    line=dict(width=2, color="rgba(0,0,0,0)"),
                    line_shape="hv",
                    hovertemplate=(
                        "idx=%{customdata[0]}<br>"
                        "range_start_idx=%{customdata[1]}<br>"
                        "range_hi=%{customdata[2]:.5f}<br>"
                        "range_lo=%{customdata[3]:.5f}<br>"
                        "last_update_idx=%{customdata[4]}<br>"
                        "structure_id=%{customdata[5]}"
                        "<extra></extra>"
                    ),
                    customdata=customdata_hi,
                )
            )

            # Bottom line (range_lo)
            fig.add_trace(
                go.Scatter(
                    x=times_lo,
                    y=vals_lo,
                    mode="lines",
                    name=f"range:bottom:sid{sid}",
                    showlegend=False,
                    line=dict(width=2, color="rgba(0,0,0,0)"),
                    line_shape="hv",
                    hovertemplate=(
                        "idx=%{customdata[0]}<br>"
                        "range_start_idx=%{customdata[1]}<br>"
                        "range_hi=%{customdata[2]:.5f}<br>"
                        "range_lo=%{customdata[3]:.5f}<br>"
                        "last_update_idx=%{customdata[4]}<br>"
                        "structure_id=%{customdata[5]}"
                        "<extra></extra>"
                    ),
                    customdata=customdata_lo,
                )
            )


    # -------------------------------------------------
    # Week 5: Structure overlays (confirmed BOS/CTS swing line)
    # -------------------------------------------------
    # Replace old "structure_levels horizontal lines" with:
    # - dots at each CONFIRMED CTS and CONFIRMED BOS
    # - straight lines connecting them in time order
    # - final line from last confirmed point to last candle close
    # Use events (survives structure overwrites) instead of df columns
    if struct_cfg.get("levels", False):
        time_by_idx = {int(ii): tt for ii, tt in zip(dfx.index.to_numpy(), dfx[COL_TIME])}
        structure_events = dfx.attrs.get("structure_events", [])

        # Get CTS_CONFIRMED and BOS_CONFIRMED events
        cts_events = [ev for ev in structure_events if getattr(ev, "type", None) == "CTS_CONFIRMED"]
        bos_events = [ev for ev in structure_events if getattr(ev, "type", None) == "BOS_CONFIRMED"]

        # Find most recent structure_id for opacity
        all_sids = set()
        for ev in cts_events + bos_events:
            all_sids.add(int(ev.meta.get("structure_id", 0)))
        most_recent_sid = max(all_sids) if all_sids else 0

        # Build points from events: (idx, time, price, kind, sid, cycle_id, sd, opacity, has_overlap)
        points = []

        # Group events by idx for overlap detection
        from collections import defaultdict
        cts_by_idx = defaultdict(list)
        bos_by_idx = defaultdict(list)

        for ev in cts_events:
            # CTS confirmed: use cts_anchor_idx if available, else event idx
            p_idx = int(ev.meta.get("cts_anchor_idx", ev.idx))
            if p_idx in time_by_idx:
                cts_by_idx[p_idx].append(ev)

        for ev in bos_events:
            p_idx = ev.idx
            if p_idx in time_by_idx:
                bos_by_idx[p_idx].append(ev)

        # Sort events at each idx by structure_id
        for idx in cts_by_idx:
            cts_by_idx[idx].sort(key=lambda e: int(e.meta.get("structure_id", 0)))
        for idx in bos_by_idx:
            bos_by_idx[idx].sort(key=lambda e: int(e.meta.get("structure_id", 0)))

        # Build CTS points - no time offset, track overlap for hover positioning
        # CTS_CONFIRMED has ev.price=None; look up cts_price from DataFrame at confirmation candle (ev.idx)
        pts_cts = []
        for idx, evs in sorted(cts_by_idx.items()):
            has_overlap = len(evs) > 1
            for ev in evs:
                sid = int(ev.meta.get("structure_id", 0))
                cycle = int(ev.meta.get("cycle_id", 0))
                sd = int(ev.meta.get("struct_direction", 0))

                # CTS price: look up from DataFrame at confirmation candle (ev.idx has cts_price set)
                if ev.price is not None:
                    price = float(ev.price)
                elif ev.idx in dfx.index and "cts_price" in dfx.columns:
                    price = float(dfx.loc[ev.idx, "cts_price"])
                else:
                    price = 0.0

                opacity = 1.0 if sid == most_recent_sid else 0.5
                x_time = time_by_idx[idx]

                pts_cts.append((idx, x_time, price, "CTS", sid, cycle, sd, opacity, has_overlap))
                points.append((idx, time_by_idx[idx], price, "CTS", sid, cycle, sd))

        # Build BOS points - no time offset, track overlap for hover positioning
        # BOS_CONFIRMED has ev.price set correctly
        pts_bos = []
        for idx, evs in sorted(bos_by_idx.items()):
            has_overlap = len(evs) > 1
            for ev in evs:
                sid = int(ev.meta.get("structure_id", 0))
                cycle = int(ev.meta.get("cycle_id", 0))
                sd = int(ev.meta.get("struct_direction", 0))
                price = float(ev.price) if ev.price is not None else 0.0
                opacity = 1.0 if sid == most_recent_sid else 0.5
                x_time = time_by_idx[idx]

                pts_bos.append((idx, x_time, price, "BOS", sid, cycle, sd, opacity, has_overlap))
                points.append((idx, time_by_idx[idx], price, "BOS", sid, cycle, sd))

        # sort by point index (time order)
        points.sort(key=lambda x: x[0])

        # Build reversal confirmed idx per structure_id (lines end at reversal)
        # Use events instead of df columns (df columns get overwritten by subsequent structures)
        rev_confirmed_by_sid_lines = _get_reversal_confirmed_by_sid(structure_events)

        if len(points) >= 1:
            # Group points by structure_id - don't connect across structures
            from collections import defaultdict
            points_by_sid = defaultdict(list)
            for p in points:
                sid = p[4]  # structure_id is at index 4
                points_by_sid[sid].append(p)

            # Draw separate line for each structure_id
            for sid in sorted(points_by_sid.keys()):
                sid_points = points_by_sid[sid]
                sid_points.sort(key=lambda x: x[0])  # sort by idx

                # Build line points from confirmed BOS/CTS points
                x_line = [p[1] for p in sid_points]
                y_line = [p[2] for p in sid_points]

                # Only extend line to last candle for the most recent structure
                # Prior structures end at their last confirmed BOS/CTS point (no extension to reversal)
                if sid == most_recent_sid:
                    end_time = dfx[COL_TIME].iloc[-1]
                    end_price = float(dfx[COL_C].iloc[-1])
                    x_line.append(end_time)
                    y_line.append(end_price)

                # Opacity: base from style, multiplier for prior structures
                line_style = _style("structure.swing_line").copy()
                base_opacity = float(line_style.get("opacity", 0.9))
                opacity_mult = 1.0 if sid == most_recent_sid else 0.5
                if "line" in line_style:
                    line_style["line"] = dict(line_style["line"])
                else:
                    line_style["line"] = {}
                line_style["opacity"] = base_opacity * opacity_mult

                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        name=f"Structure (CTS/BOS) sid={sid}",
                        hoverinfo="skip",
                        line_shape="linear",
                        showlegend=(sid == most_recent_sid),  # Only show legend for most recent
                        **line_style,
                    )
                )

            # Markers: confirmed CTS points (with opacity per structure)
            # pts_cts format: (idx, x_time, price, "CTS", sid, cycle, sd, opacity, has_overlap)
            if pts_cts:
                # Group by: (opacity, has_overlap, sid for overlap points)
                # Non-overlap points: group by opacity only
                # Overlap points: separate trace per sid with mirrored hover labels
                cts_full_no_overlap = [p for p in pts_cts if p[7] == 1.0 and not p[8]]
                cts_faded_no_overlap = [p for p in pts_cts if p[7] < 1.0 and not p[8]]
                cts_overlap_by_sid = defaultdict(list)
                for p in pts_cts:
                    if p[8]:  # has_overlap
                        cts_overlap_by_sid[p[4]].append(p)  # group by sid

                # Non-overlap traces (normal hover)
                for pts, op_label in [(cts_full_no_overlap, ""), (cts_faded_no_overlap, " (prior)")]:
                    if pts:
                        style = _style("structure.cts").copy()
                        if op_label:
                            if "marker" in style:
                                style["marker"] = dict(style["marker"])
                                style["marker"]["opacity"] = 0.5
                        fig.add_trace(
                            go.Scatter(
                                x=[p[1] for p in pts],
                                y=[p[2] for p in pts],
                                mode="markers",
                                name=f"CTS (confirmed){op_label}",
                                customdata=[[p[0], p[3], p[2], p[4], p[5], p[6]] for p in pts],
                                hovertemplate=(
                                    "idx=%{customdata[0]}<br>"
                                    "kind=%{customdata[1]}<br>"
                                    "price=%{customdata[2]:.5f}<br>"
                                    "sid=%{customdata[3]}<br>"
                                    "cycle_id=%{customdata[4]}<br>"
                                    "struct_direction=%{customdata[5]}"
                                    "<extra></extra>"
                                ),
                                **style,
                            )
                        )

                # Overlap traces: separate per sid with mirrored hover labels
                for sid in sorted(cts_overlap_by_sid.keys()):
                    pts = cts_overlap_by_sid[sid]
                    if pts:
                        style = _style("structure.cts").copy()
                        opacity = pts[0][7]  # all points for this sid have same opacity
                        if opacity < 1.0:
                            if "marker" in style:
                                style["marker"] = dict(style["marker"])
                                style["marker"]["opacity"] = 0.5
                        # Hover label: sid 0 (prior) on left, higher sid on right
                        xanchor = "right" if sid < most_recent_sid else "left"
                        op_label = " (prior)" if sid < most_recent_sid else ""
                        fig.add_trace(
                            go.Scatter(
                                x=[p[1] for p in pts],
                                y=[p[2] for p in pts],
                                mode="markers",
                                name=f"CTS (confirmed){op_label} [overlap]",
                                customdata=[[p[0], p[3], p[2], p[4], p[5], p[6]] for p in pts],
                                hovertemplate=(
                                    "idx=%{customdata[0]}<br>"
                                    "kind=%{customdata[1]}<br>"
                                    "price=%{customdata[2]:.5f}<br>"
                                    "sid=%{customdata[3]}<br>"
                                    "cycle_id=%{customdata[4]}<br>"
                                    "struct_direction=%{customdata[5]}"
                                    "<extra></extra>"
                                ),
                                hoverlabel=dict(xanchor=xanchor),
                                showlegend=False,
                                **style,
                            )
                        )

            # Markers: confirmed BOS points (with opacity per structure)
            # pts_bos format: (idx, x_time, price, "BOS", sid, cycle, sd, opacity, has_overlap)
            if pts_bos:
                # Group by: (opacity, has_overlap, sid for overlap points)
                bos_full_no_overlap = [p for p in pts_bos if p[7] == 1.0 and not p[8]]
                bos_faded_no_overlap = [p for p in pts_bos if p[7] < 1.0 and not p[8]]
                bos_overlap_by_sid = defaultdict(list)
                for p in pts_bos:
                    if p[8]:  # has_overlap
                        bos_overlap_by_sid[p[4]].append(p)  # group by sid

                # Non-overlap traces (normal hover)
                for pts, op_label in [(bos_full_no_overlap, ""), (bos_faded_no_overlap, " (prior)")]:
                    if pts:
                        style = _style("structure.bos").copy()
                        if op_label:
                            if "marker" in style:
                                style["marker"] = dict(style["marker"])
                                style["marker"]["opacity"] = 0.5
                        fig.add_trace(
                            go.Scatter(
                                x=[p[1] for p in pts],
                                y=[p[2] for p in pts],
                                mode="markers",
                                name=f"BOS (confirmed){op_label}",
                                customdata=[[p[0], p[3], p[2], p[4], p[5], p[6]] for p in pts],
                                hovertemplate=(
                                    "idx=%{customdata[0]}<br>"
                                    "kind=%{customdata[1]}<br>"
                                    "price=%{customdata[2]:.5f}<br>"
                                    "sid=%{customdata[3]}<br>"
                                    "cycle_id=%{customdata[4]}<br>"
                                    "struct_direction=%{customdata[5]}"
                                    "<extra></extra>"
                                ),
                                **style,
                            )
                        )

                # Overlap traces: separate per sid with mirrored hover labels
                for sid in sorted(bos_overlap_by_sid.keys()):
                    pts = bos_overlap_by_sid[sid]
                    if pts:
                        style = _style("structure.bos").copy()
                        opacity = pts[0][7]
                        if opacity < 1.0:
                            if "marker" in style:
                                style["marker"] = dict(style["marker"])
                                style["marker"]["opacity"] = 0.5
                        # Hover label: sid 0 (prior) on left, higher sid on right
                        xanchor = "right" if sid < most_recent_sid else "left"
                        op_label = " (prior)" if sid < most_recent_sid else ""
                        fig.add_trace(
                            go.Scatter(
                                x=[p[1] for p in pts],
                                y=[p[2] for p in pts],
                                mode="markers",
                                name=f"BOS (confirmed){op_label} [overlap]",
                                customdata=[[p[0], p[3], p[2], p[4], p[5], p[6]] for p in pts],
                                hovertemplate=(
                                    "idx=%{customdata[0]}<br>"
                                    "kind=%{customdata[1]}<br>"
                                    "price=%{customdata[2]:.5f}<br>"
                                    "sid=%{customdata[3]}<br>"
                                    "cycle_id=%{customdata[4]}<br>"
                                    "struct_direction=%{customdata[5]}"
                                    "<extra></extra>"
                                ),
                                hoverlabel=dict(xanchor=xanchor),
                                showlegend=False,
                                **style,
                            )
                        )

    # -------------------------------------------------
    # Reversal watch overlay (frozen BOS barrier)
    # -------------------------------------------------
    if all(c in dfx.columns for c in ["reversal_watch_active", "reversal_bos_th_frozen"]):
        watch_active = dfx["reversal_watch_active"].astype(bool)
        prev_active = watch_active.shift(1, fill_value=False)
        watch_y = dfx["reversal_bos_th_frozen"].where(watch_active, float("nan"))

        # -------------------------------------------------
        # Reversal candidate labels (rc)
        # Label EVERY candle that closes beyond bos_threshold
        # -------------------------------------------------
        # Reversal candidate markers: use REVERSAL_WATCH_START events from structure analysis
        # (events survive rewinds, unlike dataframe columns)
        # REVERSAL_WATCH_START marks all close-breaks; REVERSAL_CANDIDATE indicates a valid pattern
        # -------------------------------------------------
        structure_events = dfx.attrs.get("structure_events", [])
        reversal_watch_starts = [
            ev for ev in structure_events
            if getattr(ev, "type", None) == "REVERSAL_WATCH_START"
        ]
        reversal_candidates = [
            ev for ev in structure_events
            if getattr(ev, "type", None) == "REVERSAL_CANDIDATE"
        ]
        # Map anchor_idx to REVERSAL_CANDIDATE event (if pattern found)
        rc_by_anchor = {int(ev.meta.get("anchor_idx", ev.idx)): ev for ev in reversal_candidates}

        if reversal_watch_starts:
            # Group events by idx to detect overlaps (multiple structures at same candle)
            from collections import defaultdict
            events_by_idx = defaultdict(list)
            for ev in reversal_watch_starts:
                idx = int(ev.meta.get("anchor_idx", ev.idx))
                if idx in dfx.index:
                    events_by_idx[idx].append(ev)

            # Sort events at each idx by structure_id for consistent left/right ordering
            for idx in events_by_idx:
                events_by_idx[idx].sort(key=lambda e: int(e.meta.get("structure_id", 0)))

            # Build plot data with left/right offset for overlaps
            x_vals = []
            y_vals = []
            customdata = []

            # Time offset for left/right positioning (fraction of candle width)
            time_delta = pd.Timedelta(hours=0.3)  # Small offset for H1 timeframe

            for idx, evs in sorted(events_by_idx.items()):
                row = dfx.loc[idx]
                base_time = pd.to_datetime(row[COL_TIME])
                base_y = float(row[COL_C])
                has_overlap = len(evs) > 1

                for i, ev in enumerate(evs):
                    bos_frozen = float(ev.meta.get("bos_frozen", 0))
                    sid = int(ev.meta.get("structure_id", 0))
                    sd = int(ev.meta.get("struct_direction", 0))
                    rc_ev = rc_by_anchor.get(idx)
                    pattern = str(rc_ev.meta.get("pattern", "none")) if rc_ev else "none"

                    # Offset: first sid -> left, second sid -> right; no overlap -> center
                    if has_overlap:
                        if i == 0:
                            x_time = base_time - time_delta  # left
                        else:
                            x_time = base_time + time_delta  # right
                    else:
                        x_time = base_time  # center

                    x_vals.append(x_time)
                    y_vals.append(base_y)
                    customdata.append((idx, bos_frozen, base_y, pattern, sid, sd))

            if x_vals:
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="markers",
                        name="reversal candidate",
                        showlegend=True,
                        hovertemplate=(
                            "idx=%{customdata[0]}<br>"
                            "event=reversal_candidate<br>"
                            "bos_frozen=%{customdata[1]:.5f}<br>"
                            "close=%{customdata[2]:.5f}<br>"
                            "pattern=%{customdata[3]}<br>"
                            "sid=%{customdata[4]}<br>"
                            "struct_direction=%{customdata[5]}"
                            "<extra></extra>"
                        ),
                        customdata=customdata,
                        **_style("structure.reversal_watch_start"),  # purple X
                    )
                )


    # -------------------------------------------------
    # Week 6: KL Zones overlays (rectangles + confirm line + hover)
    # -------------------------------------------------
    zones = dfx.attrs.get("kl_zones", [])
    if zone_cfg.get("KL", False) and zones:
        # Determine "structures to plot" FROM ZONES:
        # Show zones from the N most recent structure_ids (default: 1)
        by_struct = {}
        for z in zones:
            sid = int(z.meta.get("structure_id", 0))
            side = str(z.side)
            by_struct.setdefault(sid, set()).add(side)

        # Get the N most recent structure_ids
        num_structures = int(zone_cfg.get("num_structures", 1))
        all_sids = sorted(by_struct.keys(), reverse=True)  # descending order (most recent first)
        selected_sids = set(all_sids[:num_structures])
        most_recent_sid = all_sids[0] if all_sids else 0  # Track most recent for opacity tiers

        zones_cur = [z for z in zones if int(z.meta.get("structure_id", 0)) in selected_sids]

        print("[chart][kl] zone structures sides:", {k: sorted(list(v)) for k, v in sorted(by_struct.items())})
        print("[chart][kl] selected_structure_ids:", sorted(selected_sids))
        print("[chart][kl] zones_cur:", [
            (z.side,
             z.meta.get("base_idx"),
             z.meta.get("confirmed_idx"),
             z.meta.get("cycle_id"),
             z.meta.get("structure_id"),
             z.meta.get("active"))
            for z in zones_cur
        ])

        # Visible window bounds
        t_last = dfx[COL_TIME].iloc[-1]

        # Deterministic draw order:
        # 1) inactive first (underneath), active last (on top)
        # 2) older start_time first, newer last
        def _zone_sort_key(z):
            active = bool(z.meta.get("active", False))
            st = pd.to_datetime(z.start_time, utc=True)
            # active False -> 0 (draw earlier), active True -> 1 (draw later/on top)
            return (1 if active else 0, st)

        zones_cur = sorted(zones_cur, key=_zone_sort_key)

        # cap AFTER sorting so we keep most recent ordering behavior stable
        if len(zones_cur) > 50:
            zones_cur = zones_cur[-50:]

        # helper for rgba fill
        def _fill_rgba(side: str, opacity: float) -> str:
            # green/red with opacity
            if side == "buy":
                return f"rgba(0, 180, 0, {opacity})"
            return f"rgba(220, 0, 0, {opacity})"

        def _line_rgba(side: str, opacity: float) -> str:
            if side == "buy":
                return f"rgba(0, 180, 0, {opacity})"
            return f"rgba(220, 0, 0, {opacity})"

        # Build hover lines (shapes themselves don't hover)
        # We'll add top/bottom transparent hv lines per zone.
        for z in zones_cur:
            side = str(z.side)
            active = bool((z.meta or {}).get("active", False)) and (z.end_time is None)
            zone_sid = int(z.meta.get("structure_id", 0))

            stz = _zone_style(side)

            # 3-tier opacity: active=100%, non-active+recent_sid=50%, non-active+prior_sid=25%
            if active:
                opacity_mult = 1.0
            elif zone_sid == most_recent_sid:
                opacity_mult = 0.5
            else:
                opacity_mult = 0.2

            base_fill_op = float(stz.get("fill_opacity_active", 0.18))
            base_line_op = float(stz.get("confirm_opacity_active", 0.9))
            fill_op = base_fill_op * opacity_mult
            line_op = base_line_op * opacity_mult
            rgb = str(stz.get("rgb", "0,180,0" if side == "buy" else "220,0,0"))
            confirm_w = int(stz.get("confirm_line_width", 2))

            fillcolor = _rgba_from_rgb(rgb, fill_op)
            linecolor = _rgba_from_rgb(rgb, line_op)

            x_zone0 = pd.to_datetime(z.start_time, utc=True)
            x_zone1 = pd.to_datetime(z.end_time, utc=True) if z.end_time is not None else pd.to_datetime(t_last, utc=True)

            # Guard: if end before start (shouldn't happen, but safe)
            if x_zone1 <= x_zone0:
                x_zone1 = x_zone0

            # Confirm time (used for placing the vertical line in the correct segment)
            conf_idx = int(z.meta.get("confirmed_idx", -1))
            conf_time = None
            if conf_idx in dfx.index:
                conf_time = pd.to_datetime(dfx.loc[conf_idx, COL_TIME], utc=True)

            # Hover styling (keep existing behavior)
            hover_st = STYLE.get("zone.kl.hover_line", {})
            hover_line = hover_st.get("line", {"width": 6, "color": "rgba(0,0,0,0)"})
            hover_showlegend = bool(hover_st.get("showlegend", False))

            # Hover metadata (keep existing fields)
            base_pattern = str(z.meta.get("base_pattern", ""))
            base_idx = int(z.meta.get("base_idx", -1))
            cycle_id = int(z.meta.get("cycle_id", 0))
            structure_id = int(z.meta.get("structure_id", -1))
            struct_direction = int(z.meta.get("struct_direction", 0))

            # ------------------------------------------------------------------
            # NEW: draw "stepwise" zone rectangles using meta["bounds_steps"]
            # Each step begins at start_idx and applies forward until next step.
            # ------------------------------------------------------------------
            steps = list((z.meta or {}).get("bounds_steps", []))

            # Fallback: no steps -> behave like old code (single segment)
            if not steps:
                steps = [{
                    "start_idx": base_idx,
                    "top": float(max(z.top, z.bottom)),
                    "bottom": float(min(z.top, z.bottom)),
                    "event": "FALLBACK",
                }]

            # Sort steps by start_idx
            steps = sorted(steps, key=lambda s: int(s.get("start_idx", -1)))

            def _idx_to_time(ii: int):
                if ii in dfx.index:
                    return pd.to_datetime(dfx.loc[ii, COL_TIME], utc=True)
                return None

            # For each segment, compute x0/x1 bounds
            for k, s in enumerate(steps):
                seg_start_idx = int(s.get("start_idx", -1))
                seg_x0 = _idx_to_time(seg_start_idx)
                if seg_x0 is None:
                    continue

                # clamp to the zone's actual window
                if seg_x0 < x_zone0:
                    seg_x0 = x_zone0

                # segment end = next step start time, else zone end
                if k + 1 < len(steps):
                    next_idx = int(steps[k + 1].get("start_idx", -1))
                    nxt = _idx_to_time(next_idx) or x_zone1
                    # end *just before* the next segment start to avoid overlap
                    seg_x1 = nxt - pd.Timedelta(microseconds=1)
                else:
                    seg_x1 = x_zone1

                if seg_x1 <= seg_x0:
                    seg_x1 = seg_x0

                seg_top = float(s.get("top", z.top))
                seg_bot = float(s.get("bottom", z.bottom))
                y0 = float(min(seg_bot, seg_top))
                y1 = float(max(seg_bot, seg_top))

                # Rectangle segment (uses same fillcolor/linecolor computed above)
                fig.add_shape(
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=seg_x0,
                    x1=seg_x1,
                    y0=y0,
                    y1=y1,
                    fillcolor=fillcolor,
                    line=dict(width=0),
                    layer="below",
                )

                # Confirm line should appear only in the segment that contains conf_time
                if conf_time is not None and (seg_x0 <= conf_time <= seg_x1):
                    fig.add_shape(
                        type="line",
                        xref="x",
                        yref="y",
                        x0=conf_time,
                        x1=conf_time,
                        y0=y0,
                        y1=y1,
                        line=dict(color=linecolor, width=confirm_w),
                        layer="below",
                    )

                # Hover lines for this segment (shapes don't hover)
                # Use multiple points (one per candle) so hover works across entire horizontal edge
                seg_times = dfx[COL_TIME][(dfx[COL_TIME] >= seg_x0) & (dfx[COL_TIME] <= seg_x1)]
                if len(seg_times) == 0:
                    seg_times = pd.Series([seg_x0, seg_x1])

                hover_customdata = [[
                    side,
                    structure_id,
                    struct_direction,
                    base_pattern,
                    base_idx,
                    conf_idx,
                    cycle_id,
                    y1,
                    y0,
                ]] * len(seg_times)

                # Top hover line
                fig.add_trace(
                    go.Scatter(
                        x=seg_times,
                        y=[y1] * len(seg_times),
                        mode="lines",
                        name="KL zone" if active else "KL zone (inactive)",
                        showlegend=hover_showlegend,
                        line=hover_line,
                        line_shape="hv",
                        hovertemplate=(
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
                        customdata=hover_customdata,
                    )
                )

                # Bottom hover line
                fig.add_trace(
                    go.Scatter(
                        x=seg_times,
                        y=[y0] * len(seg_times),
                        mode="lines",
                        name="KL zone" if active else "KL zone (inactive)",
                        showlegend=hover_showlegend,
                        line=hover_line,
                        line_shape="hv",
                        hovertemplate=(
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
                        customdata=hover_customdata,
                    )
                )


    fig.update_layout(
        title=title,
        xaxis_title="Time (UTC)",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        legend_title="Overlays",
        height=800,
    )

    # --- Global chart styling (background + grid) ---
    fig.update_layout(**_style("chart.layout"))
    fig.update_xaxes(**_style("chart.axis"))
    fig.update_yaxes(**_style("chart.axis"))

    # fig.update_xaxes(
    #     rangebreaks=[
    #         dict(bounds=["sat", "mon"])  # hide weekend gaps
    #     ]
    # )

    # times = pd.to_datetime(dfx[COL_TIME])

    # # Python weekday: Mon=0 ... Sun=6
    # wd = times.dt.dayofweek
    # hr = times.dt.hour

    # # Break window (UTC):
    # # - Friday (4) >= 17:00
    # # - All Saturday (5)
    # # - Sunday (6) < 22:00
    # is_break = ((wd == 4) & (hr >= 22)) | (wd == 5) | ((wd == 6) & (hr < 21))

    # break_times = times[is_break]

    # fig.update_xaxes(
    #     rangebreaks=[
    #         dict(values=break_times.tolist())
    #     ]
    # )

    # --- Dynamic gap removal (compress all no-candle periods) ---
    t = pd.to_datetime(dfx[COL_TIME]).sort_values().reset_index(drop=True)

    # infer expected candle spacing robustly
    dt = t.diff()
    expected = dt[dt.notna()].median()
    if pd.isna(expected) or expected <= pd.Timedelta(0):
        expected = pd.Timedelta(minutes=15)

    gap_mask = dt > (expected *2.0)

    missing = []
    t_values = t.to_list()
    for i in range(1, len(t_values)):
        if bool(gap_mask.iloc[i]):
            start = t_values[i - 1] + expected
            end   = t_values[i] - expected
            if start <= end:
                missing.extend(pd.date_range(start, end, freq=expected).to_pydatetime())

    # paranoia guard: never remove real candle timestamps
    present = set(t_values)
    missing = [x for x in missing if x not in present]

    if missing:
        fig.update_xaxes(
            rangebreaks=[
                dict(values=missing, dvalue=int(expected / pd.Timedelta(milliseconds=1)))
            ]
        )

    html_path = out_dir / f"{basename}.html"
    png_path = out_dir / f"{basename}.png"

    print("DEBUG traces:", len(fig.data))
    print("DEBUG shapes:", len(fig.layout.shapes) if fig.layout.shapes else 0)

    fig.write_html(str(html_path), include_plotlyjs="cdn")
    # PNG export requires kaleido
    fig.write_image(str(png_path), scale=2)

    return ChartExportPaths(html_path=html_path, png_path=png_path)
