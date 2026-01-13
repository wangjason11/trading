from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from engine_v2.common.types import COL_C, COL_H, COL_L, COL_O, COL_TIME
from engine_v2.charting.style_registry import STYLE
from engine_v2.common.types import PatternStatus


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

    customdata = list(zip(
        candle_idx,
        dfx["mid_price"].astype(float),
        dfx["body_pct"].astype(float),
        dfx["candle_type"].astype(str),
        dfx["body_len"].astype(float),
        dfx["candle_len"].astype(float),
        range_break_frac,
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
                "range_break_frac=%{customdata[6]:.2%}"
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
    # Week 5 Phase 1A: Market state change labels (bo/pb/pbr/rev)
    # -------------------------------------------------
    if state_cfg.get("labels", False) and "market_state" in dfx.columns:
        # map to short labels
        label_map = {
            "breakout": "bo",
            "pullback": "pb",
            "pullback_range": "pbr",
            "reversal": "rev",
        }

        ms = dfx["market_state"].astype(str)
        prev = ms.shift(1).fillna("")
        changed = (ms != prev) & ms.isin(list(label_map.keys()))
        sub = dfx[changed].copy()

        if not sub.empty:
            # choose y position: breakout above high, pullback/pbr below low, reversal based on candle dir fallback
            # (uses wick_offset already computed)
            def _label_y(row):
                st = str(row["market_state"])
                if st == "breakout":
                    return float(row[COL_H]) + float(wick_offset.loc[row.name])
                if st in ("pullback", "pullback_range"):
                    return float(row[COL_L]) - float(wick_offset.loc[row.name])
                # reversal: if we have pat_dir use that, else candle direction
                dir_hint = None
                if "pat_dir" in dfx.columns:
                    try:
                        dir_hint = int(row.get("pat_dir", 0))
                    except Exception:
                        dir_hint = 0
                if not dir_hint and "direction" in dfx.columns:
                    try:
                        dir_hint = int(row.get("direction", 0))
                    except Exception:
                        dir_hint = 0
                return (float(row[COL_H]) + float(wick_offset.loc[row.name])) if dir_hint >= 0 else (float(row[COL_L]) - float(wick_offset.loc[row.name]))

            y = sub.apply(_label_y, axis=1)
            text = sub["market_state"].map(label_map).tolist()

            fig.add_trace(
                go.Scatter(
                    x=sub[COL_TIME],
                    y=y,
                    mode="text",
                    text=text,
                    name="state labels",
                    textposition="middle center",
                    showlegend=False,
                    hovertemplate=(
                        "idx=%{customdata[0]}<br>"
                        "state=%{customdata[1]}<extra></extra>"
                    ),
                    customdata=list(zip(sub.index.to_numpy(), sub["market_state"].astype(str))),
                )
            )

    # -------------------------------------------------
    # Week 5 Phase 1B: Active range rectangles (segmented on expansion)
    # -------------------------------------------------
    if range_vis_cfg.get("rectangles", False) and all(c in dfx.columns for c in ["range_active", "range_hi", "range_lo", "range_start_idx"]):
        # We'll create rectangle segments whenever bounds change while range_active==1.
        # Each segment starts at:
        #   - first segment: max(range_start_idx, first visible idx)
        #   - subsequent segments: the candle idx where expansion occurred
        # Each segment ends at the next segment start, or at the breakout/reversal candle that ends range.

        # helper: index label -> time, but only if label exists in dfx
        time_by_idx = {int(i): t for i, t in zip(dfx.index.to_numpy(), dfx[COL_TIME])}

        in_range = False
        seg_start_idx = None
        cur_hi = None
        cur_lo = None

        # cap shapes to avoid runaway if needed
        shapes_added = 0
        MAX_SHAPES = 500

        idx_list = list(map(int, dfx.index.to_numpy()))
        for pos, idx in enumerate(idx_list):
            ra = int(dfx.iloc[pos]["range_active"]) if "range_active" in dfx.columns else 0
            st = str(dfx.iloc[pos]["market_state"]) if "market_state" in dfx.columns else ""

            # treat reversal as a hard end (even if range_active doesn't flip yet due to early stop)
            is_end_state = (st == "reversal")

            if not in_range:
                if ra == 1:
                    # start new range block
                    in_range = True
                    rs = int(dfx.iloc[pos]["range_start_idx"])
                    seg_start_idx = rs if rs in time_by_idx else idx  # clamp to visible window
                    cur_hi = float(dfx.iloc[pos]["range_hi"])
                    cur_lo = float(dfx.iloc[pos]["range_lo"])
                continue

            # in_range == True
            # detect expansion
            hi = float(dfx.iloc[pos]["range_hi"])
            lo = float(dfx.iloc[pos]["range_lo"])
            expanded = (hi != cur_hi) or (lo != cur_lo)

            # detect end: range_active flips off OR reversal state
            ended = (ra == 0) or is_end_state

            if expanded or ended:
                # close previous segment at current candle idx time
                x0 = time_by_idx.get(int(seg_start_idx), None)
                x1 = time_by_idx.get(int(idx), None)

                if x0 is not None and x1 is not None and shapes_added < MAX_SHAPES:
                    y0 = min(cur_lo, cur_hi)
                    y1 = max(cur_lo, cur_hi)
                    fig.add_shape(
                        type="rect",
                        xref="x",
                        yref="y",
                        x0=x0,
                        x1=x1,
                        y0=y0,
                        y1=y1,
                        fillcolor="yellow",
                        opacity=0.2,
                        line_width=0,
                        layer="below",
                    )
                    shapes_added += 1

                # start next segment if still active
                if ended:
                    in_range = False
                    seg_start_idx = None
                    cur_hi = None
                    cur_lo = None
                else:
                    seg_start_idx = idx
                    cur_hi = hi
                    cur_lo = lo

        # If range continued through the end of the visible window, close it to last candle
        if in_range and seg_start_idx is not None and shapes_added < MAX_SHAPES:
            last_idx = idx_list[-1]
            x0 = time_by_idx.get(int(seg_start_idx), None)
            x1 = time_by_idx.get(int(last_idx), None)
            if x0 is not None and x1 is not None:
                y0 = min(cur_lo, cur_hi)
                y1 = max(cur_lo, cur_hi)
                fig.add_shape(
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=x0,
                    x1=x1,
                    y0=y0,
                    y1=y1,
                    fillcolor="yellow",
                    opacity=0.2,
                    line_width=0,
                    layer="below",
                )

    # -------------------------------------------------
    # Range hover lines (top/bottom only)
    # Shapes (rectangles) don't support hover, so we add
    # thin line traces that carry hover data.
    # -------------------------------------------------
    if range_vis_cfg.get("rectangles", False) and all(c in dfx.columns for c in ["range_active", "range_hi", "range_lo", "range_start_idx"]):
        active = dfx["range_active"].astype(int) == 1
        sub = dfx[active].copy()

        # --- NEW: compute the candle idx/time where range bounds were last updated ---
        # We treat an "update" as: while range_active==1, range_hi or range_lo changes from the prior active candle.
        # Then we forward-fill so every candle can say "range_lo was last set at idx=X".

        # Ensure we have time handy
        sub["_t"] = sub[COL_TIME]

        # Detect changes vs previous active candle (within the filtered sub)
        prev_hi = sub["range_hi"].shift(1)
        prev_lo = sub["range_lo"].shift(1)

        changed = (sub["range_hi"] != prev_hi) | (sub["range_lo"] != prev_lo)

        # Also mark the first visible active candle as an update (so it has a seed)
        if len(sub) > 0:
            changed.iloc[0] = True

        sub["range_last_update_idx"] = sub.index.where(changed).astype("float")
        sub["range_last_update_time"] = sub["_t"].where(changed)

        # Forward-fill update info across the active segment
        sub["range_last_update_idx"] = sub["range_last_update_idx"].ffill().astype(int)
        sub["range_last_update_time"] = sub["range_last_update_time"].ffill()

        # Clean helper col
        sub.drop(columns=["_t"], inplace=True)

        if not sub.empty:
            # Top line (range_hi)
            fig.add_trace(
                go.Scatter(
                    x=sub[COL_TIME],
                    y=sub["range_hi"].astype(float),
                    mode="lines",
                    name="range:top",
                    showlegend=False,
                    line=dict(width=2, color="rgba(0,0,0,0)"),
                    line_shape="hv",
                    hovertemplate=(
                        "idx=%{customdata[0]}<br>"
                        "range_start_idx=%{customdata[1]}<br>"
                        "range_hi=%{customdata[2]:.5f}<br>"
                        "range_lo=%{customdata[3]:.5f}<br>"
                        "last_update_idx=%{customdata[4]}<br>"
                        "last_update_time=%{customdata[5]}"
                        "<extra></extra>"
                    ),
                    customdata=list(zip(
                        sub.index.to_numpy(),
                        sub["range_start_idx"].astype(int),
                        sub["range_hi"].astype(float),
                        sub["range_lo"].astype(float),
                        sub["range_last_update_idx"].astype(int),
                        sub["range_last_update_time"].astype(str),
                    )),
                )
            )

            # Bottom line (range_lo)
            fig.add_trace(
                go.Scatter(
                    x=sub[COL_TIME],
                    y=sub["range_lo"].astype(float),
                    mode="lines",
                    name="range:bottom",
                    showlegend=False,
                    line=dict(width=2, color="rgba(0,0,0,0)"),
                    line_shape="hv",
                    hovertemplate=(
                        "idx=%{customdata[0]}<br>"
                        "range_start_idx=%{customdata[1]}<br>"
                        "range_hi=%{customdata[2]:.5f}<br>"
                        "range_lo=%{customdata[3]:.5f}<br>"
                        "last_update_idx=%{customdata[4]}<br>"
                        "last_update_time=%{customdata[5]}"
                        "<extra></extra>"
                    ),
                    customdata=list(zip(
                        sub.index.to_numpy(),
                        sub["range_start_idx"].astype(int),
                        sub["range_hi"].astype(float),
                        sub["range_lo"].astype(float),
                        sub["range_last_update_idx"].astype(int),
                        sub["range_last_update_time"].astype(str),
                    )),
                )
            )


    # Structure levels overlays
    levels = structure_levels or []
    if struct_cfg.get("levels", False) and levels:
        x0 = dfx[COL_TIME].iloc[0]
        x1 = dfx[COL_TIME].iloc[-1]

        xs, ys = [], []
        for lv in levels[-200:]:
            xs += [x0, x1, None]
            y = float(lv.price)
            ys += [y, y, None]

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name="BOS/CTS",
                line=_style("structure.level").get("line", dict(width=1, dash="dot")),
            )
        )

    # Zones overlays
    zones = dfx.attrs.get("kl_zones", [])
    if zone_cfg.get("KL", False):
        for z in zones[-50:]:  # cap for chart performance
            # plotly wants y0<y1
            y0 = min(z.bottom, z.top)
            y1 = max(z.bottom, z.top)
            fig.add_hrect(
                y0=y0,
                y1=y1,
                fillcolor="green" if z.side == "buy" else "red",
                opacity=0.12,
                line_width=0,
            )


    fig.update_layout(
        title=title,
        xaxis_title="Time (UTC)",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        legend_title="Overlays",
        height=800,
    )

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

    gap_mask = dt > (expected * 1.5)

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
