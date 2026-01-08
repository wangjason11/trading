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
            "break_confirm": False,
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
    struct_cfg = cfg.get("structure", {}) or {}
    zone_cfg = cfg.get("zones", {}) or {}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfx = df.copy()
    if max_points is not None and len(dfx) > max_points:
        dfx = dfx.iloc[-max_points:].copy()

    # Ensure datetime
    dfx[COL_TIME] = pd.to_datetime(dfx[COL_TIME], utc=True)

    fig = go.Figure()

    # Offset to keep markers away from candle wicks
    wick_offset = (dfx[COL_H] - dfx[COL_L]) * 0.15

    # Candles
    candle_idx = dfx.index.to_numpy()

    customdata = list(zip(
        candle_idx,
        dfx["mid_price"].astype(float),
        dfx["body_pct"].astype(float),
        dfx["candle_type"].astype(str),
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
                "body_pct=%{customdata[2]:.2%}<br>"
                "mid_price=%{customdata[1]:.5f}"
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

    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"])  # hide weekend gaps
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
