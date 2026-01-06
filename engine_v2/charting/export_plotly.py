from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from engine_v2.common.types import COL_C, COL_H, COL_L, COL_O, COL_TIME
from engine_v2.charting.style_registry import STYLE


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

    # Candles
    fig.add_trace(
        go.Candlestick(
            x=dfx[COL_TIME],
            open=dfx[COL_O],
            high=dfx[COL_H],
            low=dfx[COL_L],
            close=dfx[COL_C],
            name="OHLC",
            showlegend=False,
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
