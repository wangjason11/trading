from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from engine_v2.common.types import COL_C, COL_H, COL_L, COL_O, COL_TIME


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
) -> ChartExportPaths:
    """
    Export an interactive HTML + PNG candlestick chart with basic overlays.

    Expected columns (minimum):
      - time, o, h, l, c
    Optional overlays (if present):
      - candle_type (pinbar)
      - engulfing (-1/0/1)
      - star (-1/0/1)
    """
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
        )
    )

    # Pinbar markers
    if "candle_type" in dfx.columns:
        pin = dfx[dfx["candle_type"] == "pinbar"]
        if not pin.empty:
            fig.add_trace(
                go.Scatter(
                    x=pin[COL_TIME],
                    y=pin[COL_H],
                    mode="markers",
                    name="pinbar",
                )
            )

    # Engulfing markers
    if "engulfing" in dfx.columns:
        bull = dfx[dfx["engulfing"] == 1]
        bear = dfx[dfx["engulfing"] == -1]
        if not bull.empty:
            fig.add_trace(
                go.Scatter(
                    x=bull[COL_TIME],
                    y=bull[COL_L],
                    mode="markers",
                    name="engulfing +1",
                )
            )
        if not bear.empty:
            fig.add_trace(
                go.Scatter(
                    x=bear[COL_TIME],
                    y=bear[COL_H],
                    mode="markers",
                    name="engulfing -1",
                )
            )

    # Star markers
    if "star" in dfx.columns:
        bull = dfx[dfx["star"] == 1]
        bear = dfx[dfx["star"] == -1]
        if not bull.empty:
            fig.add_trace(
                go.Scatter(
                    x=bull[COL_TIME],
                    y=bull[COL_L],
                    mode="markers",
                    name="star +1",
                )
            )
        if not bear.empty:
            fig.add_trace(
                go.Scatter(
                    x=bear[COL_TIME],
                    y=bear[COL_H],
                    mode="markers",
                    name="star -1",
                )
            )


    # Structure levels overlays
    levels = structure_levels or []
    if levels:
        x0 = dfx[COL_TIME].iloc[0]
        x1 = dfx[COL_TIME].iloc[-1]

        xs = []
        ys = []
        for lv in levels[-200:]:  # cap for performance
            xs += [x0, x1, None]
            ys += [lv.price, lv.price, None]

        fig.add_trace(
            go.Scattergl(
                x=xs,
                y=ys,
                mode="lines",
                name="BOS/CTS",
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

    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"])  # hide weekend gaps
        ]
    )

    html_path = out_dir / f"{basename}.html"
    png_path = out_dir / f"{basename}.png"

    fig.write_html(str(html_path), include_plotlyjs="cdn")
    # PNG export requires kaleido
    fig.write_image(str(png_path), scale=2)

    return ChartExportPaths(html_path=html_path, png_path=png_path)
