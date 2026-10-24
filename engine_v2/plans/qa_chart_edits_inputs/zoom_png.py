"""Zoomed PNG of a saved chart over an M15 idx range — for a chart look at one spot, before vs after.

    PYTHONPATH=. python engine_v2/plans/qa_chart_edits_inputs/zoom_png.py CHART_HTML LO HI YLO YHI OUT_PNG [TITLE]

The x range is mapped from M15 idx through `m15_frame.frame` (the M15 chart's candles; an M15 chart only). Needs
plotly + kaleido. Used for the replacement-break look (2026-10-24): confluence 2990-3560, counter 3560-3880.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
import plotly.graph_objects as go  # noqa: E402
from engine_v2.debug.chart_census import load_fig  # noqa: E402
from m15_frame import frame  # noqa: E402
sys.argv = _argv

path, lo, hi, ylo, yhi, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), \
    float(sys.argv[5]), sys.argv[6]
title = sys.argv[7] if len(sys.argv) > 7 else f"{os.path.basename(path)}: M15 {lo}-{hi}"
data, layout = load_fig(path)
f = frame(path)
fig = go.Figure(data=data, layout=layout)
fig.update_layout(width=1400, height=750, showlegend=False, title=title)
fig.update_xaxes(range=[f.loc[lo, "time"], f.loc[hi, "time"]], rangeslider_visible=False)
fig.update_yaxes(range=[ylo, yhi])
fig.write_image(out)
print("wrote", out)
