"""Zoomed PNG of a saved M15 chart that keeps the volume panel right — for the WVMI LP marks' look ("WVMI back on the
chart" step 5, 2026-10-26).

    PYTHONPATH=. python engine_v2/plans/qa_chart_edits_inputs/zoom_png_volume.py CHART_HTML LO HI OUT_PNG [TITLE]

`zoom_png.py` sets EVERY y axis to the price range (fine before step 5; it flattens the volume bars and the marks'
axis `y3`). This one sets the price axis to the window's low / high (+ a margin), the volume axis `yaxis2` the way the
chart's zoom JS does (`[0, the visible bars' max x volume.headroom]`) and leaves `yaxis3` (the marks' fixed [0, 1])
alone. The x range is mapped from M15 idx through `m15_frame.frame`. Needs plotly + kaleido.
"""
import base64
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
import plotly.graph_objects as go  # noqa: E402
from engine_v2.charting.style_registry import STYLE  # noqa: E402
from engine_v2.debug.chart_census import load_fig  # noqa: E402
from m15_frame import frame  # noqa: E402
sys.argv = _argv


def _arr(v):
    if isinstance(v, dict) and "bdata" in v:
        return np.frombuffer(base64.b64decode(v["bdata"]), dtype=np.dtype(v["dtype"]))
    return np.asarray(v)


path, lo, hi, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
title = sys.argv[5] if len(sys.argv) > 5 else f"{os.path.basename(path)}: M15 {lo}-{hi}"
data, layout = load_fig(path)
f = frame(path)
t0, t1 = f.loc[lo, "time"], f.loc[hi, "time"]
w = f.loc[lo:hi]
pad = (w["high"].max() - w["low"].min()) * 0.05
bar, = [t for t in data if t.get("type") == "bar" and t.get("yaxis") == "y2"]
vx = pd.to_datetime(pd.Series(bar["x"]))
vy = _arr(bar["y"]).astype(float)
vmax = float(vy[((vx >= t0) & (vx <= t1)).to_numpy()].max())

fig = go.Figure(data=data, layout=layout)
fig.update_layout(width=1400, height=750, showlegend=False, title=title)
fig.update_xaxes(range=[t0, t1], rangeslider_visible=False)
fig.update_layout(yaxis=dict(range=[w["low"].min() - pad, w["high"].max() + pad]),
                  yaxis2=dict(range=[0, vmax * float(STYLE["volume.headroom"]["factor"])]))
fig.write_image(out)
print("wrote", out)
