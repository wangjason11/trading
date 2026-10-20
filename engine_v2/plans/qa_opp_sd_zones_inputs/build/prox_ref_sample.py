"""SAMPLE for the user's chart look (PLAN_R §5.2, not locked): the H1 chart of the step-2 replay + one stepped
"current proximity reference" line per side per overlay-driven cycle (sd: solid teal, opp_sd: dashed orange), the
hover naming the reference (parent BOS / POI / CTS, or sub_N cC KIND) and the gap. Reads step 2's overlay `ref_log`
7-tuples (key, h, sd rung, opp_sd rung, gap, narrow, gate) and the replay's H1 chart; run from the repo root after a
replay. Writes H1_proximity_reference_SAMPLE.html to $PROX_REF_SAMPLE_OUT (default: the system temp dir), never
into the repo."""
import contextlib
import io
import os
import sys
import tempfile

sys.argv = [sys.argv[0]]
import plotly.graph_objects as go

import engine_v2.run_replay as rr
from engine_v2.config import CONFIG
import engine_v2.pipeline.orchestrator as orch
from engine_v2.debug.chart_census import load_fig

OUT = os.environ.get("PROX_REF_SAMPLE_OUT", tempfile.gettempdir())
P = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"

df, eff, _ = rr.fetch_history_with_auto_extend(pair=CONFIG.pair, timeframe=CONFIG.timeframe,
    start=rr._floor_to_day_start(CONFIG.start), end=CONFIG.end, lookback_days=183, min_history=50,
    extend_days=4, max_extend_iters=8)
df.attrs["pair"] = CONFIG.pair
with contextlib.redirect_stdout(io.StringIO()):
    res = orch.run_pipeline(df, lower_timeframes=CONFIG.lower_timeframes)
ov, pool, h1 = res.meta["proximity_overlay"], res.meta["sub_pool"], res.df


def name(r):
    if r is None:
        return None
    if r.source == "parent":
        return f"parent {r.zone_kind}"
    return f"sub_{pool.get_by_pool_idx(r.sub_pool_idx).sub_id} c{r.sub_cycle_id} {r.zone_kind}"


data, layout = load_fig(f"artifacts/charts/{P}.html")
fig = go.Figure(data=data, layout=layout)
style = {"sd": dict(color="rgba(0,150,136,0.9)", dash="solid"), "opp_sd": dict(color="rgba(230,120,0,0.9)", dash="dash")}
for key in sorted(ov.scans):
    rows = [e for e in ov.ref_log if e[0] == key]
    for side, col in (("sd", 2), ("opp_sd", 3)):
        xs, ys, txt = [], [], []
        for e in rows:
            r = e[col]
            if r is None:
                continue
            xs.append(h1.loc[e[1], "time"])
            ys.append(r.inner)
            txt.append(f"cycle {key} H1 {e[1]}<br>{side} proximity reference: {name(r)} {r.inner:.5f}"
                       f"<br>gap {e[4]} pips{' NARROW' if e[5] else ''}" + (f"<br>blocked: {e[6]}" if e[6] else ""))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=1.6, shape="hv", **style[side]),
                                 name=f"prox ref {side} {key}", text=txt, hovertemplate="%{text}<extra></extra>"))
path = os.path.join(OUT, "H1_proximity_reference_SAMPLE.html")
fig.write_html(path, include_plotlyjs="cdn")
print(path)
