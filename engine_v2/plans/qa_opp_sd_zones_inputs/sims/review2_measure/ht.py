import sys
sys.argv=[sys.argv[0]]
from engine_v2.debug.chart_census import load_fig
a,_=load_fig("artifacts/charts/NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5.html")
for t in a:
    if t.get("name")=="zone_proximity:trigger": print(t["hovertemplate"]); print(t["marker"])
