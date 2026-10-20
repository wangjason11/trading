import sys, json
sys.argv=[sys.argv[0]]
from engine_v2.debug.chart_census import load_fig
for suf in ("","_M15_counter","_M15_confluence"):
    for lab,p in (("SQ","artifacts/commits/week9-entries-decisions/20261014_170919_da8c15e/"),("NOW","artifacts/charts/")):
        a,l=load_fig(p+"NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"+suf+".html")
        print(suf or "H1",lab,len(a),len(l.get("shapes",[])),len(l.get("annotations",[])))
