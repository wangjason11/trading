import sys
sys.argv=[sys.argv[0]]
from engine_v2.debug.chart_census import load_fig
for suf in ("_M15_counter","_M15_confluence"):
    for lab,p in (("SQ","artifacts/commits/week9-entries-decisions/20261014_170919_da8c15e/"),("NOW","artifacts/charts/")):
        a,_=load_fig(p+"NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"+suf+".html")
        for t in a:
            if "proximity" in (t.get("name") or "").lower():
                print(suf,lab,t["name"],len(t["x"]))
                for c in t.get("customdata") or []: print("   ",c)
