import sys, json
sys.argv=[sys.argv[0]]
from engine_v2.debug.chart_census import load_fig
from collections import Counter
S="artifacts/commits/week9-entries-decisions/20261014_170919_da8c15e/"
N="artifacts/charts/"
fn="NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"
for suf in ["", "_M15_counter", "_M15_confluence"]:
    a,la=load_fig(S+fn+suf+".html"); b,lb=load_fig(N+fn+suf+".html")
    print(suf or "H1", len(a), len(b), "layout equal", json.dumps(la,sort_keys=True)==json.dumps(lb,sort_keys=True))
    ca=Counter(json.dumps(t,sort_keys=True) for t in a); cb=Counter(json.dumps(t,sort_keys=True) for t in b)
    oa=ca-cb; ob=cb-ca
    print("  only S_Q:", sum(oa.values()), " only now:", sum(ob.values()))
    if suf=="":
        for k in oa: print("  S_Q trace name:", json.loads(k).get("name"))
        for k in ob: print("  NOW trace name:", json.loads(k).get("name"))
        # dump the marker traces
        for lab,data in (("SQ",a),("NOW",b)):
            for t in data:
                if "zone_proximity" in (t.get("name") or "") or "proximity" in json.dumps(t.get("legendgroup","")):
                    print(lab, "TRACE", t.get("name"), "n=", len(t.get("x") or []))
                    xs=t.get("x"); ys=t.get("y"); ht=t.get("hovertext") or t.get("text"); cd=t.get("customdata")
                    print("  keys:", sorted(t.keys()))
                    for i in range(len(xs)):
                        print("   ", xs[i], ys[i], (ht[i] if isinstance(ht,list) else ht), (cd[i] if cd else None))
