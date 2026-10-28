"""PLAN_W 9b — the Aug–Sep chart-look replays: how the 'M15 fib' hover changed (distinct before / after strings per
point, compacted) and the new 'M15 POI zone' traces' hover.

    PYTHONIOENCODING=utf-8 PYTHONPATH=. python engine_v2/plans/plan_w_inputs/w9b_aug_sep_hover_check.py BASE_TREE VAR_TREE [OUT.txt]
"""
import collections
import json
import sys

_argv = sys.argv[:]
sys.argv = sys.argv[:1]
from engine_v2.debug.chart_census import load_fig  # noqa: E402
sys.argv = _argv

base, var = sys.argv[1], sys.argv[2]
out = []
for lens in ("confluence", "counter"):
    n = f"NZD_USD_H1_2025-08-01_2025-09-20_sd1_eps0p0001_rk2-5_M15_{lens}.html"
    (a, _), (b, _) = load_fig(f"{base}/artifacts/charts/{n}"), load_fig(f"{var}/artifacts/charts/{n}")
    fa = [t for t in a if t.get("name") == "M15 fib"]
    fb = [t for t in b if t.get("name") == "M15 fib"]
    for ta, tb in zip(fa, fb):
        keys = sorted(k for k in set(ta) | set(tb) if json.dumps(ta.get(k), default=str) != json.dumps(tb.get(k), default=str))
        if not keys:
            continue
        pairs = collections.Counter()
        for x, y in zip(ta.get("customdata") or [], tb.get("customdata") or []):
            if x != y:
                dx = [s for s in x if s not in y]
                dy = [s for s in y if s not in x]
                pairs[(json.dumps(dx), json.dumps(dy))] += 1
        out.append(f"{lens}: 'M15 fib' differs in {keys}; {sum(pairs.values())} of {len(ta.get('customdata') or [])} "
                   f"points changed, {len(pairs)} distinct change(s):")
        for (dx, dy), c in pairs.items():
            out.append(f"   x{c}: base {dx}\n         9b   {dy}")
    names_a = collections.Counter(json.dumps(t, sort_keys=True, default=str) for t in a if t.get("name") == "M15 POI zone")
    for t in b:
        if t.get("name") == "M15 POI zone" and json.dumps(t, sort_keys=True, default=str) not in names_a:
            cd = t.get("customdata")
            hover = t.get("hovertemplate") or ""
            first = cd[0] if isinstance(cd, list) and cd else cd
            out.append(f"{lens}: NEW 'M15 POI zone' trace, {len(t.get('x') or [])} points, x {t.get('x', [None])[0]} .. "
                       f"{(t.get('x') or [None])[-1]}\n   hovertemplate: {hover[:300]}\n   customdata[0]: {str(first)[:900]}")
text = "\n".join(out) + "\n"
print(text)
if len(sys.argv) > 3:
    open(sys.argv[3], "w", encoding="utf-8").write(text)
