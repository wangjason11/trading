"""Every Fib zone hover as rendered on a chart dir (distinct texts per trace), to ground the 2026-10-23 fib-hover spec.

    PYTHONPATH=. python engine_v2/plans/qa_chart_edits_inputs/fib_hovers.py SAVE_DIR
"""
import os, re, sys
from collections import defaultdict
_argv = sys.argv[:]; sys.argv = [sys.argv[0]]
from engine_v2.debug.chart_census import load_fig
sys.argv = _argv
D = sys.argv[1]
STEM = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"
_FIELD = re.compile(r"%\{customdata\[(\d+)\](?::([^}]*))?\}")
def fill(tmpl, cd):
    def rep(m):
        v = cd[int(m.group(1))]
        return format(v, m.group(2)) if m.group(2) else str(v)
    return _FIELD.sub(rep, tmpl).replace("<extra></extra>", "")
for chart, suffix in (("h1", ".html"), ("counter", "_M15_counter.html"), ("confluence", "_M15_confluence.html")):
    data, _ = load_fig(os.path.join(D, STEM + suffix))
    names = sorted({str(t.get("name", "")) for t in data})
    seen = defaultdict(list)
    for t in data:
        name = str(t.get("name", ""))
        if "fib" not in name.lower():
            continue
        tmpl = t.get("hovertemplate")
        if tmpl and t.get("customdata") is not None:
            for cd in t["customdata"]:
                tx = fill(tmpl, cd)
                if tx not in seen[name]:
                    seen[name].append(tx)
        elif t.get("hovertext") or t.get("text"):
            for tx in (t.get("hovertext") or t.get("text") or []):
                if tx not in seen[name]:
                    seen[name].append(tx)
    for name, texts in seen.items():
        print(f"--- [{chart}] {name}: {len(texts)} distinct")
        for tx in texts:
            print("   " + " // ".join(str(tx).split("<br>")))
