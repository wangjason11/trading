"""Census of the M15 charts' traces + shapes, classified PARENT (H1-derived) vs the lens's own, for item 5's final
part (the parent-overlay toggle, 2026-10-27). Traces: grouped by name (an unnamed trace by its hover's first line);
shapes: by (type, xref, yref, layer, line dash, fill). Prints the legend entries too (showlegend / legendgroup).

    PYTHONPATH=. python engine_v2/plans/qa_chart_edits_inputs/parent_overlay_census.py SAVE_DIR
"""
import glob
import os
import sys
from collections import Counter

_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from engine_v2.debug.chart_census import load_fig  # noqa: E402
sys.argv = _argv

SAVE = sys.argv[1]
STEM = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_"


def trace_key(t):
    n = t.get("name")
    if n:
        # strip per-sid / per-side suffixes so the groups stay readable
        for cut in (" sid=", " h1s", " (sid="):
            if cut in n:
                n = n.split(cut)[0] + cut.strip() + "*"
        return n
    ht = str(t.get("hovertemplate") or "")
    first = ht.split("<br>")[0]
    return f"<unnamed> {first[:40]} | {ht.split('<br>')[1][:30] if '<br>' in ht else ''}"


for suffix in ("rk2-5_M15_counter.html", "rk2-5_M15_confluence.html"):
    path = glob.glob(os.path.join(SAVE, STEM + suffix))[0]
    traces, layout = load_fig(path)
    print(f"=== {suffix}: {len(traces)} traces, {len(layout.get('shapes') or [])} shapes")
    c = Counter(trace_key(t) for t in traces)
    for k, n in sorted(c.items()):
        print(f"  {n:4d}  {k}")
    leg = [(t.get("name"), t.get("legendgroup"), t.get("showlegend")) for t in traces
           if t.get("showlegend") is not False or t.get("legendgroup")]
    lc = Counter(leg)
    print("  legend-relevant (name, legendgroup, showlegend):")
    for k, n in sorted(lc.items(), key=str):
        print(f"    {n:4d}  {k}")
    sc = Counter()
    for s in layout.get("shapes") or []:
        line = s.get("line") or {}
        sc[(s.get("type"), s.get("xref"), s.get("yref"), s.get("layer"), line.get("dash"),
            str(line.get("color"))[:28], str(s.get("fillcolor"))[:28], s.get("legendgroup"))] += 1
    print("  shapes (type, xref, yref, layer, dash, line color, fill, legendgroup):")
    for k, n in sorted(sc.items(), key=str):
        print(f"    {n:4d}  {k}")
