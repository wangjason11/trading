"""Every distinct hover family per chart (trace name with digits normalised) + its first hover lines — the inventory
behind the register's list of hovers the 2026-10-22 → 24 label pass did not touch (H1 + the counter chart).

    PYTHONPATH=. python engine_v2/plans/qa_chart_edits_inputs/hover_inventory.py SAVE_DIR
"""
import re
import sys
_argv = sys.argv[:]; sys.argv = [sys.argv[0]]
from engine_v2.debug.chart_census import load_fig  # noqa: E402
sys.argv = _argv
D = sys.argv[1]
STEM = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"
for suf in ("", "_M15_counter"):
    data, _layout = load_fig(f"{D}/{STEM}{suf}.html")
    seen = {}
    for t in data:
        name = re.sub(r"\d+", "#", str(t.get("name") or "(unnamed)"))
        ht = t.get("hovertemplate") or ""
        if isinstance(ht, list):
            ht = ht[0] if ht else ""
        if not ht and t.get("hovertext"):
            h = t["hovertext"]; ht = "TEXT:" + str(h[0] if isinstance(h, list) and h else h)
        if not ht and t.get("hoverinfo") == "skip":
            ht = "(skip)"
        key = (name, re.sub(r"\d+", "#", ht.split("<br>")[0])[:30])
        if key not in seen:
            seen[key] = ht.replace("<br>", " / ")[:150]
    print(f"===== {suf or 'H1'}: {len(seen)} families")
    for (n, _), h in sorted(seen.items()):
        print(f"{n[:40]:40s} {h}")
