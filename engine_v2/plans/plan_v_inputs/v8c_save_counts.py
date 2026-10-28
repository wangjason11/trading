"""PLAN_V 8c: S_V8b vs S_V8c — the two lens `*_wvmi_lps.csv` line counts + the 'M15 WVMI LP sell' dot counts.

    PYTHONIOENCODING=utf-8 python engine_v2/plans/plan_v_inputs/v8c_save_counts.py > engine_v2/plans/plan_v_inputs/v8c_save_counts.txt
"""
import json
import pathlib
import re

ROOT = pathlib.Path("artifacts/commits/week9-entries-decisions")
SAVES = {"S_V8b": "20261028_141322_245620c", "S_V8c": "20261028_153518_1318c82"}
PFX = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5_M15_"
TRACE = "M15 WVMI LP sell"


def _figure(html_path):
    s = html_path.read_text(encoding="utf-8")
    m = re.search(r'Plotly\.newPlot\(\s*"[^"]+",\s*(\[.*?\]),\s*(\{.*?\}),\s*(\{.*?\})\s*\)', s, re.S)
    return json.loads(m.group(1))


for lens in ("confluence", "counter"):
    row = []
    for name, folder in SAVES.items():
        d = ROOT / folder
        n_lines = len((d / f"{PFX}{lens}_wvmi_lps.csv").read_bytes().splitlines())
        traces = [t for t in _figure(d / f"{PFX}{lens}.html") if t.get("name") == TRACE]
        xs = traces[0]["x"] if traces else []
        n_dots = len(xs["bdata"]) if isinstance(xs, dict) else len(xs)
        row.append((name, n_lines, len(traces), n_dots, xs))
    (a, la, ta, da, xa), (b, lb, tb, db, xb) = row
    new_x = sorted(set(xb) - set(xa)) if isinstance(xa, list) and isinstance(xb, list) else "?"
    print(f"{lens}: *_wvmi_lps.csv lines {a} {la} -> {b} {lb}; '{TRACE}' traces {ta}/{tb}, dots {da} -> {db}; "
          f"new x {new_x}")
