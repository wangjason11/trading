"""PLAN_M — a FULL-figure fingerprint of the three replay charts (PLAN_M §8: landing 2 = the trial, figures included).

    PYTHONPATH=. python engine_v2/plans/plan_m_inputs/fig_hash.py CHARTS_DIR [NAME_PREFIX]

NAME_PREFIX (default empty) picks one run's charts in a folder holding several (the repo's `artifacts/charts`: e.g.
`NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_`).

Per chart: trace / shape counts and the sha256 of the canonical figure — the SORTED multiset of every trace's full
JSON (hover text included) + the full layout JSON (shapes sorted) — so an equal hash = an equal figure up to trace
order (`cmp_fig_full.py` is positional; this is keyed by content). Compare two runs' outputs line for line.
"""
import glob
import hashlib
import json
import os
import sys

_argv = sys.argv[:]
sys.argv = [sys.argv[0]]  # chart_census runs a census on import when given args
from engine_v2.debug.chart_census import load_fig  # noqa: E402
sys.argv = _argv

D = sys.argv[1]
PREFIX = sys.argv[2] if len(sys.argv) > 2 else ""
for suffix in ("rk2-5.html", "rk2-5_M15_confluence.html", "rk2-5_M15_counter.html"):
    got = glob.glob(os.path.join(D, PREFIX + "*" + suffix))
    assert len(got) == 1, (D, suffix, got)
    data, layout = load_fig(got[0])
    traces = sorted(json.dumps(t, sort_keys=True, default=str) for t in data)
    lay = dict(layout)
    shapes = sorted(json.dumps(s, sort_keys=True, default=str) for s in lay.pop("shapes", []))
    blob = json.dumps({"traces": traces, "shapes": shapes, "layout": lay}, sort_keys=True, default=str)
    print(f"{suffix}: traces {len(traces)} shapes {len(shapes)} sha256 {hashlib.sha256(blob.encode()).hexdigest()}")
