"""Every KL / POI zone hover as rendered on a chart dir (one line set per distinct customdata row), + the zone CSVs'
lifecycle fields, to ground the 2026-10-23 zone-hover spec questions.

    PYTHONPATH=. python zone_hovers.py SAVE_DIR
"""
import ast
import glob
import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from engine_v2.debug.chart_census import load_fig  # noqa: E402
sys.argv = _argv

D = sys.argv[1]
STEM = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"
ZONE = re.compile(r"^(KL zone|POI zone|M15 KL zone|M15 POI zone|H1 KL zone \(overlay\)|H1 POI zone \(overlay\))")
_FIELD = re.compile(r"%\{customdata\[(\d+)\](?::([^}]*))?\}")


def fill(tmpl, cd):
    def rep(m):
        v = cd[int(m.group(1))]
        return format(v, m.group(2)) if m.group(2) else str(v)
    return _FIELD.sub(rep, tmpl).replace("<extra></extra>", "")


print("=" * 30, "RENDERED HOVERS (distinct)")
for chart, suffix in (("h1", ".html"), ("counter", "_M15_counter.html"), ("confluence", "_M15_confluence.html")):
    data, _ = load_fig(os.path.join(D, STEM + suffix))
    seen = defaultdict(list)
    for t in data:
        name = str(t.get("name", ""))
        if not ZONE.match(name):
            continue
        for cd in t.get("customdata") or []:
            text = fill(t["hovertemplate"], cd)
            if text not in seen[name]:
                seen[name].append(text)
    for name, texts in seen.items():
        print(f"--- [{chart}] {name}: {len(texts)} distinct hovers")
        for tx in texts:
            print("   " + " // ".join(tx.split("<br>")))


def metas(path):
    df = pd.read_csv(path)
    out = []
    for _, r in df.iterrows():
        m = ast.literal_eval(r["meta"]) if isinstance(r["meta"], str) else {}
        out.append((r, m))
    return out


print()
print("=" * 30, "ZONE LIFECYCLE FIELDS")
for chart, pre in (("h1", ""), ("counter", "_M15_counter"), ("confluence", "_M15_confluence")):
    print(f"--- [{chart}] KL")
    for r, m in metas(os.path.join(D, f"{STEM}{pre}_kl_zones.csv")):
        steps = m.get("bounds_steps") or []
        sizes = [round(abs(s["top"] - s["bottom"]) / 0.0001, 1) for s in steps]
        print(f"   {r['source_kind']:3} sub={m.get('sub_id', '-')} sid={m['structure_id']} c={m['cycle_id']} "
              f"conf={m['confirmed_idx']} act={[(a['idx'], a['active']) for a in m['activation_history']]} "
              f"end={m['end_idx']} {m['end_reason']} status={m['status']} base={m.get('base_idx')} pips={sizes}")
    print(f"--- [{chart}] POI")
    for r, m in metas(os.path.join(D, f"{STEM}{pre}_poi_zones.csv")):
        size = round(abs(r["top"] - r["bottom"]) / 0.0001, 1)
        print(f"   sub={m.get('sub_id', '-')} sid={m['structure_id']} c={m['cycle_id']} ic={r['ic_idx']} "
              f"cts_est={m.get('cts_established_idx')} conf={m['confirmed_idx']} "
              f"act={[(a['idx'], a['active']) for a in m['activation_history']]} end={m['end_idx']} "
              f"{m['end_reason']} status={m['status']} pips={size}")
