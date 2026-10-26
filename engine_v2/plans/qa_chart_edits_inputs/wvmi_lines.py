"""Which WVMI role candles (FB / LB / FP / LP) sit on a DRAWN wave-candle line of the same owner, cycle and role, per
chart — the measurement behind "WVMI back on the chart" (the user un-tabled it 2026-10-24). Parses the wave-candle
hover format of 2026-10-23 (`<b>Wave Candle</b>`, `idx=`, `<BOS|CTS> zone: sid=S | cycle_id=N` / `sub_id=X | …`,
`Role: <FB|LB|FP|LP> (…)`; an LP names its owning cycle: `LP (last pullback of cycle N)`).

    PYTHONIOENCODING=utf-8 PYTHONPATH=. python engine_v2/plans/qa_chart_edits_inputs/wvmi_lines.py SAVE_DIR

Per record and role: `=` a drawn line of the same owner + cycle + role at that candle; `~<what>` a drawn line at that
candle but another owner / cycle / role; `.` no drawn line at that candle (any owner).
"""
import os
import re
import sys
from collections import defaultdict

import pandas as pd

_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from engine_v2.debug.chart_census import load_fig  # noqa: E402
sys.argv = _argv

D = sys.argv[1]
STEM = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"
_RE = re.compile(r"TF=(\w+)<br><b>Wave Candle</b><br>idx=(\d+).*?(BOS|CTS) zone: (sid|sub_id)=(\d+) \| cycle_id=(\d+)"
                 r".*?Role: (FB|LB|FP|LP) \(([^)]*)\)")


def drawn_lines(path):
    """{idx: [(tf, owner_kind, owner, role, role_cycle, zone_kind)]} for every wave-candle line on the chart."""
    out = defaultdict(list)
    traces, _ = load_fig(path)
    for t in traces:
        m = _RE.search(t.get("hovertemplate") or "")
        if not m:
            continue
        tf, idx, zk, ok, owner, cyc, role, name = m.groups()
        c = re.search(r"of cycle (\d+)", name)
        role_cycle = int(c.group(1)) if (role == "LP" and c) else (None if role == "LP" else int(cyc))
        out[int(idx)].append((tf, ok, int(owner), role, role_cycle, zk))
    return out


def report(chart, path, records, tf, owner_kind):
    lines = drawn_lines(path)
    n_lines = sum(len(v) for v in lines.values())
    print(f"===== {chart}: {n_lines} wave-candle lines; {len(records)} {tf} WVMI records")
    for _, r in records.iterrows():
        owner = int(r["sub_id"]) if owner_kind == "sub_id" else int(r["bos_structure_id"])
        cyc = int(r["bos_cycle_id"])
        cells = []
        for role in ("fb", "lb", "fp", "lp"):
            i = r[f"{role}_idx"]
            if pd.isna(i):
                cells.append(f"{role.upper()}=NA")
                continue
            here = [x for x in lines.get(int(i), []) if x[0] == tf]
            same = [x for x in here if x[1] == owner_kind and x[2] == owner and x[3] == role.upper() and x[4] == cyc]
            if same:
                mark = "="
            elif here:
                mark = "~" + ",".join(f"{x[1][:3]}{x[2]}c{x[4]}{x[3]}" for x in here)
            else:
                mark = "."
            cells.append(f"{role.upper()} {int(i)}{mark}")
        tag = f"{owner_kind}={owner} c{cyc} {r['zone_side']:4} {r['status']:7}{' COLLAPSED' if r['cycle_collapsed'] else ''}"
        mom = f"bo={r['breakout_momentum']:.3f} pb={r['pullback_momentum']:.3f}" if not pd.isna(r["pullback_momentum"]) \
            else f"bo={r['breakout_momentum']:.3f} pb=NA"
        print(f"  {tag:42} {mom:22} " + "  ".join(cells))


h1 = pd.read_csv(os.path.join(D, f"{STEM}_wvmi.csv"))
report("H1", os.path.join(D, f"{STEM}.html"), h1, "1H", "sid")
for lens in ("counter", "confluence"):
    sub = pd.read_csv(os.path.join(D, f"{STEM}_M15_{lens}_wvmi.csv"))
    path = os.path.join(D, f"{STEM}_M15_{lens}.html")
    report(f"M15 {lens} (sub lines)", path, sub, "15M", "sub_id")
    report(f"M15 {lens} (H1 overlay lines)", path, h1, "1H", "sid")
