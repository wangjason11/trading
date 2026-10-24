"""Which WVMI role candles (fb / lb / fp / lp) sit on a drawn wave-candle line, per chart — the old hover keyed WVMI by idx
(the measurement behind the register's §G "WVMI back on the chart", TABLED 2026-10-23).

    PYTHONPATH=. python engine_v2/plans/qa_chart_edits_inputs/wc_hovers.py SAVE_DIR > wc_hovers.log
    python engine_v2/plans/qa_chart_edits_inputs/wvmi_cover.py wc_hovers.log SAVE_DIR

Parses `wc_hovers.py` output in the wave-candle hover format of S_fib_hover `20261023_191547_563f5f8` (before the
2026-10-23 wave-candle hover edit) — re-check the field positions before running it on a later save.
"""
import os
import re
import sys

import pandas as pd

L = open(sys.argv[1], encoding="utf-8").read().split("\n")
D = sys.argv[2]
STEM = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"
drawn = {}
chart = None
for line in L:
    m = re.match(r"--- \[(\w+)\]", line)
    if m:
        chart = m.group(1)
        drawn[chart] = {}
        continue
    if chart and line.startswith("   TF="):
        f = line.strip().split(" // ")
        tf = f[0]
        idx = int(re.match(r"idx=(\d+)", f[2]).group(1))
        drawn[chart][(tf, idx)] = (f[3], f[4])
    elif line.startswith("NZD"):
        chart = None

for chart, suffix, tf in (("h1", "_wvmi.csv", "TF=1H"), ("counter", "_M15_counter_wvmi.csv", "TF=15M"),
                          ("confluence", "_M15_confluence_wvmi.csv", "TF=15M")):
    df = pd.read_csv(os.path.join(D, STEM + suffix))
    print(f"== {chart}: {len(df)} WVMI records; drawn {tf} lines {sum(1 for k in drawn[chart] if k[0] == tf)}")
    for _, r in df.iterrows():
        who = f"sid {r.bos_structure_id}" if pd.isna(r.sub_id) else f"sub {int(r.sub_id)}"
        cells = []
        for role in ("fb", "lb", "fp", "lp"):
            i = r[f"{role}_idx"]
            if pd.isna(i):
                cells.append(f"{role.upper()} -")
                continue
            hit = drawn[chart].get((tf, int(i)))
            cells.append(f"{role.upper()} {int(i)} " + ("drawn as " + hit[1].split('(')[0].replace('Role: ', '').strip()
                                                         if hit else "NOT drawn"))
        print(f"   {who} c{r.bos_cycle_id} collapsed={r.cycle_collapsed} brk={r.breakout_momentum:.3f} "
              f"pb={r.pullback_momentum if pd.isna(r.pullback_momentum) else round(r.pullback_momentum, 3)} | "
              + " | ".join(cells))
