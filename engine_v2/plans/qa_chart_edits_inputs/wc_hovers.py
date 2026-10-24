"""Every wave-candle hover on a chart dir (distinct texts) + the WVMI CSVs' role indices, for the 2026-10-23 spec.

    PYTHONPATH=. python engine_v2/plans/qa_chart_edits_inputs/wc_hovers.py SAVE_DIR
"""
import glob
import os
import sys
from collections import Counter

import pandas as pd

_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from engine_v2.debug.chart_census import load_fig  # noqa: E402
sys.argv = _argv

D = sys.argv[1]
STEM = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"
for chart, suffix in (("h1", ".html"), ("counter", "_M15_counter.html"), ("confluence", "_M15_confluence.html")):
    data, _ = load_fig(os.path.join(D, STEM + suffix))
    texts = []
    for t in data:
        ht = t.get("hovertemplate") or ""
        if "<b>Wave Candle</b>" in ht:
            texts.append(ht.replace("<extra></extra>", ""))
    kinds = Counter(tx.split("<br>")[0] for tx in texts)
    print(f"--- [{chart}] {len(texts)} wave-candle hover traces, distinct {len(set(texts))}; by TF {dict(kinds)}")
    for tx in texts:
        print("   " + " // ".join(tx.split("<br>")))

print()
for p in sorted(glob.glob(os.path.join(D, "*wvmi*.csv"))):
    df = pd.read_csv(p)
    print(os.path.basename(p), len(df), "rows; columns:", list(df.columns))
    print(df.head(12).to_string())
