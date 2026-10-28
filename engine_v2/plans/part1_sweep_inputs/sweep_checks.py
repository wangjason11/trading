"""SESSION ORDER item 10: the on-window facts the sweep list quotes (S_W9d + its run.log).

Usage (repo root): python engine_v2/plans/part1_sweep_inputs/sweep_checks.py SAVE_DIR RUN_LOG
  SAVE_DIR = artifacts/commits/week9-entries-decisions/20261028_182318_7c9477e ; RUN_LOG = run.log (the S_W9d replay's)
"""
import ast
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

save, log = Path(sys.argv[1]), Path(sys.argv[2])

print("== REVERSAL_CANDIDATE meta['pattern'] per events CSV (GOTCHAS: always '?') ==")
for p in sorted(save.glob("*structure_events.csv")):
    d = pd.read_csv(p)
    rc = d[d["event_type"] == "REVERSAL_CANDIDATE"] if "event_type" in d else d[d.iloc[:, 2] == "REVERSAL_CANDIDATE"]
    meta_col = "meta" if "meta" in d else d.columns[4]
    pats = Counter(ast.literal_eval(m).get("pattern") for m in rc[meta_col])
    print(f"  {p.name.split('rk2-5_')[-1]}: {len(rc)} candidates, pattern values {dict(pats)}")

print("== unified_probe iterations + finalize conditions (run.log) ==")
text = log.read_text(encoding="utf-8", errors="replace")
print("  iter=:", dict(sorted(Counter(int(x) for x in re.findall(r"iter=(\d+)", text)).items())))
print("  cond=:", dict(Counter(re.findall(r"cond=([a-z_0-9]+)", text))))
print("  max_iterations hits:", len(re.findall(r"max_iterations", text)))
for p in sorted(save.glob("*_M15_*_triggers.csv")):
    d = pd.read_csv(p)
    if "probe_finalize_condition" in d:
        print(f"  {p.name.split('rk2-5_')[-1]}: probe_finalize_condition {d['probe_finalize_condition'].value_counts().to_dict()}")
