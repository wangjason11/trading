"""PLAN_J §2: structure-events CSV deltas (row multiset + positions) of a variant replay vs a base dir.
  python evdiff.py BASE_DEBUG_DIR VARIANT_DEBUG_DIR
"""
import sys, csv
from collections import Counter
P = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"
base, var = sys.argv[1], sys.argv[2]
for suf in ("_structure_events.csv", "_M15_confluence_structure_events.csv", "_M15_counter_structure_events.csv"):
    a = list(csv.reader(open(f"{base}/{P}{suf}")))
    b = list(csv.reader(open(f"{var}/{P}{suf}")))
    ca, cb = Counter(map(tuple, a)), Counter(map(tuple, b))
    only_a, only_b = ca - cb, cb - ca
    print(f"== {suf}: rows {len(a)} -> {len(b)}; same multiset: {not only_a and not only_b}; "
          f"positional diffs {sum(1 for x, y in zip(a, b) if x != y)}")
    for r in sorted(only_a, key=lambda r: (r[0].isdigit() and int(r[0]), r)):
        print("   - ", r[:4], r[4][:150])
    for r in sorted(only_b, key=lambda r: (r[0].isdigit() and int(r[0]), r)):
        print("   + ", r[:4], r[4][:150])
