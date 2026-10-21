"""Patch a `git archive` copy of the tree (argv[1]) so EVERY production imbalance fill threshold is 0.65 instead of
0.70 (the user's change 2, 2026-10-20 write-up) — a VARIANT replay to measure the 65 % change by itself (its whole
cascade: FibTracker activation + the cross-cycle dead-cycle walk, MS's embedded tracker + in-flight IC read, IC
conditions / rules, the POI activation sweep). Each edit is exact-once (the readers listed by grep 2026-10-20).

    git archive HEAD | tar -x -C <short %TEMP% dir>; python <this> <that dir>
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
EDITS = [
    ("engine_v2/common/types.py", "        fill_threshold: float = 0.70,", "        fill_threshold: float = 0.65,"),
    ("engine_v2/zones/poi_zones.py", "    fill_threshold: float = 0.70  # 70% = filled",
     "    fill_threshold: float = 0.65  # 70% = filled"),
    ("engine_v2/zones/fib_tracker.py", "FIB_FILL_THRESHOLD = 0.70", "FIB_FILL_THRESHOLD = 0.65"),
    ("engine_v2/zones/cross_cycle_fib.py", "    fill_threshold: float = 0.70,", "    fill_threshold: float = 0.65,"),
]
for rel, old, new in EDITS:
    p = ROOT / rel
    b = p.read_bytes().decode("utf-8")
    assert b.count(old) == 1, (rel, b.count(old))
    p.write_bytes(b.replace(old, new).encode("utf-8"))
# patterns/imbalance.py holds two identical defaults (has_unfilled_imbalance + iter_unfilled_imbalances)
p = ROOT / "engine_v2/patterns/imbalance.py"
b = p.read_bytes().decode("utf-8")
old = "    fill_threshold: float = 0.70,"
assert b.count(old) == 2, b.count(old)
p.write_bytes(b.replace(old, "    fill_threshold: float = 0.65,").encode("utf-8"))
print("patched", len(EDITS) + 1, "files")
