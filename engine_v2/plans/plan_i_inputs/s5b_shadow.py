"""PLAN_I S5b shadow (R9-3, F19) on the UNCHANGED code: wrap FibTracker._finalize_lifecycle_fields, and before it
runs compare every (sid, cycle)'s stream terminal (already in `_terminal`: new_cycle / Option A / reversal) with the
cycle table's end (`compute_cycle_lifecycle`, the same args finalize gets). A row "would change" under S5b iff the
stream terminal exists and the table end is STRICTLY earlier. Also lists every tie / later-table row for the record.

    cd WORKDIR && PYTHONPATH=REPO python s5b_shadow.py OUT.json
"""
import json
import os
import runpy
import sys

out_path = os.path.abspath(sys.argv[1])
import engine_v2.zones.fib_tracker as ft  # noqa: E402
from engine_v2.zones.structure_lifecycle import compute_cycle_lifecycle, compute_reversal_idx_by_sid  # noqa: E402

ROWS = []
CALLS = []
_orig = ft.FibTracker._finalize_lifecycle_fields


def _shadow(self, events=None, lifecycle_floor=None, lifecycle_cap=None, cap_reason="lifecycle_end"):
    pfx = sys._getframe(1).f_locals.get("pfx", "?")
    call = len(CALLS)
    fib_cycles = {(int(k[0]), int(k[1])) for k in self._fibs if isinstance(k, tuple) and len(k) >= 2}
    CALLS.append({"call": call, "pfx": pfx, "floor": lifecycle_floor, "cap": lifecycle_cap,
                  "cap_reason": cap_reason, "n_fib_cycles": len(fib_cycles), "n_records": len(self._fibs)})
    if events is not None:
        rev = compute_reversal_idx_by_sid(events)
        table = compute_cycle_lifecycle(events, rev, lifecycle_floor, lifecycle_cap, cap_reason)
        for (s, c) in sorted(fib_cycles):
            row = table.get((s, c))
            term = self._terminal.get((s, c))
            tend = None if row is None else row[1]
            treason = None if row is None else (row[2] or cap_reason)
            rel = None
            if term is not None and tend is not None:
                rel = "table_earlier" if tend < term[0] else ("tie" if tend == term[0] else "table_later")
            ROWS.append({"call": call, "pfx": pfx, "sid": s, "cycle": c,
                         "stream": None if term is None else [term[0], term[1]],
                         "table": None if row is None else [row[0], tend, treason],
                         "rel": rel, "would_change": rel == "table_earlier"})
    return _orig(self, events, lifecycle_floor, lifecycle_cap, cap_reason)


ft.FibTracker._finalize_lifecycle_fields = _shadow


def _dump():
    with open(out_path, "w") as fh:
        json.dump({"calls": CALLS, "rows": ROWS}, fh, indent=1, default=str)
    n = sum(r["would_change"] for r in ROWS)
    rels = {}
    for r in ROWS:
        rels[str(r["rel"])] = rels.get(str(r["rel"]), 0) + 1
    print(f"[s5b_shadow] finalize calls {len(CALLS)}, fib cycles {len(ROWS)}, rel {rels}, WOULD CHANGE {n} -> {out_path}")


import atexit  # noqa: E402
atexit.register(_dump)
os.makedirs("artifacts/debug", exist_ok=True)
os.makedirs("artifacts/charts", exist_ok=True)
sys.argv = ["run_replay"]
runpy.run_module("engine_v2.run_replay", run_name="__main__")
