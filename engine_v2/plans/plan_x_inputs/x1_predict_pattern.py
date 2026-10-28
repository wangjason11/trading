"""PLAN_X X1 prediction: each REVERSAL_CANDIDATE's `pattern` after the fix = the `pat` of the STATE_CHANGED -> reversal
that applied the same PatternEvent (same structure stream, apply candle == the candidate's apply_idx). A candidate never
realised (a watch expiry discarded it) has no STATE_CHANGED to read -> printed as UNREALISED (its name is then only
knowable from the detector; none expected on the window).

Usage: python x1_predict_pattern.py SAVE_DIR
"""
import ast
import sys
from pathlib import Path

import pandas as pd

for p in sorted(Path(sys.argv[1]).glob("*structure_events.csv")):
    d = pd.read_csv(p)
    d["m"] = d["meta"].map(ast.literal_eval)
    rc = d[d["type"] == "REVERSAL_CANDIDATE"]
    sc = d[(d["type"] == "STATE_CHANGED")]
    sc = sc[sc["m"].map(lambda m: m.get("to") == "reversal")]
    print(f"== {p.name.split('rk2-5_')[-1]}: {len(rc)} candidates, {len(sc)} STATE_CHANGED->reversal")
    for _, r in rc.iterrows():
        m = r["m"]
        key = (m.get("structure_path_id"), m.get("parent_sid"), m.get("parent_cycle_id"), m.get("sub_id"))
        hits = sc[(sc["idx"] == m["apply_idx"]) & (sc["m"].map(lambda x: x.get("structure_id")) == m["structure_id"])]
        if "sub_id" in m:
            hits = hits[hits["m"].map(lambda x: x.get("sub_id")) == m["sub_id"]]
        pats = sorted({h.get("pat") for h in hits["m"]})
        print(f"  anchor={m['pattern_anchor_idx']} apply={m['apply_idx']} sid={m['structure_id']} sub_id={m.get('sub_id')} "
              f"pattern now={m['pattern']!r} -> predicted {pats if pats else 'UNREALISED'}")
