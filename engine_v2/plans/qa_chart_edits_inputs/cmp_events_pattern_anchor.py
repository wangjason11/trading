"""Step 3's structure-events delta ("WVMI back on the chart", 2026-10-26): every CTS_CONFIRMED by a pullback pattern
gains `'pattern_anchor_idx': N` in its meta cell, right after `'via'`; nothing else in the three structure-events CSVs
changes. Checks BASE vs CUR row by row (same rows, same order): every cell equal except those meta cells, whose dicts
equal BASE's plus exactly that key in that position; prints the per-file count and N per row (the pullback pattern's
first candle, entity-absolute on the M15 lenses). Written for step 3; reads a save folder's or `artifacts/debug`'s
CSVs.

    PYTHONIOENCODING=utf-8 python engine_v2/plans/qa_chart_edits_inputs/cmp_events_pattern_anchor.py BASE_DIR CUR_DIR
"""
import ast
import glob
import os
import sys

import pandas as pd

BASE, CUR = sys.argv[1], sys.argv[2]
ok = True
for suffix in ("rk2-5_structure_events.csv", "rk2-5_M15_counter_structure_events.csv",
               "rk2-5_M15_confluence_structure_events.csv"):
    (b_path,) = glob.glob(os.path.join(BASE, "*" + suffix))
    (c_path,) = glob.glob(os.path.join(CUR, "*" + suffix))
    b, c = pd.read_csv(b_path, dtype=str), pd.read_csv(c_path, dtype=str)
    assert list(b.columns) == list(c.columns) and len(b) == len(c), (suffix, b.shape, c.shape)
    added = []
    for i in range(len(b)):
        rb, rc = b.iloc[i], c.iloc[i]
        for col in b.columns:
            if col == "meta" or rb[col] == rc[col] or (pd.isna(rb[col]) and pd.isna(rc[col])):
                continue
            ok = False
            print("CELL", suffix, i, col, rb[col], "->", rc[col])
        if rb["meta"] == rc["meta"]:
            assert not (rc["type"] == "CTS_CONFIRMED" and "'confirmation_method': 'pullback'" in rc["meta"]), (suffix, i)
            continue
        mb, mc = ast.literal_eval(rb["meta"]), ast.literal_eval(rc["meta"])
        kb, kc = list(mb), list(mc)
        exp = kb[:kb.index("via") + 1] + ["pattern_anchor_idx"] + kb[kb.index("via") + 1:]
        good = (rc["type"] == "CTS_CONFIRMED" and mb.get("confirmation_method") == "pullback" and kc == exp
                and {k: v for k, v in mc.items() if k != "pattern_anchor_idx"} == mb
                and int(mc["pattern_anchor_idx"]) < int(rc["idx"]))
        if not good:
            ok = False
            print("META", suffix, i, rb["meta"], "->", rc["meta"])
        added.append((int(rc["idx"]), mc["structure_id"], mc["cycle_id"], mc["pattern_anchor_idx"]))
    print(f"{suffix}: {len(added)} rows gained pattern_anchor_idx (moment, sid, cycle, anchor): {added}")
print("RESULT:", "OK" if ok else "UNEXPECTED DELTAS")
