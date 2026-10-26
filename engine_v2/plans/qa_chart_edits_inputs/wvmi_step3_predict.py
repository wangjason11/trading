"""Step 3 prediction ("WVMI back on the chart", 2026-10-26): every WVMI record of S_h1_ungated (all three _wvmi.csv
files, collapsed included) vs the decided LP rule (`wvmi_asof.py`'s `rule_new`, run on the PRE-step-3 code) — routes,
the pattern span, the lock warning per record, the final-LP changes. NOTE: counts per RECORD, not per tracker run
(the warning printed 10 times for its 1 record — GOTCHAS "A log line from a sub's downstream prints once per TRACKER
RUN").

    PYTHONIOENCODING=utf-8 python engine_v2/plans/qa_chart_edits_inputs/wvmi_step3_predict.py ASOF_JSON
"""
import json, sys
from collections import Counter
import pandas as pd

J = json.load(open(sys.argv[1]))
S = ("artifacts/commits/week9-entries-decisions/20261026_183615_76b5a17/"
     "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5")


def pick(lens, row):
    out = []
    for c in J:
        if c["lens"] != lens:
            continue
        if lens != "H1" and (c["sub_id"] is None or int(c["sub_id"]) != int(row["sub_id"])):
            continue
        if lens == "H1" and c["sid"] != int(row["bos_structure_id"]):
            continue
        if c["cyc"] != int(row["bos_cycle_id"]):
            continue
        off = c["off"]
        lp = None if pd.isna(row["lp_idx"]) else int(row["lp_idx"])
        clp = None if c["lp_final"] is None else int(c["lp_final"]) + off
        if clp != lp or int(c["fp"]) + off != int(row["fp_idx"]) or bool(c["locked"]) != bool(row["lp_locked"]) \
                or bool(c["collapsed"]) != bool(row["cycle_collapsed"]):
            continue
        out.append(c)
    return out


routes, warn_lock, warn_bound, changes, nomatch, ambiguous = Counter(), [], [], [], [], []
span_bad, r4_diff, c_past_bound, lock_missing = [], [], [], []
for lens, csv in (("H1", S + "_wvmi.csv"), ("counter", S + "_M15_counter_wvmi.csv"),
                  ("confluence", S + "_M15_confluence_wvmi.csv")):
    rows = pd.read_csv(csv)
    for _, row in rows.iterrows():
        cs = pick(lens, row)
        tag = (f"{lens} {'sid' if lens == 'H1' else 'sub'}="
               f"{int(row['bos_structure_id'] if lens == 'H1' else row['sub_id'])} c{int(row['bos_cycle_id'])}"
               f"{' [collapsed]' if row['cycle_collapsed'] else ''}")
        if not cs:
            nomatch.append(tag); continue
        # several captures can match (a sub projected more than once): they must agree on the new rule
        keys = {(tuple(map(tuple, [x[:2] for x in c["new"]])), tuple(map(tuple, [x[:2] for x in c["new_to_bound"]])),
                 c["lock_lp"]) for c in cs}
        if len(keys) > 1:
            ambiguous.append(tag)
        c = cs[0]; o = c["off"]
        routes[(lens, c["new_route"], bool(row["cycle_collapsed"]))] += 1
        if c["pat_last_is_c"] is False:
            span_bad.append(tag)
        if c["c"] > c["bound"]:
            c_past_bound.append((tag, c["c"] + o, c["bound"] + o))
        r4 = [tuple(x[:2]) for x in c["r4"]]; nw = [tuple(x[:2]) for x in c["new"]]
        if r4 != nw:
            r4_diff.append((tag, r4[:3], nw[:3], len(r4), len(nw)))
        last_lock = c["new"][-1][1] if c["new"] else None          # as of the lock (<= min(bound, b - 1))
        last_bound = c["new_to_bound"][-1][1] if c["new_to_bound"] else None  # the engine's pre-lock search
        if c["locked"]:
            if not c["lock_lp_in_frame"]:
                lock_missing.append(tag)
            if c["lock_lp"] != last_lock:
                warn_lock.append((tag, None if last_lock is None else last_lock + o, c["lock_lp"] + o if c["lock_lp"] is not None else None))
            if c["lock_lp"] != last_bound:
                warn_bound.append((tag, None if last_bound is None else last_bound + o, c["lock_lp"] + o if c["lock_lp"] is not None else None))
            new_final = c["lock_lp"] if c["lock_lp_in_frame"] else last_bound
            new_pm = None
        else:
            new_final = last_bound
            new_pm = c["new_to_bound"][-1][2] if c["new_to_bound"] else None
        old = None if pd.isna(row["lp_idx"]) else int(row["lp_idx"])
        nf = None if new_final is None else new_final + o
        if nf != old:
            changes.append((tag, old, row["pullback_momentum"], nf, None if new_pm is None else round(new_pm, 3),
                            c["new_route"], len(c["new_to_bound"])))

print("no match:", nomatch)
print("ambiguous captures (disagree on the new rule):", ambiguous)
print("routes:", dict(routes))
print("pattern span [anchor, moment] != r4's [anchor, max(end, conf)]:", span_bad)
print("CTS moment past the LP bound (end - 1):", c_past_bound)
print("r4 != new (as of the lock):", len(r4_diff))
for d in r4_diff:
    print("   ", d)
print("locked records whose lock LP is not in the frame:", lock_missing)
print("WARNING count, lock LP != last temp LP as of the lock (<= b-1):", len(warn_lock))
for w in warn_lock:
    print("   ", w)
print("WARNING count, lock LP != last temp LP to end-1 (the engine's pre-lock search):", len(warn_bound))
for w in warn_bound:
    print("   ", w)
print("FINAL LP changes:", len(changes))
for ch in changes:
    print("   ", ch)
