"""Per visible WVMI record of S_repl_break: FP's close vs the first temporary LP's close (rule_r4, from `wvmi_asof.py`'s
JSON), and the pullback pattern's span. Written for "WVMI back on the chart" round 6, 2026-10-26.

    PYTHONIOENCODING=utf-8 PYTHONPATH=. python engine_v2/plans/qa_chart_edits_inputs/wvmi_fp_vs_first.py ASOF_JSON
"""
import json, sys
import pandas as pd
sys.path.insert(0, "engine_v2/plans/qa_chart_edits_inputs")
import m15_frame
J = json.load(open(sys.argv[1]))
S = "artifacts/commits/week9-entries-decisions/20261024_171443_56517e7/NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"
H = pd.read_csv("artifacts/commits/week9-entries-decisions/20261024_171443_56517e7/NZD_USD_H1_2025-11-15_2026-01-20_final.csv")
M = m15_frame.frame(S + "_M15_confluence.html")
def close(lens, i):
    return float(H.loc[i, "c"]) if lens == "H1" else float(M.loc[i, "close"])
def pick(lens, row):
    out = []
    for c in J:
        if c["lens"] != lens or c["extra"]:
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
rows_all = []
for lens, csv in (("H1", S + "_wvmi.csv"), ("counter", S + "_M15_counter_wvmi.csv"), ("confluence", S + "_M15_confluence_wvmi.csv")):
    for _, row in pd.read_csv(csv).iterrows():
        cs = pick(lens, row)
        if not cs or row["cycle_collapsed"]:
            continue
        rows_all.append((lens, row, cs[0]))
for c in J:
    if c["extra"] and not c["collapsed"]:
        rows_all.append(("H1", {"bos_structure_id": c["sid"], "bos_cycle_id": c["cyc"], "zone_side": c["side"], "extra": True}, c))
for lens, row, c in rows_all:
    o = c["off"]; fp = c["fp"] + o
    pb = -1 if c["side"] == "buy" else 1
    first = c["r4"][0][1] + o if c["r4"] else None
    i = c["r4_info"]
    tag = f"{lens} {'sid' if lens == 'H1' else 'sub'}={int(row['bos_structure_id'] if lens == 'H1' else row['sub_id'])} c{int(row['bos_cycle_id'])}{' (ungated)' if c['extra'] else ''}"
    cf, cl = close(lens, fp), close(lens, first)
    beyond = (cl - cf) * pb
    span = f"pat {i['pat_start'] + o}-{(i['pat_conf'] if i['pat_conf'] is not None else i['pat_end']) + o}" if "pat_start" in i else f"prox p={i['p'] + o}"
    print(f"{tag:30} {c['side']:4} cyc0={c['cyc'] == 0!s:5} {span:16} FP {fp} close {cf:.5f} | first LP {first} close {cl:.5f} | first beyond FP by {beyond * 1e4:+.1f} pips")
