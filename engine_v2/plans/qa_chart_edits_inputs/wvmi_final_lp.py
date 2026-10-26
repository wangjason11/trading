"""The final temporary LP of every UNLOCKED WVMI record of S_h1_ungated (S_repl_break until step 3, 2026-10-26): today's engine rule (closest to the BOS zone's
outer, the saved `_wvmi.csv`) vs the new rule (rule_r4 = the furthest close + the first-LP rules, `wvmi_asof.py`),
with the pullback momentum; CHANGE marks a record whose final LP moves (step 3's prediction, "WVMI back on the chart"
round 5: 6 records). Written 2026-10-26.

    PYTHONIOENCODING=utf-8 python engine_v2/plans/qa_chart_edits_inputs/wvmi_final_lp.py ASOF_JSON
"""
import json, sys
import pandas as pd
J = json.load(open(sys.argv[1]))
S = "artifacts/commits/week9-entries-decisions/20261026_183615_76b5a17/NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"
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
for lens, csv in (("H1", S + "_wvmi.csv"), ("counter", S + "_M15_counter_wvmi.csv"), ("confluence", S + "_M15_confluence_wvmi.csv")):
    rows = pd.read_csv(csv)
    for _, row in rows.iterrows():
        cs = pick(lens, row)
        if not cs:
            print(lens, "NO MATCH", row["bos_cycle_id"]); continue
        c = cs[0]; o = c["off"]
        tag = f"{lens} {'sid' if lens == 'H1' else 'sub'}={int(row['bos_structure_id'] if lens == 'H1' else row['sub_id'])} c{int(row['bos_cycle_id'])}"
        if c["locked"]:
            continue
        ef = c["ext"][-1] if c["ext"] else None
        r4 = c["r4"][-1] if c["r4"] else None
        e_lp = None if ef is None or ef[1] is None else ef[1] + o
        r_lp = None if r4 is None else r4[1] + o
        print(f"{tag:28} collapsed={bool(row['cycle_collapsed'])!s:5} csv lp={row['lp_idx']} pm={row['pullback_momentum']}  "
              f"ext_last={e_lp} pm={None if ef is None or ef[2] is None else round(ef[2],3)}  r4_last={r_lp} pm={None if r4 is None else round(r4[2],3)}"
              f"  {'CHANGE' if e_lp != r_lp else ''}")
