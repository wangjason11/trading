"""Round-4 report ("WVMI back on the chart", 2026-10-26): the user's first-LP rules (rule_r4) vs the furthest-close
rule, per visible WVMI record of S_repl_break, + the records an ungated H1 would add. Reads `wvmi_asof.py`'s JSON.

    PYTHONIOENCODING=utf-8 python engine_v2/plans/qa_chart_edits_inputs/wvmi_r4_report.py ASOF_JSON
"""
import json
import sys

import pandas as pd

J = json.load(open(sys.argv[1]))
S = "artifacts/commits/week9-entries-decisions/20261024_171443_56517e7/NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"
print("lenses in json:", sorted({c["lens"] for c in J}))


def pick(lens, row):
    sid_key = "sub_id"
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


def ab(c, seq):
    return [(t + c["off"], i + c["off"] if i is not None else None, None if p is None else round(p, 3)) for t, i, p in seq]


tot = {}
for chart, lens, csv in (("H1", "H1", S + "_wvmi.csv"), ("counter", "counter", S + "_M15_counter_wvmi.csv"),
                         ("confluence", "confluence", S + "_M15_confluence_wvmi.csv")):
    rows = pd.read_csv(csv)
    print(f"\n===== {chart}: {len(rows)} rows")
    n_fur = n_r4 = 0
    for _, row in rows.iterrows():
        cs = pick(lens, row)
        tag = f"{'sid' if lens == 'H1' else 'sub'}={int(row['bos_structure_id'] if lens == 'H1' else row['sub_id'])} c{int(row['bos_cycle_id'])}"
        if len(cs) == 0:
            print(" ", tag, "NO CAPTURE MATCH"); continue
        c = cs[0]
        if any((x["r4"], x["fur"]) != (c["r4"], c["fur"]) for x in cs[1:]):
            print(" ", tag, "AMBIGUOUS captures differ", len(cs))
        if row["cycle_collapsed"]:
            print(" ", tag, "collapsed - hidden"); continue
        fur, r4 = ab(c, c["fur"]), ab(c, c["r4"])
        n_fur += len(fur); n_r4 += len(r4)
        info = dict(c["r4_info"])
        for k in ("p", "deepest_prior_qualified", "pat_start", "pat_end", "pat_conf", "pre_pattern_deeper"):
            if info.get(k) is not None:
                info[k] = info[k] + c["off"]
        if "pat_qualified" in info:
            info["pat_qualified"] = [i + c["off"] for i in info["pat_qualified"]]
        ec = c["end_cand"]
        if ec:
            ec = dict(ec, e=ec["e"] + c["off"])
        same = [x[:2] for x in fur] == [x[:2] for x in r4]
        print(f"  {tag} {row['zone_side']:4} c={c['c'] + c['off']} life={[None if v is None or isinstance(v, str) else v + c['off'] for v in c['life']]} "
              f"locked={c['locked']} lp_final={row['lp_idx']} fp={row['fp_idx']}")
        print(f"     info={info}")
        print(f"     fur first={fur[:1]} n={len(fur)} last={fur[-1:]}")
        print(f"     r4  first={r4[:1]} n={len(r4)} last={r4[-1:]}  {'SAME' if same else 'DIFF'}")
        if not same:
            print(f"     fur={[x[:2] for x in fur]}")
            print(f"     r4 ={[x[:2] for x in r4]}")
        if ec:
            print(f"     end candle {ec}")
        if c["locked"] and r4 and r4[-1][1] != int(row["lp_idx"]):
            print(f"     LOCKED LP {row['lp_idx']} != last r4 {r4[-1][1]}")
    tot[chart] = (n_fur, n_r4)
    print(f"  dots fur={n_fur} r4={n_r4}")

print("\n===== H1 ungated extras")
for c in J:
    if c["extra"]:
        print(f"  sid={c['sid']} c{c['cyc']} {c['side']} c={c['c']} life={c['life']} locked={c['locked']} collapsed={c['collapsed']} "
              f"fp={c['fp']} lp_final={c['lp_final']} bm={c['bm']:.3f}")
        print(f"     info={c['r4_info']}")
        print(f"     fur n={len(c['fur'])} first={c['fur'][:1]} last={c['fur'][-1:]}")
        print(f"     r4  n={len(c['r4'])} first={c['r4'][:1]} last={c['r4'][-1:]}")
print(tot)
