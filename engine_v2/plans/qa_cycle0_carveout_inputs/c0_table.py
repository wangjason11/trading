"""Cycle-0 table per structure from a save's structure-events CSVs: argv[1] = the save / debug dir.

One row per structure (H1: structure_id; M15 lenses: sub_id): the start, BOS_0 (anchor, price), CTS_0 (moment,
anchor, price), the gap at CTS_0's establishment and its largest value before the CTS confirmed, how / when cycle 0's
CTS confirmed, cycle 1's establishment, the number of cycles.
"""
import ast
import sys
from pathlib import Path

import pandas as pd

D = Path(sys.argv[1])
PIP = 0.0001
LENS = {"H1": "*_rk2-5_structure_events.csv", "counter": "*_M15_counter_structure_events.csv",
        "confluence": "*_M15_confluence_structure_events.csv"}

for lens, pat in LENS.items():
    f = sorted(D.glob(pat))[0]
    ev = pd.read_csv(f)
    ev["m"] = ev["meta"].apply(ast.literal_eval)
    key = "structure_id" if lens == "H1" else "sub_id"
    ev["k"] = ev["m"].apply(lambda m: m.get(key))
    ev["cyc"] = ev["m"].apply(lambda m: m.get("cycle_id"))
    print(f"=== {lens} ({f.name})")
    for k, g in ev.groupby("k", sort=True):
        g0 = g[g["cyc"] == 0]
        est = g0[g0["type"] == "CTS_ESTABLISHED"]
        bos = g0[g0["type"] == "BOS_CONFIRMED"]
        if est.empty:
            print(f"  {key}={k}: no cycle 0")
            continue
        e, b = est.iloc[0], bos.iloc[0]
        sd = e["m"]["struct_direction"]
        conf = g0[g0["type"] == "CTS_CONFIRMED"]
        upd = g0[g0["type"] == "CTS_UPDATED"]
        bthu = g0[g0["type"] == "BOS_THRESHOLD_UPDATED"]
        c_idx = int(conf.iloc[0]["idx"]) if not conf.empty else None
        via = conf.iloc[0]["m"].get("via") if not conf.empty else None
        # the gap's widest point before the confirmation: CTS moves by CTS_UPDATED, BOS by BOS_THRESHOLD_UPDATED
        lim = c_idx if c_idx is not None else 10 ** 9
        cts_p = [e["price"]] + list(upd[upd["idx"] < lim]["price"])
        bos_p = [b["price"]] + list(bthu[bthu["idx"] < lim]["price"])
        cts_x = max(cts_p) if sd == 1 else min(cts_p)
        bos_x = min(bos_p) if sd == 1 else max(bos_p)
        gap0 = abs(e["price"] - b["price"]) / PIP
        gapx = abs(cts_x - bos_x) / PIP
        est1 = g[(g["type"] == "CTS_ESTABLISHED") & (g["cyc"] == 1)]
        n_cyc = int(g["cyc"].dropna().max()) + 1
        uc = e["m"].get("use_case", "main" if k == 0 else "reversal")
        print(f"  {key}={k} sd={sd:+d} uc={uc} parent=({e['m'].get('parent_sid')},{e['m'].get('parent_cycle_id')}) "
              f"BOS0 anchor={b['m']['bos_anchor_idx']} {b['price']:.5f} | CTS0 est={int(e['idx'])} "
              f"anchor={e['m']['cts_anchor_idx']} {e['price']:.5f} via={e['m']['via']} | gap@est={gap0:.1f}p "
              f"max-before-conf={gapx:.1f}p (CTS_UPD {len(upd)}, BOS_TH_UPD {len(bthu)}) | conf={c_idx} {via} "
              f"pat_anchor={conf.iloc[0]['m'].get('pattern_anchor_idx') if not conf.empty else None} | "
              f"c1 est={int(est1.iloc[0]['idx']) if not est1.empty else None} | cycles={n_cyc}")
