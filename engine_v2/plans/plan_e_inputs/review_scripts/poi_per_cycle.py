"""POI-per-cycle census of a replay folder (Week 9 Part 1 item 1 baseline, 2026-09-30).

    python engine_v2/plans/plan_e_inputs/review_scripts/poi_per_cycle.py [FOLDER] [--rows]

FOLDER = a save folder or `artifacts/debug` (default). For each of the 3 POI CSVs (H1, M15 confluence, M15 counter):
ICs per cycle and fibs per cycle (distinct `(bos_anchor_idx, cts_anchor_idx)`), then every cycle with more than one
POI — each IC's variants, inner (top for buy / bottom for sell — the edge price reaches first), size in pips, first
activation — marking the NEAREST inner: the only POI MS's sd-proximity CTS confirmation
(`market_structure._check_sd_proximity_at_candle`) and `zone_proximity.check_zone_proximity` read (max inner buy /
min inner sell, BOS inner included). `--rows` also prints every POI row.
Week 8 save `20260930_151531_8b11bc2`: ICs per cycle H1 {1: 1, 2: 2}, confluence {1: 10, 2: 7, 3: 2}, counter
{1: 4, 2: 1, 3: 2}; one fib per cycle on every lens; the nearest POI is the V30 pick in 8 of 11 unique multi-IC
cycles (not: confluence sub 1 (0,1), sub 6 (0,0), sub 7 (0,1)).
"""
import ast
import glob
import os
import sys

import pandas as pd

args = [a for a in sys.argv[1:] if not a.startswith("--")]
FOLDER = args[0] if args else "artifacts/debug"
paths = sorted(glob.glob(os.path.join(FOLDER, "*_poi_zones.csv")))
assert paths, f"no *_poi_zones.csv in {FOLDER}"

for path in paths:
    name = os.path.basename(path)
    lens = "conf" if "_M15_confluence_" in name else "ctr" if "_M15_counter_" in name else "H1"
    d = pd.read_csv(path)
    m = d["meta"].apply(ast.literal_eval)
    t = pd.DataFrame({
        "sub": m.apply(lambda x: x.get("sub_id", "-")),
        "sid": m.apply(lambda x: x["structure_id"]),
        "cyc": m.apply(lambda x: x["cycle_id"]),
        "sd": m.apply(lambda x: x["struct_direction"]),
        "ic": d["ic_idx"],
        # the pick's rank since PLAN_H landing B (2026-10-06); older saves: the V variants
        "ver": m.apply(lambda x: x["ip_rank"] if "ip_rank" in x else "/".join(v[1:] for v in x["versions"])),
        "inner": [r.top if x["struct_direction"] == 1 else r.bottom for r, x in zip(d.itertuples(), m)],
        "pips": ((d["top"] - d["bottom"]) * 1e4).round(1),
        "bos": m.apply(lambda x: x["bos_anchor_idx"]),
        "cts": m.apply(lambda x: x["cts_anchor_idx"]),
        "first_act": m.apply(lambda x: next((e["idx"] for e in x["activation_history"] if e["active"]), None)),
        "status": d["status"],
    })
    keys = ["sub", "sid", "cyc"]
    ics = t.groupby(keys).size().value_counts().sort_index().to_dict()
    fibs = t.drop_duplicates(keys + ["bos", "cts"]).groupby(keys).size().value_counts().sort_index().to_dict()
    print(f"===== {lens}: {len(t)} POI rows | ICs per cycle {ics} | fibs per cycle {fibs}")
    if "--rows" in sys.argv:
        print(t.to_string(index=False))
    for (sub, sid, cyc), g in t.groupby(keys):
        if len(g) < 2:
            continue
        sd = int(g["sd"].iloc[0])
        near = g["inner"].max() if sd == 1 else g["inner"].min()
        print(f"  sub {sub} ({sid},{cyc}) sd={sd:+d}: " + " | ".join(
            f"IC {r.ic} V{r.ver} inner {r.inner:.5f} {r.pips}p act@{r.first_act}"
            + (" <-NEAREST" if r.inner == near else "") for r in g.itertuples()))
