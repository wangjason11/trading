"""PLAN_H §8.4a item 2 — the `*_poi_zones.csv` FORMAT delta of landing B, keyed by column (not positional).

    python engine_v2/plans/plan_h_inputs/cmp_poi_b.py V3_DEBUG CUR_DEBUG S_A_DIR

Per lens (H1, M15 confluence, M15 counter), B (CUR) against V3 (the behaviour-only variant) row by row:
  * the columns: V3's with `versions` -> `ip_rank` at the same position;
  * every non-meta cell equal, but `ip_rank` == V3's `versions` [rank] (V3 stored the pick's rank there);
  * meta: V3's with `versions` / `current_versions` replaced IN PLACE by the 10 IP keys, activation entries without
    `versions` (V3's `["touch"]`), every other key / value / order equal;
  * the IP values == S_A's picked `*_ip_candidates.csv` row of the same (sub_id, structure_id, cycle_id, ic_idx).
Exit 1 on any difference.
"""
import ast
import re
import sys

import pandas as pd

V3, CUR, SA = sys.argv[1:4]
RK = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"
IP_KEYS = ["ip_rank", "ip_leg_candle_counts", "pattern_anchor_idx", "opp_leg_first_idx", "pattern_end_idx",
           "pattern_confirmed_idx", "ip_tiebreak", "ip_tb_runner_up_ic_idx", "ip_tied_count",
           "ip_tb_imbalance_first_c2_idx"]


def meta(s):
    s2 = re.sub(r"np\.(?:int|float)\d*\(([^)]*)\)", r"\1", s)
    s2 = re.sub(r"Timestamp\('([^']*)'[^)]*\)", r"'\1'", s2)
    return ast.literal_eval(s2)


def nan_none(v):
    return None if (v is None or (isinstance(v, float) and v != v)) else v


bad = 0
for sfx in ["", "_M15_confluence", "_M15_counter"]:
    a = pd.read_csv(f"{V3}/{RK}{sfx}_poi_zones.csv")
    b = pd.read_csv(f"{CUR}/{RK}{sfx}_poi_zones.csv")
    ip = pd.read_csv(f"{SA}/{RK}{sfx}_ip_candidates.csv")
    picks = {(nan_none(r["sub_id"]), int(r["structure_id"]), int(r["cycle_id"]), int(r["ic_idx"])): r
             for _, r in ip[ip["picked"] == True].iterrows()}  # noqa: E712
    exp_cols = ["ip_rank" if c == "versions" else c for c in a.columns]
    errs = []
    if list(b.columns) != exp_cols:
        errs.append(f"columns {list(b.columns)} != {exp_cols}")
    if len(a) != len(b):
        errs.append(f"rows {len(a)} != {len(b)}")
    for i in range(min(len(a), len(b))):
        ra, rb = a.iloc[i], b.iloc[i]
        for c in a.columns:
            if c in ("versions", "meta"):
                continue
            if str(ra[c]) != str(rb[c]):
                errs.append(f"row {i} {c}: {ra[c]} != {rb[c]}")
        ma, mb = meta(ra["meta"]), meta(rb["meta"])
        if ast.literal_eval(ra["versions"]) != [rb["ip_rank"]]:
            errs.append(f"row {i} ip_rank {rb['ip_rank']} vs V3 versions {ra['versions']}")
        exp = {}
        for k, v in ma.items():
            if k == "versions":
                for kk in IP_KEYS:
                    exp[kk] = mb.get(kk, "<missing>")
            elif k == "current_versions":
                continue
            elif k == "activation_history":
                exp[k] = [{kk: vv for kk, vv in e.items() if kk != "versions"} for e in v]
            else:
                exp[k] = v
        if list(mb.items()) != list(exp.items()):
            errs.append(f"row {i} meta keys/values: {[k for k in mb if mb.get(k) != exp.get(k)] or 'ORDER'}")
        key = (mb.get("sub_id"), int(rb["structure_id"]), int(rb["cycle_id"]), int(rb["ic_idx"]))
        pr = picks.get(key)
        if pr is None:
            errs.append(f"row {i} {key}: no picked IP row in S_A")
            continue
        for k in IP_KEYS:
            want = nan_none(pr[k])
            if isinstance(want, float) and want.is_integer():
                want = int(want)
            if mb[k] != want:
                errs.append(f"row {i} {key} {k}: {mb[k]!r} != S_A {want!r}")
    n_tb = sum(1 for i in range(len(b)) if meta(b.iloc[i]["meta"])["ip_tiebreak"] is not None)
    print(f"{sfx or 'H1'}: {len(b)} rows, {n_tb} with a tiebreak — {'OK' if not errs else f'{len(errs)} ERRORS'}")
    for e in errs[:20]:
        print("   ", e)
    bad += len(errs)
sys.exit(1 if bad else 0)
