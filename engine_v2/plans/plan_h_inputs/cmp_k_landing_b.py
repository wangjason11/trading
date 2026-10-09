"""PLAN_K landing B's check: the current replay vs S_K_A (landing A's save) == the variant replay V (PLAN_K §11.3)
+ the `no_ic_candidate` relabel (PLAN_H §6 RX1-N; the user: "Both in landing B").

Usage (repo root): python engine_v2/plans/plan_h_inputs/cmp_k_landing_b.py BASE_SAVE [CUR_DEBUG]

1. every CSV md5-equal, except the confluence + counter `*_poi_zones.csv` and `*_ip_candidates.csv`;
2. POI zones: the same rows in the same order; the ONLY deltas —
   confluence sub 0 cyc 3: IC 1604 1A.i -> 1447 1A.iv (TB2 over 1660, reference imbalance 1658; zone 0.57744-0.57768);
   sub 3 cyc 2 (confluence AND counter): the pick 2969 kept, its TB1 tie (runner-up 2988, 2 tied) gone;
3. IP candidates: the same rows; only `picked` and the 4 tie-break columns change, on exactly the rows of item 2,
   + confluence sub 1 cyc 2's `IP: none` row: `reason` no_pattern -> no_ic_candidate (3 candles pass the §3.1
   conditions, none touches the zone — no IC candidate).
Exit 1 on any unexpected delta.
"""
import ast
import hashlib
import sys
from pathlib import Path

import pandas as pd

BASE = Path(sys.argv[1])
CUR = Path(sys.argv[2] if len(sys.argv) > 2 else "artifacts/debug")
P = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"
TB = ("ip_tiebreak", "ip_tb_runner_up_ic_idx", "ip_tied_count", "ip_tb_imbalance_first_c2_idx")
bad = []


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def meta(s):
    return ast.literal_eval(s) if isinstance(s, str) and s else {}


def same(a, b):
    return (a == b) or (pd.isna(a) and pd.isna(b))


EXPECTED_FILES = {f"{P}_M15_{lens}_{kind}.csv" for lens in ("confluence", "counter")
                  for kind in ("poi_zones", "ip_candidates")}
base_csvs = sorted(p.name for p in BASE.glob("*.csv"))
cur_csvs = sorted(p.name for p in CUR.glob("*.csv"))
if base_csvs != cur_csvs:
    bad.append(f"file lists differ: only base {sorted(set(base_csvs) - set(cur_csvs))}, "
               f"only cur {sorted(set(cur_csvs) - set(base_csvs))}")
changed = {n for n in base_csvs if n in cur_csvs and md5(BASE / n) != md5(CUR / n)}
print(f"{len(base_csvs)} CSVs; md5-equal {len(base_csvs) - len(changed)}; changed {len(changed)}")
for n in sorted(changed - EXPECTED_FILES):
    bad.append(f"unexpected change: {n}")
for n in sorted(EXPECTED_FILES - changed):
    bad.append(f"expected a change, md5-equal: {n}")

# (lens, sub_id, sid, cycle) -> (non-meta column deltas, meta deltas), base -> cur
SUB0_C3_META = {"ip_leg_candle_counts": ("1-1-1", "2-1-3"), "ip_rank": ("1A.i", "1A.iv"),
                "ip_tiebreak": (None, "TB2"), "ip_tb_runner_up_ic_idx": (None, 1660), "ip_tied_count": (None, 2),
                "ip_tb_imbalance_first_c2_idx": (None, 1658), "opp_leg_first_idx": (1604, 1447),
                "pattern_anchor_idx": (1603, 1445), "pattern_confirmed_idx": (1606, 1450),
                "pattern_end_idx": (1605, 1450)}
SUB0_C3_COLS = {"ic_idx": (1604, 1447), "start_time": ("2025-12-09 15:00:00+00:00", "2025-12-07 23:45:00+00:00"),
                "top": (0.5788, 0.57768), "bottom": (0.57734, 0.57744), "ip_rank": ("1A.i", "1A.iv")}
SUB3_C2_META = {"ip_tiebreak": ("TB1", None), "ip_tb_runner_up_ic_idx": (2988, None), "ip_tied_count": (2, None)}
EXPECTED_POI = {("confluence", 0, 0, 3): (SUB0_C3_COLS, SUB0_C3_META),
                ("confluence", 3, 0, 2): ({}, SUB3_C2_META),
                ("counter", 3, 0, 2): ({}, SUB3_C2_META)}
seen = set()
for lens in ("confluence", "counter"):
    n = f"{P}_M15_{lens}_poi_zones.csv"
    b, c = pd.read_csv(BASE / n), pd.read_csv(CUR / n)
    if len(b) != len(c) or list(b.columns) != list(c.columns):
        bad.append(f"{n}: shape / columns differ")
        continue
    for i in range(len(b)):
        rb, rc = b.iloc[i], c.iloc[i]
        mb, mc = meta(rb["meta"]), meta(rc["meta"])
        key = (lens, mb.get("sub_id"), int(rb["structure_id"]), int(rb["cycle_id"]))
        cols = {k: (rb[k], rc[k]) for k in b.columns if k != "meta" and not same(rb[k], rc[k])}
        if list(mb) != list(mc):
            bad.append(f"{n} row {i}: meta key list differs")
        metas = {k: (mb.get(k), mc.get(k)) for k in mb if mb.get(k) != mc.get(k)}
        if not cols and not metas:
            continue
        seen.add(key)
        exp = EXPECTED_POI.get(key)
        if exp is None:
            bad.append(f"{n} row {i} {key}: unexpected delta {cols} {metas}")
            continue
        ecols, emeta = exp
        if {k: tuple(v) for k, v in cols.items()} != ecols or metas != emeta:
            bad.append(f"{n} row {i} {key}: delta {cols} {metas} != expected {ecols} {emeta}")
        print(f"  POI {key}: {cols or ''} {metas}")
for k in set(EXPECTED_POI) - seen:
    bad.append(f"expected POI delta missing: {k}")

# IP candidates: (lens, sub, sid, cycle, ic) -> the changed columns
EXPECTED_IP = {("confluence", 0, 0, 3, 1447): {"picked", *TB}, ("confluence", 0, 0, 3, 1604): {"picked"},
               ("confluence", 3, 0, 2, 2969): {"ip_tiebreak", "ip_tb_runner_up_ic_idx", "ip_tied_count"},
               ("counter", 3, 0, 2, 2969): {"ip_tiebreak", "ip_tb_runner_up_ic_idx", "ip_tied_count"},
               ("confluence", 1, 0, 2, None): {"reason"}}
RELABEL = {("confluence", 1, 0, 2, None): ("no_pattern", "no_ic_candidate")}
seen = set()
for lens in ("confluence", "counter"):
    n = f"{P}_M15_{lens}_ip_candidates.csv"
    b, c = pd.read_csv(BASE / n), pd.read_csv(CUR / n)
    if len(b) != len(c) or list(b.columns) != list(c.columns):
        bad.append(f"{n}: shape / columns differ")
        continue
    for i in range(len(b)):
        rb, rc = b.iloc[i], c.iloc[i]
        d = {k for k in b.columns if not same(rb[k], rc[k])}
        if not d:
            continue
        key = (lens, int(rb["sub_id"]), int(rb["structure_id"]), int(rb["cycle_id"]),
               int(rb["ic_idx"]) if not pd.isna(rb["ic_idx"]) else None)
        seen.add(key)
        if key in RELABEL and (rb["reason"], rc["reason"]) != RELABEL[key]:
            bad.append(f"{n} row {i} {key}: reason {rb['reason']!r} -> {rc['reason']!r} != {RELABEL[key]}")
        if EXPECTED_IP.get(key) != d:
            bad.append(f"{n} row {i} {key}: changed {sorted(d)} != expected {sorted(EXPECTED_IP.get(key, ()))}")
        print(f"  IP {key}: {sorted(d)}")
for k in set(EXPECTED_IP) - seen:
    bad.append(f"expected IP delta missing: {k}")

print("\nRESULT:", "OK" if not bad else "UNEXPECTED")
for x in bad:
    print("  -", x)
sys.exit(1 if bad else 0)
