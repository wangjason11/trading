"""The IP tiebreaker update's check (PLAN_H §6 TBU-1 … TBU-5, §8.9): the current replay vs S_K_B (or any save of the
PLAN_K landing-B format) == the prediction measured BEFORE the build on 2d199fd (`tb_imb_measure.py` /
`tb_imb_dump.py`, PLAN_H §8.9 "The IP tiebreaker update").

SUPERSEDED since PLAN_P (2026-10-13): it reads the tiebreak value `immediate_unfilled_imbalance`, which no save
from S_P on holds; diff S_P-format saves with `qa_ic_crossfib_inputs/cmp_trial.py`. Kept as the record.

Usage (repo root): python engine_v2/plans/plan_h_inputs/cmp_tbu.py BASE_SAVE [CUR_DEBUG]

1. every CSV md5-equal, except the 3 `*_ip_candidates.csv` (the new column) and the confluence + counter
   `*_poi_zones.csv`;
2. POI zones: the same rows in the same order; the ONLY deltas —
   confluence sub 0 cyc 3: IC 1447 -> 1660 (1A.iv; the immediate unfilled imbalance over 1447, 2 tied, its gap 1662;
   zone 0.57710-0.57738, start 2025-12-10 05:00), its legs = the BASE CSV's own classification of 1660; the
   activation-derived meta (confirmed_idx, activation_history, status, end_idx, end_reason) may change on that row
   only (printed for the review);
   the relabels "TB2" -> "most_recent_before_imbalance": confluence sub 1 cyc 1 (1875), sub 7 cyc 2 (4083, both lenses);
3. IP candidates: the base columns equal except `picked` + the 4 tiebreak columns on sub 0 cyc 3's 1447 / 1660 and
   `ip_tiebreak` on the 1875 / 4083 picked rows; the new column `immediate_imbalance_first_c2_idx` right after
   `ic_ref_reason`, on every row either empty or a gap candle in IC + 1 … min(IC + 2, the CTS anchor), empty on every
   `IP: none` row, and on the tied ICs exactly the measured values (1660 -> 1662; 1447, 1842, 1871, 1875, 4083, 4136
   empty — their immediate gaps filled or absent).
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
NEW = "immediate_imbalance_first_c2_idx"
OLD_LAST = "most_recent_before_imbalance"
ACTIVATION = {"confirmed_idx", "activation_history", "status", "end_idx", "end_reason"}
bad = []


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def meta(s):
    return ast.literal_eval(s) if isinstance(s, str) and s else {}


def same(a, b):
    return (a == b) or (pd.isna(a) and pd.isna(b))


IP_FILES = {f"{P}_ip_candidates.csv"} | {f"{P}_M15_{lens}_ip_candidates.csv" for lens in ("confluence", "counter")}
EXPECTED_FILES = IP_FILES | {f"{P}_M15_{lens}_poi_zones.csv" for lens in ("confluence", "counter")}
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

# the base's own classification of 1660 (unchanged by the update) -> the moved POI's legs
bip = pd.read_csv(BASE / f"{P}_M15_confluence_ip_candidates.csv")
r1660 = bip[(bip.sub_id == 0) & (bip.cycle_id == 3) & (bip.ic_idx == 1660)].iloc[0]
assert r1660["ip_rank"] == "1A.iv", r1660["ip_rank"]
LEGS = {k: r1660[k] for k in ("ip_leg_candle_counts", "pattern_anchor_idx", "opp_leg_first_idx", "pattern_end_idx",
                              "pattern_confirmed_idx")}
SUB0_C3_COLS = {"ic_idx": (1447, 1660), "start_time": ("2025-12-07 23:45:00+00:00", "2025-12-10 05:00:00+00:00"),
                "top": (0.57768, 0.57738), "bottom": (0.57744, 0.5771)}
SUB0_C3_META = {"ip_tiebreak": ("TB2", "immediate_unfilled_imbalance"),
                "ip_tb_runner_up_ic_idx": (1660, 1447), "ip_tb_imbalance_first_c2_idx": (1658, 1662)}
RELABEL_META = {"ip_tiebreak": ("TB2", OLD_LAST)}
EXPECTED_POI = {("confluence", 0, 0, 3): "moved",
                ("confluence", 1, 0, 1): RELABEL_META,
                ("confluence", 7, 0, 2): RELABEL_META,
                ("counter", 7, 0, 2): RELABEL_META}
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
        elif exp == "moved":
            want_cols = {k: tuple(v) for k, v in cols.items()}
            if want_cols != SUB0_C3_COLS:
                bad.append(f"{n} row {i} {key}: columns {want_cols} != {SUB0_C3_COLS}")
            tb_meta = {k: v for k, v in metas.items() if k in TB}
            if tb_meta != SUB0_C3_META or mc["ip_tied_count"] != 2:
                bad.append(f"{n} row {i} {key}: tiebreak {tb_meta} / tied {mc['ip_tied_count']} != {SUB0_C3_META} / 2")
            for k, v in LEGS.items():
                if mc[k] != (int(v) if k != "ip_leg_candle_counts" else v):
                    bad.append(f"{n} row {i} {key}: {k} {mc[k]} != the base classification of 1660 {v}")
            if mc["ip_rank"] != "1A.iv":
                bad.append(f"{n} row {i} {key}: ip_rank {mc['ip_rank']}")
            other = set(metas) - set(TB) - set(LEGS) - ACTIVATION
            if other:
                bad.append(f"{n} row {i} {key}: unexpected meta deltas {sorted(other)}")
            print(f"  POI {key} MOVED: {cols} tiebreak {tb_meta} legs {LEGS}")
            for k in sorted(set(metas) & ACTIVATION):
                print(f"      activation-derived {k}: {metas[k][0]!r} -> {metas[k][1]!r}")
        else:
            if cols or metas != exp:
                bad.append(f"{n} row {i} {key}: delta {cols} {metas} != expected {exp}")
            print(f"  POI {key}: {metas}")
for k in set(EXPECTED_POI) - seen:
    bad.append(f"expected POI delta missing: {k}")

# IP candidates
EXPECTED_IP = {("confluence", 0, 0, 3, 1447): {"picked", *TB}, ("confluence", 0, 0, 3, 1660): {"picked", *TB},
               ("confluence", 1, 0, 1, 1875): {"ip_tiebreak"}, ("confluence", 7, 0, 2, 4083): {"ip_tiebreak"},
               ("counter", 7, 0, 2, 4083): {"ip_tiebreak"}}
TIED_NEW = {(0, 3, 1447): None, (0, 3, 1660): 1662, (1, 1, 1842): None, (1, 1, 1871): None, (1, 1, 1875): None,
            (7, 2, 4083): None, (7, 2, 4136): None}
seen, seen_tied, n_new = set(), set(), 0
for n in sorted(IP_FILES):
    lens = "confluence" if "_confluence_" in n else "counter" if "_counter_" in n else "H1"
    b, c = pd.read_csv(BASE / n), pd.read_csv(CUR / n)
    cols = list(c.columns)
    want_cols = list(b.columns)
    want_cols.insert(want_cols.index("ic_ref_reason") + 1, NEW)
    if cols != want_cols or len(b) != len(c):
        bad.append(f"{n}: columns / length differ ({len(b)} vs {len(c)})")
        continue
    for i in range(len(b)):
        rb, rc = b.iloc[i], c.iloc[i]
        d = {k for k in b.columns if not same(rb[k], rc[k])}
        ic = None if pd.isna(rb["ic_idx"]) else int(rb["ic_idx"])
        sub = None if lens == "H1" else int(rb["sub_id"])
        key = (lens, sub, int(rb["structure_id"]), int(rb["cycle_id"]), ic)
        v = rc[NEW]
        if ic is None:
            if not pd.isna(v):
                bad.append(f"{n} row {i} {key}: {NEW} {v} on an IP: none row")
        elif not pd.isna(v):
            n_new += 1
            if not (ic + 1 <= int(v) <= min(ic + 2, int(rc["cts_anchor_idx"]))):
                bad.append(f"{n} row {i} {key}: {NEW} {v} outside IC + 1 … min(IC + 2, CTS anchor)")
        tk = (sub, int(rb["cycle_id"]), ic)
        if lens != "H1" and tk in TIED_NEW:
            seen_tied.add(tk)
            got = None if pd.isna(v) else int(v)
            if got != TIED_NEW[tk]:
                bad.append(f"{n} row {i} {key}: {NEW} {got} != the measured {TIED_NEW[tk]}")
        if not d:
            continue
        seen.add(key)
        if EXPECTED_IP.get(key) != d:
            bad.append(f"{n} row {i} {key}: changed {sorted(d)} != expected {sorted(EXPECTED_IP.get(key, ()))}")
        print(f"  IP {key}: " + ", ".join(f"{k} {rb[k]!r} -> {rc[k]!r}" for k in sorted(d)))
for k in set(EXPECTED_IP) - seen:
    bad.append(f"expected IP delta missing: {k}")
for k in set(TIED_NEW) - seen_tied:
    bad.append(f"tied IC row missing: {k}")
print(f"  {NEW}: {n_new} candidate rows with an immediate unfilled imbalance (all within IC + 1 … IC + 2)")

print("\nRESULT:", "OK" if not bad else "UNEXPECTED")
for x in bad:
    print("  -", x)
sys.exit(1 if bad else 0)
