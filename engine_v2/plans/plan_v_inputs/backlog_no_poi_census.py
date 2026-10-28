"""The register D.2 note "sub 3 cyc 3 has no POI record" (PLAN_H §8.9, 2026-10-06), re-measured on S_V8c.

Every sub fib record per lens that is ENDED and was NEVER LOCKED (the case the POI record gate skips), and whether a POI
record exists for its (sub, cycle). Sub numbers have moved since 2026-10-06 (PLAN_R: 8 -> 11 subs), so the census lists
every such record rather than looking up "sub 3 cyc 3".

    PYTHONIOENCODING=utf-8 python engine_v2/plans/plan_v_inputs/backlog_no_poi_census.py > engine_v2/plans/plan_v_inputs/backlog_no_poi_census.txt
"""
import ast

import pandas as pd

S = "artifacts/commits/week9-entries-decisions/20261028_153518_1318c82/"
P = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"


def _meta(df):
    return df["meta"].map(lambda s: ast.literal_eval(s) if isinstance(s, str) else {})


for lens in ("confluence", "counter"):
    fib = pd.read_csv(S + f"{P}_M15_{lens}_fib_lifecycle.csv")
    poi = pd.read_csv(S + f"{P}_M15_{lens}_poi_zones.csv")
    fm, pm = _meta(fib), _meta(poi)
    fib["sub_id"] = fm.map(lambda m: m.get("sub_id"))
    fib["activated_at"] = fm.map(lambda m: m.get("activated_at"))
    poi["sub_id"] = pm.map(lambda m: m.get("sub_id"))
    poi_keys = set(zip(poi["sub_id"], poi["cycle_id"]))
    # one row per (sub, cycle): the last version
    last = fib.sort_values("version").groupby(["sub_id", "cycle_id"], dropna=False).tail(1)
    never_locked_ended = last[(~last["locked"]) & last["end_idx"].notna()]
    print(f"== {lens}: {len(last)} sub fib records (last version per sub / cycle), "
          f"{len(never_locked_ended)} ended + never locked; POI records: {len(poi)}")
    print(f"   locked: {last['locked'].value_counts().to_dict()}, ended: {last['end_idx'].notna().sum()} "
          f"of {len(last)}")
    no_poi = last[[(s, c) not in poi_keys for s, c in zip(last["sub_id"], last["cycle_id"])]]
    print(f"   every (sub, cycle) with NO POI record ({len(no_poi)}), any reason:")
    for _, r in no_poi.iterrows():
        print(f"     sub {r['sub_id']} cyc {r['cycle_id']}: fib {r['bos_idx']} -> {r['cts_idx']}, status={r['status']}, "
              f"active={r['active']}, locked={r['locked']}, live {r['start_idx']} -> {r['end_idx']}, "
              f"end_reason={r['end_reason']}, activated_at={r['activated_at']}")
    for _, r in never_locked_ended.iterrows():
        has_poi = (r["sub_id"], r["cycle_id"]) in poi_keys
        print(f"   sub {r['sub_id']} cyc {r['cycle_id']}: fib {r['bos_idx']} -> {r['cts_idx']}, "
              f"live {r['start_idx']:.0f} -> {r['end_idx']:.0f}, end_reason={r['end_reason']}, "
              f"activated_at={r['activated_at']}, POI record: {'yes' if has_poi else 'NO'}")

# the (sub, cycle)s above with no POI row: their IC outcome from *_ic_candidates.csv
for lens in ("confluence", "counter"):
    ic = pd.read_csv(S + f"{P}_M15_{lens}_ic_candidates.csv")
    out = ic.groupby(["sub_id", "cycle_id"])["ip_outcome"].agg(lambda s: sorted(set(map(str, s))))
    print(f"== {lens}: ip_outcome per (sub, cycle) in *_ic_candidates.csv")
    for (sub, cyc), v in out.items():
        print(f"   sub {sub} cyc {cyc}: {v}")

# the locked fibs with no POI row: is any of their rows an IC candidate?
for lens, keys in (("confluence", [(1, 0), (1, 3), (2, 0), (3, 0)]), ("counter", [(3, 0)])):
    ic = pd.read_csv(S + f"{P}_M15_{lens}_ic_candidates.csv")
    for sub, cyc in keys:
        g = ic[(ic["sub_id"] == sub) & (ic["cycle_id"] == cyc)]
        print(f"== {lens} sub {sub} cyc {cyc}: {len(g)} rows, IC candidates {int(g['is_ic_candidate'].sum())}, "
              f"ic_candidate_count {sorted(set(g['ic_candidate_count'].astype(str)))}, "
              f"reasons {sorted(set(g['reason'].astype(str)))[:4]}")
