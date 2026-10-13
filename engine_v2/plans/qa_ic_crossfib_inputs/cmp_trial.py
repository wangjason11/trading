"""KEYED diff of a replay against a save, for the IC-change trial (2026-10-13) — positional `cmp_save.py` breaks on
the IC CSV's new `ic_zone_band` column and the POI `meta` cell. Per lens (H1, M15 counter, M15 confluence):

  - the IC OUTCOME of every POI record (the `*_ic_candidates.csv` picked / no-IC row), keyed by
    (sub_id, structure_id, cycle_id, bos_anchor_idx, cts_anchor_idx);
  - the drawn POIs (`*_poi_zones.csv`), same key: ic_idx, status, confirmed_idx;
  - the structure events (multiset of sub / sid / cycle / type / idx / price): removed + added;
  - the fib records (`*_fib_lifecycle.csv`), keyed by path / sid / cycle / version;
  - the subs (`*_subs.csv`, by sub_id) and the triggers (by lens / parent / type / trigger_idx);
  - row counts of every other CSV.

    python <this> BASE_DIR CUR_DIR
"""
import ast
import glob
import os
import sys
from collections import Counter

import pandas as pd

BASE, CUR = sys.argv[1], sys.argv[2]
STEM = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"
LENSES = [("H1", ""), ("M15 counter", "_M15_counter"), ("M15 confluence", "_M15_confluence")]


def read(d, lens, name):
    p = os.path.join(d, f"{STEM}{lens}_{name}.csv")
    return pd.read_csv(p, keep_default_na=False, dtype=str) if os.path.exists(p) else None


def meta(s):
    try:
        return ast.literal_eval(s) if s else {}
    except (ValueError, SyntaxError):
        return {}


def ic_outcomes(df):
    out = {}
    if df is None:
        return out
    for _, r in df.iterrows():
        if not (r["picked"] == "True" or r["ic_idx"] == ""):
            continue
        k = (r.get("sub_id", ""), r["structure_id"], r["cycle_id"], r["bos_anchor_idx"], r["cts_anchor_idx"])
        if r["ic_idx"] == "":
            v = f"none ({r['reason']})"
        elif r["ip_outcome"] == "pattern":
            v = f"{r['ic_idx']} IP {r['ip_rank']}" + (f" tb={r['ip_tiebreak']}" if r["ip_tiebreak"] else "")
        else:
            v = f"{r['ic_idx']} no pattern" + (f" rank={r['ranking_rule']}" if r["ranking_rule"] else " (only cand)")
        band = r.get("ic_zone_band", "")
        if band and band != "61.8-80":
            v += f" [zone {band}]"
        out[k] = v
    return out


def poi_rows(df):
    out = {}
    if df is None:
        return out
    for _, r in df.iterrows():
        m = meta(r["meta"])
        k = (str(m.get("sub_id", "")), r["structure_id"], r["cycle_id"], str(m.get("bos_anchor_idx")),
             str(m.get("cts_anchor_idx")))
        out[k] = f"ic {r['ic_idx']} [{r['bottom']}, {r['top']}] status={r['status']} confirmed={r['confirmed_idx']}"
    return out


def events(df):
    c = Counter()
    if df is None:
        return c
    for _, r in df.iterrows():
        m = meta(r["meta"])
        c[(str(m.get("sub_id", "")), str(m.get("structure_id")), str(m.get("cycle_id")), r["type"], r["idx"],
           r["price"])] += 1
    return c


def fibs(df):
    out = {}
    if df is None:
        return out
    for _, r in df.iterrows():
        k = (r["structure_path_id"], r["structure_id"], r["cycle_id"], r["version"])
        out[k] = (f"bos {r['bos_idx']} cts {r['cts_idx']} life [{r['start_idx']}, {r['end_idx']}] {r['end_reason']} "
                  f"{r['status']} active={r['active']} locked={r['locked']}")
    return out


def keyed(title, a, b, limit=60):
    keys = sorted(set(a) | set(b), key=lambda k: tuple(str(x).zfill(6) for x in k))
    diff = [(k, a.get(k, "—"), b.get(k, "—")) for k in keys if a.get(k) != b.get(k)]
    print(f"  {title}: {len(a)} → {len(b)} keys, {len(diff)} differ")
    for k, x, y in diff[:limit]:
        print(f"    {k}: {x}  →  {y}")


for name, lens in LENSES:
    print(f"=== {name}")
    keyed("IC outcome per POI record", ic_outcomes(read(BASE, lens, "ic_candidates")),
          ic_outcomes(read(CUR, lens, "ic_candidates")))
    keyed("drawn POIs", poi_rows(read(BASE, lens, "poi_zones")), poi_rows(read(CUR, lens, "poi_zones")))
    ea, eb = events(read(BASE, lens, "structure_events")), events(read(CUR, lens, "structure_events"))
    rem, add = ea - eb, eb - ea
    print(f"  structure events: {sum(ea.values())} → {sum(eb.values())}; removed {sum(rem.values())}, "
          f"added {sum(add.values())}")
    for k in sorted(rem, key=lambda k: (k[0].zfill(3), int(k[4]))):
        print(f"    - {k}")
    for k in sorted(add, key=lambda k: (k[0].zfill(3), int(k[4]))):
        print(f"    + {k}")
    keyed("fib records", fibs(read(BASE, lens, "fib_lifecycle")), fibs(read(CUR, lens, "fib_lifecycle")))
    if lens:
        sa, sb = read(BASE, lens, "subs"), read(CUR, lens, "subs")
        keyed("subs", {r["sub_id"]: " ".join(f"{c}={r[c]}" for c in ("direction", "starting_idx", "start_idx",
                                                                     "end_idx", "end_reason"))
                       for _, r in sa.iterrows()},
              {r["sub_id"]: " ".join(f"{c}={r[c]}" for c in ("direction", "starting_idx", "start_idx", "end_idx",
                                                             "end_reason"))
               for _, r in sb.iterrows()})
        ta, tb = read(BASE, lens, "triggers"), read(CUR, lens, "triggers")
        tk = ("parent_sid", "parent_cycle_id", "trigger_type", "trigger_idx")
        keyed("triggers", {tuple(r[c] for c in tk): f"sub {r['sub_id']} start {r['start_idx']} end {r['end_idx']}"
                           for _, r in ta.iterrows()},
              {tuple(r[c] for c in tk): f"sub {r['sub_id']} start {r['start_idx']} end {r['end_idx']}"
               for _, r in tb.iterrows()})
    for n in ("kl_zones", "wvmi"):
        a, b = read(BASE, lens, n), read(CUR, lens, n)
        if a is not None and b is not None:
            same = a.shape == b.shape and (a.values == b.values).all()
            print(f"  {n}: rows {len(a)} → {len(b)}{'' if same else ' (DIFFER)'}")
u = [os.path.basename(p) for p in glob.glob(os.path.join(BASE, "*unresolved_triggers.csv"))]
for n in u:
    a, b = pd.read_csv(os.path.join(BASE, n), dtype=str), pd.read_csv(os.path.join(CUR, n), dtype=str)
    print(f"unresolved triggers: {len(a)} → {len(b)}{'' if a.equals(b) else ' (DIFFER)'}")
