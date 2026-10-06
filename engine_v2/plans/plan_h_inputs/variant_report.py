"""PLAN_H §8.4 — what a variant replay moved against a save, KEYED (not positional).

    python engine_v2/plans/plan_h_inputs/variant_report.py BASE_DIR CUR_DEBUG [--rows]

1. md5 of every BASE CSV vs CUR (equal / DIFF / missing).
2. POIs per lens: (sub key, sid, cycle, IC) → confirmed_idx, status, versions, activation flips. A sub is keyed
   by (direction, starting_idx) from its lens's `*_subs.csv` (`sub_id` = creation order — PLAN_H §8.8 PR3).
3. Structure events per lens: removed / added (sub key, type, idx, price) — `idx` = the MOMENT (Plan E).
4. M15 triggers + unresolved triggers: removed / added rows keyed on their identity columns.
5. Subs: removed / added / changed (direction, starting_idx).
6. `_final.csv`: the changed columns and their first changed row.
"""
import ast
import glob
import hashlib
import os
import re
import sys

import pandas as pd

BASE, CUR = sys.argv[1], sys.argv[2]
ROWS = "--rows" in sys.argv
PFX = "NZD_USD_H1_2025-11-15_2026-01-20"
RK = f"{PFX}_sd-1_eps0p0001_rk2-5"
LENSES = {"H1": "", "conf": "_M15_confluence", "counter": "_M15_counter"}


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def rd(d, name):
    p = os.path.join(d, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def meta(s):
    if not isinstance(s, str):
        return {}
    s2 = re.sub(r"np\.(?:int|float)\d*\(([^)]*)\)", r"\1", s)
    s2 = re.sub(r"np\.(?:True_|False_)", lambda m: "True" if "True" in m.group(0) else "False", s2)
    s2 = re.sub(r"Timestamp\('([^']*)'[^)]*\)", r"'\1'", s2)
    try:
        return ast.literal_eval(s2)
    except Exception:
        return {"_raw": s}


print("=== 1. md5")
for p in sorted(glob.glob(os.path.join(BASE, "*.csv"))):
    n = os.path.basename(p)
    q = os.path.join(CUR, n)
    st = "missing" if not os.path.exists(q) else ("equal" if md5(p) == md5(q) else "DIFF")
    if st != "equal":
        print(f"  {st:8s} {n[len(PFX):]}")
print("  (others equal)")


def subkeys(d, lens_sfx):
    s = rd(d, f"{RK}{lens_sfx}_subs.csv")
    if s is None:
        return {}
    return {int(r.sub_id): (int(r.direction), int(r.starting_idx)) for r in s.itertuples()}


def poi_rows(d, lens, sfx):
    df = rd(d, f"{RK}{sfx}_poi_zones.csv")
    sk = subkeys(d, sfx) if lens != "H1" else {}
    out = {}
    for r in df.itertuples():
        m = meta(r.meta)
        sub = m.get("sub_id")
        key = (sk.get(int(sub), ("?", sub)) if sub is not None else None, int(r.structure_id), int(r.cycle_id),
               int(r.ic_idx))
        flips = ",".join(f"{e['idx']}{'A' if e['active'] else 'D'}" for e in m.get("activation_history", []))
        reasons = sorted({e.get("reason") for e in m.get("activation_history", []) if not e["active"]})
        out[key] = (r.confirmed_idx if pd.notna(r.confirmed_idx) else None, r.status, r.versions, flips,
                    "/".join(reasons))
    return out


print("\n=== 2. POIs (key = (sub (dir, starting_idx) | None, sid, cycle, ic_idx) -> confirmed, status, versions, "
      "flips, deact reasons)")
for lens, sfx in LENSES.items():
    b, c = poi_rows(BASE, lens, sfx), poi_rows(CUR, lens, sfx)
    gone = sorted(set(b) - set(c), key=str)
    new = sorted(set(c) - set(b), key=str)
    chg = sorted((k for k in set(b) & set(c) if b[k] != c[k]), key=str)
    print(f"  {lens}: base {len(b)} rows / {len({k[:3] for k in b})} cycles -> cur {len(c)} rows / "
          f"{len({k[:3] for k in c})} cycles; removed {len(gone)}, added {len(new)}, kept-changed {len(chg)}, "
          f"kept-equal {len(set(b) & set(c)) - len(chg)}")
    if ROWS or lens == "H1":
        for k in gone:
            print(f"    - {k} {b[k]}")
        for k in new:
            print(f"    + {k} {c[k]}")
        for k in chg:
            print(f"    ~ {k} {b[k]} -> {c[k]}")


def ev_rows(d, sfx):
    df = rd(d, f"{RK}{sfx}_structure_events.csv")
    sk = subkeys(d, sfx) if sfx else {}
    out = []
    for r in df.itertuples():
        m = meta(r.meta)
        sub = m.get("sub_id")
        out.append((sk.get(int(sub), ("?", sub)) if sub is not None else None, m.get("structure_id"),
                    m.get("cycle_id"), r.type, int(r.idx), round(float(r.price), 6) if pd.notna(r.price) else None))
    return out


print("\n=== 3. structure events (sub, sid, cycle, type, idx=moment, price)")
for lens, sfx in LENSES.items():
    b, c = ev_rows(BASE, sfx), ev_rows(CUR, sfx)
    sb, sc = set(b), set(c)
    gone = sorted(sb - sc, key=str)
    new = sorted(sc - sb, key=str)
    print(f"  {lens}: base {len(b)} -> cur {len(c)}; removed {len(gone)}, added {len(new)}")
    if ROWS or lens == "H1":
        for e in gone:
            print(f"    - {e}")
        for e in new:
            print(f"    + {e}")

print("\n=== 4. M15 triggers / unresolved")
for name, keys in [("_M15_confluence_triggers", ["parent_sid", "parent_cycle_id", "trigger_type", "trigger_idx"]),
                   ("_M15_counter_triggers", ["parent_sid", "parent_cycle_id", "trigger_type", "trigger_idx"]),
                   ("_M15_unresolved_triggers", ["lens", "parent_sid", "parent_cycle_id", "trigger_type",
                                                  "trigger_idx"])]:
    b, c = rd(BASE, f"{RK}{name}.csv"), rd(CUR, f"{RK}{name}.csv")
    show = [x for x in ["starting_idx", "direction", "start_idx", "end_idx", "end_reason", "sub_id", "reason"]
            if x in b.columns]
    bk = {tuple(r[keys]): tuple(str(x) for x in r[show]) for _, r in b.iterrows()}
    ck = {tuple(r[keys]): tuple(str(x) for x in r[show]) for _, r in c.iterrows()}
    print(f"  {name}: base {len(b)} -> cur {len(c)}  ({keys} -> {show})")
    for k in sorted(set(bk) - set(ck), key=str):
        print(f"    - {k} {bk[k]}")
    for k in sorted(set(ck) - set(bk), key=str):
        print(f"    + {k} {ck[k]}")
    for k in sorted((k for k in set(bk) & set(ck) if bk[k] != ck[k]), key=str):
        print(f"    ~ {k} {bk[k]} -> {ck[k]}")

print("\n=== 5. subs (direction, starting_idx)")
for sfx in ["_M15_confluence", "_M15_counter"]:
    b, c = rd(BASE, f"{RK}{sfx}_subs.csv"), rd(CUR, f"{RK}{sfx}_subs.csv")
    cols = [x for x in b.columns if x not in ("sub_id",)]
    bk = {(int(r.direction), int(r.starting_idx)): (int(r.sub_id), tuple(str(x) for x in r[cols])) for _, r in b.iterrows()}
    ck = {(int(r.direction), int(r.starting_idx)): (int(r.sub_id), tuple(str(x) for x in r[cols])) for _, r in c.iterrows()}
    print(f"  {sfx}: base {len(b)} -> cur {len(c)}")
    for k in sorted(set(bk) - set(ck)):
        print(f"    - {k} sub_id {bk[k][0]}")
    for k in sorted(set(ck) - set(bk)):
        print(f"    + {k} sub_id {ck[k][0]}")
    for k in sorted(set(bk) & set(ck)):
        if bk[k] != ck[k]:
            d = [(cols[i], bk[k][1][i], ck[k][1][i]) for i in range(len(cols)) if str(bk[k][1][i]) != str(ck[k][1][i])]
            print(f"    ~ {k} sub_id {bk[k][0]}->{ck[k][0]} {d}")

print("\n=== 6. _final.csv")
b, c = rd(BASE, f"{PFX}_final.csv"), rd(CUR, f"{PFX}_final.csv")
if list(b.columns) != list(c.columns):
    print("  header differs:", set(b.columns) ^ set(c.columns))
for col in b.columns:
    if col not in c.columns:
        continue
    eq = (b[col].astype(str) == c[col].astype(str))
    if not eq.all():
        first = int((~eq).to_numpy().argmax())
        print(f"  {col}: {int((~eq).sum())} rows differ, first at row {first}")
