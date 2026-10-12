"""Status quo of every structure's cross-fib limit on a save (read-only).

    python limit_table.py <save_dir>

Per structure: P, source, M, from_sub_id, started_by; per fib record: cycle, kind (single / cross vN), via,
bos / cts idx, active window, the IC reference (price, source, break).
"""
import ast
import glob
import sys

import pandas as pd

SAVE = sys.argv[1]


def one(p):
    h = glob.glob(f"{SAVE}/{p}")
    assert len(h) == 1, (p, h)
    return h[0]


raw = pd.read_csv(one("*_H1_*_raw.csv"))
H1T = pd.to_datetime(raw[[c for c in raw.columns if c.lower() in ("time", "timestamp", "datetime")][0]])


def tm(i):
    if i is None or pd.isna(i):
        return "-"
    i = int(i)
    return (H1T.iloc[i // 4] + pd.Timedelta(minutes=15 * (i % 4))).strftime("%m-%d %H:%M")


def th(i):
    if i is None or pd.isna(i):
        return "-"
    return H1T.iloc[int(i)].strftime("%m-%d %H:%M")


def rows(path, tf):
    f = pd.read_csv(path)
    f["m"] = f["meta"].map(ast.literal_eval)
    out = []
    for _, r in f.iterrows():
        m = r["m"]
        key = ("H1", int(r["structure_id"])) if tf == "H1" else ("sub", m.get("sub_id"))
        out.append((key, r, m))
    return out


allr = rows(one("*_rk2-5_fib_lifecycle.csv"), "H1")
seen = set()
for lens in ("confluence", "counter"):
    for key, r, m in rows(one(f"*_M15_{lens}_fib_lifecycle.csv"), "M15"):
        k2 = (key, int(r["cycle_id"]), str(r["version"]), r["bos_idx"], r["cts_idx"])
        if k2 in seen:
            continue
        seen.add(k2)
        allr.append((key, r, m))

by = {}
for key, r, m in allr:
    by.setdefault(key, []).append((r, m))

for key in sorted(by, key=lambda k: (k[0] != "H1", k[1])):
    recs = by[key]
    m0 = recs[0][1]
    t = th if key[0] == "H1" else tm
    # (+ PLAN_M landing 1: the two keys of a moved `P`, present only when it moved by the structure's end)
    lim_keys = ("cross_fib_limit_price", "cross_fib_limit_source", "cross_fib_limit_cycle",
                "cross_fib_limit_from_sub_id", "started_by", "cross_fib_limit_price_at_start",
                "cross_fib_limit_move_count")
    lim = tuple(m0.get(k) for k in lim_keys)
    # every record of a structure must carry the same limit
    for _, m in recs:
        assert tuple(m.get(k) for k in lim_keys) == lim, key
    moved = "" if lim[6] is None else f" (moved {lim[6]}x from {lim[5]})"
    print(f"\n{key[0]} {key[1]}: P={lim[0]}{moved} source={lim[1]} M={lim[2]} from_sub={lim[3]} started_by={lim[4]}")
    for r, m in sorted(recs, key=lambda x: (int(x[0]["cycle_id"]), str(x[0]["version"]))):
        kind = "single" if pd.isna(r["version"]) else f"cross v{int(r['version'])} from cyc {m.get('cross_start_cycle')}"
        via = m.get("via", "")
        icr = ""
        if "ic_ref_price" in m:
            icr = (f"IC ref {m['ic_ref_price']} ({m['ic_ref_source']}) break "
                   f"{m.get('ic_ref_break_pattern_anchor_idx', 'none')}")
        else:
            icr = "IC ref: none"
        print(f"   cyc {int(r['cycle_id'])} {kind:22s} via={via:16s} BOS {int(r['bos_idx'])} ({t(r['bos_idx'])}) "
              f"CTS {int(r['cts_idx'])} ({t(r['cts_idx'])}) | {r['status']:8s} start {t(r['start_idx'])} end "
              f"{t(r['end_idx'])} | {icr}")
