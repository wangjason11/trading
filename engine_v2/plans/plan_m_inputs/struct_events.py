"""Print the CTS / BOS / state events of H1 sids and named subs on a save (read-only).

    python struct_events.py <save_dir> H1:0 H1:1 sub:3 ...
"""
import ast
import glob
import sys

import pandas as pd

SAVE = sys.argv[1]
WANT = sys.argv[2:]
TYPES = {"CTS_ESTABLISHED", "CTS_UPDATED", "CTS_CONFIRMED", "CTS_THRESHOLD_UPDATED", "BOS_CONFIRMED",
         "BOS_THRESHOLD_UPDATED", "STATE_CHANGED", "REVERSAL_CANDIDATE", "RANGE_STARTED", "RANGE_ENDED"}
SKIP_UPDATED = "--all-updates" not in WANT
WANT = [w for w in WANT if not w.startswith("--")]


def one(p):
    h = glob.glob(f"{SAVE}/{p}")
    assert len(h) == 1, (p, h)
    return h[0]


raw = pd.read_csv(one("*_H1_*_raw.csv"))
H1T = pd.to_datetime(raw[[c for c in raw.columns if c.lower() in ("time", "timestamp", "datetime")][0]])


def tm(i):
    i = int(i)
    return (H1T.iloc[i // 4] + pd.Timedelta(minutes=15 * (i % 4))).strftime("%m-%d %H:%M")


def th(i):
    return H1T.iloc[int(i)].strftime("%m-%d %H:%M")


def load(p):
    e = pd.read_csv(p)
    e["m"] = e["meta"].map(ast.literal_eval)
    return e


h1 = load(one("*_rk2-5_structure_events.csv"))
lev = {lens: load(one(f"*_M15_{lens}_structure_events.csv")) for lens in ("confluence", "counter")}

for w in WANT:
    kind, n = w.split(":")
    n = int(n)
    if kind == "H1":
        e = h1[h1["m"].map(lambda m: m.get("structure_id") == n)]
        t = th
    else:
        e = None
        for lens in ("confluence", "counter"):
            x = lev[lens][lev[lens]["m"].map(lambda m: m.get("sub_id") == n)]
            if len(x):
                e = x
                break
        t = tm
    print(f"\n=== {w} ({len(e)} events) ===")
    last_upd = {}
    for _, r in e.iterrows():
        ty = r["type"]
        if ty not in TYPES:
            continue
        m = r["m"]
        if ty == "CTS_UPDATED" and SKIP_UPDATED:
            continue
        cyc = m.get("cycle_id", r.get("cycle_id", ""))
        extra = []
        for k in ("source", "cts_anchor_idx", "bos_anchor_idx", "pattern_anchor_idx", "bos_frozen", "cts_frozen", "from", "to",
                  "reason", "via", "threshold", "old_threshold", "new_threshold", "prev_threshold"):
            if k in m:
                v = m[k]
                if k.endswith("_idx") and v is not None and not pd.isna(v):
                    v = f"{int(v)} ({t(v)})"
                extra.append(f"{k}={v}")
        print(f"  {ty:22s} idx {int(r['idx']):5d} ({t(r['idx'])}) cyc {cyc} price {r['price']} | {' '.join(extra)}")
