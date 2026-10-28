"""PLAN_V 8c — what the agreed direction moves: the volume-only doji census, then every wave candle and WVMI record that
differs between `v8c_measure.py base` and `variant`, each changed candle with its row (from the base run's frames).

    python engine_v2/plans/plan_v_inputs/v8c_diff.py BASE.json VARIANT.json
"""
import json
import sys

import pandas as pd

base_p, var_p = sys.argv[1], sys.argv[2]
base, var = (json.load(open(p, encoding="utf-8")) for p in (base_p, var_p))
frames = {"H1": pd.read_csv(base_p.replace(".json", "_h1_frame.csv"), index_col="idx"),
          "M15": pd.read_csv(base_p.replace(".json", "_m15_frame.csv"), index_col="idx")}


def tf(owner):
    return "H1" if owner == "H1" else "M15"


def row(tf_, idx):
    if idx is None:
        return "-"
    r = frames[tf_].loc[int(idx)]
    kind = ("VOLUME-ONLY DOJI" if r.direction == 0 and r.vol_dir != 0 else
            "true doji" if r.direction == 0 else "split" if r.vol_dir == -r.direction else "")
    return (f"{tf_} {int(idx)} {r.time[:16]} o={r.o:.5f} h={r.h:.5f} l={r.l:.5f} c={r.c:.5f} vol={r.volume:.0f} "
            f"dir={r.direction:+d} vol_dir={r.vol_dir:+d} {r.candle_type} {kind}")


print("== the volume-only dojis (direction 0, vol_dir +-1) on the window's frames ==")
for name in ("h1_dojis", "m15_dojis"):
    ds = base[name]
    vo = [d for d in ds if d["vol_dir"] != 0]
    print(f"{name}: {len(ds)} dojis, {len(vo)} volume-only "
          f"(+1: {sum(d['vol_dir'] == 1 for d in vo)}, -1: {sum(d['vol_dir'] == -1 for d in vo)})")

print("\n== wave candles that differ (key: owner, structure_id, cycle_id, source_kind; [last, first]) ==")
key = lambda w: (w["owner"], w["structure_id"], w["cycle_id"], w["source_kind"])  # noqa: E731
bw, vw = {key(w): w for w in base["waves"]}, {key(w): w for w in var["waves"]}
assert len(bw) == len(base["waves"]) and len(vw) == len(var["waves"]), "duplicate wave keys"
n = 0
for k in sorted(set(bw) | set(vw), key=str):
    b, v = bw.get(k), vw.get(k)
    if b is not None and v is not None and (b["last"], b["first"]) == (v["last"], v["first"]):
        continue
    n += 1
    print(f"{k} {b['zone_side'] if b else v['zone_side']}: base [{b and b['last']}, {b and b['first']}] -> "
          f"variant [{v and v['last']}, {v and v['first']}]")
    for pos in ("last", "first"):
        if b is None or v is None or b[pos] != v[pos]:
            print(f"    {pos}: base   {row(tf(k[0]), b and b[pos])}")
            print(f"    {pos}: variant {row(tf(k[0]), v and v[pos])}")
print(f"{n} of {len(bw)} wave results differ")

print("\n== WVMI records that differ (key: owner, bos_structure_id, bos_cycle_id) ==")
rkey = lambda r: (r["owner"], r["bos_structure_id"], r["bos_cycle_id"])  # noqa: E731
br, vr = {rkey(r): r for r in base["wvmi"]}, {rkey(r): r for r in var["wvmi"]}
n = 0
for k in sorted(set(br) | set(vr), key=str):
    b, v = br.get(k), vr.get(k)
    if b == v:
        continue
    n += 1
    print(f"{k}:")
    for f in sorted(set(b or {}) | set(v or {})):
        if f == "lp_rows" or (b or {}).get(f) == (v or {}).get(f):
            continue
        print(f"    {f}: {(b or {}).get(f)!r} -> {(v or {}).get(f)!r}")
    bl = {r["lp_idx"]: r for r in (b or {}).get("lp_rows", [])}
    vl = {r["lp_idx"]: r for r in (v or {}).get("lp_rows", [])}
    print(f"    lp_rows: {len(bl)} -> {len(vl)}")
    for i in sorted(set(bl) | set(vl)):
        if bl.get(i) == vl.get(i):
            continue
        if i not in bl:
            r = vl[i]
            print(f"      + NEW LP row {i}: {r['lp_status']} as_of {r['as_of_idx']}, {r['lp_status_reason']} at "
                  f"{r['lp_status_idx']}, pullback_momentum {r['pullback_momentum']:.4f}")
            print(f"          {row(tf(k[0]), i)}")
        elif i not in vl:
            print(f"      - GONE LP row {i}")
        else:
            ch = {f: (bl[i][f], vl[i][f]) for f in bl[i] if bl[i][f] != vl[i][f]}
            print(f"      ~ LP row {i}: " + ", ".join(f"{f} {a} -> {c}" for f, (a, c) in ch.items()))
print(f"{n} of {len(br)} WVMI records differ")
