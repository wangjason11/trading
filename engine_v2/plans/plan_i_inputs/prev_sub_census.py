"""PLAN_I round 2 — the previous-sub census (2026-10-07; PLAN_I §3.3). Read-only; prints tables.

    python engine_v2/plans/plan_i_inputs/prev_sub_census.py artifacts/commits/week9-entries-decisions/20261006_224610_42c137c

(A) the user's x-1 rule (R2-1) per sub: x-1's direction vs x, the reference it gives at x's start_idx; (B) is every new
cycle's establishing CTS anchor beyond cycle c-1's cts_threshold at that moment? (C) per sub: BOS_0's anchor vs
starting_idx (the R2-2 fallback price), CTS_0 established / first CTS_CONFIRMED vs start_idx (R2-b's gap).
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


def load(p):
    e = pd.read_csv(p)
    e["m"] = e["meta"].map(ast.literal_eval)
    return e


raw = pd.read_csv(one("*_H1_*_raw.csv"))
H1T = pd.to_datetime(raw[[c for c in raw.columns if c.lower() in ("time", "timestamp", "datetime")][0]])


def tm(i):
    return (H1T.iloc[int(i) // 4] + pd.Timedelta(minutes=15 * (int(i) % 4))).strftime("%m-%d %H:%M")


def th(i):
    return H1T.iloc[int(i)].strftime("%m-%d %H:%M")


subs = {}
for lens in ("confluence", "counter"):
    for _, r in pd.read_csv(one(f"*_M15_{lens}_subs.csv")).iterrows():
        subs.setdefault(int(r["sub_id"]), (lens, r))
lev = {lens: load(one(f"*_M15_{lens}_structure_events.csv")) for lens in ("confluence", "counter")}


def sev(s):
    lens, _ = subs[s]
    e = lev[lens]
    return e[e["m"].map(lambda m: m.get("sub_id") == s)].sort_values("idx", kind="stable")


def bos_thr(e, at):
    thr, src = None, None
    for _, r in e.iterrows():
        if int(r["idx"]) > at:
            break
        if r["type"] in ("BOS_CONFIRMED", "BOS_THRESHOLD_UPDATED"):
            thr, src = float(r["price"]), f"{r['type']}@{int(r['idx'])} cyc {r['m'].get('cycle_id')}"
    return thr, src


def cts_thr_confirmed(e, at):
    """cts_threshold attached to the most recent CONFIRMED CTS as of `at`: that cycle's CTS_CONFIRMED price, moved by
    its CTS_THRESHOLD_UPDATED events (<= at)."""
    conf = e[(e["type"] == "CTS_CONFIRMED") & (e["idx"] <= at)]
    if conf.empty:
        return None, "no confirmed CTS"
    c = conf["m"].iloc[-1]["cycle_id"]
    thr, src = None, None
    for _, r in e.iterrows():
        if int(r["idx"]) > at:
            break
        if r["m"].get("cycle_id") != c:
            continue
        if r["type"] in ("CTS_CONFIRMED", "CTS_THRESHOLD_UPDATED"):
            thr, src = float(r["price"]), f"{r['type']}@{int(r['idx'])} cyc {c}"
    return thr, src


print("(A) the x-1 rule — subs started by first_counter / subsequent_* (and every sub, for illustration)")
order = sorted(subs, key=lambda s: (int(subs[s][1]["start_idx"]), s))
print("   start order by start_idx:", [(s, int(subs[s][1]["start_idx"])) for s in order], "== id order:", order == sorted(subs))
for s in sorted(subs):
    lens, r = subs[s]
    if s == 0:
        continue
    x1 = s - 1
    _, p = subs[x1]
    at = int(r["start_idx"])
    end1 = p["end_idx"]
    at_eff = at if pd.isna(end1) else min(at, int(end1))
    e1 = sev(x1)
    same = int(p["direction"]) == int(r["direction"])
    if same:
        val, src = cts_thr_confirmed(e1, at_eff)
    else:
        val, src = bos_thr(e1, at_eff)
    print(f"   sub {s} ({lens}, dir {int(r['direction']):+d}, started_by {r['first_record_trigger_type']}, start {at} {tm(at)}): "
          f"x-1 = sub {x1} (dir {int(p['direction']):+d}, start {int(p['start_idx'])}, end {end1}) -> "
          f"{'SAME: cts_threshold' if same else 'OPPOSITE: bos_threshold'} as of {at_eff} = {val} [{src}]; "
          f"x-1 starting_idx {int(p['starting_idx'])} ({tm(int(p['starting_idx']))})")

# hypothetical same-direction illustration: sub 6's same-direction predecessor sub 4 (it replaced it at 3819)
v, src = cts_thr_confirmed(sev(4), 3819)
print(f"   [illustration] sub 4 (sell) cts_threshold of its latest confirmed CTS as of 3819 = {v} [{src}]")
for _, r in sev(4).iterrows():
    if r["type"].startswith("CTS") or r["type"].startswith("BOS"):
        print(f"      sub 4 {r['type']:24s} idx {int(r['idx'])} ({tm(int(r['idx']))}) price {r['price']} cyc {r['m'].get('cycle_id')}")

print("\n(B) is a new cycle's CTS (at CTS_ESTABLISHED and final) beyond cycle c-1's cts_threshold at that moment?")


def check(name, e, sd, tfun):
    est = e[e["type"] == "CTS_ESTABLISHED"]
    for _, r in est.iterrows():
        c = r["m"]["cycle_id"]
        if c == 0:
            continue
        mom = int(r["idx"])
        prior = e[(e["m"].map(lambda m: m.get("cycle_id") == c - 1)) & (e["idx"] < mom)]
        thr = None
        for _, q in prior.iterrows():
            if q["type"] in ("CTS_CONFIRMED", "CTS_THRESHOLD_UPDATED"):
                thr = float(q["price"])
        conf_prev = prior[prior["type"] == "CTS_CONFIRMED"]
        cprice = float(conf_prev["price"].iloc[-1]) if len(conf_prev) else None
        p = float(r["price"])
        beyond = None if thr is None else ((p > thr) if sd == 1 else (p < thr))
        print(f"   {name} cyc {c}: EST {mom} ({tfun(mom)}) via {r['m'].get('via')} anchor {r['m'].get('cts_anchor_idx')} "
              f"{p:.5f} | c-1 confirmed CTS {cprice} | c-1 cts_threshold {thr} | beyond thr: {beyond}")


h1 = load(one("*rk2-5_structure_events.csv"))
for sid in sorted(h1["m"].map(lambda m: m.get("structure_id")).dropna().unique()):
    e = h1[h1["m"].map(lambda m: m.get("structure_id") == sid)].sort_values("idx", kind="stable")
    check(f"H1 sid {int(sid)}", e, int(e["m"].iloc[0]["struct_direction"]), th)
for s in sorted(subs):
    check(f"sub {s}", sev(s), int(subs[s][1]["direction"]), tm)

print("\n(C) per sub: BOS_0 anchor vs starting_idx; CTS_0 EST / first CTS_CONFIRMED vs start_idx")
for s in sorted(subs):
    lens, r = subs[s]
    e = lev[lens]
    e = e[e["m"].map(lambda m: m.get("sub_id") == s)].sort_values("idx", kind="stable")
    b0 = e[(e["type"] == "BOS_CONFIRMED") & e["m"].map(lambda m: m.get("cycle_id") == 0)]
    c0 = e[(e["type"] == "CTS_ESTABLISHED") & e["m"].map(lambda m: m.get("cycle_id") == 0)]
    cc = e[e["type"] == "CTS_CONFIRMED"]
    print(f"sub {s} dir {int(r['direction']):+d} starting {int(r['starting_idx'])} start {int(r['start_idx'])} | "
          f"BOS_0 anchor {b0['m'].iloc[0].get('bos_anchor_idx')} price {b0['price'].iloc[0]} at {int(b0['idx'].iloc[0])} | "
          f"CTS_0 EST {int(c0['idx'].iloc[0])} | first CTS_CONFIRMED {int(cc['idx'].iloc[0]) if len(cc) else None}")
