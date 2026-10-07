"""PLAN_I round 1 — the reference-price census on a save (2026-10-07).

For every structure the user's rule covers (H1 main sids + every unique M15 sub): its starting trigger, the reference
price(s) the rule gives (several READINGS where the rule is still open), each cycle's CTS anchor (the CONFIRMED one, or
the last CTS_UPDATED when never confirmed), the cap cycle M (the first cycle whose CTS anchor EXCEEDS the reference,
strictly, in the structure's direction) and today's fib record per cycle (single / cross / cross_failed single).

    python engine_v2/plans/plan_i_inputs/ref_census.py artifacts/commits/week9-entries-decisions/20261006_224610_42c137c

Read-only; prints tables. Times are UTC as the charts show them (M15 i = H1 i // 4 + 15 min x (i % 4)).
"""
import ast
import glob
import sys

import pandas as pd

SAVE = sys.argv[1]


def _one(pat):
    hits = glob.glob(f"{SAVE}/{pat}")
    assert len(hits) == 1, (pat, hits)
    return hits[0]


raw = pd.read_csv(_one("*_H1_*_raw.csv"))
H1T = pd.to_datetime(raw[[c for c in raw.columns if c.lower() in ("time", "timestamp", "datetime")][0]])


def t_h1(i):
    return "" if i is None else H1T.iloc[int(i)].strftime("%m-%d %H:%M")


def t_m15(i):
    return "" if i is None else (H1T.iloc[int(i) // 4] + pd.Timedelta(minutes=15 * (int(i) % 4))).strftime("%m-%d %H:%M")


def load_events(path):
    ev = pd.read_csv(path)
    ev["m"] = ev["meta"].map(ast.literal_eval)
    return ev


def cycles_of(ev, sd):
    """{cycle: dict(est, anchor, price, confirmed)} — the CONFIRMED anchor, else the last CTS_UPDATED / the EST."""
    out = {}
    for _, r in ev.iterrows():
        m, t = r["m"], r["type"]
        if t not in ("CTS_ESTABLISHED", "CTS_UPDATED", "CTS_CONFIRMED"):
            continue
        c = int(m["cycle_id"])
        anchor = int(m.get("cts_anchor_idx", r["idx"]))
        if t == "CTS_ESTABLISHED":
            out[c] = dict(est=int(r["idx"]), anchor=anchor, price=float(r["price"]), confirmed=None)
        elif c in out and out[c]["confirmed"] is None:
            out[c].update(anchor=anchor, price=float(r["price"]))
            if t == "CTS_CONFIRMED":
                out[c]["confirmed"] = int(r["idx"])
    return out


def bos_threshold_at(ev, idx):
    """The MS bos_threshold as of candle `idx` (inclusive): the last BOS_CONFIRMED price / BOS_THRESHOLD_UPDATED."""
    thr = None
    for _, r in ev.sort_values("idx", kind="stable").iterrows():
        if int(r["idx"]) > idx:
            break
        if r["type"] in ("BOS_CONFIRMED", "BOS_THRESHOLD_UPDATED"):
            thr = float(r["price"])
    return thr


def cap_cycle(cyc, ref, sd):
    if ref is None:
        return None
    for c in sorted(cyc):
        p = cyc[c]["price"]
        if (p > ref) if sd == 1 else (p < ref):
            return c
    return "not reached"


def fib_kinds(fib_csv, key_cols):
    df = pd.read_csv(fib_csv)
    out = {}
    for _, r in df.iterrows():
        m = ast.literal_eval(r["meta"])
        kind = ("cross from c%s" % m.get("cross_start_cycle", 0)) if m.get("cross_cycle") else (
            "single (cross_failed)" if m.get("via") == "cross_failed" else "single")
        if m.get("cross_cycle") and m.get("scenario") == 2:
            kind = "cross from c0 (main Scenario 2)"
        key = tuple(m.get(k, r.get(k)) if k == "sub_id" else int(r[k]) for k in key_cols)
        if r["active"] is False and m.get("cross_cycle"):
            kind += " [dead version]"
        out.setdefault(key, []).append(kind)
    return out


def show(title, sd, cyc, ref_rows, fibs, key_prefix, tfun):
    print(f"\n=== {title}  (sd={sd:+d})")
    for name, ref in ref_rows:
        M = cap_cycle(cyc, ref, sd)
        print(f"  reference [{name}]: {ref}  -> cap cycle M = {M}")
    for c in sorted(cyc):
        d = cyc[c]
        kinds = fibs.get(key_prefix + (c,), ["(no fib record)"])
        conf = f"conf {d['confirmed']}" if d["confirmed"] is not None else "NEVER CONFIRMED (last update)"
        print(f"   cyc {c}: est {d['est']} ({tfun(d['est'])}), CTS anchor {d['anchor']} ({tfun(d['anchor'])}) "
              f"{d['price']:.5f} [{conf}] | today: {', '.join(kinds)}")


# --- H1 main --------------------------------------------------------------------------------------------------------
h1ev = load_events(_one("*rk2-5_structure_events.csv"))
h1fib = fib_kinds(_one("*rk2-5_fib_lifecycle.csv"), ("structure_id", "cycle_id"))
for sid in sorted(h1ev["m"].map(lambda m: m.get("structure_id")).dropna().unique()):
    ev = h1ev[h1ev["m"].map(lambda m: m.get("structure_id") == sid)]
    sd = int(ev["m"].iloc[0]["struct_direction"])
    prev = h1ev[h1ev["m"].map(lambda m: m.get("structure_id") == sid - 1)]
    rc = prev[prev["type"] == "REVERSAL_CANDIDATE"]
    ref = float(rc["m"].iloc[-1]["bos_frozen"]) if len(rc) else None
    show(f"H1 main sid {int(sid)} — started by {'reversal' if ref is not None else 'the first MS run (start trading)'}",
         sd, cycles_of(ev, sd), [("reversal bos_frozen", ref)], h1fib, (int(sid),), t_h1)


# --- H1 parent CTS thresholds (for first_confluence) ----------------------------------------------------------------
def parent_cts_threshold(S, C):
    """The CTS threshold a pattern broke to establish parent (S, C)'s CTS: cycle C-1's cts_threshold as of (S, C)'s
    CTS_ESTABLISHED moment — the last CTS_THRESHOLD_UPDATED of (S, C-1) before it, else (S, C-1)'s confirmed price."""
    if C == 0:
        return None
    ev = h1ev[h1ev["m"].map(lambda m: m.get("structure_id") == S)]
    est = ev[(ev["type"] == "CTS_ESTABLISHED") & ev["m"].map(lambda m: m.get("cycle_id") == C)]
    moment = int(est["idx"].iloc[0])
    prior = ev[ev["m"].map(lambda m: m.get("cycle_id") == C - 1) & (ev["idx"] < moment)]
    thr = None
    for _, r in prior.sort_values("idx", kind="stable").iterrows():
        if r["type"] in ("CTS_CONFIRMED", "CTS_THRESHOLD_UPDATED"):
            thr = float(r["price"])
    return thr


# --- M15 unique subs ------------------------------------------------------------------------------------------------
subs = {}
for lens in ("confluence", "counter"):
    s = pd.read_csv(_one(f"*_M15_{lens}_subs.csv"))
    for _, r in s.iterrows():
        subs.setdefault(int(r["sub_id"]), (lens, r))
lens_ev = {lens: load_events(_one(f"*_M15_{lens}_structure_events.csv")) for lens in ("confluence", "counter")}
lens_fib = {lens: fib_kinds(_one(f"*_M15_{lens}_fib_lifecycle.csv"), ("sub_id", "structure_id", "cycle_id"))
            for lens in ("confluence", "counter")}


def sub_events(sid):
    lens, _ = subs[sid]
    ev = lens_ev[lens]
    return ev[ev["m"].map(lambda m: m.get("sub_id") == sid)]


for sid in sorted(subs):
    lens, r = subs[sid]
    sd = int(r["direction"])
    ev = sub_events(sid)
    trig, S, C, tidx = (r["first_record_trigger_type"], int(r["first_record_parent_sid"]),
                        int(r["first_record_parent_cycle_id"]), int(r["first_record_trigger_idx"]))
    rows = []
    if trig == "reversal":
        rev = [o for o in subs if o != sid and int(subs[o][1]["direction"]) == -sd
               and pd.notna(subs[o][1]["natural_reversal_idx"]) and int(subs[o][1]["natural_reversal_idx"]) == tidx]
        assert len(rev) == 1, (sid, rev)
        oev = sub_events(rev[0])
        rc = oev[(oev["type"] == "REVERSAL_CANDIDATE") & oev["m"].map(lambda m: m.get("apply_idx") == tidx)]
        rows.append((f"bos_frozen of sub {rev[0]}'s applied reversal", float(rc["m"].iloc[0]["bos_frozen"])))
    elif trig == "first_confluence":
        rows.append((f"H1 ({S},{C - 1}) cts_threshold broken by ({S},{C}) CTS_ESTABLISHED" if C else
                     "parent cycle 0: none", parent_cts_threshold(S, C)))
    else:
        started = {o: subs[o][1] for o in subs if o != sid and int(subs[o][1]["start_idx"]) <= tidx}
        opp = [o for o in started if int(started[o]["direction"]) == -sd]
        if opp:
            o = max(opp, key=lambda k: int(started[k]["start_idx"]))
            rows.append((f"(a) most recent opposite-direction sub {o}: its bos_threshold at {tidx}",
                         bos_threshold_at(sub_events(o), tidx)))
        same_lens = [o for o in started if lens in ast.literal_eval(started[o]["lenses"])]
        if same_lens:
            o = max(same_lens, key=lambda k: int(started[k]["start_idx"]))
            rows.append((f"(b) previous sub on the {lens} lens, sub {o} (dir {int(started[o]['direction']):+d}): "
                         f"its bos_threshold at {tidx}", bos_threshold_at(sub_events(o), tidx)))
        repl = [o for o in started if started[o]["end_reason"] == "same_dir_replacement"
                and pd.notna(started[o]["end_idx"]) and int(started[o]["end_idx"]) == int(r["start_idx"])]
        for o in repl:
            rows.append((f"(c) the sub it replaces, sub {o} (dir {int(started[o]['direction']):+d}): "
                         f"its bos_threshold at {tidx}", bos_threshold_at(sub_events(o), tidx)))
    show(f"M15 sub {sid} ({lens}{'+' if r['lenses'].count(',') > 1 else ''}) starting {int(r['starting_idx'])} "
         f"({t_m15(int(r['starting_idx']))}) live {int(r['start_idx'])}–{r['end_idx']} — FIRST trigger {trig} "
         f"parent ({S},{C}) at {tidx} ({t_m15(tidx)})", sd, cycles_of(ev, sd), rows, lens_fib[lens], (sid, 0), t_m15)
