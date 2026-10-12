"""Census on a save: the three Naming-Standard audit keys + KL / POI zone sizes (status quo for item 3)."""
import ast
import statistics as st
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

SAVE = Path(sys.argv[1])
P = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"
LENSES = {"H1": P, "confluence": P + "_M15_confluence", "counter": P + "_M15_counter"}
PIP = 1e-4


def q(vals):
    if not vals:
        return "n=0"
    s = sorted(vals)
    n = len(s)
    pct = lambda p: s[min(n - 1, int(round(p * (n - 1))))]
    return (f"n={n} min={s[0]:.1f} p25={pct(.25):.1f} med={st.median(s):.1f} p75={pct(.75):.1f} "
            f"p90={pct(.9):.1f} max={s[-1]:.1f}")


for lens, pre in LENSES.items():
    print(f"\n===== {lens}")
    ev = pd.read_csv(SAVE / f"{pre}_structure_events.csv")
    ev["m"] = ev["meta"].map(ast.literal_eval)
    # RANGE_STARTED
    rs = ev[ev.type == "RANGE_STARTED"]
    print("RANGE_STARTED by reason:", Counter(m.get("reason", "<label path: no reason>") for m in rs.m))
    lab = rs[rs.m.map(lambda m: "start_idx" in m)]
    gaps = [int(r.m["confirm_idx"]) - int(r.m["start_idx"]) for r in lab.itertuples()]
    eq = sum(int(r.m["confirm_idx"]) == int(r.idx) for r in lab.itertuples())
    print(f"  label path: {len(lab)} rows; confirm_idx == ev.idx on {eq}; confirm-start gap {Counter(gaps)}")
    for r in lab.head(3).itertuples():
        print(f"    e.g. ev.idx={r.idx} start_idx={r.m['start_idx']} confirm_idx={r.m['confirm_idx']} sid={r.m['structure_id']}")
    # STATE_CHANGED effective_idx
    sc = ev[ev.type == "STATE_CHANGED"]
    diff = sc[sc.apply(lambda r: int(r.m["effective_idx"]) != int(r.idx), axis=1)]
    print(f"STATE_CHANGED: {len(sc)} rows; effective_idx != ev.idx on {len(diff)}; reasons of those:",
          Counter(m.get("reason") for m in diff.m), "; all reasons:", Counter(m.get("reason") for m in sc.m))
    # KL zones
    kl = pd.read_csv(SAVE / f"{pre}_kl_zones.csv")
    kl["m"] = kl["meta"].map(ast.literal_eval)
    init_sizes, final_sizes, nexp, init_eq_base = [], [], Counter(), 0
    exp_rows = []
    for r in kl.itertuples():
        steps = r.m["bounds_steps"]
        init = steps[0]
        assert init["event"] == "INIT"
        init_eq_base += int(init["start_idx"] == r.m["base_idx"])
        init_sizes.append((init["top"] - init["bottom"]) / PIP)
        final_sizes.append((r.top - r.bottom) / PIP)
        nexp[len(steps) - 1] += 1
        for s in steps[1:]:
            exp_rows.append((r.source_kind, r.m["structure_id"], r.m["cycle_id"], s["event"], s["start_idx"],
                             r.m["confirmed_idx"], r.m.get("end_idx")))
    print(f"KL: {len(kl)} zones; INIT start_idx == base_idx on {init_eq_base}; expansions per zone {dict(nexp)}")
    print(f"  INIT size pips  {q(init_sizes)}")
    print(f"  final size pips {q(final_sizes)}")
    print(f"  by kind: ", {k: q([(t - b) / PIP for t, b, sk in zip(kl.top, kl.bottom, kl.source_kind) if sk == k])
                          for k in sorted(set(kl.source_kind))})
    print(f"  by base_pattern: ", Counter(m["base_pattern"] for m in kl.m))
    print(f"  expansion steps: {len(exp_rows)}; events {Counter(e[3] for e in exp_rows)}; "
          f"step start_idx < zone confirmed_idx on {sum(e[4] < e[5] for e in exp_rows)}")
    # POI zones
    pz = pd.read_csv(SAVE / f"{pre}_poi_zones.csv") if (SAVE / f"{pre}_poi_zones.csv").exists() else None
    if pz is not None:
        print(f"POI: {len(pz)} zones; size pips {q(list((pz.top - pz.bottom) / PIP))}")

raw = pd.read_csv(SAVE / f"{P.split('_sd-1')[0]}_raw.csv")
rng = (raw.h - raw.l) / PIP
tr = pd.concat([raw.h - raw.l, (raw.h - raw.c.shift()).abs(), (raw.l - raw.c.shift()).abs()], axis=1).max(axis=1) / PIP
atr = tr.rolling(14).mean().dropna()
print(f"\nH1 candle range pips {q(list(rng))}\nH1 ATR14 pips {q(list(atr))}")
