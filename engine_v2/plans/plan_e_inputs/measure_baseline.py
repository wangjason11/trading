"""Plan E baseline measurements on a /commit-save folder (read-only).

Usage: python engine_v2/plans/plan_e_inputs/measure_baseline.py <save_dir>

Re-derives the counts Plan E's per-stage predictions rest on:
  E1  rows carrying event meta `anchor_idx` per type / lens (+ levels rows)
  E2  CTS_ESTABLISHED / BOS_CONFIRMED row counts (the new meta keys' footprint)
  E4  CTS_ESTABLISHED + BOS_CONFIRMED idx vs confirmed_at (the flip's idx cells)
      and KL BOS-zone rows (source_event_idx re-valued)
  E3a fib_lifecycle rows whose end_idx / meta activated_at sit on a lagging EST anchor
"""
import ast
import glob
import os
import sys
from collections import Counter

import pandas as pd

save = sys.argv[1]
PFX = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"
LENSES = {"H1": "", "conf": "_M15_confluence", "counter": "_M15_counter"}


def meta(s):
    return ast.literal_eval(s) if isinstance(s, str) and s.startswith("{") else {}


est_lag = []
for lens, suf in LENSES.items():
    ev = pd.read_csv(os.path.join(save, f"{PFX}{suf}_structure_events.csv"))
    ev["m"] = ev["meta"].map(meta)
    has_anchor = ev[ev["m"].map(lambda m: "anchor_idx" in m)]
    print(f"[{lens}] events={len(ev)} anchor_idx rows by type: {dict(Counter(has_anchor['type']))}")
    for t in ("CTS_ESTABLISHED", "BOS_CONFIRMED"):
        sub = ev[ev["type"] == t]
        lags = [(int(r.idx), int(r.m["confirmed_at"]), r.m.get("sub_id"), r.m.get("cycle_id"))
                for r in sub.itertuples() if int(r.m["confirmed_at"]) != int(r.idx)]
        print(f"[{lens}] {t}: rows={len(sub)} idx!=confirmed_at: {len(lags)}")
        if t == "CTS_ESTABLISHED":
            for l in lags:
                print(f"      EST lag  idx={l[0]} moment={l[1]} sub={l[2]} cyc={l[3]}")
        else:
            gaps = sorted(l[1] - l[0] for l in lags)
            print(f"      BOS lag range {gaps[:1]}..{gaps[-1:]}"
                  + ("" if lens != "H1" else f"  pairs={[(l[0], l[1]) for l in lags]}"))
    upd = ev[ev["type"] == "CTS_UPDATED"]
    pat = upd[upd["m"].map(lambda m: m.get("via") != "replay_raw")]
    print(f"[{lens}] CTS_UPDATED rows={len(upd)} pattern-path={len(pat)}")
    kl = pd.read_csv(os.path.join(save, f"{PFX}{suf}_kl_zones.csv"))
    col = "source_kind" if "source_kind" in kl.columns else None
    if col:
        print(f"[{lens}] KL rows={len(kl)} by source_kind: {dict(Counter(kl[col]))}")
    else:
        print(f"[{lens}] KL rows={len(kl)} cols={list(kl.columns)[:12]}")

lv = pd.read_csv(os.path.join(save, f"{PFX}_structure_levels.csv"))
lv_m = lv["meta"].map(meta) if "meta" in lv.columns else pd.Series([{}] * len(lv))
print(f"[H1 levels] rows={len(lv)} with meta anchor_idx={int(lv_m.map(lambda m: 'anchor_idx' in m).sum())}"
      f" kinds={dict(Counter(lv.get('kind', lv.columns[:1])))}")

final = pd.read_csv(os.path.join(save, "NZD_USD_H1_2025-11-15_2026-01-20_final.csv"), nrows=1)
print("[H1 final.csv] has pending_reversal_anchor_idx:", "pending_reversal_anchor_idx" in final.columns)

for lens, suf in (("conf", "_M15_confluence"), ("counter", "_M15_counter")):
    fl = pd.read_csv(os.path.join(save, f"{PFX}{suf}_fib_lifecycle.csv"))
    print(f"[{lens}] fib_lifecycle rows={len(fl)} cols={list(fl.columns)}")
    for r in fl.itertuples():
        m = meta(getattr(r, "meta", "{}"))
        if getattr(r, "sub_id", None) in (3,) or str(getattr(r, "end_idx", "")) in ("2828.0", "2828"):
            print(f"      {lens} line {r.Index + 2}: " + ", ".join(f"{c}={getattr(r, c)}" for c in fl.columns if c != "meta")
                  + f" | activated_at={m.get('activated_at')}")
