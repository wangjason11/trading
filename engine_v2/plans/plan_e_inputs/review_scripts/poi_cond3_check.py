"""Independent per-candle re-derivation of POI condition 3 against every POI's
`activation_history`, over the FULL window (2026-09-30 two-stroke review).

    PYTHONPATH=. python engine_v2/plans/plan_e_inputs/review_scripts/poi_cond3_check.py

Reads the replay outputs in artifacts/debug (run from the repo root after a replay).
Condition 3 at candle t: an sd-direction imbalance FORMED by t (`formed_at <= t`), its
formed prefix `[start, min(end, t-1)]` overlapping `[ic+1, t]`, and not two-stroke-filled
by t (`confirmed_fill_idx` None or > t) — `has_unfilled_imbalance(..., evaluated_at=t)`.
Imbalances: H1 from the exported `_imbalance_instances.csv`; M15 from the engine's own
`prepare_lower_tf_data` on a fresh OANDA fetch of the replay's M15 span (the sub projections
use the full-frame list). Per POI with history: every active stretch has cond3 TRUE at every
candle, every `D` has it FALSE, every inactive stretch FALSE until the next `A`. Also lists
(not errors) the never-active POIs and the candles where cond3 already held before the first
`A` — those are the sub lifecycle floor or conditions 1 / 5 (on the reference window every
such first `A` is the sub's `start_idx` or lands on a `CTS_UPDATED`).
Reference window 2026-09-30: H1 4 / confluence 23 / counter 10 POIs with history, all consistent.
"""
import ast, json, os
import pandas as pd
from engine_v2.multitf.data_bridge import fetch_lower_tf_data, prepare_lower_tf_data
from engine_v2.zones.poi_zones import _compute_fill_idx_cache
D = "artifacts/debug/"; S = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"
def M(x):
    try: return json.loads(x)
    except Exception: return ast.literal_eval(x)
# H1 instances: (start, end, dir, formed_at, confirmed)
h1i = pd.read_csv(D + S + "_imbalance_instances.csv")
H1 = [(int(r.start_idx), int(r.end_idx), int(r.direction), int(r.start_idx) + 1, M(r.meta)["confirmed_fill_idx"]) for r in h1i.itertuples()]
h1 = pd.read_csv(D + "NZD_USD_H1_2025-11-15_2026-01-20_final.csv")
s, e = pd.to_datetime(h1["time"].iloc[0], utc=True), pd.to_datetime(h1["time"].iloc[-1], utc=True)
m15 = prepare_lower_tf_data(fetch_lower_tf_data("NZD_USD", "M15", s, e))
cache = _compute_fill_idx_cache(m15, m15.attrs["imbalances"], 0.70)
M15 = [(i.start_idx, i.end_idx, i.direction, i.formed_at, cache[id(i)][1]) for i in m15.attrs["imbalances"]]
assert all(f == st + 1 for st, _, _, f, _ in M15), "formed_at != start+1 somewhere"

def cond3(insts, sd, ic, t):
    for st, en, d, fa, cf in insts:
        if d != sd or fa > t:
            continue
        if min(en, t - 1) < ic + 1:   # formed prefix [st, min(en, t-1)] vs [ic+1, t]; st <= t-1 already
            continue
        if cf is None or cf > t:
            return True
    return False

def check(tag, f, insts, last_idx):
    d = pd.read_csv(f)
    n_ok = 0; notes = []
    for _, r in d.iterrows():
        m = M(r.meta); sd = 1 if r.side == "buy" else -1; ic = int(r.ic_idx)
        who = f"sub{m['sub_id']}" if m.get("sub_id") is not None else "main"
        key = f"{tag} {who} s{m['structure_id']}c{m['cycle_id']} ic={ic}"
        hist = m["activation_history"]; end = m.get("end_idx")
        stop = (end if end is not None else last_idx + 1)
        start = max(int(m["cts_established_idx"]), ic)
        if not hist:
            on = [t for t in range(start, stop) if cond3(insts, sd, ic, t)]
            notes.append(f"{key}: never active; cond3 true on {len(on)} of {stop - start} candles in [{start},{stop})"
                         + (f" first {on[0]}" if on else ""))
            continue
        bad = []
        first_a = hist[0]["idx"]
        pre = [t for t in range(start, first_a) if cond3(insts, sd, ic, t)]
        if pre:
            notes.append(f"{key}: cond3 already true before the first A={first_a} on {len(pre)} candles (from {pre[0]}) -> cond 1/5 or the floor held it")
        for k, h in enumerate(hist):
            nxt = hist[k + 1]["idx"] if k + 1 < len(hist) else stop
            if h["active"]:
                off = [t for t in range(h["idx"], nxt) if not cond3(insts, sd, ic, t)]
                if off: bad.append(f"A@{h['idx']}: cond3 FALSE inside the active stretch at {off[:3]}")
            else:
                if cond3(insts, sd, ic, h["idx"]): bad.append(f"D@{h['idx']} ({h.get('reason')}): cond3 still TRUE")
                on = [t for t in range(h["idx"], nxt) if cond3(insts, sd, ic, t)]
                if on: bad.append(f"D@{h['idx']}: cond3 TRUE inside the inactive stretch at {on[:3]}")
        if bad: notes.append(f"{key}: " + "; ".join(bad))
        else: n_ok += 1
    print(f"{tag}: {len(d)} POIs, {n_ok} with history fully consistent with cond3")
    for x in notes: print("   ", x)
check("H1", D + S + "_poi_zones.csv", H1, len(h1) - 1)
check("confluence", D + S + "_M15_confluence_poi_zones.csv", M15, len(m15) - 1)
check("counter", D + S + "_M15_counter_poi_zones.csv", M15, len(m15) - 1)
