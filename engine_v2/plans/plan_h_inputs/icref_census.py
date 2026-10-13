"""Census of the user's IC reference conditions (2026-10-09 write-up) on a save — a FIRST reading, not decided.

SUPERSEDED for crosses since PLAN_O (2026-10-13): the reading below takes a cross's start cycle `a`; the rule is now
its TARGET cycle T (CTS_{T-1}, cycle T's pattern) — a re-run computes a cross's reference from its start cycle, wrong
whenever start != T-1. Kept as PLAN_K's record.

Reading used here (to be confirmed with the user):
  - the fib record's start cycle a (a cross: meta cross_start_cycle; a single: its cycle_id), target cycle T;
  - refs: CTS_{a-1} = the CTS_CONFIRMED price of cycle a-1 (a >= 1);
          STRUCT = the anchor counterpart of the structure's cross-fib limit P (only while T <= M, or M None and P set):
            reversal_bos_frozen / prev_sub_bos_threshold -> the previous structure's / sub's latest BOS_CONFIRMED price
            at the structure start; parent_cts_threshold -> the parent's CTS_CONFIRMED price of cycle c-1; none -> no ref;
  - chosen = the one further opposite (sd +1: the LOWER, sd -1: the HIGHER); equal -> STRUCT (RX1-K5);
  - idx bound (DECIDED, PLAN_H RX1-1 / RX1-2b / W1-W3): chosen CTS_{a-1} -> cycle a's CTS_ESTABLISHED
    pattern_anchor_idx; chosen STRUCT, a reversal start -> the realised reversal pattern's anchor (the reversing
    structure's REVERSAL_CANDIDATE whose apply == its STATE_CHANGED(to=reversal) idx); a non-reversal start -> the
    cross-limit breakout watch on P (every candle closing past P opens a watch: detect_best_for_anchor(A, sd, P)
    confirming within 5; the earliest confirmation wins, ties -> the earlier anchor); none -> the bound passes.
    (Earlier readings: cycle M's establishment pattern — wrong at M = 0; "the first CTS event past P" — sub 7 4178.)
  - pass: price sd +1 h < ref, sd -1 l > ref (strict); idx < bound (strict).
Usage (repo root): python engine_v2/plans/plan_h_inputs/icref_census.py <save> [price]
  price = the price condition only (no idx bound)
"""
import ast
import base64
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from engine_v2.zones.poi_zones import POIConfig, fib_zone_bounds  # noqa: E402

P = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"
SAVE = sys.argv[1]
PRICE_ONLY = len(sys.argv) > 2 and sys.argv[2] == "price"
RANKS = ["1A", "1B", "2A", "2B", "2C", "3A", "3B", "3C", "3D"]
SHAPES = ["i", "ii", "iii", "iv"]
TB1 = {"maru": 0, "normal": 1, "pinbar": 2}
CFG = POIConfig()


def m15_ohlc(lens):
    keep = sys.argv
    sys.argv = [sys.argv[0]]
    from engine_v2.debug.chart_census import load_fig
    sys.argv = keep
    traces, _ = load_fig(f"{SAVE}/{P}_M15_{lens}.html")

    def arr(a):
        if isinstance(a, dict) and "bdata" in a:
            return np.frombuffer(base64.b64decode(a["bdata"]), dtype=np.dtype(a["dtype"]))
        return np.asarray(a)
    parts = [pd.DataFrame({"time": arr(t["x"]), "h": arr(t["high"]).astype(float), "l": arr(t["low"]).astype(float)})
             for t in traces if t.get("type") == "candlestick"]
    return pd.concat(parts).drop_duplicates("time").sort_values("time").reset_index(drop=True)


def events(suffix):
    ev = pd.read_csv(f"{SAVE}/{P}{suffix}_structure_events.csv")
    ev["m"] = ev["meta"].apply(ast.literal_eval)
    ev["key"] = ev["m"].apply(lambda m: m.get("sub_id", m.get("structure_id")))
    ev["cyc"] = ev["m"].apply(lambda m: m.get("cycle_id"))
    return ev


def tables(ev):
    out = {"cts_conf": {}, "est_pat": {}, "bos": {}}
    for _, r in ev.iterrows():
        k = (r["key"], r["cyc"])
        if r["type"] == "CTS_CONFIRMED":
            out["cts_conf"][k] = (float(r["price"]), int(r["m"]["cts_anchor_idx"]), int(r["idx"]))
        elif r["type"] == "CTS_ESTABLISHED":
            out["est_pat"][k] = (int(r["m"]["pattern_anchor_idx"]), int(r["idx"]), float(r["price"]))
        elif r["type"] == "BOS_CONFIRMED":
            out["bos"].setdefault(r["key"], []).append((int(r["idx"]), float(r["price"]), int(r["m"]["bos_anchor_idx"]),
                                                       r["cyc"]))
    return out


def latest_bos(tab, key, at):
    rows = [b for b in tab["bos"].get(key, []) if b[0] <= at]
    return max(rows) if rows else None


h1 = pd.read_csv(f"{SAVE}/NZD_USD_H1_2025-11-15_2026-01-20_final.csv")
H1_EV = events("")
h1_tab = tables(H1_EV)
m15_ev = pd.concat([events("_M15_confluence"), events("_M15_counter")]).drop_duplicates(["idx", "type", "meta"])
m15_tab = tables(m15_ev)
M15_EV = m15_ev
subs = pd.concat([pd.read_csv(f"{SAVE}/{P}_M15_{l}_subs.csv") for l in ("confluence", "counter")]
                 ).drop_duplicates("sub_id").set_index("sub_id")


def struct_ref(lens, key, meta):
    src = meta.get("cross_fib_limit_source")
    if src in (None, "none", "prev_sub_no_cts"):
        return None, src
    if lens == "H1":
        assert src == "reversal_bos_frozen"
        b = latest_bos(h1_tab, key - 1, 10 ** 9)
        return b[1], f"{src}: sid {key - 1} BOS {b[2]}"
    start = int(subs.loc[key, "start_idx"])
    frm = meta.get("cross_fib_limit_from_sub_id")
    if src in ("reversal_bos_frozen", "prev_sub_bos_threshold"):
        b = latest_bos(m15_tab, frm, start)
        return b[1], f"{src}: sub {frm} BOS {b[2]}"
    if src == "parent_cts_threshold":
        ps, pc = int(subs.loc[key, "first_record_parent_sid"]), int(subs.loc[key, "first_record_parent_cycle_id"])
        c = h1_tab["cts_conf"][(ps, pc - 1)]
        return c[0], f"{src}: H1 ({ps},{pc - 1}) CTS {c[1]}"
    return None, f"{src}: NOT MAPPED"


_FEAT = {}


def _m15_features():
    if "df" not in _FEAT:
        from engine_v2.features.candles_v2 import compute_candle_features
        from engine_v2.features.candle_params import CandleParams
        from engine_v2.patterns.structure_patterns import BreakoutPatterns
        keep = sys.argv
        sys.argv = [sys.argv[0]]
        from engine_v2.debug.chart_census import load_fig
        sys.argv = keep
        traces, _ = load_fig(f"{SAVE}/{P}_M15_confluence.html")

        def arr(a):
            if isinstance(a, dict) and "bdata" in a:
                return np.frombuffer(base64.b64decode(a["bdata"]), dtype=np.dtype(a["dtype"]))
            return np.asarray(a)
        parts = [pd.DataFrame({"time": arr(t["x"]), "o": arr(t["open"]).astype(float),
                               "h": arr(t["high"]).astype(float), "l": arr(t["low"]).astype(float),
                               "c": arr(t["close"]).astype(float)})
                 for t in traces if t.get("type") == "candlestick"]
        raw = pd.concat(parts).drop_duplicates("time").sort_values("time").reset_index(drop=True)
        raw["volume"] = 1.0
        raw.attrs["pair"] = "NZD_USD"
        _FEAT["df"] = compute_candle_features(raw, CandleParams(pinbar_body_pip_floor=1.0))
        _FEAT["bp"] = BreakoutPatterns(_FEAT["df"])
    return _FEAT["df"], _FEAT["bp"]


def watch_break(start, sd, thr):
    """The cross-limit breakout watch on `thr` from `start`: (anchor, confirmed) of the earliest confirmation."""
    from engine_v2.patterns.structure_patterns import PatternStatus
    df, bp = _m15_features()
    best = None
    for A in range(int(start), len(df)):
        c = float(df.loc[A, "c"])
        if not (c > thr if sd == 1 else c < thr):
            continue
        ev = bp.detect_best_for_anchor(A, sd, thr)
        if ev is None:
            continue
        ap = ev.confirmation_idx if ev.status == PatternStatus.CONFIRMED else ev.end_idx
        if ap is None or ap > A + 5:
            continue
        if best is None or (ap, A) < (best[1], best[0]):
            best = (A, int(ap))
        if best is not None and A > best[1]:
            break
    return best


def break_idx(lens, key, meta, sd):
    src_ = meta.get("cross_fib_limit_source")
    if src_ == "reversal_bos_frozen":
        ev = H1_EV if lens == "H1" else M15_EV
        rk = key - 1 if lens == "H1" else meta.get("cross_fib_limit_from_sub_id")
        sc = ev[(ev["key"] == rk) & (ev["type"] == "STATE_CHANGED")
                & (ev["m"].apply(lambda m: m.get("to")) == "reversal")]
        assert len(sc) == 1, (lens, key, len(sc))
        apply_ = int(sc.iloc[0]["idx"])
        rc = ev[(ev["key"] == rk) & (ev["type"] == "REVERSAL_CANDIDATE")
                & (ev["m"].apply(lambda m: m.get("apply_idx")) == apply_)]
        assert len(rc) == 1, (lens, key, apply_, len(rc))
        return int(rc.iloc[0]["m"]["pattern_anchor_idx"])
    assert lens != "H1", (lens, key, src_)
    start = int(subs.loc[key, "starting_idx"])
    b = watch_break(start, sd, float(meta["cross_fib_limit_price"]))
    return None if b is None else b[0]


def run(lens):
    suffix = "" if lens == "H1" else f"_M15_{lens}"
    ohlc = h1 if lens == "H1" else m15_ohlc(lens)
    tab = h1_tab if lens == "H1" else m15_tab
    fib = pd.read_csv(f"{SAVE}/{P}{suffix}_fib_lifecycle.csv")
    ipc = pd.read_csv(f"{SAVE}/{P}{suffix}_ip_candidates.csv")
    if "sub_id" not in ipc:
        ipc["sub_id"] = ipc["structure_id"]
    ipc["sub_id"] = ipc["sub_id"].fillna(ipc["structure_id"]).astype(int)
    print(f"\n######## {lens}")
    for _, r in fib.iterrows():
        m = ast.literal_eval(r["meta"])
        key = m.get("sub_id", int(r["structure_id"]))
        T = int(r["cycle_id"])
        bos, cts = int(r["bos_idx"]), int(r["cts_idx"])
        g = ipc[(ipc["sub_id"] == key) & (ipc["cycle_id"] == T) & (ipc["bos_anchor_idx"] == bos)
                & (ipc["cts_anchor_idx"] == cts)]
        if g.empty:
            continue  # no POI record (gate) / a dead version
        sd = int(g.iloc[0]["sd"])
        a = int(m["cross_start_cycle"]) if m.get("cross_cycle") else T
        Mx = m.get("cross_fib_limit_cycle")
        refs = []
        if a >= 1:
            c = tab["cts_conf"][(key, a - 1)]
            refs.append(("CTS", c[0], f"CTS_{a - 1} {c[1]}", tab["est_pat"][(key, a)][0]))
        sp, sdesc = struct_ref(lens, key, m)
        struct_live = sp is not None and (Mx is None or T <= Mx)
        if struct_live:
            bound = break_idx(lens, key, m, sd)
            refs.append(("STRUCT", sp, sdesc, bound))
        if refs:
            # further opposite; EQUAL prices -> the STRUCTURE reference (PLAN_H RX1-K5 / PLAN_K K-Q5)
            if sd == 1:
                kind, ref, desc, bound = min(refs, key=lambda x: (x[1], 0 if x[0] == "STRUCT" else 1))
            else:
                kind, ref, desc, bound = max(refs, key=lambda x: (x[1], 1 if x[0] == "STRUCT" else 0))
        else:
            kind = ref = desc = bound = None
        # the zone vs the reference (structural impossibility)
        hb = float(ohlc.loc[bos, "h"]); lb = float(ohlc.loc[bos, "l"])
        hc = float(ohlc.loc[cts, "h"]); lc = float(ohlc.loc[cts, "l"])
        hi, lo = (hc, lb) if sd == 1 else (hb, lc)
        zlo, zhi = fib_zone_bounds(hi, lo, sd, CFG)
        impossible = ref is not None and ((sd == 1 and zlo >= ref) or (sd == -1 and zhi <= ref))
        cand = g[g["ic_idx"].notna()].copy()
        cand["ic"] = cand["ic_idx"].astype(int)

        def ok(i):
            if ref is None:
                return True
            p = float(ohlc.loc[i, "h"]) < ref if sd == 1 else float(ohlc.loc[i, "l"]) > ref
            return p and (PRICE_ONLY or bound is None or i < bound)
        cand["pass"] = cand["ic"].apply(ok)
        pats = cand[(cand["touches_zone"] == True) & cand["ip_rank"].notna()]  # noqa: E712
        surv = pats[pats["pass"]]

        def best(df):
            if df.empty:
                return None
            k = df.apply(lambda x: (RANKS.index(x["ip_rank"].split(".")[0]), SHAPES.index(x["ip_rank"].split(".")[1]),
                                    TB1[x["candle_type"]]), axis=1)
            top = k.min()
            tied = df[k == top]
            return tuple(sorted(tied["ic"])), tied.iloc[0]["ip_rank"]
        old = g[g["picked"] == True]  # noqa: E712
        old_pick = (int(old.iloc[0]["ic_idx"]), old.iloc[0]["ip_rank"]) if len(old) else None
        new = best(surv)
        flag = "SAME" if (old_pick is None and new is None) or (
            old_pick and new and old_pick[0] in new[0] and len(new[0]) == 1) else "CHANGED"
        if old_pick and new and old_pick[0] in new[0] and len(new[0]) > 1:
            flag = "TIE(recheck TB2)"
        print(f"\n{'sub' if lens != 'H1' else 'sid'} {key} cyc {T} (start cyc {a}, M={Mx}) sd={sd:+d} fib {bos}->{cts}"
              f" zone[{zlo:.5f},{zhi:.5f}] | refs {[(x[0], round(x[1], 5), x[2], x[3]) for x in refs]}"
              f" -> {kind} {ref} bound {bound}{'  ZONE-BEYOND-REF' if impossible else ''}")
        print(f"   cand {len(cand)} -> pass {int(cand['pass'].sum())} | touching {int((cand['touches_zone'] == True).sum())}"  # noqa: E712
              f" -> pass {int((cand['pass'] & (cand['touches_zone'] == True)).sum())}"  # noqa: E712
              f" | patterns(touch) {len(pats)} -> {len(surv)} | pick {old_pick} -> {new}  [{flag}]")
        if not struct_live and sp is not None:
            print(f"   (struct ref {sdesc} = {sp} dropped: T > M)")
        elif sp is None and sdesc not in (None, "none"):
            print(f"   (struct ref: {sdesc})")


for lens in ("H1", "confluence", "counter"):
    run(lens)
