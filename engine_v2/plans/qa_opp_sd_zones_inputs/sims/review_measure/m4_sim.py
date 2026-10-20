"""Independent re-implementation of PLAN_R §1 (ladders, sticky source sub, pass, lock, signed half-up gap, counters)
on H1 OHLC + parent zones (H1 pipeline capture) + sub zones (S_Q CSVs; the two new subs from shadow_full2.log's dump).
Variants: PASS_FROM = "usable" (closes tested only while the zone is a candidate) | "scan_start" (§1 literal: a zone
that becomes usable later is passed if ANY close since the scan start was beyond its outer).
SRC_QUAL = whether a candidate source sub must have a QUALIFYING zone (sd filter) to become the sd source."""
import os, sys, math, pickle, ast
import pandas as pd
R = os.path.dirname(os.path.abspath(__file__))
h1, m15 = pickle.load(open(os.path.join(R, "frames.pkl"), "rb"))
CY = pickle.load(open(os.path.join(R, "h1_cycles.pkl"), "rb"))
O = h1["o"].astype(float).values; H = h1["h"].astype(float).values; L = h1["l"].astype(float).values; C = h1["c"].astype(float).values
PIP = 0.0001; PROX = 8; NARROW = 40
S = "artifacts/commits/week9-entries-decisions/20261014_170919_da8c15e/NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"

def rnd(g): return int(math.floor(g + 0.5 + 1e-9))

# ---------- sub zones from S_Q ----------
subs = pd.concat([pd.read_csv(S + f"_M15_{l}_subs.csv") for l in ("counter", "confluence")]).drop_duplicates("sub_id")
key_of = {int(r.sub_id): (int(r.direction), int(r.starting_idx)) for r in subs.itertuples()}
ev = pd.concat([pd.read_csv(S + f"_M15_{l}_structure_events.csv") for l in ("counter", "confluence")])
ev["m"] = ev["meta"].apply(ast.literal_eval)
mom = {}
for r in ev.itertuples():
    if r.type in ("BOS_CONFIRMED", "CTS_CONFIRMED"):
        k = (key_of[r.m["sub_id"]], int(r.m["cycle_id"]), r.type)
        mom.setdefault(k, set()).add(int(r.idx))
for k, v in mom.items():
    assert len(v) == 1, (k, v)
mom = {k: next(iter(v)) for k, v in mom.items()}
ZONES = {}   # key -> list of dict(cyc, kind, inner, steps[(from_m15, outer)], moment)
seen = set()
for l in ("counter", "confluence"):
    kl = pd.read_csv(S + f"_M15_{l}_kl_zones.csv")
    for r in kl.itertuples():
        m = ast.literal_eval(r.meta); k = key_of[m["sub_id"]]; c = int(m["cycle_id"])
        if (k, c, "BOS") in seen or r.source_kind != "BOS":
            continue
        seen.add((k, c, "BOS"))
        outer_key = "bottom" if r.side == "buy" else "top"
        steps = [(int(s.get("expanded_idx", s.get("base_idx"))), float(s[outer_key])) for s in m["bounds_steps"]]
        ZONES.setdefault(k, []).append(dict(cyc=c, kind="BOS", inner=float(m["inner"]), steps=steps,
                                             moment=mom[(k, c, "BOS_CONFIRMED")]))
    po = pd.read_csv(S + f"_M15_{l}_poi_zones.csv")
    for r in po.itertuples():
        m = ast.literal_eval(r.meta); k = key_of[m["sub_id"]]; c = int(r.cycle_id)
        if (k, c, "POI") in seen:
            continue
        seen.add((k, c, "POI"))
        inner = float(r.top) if r.side == "buy" else float(r.bottom); outer = float(r.bottom) if r.side == "buy" else float(r.top)
        cc = mom.get((k, c, "CTS_CONFIRMED"))
        if cc is None:
            continue   # no CTS_CONFIRMED -> never usable (rule 4b)
        ZONES.setdefault(k, []).append(dict(cyc=c, kind="POI", inner=inner, steps=[(0, outer)], moment=cc))
# new subs (shadow_full2.log [dump] lines)
ZONES[(1, 3230)] = [dict(cyc=0, kind="BOS", inner=0.57442, steps=[(0, 0.57418)], moment=3232),
                    dict(cyc=0, kind="POI", inner=0.57658, steps=[(0, 0.57628)], moment=3489)]
ZONES[(1, 4183)] = [dict(cyc=0, kind="BOS", inner=0.57818, steps=[(0, 0.57811)], moment=4189)]
ZONES[(-1, 4179)] = []
# record starts per parent cycle (shadow_full2.log [sweep] record lines; earliest record per sub)
REC = {(0, 1): {(1, 2365): 2611, (-1, 2639): 2829, (1, 3230): 3283, (-1, 3304): 3491},
       (1, 2): {(-1, 3304): 3611, (1, 3621): 3743, (-1, 3760): 3819, (1, 4027): 4143, (-1, 4179): 4183, (1, 4183): 4187}}

def outer_at(z, m15_idx):
    o = z["steps"][0][1]
    for f, v in z["steps"]:
        if f <= m15_idx: o = v
    return o

def run(key, overlay=True, pass_from="usable", src_qual=False, verbose=False, hold=True):
    cy = CY[key]; sd = cy["sd"]; lo, hi = cy["lo"], cy["hi"]
    bos_in, bos_out, _, bos_steps = cy["bos"]; cts_in, cts_out, _, cts_steps = cy["cts"]
    def psteps(steps, side):
        k = "bottom" if side == "buy" else "top"
        return [(int(s.get("expanded_idx", s.get("base_idx"))), float(s[k])) for s in steps]
    P = []   # parent zones: (label, side, inner, steps_h1, last)
    pois = cy["pois"]
    poi_inner = None
    for (top, bot, side) in pois:
        inner = top if side == "buy" else bot; outer = bot if side == "buy" else top
        if sd * (inner - bos_in) > 0:   # closer than BOS
            P.append(dict(label="P.POI", side="sd", inner=inner, steps=[(0, outer)], last=False, par=True, kind="POI", cyc=None, sub=None))
            poi_inner = inner
    P.append(dict(label="P.BOS", side="sd", inner=bos_in, steps=psteps(bos_steps, cy["bos"][2]), last=True, par=True, kind="BOS", cyc=None, sub=None))
    P.append(dict(label="P.CTS", side="opp", inner=cts_in, steps=psteps(cts_steps, cy["cts"][2]), last=True, par=True, kind="CTS", cyc=None, sub=None))
    def h1_outer(z, h):
        o = z["steps"][0][1]
        for f, v in z["steps"]:
            if f <= h: o = v
        return o
    # closeness: sd side zones -> closest to opp = max inner if sd==1 else min; opp side -> min if sd==1 else max
    def better(side, a, b):
        return (sd * (a - b) > 0) if side == "sd" else (sd * (a - b) < 0)
    def beyond_outer(side, close, outer):   # pass test
        if side == "sd": return close < outer if sd == 1 else close > outer
        return close > outer if sd == 1 else close < outer
    passed = set(); lock = {"sd": False, "opp": False}; source = {"sd": None, "opp": None}
    prev_ref = {"sd": None, "opp": None}; prev_narrow = None; cnt = None; expected = "sd"; out = []; log = []
    rec = REC.get(key, {}) if overlay else {}
    def sub_side(k): return "sd" if k[0] == sd else "opp"
    def qualifies(side, z):
        if side == "opp": return True
        return poi_inner is not None and sd * (z["inner"] - poi_inner) < 0
    def usable(k, z, h):
        return z["moment"] <= 4 * h - 1 and rec[k] <= 4 * h - 1
    def zid(k, z): return f"{k}c{z['cyc']}.{z['kind']}"
    # scan_start variant: precompute per sub zone the first close (from scan start) beyond its outer, using its first outer
    def ever_passed_before(side, k, z, h):
        o = z["steps"][0][1]
        return any(beyond_outer(side, C[j], o) for j in range(lo, h))
    for h in range(lo, hi + 1):
        rungs = {}
        for side in ("sd", "opp"):
            # sticky source
            cands = [k for k in rec if sub_side(k) == side and rec[k] <= 4 * h - 1 and
                     any(usable(k, z, h) and (qualifies(side, z) if (src_qual and side == "sd") else True) for z in ZONES[k])]
            newer = [k for k in cands if source[side] is None or rec[k] > rec[source[side]]]
            if newer:
                source[side] = max(newer, key=lambda k: rec[k])
            zs = []
            if source[side] is not None:
                k = source[side]
                uz = [z for z in ZONES[k] if usable(k, z, h)]
                if uz:
                    mc = max(z["cyc"] for z in uz)
                    for z in uz:
                        if z["cyc"] == mc and qualifies(side, z):
                            if pass_from == "scan_start" and zid(k, z) not in passed and ever_passed_before(side, k, z, h):
                                passed.add(zid(k, z))
                            zs.append(dict(label=zid(k, z), side=side, inner=z["inner"], steps=None, z=z, last=False, par=False, kind=z["kind"]))
            pz = [p for p in P if p["side"] == side]
            rungs[side] = pz + zs
        refs = {}
        for side in ("sd", "opp"):
            last = [p for p in rungs[side] if p["last"]][0]
            if lock[side]:
                refs[side] = last; continue
            avail = [r for r in rungs[side] if r["label"] not in passed]
            # tie order: parent first, then sub BOS before POI (stable sort by key then strict 'better')
            order = sorted(avail, key=lambda r: (0 if r["par"] else 1, 0 if r["kind"] in ("BOS", "CTS") else 1))
            best = None
            for r in order:
                if best is None or better(side, r["inner"], best["inner"]): best = r
            if prev_ref[side] is not None and prev_ref[side]["label"] in passed and best["last"]:
                lock[side] = True
            refs[side] = best
        g = rnd(sd * (refs["opp"]["inner"] - refs["sd"]["inner"]) / PIP); narrow = g <= NARROW
        if narrow and not prev_narrow:
            cnt = {"sd": 0, "opp": 0}
        elif narrow and prev_narrow:
            for side in ("sd", "opp"):
                old = prev_ref[side]["inner"]; new = refs[side]["inner"]
                outward = (sd * (new - old) < 0) if side == "sd" else (sd * (new - old) > 0)
                if outward: cnt[side] = 0
        elif not narrow:
            cnt = None
        changed = any(prev_ref[s] is None or prev_ref[s]["label"] != refs[s]["label"] for s in ("sd", "opp")) or prev_narrow != narrow
        if changed:
            log.append(f"  {h}: sd={refs['sd']['label']}({refs['sd']['inner']}) opp={refs['opp']['label']}({refs['opp']['inner']}) gap={g}{' NARROW' if narrow else ''} cnt={cnt} lock={lock}")
        prev_ref = dict(refs); prev_narrow = narrow
        z = refs[expected]; side = expected
        if side == "sd": hit = (L[h] <= z["inner"] + PROX * PIP) if sd == 1 else (H[h] >= z["inner"] - PROX * PIP)
        else:            hit = (H[h] >= z["inner"] - PROX * PIP) if sd == 1 else (L[h] <= z["inner"] + PROX * PIP)
        if hit:
            if narrow and cnt[side] >= 1:
                log.append(f"  {h}: {side} BLOCKED on {z['label']} cnt={cnt}")
            else:
                if narrow: cnt[side] += 1
                out.append((h, side, z["label"], z["inner"]))
                log.append(f"  {h}: TRIGGER {side} {z['label']} {z['inner']} (l={L[h]} h={H[h]} c={C[h]}) gap={g}")
                expected = "opp" if expected == "sd" else "sd"
        # pass test on close(h), every non-last candidate rung
        for side in ("sd", "opp"):
            for r in rungs[side]:
                if r["last"] or r["label"] in passed: continue
                o = h1_outer(r, h) if r["par"] else outer_at(r["z"], 4 * h + 3)
                if beyond_outer(side, C[h], o):
                    passed.add(r["label"]); log.append(f"  {h}: close {C[h]} beyond {r['label']} outer {o} -> passed from {h+1}")
    if verbose: print("\n".join(log))
    return out

if __name__ == "__main__":
    print("=== step 1 (parent only) ===")
    for k in sorted(CY):
        print(k, CY[k]["lo"], CY[k]["hi"], run(k, overlay=False, verbose=True))
    for pf in ("usable", "scan_start"):
        for sq in (False, True):
            print(f"=== overlay pass_from={pf} src_qual={sq} ===")
            for k in ((0, 1), (1, 2)):
                print(k, run(k, overlay=True, pass_from=pf, src_qual=sq, verbose=(pf == "usable" and not sq) or (pf == "scan_start" and not sq)))
