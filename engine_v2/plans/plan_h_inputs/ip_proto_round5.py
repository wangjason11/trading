"""Institutional-pattern (IP) IC prototype — round 5 readings (user answers 2026-10-01 .. 10-05).

Usage (repo root):  PYTHONPATH=. python engine_v2/plans/plan_h_inputs/ip_proto.py [LENS] [D3] [EDGE]
  LENS  H1 (default) | confluence | counter   — the M15 lenses rebuild their frame from the save's M15 chart
  D3    run (default) | last   — open D3c: a 3D initial = a 1-candle RUN, or the run's most recent candle (cap 1)
  EDGE  include (default) | best | trim   — open E1: a true doji at the pattern's OUTER edge (before the initial
        leg's first candle / after the continuation's last, the run ending there) is a member / placed where the
        pattern ranks better / never a member

DECIDED (PLAN_H §6):
  - AGREED DIRECTION (V1a): candle and volume direction (sign(c - prev c)) must agree; when one is 0 the other decides;
    both 0 = a TRUE DOJI; opposite = a SPLIT candle (breaks every leg). A volume-only doji is an ordinary directional
    candle: it can be the IC (VD, 2026-10-05) -> candidates = agreed -sd (not candle direction -sd);
  - candidates: §3.1 (with the agreed direction) + touches the 61.8-80 zone (inclusive); the IC = the LAST opp candle;
  - legs = the max consecutive run, capped at 3 (initial: most recent; continuation: first), cut at BOS / CTS (B1a);
    true dojis never break a leg and are FULL members of the leg they are assigned to (cap + multi; P6c-final);
  - true doji between initial and opp: tried in each leg, best (rank, shape) wins (J1 = a, 2026-10-05); joining the
    initial leg at the cap pushes its earliest candle out (the cap window slides); equal -> rule A / B;
    between opp and continuation: always the continuation (P6a);
  - stacked legs = the SAME classifier + params as a single candle (P2); big ratio against the 5 marus before the
    pattern's first candle (P3);
  - ranks 1A..3D, letter then shape; TB1 IC type maru > normal > pinbar; TB2 the most recent tied IC before the
    earliest unfilled sd imbalance after the OLDEST tied IC (TB2a, 2026-10-05).
"""
import base64
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\wangj\OneDrive\Documents\codingproj\Project Retire\forex_engine_v2")
from engine_v2.patterns.imbalance import compute_imbalance, has_unfilled_imbalance  # noqa: E402
from engine_v2.features.candles_v2 import compute_candle_features, compute_candle_metrics, classify_candles  # noqa: E402
from engine_v2.features.candle_params import CandleParams  # noqa: E402

SAVE = (r"C:\Users\wangj\OneDrive\Documents\codingproj\Project Retire\forex_engine_v2\artifacts"
        r"\commits\week8-volmom-multitf\20260930_151531_8b11bc2" + "\\")
PFX = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"
PIP = 0.0001
FILL = 0.70  # POIConfig.fill_threshold
LENS = sys.argv[1] if len(sys.argv) > 1 else "H1"
D3 = sys.argv[2] if len(sys.argv) > 2 else "run"
EDGE = sys.argv[3] if len(sys.argv) > 3 else "include"
CAP = 3  # H1 3, M15 3 (H1 decision)
FLOOR = 2.2 if LENS == "H1" else 1.0  # DEFAULT_PINBAR_BODY_PIP_FLOOR_BY_TF
PROX = 8 if LENS == "H1" else 5  # zone-proximity pips
PARAMS = CandleParams(pinbar_body_pip_floor=FLOOR)
SPLIT = 9


def load_frame():
    if LENS == "H1":
        return pd.read_csv(SAVE + "NZD_USD_H1_2025-11-15_2026-01-20_final.csv")
    sys.argv, keep = [sys.argv[0]], sys.argv  # chart_census runs a census on import when given args
    from engine_v2.debug.chart_census import load_fig
    sys.argv = keep
    traces, _ = load_fig(SAVE + f"{PFX}_M15_{LENS}.html")

    def arr(a):
        if isinstance(a, dict) and "bdata" in a:
            return np.frombuffer(base64.b64decode(a["bdata"]), dtype=np.dtype(a["dtype"]))
        return np.asarray(a)
    # the OHLC trace + the two imbalance-candle traces together are the M15 frame (entity index == row order)
    parts = [pd.DataFrame({"time": arr(t["x"]), "o": arr(t["open"]).astype(float), "h": arr(t["high"]).astype(float),
                           "l": arr(t["low"]).astype(float), "c": arr(t["close"]).astype(float)})
             for t in traces if t.get("type") == "candlestick"]
    raw = pd.concat(parts).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    raw["volume"] = 1.0  # vol_dir needs only closes; the column just switches the volume features on
    raw.attrs["pair"] = "NZD_USD"
    return compute_candle_features(raw, PARAMS)


df = compute_imbalance(load_frame())
D = df["direction"].values.astype(int)
V = df["vol_dir"].values.astype(int)
N = len(df)
IMBS = df.attrs["imbalances"]


def agreed(j):
    d, v = D[j], V[j]
    if d == 0:
        return v  # 0 = true doji
    return d if v in (0, d) else SPLIT


A = np.array([agreed(j) for j in range(N)])
RANKS = ["1A", "1B", "2A", "2B", "2C", "3A", "3B", "3C", "3D"]
SHAPES = ["i", "ii", "iii", "iv"]
SHAPE = {(False, False): "i", (True, False): "ii", (False, True): "iii", (True, True): "iv"}
SHAPE_3D = {(False, False): "i", (False, True): "ii"}
TYPE_PREF = {"maru": 0, "normal": 1, "pinbar": 2}


def stacked(idxs, ref):
    seg = df.loc[idxs]
    o, c = float(seg["o"].iloc[0]), float(seg["c"].iloc[-1])
    h, l = float(seg["h"].max()), float(seg["l"].min())
    if len(idxs) == 1:
        typ = df.at[idxs[0], "candle_type"]
    else:
        one = pd.DataFrame({"o": [o], "h": [h], "l": [l], "c": [c]})
        one.attrs["pair"] = "NZD_USD"
        typ = classify_candles(compute_candle_metrics(one), PARAMS)["candle_type"].iloc[0]
    ratio = round((h - l) / ref, 2) if ref > 0 else 0.0
    return dict(typ=typ, body=abs(c - o), ratio=ratio,
                big_maru=(typ == "maru" and ratio >= PARAMS.big_maru_threshold),
                big_normal=(typ == "normal" and ratio >= PARAMS.big_normal_threshold))


def rank_of(init_i, opp_i, cont_i, ic, only_3d=False):
    """(rank, shape, detail) or (None, reason). only_3d: the cap-1 3D reading (D3 'last') - accept 3D only."""
    ref = float(df.at[init_i[0], "prior_maru_max_len_as0"])
    a, m, e = stacked(init_i, ref), stacked(opp_i, ref), stacked(cont_i, ref)
    detail = (f"{len(init_i)}-{len(opp_i)}-{len(cont_i)} [{init_i[0]}..{init_i[-1]}|{opp_i[0]}..{opp_i[-1]}|"
              f"{cont_i[0]}..{cont_i[-1]}] init {a['typ']} r{a['ratio']:.2f} / cont {e['typ']} r{e['ratio']:.2f} "
              f"body {e['body']/PIP:.1f}p vs opp {m['body']/PIP:.1f}p")
    if not (e["big_maru"] and e["body"] > m["body"]):
        return None, "cont fails: " + detail
    mi, ci = len(init_i) > 1, len(cont_i) > 1
    if len(opp_i) > 1:
        if a["typ"] == "maru" and not only_3d:
            return ("3C", SHAPE[(mi, ci)], detail), None
        if a["big_normal"] and not mi:
            return ("3D", SHAPE_3D[(mi, ci)], detail), None
        return None, "init fails (multi opp): " + detail
    if only_3d:
        return None, "3D needs a multi opp leg"
    mid = df.at[ic, "candle_type"]
    if a["typ"] == "maru":
        return ({"maru": "1A" if a["big_maru"] else "1B", "normal": "2A", "pinbar": "3A"}[mid], SHAPE[(mi, ci)],
                detail), None
    if a["big_normal"]:
        return ({"maru": "2B", "normal": "2C", "pinbar": "3B"}[mid], SHAPE[(mi, ci)], detail), None
    return None, "init fails: " + detail


def key(res):
    return (RANKS.index(res[0]), SHAPES.index(res[1]))


def edge_variants(leg, lead):
    """EDGE variants of a leg whose outer end (lead: its start, else its end) may hold true dojis."""
    seq = leg if lead else leg[::-1]
    k = 0
    while k < len(seq) - 1 and A[seq[k]] == 0:
        k += 1
    if k == 0 or EDGE == "include":
        return [leg]
    if EDGE == "trim":
        return [leg[k:]] if lead else [leg[:len(leg) - k]]
    return [leg[t:] if lead else leg[:len(leg) - t] for t in range(k + 1)]


def classify(ic, sd, bos, cts):
    notes = []
    # after the IC: true dojis join the continuation (P6a); the first other candle must be agreed sd
    j = ic + 1
    while j < N and A[j] == 0:
        j += 1
    if j >= N or A[j] != sd:
        why = "next is opp (not the last of its opp leg)" if j < N and A[j] == -sd else f"next {j} not sd"
        return None, [why]
    run, j = [], ic + 1
    while j < N and j <= cts and A[j] in (0, sd):
        run.append(j)
        j += 1
    cont = run[:CAP]
    if not any(A[x] == sd for x in cont):
        return None, ["continuation has no sd candle before the cap / CTS"]
    conts = edge_variants(cont, lead=False) if len(run) <= CAP else [cont]
    if A[cont[-1]] == 0 and len(run) <= CAP:
        notes.append(f"EDGE doji ends the continuation ({cont[-1]})")
    # opp leg: back from the IC over opp candles and true dojis; leading true dojis = the junction
    opp, j = [ic], ic - 1
    while j >= 0 and A[j] in (0, -sd):
        opp.insert(0, j)
        j -= 1
    junction = []
    while len(opp) > 1 and A[opp[0]] == 0:
        junction.append(opp.pop(0))
    k = j
    if k < 0 or A[k] != sd:
        return None, notes + [f"before opp: {k} not an initial candle (agreed {A[k] if k >= 0 else None})"]
    full, j = [], k  # the whole initial run (cut at BOS); each placement caps it
    while j >= 0 and A[j] in (0, sd):
        if j < bos:
            notes.append(f"init cut at BOS {bos}")
            break
        full.insert(0, j)
        j -= 1
    if not full:
        return None, notes + ["no initial candle at or after the BOS anchor"]
    if A[full[0]] == 0 and len(full) <= CAP:
        notes.append(f"EDGE doji starts the initial run ({full[0]})")
    if any(A[x] == 0 for x in full + opp + cont):
        notes.append("true doji in pattern")
    ab = min(max(CAP - len(full), 0), len(junction))  # rule A / B: fill the initial leg up to the cap
    best, why = None, None
    for s_ in range(len(junction) + 1):
        seq = full + junction[:s_]
        opp_s = sorted(junction[s_:] + opp)
        init3 = seq[-CAP:]
        inits = [(x, False) for x in (edge_variants(init3, lead=True) if len(seq) <= CAP else [init3])]
        if D3 == "last" and len(seq) > 1:
            inits.append((seq[-1:], True))
        for init, only3d in inits:
            if not any(A[x] == sd for x in init):
                continue
            for c in conts:
                res, reason = rank_of(init, opp_s, c, ic, only_3d=only3d)
                if res and (best is None or key(res) < key(best[0])
                            or (key(res) == key(best[0]) and s_ == ab and best[1] != ab)):
                    best = (res, s_, init, c)
                why = why or reason
    if junction:
        notes.append(f"junction dojis {junction}: {best[1] if best else '-'} to initial (rule A/B: {ab})")
    if best and best[0][0] == "3D" and len(full) > 1:
        notes.append(f"3D on the run's last candle {best[2][0]} (run {full[0]}..{full[-1]})")
    if best and (best[2][0] != (full + junction[:best[1]])[-CAP:][0] or best[3] != cont):
        notes.append(f"EDGE variant used: init {best[2]} cont {best[3]}")
    return (best[0] if best else None), notes + ([] if best else [why])


def oldest_unfilled_after(i0, sd, cts):
    """The earliest unfilled sd imbalance with a c2 in (i0, cts] — §3.1's question, returning the instance."""
    hits = [x for x in IMBS if x.direction == sd and x.overlaps(i0 + 1, cts) and not x.is_filled(df, cts, FILL)]
    return min(hits, key=lambda x: x.start_idx) if hits else None


def pick(found, sd, cts):
    best = min(key(r) for r, _ in found)
    grp = [i for r, i in found if key(r) == best]
    t = min(TYPE_PREF[df.at[i, "candle_type"]] for i in grp)
    grp = [i for i in grp if TYPE_PREF[df.at[i, "candle_type"]] == t]
    imb = oldest_unfilled_after(min(grp), sd, cts)
    before = [i for i in grp if i < imb.start_idx]
    return max(before), grp, imb


def fib_rows():
    if LENS == "H1":
        fb = pd.read_csv(SAVE + f"{PFX}_fib_lifecycle.csv")
        return [(f"sid {r.structure_id} cyc {r.cycle_id}", int(r.bos_idx), int(r.cts_idx),
                 int(df.at[int(r.cts_idx), "struct_direction"])) for r in fb.itertuples()]
    fb = pd.read_csv(SAVE + f"{PFX}_M15_{LENS}_fib_lifecycle.csv")
    subs = pd.read_csv(SAVE + f"{PFX}_M15_{LENS}_subs.csv").set_index("sub_id")["direction"]
    out, seen = [], set()
    for r in fb.itertuples():
        if not (r.active or r.locked):
            continue
        sub = int(eval(r.meta)["sub_id"])
        k = (sub, int(r.bos_idx), int(r.cts_idx))
        if k in seen:
            continue
        seen.add(k)
        out.append((f"sub {sub} cyc {r.cycle_id} (parent {r.parent_sid}/{r.parent_cycle_id})", k[1], k[2],
                    int(subs[sub])))
    return out


print(f"[{LENS} D3={D3} edge={EDGE}]  frame {N} candles; direction 0: {(D == 0).sum()} "
      f"(true doji {(A == 0).sum()}); split {(A == SPLIT).sum()}")
for name, bos, cts, sd in fib_rows():
    bos_p = float(df.at[bos, "l"] if sd == 1 else df.at[bos, "h"])
    cts_p = float(df.at[cts, "h"] if sd == 1 else df.at[cts, "l"])
    rng = abs(cts_p - bos_p)
    z618, z80 = (cts_p - rng * .618, cts_p - rng * .80) if sd == 1 else (cts_p + rng * .618, cts_p + rng * .80)
    zlo, zhi = min(z618, z80), max(z618, z80)
    print(f"\n=== {name} sd={sd:+d}  BOS {bos} {bos_p:.5f} -> CTS {cts} {cts_p:.5f}  zone {zlo:.5f}-{zhi:.5f}")
    found = []
    for i in range(bos, cts + 1):
        if A[i] != -sd and D[i] != -sd:
            continue
        h, l = float(df.at[i, "h"]), float(df.at[i, "l"])
        if not (h >= zlo and l <= zhi):
            continue
        if not has_unfilled_imbalance(df, i + 1, cts, check_to_idx=cts, direction=sd, fill_threshold=FILL,
                                      evaluated_at=None):
            continue
        if A[i] != -sd:  # candle direction -sd but a split candle: not a candidate (V1a)
            print(f"  DROP {i:4d} {df.at[i,'candle_type']:6s} vol {V[i]:+d}  split candle (a candle-direction candidate)")
            continue
        if D[i] == 0:
            print(f"  NEW  {i:4d} volume-only doji (vol {V[i]:+d}, type {df.at[i,'candle_type']}) is a candidate (VD)")
        res, notes = classify(i, sd, bos, cts)
        tag = f"{res[0]}.{res[1]}  {res[2]}" if res else "-"
        print(f"  CAND {i:4d} {df.at[i,'candle_type']:6s} vol {V[i]:+d}  -> {tag}"
              + (f"   notes: {'; '.join(notes)}" if notes else ""))
        if res:
            found.append((res, i))
    if not found:
        print("  NO PATTERN -> no POI")
        continue
    ic, grp, imb = pick(found, sd, cts)
    lab = [f"{r[0]}.{r[1]}" for r, x in found if x == ic][0]
    inner = float(df.at[ic, "h"] if sd == 1 else df.at[ic, "l"])
    tb = (f"TB group {grp}; oldest unfilled sd imbalance after {min(grp)}: c2 {imb.start_idx}..{imb.end_idx}"
          if len(grp) > 1 else "TB not needed")
    print(f"  PICK {ic} {lab}  inner {inner:.5f}  ({tb})")
    for j in range(cts + 1, min(cts + 200, N)):
        if (float(df.at[j, "l"]) <= inner + PROX * PIP) if sd == 1 else (float(df.at[j, "h"]) >= inner - PROX * PIP):
            print(f"  first candle after CTS within {PROX}p of the pick's inner: {j}")
            break
