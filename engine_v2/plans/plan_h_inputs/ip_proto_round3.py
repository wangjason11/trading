"""Institutional-pattern (IP) IC prototype — round 3 readings (user answers 2026-10-01 / 10-02).

Usage: python ip_proto.py [VOL_MODE] [CLIP_MODE] [STACK]    (defaults: qual clip full)

DECIDED (PLAN_H §6):
  - candidates: §3.1 + touches the 61.8-80 zone (inclusive); the IC = the LAST opp-qualified candle of its opp leg;
  - legs = the max consecutive run available, capped at 3 (initial: most recent; continuation: first); the initial leg
    never before the BOS anchor, the continuation never past the CTS anchor — a crossing run is CUT there (B1a);
  - every non-doji leg candle needs candle direction AND vol_dir with its leg (initial / continuation: sd; opp: -sd);
    a candle failing it is a hard break (V1);
  - dojis (direction 0) never break a leg; a doji between initial and opp joins the initial leg unless that leg is
    already at the cap (then the opp leg); a doji between opp and continuation joins the continuation (P6a);
  - stacked legs classified by the SAME classifier as a single candle (P2); big ratio = stacked length / the max length
    of the 5 marus strictly before the pattern's first candle, the CandleParams thresholds (P3, item 5);
  - ranks 1A..3D, letter then shape; tiebreakers among equal rank+shape (TB, 2026-10-02): IC type maru>normal>pinbar,
    then length within TIE_RATIO of the best = tie, then body within TIE_RATIO of the best = tie, then the most recent.
PROVISIONAL (open in PLAN_H §5):
  - VOL_MODE 'qual' = vol_dir in (leg dir, 0) — the GLOSSARY "qualified candle"; 'strict' = vol_dir == leg dir;
    'off' = candle direction only (round 2);
  - a doji's own vol_dir is not checked; a doji counts toward the cap and makes its leg multi;
  - the TB tie bands are measured against the BEST value in the group (not pairwise).
"""
import sys
import pandas as pd

sys.path.insert(0, r"C:\Users\wangj\OneDrive\Documents\codingproj\Project Retire\forex_engine_v2")
from engine_v2.patterns.imbalance import compute_imbalance, has_unfilled_imbalance
from engine_v2.features.candles_v2 import compute_candle_metrics, classify_candles
from engine_v2.features.candle_params import CandleParams

SAVE = (r"C:\Users\wangj\OneDrive\Documents\codingproj\Project Retire\forex_engine_v2\artifacts"
        r"\commits\week8-volmom-multitf\20260930_151531_8b11bc2")
PIP = 0.0001
CAP = 3
TIE_RATIO = 0.80
VOL_MODE = sys.argv[1] if len(sys.argv) > 1 else "qual"
CLIP_MODE = sys.argv[2] if len(sys.argv) > 2 else "clip"
STACK = sys.argv[3] if len(sys.argv) > 3 else "full"
PARAMS = CandleParams(pinbar_body_pip_floor=2.2)  # H1 floor

df = pd.read_csv(SAVE + r"\NZD_USD_H1_2025-11-15_2026-01-20_final.csv")
df = compute_imbalance(df)
fibs = pd.read_csv(SAVE + r"\NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5_fib_lifecycle.csv")
D = df["direction"].values
V = df["vol_dir"].values
N = len(df)

RANKS = ["1A", "1B", "2A", "2B", "2C", "3A", "3B", "3C", "3D"]
SHAPES = ["i", "ii", "iii", "iv"]
SHAPE = {(False, False): "i", (True, False): "ii", (False, True): "iii", (True, True): "iv"}
SHAPE_3D = {(False, False): "i", (False, True): "ii"}
TYPE_PREF = {"maru": 0, "normal": 1, "pinbar": 2}


def qual(j, dirn):
    if D[j] != dirn:
        return False
    if VOL_MODE == "off":
        return True
    return V[j] == dirn or (VOL_MODE == "qual" and V[j] == 0)


def stacked(idxs, ref):
    seg = df.loc[idxs]
    o, c = float(seg["o"].iloc[0]), float(seg["c"].iloc[-1])
    h, l = float(seg["h"].max()), float(seg["l"].min())
    if len(idxs) == 1:
        typ = df.at[idxs[0], "candle_type"]
    elif STACK == "full":
        one = pd.DataFrame({"o": [o], "h": [h], "l": [l], "c": [c]})
        one.attrs["pair"] = "NZD_USD"
        typ = classify_candles(compute_candle_metrics(one), PARAMS)["candle_type"].iloc[0]
    else:
        bp = round(abs(c - o) / (h - l), 2) if h > l else 0.0
        typ = "maru" if bp >= 0.65 else ("pinbar" if bp < 0.40 else "normal")
    ratio = round((h - l) / ref, 2) if ref > 0 else 0.0
    return dict(typ=typ, body=abs(c - o), ratio=ratio,
                big_maru=(typ == "maru" and ratio >= PARAMS.big_maru_threshold),
                big_normal=(typ == "normal" and ratio >= PARAMS.big_normal_threshold))


def legs(ic, sd, bos, cts):
    """(init, opp, cont) idx lists or None, + notes. The IC must already be opp-qualified."""
    notes = []
    # continuation side: dojis right after the IC belong to the continuation; the next directional candle decides
    j = ic + 1
    while j < N and D[j] == 0:
        j += 1
    if j >= N or not qual(j, sd):
        why = "next is opp (not the last of its opp leg)" if j < N and qual(j, -sd) else f"next {j} unqualified"
        return None, [why]
    cont, j = [], ic + 1
    while j < N and j <= cts and len(cont) < CAP and (D[j] == 0 or qual(j, sd)):
        cont.append(j)
        j += 1
    while cont and D[cont[-1]] == 0:  # a trailing doji only joins if a later sd candle is in the leg
        cont.pop()
    if j <= N - 1 and j > cts and len(cont) < CAP and qual(j, sd):
        notes.append(f"cont cut at CTS {cts}")
    # opp leg: walk back from the IC over opp-qualified candles and dojis
    opp, j = [ic], ic - 1
    while j >= 0 and (D[j] == 0 or qual(j, -sd)):
        opp.insert(0, j)
        j -= 1
    junction = []
    while len(opp) > 1 and D[opp[0]] == 0:  # leading dojis = between the initial and opp legs
        junction.append(opp.pop(0))
    junction.sort()
    k = j  # the candle before the opp span (+ junction dojis)
    if k < 0 or not qual(k, sd):
        return None, notes + [f"before opp: {k} not an initial candle"]
    init, j = [], k
    while j >= 0 and len(init) < CAP and (D[j] == 0 or qual(j, sd)):
        if j < bos:
            notes.append(f"init cut at BOS {bos}")
            break
        init.insert(0, j)
        j -= 1
    for z in junction:  # P6a: join the initial leg unless it is at the cap
        if len(init) < CAP:
            init.append(z)
        else:
            opp.insert(0, z)
    opp.sort()
    if any(D[x] == 0 for x in init + opp + cont):
        notes.append("doji in pattern")
    if not init or not cont:
        return None, notes
    return (init, opp, cont), notes


def classify(ic, sd, bos, cts):
    if not qual(ic, -sd):
        return None, [f"IC vol_dir {V[ic]} not opp"]
    lg, notes = legs(ic, sd, bos, cts)
    if lg is None:
        return None, notes
    init_i, opp_i, cont_i = lg
    ref = float(df.at[init_i[0], "prior_maru_max_len_as0"])
    a, m, e = stacked(init_i, ref), stacked(opp_i, ref), stacked(cont_i, ref)
    detail = (f"{len(init_i)}-{len(opp_i)}-{len(cont_i)} [{init_i[0]}..{init_i[-1]}|{opp_i[0]}..{opp_i[-1]}|"
              f"{cont_i[0]}..{cont_i[-1]}] init {a['typ']} r{a['ratio']:.2f} / cont {e['typ']} r{e['ratio']:.2f} "
              f"body {e['body']/PIP:.1f}p vs opp {m['body']/PIP:.1f}p")
    if not (e["big_maru"] and e["body"] > m["body"]):
        return None, notes + ["cont fails: " + detail]
    mi, ci = len(init_i) > 1, len(cont_i) > 1
    if len(opp_i) > 1:
        if a["typ"] == "maru":
            rank, shape = "3C", SHAPE[(mi, ci)]
        elif a["big_normal"] and not mi:
            rank, shape = "3D", SHAPE_3D[(mi, ci)]
        else:
            return None, notes + ["init fails (multi opp): " + detail]
    else:
        mid = df.at[ic, "candle_type"]
        if a["typ"] == "maru":
            rank = {"maru": "1A" if a["big_maru"] else "1B", "normal": "2A", "pinbar": "3A"}[mid]
        elif a["big_normal"]:
            rank = {"maru": "2B", "normal": "2C", "pinbar": "3B"}[mid]
        else:
            return None, notes + ["init fails: " + detail]
        shape = SHAPE[(mi, ci)]
    return (rank, shape, detail), notes


def pick(found):
    """found: [(rank, shape, ic)]. Rank + shape, then the TB bands against the group's best, then the most recent."""
    best = min((RANKS.index(r), SHAPES.index(s)) for r, s, _ in found)
    grp = [ic for r, s, ic in found if (RANKS.index(r), SHAPES.index(s)) == best]
    t = min(TYPE_PREF[df.at[i, "candle_type"]] for i in grp)
    grp = [i for i in grp if TYPE_PREF[df.at[i, "candle_type"]] == t]
    for f in (lambda i: float(df.at[i, "h"] - df.at[i, "l"]), lambda i: abs(float(df.at[i, "c"] - df.at[i, "o"]))):
        top = max(f(i) for i in grp)
        grp = [i for i in grp if top <= 0 or round(f(i) / top, 2) >= TIE_RATIO]
    return max(grp), len(grp)


for _, fb in fibs.iterrows():
    sid, cyc, bos, cts = int(fb.structure_id), int(fb.cycle_id), int(fb.bos_idx), int(fb.cts_idx)
    sd = int(df.at[cts, "struct_direction"])
    bos_p = float(df.at[bos, "l"] if sd == 1 else df.at[bos, "h"])
    cts_p = float(df.at[cts, "h"] if sd == 1 else df.at[cts, "l"])
    rng = abs(cts_p - bos_p)
    z618, z80 = (cts_p - rng * .618, cts_p - rng * .80) if sd == 1 else (cts_p + rng * .618, cts_p + rng * .80)
    zlo, zhi = min(z618, z80), max(z618, z80)
    print(f"\n=== H1 sid {sid} cyc {cyc} sd={sd:+d}  BOS {bos} {bos_p:.5f} -> CTS {cts} {cts_p:.5f}  "
          f"zone {zlo:.5f}-{zhi:.5f}  [vol={VOL_MODE} clip={CLIP_MODE} stack={STACK}]")
    found = []
    for i in range(bos, cts + 1):
        if D[i] != -sd:
            continue
        h, l = float(df.at[i, "h"]), float(df.at[i, "l"])
        if not (h >= zlo and l <= zhi):
            continue
        if not has_unfilled_imbalance(df, i + 1, cts, check_to_idx=cts, direction=sd, evaluated_at=None):
            continue
        res, notes = classify(i, sd, bos, cts)
        tag = f"{res[0]}.{res[1]}  {res[2]}" if res else "-"
        print(f"  CAND {i:4d} {df.at[i,'candle_type']:6s} vol {V[i]:+d} h {h:.5f} l {l:.5f}  -> {tag}"
              + (f"   notes: {'; '.join(notes)}" if notes else ""))
        if res:
            found.append((res[0], res[1], i))
    if found:
        ic, tied = pick(found)
        inner = float(df.at[ic, "h"] if sd == 1 else df.at[ic, "l"])
        lab = [f"{r}.{s}" for r, s, x in found if x == ic][0]
        print(f"  PICK {ic} {lab}  inner {inner:.5f}  (TB group after the bands: {tied})")
        for j in range(cts + 1, min(cts + 120, N)):
            hit = (float(df.at[j, "l"]) <= inner + 8 * PIP) if sd == 1 else (float(df.at[j, "h"]) >= inner - 8 * PIP)
            if hit:
                print(f"  first candle after CTS within 8p of the pick's inner: {j}")
                break
    else:
        print("  NO PATTERN -> no POI")
