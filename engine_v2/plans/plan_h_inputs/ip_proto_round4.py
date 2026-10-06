"""Institutional-pattern (IP) IC prototype — round 4 readings (user answers 2026-10-01 / 10-02).

Usage (repo root):  PYTHONPATH=. python engine_v2/plans/plan_h_inputs/ip_proto.py [LENS] [DIR] [JUNCTION] [VDOJI]
  LENS      H1 (default) | confluence | counter   — the M15 lenses rebuild their frame from the save's M15 chart
  DIR       agreed (default, V1 / V1a round 4) | off (candle direction only, round 2)
  JUNCTION  best (default, provisional J1) | ab (rule A / B literally)
  VDOJI     skip (default, provisional VD) | fail

DECIDED (PLAN_H §6):
  - candidates: §3.1 + touches the 61.8-80 zone (inclusive); the IC = the LAST candle of its opp leg;
  - AGREED DIRECTION (V1 / V1a, 2026-10-02): candle direction and volume direction (sign(c - prev c)) must agree;
    when one of them is 0 the other decides; both 0 = a TRUE DOJI; opposite = a SPLIT candle (breaks every leg);
  - legs = the max consecutive run available, capped at 3 (initial: most recent; continuation: first), cut at the
    BOS / CTS anchors (B1a); true dojis never break a leg and are full members of the leg they join: they count
    toward the cap and make the leg multi (P6b / P6c); a pattern may start or end on a doji (P6d);
  - true doji between initial and opp: the initial leg unless it is at the cap (then opp) (P6a A / B); between opp and
    continuation: always the continuation (P6a);
  - stacked legs = the SAME classifier + params as a single candle (P2); big ratio against the 5 marus before the
    pattern's first candle (P3);
  - ranks 1A..3D, letter then shape; tiebreakers (TB, round 4): IC type maru > normal > pinbar, then the most recent
    IC before the OLDEST unfilled sd imbalance.
PROVISIONAL (PLAN_H §5, round 4):
  - J1 JUNCTION 'best': a true doji between initial and opp goes where it yields the better pattern (rank, shape);
    equal -> rule A / B. 'ab' = rule A / B always (then a single big_normal + doji can never be 3D);
  - VD VDOJI 'skip': a volume-only doji (direction 0, vol_dir -sd) after the IC stays in the opp leg and the IC remains
    the last BODIED opp candle; 'fail': it is the last opp candle, so the IC is not last -> no pattern from that leg;
  - TB2 'oldest unfilled imbalance' = the earliest unfilled sd imbalance after the OLDEST tied IC (its c2 > the IC).
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
DIRMODE = sys.argv[2] if len(sys.argv) > 2 else "agreed"
JUNCTION = sys.argv[3] if len(sys.argv) > 3 else "best"
VDOJI = sys.argv[4] if len(sys.argv) > 4 else "skip"
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
    if DIRMODE == "off":
        return d
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


def rank_of(init_i, opp_i, cont_i, ic):
    """(rank, shape, detail) or (None, reason)."""
    ref = float(df.at[init_i[0], "prior_maru_max_len_as0"])
    a, m, e = stacked(init_i, ref), stacked(opp_i, ref), stacked(cont_i, ref)
    detail = (f"{len(init_i)}-{len(opp_i)}-{len(cont_i)} [{init_i[0]}..{init_i[-1]}|{opp_i[0]}..{opp_i[-1]}|"
              f"{cont_i[0]}..{cont_i[-1]}] init {a['typ']} r{a['ratio']:.2f} / cont {e['typ']} r{e['ratio']:.2f} "
              f"body {e['body']/PIP:.1f}p vs opp {m['body']/PIP:.1f}p")
    if not (e["big_maru"] and e["body"] > m["body"]):
        return None, "cont fails: " + detail
    mi, ci = len(init_i) > 1, len(cont_i) > 1
    if len(opp_i) > 1:
        if a["typ"] == "maru":
            return ("3C", SHAPE[(mi, ci)], detail), None
        if a["big_normal"] and not mi:
            return ("3D", SHAPE_3D[(mi, ci)], detail), None
        return None, "init fails (multi opp): " + detail
    mid = df.at[ic, "candle_type"]
    if a["typ"] == "maru":
        return ({"maru": "1A" if a["big_maru"] else "1B", "normal": "2A", "pinbar": "3A"}[mid], SHAPE[(mi, ci)],
                detail), None
    if a["big_normal"]:
        return ({"maru": "2B", "normal": "2C", "pinbar": "3B"}[mid], SHAPE[(mi, ci)], detail), None
    return None, "init fails: " + detail


def key(res):
    return (RANKS.index(res[0]), SHAPES.index(res[1]))


def classify(ic, sd, bos, cts):
    notes = []
    if A[ic] != -sd:
        return None, [f"IC split (vol_dir {V[ic]:+d})"]
    # opp tail after the IC: true dojis, and (VDOJI skip) volume-only opp dojis; then an sd candle must follow
    j, tail_opp, tail = ic + 1, [], []
    while j < N and (A[j] == 0 or (VDOJI == "skip" and D[j] == 0 and A[j] == -sd)):
        tail.append(j)
        if A[j] == -sd:
            tail_opp = list(tail)  # everything up to the last volume-only opp doji stays in the opp leg
            notes.append(f"volume-only doji {j} after the IC")
        j += 1
    if j >= N or A[j] != sd:
        why = "next is opp (not the last of its opp leg)" if j < N and A[j] == -sd else f"next {j} not sd"
        return None, notes + [why]
    cont_start = (tail_opp[-1] + 1) if tail_opp else ic + 1
    cont, j = [], cont_start
    while j < N and j <= cts and len(cont) < CAP and A[j] in (0, sd):
        cont.append(j)
        j += 1
    if not any(A[x] == sd for x in cont):
        return None, notes + ["continuation has no sd candle before the cap / CTS"]
    # opp leg: back from the IC over opp candles and true dojis; leading true dojis = the junction
    opp, j = [ic] + tail_opp, ic - 1
    while j >= 0 and A[j] in (0, -sd):
        opp.insert(0, j)
        j -= 1
    junction = []
    while len(opp) > 1 and A[opp[0]] == 0:
        junction.append(opp.pop(0))
    k = j
    if k < 0 or A[k] != sd:
        return None, notes + [f"before opp: {k} not an initial candle (agreed {A[k] if k >= 0 else None})"]
    init, j = [], k
    while j >= 0 and len(init) < CAP and A[j] in (0, sd):
        if j < bos:
            notes.append(f"init cut at BOS {bos}")
            break
        init.insert(0, j)
        j -= 1
    if not init:
        return None, notes + ["no initial candle at or after the BOS anchor"]
    if any(A[x] == 0 for x in init + opp + cont):
        notes.append("true doji in pattern")
    # junction placement: s junction dojis to the initial leg, the rest to opp
    room = CAP - len(init)
    ab = min(room, len(junction))  # rule A / B, doji by doji
    options = [ab] if JUNCTION == "ab" else list(range(0, min(room, len(junction)) + 1))
    best, why = None, None
    for s in options:
        res, reason = rank_of(init + junction[:s], sorted(junction[s:] + opp), cont, ic)
        if res and (best is None or key(res) < key(best[0]) or (key(res) == key(best[0]) and s == ab)):
            best = (res, s)
        why = why or reason
    if junction:
        notes.append(f"junction dojis {junction}: {best[1] if best else '-'} to initial")
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


print(f"[{LENS} dir={DIRMODE} junction={JUNCTION} vdoji={VDOJI}]  frame {N} candles; direction 0: {(D == 0).sum()} "
      f"(true doji {(A == 0).sum() if DIRMODE == 'agreed' else '-'}); split {(A == SPLIT).sum()}")
for name, bos, cts, sd in fib_rows():
    bos_p = float(df.at[bos, "l"] if sd == 1 else df.at[bos, "h"])
    cts_p = float(df.at[cts, "h"] if sd == 1 else df.at[cts, "l"])
    rng = abs(cts_p - bos_p)
    z618, z80 = (cts_p - rng * .618, cts_p - rng * .80) if sd == 1 else (cts_p + rng * .618, cts_p + rng * .80)
    zlo, zhi = min(z618, z80), max(z618, z80)
    print(f"\n=== {name} sd={sd:+d}  BOS {bos} {bos_p:.5f} -> CTS {cts} {cts_p:.5f}  zone {zlo:.5f}-{zhi:.5f}")
    found = []
    for i in range(bos, cts + 1):
        if D[i] != -sd:
            continue
        h, l = float(df.at[i, "h"]), float(df.at[i, "l"])
        if not (h >= zlo and l <= zhi):
            continue
        if not has_unfilled_imbalance(df, i + 1, cts, check_to_idx=cts, direction=sd, fill_threshold=FILL,
                                      evaluated_at=None):
            continue
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
