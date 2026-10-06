"""Prototype of the institutional-pattern (IP) IC selection on the saved H1 frame.

DEFAULT READINGS (each is an open question in the spec discussion):
  R1 single-candle types = candle_type column (maru incl. special-maru + pip floor).
  R2 big flags of a segment = stacked length / prior-5-maru max length strictly before the
     segment's FIRST candle (single candle: the *_as0 column); big_maru >= 0.65, big_normal >= 0.5.
  R3 stacked type = round(body/len, 2) bands only (maru >= .65, pinbar < .40, else normal);
     no special-maru, no pip floor.
  R4 initial leg = last k in {1,2,3} candles before the opp leg, all direction == sd;
     continuation leg = first k in {1,2,3} candles after the IC, all direction == sd.
  R5 opp leg = maximal run of direction == -sd candles ending at the IC (a direction-0 candle stops it).
  R6 no BOS / CTS boundary on the legs (reported if crossed).
  R7 anchor 'big_normal' = stacked/single type normal AND big_normal ratio.
  R8 zone touch inclusive: IC.h >= zone_lo and IC.l <= zone_hi on the FINAL fib.
"""
import sys
import pandas as pd

sys.path.insert(0, r"C:\Users\wangj\OneDrive\Documents\codingproj\Project Retire\forex_engine_v2")
from engine_v2.patterns.imbalance import compute_imbalance, has_unfilled_imbalance

SAVE = (r"C:\Users\wangj\OneDrive\Documents\codingproj\Project Retire\forex_engine_v2\artifacts"
        r"\commits\week8-volmom-multitf\20260930_151531_8b11bc2")
PIP = 0.0001
CAP = 3

df = pd.read_csv(SAVE + r"\NZD_USD_H1_2025-11-15_2026-01-20_final.csv")
df = compute_imbalance(df)
fibs = pd.read_csv(SAVE + r"\NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5_fib_lifecycle.csv")

TIER_ORDER = ["1A", "1B", "2A", "2B", "2C", "3A", "3B", "3C"]
SHAPE_ORDER = {(1, 1): "i", (2, 1): "ii", (1, 2): "iii", (2, 2): "iv"}  # (init multi?, cont multi?) -> 1/2


def stacked(i0, i1):
    seg = df.iloc[i0:i1 + 1]
    o, c = float(seg["o"].iloc[0]), float(seg["c"].iloc[-1])
    h, l = float(seg["h"].max()), float(seg["l"].min())
    body, length = abs(c - o), h - l
    if i0 == i1:
        typ = df.at[i0, "candle_type"]
    else:
        bp = round(body / length, 2) if length > 0 else 0.0
        typ = "maru" if bp >= 0.65 else ("pinbar" if bp < 0.40 else "normal")
    prior = float(df.at[i0, "prior_maru_max_len_as0"])
    ratio = round(length / prior, 2) if prior > 0 else 0.0
    return dict(typ=typ, body=body, length=length, ratio=ratio,
                big_maru=(typ == "maru" and ratio >= 0.65), big_normal=(ratio >= 0.5))


def classify(ic, sd):
    """All qualifying (rank, detail) for IC `ic`; [] if none."""
    d = df["direction"].values
    # R5 opp leg
    s = ic
    while s - 1 >= 0 and d[s - 1] == -sd:
        s -= 1
    opp = stacked(s, ic)
    opp_n = ic - s + 1
    out = []
    for ki in range(1, CAP + 1):
        a0 = s - ki
        if a0 < 0 or any(d[j] != sd for j in range(a0, s)):
            break
        init = stacked(a0, s - 1)
        for kc in range(1, CAP + 1):
            c1 = ic + kc
            if c1 >= len(df) or any(d[j] != sd for j in range(ic + 1, c1 + 1)):
                break
            cont = stacked(ic + 1, c1)
            if not (cont["big_maru"] and cont["body"] > opp["body"]):
                continue
            mid = "multi" if opp_n > 1 else df.at[ic, "candle_type"]
            anchor_maru = init["typ"] == "maru"
            anchor_bm = init["big_maru"]
            anchor_bn = init["typ"] == "normal" and init["big_normal"]
            tier = None
            if mid == "multi":
                tier = "3C" if anchor_maru else None
            elif mid == "maru":
                tier = "1A" if anchor_bm else ("1B" if anchor_maru else ("2B" if anchor_bn else None))
            elif mid == "normal":
                tier = "2A" if anchor_maru else ("2C" if anchor_bn else None)
            elif mid == "pinbar":
                tier = "3A" if anchor_maru else ("3B" if anchor_bn else None)
            if tier is None:
                continue
            shape = SHAPE_ORDER[(1 if ki == 1 else 2, 1 if kc == 1 else 2)]
            out.append(((TIER_ORDER.index(tier), shape), f"{tier}.{shape}",
                        f"{ki}-{opp_n}-{kc} legs[{a0}..{s-1}|{s}..{ic}|{ic+1}..{c1}]"))
    return sorted(out)


for _, fb in fibs.iterrows():
    sid, cyc, bos, cts = int(fb.structure_id), int(fb.cycle_id), int(fb.bos_idx), int(fb.cts_idx)
    sd = int(df.at[cts, "struct_direction"]) if not pd.isna(df.at[cts, "struct_direction"]) else 0
    bos_p = float(df.at[bos, "l"] if sd == 1 else df.at[bos, "h"])
    cts_p = float(df.at[cts, "h"] if sd == 1 else df.at[cts, "l"])
    rng = abs(cts_p - bos_p)
    if sd == 1:
        z618, z80 = cts_p - rng * .618, cts_p - rng * .80
    else:
        z618, z80 = cts_p + rng * .618, cts_p + rng * .80
    zlo, zhi = min(z618, z80), max(z618, z80)
    print(f"\n=== H1 sid {sid} cyc {cyc} sd={sd:+d}  BOS {bos} {bos_p:.5f} -> CTS {cts} {cts_p:.5f}  "
          f"zone {zlo:.5f}-{zhi:.5f} ({(zhi-zlo)/PIP:.1f}p)")
    for i in range(bos, cts + 1):
        if df.at[i, "direction"] != -sd:
            continue
        h, l = float(df.at[i, "h"]), float(df.at[i, "l"])
        touch = h >= zlo and l <= zhi
        ov = max(0.0, min(h, zhi) - max(l, zlo)) / (h - l) if h > l else 0.0
        imb = has_unfilled_imbalance(df, i + 1, cts, check_to_idx=cts, direction=sd, evaluated_at=None)
        cand = touch and imb
        pats = classify(i, sd) if cand else []
        best = f"{pats[0][1]} {pats[0][2]}" if pats else "-"
        print(f"  {i:4d} {df.at[i,'candle_type']:6s} body {abs(df.at[i,'c']-df.at[i,'o'])/PIP:4.1f}p "
              f"len {(h-l)/PIP:4.1f}p ov {ov*100:5.1f}% touch={'Y' if touch else 'n'} imb={'Y' if imb else 'n'}"
              f"{'  CAND' if cand else ''}  best={best}  (+{max(0, len(pats)-1)} more)")
