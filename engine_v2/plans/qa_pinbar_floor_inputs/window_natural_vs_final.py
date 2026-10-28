"""Rolling floor: which pinbars fill the 90-pinbar window (2026-10-28, the user: "try both using body-% pinbars only
and also the ones the floor itself demoted ... see how much they would differ"; their guess: the demoted, small-body
candles may even out the high-volatility stretches).

Two windows, both strictly before the candle, floor = the percentile of their bodies (numpy linear interpolation),
a natural maru / normal with a body STRICTLY below its floor becomes a pinbar (today's `<` rule):
  natural : the 90 most recent body-% pinbars (the floor off) -- a candle's floor depends only on the candles before it
  final   : the 90 most recent FINAL pinbars (body-% pinbars + the ones the floor demoted) -- computed candle by
            candle, so each demotion moves the next floors.

Shadow only (the engine is not called with it). H1 + M15 only (the replay never builds an M5 frame). Each frame gets
a SEED: the history before the frame's first candle (fetched with `fetch_lower_tf_data`, classified, then dropped),
so every frame candle has a full window. The final window is run with two seed lengths (SEED_DAYS) to see whether
its frame result depends on where the computation started. Bodies rounded to the exact 0.1 pip; floors to 1e-6.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _frames as F

from engine_v2.config import CONFIG
from engine_v2.features.candle_classifier import candle_params_for_timeframe
from engine_v2.features.candle_params import CandleParams
from engine_v2.multitf.data_bridge import fetch_lower_tf_data

WINDOW = 90
PCTS = (80, 85, 90)
MAIN_P = 90
SEED_DAYS = (30, 60)               # calendar days before the frame's first candle; the first is the reported one
TFS = ("H1", "M15")
CANDLES_PER_DAY = {"H1": 24, "M15": 96}
HOURS_PER_CANDLE = {"H1": 1.0, "M15": 0.25}


def with_seed(raw: pd.DataFrame, tf: str, days: int):
    tmin = pd.to_datetime(raw["time"]).min()
    seed = fetch_lower_tf_data(CONFIG.pair, tf, tmin - pd.Timedelta(days=days), tmin)
    seed = seed[pd.to_datetime(seed["time"]) < tmin]
    cols = [c for c in raw.columns if c in seed.columns]
    return pd.concat([seed[cols], raw[cols]], ignore_index=True), len(seed)


def natural_floors(body, ctype, p):
    n = len(body)
    pos = np.flatnonzero(ctype == "pinbar")
    k = np.searchsorted(pos, np.arange(n), side="left")
    fl, oldest = np.full(n, np.nan), np.full(n, -1)
    for i in np.flatnonzero(k >= WINDOW):
        w = pos[k[i] - WINDOW:k[i]]
        fl[i], oldest[i] = round(float(np.percentile(body[w], p)), 6), w[0]
    return fl, oldest


def final_floors(body, ctype, p):
    n = len(body)
    fl, oldest = np.full(n, np.nan), np.full(n, -1)
    pins = []
    for i in range(n):
        if len(pins) >= WINDOW:
            w = pins[-WINDOW:]
            fl[i], oldest[i] = round(float(np.percentile(body[w], p)), 6), w[0]
        if ctype[i] == "pinbar" or body[i] < np.nan_to_num(fl[i], nan=-1.0):
            pins.append(i)
    return fl, oldest


def demoted(body, ctype, fl):
    return (ctype != "pinbar") & (body < np.nan_to_num(fl, nan=-1.0))


def pct(a, b):
    return a / b * 100 if b else float("nan")


def main():
    raws = F.fetch_frames(TFS)
    dist_frac = CandleParams().pinbar_distance
    out = []
    w = out.append
    for tf in TFS:
        raw = raws[tf]
        nf = len(raw)
        runs = {}
        for days in SEED_DAYS:
            full, s = with_seed(raw, tf, days)
            nat = F.classify(full, CandleParams())
            assert len(nat) - s == nf
            runs[days] = (nat, s)
        nat, s = runs[SEED_DAYS[0]]
        body = np.round(nat["body_len"].to_numpy(dtype=float) / F.PIP, 1)
        rng = np.round(nat["candle_len"].to_numpy(dtype=float) / F.PIP, 1)
        ctype = nat["candle_type"].to_numpy()
        length = nat["candle_len"].to_numpy(dtype=float)
        up, lo = nat["upper_wick"].to_numpy(dtype=float), nat["lower_wick"].to_numpy(dtype=float)
        d = dist_frac * length
        has_dir = ((up < d) & (lo > d)) | ((lo < d) & (up > d))
        frame = np.arange(len(nat)) >= s
        mar, nor = frame & (ctype == "maru"), frame & (ctype == "normal")
        w(f"## {tf}: frame candles {nf} (idx 0 = {raw['time'].iloc[0]}); seed {s} candles ({SEED_DAYS[0]} days) before it")
        w(f"   natural in the frame: maru {int(mar.sum())}, normal {int(nor.sum())}, pinbar {int((frame & (ctype == 'pinbar')).sum())}")

        # sanity: the natural floor without a seed (the 2026-10-27 measurement) = the seeded one wherever both exist
        fr = F.classify(raw, CandleParams())
        fb = np.round(fr["body_len"].to_numpy(dtype=float) / F.PIP, 1)
        f_only, _ = natural_floors(fb, fr["candle_type"].to_numpy(), MAIN_P)
        seeded, _ = natural_floors(body, ctype, MAIN_P)
        both = ~np.isnan(f_only)
        assert np.array_equal(f_only[both], seeded[s:][both]), "seeded natural floor differs from the frame-only one"
        w(f"   check: the seeded natural p{MAIN_P} floor = the frame-only one on all {int(both.sum())} candles where both exist")

        # 1. demotions, natural vs final window, p80 / p85 / p90 (+ today's fixed floor)
        fixed = candle_params_for_timeframe(tf).pinbar_body_pip_floor
        w(f"\n   1. demoted in the frame (all {nf} candles):")
        rows = [(f"fixed {fixed}", np.where(frame, fixed, np.nan))]
        fl_main = {}
        for p in PCTS:
            fn, on = natural_floors(body, ctype, p)
            ff, of = final_floors(body, ctype, p)
            rows += [(f"natural p{p}", fn), (f"final   p{p}", ff)]
            if p == MAIN_P:
                fl_main = {"natural": (fn, on), "final": (ff, of)}
        for name, fl in rows:
            dm = demoted(body, ctype, fl) & frame
            a, b = int((dm & mar).sum()), int((dm & nor).sum())
            w(f"      {name:12s}: demoted {int(dm.sum()):4d} ({pct(dm.sum(), nf):4.1f}% of the frame)  maru {a:4d} "
              f"({pct(a, mar.sum()):4.1f}%)  normal {b:4d} ({pct(b, nor.sum()):4.1f}%)  with a pin direction {int((dm & has_dir).sum())}")

        fn, on = fl_main["natural"]
        ff, of = fl_main["final"]
        dn, df_ = demoted(body, ctype, fn) & frame, demoted(body, ctype, ff) & frame
        w(f"\n   2. p{MAIN_P} overlap: demoted by both {int((dn & df_).sum())}, natural only {int((dn & ~df_).sum())}, "
          f"final only {int((df_ & ~dn).sum())}")

        # 3. the floors themselves, over the frame's candles
        w(f"\n   3. p{MAIN_P} floor over the frame's candles (pips):  p10 / p25 / median / p75 / p90 / min / max")
        for name, fl in (("natural", fn), ("final", ff)):
            v = fl[frame]
            q = np.percentile(v, (10, 25, 50, 75, 90))
            w(f"      {name:8s}: " + " / ".join(f"{x:.2f}" for x in q) + f"   min {v.min():.2f}  max {v.max():.2f}")
        gap = (fn - ff)[frame]
        w(f"      natural minus final: median {np.median(gap):+.2f}, p10 {np.percentile(gap, 10):+.2f}, "
          f"p90 {np.percentile(gap, 90):+.2f}; final lower on {int((gap > 1e-9).sum())}, equal on "
          f"{int((abs(gap) <= 1e-9).sum())}, higher on {int((gap < -1e-9).sum())} of {nf} candles")

        # 4. how far back the window reaches (the lag)
        hp = HOURS_PER_CANDLE[tf]
        w(f"\n   4. the window's reach back (from the candle to its window's oldest pinbar), in {tf} candles (hours):")
        for name, o in (("natural", on), ("final", of)):
            span = (np.arange(len(nat)) - o)[frame]
            q = np.percentile(span, (10, 50, 90))
            w(f"      {name:8s}: p10 {q[0]:.0f} ({q[0] * hp:.0f} h)  median {q[1]:.0f} ({q[1] * hp:.0f} h)  "
              f"p90 {q[2]:.0f} ({q[2] * hp:.0f} h)")

        # 5. tracking the current volatility: the floor vs the previous day's median candle range
        cpd = CANDLES_PER_DAY[tf]
        vol = pd.Series(rng).rolling(cpd).median().shift(1).to_numpy()
        ok = frame & ~np.isnan(vol)
        w(f"\n   5. Spearman correlation of the floor with the previous day's ({cpd} candles) median range, "
          f"{int(ok.sum())} candles:")
        for name, fl in (("natural", fn), ("final", ff)):
            c = pd.Series(fl[ok]).rank().corr(pd.Series(vol[ok]).rank())    # Spearman = Pearson on ranks (no scipy)
            w(f"      {name:8s}: {c:.3f}")

        # 6. week by week (Mon-Sun, the FX week: time + 3 h)
        t = pd.to_datetime(nat["time"]) + pd.Timedelta(hours=3)
        wk = t.dt.tz_localize(None).dt.to_period("W-SUN") if t.dt.tz is not None else t.dt.to_period("W-SUN")
        wk = wk.to_numpy()
        w(f"\n   6. week by week (p{MAIN_P}): candles | median body / median range of the week's candles | median floor "
          f"natural / final | demoted natural / final (% of the week's marus + normals)")
        for per in pd.unique(wk[frame]):
            m = frame & (wk == per)
            mn = m & ((ctype == "maru") | (ctype == "normal"))
            a, b = int((dn & m).sum()), int((df_ & m).sum())
            w(f"      {str(per.start_time.date())}: {int(m.sum()):4d} | {np.median(body[m]):4.1f} / {np.median(rng[m]):5.1f} | "
              f"{np.median(fn[m]):5.2f} / {np.median(ff[m]):5.2f} | {a:3d} ({pct(a, mn.sum()):4.1f}%) / {b:3d} ({pct(b, mn.sum()):4.1f}%)")

        # 7. does the final window's frame result depend on the seed length?
        nat2, s2 = runs[SEED_DAYS[1]]
        b2 = np.round(nat2["body_len"].to_numpy(dtype=float) / F.PIP, 1)
        c2 = nat2["candle_type"].to_numpy()
        assert np.array_equal(b2[s2:], body[s:]) and np.array_equal(c2[s2:], ctype[s:])
        w(f"\n   7. seed {SEED_DAYS[0]} days ({s} candles) vs {SEED_DAYS[1]} days ({s2} candles), the frame's candles:")
        for name, fun in (("natural", natural_floors), ("final", final_floors)):
            f1 = fun(body, ctype, MAIN_P)[0][s:]
            f2 = fun(b2, c2, MAIN_P)[0][s2:]
            d1, d2 = demoted(body[s:], ctype[s:], f1), demoted(b2[s2:], c2[s2:], f2)
            diff_fl = np.flatnonzero(np.abs(f1 - f2) > 1e-9)
            diff_ty = np.flatnonzero(d1 != d2)
            w(f"      {name:8s}: floors differ on {len(diff_fl)} candles"
              + (f" (idx {diff_fl[0]}..{diff_fl[-1]})" if len(diff_fl) else "")
              + f"; candle types differ on {len(diff_ty)}" + (f" (first idx {diff_ty[0]})" if len(diff_ty) else ""))

        # 8. how short a seed is enough: cut the long seed to its last D days, compare the frame with the long one
        t2 = pd.to_datetime(nat2["time"])
        t0 = t2.iloc[s2]
        ref = {name: demoted(b2[s2:], c2[s2:], fun(b2, c2, MAIN_P)[0][s2:])
               for name, fun in (("natural", natural_floors), ("final", final_floors))}
        w(f"\n   8. a seed cut to its last D days vs the {SEED_DAYS[1]}-day one (frame candles whose type differs; "
          f"'no floor' = frame candles without a full window):")
        for days in (2, 3, 4, 6, 8, 10, 12, 15, 20, 25):
            keep = np.flatnonzero((t2 >= t0 - pd.Timedelta(days=days)).to_numpy())
            bb, cc = b2[keep], c2[keep]
            ss = int((keep < s2).sum())
            npins = int((cc[:ss] == "pinbar").sum())
            parts = [f"D={days:2d}: seed {ss:4d} candles, {npins:3d} natural pinbars"]
            for name, fun in (("natural", natural_floors), ("final", final_floors)):
                fl = fun(bb, cc, MAIN_P)[0][ss:]
                dd = demoted(bb[ss:], cc[ss:], fl)
                parts.append(f"{name} differ {int((dd != ref[name]).sum()):3d}, no floor {int(np.isnan(fl).sum()):4d}")
            w("      " + " | ".join(parts))
        w("")

    with open(F.out_path(__file__), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
