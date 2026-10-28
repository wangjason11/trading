"""PLAN_U's prediction (2026-10-28): every H1 / M15 candle whose candle type changes at each landing.

  today (S_controls)  the fixed floor (H1 2.2 / M15 1.0), the engine's own `classify_candles`
  U2                  the rolling floor, natural window: p90 of the 90 most recent body-% pinbars strictly before
  U3                  the rolling floor, final window: p90 of the 90 most recent final pinbars strictly before

The floors are computed the way PLAN_U §3 specifies the engine to: on the raw `body_len` floats (price units),
numpy linear interpolation, a natural maru / normal with `body_len < floor - EPS` becomes a pinbar; the seed = the
history before idx 0 back to SEED_DAYS (both windows converge well before it — `window_natural_vs_final.txt` §7/§8).
The same sets are recomputed on bodies rounded to the 0.1 pip (the shadow scripts' way) and asserted equal.
`pinbar_dir` of a changed candle comes from `classify_candles` itself (the floor passed as a fixed value per candle
is not possible today, so it is re-derived with the same wick rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import _frames as F

from engine_v2.config import CONFIG
from engine_v2.features.candle_classifier import candle_params_for_timeframe
from engine_v2.features.candle_params import CandleParams
from engine_v2.features.candles_v2 import EPS
from engine_v2.multitf.data_bridge import fetch_lower_tf_data

WINDOW, PCT, SEED_DAYS = 90, 90, 60
TFS = ("H1", "M15")


def floors(body, ctype, source, eps=EPS):
    """Per-candle floor (NaN without a full window), the window strictly before the candle."""
    n = len(body)
    fl = np.full(n, np.nan)
    pins = []
    for i in range(n):
        if len(pins) >= WINDOW:
            fl[i] = float(np.percentile(body[pins[-WINDOW:]], PCT))
        demote = ctype[i] != "pinbar" and not np.isnan(fl[i]) and body[i] < fl[i] - eps
        if ctype[i] == "pinbar" or (source == "final" and demote):
            pins.append(i)
    return fl


def demoted(body, ctype, fl, eps):
    return (ctype != "pinbar") & (body < np.nan_to_num(fl, nan=-1.0) - eps)


def main():
    raws = F.fetch_frames(TFS)
    dist_frac = CandleParams().pinbar_distance
    out = []
    w = out.append
    for tf in TFS:
        raw = raws[tf]
        tmin = pd.to_datetime(raw["time"]).min()
        seed = fetch_lower_tf_data(CONFIG.pair, tf, tmin - pd.Timedelta(days=SEED_DAYS), tmin)
        seed = seed[pd.to_datetime(seed["time"]) < tmin]
        cols = [c for c in raw.columns if c in seed.columns]
        full = pd.concat([seed[cols], raw[cols]], ignore_index=True)
        s = len(seed)
        nat = F.classify(full, CandleParams())
        body = nat["body_len"].to_numpy(dtype=float)
        ctype = nat["candle_type"].to_numpy()
        body01 = np.round(body / F.PIP, 1)

        today = F.classify(raw, candle_params_for_timeframe(tf))["candle_type"].to_numpy()
        fr_nat = ctype[s:]
        types = {"today": today}
        fls = {}
        for name, source in (("U2", "natural"), ("U3", "final")):
            fl = floors(body, ctype, source)
            dm = demoted(body, ctype, fl, EPS)
            # the same set on 0.1-pip bodies (the shadow scripts' way)
            fl01 = np.round(floors(body01, ctype, source, eps=1e-9), 6)
            dm01 = demoted(body01, ctype, fl01, 1e-9)
            assert np.array_equal(dm, dm01), f"{tf} {name}: raw-float and 0.1-pip demotions differ"
            assert not np.isnan(fl[s:]).any(), f"{tf} {name}: a frame candle without a full window"
            t = np.where(dm, "pinbar", ctype)[s:]
            types[name] = t
            fls[name] = fl[s:] / F.PIP

        length = nat["candle_len"].to_numpy(dtype=float)[s:]
        up, lo = nat["upper_wick"].to_numpy(dtype=float)[s:], nat["lower_wick"].to_numpy(dtype=float)[s:]
        d = dist_frac * length
        pin_dir = np.where((up < d) & (lo > d), 1, np.where((lo < d) & (up > d), -1, 0))
        times = raw["time"].astype(str).to_numpy()
        b = body01[s:]

        w(f"## {tf}: {len(raw)} candles, seed {s} candles ({SEED_DAYS} days)")
        for a, c in (("today", "U2"), ("U2", "U3"), ("today", "U3")):
            ch = np.flatnonzero(types[a] != types[c])
            to_pin = int(((types[c] == "pinbar") & (types[a] != "pinbar"))[ch].sum())
            w(f"   {a} -> {c}: {len(ch)} candles change type ({to_pin} become pinbars, {len(ch) - to_pin} stop being "
              f"pinbars); from maru {int((fr_nat[ch] == 'maru').sum())}, from normal {int((fr_nat[ch] == 'normal').sum())}; "
              f"with a pin direction {int((pin_dir[ch] != 0).sum())}")
        for name in ("today", "U2", "U3"):
            vc = pd.Series(types[name]).value_counts()
            w(f"   {name:5s} types: maru {vc.get('maru', 0)}, normal {vc.get('normal', 0)}, pinbar {vc.get('pinbar', 0)}")
        w(f"\n   every candle whose type differs between any two of today / U2 / U3:")
        w(f"   {'idx':>5s}  {'time':25s}  {'body-%':6s}  {'body':>5s}  {'U2 floor':>8s}  {'U3 floor':>8s}  "
          f"{'today':6s}  {'U2':6s}  {'U3':6s}  pin_dir")
        diff = np.flatnonzero((types["today"] != types["U2"]) | (types["U2"] != types["U3"]))
        for i in diff:
            w(f"   {i:5d}  {times[i]:25s}  {fr_nat[i]:6s}  {b[i]:5.1f}  {fls['U2'][i]:8.3f}  {fls['U3'][i]:8.3f}  "
              f"{types['today'][i]:6s}  {types['U2'][i]:6s}  {types['U3'][i]:6s}  {pin_dir[i]:+d}")
        w("")

    with open(F.out_path(__file__), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(l for l in out if not l.startswith("   ") or "->" in l or "types:" in l))


if __name__ == "__main__":
    main()
