"""PLAN_U cold review (2026-10-28, reviewer A): an INDEPENDENT re-derivation of the U2 / U3 predictions from the plan's
literal text — its own floor loop (`floors`: natural vs final window), the literal seed guard (OANDA `get_history`, 400
TF-candle steps until 180 body-% pinbars, the seed strictly before the frame), checked against a 60-day seed. Its
results matched `plan_u_prediction.txt` on every count (PLAN_U §8). Use it as the oracle at U2 / U3: the engine's
types must equal its `U2` / `U3` columns.

Imports `candle_params_for_timeframe` (removed at U2), so run it against `b64b10a` or earlier — a worktree / copy
of that commit as REPO (the reviewer used one):
    PYTHONIOENCODING=utf-8 python engine_v2/plans/qa_pinbar_floor_inputs/review_rederive_predictions.py [REPO]
Caches the fetched frames in %TEMP%/plan_u_rederive_frames.pkl (delete it to refetch).
"""
import os, sys, pickle
from datetime import timedelta
REPO = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
import numpy as np
import pandas as pd

from engine_v2.config import CONFIG
from engine_v2.data.provider_oanda import get_history
from engine_v2.features.candle_classifier import candle_params_for_timeframe
from engine_v2.features.candle_params import CandleParams
from engine_v2.features.candles_v2 import EPS, classify_candles, compute_candle_metrics
try:  # since PLAN_U U1 (b64b10a) a params floor needs its per-candle array
    from engine_v2.features.candles_v2 import fixed_body_floor
except ImportError:  # before U1
    fixed_body_floor = None
from engine_v2.multitf.data_bridge import fetch_lower_tf_data
from engine_v2.run_replay import _floor_to_day_start, fetch_history_with_auto_extend

CACHE = os.path.join(os.environ["TEMP"], "plan_u_rederive_frames.pkl")
TFD = {"H1": timedelta(hours=1), "M15": timedelta(minutes=15)}


def classify(raw, params):
    d = raw.reset_index(drop=True).copy()
    d.attrs["pair"] = CONFIG.pair
    d = compute_candle_metrics(d)
    if fixed_body_floor is None:
        return classify_candles(d, params)
    return classify_candles(d, params, body_floor=fixed_body_floor(d, params))


def fetch_all():
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    h1, _, _ = fetch_history_with_auto_extend(
        pair=CONFIG.pair, timeframe=CONFIG.timeframe, start=_floor_to_day_start(CONFIG.start),
        end=CONFIG.end, lookback_days=183, min_history=50, extend_days=4, max_extend_iters=8)
    h1 = h1.reset_index(drop=True)
    h1s = pd.to_datetime(h1["time"].iloc[0], utc=True)
    h1e = pd.to_datetime(h1["time"].iloc[-1], utc=True)
    m15 = fetch_lower_tf_data(CONFIG.pair, "M15", h1s, h1e).reset_index(drop=True)
    frames = {"H1": h1, "M15": m15}
    seeds = {}
    for tf, fr in frames.items():
        first = pd.to_datetime(fr["time"].iloc[0], utc=True).to_pydatetime()
        steps = []
        for k in range(1, 9):
            s = get_history(pair=CONFIG.pair, timeframe=tf, start=first - k * 400 * TFD[tf], end=first)
            steps.append(s)
        seeds[tf] = steps
        # 60-day seed too (the script's)
    sixty = {tf: fetch_lower_tf_data(CONFIG.pair, tf, pd.to_datetime(frames[tf]["time"].iloc[0], utc=True) - pd.Timedelta(days=60),
                                     pd.to_datetime(frames[tf]["time"].iloc[0], utc=True)) for tf in frames}
    out = (frames, seeds, sixty)
    pickle.dump(out, open(CACHE, "wb"))
    return out


def floors(body, ctype, source, W=90, P=90.0):
    n = len(body)
    fl = np.full(n, np.nan)
    pins = []
    final = ctype.copy()
    for i in range(n):
        if len(pins) >= W:
            fl[i] = np.percentile(body[np.array(pins[-W:])], P)
        is_dem = ctype[i] in ("maru", "normal") and body[i] < fl[i] - EPS
        if is_dem:
            final[i] = "pinbar"
        if source == "natural":
            if ctype[i] == "pinbar":
                pins.append(i)
        else:
            if final[i] == "pinbar":
                pins.append(i)
    return fl, final


def main():
    frames, seeds, sixty = fetch_all()
    for tf in ("H1", "M15"):
        fr = frames[tf]
        first = pd.to_datetime(fr["time"].iloc[0], utc=True)
        print(f"## {tf}: frame {len(fr)} candles, first {first}, last {fr['time'].iloc[-1]}")
        # the seed guard, literal
        chosen = None
        for k, s in enumerate(seeds[tf], start=1):
            t = pd.to_datetime(s["time"], utc=True)
            incl = int((t >= first).sum())
            s2 = s[t < first].reset_index(drop=True)
            npins = int((classify(s2, CandleParams())["candle_type"] == "pinbar").sum())
            print(f"   step {k}: fetched {len(s)} rows ({incl} at/after the frame's first), kept {len(s2)}, "
                  f"first {s2['time'].iloc[0]}, body-% pinbars {npins}")
            if chosen is None and npins >= 180:
                chosen = (k, s2, npins)
        k, seed, npins = chosen
        print(f"   guard stops at step {k}: {len(seed)} candles, {npins} pinbars")
        cols = ["time", "o", "h", "l", "c", "volume"]
        today = classify(fr, candle_params_for_timeframe(tf))
        nat_fr = classify(fr, CandleParams())
        res = {}
        for seedname, sd in (("guard", seed), ("60d", sixty[tf][pd.to_datetime(sixty[tf]["time"], utc=True) < first])):
            full = pd.concat([sd[cols], fr[cols]], ignore_index=True)
            m = classify(full, CandleParams())
            s = len(sd)
            body = m["body_len"].to_numpy(float)
            ct = m["candle_type"].to_numpy()
            assert np.array_equal(ct[s:], nat_fr["candle_type"].to_numpy())
            for source in ("natural", "final"):
                fl, final = floors(body, ct, source)
                res[(seedname, source)] = (fl[s:], final[s:])
                assert not np.isnan(fl[s:]).any()
        for source in ("natural", "final"):
            a, b = res[("guard", source)], res[("60d", source)]
            print(f"   {source}: guard seed vs 60d seed: types differ on {int((a[1] != b[1]).sum())}, "
                  f"floors differ on {int((np.abs(a[0] - b[0]) > 1e-15).sum())}")
        tt = today["candle_type"].to_numpy()
        nt = nat_fr["candle_type"].to_numpy()
        isfp_today = (tt == "pinbar") & (nt != "pinbar")
        print(f"   today: floor pinbars {int(isfp_today.sum())} (maru {int((isfp_today & (nt=='maru')).sum())}, normal {int((isfp_today & (nt=='normal')).sum())})")
        u2 = res[("guard", "natural")][1]
        u3 = res[("guard", "final")][1]
        pdist = CandleParams().pinbar_distance
        L = nat_fr["candle_len"].to_numpy(float); up = nat_fr["upper_wick"].to_numpy(float); lo = nat_fr["lower_wick"].to_numpy(float)
        d = pdist * L
        hasdir = ((up < d) & (lo > d)) | ((lo < d) & (up > d))
        for a_name, a, b_name, b in (("today", tt, "U2", u2), ("U2", u2, "U3", u3), ("today", tt, "U3", u3)):
            ch = a != b
            become = ch & (b == "pinbar")
            print(f"   {a_name}->{b_name}: {int(ch.sum())} change ({int(become.sum())} become, {int((ch & (a=='pinbar')).sum())} stop); "
                  f"maru {int((ch & (nt=='maru')).sum())} normal {int((ch & (nt=='normal')).sum())}; pin dir {int((ch & hasdir).sum())}")
        for name, t in (("today", tt), ("U2", u2), ("U3", u3)):
            vc = pd.Series(t).value_counts()
            print(f"   {name} types maru/normal/pinbar: {vc.get('maru',0)}/{vc.get('normal',0)}/{vc.get('pinbar',0)}")
        fl2 = res[("guard", "natural")][0] / 0.0001
        fl3 = res[("guard", "final")][0] / 0.0001
        print(f"   U2 floor min {fl2.min():.4f}, U3 floor min {fl3.min():.4f}")
        # special-maru among the changed
        sm = nat_fr["is_special_maru"].to_numpy(bool)
        print(f"   special marus demoted U2 {int((sm & (u2=='pinbar')).sum())}, today {int((sm & (tt=='pinbar')).sum())}")
        # bodies exactly at floor (within 1e-9 pip) — EPS sensitivity
        body = nat_fr["body_len"].to_numpy(float)
        for nm, fl in (("U2", res[("guard", "natural")][0]), ("U3", res[("guard", "final")][0])):
            near = (nt != "pinbar") & (np.abs(body - fl) < 1e-9)
            print(f"   {nm}: maru/normal with body within 1e-9 of the floor: {int(near.sum())}; "
                  f"of those body<floor raw (no EPS): {int((near & (body < fl)).sum())}")
        # floors whose interpolation fraction is nonzero
        print()


if __name__ == "__main__":
    main()
