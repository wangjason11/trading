"""PLAN_U U3 /compare: WHY each moved event / zone / WVMI record moved — the classification columns that differ
between U2 (the natural window) and U3 (the final window) in the window of each change `u3_compare.py` lists, plus
the moved records on both sides; then every other differing CSV's changed columns (positional, same rows).

    PYTHONIOENCODING=utf-8 python engine_v2/plans/qa_pinbar_floor_inputs/u3_causes.py BASE_DIR CUR_DIR [OUT.txt]

BASE_DIR = the S_U2 save, CUR_DIR = the U3 variant's artifacts/debug. The frames are re-fetched with the engine's own
path (cached in %TEMP%/plan_u_u3_causes.pkl) and classified twice through `compute_candle_features` with the floor
seed (= `apply_candle_classification`, which takes `CandleParams()`): `pinbar_floor_window_source` "natural" (U2)
and "final" (U3). The H1 types of both are checked against BASE's / CUR's `*_final.csv` first (they must be equal).
"""
from __future__ import annotations

import ast
import filecmp
import os
import pickle
import sys

import _frames  # noqa: F401  (sys.path)
import pandas as pd

from _frames import HERE
from engine_v2.config import CONFIG
from engine_v2.features.candle_params import CandleParams
from engine_v2.features.candles_v2 import compute_candle_features
from engine_v2.multitf.data_bridge import fetch_lower_tf_data
from engine_v2.run_replay import _floor_to_day_start, fetch_floor_seeds, fetch_history_with_auto_extend

CACHE = os.path.join(os.environ["TEMP"], "plan_u_u3_causes.pkl")
P = "NZD_USD_H1_2025-11-15_2026-01-20"
Q = P + "_sd-1_eps0p0001_rk2-5"
COLS = ["candle_type", "pinbar_dir", "normal_dir", "is_special_maru", "is_big_maru_as0", "is_big_normal_as0",
        "is_big_maru_as1", "is_big_normal_as1", "is_big_maru_as2", "is_big_normal_as2", "big_ratio_as0",
        "prior_maru_max_len_as0", "pat", "pat_status", "pat_start_idx", "pat_end_idx"]

# (TF, lens file suffix, label, first idx, last idx, event keys to print: (sub_id, cycle_id, type), WVMI key)
WINDOWS = [
    ("H1", "", "H1 WVMI (sid 0, cycle 0): FP 473 -> 468 (pullback momentum 0.237 -> 0.645); its LP rows' hovers",
     447, 474, [], ("", "0", "0")),
    ("M15", "_M15_confluence", "confluence sub 0 cycle 1: the BOS KL zone base 916 'no base star' -> 917 'base' "
     "(top 0.57066 -> 0.57065); WVMI (0, 0, 1) FB 918 -> 920 (breakout momentum 0.113 -> 0.221)", 910, 922, [],
     ("0", "0", "1")),
    ("M15", "_M15_confluence", "confluence sub 1 cycle 4: CTS_UPDATED 2364 double_maru -> one_maru_continuous",
     2358, 2365, [(1, 4, "CTS_UPDATED")], None),
]


def frames():
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    h1, _, _ = fetch_history_with_auto_extend(
        pair=CONFIG.pair, timeframe=CONFIG.timeframe, start=_floor_to_day_start(CONFIG.start),
        end=CONFIG.end, lookback_days=183, min_history=50, extend_days=4, max_extend_iters=8)
    h1 = h1.reset_index(drop=True)
    h1.attrs["pair"] = CONFIG.pair
    m15 = fetch_lower_tf_data(CONFIG.pair, "M15", pd.to_datetime(h1["time"].iloc[0], utc=True),
                              pd.to_datetime(h1["time"].iloc[-1], utc=True))
    seeds = fetch_floor_seeds(CONFIG.pair, CONFIG.timeframe, h1["time"].iloc[0], ("M15",))
    out = ({"H1": h1, "M15": m15}, seeds)
    pickle.dump(out, open(CACHE, "wb"))
    return out


def classify(raw, seed, source):
    return compute_candle_features(raw.copy(), CandleParams(pinbar_floor_window_source=source),
                                   anchor_shifts=(0, 1, 2), floor_seed=seed)


def events(d, sfx):
    ev = pd.read_csv(os.path.join(d, f"{Q}{sfx}_structure_events.csv"))
    rows = []
    for r in ev.itertuples():
        m = ast.literal_eval(r.meta)
        rows.append((m.get("sub_id"), m.get("cycle_id"), r.type, int(r.idx), m))
    return rows


def read(d, name):
    return pd.read_csv(os.path.join(d, name), dtype=str, keep_default_na=False)


def main():
    B, C = sys.argv[1], sys.argv[2]
    OUT = sys.argv[3] if len(sys.argv) > 3 else str(HERE / "u3_causes.txt")
    raws, seeds = frames()
    cls = {tf: (classify(raw, seeds[tf], "natural"), classify(raw, seeds[tf], "final")) for tf, raw in raws.items()}
    lines = []
    for side, d, k in (("BASE (S_U2)", B, 0), ("CUR (U3)", C, 1)):
        final = read(d, f"{P}_final.csv")
        assert (final["candle_type"].to_numpy() == cls["H1"][k]["candle_type"].to_numpy()).all(), side
        lines.append(f"H1 {('U2', 'U3')[k]} types re-derived here == {side} *_final.csv (all {len(final)})")
    for tf, sfx, label, lo, hi, keys, wkey in WINDOWS:
        u2, u3 = cls[tf]
        lines += ["", f"## {tf} {label}", f"   candles {lo}..{hi} whose classification differs (U2 -> U3):"]
        for i in range(lo, hi + 1):
            d = [f"{c} {u2.loc[i, c]} -> {u3.loc[i, c]}" for c in COLS
                 if c in u2.columns and str(u2.loc[i, c]) != str(u3.loc[i, c])]
            if d:
                lines.append(f"     {i} {u3.loc[i, 'time']} ({u3.loc[i, 'candle_type_before_floor']}, body "
                             f"{u3.loc[i, 'body_len'] / 1e-4:.1f}, floor U2 {u2.loc[i, 'pinbar_body_floor'] / 1e-4:.2f} "
                             f"U3 {u3.loc[i, 'pinbar_body_floor'] / 1e-4:.2f}): " + "; ".join(d))
        for side, d in (("S_U2", B), ("U3", C)):
            for sub, cyc, typ, idx, m in events(d, sfx):
                if (sub, cyc, typ) in keys and lo <= idx <= hi:
                    keep = {k: m[k] for k in ("cts_anchor_idx", "bos_anchor_idx", "pattern_anchor_idx", "via",
                                              "confirmation_method", "confirmed_at") if k in m}
                    lines.append(f"   {side} {typ} idx={idx} {keep}")
            if wkey:
                w = read(d, f"{Q}{sfx}_wvmi.csv")
                r = w[(w["sub_id"] == wkey[0]) & (w["bos_structure_id"] == wkey[1]) & (w["bos_cycle_id"] == wkey[2])]
                for row in r.to_dict("records"):
                    lines.append(f"   {side} WVMI " + ", ".join(f"{c}={row[c]}" for c in (
                        "fb_idx", "lb_idx", "fp_idx", "lp_idx", "breakout_momentum", "pullback_momentum", "status")))
    lines += ["", "## Every other differing CSV: its changed columns (positional, same row count)"]
    covered = {f"{P}_final.csv"}
    for name in sorted(os.listdir(B)):
        if not name.endswith(".csv") or name in covered or filecmp.cmp(os.path.join(B, name), os.path.join(C, name),
                                                                       shallow=False):
            continue
        a, z = read(B, name), read(C, name)
        if len(a) != len(z) or list(a.columns) != list(z.columns):
            lines.append(f"   {name.replace(Q, '')}: rows {len(a)} -> {len(z)} (not positional)")
            continue
        cols = [(c, int((a[c] != z[c]).sum())) for c in a.columns if (a[c] != z[c]).any()]
        lines.append(f"   {name.replace(Q, '')}: {len(a)} rows; " + ", ".join(f"{c} {n}" for c, n in cols))
        if name.endswith("_structure_events.csv"):
            for i in range(len(a)):
                if a.loc[i, "meta"] != z.loc[i, "meta"]:
                    ma, mz = ast.literal_eval(a.loc[i, "meta"]), ast.literal_eval(z.loc[i, "meta"])
                    d = {k: (ma.get(k), mz.get(k)) for k in sorted(set(ma) | set(mz)) if ma.get(k) != mz.get(k)}
                    lines.append(f"      row {i} {a.loc[i, 'type']} idx={a.loc[i, 'idx']} sub={ma.get('sub_id')} "
                                 f"cycle={ma.get('cycle_id')}: {d}")
    text = "\n".join(lines)
    print(text)
    open(OUT, "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
