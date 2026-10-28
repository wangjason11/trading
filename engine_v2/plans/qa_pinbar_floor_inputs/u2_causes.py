"""PLAN_U U2 /compare: WHY each moved event / zone / WVMI record moved — the classification columns that differ
between U1 (the fixed floor H1 2.2 / M15 1.0 pips) and U2 (the rolling floor + the seed) in the window of each
change `u2_compare.py` lists, plus the moved events' meta on both sides.

    PYTHONIOENCODING=utf-8 python engine_v2/plans/qa_pinbar_floor_inputs/u2_causes.py BASE_DIR CUR_DIR [OUT.txt]

The frames are re-fetched with the engine's own path (cached in %TEMP%/plan_u_u2_causes.pkl) and classified twice
through `apply_candle_classification`: U1 via the `body_floor` seam (the fixed floor on every row, = S_U1's
`pinbar_body_floor`), U2 with the floor seed (= the replay). The H1 U2 types are checked against CUR's
`*_final.csv` first (they must be equal).
"""
from __future__ import annotations

import ast
import os
import pickle
import sys

import _frames  # noqa: F401  (sys.path)
import numpy as np
import pandas as pd

from _frames import HERE
from engine_v2.config import CONFIG
from engine_v2.features.candle_classifier import apply_candle_classification
from engine_v2.multitf.data_bridge import fetch_lower_tf_data
from engine_v2.run_replay import _floor_to_day_start, fetch_floor_seeds, fetch_history_with_auto_extend

B, C = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else str(HERE / "u2_causes.txt")
CACHE = os.path.join(os.environ["TEMP"], "plan_u_u2_causes.pkl")
Q = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"
FIXED = {"H1": 2.2e-4, "M15": 1.0e-4}
COLS = ["candle_type", "pinbar_dir", "normal_dir", "is_big_maru_as0", "is_big_normal_as0", "is_big_maru_as1",
        "is_big_normal_as1", "is_big_maru_as2", "is_big_normal_as2", "big_ratio_as0", "prior_maru_max_len_as0"]

# (TF, lens file suffix, label, first idx, last idx, event keys to print: (sub_id, cycle_id, type))
WINDOWS = [
    ("H1", "", "H1 sid 0 cycle 0: CTS_CONFIRMED 439 (anchor 430) -> 485 (anchor 448); the CTS KL zone, fib, WVMI follow",
     425, 486, [(None, 0, "CTS_CONFIRMED")]),
    ("M15", "_M15_counter", "sub 6 cycle 0: CTS_ESTABLISHED / BOS_CONFIRMED 3656 (double_maru) -> 3657 "
     "(one_maru_continuous)", 3645, 3658, [(6, 0, "CTS_ESTABLISHED")]),
    ("M15", "_M15_confluence", "sub 0 cycle 0: the pattern-path CTS_UPDATEDs 780 / 784 -> 783", 774, 785,
     [(0, 0, "CTS_UPDATED")]),
    ("M15", "_M15_confluence", "sub 0 cycle 3: CTS_CONFIRMED 1748 -> 1751; WVMI (0, 3) FP 1747 -> 1748", 1738, 1752,
     [(0, 3, "CTS_CONFIRMED")]),
    ("M15", "_M15_confluence", "sub 1 cycle 0: CTS_CONFIRMED 1818 -> 1833, the range 1818 -> 1821; WVMI (1, 0) FP "
     "1817 -> 1828", 1810, 1834, [(1, 0, "CTS_CONFIRMED")]),
    ("M15", "_M15_confluence", "sub 1 cycle 1: CTS_RECONFIRMED 1904 -> 1903", 1896, 1905, [(1, 1, "CTS_RECONFIRMED")]),
    ("M15", "_M15_confluence", "sub 1 cycle 4: CTS_UPDATED 2364 one_maru_continuous -> double_maru", 2358, 2365,
     [(1, 4, "CTS_UPDATED")]),
    ("M15", "_M15_counter", "sub 3 cycle 2: the BOS KL zone base 2914 'no base star' -> 2915 'base'; WVMI (3, 1) LP "
     "2914 weight; WVMI (3, 2) FB 2916 -> 2917", 2908, 2920, []),
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


def events(d, sfx):
    ev = pd.read_csv(os.path.join(d, f"{Q}{sfx}_structure_events.csv"))
    rows = []
    for r in ev.itertuples():
        m = ast.literal_eval(r.meta)
        rows.append((m.get("sub_id"), m.get("cycle_id"), r.type, int(r.idx), m))
    return rows


def main():
    raws, seeds = frames()
    cls = {}
    for tf, raw in raws.items():
        u1 = apply_candle_classification(raw.copy(), timeframe=tf, body_floor=np.full(len(raw), FIXED[tf])).df
        u2 = apply_candle_classification(raw.copy(), timeframe=tf, floor_seed=seeds[tf]).df
        cls[tf] = (u1, u2)
    lines = []
    final = pd.read_csv(os.path.join(C, "NZD_USD_H1_2025-11-15_2026-01-20_final.csv"))
    assert (final["candle_type"].to_numpy() == cls["H1"][1]["candle_type"].to_numpy()).all()
    lines.append("H1 U2 types re-derived here == CUR *_final.csv (all 1058)")
    for tf, sfx, label, lo, hi, keys in WINDOWS:
        u1, u2 = cls[tf]
        lines += ["", f"## {tf} {label}", f"   candles {lo}..{hi} whose classification differs (U1 -> U2):"]
        for i in range(lo, hi + 1):
            d = [f"{c} {u1.loc[i, c]} -> {u2.loc[i, c]}" for c in COLS if str(u1.loc[i, c]) != str(u2.loc[i, c])]
            if d:
                lines.append(f"     {i} ({u2.loc[i, 'candle_type_before_floor']}, body "
                             f"{u2.loc[i, 'body_len'] / 1e-4:.1f}, floor {u2.loc[i, 'pinbar_body_floor'] / 1e-4:.2f}): "
                             + "; ".join(d))
        for side, d in (("S_U1", B), ("U2", C)):
            for sub, cyc, typ, idx, m in events(d, sfx):
                if (sub, cyc, typ) in keys and lo <= idx <= hi:
                    keep = {k: m[k] for k in ("cts_anchor_idx", "bos_anchor_idx", "pattern_anchor_idx", "via",
                                              "confirmation_method", "confirmed_at") if k in m}
                    lines.append(f"   {side} {typ} idx={idx} {keep}")
    text = "\n".join(lines)
    print(text)
    open(OUT, "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
