"""Prototype of the user's cross-limit breakout watch on the S_S6 M15 frame (rebuilt from the lens chart).

Reading (option a): from the sub's starting_idx, every candle A whose CLOSE is past P in the sub's direction opens a
watch: BreakoutPatterns.detect_best_for_anchor(A, sd, P) (the reversal watch's detector, the cross limit as the break
threshold); the first A whose pattern confirms (apply <= A + 5) = the break; its pattern anchor = A.
Also listed: every close-past candle with its detector result, and the same with the anchor price behind P.
"""
import base64
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from engine_v2.features.candles_v2 import compute_candle_features  # noqa: E402
from engine_v2.features.candle_params import CandleParams  # noqa: E402
from engine_v2.patterns.structure_patterns import BreakoutPatterns, PatternStatus  # noqa: E402

P = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"
SAVE = sys.argv[1]
keep = sys.argv
sys.argv = [sys.argv[0]]
from engine_v2.debug.chart_census import load_fig  # noqa: E402
sys.argv = keep
traces, _ = load_fig(f"{SAVE}/{P}_M15_confluence.html")


def arr(a):
    if isinstance(a, dict) and "bdata" in a:
        return np.frombuffer(base64.b64decode(a["bdata"]), dtype=np.dtype(a["dtype"]))
    return np.asarray(a)


parts = [pd.DataFrame({"time": arr(t["x"]), "o": arr(t["open"]).astype(float), "h": arr(t["high"]).astype(float),
                       "l": arr(t["low"]).astype(float), "c": arr(t["close"]).astype(float)})
         for t in traces if t.get("type") == "candlestick"]
raw = pd.concat(parts).drop_duplicates("time").sort_values("time").reset_index(drop=True)
raw["volume"] = 1.0
raw.attrs["pair"] = "NZD_USD"
df = compute_candle_features(raw, CandleParams(pinbar_body_pip_floor=1.0))
bp = BreakoutPatterns(df)


def apply_idx(ev):
    return ev.confirmation_idx if ev.status == PatternStatus.CONFIRMED else ev.end_idx


for sub, sd, start, end, thr, label in [(7, 1, 4027, len(df) - 1, 0.57827, "P"), (7, 1, 4027, len(df) - 1, 0.57806, "anchor"),
                                        (4, -1, 3304, 3819, 0.57376, "P"), (4, -1, 3304, 3819, 0.574, "anchor"),
                                        (5, 1, 3621, 4159, 0.58105, "P"), (6, -1, 3760, 4200, 0.57112, "P")]:
    print(f"\n== sub {sub} sd {sd:+d} threshold {thr} ({label}), candles {start}..{end}")
    first = None
    shown = 0
    for A in range(start, end + 1):
        c = float(df.loc[A, "c"])
        if not (c > thr if sd == 1 else c < thr):
            continue
        ev = bp.detect_best_for_anchor(A, sd, thr)
        ok = ev is not None and apply_idx(ev) is not None and apply_idx(ev) <= A + 5
        if shown < 6:
            print(f"   close-past {A} c {c:.5f} {df.loc[A, 'candle_type']} dir {int(df.loc[A, 'direction'])}"
                  f" -> {None if ev is None else (ev.name, ev.status.name, ev.start_idx, ev.end_idx, apply_idx(ev))}")
            shown += 1
        if ok and first is None:
            first = (A, ev.name, apply_idx(ev))
    print(f"   BREAK: {first}")
