"""PLAN_U U3, for the user's chart look: every candle whose type differs between U2 (natural window) and U3 (final
window) — idx, time, body-% type, body, the floor on each side, the type on each side, the pin direction it has as a
pinbar. H1 from the two replays' own `*_final.csv` (BASE = S_U2, CUR = the U3 variant's artifacts/debug); M15 (no
per-candle CSV) from the engine's classification of both modes (`compute_candle_features` + the floor seed — the
`u3_causes.py` cache), checked against the M15 count on the lens hovers (`m15_hover_floor_count.py`: U3 410).

    PYTHONIOENCODING=utf-8 python engine_v2/plans/qa_pinbar_floor_inputs/u3_changed_candles.py BASE_DIR CUR_DIR [OUT]
"""
from __future__ import annotations

import os
import sys

import _frames  # noqa: F401  (sys.path)
import numpy as np
import pandas as pd

from _frames import HERE
from engine_v2.features.candle_params import CandleParams
from engine_v2.features.candles_v2 import classify_candles
from u3_causes import classify, frames

B, C = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else str(HERE / "u3_changed_candles.txt")
P = "NZD_USD_H1_2025-11-15_2026-01-20"
PIP = 1e-4


def table(tf, u2, u3, lines):
    t2, t3 = u2["candle_type"].to_numpy(), u3["candle_type"].to_numpy()
    before = u3["candle_type_before_floor"].to_numpy()
    # the pin direction as a pinbar: the engine's wick rule with every maru / normal made a floor pinbar
    pdir = classify_candles(u3, CandleParams(), body_floor=np.full(len(u3), np.inf))["pinbar_dir"].to_numpy()
    ch = np.flatnonzero(t2 != t3)
    become = int((t3[ch] == "pinbar").sum())
    f2 = u2["pinbar_body_floor"].to_numpy(float) / PIP
    f3 = u3["pinbar_body_floor"].to_numpy(float) / PIP
    d = np.round(f3 - f2, 6)
    small = np.abs(d[ch]) <= 0.010001
    lines += [f"## {tf}: the floor (pips) differs on {int((d != 0).sum())} of {len(d)} candles; U3 - U2 mean "
              f"{d.mean():+.3f}, min {d.min():+.2f}, max {d.max():+.2f}; U3 lower on {int((d < 0).sum())}, higher on "
              f"{int((d > 0).sum())}; median floor U2 {np.median(f2):.2f} U3 {np.median(f3):.2f}",
              f"   of the type changes: {int(small.sum())} with a floor move of at most 0.01 pip (the body on the "
              f"0.01-pip grid, at the floor on one side), {int((~small).sum())} with a larger one"]
    lines += [f"## {tf}: {len(ch)} candles change type U2 -> U3 ({become} become pinbars, {len(ch) - become} stop); "
              f"from maru {int((before[ch] == 'maru').sum())}, normal {int((before[ch] == 'normal').sum())}; with a "
              f"pin direction {int((pdir[ch] != 0).sum())}",
              f"   types maru/normal/pinbar U2 {'/'.join(str(int((t2 == k).sum())) for k in ('maru', 'normal', 'pinbar'))}"
              f" -> U3 {'/'.join(str(int((t3 == k).sum())) for k in ('maru', 'normal', 'pinbar'))}",
              "     idx  time (UTC)           body-%  body  floor U2  floor U3  U2      U3      pin_dir"]
    for i in ch:
        lines.append(f"   {i:5d}  {str(u3['time'].iloc[i])[:16]}  {before[i]:<6}  {u3['body_len'].iloc[i] / PIP:4.1f}"
                     f"  {float(u2['pinbar_body_floor'].iloc[i]) / PIP:8.2f}  {float(u3['pinbar_body_floor'].iloc[i]) / PIP:8.2f}"
                     f"  {t2[i]:<6}  {t3[i]:<6}  {int(pdir[i]):+d}")
    lines.append("")


def main():
    lines = []
    num = ["body_len", "pinbar_body_floor"]
    fb = pd.read_csv(os.path.join(B, f"{P}_final.csv"))
    fc = pd.read_csv(os.path.join(C, f"{P}_final.csv"))
    for d in (fb, fc):
        d[num] = d[num].astype(float)
    raws, seeds = frames()
    h1u3 = classify(raws["H1"], seeds["H1"], "final")
    assert (h1u3["candle_type"].to_numpy() == fc["candle_type"].to_numpy()).all()
    # H1: the types, times, bodies and floors as the two replays wrote them; the wick columns from the engine frame
    fc_full = h1u3.copy()
    for c in ("time", "candle_type", "candle_type_before_floor", "body_len", "pinbar_body_floor"):
        fc_full[c] = fc[c].to_numpy()
    table("H1 (the replays' *_final.csv)", fb, fc_full, lines)
    m2 = classify(raws["M15"], seeds["M15"], "natural")
    m3 = classify(raws["M15"], seeds["M15"], "final")
    assert int(m3["is_floor_pinbar"].sum()) == 410 and int(m2["is_floor_pinbar"].sum()) == 436
    table("M15 (the engine's classification; is_floor_pinbar U2 436 / U3 410 = the lens hovers)", m2, m3, lines)
    text = "\n".join(lines)
    print(text)
    open(OUT, "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
