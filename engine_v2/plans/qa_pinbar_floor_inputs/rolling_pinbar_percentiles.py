"""The rolling pinbar body percentiles (the user's request, 2026-10-27; p85 added the same day).

For every candle i of a TF: take the 90 most recent pinbars STRICTLY BEFORE i (natural classification —
the floor off, so the window is not shaped by the floor it would calibrate), and log the 80th, 85th and
90th percentile of their BODY lengths (pips; numpy's default linear interpolation). The series starts at
the first candle with at least 90 prior pinbars. Then print the deciles of each series (one value per
candle, so a window is weighted by the candles it stays current for)."""
from __future__ import annotations

import numpy as np

import _frames as F

from engine_v2.features.candle_classifier import candle_params_for_timeframe
from engine_v2.features.candle_params import CandleParams

WINDOW = 90
SERIES = (80, 85, 90)
QS = [10, 20, 30, 40, 50, 60, 70, 80, 90]


def rolling(nat):
    # Prices are quoted to 0.1 pip, so every body IS a multiple of 0.1 pip: rounding removes only the float
    # noise of body_len / PIP (2.2 can come out 2.1999...), which would otherwise miscount "below the floor".
    body = np.round(nat["body_len"].to_numpy(dtype=float) / F.PIP, 1)
    pos = np.flatnonzero(nat["candle_type"].to_numpy() == "pinbar")
    pb = body[pos]
    first, rows = None, []
    for i in range(len(nat)):
        k = int(np.searchsorted(pos, i, side="left"))   # pinbars strictly before i
        if k < WINDOW:
            continue
        if first is None:
            first = i
        rows.append(np.percentile(pb[k - WINDOW:k], SERIES))
    rows = np.array(rows)
    return first, {p: rows[:, j] for j, p in enumerate(SERIES)}


def main():
    raws = F.fetch_frames()
    with open(F.out_path(__file__), "w", encoding="utf-8") as fh:
        fh.write(f"window = the {WINDOW} most recent natural pinbars strictly before each candle; body pips\n\n")
        for tf, raw in raws.items():
            nat = F.classify(raw, CandleParams())
            first, series = rolling(nat)
            floor = candle_params_for_timeframe(tf).pinbar_body_pip_floor
            fh.write(f"## {tf} candles={len(nat)} pinbars={int((nat.candle_type == 'pinbar').sum())} "
                     f"series from idx {first} ({nat.loc[first, 'time']}) = {len(series[SERIES[0]])} candles; "
                     f"floor now={floor}\n")
            hdr = " ".join(f"{'p' + str(q):>5}" for q in QS)
            fh.write(f"  {'series':6} {'min':>5} {hdr} {'max':>5} {'mean':>5}\n")
            for p, s in series.items():
                fh.write(f"  {'p' + str(p):6} {s.min():5.2f} " + " ".join(f"{np.percentile(s, q):5.2f}" for q in QS)
                         + f" {s.max():5.2f} {s.mean():5.2f}\n")
                below, at = np.round(s, 6) < floor, np.round(s, 6) == floor
                fh.write(f"         candles where it is below today's floor: {int(below.sum())} "
                         f"({below.mean() * 100:.1f}%), exactly at it: {int(at.sum())} ({at.mean() * 100:.1f}%)\n")
            fh.write("\n")


if __name__ == "__main__":
    main()
