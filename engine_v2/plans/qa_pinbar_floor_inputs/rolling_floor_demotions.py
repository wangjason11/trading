"""What a ROLLING floor would demote (2026-10-27, the user's proposed rule: the floor of a candle = the p90 of the
bodies of the 90 most recent natural pinbars strictly before it; p80 / p85 shown for scale).

Shadow only (the engine is not called with it): natural classification (the floor off); a natural maru / normal
whose body is STRICTLY below its candle's rolling floor would become a pinbar (today's `<` rule). Bodies are rounded
to the exact 0.1 pip. Compared with today's fixed floor over the SAME candles (those with a full window); the
candles without one (the warm-up at the frame's start) are counted separately. `pinbar_dir` of a demoted candle is
re-derived with `classify_candles`' wick rule (`pinbar_distance`)."""
from __future__ import annotations

import numpy as np

import _frames as F

from engine_v2.features.candle_classifier import candle_params_for_timeframe
from engine_v2.features.candle_params import CandleParams

WINDOW = 90
SERIES = (80, 85, 90)


def main():
    raws = F.fetch_frames()
    dist_frac = CandleParams().pinbar_distance
    with open(F.out_path(__file__), "w", encoding="utf-8") as fh:
        for tf, raw in raws.items():
            nat = F.classify(raw, CandleParams())
            n = len(nat)
            body = np.round(nat["body_len"].to_numpy(dtype=float) / F.PIP, 1)
            ctype = nat["candle_type"].to_numpy()
            pos = np.flatnonzero(ctype == "pinbar")
            k = np.searchsorted(pos, np.arange(n), side="left")      # natural pinbars strictly before each candle
            full = k >= WINDOW
            floors = {p: np.full(n, np.nan) for p in SERIES}
            for i in np.flatnonzero(full):
                vals = np.percentile(body[pos[k[i] - WINDOW:k[i]]], SERIES)
                for p, v in zip(SERIES, vals):
                    floors[p][i] = round(float(v), 6)
            span = full
            mar, nor = (ctype == "maru") & span, (ctype == "normal") & span
            fixed = candle_params_for_timeframe(tf).pinbar_body_pip_floor
            length = nat["candle_len"].to_numpy(dtype=float)
            up, lo = nat["upper_wick"].to_numpy(dtype=float), nat["lower_wick"].to_numpy(dtype=float)
            d = dist_frac * length
            has_dir = ((up < d) & (lo > d)) | ((lo < d) & (up > d))
            fh.write(f"## {tf}: candles {n}; with a full window {int(span.sum())} (from idx "
                     f"{int(np.flatnonzero(span)[0])}); warm-up without one {int((~span).sum())}\n")
            fh.write(f"   natural in the span: maru {int(mar.sum())}, normal {int(nor.sum())}\n")
            rows = [("fixed " + str(fixed), np.where(span, fixed, np.nan))] + [(f"rolling p{p}", floors[p]) for p in SERIES]
            for name, fl in rows:
                dem = (mar | nor) & (body < np.nan_to_num(fl, nan=-1.0))
                dm, dn = int((dem & mar).sum()), int((dem & nor).sum())
                fh.write(f"   {name:13s}: demoted {int(dem.sum()):5d} ({dem.sum() / span.sum() * 100:4.1f}% of the span)"
                         f"  maru {dm:5d} ({dm / mar.sum() * 100:4.1f}%)  normal {dn:5d} ({dn / nor.sum() * 100:4.1f}%)"
                         f"  with a pin direction {int((dem & has_dir).sum())}\n")
            fh.write("\n")


if __name__ == "__main__":
    main()
