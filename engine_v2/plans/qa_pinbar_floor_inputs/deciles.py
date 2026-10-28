"""Body / range deciles (pips) per TF and NATURAL candle type (the floor off), with the candle count of
each row, the type shares, and the type counts with today's floor. The tables in
`features/CANDLE_BODY_FLOOR_NOTES.md` "Measured distribution" come from this script (2026-10-27)."""
from __future__ import annotations

import _frames as F

from engine_v2.features.candle_classifier import candle_params_for_timeframe
from engine_v2.features.candle_params import CandleParams

QS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TYPES = ("maru", "normal", "pinbar")


def main():
    raws = F.fetch_frames()
    with open(F.out_path(__file__), "w", encoding="utf-8") as fh:
        for tf, raw in raws.items():
            nat = F.classify(raw, CandleParams())
            cur = F.classify(raw, candle_params_for_timeframe(tf))
            body, rng = nat["body_len"] / F.PIP, nat["candle_len"] / F.PIP
            n = len(nat)
            fh.write(f"## {tf} candles={n} median body={body.median():.2f}p "
                     f"floor now={candle_params_for_timeframe(tf).pinbar_body_pip_floor}\n")
            fh.write("  natural types: " + "  ".join(
                f"{t} {int((nat.candle_type == t).sum())} ({(nat.candle_type == t).mean() * 100:.1f}%)" for t in TYPES)
                + "\n  with the floor: " + "  ".join(f"{t} {int((cur.candle_type == t).sum())}" for t in TYPES) + "\n")
            hdr = " ".join(f"{'p' + str(int(q * 100)):>5}" for q in QS)
            fh.write(f"  {'metric':6} {'group':7} {'candles':>7} {hdr}\n")
            for metric, s in (("body", body), ("range", rng)):
                for g in ("all",) + TYPES:
                    x = s if g == "all" else s[nat["candle_type"] == g]
                    fh.write(f"  {metric:6} {g:7} {len(x):7d} " + " ".join(f"{x.quantile(q):5.1f}" for q in QS) + "\n")
            fh.write("  body deciles / median body: " + " ".join(
                f"{body.quantile(q) / body.median():4.2f}" for q in QS) + "\n\n")


if __name__ == "__main__":
    main()
