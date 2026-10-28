"""PLAN_U U3, for the user's question "final is with demoted included?": the two windows at one H1 candle, side by
side — which 90 pinbars fill each, how many of the final window's are floor-demoted, the time each reaches back to,
and the p90 (= the candle's floor). Default candle: H1 446 (2025-12-11 12:00, the largest floor drop U2 -> U3).

Rebuilds the windows with the engine's own pieces: `compute_candle_features` (both modes, the floor seed — the
`u3_causes.py` cache) for the per-candle floors, and the seed + frame body-% types in time order for the windows,
replaying `rolling_pinbar_body_floor`'s membership rule (natural: body-% pinbars; final: + the rows it demoted).
Asserted: each window's p90 == that mode's `pinbar_body_floor` at the candle.

    PYTHONIOENCODING=utf-8 python engine_v2/plans/qa_pinbar_floor_inputs/u3_window_example.py [IDX] [OUT]
"""
from __future__ import annotations

import sys

import _frames  # noqa: F401  (sys.path)
import numpy as np
import pandas as pd

from _frames import HERE
from engine_v2.features.candle_params import CandleParams
from engine_v2.features.pinbar_body_floor import _body_pct_types, rolling_pinbar_body_floor
from u3_causes import classify, frames

IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 446
OUT = sys.argv[2] if len(sys.argv) > 2 else str(HERE / "u3_window_example.txt")
PIP = 1e-4


def main():
    raws, seeds = frames()
    raw, seed = raws["H1"], seeds["H1"]
    lines = []
    sd = _body_pct_types(seed, CandleParams())
    fr = _body_pct_types(raw, CandleParams())
    body = np.concatenate([sd["body_len"].to_numpy(float), fr["body_len"].to_numpy(float)])
    btype = np.concatenate([sd["candle_type"].to_numpy(), fr["candle_type"].to_numpy()])
    times = pd.concat([sd["time"], fr["time"]]).reset_index(drop=True)
    s = len(sd)
    k = s + IDX
    u = {m: classify(raw, seed, m) for m in ("natural", "final")}
    lines.append(f"H1 candle {IDX} ({str(raw['time'].iloc[IDX])[:16]} UTC): body-% type "
                 f"{u['final'].loc[IDX, 'candle_type_before_floor']}, body {body[k] / PIP:.1f} pips")
    for mode in ("natural", "final"):
        p = CandleParams(pinbar_floor_window_source=mode)
        floor, demoted = rolling_pinbar_body_floor(body, btype, p)
        member = (btype == "pinbar") | (demoted if mode == "final" else False)
        rows = np.flatnonzero(member[:k])[-p.pinbar_floor_window:]
        w = body[rows]
        f = float(np.percentile(w, p.pinbar_floor_percentile))
        assert abs(f - float(u[mode].loc[IDX, "pinbar_body_floor"])) < 1e-15, (mode, f)
        n_dem = int(demoted[rows].sum()) if mode == "final" else 0
        top = np.sort(w)[::-1][:12] / PIP
        lines += [
            "",
            f"## {mode}: the 90 most recent {'body-% pinbars' if mode == 'natural' else 'pinbars (body-% + floor-demoted)'}"
            f" before {IDX}",
            f"   reaches back to {str(times[rows[0]])[:16]} ({(times[k] - times[rows[0]]).total_seconds() / 3600:.0f} h); "
            f"{n_dem} of the 90 are floor-demoted"
            + (f" (their bodies {w[demoted[rows]].min() / PIP:.1f} – {w[demoted[rows]].max() / PIP:.1f} pips)" if n_dem else ""),
            f"   bodies: median {np.median(w) / PIP:.2f}, the 12 largest {', '.join(f'{x:.1f}' for x in top)} pips",
            f"   p90 (linear, the 81st / 82nd of 90 sorted) = {f / PIP:.2f} pips = the floor of {IDX} -> "
            f"{u[mode].loc[IDX, 'candle_type']}",
        ]
    text = "\n".join(lines)
    print(text)
    open(OUT, "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
