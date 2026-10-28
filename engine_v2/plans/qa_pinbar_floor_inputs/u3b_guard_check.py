"""PLAN_U U3b (§6b), before the replay: the BUILT seed guard (`run_replay.fetch_floor_seeds` — the count AND
`floor_seed_converged`) on this window vs the §6b prediction (H1 3 steps, 840 candles / 355 body-% pinbars; M15 3 steps,
816 / 310), each step's count and convergence, and the frame floors with the new seed vs the 2-window seed (S_U3's) —
equal on every candle expected. H1 also vs S_U3's `*_final.csv` `pinbar_body_floor` column.

    PYTHONIOENCODING=utf-8 python engine_v2/plans/qa_pinbar_floor_inputs/u3b_guard_check.py [OUT]
"""
from __future__ import annotations

import contextlib
import io

import _frames  # noqa: F401  (sys.path)
import numpy as np
import pandas as pd

from _frames import REPO, fetch_frames, out_path
from engine_v2.config import CONFIG
from engine_v2.features.candle_params import CandleParams
from engine_v2.features.pinbar_body_floor import _body_pct_types, floor_seed_converged, pinbar_body_floor_for_frame
from engine_v2.run_replay import _get_history_retrying, _timeframe_to_timedelta, fetch_floor_seeds

OUT = out_path(__file__)
S_U3 = REPO / "artifacts/commits/week9-entries-decisions/20261028_123113_32c544d"
PRED = {"H1": (3, 840, 355), "M15": (3, 816, 310)}


def span_seed(tf, first, k):
    """The guard's step-k span, fetched like `fetch_floor_seed` (no stop rule)."""
    start = (first - k * 400 * _timeframe_to_timedelta(tf)).to_pydatetime()
    raw = _get_history_retrying(pair=CONFIG.pair, timeframe=tf, start=start, end=first.to_pydatetime(),
                                prefix="[floor_seed]", allow_empty=True)
    raw = raw[pd.to_datetime(raw["time"], utc=True) < first].reset_index(drop=True)
    raw.attrs["pair"] = CONFIG.pair
    return raw


def main():
    p = CandleParams()
    raws = fetch_frames(("H1", "M15"))
    first = pd.to_datetime(raws["H1"]["time"].iloc[0], utc=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        seeds = fetch_floor_seeds(CONFIG.pair, CONFIG.timeframe, first, ("M15",))
    log = [l for l in buf.getvalue().splitlines() if l.startswith("[floor_seed]")]
    lines = [f"window {p.pinbar_floor_window}, p{p.pinbar_floor_percentile:g}, source {p.pinbar_floor_window_source}, "
             f"seed windows {p.pinbar_floor_seed_windows}; frame first candle {first}", "",
             "## The guard's lines (fetch_floor_seeds, the built guard)"] + [f"   {l}" for l in log] + [""]
    ok_all = True
    for tf in ("H1", "M15"):
        lines.append(f"## {tf}: each step's span (400 candles of TF time per step)")
        for k in (1, 2, 3, 4):
            sd = _body_pct_types(span_seed(tf, first, k), p)
            n_pins = int((sd["candle_type"] == "pinbar").sum())
            conv = floor_seed_converged(sd["body_len"], sd["candle_type"], p)
            nat = floor_seed_converged(sd["body_len"], sd["candle_type"],
                                       CandleParams(pinbar_floor_window_source="natural"))
            lines.append(f"   step {k}: {len(sd)} candles from {str(sd['time'].iloc[0])[:16]}, {n_pins} body-% pinbars "
                         f"(>= 180: {n_pins >= 180}); converged final: {conv}, natural: {nat}")
        seed = seeds[tf]
        n_pins = int((_body_pct_types(seed, p)["candle_type"] == "pinbar").sum())
        k_got = next(int(l.split(", ")[1].split()[0]) for l in log if l.startswith(f"[floor_seed] {tf}:"))
        got = (k_got, len(seed), n_pins)
        ok = got == PRED[tf]
        ok_all &= ok
        lines.append(f"   the guard: {got[0]} steps, {got[1]} candles {str(seed['time'].iloc[0])[:16]} -> "
                     f"{str(seed['time'].iloc[-1])[:16]}, {got[2]} body-% pinbars; predicted {PRED[tf]} -> "
                     f"{'MATCH' if ok else 'DIFFER'}")
        new_f, new_d = pinbar_body_floor_for_frame(raws[tf], seed, p)
        old_f, old_d = pinbar_body_floor_for_frame(raws[tf], span_seed(tf, first, 2), p)
        same = bool(np.array_equal(new_f, old_f, equal_nan=True) and np.array_equal(new_d, old_d))
        ok_all &= same
        lines.append(f"   frame ({len(raws[tf])} candles): floors + demotions with the new seed == the 2-window seed's "
                     f"(S_U3): {same}")
        if tf == "H1":
            csv = next(S_U3.glob("*_H1_*_final.csv"))
            # round_trip: the default C parser can be 1 ulp off the written repr (GOTCHAS, the floor-seed entry's rule "a pandas CSV round-trip")
            saved = pd.read_csv(csv, usecols=["pinbar_body_floor", "is_floor_pinbar"], float_precision="round_trip")
            eq = bool(np.array_equal(saved["pinbar_body_floor"].to_numpy(float), new_f, equal_nan=True)
                      and np.array_equal(saved["is_floor_pinbar"].to_numpy(bool), new_d))
            ok_all &= eq
            lines.append(f"   H1 frame floors + is_floor_pinbar == S_U3's {csv.name}: {eq}")
        lines.append("")
    lines.append(f"ALL AS PREDICTED: {ok_all}")
    text = "\n".join(lines)
    print(text)
    open(OUT, "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
