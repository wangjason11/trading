"""PLAN_U U3 doc check: how far does a frame classified WITHOUT its floor seed (`floor_seed=None` — tests, standalone
scripts) differ from the replay's seeded classification, per window mode? Natural: only before the frame's 90th
pinbar (no floor there; after it the windows are identical). Final: also after it, until the two runs' windows
converge (a demotion the unseeded run misses or adds changes the next windows).

The engine's `compute_candle_features` both ways (the `u3_causes.py` cache: the replay's frames + seeds), per TF
and mode: the frame candles with no floor (unseeded), the candles whose floor / type differ, the last such idx.

    PYTHONIOENCODING=utf-8 python engine_v2/plans/qa_pinbar_floor_inputs/u3_frame_only.py [OUT]
"""
from __future__ import annotations

import sys

import _frames  # noqa: F401  (sys.path)
import numpy as np

from _frames import HERE
from u3_causes import classify, frames

OUT = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "u3_frame_only.txt")


def main():
    raws, seeds = frames()
    lines = ["the frame classified with its seed (= the replay) vs without (floor_seed=None)", ""]
    for tf in ("H1", "M15"):
        for mode in ("natural", "final"):
            a = classify(raws[tf], seeds[tf], mode)
            b = classify(raws[tf], None, mode)
            fa, fb = a["pinbar_body_floor"].to_numpy(float), b["pinbar_body_floor"].to_numpy(float)
            nan_b = np.flatnonzero(np.isnan(fb))
            both = ~np.isnan(fb)
            df = np.flatnonzero(both & (fa != fb))
            dt = np.flatnonzero(a["candle_type"].to_numpy() != b["candle_type"].to_numpy())
            dt_after = dt[dt > (nan_b[-1] if len(nan_b) else -1)]
            lines.append(
                f"{tf} {mode:<7}: unseeded has no floor on idx 0 .. {nan_b[-1] if len(nan_b) else '-'} "
                f"({len(nan_b)} candles); where both have one, floors differ on {len(df)}"
                + (f" (idx {df[0]} .. {df[-1]})" if len(df) else "")
                + f"; types differ on {len(dt)} in all, {len(dt_after)} after the no-floor stretch"
                + (f" (last idx {dt_after[-1]})" if len(dt_after) else ""))
        lines.append("")
    text = "\n".join(lines)
    print(text)
    open(OUT, "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
