"""PLAN_U — count the M15 floor pinbars from the two lens charts' candle hovers (the M15 frame is not saved as a CSV).

Run from the repo root after a replay:
    PYTHONIOENCODING=utf-8 python engine_v2/plans/qa_pinbar_floor_inputs/m15_hover_floor_count.py [CHART_DIR]

Reads the floor line (customdata slot 14 since U1, `_candle_floor_hover`) once per M15 candle idx on each lens and prints
the candle count, the floor values and the floor pinbars by their body-% type. Measured: U1 4228 candles at 0.000100,
142 floor pinbars (34 maru / 108 normal) on both lenses. Expected: U2 436 (PLAN_U §8 A4), U3 410 — U2 / U3 give each
candle its own floor, so the floor-value counter then lists many values (read the floor-pinbar total).
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # the repo root

chart_dir = sys.argv[1] if len(sys.argv) > 1 else "artifacts/charts"
sys.argv = [sys.argv[0]]
from engine_v2.debug.chart_census import load_fig  # noqa: E402

STEM = chart_dir.rstrip("/\\") + "/NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"
for lens in ("_M15_counter", "_M15_confluence"):
    data, _ = load_fig(STEM + lens + ".html")
    seen, floors, before = set(), Counter(), Counter()
    for tr in data:
        if tr.get("type") != "candlestick":
            continue
        for cd in tr["customdata"]:
            i = int(cd[0])
            if i in seen:
                continue
            seen.add(i)
            line = cd[14]
            floors[line.split(" | ")[0]] += 1
            if "candle_type_before_floor=" in line:
                before[line.split("=")[-1]] += 1
    print(lens, "candles", len(seen), "floor values", len(floors), dict(floors.most_common(5)),
          "floor pinbars", sum(before.values()), dict(before))
