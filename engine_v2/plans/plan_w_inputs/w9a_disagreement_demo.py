"""PLAN_W 9a — the two label-based fill paths on a frame with a missing label (synthetic: the engine never builds one —
`w9a_index_census.txt`). The test suite's bullish fixture (`tests/test_imbalance.py::_SAME_CANDLE_BULL` + one
candle), labelled 100.., with ONE label dropped; the instance is the one found on the full frame.

    PYTHONPATH=. python engine_v2/plans/plan_w_inputs/w9a_disagreement_demo.py OUT.txt   (from the repo root)
"""
import sys

import pandas as pd

from engine_v2.common.types import IMBALANCE_FILL_THRESHOLD, ImbalanceInstance
from engine_v2.patterns.imbalance import compute_imbalance
from engine_v2.zones.poi_zones import _compute_fill_idx_cache

ROWS = [  # (o, h, l, c, note)
    (0.95, 1.00, 0.90, 0.95, "c1 (gap_bottom 1.00)"),
    (1.02, 1.08, 1.01, 1.07, "c2 bullish imbalance"),
    (1.10, 1.15, 1.10, 1.14, "c3 (gap_top 1.10); stroke 1 = low <= 1.10 - 0.10 x 0.65 = 1.035"),
    (1.12, 1.13, 1.08, 1.12, "no arm (low 1.08)"),
    (1.05, 1.13, 1.02, 1.09, "low 1.02 ARMS (close 1.09 inside)"),
    (1.09, 1.14, 1.08, 1.12, "close 1.12 >= 1.10 -> stroke 2: FILLED"),
]
OFFSET = 100


def frame():
    df = pd.DataFrame([r[:4] for r in ROWS], columns=["o", "h", "l", "c"])
    df["direction"] = [1 if c > o else (-1 if c < o else 0) for o, _, _, c in df[["o", "h", "l", "c"]].values]
    out = compute_imbalance(df)
    inst, = out.attrs["imbalances"]
    out.index = pd.Index([i + OFFSET for i in range(len(out))], dtype="int64")
    return out, ImbalanceInstance(start_idx=inst.start_idx + OFFSET, end_idx=inst.end_idx + OFFSET,
                                  direction=inst.direction, gap_top=inst.gap_top, gap_bottom=inst.gap_bottom,
                                  gap_size=inst.gap_size)


lines = ["The frame (labels 100..105; candle 104 arms, 105 fills):",
         "  label     o     h     l     c   note"]
for i, (o, h, l, c, note) in enumerate(ROWS):
    lines.append(f"  {i + OFFSET:>5}  {o:.2f}  {h:.2f}  {l:.2f}  {c:.2f}   {note}")
lines.append("")
lines.append("case                          is_filled(t=105)   cache (armed, filled)   agree?")
for name, drop in [("no label missing", None), ("label 102 missing (end_idx+1)", 102),
                   ("label 103 missing (later)", 103)]:
    df, inst = frame()
    if drop is not None:
        df = df.drop(index=drop)
    filled = inst.is_filled(df, check_to_idx=105, fill_threshold=IMBALANCE_FILL_THRESHOLD)
    armed_idx, fill_idx = _compute_fill_idx_cache(df, [inst], IMBALANCE_FILL_THRESHOLD)[id(inst)]
    agree = filled == (fill_idx is not None and fill_idx <= 105)
    lines.append(f"{name:<30}{str(filled):<19}{str((armed_idx, fill_idx)):<24}{'yes' if agree else 'NO'}")
text = "\n".join(lines) + "\n"
print(text)
with open(sys.argv[1], "w", encoding="utf-8") as f:
    f.write(text)
