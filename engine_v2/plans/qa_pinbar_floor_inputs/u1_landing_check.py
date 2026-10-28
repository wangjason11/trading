"""PLAN_U U1's landing check (§4.2): the replay's 30 CSVs vs a save — every CSV byte-equal except the H1
`*_final.csv`, which = the save's + the 3 body-floor columns (their fields removed → the save's rows, field for
field, as text); the new columns' values.

    python engine_v2/plans/qa_pinbar_floor_inputs/u1_landing_check.py SAVE_DIR [CUR_DIR=artifacts/debug]
"""
from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import pandas as pd

NEW = ["candle_type_before_floor", "pinbar_body_floor", "is_floor_pinbar"]


def _rows(b: bytes):
    return list(csv.reader(io.StringIO(b.decode("utf-8"))))


def main():
    save = Path(sys.argv[1])
    cur = Path(sys.argv[2] if len(sys.argv) > 2 else "artifacts/debug")
    names = sorted(p.name for p in save.glob("*.csv"))
    print(f"{len(names)} CSVs in the save")
    bad = 0
    for n in names:
        if not (cur / n).exists():
            print(f"  MISSING  {n}")
            bad += 1
            continue
        a, b = (save / n).read_bytes(), (cur / n).read_bytes()
        if not n.endswith("_final.csv"):
            same = a == b
            print(f"  {'EQUAL ' if same else 'DIFFER'}  {n}")
            bad += not same
            continue
        rows_a, rows_b = _rows(a), _rows(b)
        missing = [c for c in NEW if c not in rows_b[0]]
        assert not missing, missing
        keep = [i for i, c in enumerate(rows_b[0]) if c not in NEW]
        stripped = [[r[i] for i in keep] for r in rows_b]
        same = stripped == rows_a
        print(f"  {'EQUAL ' if same else 'DIFFER'}  {n} (the 3 new columns' fields removed; {len(rows_a) - 1} rows)")
        if not same:
            diff = [k for k, (x, y) in enumerate(zip(rows_a, stripped)) if x != y]
            print(f"     header equal {rows_a[0] == stripped[0]}; rows {len(rows_a)} vs {len(stripped)}; "
                  f"first differing rows {diff[:3]}")
        bad += not same
        df = pd.read_csv(io.BytesIO(b))
        fl = df["pinbar_body_floor"]
        print(f"     pinbar_body_floor: unique {sorted(fl.dropna().unique().tolist())}, NaN {int(fl.isna().sum())}")
        print(f"     is_floor_pinbar: {int(df['is_floor_pinbar'].sum())} "
              f"(from {df.loc[df['is_floor_pinbar'], 'candle_type_before_floor'].value_counts().to_dict()})")
        print(f"     candle_type_before_floor != candle_type on "
              f"{int((df['candle_type_before_floor'] != df['candle_type']).sum())}")
    print("ALL AS PREDICTED" if not bad else f"{bad} NOT AS PREDICTED")


if __name__ == "__main__":
    main()
