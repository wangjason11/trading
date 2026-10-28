"""PLAN_W 9a census — which index does every fill check see on a real run?

Runs the pipeline on the CONFIG window (with its floor seeds, as the replay does; no exports) with the two
label-based fill implementations wrapped by a counter:
  - `ImbalanceInstance.is_filled` (common/types.py): the fast path needs a 0..n-1 RangeIndex
    (`step == 1, start == 0, stop == n`); anything else runs the label-based fallback loop;
  - `poi_zones._compute_fill_idx_cache`: its identity path needs an int index with `[0] == 0` and `[-1] == n - 1`;
    anything else maps labels through a dict and gives `(None, None)` when the label `end_idx + 1` is missing.
For every call it records the frame's index kind and, off the fast / identity path, whether a label the scan
would cross is missing (the case the two disagree on: the fallback skips it, the cache stops).

    PYTHONPATH=. python engine_v2/plans/plan_w_inputs/w9a_index_census.py OUT.txt   (from the repo root)
"""
import collections
import sys

import engine_v2.run_replay as rr
from engine_v2.common.types import ImbalanceInstance
from engine_v2.zones import poi_zones

out_path = sys.argv[1]
calls = collections.Counter()
examples = {}


def _kind(df):
    idx = df.index
    n = len(df)
    if getattr(idx, "step", None) == 1 and getattr(idx, "start", None) == 0 and getattr(idx, "stop", None) == n:
        return "RangeIndex(0, n)"
    return f"{type(idx).__name__} dtype={idx.dtype} first={idx[0] if n else None} last={idx[-1] if n else None} n={n}"


_orig_is_filled = ImbalanceInstance.is_filled
_orig_cache = poi_zones._compute_fill_idx_cache


def is_filled(self, df, check_to_idx, fill_threshold=ImbalanceInstance.is_filled.__defaults__[0]):
    kind = _kind(df)
    key = ("is_filled", kind)
    if kind != "RangeIndex(0, n)":
        missing = [i for i in range(self.end_idx + 1, check_to_idx + 1) if i not in df.index]
        key = key + (f"missing labels in scan: {len(missing)}",)
        examples.setdefault(key, (self.start_idx, self.end_idx, check_to_idx, missing[:5]))
    calls[key] += 1
    return _orig_is_filled(self, df, check_to_idx, fill_threshold)


def cache(df, imbalances, fill_threshold):
    kind = _kind(df)
    arr = df.index.to_numpy()
    identity = arr.dtype.kind in "iu" and len(df) > 0 and arr[0] == 0 and arr[-1] == len(df) - 1
    key = ("_compute_fill_idx_cache", kind, "identity path" if identity else "dict path")
    if not identity:
        labels = set(int(v) for v in arr)
        miss = sum(1 for inst in imbalances if inst.end_idx + 1 not in labels)
        key = key + (f"instances whose end_idx + 1 is missing: {miss}",)
    calls[key] += 1
    calls[key[:3] + ("instances",)] += len(imbalances)
    return _orig_cache(df, imbalances, fill_threshold)


ImbalanceInstance.is_filled = is_filled
poi_zones._compute_fill_idx_cache = cache

df, effective_start, _ = rr.fetch_history_with_auto_extend(
    pair=rr.CONFIG.pair, timeframe=rr.CONFIG.timeframe, start=rr._floor_to_day_start(rr.CONFIG.start),
    end=rr.CONFIG.end, lookback_days=183, min_history=50, extend_days=4, max_extend_iters=8,
)
df.attrs["pair"] = rr.CONFIG.pair
seeds = rr.fetch_floor_seeds(rr.CONFIG.pair, rr.CONFIG.timeframe, df["time"].iloc[0], rr.CONFIG.lower_timeframes)
res = rr.run_pipeline(df, lower_timeframes=rr.CONFIG.lower_timeframes, floor_seeds=seeds)

lines = [f"window {rr.CONFIG.pair} {rr.CONFIG.timeframe} {rr.CONFIG.start} -> {rr.CONFIG.end}; "
         f"H1 frame {len(df)} candles", ""]
for key, n in sorted(calls.items(), key=lambda kv: (kv[0][0], -kv[1])):
    lines.append(f"{n:>9}  " + " | ".join(key))
    if key in examples:
        lines.append(f"           example (start_idx, end_idx, check_to_idx, first missing): {examples[key]}")
text = "\n".join(lines) + "\n"
print(text)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(text)
