from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class RangeLabelConfig:
    min_lookahead: int = 2
    max_lookahead: int = 5


def apply_is_range_labels(df: pd.DataFrame, cfg: RangeLabelConfig = RangeLabelConfig()) -> pd.DataFrame:
    """
    Implements user's definition:

    Candle i becomes is_range if there exists k in [min_lookahead, max_lookahead]
    such that close[i+k] is within [low[i], high[i]].

    Implemented event-style:
      - iterate t (current close index)
      - t can confirm earlier i=t-k candles
      - once is_range[i] == 1, never changes

    Adds:
      - is_range (0/1)
      - is_range_confirm_idx (index that confirmed it)
      - is_range_lag (k that confirmed it)
    """
    out = df.copy()
    n = len(out)

    # initialize if not present
    if "is_range" not in out.columns:
        out["is_range"] = 0
    if "is_range_confirm_idx" not in out.columns:
        out["is_range_confirm_idx"] = -1
    if "is_range_lag" not in out.columns:
        out["is_range_lag"] = -1

    # enforce int types
    out["is_range"] = out["is_range"].astype(int)
    out["is_range_confirm_idx"] = out["is_range_confirm_idx"].astype(int)
    out["is_range_lag"] = out["is_range_lag"].astype(int)

    # Numpy positional views for the hot double loop. Per-cell `out.iloc[i][col]`
    # builds a full-row Series on every access, which triggers pandas
    # `__finalize__` -> `deepcopy(df.attrs)` (the attrs carry the imbalances
    # instance list) -> hundreds of millions of deepcopy calls on real windows.
    # The loop logic below is byte-identical to the prior `.iloc`/`.iat` version
    # (same t-then-k order, same lock-in skip, same inclusive [lo, hi] test);
    # only the read/write substrate changed to numpy arrays. Written back once
    # at the end. See memory `project_ms_optimization_opportunity.md` (same
    # `.iloc`-per-cell antipattern the MS sprint removed).
    c_arr = out["c"].to_numpy(dtype=float)
    l_arr = out["l"].to_numpy(dtype=float)
    h_arr = out["h"].to_numpy(dtype=float)
    is_range = out["is_range"].to_numpy(dtype=int).copy()
    confirm_idx = out["is_range_confirm_idx"].to_numpy(dtype=int).copy()
    lag_arr = out["is_range_lag"].to_numpy(dtype=int).copy()

    for t in range(n):
        c_t = c_arr[t]

        for k in range(cfg.min_lookahead, cfg.max_lookahead + 1):
            i = t - k
            if i < 0:
                continue

            # already locked-in
            if is_range[i] == 1:
                continue

            if l_arr[i] <= c_t <= h_arr[i]:
                is_range[i] = 1
                confirm_idx[i] = t
                lag_arr[i] = k

    out["is_range"] = is_range
    out["is_range_confirm_idx"] = confirm_idx
    out["is_range_lag"] = lag_arr

    return out
