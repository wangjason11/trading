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

    for t in range(n):
        c_t = float(out.iloc[t]["c"])

        for k in range(cfg.min_lookahead, cfg.max_lookahead + 1):
            i = t - k
            if i < 0:
                continue

            # already locked-in
            if int(out.iloc[i]["is_range"]) == 1:
                continue

            lo_i = float(out.iloc[i]["l"])
            hi_i = float(out.iloc[i]["h"])

            if lo_i <= c_t <= hi_i:
                out.iat[i, out.columns.get_loc("is_range")] = 1
                out.iat[i, out.columns.get_loc("is_range_confirm_idx")] = t
                out.iat[i, out.columns.get_loc("is_range_lag")] = k

    return out
