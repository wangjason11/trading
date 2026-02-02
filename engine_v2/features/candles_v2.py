from __future__ import annotations

import numpy as np
import pandas as pd

from engine_v2.common.types import COL_C, COL_H, COL_L, COL_O, COL_TIME, Direction
from engine_v2.features.candle_params import CandleParams


EPS = 1e-12


def compute_candle_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds core metrics:
      - body_len, candle_len, upper_wick, lower_wick
      - direction (-1/0/1)
      - mid_price
      - body_pct (body_len / candle_len)
    """
    out = df.copy()

    o = out[COL_O].astype(float)
    h = out[COL_H].astype(float)
    l = out[COL_L].astype(float)
    c = out[COL_C].astype(float)

    body_len = (c - o).abs()
    candle_len = (h - l).clip(lower=0.0)

    # wicks
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l

    # direction
    direction = np.where(c > o, 1, np.where(c < o, -1, 0)).astype(int)

    out["body_len"] = body_len
    out["candle_len"] = candle_len
    out["upper_wick"] = upper_wick
    out["lower_wick"] = lower_wick
    out["direction"] = direction  # type: ignore[assignment]
    out["mid_price"] = (h + l) / 2.0

    out["body_pct"] = np.divide(
        body_len,
        candle_len,
        out=np.zeros_like(body_len, dtype=float),
        where=candle_len > EPS,
    )

    return out


def classify_candles(df: pd.DataFrame, params: CandleParams) -> pd.DataFrame:
    """
    Produces:
      - candle_type: maru|pinbar|normal
      - pinbar_dir: +1 (up) / -1 (down) / 0 (none)
      - normal_dir: +1 (up) / -1 (down) / 0 (none) - computed after is_special_maru reclassification
      - is_special_maru: bool (independent flag; can be used by patterns later)
    """
    out = df.copy()

    # Default
    out["candle_type"] = "normal"
    out["pinbar_dir"] = 0
    out["normal_dir"] = 0
    out["is_special_maru"] = False

    body_pct = out["body_pct"].astype(float)
    upper = out["upper_wick"].astype(float)
    lower = out["lower_wick"].astype(float)
    length = out["candle_len"].astype(float)
    direction = out["direction"].astype(int)

    # Primary classification (mutually exclusive)
    is_maru = round(body_pct, 2) >= params.maru
    is_pinbar = round(body_pct, 2) < params.pinbar

    # If both true (possible if maru <= pinbar), maru wins
    out.loc[is_pinbar, "candle_type"] = "pinbar"
    out.loc[is_maru, "candle_type"] = "maru"

    # Pinbar direction: body near top => up pinbar; body near bottom => down pinbar
    # Use pinbar_distance as fraction of candle_len.
    dist = params.pinbar_distance * length
    # up pinbar: lower_wick large, upper_wick small (body near top)
    is_up_pin = (out["candle_type"] == "pinbar") & (upper < dist) & (lower > dist)
    # down pinbar: upper_wick large, lower_wick small (body near bottom)
    is_dn_pin = (out["candle_type"] == "pinbar") & (lower < dist) & (upper > dist)

    out.loc[is_up_pin, "pinbar_dir"] = 1
    out.loc[is_dn_pin, "pinbar_dir"] = -1

    # Special maru flag (independent): looser body_pct threshold + body location constraint
    # This is NOT the primary candle_type, but meeting this condition means the candle becomes maru.
    # Bullish (dir=+1): small upper wick (body near top)
    # Bearish (dir=-1): small lower wick (body near bottom)
    is_special_maru = (round(body_pct, 2) >= params.special_maru) & (
        ((upper <= params.special_maru_distance * length) & (direction == 1)) |
        ((lower <= params.special_maru_distance * length) & (direction == -1))
    )
    out.loc[is_special_maru, "is_special_maru"] = True
    out.loc[is_special_maru, "candle_type"] = "maru"

    # Normal direction: based on where the body midpoint sits within the candle range
    # Computed AFTER is_special_maru reclassification so only true normals get a direction
    # Normalize body midpoint to [0, 1] where 0 = low, 1 = high
    o = out[COL_O].astype(float)
    c = out[COL_C].astype(float)
    h = out[COL_H].astype(float)
    l = out[COL_L].astype(float)
    body_mid = (o + c) / 2.0
    # Normalized position: 0 = at low, 1 = at high
    body_mid_normalized = np.divide(
        body_mid - l,
        h - l,
        out=np.full(len(out), 0.5, dtype=float),  # default to 0.5 for doji (h == l)
        where=(h - l) > EPS,
    )
    x = params.normal_distance
    # up normal: body midpoint in top x portion (normalized > 1 - x)
    is_up_normal = (out["candle_type"] == "normal") & (body_mid_normalized > (1 - x))
    # down normal: body midpoint in bottom x portion (normalized < x)
    is_dn_normal = (out["candle_type"] == "normal") & (body_mid_normalized < x)

    out.loc[is_up_normal, "normal_dir"] = 1
    out.loc[is_dn_normal, "normal_dir"] = -1

    return out


def _prior_maru_max_len(df: pd.DataFrame, *, idx: int, lookback: int, anchor_shift: int) -> float:
    """
    Max candle_len among the previous `lookback` maru candles strictly before (idx - anchor_shift).
    """
    cutoff = idx - anchor_shift
    if cutoff <= 0:
        return 0.0

    # Look backward and collect maru candle lengths
    maru_mask = (df["candle_type"].values == "maru")
    lengths = df["candle_len"].values.astype(float)

    found = []
    j = cutoff - 1
    while j >= 0 and len(found) < lookback:
        if maru_mask[j]:
            found.append(lengths[j])
        j -= 1

    if not found:
        return 0.0
    return float(np.max(found))


def classify_big_flags(
    df: pd.DataFrame,
    params: CandleParams,
    *,
    anchor_shifts: tuple[int, ...] = (0,),
) -> pd.DataFrame:
    """
    Adds big flags. Because patterns may need "big_normal/big_maru" evaluated relative to a pattern start,
    we compute per-anchor variants:

      - is_big_maru_as{shift}
      - is_big_normal_as{shift}

    Meaning: for a candle at index i, we compare its candle_len to the max candle_len of the
    previous `lookback` maru candles strictly before i-shift.

    In patterns:
      - 2-candle pattern starting at idx: evaluate candle idx+1 with shift=1
      - 3-candle pattern starting at idx: evaluate candle idx+2 with shift=2
    """
    out = df.copy()
    n = len(out)

    # Initialize
    for s in anchor_shifts:
        out[f"is_big_maru_as{s}"] = False
        out[f"is_big_normal_as{s}"] = False
        out[f"prior_maru_max_len_as{s}"] = 0.0
        out[f"big_ratio_as{s}"] = 0.0

    lengths = out["candle_len"].astype(float).values
    ctype = out["candle_type"].values

    for i in range(n):
        for s in anchor_shifts:
            prior_max = _prior_maru_max_len(out, idx=i, lookback=params.lookback, anchor_shift=s)
            out.loc[i, f"prior_maru_max_len_as{s}"] = prior_max

            if prior_max <= EPS:
                continue

            ratio = round(lengths[i] / prior_max, 2)
            out.loc[i, f"big_ratio_as{s}"] = ratio

            # if ctype[i] == "maru" and ratio >= params.big_maru_threshold:
            
            # big_maru alternatively can apply to every candle
            if ratio >= params.big_maru_threshold:
                out.loc[i, f"is_big_maru_as{s}"] = True

            # big_normal applies to maru or normal (as you described)
            # if ctype[i] in ("maru", "normal") and ratio >= params.big_normal_threshold:
            
            # big_normal alternatively can apply to every candle
            if ratio >= params.big_normal_threshold:
                out.loc[i, f"is_big_normal_as{s}"] = True

    return out


def compute_candle_features(
    df: pd.DataFrame,
    params: CandleParams,
    *,
    anchor_shifts: tuple[int, ...] = (0, 1, 2),
) -> pd.DataFrame:
    """
    Convenience: metrics -> primary classification -> big flags.
    """
    out = compute_candle_metrics(df)
    out = classify_candles(out, params)
    out = classify_big_flags(out, params, anchor_shifts=anchor_shifts)
    return out
