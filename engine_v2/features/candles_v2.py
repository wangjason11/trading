from __future__ import annotations

import numpy as np
import pandas as pd

from engine_v2.common.types import COL_C, COL_H, COL_L, COL_O, COL_TIME, COL_V, Direction
from engine_v2.features.candle_params import CandleParams


EPS = 1e-12

# The agreed direction's split-candle value (`agreed_direction`): a candle whose
# candle direction and volume direction are opposite belongs to no leg.
AGREED_SPLIT = 2


def agreed_direction(direction, vol_dir) -> np.ndarray:
    """The AGREED direction of each candle (PLAN_H §1 item 2, V1a — GLOSSARY
    "Agreed direction"): candle `direction` (sign(c - o)) and `vol_dir`
    (sign(c - prev c), `compute_volume_features`) together.

      direction ±1, vol_dir the same or 0 -> the candle's direction
      direction ±1, vol_dir the opposite  -> `AGREED_SPLIT` (a split candle)
      direction 0,  vol_dir ±1            -> vol_dir (a volume-only doji)
      direction 0,  vol_dir 0             -> 0 (a true doji)

    Element-wise on arrays; a scalar pair gives a 0-d array (`int(...)` it).
    For d = ±1, "qualified in d" (GLOSSARY "Qualified candle": direction == d
    and vol_dir in (d, 0)) == `direction == d and agreed == d`.
    """
    d = np.asarray(direction).astype(int)
    v = np.asarray(vol_dir).astype(int)
    return np.where(d == 0, v, np.where((v == 0) | (v == d), d, AGREED_SPLIT))


def big_ratio_flags(length: float, prior_max: float, params: CandleParams):
    """The single-candle big-flag rule: `(ratio, big_maru_ratio, big_normal_ratio)`
    of a candle (or a stacked leg) of range `length` against the max range of the
    prior `lookback` marus (`prior_max`), or None when there is no prior maru
    (`prior_max <= EPS` -> no flag). Ratio-only — the type gate (big_maru needs
    a maru, big_normal a normal) is the caller's. `classify_big_flags` and the
    institutional pattern's stacks read this one rule."""
    if prior_max <= EPS:
        return None
    ratio = round(length / prior_max, 2)
    return ratio, ratio >= params.big_maru_threshold, ratio >= params.big_normal_threshold


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


def floor_demotes(body_len, type_before_floor, floor):
    """The ONE demotion predicate (PLAN_U §8 A1): a body-% maru / normal whose body is strictly below its floor
    (`body < floor - EPS`; NaN = no floor -> never). Step 4 of `classify_candles` and the rolling floor's "final"
    window (`features/pinbar_body_floor.py`) both read it. Element-wise; scalars give a 0-d array."""
    body = np.asarray(body_len, dtype=float)
    fl = np.asarray(floor, dtype=float)
    before = np.asarray(type_before_floor)
    return ~np.isnan(fl) & (body < np.nan_to_num(fl, nan=0.0) - EPS) & (before != "pinbar")


def classify_candles(df: pd.DataFrame, params: CandleParams, body_floor=None) -> pd.DataFrame:
    """
    Produces:
      - candle_type: maru|pinbar|normal
      - pinbar_dir: +1 (up) / -1 (down) / 0 (none)
      - normal_dir: +1 (up) / -1 (down) / 0 (none) - computed after is_special_maru reclassification
      - is_special_maru: bool (independent flag; currently unused downstream)
      - candle_type_before_floor: the type after step 3 (the body-% type), before the floor
      - pinbar_body_floor: the floor this candle was classified against (price units; NaN = no floor)
      - is_floor_pinbar: a pinbar because of step 4 only (a body-% maru / normal under the floor)

    Type-write order (last write wins):
      1. default normal
      2. body_pct bands -> pinbar / maru
      3. special-maru promotion (0.50-0.64 band + tiny close-side wick) -> maru
      4. the body floor -> pinbar (`body_floor`: one floor per row, price units,
         None / NaN = no floor; trumps all of the above). Direction sub-flags
         (pinbar_dir, normal_dir) are computed AFTER this, so the reclassified
         pinbars are directioned too.
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

    # The body floor -> pinbar (applied LAST; trumps maru/normal/pinbar).
    # A candle whose REAL BODY is shorter than its floor carries no directional
    # conviction, so it is reclassified `pinbar` regardless of its body_pct (and
    # regardless of the special-maru promotion just above). This is why the
    # legacy `is_big_maru` body-pip floor was removed: these small candles can no
    # longer be maru in the first place. The floor is a per-row input in price
    # units (`body_floor`; None / NaN = no floor) — `compute_candle_features`
    # builds it (the rolling floor, `features/pinbar_body_floor.py`). Strict `<`
    # with an EPS guard so a body sitting exactly at the floor keeps its
    # original type (a nominal N-pip body can float-round to N*pip - ~2e-15) —
    # `floor_demotes`, the predicate the rolling floor's "final" window shares.
    out["candle_type_before_floor"] = out["candle_type"]
    floor = (np.full(len(out), np.nan) if body_floor is None
             else np.asarray(body_floor, dtype=float).reshape(-1))
    if len(floor) != len(out):
        raise ValueError(f"[candles_v2] body_floor has {len(floor)} values for {len(out)} candles")
    out["pinbar_body_floor"] = floor
    demoted = floor_demotes(out["body_len"].to_numpy(dtype=float),
                            out["candle_type_before_floor"].to_numpy(), floor)
    out["is_floor_pinbar"] = demoted
    out.loc[demoted, "candle_type"] = "pinbar"
    # Keep the (downstream-unused) flag self-consistent: a tiny-body pinbar
    # is not a maru. No behavioral effect — nothing reads is_special_maru.
    out.loc[demoted, "is_special_maru"] = False

    # Pinbar direction: body near top => up pinbar; body near bottom => down pinbar.
    # Computed AFTER candle_type is final (incl. special-maru promotion AND the
    # body-pip-floor reclassification above) so every final pinbar — including
    # the newly reclassified ones — is directioned by the same wick-geometry
    # rule. Moving this below special-maru is byte-neutral for pre-existing
    # pinbars: special-maru only ever promotes normals, never touches pinbars.
    # Use pinbar_distance as fraction of candle_len.
    dist = params.pinbar_distance * length
    # up pinbar: lower_wick large, upper_wick small (body near top)
    is_up_pin = (out["candle_type"] == "pinbar") & (upper < dist) & (lower > dist)
    # down pinbar: upper_wick large, lower_wick small (body near bottom)
    is_dn_pin = (out["candle_type"] == "pinbar") & (lower < dist) & (upper > dist)

    out.loc[is_up_pin, "pinbar_dir"] = 1
    out.loc[is_dn_pin, "pinbar_dir"] = -1

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

            flags = big_ratio_flags(lengths[i], prior_max, params)
            if flags is None:
                continue

            ratio, is_big_maru, is_big_normal = flags
            out.loc[i, f"big_ratio_as{s}"] = ratio

            # big_maru / big_normal are ratio-only (current candle_len vs the
            # prior-5-maru max range). The former additive body-pip floor on
            # `is_big_maru` was removed: small-bodied candles are now
            # reclassified `pinbar` in `classify_candles` (step 4, the per-candle
            # `pinbar_body_floor`), so they can never be maru — nor
            # enter the prior-maru pool — in the first place. The floor that
            # used to live here now lives at the type-assignment layer.
            if is_big_maru:
                out.loc[i, f"is_big_maru_as{s}"] = True

            if is_big_normal:
                out.loc[i, f"is_big_normal_as{s}"] = True

    return out


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds volume-based features:
      - vol_dir: +1 if close > prev_close, -1 if close < prev_close, 0 otherwise
        (Note: different from candle 'direction' which compares open vs close of same candle)
      - vol_ema20: EMA(volume, 20)
      - vol_spike_ratio: round(volume / vol_ema20, 2)
      - is_vol_spike: True if vol_spike_ratio >= 1.2
    """
    out = df.copy()

    # Check if volume column exists
    if COL_V not in out.columns:
        # Volume not available - add placeholder columns
        out["vol_dir"] = 0
        out["vol_ema20"] = float("nan")
        out["vol_spike_ratio"] = float("nan")
        out["is_vol_spike"] = False
        return out

    c = out[COL_C].astype(float)
    volume = out[COL_V].astype(float)

    # Volume direction: compare current close to previous close
    prev_close = c.shift(1)
    out["vol_dir"] = np.where(
        c > prev_close, 1,
        np.where(c < prev_close, -1, 0)
    ).astype(int)

    # Volume EMA(20)
    out["vol_ema20"] = volume.ewm(span=20, adjust=False).mean()

    # Volume spike ratio: round(volume / vol_ema20, 2)
    vol_ema = out["vol_ema20"].astype(float)
    out["vol_spike_ratio"] = np.round(
        np.divide(
            volume,
            vol_ema,
            out=np.zeros_like(volume, dtype=float),
            where=vol_ema > EPS,
        ),
        2
    )

    # Volume spike indicator: ratio >= 1.2
    out["is_vol_spike"] = out["vol_spike_ratio"] >= 1.2

    return out


def compute_candle_features(
    df: pd.DataFrame,
    params: CandleParams,
    *,
    anchor_shifts: tuple[int, ...] = (0, 1, 2),
    floor_seed: pd.DataFrame | None = None,
    body_floor=None,
) -> pd.DataFrame:
    """
    Convenience: metrics -> primary classification -> big flags -> volume features.
    The body floor of `classify_candles` step 4 = the rolling floor over `floor_seed` + this frame
    (`pinbar_body_floor_for_frame`; `floor_seed` = None -> frame-only windows). The seed fills only the
    windows — the prior-maru pool, big flags and volume features stay this frame's. `body_floor` (one
    floor per row, price units) overrides the rolling floor: a test seam for frames too short to hold a
    window (PLAN_U §8 A5).
    """
    out = compute_candle_metrics(df)
    if body_floor is not None:
        out = classify_candles(out, params, body_floor=body_floor)
    else:
        from engine_v2.features.pinbar_body_floor import pinbar_body_floor_for_frame

        floor, demoted = pinbar_body_floor_for_frame(out, floor_seed, params)
        out = classify_candles(out, params, body_floor=floor)
        # the floor function's own demotions (its "final" window) are the classifier's (PLAN_U §8 A1)
        assert np.array_equal(out["is_floor_pinbar"].to_numpy(dtype=bool), demoted), \
            "[candles_v2] the rolling floor's demotions differ from classify_candles step 4"
    out = classify_big_flags(out, params, anchor_shifts=anchor_shifts)
    out = compute_volume_features(out)
    return out
