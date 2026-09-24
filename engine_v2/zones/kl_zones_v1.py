# engine_v2/zones/kl_zones_v1.py
from __future__ import annotations

from dataclasses import replace
from typing import List, Optional

import numpy as np
import pandas as pd

from engine_v2.common.types import KLZone
from engine_v2.structure import event_fields as ef
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.zones.structure_lifecycle import (
    compute_cycle_lifecycle,
    compute_reversal_idx_by_sid,
    compute_struct_start_by_sid,
)


# Reversal-idx-per-sid now lives in `structure_lifecycle.compute_reversal_idx_by_sid`
# (B2 dedup, 2026-05-27) — shared with POI; previously duplicated verbatim here
# and in `poi_zones`.


# -------------------------
# Pattern Identification Functions (structure-aware)
# -------------------------

def identify_inside_bar_pattern(
    df: pd.DataFrame,
    anchor_idx: int,
) -> tuple[str, int] | None:
    """
    Check if anchor candle qualifies as "base inside bar" pattern.

    Looks at up to 5 candles left and 5 candles right of anchor.
    If at least 2 candles have their entire range (low to high) within
    the anchor candle's low-high range, returns the pattern.

    Parameters
    ----------
    df : DataFrame
        OHLC data
    anchor_idx : int
        Index of the BOS/CTS candle

    Returns
    -------
    tuple or None
        ("base inside bar", anchor_idx) if pattern matches, None otherwise
    """
    if anchor_idx not in df.index:
        return None

    anchor_low = float(df.loc[anchor_idx, "l"])
    anchor_high = float(df.loc[anchor_idx, "h"])

    # Check up to 5 candles left and 5 candles right
    left_start = max(0, anchor_idx - 5)
    right_end = min(len(df) - 1, anchor_idx + 5)

    inside_count = 0

    for idx in range(left_start, right_end + 1):
        if idx == anchor_idx:
            continue
        if idx not in df.index:
            continue

        candle_low = float(df.loc[idx, "l"])
        candle_high = float(df.loc[idx, "h"])

        # Check if entire candle is within anchor's range
        if candle_low >= anchor_low and candle_high <= anchor_high:
            inside_count += 1

        if inside_count >= 2:
            return ("base inside bar", anchor_idx)

    return None


def identify_2candle_pattern(
    df: pd.DataFrame,
    idx1: int,
    idx2: int,
    *,
    length_threshold: float = 0.7,
) -> tuple[str, int] | None:
    """
    Check 2-candle base patterns.

    Patterns: no base, no base 1st big, no base 2nd big,
              no base long tails up, no base long tails down

    Parameters
    ----------
    df : DataFrame
        OHLC data with candle classification columns
    idx1 : int
        Index of the 1st candle in the pattern
    idx2 : int
        Index of the 2nd candle in the pattern
    length_threshold : float
        Threshold for determining 1st big vs 2nd big (default 0.7)

    Returns
    -------
    tuple or None
        (pattern_name, idx1) if pattern matches, None otherwise
    """
    if idx1 not in df.index or idx2 not in df.index:
        return None

    # Gate conditions
    big0 = int(df.loc[idx1, "is_big_normal_as0"]) == 1
    big1 = int(df.loc[idx2, "is_big_normal_as1"]) == 1
    dir1 = int(df.loc[idx1, "direction"])
    dir2 = int(df.loc[idx2, "direction"])
    dir_flip = dir1 != dir2

    if not (big0 and big1 and dir_flip):
        return None

    # Check candle types
    ctype1 = str(df.loc[idx1, "candle_type"])
    ctype2 = str(df.loc[idx2, "candle_type"])

    is_maru_or_normal_1 = ctype1 in ("maru", "normal")
    is_maru_or_normal_2 = ctype2 in ("maru", "normal")

    pdir1 = int(df.loc[idx1, "pinbar_dir"])
    pdir2 = int(df.loc[idx2, "pinbar_dir"])

    is_up_pinbar_1 = ctype1 == "pinbar" and pdir1 == 1
    is_dn_pinbar_1 = ctype1 == "pinbar" and pdir1 == -1
    is_up_pinbar_2 = ctype2 == "pinbar" and pdir2 == 1
    is_dn_pinbar_2 = ctype2 == "pinbar" and pdir2 == -1

    clen1 = float(df.loc[idx1, "candle_len"])
    clen2 = float(df.loc[idx2, "candle_len"])

    # Determine pattern
    if is_maru_or_normal_1 and is_maru_or_normal_2:
        # Check size comparison
        if clen1 < length_threshold * clen2:
            return ("no base 2nd big", idx1)
        elif clen1 * length_threshold > clen2:
            return ("no base 1st big", idx1)
        else:
            return ("no base", idx1)
    elif is_up_pinbar_1 and is_up_pinbar_2:
        return ("no base long tails up", idx1)
    elif is_dn_pinbar_1 and is_dn_pinbar_2:
        return ("no base long tails down", idx1)

    return None


def identify_1candle_pattern(
    df: pd.DataFrame,
    anchor_idx: int,
) -> tuple[str, int] | None:
    """
    Check 1-candle (pinbar) base patterns.

    Patterns: no base big tail up, no base big tail down

    Parameters
    ----------
    df : DataFrame
        OHLC data with candle classification columns
    anchor_idx : int
        Index of the BOS/CTS candle

    Returns
    -------
    tuple or None
        (pattern_name, anchor_idx) if pattern matches, None otherwise
    """
    if anchor_idx not in df.index:
        return None

    ctype = str(df.loc[anchor_idx, "candle_type"])
    pdir = int(df.loc[anchor_idx, "pinbar_dir"])
    is_big = int(df.loc[anchor_idx, "is_big_maru_as0"]) == 1

    if ctype != "pinbar" or not is_big:
        return None

    if pdir == 1:
        return ("no base big tail up", anchor_idx)
    elif pdir == -1:
        return ("no base big tail down", anchor_idx)

    return None


def identify_3candle_pattern(
    df: pd.DataFrame,
    idx1: int,
    idx2: int,
    idx3: int,
    *,
    length_threshold: float = 0.7,
) -> tuple[str, int] | None:
    """
    Check 3-candle (star) base patterns.

    Patterns: no base star, no base star 1st big, no base star 2nd big

    Star pattern: maru/normal + pinbar + maru/normal with opposite direction
    between candle 1 and candle 3.

    Parameters
    ----------
    df : DataFrame
        OHLC data with candle classification columns
    idx1 : int
        Index of the 1st candle (before anchor)
    idx2 : int
        Index of the 2nd candle (anchor/middle)
    idx3 : int
        Index of the 3rd candle (after anchor)
    length_threshold : float
        Threshold for determining 1st big vs 2nd big (default 0.7)

    Returns
    -------
    tuple or None
        (pattern_name, idx1) if pattern matches, None otherwise
    """
    if idx1 not in df.index or idx2 not in df.index or idx3 not in df.index:
        return None

    # Star conditions
    dir1 = int(df.loc[idx1, "direction"])
    dir3 = int(df.loc[idx3, "direction"])
    opposite_dir = dir1 != dir3

    ctype1 = str(df.loc[idx1, "candle_type"])
    ctype2 = str(df.loc[idx2, "candle_type"])
    ctype3 = str(df.loc[idx3, "candle_type"])

    star0 = ctype1 in ("maru", "normal") and int(df.loc[idx1, "is_big_normal_as0"]) == 1
    star1 = ctype2 == "pinbar"
    star2 = ctype3 in ("maru", "normal") and int(df.loc[idx3, "is_big_normal_as2"]) == 1

    if not (opposite_dir and star0 and star1 and star2):
        return None

    clen1 = float(df.loc[idx1, "candle_len"])
    clen3 = float(df.loc[idx3, "candle_len"])

    # Determine pattern variant
    if clen1 < length_threshold * clen3:
        return ("no base star 2nd big", idx1)
    elif clen1 * length_threshold > clen3:
        return ("no base star 1st big", idx1)
    else:
        return ("no base star", idx1)


# -------------------------
# Base Window Features (computed on-the-fly)
# -------------------------

def compute_base_window_features(
    df: pd.DataFrame,
    base_idx: int,
    pattern_name: str,
) -> dict:
    """
    Compute base window features on-the-fly based on pattern type and base_idx.

    Parameters
    ----------
    df : DataFrame
        OHLC data
    base_idx : int
        Index of the first candle in the pattern window
    pattern_name : str
        The identified pattern name

    Returns
    -------
    dict
        Contains: base_low, base_high, base_min_close_open, base_max_close_open
        Some values may be NaN for patterns that use find_base_threshold instead.
    """
    # Determine window size based on pattern
    if pattern_name in ("base", "base inside bar"):
        # 1-candle window, uses find_base_threshold for inner
        base_low = float(df.loc[base_idx, "l"])
        base_high = float(df.loc[base_idx, "h"])
        return {
            "base_low": base_low,
            "base_high": base_high,
            "base_min_close_open": float("nan"),
            "base_max_close_open": float("nan"),
        }

    elif pattern_name in ("no base big tail up", "no base big tail down"):
        # 1-candle pinbar pattern
        base_low = float(df.loc[base_idx, "l"])
        base_high = float(df.loc[base_idx, "h"])
        o = float(df.loc[base_idx, "o"])
        c = float(df.loc[base_idx, "c"])
        return {
            "base_low": base_low,
            "base_high": base_high,
            "base_min_close_open": min(o, c),
            "base_max_close_open": max(o, c),
        }

    elif pattern_name.startswith("no base star"):
        # 3-candle star pattern: base_idx, base_idx+1, base_idx+2
        idx1, idx2, idx3 = base_idx, base_idx + 1, base_idx + 2

        l1 = float(df.loc[idx1, "l"])
        l2 = float(df.loc[idx2, "l"])
        l3 = float(df.loc[idx3, "l"])
        h1 = float(df.loc[idx1, "h"])
        h2 = float(df.loc[idx2, "h"])
        h3 = float(df.loc[idx3, "h"])

        # For star patterns, min/max close_open uses candles 1 and 3 only (not middle pinbar)
        o1 = float(df.loc[idx1, "o"])
        c1 = float(df.loc[idx1, "c"])
        o3 = float(df.loc[idx3, "o"])
        c3 = float(df.loc[idx3, "c"])

        return {
            "base_low": min(l1, l2, l3),
            "base_high": max(h1, h2, h3),
            "base_min_close_open": min(o1, c1, o3, c3),
            "base_max_close_open": max(o1, c1, o3, c3),
        }

    else:
        # 2-candle patterns (no base, no base 1st big, no base 2nd big, long tails)
        idx1, idx2 = base_idx, base_idx + 1

        l1 = float(df.loc[idx1, "l"])
        l2 = float(df.loc[idx2, "l"])
        h1 = float(df.loc[idx1, "h"])
        h2 = float(df.loc[idx2, "h"])

        o1 = float(df.loc[idx1, "o"])
        c1 = float(df.loc[idx1, "c"])
        o2 = float(df.loc[idx2, "o"])
        c2 = float(df.loc[idx2, "c"])

        return {
            "base_low": min(l1, l2),
            "base_high": max(h1, h2),
            "base_min_close_open": min(o1, c1, o2, c2),
            "base_max_close_open": max(o1, c1, o2, c2),
        }



# -------------------------
# find_base_threshold (ported intent, fixed)
# -------------------------

def find_base_threshold(df: pd.DataFrame, idx: int, struct_direction: int, *, bos: bool = True) -> float:
    """Inner threshold for a 1-candle base ("base" / "base inside bar").

    Built from the ±5 neighbours' INNER-EDGE body point — the body extreme on
    the side of the zone facing the price interior — bounded so the inner can
    never sit beyond the outer extreme established by the base candle:

      - outer = base_high (BOS sd=-1 / CTS sd=+1): zone opens DOWN, inner-edge =
        bottom of body = min(o,c); kept points must be <= base_high; inner is
        the 2nd-highest such point (closest to the outer).
      - outer = base_low  (BOS sd=+1 / CTS sd=-1): zone opens UP, inner-edge =
        top of body = max(o,c); kept points must be >= base_low; inner is the
        2nd-lowest such point.

    Only the single inner-edge point is tested (not BOTH o and c): a neighbour
    whose far-side body extreme is beyond the outer still qualifies on its near
    side. This guarantees the base extreme stays the outer edge, the inner sits
    within it (containing the base), and it widens the zone / avoids degenerate
    too-narrow bases that requiring both o and c produced. All ±5 neighbours are
    pooled and ranked by price (no pre/post precedence). Falls back to the
    single closest qualifying neighbour, else NaN.
    """
    n = len(df)
    i = int(idx)
    sd = int(struct_direction)
    if n == 0:
        return float("nan")

    left = max(0, i - 5)
    right = min(n, i + 6)
    neighbor_df = pd.concat([df.iloc[left:i], df.iloc[i + 1:right]], axis=0)
    if neighbor_df.empty:
        return float("nan")

    o = neighbor_df["o"].astype(float)
    c = neighbor_df["c"].astype(float)

    # Outer extreme (set by the base candle) + the inner-edge body point that
    # faces the zone interior. `use_low_outer` mirrors find_pinbar_threshold.
    use_low_outer = (bos and sd == 1) or ((not bos) and sd == -1)

    if use_low_outer:
        outer = float(df.loc[i, "l"])
        sel = np.maximum(o, c)        # top of body (c of bullish, o of bearish)
        sel = sel[sel >= outer]       # keep inner-side points (above the low)
        levels = sorted(set(sel))     # ascending: closest to the low outer first
        cmp = lambda lvl: int((sel <= lvl).sum())
    else:
        outer = float(df.loc[i, "h"])
        sel = np.minimum(o, c)        # bottom of body (o of bullish, c of bearish)
        sel = sel[sel <= outer]       # keep inner-side points (below the high)
        levels = sorted(set(sel), reverse=True)  # descending: closest to high first
        cmp = lambda lvl: int((sel >= lvl).sum())

    result = None
    for level in levels:
        count = cmp(level)
        if count >= 1:
            result = float(level)
        if count >= 2:
            return float(level)
    return float(result) if result is not None else float("nan")

def find_pinbar_threshold(
    df: pd.DataFrame,
    base_idx: int,
    *,
    bos: bool,
    struct_direction: int,
) -> float:
    """
    Returns the INNER threshold for a single-big-pinbar base.
    Outer extreme is handled by zone_thresholds via base_low/base_high;
    here we choose the neighbor O/C closest to the correct extreme reference.

    Extreme reference (outer) depends on BOS/CTS and struct_direction:
      - BOS, sd=+1  -> reference = LOW
      - CTS, sd=+1  -> reference = HIGH
      - BOS, sd=-1  -> reference = HIGH
      - CTS, sd=-1  -> reference = LOW

    (c) inner-side bound (2026-06-20): the chosen inner must sit WITHIN the
    outer (= the base candle's own extreme `ref`), never beyond it — otherwise
    the zone inverts (inner past the base extreme, rectangle sitting outside the
    base). Candidates beyond the outer are dropped before the closest pick:
    outer=LOW (use_low_ref) -> keep candidates >= ref; outer=HIGH -> keep <= ref.
    This is the ONLY change vs the original closest-neighbour pinbar rule — it
    preserves the tight pinbar zone (unchanged whenever the closest neighbour is
    already within the outer, which is the normal case) and adds inversion
    safety for the edge case. Deliberately NOT the base/inside-bar inner-edge
    rule (no ±5 pooling, no 2nd-closest widening): pinbar zones key off the
    pinbar's own body/tail and must stay tight (decision 2026-06-20 after the
    inner-edge rule was found to over-widen pinbar zones). If every neighbour is
    beyond the outer (or no neighbour exists), fall back to the base candle's
    own body point closest to ref — itself always within the outer.
    """
    n = len(df)
    i = int(base_idx)
    if n == 0:
        return float("nan")

    sd = int(struct_direction)

    # Select reference extreme for "outer"
    use_low_ref = (bos and sd == 1) or ((not bos) and sd == -1)
    ref = float(df.loc[i, "l"] if use_low_ref else df.loc[i, "h"])

    # Base candle's own body point closest to ref — always within the outer
    # (o, c lie inside [l, h]). Used as the fallback for edges / all-beyond.
    own_o = float(df.loc[i, "o"])
    own_c = float(df.loc[i, "c"])
    own = own_o if abs(own_o - ref) <= abs(own_c - ref) else own_c

    # Need neighbors; if missing, fall back to the base candle's body point.
    if i - 1 < 0 or i + 1 >= n:
        return own

    candidates = [
        float(df.loc[i - 1, "o"]),
        float(df.loc[i - 1, "c"]),
        float(df.loc[i + 1, "o"]),
        float(df.loc[i + 1, "c"]),
    ]

    # Inner-side bound: drop neighbour points that sit beyond the outer.
    if use_low_ref:
        bounded = [x for x in candidates if x >= ref]
    else:
        bounded = [x for x in candidates if x <= ref]

    if not bounded:
        # Every neighbour point is beyond the outer -> the original rule would
        # have inverted the zone; fall back to the base body point instead.
        return own

    inner = min(bounded, key=lambda x: abs(x - ref))
    return float(inner)


# -------------------------
# zone_thresholds (updated to compute features on-the-fly)
# -------------------------

def zone_thresholds(
    df: pd.DataFrame,
    base_idx: int,
    struct_direction: int,
    zone_pattern: str,
    *,
    bos: bool,
) -> tuple[float, float]:
    """
    Returns (outer, inner) bounds (your legacy meaning).
    We'll convert to (top/bottom) for charting when creating KLZone.

    Now computes base window features on-the-fly instead of reading from df columns.
    """
    sd = int(struct_direction)

    # Compute base window features on-the-fly
    features = compute_base_window_features(df, base_idx, zone_pattern)
    base_low = features["base_low"]
    base_high = features["base_high"]
    base_min_close_open = features["base_min_close_open"]
    base_max_close_open = features["base_max_close_open"]

    if zone_pattern in ("no base big tail up", "no base big tail down"):
        inner = find_pinbar_threshold(df, base_idx, bos=bos, struct_direction=struct_direction)
        if bos:
            return (base_low, inner) if sd == 1 else (base_high, inner)
        else:
            return (base_high, inner) if sd == 1 else (base_low, inner)

    if zone_pattern in ["no base 2nd big", "no base star 2nd big"]:
        inner = float(df.loc[base_idx + 1, "mid_price"])
        if bos:
            return (base_low, inner) if sd == 1 else (base_high, inner)
        else:
            return (base_high, inner) if sd == 1 else (base_low, inner)

    if zone_pattern in ["no base 1st big", "no base star 1st big"]:
        # For 2-candle "1st big", inner is close of 2nd candle (base_idx + 1)
        # For 3-candle star "1st big", inner is close of 3rd candle (base_idx + 2)
        if zone_pattern == "no base star 1st big":
            inner = float(df.loc[base_idx + 2, "c"])
        else:
            inner = float(df.loc[base_idx + 1, "c"])
        if bos:
            return (base_low, inner) if sd == 1 else (base_high, inner)
        else:
            return (base_high, inner) if sd == 1 else (base_low, inner)

    if zone_pattern in ["no base", "no base star"]:
        inner = float(df.loc[base_idx, "o"])
        if bos:
            return (base_low, inner) if sd == 1 else (base_high, inner)
        else:
            return (base_high, inner) if sd == 1 else (base_low, inner)

    # Pinbar long-tail patterns
    if zone_pattern == "no base long tails up":
        # inner uses base_min_close_open
        inner = base_min_close_open
        if bos:
            # legacy condition was "up pinbar and direction==+1" in bos branch
            return (base_low, inner) if sd == 1 else (base_high, base_max_close_open)
        else:
            # legacy condition was "up pinbar and direction==-1" in non-bos branch
            return (base_low, inner) if sd == -1 else (base_high, base_max_close_open)

    if zone_pattern == "no base long tails down":
        inner = base_max_close_open
        if bos:
            return (base_high, inner) if sd == -1 else (base_low, base_min_close_open)
        else:
            return (base_high, inner) if sd == 1 else (base_low, base_min_close_open)

    # Default fallback: "base", "base inside bar", or unknown patterns
    thr = float(find_base_threshold(df, base_idx, sd, bos=bos))
    if bos:
        return (base_low, thr) if sd == 1 else (base_high, thr)
    else:
        return (base_high, thr) if sd == 1 else (base_low, thr)


# -------------------------
# Pattern Identification Orchestration
# -------------------------

def identify_base_pattern(
    df: pd.DataFrame,
    anchor_idx: int,
    struct_direction: int,
    *,
    bos: bool,
    length_threshold: float = 0.7,
) -> tuple[str, int]:
    """
    Orchestrate pattern identification for a BOS/CTS event.

    Order of checks:
    1. Inside bar pattern (new)
    2. 2-candle patterns
    3. 1-candle (pinbar) patterns
    4. 3-candle (star) patterns
    5. Default to "base"

    Parameters
    ----------
    df : DataFrame
        OHLC data with candle classification columns
    anchor_idx : int
        Index of the BOS/CTS candle
    struct_direction : int
        Structure direction (+1 bullish, -1 bearish)
    bos : bool
        True if BOS event, False if CTS event
    length_threshold : float
        Threshold for 1st big vs 2nd big patterns

    Returns
    -------
    tuple
        (pattern_name, base_idx)
    """
    sd = int(struct_direction)

    # 1. Check inside bar pattern first
    result = identify_inside_bar_pattern(df, anchor_idx)
    if result is not None:
        return result

    # 2. Compute indices for 2-candle patterns based on BOS/CTS and direction
    anchor_dir = int(df.loc[anchor_idx, "direction"])

    if bos:
        if anchor_dir == sd:
            # BOS candle is 2nd, candle before is 1st
            idx1, idx2 = anchor_idx - 1, anchor_idx
        else:
            # BOS candle is 1st, candle after is 2nd
            idx1, idx2 = anchor_idx, anchor_idx + 1
    else:  # CTS
        if anchor_dir == sd:
            # CTS candle is 1st, candle after is 2nd
            idx1, idx2 = anchor_idx, anchor_idx + 1
        else:
            # CTS candle is 2nd, candle before is 1st
            idx1, idx2 = anchor_idx - 1, anchor_idx

    # Check 2-candle patterns
    result = identify_2candle_pattern(df, idx1, idx2, length_threshold=length_threshold)
    if result is not None:
        return result

    # 3. Check 1-candle (pinbar) pattern
    result = identify_1candle_pattern(df, anchor_idx)
    if result is not None:
        return result

    # 4. Check 3-candle (star) pattern with pre-conditions
    # Candle positions: idx1 = anchor-1, idx2 = anchor, idx3 = anchor+1
    star_idx1 = anchor_idx - 1
    star_idx2 = anchor_idx
    star_idx3 = anchor_idx + 1

    # Pre-conditions
    can_check_star = False
    if star_idx1 in df.index and star_idx3 in df.index:
        if bos:
            # For BOS: candle 3 direction must == struct_direction
            star_dir3 = int(df.loc[star_idx3, "direction"])
            can_check_star = (star_dir3 == sd)
        else:
            # For CTS: candle 1 direction must == struct_direction
            star_dir1 = int(df.loc[star_idx1, "direction"])
            can_check_star = (star_dir1 == sd)

    if can_check_star:
        result = identify_3candle_pattern(df, star_idx1, star_idx2, star_idx3, length_threshold=length_threshold)
        if result is not None:
            return result

    # 5. Default to "base"
    return ("base", anchor_idx)


# -------------------------
# Public API: derive zones from structure events
# -------------------------

"""
Zone Semantics (canonical)

Identifiers
- structure_id: market structure unit id (directional regime). Starts at 0. Increments on reversal.
- cts_cycle_id: internal CTS/BOS cycle id within a structure. Starts at 0.

StructureEvent indexing
- ev.idx: the raw index (today the BOS anchor / the CTS_CONFIRMED confirmation candle; Plan E E4
  flips BOS to the moment). The BOS level is read from meta["bos_anchor_idx"] (`ef.bos_anchor_idx`).
- ev.meta["confirmed_at"]: the candle index where that level was confirmed (breakout/pullback timing).

Zone indexing
- meta["base_idx"]: anchor candle of the zone base pattern (where rectangle begins).
- meta["source_event_idx"]: the source event's raw ev.idx — write-only (no reader; Plan E E4b-pre deletes it).
- meta["confirmed_idx"]: the candle index where the zone becomes confirmed for charting:
    - BOS-derived zones: confirmed_idx = ev.meta["confirmed_at"] (breakout candle)
    - CTS-derived zones: confirmed_idx = ev.idx (pullback candle)

Chart rules
- Show zones for the most recent structure_id.
- Within that structure, the most recent buy and sell zones have higher opacity (active=True).
"""

def derive_kl_zones_v1(
    df: pd.DataFrame,
    events: List[StructureEvent],
    *,
    struct_direction: int,
    length_threshold: float = 0.7,
    source_kinds: Optional[List[str]] = None,
    lifecycle_floor: Optional[int] = None,
    lifecycle_cap: Optional[int] = None,
    cap_reason: str = "lifecycle_end",
) -> List[KLZone]:
    """
    Event-driven KL Zones v1:
    - On CTS_CONFIRMED / BOS_CONFIRMED, identify base pattern (structure-aware)
    - Build zone using identified pattern + thresholds logic
    - Maintain 1 active buy + 1 active sell (most recent)
    - Stamp cycle_id from df["cts_cycle_id"] if present
    """
    dfx = df

    zones: List[KLZone] = []
    active_buy_idx: Optional[int] = None
    active_sell_idx: Optional[int] = None

    sd = int(struct_direction)
    sid = -1  # NEW: current event's structure_id (refreshed per event from ev.meta)
    current_sid: Optional[int] = None  # Track current structure for reversal detection

    # Pre-compute reversal confirmed indices per structure_id (for deactivating zones on reversal)
    rev_confirmed_by_sid = compute_reversal_idx_by_sid(events)

    # Debugging
    # (idx, anchor, sid, cycle): the raw idx AND the anchor, so the Plan E E4
    # flip shows as idx moving while the anchor stays (PLAN_E §6.3 B12).
    print("[kl_zones][events] BOS_CONFIRMED:", [
        (int(ev.idx), ef.bos_anchor_idx(ev), ev.meta.get("structure_id"), ev.meta.get("cycle_id"))
        for ev in events if ev.type == "BOS_CONFIRMED"
    ])
    print("[kl_zones][events] CTS_ESTABLISHED:", [
        (int(ev.idx), ef.cts_anchor_idx(ev), ev.meta.get("structure_id"), ev.meta.get("cycle_id"))
        for ev in events if ev.type == "CTS_ESTABLISHED"
    ])
    print("[kl_zones][events] CTS_CONFIRMED:", [
        (int(ev.idx), ev.meta.get("structure_id"), ev.meta.get("cycle_id"))
        for ev in events if ev.type == "CTS_CONFIRMED"
    ])

    print("[kl_zones][events] BOS_CONFIRMED:", [
        (int(ev.idx),
        ef.bos_anchor_idx(ev),
        (ev.meta or {}).get("confirmed_at"),
        (ev.meta or {}).get("structure_id"),
        (ev.meta or {}).get("cycle_id"),
        (ev.meta or {}).get("source"))
        for ev in events if ev.type == "BOS_CONFIRMED"
    ])

    # convenience
    def _time(i: int):
        return pd.to_datetime(dfx.loc[i, "time"], utc=True)

    def _cycle_id(i: int) -> int:
        return int(dfx.loc[i, "cts_cycle_id"]) if "cts_cycle_id" in dfx.columns else 0

    for ev in events:
        if ev.type not in ("BOS_CONFIRMED", "CTS_CONFIRMED", "CTS_THRESHOLD_UPDATED", "BOS_THRESHOLD_UPDATED", "CTS_ESTABLISHED"):
            continue

        # source_kinds filter: skip events not matching allowed kinds
        if source_kinds is not None:
            if ev.type == "BOS_CONFIRMED" and "BOS" not in source_kinds:
                continue
            if ev.type in ("CTS_CONFIRMED", "CTS_ESTABLISHED", "CTS_THRESHOLD_UPDATED") and "CTS" not in source_kinds:
                continue
            if ev.type == "BOS_THRESHOLD_UPDATED" and "BOS" not in source_kinds:
                continue

        # NEW: make sd/sid event-accurate for ALL event types (zones can span multiple structures now)
        sd = int((ev.meta or {}).get("struct_direction", sd))
        sid = int((ev.meta or {}).get("structure_id", sid))

        if ev.type in ("CTS_THRESHOLD_UPDATED", "BOS_THRESHOLD_UPDATED"):
            # NOTE: sd/sid already refreshed above; you can keep these lines or delete them.
            sd = int((ev.meta or {}).get("struct_direction", struct_direction))
            sid = int((ev.meta or {}).get("structure_id", -1))
            price = float(ev.price)

            # NEW: do not apply expansions on/after reversal candle
            try:
                if "market_state" in dfx.columns:
                    ms = str(dfx.loc[int(ev.idx), "market_state"]).lower()
                    if ms == "reversal":
                        continue
            except Exception:
                pass

            # map event -> which side’s active zone expands
            if ev.type == "CTS_THRESHOLD_UPDATED":
                # CTS-side zone mapping should mirror your CTS_CONFIRMED mapping
                side = "sell" if sd == 1 else "buy"
                zi = active_sell_idx if side == "sell" else active_buy_idx
            else:
                # BOS-side zone mapping should mirror your BOS_CONFIRMED mapping
                side = "buy" if sd == 1 else "sell"
                zi = active_buy_idx if side == "buy" else active_sell_idx

            if zi is None:
                continue

            z0 = zones[zi]

            # guard: only expand zones belonging to same structure_id
            if int((z0.meta or {}).get("structure_id", -999)) != sid:
                continue

            top = float(z0.top)
            bot = float(z0.bottom)

            # expand only in the more extreme direction:
            # buy zones expand DOWN (bottom decreases)
            # sell zones expand UP (top increases)
            if side == "buy":
                bot2, top2 = min(bot, price), top
            else:
                bot2, top2 = bot, max(top, price)

            if top2 != top or bot2 != bot:
                steps = list((z0.meta or {}).get("bounds_steps", []))
                steps.append({
                    "start_idx": int(ev.idx),       # expansion happens HERE
                    "top": float(top2),
                    "bottom": float(bot2),
                    "event": str(ev.type),
                    "price": float(price),
                })

                zones[zi] = replace(
                    z0,
                    top=float(top2),
                    bottom=float(bot2),
                    meta={
                        **(z0.meta or {}),
                        "bounds_steps": steps,
                        "expanded": True,
                        "expanded_last_idx": int(ev.idx),
                        "expanded_last_price": float(price),
                        "expanded_last_event": str(ev.type),
                    },
                )

            continue

        # CTS_ESTABLISHED: release the active CTS-zone tracker so the prior
        # cycle's CTS zone is no longer the expansion target. Phase 3: the
        # zone's end_time is owned by the cycle-end post-pass (no end-write
        # here — the old "cts_established" early-end is subsumed by the
        # cycle inheriting next-cycle CTS-established as its end).
        if ev.type == "CTS_ESTABLISHED":
            cts_side = "sell" if sd == 1 else "buy"
            zi = active_sell_idx if cts_side == "sell" else active_buy_idx
            if zi is not None:
                z0 = zones[zi]
                if (int((z0.meta or {}).get("structure_id", -999)) == sid
                        and z0.source_kind == "CTS"):
                    if cts_side == "sell":
                        active_sell_idx = None
                    else:
                        active_buy_idx = None
            continue

        # The event's RAW idx, exported write-only as meta `source_event_idx`
        # (a declared raw reader; Plan E E4b-pre deletes it). confirmed_at is the
        # candle that CONFIRMED the zone.
        source_event_idx = int(ev.idx)
        confirmed_idx = int((ev.meta or {}).get("confirmed_at", source_event_idx))
        bos = (ev.type == "BOS_CONFIRMED")

        # Anchor for pattern identification differs by event type
        if bos:
            anchor_idx = ef.bos_anchor_idx(ev)   # the BOS anchor: the zone's base candle
        else:
            anchor_idx = int((ev.meta or {}).get("cts_anchor_idx", source_event_idx))

        # Identify base pattern (structure-aware)
        pat, base_idx = identify_base_pattern(dfx, anchor_idx, sd, bos=bos, length_threshold=length_threshold)
        outer, inner = zone_thresholds(dfx, base_idx, sd, pat, bos=bos)

        # Resolve structure_id (authoritative = event meta, fallback = df)
        sid = int((ev.meta or {}).get("structure_id", -1))
        if sid < 0 and "structure_id" in dfx.columns:
            sid = int(dfx.loc[confirmed_idx, "structure_id"])

        # --- Structure change detection: release trackers when a new structure
        # begins after a reversal. Phase 3: per-zone reversal end-writes removed
        # — the cycle-end post-pass caps prior-structure zones at the reversal
        # idx via end_reason="reversal".
        if current_sid is not None and sid != current_sid and current_sid in rev_confirmed_by_sid:
            active_buy_idx = None
            active_sell_idx = None

        current_sid = sid

        # Side mapping (locked)
        if sd == 1:
            side = "buy" if bos else "sell"
        else:
            side = "sell" if bos else "buy"

        top = float(max(outer, inner))
        bottom = float(min(outer, inner))

        # CTS-zone-specific: capture confirmation_method from the event
        # ("pullback" or "sd_zone_proximity"). Will be upgraded to "pullback"
        # by a subsequent CTS_RECONFIRMED event for the same (sid, cycle_id)
        # — handled below outside the zone-creation switch.
        cts_confirmation_method = None
        if not bos:
            cts_confirmation_method = (ev.meta or {}).get("confirmation_method", "pullback")

        z = KLZone(
            start_time=_time(base_idx),
            end_time=None,
            side=side,
            top=top,
            bottom=bottom,
            source_kind="BOS" if bos else "CTS",
            source_time=_time(confirmed_idx),
            source_price=float(ev.price) if ev.price is not None else float("nan"),
            strength=0.0,
            meta={
                "structure_id": sid,
                "struct_direction": sd,
                "cycle_id": _cycle_id(confirmed_idx),

                # Zone confirmation semantics
                "confirmed_idx": confirmed_idx,          # breakout / pullback candle
                "source_event_idx": source_event_idx,    # BOS/CTS level candle

                # Zone base
                "anchor_idx": anchor_idx,
                "base_idx": base_idx,
                "base_pattern": pat,
                "outer": float(outer),
                "inner": float(inner),

                # CTS-only: how this CTS was confirmed. May be upgraded later
                # to "pullback" by a CTS_RECONFIRMED event for the same cycle.
                **({"confirmation_method": cts_confirmation_method} if not bos else {}),

                "bounds_steps": [
                    {
                        "start_idx": int(base_idx),   # segment begins at base anchor candle
                        "top": float(top),
                        "bottom": float(bottom),
                        "event": "INIT",
                    }
                ],
            },
        )

        # Track the most-recent zone per side (the expansion target for
        # *_THRESHOLD_UPDATED). Phase 3: the prior same-side zone's end_time is
        # owned by the cycle-end post-pass (no end-write on replacement).
        if side == "buy":
            active_buy_idx = len(zones)
        else:
            active_sell_idx = len(zones)

        zones.append(z)

    # ------------------------------------------------------------------
    # Post-pass: process CTS_RECONFIRMED events. When CTS was first
    # confirmed via sd zone proximity AND a valid pullback fired later,
    # upgrade the CTS zone's confirmation_method to "pullback" and record
    # pb_reconfirm_idx. The original CTS_CONFIRMED event stays immutable.
    # ------------------------------------------------------------------
    for ev in events:
        if ev.type != "CTS_RECONFIRMED":
            continue
        ev_sid = (ev.meta or {}).get("structure_id")
        ev_cycle = (ev.meta or {}).get("cycle_id")
        if ev_sid is None or ev_cycle is None:
            continue

        for zi, z in enumerate(zones):
            if z.source_kind != "CTS":
                continue
            zmeta = z.meta or {}
            if zmeta.get("structure_id") != ev_sid:
                continue
            if zmeta.get("cycle_id") != ev_cycle:
                continue
            # Found the CTS zone for this (sid, cycle). Upgrade it.
            zones[zi] = replace(
                z,
                meta={
                    **zmeta,
                    "confirmation_method": "pullback",
                    "pb_reconfirm_idx": int(ev.idx),
                },
            )
            break

    # ------------------------------------------------------------------
    # Unified lifecycle (Phase 3 2026-05-26; END pass-through B2 2026-05-27).
    # Each zone INHERITS its owning cycle's resolved end from the shared
    # `compute_cycle_lifecycle` table — ends are derived from starts, never
    # computed per-zone (PART4_REFACTOR_SPEC §5 "End resolution as
    # start-passthrough"). The zone keeps its OWN start (first-active), clamped
    # up to the structure lifecycle-start (B1) — nothing inherits cycle start.
    # ------------------------------------------------------------------
    last_candle = int(dfx.index.max())

    # struct_start (per sid) for the first-active clamp (B1, unchanged). The
    # cycle lifecycle table (start, end, end_reason) for END inheritance: end =
    # min(next-cycle clamped start, reversal, lifecycle_cap). For main cap=None;
    # for subs the projection (`render_sub_projection`) supplies the slice-local
    # cap (+ cap_reason) — the unique sub's lifecycle end (PART4 §17.9).
    struct_start_by_sid = compute_struct_start_by_sid(
        events, rev_confirmed_by_sid, lifecycle_floor,
    )
    cycle_life = compute_cycle_lifecycle(
        events, rev_confirmed_by_sid, lifecycle_floor, lifecycle_cap, cap_reason,
    )

    for zi, z in enumerate(zones):
        zmeta = z.meta or {}
        zsid = zmeta.get("structure_id")
        zcyc = zmeta.get("cycle_id")
        raw_confirmed = int(zmeta.get("confirmed_idx"))
        # Clamp first-active (the zone's OWN start) to the structure
        # lifecycle-start (no zone may activate before its structure is alive).
        sstart = struct_start_by_sid.get(int(zsid)) if zsid is not None else None
        confirmed_idx = max(raw_confirmed, int(sstart)) if sstart is not None else raw_confirmed

        # end_idx/end_reason inherited from the cycle (pass-through).
        life = cycle_life.get((int(zsid), int(zcyc))) if (zsid is not None and zcyc is not None) else None
        end_idx = life[1] if life is not None else None
        end_reason = life[2] if life is not None else None

        end_time = _time(end_idx) if end_idx is not None else None

        # KL has a single active interval from confirmed_idx (no condition-state
        # flips). Collapsed (never active) if it would start at/after end_idx.
        if end_idx is not None and confirmed_idx >= end_idx:
            activation_history = []
        else:
            activation_history = [
                {"idx": confirmed_idx, "active": True, "reason": "confirmed"}
            ]

        # Derived 3-state status. Collapsed (no activation) is "inactive" — it
        # never became active, so not "ended" (standardized 2026-05-27; matches
        # the prior per-sid cap, replacing the inner derivation's "ended").
        if not activation_history:
            status = "inactive"
        elif end_idx is not None and end_idx <= last_candle:
            status = "ended"
        else:
            status = "active"

        new_meta = {
            k: v for k, v in zmeta.items()
            if k not in ("active", "deactivated_by")
        }
        new_meta.update({
            # confirmed_idx is the clamped first-active idx (>= structure
            # lifecycle-start); equals the raw structural confirm when unclamped.
            "confirmed_idx": confirmed_idx,
            "end_idx": end_idx,
            "end_reason": end_reason,
            "activation_history": activation_history,
            "status": status,
        })
        zones[zi] = replace(z, end_time=end_time, meta=new_meta)

    return zones


# ---------------------------------------------------------------------------
# Single-zone primitives — used by MarketStructure's dual CTS proximity check.
#
# Wired from `structure/structure_engine.py` as the `bos_inner_resolver` /
# `poi_inners_resolver` callables passed to MarketStructure.__init__ (Part 4
# §13.5.b). Defined here (rather than inside structure/) so the structure→zones
# import inversion stays gone — structure/ never imports zones/.
# ---------------------------------------------------------------------------

def compute_bos_inner_from_event(
    df: pd.DataFrame,
    bos_idx: int,
    struct_direction: int,
    length_threshold: float = 0.7,
) -> Optional[float]:
    """Derive a single BOS zone's inner price for proximity checking.

    Mirrors the per-zone derivation inside `derive_kl_zones_v1` (identify
    base pattern → resolve threshold → return inner). Returns None if the
    base pattern can't be identified, so the caller's proximity check
    naturally falls back to no-trigger.
    """
    if bos_idx not in df.index:
        return None

    try:
        zone_pattern, base_idx = identify_base_pattern(
            df,
            anchor_idx=int(bos_idx),
            struct_direction=int(struct_direction),
            bos=True,
            length_threshold=length_threshold,
        )
        if base_idx is None or base_idx not in df.index:
            return None

        _outer, inner = zone_thresholds(
            df,
            base_idx=int(base_idx),
            struct_direction=int(struct_direction),
            zone_pattern=zone_pattern,
            bos=True,
        )
        if inner is None:
            return None
        return float(inner)
    except Exception:
        return None
