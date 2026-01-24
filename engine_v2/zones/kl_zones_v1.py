# engine_v2/zones/kl_zones_v1.py
from __future__ import annotations

from dataclasses import replace
from typing import List, Optional

import numpy as np
import pandas as pd

from engine_v2.common.types import KLZone
from engine_v2.structure.market_structure import StructureEvent


# -------------------------
# Base features (ported)
# -------------------------

def compute_base_features(df: pd.DataFrame, *, length_threshold: float = 0.7) -> pd.DataFrame:
    """
    Port of legacy BasePatterns.base_patterns with corrected pandas mechanics.
    Keeps branching logic identical, including the corrected typo for 'no base 1st big'.

    Requires df columns:
      - is_big_normal_as0, is_big_normal_as1
      - direction
      - candle_type
      - pinbar_dir
      - candle_len
      - o/h/l/c
    """
    dfx = df.copy()

    # --- column exists check ---
    # big-normal flags
    if "is_big_normal_as0" not in dfx.columns:
        raise KeyError("KL zones: missing is_big_normal_as0 flag")

    if "is_big_normal_as1" not in dfx.columns:
        raise KeyError("KL zones: missing is_big_normal_as1 flag")

    # candle length
    if "candle_len" not in dfx.columns:
        raise KeyError("KL zones: missing candle length (expected candle_len)")

    big0 = dfx["is_big_normal_as0"].astype(int)
    big1 = dfx["is_big_normal_as1"].astype(int)
    dir_flip = dfx["direction"].shift(1) != dfx["direction"]

    gate = (big0 == 1) & (big1 == 1) & dir_flip

    prev_is_maru_or_normal = dfx["candle_type"].shift(1).isin(["maru", "normal"])
    prev_big_normal = dfx["is_big_normal_as0"].shift(1).fillna(0).astype(int) == 1

    # Pinbar mapping: up/down pinbar = candle_type=="pinbar" and pinbar_dir==±1
    prev_up_pinbar = (dfx["candle_type"].shift(1) == "pinbar") & (dfx["pinbar_dir"].shift(1) == 1)
    prev_dn_pinbar = (dfx["candle_type"].shift(1) == "pinbar") & (dfx["pinbar_dir"].shift(1) == -1)
    cur_up_pinbar = (dfx["candle_type"] == "pinbar") & (dfx["pinbar_dir"] == 1)
    cur_dn_pinbar = (dfx["candle_type"] == "pinbar") & (dfx["pinbar_dir"] == -1)

    clen = dfx["candle_len"].astype(float)
    prev_clen = clen.shift(1)

    # Corrected typo from you:
    #   prev_clen < length_threshold * clen  => "no base 2nd big"
    #   prev_clen * length_threshold > clen  => "no base 1st big"
    nb2 = prev_clen < (length_threshold * clen)
    nb1 = (prev_clen * length_threshold) > clen

    base_pattern = np.where(
        gate,
        np.where(
            prev_is_maru_or_normal & prev_big_normal,
            np.where(nb2, "no base 2nd big", np.where(nb1, "no base 1st big", "no base")),
            np.where(
                prev_up_pinbar & cur_up_pinbar,
                "no base long tails up",
                np.where(prev_dn_pinbar & cur_dn_pinbar, "no base long tails down", "base"),
            ),
        ),
        "base",
    )

    dfx["base_pattern"] = base_pattern

    # Always compute 2-candle window features (base_idx uses base_idx & base_idx+1 regardless of pattern)
    dfx["base_low"] = np.minimum(dfx["l"], dfx["l"].shift(-1))
    dfx["base_high"] = np.maximum(dfx["h"], dfx["h"].shift(-1))

    c0 = dfx["c"].astype(float)
    c1 = dfx["c"].shift(-1).astype(float)
    o0 = dfx["o"].astype(float)
    o1 = dfx["o"].shift(-1).astype(float)

    min_co = np.minimum(np.minimum(c0, c1), np.minimum(o0, o1))
    max_co = np.maximum(np.maximum(c0, c1), np.maximum(o0, o1))

    dfx["base_min_close_open"] = min_co
    dfx["base_max_close_open"] = max_co

    return dfx


# -------------------------
# Base index resolver (ported)
# -------------------------

def resolve_base_idx_and_pattern(
    df: pd.DataFrame,
    confirmed_idx: int,
    struct_direction: int,
    *,
    bos: bool,
) -> tuple[int, str]:
    """
    Port of legacy check_base_pattern.

    bos=True (BOS-confirmed event):
      if struct_direction == direction[idx] => (idx-1) else idx

    bos=False (CTS-confirmed event):
      if struct_direction == direction[idx] => idx else (idx-1)

    Clamp so base_idx+1 is valid for 2-candle window usage.
    """
    i = int(confirmed_idx)
    sd = int(struct_direction)
    conf_dir = int(df.loc[i, "direction"])

    if bos:
        base_idx = (i - 1) if (sd == conf_dir) else i
    else:
        base_idx = i if (sd == conf_dir) else (i - 1)

    # Ensure base_idx allows base_idx+1 access
    base_idx = max(0, min(base_idx, len(df) - 2))
    zone_pattern = str(df.loc[base_idx, "base_pattern"])
    return base_idx, zone_pattern


# -------------------------
# find_base_threshold (ported intent, fixed)
# -------------------------

def find_base_threshold(df: pd.DataFrame, idx: int, starting_direction: int, *, bos: bool = True) -> float:
    left = max(0, int(idx) - 5)
    right = min(len(df), int(idx) + 6)

    neighbor_df = df.iloc[left:idx].copy()
    neighbor_df = pd.concat([neighbor_df, df.iloc[idx + 1:right]], axis=0)

    if neighbor_df.empty:
        return float("nan")

    candidates_desc = sorted(set(np.minimum(neighbor_df["o"], neighbor_df["c"])), reverse=True)
    candidates_asc = sorted(set(np.maximum(neighbor_df["o"], neighbor_df["c"])), reverse=False)

    sd = int(starting_direction)
    result = None

    if bos:
        levels = candidates_asc if sd == 1 else candidates_desc
        for level in levels:
            if sd == 1:
                count = ((neighbor_df["o"] <= level) & (neighbor_df["c"] <= level)).sum()
            else:
                count = ((neighbor_df["o"] >= level) & (neighbor_df["c"] >= level)).sum()
            if count >= 2:
                result = float(level)
            if count >= 3:
                return float(level)
        return float(result) if result is not None else float("nan")

    else:
        levels = candidates_desc if sd == 1 else candidates_asc
        for level in levels:
            if sd == 1:
                count = ((neighbor_df["o"] >= level) & (neighbor_df["c"] >= level)).sum()
            else:
                # fixed typo: "<=3 level" -> "<= level"
                count = ((neighbor_df["o"] <= level) & (neighbor_df["c"] <= level)).sum()
            if count >= 2:
                result = float(level)
            if count >= 3:
                return float(level)
        return float(result) if result is not None else float("nan")


# -------------------------
# zone_thresholds (ported mapping)
# -------------------------

def zone_thresholds(
    df: pd.DataFrame,
    base_idx: int,
    starting_direction: int,
    zone_pattern: str,
    *,
    bos: bool,
) -> tuple[float, float]:
    """
    Returns (outer, inner) bounds (your legacy meaning).
    We'll convert to (top/bottom) for charting when creating KLZone.
    """
    sd = int(starting_direction)

    base_low = float(df.loc[base_idx, "base_low"])
    base_high = float(df.loc[base_idx, "base_high"])

    if zone_pattern == "no base 2nd big":
        inner = float(df.loc[base_idx + 1, "mid_price"])
        if bos:
            return (base_low, inner) if sd == 1 else (base_high, inner)
        else:
            return (base_high, inner) if sd == 1 else (base_low, inner)

    if zone_pattern == "no base 1st big":
        inner = float(df.loc[base_idx + 1, "c"])
        if bos:
            return (base_low, inner) if sd == 1 else (base_high, inner)
        else:
            return (base_high, inner) if sd == 1 else (base_low, inner)

    if zone_pattern == "no base":
        inner = float(df.loc[base_idx, "o"])
        if bos:
            return (base_low, inner) if sd == 1 else (base_high, inner)
        else:
            return (base_high, inner) if sd == 1 else (base_low, inner)

    # Pinbar long-tail patterns
    if zone_pattern == "no base long tails up":
        # inner uses base_min_close_open
        inner = float(df.loc[base_idx, "base_min_close_open"])
        if bos:
            # legacy condition was "up pinbar and direction==+1" in bos branch
            return (base_low, inner) if sd == 1 else (base_high, float(df.loc[base_idx, "base_max_close_open"]))
        else:
            # legacy condition was "up pinbar and direction==-1" in non-bos branch
            return (base_low, inner) if sd == -1 else (base_high, float(df.loc[base_idx, "base_max_close_open"]))

    if zone_pattern == "no base long tails down":
        inner = float(df.loc[base_idx, "base_max_close_open"])
        if bos:
            return (base_high, inner) if sd == -1 else (base_low, float(df.loc[base_idx, "base_min_close_open"]))
        else:
            return (base_high, inner) if sd == 1 else (base_low, float(df.loc[base_idx, "base_min_close_open"]))

    thr = float(find_base_threshold(df, base_idx, sd, bos=bos))
    if bos:
        return (base_low, thr) if sd == 1 else (base_high, thr)
    else:
        return (base_high, thr) if sd == 1 else (base_low, thr)


# -------------------------
# Public API: derive zones from structure events
# -------------------------

def derive_kl_zones_v1(
    df: pd.DataFrame,
    events: List[StructureEvent],
    *,
    struct_direction: int,
    length_threshold: float = 0.7,
) -> List[KLZone]:
    """
    Event-driven KL Zones v1:
    - Precompute base features
    - On CTS_CONFIRMED / BOS_CONFIRMED, build a zone using legacy base pattern + thresholds logic
    - Maintain 1 active buy + 1 active sell (most recent)
    - Stamp cycle_id from df["cts_cycle_id"] if present
    """
    dfx = df
    if "base_pattern" not in dfx.columns:
        dfx = compute_base_features(dfx, length_threshold=length_threshold)

    zones: List[KLZone] = []
    active_buy_idx: Optional[int] = None
    active_sell_idx: Optional[int] = None

    sd = int(struct_direction)

    # convenience
    def _time(i: int):
        return pd.to_datetime(dfx.loc[i, "time"], utc=True)

    def _cycle_id(i: int) -> int:
        return int(dfx.loc[i, "cts_cycle_id"]) if "cts_cycle_id" in dfx.columns else 0

    for ev in events:
        if ev.type not in ("BOS_CONFIRMED", "CTS_CONFIRMED"):
            continue

        confirmed_idx = int(ev.idx)
        bos = (ev.type == "BOS_CONFIRMED")

        base_idx, pat = resolve_base_idx_and_pattern(dfx, confirmed_idx, sd, bos=bos)
        outer, inner = zone_thresholds(dfx, base_idx, sd, pat, bos=bos)

        # Side mapping (locked)
        if sd == 1:
            side = "buy" if bos else "sell"
        else:
            side = "sell" if bos else "buy"

        top = float(max(outer, inner))
        bottom = float(min(outer, inner))

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
                "cycle_id": _cycle_id(confirmed_idx),
                "confirmed_idx": confirmed_idx,
                "base_idx": base_idx,
                "base_pattern": pat,
                "outer": float(outer),
                "inner": float(inner),
                "active": True,  # updated below
            },
        )

        # Enforce 1 active per side: deactivate previous active of same side
        if side == "buy":
            if active_buy_idx is not None:
                prev = zones[active_buy_idx]
                zones[active_buy_idx] = replace(prev, meta={**prev.meta, "active": False})
            active_buy_idx = len(zones)
        else:
            if active_sell_idx is not None:
                prev = zones[active_sell_idx]
                zones[active_sell_idx] = replace(prev, meta={**prev.meta, "active": False})
            active_sell_idx = len(zones)

        zones.append(z)

    return zones
