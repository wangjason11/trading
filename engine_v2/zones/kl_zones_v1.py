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

"""
Zone Semantics (canonical)

Identifiers
- structure_id: market structure unit id (directional regime). Starts at 0. Increments on reversal.
- cts_cycle_id: internal CTS/BOS cycle id within a structure. Starts at 0.

StructureEvent indexing
- ev.idx: the *level index* (where the BOS/CTS level is anchored; often an earlier extreme).
- ev.meta["confirmed_at"]: the candle index where that level was confirmed (breakout/pullback timing).

Zone indexing
- meta["base_idx"]: anchor candle of the zone base pattern (where rectangle begins).
- meta["source_event_idx"]: the StructureEvent level index used to derive the zone (ev.idx).
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

    # Debugging
    print("[kl_zones][events] BOS_CONFIRMED:", [
        (int(ev.idx), ev.meta.get("structure_id"), ev.meta.get("cycle_id"), ev.meta.get("bos_prev"))
        for ev in events if ev.type == "BOS_CONFIRMED"
    ])
    print("[kl_zones][events] CTS_ESTABLISHED:", [
        (int(ev.idx), ev.meta.get("structure_id"), ev.meta.get("cycle_id"))
        for ev in events if ev.type == "CTS_ESTABLISHED"
    ])
    print("[kl_zones][events] CTS_CONFIRMED:", [
        (int(ev.idx), ev.meta.get("structure_id"), ev.meta.get("cycle_id"))
        for ev in events if ev.type == "CTS_CONFIRMED"
    ])

    print("[kl_zones][events] BOS_CONFIRMED:", [
        (int(ev.idx),
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
        if ev.type not in ("BOS_CONFIRMED", "CTS_CONFIRMED"):
            continue

        # Event idx is the BOS/CTS LEVEL index; confirmed_at is the candle that CONFIRMED it.
        source_event_idx = int(ev.idx)
        confirmed_idx = int((ev.meta or {}).get("confirmed_at", source_event_idx))
        bos = (ev.type == "BOS_CONFIRMED")

        # ✅ base-resolution anchor differs by event type
        if bos:
            base_source_idx = source_event_idx
        else:
            base_source_idx = int((ev.meta or {}).get("cts_anchor_idx", source_event_idx))

        base_idx, pat = resolve_base_idx_and_pattern(dfx, base_source_idx, sd, bos=bos)
        outer, inner = zone_thresholds(dfx, base_idx, sd, pat, bos=bos)

        # Resolve structure_id (authoritative = event meta, fallback = df)
        sid = int((ev.meta or {}).get("structure_id", -1))
        if sid < 0 and "structure_id" in dfx.columns:
            sid = int(dfx.loc[confirmed_idx, "structure_id"])

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
                "structure_id": sid,
                "cycle_id": _cycle_id(confirmed_idx),

                # Zone confirmation semantics
                "confirmed_idx": confirmed_idx,          # breakout / pullback candle
                "source_event_idx": source_event_idx,    # BOS/CTS level candle

                # Zone base
                "base_source_idx": base_source_idx,
                "base_idx": base_idx,
                "base_pattern": pat,
                "outer": float(outer),
                "inner": float(inner),

                "active": True,
            },
        )

        # Enforce 1 active per side: deactivate previous active of same side
        deactivate_time = _time(confirmed_idx)  # zone becomes inactive when the NEW zone confirms

        if side == "buy":
            if active_buy_idx is not None:
                prev = zones[active_buy_idx]
                zones[active_buy_idx] = replace(
                    prev,
                    end_time=deactivate_time,
                    meta={**prev.meta, "active": False},
                )
            active_buy_idx = len(zones)
        else:
            if active_sell_idx is not None:
                prev = zones[active_sell_idx]
                zones[active_sell_idx] = replace(
                    prev,
                    end_time=deactivate_time,
                    meta={**prev.meta, "active": False},
                )
            active_sell_idx = len(zones)

        zones.append(z)

    # --- Terminal structure end: if reversal occurs, end any still-active zones at the reversal candle ---
    # if "market_state" in dfx.columns and "structure_id" in dfx.columns:
    #     # Find first reversal candle for each structure_id present in the dataframe
    #     rev_mask = (dfx["market_state"].astype(str) == "reversal")
    #     if rev_mask.any():
    #         # We only care about ending zones for the structure(s) that actually reversed in this df
    #         # Compute first reversal idx per structure_id
    #         rev_df = dfx.loc[rev_mask, ["structure_id"]].copy()
    #         rev_df["idx"] = rev_df.index.astype(int)
    #         first_rev_by_sid = rev_df.groupby("structure_id")["idx"].min().to_dict()

    #         # Force-close any zones still active in those structures
    #         for zi, z in enumerate(zones):
    #             sid = (z.meta or {}).get("structure_id", None)
    #             if sid is None:
    #                 continue
    #             if sid not in first_rev_by_sid:
    #                 continue
    #             if z.end_time is not None:
    #                 continue

    #             rev_idx = int(first_rev_by_sid[sid])
    #             rev_time = _time(rev_idx)

    #             zones[zi] = replace(
    #                 z,
    #                 end_time=rev_time,
    #                 meta={**z.meta, "active": False, "ended_reason": "reversal", "ended_idx": rev_idx},
    #             )

    return zones
