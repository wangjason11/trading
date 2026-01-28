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

    # -------------------------
    # 2 Candle Pattern Base Patterns
    # -------------------------
    big0 = dfx["is_big_normal_as0"].astype(int)
    big1 = dfx["is_big_normal_as1"].shift(-1).fillna(0).astype(int)
    dir_flip = dfx["direction"] != dfx["direction"].shift(-1)

    gate = (big0 == 1) & (big1 == 1) & dir_flip

    cur_is_maru_or_normal = dfx["candle_type"].isin(["maru", "normal"])
    next_is_maru_or_normal = dfx["candle_type"].shift(-1).isin(["maru", "normal"])

    # Pinbar mapping: up/down pinbar = candle_type=="pinbar" and pinbar_dir==±1
    cur_up_pinbar = (dfx["candle_type"] == "pinbar") & (dfx["pinbar_dir"] == 1)
    cur_dn_pinbar = (dfx["candle_type"] == "pinbar") & (dfx["pinbar_dir"] == -1)
    next_up_pinbar = (dfx["candle_type"].shift(-1) == "pinbar") & (dfx["pinbar_dir"].shift(-1) == 1)
    next_dn_pinbar = (dfx["candle_type"].shift(-1) == "pinbar") & (dfx["pinbar_dir"].shift(-1) == -1)

    clen = dfx["candle_len"].astype(float)
    next_clen = clen.shift(-1)

    # Corrected typo from you:
    #   prev_clen < length_threshold * clen  => "no base 2nd big"
    #   prev_clen * length_threshold > clen  => "no base 1st big"
    nb2 = clen < (length_threshold * next_clen)
    nb1 = (clen * length_threshold) > next_clen

    base_pattern = np.where(
        gate,
        np.where(
            cur_is_maru_or_normal & next_is_maru_or_normal,
            np.where(nb2, "no base 2nd big", np.where(nb1, "no base 1st big", "no base")),
            np.where(
                cur_up_pinbar & next_up_pinbar,
                "no base long tails up",
                np.where(cur_dn_pinbar & next_dn_pinbar, "no base long tails down", "base"),
            ),
        ),
        "base",
    )

    dfx["base_pattern"] = base_pattern

    # Always compute 2-candle window features (base_idx uses base_idx & base_idx+1 regardless of pattern)
    dfx["base_low"] = np.where(dfx["base_pattern"] != "base", np.minimum(dfx["l"], dfx["l"].shift(-1)), float("nan"))
    dfx["base_high"] = np.where(dfx["base_pattern"] != "base", np.maximum(dfx["h"], dfx["h"].shift(-1)), float("nan"))

    c0 = dfx["c"].astype(float)
    c1 = dfx["c"].shift(-1).astype(float)
    o0 = dfx["o"].astype(float)
    o1 = dfx["o"].shift(-1).astype(float)

    min_co = np.minimum(np.minimum(c0, c1), np.minimum(o0, o1))
    max_co = np.maximum(np.maximum(c0, c1), np.maximum(o0, o1))

    dfx["base_min_close_open"] = np.where(dfx["base_pattern"] != "base", min_co, float("nan"))
    dfx["base_max_close_open"] = np.where(dfx["base_pattern"] != "base", max_co, float("nan"))

    # -------------------------
    # Single Large Pinbar Base Pattern
    # -------------------------
    big_up_pinbar = (dfx["candle_type"] == "pinbar") & (dfx["pinbar_dir"] == 1) & (dfx["is_big_maru_as0"] == 1)
    big_down_pinbar = (dfx["candle_type"] == "pinbar") & (dfx["pinbar_dir"] == -1) & (dfx["is_big_maru_as0"] == 1)

    base_pattern = np.where(
        dfx["base_pattern"] == "base",
        np.where(big_up_pinbar,
            "no base big tail up",
            np.where(big_down_pinbar, "no base big tail down", "base")
        ),
        dfx["base_pattern"],
    )

    dfx["base_pattern"] = base_pattern

    is_big_pinbar = dfx["base_pattern"].isin(["no base big tail up", "no base big tail down"])
    dfx.loc[is_big_pinbar, "base_low"]  = dfx.loc[is_big_pinbar, "l"].astype(float)
    dfx.loc[is_big_pinbar, "base_high"] = dfx.loc[is_big_pinbar, "h"].astype(float)

    # For pinbar, these are “inner candidates” used by threshold logic; keeping them as candle body bounds is reasonable
    pin_o = dfx["o"].astype(float)
    pin_c = dfx["c"].astype(float)
    dfx.loc[is_big_pinbar, "base_min_close_open"] = np.minimum(pin_o, pin_c)[is_big_pinbar]
    dfx.loc[is_big_pinbar, "base_max_close_open"] = np.maximum(pin_o, pin_c)[is_big_pinbar]

    # -------------------------
    # Star Base Pattern
    # -------------------------
    opposite_dir = dfx["direction"] != dfx["direction"].shift(-2)
    star0 = dfx["candle_type"].isin(["maru", "normal"]) & (dfx["is_big_normal_as0"] == 1)
    star1 = dfx["candle_type"].shift(-1).isin(["pinbar"])
    star2 = dfx["candle_type"].shift(-2).isin(["maru", "normal"]) & (dfx["is_big_normal_as2"].shift(-2).fillna(0) == 1)

    star_gate = opposite_dir & star0 & star1 & star2

    clen_star = dfx["candle_len"].astype(float)
    end_clen = clen.shift(-2)
    star_2big = clen_star < (length_threshold * end_clen)
    star_1big = (clen_star * length_threshold) > end_clen

    base_pattern = np.where(
        dfx["base_pattern"] == "base",
        np.where(star_gate,
            np.where(star_2big,
                "no base star 2nd big", 
                np.where(star_1big, "no base star 1st big", "no base star")
            ),
            "base"
        ),
        dfx["base_pattern"],
    )

    dfx["base_pattern"] = base_pattern

    # After you finalize dfx["base_pattern"] for star patterns...
    # Always compute star window features (base_idx uses base_idx & base_idx+2)
    is_star = dfx["base_pattern"].astype(str).str.startswith("no base star")

    # 3-candle window features ONLY for star patterns
    star_low  = np.minimum(dfx["l"].astype(float), dfx["l"].shift(-1).astype(float), dfx["l"].shift(-2).astype(float))
    star_high = np.maximum(dfx["h"].astype(float), dfx["h"].shift(-1).astype(float), dfx["h"].shift(-2).astype(float))

    dfx["base_low"]  = np.where(is_star, star_low,  dfx["base_low"])
    dfx["base_high"] = np.where(is_star, star_high, dfx["base_high"])

    c0 = dfx["c"].astype(float)
    c2 = dfx["c"].shift(-2).astype(float)
    o0 = dfx["o"].astype(float)
    o2 = dfx["o"].shift(-2).astype(float)

    star_min_co = np.minimum(np.minimum(c0, c2), np.minimum(o0, o2))
    star_max_co = np.maximum(np.maximum(c0, c2), np.maximum(o0, o2))

    dfx["base_min_close_open"] = np.where(is_star, star_min_co, dfx["base_min_close_open"])
    dfx["base_max_close_open"] = np.where(is_star, star_max_co, dfx["base_max_close_open"])

    dfx["base_low"] = np.where(dfx["base_pattern"] == "base", dfx["l"], dfx["base_low"])
    dfx["base_high"] = np.where(dfx["base_pattern"] == "base", dfx["h"], dfx["base_high"])

    return dfx


# -------------------------
# Base index resolver (ported)
# -------------------------

def resolve_base_idx_and_pattern(
    df: pd.DataFrame,
    base_source_idx: int,
    struct_direction: int,
    *,
    bos: bool,
) -> tuple[int, str]:
    """
    Port of legacy check_base_pattern for 2 candle patterns

    bos=True (BOS-confirmed event):
      if struct_direction == direction[idx] => (idx-1) else idx

    bos=False (CTS-confirmed event):
      if struct_direction == direction[idx] => idx else (idx-1)

    Adds additional 1 large pinbar pattern & star patterns (3 candles)

    Clamp so base_idx+2 is valid for 3-candle window usage.
    """

    i = int(base_source_idx)
    sd = int(struct_direction)
    conf_dir = int(df.loc[i, "direction"])
    candle_type = df.loc[i, "candle_type"]
    base_pattern = df.loc[i, "base_pattern"]
    prev_base_pattern = df.loc[i-1, "base_pattern"]
    prev_base_pattern_dir = int(df.loc[i-1, "direction"])

    base_idx = i  # default

    # Check for single large pinbar
    if base_pattern in ["no base big tail up", "no base big tail down"]:
        if bos and struct_direction == 1 and base_pattern == "no base big tail up":
            base_idx = i
        elif bos and struct_direction == -1 and base_pattern == "no base big tail down":
            base_idx = i
        elif not bos and struct_direction == 1 and base_pattern == "no base big tail down":
            base_idx = i
        elif not bos and struct_direction == -1 and base_pattern == "no base big tail up":
            base_idx = i

    # Check for star pattern
    elif prev_base_pattern in ["no base star 2nd big", "no base star 1st big", "no base star"]:
        if bos and prev_base_pattern_dir != sd:
            base_idx = i - 1
        elif not bos and prev_base_pattern_dir == sd:
            base_idx = i - 1

    # Check for remaining 2-candle patterns
    elif bos:
        if sd == conf_dir and prev_base_pattern != "base":
            base_idx = i - 1
        elif sd != conf_dir and base_pattern != "base":
            base_idx = i
        # base_idx = (i - 1) if (sd == conf_dir) else i
    elif not bos:
        if sd == conf_dir and base_pattern != "base":
            base_idx = i
        elif sd != conf_dir and prev_base_pattern != "base":
            base_idx = i - 1
        # base_idx = i if (sd == conf_dir) else (i - 1)
    # else:
    #     base_idx = i

    # Ensure base_idx allows base_idx+2 access
    base_idx = max(0, min(base_idx, len(df) - 3))
    zone_pattern = str(df.loc[base_idx, "base_pattern"])
    return base_idx, zone_pattern

# -------------------------
# find_base_threshold (ported intent, fixed)
# -------------------------

def find_base_threshold(df: pd.DataFrame, idx: int, struct_direction: int, *, bos: bool = True) -> float:
    left = max(0, int(idx) - 5)
    right = min(len(df), int(idx) + 6)

    neighbor_df = df.iloc[left:idx].copy()
    neighbor_df = pd.concat([neighbor_df, df.iloc[idx + 1:right]], axis=0)

    if neighbor_df.empty:
        return float("nan")

    candidates_desc = sorted(set(np.minimum(neighbor_df["o"], neighbor_df["c"])), reverse=True)
    candidates_asc = sorted(set(np.maximum(neighbor_df["o"], neighbor_df["c"])), reverse=False)

    sd = int(struct_direction)
    result = None

    if bos:
        levels = candidates_asc if sd == 1 else candidates_desc
        for level in levels:
            if sd == 1:
                count = ((neighbor_df["o"] <= level) & (neighbor_df["c"] <= level)).sum()
            else:
                count = ((neighbor_df["o"] >= level) & (neighbor_df["c"] >= level)).sum()
            if count >= 1:
                result = float(level)
            if count >= 2:
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
    """
    n = len(df)
    i = int(base_idx)
    if n == 0:
        return float("nan")

    sd = int(struct_direction)

    # Select reference extreme for "outer"
    use_low_ref = (bos and sd == 1) or ((not bos) and sd == -1)
    ref = float(df.loc[i, "l"] if use_low_ref else df.loc[i, "h"])

    # Need neighbors; if missing, fall back to this candle's body point closest to ref
    if i - 1 < 0 or i + 1 >= n:
        o = float(df.loc[i, "o"])
        c = float(df.loc[i, "c"])
        return o if abs(o - ref) <= abs(c - ref) else c

    candidates = [
        float(df.loc[i - 1, "o"]),
        float(df.loc[i - 1, "c"]),
        float(df.loc[i + 1, "o"]),
        float(df.loc[i + 1, "c"]),
    ]

    inner = min(candidates, key=lambda x: abs(x - ref))
    return float(inner)


# -------------------------
# zone_thresholds (ported mapping)
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
    """
    sd = int(struct_direction)

    base_low = float(df.loc[base_idx, "base_low"])
    base_high = float(df.loc[base_idx, "base_high"])

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
        inner = float(df.loc[base_idx + 2, "c"])
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
    sid = -1  # NEW: current event's structure_id (refreshed per event from ev.meta)

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
        if ev.type not in ("BOS_CONFIRMED", "CTS_CONFIRMED", "CTS_THRESHOLD_UPDATED", "BOS_THRESHOLD_UPDATED"):
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
                "struct_direction": sd,
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

                "bounds_steps": [
                    {
                        "start_idx": int(base_idx),   # segment begins at base anchor candle
                        "top": float(top),
                        "bottom": float(bottom),
                        "event": "INIT",
                    }
                ],

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
    if "market_state" in dfx.columns and "structure_id" in dfx.columns:
        rev_mask = (dfx["market_state"].astype(str).str.lower() == "reversal")
        if rev_mask.any():
            rev_df = dfx.loc[rev_mask, ["structure_id"]].copy()
            rev_df["idx"] = rev_df.index.astype(int)
            first_rev_by_sid = rev_df.groupby("structure_id")["idx"].min().to_dict()

            for zi, z in enumerate(zones):
                sid = (z.meta or {}).get("structure_id", None)
                if sid is None or sid not in first_rev_by_sid:
                    continue
                if z.end_time is not None:
                    continue

                rev_idx = int(first_rev_by_sid[sid])
                zones[zi] = replace(
                    z,
                    end_time=_time(rev_idx),
                    meta={**(z.meta or {}), "active": False},
                )


    return zones
