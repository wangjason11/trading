"""PLAN_I session 2 — PROTOTYPE of the cross-fib LIMIT reference price P (§1.2), copied into the variant trees by
`variant_patch_i.py` as `engine_v2/multitf/cross_fib_limit.py`. NOT imported by the repo's own code.

P by the structure's starting trigger (PLAN_I §1.2 / §1.4; decisions R0, R0a, Q2, Q3, R2-1, R2-2, R2-4, R3-5,
R4-5, R4-6, R5-3):

- main sid 0 (start trading) -> none; main sid N >= 1 (reversal) -> the APPLIED reversal's `bos_frozen`
  (`STATE_CHANGED(to=reversal)` of sid N-1).
- sub reversal -> the reversing sub's applied `bos_frozen` (its geometry's `STATE_CHANGED(to=reversal)` at R).
- sub first_confluence on parent cycle 0 -> none; on c >= 1 -> the parent's CTS_ESTABLISHED (S, c)
  `meta["cts_frozen"]` (the frozen c-1 `cts_threshold` the breakout broke; stamped by the V2 patch).
- sub first_counter / subsequent_* -> the previous unique sub x-1 read at x's record start s, INCLUSIVE, by
  MOMENT (knowable-at), its post-end events excluded (read until min(s, x-1's end)); same direction -> the
  `cts_threshold` of its most recent CONFIRMED cycle (CTS_CONFIRMED price, moved by that cycle's
  CTS_THRESHOLD_UPDATED); else its latest CTS anchor price (R2-b); opposite -> its `bos_threshold` (latest
  BOS_CONFIRMED / BOS_THRESHOLD_UPDATED price); else its starting candle's extreme (R2-2).

x-1 at the FIRE of x's first trigger (when the geometry MS runs): the sub with the largest (known start, creation
index) among the subs whose known start <= s (known start = `sub.start_idx`, else the min `start_idx` of its
non-zero-length records — final once created). == the start-numbered x-1 whenever x's record start == its fire
(PLAN_I §9: every non-lagged trigger); the lag cases are logged.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from engine_v2.multitf.sub_structure_pool import knowable_at_idx

LOG: List[dict] = []


def beyond(price: float, ref: float, sd: int) -> bool:
    """Strictly past `ref` in the structure's direction (Q2 / N-b)."""
    return price > ref if int(sd) == 1 else price < ref


def _limit(price, source, started_by=None, from_sub=None) -> Dict[str, Any]:
    return {"price": (None if price is None else float(price)), "source": source,
            "started_by": started_by, "from_sub": from_sub}


def main_limits(events) -> Dict[int, Dict[str, Any]]:
    sids = sorted({int(e.meta["structure_id"]) for e in events if e.meta.get("structure_id") is not None})
    frozen: Dict[int, float] = {}
    for e in events:
        if e.type == "STATE_CHANGED" and e.meta.get("to") == "reversal":
            s = int(e.meta["structure_id"])
            assert s not in frozen, f"two reversals for sid {s}"
            frozen[s] = float(e.meta["bos_frozen"])
    out = {}
    for s in sids:
        out[s] = (_limit(None, "none", "start_trading") if s == 0
                  else _limit(frozen[s - 1], "reversal_bos_frozen", "reversal"))
    LOG.append({"kind": "main_limits", "limits": {str(k): {kk: vv for kk, vv in v.items() if kk != "from_sub"}
                                                  for k, v in out.items()}})
    return out


def _known_start(sub) -> Optional[int]:
    if sub.start_idx is not None:
        return int(sub.start_idx)
    starts = [int(r.start_idx) for r in sub.records if not r.is_zero_length]
    return min(starts) if starts else None


def prev_sub(pool, s: int):
    best, best_key = None, None
    for sub in pool.all():
        ks = _known_start(sub)
        if ks is None or ks > int(s):
            continue
        k = (ks, int(sub.sub_id))
        if best_key is None or k > best_key:
            best, best_key = sub, k
    return best


def _abs_events_until(sub, until: int):
    bounded, sb = sub.geometry
    return [e for e in bounded.events if knowable_at_idx(e) + int(sb) <= int(until)], bounded, int(sb)


def read_prev(prev, s: int, x_dir: int) -> Dict[str, Any]:
    until = int(s) if (prev.end_idx is None or int(prev.end_idx) >= int(s)) else int(prev.end_idx)
    evs, bounded, sb = _abs_events_until(prev, until)
    same = int(prev.direction) == int(x_dir)
    col = None
    loc = until - sb
    if 0 <= loc < len(bounded.df):
        cname = "cts_threshold" if same else "bos_threshold"
        if cname in bounded.df.columns:
            v = bounded.df[cname].iloc[loc]
            col = None if v != v else float(v)
    if same:
        conf_cycle, thr = None, None
        for e in evs:
            if e.type == "CTS_CONFIRMED":
                conf_cycle, thr = int(e.meta["cycle_id"]), float(e.price)
            elif (e.type == "CTS_THRESHOLD_UPDATED" and conf_cycle is not None
                  and int(e.meta["cycle_id"]) == conf_cycle):
                thr = float(e.price)
        if thr is not None:
            out = _limit(thr, "prev_sub_cts_threshold", from_sub=prev)
        else:
            latest = None
            for e in evs:
                if e.type in ("CTS_ESTABLISHED", "CTS_UPDATED", "CTS_CONFIRMED"):
                    latest = float(e.price)
            out = (_limit(latest, "prev_sub_latest_cts", from_sub=prev) if latest is not None
                   else _limit(None, "none", from_sub=prev))
    else:
        thr = None
        for e in evs:
            if e.type in ("BOS_CONFIRMED", "BOS_THRESHOLD_UPDATED"):
                thr = float(e.price)
        if thr is not None:
            out = _limit(thr, "prev_sub_bos_threshold", from_sub=prev)
        else:
            sl = int(prev.starting_idx) - sb
            p = bounded.df["h"].iloc[sl] if int(prev.direction) == -1 else bounded.df["l"].iloc[sl]
            out = _limit(float(p), "prev_sub_starting_price", from_sub=prev)
    LOG.append({"kind": "read_prev", "prev_sub_id": int(prev.sub_id), "prev_dir": int(prev.direction),
                "x_dir": int(x_dir), "s": int(s), "until": until, "price": out["price"], "source": out["source"],
                "df_column_at_until": col})
    return out


def sub_limit(sweep, t, start_idx: int, reversing_sub, h1_events) -> Dict[str, Any]:
    tt = t.trigger_type
    if tt == "reversal":
        bounded, sb = reversing_sub.geometry
        sc = [e for e in bounded.events if e.type == "STATE_CHANGED" and e.meta.get("to") == "reversal"]
        assert len(sc) == 1, f"reversing sub {reversing_sub.sub_id}: {len(sc)} reversals"
        assert int(sc[0].idx) + int(sb) == int(reversing_sub.natural_reversal_idx)
        out = _limit(float(sc[0].meta["bos_frozen"]), "reversal_bos_frozen", from_sub=reversing_sub)
    elif tt == "first_confluence":
        S, C = t.cycle
        if int(C) == 0:
            out = _limit(None, "none")
        else:
            est = [e for e in h1_events if e.type == "CTS_ESTABLISHED"
                   and int(e.meta["structure_id"]) == int(S) and int(e.meta["cycle_id"]) == int(C)]
            assert len(est) == 1, (S, C, len(est))
            v = est[0].meta["cts_frozen"]
            assert v is not None, (S, C)
            out = _limit(float(v), "parent_cts_threshold")
    else:
        prev = prev_sub(sweep.pool, int(start_idx))
        out = _limit(None, "none") if prev is None else read_prev(prev, int(start_idx), int(t.direction))
    LOG.append({"kind": "sub_limit", "trigger": tt, "lens": t.lens, "parent": list(t.cycle),
                "trigger_idx": int(t.trigger_idx), "start": int(start_idx), "dir": int(t.direction),
                "price": out["price"], "source": out["source"],
                "from_sub": None if out["from_sub"] is None else int(out["from_sub"].sub_id),
                "lagged": int(start_idx) != int(t.trigger_idx)})
    return out
