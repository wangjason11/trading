"""The cross-fib LIMIT reference price `P` of a unique sub (PLAN_I §1.2 / §9.3 D2) — the rule by the sub's
STARTING trigger, computed ONCE (R2-4).

`P` is computed when the sub's MS geometry is built — at its creating trigger's FIRE, from that trigger at its
record's start (R6-2 (a)); the orchestrator re-derives it after the sweep from the sub's started_by record at the
sub's `start_idx` with the start-numbered x−1 and WARNING-logs a mismatch (`check_cross_fib_limits`, the R6-2
tripwire; the stored `P` stays). Both go through `cross_fib_limit_for` — one formula.

The rule (§1.2):
- `reversal` → the REVERSING sub's applied `bos_frozen` (its geometry's `STATE_CHANGED(to=reversal)`, whose idx
  + `slice_begin` is the reversing sub's `natural_reversal_idx` — asserted; never None);
- `first_confluence` on parent cycle 0 → none; on c ≥ 1 → the parent's `CTS_ESTABLISHED` (S, c)
  `meta["cts_frozen"]` (the frozen c−1 `cts_threshold` the breakout broke; never None);
- `first_counter` / `subsequent_confluence` / `subsequent_counter` → the PREVIOUS unique sub x−1 read at x's start
  `s` (`read_prev_sub`); no x−1 → none.

Pure: no pandas import (the geometry's `bounded.df` is read through `.iloc`), no MS / chart import.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional, Tuple

from engine_v2.multitf.sub_structure_pool import PooledStructure, knowable_at_idx
from engine_v2.structure import event_fields as ef
from engine_v2.zones.cross_cycle_fib import CrossFibLimit

# x−1 types (§1.2): the previous unique sub supplies P.
PREV_SUB_TYPES = frozenset({"first_counter", "subsequent_confluence", "subsequent_counter"})

NO_LIMIT = CrossFibLimit(price=None, source="none")


def known_start(sub: PooledStructure) -> Optional[int]:
    """A sub's start as known DURING the sweep: `start_idx` once phase 2 set it, else the earliest `start_idx`
    of its non-zero-length records (it can still move or vanish later — R6-2's cases; the tripwire catches
    them); None when it has none."""
    if sub.start_idx is not None:
        return int(sub.start_idx)
    starts = [int(r.start_idx) for r in sub.records if not r.is_zero_length]
    return min(starts) if starts else None


def prev_sub_at_fire(subs: Iterable[PooledStructure], s: int) -> Optional[PooledStructure]:
    """x−1 at the fire (R6-2 (a)): the sub with the largest `(known start, pool_idx)` among the subs whose known
    start ≤ `s` — the start-numbered x−1 whenever nothing lags (§9.2 F6)."""
    best, best_key = None, None
    for sub in subs:
        ks = known_start(sub)
        if ks is None or ks > int(s):
            continue
        k = (ks, int(sub.pool_idx))
        if best_key is None or k > best_key:
            best, best_key = sub, k
    return best


def _events_by_moment_until(sub: PooledStructure, until: int) -> List[Any]:
    """The sub's geometry events whose MOMENT (`knowable_at_idx` + `slice_begin`, entity-absolute) is ≤ `until`
    (inclusive, R4-5), in moment order — at an equal moment the LATER-EMITTED one last (a stable sort)."""
    bounded, slice_begin = sub.geometry
    sb = int(slice_begin)
    keyed = [(knowable_at_idx(e) + sb, e) for e in bounded.events]
    keyed = [(m, e) for m, e in keyed if m <= int(until)]
    keyed.sort(key=lambda me: me[0])
    return [e for _m, e in keyed]


def read_prev_sub(prev: PooledStructure, s: int, x_direction: int,
                  *, log: Callable[[str], None] = print) -> CrossFibLimit:
    """x−1's reference at x's start `s` (R2-1, R2-2, R3-5, R4-5, R4-6, R5-3, R6-3 (vi)).

    Read by MOMENT up to `min(s, x−1.end_idx)` when x−1's end is known — inclusive; its post-end events never
    count (its geometry runs on to the data edge), so an x−1 ended by a reversal / a new parent cycle yields its
    frozen thresholds. Same direction → the `cts_threshold` of its most recent CONFIRMED cycle (that cycle's
    `CTS_CONFIRMED` price, moved by its `CTS_THRESHOLD_UPDATED`), else its latest CTS anchor price (EST / UPD /
    CONF), else no P (`prev_sub_no_cts`, a WARNING); opposite → its latest `BOS_CONFIRMED` /
    `BOS_THRESHOLD_UPDATED` price, else its starting candle's extreme (the high of a sell sub, the low of a buy
    sub = its BOS_0 price).

    The anchor behind `P` (`ref_anchor_price`, PLAN_K §3.3, RX1-2), read in the SAME read: the confirmed cycle's
    `CTS_CONFIRMED` price (before its `CTS_THRESHOLD_UPDATED`s); `P` itself for the latest CTS anchor and the
    starting extreme (already anchor prices); the latest `BOS_CONFIRMED` price — the BOS whose threshold `P` is —
    for the BOS threshold."""
    until = int(s) if (prev.end_idx is None or int(prev.end_idx) >= int(s)) else int(prev.end_idx)
    evs = _events_by_moment_until(prev, until)
    from_pool = int(prev.pool_idx)
    if int(prev.direction) == int(x_direction):
        conf_cycle, thr, conf_price = None, None, None
        for e in evs:
            if e.type == "CTS_CONFIRMED":
                conf_cycle, thr, conf_price = int(e.meta["cycle_id"]), float(e.price), float(e.price)
            elif (e.type == "CTS_THRESHOLD_UPDATED" and conf_cycle is not None
                  and int(e.meta["cycle_id"]) == conf_cycle):
                thr = float(e.price)
        if thr is not None:
            return CrossFibLimit(thr, "prev_sub_cts_threshold", from_pool_idx=from_pool, ref_anchor_price=conf_price)
        latest = None
        for e in evs:
            if e.type in ("CTS_ESTABLISHED", "CTS_UPDATED", "CTS_CONFIRMED"):
                latest = float(e.price)
        if latest is not None:
            return CrossFibLimit(latest, "prev_sub_latest_cts", from_pool_idx=from_pool, ref_anchor_price=latest)
        log(f"WARNING [cross_fib_limit] x-1 pool#{from_pool} (same direction {int(x_direction)}) has no CTS "
            f"anchor by {until} -> no P (prev_sub_no_cts)")
        return CrossFibLimit(None, "prev_sub_no_cts", from_pool_idx=from_pool)
    thr, bos_price = None, None
    for e in evs:
        if e.type in ("BOS_CONFIRMED", "BOS_THRESHOLD_UPDATED"):
            thr = float(e.price)
        if e.type == "BOS_CONFIRMED":
            bos_price = float(e.price)
    if thr is not None:
        assert bos_price is not None, f"[cross_fib_limit] x-1 pool#{from_pool}: a BOS threshold with no BOS_CONFIRMED"
        return CrossFibLimit(thr, "prev_sub_bos_threshold", from_pool_idx=from_pool, ref_anchor_price=bos_price)
    bounded, slice_begin = prev.geometry
    loc = int(prev.starting_idx) - int(slice_begin)
    col = "h" if int(prev.direction) == -1 else "l"
    start_price = float(bounded.df[col].iloc[loc])
    return CrossFibLimit(start_price, "prev_sub_starting_price", from_pool_idx=from_pool, ref_anchor_price=start_price)


def reversal_limit(reversing_sub: PooledStructure) -> CrossFibLimit:
    """A `reversal`-started sub: the REVERSING sub's applied `bos_frozen` (§1.2) — its geometry's one
    `STATE_CHANGED(to=reversal)`, which must sit at its `natural_reversal_idx`. + the IC reference's inputs
    (PLAN_K A10): `ref_anchor_price` = the reversing sub's most recent `BOS_CONFIRMED` price by the reversal's
    moment, and the break = the REALISED reversal pattern (`ef.pattern_anchor_idx` on that `STATE_CHANGED`, and its
    idx), ENTITY-ABSOLUTE (+ the reversing geometry's `slice_begin`) — `CrossFibLimit.localized` moves it into a
    sub's slice at each entry point (PLAN_K §3.5)."""
    bounded, slice_begin = reversing_sub.geometry
    sc = [e for e in bounded.events if e.type == "STATE_CHANGED" and e.meta.get("to") == "reversal"]
    assert len(sc) == 1, f"[cross_fib_limit] reversing pool#{reversing_sub.pool_idx}: {len(sc)} reversals"
    assert int(sc[0].idx) + int(slice_begin) == int(reversing_sub.natural_reversal_idx), (
        f"[cross_fib_limit] reversing pool#{reversing_sub.pool_idx}: reversal at "
        f"{int(sc[0].idx) + int(slice_begin)} != natural_reversal_idx {reversing_sub.natural_reversal_idx}")
    frozen = sc[0].meta["bos_frozen"]
    assert frozen is not None, f"[cross_fib_limit] reversing pool#{reversing_sub.pool_idx}: bos_frozen None"
    boses = [e for e in bounded.events if e.type == "BOS_CONFIRMED" and ef.event_moment(e) <= int(sc[0].idx)]
    assert boses, f"[cross_fib_limit] reversing pool#{reversing_sub.pool_idx}: no BOS_CONFIRMED before its reversal"
    last_bos = max(boses, key=ef.event_moment)
    sb = int(slice_begin)
    return CrossFibLimit(float(frozen), "reversal_bos_frozen", from_pool_idx=int(reversing_sub.pool_idx),
                         ref_anchor_price=float(last_bos.price),
                         break_pattern_anchor_idx=ef.pattern_anchor_idx(sc[0]) + sb,
                         break_confirmed_idx=int(sc[0].idx) + sb)


def parent_cts_limit(h1_events: Iterable[Any], parent_sid: int, parent_cycle_id: int) -> CrossFibLimit:
    """A `first_confluence` on parent cycle c >= 1: the parent's `CTS_ESTABLISHED` (S, c) `cts_frozen` (Q3) —
    cycle c−1's frozen `cts_threshold` the breakout broke; the anchor behind it (`ref_anchor_price`, PLAN_K §3.3) =
    the parent cycle c−1's `CTS_CONFIRMED` price."""
    h1_events = list(h1_events)
    est = [e for e in h1_events if e.type == "CTS_ESTABLISHED"
           and int(e.meta["structure_id"]) == int(parent_sid) and int(e.meta["cycle_id"]) == int(parent_cycle_id)]
    assert len(est) == 1, f"[cross_fib_limit] parent ({parent_sid},{parent_cycle_id}): {len(est)} CTS_ESTABLISHED"
    frozen = est[0].meta["cts_frozen"]
    assert frozen is not None, f"[cross_fib_limit] parent ({parent_sid},{parent_cycle_id}): cts_frozen None"
    conf = [e for e in h1_events if e.type == "CTS_CONFIRMED"
            and int(e.meta["structure_id"]) == int(parent_sid)
            and int(e.meta["cycle_id"]) == int(parent_cycle_id) - 1]
    assert len(conf) == 1, (
        f"[cross_fib_limit] parent ({parent_sid},{int(parent_cycle_id) - 1}): {len(conf)} CTS_CONFIRMED")
    return CrossFibLimit(float(frozen), "parent_cts_threshold", ref_anchor_price=float(conf[0].price))


def cross_fib_limit_for(
    trigger_type: str,
    parent_cycle: Tuple[int, int],
    direction: int,
    s: int,
    *,
    h1_events: Iterable[Any],
    reversing_sub: Optional[PooledStructure],
    prev_sub: Optional[PooledStructure],
    log: Callable[[str], None] = print,
) -> CrossFibLimit:
    """THE rule (§1.2) for a unique sub started by `trigger_type` at `s` (its start): the one formula behind the
    fire-time `P` (the sweep) and the post-sweep re-derivation (`check_cross_fib_limits`) — they differ only in
    which `reversing_sub` / `prev_sub` they hand in."""
    if trigger_type == "reversal":
        assert reversing_sub is not None, "[cross_fib_limit] a reversal needs its reversing sub"
        return reversal_limit(reversing_sub)
    if trigger_type == "first_confluence":
        S, C = parent_cycle
        return NO_LIMIT if int(C) == 0 else parent_cts_limit(h1_events, int(S), int(C))
    assert trigger_type in PREV_SUB_TYPES, f"[cross_fib_limit] unknown trigger type {trigger_type!r}"
    return NO_LIMIT if prev_sub is None else read_prev_sub(prev_sub, int(s), int(direction), log=log)


def check_cross_fib_limits(pool: Any, h1_events: Iterable[Any], *, log: Callable[[str], None] = print) -> int:
    """The R6-2 tripwire (PLAN_I §9.3 D2), after the sweep has numbered the subs: for every started sub whose
    started_by record (the first live one by `(start_idx, seq)`) is not its creating record (the first by `seq`),
    or whose type reads x−1, re-derive `P` the §1.2 way — from the started_by record, at the sub's `start_idx`,
    x−1 = the sub numbered `sub_id − 1` (a reversal's reversing sub = THE sub whose `natural_reversal_idx` is the
    record's `trigger_idx`, of the opposite direction, with a record live at it) — and compare `(price, source,
    from_sub_id)` with the stored one. A mismatch is a `WARNING [cross_fib_limit]` and the stored `P` stays
    (R6-2 (a)). Returns the mismatch count."""
    h1_events = list(h1_events)
    by_id = {s.sub_id: s for s in pool.all() if s.sub_id is not None}
    checked = mismatched = 0
    for sub_id in sorted(by_id):
        sub = by_id[sub_id]
        first = min(sub.live_records(), key=lambda r: (r.start_idx, r.seq))
        creating = min(sub.records, key=lambda r: r.seq)
        if first is creating and first.trigger_type not in PREV_SUB_TYPES:
            continue
        checked += 1
        reversing = None
        if first.trigger_type == "reversal":
            R = int(first.trigger_idx)
            cands = [o for o in pool.all()
                     if o.natural_reversal_idx is not None and int(o.natural_reversal_idx) == R
                     and int(o.direction) == -int(sub.direction)
                     and any(r.is_live_at_reversal(R) for r in o.records)]
            if len(cands) != 1:
                # A warn-only tripwire never stops the replay (R6-2 (a)): two subs reversing at R with live records
                # leave the reversing sub ambiguous — say so and skip the comparison.
                mismatched += 1
                log(f"WARNING [cross_fib_limit] sub {sub_id}: {len(cands)} candidate reversing subs at {R} — "
                    f"its P cannot be re-derived (PLAN_I R6-2)")
                continue
            reversing = cands[0]
        prev = by_id.get(sub_id - 1) if first.trigger_type in PREV_SUB_TYPES else None
        got = cross_fib_limit_for(
            first.trigger_type, (first.parent_sid, first.parent_cycle_id), int(sub.direction), int(sub.start_idx),
            h1_events=h1_events, reversing_sub=reversing, prev_sub=prev,
            log=lambda _m: None,          # a `prev_sub_no_cts` WARNING was already logged at the fire
        )
        got_from = None if got.from_pool_idx is None else pool.get_by_pool_idx(got.from_pool_idx).sub_id
        stored = sub.cross_fib_limit
        if ((got.price, got.source, got_from, got.ref_anchor_price, got.break_pattern_anchor_idx,
             got.break_confirmed_idx)
                != (stored.price, stored.source, stored.from_sub_id, stored.ref_anchor_price,
                    stored.break_pattern_anchor_idx, stored.break_confirmed_idx)):
            mismatched += 1
            log(f"WARNING [cross_fib_limit] sub {sub_id}: stored P={stored.price} {stored.source} "
                f"from sub {stored.from_sub_id} != re-derived at its start {sub.start_idx} from its started_by "
                f"record {first.identity} ({first.trigger_type}): P={got.price} {got.source} from sub {got_from} "
                f"— keeping the stored P (PLAN_I R6-2)")
    log(f"[cross_fib_limit] tripwire: {checked} subs re-derived at their start, {mismatched} mismatches")
    return mismatched
