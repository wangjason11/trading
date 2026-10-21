"""The proximity overlay — the parent cycles' proximity scans driven INSIDE the lifecycle sweep (PLAN_R step 2, §5).

The overlay is circular: opp_sd and sd triggers read sub zones, and subs come from triggers. So, per NON-collapsed parent
cycle, the SAME `zones.zone_proximity.CycleProximityScan` the batch scan uses is driven tick by tick from the sweep —
one `proximity_tick` moment per H1 candle `h` of the cycle's scan window, at M15 `LOH(h − 1)`, in a phase after
`SUB_END` (`lifecycle_sweep._PHASE_PROXIMITY`): every pool moment ≤ `LOH(h − 1)` is processed — "known before the H1
candle opens". A tick (`ProximityOverlay.on_tick`):

  1. `close(h − 1)` — candle h − 1's pass test, every outer as of `LOH(h − 1)`;
  2. refreshes each side's PROXIMITY SOURCE SUB and its overlay rungs from the pool (PLAN_R §1, below);
  3. `step(h)` on candle h's OHLC (price data, not engine state);
  4. on a trigger, the `subsequent_*` detectors run on the cycle's list (they read only list[i−1], list[i−2] and the
     H1 OHLC in [prior.idx, h] — causal) and each new trigger is returned as a `SweepTrigger` at `LOH(h)`, phase FIRE —
     after this tick, so never queued in the past. An H1 hour with no M15 bar (`LOH(h) == LOH(h − 1)`) still steps
     (the H1 list matches the batch) but pushes nothing: it is returned UNRESOLVED (`no_m15_hour`).

The proximity source sub of a side (PLAN_R §1 "Which sub feeds a side", the user round 9: "A hold"): STICKY. It starts
empty. Candidates = the unique subs of that side's direction (sd side: the parent's; opp_sd: the counter direction)
with a NON-zero-length record under this parent cycle, either lens, started by `LOH(h − 1)`. A sub CONTRIBUTES its
most recent cycle whose BOS zone is usable (STRICT — passing that cycle's zones never brings an older cycle back): its
BOS zone and its POI (usable from its cycle's CTS_CONFIRMED); on the sd side only zones strictly beyond the parent POI
inner (none when the parent has no POI rung). At each tick the source moves to the NEWEST candidate (a later unique-sub
`start_idx`, ties → the higher pool idx) that is newer than the current one and contributes a REACHABLE zone — one on
the parent's side of the side's last rung (sd: inside the parent BOS; opp_sd: inside the parent CTS), so it could become
the reference (the user, 2026-10-20: "Contributed + reachable" — a newer sub that contributes nothing reachable must
not drop the side to the parent's last rung: the hold). It never moves back, never because its sub ended or was
replaced; an ENDED source is FROZEN: only zones whose moment ≤ its end, outer steps up to its end.

A zone is USABLE once its moment ≤ `LOH(h − 1)` (and the source's record has started): a BOS zone from its cycle's
BOS_CONFIRMED event moment, a POI from its cycle's CTS_CONFIRMED event moment — gated on the EVENTS, never on a zone's
`confirmed_idx` or a POI's presence. Each zone is read AS OF ITS OWN USABLE MOMENT (`build_sub_zone_view`: the same
`project_to_window`, capped there) — causal by construction; the natural-end view would hold cycles formed after a sub
ended, and a POI pick is not mechanism-guaranteed final.

The opp_sd lock (PLAN_R §1 Locking, Q2 = (b), Q3 = (ii)): a lock on the parent CTS is lifted when the opp_sd overlay
presents a zone NEWER than the passed ones — a newer source sub, or the source's newer cycle: the lock records the
PASSED reference's own (sub, cycle), and any other overlay basis lifts it (bases only move forward). The sd lock
stays.

Reads only `pool.records_for` (both lenses), non-zero-length records started by `LOH(h − 1)`, and PROCESSED `end_idx`
values (set at their own moment) — never the sweep's `_active` (it drops the ended records the hold needs) nor
`natural_reversal_idx` (future-known from the sub's creation).
"""
from __future__ import annotations

import contextlib
import io
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from engine_v2.multitf.sub_structure_pool import LENS_CONFLUENCE, LENS_COUNTER, PooledStructure, SubStructurePool
from engine_v2.structure import event_fields as ef
from engine_v2.zones.zone_proximity import CycleProximityScan, ProximityRung, ZoneProximityTrigger

Cycle = Tuple[int, int]
SIDES = ("sd", "opp_sd")
# A capped view projection re-runs the downstream pipeline; its stdout is captured and only lines the post-replay
# grep would flag are re-printed (prefixed), so the view never hides one and never floods run.log.
# = the post-replay grep pattern (the `/compare` skill §2b, WORKFLOWS "Post-replay log grep") — keep them equal.
_VIEW_LOG_KEEP = re.compile(r"error|warning|skipping|unavailable|degenerate|collapsed|pending|no sid|retry",
                            re.IGNORECASE)


@dataclass(frozen=True)
class SubZone:
    """One zone of a pooled sub as the overlay reads it: its inner AS OF its own usable moment (PLAN_R §5.2)."""
    cycle_id: int
    kind: str                                    # "BOS" | "POI"
    inner: float
    usable_at: int                               # M15, entity-absolute: the cycle's BOS_CONFIRMED / CTS_CONFIRMED moment
    outer_steps: Tuple[Tuple[int, float], ...]   # (M15 moment, outer), ascending — a POI: ((usable_at, outer),)

    def outer_at(self, m15_idx: int) -> float:
        """The outer as of M15 `m15_idx` (each expansion counts from its moment)."""
        outer = self.outer_steps[0][1]
        for moment, value in self.outer_steps:
            if moment <= m15_idx:
                outer = value
        return outer


def build_sub_zone_view(
    sub: PooledStructure,
    project: Callable[[PooledStructure, Optional[int]], Dict[str, Any]],
    log: Callable[[str], None] = print,
) -> List[SubZone]:
    """Every BOS zone / POI of `sub`, each read AS OF ITS OWN USABLE MOMENT (PLAN_R §5.2).

    Per cycle of the sub's geometry (its events run to the data edge): the BOS zone from `project(sub, BOS_CONFIRMED
    moment)`, the POI from `project(sub, CTS_CONFIRMED moment)` (`project(sub, cap)` = `project_to_window` capped at the
    entity-absolute M15 `cap`; None = the natural end). A cycle whose capped projection holds no such zone has none.
    A BOS zone's outer EXPANDS after its moment: its steps come from the natural-end projection, each counted from its
    moment (`expanded_idx`) at use — inners never move, which is asserted against the as-of-moment inner.
    """
    bounded, slice_begin = sub.geometry
    sb = int(slice_begin)
    moments: Dict[Tuple[int, str], int] = {}
    for ev in bounded.events:
        if ev.type not in ("BOS_CONFIRMED", "CTS_CONFIRMED"):
            continue
        key = (int(ev.meta.get("cycle_id", 0)), ev.type)
        m = int(ef.event_moment(ev)) + sb
        assert moments.get(key, m) == m, (
            f"[prox_overlay] pool#{sub.pool_idx} cycle {key[0]}: two {key[1]} moments {moments[key]} / {m}"
        )
        moments[key] = m
    if not moments:
        return []
    latest = _quiet_project(project, sub, None, log)
    latest_bos = {int(z.meta["cycle_id"]): z for z in latest["kl_zones"] if z.source_kind == "BOS"}
    zones: List[SubZone] = []
    for (cycle_id, ev_type), m in sorted(moments.items(), key=lambda kv: (kv[1], kv[0])):
        down = _quiet_project(project, sub, m, log)
        if ev_type == "BOS_CONFIRMED":
            kl = [z for z in down["kl_zones"] if z.source_kind == "BOS" and int(z.meta["cycle_id"]) == cycle_id]
            # every BOS_CONFIRMED derives its BOS zone (kl_zones_v1): none = the zone's cycle and the event's disagree
            assert len(kl) == 1, f"[prox_overlay] pool#{sub.pool_idx} c{cycle_id}: {len(kl)} BOS zones at {m}"
            inner = float(kl[0].meta["inner"])
            full = latest_bos.get(cycle_id)
            assert full is not None and float(full.meta["inner"]) == inner, (
                f"[prox_overlay] pool#{sub.pool_idx} c{cycle_id}: the BOS zone's inner moved after its moment {m} "
                f"({inner} -> {None if full is None else full.meta['inner']}) — PLAN_R §1 relies on inners never moving"
            )
            edge = "bottom" if full.side == "buy" else "top"
            steps = full.meta.get("bounds_steps") or []
            assert steps and steps[0]["event"] == "INIT", f"[prox_overlay] pool#{sub.pool_idx} c{cycle_id}: no INIT step"
            outer_steps = [(m, float(steps[0][edge]))]
            for s in steps[1:]:
                outer_steps.append((int(s["expanded_idx"]) + sb, float(s[edge])))
            zones.append(SubZone(cycle_id, "BOS", inner, m, tuple(outer_steps)))
        else:
            pz = [p for p in down["poi_zones"] if int(p.meta["cycle_id"]) == cycle_id]
            if not pz:
                continue
            assert len(pz) == 1, f"[prox_overlay] pool#{sub.pool_idx} c{cycle_id}: {len(pz)} POIs at {m}"
            p = pz[0]
            inner, outer = (float(p.top), float(p.bottom)) if p.side == "buy" else (float(p.bottom), float(p.top))
            zones.append(SubZone(cycle_id, "POI", inner, m, ((m, outer),)))
    return zones


def project_sub_as_of(sub: PooledStructure, cap_abs: Optional[int]) -> Dict[str, Any]:
    """One sub's zones read as of an entity-absolute M15 moment (PLAN_R §5.2): the SAME `project_to_window` the
    sub's real projection uses, floorless, capped there (None = the natural end), with the sub's cross-fib limit
    localized to its slice."""
    from engine_v2.multitf.pooled_structure_build import project_to_window

    bounded, slice_begin = sub.geometry
    sb = int(slice_begin)
    return project_to_window(
        bounded, floor=None, cap=(None if cap_abs is None else int(cap_abs) - sb),
        cap_reason="proximity_view", direction=int(sub.direction), timeframe="M15",
        log_prefix=f"PROX_VIEW{sub.pool_idx}", cross_fib_limits={0: sub.cross_fib_limit.localized(sb)},
    )


def _quiet_project(project, sub, cap, log) -> Dict[str, Any]:
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            return project(sub, cap)
    finally:     # re-printed even when the projection raises
        for line in buf.getvalue().splitlines():
            if _VIEW_LOG_KEEP.search(line):
                log(f"[prox_view] pool#{sub.pool_idx} cap={cap}: {line}")


class ProximityOverlay:
    """The overlay's per-cycle driver (module docstring). `scans` = the non-collapsed cycles' `CycleProximityScan`s
    (`zone_proximity.build_cycle_proximity_scans`, the batch's own builder); `loh(h)` = the H1 → M15 last-of-hour map;
    `project(sub, cap)` = the capped `project_to_window`; `make_triggers(cycle, trigger_list, h)` → `[(detector
    trigger, SweepTrigger)]` for the `subsequent_*` triggers the cycle's list yields AT candle `h`."""

    def __init__(
        self,
        *,
        scans: Dict[Cycle, CycleProximityScan],
        pool: SubStructurePool,
        loh: Callable[[int], int],
        project: Callable[[PooledStructure, Optional[int]], Dict[str, Any]],
        make_triggers: Callable[[Cycle, List[ZoneProximityTrigger], int], List[Tuple[Any, Any]]],
        log: Callable[[str], None] = print,
    ) -> None:
        self.scans = dict(scans)
        self.pool = pool
        self._loh_fn = loh
        self._loh_cache: Dict[int, int] = {}
        self.project = project
        self.make_triggers = make_triggers
        self.log = log
        self._views: Dict[int, List[SubZone]] = {}
        self._source: Dict[Tuple[Cycle, str], Optional[int]] = {}       # pool_idx of each side's source sub
        self._basis: Dict[Tuple[Cycle, str], Optional[Tuple[int, int]]] = {}   # (pool_idx, cycle) of the overlay
        self._opp_lock_basis: Dict[Cycle, Optional[Tuple[int, int]]] = {}
        self.generated: List[Tuple[Any, Any]] = []                     # (detector trigger, SweepTrigger) pushed
        self.used: Dict[Tuple[Cycle, int], ProximityRung] = {}          # sub-sourced trigger → the rung it fired on
        # per tick: (cycle, h, sd reference, opp_sd reference, gap, narrow, the gate that blocked the trigger test)
        self.ref_log: List[Tuple[Cycle, int, Optional[ProximityRung], Optional[ProximityRung], Optional[int], bool,
                                 Optional[str]]] = []
        self.pass_log: List[Tuple[Cycle, int, List[Tuple[Any, ...]]]] = []   # (cycle, the candle whose close, keys)

    # --- the sweep's interface ---
    def initial_ticks(self) -> List[Tuple[int, tuple, Any]]:
        """One `(idx, order, payload)` per candle of every cycle's scan window: idx = `LOH(h − 1)`, order = (S, C, h)."""
        out: List[Tuple[int, tuple, Any]] = []
        for key in sorted(self.scans):
            scan = self.scans[key]
            assert scan.scan_start >= 1, f"[prox_overlay] {key} scan starts at H1 {scan.scan_start}: no LOH(h - 1)"
            for h in range(scan.scan_start, scan.scan_end + 1):
                out.append((self.loh(h - 1), (key[0], key[1], h), (key, h)))
        return out

    def on_tick(self, payload: Tuple[Cycle, int], idx: int) -> Tuple[List[Any], List[Tuple[Any, str, str]]]:
        """One proximity tick (module docstring). Returns (SweepTriggers to FIRE, (SweepTrigger, reason, detail)
        to log unresolved)."""
        key, h = payload
        scan = self.scans[key]
        known = self.loh(h - 1)
        assert int(idx) == known, f"[prox_overlay] {key} tick for H1 {h} at {idx}, expected LOH({h - 1}) = {known}"
        if h > scan.scan_start:
            passed = scan.close(h - 1)
            if passed:
                self.pass_log.append((key, h - 1, [r.key for r in passed]))
        self._refresh(key, h, known)
        passed_opp = scan.refs["opp_sd"]              # the reference before this pick: a lock here = it was passed
        was_locked = scan.locked["opp_sd"]
        trig = scan.step(h)
        if scan.locked["opp_sd"] and not was_locked:
            assert passed_opp is not None and passed_opp.source == "sub", (
                f"[prox_overlay] {key} H1 {h}: opp_sd locked off {passed_opp} — only an overlay rung can be passed")
            self._opp_lock_basis[key] = (passed_opp.sub_pool_idx, passed_opp.sub_cycle_id)
        self.ref_log.append((key, h, scan.refs["sd"], scan.refs["opp_sd"], scan.ref_gap_pips, scan.narrow_prox_ref,
                             scan.gate))
        if trig is None:
            return [], []
        ref = scan.refs[trig.direction]
        if ref is not None and ref.source == "sub":
            self.used[(key, h)] = ref
        made = self.make_triggers(key, scan.triggers, h)
        self.generated.extend(made)         # in the final subsequent_* lists either way (like a collapsed cycle's)
        if self.loh(h) == known:
            return [], [(st, "no_m15_hour", f"H1 {h} has no M15 bar (LOH {known} == LOH of {h - 1})")
                        for _var, st in made]
        return [st for _var, st in made], []

    # --- helpers ---
    def loh(self, h: int) -> int:
        h = int(h)
        if h not in self._loh_cache:
            self._loh_cache[h] = int(self._loh_fn(h))
        return self._loh_cache[h]

    def _counting_from(self, scan: CycleProximityScan, m15_idx: int) -> int:
        """The first H1 candle of the scan whose tick knows M15 `m15_idx` (LOH(h − 1) ≥ it)."""
        for h in range(scan.scan_start, scan.scan_end + 1):
            if self.loh(h - 1) >= m15_idx:
                return h
        return scan.scan_end + 1

    def view(self, sub: PooledStructure) -> List[SubZone]:
        if sub.pool_idx not in self._views:
            self._views[sub.pool_idx] = build_sub_zone_view(sub, self.project, self.log)
        return self._views[sub.pool_idx]

    def _record_start(self, sub: PooledStructure, key: Cycle, known: int) -> Optional[int]:
        """The earliest start of `sub`'s non-zero-length records under `key` started by `known`, else None."""
        starts = [r.start_idx for lens in (LENS_CONFLUENCE, LENS_COUNTER)
                  for r in self.pool.records_for(lens, key[0], key[1])
                  if r.pool_idx == sub.pool_idx and not r.is_zero_length and r.start_idx <= known]
        return min(starts) if starts else None

    def _candidates(self, key: Cycle, direction: int, known: int) -> List[PooledStructure]:
        seen: Dict[int, PooledStructure] = {}
        for lens in (LENS_CONFLUENCE, LENS_COUNTER):
            for r in self.pool.records_for(lens, key[0], key[1]):
                if r.direction == direction and not r.is_zero_length and r.start_idx <= known:
                    seen.setdefault(r.pool_idx, self.pool.get_by_pool_idx(r.pool_idx))
        return list(seen.values())

    def _usable(self, sub: PooledStructure, known: int) -> List[SubZone]:
        """`sub`'s zones known by `known` — frozen at its PROCESSED end when it has ended."""
        lim = known if sub.end_idx is None else min(known, int(sub.end_idx))
        return [z for z in self.view(sub) if z.usable_at <= lim]

    def _parent_poi(self, scan: CycleProximityScan) -> Optional[ProximityRung]:
        """The cycle's sd POI rung nearest the opp_sd side (the sd overlay's qualifying bound), or None."""
        pois = [r for r in scan.parent_rungs if r.side == "sd" and r.zone_kind == "POI"]
        return max(pois, key=lambda r: scan.sd * r.inner) if pois else None

    def _contributed(self, scan: CycleProximityScan, side: str, sub: PooledStructure,
                     known: int) -> Tuple[Optional[int], List[SubZone]]:
        """`sub`'s contribution to `side` at `known`: (its most recent cycle whose BOS zone is usable — STRICT, that
        cycle's zones that would be rungs: on the sd side only those strictly beyond the parent POI inner)."""
        zones = self._usable(sub, known)
        bos_cycles = [z.cycle_id for z in zones if z.kind == "BOS"]
        if not bos_cycles:
            return None, []
        cyc = max(bos_cycles)
        parent_poi = self._parent_poi(scan)
        out = []
        for z in sorted((z for z in zones if z.cycle_id == cyc), key=lambda z: z.kind != "BOS"):
            # sd side: strictly beyond the parent POI inner, in the pullback direction; never without a parent POI.
            if side == "opp_sd" or (parent_poi is not None and scan.sd * (z.inner - parent_poi.inner) < 0):
                out.append(z)
        return cyc, out

    @staticmethod
    def _reachable(scan: CycleProximityScan, side: str, z: SubZone) -> bool:
        """Strictly on the parent's side of the side's last rung (sd: inside the BOS; opp_sd: inside the CTS) — it
        could become the reference (an equal inner loses the tie to the parent zone). No last rung → reachable."""
        last = next((r for r in scan.parent_rungs if r.side == side and r.last), None)
        toward = scan.sd if side == "sd" else -scan.sd
        return last is None or toward * (z.inner - last.inner) > 0

    def _refresh(self, key: Cycle, h: int, known: int) -> None:
        scan = self.scans[key]
        sd = scan.sd

        for side in SIDES:
            direction = sd if side == "sd" else -sd
            cur = self._source.get((key, side))
            cur_sub = self.pool.get_by_pool_idx(cur) if cur is not None else None
            newer = [s for s in self._candidates(key, direction, known)
                     if cur_sub is None or (s.start_idx, s.pool_idx) > (cur_sub.start_idx, cur_sub.pool_idx)]
            # the user, 2026-10-20: a newer sub takes over only when it CONTRIBUTES a REACHABLE zone (the hold)
            eligible = [s for s in newer
                        if any(self._reachable(scan, side, z) for z in self._contributed(scan, side, s, known)[1])]
            if eligible:
                new_src = max(eligible, key=lambda s: (s.start_idx, s.pool_idx))
                self._source[(key, side)] = new_src.pool_idx
                self.log(f"[prox_overlay] {key} H1 {h} {side} source -> pool#{new_src.pool_idx} "
                         f"key=({new_src.direction},{new_src.starting_idx})"
                         + (f" (was pool#{cur})" if cur is not None else ""))
            src = self._source.get((key, side))
            rungs: List[ProximityRung] = []
            basis: Optional[Tuple[int, int]] = None
            if src is not None:
                sub = self.pool.get_by_pool_idx(src)
                cyc, zones = self._contributed(scan, side, sub, known)
                if cyc is not None:
                    basis = (src, cyc)
                    rec_start = self._record_start(sub, key, known)
                    rungs = [self._rung(scan, side, sub, z, rec_start) for z in zones]
            if side == "opp_sd" and scan.locked["opp_sd"] and basis is not None \
                    and basis != self._opp_lock_basis.get(key):
                scan.unlock("opp_sd")
                self.log(f"[prox_overlay] {key} H1 {h} opp_sd lock lifted: a newer counter-sub zone, pool#{basis[0]} "
                         f"c{basis[1]} (the passed one was pool#{self._opp_lock_basis[key][0]} "
                         f"c{self._opp_lock_basis[key][1]}); the reference = the gate's pick from the CTS")
                self._opp_lock_basis[key] = None
            self._basis[(key, side)] = basis
            scan.set_overlay(side, rungs)

    def _rung(self, scan: CycleProximityScan, side: str, sub: PooledStructure, z: SubZone,
              rec_start: Optional[int]) -> ProximityRung:
        def outer_at(i: int, _sub=sub, _z=z) -> float:
            # as of candle i's close, frozen at the sub's processed end
            t = self.loh(i)
            if _sub.end_idx is not None:
                t = min(t, int(_sub.end_idx))
            return _z.outer_at(t)
        usable_from = max(z.usable_at, rec_start if rec_start is not None else z.usable_at)
        return ProximityRung(
            key=("sub", sub.pool_idx, z.cycle_id, z.kind), side=side, zone_kind=z.kind, inner=z.inner,
            outer_at=outer_at, source="sub", tie_rank=1 if z.kind == "BOS" else 2,
            sub_pool_idx=sub.pool_idx, sub_cycle_id=z.cycle_id, usable_idx=self._counting_from(scan, usable_from),
        )
