"""The lifecycle sweep — the sub-structure driver (PART4 §17.6, Plan C §4).

Replaces the Phase-1 two-entity cadence chain (`_ChainCursor` /
`build_two_entity_parent_cycle` / `build_parent_cycle_chain`). Every rule in
§17.4–§17.5 is monotone and write-once, so a priority-queue sweep over
MOMENTS is provably identical to iterating candle-by-candle (the real
per-candle driver is Phase 3; this step body is that loop body).

Pure orchestration: the probe, the geometry build and the reversal handoff are
INJECTED callables so the predicted-table test can stub them. This module
imports no pandas / MS / chart code.

Moments and phases — at each idx, phases run in this order, each to completion
before the next (start-before-end at the same idx is LOAD-BEARING: it is what
keeps a sub continuous across a same-candle handover under the strict `>`
filter of §17.5):

| phase | moment            | action                                                        |
|-------|-------------------|---------------------------------------------------------------|
| 0     | TRIGGER_FIRE /    | resolve (degenerate check → probe → geometry) → create record |
|       | REVERSAL_SPAWN    | → queue its RECORD_START and any end already known            |
| 1     | RECORD_START      | freeze zero-length if the sub already ended BEFORE this idx;  |
|       |                   | else mark active + queue a same_dir_replacement candidate     |
|       |                   | for the same-lens incumbent (different sub)                   |
| 2     | SUB_START         | set sub.start_idx if unset and a live record started here     |
| 3     | RECORD_END        | apply the highest-priority end entry at this idx; entries     |
|       |                   | for an already-ended record are stale and dropped             |
| 4     | SUB_END           | re-evaluate the §17.5 end rule for every sub touched here     |

Order within a phase: TRIGGER_FIRE by `(lens_rank, type_rank,
trigger_event_idx)` (confluence before counter, `first_*` before
`subsequent_*` — today's cadence order; the second trigger's sibling read at
`hi = t` inclusive can see the first's sub); REVERSAL_SPAWN sorts BEFORE
TRIGGER_FIRE at the same idx (the successor must exist before an H1 trigger at
that candle runs its sibling read); every other moment orders by record `seq`.
The heap is idx-monotone: a moment is never queued in the past (asserted).

Logging (stdout, greppable — the `/compare` skill greps
`warning|skipping|unavailable|degenerate|pending|no sid`): `[sweep] fire …`,
`[sweep] record …`, `[sweep] replace …`, `[sweep] spawn …`, `[sweep] sub_end …`,
`[sweep] UNRESOLVED (skipping) reason=…`.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple

from engine_v2.multitf.parent_tables import ParentTables
from engine_v2.multitf.sub_structure_pool import (
    LENS_CONFLUENCE,
    LENS_COUNTER,
    PooledStructure,
    StructureKey,
    SubStructurePool,
    TriggerRecord,
    UnresolvedTrigger,
    _END_REASON_PRIORITY,
)


# --- Inputs / outputs ---------------------------------------------------------

@dataclass(frozen=True)
class SweepTrigger:
    """One H1-derived (or sweep-synthesised reversal) trigger, already
    lens-tagged (`resolve_lens`) and LOH-mapped (`trigger_idx`)."""
    lens: str
    parent_sid: int
    parent_cycle_id: int
    trigger_type: str            # first_confluence | subsequent_confluence | first_counter
                                 # | subsequent_counter | reversal
    trigger_idx: int             # LOH(trigger_event_idx); native M15 for reversal
    direction: int               # lower_sd
    trigger_event_idx: int       # H1 idx (order key); == trigger_idx for reversal
    source: Any = None           # the MultiTFTrigger (opaque)
    pending: bool = False        # first_confluence with status != "finalized"
    probe_input_idx: Optional[int] = None   # whatever was known (H1 for the four types;
                                            # the M15 handoff input for reversal)

    @property
    def cycle(self) -> Tuple[int, int]:
        return (int(self.parent_sid), int(self.parent_cycle_id))


class ResolvedStart(NamedTuple):
    """A resolver's success value (§5.2: the four-tuple + `finalize_condition`)."""
    starting_idx: int
    validated_parent_idx: Optional[int]
    bos0_inner: Optional[float]
    finalize_idx: int
    finalize_condition: str
    probe_input_idx: Optional[int] = None   # the probe's initial input (M15) — informational
    cache_hit: bool = False


class ProbeFailure(NamedTuple):
    """A resolver's failure value (`None` is accepted too)."""
    detail: str
    probe_input_idx: Optional[int] = None


@dataclass
class SweepResult:
    pool: SubStructurePool
    unresolved: List[UnresolvedTrigger]
    spawned: List[SweepTrigger] = field(default_factory=list)


# --- Ordering -----------------------------------------------------------------

_PHASE_FIRE, _PHASE_RECORD_START, _PHASE_SUB_START, _PHASE_RECORD_END, _PHASE_SUB_END = range(5)
_KIND_REVERSAL_SPAWN, _KIND_TRIGGER_FIRE = 0, 1     # spawn sorts before fire at the same idx

_LENS_RANK = {LENS_CONFLUENCE: 0, LENS_COUNTER: 1}
_TYPE_RANK = {
    "first_confluence": 0, "first_counter": 0,
    "subsequent_confluence": 1, "subsequent_counter": 1,
    "reversal": 2,
}


def _fire_order_key(t: SweepTrigger) -> Tuple[int, int, int]:
    return (
        _LENS_RANK.get(t.lens, 9),
        _TYPE_RANK.get(t.trigger_type, 9),
        int(t.trigger_event_idx),
    )


# --- The sweep ----------------------------------------------------------------

class _Sweep:
    def __init__(
        self,
        *,
        pool: SubStructurePool,
        tables: ParentTables,
        resolve_start: Callable[[SweepTrigger, int], Any],
        build_geometry: Callable[[StructureKey, Optional[float]], Optional[Tuple[PooledStructure, bool]]],
        resolve_reversal: Callable[[PooledStructure, int, int], Any],
        synth_reversal_trigger: Callable[[Any, int, int], Any],
        parent_path: str,
        sub_tf: str,
        log: Callable[[str], None],
    ) -> None:
        self.pool = pool
        self.tables = tables
        self.resolve_start = resolve_start
        self.build_geometry = build_geometry
        self.resolve_reversal = resolve_reversal
        self.synth_reversal_trigger = synth_reversal_trigger
        self.parent_path = parent_path
        self.sub_tf = sub_tf
        self.log = log
        self._heap: List[tuple] = []
        self._push_counter = 0
        self._cur_idx: Optional[int] = None
        self._cur_phase: int = -1
        self._active: Set[int] = set()              # record seqs currently active
        self._touched: Set[int] = set()             # sub_ids touched at the current idx
        self.spawned: List[SweepTrigger] = []
        self._by_seq: Dict[int, TriggerRecord] = {}

    # --- heap ---
    def _push(self, idx: int, phase: int, order: tuple, payload: Any) -> None:
        idx = int(idx)
        if self._cur_idx is not None:
            assert (idx, phase) >= (self._cur_idx, self._cur_phase), (
                f"[sweep] moment queued in the past: ({idx},{phase}) while at "
                f"({self._cur_idx},{self._cur_phase})"
            )
        self._push_counter += 1
        heapq.heappush(self._heap, (idx, phase, tuple(order), self._push_counter, payload))

    # --- entry ---
    def run(self, triggers: List[SweepTrigger]) -> None:
        for t in triggers:
            self._push(t.trigger_idx, _PHASE_FIRE, (_KIND_TRIGGER_FIRE,) + _fire_order_key(t),
                       ("fire", t))
        last_idx: Optional[int] = None
        while self._heap:
            idx, phase, _order, _n, payload = heapq.heappop(self._heap)
            if last_idx is not None:
                assert idx >= last_idx, f"[sweep] heap popped {idx} after {last_idx}"
            if idx != last_idx:
                self._touched = set()
            last_idx = idx
            self._cur_idx, self._cur_phase = idx, phase
            kind = payload[0]
            if kind == "fire":
                self._phase0_fire(payload[1], idx)
            elif kind == "spawn":
                self._phase0_spawn(payload[1], idx)
            elif kind == "record_start":
                self._phase1_record_start(payload[1], idx)
            elif kind == "sub_start":
                self._phase2_sub_start(payload[1], idx)
            elif kind == "record_end":
                rec = payload[1]
                entries = [(idx, payload[2], payload[3])]
                # Every RECORD_END entry for this record at this idx is adjacent
                # in pop order (same idx / phase / seq) — drain them and apply once.
                while (self._heap and self._heap[0][0] == idx
                       and self._heap[0][1] == _PHASE_RECORD_END
                       and self._heap[0][4][0] == "record_end"
                       and self._heap[0][4][1] is rec):
                    _i, _p, _o, _n, pl = heapq.heappop(self._heap)
                    entries.append((idx, pl[2], pl[3]))
                self._phase3_record_end(rec, idx, entries)
            elif kind == "sub_end":
                self._phase4_sub_end(payload[1], idx)
            else:  # pragma: no cover
                raise AssertionError(f"[sweep] unknown moment {kind}")
        self._after_sweep()

    # --- phase 0 ---
    def _unresolved(self, t: SweepTrigger, reason: str, detail: str,
                    probe_input_idx: Optional[int] = None) -> None:
        pii = probe_input_idx if probe_input_idx is not None else t.probe_input_idx
        u = UnresolvedTrigger(
            lens=t.lens, parent_sid=int(t.parent_sid), parent_cycle_id=int(t.parent_cycle_id),
            trigger_type=t.trigger_type, trigger_idx=int(t.trigger_idx), direction=int(t.direction),
            probe_input_idx=(int(pii) if pii is not None else None), reason=reason, detail=detail,
        )
        self.pool.unresolved.append(u)
        self.log(
            f"[sweep] UNRESOLVED (skipping) reason={reason} lens={t.lens} "
            f"parent=({t.parent_sid},{t.parent_cycle_id}) type={t.trigger_type} "
            f"trigger_idx={t.trigger_idx} detail={detail}"
        )

    def _phase0_fire(self, t: SweepTrigger, idx: int) -> None:
        self.log(
            f"[sweep] fire lens={t.lens} parent=({t.parent_sid},{t.parent_cycle_id}) "
            f"type={t.trigger_type} trigger_idx={t.trigger_idx} dir={t.direction}"
        )
        if t.pending:
            self._unresolved(t, "pending", "first_confluence probe not finalized")
            return
        self._resolve_and_record(t, idx)

    def _phase0_spawn(self, sub: PooledStructure, R: int) -> None:
        live = [r for r in sub.records if r.is_live_at_reversal(R)]
        if not live:
            self.log(
                f"[sweep] spawn sub_id={sub.sub_id} R={R}: no live record at R — reversal inert"
            )
            return
        for r in live:
            src = self.synth_reversal_trigger(r.source_trigger, -int(sub.direction), int(R))
            t = SweepTrigger(
                lens=r.lens, parent_sid=r.parent_sid, parent_cycle_id=r.parent_cycle_id,
                trigger_type="reversal", trigger_idx=int(R), direction=-int(sub.direction),
                trigger_event_idx=int(R), source=src, pending=False, probe_input_idx=None,
            )
            self.spawned.append(t)
            self.log(
                f"[sweep] spawn sub_id={sub.sub_id} R={R} from record {r.identity} "
                f"-> reversal trigger lens={t.lens} dir={t.direction}"
            )
            self._resolve_and_record(t, R, reversing_sub=sub)

    def _resolve_and_record(self, t: SweepTrigger, idx: int,
                            reversing_sub: Optional[PooledStructure] = None) -> None:
        S, C = t.cycle
        # 1. degenerate parent cycle → log, don't build (no probe, no MS).
        assert self.tables.has_cycle(S, C), (
            f"[sweep] trigger {t.trigger_type} in ({S},{C}) has no parent CTS_ESTABLISHED"
        )
        if self.tables.is_degenerate(S, C):
            self._unresolved(
                t, "degenerate_parent_cycle",
                f"floor {self.tables.floor(S, C)} >= end {self.tables.end(S, C)}",
            )
            return
        # 2. probe (the cache lives inside the resolver).
        if t.trigger_type == "reversal":
            assert reversing_sub is not None
            res = self.resolve_reversal(reversing_sub, int(idx), int(t.direction))
        else:
            res = self.resolve_start(t, int(t.trigger_idx))
        if res is None:
            self._unresolved(t, "probe_failed", "resolver returned None")
            return
        if isinstance(res, ProbeFailure):
            self._unresolved(t, "probe_failed", res.detail, res.probe_input_idx)
            return
        assert isinstance(res, ResolvedStart), f"[sweep] bad resolver result {res!r}"
        finalize_idx = int(res.finalize_idx)
        if t.trigger_type not in ("first_confluence", "reversal") and not res.cache_hit:
            # §5.2: sibling types are Phase-1 probes bounded at hi = trigger_idx →
            # finalize == trigger_idx by construction (a cache hit inherits an
            # earlier finalize and is exempt).
            assert finalize_idx == int(t.trigger_idx), (
                f"[sweep] {t.trigger_type} finalize_idx={finalize_idx} != trigger_idx={t.trigger_idx}"
            )
        # 3. geometry — build first, pool entry only on success.
        key = StructureKey(self.parent_path, self.sub_tf, int(t.direction), int(res.starting_idx))
        built = self.build_geometry(key, res.bos0_inner)
        if built is None:
            self._unresolved(t, "geometry_failed", f"no geometry for key {tuple(key)}",
                             res.probe_input_idx)
            return
        sub, created = built
        if created and sub.natural_reversal_idx is not None:
            R = int(sub.natural_reversal_idx)
            if R > int(idx):
                self._push(R, _PHASE_FIRE, (_KIND_REVERSAL_SPAWN, sub.sub_id), ("spawn", sub))
            # R <= idx: reversed before this trigger — nothing to spawn (no record
            # can be live at R); step 6 makes this record zero-length.
        if (not created and res.bos0_inner is not None and sub.bos0_inner is not None
                and abs(float(sub.bos0_inner) - float(res.bos0_inner)) > 1e-9):
            self.log(
                f"WARNING [sweep] bos0_inner mismatch on sub_id={sub.sub_id} key={tuple(key)}: "
                f"pool={sub.bos0_inner} probe={res.bos0_inner} (keeping the first)"
            )
        # 4. same (lens, parent) → same record: absorb.
        existing = [r for r in self.pool.records_for(t.lens, S, C) if r.sub_id == sub.sub_id]
        if existing:
            existing[0].extra_trigger_idxs.append(int(t.trigger_idx))
            self.log(
                f"[sweep] record {existing[0].identity} absorbs trigger {t.trigger_type}@{t.trigger_idx}"
            )
            return
        # 5. the record.
        floor = int(self.tables.floor(S, C))
        parent_sd = self.tables.parent_sd.get(S)
        assert parent_sd is not None, f"[sweep] no parent_sd for sid {S}"
        rec = TriggerRecord(
            lens=t.lens, parent_sid=S, parent_cycle_id=C,
            trigger_sub_sid=self.pool.next_trigger_sub_sid(t.lens, S, C),
            sub_id=sub.sub_id,
            trigger_type=t.trigger_type, trigger_idx=int(t.trigger_idx),
            probe_finalize_idx=finalize_idx,
            probe_finalize_condition=str(res.finalize_condition),
            validated_parent_idx=(int(res.validated_parent_idx)
                                  if res.validated_parent_idx is not None else None),
            starting_idx=int(sub.starting_idx), direction=int(sub.direction), sub_tf=self.sub_tf,
            relative_dir=("confluence" if int(t.direction) == int(parent_sd) else "counter"),
            parent_floor_idx=floor,
            start_idx=max(finalize_idx, int(t.trigger_idx), floor),
            source_trigger=t.source,
            seq=self.pool.next_seq(),
        )
        self._by_seq[rec.seq] = rec
        # 6. known-in-advance ends.
        if sub.end_idx is not None and rec.start_idx > sub.end_idx:
            # post-end re-trigger: the sub is frozen — link, never active.
            rec.trigger_end_idx = int(sub.end_idx)
            rec.end_reason = sub.end_reason
            ender = next((r for r in sub.records if r.end_idx == sub.end_idx
                          and r.end_reason == sub.end_reason), None)
            rec.ended_by_sub_id = ender.ended_by_sub_id if ender is not None else None
        else:
            cands: List[Tuple[int, str]] = []
            if sub.natural_reversal_idx is not None:
                cands.append((int(sub.natural_reversal_idx), "reversal"))
            pe = self.tables.end(S, C)
            if pe is not None:
                cands.append((int(pe), "parent_end"))
            e = min(cands, key=lambda c: (c[0], _END_REASON_PRIORITY[c[1]])) if cands else None
            if e is not None and e[0] <= rec.start_idx:
                rec.trigger_end_idx, rec.end_reason = e          # zero-length now
            elif e is not None:
                self._push(e[0], _PHASE_RECORD_END, (rec.seq,), ("record_end", rec, e[1], None))
        rec.end_idx = (max(rec.trigger_end_idx, rec.start_idx)
                       if rec.trigger_end_idx is not None else None)
        # 7. register + queue the start.
        self.pool.add_record(rec)
        self.log(
            f"[sweep] record {rec.identity} sub_id={rec.sub_id} type={rec.trigger_type} "
            f"trigger_idx={rec.trigger_idx} finalize={rec.probe_finalize_idx} floor={floor} "
            f"start_idx={rec.start_idx} known_end={rec.trigger_end_idx}/{rec.end_reason} "
            f"relative_dir={rec.relative_dir} zero_length={rec.is_zero_length}"
        )
        if not rec.is_zero_length:
            self._push(rec.start_idx, _PHASE_RECORD_START, (rec.seq,), ("record_start", rec))

    # --- phase 1 ---
    def _phase1_record_start(self, rec: TriggerRecord, t: int) -> None:
        sub = self.pool.get_by_id(rec.sub_id)
        assert rec.trigger_end_idx is None or rec.trigger_end_idx > t, (
            f"[sweep] record {rec.identity} reached RECORD_START at {t} already ended "
            f"at {rec.trigger_end_idx}"
        )
        if sub.end_idx is not None and sub.end_idx < rec.start_idx:
            # The sub froze while this record was pending (strict <: == t is the
            # same-candle handover, which keeps the sub alive).
            rec.trigger_end_idx = int(sub.end_idx)
            rec.end_reason = sub.end_reason
            rec.end_idx = max(rec.trigger_end_idx, rec.start_idx)
            ender = next((r for r in sub.records if r is not rec and r.end_idx == sub.end_idx
                          and r.end_reason == sub.end_reason), None)
            rec.ended_by_sub_id = ender.ended_by_sub_id if ender is not None else None
            self.log(
                f"[sweep] record {rec.identity} start {t} after sub_id={sub.sub_id} ended "
                f"{sub.end_idx}/{sub.end_reason} -> frozen zero-length"
            )
            return
        self._touched.add(sub.sub_id)
        # The incumbent = the record of this (lens, S, C, direction) that has
        # STARTED (its RECORD_START ran — `_active`) and not ended. The interval
        # rule alone would also see a same-idx record whose start moment is
        # still queued (a later seq); the collision rule is "the later seq
        # replaces the earlier", so only started records count here.
        incs = [
            r for r in self.pool.records_for(rec.lens, rec.parent_sid, rec.parent_cycle_id)
            if r is not rec and r.direction == rec.direction and r.seq in self._active
            and r.is_active_at(t)
        ]
        if len(incs) > 1:
            # Acausal: an earlier incumbent whose replacement end is queued for
            # phase 3 of THIS idx plus a same-idx starter. §4.2 says degrade
            # ("the later seq replaces the earlier" + WARNING), never crash —
            # every started incumbent is replaced by this record.
            self.log(
                f"WARNING [sweep] {len(incs)} started records for ({rec.lens},"
                f"{rec.parent_sid},{rec.parent_cycle_id},dir={rec.direction}) at {t}: "
                f"{[r.identity for r in incs]} — all replaced by {rec.identity}"
            )
        self._active.add(rec.seq)
        for inc in incs:
            if inc.sub_id == rec.sub_id or inc.trigger_end_idx is not None:
                continue
            if inc.start_idx == rec.start_idx:
                # Acausal case (§4.2): the later seq replaces the earlier. The
                # incumbent is zero-length by construction (end == start), so it
                # is frozen here and now — a zero-length record participates in
                # nothing, and deferring to phase 3 would let phase 2 read it as
                # live for one moment.
                self.log(
                    f"WARNING [sweep] same-idx start collision at {t}: {inc.identity} "
                    f"(sub {inc.sub_id}) replaced by {rec.identity} (sub {rec.sub_id})"
                )
                inc.trigger_end_idx = int(t)
                inc.end_reason = "same_dir_replacement"
                inc.ended_by_sub_id = rec.sub_id
                inc.end_idx = max(int(t), inc.start_idx)
                self._active.discard(inc.seq)
            else:
                self.log(
                    f"[sweep] replace {inc.identity} sub_id={inc.sub_id} by sub_id={rec.sub_id} at {t}"
                )
                self._push(t, _PHASE_RECORD_END, (inc.seq,),
                           ("record_end", inc, "same_dir_replacement", rec.sub_id))
        self._push(t, _PHASE_SUB_START, (rec.seq,), ("sub_start", rec))
        # A live record STARTED here → the sub is re-evaluated in phase 4 (a no-op
        # by monotonicity — a start only shrinks the candidate set — kept for
        # fidelity with §4.4).
        self._push(t, _PHASE_SUB_END, (sub.sub_id,), ("sub_end", sub.sub_id))

    # --- phase 2 ---
    def _phase2_sub_start(self, rec: TriggerRecord, t: int) -> None:
        sub = self.pool.get_by_id(rec.sub_id)
        if sub.start_idx is None and not rec.is_zero_length and rec.start_idx == t:
            sub.start_idx = int(t)
            self.log(f"[sweep] sub_start sub_id={sub.sub_id} start_idx={t} via {rec.identity}")

    # --- phase 3 ---
    def _phase3_record_end(self, rec: TriggerRecord, t: int,
                           entries: List[Tuple[int, str, Optional[int]]]) -> None:
        if rec.trigger_end_idx is not None:
            return                                  # stale entries (ended earlier)
        best = min(entries, key=lambda e: _END_REASON_PRIORITY.get(e[1], 99))
        rec.trigger_end_idx = int(t)
        rec.end_reason = best[1]
        rec.ended_by_sub_id = best[2]
        rec.end_idx = max(int(t), rec.start_idx)
        self._active.discard(rec.seq)
        self._touched.add(rec.sub_id)
        self.log(
            f"[sweep] record_end {rec.identity} sub_id={rec.sub_id} at {t} "
            f"reason={rec.end_reason} ended_by={rec.ended_by_sub_id}"
        )
        self._push(t, _PHASE_SUB_END, (rec.sub_id,), ("sub_end", rec.sub_id))

    # --- phase 4 ---
    def _phase4_sub_end(self, sub_id: int, t: int) -> None:
        sub = self.pool.get_by_id(sub_id)
        if sub.end_idx is not None:
            return
        live = [r for r in sub.records if not r.is_zero_length and r.start_idx <= t]
        if not live:
            return
        max_start = max(r.start_idx for r in live)
        cands = [
            (r.end_idx, _END_REASON_PRIORITY.get(r.end_reason, 99), r)
            for r in live if r.end_idx is not None and r.end_idx > max_start
        ]
        if not cands:
            return
        e, _p, r = min(cands, key=lambda c: (c[0], c[1], c[2].seq))
        sub.end_idx = int(e)
        sub.end_reason = r.end_reason
        self.log(
            f"[sweep] sub_end sub_id={sub.sub_id} end_idx={sub.end_idx} reason={sub.end_reason} "
            f"via {r.identity} (max_start={max_start})"
        )

    # --- after ---
    def _after_sweep(self) -> None:
        for sub in self.pool.all():
            sub.relative_dir_segments = _relative_dir_segments(sub)
            for r in sub.records:
                if r.is_zero_length:
                    continue
                assert r.end_idx is None or r.start_idx < r.end_idx, (
                    f"[sweep] live record {r.identity} has start {r.start_idx} >= end {r.end_idx}"
                )
                assert sub.end_idx is None or r.start_idx <= sub.end_idx, (
                    f"[sweep] live record {r.identity} starts {r.start_idx} after its sub "
                    f"{sub.sub_id} ended {sub.end_idx}"
                )
            if sub.start_idx is not None:
                assert sub.end_idx is None or sub.end_idx > sub.start_idx, (
                    f"[sweep] sub {sub.sub_id} window [{sub.start_idx},{sub.end_idx}] is empty"
                )
            else:
                assert not sub.live_records(), (
                    f"[sweep] sub {sub.sub_id} has live records but no start_idx"
                )
        # Per (lens, S, C, direction): live windows non-overlapping as half-open intervals.
        by_scope: Dict[Tuple[str, int, int, int], List[TriggerRecord]] = {}
        for r in self.pool.all_records():
            if r.is_zero_length:
                continue
            by_scope.setdefault((r.lens, r.parent_sid, r.parent_cycle_id, r.direction), []).append(r)
        for scope, recs in by_scope.items():
            recs.sort(key=lambda r: (r.start_idx, r.seq))
            for a, b in zip(recs, recs[1:]):
                a_end = a.end_idx
                assert a_end is not None and a_end <= b.start_idx, (
                    f"[sweep] overlapping live records in {scope}: {a.identity} "
                    f"[{a.start_idx},{a.end_idx}] vs {b.identity} [{b.start_idx},{b.end_idx}]"
                )


def _relative_dir_segments(sub: PooledStructure) -> List[Tuple[int, str]]:
    """§17.3 step function over `[start_idx, end_idx]`: at each record start/end,
    the active record with the latest `start_idx` → its `relative_dir`; carry
    forward when none is active."""
    if sub.start_idx is None:
        return []
    live = sub.live_records()
    points: Set[int] = set()
    for r in live:
        points.add(int(r.start_idx))
        if r.end_idx is not None:
            points.add(int(r.end_idx))
    if sub.end_idx is not None:
        points = {p for p in points if p <= sub.end_idx}
    out: List[Tuple[int, str]] = []
    cur: Optional[str] = None
    for p in sorted(points):
        if p < sub.start_idx:
            continue
        active = [r for r in live if r.start_idx <= p and (r.end_idx is None or r.end_idx > p)]
        if active:
            latest = max(active, key=lambda r: (r.start_idx, r.seq))
            rd = latest.relative_dir
        else:
            rd = cur
        if rd is not None and rd != cur:
            out.append((int(p), rd))
            cur = rd
    return out


def run_lifecycle_sweep(
    triggers: List[SweepTrigger],
    *,
    pool: SubStructurePool,
    tables: ParentTables,
    resolve_start: Callable[[SweepTrigger, int], Any],
    build_geometry: Callable[[StructureKey, Optional[float]], Optional[Tuple[PooledStructure, bool]]],
    resolve_reversal: Callable[[PooledStructure, int, int], Any],
    synth_reversal_trigger: Optional[Callable[[Any, int, int], Any]] = None,
    parent_path: str = "H1.main",
    sub_tf: str = "M15",
    log: Callable[[str], None] = print,
) -> SweepResult:
    """Run the §17.6 sweep over `triggers` and populate `pool` in place.

    - `resolve_start(trigger, hi)` → `ResolvedStart` | `ProbeFailure` | None for
      the four H1 types (`hi` = the trigger's `trigger_idx`, the sibling window's
      inclusive upper edge).
    - `build_geometry(key, bos0_inner)` → `(sub, created)` or None (the §5.4
      builder: MS first, pool entry only on success).
    - `resolve_reversal(sub, R, probe_direction)` → the reversal handoff over
      `sub`'s geometry, entity-absolute.
    - `synth_reversal_trigger(source_trigger, sd, R)` → the successor's
      `MultiTFTrigger`; default = `entity_df_mutation._synth_reversal_trigger`.
    """
    if synth_reversal_trigger is None:
        from engine_v2.multitf.entity_df_mutation import _synth_reversal_trigger
        synth_reversal_trigger = _synth_reversal_trigger
    sweep = _Sweep(
        pool=pool, tables=tables, resolve_start=resolve_start, build_geometry=build_geometry,
        resolve_reversal=resolve_reversal, synth_reversal_trigger=synth_reversal_trigger,
        parent_path=parent_path, sub_tf=sub_tf, log=log,
    )
    sweep.run(list(triggers))
    return SweepResult(pool=pool, unresolved=pool.unresolved, spawned=sweep.spawned)
