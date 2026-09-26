"""Unit tests for the Plan C lifecycle sweep — §9.3 sweep semantics + the §9.1
by-design rewrites that concern the sweep (PLAN_C_lifecycle_rewrite.md §2, §4;
PART4_REFACTOR_SPEC.md §17.4–§17.7).

Written TESTS-FIRST against `PLAN_C_API_CONTRACT.md`. The production modules
(`multitf/lifecycle_sweep.py`, `multitf/parent_tables.py`, the rewritten
`TriggerRecord` / `PooledStructure` / `UnresolvedTrigger`) do not exist at the
base — this file fails at collection with `ModuleNotFoundError` there.

Every fixture is synthetic and hand-derived; each expected value carries a
comment deriving it from the rule. All idxs are entity-absolute M15 ints. The
sweep is pure orchestration: the probe (`resolve_start` / `resolve_reversal`)
and the geometry builder (`build_geometry`) are injected stubs that create pool
entries through `pool.get_or_create(key)` and set `natural_reversal_idx`.

Conventions
- `_tables(...)` builds a `ParentTables` from M15 floors/ends; the sweep reads
  only `floor_m15` / `end_m15` / `degenerate` / `parent_sd`.
- `trigger_event_idx` is an ORDER KEY only (`trigger_idx // 4`); the sweep has
  no df and cannot check LOH consistency.
- Trigger-type ↔ direction semantics (which detector fires for which
  direction) are the detectors' business; the sweep keys everything on the
  trigger's `lens` / `direction` / `(S, C)` as given.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace as _dc_replace
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytest

from engine_v2.multitf.parent_tables import ParentTables
from engine_v2.multitf.types import MultiTFTrigger
from engine_v2.multitf.lifecycle_sweep import (
    ProbeFailure,
    ResolvedStart,
    SweepTrigger,
    run_lifecycle_sweep,
)
from engine_v2.multitf.sub_structure_pool import (
    LENS_CONFLUENCE as CONF,
    LENS_COUNTER as CTR,
    PooledStructure,
    StructureKey,
    SubStructurePool,
    UnresolvedTrigger,
    _END_REASON_PRIORITY,
)

PARENT = "H1.main"
SUB_TF = "M15"


# =============================================================================
# Fixture helpers
# =============================================================================

def _tables(
    cycles: Dict[Tuple[int, int], Tuple[int, Optional[int]]],
    parent_sd: Dict[int, int],
) -> ParentTables:
    """`ParentTables` from M15 values: `cycles = {(S, C): (floor_m15, end_m15|None)}`.

    `degenerate[(S,C)] = end_m15 is not None and floor_m15 >= end_m15` (§3).
    The H1-side dicts are derived with the inverse of LOH(h) = 4h + 3 (exact for
    values ≡ 3 mod 4, floor-divided otherwise) and are provenance only — the
    sweep reads `floor_m15` / `end_m15` / `degenerate` / `parent_sd`
    (`ParentTables.floor()/end()` are the M15 accessors per the contract).
    `cts_moment` = `floor_h1` (struct_start <= moment assumed); `struct_start[S]`
    = the sid's earliest floor; `rev_by_sid[S]` = the sid's last cycle end.
    """
    floor_m15 = {k: int(f) for k, (f, _e) in cycles.items()}
    end_m15 = {k: (None if e is None else int(e)) for k, (_f, e) in cycles.items()}
    floor_h1 = {k: (f - 3) // 4 for k, f in floor_m15.items()}
    end_h1 = {k: (None if e is None else (e - 3) // 4) for k, e in end_m15.items()}
    struct_start = {
        S: min(f for (s, _c), f in floor_h1.items() if s == S) for S in parent_sd
    }
    rev_by_sid: Dict[int, int] = {}
    for S in parent_sd:
        last_c = max(c for (s, c) in cycles if s == S)
        e = end_h1[(S, last_c)]
        if e is not None:
            rev_by_sid[S] = e
    degenerate = {
        k: (end_m15[k] is not None and floor_m15[k] >= end_m15[k]) for k in cycles
    }
    return ParentTables(
        rev_by_sid=rev_by_sid,
        struct_start=struct_start,
        cts_moment=dict(floor_h1),
        parent_sd=dict(parent_sd),
        floor_h1=floor_h1,
        end_h1=end_h1,
        floor_m15=floor_m15,
        end_m15=end_m15,
        degenerate=degenerate,
    )


def _trig(
    lens: str, S: int, C: int, ttype: str, t: int, d: int,
    *, pending: bool = False, parent_input: Optional[int] = None,
) -> SweepTrigger:
    """A lens-tagged, LOH-mapped H1 trigger. `trigger_event_idx = t // 4` is an
    order key only; `parent_input` = its H1 input candle (`parent_input_idx`)."""
    return SweepTrigger(
        lens=lens, parent_sid=S, parent_cycle_id=C, trigger_type=ttype,
        trigger_idx=t, direction=d, trigger_event_idx=t // 4, source=None,
        pending=pending, parent_input_idx=parent_input,
    )


def _rs(
    starting_idx: int, finalize_idx: int, *,
    parent_bos_anchor: Optional[int] = None, bos0: Optional[float] = 1.0,   # pass-through stub values
    # (the real rule — FC-only — is pinned in test_first_trigger_migration)
    cond: str = "stub", cache_hit: bool = False, probe_input: Optional[int] = None,
) -> ResolvedStart:
    return ResolvedStart(
        starting_idx=starting_idx, parent_bos_anchor_idx=parent_bos_anchor,
        bos0_inner=bos0, finalize_idx=finalize_idx, finalize_condition=cond,
        probe_input_idx=probe_input, cache_hit=cache_hit,
    )


def _resolver(mapping: Dict[Any, Any]):
    """`resolve_start(trigger, hi)` stub keyed on `(lens, trigger_idx)` (fallback:
    `trigger_idx`). Values: `ResolvedStart` | `ProbeFailure` | None. Records
    every call on `.calls` as `(trigger, hi)`."""
    calls: List[Tuple[SweepTrigger, int]] = []

    def resolve_start(trigger: SweepTrigger, hi: int):
        calls.append((trigger, hi))
        key = (trigger.lens, trigger.trigger_idx)
        if key in mapping:
            return mapping[key]
        return mapping[trigger.trigger_idx]

    resolve_start.calls = calls          # type: ignore[attr-defined]
    return resolve_start


def _geom(
    pool: SubStructurePool,
    nat_rev: Dict[Tuple[int, int], Optional[int]],
    *, fail_once: Optional[set] = None,
):
    """`build_geometry(key, bos0_inner)` stub — the §5.4 builder: on a miss the
    pool entry is created (here: only when the build "succeeds") with the stubbed
    `natural_reversal_idx` from `nat_rev[(direction, starting_idx)]`; keys in
    `fail_once` return None ONCE (no pool entry, no sub_id consumed), then build.
    Records every call on `.calls` as `(key, bos0_inner)`."""
    calls: List[Tuple[StructureKey, Optional[float]]] = []
    failing = set(fail_once or ())

    def build_geometry(key: StructureKey, bos0_inner: Optional[float]):
        calls.append((key, bos0_inner))
        dk = (key.direction, key.starting_idx)
        if dk in failing:
            failing.discard(dk)
            return None
        sub, created = pool.get_or_create(key)
        if created:
            sub.natural_reversal_idx = nat_rev.get(dk)
            sub.bos0_inner = bos0_inner
            sub.geometry = ("stub-geometry", 0)     # (bounded, slice_begin) shape
        return sub, created

    build_geometry.calls = calls         # type: ignore[attr-defined]
    return build_geometry


def _rev_resolver(mapping: Dict[Tuple[int, int], Any]):
    """`resolve_reversal(sub, R, probe_direction)` stub keyed on the REVERSING
    sub's `(direction, starting_idx)` → successor `starting_idx` (or a
    `ProbeFailure`). Success = `ResolvedStart(finalize_idx=R)` — the handoff's
    finalize is R by construction (§4.3 spawn: start = max(R, R, floor) = R).
    Unknown subs → `ProbeFailure`. Records `(sub_id, R, probe_direction)`."""
    calls: List[Tuple[int, int, int]] = []

    def resolve_reversal(sub: PooledStructure, R: int, probe_direction: int):
        calls.append((sub.sub_id, R, probe_direction))
        v = mapping.get((sub.direction, sub.starting_idx))
        if v is None:
            return ProbeFailure(detail="no successor stubbed", probe_input_idx=R)
        if isinstance(v, ProbeFailure):
            return v
        return _rs(int(v), R, cond="reversal_handoff", probe_input=R)

    resolve_reversal.calls = calls       # type: ignore[attr-defined]
    return resolve_reversal


def _synth_stub(source_trigger: Any, sd: int, R: int) -> Any:
    """Stand-in for `entity_df_mutation._synth_reversal_trigger` (which would
    `dataclasses.replace(None, ...)` on our source-less triggers)."""
    return ("synth-reversal", source_trigger, int(sd), int(R))


def _run(
    triggers: List[SweepTrigger], *, pool: SubStructurePool, tables: ParentTables,
    resolve=None, geom=None, resolve_rev=None,
):
    """Run the sweep with stubs + a capturing `log`; returns `(result, logs)`."""
    logs: List[str] = []

    def _log(*args: Any, **_kw: Any) -> None:
        logs.append(" ".join(str(a) for a in args))

    result = run_lifecycle_sweep(
        triggers,
        pool=pool,
        tables=tables,
        resolve_start=resolve if resolve is not None else _resolver({}),
        build_geometry=geom if geom is not None else _geom(pool, {}),
        resolve_reversal=resolve_rev if resolve_rev is not None else _rev_resolver({}),
        synth_reversal_trigger=_synth_stub,
        parent_path=PARENT,
        sub_tf=SUB_TF,
        log=_log,
    )
    return result, logs


def _sub(pool: SubStructurePool, direction: int, starting_idx: int) -> PooledStructure:
    s = pool.get(StructureKey(PARENT, SUB_TF, direction, starting_idx))
    assert s is not None, f"no sub ({direction}, {starting_idx}) in the pool"
    return s


def _window(x) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    return (x.start_idx, x.end_idx, x.end_reason)


def _assert_sweep_invariants(pool: SubStructurePool) -> None:
    """The §4.5 post-sweep asserts, re-derived from the pool (item 12)."""
    for sub in pool.all():
        for r in sub.records:
            if not r.is_zero_length:
                assert r.end_idx is None or r.start_idx < r.end_idx, r
        if sub.start_idx is not None:
            assert sub.end_idx is None or sub.end_idx > sub.start_idx, sub
    groups: Dict[Tuple[str, int, int, int], list] = defaultdict(list)
    for r in pool.all_records():
        if not r.is_zero_length:
            groups[(r.lens, r.parent_sid, r.parent_cycle_id, r.direction)].append(r)
    for recs in groups.values():
        recs.sort(key=lambda r: (r.start_idx, r.seq))
        for a, b in zip(recs, recs[1:]):
            # half-open [start, end): a handover may share the boundary candle
            assert a.end_idx is not None and a.end_idx <= b.start_idx, (a, b)


# =============================================================================
# Reusable groundtruth scenarios (subs keyed by (direction, starting_idx))
# =============================================================================

def _scenario_2365(pool: SubStructurePool):
    """Groundtruth subs 0→1→2→3 on H1 sid 0 (parent_sd +1):
    (0,0) floor 463 end 2611; (0,1) floor 2611 end 3611.

    FC(0,0) t=463 → (+1,454) fin 1020, R=1940 → rev → (−1,1797) R=2470 → rev →
    (+1,2365) R=2829; FC(0,1) t=2611 → (+1,2365) fin 2608 (same sub);
    (+1,2365) rev at 2829 → (−1,2639) R=None.
    """
    tables = _tables({(0, 0): (463, 2611), (0, 1): (2611, 3611)}, {0: +1})
    triggers = [
        _trig(CONF, 0, 0, "first_confluence", 463, +1),
        _trig(CONF, 0, 1, "first_confluence", 2611, +1),
    ]
    resolve = _resolver({
        (CONF, 463): _rs(454, 1020, parent_bos_anchor=96, cond="second_cts_reached"),
        (CONF, 2611): _rs(2365, 2608, parent_bos_anchor=591, cond="second_cts_reached"),
    })
    geom = _geom(pool, {(+1, 454): 1940, (-1, 1797): 2470, (+1, 2365): 2829, (-1, 2639): None})
    rev = _rev_resolver({(+1, 454): 1797, (-1, 1797): 2365, (+1, 2365): 2639})
    return _run(triggers, pool=pool, tables=tables, resolve=resolve, geom=geom, resolve_rev=rev)


def _scenario_3304(pool: SubStructurePool):
    """Groundtruth subs 7→9→10 on H1 sid 1 (parent_sd −1): (1,2) floor 3611 end None.

    FC(1,2) t=3611 → (−1,3304) fin 3621, R=None; subseq_conf(1,2) t=3819 →
    (−1,3760) fin 3819, R=4200 → rev → (+1,4027) R=None.
    """
    tables = _tables({(1, 2): (3611, None)}, {1: -1})
    triggers = [
        _trig(CONF, 1, 2, "first_confluence", 3611, -1),
        _trig(CONF, 1, 2, "subsequent_confluence", 3819, -1),
    ]
    resolve = _resolver({
        (CONF, 3611): _rs(3304, 3621, parent_bos_anchor=826, cond="no_retrace"),
        (CONF, 3819): _rs(3760, 3819, parent_bos_anchor=954),
    })
    geom = _geom(pool, {(-1, 3304): None, (-1, 3760): 4200, (+1, 4027): None})
    rev = _rev_resolver({(-1, 3760): 4027})
    return _run(triggers, pool=pool, tables=tables, resolve=resolve, geom=geom, resolve_rev=rev)


# =============================================================================
# 1. Record floor: start_idx = max(probe_finalize_idx, trigger_idx, parent_floor_idx)
# =============================================================================

def test_record_start_floor_finalize_binds():
    """FC(0,0): trigger 463, finalize 1020, floor 463 → start = max(1020, 463, 463) = 1020
    (the groundtruth's sub 0). Historical fields stay raw."""
    pool = SubStructurePool()
    tables = _tables({(0, 0): (463, 2611)}, {0: +1})
    resolve = _resolver({(CONF, 463): _rs(454, 1020, parent_bos_anchor=96, cond="second_cts_reached",
                                          probe_input=385)})
    geom = _geom(pool, {(+1, 454): 1940, (-1, 1797): None})
    rev = _rev_resolver({(+1, 454): 1797})
    result, _ = _run(
        [_trig(CONF, 0, 0, "first_confluence", 463, +1)],
        pool=pool, tables=tables, resolve=resolve, geom=geom, resolve_rev=rev,
    )
    s0 = _sub(pool, +1, 454)
    (rec,) = s0.records
    assert (rec.trigger_idx, rec.probe_finalize_idx, rec.parent_floor_idx) == (463, 1020, 463)
    assert rec.start_idx == 1020                     # max(1020, 463, 463)
    assert rec.probe_finalize_condition == "second_cts_reached"
    assert rec.parent_bos_anchor_idx == 96           # H1 (FC only)
    assert rec.probe_input_idx == 385                # the probe's M15 input, copied from the resolver
    assert (rec.trigger_sub_sid, rec.relative_dir, rec.sub_id) == (0, CONF, s0.sub_id)
    # sub start = the first non-zero-length record's start_idx (§2.2)
    assert s0.start_idx == 1020
    # own reversal R=1940 > start 1020 → record + sub end there (§4.3 step 6 / §4.4)
    assert _window(rec) == (1020, 1940, "reversal")
    assert _window(s0) == (1020, 1940, "reversal")
    # successor born at R: start = max(finalize R, trigger R, floor 463) = 1940 (§4.3)
    s1 = _sub(pool, -1, 1797)
    assert s1.start_idx == 1940 and s1.records[0].trigger_type == "reversal"
    assert result.unresolved == []


def test_record_copies_probe_input_idx_for_every_resolved_type():
    """PLAN_E E5·4b: the sweep copies `ResolvedStart.probe_input_idx` into the record for EVERY
    resolved type — FC, a sibling type, and a reversal-born successor — and
    `parent_bos_anchor_idx` as given (the FC-only rule lives in the resolvers, pinned in
    test_first_trigger_migration / test_reversal_resolver). The (1,2) scenario of
    `_scenario_3304`; the inputs are synthetic and != `starting_idx` so a type-conditional copy
    (e.g. mirroring the FC-only `parent_bos_anchor_idx` rule) or a `starting_idx` copy is caught.
    (E5·4 landing review, mutation lens: kills the FC-only / no-reversal / no-sibling copies.)"""
    pool = SubStructurePool()
    tables = _tables({(1, 2): (3611, None)}, {1: -1})
    triggers = [
        _trig(CONF, 1, 2, "first_confluence", 3611, -1),
        _trig(CONF, 1, 2, "subsequent_confluence", 3819, -1),
    ]
    resolve = _resolver({
        (CONF, 3611): _rs(3304, 3621, parent_bos_anchor=826, cond="no_retrace", probe_input=3305),
        (CONF, 3819): _rs(3760, 3819, probe_input=3750),     # sibling: no parent BOS anchor
    })
    geom = _geom(pool, {(-1, 3304): None, (-1, 3760): 4200, (+1, 4027): None})

    def rev(sub: PooledStructure, R: int, probe_direction: int):
        # (-1,3760) reverses at 4200 -> successor (+1,4027) probed from input 4000
        if (sub.direction, sub.starting_idx) == (-1, 3760):
            return _rs(4027, R, cond="reversal_handoff", probe_input=4000)
        return ProbeFailure(detail="no successor stubbed", probe_input_idx=None)

    _run(triggers, pool=pool, tables=tables, resolve=resolve, geom=geom, resolve_rev=rev)
    got = sorted(
        (r.trigger_type, r.starting_idx, r.parent_bos_anchor_idx, r.probe_input_idx)
        for r in pool.all_records()
    )
    assert got == [
        ("first_confluence", 3304, 826, 3305),
        ("reversal", 4027, None, 4000),
        ("subsequent_confluence", 3760, None, 3750),
    ]


def test_unresolved_reversal_row_has_no_parent_input():
    """PLAN_E §9.2: a sweep-synthesised reversal has no parent-trigger input — its unresolved row
    carries `parent_input_idx` None and the M15 input the handoff resolver had derived (its
    `ProbeFailure.probe_input_idx`), never the reversing record's H1 input (96 here)."""
    pool = SubStructurePool()
    tables = _tables({(0, 0): (463, 2611)}, {0: +1})
    resolve = _resolver({(CONF, 463): _rs(454, 1020, parent_bos_anchor=96, probe_input=385)})
    geom = _geom(pool, {(+1, 454): 1940})
    rev = _rev_resolver({(+1, 454): ProbeFailure(
        detail="degenerate reversal probe window (1950 >= 1940)", probe_input_idx=1950)})
    result, _ = _run(
        [_trig(CONF, 0, 0, "first_confluence", 463, +1, parent_input=96)],
        pool=pool, tables=tables, resolve=resolve, geom=geom, resolve_rev=rev,
    )
    (u,) = result.unresolved
    assert (u.trigger_type, u.trigger_idx, u.reason) == ("reversal", 1940, "probe_failed")
    assert (u.parent_input_idx, u.probe_input_idx) == (None, 1950)

def test_record_start_floor_trigger_binds_on_cache_hit_inherited_finalize():
    """A re-trigger whose probe was a cache hit inherits the earlier finalize RAW
    (§2.1): subseq_conf(1,2) trigger 3819, inherited finalize 3487, floor 3611 →
    start = max(3487, 3819, 3611) = 3819 — the trigger term binds alone.

    PLAN-AMBIGUITY: §4.3 step 2 / §5.2 assert `finalize_idx == trigger_idx` for
    every non-FC type, while §2.1/§3 say a CACHE-HIT record inherits the cached
    finalize raw (which "normally precedes its own trigger_idx") and that this is
    exactly the case where the trigger term is load-bearing. Both can only hold
    if the assert exempts cache hits (`ResolvedStart.cache_hit=True`), which is
    what this fixture encodes.
    """
    pool = SubStructurePool()
    tables = _tables({(1, 2): (3611, None)}, {1: -1})
    resolve = _resolver({
        (CONF, 3819): _rs(3760, 3487, parent_bos_anchor=954, cond="cached", cache_hit=True),
    })
    geom = _geom(pool, {(-1, 3760): None})
    _run(
        [_trig(CONF, 1, 2, "subsequent_confluence", 3819, -1)],
        pool=pool, tables=tables, resolve=resolve, geom=geom,
    )
    (rec,) = _sub(pool, -1, 3760).records
    assert (rec.trigger_idx, rec.probe_finalize_idx, rec.parent_floor_idx) == (3819, 3487, 3611)
    assert rec.start_idx == 3819                     # max(3487, 3819, 3611)
    assert _sub(pool, -1, 3760).start_idx == 3819


def test_record_start_floor_parent_floor_binds():
    """FC on a retroactive parent (the FC(1,0) shape, non-degenerate here): trigger
    2815, finalize 2844, floor 3611 → start = max(2844, 2815, 3611) = 3611.
    `probe_finalize_idx` / `trigger_idx` are NOT adjusted (historical)."""
    pool = SubStructurePool()
    tables = _tables({(1, 0): (3611, None)}, {1: -1})
    resolve = _resolver({(CONF, 2815): _rs(2803, 2844, parent_bos_anchor=689, cond="no_retrace")})
    geom = _geom(pool, {(-1, 2803): None})
    _run(
        [_trig(CONF, 1, 0, "first_confluence", 2815, -1)],
        pool=pool, tables=tables, resolve=resolve, geom=geom,
    )
    s = _sub(pool, -1, 2803)
    (rec,) = s.records
    assert (rec.trigger_idx, rec.probe_finalize_idx, rec.parent_floor_idx) == (2815, 2844, 3611)
    assert rec.start_idx == 3611                     # max(2844, 2815, 3611)
    assert _window(s) == (3611, None, None)          # no end condition → open


# =============================================================================
# 2. Zero-length exclusion from ALL FOUR aggregations
# =============================================================================

def test_zero_length_record_participates_in_nothing():
    """Sub S=(+1,150), own reversal R=499. Record B (conf FC(0,0) t=103, fin 150):
    start = max(150,103,103) = 150 → live, ends at R. Record A (ctr
    subsequent_counter(0,0) t=499, fin 499, resolving to S): start =
    max(499,499,103) = 499; its known end min((499,reversal),(603,parent_end)) =
    499 <= 499 → zero-length at creation (§4.3 step 6; the sub has NOT ended yet
    at phase 0 of 499).

    Aggregations (§2.1 "participates in nothing"):
    - sub.start_idx = 150 (B), not touched by A;
    - max_start at 499 = 150 (B only). If A counted, max_start = 499 and B's end
      499 > 499 would be FALSE → the sub would never end. Expected: sub ends 499;
    - A's end (499) is not an end candidate (coincides with B's here — the sub's
      end/reason must come from B);
    - lenses() = {confluence} — A's counter lens is NOT added.
    """
    pool = SubStructurePool()
    tables = _tables({(0, 0): (103, 603)}, {0: +1})
    resolve = _resolver({
        (CONF, 103): _rs(150, 150),
        (CTR, 499): _rs(150, 499),                   # converges on S; non-FC: finalize == t
    })
    geom = _geom(pool, {(+1, 150): 499, (-1, 480): None})
    rev = _rev_resolver({(+1, 150): 480})            # S's reversal spawns (−1,480) from B
    result, _ = _run(
        [
            _trig(CONF, 0, 0, "first_confluence", 103, +1),
            _trig(CTR, 0, 0, "subsequent_counter", 499, +1),
        ],
        pool=pool, tables=tables, resolve=resolve, geom=geom, resolve_rev=rev,
    )
    S = _sub(pool, +1, 150)
    assert len(S.records) == 2
    B, A = S.records
    assert (B.lens, B.trigger_idx, B.start_idx) == (CONF, 103, 150)
    assert (A.lens, A.trigger_idx, A.start_idx) == (CTR, 499, 499)
    # A: known-in-advance end at creation
    assert A.is_zero_length
    assert (A.trigger_end_idx, A.end_reason, A.end_idx) == (499, "reversal", 499)
    assert A.sub_id == S.sub_id and A.trigger_sub_sid == 0     # (counter,0,0) scope starts at 0
    # B: live, ends at the own reversal
    assert not B.is_zero_length
    assert _window(B) == (150, 499, "reversal")
    # the four aggregations
    assert S.start_idx == 150
    assert (S.end_idx, S.end_reason) == (499, "reversal")     # max_start = 150 (A excluded)
    assert S.live_records() == [B]
    assert S.lenses() == {CONF}
    # the reversal still spawned from B (live at R); A was not live at R
    succ = _sub(pool, -1, 480)
    assert succ.start_idx == 499 and succ.lenses() == {CONF}
    assert result.unresolved == []
    _assert_sweep_invariants(pool)


# =============================================================================
# 3. Strict '>' with same-candle handover (both directions)
# =============================================================================

def test_handover_2365_same_candle_keeps_sub_continuous():
    """Groundtruth sub 2 (+1,2365): record A = conf reversal (0,0) [2470, 2611]
    parent_end (min(R 2829, end(0,0) 2611)); record B = conf FC (0,1) t=2611,
    fin 2608 → start = max(2608, 2611, 2611) = 2611 — the SAME candle A ends.
    Phase order at 2611: B starts (phase 1) before A ends (phase 3), so at
    phase 4 max_start = 2611 and A's end 2611 > 2611 is false → the sub persists
    and ends at its own reversal 2829 through B. Sub window [2470, 2829].
    """
    pool = SubStructurePool()
    result, _ = _scenario_2365(pool)
    s2 = _sub(pool, +1, 2365)
    A, B = s2.records
    assert (A.lens, A.parent_cycle_id, A.trigger_type, A.trigger_sub_sid) == (CONF, 0, "reversal", 2)
    assert (B.lens, B.parent_cycle_id, B.trigger_type, B.trigger_sub_sid) == (CONF, 1, "first_confluence", 0)
    assert _window(A) == (2470, 2611, "parent_end")
    assert _window(B) == (2611, 2829, "reversal")
    assert (B.trigger_idx, B.probe_finalize_idx, B.parent_floor_idx) == (2611, 2608, 2611)
    assert _window(s2) == (2470, 2829, "reversal")   # continuous across the handover
    assert s2.live_records() == [A, B] and s2.lenses() == {CONF}
    # the rest of the chain, for the record (groundtruth subs 0, 1, 3)
    assert _window(_sub(pool, +1, 454)) == (1020, 1940, "reversal")
    assert _window(_sub(pool, -1, 1797)) == (1940, 2470, "reversal")
    s3 = _sub(pool, -1, 2639)
    assert _window(s3) == (2829, 3611, "parent_end")   # min(R None, end(0,1) 3611)
    assert s3.records[0].trigger_sub_sid == 1           # (conf,0,1): B took 0
    assert _sub(pool, -1, 1797).records[0].relative_dir == CTR   # −1 under parent +1
    assert result.unresolved == []
    _assert_sweep_invariants(pool)


def test_handover_3304_same_dir_replacement_at_new_record_start():
    """Groundtruth subs 7 → 9: rec7 = conf FC(1,2) t=3611 fin 3621 → start 3621;
    rec9 = conf subseq_conf(1,2) t=3819 fin 3819 → start 3819 (a NEW sub, same
    lens/parent/direction). At 3819 rec9 starts (phase 1) → the incumbent rec7
    is queued to end same_dir_replacement at 3819 (§4.3′) → rec7 [3621, 3819]
    ended_by sub 9; sub 7 ends 3819 (3819 > max_start 3621); sub 9 starts 3819
    and ends at its own reversal 4200.
    """
    pool = SubStructurePool()
    result, _ = _scenario_3304(pool)
    s7, s9, s10 = _sub(pool, -1, 3304), _sub(pool, -1, 3760), _sub(pool, +1, 4027)
    (rec7,) = s7.records
    (rec9,) = s9.records
    assert _window(rec7) == (3621, 3819, "same_dir_replacement")
    assert rec7.ended_by_sub_id == s9.sub_id
    assert (rec7.trigger_sub_sid, rec9.trigger_sub_sid) == (0, 1)
    assert _window(s7) == (3621, 3819, "same_dir_replacement")
    assert _window(rec9) == (3819, 4200, "reversal")
    assert _window(s9) == (3819, 4200, "reversal")
    # the handover shares the boundary candle: half-open windows do not overlap
    assert rec7.end_idx == rec9.start_idx == 3819
    # sub 10 born from sub 9's reversal on the confluence lens: tss 2, counter-relative
    (rec10,) = s10.records
    assert (rec10.trigger_type, rec10.start_idx, rec10.trigger_sub_sid, rec10.relative_dir) == (
        "reversal", 4200, 2, CTR)
    assert _window(s10) == (4200, None, None)
    assert result.unresolved == []
    _assert_sweep_invariants(pool)


def test_end_strictly_before_next_start_freezes_late_record_zero_length():
    """The opposite direction of the rule (§4.4 case 1). Sub S=(+1,150):
    record A = conf FC(0,0) t=103 fin 150 → [150, 499] parent_end (end(0,0)=499).
    Record B = conf FC(0,1) t=499, a different-input probe converging on S with
    its OWN finalize 520 → start = max(520, 499, 499) = 520 — strictly AFTER 499.
    At 499 phase 4 only STARTED records count: live = [A], max_start 150, A's end
    499 > 150 → the sub ends at 499 in real time. At 520 phase 1 B finds the sub
    frozen (499 < 520) → B freezes zero-length with the sub's end/reason, linked
    (sub_id set), never active.
    """
    pool = SubStructurePool()
    tables = _tables({(0, 0): (103, 499), (0, 1): (499, 903)}, {0: +1})
    resolve = _resolver({
        (CONF, 103): _rs(150, 150),
        (CONF, 499): _rs(150, 520, cond="second_cts_reached"),   # own finalize (converging probe)
    })
    geom = _geom(pool, {(+1, 150): None})
    result, _ = _run(
        [
            _trig(CONF, 0, 0, "first_confluence", 103, +1),
            _trig(CONF, 0, 1, "first_confluence", 499, +1),
        ],
        pool=pool, tables=tables, resolve=resolve, geom=geom,
    )
    S = _sub(pool, +1, 150)
    A, B = S.records
    assert _window(A) == (150, 499, "parent_end")
    assert _window(S) == (150, 499, "parent_end")        # frozen at A's end, not moved by B
    assert B.start_idx == 520 and B.trigger_sub_sid == 0 and B.sub_id == S.sub_id
    assert B.is_zero_length
    assert (B.trigger_end_idx, B.end_reason) == (499, "parent_end")   # the sub's frozen end
    # PLAN-AMBIGUITY: §4.4 phase 1's freeze pseudo-code writes trigger_end_idx /
    # end_reason only; §2.1 defines end_idx = max(trigger_end_idx, start_idx) for
    # every ended record → 520 encoded here.
    assert B.end_idx == 520
    assert S.live_records() == [A] and S.lenses() == {CONF}
    assert result.unresolved == []
    _assert_sweep_invariants(pool)


# =============================================================================
# 4. Per-lens replacement; cross-lens non-replacement
# =============================================================================

def test_same_lens_replacement_ends_incumbent_at_new_record_start_idx():
    """(conf,0,0,+1): Q = subseq_conf t=403 fin 403 → start 403 (sub (+1,300));
    P = FC t=103 fin 1020 → start 1020 (sub (+1,150)) — P is created FIRST but
    starts LAST. At 1020 P's RECORD_START finds Q active → Q ends at 1020 =
    P's true start_idx (NOT P's trigger_idx 103), reason same_dir_replacement,
    ended_by P. Sub Q ends 1020 (1020 > max_start 403); P stays open.
    """
    pool = SubStructurePool()
    tables = _tables({(0, 0): (103, None)}, {0: +1})
    resolve = _resolver({
        (CONF, 103): _rs(150, 1020, cond="second_cts_reached"),
        (CONF, 403): _rs(300, 403),
    })
    geom = _geom(pool, {(+1, 150): None, (+1, 300): None})
    result, _ = _run(
        [
            _trig(CONF, 0, 0, "first_confluence", 103, +1),
            _trig(CONF, 0, 0, "subsequent_confluence", 403, +1),
        ],
        pool=pool, tables=tables, resolve=resolve, geom=geom,
    )
    P, Q = _sub(pool, +1, 150), _sub(pool, +1, 300)
    (recP,), (recQ,) = P.records, Q.records
    assert (recP.trigger_sub_sid, recQ.trigger_sub_sid) == (0, 1)    # creation order
    assert _window(recQ) == (403, 1020, "same_dir_replacement")
    assert recQ.ended_by_sub_id == P.sub_id
    assert recQ.end_idx == recP.start_idx == 1020 != recP.trigger_idx
    assert _window(Q) == (403, 1020, "same_dir_replacement")
    assert _window(P) == (1020, None, None)
    assert result.unresolved == []
    _assert_sweep_invariants(pool)


def test_cross_lens_reversal_successor_does_not_end_counter_record():
    """INVERTED `test_cross_chain_reversal_ends_active_counter_sub` (§9.1).
    K=(−1,50) has a COUNTER-lens record (first_counter(0,0) t=83 → start 83).
    C=(+1,100) (conf FC t=63 fin 150 → start 150) reverses at 500 → its successor
    R=(−1,480) is born on the CONFLUENCE lens (sticky), direction −1. Replacement
    is same-lens only (§4.3′: `active_record` is keyed on lens) → K's counter
    record is NOT ended; K stays open.
    """
    pool = SubStructurePool()
    tables = _tables({(0, 0): (63, None)}, {0: +1})
    resolve = _resolver({
        (CONF, 63): _rs(100, 150),
        (CTR, 83): _rs(50, 83),
    })
    geom = _geom(pool, {(+1, 100): 500, (-1, 50): None, (-1, 480): None})
    rev = _rev_resolver({(+1, 100): 480})
    result, _ = _run(
        [
            _trig(CONF, 0, 0, "first_confluence", 63, +1),
            _trig(CTR, 0, 0, "first_counter", 83, -1),
        ],
        pool=pool, tables=tables, resolve=resolve, geom=geom, resolve_rev=rev,
    )
    K, C, R = _sub(pool, -1, 50), _sub(pool, +1, 100), _sub(pool, -1, 480)
    assert _window(C) == (150, 500, "reversal")
    (recR,) = R.records
    assert (recR.lens, recR.direction, recR.start_idx, recR.trigger_type) == (CONF, -1, 500, "reversal")
    assert R.lenses() == {CONF} and R.direction == -1
    # K: counter lens, −1, still open — a confluence-lens −1 record never ends it
    (recK,) = K.records
    assert _window(recK) == (83, None, None)
    assert _window(K) == (83, None, None)
    assert result.unresolved == []
    _assert_sweep_invariants(pool)


def test_same_lens_reversal_successor_ends_counter_incumbent():
    """The same-lens case that DOES replace. All on the COUNTER lens, parent +1:
    K=(−1,50) first_counter t=83 → [83, 300] own reversal (R=300) → successor
    C=(+1,280) counter-lens reversal record start 300 (R=500);
    K'=(−1,380) subsequent_counter t=403 → start 403 (no incumbent: K ended 300,
    C is +1); C reverses at 500 → successor R=(−1,480) on the COUNTER lens →
    at 500 its RECORD_START finds K' (ctr, −1, active) → K' ends 500
    same_dir_replacement, ended_by R. trigger_sub_sid in (ctr,0,0): K 0, C 1,
    K' 2, R 3 (creation order).
    """
    pool = SubStructurePool()
    tables = _tables({(0, 0): (63, None)}, {0: +1})
    resolve = _resolver({
        (CTR, 83): _rs(50, 83),
        (CTR, 403): _rs(380, 403),
    })
    geom = _geom(pool, {(-1, 50): 300, (+1, 280): 500, (-1, 380): None, (-1, 480): None})
    rev = _rev_resolver({(-1, 50): 280, (+1, 280): 480})
    result, _ = _run(
        [
            _trig(CTR, 0, 0, "first_counter", 83, -1),
            _trig(CTR, 0, 0, "subsequent_counter", 403, -1),
        ],
        pool=pool, tables=tables, resolve=resolve, geom=geom, resolve_rev=rev,
    )
    K, C, K2, R = _sub(pool, -1, 50), _sub(pool, +1, 280), _sub(pool, -1, 380), _sub(pool, -1, 480)
    assert _window(K) == (83, 300, "reversal")
    assert _window(C) == (300, 500, "reversal")
    (recK2,) = K2.records
    (recR,) = R.records
    assert (recR.lens, recR.direction, recR.start_idx) == (CTR, -1, 500)
    assert _window(recK2) == (403, 500, "same_dir_replacement")
    assert recK2.ended_by_sub_id == R.sub_id
    assert _window(K2) == (403, 500, "same_dir_replacement")
    assert [r.trigger_sub_sid for r in pool.records_for(CTR, 0, 0)] == [0, 1, 2, 3]
    assert _window(R) == (500, None, None)
    assert result.unresolved == []
    _assert_sweep_invariants(pool)


# =============================================================================
# 5. Post-end re-trigger → zero-length with the frozen end; the ABSORBED case
# =============================================================================

def test_post_end_retrigger_in_next_cycle_is_zero_length_with_frozen_end():
    """§4.3 step 6. S=(+1,150): conf FC(0,0) t=103 fin 150 → [150, 400] own
    reversal (R=400; the successor (−1,380) is born at 400). Later FC(0,1)
    t=503 (cache hit, inherited finalize 150) resolves to S → a NEW record in
    (conf,0,1) (different parent cycle → not absorbed) with start =
    max(150, 503, 503) = 503 > S.end 400 → frozen: trigger_end_idx = 400,
    end_reason = reversal, ended_by copied (None — reversal), end_idx =
    max(400, 503) = 503, zero-length. S's lifecycle unchanged.
    """
    pool = SubStructurePool()
    tables = _tables({(0, 0): (103, 503), (0, 1): (503, None)}, {0: +1})
    resolve = _resolver({
        (CONF, 103): _rs(150, 150),
        (CONF, 503): _rs(150, 150, cache_hit=True),
    })
    geom = _geom(pool, {(+1, 150): 400, (-1, 380): None})
    rev = _rev_resolver({(+1, 150): 380})
    result, _ = _run(
        [
            _trig(CONF, 0, 0, "first_confluence", 103, +1),
            _trig(CONF, 0, 1, "first_confluence", 503, +1),
        ],
        pool=pool, tables=tables, resolve=resolve, geom=geom, resolve_rev=rev,
    )
    S = _sub(pool, +1, 150)
    rec0, rec1 = S.records
    assert _window(rec0) == (150, 400, "reversal") and rec0.ended_by_sub_id is None
    assert _window(S) == (150, 400, "reversal")             # unchanged by the re-trigger
    assert (rec1.parent_cycle_id, rec1.trigger_sub_sid, rec1.sub_id) == (1, 0, S.sub_id)
    assert rec1.start_idx == 503
    assert rec1.is_zero_length
    assert (rec1.trigger_end_idx, rec1.end_reason, rec1.ended_by_sub_id) == (400, "reversal", None)
    assert rec1.end_idx == 503                              # max(400, 503)
    assert S.live_records() == [rec0]
    # the successor born at 400 ran to (0,0)'s end
    assert _window(_sub(pool, -1, 380)) == (400, 503, "parent_end")
    assert result.unresolved == []
    _assert_sweep_invariants(pool)


def test_post_end_retrigger_copies_ended_by_from_a_replacement_end():
    """§9.1's second case for the rewritten over-extension test: S replaced in
    (conf,0,0) by R at 400, then re-triggered in the NEXT cycle (conf,0,1) at
    500 → zero-length record carrying S's frozen end AND `ended_by_sub_id`
    copied from the ending record (R's sub_id).
    """
    pool = SubStructurePool()
    tables = _tables({(0, 0): (200, 500), (0, 1): (500, None)}, {0: +1})
    resolve = _resolver({
        (CONF, 200): _rs(100, 200),
        (CONF, 400): _rs(300, 400),
        (CONF, 500): _rs(100, 100, cache_hit=True),
    })
    geom = _geom(pool, {(+1, 100): None, (+1, 300): None})
    result, _ = _run(
        [
            _trig(CONF, 0, 0, "first_confluence", 200, +1),
            _trig(CONF, 0, 0, "subsequent_confluence", 400, +1),
            _trig(CONF, 0, 1, "first_confluence", 500, +1),
        ],
        pool=pool, tables=tables, resolve=resolve, geom=geom,
    )
    S, R = _sub(pool, +1, 100), _sub(pool, +1, 300)
    rec0, rec1 = S.records
    assert _window(rec0) == (200, 400, "same_dir_replacement") and rec0.ended_by_sub_id == R.sub_id
    assert _window(S) == (200, 400, "same_dir_replacement")
    assert rec1.is_zero_length and rec1.start_idx == 500     # max(100, 500, 500)
    assert (rec1.trigger_end_idx, rec1.end_reason, rec1.ended_by_sub_id) == (
        400, "same_dir_replacement", R.sub_id)
    assert rec1.end_idx == 500
    assert _window(R) == (400, 500, "parent_end")           # end(0,0) = 500
    assert result.unresolved == []
    _assert_sweep_invariants(pool)


def test_retrigger_in_same_lens_and_parent_is_absorbed_fixed_semantics():
    """REWRITTEN `test_replaced_then_retriggered_overextension_is_the_known_edge`
    (§9.1): three triggers all in (confluence,0,0). S=(+1,100): FC t=200 fin 200
    → start 200. R=(+1,300): subseq t=400 → start 400 → S's record ends 400
    same_dir_replacement (§4.3′). Re-trigger at 600 resolves to S → same
    (lens, parent) → ABSORBED into S's existing record (§4.3 step 4):
    extra_trigger_idxs == [600], no new record, no trigger_sub_sid consumed.
    S's record AND sub still end at 400 — the over-extension is gone.
    """
    pool = SubStructurePool()
    tables = _tables({(0, 0): (200, None)}, {0: +1})
    resolve = _resolver({
        (CONF, 200): _rs(100, 200),
        (CONF, 400): _rs(300, 400),
        (CONF, 600): _rs(100, 600),
    })
    geom = _geom(pool, {(+1, 100): None, (+1, 300): None})
    result, _ = _run(
        [
            _trig(CONF, 0, 0, "first_confluence", 200, +1),
            _trig(CONF, 0, 0, "subsequent_confluence", 400, +1),
            _trig(CONF, 0, 0, "subsequent_confluence", 600, +1),
        ],
        pool=pool, tables=tables, resolve=resolve, geom=geom,
    )
    S, R = _sub(pool, +1, 100), _sub(pool, +1, 300)
    assert len(S.records) == 1
    (recS,) = S.records
    assert recS.extra_trigger_idxs == [600]
    assert _window(recS) == (200, 400, "same_dir_replacement") and recS.ended_by_sub_id == R.sub_id
    assert _window(S) == (200, 400, "same_dir_replacement")
    assert _window(R) == (400, None, None)
    assert len(pool.all()) == 2 and len(pool.records_for(CONF, 0, 0)) == 2
    # trigger_sub_sid: S took 0, R took 1, the absorbed re-trigger consumed none → next is 2
    assert [r.trigger_sub_sid for r in pool.records_for(CONF, 0, 0)] == [0, 1]
    assert pool.next_trigger_sub_sid(CONF, 0, 0) == 2
    assert result.unresolved == []
    _assert_sweep_invariants(pool)


# =============================================================================
# 6. Reversal spawn only when a live record covers R
# =============================================================================

def test_reversal_spawn_one_successor_with_a_record_per_live_lens():
    """W=(+1,100) conf FC t=63 fin 120 → [120, 300] (R=300) → X=(−1,280) born on
    the CONFLUENCE lens at 300. first_counter(0,0) t=403 (d=−1, fin 403)
    converges on X → a COUNTER-lens record of X [403, …]. X reverses at 800 with
    BOTH records live → two synthesised reversal triggers (one per live lens)
    → they dedup into ONE successor sub Y=(+1,780) with a record per lens
    (conf tss 2, ctr tss 1). X ends 800 (max_start 403 < 800).
    """
    pool = SubStructurePool()
    tables = _tables({(0, 0): (63, None)}, {0: +1})
    resolve = _resolver({
        (CONF, 63): _rs(100, 120),
        (CTR, 403): _rs(280, 403),
    })
    geom = _geom(pool, {(+1, 100): 300, (-1, 280): 800, (+1, 780): None})
    rev = _rev_resolver({(+1, 100): 280, (-1, 280): 780})
    result, _ = _run(
        [
            _trig(CONF, 0, 0, "first_confluence", 63, +1),
            _trig(CTR, 0, 0, "first_counter", 403, -1),
        ],
        pool=pool, tables=tables, resolve=resolve, geom=geom, resolve_rev=rev,
    )
    W, X, Y = _sub(pool, +1, 100), _sub(pool, -1, 280), _sub(pool, +1, 780)
    assert len(pool.all()) == 3
    assert _window(W) == (120, 300, "reversal")
    xc, xk = X.records
    assert (xc.lens, xc.trigger_type, xc.start_idx) == (CONF, "reversal", 300)
    assert (xk.lens, xk.trigger_type, xk.start_idx) == (CTR, "first_counter", 403)
    assert _window(xc) == (300, 800, "reversal") and _window(xk) == (403, 800, "reversal")
    assert _window(X) == (300, 800, "reversal") and X.lenses() == {CONF, CTR}
    # ONE successor, a record per live lens, both reversal-born at R=800
    assert len(Y.records) == 2 and Y.lenses() == {CONF, CTR}
    assert {(r.lens, r.trigger_type, r.trigger_idx, r.start_idx, r.direction) for r in Y.records} == {
        (CONF, "reversal", 800, 800, +1), (CTR, "reversal", 800, 800, +1)}
    by_lens = {r.lens: r for r in Y.records}
    assert (by_lens[CONF].trigger_sub_sid, by_lens[CTR].trigger_sub_sid) == (2, 1)
    assert _window(Y) == (800, None, None)
    # spawned triggers: one at 300 (conf), two at 800 (conf + ctr)
    assert sorted((t.lens, t.trigger_idx, t.direction, t.trigger_type) for t in result.spawned) == [
        (CONF, 300, -1, "reversal"), (CONF, 800, +1, "reversal"), (CTR, 800, +1, "reversal")]
    assert all((t.parent_sid, t.parent_cycle_id) == (0, 0) for t in result.spawned)
    assert result.unresolved == []
    _assert_sweep_invariants(pool)


def test_reversal_spawn_record_whose_parent_ends_at_R_is_still_live():
    """W=(+1,100) conf FC(0,0) t=63 fin 120 → start 120; own reversal R=503 ==
    end(0,0)=503. At 503 phase 0 (REVERSAL_SPAWN, before any RECORD_END) W's
    record is still live at R (§4.3: `trigger_end_idx is None or >= R`) → the
    successor IS spawned and probed.

    PLAN-AMBIGUITY: the successor's record lives in the SAME (S,C) = (0,0) whose
    end is 503 == its start → by §4.3 step 6 it is zero-length at creation
    (parent_end), so the spawn "succeeds" but produces a successor with no live
    record (start_idx None). Encoded literally.
    """
    pool = SubStructurePool()
    tables = _tables({(0, 0): (63, 503), (0, 1): (503, None)}, {0: +1})
    resolve = _resolver({(CONF, 63): _rs(100, 120)})
    geom = _geom(pool, {(+1, 100): 503, (-1, 480): None})
    rev = _rev_resolver({(+1, 100): 480})
    result, _ = _run(
        [_trig(CONF, 0, 0, "first_confluence", 63, +1)],
        pool=pool, tables=tables, resolve=resolve, geom=geom, resolve_rev=rev,
    )
    W = _sub(pool, +1, 100)
    assert W.start_idx == 120 and W.end_idx == 503
    assert rev.calls == [(W.sub_id, 503, -1)]
    assert [(t.lens, t.trigger_idx, t.direction) for t in result.spawned] == [(CONF, 503, -1)]
    Y = _sub(pool, -1, 480)
    (recY,) = Y.records
    assert (recY.trigger_type, recY.trigger_idx, recY.start_idx, recY.parent_cycle_id) == (
        "reversal", 503, 503, 0)
    assert recY.is_zero_length and (recY.trigger_end_idx, recY.end_reason) == (503, "parent_end")
    assert Y.start_idx is None and Y.lenses() == set()
    assert result.unresolved == []
    # Cold review 2026-09-20: inside the sweep this scenario cannot distinguish the
    # closed-right `>= R` of `is_live_at_reversal` from a strict `>` — at phase 0
    # of R the record's trigger_end_idx is still None (phase 3 writes it later
    # the same idx). The `>=` is therefore pinned on the helper directly: a
    # record whose trigger_end_idx == R is live AT R for the spawn rule, but not
    # `is_active_at(R)` (half-open incumbent rule).
    (recW,) = W.records
    assert recW.trigger_end_idx == 503
    assert recW.is_live_at_reversal(503) is True and recW.is_active_at(503) is False
    assert recW.is_live_at_reversal(504) is False


def test_reversal_with_no_live_record_at_R_spawns_nothing():
    """W=(+1,100) conf FC(0,0) t=63 fin 120 → [120, 303] parent_end (end(0,0)=303
    < R=500 → min). The REVERSAL_SPAWN queued at 500 finds no record live at
    500 (W's record ended 303 < 500) → no successor, no record, no probe call,
    no raise. The reversal is inert (§4.3).
    """
    pool = SubStructurePool()
    tables = _tables({(0, 0): (63, 303), (0, 1): (303, None)}, {0: +1})
    resolve = _resolver({(CONF, 63): _rs(100, 120)})
    geom = _geom(pool, {(+1, 100): 500})
    rev = _rev_resolver({(+1, 100): 480})
    result, _ = _run(
        [_trig(CONF, 0, 0, "first_confluence", 63, +1)],
        pool=pool, tables=tables, resolve=resolve, geom=geom, resolve_rev=rev,
    )
    W = _sub(pool, +1, 100)
    assert _window(W) == (120, 303, "parent_end")
    assert W.natural_reversal_idx == 500
    assert pool.all() == [W]
    assert rev.calls == [] and result.spawned == []
    assert result.unresolved == []


@pytest.mark.parametrize("R", [50, 63])       # R < t and R == t
def test_reversal_at_or_before_trigger_idx_queues_nothing_and_record_is_zero_length(R):
    """§4.3 step 3: a structure whose natural reversal R <= trigger_idx t reversed
    before this trigger fired → nothing is queued (never a moment in the past)
    and step 6 makes the record zero-length: FC(0,0) t=63 fin 120 → start 120;
    cands = [(R, reversal)], R <= 120 → trigger_end_idx = R, end_idx = 120.
    """
    pool = SubStructurePool()
    tables = _tables({(0, 0): (63, None)}, {0: +1})
    resolve = _resolver({(CONF, 63): _rs(20, 120)})
    geom = _geom(pool, {(+1, 20): R})
    rev = _rev_resolver({(+1, 20): 10})
    result, _ = _run(
        [_trig(CONF, 0, 0, "first_confluence", 63, +1)],
        pool=pool, tables=tables, resolve=resolve, geom=geom, resolve_rev=rev,
    )
    (S,) = pool.all()
    (rec,) = S.records
    assert rec.start_idx == 120 and rec.is_zero_length
    assert (rec.trigger_end_idx, rec.end_reason, rec.end_idx) == (R, "reversal", 120)
    assert S.start_idx is None and S.end_idx is None and S.lenses() == set()
    assert rev.calls == [] and result.spawned == []
    assert result.unresolved == []


# =============================================================================
# 7. Same-idx start collision
# =============================================================================

def test_same_idx_start_collision_later_seq_replaces_earlier_and_warns():
    """Two records of the same (conf,1,0,−1) start at the same idx for DIFFERENT
    subs (an acausal retroactive-parent case): FC(1,0) t=2815 fin 2844 →
    A=(−1,2803) start max(2844,2815,3611) = 3611; subseq_conf(1,0) t=2963 fin
    2963 → B=(−1,2900) start max(2963,2963,3611) = 3611. §4.2: the later seq
    (B) replaces the earlier (A) at 3611 and `WARNING [sweep] same-idx start
    collision` is logged. A's record ends 3611 == its start → zero-length,
    same_dir_replacement by B; B's record is live and open.

    PLAN-AMBIGUITY: §4.3′'s `active_record` is interval-based (start_idx <= t),
    under which A's own RECORD_START (processed first, lower seq) would ALSO see
    B as an incumbent and end it — "later replaces earlier" requires the
    incumbent lookup to see only records whose RECORD_START has already run
    ("mark active"). Encoded as the plan states the outcome.
    """
    pool = SubStructurePool()
    tables = _tables({(1, 0): (3611, None)}, {1: -1})
    resolve = _resolver({
        (CONF, 2815): _rs(2803, 2844, cond="no_retrace"),
        (CONF, 2963): _rs(2900, 2963),
    })
    geom = _geom(pool, {(-1, 2803): None, (-1, 2900): None})
    result, logs = _run(
        [
            _trig(CONF, 1, 0, "first_confluence", 2815, -1),
            _trig(CONF, 1, 0, "subsequent_confluence", 2963, -1),
        ],
        pool=pool, tables=tables, resolve=resolve, geom=geom,
    )
    A, B = _sub(pool, -1, 2803), _sub(pool, -1, 2900)
    (recA,), (recB,) = A.records, B.records
    assert recA.seq < recB.seq and recA.start_idx == recB.start_idx == 3611
    assert recA.is_zero_length
    assert (recA.trigger_end_idx, recA.end_reason, recA.ended_by_sub_id) == (
        3611, "same_dir_replacement", B.sub_id)
    assert not recB.is_zero_length and _window(recB) == (3611, None, None)
    assert _window(B) == (3611, None, None) and B.lenses() == {CONF}
    assert A.end_idx is None and A.lenses() == set()        # no live record → no lens
    assert any("WARNING [sweep] same-idx start collision" in line for line in logs), logs
    assert result.unresolved == []
    _assert_sweep_invariants(pool)


# =============================================================================
# 8. relative_dir step function incl. a parent-sid flip
# =============================================================================

def test_relative_dir_segments_step_function_with_parent_sid_flip():
    """X=(+1,150) spans H1 sids: (0,0) parent_sd +1, floor 103, end 503 (sid 0
    reverses); (1,0) parent_sd −1, floor 503, end None.
    rec0 = conf FC(0,0) t=103 fin 150 → [150, 503] parent_end, relative_dir
    confluence (+1 == +1); rec1 = ctr first_counter(1,0) t=503 fin 503 → start
    503, relative_dir counter (+1 != −1) — same candle as rec0's end → handover,
    X continuous and open. Segments: at 150 the active record with the latest
    start is rec0 → confluence; at 503 rec1 → counter ⇒
    [(150, confluence), (503, counter)].
    Carry-forward: Z=(−1,180) has a single counter record [203, 503]; at its end
    no record is active → no new segment ⇒ [(203, counter)].
    """
    pool = SubStructurePool()
    tables = _tables({(0, 0): (103, 503), (1, 0): (503, None)}, {0: +1, 1: -1})
    resolve = _resolver({
        (CONF, 103): _rs(150, 150),
        (CTR, 203): _rs(180, 203),
        (CTR, 503): _rs(150, 503),
    })
    geom = _geom(pool, {(+1, 150): None, (-1, 180): None})
    result, _ = _run(
        [
            _trig(CONF, 0, 0, "first_confluence", 103, +1),
            _trig(CTR, 0, 0, "first_counter", 203, -1),
            _trig(CTR, 1, 0, "first_counter", 503, +1),
        ],
        pool=pool, tables=tables, resolve=resolve, geom=geom,
    )
    X, Z = _sub(pool, +1, 150), _sub(pool, -1, 180)
    rec0, rec1 = X.records
    assert (rec0.parent_sid, rec0.relative_dir, _window(rec0)) == (0, CONF, (150, 503, "parent_end"))
    assert (rec1.parent_sid, rec1.relative_dir, _window(rec1)) == (1, CTR, (503, None, None))
    assert _window(X) == (150, None, None)                   # 503 > max_start 503 is false
    assert list(X.relative_dir_segments) == [(150, CONF), (503, CTR)]
    assert X.lenses() == {CONF, CTR}
    (recZ,) = Z.records
    assert _window(recZ) == (203, 503, "parent_end") and recZ.relative_dir == CTR
    assert list(Z.relative_dir_segments) == [(203, CTR)]   # carry-forward at 503
    assert result.unresolved == []
    _assert_sweep_invariants(pool)


# =============================================================================
# 9. trigger_sub_sid per (lens, S, C)
# =============================================================================

def test_trigger_sub_sid_starts_at_zero_per_scope_and_counts_new_subs_only():
    """REWRITTEN `test_two_cycles_sub_sid_resets` (§9.1). Scopes (conf,0,0),
    (ctr,0,0), (conf,0,1), (ctr,0,1) each start at 0; +1 only when a trigger in
    that scope resolves to a NEW sub; an absorbed re-trigger consumes none.
      T1 conf FC(0,0) t=103 → A=(+1,150)            tss 0 (conf,0,0)
      T2 ctr first_counter(0,0) t=203 → B=(−1,180)  tss 0 (ctr,0,0)   independent per lens
      T3 conf subseq(0,0) t=303 → A again           absorbed (extra [303]), no tss
      T4 conf subseq(0,0) t=403 → C=(+1,350)        tss 1 (conf,0,0); replaces A at 403
      T5 conf FC(0,1) t=503 → C (cache hit)          tss 0 (conf,0,1)   resets per cycle
      T6 ctr subseq_ctr(0,1) t=603 → D=(−1,580)     tss 0 (ctr,0,1)
    """
    pool = SubStructurePool()
    tables = _tables({(0, 0): (103, 503), (0, 1): (503, None)}, {0: +1})
    resolve = _resolver({
        (CONF, 103): _rs(150, 150),
        (CTR, 203): _rs(180, 203),
        (CONF, 303): _rs(150, 303),
        (CONF, 403): _rs(350, 403),
        (CONF, 503): _rs(350, 350, cache_hit=True),
        (CTR, 603): _rs(580, 603),
    })
    geom = _geom(pool, {(+1, 150): None, (-1, 180): None, (+1, 350): None, (-1, 580): None})
    triggers = [
        _trig(CONF, 0, 0, "first_confluence", 103, +1),
        _trig(CTR, 0, 0, "first_counter", 203, -1),
        _trig(CONF, 0, 0, "subsequent_confluence", 303, +1),
        _trig(CONF, 0, 0, "subsequent_confluence", 403, +1),
        _trig(CONF, 0, 1, "first_confluence", 503, +1),
        _trig(CTR, 0, 1, "subsequent_counter", 603, -1),
    ]
    result, _ = _run(triggers, pool=pool, tables=tables, resolve=resolve, geom=geom)
    A, B, C, D = _sub(pool, +1, 150), _sub(pool, -1, 180), _sub(pool, +1, 350), _sub(pool, -1, 580)
    assert [s.sub_id for s in pool.all()] == [A.sub_id, B.sub_id, C.sub_id, D.sub_id] == [0, 1, 2, 3]
    (recA,), (recB,), (recD,) = A.records, B.records, D.records
    recC0, recC1 = C.records
    assert (recA.lens, recA.parent_cycle_id, recA.trigger_sub_sid) == (CONF, 0, 0)
    assert (recB.lens, recB.parent_cycle_id, recB.trigger_sub_sid) == (CTR, 0, 0)
    assert recA.extra_trigger_idxs == [303]
    assert (recC0.lens, recC0.parent_cycle_id, recC0.trigger_sub_sid) == (CONF, 0, 1)
    assert (recC1.lens, recC1.parent_cycle_id, recC1.trigger_sub_sid) == (CONF, 1, 0)
    assert (recD.lens, recD.parent_cycle_id, recD.trigger_sub_sid) == (CTR, 1, 0)
    # lifecycle side-checks (derivations in the docstring)
    assert _window(A) == (150, 403, "same_dir_replacement")
    assert _window(B) == (203, 503, "parent_end")
    assert _window(recC0) == (403, 503, "parent_end") and _window(recC1) == (503, None, None)
    assert _window(C) == (403, None, None)                  # handover at 503 (strict >)
    assert _window(D) == (603, None, None)
    # the counters after the sweep (next_trigger_sub_sid CONSUMES — read once each)
    assert pool.next_trigger_sub_sid(CONF, 0, 0) == 2
    assert pool.next_trigger_sub_sid(CTR, 0, 0) == 1
    assert pool.next_trigger_sub_sid(CONF, 0, 1) == 1
    assert pool.next_trigger_sub_sid(CTR, 0, 1) == 1
    assert pool.next_trigger_sub_sid(CTR, 5, 5) == 0        # a fresh scope starts at 0
    # the resolver is called with hi = the trigger's own trigger_idx
    assert [hi for _t, hi in resolve.calls] == [103, 203, 303, 403, 503, 603]
    assert all(hi == t.trigger_idx for t, hi in resolve.calls)
    assert result.unresolved == []
    _assert_sweep_invariants(pool)


# =============================================================================
# 10. Equal-idx end-reason priority on a record
# =============================================================================

def test_end_reason_priority_order_is_reversal_parent_end_replacement():
    """§2.4 / §4.4: reversal > parent_end > same_dir_replacement (lower = wins).
    The base's dict ranks same_dir_replacement above parent_end — pinned here to
    the plan's order (the contract resolves the disagreement toward the plan).
    parent_end vs same_dir_replacement at one idx is unreachable through the
    sweep (a replacement record starting at its own cycle's end candle is
    zero-length by §4.3 step 6), so the ordering is pinned on the table itself.
    """
    assert (
        _END_REASON_PRIORITY["reversal"]
        < _END_REASON_PRIORITY["parent_end"]
        < _END_REASON_PRIORITY["same_dir_replacement"]
    )


def test_equal_idx_reversal_beats_parent_end_on_a_record():
    """S=(+1,150) conf FC(0,0) t=103 fin 150 → start 150; own reversal R=503 ==
    end(0,0)=503 → both end conditions at 503 → reason 'reversal'.

    PLAN-AMBIGUITY: §4.3 step 6's `e = min(cands)` over (idx, reason) tuples
    would tie-break alphabetically ("parent_end" < "reversal"); §2.4/§4.4 say
    reversal > parent_end at an equal idx. Encoded as the priority rule.
    """
    pool = SubStructurePool()
    tables = _tables({(0, 0): (103, 503), (0, 1): (503, None)}, {0: +1})
    resolve = _resolver({(CONF, 103): _rs(150, 150)})
    geom = _geom(pool, {(+1, 150): 503, (-1, 480): None})
    rev = _rev_resolver({(+1, 150): 480})
    _run([_trig(CONF, 0, 0, "first_confluence", 103, +1)],
         pool=pool, tables=tables, resolve=resolve, geom=geom, resolve_rev=rev)
    S = _sub(pool, +1, 150)
    (rec,) = S.records
    assert _window(rec) == (150, 503, "reversal")
    assert _window(S) == (150, 503, "reversal")


def test_equal_idx_reversal_beats_same_dir_replacement_on_a_record():
    """Two RECORD_END entries at one idx for one record (§4.4 phase 3 picks the
    highest priority): A=(+1,150) conf FC(0,0) t=103 fin 150 → start 150, own
    reversal R=503 (entry queued at creation); B=(+1,450) conf subseq(0,0) t=503
    fin 503 → start 503 → at 503 phase 1 B finds A active and queues a
    same_dir_replacement entry at 503 too → A ends 503 with reason 'reversal'
    and ended_by_sub_id None (the winning entry carries no replacer).
    """
    pool = SubStructurePool()
    tables = _tables({(0, 0): (103, None)}, {0: +1})
    resolve = _resolver({
        (CONF, 103): _rs(150, 150),
        (CONF, 503): _rs(450, 503),
    })
    geom = _geom(pool, {(+1, 150): 503, (+1, 450): None, (-1, 480): None})
    rev = _rev_resolver({(+1, 150): 480})
    result, _ = _run(
        [
            _trig(CONF, 0, 0, "first_confluence", 103, +1),
            _trig(CONF, 0, 0, "subsequent_confluence", 503, +1),
        ],
        pool=pool, tables=tables, resolve=resolve, geom=geom, resolve_rev=rev,
    )
    A, B = _sub(pool, +1, 150), _sub(pool, +1, 450)
    (recA,), (recB,) = A.records, B.records
    assert _window(recA) == (150, 503, "reversal") and recA.ended_by_sub_id is None
    assert _window(A) == (150, 503, "reversal")
    assert _window(recB) == (503, None, None) and _window(B) == (503, None, None)
    assert result.unresolved == []
    _assert_sweep_invariants(pool)


# =============================================================================
# 11. A sub with no live record
# =============================================================================

def test_sub_with_only_zero_length_records_has_no_start_and_no_lens():
    """S=(+1,200), own reversal R=280, found by FC(0,0) t=263 fin 270 under floor
    303 → start = max(270, 263, 303) = 303; cands [(280, reversal)], 280 <= 303
    → zero-length. R=280 > t=263 → a REVERSAL_SPAWN is queued at 280, where no
    record is live → nothing. The sub has no live record: start_idx None,
    end_idx None, lenses() == set() — logged, not rendered (§4.5). No raise.
    """
    pool = SubStructurePool()
    tables = _tables({(0, 0): (303, None)}, {0: +1})
    resolve = _resolver({(CONF, 263): _rs(200, 270)})
    geom = _geom(pool, {(+1, 200): 280})
    rev = _rev_resolver({(+1, 200): 260})
    result, _ = _run(
        [_trig(CONF, 0, 0, "first_confluence", 263, +1)],
        pool=pool, tables=tables, resolve=resolve, geom=geom, resolve_rev=rev,
    )
    (S,) = pool.all()
    (rec,) = S.records
    assert rec.is_zero_length and (rec.start_idx, rec.trigger_end_idx, rec.end_reason) == (
        303, 280, "reversal")
    assert S.start_idx is None and S.end_idx is None and S.end_reason is None
    assert S.lenses() == set() and S.live_records() == []
    assert rev.calls == [] and result.spawned == []
    assert result.unresolved == []
    _assert_sweep_invariants(pool)


# =============================================================================
# 12. Post-sweep invariants
# =============================================================================

def test_post_sweep_invariants_hold_on_the_groundtruth_chains():
    """§4.5: every non-zero-length record has start_idx < end_idx or end_idx
    None; every started sub has end_idx None or > start_idx; per
    (lens,S,C,direction) the live windows are non-overlapping half-open
    intervals — a handover shares the boundary candle (rec7 [3621,3819) /
    rec9 [3819,4200)). Checked on both groundtruth chains in one pool.
    """
    pool = SubStructurePool()
    _scenario_2365(pool)
    _scenario_3304(pool)
    assert len(pool.all()) == 7                      # 454, 1797, 2365, 2639, 3304, 3760, 4027
    assert len(pool.all_records()) == 8
    assert all(not r.is_zero_length for r in pool.all_records())
    _assert_sweep_invariants(pool)
    # the shared boundary candle, explicitly
    (rec7,), (rec9,) = _sub(pool, -1, 3304).records, _sub(pool, -1, 3760).records
    assert rec7.end_idx == rec9.start_idx == 3819
    assert rec7.is_active_at(3818) and not rec7.is_active_at(3819)
    assert rec9.is_active_at(3819)
    # seq is global creation order
    seqs = [r.seq for r in pool.all_records()]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


# =============================================================================
# 13. Triggers are independent
# =============================================================================

@pytest.mark.parametrize(
    "fc_value, fc_pending, exp_reason, exp_detail_fragment, exp_probe_input",
    [
        (ProbeFailure(detail="no CTS reference zone", probe_input_idx=90), False, "probe_failed",
         "no CTS reference zone", 90),
        (None, False, "probe_failed", "None", None),
        ("never-called", True, "pending", None, None),
    ],
)
def test_subsequent_is_processed_when_the_cycle_first_trigger_failed(
    fc_value, fc_pending, exp_reason, exp_detail_fragment, exp_probe_input,
):
    """§4.3 "Triggers are independent": FC(0,0) t=103 fails (resolver returns
    ProbeFailure / None) or is pending (status != finalized → never probed) →
    UnresolvedTrigger with the matching reason/detail; the subsequent_confluence
    at 403 still resolves and builds its record (tss 0 — the failed FC consumed
    no trigger_sub_sid, no sub_id). One frame per input column (PLAN_E §9.2):
    every row carries the trigger's H1 `parent_input_idx` (25); `probe_input_idx`
    is only the resolver's M15 value — None when it gave none, never the H1 input.
    """
    pool = SubStructurePool()
    tables = _tables({(0, 0): (103, None)}, {0: +1})
    resolve = _resolver({
        (CONF, 103): fc_value,
        (CONF, 403): _rs(350, 403),
    })
    geom = _geom(pool, {(+1, 350): None})
    result, logs = _run(
        [
            _trig(CONF, 0, 0, "first_confluence", 103, +1, pending=fc_pending, parent_input=25),
            _trig(CONF, 0, 0, "subsequent_confluence", 403, +1),
        ],
        pool=pool, tables=tables, resolve=resolve, geom=geom,
    )
    assert result.unresolved is pool.unresolved
    assert len(result.unresolved) == 1
    u = result.unresolved[0]
    assert isinstance(u, UnresolvedTrigger)
    assert (u.lens, u.parent_sid, u.parent_cycle_id, u.trigger_type, u.trigger_idx, u.direction) == (
        CONF, 0, 0, "first_confluence", 103, +1)
    assert u.reason == exp_reason
    assert u.parent_input_idx == 25
    assert u.probe_input_idx == exp_probe_input
    if exp_detail_fragment is not None:
        assert exp_detail_fragment in u.detail
    if fc_pending:
        assert [t.trigger_idx for t, _hi in resolve.calls] == [403]    # pending → never probed
    else:
        assert [t.trigger_idx for t, _hi in resolve.calls] == [103, 403]
    assert any("UNRESOLVED (skipping)" in line for line in logs), logs
    # the subsequent still built
    (S,) = pool.all()
    assert S.sub_id == 0 and S.starting_idx == 350
    (rec,) = S.records
    assert (rec.trigger_type, rec.trigger_sub_sid, _window(rec)) == ("subsequent_confluence", 0, (403, None, None))
    assert _window(S) == (403, None, None)


# =============================================================================
# 14. Degenerate parent cycle; geometry_failed
# =============================================================================

def test_degenerate_cycle_trigger_is_logged_and_never_probed():
    """§4.3 step 1 / §17.7: (1,0) floor 3611 vs end 2995 (inverted) and (1,1)
    floor 3611 vs end 3611 (zero) are degenerate → their FC triggers become
    UnresolvedTrigger(reason=degenerate_parent_cycle) whose detail names the
    floor/end; the resolver and the geometry builder are never called; the pool
    stays empty (no sub_id consumed). (1,2) is real but has no trigger here.
    """
    pool = SubStructurePool()
    tables = _tables({(1, 0): (3611, 2995), (1, 1): (3611, 3611), (1, 2): (3611, None)}, {1: -1})
    assert tables.degenerate == {(1, 0): True, (1, 1): True, (1, 2): False}
    resolve = _resolver({
        (CONF, 2815): _rs(2803, 2844),
        (CONF, 2995): _rs(2915, 3047),
    })
    geom = _geom(pool, {(-1, 2803): None, (-1, 2915): None})
    result, logs = _run(
        [
            _trig(CONF, 1, 0, "first_confluence", 2815, -1, parent_input=689),
            _trig(CONF, 1, 1, "first_confluence", 2995, -1, parent_input=728),
        ],
        pool=pool, tables=tables, resolve=resolve, geom=geom,
    )
    assert resolve.calls == [] and geom.calls == []
    assert pool.all() == []
    assert [(u.parent_sid, u.parent_cycle_id, u.trigger_idx, u.reason) for u in result.unresolved] == [
        (1, 0, 2815, "degenerate_parent_cycle"), (1, 1, 2995, "degenerate_parent_cycle")]
    u10, u11 = result.unresolved
    assert "3611" in u10.detail and "2995" in u10.detail     # "floor 3611 >= end 2995"
    assert "3611" in u11.detail
    # never probed → the H1 input only; no M15 input (PLAN_E §9.2)
    assert (u10.direction, u10.trigger_type, u10.parent_input_idx, u10.probe_input_idx) == (
        -1, "first_confluence", 689, None)
    assert sum("UNRESOLVED (skipping)" in line for line in logs) == 2, logs


def test_geometry_failed_consumes_no_sub_id_and_later_trigger_can_build_the_key():
    """§4.3 step 3 / §5.4: FC(1,2) t=3611 → (−1,3304) but the geometry build
    fails → UnresolvedTrigger(reason=geometry_failed), NO pool entry (pool.all()
    unchanged), no sub_id consumed. A later subseq_conf(1,2) t=3819 resolving to
    the same key builds it fresh: sub_id 0, one record (tss 0, trigger 3819) —
    no false cache hit on a geometry-less entry.
    """
    pool = SubStructurePool()
    tables = _tables({(1, 2): (3611, None)}, {1: -1})
    resolve = _resolver({
        (CONF, 3611): _rs(3304, 3621, cond="no_retrace", probe_input=3305),
        (CONF, 3819): _rs(3304, 3819),
    })
    geom = _geom(pool, {(-1, 3304): None}, fail_once={(-1, 3304)})
    result, logs = _run(
        [
            _trig(CONF, 1, 2, "first_confluence", 3611, -1, parent_input=826),
            _trig(CONF, 1, 2, "subsequent_confluence", 3819, -1),
        ],
        pool=pool, tables=tables, resolve=resolve, geom=geom,
    )
    assert [(k.direction, k.starting_idx) for k, _b in geom.calls] == [(-1, 3304), (-1, 3304)]
    # the row keeps both inputs: the trigger's H1 826 and the resolver's M15 3305 (PLAN_E §9.2)
    assert [(u.trigger_idx, u.reason, u.parent_input_idx, u.probe_input_idx)
            for u in result.unresolved] == [(3611, "geometry_failed", 826, 3305)]
    (S,) = pool.all()
    assert S.sub_id == 0 and S.starting_idx == 3304
    (rec,) = S.records
    assert (rec.trigger_type, rec.trigger_idx, rec.trigger_sub_sid, rec.start_idx) == (
        "subsequent_confluence", 3819, 0, 3819)
    assert _window(S) == (3819, None, None)
    assert any("UNRESOLVED (skipping)" in line and "geometry_failed" in line for line in logs), logs


# =============================================================================
# 15. Cold-review gaps (2026-09-20) — phase-0 ordering, the WARNING paths, the
#     raise-don't-degrade asserts, the DEFAULT reversal synthesiser, spawn
#     liveness on a record that outlives its sub, the same-idx collision path
#     (Plan C §4.2 / §4.3 / §4.5; PART4 §17.6 / §17.7)
# =============================================================================

def _resolver_by_event_idx(mapping: Dict[int, Any]):
    """`resolve_start(trigger, hi)` stub keyed on `trigger_event_idx` — for
    fixtures where several triggers share `(lens, trigger_idx)` but must
    resolve to DIFFERENT subs. Records every call on `.calls`."""
    calls: List[Tuple[SweepTrigger, int]] = []

    def resolve_start(trigger: SweepTrigger, hi: int):
        calls.append((trigger, hi))
        return mapping[trigger.trigger_event_idx]

    resolve_start.calls = calls          # type: ignore[attr-defined]
    return resolve_start


def test_reversal_spawn_sorts_before_trigger_fire_at_the_same_idx():
    """§4.2: `REVERSAL_SPAWN` sorts BEFORE `TRIGGER_FIRE` at the same idx — "its
    successor must exist before an H1 trigger at the same candle runs its
    sibling read". X=(+1,100) conf FC(0,0) t=63 fin 120 → start 120, own
    reversal R=303 → a REVERSAL_SPAWN is queued at 303. An H1 first_counter(0,0)
    fires at trigger_idx == 303 (d=−1) and its resolver converges on the
    successor (−1,280) — the real "sibling read sees the just-born successor"
    case. The resolver snapshots `pool.all()` at call time: when the H1 trigger
    resolves, the successor MUST already be in the pool (spawn ran first).

    The end state is order-independent (the successor is built by whichever
    moment runs first and the other dedups onto it), so the snapshot and the
    log order are the discriminating assertions.
    """
    pool = SubStructurePool()
    tables = _tables({(0, 0): (63, None)}, {0: +1})
    snapshots: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    answers = {(CONF, 63): _rs(100, 120), (CTR, 303): _rs(280, 303)}

    def resolve(trigger: SweepTrigger, hi: int):
        snapshots[(trigger.lens, trigger.trigger_type)] = [
            (s.direction, s.starting_idx) for s in pool.all()]
        return answers[(trigger.lens, trigger.trigger_idx)]

    geom = _geom(pool, {(+1, 100): 303, (-1, 280): None})
    rev = _rev_resolver({(+1, 100): 280})
    result, logs = _run(
        [
            _trig(CONF, 0, 0, "first_confluence", 63, +1),
            _trig(CTR, 0, 0, "first_counter", 303, -1),
        ],
        pool=pool, tables=tables, resolve=resolve, geom=geom, resolve_rev=rev,
    )
    # the successor existed when the same-idx H1 trigger resolved (spawn ran first)
    assert snapshots[(CTR, "first_counter")] == [(+1, 100), (-1, 280)]
    assert snapshots[(CONF, "first_confluence")] == []            # nothing before the FC
    # log order at 303: the spawn line precedes the counter trigger's fire line
    i_spawn = next(i for i, l in enumerate(logs) if l.startswith("[sweep] spawn sub_id=") and "R=303" in l)
    i_fire = next(i for i, l in enumerate(logs)
                  if l.startswith("[sweep] fire lens=counter") and "type=first_counter" in l)
    assert i_spawn < i_fire, logs
    # end state (order-independent): one successor with a record per lens, both start 303
    X, Y = _sub(pool, +1, 100), _sub(pool, -1, 280)
    assert len(pool.all()) == 2
    assert _window(X) == (120, 303, "reversal")
    assert {(r.lens, r.trigger_type, r.start_idx) for r in Y.records} == {
        (CONF, "reversal", 303), (CTR, "first_counter", 303)}
    assert _window(Y) == (303, None, None) and Y.lenses() == {CONF, CTR}
    assert [(t.lens, t.trigger_idx, t.direction) for t in result.spawned] == [(CONF, 303, -1)]
    assert result.unresolved == []
    _assert_sweep_invariants(pool)


def test_trigger_fire_order_lens_then_type_at_the_same_idx():
    """§4.2 order key within phase 0 for TRIGGER_FIRE = `(lens_rank, type_rank,
    trigger_event_idx)`: confluence (0) before counter (1); `first_*` (0) before
    `subsequent_*` (1). Four triggers at trigger_idx 503 (all `trigger_event_idx`
    125 → the third term ties) are handed to the sweep in FULLY REVERSED order;
    the resolver must be called [conf first, conf subsequent, ctr first, ctr
    subsequent]. Any missing term would leave (part of) the input order intact.

    Both conf triggers resolve to (+1,100) and both ctr triggers to (−1,50), so
    each `subsequent_*` is absorbed into its `first_*`'s record (§4.3 step 4) —
    the absorbing direction (subsequent absorbed INTO first, `extra_trigger_idxs
    == [503]` on the first's record) is itself a consequence of the order.
    """
    pool = SubStructurePool()
    tables = _tables({(0, 0): (103, None)}, {0: +1})
    resolve = _resolver({(CONF, 503): _rs(100, 503), (CTR, 503): _rs(50, 503)})
    geom = _geom(pool, {(+1, 100): None, (-1, 50): None})
    triggers = [
        _trig(CTR, 0, 0, "subsequent_counter", 503, -1),
        _trig(CTR, 0, 0, "first_counter", 503, -1),
        _trig(CONF, 0, 0, "subsequent_confluence", 503, +1),
        _trig(CONF, 0, 0, "first_confluence", 503, +1),
    ]
    assert len({t.trigger_event_idx for t in triggers}) == 1          # third term ties
    result, _ = _run(triggers, pool=pool, tables=tables, resolve=resolve, geom=geom)
    assert [(t.lens, t.trigger_type) for t, _hi in resolve.calls] == [
        (CONF, "first_confluence"), (CONF, "subsequent_confluence"),
        (CTR, "first_counter"), (CTR, "subsequent_counter"),
    ]
    C, K = _sub(pool, +1, 100), _sub(pool, -1, 50)
    (recC,), (recK,) = C.records, K.records
    assert (recC.trigger_type, recC.extra_trigger_idxs) == ("first_confluence", [503])
    assert (recK.trigger_type, recK.extra_trigger_idxs) == ("first_counter", [503])
    assert len(pool.all()) == 2 and len(pool.all_records()) == 2
    assert result.unresolved == []
    _assert_sweep_invariants(pool)


def test_trigger_fire_order_equal_lens_and_type_ascends_by_trigger_event_idx():
    """§4.2's third term: with equal lens AND type at one trigger_idx, the lower
    `trigger_event_idx` fires first. Two subsequent_confluence(0,0) at
    trigger_idx 503 with trigger_event_idx 130 (given FIRST) and 125 → resolver
    call order [125, 130]. Both resolve to (+1,300): the 125 one creates the
    record (trigger_sub_sid 0), the 130 one is absorbed (§4.3 step 4).
    (`trigger_event_idx` is an ORDER KEY only here — the sweep has no df and
    cannot check LOH consistency; see the module docstring.)
    """
    pool = SubStructurePool()
    tables = _tables({(0, 0): (103, None)}, {0: +1})
    resolve = _resolver({(CONF, 503): _rs(300, 503)})
    geom = _geom(pool, {(+1, 300): None})
    triggers = [
        SweepTrigger(lens=CONF, parent_sid=0, parent_cycle_id=0, trigger_type="subsequent_confluence",
                     trigger_idx=503, direction=+1, trigger_event_idx=130),
        SweepTrigger(lens=CONF, parent_sid=0, parent_cycle_id=0, trigger_type="subsequent_confluence",
                     trigger_idx=503, direction=+1, trigger_event_idx=125),
    ]
    result, _ = _run(triggers, pool=pool, tables=tables, resolve=resolve, geom=geom)
    assert [t.trigger_event_idx for t, _hi in resolve.calls] == [125, 130]
    (S,) = pool.all()
    (rec,) = S.records
    assert rec.trigger_sub_sid == 0 and rec.extra_trigger_idxs == [503]
    assert _window(S) == (503, None, None)
    assert result.unresolved == []


@pytest.mark.parametrize(
    "first_bos0, second_bos0, expect_warn",
    [
        (0.5600, 0.5601, True),           # |delta| = 1e-4 > 1e-9 → WARNING
        (0.5600, 0.5600, False),          # equal → no line
        (0.5600, 0.5600 + 1e-12, False),  # |delta| = 1e-12 <= 1e-9 → no line (tolerance, not equality)
        (0.5600, None, False),            # probe value None → no line
        (None, 0.5600, False),            # pool value None → no line
    ],
)
def test_bos0_inner_mismatch_on_dedup_hit_warns_and_keeps_the_first(first_bos0, second_bos0, expect_warn):
    """§4.3 step 3: on a dedup hit (`created=False`) whose resolver `bos0_inner`
    differs from the pool's by more than 1e-9, log `WARNING [sweep] bos0_inner
    mismatch` and keep the first (never raise — the same (dir, start) can be
    reached via an ad-hoc BOS_0 and via a sibling CTS-zone inner). No line when
    the two are equal within tolerance, or when either side is None.

    FC(0,0) t=103 → (+1,150) built with `first_bos0`; subseq_conf(0,0) t=303
    resolves to the same key with `second_bos0` → hit → (absorbed) → the
    comparison runs. Exactly one hit → at most one line.
    """
    pool = SubStructurePool()
    tables = _tables({(0, 0): (103, None)}, {0: +1})
    resolve = _resolver({
        (CONF, 103): _rs(150, 150, bos0=first_bos0),
        (CONF, 303): _rs(150, 303, bos0=second_bos0),
    })
    geom = _geom(pool, {(+1, 150): None})
    result, logs = _run(
        [
            _trig(CONF, 0, 0, "first_confluence", 103, +1),
            _trig(CONF, 0, 0, "subsequent_confluence", 303, +1),
        ],
        pool=pool, tables=tables, resolve=resolve, geom=geom,
    )
    (S,) = pool.all()
    assert [b for _k, b in geom.calls] == [first_bos0, second_bos0]      # both reached the builder
    assert S.bos0_inner == first_bos0                                   # the first is kept
    warn = [l for l in logs if "WARNING [sweep] bos0_inner mismatch" in l]
    assert len(warn) == (1 if expect_warn else 0), logs
    if expect_warn:
        assert f"sub_id={S.sub_id}" in warn[0] and "keeping the first" in warn[0]
    # never raised; the second trigger was absorbed as usual
    (rec,) = S.records
    assert rec.extra_trigger_idxs == [303]
    assert result.unresolved == []


def test_trigger_in_a_cycle_without_cts_established_raises():
    """§3 "Asserts (raise, do not degrade)": every (S,C) with a trigger has a
    CTS_ESTABLISHED. Tables know only (0,0); a subseq_conf in (0,1) reaches
    `_resolve_and_record` → `tables.has_cycle(0,1)` is False → AssertionError
    naming the missing parent, BEFORE any probe (resolver never called)."""
    pool = SubStructurePool()
    tables = _tables({(0, 0): (103, None)}, {0: +1})
    assert not tables.has_cycle(0, 1)
    resolve = _resolver({(CONF, 403): _rs(350, 403)})
    geom = _geom(pool, {(+1, 350): None})
    with pytest.raises(AssertionError, match=r"no parent CTS_ESTABLISHED"):
        _run([_trig(CONF, 0, 1, "subsequent_confluence", 403, +1)],
             pool=pool, tables=tables, resolve=resolve, geom=geom)
    assert resolve.calls == [] and geom.calls == [] and pool.all() == []


def test_pending_first_confluence_in_a_cycle_without_cts_is_logged_not_raised():
    """§4.1: a first_confluence with status != finalized →
    UnresolvedTrigger(reason="pending") — the pending check runs in phase 0
    BEFORE the cycle assert, so a pending FC whose (S,C) has no
    CTS_ESTABLISHED (tables know only (0,0); the trigger is in (0,1)) is logged
    "pending" and nothing raises. Never probed, nothing built."""
    pool = SubStructurePool()
    tables = _tables({(0, 0): (103, None)}, {0: +1})
    assert not tables.has_cycle(0, 1)
    resolve = _resolver({(CONF, 403): _rs(350, 403)})
    geom = _geom(pool, {(+1, 350): None})
    result, logs = _run(
        [_trig(CONF, 0, 1, "first_confluence", 403, +1, pending=True, parent_input=99)],
        pool=pool, tables=tables, resolve=resolve, geom=geom,
    )
    (u,) = result.unresolved
    assert (u.reason, u.lens, u.parent_sid, u.parent_cycle_id, u.trigger_idx) == (
        "pending", CONF, 0, 1, 403)
    assert (u.parent_input_idx, u.probe_input_idx) == (99, None)     # never probed (PLAN_E §9.2)
    assert resolve.calls == [] and geom.calls == [] and pool.all() == []
    assert any("UNRESOLVED (skipping) reason=pending" in l for l in logs), logs


def test_missing_parent_sd_for_a_sid_raises():
    """§3 / §4.3 step 5: `relative_dir` needs `parent_sd[S]`; a sid with no
    direction in the tables is an assert (raise, do not degrade), not a default.
    Tables built for (0,0) then stripped of `parent_sd` (frozen → `replace`)."""
    pool = SubStructurePool()
    tables = _dc_replace(_tables({(0, 0): (103, None)}, {0: +1}), parent_sd={})
    assert tables.has_cycle(0, 0) and not tables.is_degenerate(0, 0)
    resolve = _resolver({(CONF, 103): _rs(150, 150)})
    geom = _geom(pool, {(+1, 150): None})
    with pytest.raises(AssertionError, match=r"no parent_sd for sid 0"):
        _run([_trig(CONF, 0, 0, "first_confluence", 103, +1)],
             pool=pool, tables=tables, resolve=resolve, geom=geom)


def _multi_tf_trigger(**overrides: Any) -> MultiTFTrigger:
    """A real `MultiTFTrigger` (the production record source)."""
    kwargs: Dict[str, Any] = dict(
        parent_tf="H1", parent_sid=0, parent_cycle_id=0, parent_sd=+1,
        use_case="first_confluence", lower_tf="M15", lower_sd=+1,
        meta={"trigger_event_idx": 15, "probe_end_idx": 40},
    )
    kwargs.update(overrides)
    return MultiTFTrigger(**kwargs)


def test_default_synth_reversal_trigger_builds_a_reversal_multi_tf_trigger():
    """§4.3 spawn: the successor's trigger is synthesised from the SPAWNING
    RECORD's own `source_trigger` by the DEFAULT synthesiser
    (`entity_df_mutation._synth_reversal_trigger` — not injected here): same
    parent linkage + TF, `use_case="reversal"`, `lower_sd = −X.direction`,
    `meta["reversal_apply_idx"] = R`, the source's other meta keys kept.
    X=(+1,100) conf FC(0,0) t=63 fin 120, R=303 → successor (−1,280) at 303.
    """
    pool = SubStructurePool()
    tables = _tables({(0, 0): (63, None)}, {0: +1})
    src = _multi_tf_trigger()
    trig = SweepTrigger(lens=CONF, parent_sid=0, parent_cycle_id=0, trigger_type="first_confluence",
                        trigger_idx=63, direction=+1, trigger_event_idx=15, source=src)
    logs: List[str] = []
    result = run_lifecycle_sweep(
        [trig], pool=pool, tables=tables,
        resolve_start=_resolver({(CONF, 63): _rs(100, 120)}),
        build_geometry=_geom(pool, {(+1, 100): 303, (-1, 280): None}),
        resolve_reversal=_rev_resolver({(+1, 100): 280}),
        parent_path=PARENT, sub_tf=SUB_TF, log=logs.append,       # no synth_reversal_trigger → default
    )
    X, Y = _sub(pool, +1, 100), _sub(pool, -1, 280)
    assert _window(X) == (120, 303, "reversal")
    (st,) = result.spawned
    assert (st.lens, st.trigger_type, st.trigger_idx, st.direction) == (CONF, "reversal", 303, -1)
    s = st.source
    assert isinstance(s, MultiTFTrigger) and s is not src
    assert s.use_case == "reversal"
    assert s.lower_sd == -X.direction == -1
    assert s.meta["reversal_apply_idx"] == 303
    assert (s.parent_sid, s.parent_cycle_id, s.lower_tf) == (src.parent_sid, src.parent_cycle_id, src.lower_tf)
    assert (s.parent_tf, s.parent_sd) == (src.parent_tf, src.parent_sd)
    assert s.meta["trigger_event_idx"] == 15 and s.meta["probe_end_idx"] == 40   # inherited meta kept
    # the successor record carries the synthesised source; the H1 record keeps the input object
    (recX,), (recY,) = X.records, Y.records
    assert recX.source_trigger is src and recY.source_trigger is s
    # the spawning record's source is untouched (replace() copies; meta is a new dict)
    assert src.use_case == "first_confluence" and "reversal_apply_idx" not in src.meta
    assert _window(Y) == (303, None, None) and recY.direction == -1
    assert result.unresolved == []


def test_reversal_spawns_from_a_record_that_outlives_its_ended_sub():
    """§4.3 spawn rule reads RECORD liveness (`is_live_at_reversal`), not the
    sub's window. X=(+1,100) under parent_sd −1 (floor 63, end None):
      A = conf FC(0,0) t=63 fin 120 → start 120;
      B = ctr first_counter(0,0) t=203 fin 203, converging on X → start 203;
      C = conf subseq_conf(0,0) t=503 → Y=(+1,400) → at 503 replaces A
          (same lens/parent/direction, different sub): A [120, 503]
          same_dir_replacement ended_by Y. B (counter lens) is NOT an incumbent.
    Sub X ends at 503 (phase 4: live = {A, B}, max_start 203, A.end 503 > 203)
    — the §4.5 "one lifecycle, drawn on every lens" case — while B runs on
    (trigger_end_idx None until its own end).
    X reverses at R=803 > 503: live-at-R = [B] only (A.trigger_end_idx 503 <
    803) → exactly ONE successor record, lens counter, on B's parent; successor
    start = max(R, R, floor 63) = 803. B then ends 803 reversal.
    """
    pool = SubStructurePool()
    tables = _tables({(0, 0): (63, None)}, {0: -1})
    resolve = _resolver({
        (CONF, 63): _rs(100, 120),
        (CTR, 203): _rs(100, 203),            # converges on X
        (CONF, 503): _rs(400, 503),
    })
    geom = _geom(pool, {(+1, 100): 803, (+1, 400): None, (-1, 780): None})
    rev = _rev_resolver({(+1, 100): 780})
    result, logs = _run(
        [
            _trig(CONF, 0, 0, "first_confluence", 63, +1),
            _trig(CTR, 0, 0, "first_counter", 203, +1),
            _trig(CONF, 0, 0, "subsequent_confluence", 503, +1),
        ],
        pool=pool, tables=tables, resolve=resolve, geom=geom, resolve_rev=rev,
    )
    X, Y, Z = _sub(pool, +1, 100), _sub(pool, +1, 400), _sub(pool, -1, 780)
    A, B = X.records
    assert (A.lens, B.lens) == (CONF, CTR)
    assert _window(A) == (120, 503, "same_dir_replacement") and A.ended_by_sub_id == Y.sub_id
    assert _window(X) == (120, 503, "same_dir_replacement")          # the sub froze at 503 ...
    assert _window(B) == (203, 803, "reversal")                      # ... while its counter record ran on
    assert _window(Y) == (503, None, None)
    # exactly one spawn, from B (counter lens), at R
    assert rev.calls == [(X.sub_id, 803, -1)]
    assert [(t.lens, t.parent_sid, t.parent_cycle_id, t.trigger_idx, t.direction) for t in result.spawned] == [
        (CTR, 0, 0, 803, -1)]
    (recZ,) = Z.records
    assert (recZ.lens, recZ.trigger_type, recZ.trigger_idx, recZ.start_idx, recZ.direction) == (
        CTR, "reversal", 803, 803, -1)
    assert recZ.trigger_sub_sid == 1                                # (ctr,0,0): B took 0
    assert _window(Z) == (803, None, None) and Z.lenses() == {CTR}
    assert sum(l.startswith("[sweep] spawn sub_id=") for l in logs) == 1
    assert result.unresolved == []
    _assert_sweep_invariants(pool)


def test_incumbent_plus_two_same_idx_starters_degrades_with_warning_not_raise():
    """§4.2 collision rule with a THIRD party: an open incumbent I plus two
    same-scope (conf,0,0,+1) records A then B (different subs) starting at the
    same idx 503. Cold review 2026-09-20: this path used to raise; now it
    degrades — "the later seq replaces the earlier" + WARNING.
      I = FC(0,0) t=103 fin 150 → start 150, open.
      A = subseq_conf t=503 (tei 125) → (+1,300) start 503;  B = subseq_conf
      t=503 (tei 126) → (+1,450) start 503.  A's seq < B's seq (fire order).
    Phase 1 at 503, by seq: A starts → incumbent I (started, active) → I's
    replacement queued at 503. B starts → started incumbents = {I, A} (2 → a
    WARNING) → I gets a second replacement entry; A.start_idx == 503 == B's →
    the same-idx collision: A frozen zero-length in phase 1 (trigger_end_idx
    503, same_dir_replacement, ended_by B) + WARNING. Phase 2: sub B starts 503;
    sub A never (A zero-length). Phase 3: I ends 503 same_dir_replacement.
    Phase 4: sub I ends 503 (503 > max_start 150). Post-sweep: scope
    (conf,0,0,+1) live = I [150,503) + B [503, …) — no overlap; <= 1 active.
    `ended_by` on I is either replacer: the plan does not fix the tie between
    two equal-priority entries at one idx (§4.4 phase 3).
    """
    pool = SubStructurePool()
    tables = _tables({(0, 0): (103, None)}, {0: +1})
    resolve = _resolver_by_event_idx({
        25: _rs(150, 150),                 # I: _trig's tei = 103 // 4
        125: _rs(300, 503),                # A
        126: _rs(450, 503),                # B
    })
    geom = _geom(pool, {(+1, 150): None, (+1, 300): None, (+1, 450): None})
    triggers = [
        _trig(CONF, 0, 0, "first_confluence", 103, +1),
        SweepTrigger(lens=CONF, parent_sid=0, parent_cycle_id=0, trigger_type="subsequent_confluence",
                     trigger_idx=503, direction=+1, trigger_event_idx=125),
        SweepTrigger(lens=CONF, parent_sid=0, parent_cycle_id=0, trigger_type="subsequent_confluence",
                     trigger_idx=503, direction=+1, trigger_event_idx=126),
    ]
    result, logs = _run(triggers, pool=pool, tables=tables, resolve=resolve, geom=geom)   # no raise
    I, A, B = _sub(pool, +1, 150), _sub(pool, +1, 300), _sub(pool, +1, 450)
    (recI,), (recA,), (recB,) = I.records, A.records, B.records
    assert recI.seq < recA.seq < recB.seq
    assert (recI.trigger_sub_sid, recA.trigger_sub_sid, recB.trigger_sub_sid) == (0, 1, 2)
    # I: replaced at 503
    assert _window(recI) == (150, 503, "same_dir_replacement")
    assert recI.ended_by_sub_id in {A.sub_id, B.sub_id}
    assert _window(I) == (150, 503, "same_dir_replacement")
    # A: frozen zero-length, ended by B (the later seq replaces the earlier)
    assert recA.is_zero_length
    assert (recA.trigger_end_idx, recA.end_reason, recA.ended_by_sub_id, recA.end_idx) == (
        503, "same_dir_replacement", B.sub_id, 503)
    assert A.start_idx is None and A.end_idx is None and A.lenses() == set()
    # B: live and open
    assert not recB.is_zero_length and _window(recB) == (503, None, None)
    assert _window(B) == (503, None, None) and B.lenses() == {CONF}
    # the collision is WARNING-logged (and the >1-started-incumbents degradation too)
    assert any("WARNING [sweep]" in l and "collision" in l for l in logs), logs
    assert any("WARNING [sweep] 2 started records" in l for l in logs), logs
    # post-sweep: exactly one active record in the scope at 503
    assert pool.active_record(CONF, 0, 0, +1, 503) is recB
    assert pool.active_record(CONF, 0, 0, +1, 502) is recI
    assert result.unresolved == []
    _assert_sweep_invariants(pool)
