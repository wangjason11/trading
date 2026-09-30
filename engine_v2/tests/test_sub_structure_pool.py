"""Unit tests for the sub-structure pool data model (Plan C, PART4 §17.2–§17.5).

Plan C (`plans/PLAN_C_lifecycle_rewrite.md` §2) rewrites the pool's data model:
`TriggerRecord` is a triggered INSTANCE with its own real-time lifecycle
(`start_idx` / `trigger_end_idx` / `end_idx`), `PooledStructure` (= unique sub)
aggregates one lifecycle from its records, and the pool gains the record-scoped
queries the sweep needs (`records_for`, `next_trigger_sub_sid`, `active_record`,
`get_by_id`) plus a value-shaped probe cache. `finalize_lifecycles` /
`select_lifecycle_end` are gone (the sweep in `lifecycle_sweep.py` replaces
them — tested in the sweep-unit / predicted-table files, not here).

Pure logic — no pandas / MS / charts. Names and signatures follow the Plan C
API contract (session scratchpad `PLAN_C_API_CONTRACT.md`).

Survivors (pass at the Plan-B base — they pin API the plan keeps unchanged):
`test_get_or_create_dedups_on_key`, `test_sub_id_is_monotonic_and_stable`
(§9.1: "expected to SURVIVE unchanged"), `test_direction_is_part_of_identity`,
`test_resolve_lens_named_triggers`, `test_resolve_lens_reversal_is_sticky`.
The NEW names are reached through the module object (`ssp.ProbeCacheEntry`)
rather than a module-level `from … import` so the survivors still collect and
pass at the base while every Plan-C test fails on its own missing name.
"""
from __future__ import annotations

import dataclasses
from dataclasses import fields

import pytest

import engine_v2.multitf.sub_structure_pool as ssp
from engine_v2.multitf.sub_structure_pool import (
    LENS_CONFLUENCE,
    LENS_COUNTER,
    PooledStructure,
    StructureKey,
    SubStructurePool,
    TriggerRecord,
    _END_REASON_PRIORITY,
    resolve_lens,
)

PARENT = "H1.main"


def _key(direction, start, parent_path=PARENT, sub_tf="M15") -> StructureKey:
    return StructureKey(parent_path, sub_tf, direction, start)


def _rec(
    pool: SubStructurePool,
    sub: PooledStructure,
    lens: str,
    psid: int,
    pcyc: int,
    *,
    start_idx: int,
    trigger_type: str = "first_confluence",
    trigger_idx=None,
    finalize=None,
    floor=None,
    trigger_sub_sid=None,
    relative_dir: str = "confluence",
    finalize_condition: str = "second_cts_reached",
    parent_bos_anchor_idx=None,
    probe_input_idx=None,
) -> TriggerRecord:
    """Build a §2.1 `TriggerRecord` for `sub` and append it the way the sweep
    does (§4.3 step 5 + step 7): `trigger_sub_sid` from
    `pool.next_trigger_sub_sid(lens, S, C)`, `seq` from `pool.next_seq()`,
    `sub.records.append(rec)`. Lifecycle END fields are left at their defaults
    (None) — tests apply ends with `_end()` afterwards, as phase 3 does.

    The three historical terms default to `start_idx` so that the §2.1 rule
    `start_idx == max(probe_finalize_idx, trigger_idx, parent_floor_idx)` holds
    trivially; when a test passes them explicitly the helper asserts the rule so
    a fixture cannot carry an underivable `start_idx`.

    PLAN-AMBIGUITY: the plan registers a record ONLY via `sub.records.append(rec)`
    (§4.3 step 7) and the API contract lists no pool-level registration call, so
    `records_for` / `all_records` / `active_record` are expected to read through
    `pool.all()` → `sub.records` (single source of truth). If the implementation
    adds a registration method (e.g. `pool.add_record`), the queries must still
    see records appended directly to `sub.records`.
    """
    trigger_idx = start_idx if trigger_idx is None else trigger_idx
    finalize = start_idx if finalize is None else finalize
    floor = start_idx if floor is None else floor
    assert start_idx == max(finalize, trigger_idx, floor), (
        "fixture bug: start_idx must be max(finalize, trigger_idx, floor) (§2.1)"
    )
    if trigger_sub_sid is None:
        trigger_sub_sid = pool.next_trigger_sub_sid(lens, psid, pcyc)
    rec = TriggerRecord(
        lens=lens,
        parent_sid=psid,
        parent_cycle_id=pcyc,
        trigger_sub_sid=trigger_sub_sid,
        sub_id=sub.sub_id,
        trigger_type=trigger_type,
        trigger_idx=trigger_idx,
        probe_finalize_idx=finalize,
        probe_finalize_condition=finalize_condition,
        parent_bos_anchor_idx=parent_bos_anchor_idx,
        probe_input_idx=probe_input_idx,
        starting_idx=sub.starting_idx,
        direction=sub.direction,
        sub_tf=sub.sub_tf,
        relative_dir=relative_dir,
        parent_floor_idx=floor,
        start_idx=start_idx,
        seq=pool.next_seq(),
    )
    sub.records.append(rec)
    return rec


def _end(rec: TriggerRecord, trigger_end_idx: int, reason: str, ended_by=None) -> None:
    """Apply an end the way phase 3 writes it (§4.4): `trigger_end_idx`,
    `end_reason`, `ended_by_sub_id`, and `end_idx = max(trigger_end_idx,
    start_idx)` (§2.1)."""
    rec.trigger_end_idx = trigger_end_idx
    rec.end_reason = reason
    rec.ended_by_sub_id = ended_by
    rec.end_idx = max(trigger_end_idx, rec.start_idx)


# --- Identity / dedup (SURVIVORS — unchanged API, pass at the base) -----------

def test_get_or_create_dedups_on_key():
    pool = SubStructurePool()
    k = _key(-1, 3304)
    s1, created1 = pool.get_or_create(k)
    assert created1 is True
    s2, created2 = pool.get_or_create(k)
    assert created2 is False
    assert s1 is s2                     # same object reused
    assert pool.get(k) is s1


def test_sub_id_is_monotonic_and_stable():
    pool = SubStructurePool()
    a, _ = pool.get_or_create(_key(-1, 3304))
    b, _ = pool.get_or_create(_key(1, 4027))
    _again, created = pool.get_or_create(_key(-1, 3304))
    assert (a.sub_id, b.sub_id) == (0, 1)
    assert created is False             # dedup does NOT bump the counter
    c, _ = pool.get_or_create(_key(1, 2365))
    assert c.sub_id == 2


def test_direction_is_part_of_identity():
    pool = SubStructurePool()
    up, _ = pool.get_or_create(_key(1, 100))
    dn, _ = pool.get_or_create(_key(-1, 100))
    assert up is not dn                 # same start, opposite direction = distinct


# --- Lens attribution (SURVIVORS — `resolve_lens` unchanged) ------------------

def test_resolve_lens_named_triggers():
    assert resolve_lens("first_confluence") == LENS_CONFLUENCE
    assert resolve_lens("subsequent_confluence") == LENS_CONFLUENCE
    assert resolve_lens("first_counter") == LENS_COUNTER
    assert resolve_lens("subsequent_counter") == LENS_COUNTER


def test_resolve_lens_reversal_is_sticky():
    assert resolve_lens("reversal", reversed_from_lens=LENS_CONFLUENCE) == LENS_CONFLUENCE
    assert resolve_lens("reversal", reversed_from_lens=LENS_COUNTER) == LENS_COUNTER
    with pytest.raises(ValueError):
        resolve_lens("reversal")            # requires reversed_from_lens
    with pytest.raises(ValueError):
        resolve_lens("nonsense")


# --- §2.1 TriggerRecord shape --------------------------------------------------

def test_trigger_record_fields_in_contract_order():
    """Field DECLARATION order is the `_triggers.csv` column order (§6.3 /
    contract "columns = the dataclass fields in declaration order")."""
    names = [f.name for f in fields(TriggerRecord)]
    assert names == [
        "lens", "parent_sid", "parent_cycle_id", "trigger_sub_sid",
        "sub_id",
        "trigger_type", "trigger_idx",
        "probe_finalize_idx", "probe_finalize_condition",
        "parent_bos_anchor_idx", "probe_input_idx",
        "starting_idx", "direction", "sub_tf", "relative_dir",
        "parent_floor_idx",
        "start_idx",
        "source_trigger",
        "trigger_end_idx", "end_idx", "end_reason", "ended_by_sub_id",
        "seq", "extra_trigger_idxs",
    ]
    # The rev-1 names are gone (`_dt` → `_idx`; `meta` deleted — §7).
    assert "trigger_dt" not in names
    assert "meta" not in names


def test_trigger_record_defaults_and_mutability():
    """Lifecycle-end fields default to None / `seq` 0 / `extra_trigger_idxs` a
    fresh list; the dataclass is NOT frozen (the sweep writes the lifecycle)."""
    pool = SubStructurePool()
    s, _ = pool.get_or_create(_key(1, 454))
    # start = max(finalize 1020, trigger 463, floor 463) = 1020 (FC(0,0) shape).
    rec = _rec(pool, s, LENS_CONFLUENCE, 0, 0, start_idx=1020,
               trigger_idx=463, finalize=1020, floor=463)
    assert (rec.trigger_end_idx, rec.end_idx, rec.end_reason, rec.ended_by_sub_id) == (
        None, None, None, None,
    )
    assert rec.source_trigger is None
    assert rec.extra_trigger_idxs == []
    assert rec.sub_id == s.sub_id and rec.starting_idx == 454 and rec.direction == 1
    assert rec.sub_tf == "M15"
    assert rec.is_zero_length is False
    # Distinct list objects per record (default_factory, not a shared default).
    rec2 = _rec(pool, s, LENS_COUNTER, 0, 0, start_idx=1100)
    rec.extra_trigger_idxs.append(600)
    assert rec2.extra_trigger_idxs == []
    # Not frozen: phase 3 writes are plain attribute assignments.
    _end(rec, 1940, "reversal")
    assert (rec.trigger_end_idx, rec.end_idx, rec.end_reason) == (1940, 1940, "reversal")


def test_is_zero_length_is_defined_on_trigger_end_idx_not_end_idx():
    """§2.1: `is_zero_length = trigger_end_idx is not None and trigger_end_idx <=
    start_idx` — decidable in phases 1–2 of the idx at which the record is ended,
    BEFORE phase 3 writes `end_idx`. So a record with `trigger_end_idx ==
    start_idx` and `end_idx` still None is already zero-length."""
    pool = SubStructurePool()
    s, _ = pool.get_or_create(_key(-1, 2803))
    rec = _rec(pool, s, LENS_CONFLUENCE, 1, 0, start_idx=3611)
    assert rec.is_zero_length is False              # open record
    rec.trigger_end_idx = 3611                      # == start_idx; end_idx NOT written yet
    assert rec.end_idx is None
    assert rec.is_zero_length is True
    # Strictly earlier end (inverted) is zero-length too.
    rec.trigger_end_idx = 2995
    assert rec.is_zero_length is True
    # A later end is a real window.
    rec.trigger_end_idx = 3612
    assert rec.is_zero_length is False


def test_is_active_at_interval_rule():
    """`is_active_at(t)`: not zero-length, `start_idx <= t`, and
    `trigger_end_idx is None or trigger_end_idx > t` (strict — a record ended AT
    t is not active at t; §2.4)."""
    pool = SubStructurePool()
    s, _ = pool.get_or_create(_key(1, 100))
    rec = _rec(pool, s, LENS_CONFLUENCE, 0, 0, start_idx=200)
    assert rec.is_active_at(199) is False           # before start
    assert rec.is_active_at(200) is True            # at start (inclusive)
    assert rec.is_active_at(10_000) is True         # open → active forever
    _end(rec, 400, "parent_end")
    assert rec.is_active_at(399) is True
    assert rec.is_active_at(400) is False           # strict > on trigger_end_idx
    # Zero-length: never active, even at its own start.
    z = _rec(pool, s, LENS_COUNTER, 0, 0, start_idx=300)
    z.trigger_end_idx = 300
    assert z.is_active_at(300) is False


def test_is_live_at_reversal_uses_inclusive_end():
    """§4.3 spawn rule: a record whose parent ends AT R is still live at R
    (`trigger_end_idx >= R`), unlike `is_active_at` (strict `>`)."""
    pool = SubStructurePool()
    s, _ = pool.get_or_create(_key(-1, 3760))
    rec = _rec(pool, s, LENS_CONFLUENCE, 1, 2, start_idx=3819)
    _end(rec, 4200, "reversal")
    assert rec.is_live_at_reversal(4200) is True    # end == R → live (>=)
    assert rec.is_active_at(4200) is False          # …but not active (>)
    assert rec.is_live_at_reversal(4201) is False   # ended before R
    assert rec.is_live_at_reversal(3818) is False   # not started yet
    assert rec.is_live_at_reversal(3819) is True    # start == R
    # Zero-length participates in nothing.
    z = _rec(pool, s, LENS_COUNTER, 1, 2, start_idx=3900)
    z.trigger_end_idx = 3900
    assert z.is_live_at_reversal(3900) is False


# --- §2.2 PooledStructure --------------------------------------------------------

def test_pooled_structure_defaults():
    pool = SubStructurePool()
    s, _ = pool.get_or_create(_key(1, 3621))
    assert s.records == []
    assert s.geometry is None
    assert s.natural_reversal_idx is None
    assert s.bos0_inner is None
    assert (s.start_idx, s.end_idx, s.end_reason) == (None, None, None)
    assert s.relative_dir_segments == []
    assert s.live_records() == []
    assert s.lenses() == set()
    # Unchanged accessors.
    assert (s.parent_path, s.sub_tf, s.direction, s.starting_idx) == (PARENT, "M15", 1, 3621)
    assert s.dir_key == (PARENT, "M15", 1)


def test_live_records_and_lenses_exclude_zero_length():
    """`live_records()` = non-zero-length records in creation order; `lenses()`
    = union over live records only (§2.2). A counter-lens record that is
    zero-length contributes NO lens."""
    pool = SubStructurePool()
    s, _ = pool.get_or_create(_key(-1, 2639))
    a = _rec(pool, s, LENS_CONFLUENCE, 0, 1, start_idx=2829, trigger_type="reversal",
             relative_dir="counter")
    z = _rec(pool, s, LENS_COUNTER, 0, 1, start_idx=2843, trigger_type="first_counter",
             relative_dir="counter")
    z.trigger_end_idx = 2843                          # zero-length (end == start)
    b = _rec(pool, s, LENS_COUNTER, 0, 1, start_idx=2850, trigger_type="subsequent_counter",
             relative_dir="counter")
    assert s.live_records() == [a, b]
    assert s.lenses() == {LENS_CONFLUENCE, LENS_COUNTER}
    _end(b, 2850, "parent_end")                       # b now zero-length too
    assert s.live_records() == [a]
    assert s.lenses() == {LENS_CONFLUENCE}


def test_sub_whose_only_record_is_zero_length_has_no_lens():
    """§4.5 / §17.5: a sub with no live record has NO lens (logged, not rendered)."""
    pool = SubStructurePool()
    s, _ = pool.get_or_create(_key(-1, 2915))
    rec = _rec(pool, s, LENS_CONFLUENCE, 1, 1, start_idx=3611)
    _end(rec, 3611, "parent_end")                     # end == start → zero-length
    assert rec.is_zero_length is True
    assert s.live_records() == []
    assert s.lenses() == set()
    assert s.records == [rec]                         # …but it IS logged on the sub


# --- §2.3 UnresolvedTrigger ------------------------------------------------------

def test_unresolved_trigger_is_frozen_and_pool_list_starts_empty():
    pool = SubStructurePool()
    assert pool.unresolved == []
    u = ssp.UnresolvedTrigger(
        lens=LENS_CONFLUENCE, parent_sid=1, parent_cycle_id=0,
        trigger_type="first_confluence", trigger_idx=2815, direction=-1,
        parent_input_idx=689, probe_input_idx=None, reason="degenerate_parent_cycle",
        detail="floor 3611 >= end 3611",
    )
    pool.unresolved.append(u)
    assert pool.unresolved == [u]
    with pytest.raises(dataclasses.FrozenInstanceError):
        u.reason = "pending"                          # type: ignore[misc]
    assert [f.name for f in fields(ssp.UnresolvedTrigger)] == [
        "lens", "parent_sid", "parent_cycle_id",
        "trigger_type", "trigger_idx", "direction",
        "parent_input_idx", "probe_input_idx", "reason", "detail",
    ]


# --- §2.4 pool queries -------------------------------------------------------------

def test_get_by_id_hit_and_keyerror():
    pool = SubStructurePool()
    a, _ = pool.get_or_create(_key(-1, 3304))       # sub_id 0
    b, _ = pool.get_or_create(_key(1, 4027))        # sub_id 1
    assert pool.get_by_id(0) is a
    assert pool.get_by_id(1) is b
    with pytest.raises(KeyError):
        pool.get_by_id(2)


def test_next_seq_is_strictly_increasing():
    pool = SubStructurePool()
    s0, s1, s2 = pool.next_seq(), pool.next_seq(), pool.next_seq()
    assert s0 < s1 < s2


def test_next_trigger_sub_sid_starts_at_zero_per_scope_and_consumes():
    """§2.4: starts at 0 per `(lens, parent_sid, parent_cycle_id)`; each call
    CONSUMES; independent per lens and per parent cycle (and per parent sid)."""
    pool = SubStructurePool()
    assert pool.next_trigger_sub_sid(LENS_CONFLUENCE, 0, 0) == 0
    assert pool.next_trigger_sub_sid(LENS_CONFLUENCE, 0, 0) == 1
    assert pool.next_trigger_sub_sid(LENS_CONFLUENCE, 0, 0) == 2
    # Other lens, same parent cycle → its own counter.
    assert pool.next_trigger_sub_sid(LENS_COUNTER, 0, 0) == 0
    # Same lens, next parent cycle → its own counter.
    assert pool.next_trigger_sub_sid(LENS_CONFLUENCE, 0, 1) == 0
    # Same lens, other parent sid → its own counter.
    assert pool.next_trigger_sub_sid(LENS_CONFLUENCE, 1, 0) == 0
    # The first scope kept counting where it left off.
    assert pool.next_trigger_sub_sid(LENS_CONFLUENCE, 0, 0) == 3
    assert pool.next_trigger_sub_sid(LENS_COUNTER, 0, 0) == 1


def test_records_for_is_scoped_and_in_creation_order_including_zero_length():
    """`records_for(lens, S, C)` returns EVERY record of that scope — zero-length
    included — in creation (`seq`) order, NOT grouped by sub (§2.4: "returns
    every record including zero-length ones; callers filter")."""
    pool = SubStructurePool()
    a, _ = pool.get_or_create(_key(1, 100))         # sub_id 0
    b, _ = pool.get_or_create(_key(1, 300))         # sub_id 1
    # Interleave creation across the two subs so seq order != sub order.
    r0 = _rec(pool, b, LENS_CONFLUENCE, 0, 0, start_idx=400)           # seq first, on b
    r1 = _rec(pool, a, LENS_CONFLUENCE, 0, 0, start_idx=200)           # on a
    r2 = _rec(pool, b, LENS_CONFLUENCE, 0, 0, start_idx=500)           # on b, zero-length
    r2.trigger_end_idx = 500
    other_lens = _rec(pool, a, LENS_COUNTER, 0, 0, start_idx=250)
    other_cycle = _rec(pool, a, LENS_CONFLUENCE, 0, 1, start_idx=600)
    other_sid = _rec(pool, a, LENS_CONFLUENCE, 1, 0, start_idx=700)

    assert pool.records_for(LENS_CONFLUENCE, 0, 0) == [r0, r1, r2]
    assert pool.records_for(LENS_COUNTER, 0, 0) == [other_lens]
    assert pool.records_for(LENS_CONFLUENCE, 0, 1) == [other_cycle]
    assert pool.records_for(LENS_CONFLUENCE, 1, 0) == [other_sid]
    assert pool.records_for(LENS_COUNTER, 1, 0) == []
    # trigger_sub_sid was consumed per scope in creation order (helper uses
    # pool.next_trigger_sub_sid): (conf,0,0) → 0,1,2; the other scopes → 0.
    assert [r.trigger_sub_sid for r in (r0, r1, r2)] == [0, 1, 2]
    assert (other_lens.trigger_sub_sid, other_cycle.trigger_sub_sid,
            other_sid.trigger_sub_sid) == (0, 0, 0)


def test_all_records_in_seq_order():
    pool = SubStructurePool()
    a, _ = pool.get_or_create(_key(1, 100))
    b, _ = pool.get_or_create(_key(-1, 150))
    r0 = _rec(pool, b, LENS_COUNTER, 0, 0, start_idx=160)
    r1 = _rec(pool, a, LENS_CONFLUENCE, 0, 0, start_idx=170)
    r2 = _rec(pool, b, LENS_CONFLUENCE, 0, 1, start_idx=180)
    r2.trigger_end_idx = 180                           # zero-length is still listed
    r3 = _rec(pool, a, LENS_CONFLUENCE, 1, 0, start_idx=190)
    assert pool.all_records() == [r0, r1, r2, r3]
    assert [r.seq for r in pool.all_records()] == sorted(r.seq for r in (r0, r1, r2, r3))


def test_active_record_interval_rule():
    """§2.4: active at `at_idx` iff not zero-length, `start_idx <= at_idx`,
    `trigger_end_idx is None or trigger_end_idx > at_idx`. Keyed on
    `(lens, S, C, direction)`; `exclude=` drops one record; > 1 active asserts."""
    pool = SubStructurePool()
    a, _ = pool.get_or_create(_key(1, 100))         # +1
    b, _ = pool.get_or_create(_key(1, 300))         # +1, different sub
    d, _ = pool.get_or_create(_key(-1, 120))        # -1
    ra = _rec(pool, a, LENS_CONFLUENCE, 0, 0, start_idx=200)
    rd = _rec(pool, d, LENS_CONFLUENCE, 0, 0, start_idx=210, relative_dir="counter")
    scope = (LENS_CONFLUENCE, 0, 0, 1)

    assert pool.active_record(*scope, at_idx=199) is None          # nothing started
    assert pool.active_record(*scope, at_idx=200) is ra            # start inclusive
    # Direction is part of the key: -1 record never answers a +1 query.
    assert pool.active_record(LENS_CONFLUENCE, 0, 0, -1, at_idx=250) is rd
    # Lens / parent scope: other scopes are empty.
    assert pool.active_record(LENS_COUNTER, 0, 0, 1, at_idx=250) is None
    assert pool.active_record(LENS_CONFLUENCE, 0, 1, 1, at_idx=250) is None

    # End at 400: active at 399, NOT at 400 (strict > on trigger_end_idx).
    _end(ra, 400, "same_dir_replacement", ended_by=b.sub_id)
    assert pool.active_record(*scope, at_idx=399) is ra
    assert pool.active_record(*scope, at_idx=400) is None

    # Zero-length is excluded even inside what would be its window.
    rz = _rec(pool, b, LENS_CONFLUENCE, 0, 0, start_idx=250)
    rz.trigger_end_idx = 250
    # ra [200,400) is live at 250; if rz were counted there would be TWO
    # active records → AssertionError. Returning ra proves rz is excluded.
    assert pool.active_record(*scope, at_idx=250) is ra

    # Two open records of the same scope for different subs → AssertionError
    # (the ≤1-active invariant, §17.4); `exclude=` is how phase 1 asks for the
    # incumbent while the new record is already marked active.
    rb = _rec(pool, b, LENS_CONFLUENCE, 0, 0, start_idx=380)
    with pytest.raises(AssertionError):
        pool.active_record(*scope, at_idx=390)                     # ra [200,400) + rb [380,…)
    assert pool.active_record(*scope, at_idx=390, exclude=rb) is ra
    assert pool.active_record(*scope, at_idx=390, exclude=ra) is rb
    # After ra's end only rb is active — no assert.
    assert pool.active_record(*scope, at_idx=400) is rb


# --- §5.3 probe cache -------------------------------------------------------------

def _entry(**over) -> "ssp.ProbeCacheEntry":
    base = dict(starting_idx=3304, finalize_idx=3487, finalize_condition="no_retrace",
                bos0_inner=0.5678, probe_end_idx=3623)
    base.update(over)
    return ssp.ProbeCacheEntry(**base)


def test_probe_cache_entry_is_frozen_with_contract_fields():
    e = _entry()
    assert [f.name for f in fields(ssp.ProbeCacheEntry)] == [
        "starting_idx", "finalize_idx", "finalize_condition", "bos0_inner", "probe_end_idx",
        "ref_inner",   # the reference inner the probe ran against (tripwire comparand)
    ]
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.starting_idx = 1                            # type: ignore[misc]


def test_probe_cache_miss_hit_and_first_write_wins():
    """`get_cached_probe` miss → None; `record_probe` then hit → the entry; a
    second write with a DIFFERENT entry raises AssertionError (§5.3: "on a
    re-probe of an identical (input, end_idx) the result must be byte-identical
    — assert"); the SAME entry is a no-op. Key = (parent_path, sub_tf,
    direction, initial_input_idx) — §17.8."""
    pool = SubStructurePool()
    key = (PARENT, "M15", -1, 3300)
    assert pool.get_cached_probe(*key) is None
    e = _entry()
    pool.record_probe(*key, e)
    assert pool.get_cached_probe(*key) == e
    # Same entry → no-op (identical object …)
    pool.record_probe(*key, e)
    # … and an equal-valued entry: frozen dataclasses compare by value.
    # PLAN-AMBIGUITY: "same entry" is read as VALUE equality (dataclass __eq__),
    # not identity — a byte-identical re-probe builds a new entry object.
    pool.record_probe(*key, _entry())
    assert pool.get_cached_probe(*key) == e
    # A different entry for the same key is a contract violation.
    with pytest.raises(AssertionError):
        pool.record_probe(*key, _entry(starting_idx=3305))
    assert pool.get_cached_probe(*key) == e           # first write still wins
    # Other keys are independent: direction and input are both in the key.
    assert pool.get_cached_probe(PARENT, "M15", 1, 3300) is None
    assert pool.get_cached_probe(PARENT, "M15", -1, 3301) is None
    other = _entry(starting_idx=2365, finalize_idx=2608, finalize_condition="second_cts_reached",
                   bos0_inner=None, probe_end_idx=2609)
    pool.record_probe(PARENT, "M15", 1, 2365, other)
    assert pool.get_cached_probe(PARENT, "M15", 1, 2365) == other
    assert pool.get_cached_probe(*key) == e


# --- End-reason priority ------------------------------------------------------------

def test_end_reason_priority_order():
    """Plan C §2.4 / §4.4: `reversal > parent_end > same_dir_replacement`
    (numerically 0, 1, 2 — lower wins at an equal idx). The base's dict had
    `same_dir_replacement` before `parent_end`; the plan's order is the contract."""
    assert _END_REASON_PRIORITY == {
        "reversal": 0,
        "parent_end": 1,
        "same_dir_replacement": 2,
    }
    assert (
        _END_REASON_PRIORITY["reversal"]
        < _END_REASON_PRIORITY["parent_end"]
        < _END_REASON_PRIORITY["same_dir_replacement"]
    )


# --- Deleted API (§2.2 / §2.4 / §7) -------------------------------------------------

def test_deleted_lifecycle_api_is_gone():
    # Module-level: replaced by the sweep (§4).
    assert not hasattr(ssp, "finalize_lifecycles")
    assert not hasattr(ssp, "select_lifecycle_end")
    # Pool: `for_lens` → filter over `lenses()`; the old scalar probe cache
    # (`probe_cached_start` / `record_probe_start`) → `get_cached_probe` /
    # `record_probe` with a `ProbeCacheEntry` value.
    pool = SubStructurePool()
    for name in ("for_lens", "probe_cached_start", "record_probe_start"):
        assert not hasattr(pool, name), name
    # PooledStructure: rev-1 lifecycle scalars + membership helpers + trigger list.
    s, _ = pool.get_or_create(_key(1, 100))
    for name in ("memberships", "earliest_membership", "add_trigger", "trigger_records",
                 "lifecycle_start", "lifecycle_end", "lifecycle_end_reason", "meta"):
        assert not hasattr(s, name), name
    # Kept: the priority table is used by phase 3.
    assert hasattr(ssp, "_END_REASON_PRIORITY")
    # Kept: `knowable_at_idx` (render-side clip key) — keyed on the moment for
    # BOS_CONFIRMED and, since Plan E E3b, CTS_ESTABLISHED / pattern-path
    # CTS_UPDATED; a raw CTS_UPDATED (no confirmed_at) and other types on idx.
    from engine_v2.tests._event_factory import make_bos_confirmed, make_cts_established, make_event
    assert ssp.knowable_at_idx(make_bos_confirmed(bos_anchor_idx=10, confirmed_at=14)) == 14
    assert ssp.knowable_at_idx(make_cts_established(cts_anchor_idx=10, confirmed_at=14)) == 14
    assert ssp.knowable_at_idx(make_event("CTS_UPDATED", 10, via="continuous", confirmed_at=14)) == 14
    assert ssp.knowable_at_idx(make_event("CTS_UPDATED", 10, via="replay_raw")) == 10
    assert ssp.knowable_at_idx(make_event("REVERSAL_CANDIDATE", 10, apply_idx=14)) == 10
