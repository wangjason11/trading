"""Unit tests for the sub-structure pool (Phase 2, PART4_REFACTOR_SPEC.md §17).

Stage 1 — pool identity/dedup, lens attribution, and the lifecycle rule
(`start = min(trigger_dt)`, `end = min(end-candidate >= max(trigger_dt))`),
including the canonical M15 3304 cross-parent-cycle case, the cross-chain
reversal ending, and the documented replaced-then-re-triggered over-extension
edge. Pure logic — no pandas / MS / charts.
"""
from __future__ import annotations

import pytest

from engine_v2.multitf.sub_structure_pool import (
    LENS_CONFLUENCE,
    LENS_COUNTER,
    PooledStructure,
    StructureKey,
    SubStructurePool,
    TriggerRecord,
    finalize_lifecycles,
    resolve_lens,
    select_lifecycle_end,
)

PARENT = "H1.main"


def _tr(trigger_type, dt, psid, pcyc, lens, **meta) -> TriggerRecord:
    return TriggerRecord(
        trigger_type=trigger_type, trigger_dt=dt,
        parent_sid=psid, parent_cycle_id=pcyc, lens=lens, meta=meta,
    )


def _key(direction, start, parent_path=PARENT, sub_tf="M15") -> StructureKey:
    return StructureKey(parent_path, sub_tf, direction, start)


# --- Identity / dedup ---------------------------------------------------------

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


def test_probe_cache_roundtrip_and_setdefault():
    pool = SubStructurePool()
    assert pool.probe_cached_start(PARENT, "M15", -1, 3300) is None
    pool.record_probe_start(PARENT, "M15", -1, 3300, 3304)
    assert pool.probe_cached_start(PARENT, "M15", -1, 3300) == 3304
    # First write wins (a re-probe with different end_idx accepted, §17.7).
    pool.record_probe_start(PARENT, "M15", -1, 3300, 9999)
    assert pool.probe_cached_start(PARENT, "M15", -1, 3300) == 3304


# --- Lens attribution ---------------------------------------------------------

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


def test_sub_on_both_charts_via_lens_union():
    """A unique sub with both a confluence trigger and a counter (reversal-sticky)
    trigger renders on BOTH charts (§17.4) — the 3304 shape."""
    pool = SubStructurePool()
    s, _ = pool.get_or_create(_key(-1, 3304))
    s.add_trigger(_tr("subsequent_confluence", 3350, 1, 1, LENS_CONFLUENCE))
    s.add_trigger(_tr("reversal", 3360, 1, 1, LENS_COUNTER))
    assert s.lenses() == {LENS_CONFLUENCE, LENS_COUNTER}
    assert pool.for_lens(LENS_CONFLUENCE) == [s]
    assert pool.for_lens(LENS_COUNTER) == [s]


def test_memberships_and_earliest():
    pool = SubStructurePool()
    s, _ = pool.get_or_create(_key(-1, 3304))
    s.add_trigger(_tr("subsequent_confluence", 3350, 1, 1, LENS_CONFLUENCE))
    s.add_trigger(_tr("reversal", 3360, 1, 1, LENS_COUNTER))
    s.add_trigger(_tr("first_confluence", 3500, 1, 2, LENS_CONFLUENCE))
    assert s.memberships() == [(1, 1), (1, 2)]      # distinct, first-seen order
    assert s.earliest_membership() == (1, 1)         # earliest trigger_dt


# --- select_lifecycle_end -----------------------------------------------------

def test_select_end_min_at_or_after_max_start():
    end, reason = select_lifecycle_end(
        100, [(50, "parent_end"), (150, "reversal"), (200, "parent_end")],
    )
    assert (end, reason) == (150, "reversal")


def test_select_end_none_when_all_before_max_start():
    assert select_lifecycle_end(100, [(50, "reversal"), (99, "parent_end")]) == (None, None)


def test_select_end_empty():
    assert select_lifecycle_end(100, []) == (None, None)


def test_select_end_tie_breaks_by_reason_priority():
    # reversal (0) beats same_dir_replacement (1) beats parent_end (2) at same idx.
    assert select_lifecycle_end(
        100, [(150, "parent_end"), (150, "reversal"), (150, "same_dir_replacement")],
    ) == (150, "reversal")


# --- finalize_lifecycles ------------------------------------------------------

def test_lifecycle_start_is_earliest_trigger():
    pool = SubStructurePool()
    s, _ = pool.get_or_create(_key(1, 100))
    s.add_trigger(_tr("first_confluence", 300, 0, 0, LENS_CONFLUENCE))
    s.add_trigger(_tr("subsequent_confluence", 150, 0, 0, LENS_CONFLUENCE))
    finalize_lifecycles(pool.all(), parent_end_lookup={})
    assert s.lifecycle_start == 150


def test_same_direction_replacement_ends_prior_sub():
    """New same-(parent,TF,dir) sub ends the active one at its start (§17.6
    ≤1-active-per-direction)."""
    pool = SubStructurePool()
    s1, _ = pool.get_or_create(_key(1, 100))
    s1.add_trigger(_tr("first_confluence", 200, 0, 0, LENS_CONFLUENCE))
    s2, _ = pool.get_or_create(_key(1, 300))
    s2.add_trigger(_tr("subsequent_confluence", 400, 0, 0, LENS_CONFLUENCE))
    finalize_lifecycles(pool.all(), parent_end_lookup={})
    assert (s1.lifecycle_end, s1.lifecycle_end_reason) == (400, "same_dir_replacement")
    assert s2.lifecycle_end is None          # still open (no later same-dir sub)


def test_3304_cross_parent_cycle_stays_continuous():
    """Canonical dedup case: one sub triggered in parent cycles (1,1) AND (1,2).
    The parent-cycle-1-end (between the triggers) must NOT end it — the
    `end >= max(start)` rule keeps it continuous (§17.6)."""
    pool = SubStructurePool()
    s, _ = pool.get_or_create(_key(-1, 3304))
    s.add_trigger(_tr("subsequent_confluence", 3350, 1, 1, LENS_CONFLUENCE))
    s.add_trigger(_tr("reversal", 3360, 1, 1, LENS_COUNTER))
    s.add_trigger(_tr("first_confluence", 3500, 1, 2, LENS_CONFLUENCE))
    # parent cycle 1 ends at 3400 (BETWEEN the cycle-1 triggers and the cycle-2
    # trigger); cycle 2 ends at 3700.
    parent_ends = {(1, 1): 3400, (1, 2): 3700}
    finalize_lifecycles(pool.all(), parent_end_lookup=parent_ends)
    assert s.lifecycle_start == 3350
    # NOT 3400 (that would split the sub) — continuous to 3700.
    assert (s.lifecycle_end, s.lifecycle_end_reason) == (3700, "parent_end")


def test_earlier_membership_parent_end_does_not_cap_multicycle_sub():
    """A sub spanning cycles (0,0)+(0,1) must NOT be capped at cycle (0,0)'s end
    (it continues into (0,1)); only the LATEST membership's parent-end caps it.
    Regression for the M15 2365 lifecycle bug found via Stage 3.2a logging: an
    earlier cycle's end >= max(start) was wrongly ending a multi-cycle sub."""
    pool = SubStructurePool()
    s, _ = pool.get_or_create(_key(1, 2365))
    s.add_trigger(_tr("reversal", 2470, 0, 0, LENS_CONFLUENCE))
    s.add_trigger(_tr("first_confluence", 2608, 0, 1, LENS_CONFLUENCE))
    s.natural_reversal_idx = 2829
    # (0,0) ends at 2611 (just after the last trigger) — must NOT cap it; (0,1)
    # ends at 3200. So the natural reversal 2829 (< 3200) is the terminal.
    parent_ends = {(0, 0): 2611, (0, 1): 3200}
    finalize_lifecycles(pool.all(), parent_end_lookup=parent_ends)
    assert s.lifecycle_start == 2470
    assert (s.lifecycle_end, s.lifecycle_end_reason) == (2829, "reversal")


def test_own_reversal_ends_sub():
    pool = SubStructurePool()
    s, _ = pool.get_or_create(_key(1, 100))
    s.add_trigger(_tr("first_confluence", 150, 0, 0, LENS_CONFLUENCE))
    s.natural_reversal_idx = 500
    finalize_lifecycles(pool.all(), parent_end_lookup={(0, 0): 900})
    assert (s.lifecycle_end, s.lifecycle_end_reason) == (500, "reversal")


def test_cross_chain_reversal_ends_active_counter_sub():
    """A confluence-direction sub (+1) reverses; its reversal successor is a
    counter-direction sub (-1) that is sticky to the CONFLUENCE chart but, being
    -1, ends the active counter-direction sub (§17.6 cross-chain interaction)."""
    pool = SubStructurePool()
    # Existing counter-direction sub K (-1), on the counter chart.
    k, _ = pool.get_or_create(_key(-1, 50))
    k.add_trigger(_tr("first_counter", 80, 0, 0, LENS_COUNTER))
    # Confluence-direction sub C (+1) that reverses at 500.
    c, _ = pool.get_or_create(_key(1, 100))
    c.add_trigger(_tr("first_confluence", 150, 0, 0, LENS_CONFLUENCE))
    c.natural_reversal_idx = 500
    # Reversal successor R (-1), sticky to the confluence chart, starts at 500.
    r, _ = pool.get_or_create(_key(-1, 480))
    r.add_trigger(_tr(
        "reversal", 500, 0, 0,
        resolve_lens("reversal", reversed_from_lens=LENS_CONFLUENCE),
    ))
    finalize_lifecycles(pool.all(), parent_end_lookup={})
    # K ends when R (a new -1 sub) starts — a cross-chain end.
    assert (k.lifecycle_end, k.lifecycle_end_reason) == (500, "same_dir_replacement")
    # C ends at its own reversal.
    assert (c.lifecycle_end, c.lifecycle_end_reason) == (500, "reversal")
    # R is -1 (counter direction) but renders on the confluence chart (sticky).
    assert r.lenses() == {LENS_CONFLUENCE}
    assert r.direction == -1


def test_replaced_then_retriggered_overextension_is_the_known_edge():
    """Documented v1 edge (§17.6): if a same-dir sub replaces S and THEN a later
    trigger re-resolves to S's exact (dir,start), the `end >= max(start)` rule
    extends S across the gap where it was actually replaced. Asserting the
    (accepted) over-extension so a future interval-sweep fix has a guard."""
    pool = SubStructurePool()
    s, _ = pool.get_or_create(_key(1, 100))
    s.add_trigger(_tr("first_confluence", 200, 0, 0, LENS_CONFLUENCE))
    s.add_trigger(_tr("subsequent_confluence", 600, 0, 0, LENS_CONFLUENCE))  # re-trigger
    r, _ = pool.get_or_create(_key(1, 300))
    r.add_trigger(_tr("subsequent_confluence", 400, 0, 0, LENS_CONFLUENCE))  # replacement
    finalize_lifecycles(pool.all(), parent_end_lookup={})
    # R.start (400) < max(S.start)=600, so it is filtered out — S over-extends
    # (open) across [400,600] instead of ending at 400. This is the known edge.
    assert s.lifecycle_start == 200
    assert s.lifecycle_end is None
