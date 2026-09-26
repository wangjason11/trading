"""Plan C section 9.2 -- the predicted-table test (the sweep's internal checkpoint).

Feeds `run_lifecycle_sweep` the 11 H1-derived triggers of the reference window
(NZD_USD H1 2025-11-15 -> 2026-01-20, M15 subs) with STUBBED probe results and
STUBBED geometry, builds the section-3 parent tables from a synthetic H1 event
list that reproduces the saved stream's parent facts, and asserts the FULL
predicted table from `memory/reference_pool_redesign_groundtruth.md` -- every
record's identity + lifecycle, every sub's lifecycle + lenses, the 4 unresolved
rows and the 4 sweep-spawned reversal triggers -- keyed by
`(direction, starting_idx)` (the memory table's "sub 7/8/9/10" labels are the
BASELINE's numbering; under Plan C `sub_id` is 0..7 in creation order).

Contract: `plans/PLAN_C_lifecycle_rewrite.md` sections 2-4 + 9.2; PART4 section
17.4-17.7. Pure logic -- no pandas frames, no MS, no probe: the resolvers are
dict stubs keyed by `(trigger_type, S, C)` / `(reversing starting_idx, R)`; the
geometry stub creates pool entries with the known `natural_reversal_idx`.

Every expected number below is DERIVED in a comment from the rule that produces
it; none is copied from a CSV. The sweep is hand-run in the comments of
`EXPECTED_RECORDS` / `EXPECTED_SUBS`.
"""
from __future__ import annotations

import ast
import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import pytest

from engine_v2.multitf.sub_structure_pool import (
    LENS_CONFLUENCE,
    LENS_COUNTER,
    PooledStructure,
    StructureKey,
    SubStructurePool,
    TriggerRecord,
    UnresolvedTrigger,
)
from engine_v2.multitf.parent_tables import ParentTables, build_parent_tables
from engine_v2.multitf.lifecycle_sweep import (
    ProbeFailure,  # noqa: F401  (part of the pinned API surface; unused by the stubs here)
    ResolvedStart,
    SweepResult,
    SweepTrigger,
    run_lifecycle_sweep,
)
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.structure import event_fields as ef
from engine_v2.structure.event_fields import CTS_UPDATED_RAW_VIA
from engine_v2.tests._event_factory import make_event

PARENT = "H1.main"
SUB_TF = "M15"
CONF = LENS_CONFLUENCE
CTR = LENS_COUNTER

# Optional cross-check input: the Plan-B save's H1 structure-events CSV
# (Plan C's /compare baseline). Gitignored artifact -> skip when absent.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_H1_EVENTS_CSV = (
    _REPO_ROOT / "artifacts" / "commits" / "week8-volmom-multitf"
    / "20260920_104606_189c127"
    / "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5_structure_events.csv"
)


# =============================================================================
# H1 parent facts (groundtruth "H1 parent facts")
# =============================================================================
#
#   CTS_ESTABLISHED (S,C).idx == meta["confirmed_at"] (the MOMENT):
#       (0,0)=115  (0,1)=652  (1,0)=703  (1,1)=748  (1,2)=902
#   BOS_CONFIRMED (S,C): anchor (the BOS extreme; the `idx` argument) -> confirmed_at == the CTS
#   moment (the event's idx since Plan E E4b):
#       (0,0) 96->115  (0,1) 591->652  (1,0) 689->703  (1,1) 728->748  (1,2) 826->902
#   STATE_CHANGED to=reversal: sid 0 @ 902. sid 1 never reverses.
#   parent_sd: sid 0 = +1, sid 1 = -1.
#   the struct_start base = the first CTS_ESTABLISHED moment (Plan E E3f): sid 0 =
#   115 (its BOS extreme 96 before E3f), sid 1 = 703 -> reversal handoff lifts
#   sid 1 to rev_by_sid[0] = 902.
#
# LOH(h) = 4h + 3 holds exactly on this window (groundtruth header); injected.


def _loh(parent_idx: int, _h1_df: Any, _m15_df: Any) -> Optional[int]:
    return 4 * int(parent_idx) + 3


def _ev(idx: int, category: str, ev_type: str, price: Optional[float], **meta) -> StructureEvent:
    # CTS_ESTABLISHED / BOS_CONFIRMED: the `idx` argument is the ANCHOR (`make_event`); the event's idx is the
    # contract's (the moment on both types since Plan E E4a / E4b).
    return make_event(ev_type, idx, price=price, category=category, **meta)


def _h1_events() -> List[StructureEvent]:
    """Synthetic H1 stream reproducing the parent facts above (emission shape
    of the saved CSV: CTS_ESTABLISHED + BOS_CONFIRMED per cycle, the sid-0
    reversal, plus the REVERSAL_CANDIDATE/WATCH rows that the tables must
    IGNORE -- rev_by_sid reads STATE_CHANGED->reversal only)."""
    evs = [
        # --- sid 0 (+1) ---
        _ev(115, "STRUCTURE", "CTS_ESTABLISHED", 0.56126, via="one_maru_continuous", pattern_anchor_idx=114,
            confirmed_at=115, cycle_id=0, structure_id=0, struct_direction=1),
        _ev(96, "STRUCTURE", "BOS_CONFIRMED", 0.55808, source="initial_prior_extreme",
            confirmed_at=115, pb_start=None, cycle_id=0, structure_id=0, struct_direction=1),
        _ev(652, "STRUCTURE", "CTS_ESTABLISHED", 0.58534, via="one_maru_opposite", pattern_anchor_idx=651,
            confirmed_at=652, cycle_id=1, structure_id=0, struct_direction=1),
        _ev(591, "STRUCTURE", "BOS_CONFIRMED", 0.5736, source="pullback_extreme",
            confirmed_at=652, pb_start=439, cycle_id=1, structure_id=0, struct_direction=1),
        # A prediction that could have expired -- NOT the reversal idx source.
        _ev(897, "STRUCTURE", "REVERSAL_WATCH_START", 0.5736, pattern_anchor_idx=897, bos_frozen=0.5736,
            expires_idx=902, structure_id=0, struct_direction=1),
        _ev(897, "STRUCTURE", "REVERSAL_CANDIDATE", 0.5736, pattern_anchor_idx=897, apply_idx=902,
            pattern="?", bos_frozen=0.5736, expires_idx=902, structure_id=0, struct_direction=1),
        _ev(902, "STATE", "STATE_CHANGED", None, **{"from": "pullback_range"}, to="reversal",
            structure_id=0, struct_direction=1, reason="reversal_pattern", pat="continuous",
            bos_frozen=0.5736, effective_idx=902),
        # --- sid 1 (-1) ---
        _ev(703, "STRUCTURE", "CTS_ESTABLISHED", 0.58195, via="double_maru", pattern_anchor_idx=702,
            confirmed_at=703, cycle_id=0, structure_id=1, struct_direction=-1),
        _ev(689, "STRUCTURE", "BOS_CONFIRMED", 0.58424, source="initial_prior_extreme",
            confirmed_at=703, pb_start=None, cycle_id=0, structure_id=1, struct_direction=-1),
        _ev(748, "STRUCTURE", "CTS_ESTABLISHED", 0.57764, via="one_maru_opposite", pattern_anchor_idx=747,
            confirmed_at=748, cycle_id=1, structure_id=1, struct_direction=-1),
        _ev(728, "STRUCTURE", "BOS_CONFIRMED", 0.58196, source="pullback_extreme",
            confirmed_at=748, pb_start=727, cycle_id=1, structure_id=1, struct_direction=-1),
        _ev(902, "STRUCTURE", "CTS_ESTABLISHED", 0.57228, via="continuous", pattern_anchor_idx=897,
            confirmed_at=902, cycle_id=2, structure_id=1, struct_direction=-1),
        _ev(826, "STRUCTURE", "BOS_CONFIRMED", 0.58105, source="pullback_extreme",
            confirmed_at=902, pb_start=810, cycle_id=2, structure_id=1, struct_direction=-1),
    ]
    # The orchestrator's `sorted_events` order (`ef.processing_order_key`).
    return sorted(evs, key=ef.processing_order_key)


# Expected section-3 tables (derived):
#   rev_by_sid   = {0: 902}                       # STATE_CHANGED->reversal, sid 0
#   struct_start = {0: 96, 1: 902}                # min idx 96 / 689 -> sid 1 handoff = rev[0] = 902
#   cts_moment   = {(0,0):115,(0,1):652,(1,0):703,(1,1):748,(1,2):902}
#   parent_sd    = {0: +1, 1: -1}
#   floor_h1     = max(struct_start[S], cts_moment):
#                  (0,0)=max(96,115)=115 (0,1)=max(96,652)=652
#                  (1,0)=max(902,703)=902 (1,1)=max(902,748)=902 (1,2)=max(902,902)=902
#   end_h1       = floor_h1[(S,C+1)] else rev_by_sid[S] else None:
#                  (0,0)=floor(0,1)=652 (0,1)=rev[0]=902 (1,0)=floor(1,1)=902
#                  (1,1)=floor(1,2)=902 (1,2)=rev.get(1)=None
#   floor_m15    = LOH: 115->463  652->2611  902->3611 (x3)
#   end_m15      = LOH: 652->2611  902->3611  902->3611  902->3611  None
#   degenerate   = end is not None and floor >= end:
#                  (0,0) 463>=2611 F  (0,1) 2611>=3611 F  (1,0) 3611>=3611 T
#                  (1,1) 3611>=3611 T  (1,2) end None -> F
EXPECTED_REV_BY_SID = {0: 902}
EXPECTED_STRUCT_START = {0: 115, 1: 902}   # sid 0: the CTS_0 moment (Plan E E3f; was the BOS_0 anchor 96)
EXPECTED_CTS_MOMENT = {(0, 0): 115, (0, 1): 652, (1, 0): 703, (1, 1): 748, (1, 2): 902}
EXPECTED_PARENT_SD = {0: 1, 1: -1}
EXPECTED_FLOOR_H1 = {(0, 0): 115, (0, 1): 652, (1, 0): 902, (1, 1): 902, (1, 2): 902}
EXPECTED_END_H1 = {(0, 0): 652, (0, 1): 902, (1, 0): 902, (1, 1): 902, (1, 2): None}
EXPECTED_FLOOR_M15 = {(0, 0): 463, (0, 1): 2611, (1, 0): 3611, (1, 1): 3611, (1, 2): 3611}
EXPECTED_END_M15 = {(0, 0): 2611, (0, 1): 3611, (1, 0): 3611, (1, 1): 3611, (1, 2): None}
EXPECTED_DEGENERATE = {(0, 0): False, (0, 1): False, (1, 0): True, (1, 1): True, (1, 2): False}


# =============================================================================
# The 11 H1-derived input triggers (groundtruth "The 16 triggers -> 11 unique subs")
# =============================================================================
#
# The memory table's 16 rows include 5 `reversal` triggers; those are NOT inputs
# -- the sweep spawns them from natural reversals. The inputs are the 11
# H1-derived ones. All five FC are finalized (no `pending` row).
#
#   trigger_idx = LOH(trigger_event_idx) = 4*tei + 3
#   FC / subsequent_confluence: lens confluence, direction = parent_sd
#   first_counter / subsequent_counter: lens counter, direction = -parent_sd
#   FC's trigger_event_idx = BOS.confirmed_at (== the CTS moment).
#
#   type                  (S,C)  tei   trigger_idx           dir
#   first_confluence      (0,0)  115   4*115+3  =  463       +1
#   first_confluence      (0,1)  652   4*652+3  = 2611       +1
#   first_confluence      (1,0)  703   4*703+3  = 2815       -1   degenerate cycle
#   first_confluence      (1,1)  748   4*748+3  = 2995       -1   degenerate cycle
#   first_confluence      (1,2)  902   4*902+3  = 3611       -1
#   first_counter         (0,1)  710   4*710+3  = 2843       -1
#   first_counter         (1,1)  820   4*820+3  = 3283       +1   degenerate cycle
#   first_counter         (1,2)  926   4*926+3  = 3707       +1
#   subsequent_confluence (1,1)  871   4*871+3  = 3487       -1   degenerate cycle
#   subsequent_confluence (1,2)  954   4*954+3  = 3819       -1
#   subsequent_counter    (1,2) 1020   4*1020+3 = 4083       +1


@dataclass(frozen=True)
class _Src:
    """Stand-in for the (opaque to the sweep) `MultiTFTrigger`. Carries the
    fields `_synth_reversal_trigger` touches so the synth stub can mirror it."""
    parent_tf: str
    parent_sid: int
    parent_cycle_id: int
    parent_sd: int
    use_case: str
    lower_tf: str
    lower_sd: int
    meta: Dict[str, Any]


def _synth_stub(source: _Src, sd: int, reversal_apply_idx: int) -> _Src:
    """Mirror of `entity_df_mutation._synth_reversal_trigger` on the stand-in:
    same parent linkage, `use_case="reversal"`, `lower_sd=sd`,
    `meta["reversal_apply_idx"]`."""
    assert isinstance(source, _Src), f"synth got a non-source object: {source!r}"
    return replace(
        source, use_case="reversal", lower_sd=int(sd),
        meta={**source.meta, "reversal_apply_idx": int(reversal_apply_idx)},
    )


# (trigger_type, S, C, tei, parent_input_idx) ; parent_input_idx = the H1 input
# candle: the BOS anchor for FC (groundtruth BOS column); for the sibling-referencing
# types (whose H1 input is informational — they probe from an M15 sibling CTS) a
# stand-in, the trigger event idx.
_H1_TRIGGER_SPECS: List[Tuple[str, int, int, int, Optional[int]]] = [
    ("first_confluence", 0, 0, 115, 96),
    ("first_confluence", 0, 1, 652, 591),
    ("first_confluence", 1, 0, 703, 689),
    ("first_confluence", 1, 1, 748, 728),
    ("first_confluence", 1, 2, 902, 826),
    ("first_counter", 0, 1, 710, 710),
    ("first_counter", 1, 1, 820, 820),
    ("first_counter", 1, 2, 926, 926),
    ("subsequent_confluence", 1, 1, 871, 871),
    ("subsequent_confluence", 1, 2, 954, 954),
    ("subsequent_counter", 1, 2, 1020, 1020),
]

_LENS_OF = {
    "first_confluence": CONF, "subsequent_confluence": CONF,
    "first_counter": CTR, "subsequent_counter": CTR,
}


def _h1_triggers() -> List[SweepTrigger]:
    """The 11 inputs, in DETECTOR order (all FC, then first_counter, then the
    subsequent_* types) -- deliberately NOT chronological, so the sweep's own
    ordering by `trigger_idx` is exercised."""
    out: List[SweepTrigger] = []
    for ttype, s, c, tei, parent_in in _H1_TRIGGER_SPECS:
        parent_sd = EXPECTED_PARENT_SD[s]
        lens = _LENS_OF[ttype]
        direction = parent_sd if lens == CONF else -parent_sd
        src = _Src(
            parent_tf="H1", parent_sid=s, parent_cycle_id=c, parent_sd=parent_sd,
            use_case=ttype, lower_tf=SUB_TF, lower_sd=direction,
            meta={"trigger_event_idx": tei},
        )
        out.append(SweepTrigger(
            lens=lens, parent_sid=s, parent_cycle_id=c, trigger_type=ttype,
            trigger_idx=4 * tei + 3, direction=direction, trigger_event_idx=tei,
            source=src, pending=False, parent_input_idx=parent_in,
        ))
    assert len(out) == 11
    return out


# =============================================================================
# Stubs (groundtruth probe outputs)
# =============================================================================

# (trigger_type, S, C) -> (starting_idx, finalize_idx, finalize_condition)
# from the groundtruth "first_confluence probe internals" (FC rows) and the
# "16 triggers" table (non-FC: finalize == trigger_idx, gap 0).
_PROBE: Dict[Tuple[str, int, int], Tuple[int, int, str]] = {
    ("first_confluence", 0, 0): (454, 1020, "second_cts_reached"),
    ("first_confluence", 0, 1): (2365, 2608, "second_cts_reached"),
    ("first_confluence", 1, 2): (3304, 3621, "no_retrace"),
    ("first_counter", 0, 1): (2639, 2843, "no_retrace"),
    ("first_counter", 1, 2): (3621, 3707, "no_retrace"),
    ("subsequent_confluence", 1, 2): (3760, 3819, "no_retrace"),
    ("subsequent_counter", 1, 2): (4027, 4083, "no_retrace"),
}

# (reversing sub's starting_idx, R) -> successor starting_idx; finalize = R,
# condition "no_retrace" (the reversal handoff over the reversing sub's geometry).
_REVERSAL: Dict[Tuple[int, int], int] = {
    (454, 1940): 1797,
    (1797, 2470): 2365,
    (2365, 2829): 2639,
    (3760, 4200): 4027,
}

# starting_idx -> natural_reversal_idx (groundtruth: 1940, 2470, 2829, None, None,
# None, 3589, None, None, 4200, None for the baseline's 11 subs; 3589 belongs to
# 3230/+1 which is NEVER built under Plan C -- its (1,1) trigger is degenerate).
_NAT_REV: Dict[int, Optional[int]] = {
    454: 1940, 1797: 2470, 2365: 2829, 2639: None,
    3304: None, 3621: None, 3760: 4200, 4027: None,
}

# Arbitrary but CONSISTENT per key (not part of the predicted table): the same
# value on every trigger that reaches a key, so the section-4.3 bos0_inner
# mismatch WARNING never fires here.
_BOS0: Dict[int, float] = {
    454: 0.5600, 1797: 0.5700, 2365: 0.5650, 2639: 0.5800,
    3304: 0.5750, 3621: 0.5700, 3760: 0.5720, 4027: 0.5680,
}


def _key(direction: int, starting_idx: int) -> StructureKey:
    return StructureKey(PARENT, SUB_TF, int(direction), int(starting_idx))


def _make_stubs(pool: SubStructurePool, calls: Dict[str, list]):
    def resolve_start(trigger: SweepTrigger, hi: int):
        k = (trigger.trigger_type, trigger.parent_sid, trigger.parent_cycle_id)
        calls["resolve_start"].append((k, trigger.trigger_idx, hi))
        if k not in _PROBE:
            # (1,0)/(1,1) triggers are degenerate -> must NEVER reach the resolver.
            raise AssertionError(f"resolver reached for a trigger that must be unresolved: {k}")
        assert hi == trigger.trigger_idx, (hi, trigger.trigger_idx)   # sibling window hi = t
        starting_idx, finalize_idx, cond = _PROBE[k]
        return ResolvedStart(
            starting_idx=starting_idx,
            parent_bos_anchor_idx=trigger.trigger_event_idx,   # a stub PASS-THROUGH (the sweep copies the
                                                             # resolver's value; the real rule — the H1 BOS anchor
                                                             # for FC, None for every other type (PLAN_E Q4) —
                                                             # is pinned in test_first_trigger_migration /
                                                             # test_reversal_resolver, not here)
            bos0_inner=_BOS0[starting_idx],
            finalize_idx=finalize_idx,
            finalize_condition=cond,
            probe_input_idx=None,          # the M15 input is not modelled here (pinned in
                                           # test_lifecycle_sweep_unit / test_first_trigger_migration)
            cache_hit=False,
        )

    def build_geometry(key: StructureKey, bos0_inner: Optional[float]):
        calls["build_geometry"].append((key, bos0_inner))
        assert key.parent_path == PARENT and key.sub_tf == SUB_TF, key
        if key.starting_idx not in _NAT_REV or key.starting_idx in (2803, 2915, 3230):
            raise AssertionError(f"geometry requested for a sub that must never be built: {key}")
        sub, created = pool.get_or_create(key)
        if created:
            sub.geometry = ("stub-geometry", 0)          # (bounded, slice_begin) shape
            sub.natural_reversal_idx = _NAT_REV[key.starting_idx]
            sub.bos0_inner = bos0_inner
        return sub, created

    def resolve_reversal(sub: PooledStructure, R: int, probe_direction: int):
        calls["resolve_reversal"].append((sub.starting_idx, R, probe_direction))
        assert probe_direction == -sub.direction, (probe_direction, sub.direction)
        k = (sub.starting_idx, R)
        if k not in _REVERSAL:
            raise AssertionError(f"reversal resolver reached for an unexpected (sub, R): {k}")
        return ResolvedStart(
            starting_idx=_REVERSAL[k], parent_bos_anchor_idx=None,
            bos0_inner=_BOS0[_REVERSAL[k]], finalize_idx=R, finalize_condition="no_retrace",
            probe_input_idx=None, cache_hit=False,
        )

    return resolve_start, build_geometry, resolve_reversal


# =============================================================================
# The predicted table (hand-run of the sweep; groundtruth "PREDICTED TABLE")
# =============================================================================
#
# Rules: start = max(finalize, trigger, floor); ends at record level only
# (own reversal / same-lens same-dir replacement at the NEW record's start_idx /
# parent_end = end_m15 of the record's own cycle); earliest wins; sub start =
# first live record's start; sub end = min(record end > max_start over records
# that EXIST at t); start-before-end at equal idx.
#
# Phase-0 order (fire idx; REVERSAL_SPAWN before TRIGGER_FIRE at equal idx):
#    463 FC(0,0)            -> R0 ; sub 454/+1 created, nat_rev 1940 -> spawn queued @1940
#   1940 spawn from 454     -> R1 ; sub 1797/-1 created, nat_rev 2470 -> spawn @2470
#   2470 spawn from 1797    -> R2 ; sub 2365/+1 created, nat_rev 2829 -> spawn @2829
#   2611 FC(0,1)            -> R3 ; sub 2365/+1 dedup hit (no spawn)
#   2815 FC(1,0)            -> UNRESOLVED degenerate (1,0)
#   2829 spawn from 2365    -> R4 ; sub 2639/-1 created, nat_rev None
#   2843 first_counter(0,1) -> R5 ; sub 2639/-1 dedup hit
#   2995 FC(1,1)            -> UNRESOLVED degenerate (1,1)
#   3283 first_counter(1,1) -> UNRESOLVED degenerate (1,1)
#   3487 subseq_conf(1,1)   -> UNRESOLVED degenerate (1,1)
#   3611 FC(1,2)            -> R6 ; sub 3304/-1 created, nat_rev None
#   3707 first_counter(1,2) -> R7 ; sub 3621/+1 created, nat_rev None
#   3819 subseq_conf(1,2)   -> R8 ; sub 3760/-1 created, nat_rev 4200 -> spawn @4200
#   4083 subseq_ctr(1,2)    -> R9 ; sub 4027/+1 created, nat_rev None
#   4200 spawn from 3760    -> R10; sub 4027/+1 dedup hit
#
# => sub_id 0..7 in creation order: 454, 1797, 2365, 2639, 3304, 3621, 3760, 4027.
EXPECTED_SUB_ORDER: List[Tuple[int, int]] = [
    (1, 454), (-1, 1797), (1, 2365), (-1, 2639), (-1, 3304), (1, 3621), (-1, 3760), (1, 4027),
]


class _R(NamedTuple):
    sub: Tuple[int, int]            # (direction, starting_idx) -> resolved to sub_id at assert time
    lens: str
    parent: Tuple[int, int]         # (S, C)
    tss: int                        # trigger_sub_sid (0-based per (lens, S, C), creation-ordered)
    trigger_type: str
    trigger_idx: int
    finalize_idx: int
    finalize_condition: str
    parent_bos_anchor_idx: Optional[int]
    relative_dir: str
    floor: int                      # parent_floor_idx
    start_idx: int
    trigger_end_idx: Optional[int]
    end_idx: Optional[int]
    end_reason: Optional[str]
    ended_by: Optional[Tuple[int, int]]   # (direction, starting_idx) of the replacing sub


EXPECTED_RECORDS: List[_R] = [
    # R0  FC(0,0) @463 -> probe (454, fin 1020). rd: +1 == parent_sd[0]=+1 -> confluence.
    #     start = max(fin 1020, trig 463, floor 463) = 1020.
    #     ends known at creation: own reversal 1940 vs parent_end(0,0)=2611 -> min = 1940 reversal
    #     (1940 > start 1020 -> queued). end_idx = max(1940, 1020) = 1940.
    _R((1, 454), CONF, (0, 0), 0, "first_confluence", 463, 1020, "second_cts_reached", 115,
       "confluence", 463, 1020, 1940, 1940, "reversal", None),
    # R1  REVERSAL_SPAWN of sub 454 at R=1940: R0 is live at 1940 (1020 <= 1940, open) -> one
    #     successor trigger on R0's lens/parent (conf, 0,0), dir = -(+1) = -1, trigger = fin = 1940.
    #     handoff -> starting 1797. tss = 1 (second NEW sub in (conf,0,0)). rd: -1 != +1 -> counter.
    #     start = max(1940, 1940, floor 463) = 1940. ends: own rev 2470 vs parent_end 2611 -> 2470.
    _R((-1, 1797), CONF, (0, 0), 1, "reversal", 1940, 1940, "no_retrace", None,
       "counter", 463, 1940, 2470, 2470, "reversal", None),
    # R2  spawn of sub 1797 at 2470 (R1 live) -> handoff 2365/+1, tss 2, rd confluence,
    #     start = 2470. ends: own rev 2829 vs parent_end(0,0) 2611 -> 2611 parent_end.
    _R((1, 2365), CONF, (0, 0), 2, "reversal", 2470, 2470, "no_retrace", None,
       "confluence", 463, 2470, 2611, 2611, "parent_end", None),
    # R3  FC(0,1) @2611 -> probe (2365, fin 2608) -> dedup onto sub 2365/+1. New scope
    #     (conf,0,1) -> tss 0. start = max(2608, 2611, floor 2611) = 2611 (trigger term binds;
    #     finalize 2608 < trigger is logged raw). ends: own rev 2829 vs parent_end(0,1) 3611 -> 2829.
    _R((1, 2365), CONF, (0, 1), 0, "first_confluence", 2611, 2608, "second_cts_reached", 652,
       "confluence", 2611, 2611, 2829, 2829, "reversal", None),
    # R4  spawn of sub 2365 at 2829: R3 live (2611 <= 2829, open); R2 NOT live (ended 2611 < 2829)
    #     -> exactly one successor, on (conf,0,1): handoff 2639/-1, tss 1, rd: -1 != +1 -> counter.
    #     start = max(2829, 2829, 2611) = 2829. ends: no own reversal; parent_end(0,1) 3611.
    _R((-1, 2639), CONF, (0, 1), 1, "reversal", 2829, 2829, "no_retrace", None,
       "counter", 2611, 2829, 3611, 3611, "parent_end", None),
    # R5  first_counter(0,1) @2843 -> probe (2639, fin 2843) -> dedup onto 2639/-1; (ctr,0,1) tss 0.
    #     start = max(2843, 2843, 2611) = 2843. ends: parent_end 3611.
    _R((-1, 2639), CTR, (0, 1), 0, "first_counter", 2843, 2843, "no_retrace", 710,
       "counter", 2611, 2843, 3611, 3611, "parent_end", None),
    # R6  FC(1,2) @3611 -> probe (3304, fin 3621). rd: -1 == parent_sd[1]=-1 -> confluence.
    #     start = max(3621, 3611, floor 3611) = 3621. No end known at creation (nat_rev None,
    #     end_m15(1,2) None). Replaced at 3819: R8 (same lens conf, same (1,2), same dir -1,
    #     DIFFERENT sub) starts at 3819 -> R6 ends same_dir_replacement at R8's start_idx 3819.
    _R((-1, 3304), CONF, (1, 2), 0, "first_confluence", 3611, 3621, "no_retrace", 902,
       "confluence", 3611, 3621, 3819, 3819, "same_dir_replacement", (-1, 3760)),
    # R7  first_counter(1,2) @3707 -> (3621, fin 3707); (ctr,1,2) tss 0; rd: +1 != -1 -> counter.
    #     start 3707. Replaced at 4083 by R9 (ctr, (1,2), +1, sub 4027) -> same_dir_replacement.
    _R((1, 3621), CTR, (1, 2), 0, "first_counter", 3707, 3707, "no_retrace", 926,
       "counter", 3611, 3707, 4083, 4083, "same_dir_replacement", (1, 4027)),
    # R8  subseq_conf(1,2) @3819 -> (3760, fin 3819); (conf,1,2) tss 1; rd confluence; start 3819.
    #     ends: own reversal 4200 (no parent_end) -> 4200 reversal.
    _R((-1, 3760), CONF, (1, 2), 1, "subsequent_confluence", 3819, 3819, "no_retrace", 954,
       "confluence", 3611, 3819, 4200, 4200, "reversal", None),
    # R9  subseq_ctr(1,2) @4083 -> (4027, fin 4083); (ctr,1,2) tss 1; rd counter; start 4083; open.
    _R((1, 4027), CTR, (1, 2), 1, "subsequent_counter", 4083, 4083, "no_retrace", 1020,
       "counter", 3611, 4083, None, None, None, None),
    # R10 spawn of sub 3760 at 4200 (R8 live) -> handoff 4027/+1 = dedup onto sub 4027;
    #     (conf,1,2) tss 2 (third NEW sub in that scope: 3304, 3760, 4027); rd: +1 != -1 -> counter.
    #     start = max(4200, 4200, 3611) = 4200; open (no own reversal, no parent end).
    _R((1, 4027), CONF, (1, 2), 2, "reversal", 4200, 4200, "no_retrace", None,
       "counter", 3611, 4200, None, None, None, None),
]


class _S(NamedTuple):
    sub: Tuple[int, int]
    start_idx: int
    end_idx: Optional[int]
    end_reason: Optional[str]
    lenses: frozenset
    natural_reversal_idx: Optional[int]
    n_records: int


EXPECTED_SUBS: List[_S] = [
    # 454/+1: start = R0.start 1020; at 1940 R0 ends: max_start 1020, 1940 > 1020 -> end 1940 reversal.
    _S((1, 454), 1020, 1940, "reversal", frozenset({CONF}), 1940, 1),
    # 1797/-1: [1940, 2470] reversal (R1).
    _S((-1, 1797), 1940, 2470, "reversal", frozenset({CONF}), 2470, 1),
    # 2365/+1: start = R2.start 2470. At 2611: phase 1 starts R3 BEFORE phase 3 ends R2 ->
    #   max_start = 2611, R2.end 2611 > 2611 is FALSE -> the sub persists (the handover).
    #   At 2829: R3.end 2829 > max_start 2611 -> end 2829 reversal. Both records: conf lens.
    _S((1, 2365), 2470, 2829, "reversal", frozenset({CONF}), 2829, 2),
    # 2639/-1: start = R4.start 2829 (R5 starts 2843 -> max_start 2843). At 3611 both records
    #   end parent_end: 3611 > 2843 -> end 3611 parent_end. Lenses: conf (R4) + ctr (R5) -> both.
    _S((-1, 2639), 2829, 3611, "parent_end", frozenset({CONF, CTR}), None, 2),
    # 3304/-1: [3621, 3819] same_dir_replacement (R6 replaced by sub 3760's record).
    _S((-1, 3304), 3621, 3819, "same_dir_replacement", frozenset({CONF}), None, 1),
    # 3621/+1: [3707, 4083] same_dir_replacement (R7 replaced by sub 4027's counter record).
    _S((1, 3621), 3707, 4083, "same_dir_replacement", frozenset({CTR}), None, 1),
    # 3760/-1: [3819, 4200] reversal (R8).
    _S((-1, 3760), 3819, 4200, "reversal", frozenset({CONF}), 4200, 1),
    # 4027/+1: start = R9.start 4083; R10 starts 4200 (max_start 4200); no end candidates -> open.
    #   Lenses: ctr (R9) + conf (R10) -> both. Overlaps sub 3760/-1 on confluence over
    #   [4083, 4200] -- opposite directions, allowed.
    _S((1, 4027), 4083, None, None, frozenset({CONF, CTR}), None, 2),
]

# (lens, S, C, trigger_type, trigger_idx, direction) of the 4 degenerate-cycle triggers.
EXPECTED_UNRESOLVED = [
    (CONF, 1, 0, "first_confluence", 2815, -1),
    (CONF, 1, 1, "first_confluence", 2995, -1),
    (CTR, 1, 1, "first_counter", 3283, 1),
    (CONF, 1, 1, "subsequent_confluence", 3487, -1),
]

# The sweep-synthesised reversal triggers, in fire order:
# (lens, S, C, trigger_idx == R, direction == -reversing sub's direction)
EXPECTED_SPAWNED = [
    (CONF, 0, 0, 1940, -1),   # from 454/+1
    (CONF, 0, 0, 2470, 1),    # from 1797/-1
    (CONF, 0, 1, 2829, -1),   # from 2365/+1 (only R3 (0,1) is live at 2829; R2 (0,0) ended 2611)
    (CONF, 1, 2, 4200, 1),    # from 3760/-1
]


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def tables() -> Tuple[ParentTables, List[str]]:
    lines: List[str] = []
    t = build_parent_tables(_h1_events(), None, None, loh=_loh, log=lines.append)
    return t, lines


@pytest.fixture(scope="module")
def sweep(tables) -> Dict[str, Any]:
    t, _ = tables
    pool = SubStructurePool()
    calls: Dict[str, list] = {"resolve_start": [], "build_geometry": [], "resolve_reversal": []}
    resolve_start, build_geometry, resolve_reversal = _make_stubs(pool, calls)
    lines: List[str] = []
    triggers = _h1_triggers()
    # The sweep's own post-sweep invariants (section 4.5) run inside this call;
    # a raise here IS the failure (no try/except).
    result = run_lifecycle_sweep(
        triggers, pool=pool, tables=t,
        resolve_start=resolve_start, build_geometry=build_geometry,
        resolve_reversal=resolve_reversal, synth_reversal_trigger=_synth_stub,
        parent_path=PARENT, sub_tf=SUB_TF, log=lines.append,
    )
    assert isinstance(result, SweepResult)
    assert result.pool is pool
    return {"pool": pool, "result": result, "calls": calls, "lines": lines, "triggers": triggers}


def _sub(pool: SubStructurePool, dir_start: Tuple[int, int]) -> PooledStructure:
    s = pool.get(_key(*dir_start))
    assert s is not None, f"sub {dir_start} missing from the pool"
    return s


def _rec_ident(r: TriggerRecord) -> Tuple[str, int, int, str, int]:
    return (r.lens, r.parent_sid, r.parent_cycle_id, r.trigger_type, r.trigger_idx)


# =============================================================================
# Section 3 -- parent tables
# =============================================================================

def test_parent_tables_reference_window(tables):
    t, lines = tables
    assert t.rev_by_sid == EXPECTED_REV_BY_SID
    assert t.struct_start == EXPECTED_STRUCT_START
    assert t.cts_moment == EXPECTED_CTS_MOMENT
    assert t.parent_sd == EXPECTED_PARENT_SD
    assert t.floor_h1 == EXPECTED_FLOOR_H1
    assert t.end_h1 == EXPECTED_END_H1
    assert t.floor_m15 == EXPECTED_FLOOR_M15
    assert t.end_m15 == EXPECTED_END_M15
    assert t.degenerate == EXPECTED_DEGENERATE
    assert {k for k, v in t.degenerate.items() if v} == {(1, 0), (1, 1)}
    # Accessors.
    assert t.cycles() == [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]
    assert [t.floor(s, c) for (s, c) in t.cycles()] == [463, 2611, 3611, 3611, 3611]
    assert [t.end(s, c) for (s, c) in t.cycles()] == [2611, 3611, 3611, 3611, None]
    assert [t.is_degenerate(s, c) for (s, c) in t.cycles()] == [False, False, True, True, False]


def test_parent_tables_logs_one_warning_per_degenerate_cycle(tables):
    _, lines = tables
    warnings = [ln for ln in lines if ln.startswith("WARNING [parent_tables] degenerate parent cycle")]
    assert len(warnings) == 2, lines
    norm = [w.replace(" ", "") for w in warnings]
    assert any("(1,0)" in w for w in norm), warnings
    assert any("(1,1)" in w for w in norm), warnings
    # One "[parent_tables]" line per cycle (5) besides the two warnings.
    per_cycle = [ln for ln in lines if ln.startswith("[parent_tables]")]
    assert len(per_cycle) == 5, lines


def _events_from_csv(path: Path) -> List[StructureEvent]:
    out: List[StructureEvent] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            meta = ast.literal_eval(row["meta"]) if row.get("meta") else {}
            # A save older than Plan E E2a lacks the anchor keys; there `idx` IS the
            # anchor (the pre-E4 contract), so translate the legacy row.
            key = {"CTS_ESTABLISHED": "cts_anchor_idx", "BOS_CONFIRMED": "bos_anchor_idx"}.get(row["type"])
            if key is not None:
                meta.setdefault(key, int(row["idx"]))
            # A save older than Plan E E4a / E4b stamps a CTS_ESTABLISHED /
            # BOS_CONFIRMED at its anchor; since then its idx is the moment.
            idx = int(meta["confirmed_at"]) if key is not None else int(row["idx"])
            # A save older than Plan E E3·0 lacks a pattern-path CTS_UPDATED's
            # `confirmed_at` (its apply candle — not recoverable from the row).
            # Stand in `idx`: every H1 pattern-path update on this window has
            # apply == idx (measured at E3·0), and parent_tables reads no
            # CTS_UPDATED moment.
            # A save older than Plan E E4c stamps a pattern-path CTS_UPDATED at its
            # anchor and has no `cts_anchor_idx`: with the stand-in apply == idx the
            # row's idx is anchor and moment at once.
            if row["type"] == "CTS_UPDATED" and meta.get("via") != CTS_UPDATED_RAW_VIA:
                meta.setdefault("confirmed_at", int(row["idx"]))
                meta.setdefault("cts_anchor_idx", int(row["idx"]))
                idx = int(meta["confirmed_at"])
            price = float(row["price"]) if row.get("price") not in (None, "") else None
            out.append(StructureEvent(
                idx=idx, category=row["category"], type=row["type"],
                price=price, meta=meta,
            ))
    return sorted(out, key=ef.processing_order_key)


@pytest.mark.skipif(not _H1_EVENTS_CSV.exists(), reason="Plan-B save's H1 structure_events CSV not present")
def test_parent_tables_from_saved_h1_csv_match_synthetic(tables):
    """Cross-check: the synthetic H1 stream reproduces the saved stream's tables."""
    t_syn, _ = tables
    t_csv = build_parent_tables(_events_from_csv(_H1_EVENTS_CSV), None, None, loh=_loh, log=lambda s: None)
    for name in ("rev_by_sid", "struct_start", "cts_moment", "parent_sd",
                 "floor_h1", "end_h1", "floor_m15", "end_m15", "degenerate"):
        assert getattr(t_csv, name) == getattr(t_syn, name), name


# =============================================================================
# Section 9.2 -- the predicted table
# =============================================================================

def test_sub_ids_follow_creation_order(sweep):
    pool: SubStructurePool = sweep["pool"]
    subs = pool.all()
    assert [(s.direction, s.starting_idx) for s in subs] == EXPECTED_SUB_ORDER
    assert [s.sub_id for s in subs] == list(range(8))
    for s in subs:
        assert pool.get_by_id(s.sub_id) is s
        assert s.key == _key(s.direction, s.starting_idx)
        assert s.parent_path == PARENT and s.sub_tf == SUB_TF


def test_degenerate_subs_never_exist(sweep):
    pool: SubStructurePool = sweep["pool"]
    # The baseline's subs 4/5/6 (2803/-1, 2915/-1, 3230/+1) sit in degenerate cycles:
    # never probed, never built -- and 3230's reversal at 3589 therefore never fires.
    for dir_start in ((-1, 2803), (-1, 2915), (1, 3230)):
        assert pool.get(_key(*dir_start)) is None, dir_start
    assert not any(s.starting_idx in (2803, 2915, 3230) for s in pool.all())
    # No record in the degenerate cycles, on either lens.
    for lens in (CONF, CTR):
        assert pool.records_for(lens, 1, 0) == []
        assert pool.records_for(lens, 1, 1) == []
    assert not any(r.parent_cycle_id in (0, 1) and r.parent_sid == 1 for r in pool.all_records())
    # And no trigger (input or spawned) at 3589 ever produced a record.
    assert not any(r.trigger_idx == 3589 for r in pool.all_records())


def test_records_full_predicted_table(sweep):
    pool: SubStructurePool = sweep["pool"]
    recs = pool.all_records()
    assert len(recs) == 11
    # Creation order == expected order (seq strictly increasing; trigger_sub_sid is creation-ordered).
    assert [r.seq for r in recs] == sorted(r.seq for r in recs)
    assert len({r.seq for r in recs}) == 11
    assert [_rec_ident(r) for r in recs] == [
        (e.lens, e.parent[0], e.parent[1], e.trigger_type, e.trigger_idx) for e in EXPECTED_RECORDS
    ]
    for r, e in zip(recs, EXPECTED_RECORDS):
        sub = _sub(pool, e.sub)
        ended_by = _sub(pool, e.ended_by).sub_id if e.ended_by is not None else None
        got = (r.lens, (r.parent_sid, r.parent_cycle_id), r.trigger_sub_sid, r.relative_dir,
               r.start_idx, r.end_idx, r.end_reason, r.ended_by_sub_id)
        want = (e.lens, e.parent, e.tss, e.relative_dir, e.start_idx, e.end_idx, e.end_reason, ended_by)
        assert got == want, f"record {_rec_ident(r)}: {got} != {want}"
        # foreign key + denormalised structural copies
        assert r.sub_id == sub.sub_id
        assert r.starting_idx == sub.starting_idx == e.sub[1]
        assert r.direction == sub.direction == e.sub[0]
        assert r.sub_tf == SUB_TF
        # provenance (historical)
        assert r.trigger_type == e.trigger_type
        assert r.trigger_idx == e.trigger_idx
        assert r.probe_finalize_idx == e.finalize_idx
        assert r.probe_finalize_condition == e.finalize_condition
        assert r.parent_bos_anchor_idx == e.parent_bos_anchor_idx
        assert r.parent_floor_idx == e.floor
        # lifecycle
        assert r.trigger_end_idx == e.trigger_end_idx
        assert r.is_zero_length is False
        assert r.extra_trigger_idxs == []
        # the record belongs to its sub and is found via records_for
        assert r in sub.records
        assert r in pool.records_for(r.lens, r.parent_sid, r.parent_cycle_id)


def test_trigger_sub_sid_scoped_per_lens_parent(sweep):
    pool: SubStructurePool = sweep["pool"]

    def tss(lens, s, c):
        return [r.trigger_sub_sid for r in pool.records_for(lens, s, c)]

    assert tss(CONF, 0, 0) == [0, 1, 2]      # FC, reversal 1940, reversal 2470 -> three NEW subs
    assert tss(CONF, 0, 1) == [0, 1]         # FC(0,1) (sub 2365, new to this scope), reversal 2829
    assert tss(CTR, 0, 1) == [0]             # first_counter
    assert tss(CONF, 1, 2) == [0, 1, 2]      # FC(1,2), subseq_conf, reversal 4200
    assert tss(CTR, 1, 2) == [0, 1]          # first_counter, subseq_ctr
    assert tss(CTR, 0, 0) == []
    assert tss(CONF, 1, 0) == tss(CONF, 1, 1) == tss(CTR, 1, 1) == []


def test_reversal_born_records_carry_a_synthesised_source(sweep):
    """The successor trigger is synthesised from the SPAWNING RECORD's own source
    (plan section 4.3: `_synth_reversal_trigger(r.source_trigger, sd, R)`)."""
    pool: SubStructurePool = sweep["pool"]
    by_ident = {_rec_ident(r): r for r in pool.all_records()}
    spawners = {
        (CONF, 0, 0, "reversal", 1940): (CONF, 0, 0, "first_confluence", 463),
        (CONF, 0, 0, "reversal", 2470): (CONF, 0, 0, "reversal", 1940),
        (CONF, 0, 1, "reversal", 2829): (CONF, 0, 1, "first_confluence", 2611),
        (CONF, 1, 2, "reversal", 4200): (CONF, 1, 2, "subsequent_confluence", 3819),
    }
    for rev_ident, src_ident in spawners.items():
        rev = by_ident[rev_ident]
        src = by_ident[src_ident].source_trigger
        assert isinstance(rev.source_trigger, _Src)
        assert rev.source_trigger.use_case == "reversal"
        assert rev.source_trigger.lower_sd == rev.direction
        assert rev.source_trigger.meta["reversal_apply_idx"] == rev.trigger_idx
        assert (rev.source_trigger.parent_sid, rev.source_trigger.parent_cycle_id) == (src.parent_sid, src.parent_cycle_id)
        assert rev.source_trigger.meta["trigger_event_idx"] == src.meta["trigger_event_idx"]
    # H1-born records keep the input trigger's source object as-is.
    inputs = {(t.lens, t.parent_sid, t.parent_cycle_id, t.trigger_type, t.trigger_idx): t for t in sweep["triggers"]}
    for ident, r in by_ident.items():
        if r.trigger_type != "reversal":
            assert r.source_trigger is inputs[ident].source


def test_subs_full_predicted_table(sweep):
    pool: SubStructurePool = sweep["pool"]
    assert len(pool.all()) == 8
    for e in EXPECTED_SUBS:
        s = _sub(pool, e.sub)
        got = (s.start_idx, s.end_idx, s.end_reason, frozenset(s.lenses()), s.natural_reversal_idx, len(s.records))
        want = (e.start_idx, e.end_idx, e.end_reason, e.lenses, e.natural_reversal_idx, e.n_records)
        assert got == want, f"sub {e.sub}: {got} != {want}"
        assert s.live_records() == s.records          # no zero-length record on this window
        assert s.bos0_inner == _BOS0[e.sub[1]]
    # The two timelines from the groundtruth (contiguity + the intended gap).
    conf = sorted(((s.start_idx, s.end_idx) for s in pool.all() if CONF in s.lenses()),
                  key=lambda w: w[0])
    assert conf == [(1020, 1940), (1940, 2470), (2470, 2829), (2829, 3611), (3621, 3819), (3819, 4200), (4083, None)]
    ctr = sorted(((s.start_idx, s.end_idx) for s in pool.all() if CTR in s.lenses()),
                 key=lambda w: w[0])
    assert ctr == [(2829, 3611), (3707, 4083), (4083, None)]


def test_unresolved_rows(sweep):
    pool: SubStructurePool = sweep["pool"]
    result: SweepResult = sweep["result"]
    assert result.unresolved == pool.unresolved
    assert len(pool.unresolved) == 4
    assert all(isinstance(u, UnresolvedTrigger) for u in pool.unresolved)
    got = sorted((u.lens, u.parent_sid, u.parent_cycle_id, u.trigger_type, u.trigger_idx, u.direction)
                 for u in pool.unresolved)
    assert got == sorted(EXPECTED_UNRESOLVED)
    for u in pool.unresolved:
        assert u.reason == "degenerate_parent_cycle"
        assert isinstance(u.detail, str) and u.detail
    # The degenerate trigger never reaches a resolver, so the row carries the input's own H1
    # parent_input_idx and no M15 probe_input_idx (PLAN_E §9.2).
    inputs = {(t.lens, t.parent_sid, t.parent_cycle_id, t.trigger_type, t.trigger_idx): t for t in sweep["triggers"]}
    for u in pool.unresolved:
        t = inputs[(u.lens, u.parent_sid, u.parent_cycle_id, u.trigger_type, u.trigger_idx)]
        assert (u.parent_input_idx, u.probe_input_idx) == (t.parent_input_idx, None)
    # Totals: 8 subs / 11 records / 4 unresolved.
    assert (len(pool.all()), len(pool.all_records()), len(pool.unresolved)) == (8, 11, 4)


def test_unresolved_are_logged_with_skipping(sweep):
    """Plan section 8: `[sweep] UNRESOLVED (skipping) reason=...` -- the /compare
    skill greps the word 'skipping'."""
    lines: List[str] = sweep["lines"]
    unresolved_lines = [ln for ln in lines if "UNRESOLVED (skipping)" in ln]
    assert len(unresolved_lines) == 4, lines
    assert all("degenerate_parent_cycle" in ln for ln in unresolved_lines)


def test_spawned_reversal_triggers(sweep):
    result: SweepResult = sweep["result"]
    got = [(t.lens, t.parent_sid, t.parent_cycle_id, t.trigger_idx, t.direction) for t in result.spawned]
    assert got == EXPECTED_SPAWNED
    for t in result.spawned:
        assert isinstance(t, SweepTrigger)
        assert t.trigger_type == "reversal"
        assert t.trigger_event_idx == t.trigger_idx      # native M15 for reversal
        assert t.pending is False


def test_resolver_and_geometry_call_pattern(sweep):
    calls = sweep["calls"]
    # resolve_start: exactly the 7 non-degenerate H1 triggers, in fire order, hi == trigger_idx.
    assert [(k, ti) for (k, ti, _hi) in calls["resolve_start"]] == [
        (("first_confluence", 0, 0), 463),
        (("first_confluence", 0, 1), 2611),
        (("first_counter", 0, 1), 2843),
        (("first_confluence", 1, 2), 3611),
        (("first_counter", 1, 2), 3707),
        (("subsequent_confluence", 1, 2), 3819),
        (("subsequent_counter", 1, 2), 4083),
    ]
    assert all(hi == ti for (_k, ti, hi) in calls["resolve_start"])
    # resolve_reversal: one per spawned successor, probe_direction = -reversing direction.
    assert calls["resolve_reversal"] == [(454, 1940, -1), (1797, 2470, 1), (2365, 2829, -1), (3760, 4200, 1)]
    # build_geometry: once per resolved trigger (7 + 4 = 11); 8 misses (created) + 3 dedup hits.
    keys = [(k.direction, k.starting_idx) for (k, _b) in calls["build_geometry"]]
    assert keys == [
        (1, 454), (-1, 1797), (1, 2365), (1, 2365), (-1, 2639), (-1, 2639),
        (-1, 3304), (1, 3621), (-1, 3760), (1, 4027), (1, 4027),
    ]
    assert all(b == _BOS0[k.starting_idx] for (k, b) in calls["build_geometry"])


def test_relative_dir_segments(sweep):
    pool: SubStructurePool = sweep["pool"]

    # PLAN-AMBIGUITY: section 4.5 says "step function ... at each record start/end"; the plan does
    # not say whether consecutive EQUAL values are collapsed into one segment. Asserted
    # tolerantly: first segment starts at sub.start_idx, from_idx strictly increasing and inside
    # the window, every value == the expected constant.
    def check(dir_start, expected_rd):
        s = _sub(pool, dir_start)
        segs = list(s.relative_dir_segments)
        assert segs, f"sub {dir_start} has no relative_dir_segments"
        assert all(isinstance(i, int) and isinstance(v, str) for i, v in segs), segs
        assert segs[0][0] == s.start_idx, segs
        idxs = [i for i, _ in segs]
        assert idxs == sorted(set(idxs)), segs
        if s.end_idx is not None:
            assert idxs[-1] <= s.end_idx, segs
        assert {v for _, v in segs} == {expected_rd}, segs

    # 2365/+1 under sid 0 (+1): both records confluence -> 'confluence' throughout [2470, 2829].
    check((1, 2365), "confluence")
    # 4027/+1 under sid 1 (-1): counter record (ctr,(1,2)) from 4083, then the reversal-born
    # confluence-LENS record from 4200 -- both relative_dir 'counter' (lens != relative_dir).
    check((1, 4027), "counter")
    # 2639/-1 under sid 0 (+1): counter-relative on BOTH lenses (reversal-sticky on confluence).
    check((-1, 2639), "counter")
    # Every sub on this window started (no lens-less sub).
    assert all(s.start_idx is not None for s in pool.all())


def test_active_record_interval_rule_on_the_3819_handover(sweep):
    """Section 2.4: active iff start_idx <= at and (trigger_end_idx is None or trigger_end_idx > at).
    In the 3819 handover the incumbent (3304's record) is found at 3818 and NOT at 3819."""
    pool: SubStructurePool = sweep["pool"]
    r6 = pool.active_record(CONF, 1, 2, -1, 3818)
    assert r6 is not None and r6.sub_id == _sub(pool, (-1, 3304)).sub_id
    r8 = pool.active_record(CONF, 1, 2, -1, 3819)
    assert r8 is not None and r8.sub_id == _sub(pool, (-1, 3760)).sub_id
    assert r8.is_active_at(3819) and not r6.is_active_at(3819)
    # 2611: R2 (0,0) ended at 2611 -> not active; scope (conf,0,0,+1) is empty at 2611.
    assert pool.active_record(CONF, 0, 0, 1, 2611) is None
    assert pool.active_record(CONF, 0, 0, 1, 2610).sub_id == _sub(pool, (1, 2365)).sub_id
    # (conf,0,1,+1) at 2700 -> R3 (sub 2365).
    assert pool.active_record(CONF, 0, 1, 1, 2700).sub_id == _sub(pool, (1, 2365)).sub_id
    # (ctr,1,2,+1) at 4083 -> the replacing record R9 (sub 4027), not R7 (ended at 4083).
    assert pool.active_record(CTR, 1, 2, 1, 4083).sub_id == _sub(pool, (1, 4027)).sub_id
    assert pool.active_record(CTR, 1, 2, 1, 4082).sub_id == _sub(pool, (1, 3621)).sub_id
    # `exclude` keyword (section 4.3': the incumbent read excludes the record being started).
    assert pool.active_record(CONF, 1, 2, -1, 3819, exclude=r8) is None
    # Spawn-liveness (>= R, not > R): a record whose parent ends AT R is still live at R.
    r4 = next(r for r in pool.records_for(CONF, 0, 1) if r.trigger_type == "reversal")
    assert r4.trigger_end_idx == 3611 and r4.is_live_at_reversal(3611) and not r4.is_active_at(3611)
    r2 = next(r for r in pool.records_for(CONF, 0, 0) if r.trigger_idx == 2470)
    assert not r2.is_live_at_reversal(2829)          # ended 2611 < 2829 -> did not spawn


def test_post_sweep_invariants_hold(sweep):
    """The section-4.5 asserts, re-checked independently of the sweep's own."""
    pool: SubStructurePool = sweep["pool"]
    for s in pool.all():
        for r in s.records:
            assert r.sub_id == s.sub_id
            if r.is_zero_length:
                continue
            assert r.end_idx is None or r.start_idx < r.end_idx, r
            assert s.end_idx is None or r.start_idx <= s.end_idx, r
        if s.start_idx is not None:
            assert s.end_idx is None or s.end_idx > s.start_idx, s
            assert s.start_idx == min(r.start_idx for r in s.live_records())
    # Half-open non-overlap per (lens, S, C, direction) over live records.
    groups: Dict[Tuple[str, int, int, int], List[TriggerRecord]] = {}
    for r in pool.all_records():
        if not r.is_zero_length:
            groups.setdefault((r.lens, r.parent_sid, r.parent_cycle_id, r.direction), []).append(r)
    for g, rs in groups.items():
        rs = sorted(rs, key=lambda r: r.start_idx)
        for a, b in zip(rs, rs[1:]):
            assert a.end_idx is not None and a.end_idx <= b.start_idx, (g, a, b)
    # At most one active record per (lens, S, C, direction) at every record boundary.
    for r in pool.all_records():
        for at in (r.start_idx, r.end_idx):
            if at is None:
                continue
            for g in groups:
                pool.active_record(*g, at)    # raises on > 1


# =============================================================================
# Section 2.1 / 5.3 -- the CACHE-HIT variant of the predicted table
# =============================================================================
#
# On the LANDED replay three probes were cache hits (section 5.3: same direction
# + same initial input as an earlier probe -> the earlier finalize is inherited
# RAW, `cache_hit=True`); measured in the landed `_triggers.csv`:
#
#   record                       own-run finalize   inherited finalize   inherited from
#   FC(0,1)          @2611       2608               2470 no_retrace      the reversal handoff probe at
#                                                                        R=2470 (dir +1 -> 2365)
#   first_counter(0,1) @2843     2843               2829 no_retrace      the reversal handoff probe at
#                                                                        R=2829 (dir -1 -> 2639)
#   reversal of 3760/-1 @4200    4200               4083 no_retrace      subsequent_counter(1,2)'s probe
#   (the baseline's "sub 6")                                             at 4083 (dir +1 -> 4027)
#
# Section 2.1: start_idx = max(probe_finalize_idx, trigger_idx, parent_floor_idx) --
# "trigger_idx when the structure was already known before this trigger fired
# (re-trigger: inherited finalize ...)". Each inherited finalize precedes its own
# trigger_idx, so the other two terms absorb it and every lifecycle value is
# IDENTICAL to the own-run table:
#
#   R3  FC(0,1):            max(2470, 2611, 2611) = 2611   (trigger == floor bind; was max(2608, 2611, 2611))
#   R5  first_counter(0,1): max(2829, 2843, 2611) = 2843   (trigger binds;          was max(2843, 2843, 2611))
#   R10 reversal @4200:     max(4083, 4200, 3611) = 4200   (trigger binds;          was max(4200, 4200, 3611))
#
# No end depends on a finalize (own reversal / parent_end / a replacer's
# start_idx only), so every record end, every sub window and every spawn is
# unchanged. The historical `probe_finalize_idx` on the three records IS the
# inherited value (never adjusted); `probe_finalize_condition` follows it.
#
# The own-run table cannot detect a dropped `trigger_idx` term (there, every
# non-FC finalize == trigger and FC(0,1)'s floor ties the trigger); this variant
# can (R5 -> 2829, R10 -> 4083).

# (trigger_type, S, C) -> (inherited finalize_idx, finalize_condition)
_CACHE_HIT_START: Dict[Tuple[str, int, int], Tuple[int, str]] = {
    ("first_confluence", 0, 1): (2470, "no_retrace"),
    ("first_counter", 0, 1): (2829, "no_retrace"),
}
# (reversing sub's starting_idx, R) -> (inherited finalize_idx, finalize_condition)
_CACHE_HIT_REVERSAL: Dict[Tuple[int, int], Tuple[int, str]] = {
    (3760, 4200): (4083, "no_retrace"),
}


def _make_cache_hit_stubs(pool: SubStructurePool, calls: Dict[str, list]):
    """The section-9.2 stubs with the three measured cache hits layered on:
    the resolver returns the INHERITED finalize (+ condition) with
    `cache_hit=True`; everything else is `_make_stubs` verbatim."""
    base_start, build_geometry, base_reversal = _make_stubs(pool, calls)

    def resolve_start(trigger: SweepTrigger, hi: int):
        res = base_start(trigger, hi)
        k = (trigger.trigger_type, trigger.parent_sid, trigger.parent_cycle_id)
        if k in _CACHE_HIT_START:
            fin, cond = _CACHE_HIT_START[k]
            res = res._replace(finalize_idx=fin, finalize_condition=cond, cache_hit=True)
        return res

    def resolve_reversal(sub: PooledStructure, R: int, probe_direction: int):
        res = base_reversal(sub, R, probe_direction)
        k = (sub.starting_idx, R)
        if k in _CACHE_HIT_REVERSAL:
            fin, cond = _CACHE_HIT_REVERSAL[k]
            res = res._replace(finalize_idx=fin, finalize_condition=cond, cache_hit=True)
        return res

    return resolve_start, build_geometry, resolve_reversal


@pytest.fixture(scope="module")
def sweep_cache_hit(tables) -> Dict[str, Any]:
    """Second module-level sweep: the same 11 H1 triggers (+ the 4 spawned
    reversals) with the three inherited finalizes. Independent pool."""
    t, _ = tables
    pool = SubStructurePool()
    calls: Dict[str, list] = {"resolve_start": [], "build_geometry": [], "resolve_reversal": []}
    resolve_start, build_geometry, resolve_reversal = _make_cache_hit_stubs(pool, calls)
    lines: List[str] = []
    triggers = _h1_triggers()
    result = run_lifecycle_sweep(
        triggers, pool=pool, tables=t,
        resolve_start=resolve_start, build_geometry=build_geometry,
        resolve_reversal=resolve_reversal, synth_reversal_trigger=_synth_stub,
        parent_path=PARENT, sub_tf=SUB_TF, log=lines.append,
    )
    assert isinstance(result, SweepResult) and result.pool is pool
    return {"pool": pool, "result": result, "calls": calls, "lines": lines, "triggers": triggers}


# The three inherited records, keyed by record identity (lens, S, C, type, trigger_idx).
_INHERITED_BY_IDENT: Dict[Tuple[str, int, int, str, int], Tuple[int, str]] = {
    (CONF, 0, 1, "first_confluence", 2611): (2470, "no_retrace"),
    (CTR, 0, 1, "first_counter", 2843): (2829, "no_retrace"),
    (CONF, 1, 2, "reversal", 4200): (4083, "no_retrace"),
}


def _lifecycle_row(pool: SubStructurePool, r: TriggerRecord) -> tuple:
    """Everything in a record EXCEPT the probe provenance -- what must be
    identical between the own-run and the cache-hit sweeps."""
    sub = pool.get_by_id(r.sub_id)
    return (
        _rec_ident(r), (sub.direction, sub.starting_idx), r.trigger_sub_sid, r.relative_dir,
        r.parent_floor_idx, r.start_idx, r.trigger_end_idx, r.end_idx, r.end_reason,
        (None if r.ended_by_sub_id is None
         else (pool.get_by_id(r.ended_by_sub_id).direction, pool.get_by_id(r.ended_by_sub_id).starting_idx)),
        r.is_zero_length, tuple(r.extra_trigger_idxs),
    )


def test_cache_hit_variant_records_carry_the_inherited_finalize(sweep_cache_hit):
    pool: SubStructurePool = sweep_cache_hit["pool"]
    by_ident = {_rec_ident(r): r for r in pool.all_records()}
    assert len(by_ident) == 11
    for ident, (fin, cond) in _INHERITED_BY_IDENT.items():
        r = by_ident[ident]
        # historical: the inherited value, raw -- it precedes the record's own trigger_idx
        assert (r.probe_finalize_idx, r.probe_finalize_condition) == (fin, cond), ident
        assert r.probe_finalize_idx < r.trigger_idx, ident
        # section 2.1: the trigger / floor terms absorb it -- start_idx == the own-run table's
        e = next(e for e in EXPECTED_RECORDS
                 if (e.lens, e.parent[0], e.parent[1], e.trigger_type, e.trigger_idx) == ident)
        assert r.start_idx == max(fin, r.trigger_idx, r.parent_floor_idx) == e.start_idx, ident
    # the trigger term is what binds on two of them; the floor ties it on FC(0,1)
    assert by_ident[(CTR, 0, 1, "first_counter", 2843)].start_idx == 2843 > 2829
    assert by_ident[(CONF, 1, 2, "reversal", 4200)].start_idx == 4200 > 4083
    assert by_ident[(CONF, 0, 1, "first_confluence", 2611)].start_idx == 2611 == 2611  # trigger == floor
    # every other record's probe provenance is the own-run value
    for ident, r in by_ident.items():
        if ident in _INHERITED_BY_IDENT:
            continue
        e = next(e for e in EXPECTED_RECORDS
                 if (e.lens, e.parent[0], e.parent[1], e.trigger_type, e.trigger_idx) == ident)
        assert (r.probe_finalize_idx, r.probe_finalize_condition) == (e.finalize_idx, e.finalize_condition), ident


def test_cache_hit_variant_lifecycle_table_is_identical(sweep, sweep_cache_hit):
    """Every record's lifecycle (start/trigger_end/end/reason/ended_by/tss/
    relative_dir/floor) and every sub's window/reason/lenses/record count are
    IDENTICAL to the own-run sweep AND to the predicted table."""
    p0: SubStructurePool = sweep["pool"]
    p1: SubStructurePool = sweep_cache_hit["pool"]
    # subs: same creation order, same sub_ids
    assert [(s.direction, s.starting_idx, s.sub_id) for s in p1.all()] == \
           [(s.direction, s.starting_idx, s.sub_id) for s in p0.all()]
    assert [(s.direction, s.starting_idx) for s in p1.all()] == EXPECTED_SUB_ORDER
    # records: same identities in the same creation order, same lifecycle rows
    rows0 = [_lifecycle_row(p0, r) for r in p0.all_records()]
    rows1 = [_lifecycle_row(p1, r) for r in p1.all_records()]
    assert rows1 == rows0
    assert len(rows1) == 11
    # ... and equal to the predicted table (independently of the own-run pool)
    for r, e in zip(p1.all_records(), EXPECTED_RECORDS):
        ended_by = _sub(p1, e.ended_by).sub_id if e.ended_by is not None else None
        got = (_rec_ident(r), (r.direction, r.starting_idx), r.trigger_sub_sid, r.relative_dir,
               r.parent_floor_idx, r.start_idx, r.trigger_end_idx, r.end_idx, r.end_reason, r.ended_by_sub_id)
        want = ((e.lens, e.parent[0], e.parent[1], e.trigger_type, e.trigger_idx), e.sub, e.tss,
                e.relative_dir, e.floor, e.start_idx, e.trigger_end_idx, e.end_idx, e.end_reason, ended_by)
        assert got == want, f"record {_rec_ident(r)}: {got} != {want}"
        assert not r.is_zero_length
    # subs: window / reason / lenses / natural reversal / record count
    for e in EXPECTED_SUBS:
        s1, s0 = _sub(p1, e.sub), _sub(p0, e.sub)
        got = (s1.start_idx, s1.end_idx, s1.end_reason, frozenset(s1.lenses()), s1.natural_reversal_idx, len(s1.records))
        assert got == (e.start_idx, e.end_idx, e.end_reason, e.lenses, e.natural_reversal_idx, e.n_records), e.sub
        assert got == (s0.start_idx, s0.end_idx, s0.end_reason, frozenset(s0.lenses()), s0.natural_reversal_idx, len(s0.records))
        assert list(s1.relative_dir_segments) == list(s0.relative_dir_segments), e.sub
    # unresolved rows + spawned reversal triggers: identical
    r0: SweepResult = sweep["result"]
    r1: SweepResult = sweep_cache_hit["result"]
    key_u = lambda u: (u.lens, u.parent_sid, u.parent_cycle_id, u.trigger_type, u.trigger_idx, u.direction, u.reason)
    assert sorted(map(key_u, r1.unresolved)) == sorted(map(key_u, r0.unresolved))
    assert [(t.lens, t.parent_sid, t.parent_cycle_id, t.trigger_idx, t.direction) for t in r1.spawned] == EXPECTED_SPAWNED
    # the probe / geometry call pattern is the same (a cache hit changes the VALUE, not the call)
    c0, c1 = sweep["calls"], sweep_cache_hit["calls"]
    assert c1["resolve_start"] == c0["resolve_start"]
    assert c1["resolve_reversal"] == c0["resolve_reversal"]
    assert [(k, b) for k, b in c1["build_geometry"]] == [(k, b) for k, b in c0["build_geometry"]]
