"""Sub-structure pool — Phase 2 (PART4_REFACTOR_SPEC.md §17, rev 2 2026-09-19).

The pool holds the two objects of the sub-structure model (§17.2):

- **`PooledStructure` = the unique sub** — one MS run + its cap-free geometry,
  identified by `(parent_path, sub_TF, direction, starting_idx)` with ABSOLUTE
  direction (§17.3), `sub_id` global monotonic (creation order). NOT bound to a
  parent: it spans parent cycles and parent sids, and carries ONE real-time
  lifecycle `[start_idx, end_idx]` aggregated from its records (§17.5). This is
  what we trade on.
- **`TriggerRecord` = a triggered instance** of a unique sub — identity
  `(lens, parent_sid, parent_cycle_id, trigger_sub_sid)`, FK `sub_id` (never
  None), its OWN parent-bound lifecycle (§17.4) and the chart lens.
- **`UnresolvedTrigger`** — a trigger that produced no record (§17.7).

This module is the pure data model. It is deliberately free of pandas / MS /
chart imports so it can be unit-tested in isolation and depended on by the
sweep (`multitf/lifecycle_sweep.py`), the build layer (`entity_df_mutation`),
the orchestrator and the chart. The lifecycle *evaluation* — the ordered sweep
of §17.6 — lives in `lifecycle_sweep.py`; nothing here computes a lifecycle.

Landed by Plan C (`plans/PLAN_C_lifecycle_rewrite.md`). Rev 1's post-pass
(`finalize_lifecycles`: `start = min(trigger_dt)`, pool-wide `end = min(end ≥
max start)`) is gone — see LANDMINES "Sub-Structure Pool" entries and GOTCHAS
"A Lifecycle Floor That Lives in a Build Function Is Lost…" for why.

**One principle (§17 header):** lifecycle fields (`start_idx`, `end_idx`,
`trigger_end_idx`) are REAL-TIME; pattern/element fields (`starting_idx`,
`trigger_idx`, `probe_finalize_idx`) are HISTORICAL. Historical fields are
inputs to a lifecycle value, never used as one.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple


# --- Identity -----------------------------------------------------------------

class StructureKey(NamedTuple):
    """Pool identity for a unique sub structure (§17.3).

    `direction` is ABSOLUTE (+1 / -1) — dedup keys off the actual MS run, not the
    confluence/counter label (which lives on the record as `relative_dir` /
    `lens`). `parent_path` (not just parent TF) so an M5-under-counter can't
    collide with an M5-under-confluence. `starting_idx` is entity-absolute on
    the shared sub-TF frame.
    """
    parent_path: str
    sub_tf: str
    direction: int
    starting_idx: int


# Lens (chart) labels. A sub can appear on both.
LENS_CONFLUENCE = "confluence"
LENS_COUNTER = "counter"

# Named-trigger → lens mapping (§17.3). `reversal` is NOT here — it is
# sticky-per-chart (inherits the reversing record's lens), resolved by the caller.
_USE_CASE_LENS = {
    "first_confluence": LENS_CONFLUENCE,
    "subsequent_confluence": LENS_CONFLUENCE,
    "first_counter": LENS_COUNTER,
    "subsequent_counter": LENS_COUNTER,
}

# End-reason priority at an EQUAL idx (lower wins) — §17.4: reversal is the most
# structure-intrinsic cause, then the parent boundary, then a replacement. Used
# by the sweep's phase 3 (record end) and phase 4 (sub end tie-break).
_END_REASON_PRIORITY = {
    "reversal": 0,
    "parent_end": 1,
    "same_dir_replacement": 2,
}

# The unresolved-trigger reasons (§17.7).
UNRESOLVED_REASONS = frozenset(
    {"pending", "degenerate_parent_cycle", "probe_failed", "geometry_failed"}
)


def knowable_at_idx(
    ev_type: str, ev_idx: int, confirmed_at: Optional[int] = None,
) -> int:
    """The idx at which an event became KNOWN — the clip key for projecting a
    pooled structure into a lifecycle window (§17.9 / `project_to_window`).

    `BOS_CONFIRMED.ev.idx` is the BOS EXTREME candle, but the break isn't known
    until `meta["confirmed_at"]` (later); every other event is known at `ev.idx`.
    Clipping by knowable-at (not `ev.idx`) neutralizes the
    boundary-straddling-confirmation case: a BOS whose extreme is inside the
    window but whose confirmation landed past it is correctly excluded.
    Known limit (§17.12): `CTS_ESTABLISHED` / `REVERSAL_CANDIDATE` straddle too
    and are not special-cased here.
    """
    if ev_type == "BOS_CONFIRMED" and confirmed_at is not None:
        return int(confirmed_at)
    return int(ev_idx)


def resolve_lens(use_case: str, *, reversed_from_lens: Optional[str] = None) -> str:
    """Chart-lens for a trigger (§17.3) — THE lens rule (the `"confluence" in
    sub_path_id` substring test is retired).

    The four named variations map by `use_case`. A `reversal` trigger is
    sticky-per-chart: it inherits the lens of the record whose sub reversed
    (`reversed_from_lens` — required for `use_case == "reversal"`). A sub can end
    up on both charts when records of both lenses map to it; that union is
    computed at the `PooledStructure` level (`lenses()`), not here.
    """
    if use_case == "reversal":
        if reversed_from_lens is None:
            raise ValueError(
                "resolve_lens: reversal trigger requires reversed_from_lens "
                "(sticky-per-chart, §17.3)"
            )
        return reversed_from_lens
    lens = _USE_CASE_LENS.get(use_case)
    if lens is None:
        raise ValueError(f"resolve_lens: unknown use_case {use_case!r}")
    return lens


# --- Trigger record (§17.4) ---------------------------------------------------

@dataclass
class TriggerRecord:
    """A triggered INSTANCE of a unique sub (§17.4) with its own lifecycle.

    Identity `(lens, parent_sid, parent_cycle_id, trigger_sub_sid)`; FK `sub_id`
    (never None). All idxs entity-absolute M15.

    Historical (never adjusted): `trigger_idx` (LOH of the parent trigger
    candle; native M15 for `reversal`), `probe_finalize_idx` (when THIS record's
    probe — the run keyed by (direction, initial input) — finalized; own run
    as-is, or the cached finalize on a probe-cache hit), `starting_idx` (= the
    sub's structural anchor), `validated_parent_idx`.

    Real-time: `start_idx = max(probe_finalize_idx, trigger_idx,
    parent_floor_idx)` — the record EXISTS from here and nothing earlier;
    `trigger_end_idx` = the first end condition (own reversal /
    same_dir_replacement / parent_end), or the sub's frozen end on a post-end
    re-trigger; `end_idx = max(trigger_end_idx, start_idx)`.

    Not frozen: the sweep writes the lifecycle fields in place (write-once).
    `source_trigger` is the originating `MultiTFTrigger` (opaque here; never
    exported) — it feeds `LowerTFResult.trigger` for the sub's projection when
    this is the sub's first live record.
    """
    # identity
    lens: str                        # LENS_CONFLUENCE | LENS_COUNTER
    parent_sid: int
    parent_cycle_id: int
    trigger_sub_sid: int             # per (lens, parent_sid, parent_cycle_id); creation-ordered
    # foreign key
    sub_id: int
    # provenance (historical)
    trigger_type: str                # first_confluence | subsequent_confluence | first_counter
                                     # | subsequent_counter | reversal
    trigger_idx: int
    probe_finalize_idx: int
    probe_finalize_condition: str
    validated_parent_idx: Optional[int]
    starting_idx: int
    direction: int
    sub_tf: str
    relative_dir: str                # "confluence" iff direction == parent_sd(parent sid) else "counter"
    parent_floor_idx: int            # the floor that was applied (diagnostic)
    # lifecycle (real-time)
    start_idx: int
    source_trigger: Any = None
    trigger_end_idx: Optional[int] = None
    end_idx: Optional[int] = None
    end_reason: Optional[str] = None  # "reversal" | "same_dir_replacement" | "parent_end" | None
    ended_by_sub_id: Optional[int] = None
    # bookkeeping
    seq: int = 0                     # creation order (deterministic tie-break; never identity)
    extra_trigger_idxs: List[int] = field(default_factory=list)

    @property
    def is_zero_length(self) -> bool:
        """Defined on `trigger_end_idx`, NOT `end_idx`, so it is already correct
        in phases 1–2 of the idx at which the record is ended (phase 3 writes
        `end_idx` later the same idx). A zero-length record participates in
        nothing (§17.4) — it is logged only."""
        return self.trigger_end_idx is not None and self.trigger_end_idx <= self.start_idx

    def is_active_at(self, at_idx: int) -> bool:
        """Interval rule on `trigger_end_idx` (half-open `[start_idx,
        trigger_end_idx)`): the incumbent test of the sweep's phase 1."""
        if self.is_zero_length:
            return False
        if self.start_idx > int(at_idx):
            return False
        return self.trigger_end_idx is None or self.trigger_end_idx > int(at_idx)

    def is_live_at_reversal(self, reversal_idx: int) -> bool:
        """The successor-spawn rule (§17.5): a record whose parent ends AT `R`
        is still live *at* `R` (closed on the right, unlike `is_active_at`)."""
        if self.is_zero_length:
            return False
        if self.start_idx > int(reversal_idx):
            return False
        return self.trigger_end_idx is None or self.trigger_end_idx >= int(reversal_idx)

    @property
    def identity(self) -> Tuple[str, int, int, int]:
        return (self.lens, self.parent_sid, self.parent_cycle_id, self.trigger_sub_sid)

    @property
    def scope(self) -> Tuple[str, int, int]:
        """The `(lens, parent_sid, parent_cycle_id)` scope of `trigger_sub_sid`."""
        return (self.lens, self.parent_sid, self.parent_cycle_id)


# --- Unresolved trigger (§17.7) -----------------------------------------------

@dataclass(frozen=True)
class UnresolvedTrigger:
    """A trigger that produced no record — logged, never built. No `sub_id`, no
    `trigger_sub_sid`. `reason ∈ UNRESOLVED_REASONS`; `probe_input_idx` is
    whatever was known (H1 for the four named types, M15 for `reversal`)."""
    lens: str
    parent_sid: int
    parent_cycle_id: int
    trigger_type: str
    trigger_idx: int
    direction: int
    probe_input_idx: Optional[int]
    reason: str
    detail: str


# --- Probe cache entry (§17.8) ------------------------------------------------

@dataclass(frozen=True)
class ProbeCacheEntry:
    """The full result of the first probe to finalize for a cache key
    `(parent_path, sub_tf, direction, initial_input_idx)` — the truth for every
    later probe of that key regardless of its `probe_end_idx` and reference
    zone (accepted approximation, §17.8)."""
    starting_idx: int
    finalize_idx: int
    finalize_condition: str
    bos0_inner: Optional[float]          # the FINAL iteration's BOS_0 threshold (moves on a reset)
    probe_end_idx: Optional[int]
    ref_inner: Optional[float] = None    # the reference zone's inner the probe was RUN against
                                         # (iteration 1's threshold) — the tripwire's comparand


# --- Pooled structure (§17.5) -------------------------------------------------

@dataclass
class PooledStructure:
    """A unique sub structure (§17.2/§17.5). Computed once; many records point
    at it. Identity is `key`; `sub_id` is the global creation index.

    `geometry` = `(bounded, slice_begin)` attached by the geometry builder
    (`entity_df_mutation.build_or_get_geometry`) — the natural-end MS run with
    run cap = the DATA EDGE, SLICE-LOCAL events/df + `slice_begin` for
    entity-absolute. Untyped here to keep this module pandas/MS-free.

    Lifecycle (real-time, set once by the sweep): `start_idx` = the first
    non-zero-length record's `start_idx`; `end_idx` per the §17.5 rule
    (`min` of record ends strictly `> max_start` over the records that EXIST
    at `t`); `relative_dir_segments` = the §17.3 step function.
    """
    key: StructureKey
    sub_id: int
    geometry: Any = None
    natural_reversal_idx: Optional[int] = None
    bos0_inner: Optional[float] = None
    records: List[TriggerRecord] = field(default_factory=list)
    # lifecycle (real-time; set once)
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None
    end_reason: Optional[str] = None
    relative_dir_segments: List[Tuple[int, str]] = field(default_factory=list)

    # --- convenience accessors ---
    @property
    def parent_path(self) -> str:
        return self.key.parent_path

    @property
    def sub_tf(self) -> str:
        return self.key.sub_tf

    @property
    def direction(self) -> int:
        return self.key.direction

    @property
    def starting_idx(self) -> int:
        return self.key.starting_idx

    @property
    def dir_key(self) -> Tuple[str, str, int]:
        """`(parent_path, sub_tf, direction)` — informational grouping key."""
        return (self.key.parent_path, self.key.sub_tf, self.key.direction)

    def live_records(self) -> List[TriggerRecord]:
        """Non-zero-length records, creation order (§17.4: a zero-length record
        participates in nothing)."""
        return [r for r in self.records if not r.is_zero_length]

    def lenses(self) -> Set[str]:
        """Charts this sub renders on = union of its LIVE records' lenses
        (§17.5). A sub with no live record has no lens (logged, not rendered)."""
        return {r.lens for r in self.live_records()}

    def relative_dir_at(self, at_idx: int) -> Optional[str]:
        """Read the step function at `at_idx` (None before the first segment)."""
        cur: Optional[str] = None
        for from_idx, rd in self.relative_dir_segments:
            if int(from_idx) <= int(at_idx):
                cur = rd
            else:
                break
        return cur


# --- The pool (§17.2 / §17.8) -------------------------------------------------

class SubStructurePool:
    """Session-scoped store of unique sub structures + their records + the
    unresolved-trigger log + the probe cache.

    Single-entity storage: one pool holds every sub across both lenses and all
    parent cycles. `get_or_create` is the dedup interception point — the
    geometry builder calls it ONLY after a successful MS run (§17.7: a failed
    build creates no entry and consumes no `sub_id`).
    """

    def __init__(self) -> None:
        self._by_key: Dict[StructureKey, PooledStructure] = {}
        self._by_id: Dict[int, PooledStructure] = {}
        self._next_sub_id = 0
        self._next_tss: Dict[Tuple[str, int, int], int] = defaultdict(int)
        self._seq = 0
        self.unresolved: List[UnresolvedTrigger] = []
        self._probe_cache: Dict[Tuple[str, str, int, int], ProbeCacheEntry] = {}

    # --- structure dedup ---
    def get(self, key: StructureKey) -> Optional[PooledStructure]:
        return self._by_key.get(key)

    def get_or_create(self, key: StructureKey) -> Tuple[PooledStructure, bool]:
        """Return `(structure, created)`. `created` is True only on the first
        request for `key`; False on a dedup hit."""
        existing = self._by_key.get(key)
        if existing is not None:
            return existing, False
        s = PooledStructure(key=key, sub_id=self._next_sub_id)
        self._next_sub_id += 1
        self._by_key[key] = s
        self._by_id[s.sub_id] = s
        return s, True

    def get_by_id(self, sub_id: int) -> PooledStructure:
        """The sub with `sub_id` (KeyError if absent)."""
        return self._by_id[int(sub_id)]

    def all(self) -> List[PooledStructure]:
        """Structures in creation order (deterministic → stable `sub_id`)."""
        return sorted(self._by_key.values(), key=lambda s: s.sub_id)

    # --- records ---
    def next_seq(self) -> int:
        """Global creation counter for records (the sweep's tie-break)."""
        v = self._seq
        self._seq += 1
        return v

    def next_trigger_sub_sid(self, lens: str, parent_sid: int, parent_cycle_id: int) -> int:
        """Starts at 0 per `(lens, parent_sid, parent_cycle_id)`; CONSUMES a value
        (call only when a trigger resolves to a NEW unique sub in that scope)."""
        scope = (lens, int(parent_sid), int(parent_cycle_id))
        v = self._next_tss[scope]
        self._next_tss[scope] = v + 1
        return v

    def add_record(self, rec: TriggerRecord) -> None:
        """Append a record to its sub (`sub.records` is the ONLY store — the
        scope queries below read through it, §4.3 step 7)."""
        self.get_by_id(rec.sub_id).records.append(rec)

    def records_for(self, lens: str, parent_sid: int, parent_cycle_id: int) -> List[TriggerRecord]:
        """Every record in the `(lens, parent_sid, parent_cycle_id)` scope,
        INCLUDING zero-length ones, creation (`seq`) order. Callers filter."""
        scope = (lens, int(parent_sid), int(parent_cycle_id))
        return sorted(
            (r for s in self._by_key.values() for r in s.records if r.scope == scope),
            key=lambda r: r.seq,
        )

    def all_records(self) -> List[TriggerRecord]:
        """Every record across every sub, creation (`seq`) order."""
        return sorted(
            (r for s in self._by_key.values() for r in s.records), key=lambda r: r.seq,
        )

    def active_record(
        self, lens: str, parent_sid: int, parent_cycle_id: int, direction: int,
        at_idx: int, *, exclude: Optional[TriggerRecord] = None,
    ) -> Optional[TriggerRecord]:
        """The ≤1 record of `(lens, S, C, direction)` active at `at_idx` per the
        interval rule on `trigger_end_idx` (`TriggerRecord.is_active_at`).
        Asserts at most one (§17.4 invariant)."""
        cands = [
            r for r in self.records_for(lens, parent_sid, parent_cycle_id)
            if r is not exclude and r.direction == int(direction) and r.is_active_at(at_idx)
        ]
        assert len(cands) <= 1, (
            f"[pool] >1 active record for ({lens},{parent_sid},{parent_cycle_id},"
            f"dir={direction}) at {at_idx}: "
            f"{[(r.identity, r.sub_id, r.start_idx, r.trigger_end_idx) for r in cands]}"
        )
        return cands[0] if cands else None

    # --- probe cache (§17.8) ---
    def get_cached_probe(
        self, parent_path: str, sub_tf: str, direction: int, initial_input_idx: int,
    ) -> Optional[ProbeCacheEntry]:
        return self._probe_cache.get((parent_path, sub_tf, int(direction), int(initial_input_idx)))

    def record_probe(
        self, parent_path: str, sub_tf: str, direction: int, initial_input_idx: int,
        entry: ProbeCacheEntry,
    ) -> None:
        """First write wins. A same-key probe never runs twice (a hit skips the
        probe), so a second write with a DIFFERENT entry is a bug — assert."""
        k = (parent_path, sub_tf, int(direction), int(initial_input_idx))
        prev = self._probe_cache.get(k)
        if prev is None:
            self._probe_cache[k] = entry
            return
        assert prev == entry, (
            f"[probe_cache] second write for key {k} differs: had {prev}, got {entry}"
        )
