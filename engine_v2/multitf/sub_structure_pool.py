"""Sub-structure pool — Phase 2 (PART4_REFACTOR_SPEC.md §17).

Deduplicates subordinate structures across the confluence/counter lenses and
across parent cycles: a sub is uniquely identified by
`(parent_path, sub_TF, direction, starting_idx)` with **absolute** direction, so
any trigger that resolves to the same four reuses one `PooledStructure` instead
of re-running MarketStructure + the downstream pipeline.

This module is the pure data model + lifecycle logic. It is deliberately
free of pandas / MS / chart imports so it can be unit-tested in isolation and
depended on by both the build layer (`entity_df_mutation` / orchestrator) and
the chart layer. The natural-end structure BUILDER (`build_pooled_structure`)
and the render projection land in Stage 2 — this stage is dead code (imported
by nothing on the live path) and validated by unit tests + a byte-identical
`/compare`.

Key model points (see §17 for the authoritative spec):

- **Structure vs TriggerRecord (§17.2).** `PooledStructure` = one MS run +
  geometry, computed once, run to its NATURAL end. `TriggerRecord` = per-trigger
  metadata pointing at a structure; N:1.
- **Direction is absolute; alignment is a derived per-chart label (§17.3).**
  A sub is drawn on the confluence chart iff a `TriggerRecord` assigned it there,
  else/also counter. `reversal` triggers are sticky-per-chart (inherit the lens
  of the sub they reversed from); the four named triggers map by use_case.
- **Lifecycle (§17.6).** `start = min(trigger_dt)`; `end = min(end-candidate ≥
  max(trigger_dt))` over the end set {own reversal, same-direction replacement,
  parent-lifecycle-end}. The `≥ max start` keeps a cross-parent-cycle sub
  continuous across an intervening parent-cycle boundary (canonical: M15 3304).
  See LANDMINES "Sub-Structure Pool: Lifecycle End Is min(end ≥ max start)".
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple


# --- Identity -----------------------------------------------------------------

class StructureKey(NamedTuple):
    """Pool identity for a unique sub structure (§17.3).

    `direction` is ABSOLUTE (+1 / -1) — dedup keys off the actual MS run, not the
    confluence/counter label (which is derived per-chart). `parent_path` (not just
    parent TF) so an M5-under-counter can't collide with an M5-under-confluence.
    `starting_idx` is entity-absolute on the shared sub-TF frame.
    """
    parent_path: str
    sub_tf: str
    direction: int
    starting_idx: int


# Lens (chart) labels. A sub can appear on both.
LENS_CONFLUENCE = "confluence"
LENS_COUNTER = "counter"

# Named-trigger → lens mapping (§17.4). `reversal` is NOT here — it is
# sticky-per-chart (inherits the reversing sub's lens), resolved by the caller.
_USE_CASE_LENS = {
    "first_confluence": LENS_CONFLUENCE,
    "subsequent_confluence": LENS_CONFLUENCE,
    "first_counter": LENS_COUNTER,
    "subsequent_counter": LENS_COUNTER,
}

# End-reason tie-break priority (lower wins when two candidates share an idx).
# Cosmetic only — reversal is the most structure-intrinsic cause, so it wins.
_END_REASON_PRIORITY = {
    "reversal": 0,
    "same_dir_replacement": 1,
    "parent_end": 2,
}


def knowable_at_idx(
    ev_type: str, ev_idx: int, confirmed_at: Optional[int] = None,
) -> int:
    """The idx at which an event became KNOWN — the clip key for rendering a
    pooled structure into a lens window (§17.11 knowable-at clip).

    `BOS_CONFIRMED.ev.idx` is the BOS EXTREME candle, but the break isn't known
    until `meta["confirmed_at"]` (later); every other event is known at `ev.idx`.
    Clipping by knowable-at (not `ev.idx`) matches the Phase-1 bounded-run
    semantics and neutralizes the boundary-straddling-confirmation case: a BOS
    whose extreme is inside the window but whose confirmation landed past it is
    correctly excluded (a bounded run to the window could not have known it).
    See LANDMINES "Run Cap != Lifecycle End; Knowable-At Clip on Render".
    """
    if ev_type == "BOS_CONFIRMED" and confirmed_at is not None:
        return int(confirmed_at)
    return int(ev_idx)


def resolve_lens(use_case: str, *, reversed_from_lens: Optional[str] = None) -> str:
    """Chart-lens for a trigger (§17.4).

    The four named variations map by `use_case`. A `reversal` trigger is
    sticky-per-chart: it inherits the lens of the sub it reversed from
    (`reversed_from_lens` — required for `use_case == "reversal"`). A sub can end
    up on both charts when triggers of both lenses map to it; that union is
    computed at the `PooledStructure` level (`lenses`), not here.
    """
    if use_case == "reversal":
        if reversed_from_lens is None:
            raise ValueError(
                "resolve_lens: reversal trigger requires reversed_from_lens "
                "(sticky-per-chart, §17.4)"
            )
        return reversed_from_lens
    lens = _USE_CASE_LENS.get(use_case)
    if lens is None:
        raise ValueError(f"resolve_lens: unknown use_case {use_case!r}")
    return lens


# --- Trigger record -----------------------------------------------------------

@dataclass(frozen=True)
class TriggerRecord:
    """One trigger that maps to a `PooledStructure` (§17.2).

    A start-trigger only — it carries the lifecycle-start candidate (`trigger_dt`)
    and its chart attribution (`lens`). End resolution is a pool-level
    computation over ALL subs (`finalize_lifecycles`), not stored here.

    `trigger_dt` is the lifecycle-start idx we already use as `start_trigger_idx`
    (§5): bootstrap (`first_*`) → probe finalize idx; `subsequent_*` → trigger
    candle; `reversal` → reversal-apply idx. Entity-absolute on the sub-TF frame.
    """
    trigger_type: str                 # first_confluence | subsequent_confluence
                                      # | first_counter | subsequent_counter | reversal
    trigger_dt: int                   # lifecycle-start candidate (§17.6)
    parent_sid: int
    parent_cycle_id: int
    lens: str                         # LENS_CONFLUENCE | LENS_COUNTER
    meta: Dict[str, Any] = field(default_factory=dict)


# --- Pooled structure ---------------------------------------------------------

@dataclass
class PooledStructure:
    """A unique sub structure (§17.2). Computed once; many triggers point at it.

    Identity is `key`. `sub_id` is a single monotonic pool index (§17.4). The
    geometry payload (MS run + derived elements) is attached by the Stage-2
    builder; this stage carries the identity, trigger list, natural-end marker,
    and the finalized lifecycle.
    """
    key: StructureKey
    sub_id: int
    trigger_records: List[TriggerRecord] = field(default_factory=list)

    # Set by the natural-end builder (Stage 2). The structure's own first reversal
    # (entity-absolute on the sub-TF frame); None if it ran to the edge without
    # reversing. Used as an end candidate and to spawn the reversal successor.
    natural_reversal_idx: Optional[int] = None

    # Opaque geometry payload (LowerTFResult-shaped) attached at creation by the
    # Stage-2 builder. Cap-free (natural end); the lifecycle projection is applied
    # per unique sub in the post-pass, once (§17.7). Untyped here to keep this
    # module pandas/MS-free.
    geometry: Any = None

    # Finalized by `finalize_lifecycles` (post-pass, §17.6). Entity-absolute.
    lifecycle_start: Optional[int] = None
    lifecycle_end: Optional[int] = None
    lifecycle_end_reason: Optional[str] = None

    meta: Dict[str, Any] = field(default_factory=dict)

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
        """Group key for the ≤1-active-per-(parent, TF, direction) rule (§17.6)."""
        return (self.key.parent_path, self.key.sub_tf, self.key.direction)

    def add_trigger(self, tr: TriggerRecord) -> None:
        self.trigger_records.append(tr)

    def lenses(self) -> set:
        """Charts this sub renders on = union of its trigger records' lenses.

        A sub with both a confluence-lens and a counter-lens trigger appears on
        BOTH charts (§17.4).
        """
        return {tr.lens for tr in self.trigger_records}

    def memberships(self) -> List[Tuple[int, int]]:
        """Distinct `(parent_sid, parent_cycle_id)` this sub is triggered in,
        in first-seen order (§17.4). A unique sub can span several parent cycles
        (canonical: M15 3304 → (1,1) and (1,2))."""
        out: List[Tuple[int, int]] = []
        seen = set()
        for tr in self.trigger_records:
            m = (tr.parent_sid, tr.parent_cycle_id)
            if m not in seen:
                seen.add(m)
                out.append(m)
        return out

    def earliest_membership(self) -> Optional[Tuple[int, int]]:
        """`(parent_sid, parent_cycle_id)` of the earliest trigger — the chart
        label's parent identity (§17.10)."""
        if not self.trigger_records:
            return None
        tr = min(self.trigger_records, key=lambda t: t.trigger_dt)
        return (tr.parent_sid, tr.parent_cycle_id)


# --- The pool -----------------------------------------------------------------

class SubStructurePool:
    """Session-scoped store of unique sub structures (§17.4).

    Single-entity storage: one pool holds every sub across both lenses and all
    parent cycles. `get_or_create` is the dedup interception point (§17.7): the
    first trigger to resolve a `(parent_path, sub_TF, direction, starting_idx)`
    creates the `PooledStructure` (the caller then runs MS + downstream and
    attaches geometry); later triggers reuse it and just append their
    `TriggerRecord`.

    Also carries the secondary probe cache (§17.7) keyed on the *initial* probe
    start — `(parent_path, sub_TF, direction, initial_start)` → final start —
    so a repeated probe with the same key (different `end_idx` accepted) need not
    rerun. Kept here so both caches share the pool's lifetime; wired in Stage 4.
    """

    def __init__(self) -> None:
        self._by_key: Dict[StructureKey, PooledStructure] = {}
        self._next_sub_id = 0
        # Probe cache: (parent_path, sub_tf, direction, initial_start) -> final_start
        self._probe_cache: Dict[Tuple[str, str, int, int], int] = {}

    # --- structure dedup ---
    def get(self, key: StructureKey) -> Optional[PooledStructure]:
        return self._by_key.get(key)

    def get_or_create(self, key: StructureKey) -> Tuple[PooledStructure, bool]:
        """Return `(structure, created)`. `created` is True only on the first
        request for `key` (caller must then build + attach geometry); False on a
        dedup hit (caller only appends its `TriggerRecord`)."""
        existing = self._by_key.get(key)
        if existing is not None:
            return existing, False
        s = PooledStructure(key=key, sub_id=self._next_sub_id)
        self._next_sub_id += 1
        self._by_key[key] = s
        return s, True

    def all(self) -> List[PooledStructure]:
        """Structures in creation order (deterministic → stable `sub_id`)."""
        return sorted(self._by_key.values(), key=lambda s: s.sub_id)

    def for_lens(self, lens: str) -> List[PooledStructure]:
        """Structures that render on `lens`'s chart (§17.4), in `sub_id` order."""
        return [s for s in self.all() if lens in s.lenses()]

    # --- probe cache ---
    def probe_cached_start(
        self, parent_path: str, sub_tf: str, direction: int, initial_start: int,
    ) -> Optional[int]:
        return self._probe_cache.get((parent_path, sub_tf, direction, initial_start))

    def record_probe_start(
        self, parent_path: str, sub_tf: str, direction: int,
        initial_start: int, final_start: int,
    ) -> None:
        self._probe_cache.setdefault(
            (parent_path, sub_tf, direction, initial_start), final_start,
        )


# --- Lifecycle resolution (§17.6) ---------------------------------------------

def select_lifecycle_end(
    max_start: int, candidates: Iterable[Tuple[int, str]],
) -> Tuple[Optional[int], Optional[str]]:
    """Pick the lifecycle end from `candidates` = iterable of `(idx, reason)`.

    Returns `(end_idx, end_reason)` = the min-idx candidate with `idx ≥
    max_start`, else `(None, None)` (open lifecycle). Ties on idx break by
    `_END_REASON_PRIORITY` for determinism.

    The `≥ max_start` filter is the crux (§17.6): a candidate earlier than the
    sub's LAST trigger is a boundary the sub was re-triggered past, so it must not
    end it. This keeps a cross-parent-cycle sub continuous instead of a degenerate
    same-dt end/restart. See LANDMINES "Lifecycle End Is min(end ≥ max start)".
    """
    eligible = [
        (idx, reason) for (idx, reason) in candidates if idx >= max_start
    ]
    if not eligible:
        return None, None
    idx, reason = min(
        eligible, key=lambda c: (c[0], _END_REASON_PRIORITY.get(c[1], 99)),
    )
    return idx, reason


def finalize_lifecycles(
    structures: List[PooledStructure],
    parent_end_lookup: Dict[Tuple[int, int], Optional[int]],
) -> None:
    """Finalize `lifecycle_start` / `lifecycle_end` / `lifecycle_end_reason` on
    every structure IN PLACE (post-pass, §17.6 / §17.7).

    Two passes because ends depend on other subs' starts (same-direction
    replacement), but starts depend on nothing:

      1. `start = min(trigger_dt)` over the sub's triggers.
      2. `end = select_lifecycle_end(max(trigger_dt), candidates)` where
         candidates =
           - `(natural_reversal_idx, "reversal")` if the sub reversed;
           - `(other.lifecycle_start, "same_dir_replacement")` for every OTHER sub
             sharing `(parent_path, sub_TF, direction)` — the ≤1-active-per-
             direction rule, incl. cross-chain reversal successors (a reversal
             from the confluence chain spawns a counter-direction sub that ends
             the active counter-direction sub);
           - `(parent_end_lookup[m], "parent_end")` for each parent-cycle
             membership `m` (parent-cycle-end / parent reversal, §6.5).

    `parent_end_lookup` maps `(parent_sid, parent_cycle_id)` → its lifecycle-end
    idx on the sub-TF frame (None if that parent cycle is still open → no cap).
    A structure with no triggers is skipped (degenerate; should not occur).
    """
    # Pass 1 — starts.
    for s in structures:
        if not s.trigger_records:
            continue
        s.lifecycle_start = min(tr.trigger_dt for tr in s.trigger_records)

    by_dir: Dict[Tuple[str, str, int], List[PooledStructure]] = defaultdict(list)
    for s in structures:
        if s.trigger_records:
            by_dir[s.dir_key].append(s)

    # Pass 2 — ends.
    for s in structures:
        if not s.trigger_records:
            continue
        max_start = max(tr.trigger_dt for tr in s.trigger_records)
        candidates: List[Tuple[int, str]] = []

        if s.natural_reversal_idx is not None:
            candidates.append((int(s.natural_reversal_idx), "reversal"))

        for other in by_dir[s.dir_key]:
            if other is s or other.lifecycle_start is None:
                continue
            candidates.append((int(other.lifecycle_start), "same_dir_replacement"))

        # Parent-cycle-end: ONLY the sub's LATEST membership cycle caps it. A sub
        # that spans several parent cycles lives until its last cycle ends — an
        # EARLIER membership's cycle-end is a boundary it is triggered past, not a
        # terminal (else a multi-cycle sub like M15 2365 would be cut at cycle 0's
        # end even though it continues into cycle 1). `max()` over memberships is
        # chronological (sids increase over time; cycles within a sid).
        _memberships = s.memberships()
        if _memberships:
            pe = parent_end_lookup.get(max(_memberships))
            if pe is not None:
                candidates.append((int(pe), "parent_end"))

        end, reason = select_lifecycle_end(max_start, candidates)
        s.lifecycle_end = end
        s.lifecycle_end_reason = reason
