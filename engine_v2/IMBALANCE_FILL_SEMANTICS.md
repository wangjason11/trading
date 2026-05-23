# Imbalance Fill Semantics

> Canonical reference for what it means for an `ImbalanceInstance` to be
> "filled." The fill predicate is consumed by FibTracker, MarketStructure's
> in-flight Scenario 2 snapshot, POI identification, and POI's per-candle
> activation sweep — so this doc lives outside `POI_ZONES_SPEC.md` to make
> clear it isn't POI-specific.

---

## The predicate: `ImbalanceInstance.is_filled`

Defined in `engine_v2/common/types.py::ImbalanceInstance.is_filled`. One
method, one rule:

```
is_filled(df, check_to_idx, fill_threshold=0.70) -> bool
```

An instance is **filled** iff both strokes of a two-stroke state machine
have fired by `check_to_idx`:

| Stroke | Direction | Condition |
|---|---|---|
| **Stroke 1 — armed** | bullish | first candle in `(end_idx, check_to_idx]` with `low <= gap_top - gap_size * fill_threshold` |
| | bearish | first candle in `(end_idx, check_to_idx]` with `high >= gap_bottom + gap_size * fill_threshold` |
| **Stroke 2 — confirmed** | bullish | first candle in `[armed_idx, check_to_idx]` with `close >= gap_top` |
| | bearish | first candle in `[armed_idx, check_to_idx]` with `close <= gap_bottom` |

Both strokes can fire on the same candle (rare but legal —
`armed_idx == confirmed_fill_idx`). Stroke 2 must be at `idx >= armed_idx`;
a close past the gap outer **before** any stroke 1 does not count.

The state machine is **monotonic in `check_to_idx`**: once `is_filled`
returns True at some `check_to_idx = t`, it returns True for all larger
values. Strokes latch once and never un-latch.

### Edge cases

| Case | Behavior |
|---|---|
| `gap_size <= 0` (degenerate gap) | Returns **True** — safe default; treat invalid gap as filled so it doesn't pollute "unfilled" counts |
| `end_idx >= check_to_idx` (empty scan range) | Returns **False** — a freshly-formed instance is unfilled until candles after it exist |
| Stroke 2 condition met before any stroke 1 | Returns **False** — stroke 2 only counts at `idx >= armed_idx` |
| Stroke 1 fires but stroke 2 never does | Returns **False** forever — instance stays "armed but not committed" until terminal end of any containing window |

### Why two strokes

A single-stroke definition (just the 70% retrace) treats every gap touch as
invalidation. Real markets often touch a gap and bounce — the touch alone
doesn't commit the move. The two-stroke definition requires both an
engagement (price retraces into the gap by 70%) AND a follow-through
commitment (price subsequently closes past the gap's outer edge in the
imbalance's direction). Until both happen, the gap is still "in play."

Concrete effect on POI activation: a per-candle sweep that flips
deactivate→reactivate→deactivate on every wick into a gap (the
pre-2026-05-23 behavior) collapses to a single deactivate at the candle
where stroke 2 confirms.

---

## The wrappers

### `has_unfilled_imbalance`

```
has_unfilled_imbalance(df, start_idx, end_idx, check_to_idx,
                       fill_threshold=0.70, *, direction=None) -> bool
```

In `engine_v2/patterns/imbalance.py`. For each instance in
`df.attrs["imbalances"]`:

1. **Window filter** — skip unless `inst.overlaps(start_idx, end_idx)`.
2. **Direction filter** — skip unless `direction is None` or
   `inst.direction == direction`.
3. **Fill check** — `inst.is_filled(df, check_to_idx, fill_threshold)`.

Returns `True` on the first unfilled survivor. Used by every Fib-side
consumer.

### `get_unfilled_imbalances`

Same as `has_unfilled_imbalance` but returns the list of unfilled
instances rather than a bool. No direction filter (always permissive
today). Used only by FibTracker's locking path.

---

## Consumer call-site matrix

Each consumer tunes three knobs: the **window** (which instances are in
scope), the **`check_to_idx`** (as-of time for the fill scan), and the
**direction filter**.

### POI consumers

| Site | Window | `check_to_idx` | Direction | Question |
|---|---|---|---|---|
| `zones/poi_zones.py::find_ic_candidates` | `(candidate_idx, cts_idx]` | `cts_idx` (Fib's locked CTS) | `direction=sd` | "Does this candidate IC have a post-IC sd-direction unfilled imbalance, at Fib lock time?" Gates IC qualification. Layer 1. |
| `zones/poi_zones.py::_compute_fill_idx_cache` | n/a (per-instance) | per-instance, scans to end of df | per-instance | "Per instance, what are `armed_idx` and `confirmed_fill_idx`?" Mirrors `is_filled` for efficiency in the per-candle activation sweep. Layer 2. |

### FibTracker consumers — all sd-direction strict (post-2026-05-23)

| Site | Purpose | Window | `check_to_idx` |
|---|---|---|---|
| `fib_tracker.py:120` (also `:1056`, `:1132`) | Scenario 2 cond1 — cycle 1 has unfilled | `[BOS_n, CTS_n]` (current cycle) | `cts_idx` |
| `fib_tracker.py:127` (also `:1065`, `:1139`) | Scenario 2 cond3 — did BOS_1 fill cycle 0? | `[BOS_0, CTS_0]` | `bos_idx` (prospective BOS_1) |
| `fib_tracker.py:275` | Regular Fib activation eligibility | `[BOS_n, CTS_n]` | `cts_idx` |
| `fib_tracker.py:943` | Cycle 0 snapshot for cross-cycle | `[BOS_0, CTS_0]` | `cts_idx` |
| `fib_tracker.py:1062`, `:1137` | Scenario 2 cond2 (cycle 1's own imbalance) | `[BOS_1, CTS_1]` | `cts_idx` |
| `fib_tracker.py:1073` | Normal Fib unfilled check (fallback) | `[BOS_n, CTS_n]` | `cts_idx` |
| `fib_tracker.py:1158` | Normal cycle 1 Fib check (Scenario 3 path) | `[BOS_1, CTS_1]` | `cts_idx` |
| `fib_tracker.py:1620` | `cross_cycle` mode: own cycle's imbalance | varies | varies (current candle in pre-established) |
| `fib_tracker.py:1650` | `cross_cycle` mode: dead-cycle walk | `[BOS_k, CTS_k]` for prior cycles | current candle (Interpretation B) |

### MarketStructure consumer — sd-direction strict

| Site | Purpose | Window | `check_to_idx` |
|---|---|---|---|
| `structure/market_structure.py:1791` (`_capture_cycle0_snapshot` / `_update_cycle0_data`) | In-flight Scenario 2 snapshot mirroring `fib_tracker:943` | `[BOS_0, CTS_0]` | `cts_idx` |

This mirror is load-bearing: the in-flight POI resolver
(`compute_poi_inners_for_cycle`) reads `cycle0_data["has_unfilled"]` to
make the same Scenario 2 decision as FibTracker's downstream layer.
Both have to use the same direction filter or anchor selection diverges
(see `LANDMINES.md` "Scenario 2 anchor agreement").

---

## Why every Fib + scenario consumer is strict to sd

The pre-2026-05-23 design left these calls permissive (no direction
filter), arguing that the BOS→CTS span is structurally directional and a
counter-direction imbalance inside it is geometrically unusual but should
still count when it appears.

Counter-direction imbalances inside a structural cycle's swing never
produce POIs (POIs are sd-direction by construction — see
`POI_ZONES_SPEC.md §4` "POIs are always sd-direction"). Since the
downstream consequence of a Fib activation / Scenario 2 decision is the
POI set, counter-direction imbalances driving those decisions creates a
disconnect between cause (counter-direction unfilled imbalance) and
effect (sd-direction POI). The strict filter aligns each consumer's input
with what it can actually influence downstream.

POI IC validation (`find_ic_candidates`) was already strict — it asks a
slightly different question ("did an institutional candle produce a
same-direction continuation push after the candidate?") which has always
required directional alignment. The asymmetry that used to exist between
Fib activation (permissive) and POI IC validation (strict) — discussed in
the old `POI_ZONES_SPEC.md §1` section — is now gone.

---

## The cache layer (POI-internal optimization)

`zones/poi_zones.py::_compute_fill_idx_cache` precomputes
`(armed_idx, confirmed_fill_idx)` per imbalance instance, returns a dict
keyed by `id(inst)`, AND stashes both indices on `inst.meta` for debug
exporters. This is consumed by `_compute_poi_activation_history` so its
per-candle sweep doesn't re-scan the `(end_idx, t]` window for every
candidate `t`.

**Equivalence invariant:** a caller of `is_filled(check_to_idx=t)` would
get the same boolean as
`confirmed_fill_idx is not None and confirmed_fill_idx <= t`. The cache
is a performance optimization, not a divergent semantics — any drift
between the two is a bug.

**Why it's POI-only:** the cache exists to amortize the cost of
per-candle evaluation. FibTracker and MarketStructure call
`has_unfilled_imbalance` once per CTS event, not per candle, so they
don't need it.

**Live-mode note:** under live (incremental) execution, `armed_idx` is
known at the moment stroke 1 fires; `confirmed_fill_idx` is None until
stroke 2 fires. The cache built today computes both eagerly over the
full df because backtest knows the entire price path up front. A live
adaptation would maintain both indices as state per instance, advancing
them as new candles arrive.

---

## Deliberate behavior shifts (pre → post)

The universal change (rolled in 2026-05-23) replaces the old single-stroke
70% retrace predicate with the two-stroke state machine. Where this matters:

| Consumer | Old (70% retrace only) | New (two-stroke) |
|---|---|---|
| **POI per-candle sweep** | Activation history can flip D↔A on every gap touch (e.g., M15.confluence sid=1 cyc=0 in the per-POI table showed 7 flips at idx 1893→1936) | Collapses to one deactivate at the stroke-2 candle (or none if stroke 2 never fires within the lifecycle) |
| **POI Layer 1 (`find_ic_candidates`)** | An IC qualifies if any post-IC sd-direction imbalance has not yet 70%-retraced | An IC qualifies if any post-IC sd-direction imbalance has not yet committed-filled — slightly **more** ICs qualify (some POIs that didn't exist now do) |
| **Fib activation** | Mostly invariant — in a healthy BOS→CTS swing, stroke 1 doesn't fire | Effectively unchanged |
| **Scenario 2 cond3** | Filled if cycle 0's imbalance 70%-retraces by BOS_1 | By the geometric invariant ("BOS_1 must close past gap_top to form, since BOS_1 > BOS_0 > gap_top"), if stroke 1 fires before BOS_1, stroke 2 also fires — **invariant in the dominant case** |
| **Dead-cycle walks (`cross_cycle` mode)** | Cycle marked dead the first candle its imbalance 70%-fills | Cycle stays alive longer (until stroke 2 confirms); cross-fib chains can extend further back |

The geometric invariant for cond3 means most production behavior outside
POI Layer 2 stays put. The exception is `cross_cycle` dead-cycle walks
(M15 subordinate pipelines), where `check_to_idx` is the live candle
rather than a structural event — those genuinely see "cycle stays alive
longer" under the new semantics.

---

## When to revisit

This predicate is one of the load-bearing primitives of the system. Reasons
that would justify a future redesign:

- **A consumer needs a different stroke 2 definition.** E.g., POI's per-
  candle sweep wants close-past-outer but Fib activation prefers
  high/low-past-outer. Current design has one universal predicate; adding
  consumer-specific variants means rebuilding the matrix above with
  per-consumer fill predicates.
- **Live-mode incremental computation** requires per-instance state
  rather than a full-df scan per call. The cache abstraction already
  supports this — `is_filled` would become a thin wrapper over a stateful
  per-instance tracker.
- **A third stroke is needed** (e.g., a re-entry confirmation after
  stroke 2). Today's state machine is monotonic 0→1→2; a more complex
  state machine would need new naming for the additional state.
