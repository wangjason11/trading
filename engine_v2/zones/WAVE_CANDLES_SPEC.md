# Wave Candles Spec — Week 8

Primary file: `zones/wave_candles.py`

---

## Overview

Wave candles are boundary candles between two consecutive waves at a KL zone. For each zone, the system identifies:
- **Last wave candle**: final candle of the ending wave
- **First wave candle**: initial candle of the starting wave

These feed into WVMI (volume momentum) downstream.

---

## Zone Type Matrix

| Zone Type | Last Wave Candle | First Wave Candle |
|-----------|-----------------|-------------------|
| **BOS** | Last Pullback (dir opposite to zone side) | First Breakout (dir same as zone side) |
| **CTS** | Last Breakout (dir opposite to zone side) | First Pullback (dir same as zone side) |

Direction mapping:
- Buy zone (sd=+1): pullback dir = -1, breakout dir = +1
- Sell zone (sd=-1): pullback dir = +1, breakout dir = -1

---

## Qualification Criteria

### Qualified Candle (`_is_qualified`)
Both must be true:
1. `candle direction == required_dir`
2. `vol_dir == required_dir OR vol_dir == 0`

**Rationale:** vol_dir == 0 (neutral volume) is allowed because volume data may be incomplete or neutral during legitimate wave transitions.

### Compound First-Wave (`_is_compound_first_wave`)
Match ONE of:
1. `is_big_normal_as0 == 1 AND candle_type in (maru, normal)` — strong directional candle with size
2. `is_big_maru_as0 == 1 AND candle_type == pinbar AND pinbar_dir == wave_dir` — large candle with pinbar rejection in wave direction

**Rationale:** First wave candles must show conviction — they're the initiating move of the new wave. Weak or indecisive candles don't qualify.

---

## Index fields used below

The searches mix several "anchor" values. They are **different fields** — never interchangeable
(canonical per-event table: `ARCHITECTURE.md` "`ev.idx` convention"):

| Name here | Source | What it is |
|---|---|---|
| `anchor_idx` (BOS sections) | the BOS zone's `meta["anchor_idx"]` = `BOS_CONFIRMED.meta["bos_anchor_idx"]` (Plan E E2c; `.idx` is the moment since E4b) | the BOS **anchor** |
| `cts_anchor_idx` (CTS sections) | the CTS zone's `meta["anchor_idx"]` = `CTS_CONFIRMED.meta["cts_anchor_idx"]` | the CTS **anchor at confirmation** (equals the cycle's `CTS_ESTABLISHED.meta["cts_anchor_idx"]` unless a `CTS_UPDATED` moved it) |
| `pattern_anchor_idx` | the same cycle's `CTS_ESTABLISHED.meta["pattern_anchor_idx"]` — on every `CTS_ESTABLISHED`, read by direct index (a missing key raises `KeyError`) | the breakout pattern's **first candle** (a pattern-realm anchor) — not necessarily the CTS anchor |
| the event candle in the CTS event walk (`ev_idx`) | `ef.cts_anchor_idx(ev)` of `CTS_ESTABLISHED` / `CTS_UPDATED` — `meta["cts_anchor_idx"]` on `CTS_ESTABLISHED` / pattern-path `CTS_UPDATED` (whose `ev.idx` is the moment since Plan E E4a / E4c), `ev.idx` on raw-path `CTS_UPDATED` | a **price location**: the CTS anchor (`CTS_ESTABLISHED`, pattern-path `CTS_UPDATED`) or the processed candle (raw-path `CTS_UPDATED`, `via == "replay_raw"`) |

A bare `confirmed_at` below (the CTS Last Breakout BIB Step 3 bound, and the Step 1(b) gap-scan end
after the last event) is the same cycle's `CTS_CONFIRMED.meta["confirmed_at"]` (== its `idx`, the
confirmation candle). `CTS_ESTABLISHED`'s own `meta["confirmed_at"]` (its apply candle, the
established moment) is always written out in full.

---

## BOS Wave Candles

### Last Pullback (BIB path)

Two-step search:

**Step 1 — Forward search:** `(anchor_idx+1, pattern_anchor_idx)`
- Find qualified candle closest to zone outer bound
- Accept ONLY if closer to outer than the BOS candle's own close distance
- Rationale: candidate must be a better "pullback into zone" than the BOS candle itself

**Step 2 — Backward fallback:** `[pullback_start, anchor_idx]`
- `pullback_start` priority: last STATE_CHANGED to='pullback' → prior CTS anchor → lookback window
- Lookback window: **15 candles** for cycle 0, **50 candles** for cycle 1+
- Pick qualified candle closest to zone outer bound

**Why different lookbacks:** Cycle 0 has limited history (structure just started). Cycle 1+ has more price action and the prior pullback may be significantly earlier.

### Last Pullback (Non-BIB path)

Simple window: `[anchor_idx-5, anchor_idx+5]`
- Qualified candle that touches zone AND closest to outer bound

### First Breakout

Scan forward from `last_pb_idx + 1` to next CTS_CONFIRMED (or end of data).
- First qualified candle matching compound first-wave condition.

---

## CTS Wave Candles

### Last Breakout (BIB path)

Three-step event walk:

**Step 1 — Event walk:** Iterate CTS_ESTABLISHED + CTS_UPDATED events sorted by their CTS anchor `ef.cts_anchor_idx` (the event candle `ev_idx` — see "Index fields used below"; a location walk, not `ev.idx`).
For each event:
- (a) Direct check: event candle is qualified AND wick enters zone AND closes within zone
  - **Pattern scan-back (CTS_ESTABLISHED only):** If the event candle matches, scan from `pattern_anchor_idx` to `ev_idx` (exclusive) for the first qualified candle closing within the zone. If found, return that earlier candle instead of the event candle.
  - **Rationale:** The event candle is the CTS **anchor** — the pattern extreme, the first highest high / lowest low over the breakout pattern's span, so `meta["pattern_anchor_idx"]` (the pattern's first candle) `<= ev_idx <=` the apply candle `CTS_ESTABLISHED.meta["confirmed_at"]` (not the bare `confirmed_at` of Steps 1(b)/3). It is usually the apply candle, but can be an earlier pattern candle; when the first candle itself holds the extreme (`pattern_anchor_idx == ev_idx`) the scan-back is empty. The *first* pattern candle entering the zone better represents the initial breakout moment.
  - If no earlier pattern candle qualifies, return the event candle itself.
- (b) Gap scan: scan between current event idx and next event idx (after the last event: up to `confirmed_at`, exclusive, if the cycle has a `CTS_CONFIRMED`) for qualified candle closing within zone

**Step 2 — Fallback before anchor:** `[cts_anchor_idx - 10, cts_anchor_idx)`
- First qualified candle closing within zone

**Step 3 — Fallback from anchor forward:** `[cts_anchor_idx, confirmed_at]`
- First qualified candle closing within zone

**Rationale for event walk:** CTS zones can shift via CTS_UPDATED events. The event walk checks each snapshot of the zone to find the breakout candle at the right zone position.

### Last Breakout (Non-BIB path)

Window: `[max(cts_anchor_idx-5, pattern_anchor_idx), cts_anchor_idx+5]`
- `pattern_anchor_idx` = `CTS_ESTABLISHED.meta["pattern_anchor_idx"]` (the first candle of the breakout pattern)
- If pattern anchor is after `cts_anchor_idx - 5`, the window shrinks (candles before the pattern are excluded)
- Selection: first qualified candle that **closes within** the CTS zone (not closest to outer)
- **Rationale:** For CTS zones, the first candle entering the zone represents the initial breakout moment. Unlike BOS zones which use closest-to-outer, CTS prioritizes temporal order.

### First Pullback

Scan forward from `last_bo_idx + 1` to `CTS_CONFIRMED + 10 candles` (or end of data).
- First qualified candle matching compound first-wave condition.

---

## Zone Geometry

- **outer threshold**: `zone.meta["outer"]` — the far boundary of the zone
  - Buy zone: outer = bottom (zone sits above outer)
  - Sell zone: outer = top (zone sits below outer)
- **touches zone**: Buy → `low <= zone.top`; Sell → `high >= zone.bottom`
- **closes within zone**: `zone.bottom <= close <= zone.top`
- **close distance to outer**: `abs(close - outer)` — selection metric for "closest to outer"

---

## Edge Cases

| Case | Result |
|------|--------|
| No qualified candle found for last wave | `last_wave_candle_idx = None` |
| Last wave candle found but no first wave candle matches compound condition | `first_wave_candle_idx = None` |
| BIB forward search finds candidate but it's farther from outer than BOS candle | Rejected; falls through to backward fallback |
| Zone has no events (missing CTS_ESTABLISHED) | BIB event walk yields nothing; fallback ±10 window used |
| Candle with vol_dir == 0 | Qualifies (neutral volume is acceptable) |
| Candle with direction == 0 (neutral) | Does NOT qualify for any wave candle role |

---

## Data Flow

```
KL Zones → identify_wave_candles() per zone → WaveCandleResult list
  → df.attrs["wave_candles"]
  → WVMI (volume momentum)
  → Charting (vertical lines + hover overlay)
```

---

## Chart Rendering & Lifecycle (design 2026-05-27)

Wave candles are the **charted primitive**; WVMI is a derived momentum calc on top
(see `WVMI_SPEC.md`). On-chart visibility is gated by the **cycle lifecycle**
(`compute_cycle_lifecycle`, available for every cycle), NOT by WVMI. (The WVMI
*record* lifecycle was removed 2026-05-27 — it had no consumer; lifecycle is a
cycle/wave-candle property. See `WVMI_SPEC.md` "Lifecycle".)

### Line → role mapping (the four lines per cycle) — with cross-cycle LP attribution

Each rendered line maps to a `(role, cycle_offset)` for visibility lookup:

| Drawn line | `WaveCandleResult` field | Role | Cycle attribution |
|---|---|---|---|
| `BOS_N.first` | BOS zone `first_wave_candle_idx` | **FB** First Breakout  | cycle **N** |
| `BOS_N.last`  | BOS zone `last_wave_candle_idx`  | **LP** Last Pullback   | cycle **N − 1** ← cross-cycle |
| `CTS_N.first` | CTS zone `first_wave_candle_idx` | **FP** First Pullback  | cycle **N** |
| `CTS_N.last`  | CTS zone `last_wave_candle_idx`  | **LB** Last Breakout   | cycle **N** |

> **Why `BOS.last` (LP) belongs to the PREVIOUS cycle (decided 2026-05-28):** per
> the Zone Type Matrix, `BOS_N.last_wave_candle` is the *Last Pullback before
> BOS_N began* — i.e., the last candle of the pullback that BOS_N interrupts. That
> pullback is part of **cycle N − 1**'s tail (the cycle ending right when BOS_N
> starts and CTS_N is established). LP_N (the **locked** LP for cycle N) is
> therefore physically located at `BOS_{N+1}.last_wave_candle`, not
> `BOS_N.last_wave_candle`. Practical consequence: LP_N and FB_{N+1} appear at
> the same structural-point idx (CTS_{N+1} established) and lock together — the
> cross-cycle "structural-point pair" visual, but they are independent records
> attributed to different cycles.

> **`BOS_0.last` — Option C (decided 2026-05-28):** when `wc.cycle_id + offset =
> −1` (the pre-structure pullback before a sub's first BOS), render **iff the
> sub's `cyc=0` is non-collapsed**. Rationale: a clean structure start should
> show the pullback leading into BOS_0 (consistent with how the rest of the
> structure renders); a retroactive-Scenario-2 phantom (cyc=0 collapsed via the
> chain-clamp) hides everything including this pre-structure pullback. Chart
> code special-cases the `lookup_cycle < 0` branch: looks up
> `cycle_life[(sid, 0)]` and renders only if cyc=0's `start < end`.

**LP has two STATES — both are real wave candles** (clarification 2026-05-28):

- **Locked** — the wave candle is *set and won't change*. Physically located at
  `BOS_{N+1}.last_wave_candle` (rendered via the wave_candle_results loop with the
  cross-cycle attribution above). Lock happens at the `CTS_{N+1}` ESTABLISHED
  moment (the cycle end), only when the cycle ends via `next_cycle`.
- **Temp** — the wave candle is *not yet locked; it can shift* to a closer-to-outer
  qualified candle as new bars arrive. While temp, this candle still IS the LP and
  IS the input that WVMI momentum uses (`_find_temporary_lp` is the mechanism that
  picks which candle it currently sits on). Physical position: `WVMIRecord.lp_idx`
  for cycles that have a WVMI record (the main: sd-prox-gated; a sub: every cycle with
  a CTS_CONFIRMED, ungated since Plan G 2026-09-30). The "separate WVMI-temp-LP
  rendering pass" this spec planned is NOT implemented — no chart reads `lp_idx`
  (verified 2026-09-30).

A nice consequence falls out from the cross-cycle attribution combined with the
lifecycle rules: a cycle ended without `next_cycle` (reversal / parent-end) has
**no `BOS_{N+1}`** in the same structure, so its locked LP candle physically
doesn't exist — the "LP drops when ended-without-lock" outcome is enforced
**automatically by the data**, not by an extra gate. The temp LP for such a cycle
*was* tracked while alive but, per spec, isn't rendered after the cycle ends
without lock.

### Per-candle lifecycle

Each drawn line has its own `(start_idx, end_idx)` — all attributed to the
**current** cycle. The end is shared (= the cycle's `compute_cycle_lifecycle.end`);
the start differs by role.

| Role | start_idx (clamped to struct/parent floor) | end_idx | Locks on activation? |
|---|---|---|---|
| **FB** | `CTS_n` established **moment** (`CTS_ESTABLISHED.meta["confirmed_at"]`, clamped = `compute_cycle_lifecycle.start`; NOT the CTS anchor (`CTS_ESTABLISHED.idx` until Plan E E4a made that idx the moment) — Plan C 2026-09-20) | cycle end | yes (immediate) |
| **LB** | `CTS_n` CONFIRMED | cycle end | yes (immediate) |
| **FP** | `CTS_n` CONFIRMED | cycle end | yes (immediate) |
| **LP** | `CTS_n` CONFIRMED | cycle end | **only if** cycle ended via `next_cycle` (`CTS_{n+1}` ESTABLISHED); active-temp otherwise |

**Cycle end** = `compute_cycle_lifecycle.end` — the **earliest** of three
candidates (separate machinery; not part of the start-side floor): next-cycle
clamped start (on the CTS-established moment), this-sid reversal, and — subs
only — the `lifecycle_cap` = the unique sub's real-time `end_idx` (Plan C,
2026-09-20; was the parent-cycle/parent-sid end). Absent (open last cycle / open
last structure / open sub) → `end_idx = None`. `end_reason ∈ {"next_cycle",
"reversal", <cap_reason>}` where `cap_reason` is the sub's `end_reason`:
`"reversal"` \| `"same_dir_replacement"` \| `"parent_end"` (the pre-Plan-C
`"lifecycle_end"` is no longer emitted). The LP-lock test below branches only
on `"next_cycle"`, so it is correct for every cap value by construction.

**LP locking is the asymmetric piece.** Only `end_reason == "next_cycle"` locks LP
(that's the same event that activates FB of cycle n+1). Any other end
(reversal / the sub's window cap: `reversal`, `same_dir_replacement`,
`parent_end`) leaves LP active-temp through end_idx without locking →
LP **disappears** at end_idx. FB/LB/FP lock immediately on activation, so they
carry over past `end_idx` regardless of how the cycle ended.

### Visibility rule

A line renders **iff active OR locked**, where collapse (`start ≥ end`, `==`
included) means *never activated* → not shown:

| Role | Shown on a static end-of-data chart iff… |
|---|---|
| **FB** | `start < end` (cycle didn't end before `CTS_n` established) |
| **LB**, **FP** | `start < end` (cycle didn't end before `CTS_n` confirmed) |
| **LP** | `start < end` **AND** ( `end_idx is None` *or* `end_reason == "next_cycle"` ) |

The `==` collapse case is load-bearing: chained clamped cycles share a floor idx,
producing `start == end` exactly; strict `>` would leak a zero-width phantom set.
(Matches the KL/POI/fib collapse rule.)

### Visual outcomes per cycle (static historical chart)

Reading the table: each cell shows which of cycle N's own four roles render.
Cycle N's LP is physically the `BOS_{N+1}.last_wave_candle` line, so it requires
`BOS_{N+1}` to actually exist in the same structure.

| Cycle N ended via… | FB | LB | FP | LP | Notes |
|---|---|---|---|---|---|
| `CTS_{N+1}` ESTABLISHED (normal) | ✓ locked | ✓ locked | ✓ locked | ✓ locked | LP rendered via `BOS_{N+1}.last` |
| Reversal / parent-end **after** `CTS_N` confirmed | ✓ locked | ✓ locked | ✓ locked | — | no `BOS_{N+1}` in same structure → LP candle physically doesn't exist (no extra gate needed) |
| Reversal / parent-end **between** `CTS_N` est and conf | ✓ locked | — | — | — | LB/FP never activated; LP candle doesn't exist |
| Reversal / parent-end **before** `CTS_N` established (retroactive Scenario-2, `reversal_example.png`) | — | — | — | — | collapsed; all never activated |
| Active last cycle (no end yet) | ✓ locked | ✓ locked if `CTS_K` conf fired | ✓ locked if `CTS_K` conf fired | ✓ **temp** if the cycle has a WVMI record (main: WVMI-gated; sub: always, Plan G) and `CTS_K` conf fired | LP = the **temp** wave candle `WVMIRecord.lp_idx` (the planned separate WVMI-temp-LP rendering pass is NOT implemented — not drawn); shifts each bar in live, sits at its end-of-data position in a static chart; no `BOS_{K+1}` exists yet so no locked LP. For a main cycle NOT WVMI-gated, no temp LP candle is computed. |

**Cross-cycle visual at structural points:** at the `CTS_{N+1}` ESTABLISHED
moment two records hit the same idx — **LP_N locks** (rendered via `BOS_{N+1}.last`) and
**FB_{N+1} simultaneously activates + locks** (rendered via `BOS_{N+1}.first`).
They appear together as the "structural-point pair" but are independent records
attributed to *different* cycles (LP→N, FB→N+1). This is why `BOS.last` needs
the `cycle_offset = −1` in the role mapping above. KL zones for collapsed
cycles still render as the hollow rectangle (left as-is).

### Overlap handling — kept aligned with KL zones (2026-05-28)

Wave-candle lines use the **same most-recent-sid overlap filters that KL zones
use** on each chart — `selected_sids` on the H1 main chart, the §16.5
`_owned_here` per-candle owner filter on the M15 sub chart. (Considered "chart
all structures like zones" with overlap filters dropped; user signed off
2026-05-28 keeping them after the per-candle gating diff was validated
apples-to-apples. KL/POI/etc. unchanged.)

### Rendering simplifications

- **Hover:** wave-candle info only — `idx`, the zone line `<BOS|CTS> zone: <owner> cycle=N`
  (the record's own `source_kind` + `cycle_id`; owner `sid=` on H1, `sub_id=` on an M15
  sub), a `Role:` line with the role from the line → role table above (FB / LB / FP /
  LP — the WVMI CSV's `fb_idx` / `lb_idx` / `fp_idx` / `lp_idx`), raw `Volume`. The LP
  line names the cycle it belongs to (`Role: LP (last pullback of cycle N−1)`;
  `BOS_0.last` → `LP (pre-structure pullback)`). One builder for all three hover sites:
  `wave_candles.wave_candle_hover_lines`. (Until 2026-09-30 every site hardcoded
  "BOS zone:", mislabelling the CTS lines — 55 of 110 labels on the reference window.)
  Drop the WVMI momentum block **and** `Weighted vol` (WVMI-weight-derived).
- **Opacity:** uniform across all charts (drop the 3-tier active / recent /
  prior).
- **Line style:** keep **dashed = subordinate**, **solid = main overlay**.
- Full WVMI momentum markers deferred (revisit later).

---

## Output

`WaveCandleResult` (frozen dataclass):
- `structure_id`, `cycle_id`, `source_kind` (BOS/CTS), `zone_side`
- `last_wave_candle_idx`: Optional[int]
- `first_wave_candle_idx`: Optional[int]
- `meta`: dict with `base_pattern` and `anchor_idx` (the zone's `meta["anchor_idx"]` — the BOS / CTS anchor; see "Index fields used below")
