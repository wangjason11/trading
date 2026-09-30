# CLAUDE.md — Project Context for Claude Code

> Project context for the engine. The **repo-root `CLAUDE.md`** is what loads at startup (it points
> here); this nested file comes in with work under `engine_v2/`. Keep the root file a pointer — the
> content lives here.

## Project Overview

This is an **explainable, visualization-first, event-driven** automated trading engine for Forex. The primary development loop is **backtesting + replay**.

**Core philosophy:** Research engine first, trading bot second. Every decision must be inspectable, replayable, and explainable.

---

## Current Status

**Week 8 In Progress** (branch: `week8-volmom-multitf`)

| Part | Focus | Status |
|------|-------|--------|
| Part 1 | Scenario 3 for start candle identification | Done |
| Part 2 | Volume momentum indicator (WVMI + proximity gate) | Done |
| Part 3 | Multi-timeframe analysis (subordinate structures + overlay) | Done |
| Part 4 | Pipeline / strategy / multi-TF refactor | In progress — see `PART4_REFACTOR_SPEC.md` (§17 authoritative for subs). Done: per-entity dfs (through §13.5.c.iii); the **sub-structure pool** (§17 — `TriggerRecord` + unique sub, lifecycle sweep; Plans A/B/C, 2026-09-21) + two chart-review rounds; the **zones pass** (2026-09-22 → 29): POI activation on the CTS-established moment (Plan D), imbalance c3 knowability (Plan F), `ev.idx` = the moment on every CTS / BOS event + the candle-index Naming Standard (Plan E, Post-E·1–4), and the MS latent-bug close-out (MS never rewinds since 2026-09-29). Next: the WVMI pass (§8 / §17.10, deferred), then the strategy layer |

**Pre-Week 8 fix:** Exception 2 probe relaxed from CTS_CONFIRMED to CTS_ESTABLISHED (`bbb6d32`).

**Session-level status** (the latest commits, the `/compare` baseline save, test count, next priorities) lives in
memory `MEMORY.md` "Next session priorities"; the per-stage records of the zones pass in
`plans/PLAN_E_naming_event_convention.md` §9 and memory `project_zones_timing_audit_20260922.md`.

**Note:** Original syllabus had multi-TF in Week 8. Parts 1 & 2 revisit prior-week topics to strengthen the single-TF foundation before Part 3 layers on multi-TF.

**Note:** On any given week, we may deviate slightly from the original 10-week plan. We may also return to prior week topics for additional debugging and checking how they interact with new elements we are building.

---

## Quick Commands

```bash
# Run replay pipeline (generates charts + CSVs), from the repo root.
# Capture the log: the M15 fetch gate (WORKFLOWS.md / /compare §2b) reads run.log.
python -m engine_v2.run_replay > run.log 2>&1

# Run tests — from the REPO ROOT (from engine_v2/ the OANDA smoke test cannot find oanda.cfg)
pytest

# Output location
artifacts/debug/*.csv    # Raw and final dataframes
artifacts/charts/*.html  # Interactive Plotly charts
```

**Slash Commands:**
- `/commit-save [message]` — Commit, run replay, save outputs to timestamped folder
- `/compare` — Compare current replay against last `/commit-save` to detect regressions

---

## Key Files (Hot Paths)

```
engine_v2/
├── run_replay.py                    # Entry point - run this first
├── config.py                        # Pair/timeframe/date config
├── pipeline/orchestrator.py         # Pipeline ordering (LOCKED)
├── structure/
│   ├── market_structure.py          # CTS/BOS state machine (core; dual CTS confirmation paths)
│   ├── event_fields.py              # Event index ROLES: ef.event_moment / ef.cts_anchor_idx / ef.bos_anchor_idx
│   ├── unified_probe.py             # Start-candle probe (all trigger types + main reversals)
│   ├── structure_engine.py          # Wrapper for orchestrator; wires zone-derivation resolvers
│   └── identify_start.py            # Start candle selection
├── zones/kl_zones_v1.py             # KL Zone derivation from events
├── zones/poi_zones.py               # POI Zone derivation (Fib + IC)
├── zones/fib_tracker.py             # Fibonacci lifecycle management
├── zones/wave_candles.py            # Wave candle identification
├── zones/wvmi.py                    # Wave Volume Momentum Indicator
├── zones/zone_proximity.py          # Zone proximity triggers (alternating sd/opp_sd)
├── patterns/imbalance.py            # Imbalance (FVG) pattern detection
├── patterns/structure_patterns.py   # Breakout pattern detection
├── features/candles_v2.py           # Candle classification
├── multitf/                         # Multi-TF analysis (subordinate M15 structures)
│   ├── sub_structure_pool.py        # Pool: TriggerRecord + unique sub (PART4 §17)
│   └── entity_df_mutation.py        # Sub build + mirror to entity-absolute (the *_META_IDX_KEYS shift lists)
├── charting/
│   ├── export_plotly.py             # Chart generation (H1)
│   ├── export_m15_chart.py          # M15 lens charts
│   └── style_registry.py            # Visual styling
└── debug/                           # CSV export utilities
```

---

## Pipeline Ordering (LOCKED)

```
candle features → structure patterns → imbalance → market structure → KL zones → wave candles → Fib tracking → POI zones → WVMI → charting
```

**Critical:** Base features MUST run BEFORE structure. WVMI MUST run AFTER POI zones (depends on POI zone inner bounds for activation gate). See `LANDMINES.md` for details.

---

## Key Documentation

| File | What's Inside |
|------|---------------|
| `structure/MARKET_STRUCTURE_SPEC.md` | CTS/BOS/Range/Reversal semantics |
| `zones/KL_ZONES_SPEC.md` | Zone construction, thresholds, expansion |
| `zones/POI_ZONES_SPEC.md` | POI zones (Fib + IC) specification |
| `zones/FIB_LIFECYCLE_SPEC.md` / `zones/CROSS_CYCLE_FIB_SPEC.md` | Fib lifecycle; the cross-cycle Fib mode |
| `zones/WAVE_CANDLES_SPEC.md` | Wave candle identification algorithm |
| `zones/WVMI_SPEC.md` | Wave Volume Momentum Indicator lifecycle + formulas |
| `IMBALANCE_FILL_SEMANTICS.md` | Imbalance (FVG) knowability (the c3 rule) + two-stroke fill |
| `charting/CHARTING_SPEC.md` | Chart overlay rules, style registry |
| `ARCHITECTURE.md` | System design, event contracts, the `ev.idx` convention table |
| `PART4_REFACTOR_SPEC.md` | Multi-TF refactor; §17 = the sub-structure pool (authoritative for subs) |
| `../PROJECT_PRINCIPLES.md` (repo root) | Non-negotiable guardrails |
| `WORKFLOWS.md` | Debugging checklist |
| `GOTCHAS.md` | Debugging lessons learned |
| `LANDMINES.md` | Critical constraints, things to avoid (incl. "Event Contract Rules") |
| `GLOSSARY.md` | Domain terminology; the candle-index "Naming Standard" |
---

## Guardrails (Summary)

Full details in `PROJECT_PRINCIPLES.md` (repo root). Key points:

1. **Research engine first** — every decision traceable to events
2. **Interfaces frozen** — contracts stable, internals can evolve
3. **Event-driven** — state transitions, not bulk transforms
4. **Visibility > performance** — slow but explainable wins
5. **Chart is the debugger** — if it can't be verified visually, it isn't verified
6. **No premature optimization** — no Optuna until logic is trusted

---

## Debug Checklist

When something looks wrong:

1. **Trace the full flow first** — Never debug isolated functions. One change cascades through:
   `candle classification → patterns → state machine → CTS/BOS → zones → charting`
2. **Understand before fixing** — Find *why* it's wrong, not just *what* to change
3. **Pipeline ordering** — base features before structure?
4. **structure_id filtering** — chart shows most recent structure_id only
5. **Timing indices** — check apply_idx, confirmed_at, confirmed_idx
6. **Thresholds** — zone expansion only from threshold-update events
7. **Config** — correct pair/timeframe/dates in config.py?

See `GOTCHAS.md` for detailed debugging lessons (including cascading effect examples).

---

## Documentation cadence (standing rule, 2026-09-22)

Capture learnings **continuously through the session**, not in one sweep at the end:
write the nuance down when it is said (verbatim + why it matters), then **file and
reconcile at every checkpoint** — after each landed change, at each `/compare`
pause, and at `/commit-save`. Reconciling is the new risk that frequent writes
create: edit existing entries **in place** rather than appending contradicting ones,
keep one canonical home per fact, and move code + spec + skill values + memory in the
same commit. Unresolved items go one line into `memory/_INBOX.md` (durable; drained
at the next checkpoint). Full rule + trigger list + routing table:
`memory/feedback_continuous_documentation.md`; procedure: the `/remember` skill's
"Continuous mode".

## Development Workflow

```
1. Run run_replay.py → generate baseline chart
2. Make changes
3. Run pytest (if tests exist)
4. Run run_replay.py → compare to baseline
5. Run /compare → verify no unintended changes to prior logic
6. Commit when behavior matches expectations
```

**IMPORTANT:** Run `/compare` before every commit and merge to catch regressions and unintended side effects.

**Branching:** One branch per week (e.g., `week6-kl-zones`). Merge to `main` when Definition of Done is met.
