# Workflows: Replay, Debugging, and Development Process

This doc explains how we work on this repo so changes remain safe and explainable.

---

## Replay workflow (canonical)

1. Run `run_replay.py` to fetch and replay a dataset and generate:
   - raw CSV export
   - final pipeline CSV export
   - printed summaries (pattern counts, structure levels, zone stats)
2. Attach structure levels + zones to df attrs:
   - `df.attrs["structure_levels"]`
   - `df.attrs["kl_zones"]`
3. Export chart artifacts using export_plotly.

---

## Debug checklist (when something looks wrong)

### A) Trace the full flow first (CRITICAL)

**Never debug by looking at isolated functions or fragments.** One small change can cascade through the entire system:

```
candle classification → patterns → state machine → CTS/BOS → zones → charting
```

When output differs from expectations:
1. Run replay and capture the full event stream
2. Compare events between "before" and "after" states
3. Find the FIRST divergence point (the root cause, not symptoms)
4. Trace backward: what inputs/conditions feed into that divergence?
5. Trace forward: how does that divergence cascade to later stages?

**Example:** A candle changing from `normal` to `maru` can shift a reversal by 38 candles, which shifts all downstream structure timing and zone boundaries.

### B) Confirm pipeline ordering
Zones depend on base features; base features must occur before structure.

### C) Confirm structure_id filtering
If zones "disappear", confirm the chart is selecting the most recent structure_id and the zones carry that meta field.

### D) Confirm timing indices
When something "happens too late/too early", check:
- PatternEvent.apply_idx (end_idx vs confirmation_idx)
- StructureEvent.idx vs meta["confirmed_at"]
- Zone meta["confirmed_idx"] rules (BOS vs CTS)

### E) Confirm thresholds
If ranges or zones don't expand:
- check whether the correct threshold-update event is emitted
- ensure that threshold updates are not coming from unrelated sources (e.g., only range sync updates CTS threshold)

---

## Pre-Commit Comparison (REQUIRED)

**Before every commit and merge, run `/compare`** to ensure changes don't unintentionally alter prior logic.

The `/compare` command:
1. Runs replay on the previous commit
2. Runs replay on current code
3. Compares key metrics (structure events, zones, candle patterns, Fib states)
4. Reports what stayed the same vs what changed
5. Flags unexpected changes for investigation

**Why this matters:**
- Catches regression bugs early
- Detects unintended side effects
- Ensures each iteration maintains consistency with prior work

---

## Branching + PR discipline (hard rules)

- One branch per week from `main` (e.g., `week6-kl-zones`).
- Optional short-lived day/topic branches.
- Merge to `main` only when the week's Definition of Done is met.
- Keep a replay "golden dataset" output to regression-test chart behavior.
- **Run `/compare` before each commit and merge.**

---

## Contribution guidelines

### Naming & docstrings
- Prefer explicit names tied to domain language (CTS/BOS/range/reversal watch).
- Docstrings should carry the canonical semantics (not just "what code does").

### Testing style
- The chart is the primary integration test.
- Add lightweight invariant checks in core engines to catch silent corruption (MarketStructure already does this).
