# Workflows: Replay, Debugging, and Development Process

This doc explains how we work on this repo so changes remain safe and explainable.

---

## Replay workflow (canonical)

1. Run `run_replay.py` to fetch and replay a dataset and generate:
   - raw CSV export
   - final pipeline CSV export
   - printed summaries (pattern counts, structure levels, zone stats)
   - the sub-structure pool tables (`*_M15_{lens}_subs.csv`,
     `*_M15_{lens}_triggers.csv`, `*_M15_unresolved_triggers.csv` —
     `debug/export_sub_tables.py`, written BEFORE the M15 chart loop) and the
     per-lens M15 CSVs (written inside it)
2. Attach structure levels + zones to df attrs:
   - `df.attrs["structure_levels"]`
   - `df.attrs["kl_zones"]`
3. Export chart artifacts using export_plotly.
4. **Grep the captured log for silently-skipped work** (see below) before
   trusting the output or moving to `/compare`.

---

## Post-replay log grep (catch silent skips BEFORE /compare or chart review)

The engine emits `WARNING` lines and skips work — a trigger that can't resolve,
a reference zone that's unavailable, a degenerate window, a pending probe —
**without raising**. The run exits 0 and the only trace is a log line. A
behavioral change that *accidentally* drops sids/zones looks identical to one
that *correctly* prunes them until you read those lines. After any replay whose
log you captured (`python -m engine_v2.run_replay > run.log 2>&1`):

```bash
grep -iaE "warning|skipping|unavailable|degenerate|pending|no sid" run.log
```

Report the count. A non-zero count is not automatically a bug — some skips are
correct (a probe that legitimately finds no structure). But each one must be
**explained, not ignored**. When a `/compare` shows dropped sids/zones, the
matching skip-warning usually names the exact cause — read it before theorizing.

Since the sub-structure pool (Plan C, 2026-09-20) the grep also catches, by
design, `[sweep] UNRESOLVED (skipping) reason=… lens=… parent=(S,C) type=…
trigger_idx=… detail=…` (one per trigger that produced no record: `pending` /
`degenerate_parent_cycle` / `probe_failed` / `geometry_failed` — the same rows
as `*_M15_unresolved_triggers.csv`) and `WARNING [parent_tables] degenerate
parent cycle (S,C): floor=… end=…` (one per parent cycle whose lifecycle floor
≥ its end). Each is a finding to explain against `PART4_REFACTOR_SPEC.md §17.7`
(reference window — measured on the first Plan C replay: 4 unresolved rows, all
`degenerate_parent_cycle`; expected per §17.6: the degenerate cycles are (1,0)
and (1,1)), not a bug by itself. Related
non-grep lines worth reading in the same pass: `[probe_cache] hit|APPROX hit`
and `[probe_cache] REF-ZONE DIFFERS …` (the accepted-approximation tripwire —
a probe skipped because an earlier same-direction, same-input probe already
finalized) and `WARNING [sweep] bos0_inner mismatch` / `same-idx start
collision`.

> **Why this exists:** Session 3 Step 2 (2026-05-31) dropped 5 sub sids because
> every `subsequent_*` trigger hit "sibling-CTS reference zone unavailable" —
> the warnings sat in the log the whole time but only surfaced during chart
> review. A 5-second grep would have caught it immediately. See memory
> `feedback_implement_against_docs.md`.

---

## Verifying a chart-rendering change

A chart-only change has no CSV delta to diff, so "it looks right" is the only
evidence unless you reconstruct what was drawn. The loop that worked for the
2026-09-21/22 chart review (three rule changes, zero regressions):

1. **Predict from the data first.** Derive the expected outcome from the CSVs /
   events / lifecycle tables BEFORE editing (which segments flip, which dots
   change, which counts move). State it to the user as a table — a wrong
   prediction here is cheap; a wrong rule rendered is a whole re-run.
2. **Re-render** (`python -m engine_v2.run_replay`, ~45 s of chart time) and
   **prove the CSVs are byte-identical** to the baseline save — that is what
   makes it chart-only:
   `for f in <save>/*.csv; do cmp -s "$f" "artifacts/debug/$(basename $f)"; done`
3. **Reconstruct the drawing from the saved HTML and re-derive the rule
   independently** — parse `Plotly.newPlot`'s trace list (`chart_census.load_fig`;
   mind the base64 `bdata` trap in GOTCHAS), map x back to candle idx via the
   candlestick `customdata`, then recompute the rule from the reconstructed
   geometry and assert it matches what the trace names/styles say. This catches
   a rule that is right in the abstract but wrong as applied, which counts
   cannot.
4. **Attribute every count move** with `PYTHONPATH=. python
   engine_v2/debug/chart_census.py <baseline.html> <current.html>` before
   re-baselining the numbers in the `/compare` skill. A style-only change
   renames traces, so it shows up as `-1/+1` pairs, and a trace that merges into
   an existing one nets `-1` — predict which, then check.
5. **Re-baseline the counts in `.claude/skills/compare/SKILL.md`** in the same
   commit as the rendering change, with the docs (CHARTING_SPEC + PART4 §16.5 +
   style registry). Never leave the skill's "standard counts" stale.

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
