# Workflows: Replay, Debugging, and Development Process

This doc explains how we work on this repo so changes remain safe and explainable.

---

## Replay workflow (canonical)

1. Run `run_replay.py` from the repo root, capturing its log:
   `python -m engine_v2.run_replay > run.log 2>&1` (the fetch gate in step 4
   reads `run.log`). It fetches and replays a dataset and generates:
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
4. **Check the captured log**: the M15 fetch-completeness gate first, then the
   silent-skip grep (see below). Do both before trusting the output or moving
   to `/compare`.

---

## Post-replay log grep (fetch gate + silent skips, BEFORE /compare or chart review)

The engine emits `WARNING` lines and skips work — a trigger that can't resolve,
a reference zone that's unavailable, a degenerate window, a pending probe —
**without raising**. The run exits 0 and the only trace is a log line. A
behavioral change that *accidentally* drops sids/zones looks identical to one
that *correctly* prunes them until you read those lines. After any replay whose
log you captured (`python -m engine_v2.run_replay > run.log 2>&1`), run two
checks in this order.

**1. Fetch-completeness gate (MANDATORY; no diff is trusted until it passes).**
The M15 input is not among the saved CSVs, so an incomplete M15 frame would
fake a CSV delta. `multitf/data_bridge.fetch_lower_tf_data` fails loudly since
2026-09-27 (a failed OANDA chunk request is retried twice — `[data_bridge]
RETRY k/2 …` — then the fetch raises `[data_bridge] ERROR …` and the replay
crashes; before, it printed the error and exited 0 on a truncated frame). The
gate ties the log to the outputs on disk and checks the exact candle count. The
full rationale, the same-run check, the N/A cases and the canonical
`EXPECTED_FETCH` value live in the `/compare` skill §2b; `/commit-save` Step 4b
runs the same snippet:

```bash
RAW=$(ls -t artifacts/debug/*_raw.csv | head -1); \
EXPECTED_FETCH="[data_bridge] Fetched 4228 M15 candles for NZD_USD in 5 chunks"; \
if ! { [ run.log -nt "$RAW" ] && grep -aqF "=== Replay Timing ===" run.log; }; then \
    echo "FETCH GATE: FAIL (stale log or crashed run: not newer than *_raw.csv, or no timing block)"; grep -aF "[data_bridge]" run.log; tail -n 3 run.log; \
elif ! grep -aqF "[data_bridge]" run.log && ! grep -aqF "[multi_tf:dual]" run.log; then \
    echo "FETCH GATE: N/A (no lower timeframe)"; \
elif ! grep -aqF "[data_bridge]" run.log && grep -aqF "[multi_tf:dual] no triggers" run.log; then \
    echo "FETCH GATE: N/A (no H1 triggers - no M15 fetch)"; \
elif grep -aqF "$EXPECTED_FETCH" run.log && ! grep -aqF "[data_bridge] ERROR" run.log; then \
    echo "FETCH GATE: PASS"; \
else \
    echo "FETCH GATE: FAIL"; grep -aF "[data_bridge]" run.log; \
fi
```

**On FAIL, STOP.** Report a data-fetch failure, re-run the replay, and never
interpret that run's diff. N/A is decided by the snippet, only when no M15
fetch ran: no lower timeframe (`config.py` `lower_timeframes=()`), or no H1
trigger (the `[multi_tf:dual] no triggers` line) — that run writes no M15 CSVs.
The expected line depends on the window (reference window
2025-11-15→2026-01-20: 4228 in 5 chunks); re-baseline it in `/compare`,
`/commit-save` and here in one commit whenever `config.py`'s window changes.

**2. Silent-skip grep.** `error` is in the pattern (case-insensitive) as a
backstop:

```bash
grep -iaE "error|warning|skipping|unavailable|degenerate|pending|no sid" run.log
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
2. **Re-render** (`python -m engine_v2.run_replay > run.log 2>&1`, ~45 s of
   chart time), **pass the fetch gate** (Post-replay log grep above; a partial
   M15 fetch would fake a CSV delta), and **prove the CSVs are byte-identical**
   to the baseline save. That comparison is what makes it chart-only:
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
- StructureEvent.idx vs meta["confirmed_at"]: on `CTS_ESTABLISHED` / `BOS_CONFIRMED` / pattern-path `CTS_UPDATED` `idx` IS the moment `confirmed_at` since Plan E E4a / E4b / E4c (the anchor — a price location — is `meta["cts_anchor_idx"]` / `meta["bos_anchor_idx"]` / `meta["cts_anchor_idx"]`; `price` stays the anchor's). Read roles through `structure/event_fields.py` (`ef.event_moment` / `ef.cts_anchor_idx` / `ef.bos_anchor_idx`), never a raw CTS / BOS `idx`. The canonical per-event table is `ARCHITECTURE.md` "`ev.idx` convention".
- Zone meta["confirmed_idx"] rules (BOS vs CTS)

### E) Confirm thresholds
If ranges or zones don't expand:
- check whether the correct threshold-update event is emitted
- ensure that threshold updates are not coming from unrelated sources (e.g., only range sync updates CTS threshold)

---

## Pre-Commit Comparison (REQUIRED)

**Before every commit and merge, run `/compare`** to ensure changes don't unintentionally alter prior logic.

The `/compare` command:
1. Resolves the baseline = the last `/commit-save` folder (`artifacts/commits/LATEST_<branch>`) — it does NOT re-run
   the previous commit
2. Runs a replay on the current code (or reuses this session's replay of the exact same code) and applies the M15
   fetch gate before trusting any diff
3. Compares the 24 CSVs + chart counts (structure events incl. `confirmed_at`, zones, candle patterns, Fib states,
   pool tables)
4. Reports what stayed the same vs what changed
5. Flags changes outside the plan's expected deltas for investigation

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
