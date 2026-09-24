---
name: compare
description: Compare current replay output against previous /commit-save to detect unintended changes.
user-invocable: true
allowed-tools: Bash, Read, Glob, Grep, Write
argument-hint: "[--reuse-replay]"
---

# Compare Replay Output Against Previous Commit-Save

Compare the current code's replay output against the most recent `/commit-save` output to ensure changes don't unintentionally alter prior logic.

## Baseline resolution (branch-aware, since 2026-05-23)

Saves are stored under `artifacts/commits/<branch>/<timestamp>_<hash>/`
and tracked by per-branch `LATEST_<branch>` pointers. Legacy flat saves
(`artifacts/commits/<ts>_<hash>/` with the global `LATEST` file) are
still supported as a fallback.

Resolve the previous save folder by:
1. Read `artifacts/commits/LATEST_<current-branch>` → look up
   `artifacts/commits/<current-branch>/<folder>/` (new layout)
2. If missing, fall back to `artifacts/commits/LATEST` →
   `artifacts/commits/<folder>/` (legacy flat layout)
3. If both missing, error: no baseline to compare against

The `artifacts-trunk` branch carries the union of ALL saves cherry-picked
from every branch. If you need to compare against a save from a different
branch, either checkout `artifacts-trunk` or use
`git show artifacts-trunk:artifacts/commits/<branch>/<folder>/<file>` to
read the historical contents.

## Instructions

### 1. Find Previous Commit-Save Output

```bash
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
PER_BRANCH_LATEST="artifacts/commits/LATEST_${CURRENT_BRANCH}"
LEGACY_LATEST="artifacts/commits/LATEST"

if [ -f "${PER_BRANCH_LATEST}" ]; then
    PREV_FOLDER=$(cat "${PER_BRANCH_LATEST}")
    PREV_PATH="artifacts/commits/${CURRENT_BRANCH}/${PREV_FOLDER}"
    echo "Baseline: per-branch (${CURRENT_BRANCH}) -> ${PREV_FOLDER}"
elif [ -f "${LEGACY_LATEST}" ]; then
    PREV_FOLDER=$(cat "${LEGACY_LATEST}")
    PREV_PATH="artifacts/commits/${PREV_FOLDER}"
    echo "Baseline: legacy flat layout -> ${PREV_FOLDER}"
else
    echo "ERROR: No previous commit-save found. Run /commit-save first."
    exit 1
fi

if [ ! -d "${PREV_PATH}" ]; then
    echo "ERROR: Baseline pointer references missing folder: ${PREV_PATH}"
    exit 1
fi
```

### 2. Get Current Replay Output (run vs reuse)

`/compare` needs the current code's replay output in `artifacts/debug` +
`artifacts/charts`. Two modes for obtaining it — mirrors `/commit-save`:

- **Run mode (default — plain `/compare`):** run a fresh replay:
  ```bash
  python -m engine_v2.run_replay > run.log 2>&1
  ```
- **Reuse mode (`/compare --reuse-replay`, or the user says "use the recent
  replay" / "don't re-run the replay"):** SKIP the replay and compare the
  outputs already sitting in `artifacts/debug` + `artifacts/charts` from the
  session's most recent replay.

**Best-judgement reuse (no flag needed).** If a replay for the EXACT current
code already ran earlier this session (it completed AND no source changed
since — a re-run would reproduce identical output), you MAY reuse it without
the flag and SAY you're doing so ("reusing the replay from N minutes ago — no
code changed since"). When in doubt, or if any source changed after the last
replay, run a fresh one. Reusing stale outputs silently compares the wrong
data — the same guard as `/commit-save` reuse mode.

**Always display the `=== Replay Timing ===` block** when a replay is run
(per `feedback_replay_timing_display`). In reuse mode there's no new timing.

### 2b. Post-replay log grep (fetch gate FIRST, then the silent-skip grep)

**Fetch-completeness gate: MANDATORY, and it runs before any diff is
trusted.** `multitf/data_bridge.fetch_lower_tf_data` fetches the M15 input in
14-day OANDA chunks and does **not** raise when a chunk fails. It prints
`[data_bridge] ERROR fetching M15 chunk <from>-<to>: <exception>`, carries on
with the chunks it got, and still prints `[data_bridge] Fetched N M15 candles
for NZD_USD in K chunks`, where K counts only the chunks that returned data.
The replay exits 0 on the truncated frame. The M15 input is not among the 24
saved CSVs and saves carry no `run.log`, so a partial fetch shows up only as
M15 deltas that look exactly like an engine change. This happened in a
2026-09-22 audit run: two HTTP 504s left 2500 of 4228 M15 candles, and nothing
failed. The H1 fetch does not have this problem: `provider_oanda.get_history`
raises on a non-200 response and `run_replay.py` does not catch it, so a failed
H1 fetch crashes the run.

```bash
RAW=$(ls -t artifacts/debug/*_raw.csv | head -1); \
EXPECTED_FETCH="[data_bridge] Fetched 4228 M15 candles for NZD_USD in 5 chunks"; \
if [ run.log -nt "$RAW" ] && grep -aqF "=== Replay Timing ===" run.log \
   && grep -aqF "$EXPECTED_FETCH" run.log && ! grep -aqF "[data_bridge] ERROR" run.log; then \
    echo "FETCH GATE: PASS"; \
else \
    echo "FETCH GATE: FAIL"; grep -aF "[data_bridge]" run.log; \
fi
```

- **PASS needs all four conditions.** The first two are the **same-run check**:
  `run.log` is newer than the newest `*_raw.csv` AND contains the `=== Replay
  Timing ===` block. `run_replay.py` writes `*_raw.csv` first (before
  `run_pipeline`) and prints the timing block last, so the log of the replay
  whose outputs are on disk passes both. A stale `run.log` from an earlier
  replay fails the first (a later replay run without `> run.log 2>&1` wrote a
  newer `*_raw.csv`), and a crashed run fails the second. The last two are the
  fetch check: the exact expected `Fetched` line is present AND there is no
  `[data_bridge] ERROR` line. `/commit-save` Step 4b runs this same snippet.
- **On FAIL, STOP.** Report a **data-fetch failure** and quote the
  `[data_bridge]` lines. If those lines look complete, the same-run check is
  what failed: the log does not belong to the outputs on disk. Either way,
  re-run the replay (step 2, run mode) and check again. **Never interpret the
  diff of a run that failed the gate.** Nothing it shows is evidence about the
  code.
- **Reuse mode:** the gate still applies, to the captured log of the replay
  being reused, and the same-run check is what stops reuse from passing on a
  stale log. If that replay was run without `> run.log 2>&1`, completeness
  cannot be checked, so run a fresh replay instead of comparing.
- **N/A only when the run has no lower timeframe** (`config.py`
  `lower_timeframes=()` — then no M15 CSVs exist either).
- **The expected line depends on the window.** The M15 fetch spans the H1 frame
  (`orchestrator._run_multi_tf_dual`, first to last H1 candle), which is
  `config.py`'s window after `[auto_extend]` moves its start. On the reference
  window, `config.py` says 2025-12-01, auto-extend moves it to 2025-11-15, and
  the fetch covers 2025-11-15→2026-01-20: **4228 candles in 5 chunks**. That
  value is measured, not computed: the 2026-09-22 `run.log` and
  `artifacts/replay_{msfix,session2,subrev,subrev2,zonefix}.log` all show it,
  while older logs from a shorter window (`artifacts/phase2_*.log`, May 2026)
  show `3364 … in 4 chunks`. **Re-baseline `EXPECTED_FETCH` whenever
  `config.py`'s window changes**, using a replay whose log has no
  `[data_bridge] ERROR` line. This skill holds the canonical value, and
  `/commit-save` Step 4b and `engine_v2/WORKFLOWS.md` repeat it, so update all
  three in the same commit.

**Silent-skip grep.** Once the gate passes, grep the same log for
silently-skipped work BEFORE the CSV comparison. `error` is in the pattern
(case-insensitive) as a backstop, so a fetch error or any other error line
cannot be missed:

```bash
grep -iaE "error|warning|skipping|unavailable|degenerate|pending|no sid" run.log
```

Report the count + the lines. A non-zero count is not automatically a bug
(some skips are correct), but each must be **explained, not ignored** — and
when the CSV/chart comparison below shows dropped sids/zones, the matching
skip-warning usually names the exact cause. In reuse mode, grep the captured
log of the replay being reused (the fetch gate above already requires it). See
`engine_v2/WORKFLOWS.md` "Post-replay log grep" + memory
`feedback_implement_against_docs.md` for why this exists (Session 3 Step 2
dropped 5 sids whose cause sat unread in the log).

Since the sub-structure pool (Plan C, 2026-09-20) this grep also catches, by
design, two line families — each is a **finding to explain**, not a bug by
itself:

- `[sweep] UNRESOLVED (skipping) reason=… lens=… parent=(S,C) type=…
  trigger_idx=… detail=…` — one per trigger that produced no record (`pending`
  / `degenerate_parent_cycle` / `probe_failed` / `geometry_failed`); the same
  rows land in `*_M15_unresolved_triggers.csv`. Compare the set against the
  plan's expectation (reference window, measured on the first Plan C replay:
  4 rows, all `degenerate_parent_cycle`).
- `WARNING [parent_tables] degenerate parent cycle (S,C): floor=… end=…` —
  one per parent cycle whose lifecycle floor ≥ its end (`PART4 §17.7`;
  expected on the reference window: (1,0) and (1,1)).

Also read (not caught by the grep) `[probe_cache] hit|APPROX hit …` and
`[probe_cache] REF-ZONE DIFFERS …` — a probe skipped because an earlier
same-direction, same-input probe already finalized (the §17.8 accepted
approximation's tripwire). Any lifecycle delta must be traceable to one of
these when the cache bit.

### 3. Load and Compare Data

Use Python to load both datasets and perform detailed comparison:

```python
import pandas as pd

# Load previous
prev_final = pd.read_csv(f"{prev_path}/NZD_USD_H1_..._final.csv")

# Load current
curr_final = pd.read_csv("artifacts/debug/NZD_USD_H1_..._final.csv")

# Compare
```

### 4. Comparison Categories

#### A) Row-Level Changes (Same Index, Different Values)

For each key column, identify rows where values differ:

| Column Category | Columns to Compare |
|-----------------|-------------------|
| **Candle Classification** | `candle_type`, `is_special_maru`, `pinbar_dir`, `direction`, `body_pct` |
| **Pattern Detection** | `pat` (pattern name at each row) |
| **Market Structure** | `market_state`, `structure_id`, `is_cts`, `is_bos` |
| **Zones** | `in_kl_zone`, `in_poi_zone` (if present) |
| **Imbalance** | `is_imbalance`, `imbalance_fill_pct` (if present) |

Output format:
```
ROW-LEVEL CHANGES:

candle_type (5 rows changed):
  idx  | before  | after
  707  | normal  | maru
  716  | normal  | maru
  ...

market_state (12 rows changed):
  idx  | before       | after
  710  | breakout     | pullback
  711  | range        | pullback_range
  ...
```

#### B) Event-Level Shifts (Events Moved to Different Indices)

Compare structural events between iterations:

| Event Type | What to Track |
|------------|---------------|
| **BOS_CONFIRMED** | `(idx, confirmed_at, bos_anchor_idx, structure_id, cycle_id)` |
| **CTS_CONFIRMED** | `(idx, cts_anchor_idx, structure_id, cycle_id)` |
| **CTS_ESTABLISHED** | `(idx, confirmed_at, cts_anchor_idx, structure_id, cycle_id)` |
| **STATE_CHANGED to reversal** | `(idx, structure_id)` |

On `BOS_CONFIRMED` / `CTS_ESTABLISHED`, `idx` is the price **extreme** (the
anchor), stamped after the fact; since Plan E E2a (2026-09-24) the anchor is
also in meta `bos_anchor_idx` / `cts_anchor_idx` (`== idx` until Plan E E4 flips
`idx` to the moment — from then on track the meta anchor for location shifts). The moment the event became known is `meta["confirmed_at"]`,
stored inside the `meta` column of `*_structure_events.csv`, and it is the
value that timing and lifecycle code reads. A shift in `confirmed_at` alone
leaves `idx` unchanged, so compare both. On `CTS_CONFIRMED`, `idx` ==
`confirmed_at`. The canonical per-event table is `engine_v2/ARCHITECTURE.md`
"`ev.idx` convention".

Detect:
- **Removed events**: In previous but not current
- **Added events**: In current but not previous
- **Shifted events**: Same `(structure_id, cycle_id)` (and `sub_id` on M15 —
  every sub's events carry `structure_id` 0) but different `idx` OR different
  `confirmed_at`

Output format:
```
EVENT-LEVEL SHIFTS:

BOS_CONFIRMED:
  SHIFTED: sid=1 cycle=1 moved from idx=728 to idx=826 (+98 candles)
  SHIFTED (moment only): M15.counter sub_id=3 cycle=1 idx=2758 unchanged, confirmed_at moved from 2829 to 2830 (+1 candle)

REVERSAL:
  SHIFTED: sid=0 moved from idx=710 to idx=748 (+38 candles)

CTS_CONFIRMED:
  UNCHANGED: All 5 events match
```

#### C) Aggregate Metrics

Compare high-level counts:

```
AGGREGATE METRICS:

                        Previous  Current  Change
Total candles           1058      1058     -
Maru candles            78        85       +7
Pinbar candles          120       115      -5
Structure events        290       295      +5
KL zones                10        10       -
POI zones               4         3        -1
Fib states              3         3        -
Imbalance candles       218       218      -
```

#### D) Zone Boundary Changes

For KL zones and POI zones, compare:
- Zone count per structure/cycle
- Zone boundaries (start_idx, end_idx, top, bottom)
- Zone status (active/inactive)

```
ZONE CHANGES:

KL Zones:
  sid=1 cycle=1 BOS zone: SHIFTED from idx=728 to idx=826

POI Zones:
  sid=1 cycle=1: CHANGED ic_idx from 832 to 732
```

### 5. Summary and Recommendation

```
=== COMPARE SUMMARY ===

Baseline: <per-branch | legacy> -> <folder> (commit <hash>)
Current branch: <branch>
Current code: <uncommitted changes> OR <current commit>

UNCHANGED:
- Total candles: 1058
- Imbalance count: 218
- KL zone structure: Matches

CHANGED (with analysis):
- candle_type: 7 rows changed (special_maru direction fix applied)
- Reversal: Shifted from idx=748 to idx=710 (EXPECTED: continuous pattern fix)
- BOS sid=1 cycle=1: Shifted from idx=826 to idx=728 (EXPECTED: downstream of reversal fix)

UNEXPECTED CHANGES:
- [None found]

RECOMMENDATION: Safe to commit
```

OR if unexpected changes:

```
UNEXPECTED CHANGES:
- POI zones decreased from 4 to 2 (POI logic not touched in this iteration)

RECOMMENDATION: Investigate before commit
  - Check poi_zones.py for unintended changes
  - Verify Fib state handling
```

### 6. Cascading Change Detection

When a change is detected, trace its downstream effects:

```
CASCADE ANALYSIS:

Root cause: candle_type changed at idx=707 (normal -> maru)
  v
Effect 1: continuous pattern at 707-709 no longer matches Pattern3
  v
Effect 2: Reversal watch at idx=707 fails (no valid pattern)
  v
Effect 3: Reversal shifts from idx=710 to idx=748
  v
Effect 4: sid=1 starts later, all cycle timing shifts
  v
Effect 5: BOS for sid=1 cycle=1 moves from idx=728 to idx=826

All changes are causally linked to the root change.
```

## Key Files to Compare

Per-entity CSV outputs cover **H1.main**, **M15.counter**, and
**M15.confluence**. Compare every file md5 by entity:

**H1.main** (9 files):

| File pattern |
|--|
| `*_final.csv` |
| `*_raw.csv` |
| `*_structure_levels.csv` |
| `*_kl_zones.csv` |
| `*_poi_zones.csv` |
| `*_structure_events.csv` |
| `*_imbalance_instances.csv` |
| `*_fib_lifecycle.csv` |
| `*_wvmi.csv` |

**M15.counter** and **M15.confluence** (7 files each — 5 per-lens debug CSVs
with the same filename suffixes prefixed `*_M15_counter_` /
`*_M15_confluence_`, plus the two per-lens pool tables):

| File pattern | Written by |
|--|--|
| `*_M15_{lens}_structure_events.csv` | chart loop (`run_replay.py`) |
| `*_M15_{lens}_kl_zones.csv` | chart loop |
| `*_M15_{lens}_poi_zones.csv` | chart loop |
| `*_M15_{lens}_fib_lifecycle.csv` | chart loop |
| `*_M15_{lens}_wvmi.csv` | chart loop |
| `*_M15_{lens}_subs.csv` — one row per unique sub on this lens | `debug/export_sub_tables.py`, BEFORE the chart loop |
| `*_M15_{lens}_triggers.csv` — one row per `TriggerRecord` on this lens (incl. zero-length) | `debug/export_sub_tables.py`, BEFORE the chart loop |

**Pool-wide** (1 file): `*_M15_unresolved_triggers.csv` — one row per
`UnresolvedTrigger` (`debug/export_sub_tables.py`, written once).

Total: **9 + 2×7 + 1 = 24 CSVs** per replay (POI-zones CSVs added 2026-06-08;
pool tables added by Plan C 2026-09-20). `*_M15_{lens}_sids.csv` is **GONE**
since Plan C — its role is split across `_subs.csv` / `_triggers.csv`. **Delete
stale `*_sids.csv` copies from `artifacts/debug/` before comparing**: the replay
no longer writes them, so a leftover from a pre-Plan-C run is never overwritten
and would be compared as if current. Current path is `artifacts/debug/`; baseline
path is whichever step 1 resolved to: `artifacts/commits/<branch>/<folder>/`
(new layout) or `artifacts/commits/<folder>/` (legacy flat layout).

**Files present on only one side are a finding, not a skip.** Diff the two
file *lists* first and report every `NEW (no baseline)` and `MISSING (in
baseline, not produced)` file explicitly. The pool/lifecycle redesign (Plan
C, 2026-09-20) replaced the per-lens `*_sids.csv` with `*_subs.csv` +
`*_triggers.csv` + `*_unresolved_triggers.csv`; on its first run those had
no baseline and the old file disappeared — both were called out, and the
new tables are validated against the plan's **predicted table**
(`memory/reference_pool_redesign_groundtruth.md`, matched by
`(direction, starting_idx)` — `sub_id`s are 0–7 in creation order on the
reference window) rather than a prior save; from the next `/commit-save` on
they have a baseline like any other CSV. The three pool tables are written
**before** the chart loop, decoupled from it (`export_sub_tables` in its own
try/except — a chart-export exception cannot lose them), so a MISSING pool
table means the sweep/export itself failed; the other per-lens CSVs are still
written **inside** the chart loop, so a missing one of those can mean the
chart export it is coupled to crashed — treat MISSING as a possible crash
signal either way.

Column vocabulary to expect since Plan C: `sub_id` replaces `sub_sid` on every
M15 structural artifact (and on the shared `export_wvmi` column — the H1
`_wvmi.csv` header changes with it); `end_reason ∈ {reversal,
same_dir_replacement, parent_end}` or empty (no `lifecycle_end`; `next_cycle`
only on cycle-owned zone/fib rows).

**Classify deltas against the plan's stated expectations.** When the change
being compared has a plan that lists its expected deltas (per
`memory/feedback_one_cause_per_compare.md`), the summary's EXPECTED /
UNEXPECTED split is *against that list* — an "expected byte-identical" plan
with any non-empty diff is a bug signal, not a review task.

## File learnings before moving on (checkpoint)

A finished `/compare` (and the chart-review pause that follows it) is a
**checkpoint** for the continuous-documentation rule
(`memory/feedback_continuous_documentation.md`): before starting the next change,
drain `memory/_INBOX.md`, file what this round established — the decision AND its
rationale, any scope boundary the user drew, any measured fact a future session
would re-derive — and run the reconcile checklist (edit in place, one canonical
home, `MEMORY.md` true end to end). Do not defer it to `/commit-save`.

## Figure-JSON diff (required for Plan E stages; recommended whenever charts should be identical)

Count parity cannot see a marker that MOVES (a BOS dot shifting 2–411 candles
keeps every count). Compare the figures themselves: load both `.html` files with
`engine_v2/debug/chart_census.load_fig` (it returns `(data, layout)`; import it
with `sys.argv` reset — the module runs a census on import) and compare, per
trace, `(name, x, y)`, plus `layout["shapes"]`. Report "traces_xy_equal /
shapes_equal" per chart; on a difference list the first differing traces
(name + x before → after). Plan E E2 stages must be figure-identical; the E4
variant replays must differ exactly in PLAN_E §8's cells.

## Chart Count Parity (Corroborating check)

Chart trace/shape counts tally everything rendered for an entity, so a
mismatch confirms divergence even when you can't immediately tell which
CSV moved. They are no longer the *only* sub-entity signal (the
per-entity CSVs above are primary), but they're cheap to read from
stdout and catch chart-side regressions (style registry, hover,
lifecycle gating) that the CSVs alone miss.

After the replay run, verify the three trace/shape counts in the stdout
match the baseline run's counts exactly. Per the latest run the standard
counts are (H1 and confluence set by the 2026-09-22 chart review round 2,
`aadb887`; counter by Plan F, 2026-09-24; all three last verified 2026-09-24 at
the Plan F `/compare`; config window 2025-12-01→2026-01-20, auto-extended to
2025-11-15. Window-dependent, re-baseline when the config window or chart
rendering changes; older values in **History** below):

- `H1` chart: traces=85, shapes=245
- `M15.counter` chart: traces=151, shapes=124
- `M15.confluence` chart: traces=294, shapes=233

**History:** on 2026-05-29 the counts were H1 125/261, counter 216/169, confluence
359/280. Pre-Plan C (through the Plan B save 2026-09-20) they were H1
107/250, counter 198/133, confluence 342/222. Plan C's first replay moved the
M15 counts (174/128, 321/238: 8 unique subs instead of 16 per-trigger sids,
ownership by lifecycle window, degenerate subs absent); the chart review then
added the forming layer (+ traces), hid collapsed-cycle zones on BOTH charts
(H1 107→85: sid 1's retroactive (1,0)/(1,1) zones + POI gone; M15 −22/−20) and
tinted POIs — the Plan C save (`20260921_125218_afaa326`) = 85/245, 157/125,
301/233. The 2026-09-21 chart review's wave rule + H1-overlay lifecycle filter
(PART4 §16.5 items 4–5) then moved the M15 traces 157→154 / 301→295, and the
2026-09-22 recent-vs-prior rule (§16.5 item 6) to 152 / 293 — all CSVs
byte-identical throughout. Item 6's deltas: forming swing traces replaced by
prior ones (one per overlap region: confluence 7→6, counter 3→1) and the
forming dot traces merged into their subs' live dot traces (only sub `2639/−1`'s
3611 CTS dot is still prior, on confluence). The replacement-break rule (§16.5
item 7) then added one PB-dot trace per lens → 153 / 294. Plan D (2026-09-23, POI activation on the moment) left every count unchanged: its only
figure delta was 2 shapes moved + 2 hover traces' customdata on the confluence chart.
Plan F (2026-09-24, imbalance c3 knowability) moved counter 153/125 → 151/124
(`chart_census`: −2 `POI_hover` traces of sub 5 and −1 outline rect). That is the
removed POI twin of IC 3654, attached to a cross fib FibTracker had pre-created
for a cycle 1 that sub 5 never establishes. M15 fibs are not drawn, so the dropped
fib shows only in `_fib_lifecycle.csv`. H1 and confluence counts were unchanged.

These print as `DEBUG traces:` / `DEBUG shapes:` (H1) and `[m15_chart] traces:
N, shapes: M` (each M15 entity) at the end of `python -m engine_v2.run_replay`.

A run with all 24 CSVs byte-identical but shifted chart counts means a
purely rendering-side change (e.g. style registry tweak). A run with
matching chart counts but mismatched CSVs means a logic change. Both
matrices clean = full parity.

When an M15 count moves and the cause is not obvious, run
`PYTHONPATH=. python engine_v2/debug/chart_census.py <baseline.html> <current.html>`
— it prints a per-element / per-sub trace census of the two saved charts side
by side, so the shifted count is attributed to a specific element class and
sub before any CSV digging.

## Per-Cycle Proximity Trigger Counts (Required when proximity logic changed)

CSV-level + event-level comparison can MASK cycle-specific bypass bugs.
A per-cycle rule (e.g., narrow-cycle Rules 1/2/3) can fire correctly in
one cycle and silently bypass in another while keeping aggregate event
counts unchanged — `subsequent_confluence` / `subsequent_counter` counts
shift slightly, but the relevant `BOS_CONFIRMED` / `CTS_CONFIRMED`
counts stay identical.

After any change to:
- `zones/zone_proximity.py` (Rules 2/3 scan logic, alternation, caps)
- `structure/market_structure.py` per-candle proximity-confirmation gate (Rule 1)
- Per-TF threshold tables (`DEFAULT_PROXIMITY_PIPS`, `DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS`)

ALSO run the proximity diag tool and check per-cycle trigger counts:

```bash
python -m engine_v2.debug.zone_proximity_diag
```

Output ends with `=== Per-cycle counts ===` showing `sid=N cycle=M ...
alt_list=X var3=Y var4=Z BOS-CTS gap=Wp` per cycle. Compare each cycle's
`alt_list` count against expectations:
- Narrow cycle (gap < min_gap_pips): MUST be <= 2 (Rule 3 cap: <=1 sd + <=1 opp_sd).
- Wide cycle: unbounded; depends on V/Lambda pattern length.

If a narrow cycle shows `alt_list > 2`, Rule 3 is being bypassed — investigate before commit. A specific concrete failure mode that has happened: df-col-based gap source went NaN when scan window crossed a reversal into the next sid's rows, causing Rule 3 to silently default-mode bypass for the cycle-tail (see LANDMINES "Narrow-Cycle Rules 1+2+3 Are a Triple").

## Cross-Branch Compare (advanced)

To compare against a save made on a different branch — e.g., currently
on `week8-volmom-multitf` but wanting to compare against a
`sub-debug-c2-baseline` save — the per-branch LATEST_<other-branch>
file is the entry point:

```bash
OTHER_BRANCH="sub-debug-c2-baseline"
OTHER_LATEST="artifacts/commits/LATEST_${OTHER_BRANCH}"
if [ -f "${OTHER_LATEST}" ]; then
    OTHER_FOLDER=$(cat "${OTHER_LATEST}")
    OTHER_PATH="artifacts/commits/${OTHER_BRANCH}/${OTHER_FOLDER}"
fi
```

If `${OTHER_PATH}` isn't present in the current branch's working tree
(because that save was committed only on a different branch and never
cherry-picked back), you can either:
- Read individual files via `git show artifacts-trunk:${OTHER_PATH}/<file>`
- Or `git checkout artifacts-trunk -- ${OTHER_PATH}/` to materialize them
  locally (then `git restore --staged` afterward to keep them out of any
  pending commit)

## Why This Matters

This comparison catches:
- **Regression bugs**: Prior logic broken by new changes
- **Cascading effects**: One small change affecting downstream structures
- **Shifted events**: BOS/CTS/reversal moving to different indices
- **Missing coverage**: Changes in areas not touched by recent work

**Run `/compare` before every commit** to maintain code integrity and catch issues early.

## Workflow

```
1. /commit-save          # Checkpoint current working state
2. Make changes          # Implement new feature/fix
3. /compare              # Verify changes are as expected
4. If unexpected -> investigate
5. If expected -> /commit-save  # Create new checkpoint
```
