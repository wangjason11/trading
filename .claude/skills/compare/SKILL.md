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

### 2b. Post-replay log grep (catch silent skips)

Whenever a replay was actually run in step 2 (run mode), grep its captured log
for silently-skipped work BEFORE the CSV comparison:

```bash
grep -iaE "warning|skipping|unavailable|degenerate|pending|no sid" run.log
```

Report the count + the lines. A non-zero count is not automatically a bug
(some skips are correct), but each must be **explained, not ignored** — and
when the CSV/chart comparison below shows dropped sids/zones, the matching
skip-warning usually names the exact cause. In reuse mode, grep the most recent
replay's log if it was captured; otherwise note it was unavailable. See
`engine_v2/WORKFLOWS.md` "Post-replay log grep" + memory
`feedback_implement_against_docs.md` for why this exists (Session 3 Step 2
dropped 5 sids whose cause sat unread in the log).

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
| **BOS_CONFIRMED** | `(idx, structure_id, cycle_id)` |
| **CTS_CONFIRMED** | `(idx, structure_id, cycle_id)` |
| **CTS_ESTABLISHED** | `(idx, structure_id, cycle_id)` |
| **STATE_CHANGED to reversal** | `(idx, structure_id)` |

Detect:
- **Removed events**: In previous but not current
- **Added events**: In current but not previous
- **Shifted events**: Same `(structure_id, cycle_id)` but different `idx`

Output format:
```
EVENT-LEVEL SHIFTS:

BOS_CONFIRMED:
  SHIFTED: sid=1 cycle=1 moved from idx=728 to idx=826 (+98 candles)

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

**M15.counter** and **M15.confluence** (6 files each — same filename
suffixes prefixed with `*_M15_counter_` / `*_M15_confluence_`):

| File pattern |
|--|
| `*_M15_{entity}_structure_events.csv` |
| `*_M15_{entity}_kl_zones.csv` |
| `*_M15_{entity}_poi_zones.csv` |
| `*_M15_{entity}_sids.csv` |
| `*_M15_{entity}_fib_lifecycle.csv` |
| `*_M15_{entity}_wvmi.csv` |

Total: 21 CSVs per replay (POI-zones CSVs added 2026-06-08). Current path is `artifacts/debug/`; baseline
path is whichever step 1 resolved to: `artifacts/commits/<branch>/<folder>/`
(new layout) or `artifacts/commits/<folder>/` (legacy flat layout).

**Files present on only one side are a finding, not a skip.** Diff the two
file *lists* first and report every `NEW (no baseline)` and `MISSING (in
baseline, not produced)` file explicitly. The pool/lifecycle redesign (Plan
C, 2026-09) replaces the per-lens `*_sids.csv` with `*_subs.csv` +
`*_triggers.csv` + `*_unresolved_triggers.csv`; on its first run those have
no baseline and the old file disappears — both must be called out, and the
new tables are validated against the plan's **predicted table**
(`memory/reference_pool_redesign_groundtruth.md`) rather than a prior save.
A missing per-sub CSV can also mean the chart export that it is coupled to
crashed (see `run_replay.py` — per-sub CSVs are written inside the chart
loop), so treat MISSING as a possible crash signal.

**Classify deltas against the plan's stated expectations.** When the change
being compared has a plan that lists its expected deltas (per
`memory/feedback_one_cause_per_compare.md`), the summary's EXPECTED /
UNEXPECTED split is *against that list* — an "expected byte-identical" plan
with any non-empty diff is a bug signal, not a review task.

## Chart Count Parity (Corroborating check)

Chart trace/shape counts tally everything rendered for an entity, so a
mismatch confirms divergence even when you can't immediately tell which
CSV moved. They are no longer the *only* sub-entity signal (the
per-entity CSVs above are primary), but they're cheap to read from
stdout and catch chart-side regressions (style registry, hover,
lifecycle gating) that the CSVs alone miss.

After the replay run, verify the three trace/shape counts in the stdout
match the baseline run's counts exactly. Per the latest run the standard
counts are (last verified 2026-09-19 on the Stage-3.2a tree = revert `5a658dc`,
again after Plan A, and after Plan B 2026-09-20; full 2025-12-01→2026-01-20 window; the 2026-05-29
values were H1 125/261, counter 216/169, confluence 359/280. Window-dependent,
re-baseline when the config window or chart rendering changes):

- `H1` chart: traces=107, shapes=250
- `M15.counter` chart: traces=198, shapes=133
- `M15.confluence` chart: traces=342, shapes=222

These print as `DEBUG traces:` / `DEBUG shapes:` (H1) and `[m15_chart] traces:
N, shapes: M` (each M15 entity) at the end of `python -m engine_v2.run_replay`.

A run with all 21 CSVs byte-identical but shifted chart counts means a
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
