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
trusted.** The M15 input is not among the 30 saved CSVs and saves carry no
`run.log`, so an incomplete M15 frame shows up only as M15 deltas that look
exactly like an engine change. This happened in a 2026-09-22 audit run:
`multitf/data_bridge.fetch_lower_tf_data` (14-day OANDA chunks) then printed a
failed chunk and carried on, and after two HTTP 504s the replay exited 0 on
2500 of 4228 M15 candles. **Since 2026-09-27 the fetch fails loudly:** a chunk
whose request fails (any non-200 status, or any `requests` error — a network
error, a non-JSON body) is retried twice, each retry printing one line
`[data_bridge] RETRY k/2 M15 chunk <from>-<to> in 5 s: <exception>`; a config
or payload-schema error raises at once; a chunk that still fails, or a fetch that
returns no candles at all, raises `RuntimeError: [data_bridge] ERROR …` and
the replay crashes (no `=== Replay Timing ===` block). The H1 fetch always
raised on a non-200 response; **since 2026-09-28 it retries the same way**
(`run_replay._get_history_retrying`, one call per auto-extend request, same
policy constants): `[auto_extend] RETRY k/2 H1 fetch <from>-<to> in 5 s:
<exception>`, then `RuntimeError: [auto_extend] ERROR fetching H1 …` (an empty
H1 frame raises `[auto_extend] ERROR: no H1 data …`). The snippet has no H1
branch on purpose: a failed H1 fetch is a crashed run (first branch), and the
H1 lines must never carry the `[data_bridge]` / `[multi_tf:dual]` prefixes the
N/A branches key on (`tests/test_h1_fetch_retry.py` pins that). The gate still
runs: it ties the log to the outputs on disk and checks the exact candle count,
which also catches a window change or a short response that raised nothing.

```bash
RAW=$(ls -t artifacts/debug/*_raw.csv | head -1); \
EXPECTED_FETCH="[data_bridge] Fetched 4228 M15 candles for NZD_USD in 5 chunks"; \
if ! { [ run.log -nt "$RAW" ] && grep -aqF "=== Replay Timing ===" run.log; }; then \
    echo "FETCH GATE: FAIL (stale log or crashed run: not newer than *_raw.csv, or no timing block)"; grep -aF "[data_bridge]" run.log; grep -aE "^([A-Za-z_][A-Za-z0-9_]*\.)*([A-Za-z0-9_]*(Error|Exception|Timeout)|KeyboardInterrupt|SystemExit)(: |[[:space:]]*$)" run.log; tail -n 3 run.log; \
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

- **PASS needs the same-run check AND the fetch check.** The **same-run
  check** (first branch): `run.log` is newer than the newest `*_raw.csv` AND
  contains the `=== Replay Timing ===` block. `run_replay.py` writes
  `*_raw.csv` first (before `run_pipeline`) and prints the timing block last,
  so the log of the replay whose outputs are on disk passes both. A stale
  `run.log` from an earlier replay fails the first (a later replay run without
  `> run.log 2>&1` wrote a newer `*_raw.csv`), and a crashed run — a failed
  fetch included — fails the second; its FAIL prints any `[data_bridge]` lines,
  the traceback's exception lines (a possibly dotted `…Error` / `…Exception` /
  `…Timeout` name, `KeyboardInterrupt` or `SystemExit`, with or without a
  message — a chained traceback's cause and final exception both print) and the
  log's last 3 lines. **The last 3
  lines are NOT reliably the traceback** (measured 2026-09-28 on simulated H1
  and M15 fetch crashes): under `> run.log 2>&1` stdout is block-buffered and
  the lines still buffered at the crash are flushed AFTER the traceback, so the
  tail shows e.g. `[multi_tf:dual] confluence var1 …` + the RETRY lines — the
  exception grep is what names the error. The **fetch check** (PASS branch): the exact
  expected `Fetched` line is present AND there is no `[data_bridge] ERROR`
  line (a backstop — a fetch error now ends the run). A RETRY line on a PASS
  (`[data_bridge] RETRY` M15 or `[auto_extend] RETRY` H1) is benign (the retry
  succeeded and the count is exact) — the silent-skip grep below prints it;
  mention it in the report. `/commit-save` Step 4b runs this same snippet.
- **On FAIL, STOP.** `FAIL (stale log or crashed run …)` = the same-run check
  failed: either the log belongs to an earlier replay (a later replay run
  without `> run.log 2>&1` wrote newer outputs), or this replay crashed — the
  printed exception lines name it; a failed fetch raises `RuntimeError:
  [data_bridge] ERROR …` (M15) or `RuntimeError: [auto_extend] ERROR …` (H1)
  (report a **data-fetch failure**), anything else is a crash to debug.
  Plain `FAIL` = the fetch check failed: a `Fetched` count
  other than `EXPECTED_FETCH` — report a data-fetch failure (or a window
  change, see below) and quote the `[data_bridge]` lines. Either way,
  re-run the replay (step 2, run mode) and check again. **Never interpret the
  diff of a run that failed the gate.** Nothing it shows is evidence about the
  code.
- **Reuse mode:** the gate still applies, to the captured log of the replay
  being reused, and the same-run check is what stops reuse from passing on a
  stale log. If that replay was run without `> run.log 2>&1`, completeness
  cannot be checked, so run a fresh replay instead of comparing.
- **N/A: the snippet decides it from the log, and only when no M15 fetch ran
  (no `[data_bridge]` line at all).** Two cases: (a) no `[multi_tf:dual]` line
  either — the run has no lower timeframe (`config.py` `lower_timeframes=()`);
  (b) the no-trigger line `[multi_tf:dual] no triggers — no subs built`
  (`orchestrator._run_multi_tf_dual`) — M15 is on but the window has no H1
  trigger, so nothing is fetched. Either way that run writes no M15 CSVs. The
  snippet greps only the line's ASCII prefix `[multi_tf:dual] no triggers`:
  Python writes the redirected log in cp1252, so its `—` is byte `0x97` and a
  UTF-8 grep for it never matches. Rewording the `no triggers` text makes a
  no-trigger run FAIL (fails closed); renaming the `[multi_tf:dual]` tag gives
  it N/A (a) — the right outcome under the wrong label; a run that fetched
  could read N/A only if the `[data_bridge]` and `[multi_tf:dual]` prefixes
  both vanished at once.
  `tests/test_data_bridge_fetch.py` pins both prefixes and the `Fetched`
  line's shape.
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
cannot be missed; `retry` prints a recovered fetch retry (M15 or H1), which
the gate's PASS branch does not show:

```bash
grep -iaE "error|warning|skipping|unavailable|degenerate|collapsed|pending|no sid|retry" run.log
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
design, two line families (a third, the WVMI lock warning, since 2026-10-26; a fourth, the probe/sub consistency
warning, since PLAN_T T1 2026-10-27) — each is a **finding to explain**, not a bug by
itself:

- `[sweep] UNRESOLVED (skipping) reason=… lens=… parent=(S,C) type=…
  trigger_idx=… detail=…` — one per trigger that produced no record (`pending`
  / `collapsed_parent_cycle` — `degenerate_parent_cycle` until PLAN_N N4, 2026-10-12 — / `probe_failed` /
  `geometry_failed`); the same rows land in `*_M15_unresolved_triggers.csv`. Compare the set against the
  plan's expectation (reference window, measured on the first Plan C replay: 4 rows, all the collapsed reason).
- `WARNING [parent_tables] collapsed parent cycle (S,C): floor=… end=…` —
  one per parent cycle whose lifecycle floor ≥ its end (`PART4 §17.7`;
  expected on the reference window: (1,0) and (1,1)). `collapsed` joined the grep with PLAN_N (`degenerate` stays
  for the probe-window warnings); it also matches the informational `[chart][fib] rendered … not collapsed /
  disappeared` line — the grep count on the reference window is 33 since "WVMI back on the chart" step 3
  (2026-10-26: + the 10 `[wvmi] WARNING` lines below); 23 since PLAN_R step 3 (PLAN_N's 22 + a sixth
  pandas FutureWarning, `market_structure.py:2487` since step 3 — `:2484` before, the line moves with edits above
  it — benign; 24 at PLAN_R step 2 with a `sibling-CTS unavailable`
  line for 1046's subsequent_counter, gone with 1046; 21 before PLAN_N), plus any OANDA RETRY lines of that run.
- `[wvmi] WARNING <path> sid=S cycle=C: the locked LP L (BOS cycle=…'s last wave candle) != the last temporary LP T`
  (since 2026-10-26, "WVMI back on the chart" step 3; WVMI_SPEC "3. Locked") — slice-local indices on a sub; printed
  once per tracker run that locks the record, so a pool sub's line also comes back once per proximity-view cap as
  `[prox_view] pool#N cap=…: …` (GOTCHAS "A log line from a sub's downstream prints once per TRACKER RUN"). Expected
  on the reference window: 10 lines, all the collapsed confluence sub 1 cycle 0 (90 != 119 = 1837 != 1866) — the
  confluence lens + `pool#1` at caps None / 1894 / 1902 / 1940 / 1948 / 2231 / 2236 / 2343 / 2367.
- `WARNING [sweep] probe/sub disagree on the 2nd CTS | cycle 0's CTS_CONFIRMED: pool#N key=… probe finalize=F (cond)
  vs the sub's own=O (trigger first_confluence@T)` (since PLAN_T T1, 2026-10-27;
  `entity_df_mutation.check_probe_sub_consistency`) — a first_confluence probe that RAN and CREATED its sub disagrees
  with the sub's own MS run on the moment its start was finalized from. Expected on the reference window: 0 lines (the
  one probe that runs Phase 2, FC (0,0): probe and sub both 1020). A line = the probe's start / lifecycle start no
  longer matches its own structure — PLAN_T §2.2.5 lists the known causes (a pool hit, the probe's bound).

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

On `CTS_ESTABLISHED` / `BOS_CONFIRMED` / pattern-path `CTS_UPDATED`, `idx` IS the
moment since Plan E E4a / E4b / E4c (2026-09-25; `== confirmed_at`) — track meta
`cts_anchor_idx` / `bos_anchor_idx` for location shifts (the ANCHOR — a price location; `price` stays
the anchor's price). The moment the event became known is `meta["confirmed_at"]`,
stored inside the `meta` column of `*_structure_events.csv`, and it is the
value that timing and lifecycle code reads. Since Plan E E4 an anchor shift alone
leaves `idx` unchanged (and, diffing against a pre-E4 save, a `confirmed_at` shift
alone left the old anchor-stamped `idx` unchanged) — so compare both. On `CTS_CONFIRMED`, `idx` ==
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

**H1.main** (11 files):

| File pattern |
|--|
| `*_final.csv` |
| `*_raw.csv` |
| `*_structure_levels.csv` |
| `*_kl_zones.csv` |
| `*_poi_zones.csv` |
| `*_ic_candidates.csv` — the IC selection (`*_ip_candidates.csv` until PLAN_L landing 1, 2026-10-09; PLAN_H landing A; the picked rows = the drawn POIs since landing B); + the column `ic_ref_reason` after `touches_zone` since PLAN_K landing A (2026-10-09); + `immediate_imbalance_first_c2_idx` after `ic_ref_reason` since the IP tiebreaker update (2026-10-09); + `has_unfilled_imbalance_after`, `is_ic_candidate` after it and one row per agreed-opposite candle of the window since PLAN_L landing 1; + `ip_outcome` after `is_ic_candidate` and the 4 `ranking_*` columns at the end since PLAN_L landing 2 (2026-10-10); + `ic_zone_band` after `is_ic_candidate` since PLAN_P (2026-10-13 — every row; `touches_zone` is that band's touch, `immediate_imbalance_first_c2_idx` = IC condition 3, `has_unfilled_imbalance_after` information only); + Step 1B's 5 columns `prior_imbalance_first_c2_idx`, `prior_imbalance_unfilled` (per check row), `prior_imbalance_outcome`, `prior_imbalance_passer_count`, `ic_candidate_count` (per record) after `ic_zone_band` since PLAN_S step 3 (2026-10-21 — `ip_outcome` = Step 2's outcome over the list Steps 2–3 ran on, the passers when 1B applied) |
| `*_structure_events.csv` — `STATE_CHANGED(to=reversal)` carries `meta["pattern_anchor_idx"]` since PLAN_K landing A; a PULLBACK `CTS_CONFIRMED` carries it too (the pullback pattern's first candle, right after `'via'`) since "WVMI back on the chart" step 3 (2026-10-26, S_lp_furthest `20261026_192655_f63df34` the first save with it — a save-format boundary for all 3 structure-events CSVs: across it diff them with `engine_v2/plans/qa_chart_edits_inputs/cmp_events_pattern_anchor.py BASE CUR`, positional `cmp_save.py` reports every such meta cell) |
| `*_imbalance_instances.csv` |
| `*_fib_lifecycle.csv` — no `fib_mode` column since PLAN_I S6 (2026-10-09): against an older save (S_J `20261009_013718_708a8d9` and before) diff the 3 fib CSVs with `engine_v2/plans/plan_i_inputs/cmp_s6_fib_mode.py BASE CUR` (positional `cmp_save.py` shifts the `meta` cell); the `ic_ref_*` meta keys since PLAN_K landing A |
| `*_wvmi.csv` |
| `*_wvmi_lps.csv` — the per-LP WVMI rows, one per dot ("WVMI back on the chart" step 4, 2026-10-26 — `debug/export_wvmi.export_wvmi_lps`; collapsed records none; WVMI_SPEC "Per-LP rows"); NEW since step 4 — S_lp_rows `20261026_220225_588014a` is the first save with it; against an older save it is `NEW (no baseline)` |

**Save-format boundary — PLAN_K landing A (2026-10-09; S_K_A `20261009_112137_02e9381` is the first save with it):** a save up to S_S6
`20261009_020650_5f6bae3` lacks the IP CSVs' `ic_ref_reason` column, the fib meta's `ic_ref_price` / `ic_ref_source` /
`ic_ref_break_pattern_anchor_idx` / `ic_ref_break_confirmed_idx` and the reversal `STATE_CHANGED`'s
`pattern_anchor_idx` — up to 9 CSVs (the 3 structure-events, 3 fib-lifecycle, 3 ip-candidates; 8 on the reference
window, whose counter-lens structure events hold no reversal) plus the fib hover `customdata`. Against such a save run
`engine_v2/plans/plan_h_inputs/cmp_k_landing_a.py BASE` (format deltas only, exit 1 on anything else) instead of
reading those files' diffs as engine changes.
**PLAN_K landing B (2026-10-09) — a BEHAVIOUR change, not a format one:** the IC reference condition filters the
pick, and the `IP: none` row's `reason` is `no_pattern` (IC candidates, none patterned) or `no_ic_candidate` (no
candle passes every IC condition; replaces `no_candidates`). Against S_K_A: `engine_v2/plans/plan_h_inputs/
cmp_k_landing_b.py BASE` (PLAN_K §11.3's V + the sub 1 cyc 2 relabel, exit 1 on anything else). S_K_B
`20261009_140048_97795bd` is the first save with it (`LATEST_<branch>` until S_TBU, below).
**The IP tiebreaker update (2026-10-09, PLAN_H §6 TBU) — a BEHAVIOUR + FORMAT change:** a new first tiebreaker (the
immediate unfilled imbalance) and descriptive `ip_tiebreak` values — `"TB1"` → `"ic_candle_type"`, `"TB2"` →
`"most_recent_before_imbalance"`, new `"immediate_unfilled_imbalance"` (POI meta → `*_poi_zones.csv` `meta`, the IP
CSVs, the POI hover) — and the IP CSVs' new column `immediate_imbalance_first_c2_idx` after `ic_ref_reason`. Against
S_K_B or older: `engine_v2/plans/plan_h_inputs/cmp_tbu.py BASE` (the predicted pick move + the relabels + the new
column, exit 1 on anything else) instead of reading those diffs as engine changes. S_TBU `20261009_171255_5b5c307` is the first save with it (`LATEST_<branch>` until S_L1, below).
**PLAN_L landing 1 (2026-10-09, the IC identification in steps) — a FORMAT change, the selection identical:** the three
IP CSVs are RENAMED `*_ip_candidates.csv` → `*_ic_candidates.csv`, list every agreed-opposite candle of each fib's
window (the candles failing IC condition 3 are new rows — on the reference window H1 +5, confluence +41, counter +31)
and carry two new columns `has_unfilled_imbalance_after`, `is_ic_candidate` after `immediate_imbalance_first_c2_idx`.
Every other CSV and every figure byte-identical. Against a save from PLAN_H landing A up to S_TBU: `cmp_fig_full.py`
lists exactly the 3 renamed files as MISSING + NEW (before PLAN_H landing A: 3 NEW only); compare them with `engine_v2/plans/plan_h_inputs/cmp_l1.py BASE CUR_DEBUG` (keyed; the
per-lens new-row counts are the reference window's). **Delete the 3 stale `artifacts/debug/*_ip_candidates.csv`
first** (no replay overwrites them — `cmp_save --names-from` an older save would compare them as equal, the
`*_sids.csv` trap below). The older boundary scripts (`cmp_tbu.py`, `cmp_k_landing_a/b.py`) read
`*_ip_candidates.csv`: run them with CUR = S_TBU, then `cmp_l1.py` S_TBU → current. S_L1 `20261010_114846_b5c3613` is the first
save with it (`LATEST_<branch>` until S_L2, below).
**PLAN_L landing 2 (2026-10-10, Step 3 — the ranking rules) — a BEHAVIOUR + FORMAT change:** a fib record with IC
candidates but no institutional pattern now gets a RANKED IC (POI_ZONES_SPEC §3.6) — on the reference window 7 new
POIs (H1 180, 694; confluence 670, 929, 1804, 2575, 3809), no structure / trigger / sub / WVMI change; the POI meta
gains 5 keys (`ip_outcome` before `ip_rank`, the 4 `ranking_*` after `ip_tb_imbalance_first_c2_idx` — every POI row's
`meta` cell differs) and a ranked POI's pattern keys are None (`ip_rank` column empty); the IC CSVs gain `ip_outcome`
(after `is_ic_candidate`) and the 4 `ranking_*` columns (at the end); the no-IC row's reasons are `no_ic_candidate` /
`all_ic_candidates_pinbar` (`no_pattern` is Step 2's `ip_outcome` now — the 7 records' no-IC `no_pattern` rows are
gone: H1 241 → 239, confluence 1324 → 1319 — and each record's ranked candidate row becomes its picked row); the hovers: a ranked POI's `No pattern · ranking rule …` / `No pattern · only IC candidate`, the fib
hover's `IC: none (…)` / `IC: not evaluated (…)` (were `IP: …`) and `No pattern · IC <idx> · …`. Exactly 6 CSVs differ
(the 3 POI + 3 IC CSVs); the figures differ by the 4 DRAWN new POIs — H1 180 (+ its H1-overlay copy on both
lenses), confluence 929 / 2575 / 3809; 694 / 670 / 1804 sit on collapsed cycles and are hidden like their fibs — and
the hover texts. Against S_L1:
`engine_v2/plans/plan_h_inputs/cmp_l2.py BASE [CUR_DEBUG CUR_CHARTS]` (keyed: the 21 other CSVs md5, the POI rows per
key with the meta per key, the IC CSV columns by name, the figure counts; exit 1 on anything else) — a positional
`cmp_save.py` breaks on the POI `meta` cell and the new columns. Older saves: chain through S_L1 (`cmp_l1.py`, then
`cmp_l2.py`). S_L2 `20261010_155816_6d8cc67` is the first save with it (source `6d8cc67`; `LATEST_<branch>` until S_M1, below).
**PLAN_M landing 1 (2026-10-11, the live-`P` plumbing) — NO format or behaviour change on the reference window:** a
cross-fib limit may now MOVE (CROSS_CYCLE_FIB_SPEC §3 "A moving `P`"), but no producer makes one move until landing 2,
so every CSV and figure is byte-identical to S_L2 (`cmp_fig_full.py` 27 / 27 md5, figures identical; `fig_hash.py`
equal). The fib meta gains two keys ONLY on a structure whose `P` moved by its end (`cross_fib_limit_price_at_start`,
`cross_fib_limit_move_count`, right after `cross_fib_limit_from_sub_id`) and the hover a `(moved N× from …)` suffix —
none on this window. S_M1 `20261011_204047_86d4f7b` (source `86d4f7b`; `LATEST_<branch>` until S_M2, below).
**PLAN_M landing 2 (2026-10-12, a COUNTER sub's cross limit = its parent cycle's BOS threshold) — a BEHAVIOUR
change:** subs 1 / 3 / 5 / 7 take `parent_bos_threshold` (0.55808 / 0.5736 / 0.58105 / 0.58105, no move on this
window). Exactly 6 CSVs differ vs S_M1 — both M15 lenses' `fib_lifecycle`, `ic_candidates`, `poi_zones` (sub 3 cycle 2
becomes a cross from cycle 1, pre-established at 2967, its POI's IC 2969 → 2821; `via` relabels on subs 1 / 3; the
IC-reference moves on subs 1 / 3 / 7; the limit keys of every record of the four subs); no structure / trigger /
sub / WVMI / H1 change; trace / shape COUNTS unchanged, the changed hovers + sub 3's fib / POI traces differ. Against
S_M1 (or S_L2): `engine_v2/plans/plan_m_inputs/trial_diff.py BASE CUR_DEBUG CUR_CHARTS` (keyed) — S_M2's output is
`plan_m_inputs/trial_diff_vs_S_L2.txt`; the exact outputs: `trial_md5.txt` / `trial_fig_hash.txt`. S_M2
`20261012_023137_10e4c6f` (source `10e4c6f`; `LATEST_<branch>` until S_N, below).
**PLAN_N (2026-10-12, the Naming-Standard audit) — a FORMAT change (meta keys + one meaning), figures identical:** KL
`bounds_steps` step `start_idx` → `base_idx` (INIT) / `expanded_idx` (expansion); `RANGE_STARTED` `start_idx` /
`confirm_idx` → `range_anchor_idx` / `confirmed_at` (and the pullback / proximity rows GAIN both — the anchor = their
`cts_anchor_idx`); `STATE_CHANGED` `effective_idx` = the row's `idx` on the `range_confirmed` rows (was the range
start); the unresolved reason `degenerate_parent_cycle` → `collapsed_parent_cycle` (+ the `run.log` WARNING /
`[parent_tables] … collapsed=` lines). Exactly 7 CSVs differ vs S_M2 — the 3 KL, 3 structure-events, the unresolved
table: 164 cells — and no figure. Against S_M2 (or older, chaining through it): `engine_v2/plans/plan_n_inputs/
cmp_plan_n.py BASE CUR_DEBUG --expect-cells 164` (applies the transform to BASE, requires text equality; exit 1 on
anything else); positional `cmp_save.py` would report the 164 cells as engine deltas. S_N
`20261012_165925_1b2f24c` (source `1b2f24c`, save `81a0d46`, trunk `7869a24`; `LATEST_<branch>` until S_N_hover, below).
**The PLAN_N hover follow-up (2026-10-12, option B: the H1 range hover line = the box, candle for candle) — a FIGURE
change only:** the H1 range hover lines snapshot at each event's replay candle (a `RANGE_STARTED` at its anchor). Vs
S_N exactly 4 H1 traces differ — `range:top/bottom:sid0` 773 → 780 points, `range:top/bottom:sid1` 341 → 345; 28
candle points (a label range's line now starts at its anchor; the stale confirm-candle states corrected) — while every
CSV is md5-equal and the M15 figures and the H1 shapes / layout are identical. Against S_N (or older): `fig_hash.py`
(both M15 hashes equal) + the keyed `PYTHONPATH=. python engine_v2/plans/plan_n_inputs/h1_trace_diff.py diff BASE
CUR_H1_DIR` (a dir holding ONE H1 html; expected output `plan_n_inputs/followup_h1_diff_vs_SN.txt`) instead of reading
those hover points as engine deltas. S_N_hover `20261012_194022_9f9406d` (source `9f9406d`, save `cc3d3ff`, trunk
`efd95b9`; `LATEST_<branch>` until S_O, below).
**PLAN_O (2026-10-13, a cross's IC reference = its TARGET's previous CTS, the user's change 2) — a BEHAVIOUR change, no
format change:** the window's 4 cross records' `ic_ref_*` fib meta change (H1 sid 1 0→1, sub 3 1→2, sub 7 0→1 / 0→2)
and with them their IC CSV rows' `ic_ref_reason` / `is_ic_candidate`; ONE pick moves — sub 3 cycle 2 2821 → 2837 (1A.iv,
both lenses; its `ip_outcome` flips on all 130 rows). Exactly 8 of 27 CSVs differ vs S_N_hover (3 fib-lifecycle, 3 IC,
the 2 lens POI CSVs); no structure / KL / WVMI / trigger / sub change; chart counts unchanged; the H1 figure identical
(sid 1's cross fib is on collapsed (1,1), not drawn); each lens 5 traces' customdata (sub 3's 2 POI-zone hover traces,
the 3 cross fibs' hover) + 3 shapes (sub 3's 2 POI rectangles, its confirm line). Against S_N_hover: positional
`cmp_save.py` + `cmp_hover.py` read correctly (no column change) — every delta = PLAN_O §10. S_O
`20261013_151425_ce5ac9b` (source `ce5ac9b`, save `b138424`, trunk `07f5ff6`; `LATEST_<branch>` until S_P, below).
**PLAN_P (2026-10-13, IC condition 3 = the immediate unfilled imbalance + Step 1A, the 80–100 fallback zone — the
user's changes 1 + 3) — a BEHAVIOUR change AND a save-format boundary:** the IC CSVs gain the column `ic_zone_band`
after `is_ic_candidate` (every row, the no-IC row included); the POI meta gains `ic_zone_band` (after `ip_outcome`);
the `ip_tiebreak` / `ranking_rule` value `immediate_unfilled_imbalance` is gone (the rule became IC condition 3).
Meaning changes of existing IC CSV columns: `touches_zone` = the touch of the record's band (80–100 on EVERY row of a
record where Step 1A ran); `is_ic_candidate` reads the immediate gap; `immediate_imbalance_first_c2_idx` IS condition
3 and `has_unfilled_imbalance_after` is information only — both with UNCHANGED values (the CSV always classified every
check). Against S_O: positional `cmp_save.py` breaks on the new column (every IC row shifts) and the POI `meta` cell —
use the keyed `engine_v2/plans/qa_ic_crossfib_inputs/cmp_trial.py BASE CUR` (IC outcome per record, the drawn POIs,
the structure events, fibs, subs, triggers) and `qa_ic_crossfib_inputs/cmp_hover_strings.py BASE CUR` (hover lines,
works when the trace counts differ). 22 of 27 CSVs differ; H1 POIs 5 → 2 (180, 694, 732 gone; 832 → 859); ONE
structure event — H1 sid 1 cycle 2 CTS_CONFIRMED 939 → 935 (the counter triggers 3759 → 3743 / 4159 → 4143, sub 5
[3743, 4143], sub 7 from 4143); 4 Step 1A POIs (650, 923, 1934, 4131); 0 M15 structure-event changes (1 meta cell:
confluence 1902 `trigger_inner`); chart counts 81/236, 146/135, 289/293. Every delta = PLAN_P §6 (the landing
replay is md5- / figure-hash-equal to the variant, `qa_ic_crossfib_inputs/plan_p_variant/`). S_P
`20261013_231327_23f395d` (source `23f395d`, save `69ac1d7`, trunk `ab4c982`; `LATEST_<branch>` until S_Q, below).
**PLAN_Q (2026-10-14, the zone-proximity scan reads a POI exactly like a KL zone — by location at every candle of the
scan, never its activation history; the user, session order item 3) — a BEHAVIOUR change, no format change:** on this
window the outputs move only in the hovers. Vs S_P 0 / 27 CSVs differ (no trigger, sub, WVMI or event change — the
one POI the rule would have moved, S_O's collapsed (1,1) at 820, PLAN_P had already removed); exactly 3 figure leaves:
the `zone_proximity:trigger` marker trace's `customdata[5][8]` (the H1 sid 1 cyc 2 1035 sd:POI marker's
`zone_confirmed_idx`) 998 → 902 on the H1 chart and both M15 lenses — a POI marker now shows its UNCLAMPED activation
floor, not its active stretch's start (CHARTING_SPEC §12); layouts equal; chart counts unchanged. Against S_P (or
older): `cmp_save.py` (CSVs) reads correctly; the positional `cmp_fig_full.py` or `plan_q_inputs/fig_leaf_diff.py`
shows those 3 leaves (expected output `plan_q_inputs/variant_leaf_diff_vs_S_P.txt`) — not an engine delta. Every
delta = PLAN_Q §6 (the save replay is md5- / figure-hash-equal to the variant, `plan_q_inputs/variant_*`). S_Q
`20261014_170919_da8c15e` (source `da8c15e`, save `ff8d246`, trunk `ab8e3cf`; `LATEST_<branch>` until S_R2, below).
**PLAN_R step 2 (2026-10-20, the proximity overlay — sub zones join the parent's proximity reference ladders, inside
the lifecycle sweep; the user, session order item 3) — a BEHAVIOUR change, no CSV format change:** vs S_Q 16 / 27
CSVs differ — all 16 M15 lens CSVs; the 10 H1 CSVs and the unresolved table are identical. H1 zone-proximity
triggers 6 → 10 ((0,1) +820 opp_sd on a sub POI, +872 sd:BOS; (1,2) +1045, +1046, and 954 now names a sub source);
subs 8 → 11, records 11 → 15, unresolved 4 (the same rows). `sub_id`s RENUMBER (start order): key subs, records and
their lens rows by `(direction, starting_idx)`, never by `sub_id` or row position — a positional `cmp_save.py` diff
of the lens CSVs is mostly renumbering. Figures: H1 81/236, only the `zone_proximity:trigger` marker trace differs
(a sub-sourced marker's type line names its source, e.g. `opp_sd: sub_3 c2 POI 0.58096`, and its
`zone_confirmed_idx` = that zone's usable moment), layout equal; M15 counter 146/135 → 164/144, confluence
289/293 → 257/263. Skip grep 22 → 24 (above). Every delta = PLAN_R §8 "Step 2" (the save replay is md5- /
figure-hash-equal to the validated variant; the pool table byte-identical to
`qa_opp_sd_zones_inputs/sims/shadow_full2.log`). S_R2 `20261020_115859_817d2a9` (source `817d2a9`, save `e6c3f7a`,
trunk `87535dd`; `LATEST_<branch>` until S_R3, below).
**PLAN_R step 3 (2026-10-20, the proximity reference gate replaces the narrow proximity-reference cap — a reference
update is made only if the gap after it stays above 40 / 25 / 15; the user's round-15 pivot) — a BEHAVIOUR change,
no CSV format change:** vs S_R2 16 / 27 CSVs differ — the 16 M15 lens CSVs; the 10 H1 CSVs and the unresolved table
are identical. H1 zone-proximity triggers 10 → 8 (1045 / 1046 gone: the gate skips every sub_8 zone, the superseded
sub_6 c0 POI stays the opp_sd reference); subs 11 → 9 (sub_9 (−1,4179) / sub_10 (1,4183) not built — the last two
in start order, so `sub_id`s 0–8 keep their keys), records 15 → 14, unresolved 4; sub_7 (−1,3760) runs to its
reversal 4200, sub_8 (1,4027) is not cut at 4187 and gains a confluence (1,2) reversal record — the (1,2) tail subs
are S_Q's by key. Diff the lens CSVs by sub key with `engine_v2/plans/qa_opp_sd_zones_inputs/build/kdiff_saves.py
BASE CUR` (expected output vs S_R2: `build/kdiff_step3_vs_S_R2.txt`): a textual diff of
`_M15_confluence_fib_lifecycle.csv` shows EVERY row changed — sub_8's open fib turns the `end_idx` column float
(`1020` → `1020.0`); by value only (−1,3760) differs and (1,4027) is new. Figures: H1 81/236, only the
`zone_proximity:trigger` marker trace differs (8 markers), layout equal; M15 counter 164/144 → 153/138, confluence
257/263 → 301/309. Skip grep 24 → 23. Every delta = PLAN_R §9.4 / §9.6 (predicted before the build; the save replay
is md5- / figure-hash-equal to the validated variant, and all 27 CSVs md5-equal to the step-3 shadow
`build/r6_gate_shadow.py`'s). S_R3 `20261020_225749_d424099` (source `d424099`, save `d613dfa`, trunk `630f024`; =
`LATEST_<branch>` until S_S1, below).
**PLAN_S step 1 (2026-10-21, the imbalance arming threshold 0.70 → 0.65 — one constant `IMBALANCE_FILL_THRESHOLD`,
every fill reader; the user's IC-revisit change A) — a BEHAVIOUR change, no format change:** vs S_R3 exactly 11
real cells in 5 CSVs — H1 `imbalance_instances` 7 (instances arm earlier: 275 → 267, 445 → 444, 494 → 493,
510 → 509, 868 → 866, 1042 → 1041; #92 armed + committed 761 → 600), both lenses' `fib_lifecycle` 1 (sub 3 cycle
0's last deactivate / reactivate pair 2667 / 2672 → 2722 / 2723) and `ic_candidates` 1 (an IC check, not a pick,
loses condition 3: its immediate gap 3488 fills); every figure identical (`fig_hash.py` equal to S_R3, hovers
included); no structure, POI, trigger, sub or WVMI change. Positional `cmp_save.py` reads it correctly. Every
delta = PLAN_S §2.2 (the save replay is md5- / figure-hash-equal to the variant `make_variant_fill65.py`). S_S1
`20261021_020427_69c099b` (source `69c099b`, save `b4f040b`, trunk `138c2a7`; = `LATEST_<branch>` until S_S2, below).
**PLAN_S step 2 (2026-10-21, the two fill strokes on SEPARATE candles — stroke 2 only on a candle after the arming
one, a close AT the edge counts; every fill reader; the user's IC-revisit change B) — a BEHAVIOUR change, no format
change:** vs S_S1 exactly 51 real cells in 3 CSVs — H1 `imbalance_instances` 47 (every instance that armed and
committed on ONE candle — all 47 of S_S1's, none left — now commits later: `confirmed_fill_idx` = `armed_idx` → a
later candle, e.g. 19 → 20, 954 → 964; the one armed at 183 never commits; no `armed_idx` moves), H1 `poi_zones` 2 (sid 1 cycle 2, IC 859: `confirmed_idx` 998 → 954 —
its 992 off / 998 on pair is gone, one continuous zone), confluence `poi_zones` 2 (one off moment a candle later:
sub 1 cycle 2 2192 → 2193, sub 4 cycle 0 3377 → 3378); no structure, IC, fib, trigger, sub or WVMI change.
Figures: traces unchanged and xy-equal, shapes 236 → 234, 138 → 136, 309 → 307 (that POI's off / on pair, drawn on
H1 and both M15 overlays); hovers (`cmp_hover.py`): 0 templates, the customdata of 2 traces per chart (the same POI's
confirmed candle 998 → 954). Positional `cmp_save.py` reads it correctly. Every delta = PLAN_S §3.2 (the save replay
is md5- / figure-hash-equal to the variant `make_variant_stroke2_later.py` on step 1's). S_S2
`20261021_081421_cbaed31` (source `cbaed31`, save `780e628`, trunk `f4e83c6`; = `LATEST_<branch>` until S_S3, below).
**PLAN_S step 3 (2026-10-21, IC Step 1B — the IC candidates whose closest prior sd imbalance is unfilled through the
candle before go to Steps 2–3 first; no IC from them, or no passer → all the candidates; the user's IC-revisit change
C, POI_ZONES_SPEC §3.7) — a BEHAVIOUR change AND a save-format boundary:** the IC CSVs gain 5 columns after
`ic_zone_band` (`prior_imbalance_first_c2_idx`, `prior_imbalance_unfilled` per check; `prior_imbalance_outcome`,
`prior_imbalance_passer_count`, `ic_candidate_count` per record); the POI meta gains 6 keys (`prior_imbalance_outcome`,
`prior_imbalance_first_c2_idx`, `prior_imbalance_passer_count`, `ic_candidate_count`, `prior_imbalance_outranked_ic_idx`,
`prior_imbalance_outranked_ip_rank`); `ip_outcome` now = Step 2's outcome over the list Steps 2–3 ran on. Against S_S2:
21 / 27 CSVs md5-equal; the 6 IC / POI CSVs differ (H1's by the format only) — positional `cmp_save.py` breaks on the
new columns (every IC row shifts) and the POI `meta` cell: use the keyed `qa_ic_crossfib_inputs/cmp_trial.py BASE CUR`
and `qa_ic_crossfib_inputs/cmp_hover_strings.py BASE CUR`; `qa_ic_revisit_inputs/cmp_landed_vs_variant.py` strips the
step-3 columns / keys to compare with a behaviour-only tree. Exactly two picks move: confluence sub 0 cycle 2 IC
1216 → 1142 (POI confirmed 1230 → 1224); sub 3 cycle 1, both lenses, 2808 (pattern 3B.iii) → 2821 (confirmed
2829 → 2841); no structure-event (329 / 591 / 1388), fib, KL, trigger, sub, unresolved or WVMI change. Figures: H1
identical; chart counts unchanged (81/234, 153/136, 301/307); the two moved POIs' traces + the new hover lines
(`Prior imbalance <c2> unfilled · preferred <n> of <m> IC candidates [(over pattern <rank> at <IC>)]`, `… only
preferred IC candidate` — only when 1B narrowed the field: 1142, 2821, and sub 8 cross 0→2's unmoved 4131). Skip grep
24 = the 23 + one benign `[data_bridge] RETRY` (the fetch recovered). Every delta = PLAN_S §4.4 (the variant
`make_variant_step1b.py` — 97 cells vs S_S2 — predicted before the build). S_S3 `20261021_130400_7b62a47` (source `7b62a47`,
save `538ecc1`, trunk `231d5c8`; = `LATEST_<branch>` until S_dot_hover, below).
**The BOS / CTS dot hover (2026-10-22, SESSION ORDER item 5's first chart edit — the user's spec
`plans/qa_chart_edits_inputs/user_spec_20261022_dot_hover.md`) — FIGURE-ONLY:** 27 / 27 CSVs md5-equal to S_S3; traces,
shapes and layouts unchanged (81/234, 153/136, 301/307); the `customdata` / `hovertemplate` change on the dot traces
only (H1 5, counter 14 = 11 sub + 3 H1 overlay, confluence 21 = 18 + 3). Positional `cmp_save.py` is fine across it
(it ignores hovers); a hover diff across it (`cmp_hover.py` / `cmp_hover_strings.py`) shows every dot hover changed —
the format, not an engine delta. Checks: `plans/qa_chart_edits_inputs/fig_scope.py BASE CUR` (the figure-only scope:
same traces in order, layouts equal, hover changed only on the dot traces), `dot_hovers_new.py CHART_DIR` (every dot
hover rendered; `NA (before start_idx)` ⇔ `phase=forming`). Skip grep 23. S_dot_hover `20261022_200747_ab2fcc1`
(source `ab2fcc1`, save `2c30189`, trunk `79a43bf`; = `LATEST_<branch>` until S_zone_hover, below).
**The KL / POI zone hover (2026-10-23, SESSION ORDER item 5's second chart edit — the user's spec
`plans/qa_chart_edits_inputs/user_spec_20261023_zone_hover.md`) — FIGURE-ONLY:** 27 / 27 CSVs md5-equal to
S_dot_hover; traces, shapes and layouts unchanged (81/234, 153/136, 301/307); the hover changes on the zone traces only
(two per zone / KL bounds step) — H1: KL 30, POI 4; counter: sub KL 14, sub POI 14, H1-overlay KL 30, overlay POI 4;
confluence: sub KL 76, sub POI 30, overlay KL 30, overlay POI 4. A save-format boundary for hover diffs only: the zone
hovers gain `lifecycle=` / `zone_size=` lines and lose `status=` / the sub `sub window=` / `records:` lines, and
the IC lines are labelled (`ic_reason=` / `ic_tiebreak=` / `ic_prior_imbalance=` / `ic_zone=`; the S_S3 entry above
quotes them unlabelled). Positional `cmp_save.py` is fine across it. Checks: `fig_scope.py BASE CUR '^(KL zone|POI
zone|M15 KL zone|M15 POI zone|H1 KL zone \(overlay\)|H1 POI zone \(overlay\))'` (SCOPE OK; run it as `python -m
engine_v2.plans.qa_chart_edits_inputs.fig_scope` from the repo root), `zone_hovers.py SAVE_DIR` (every zone hover + a
CSV lifecycle table; `PYTHONIOENCODING=utf-8`). Skip grep 23. S_zone_hover `20261023_153743_56bb20d` (source
`56bb20d`, save `13d7252`, trunk `8b2916e`; = `LATEST_<branch>` until S_fib_hover, below).
**The Fib zone hover (2026-10-23, SESSION ORDER item 5's third chart edit — the user's spec
`plans/qa_chart_edits_inputs/user_spec_20261023_fib_hover.md`) — FIGURE-ONLY:** 27 / 27 CSVs md5-equal to
S_zone_hover; traces and layouts unchanged (81, 153, 301); the hover changes on the fib hover traces only — H1 3
(`fib:sid0:c0` / `c1`, `fib:sid1:c2`), counter 7, confluence 17 (`M15 fib`). A save-format boundary for hover diffs
only: the fib hover's lines are relabelled and reordered (`ic_ref:` / `limit_price=` / `lifecycle=`, the anchors' block
first, `struct_direction` added), the sub's `sub window=` / `records:` lines and every pick's IC outcome line are gone
(an `ic_idx=NA (…)` line only on a fib with no IC). Positional `cmp_save.py` is fine across it. Check: `fig_scope.py
BASE CUR '^(fib:|M15 fib$)'` (SCOPE OK; `python -m engine_v2.plans.qa_chart_edits_inputs.fig_scope` from the repo
root). Skip grep 23. S_fib_hover `20261023_191547_563f5f8` (source `563f5f8`, save `bf3c5a3`, trunk `023d36e`; =
`LATEST_<branch>` until S_wave_hover, below).
**The wave candle hover (2026-10-23, SESSION ORDER item 5's fourth chart edit — the user's spec
`plans/qa_chart_edits_inputs/user_spec_20261023_wave_candle_hover.md`) — FIGURE-ONLY:** 27 / 27 CSVs md5-equal to
S_fib_hover; traces and layouts unchanged (81, 153, 301); the hover changes on the 112 wave-candle traces only
(UNNAMED traces, so `fig_scope.py`'s name regex cannot select them — match the hovertemplate's `<b>Wave Candle</b>`;
`plans/qa_chart_edits_inputs/wc_scope.py BASE CUR` does):
H1 11, counter 22 sub + 11 H1 overlay, confluence 57 + 11. A save-format boundary for hover diffs only (`cycle_id=`
after ` | `, the structure's window and `struct_direction` added, `volume=`, the sub's `first record:` line before
the role). **`debug/chart_census.py` was broken across the four hover edits and is fixed here** (the dot hover keyed
its sub dots by hover strings, the KL / POI zone hover made it crash, M15 wave candles were all `?`): it reads every
format to one `(parent_sid, parent_cycle, sub_id)` key, so `chart_census.py <S_S3 chart> <current chart>` → 0
deltas on both lenses — run it on every hover-format change (GOTCHAS "A hover-format change breaks the
chart-reading debug tools silently"). Skip grep 23. S_wave_hover `20261023_194643_18711dc` (source `18711dc`, save
`d15cb00`, trunk `293b7c7`; = `LATEST_<branch>` until S_prox_hover, below).
**The zone proximity trigger marker hover (2026-10-24, SESSION ORDER item 5's fifth chart edit — the user's spec
`plans/qa_chart_edits_inputs/user_spec_20261023_proximity_marker_hover.md`) — FIGURE-ONLY:** 27 / 27 CSVs md5-equal
to S_wave_hover; traces and layouts unchanged (81, 153, 301); the hover changes on the `zone_proximity:trigger`
trace only (one per chart, 8 markers each). A save-format boundary for hover diffs only (`TF=1H`, `sid=… |
cycle_id=…` on one line, `trigger_price=` / `zone_inner=` / `sub_trigger=` added, `zone_kind=` gone, a sub zone's
`type` without its inner; the M15 copy `idx=` / `parent_sid=… | parent_cycle_id=…`, no `struct_direction`).
Positional `cmp_save.py` is fine across it. Checks: `fig_scope.py BASE CUR '^zone_proximity:trigger$'` (SCOPE OK),
`chart_census.py` vs S_S3 (0 deltas), the run log's `[uc1_trigger]` lines unchanged (3; the chart's UC1 call for
`sub_trigger` is silenced). Skip grep 23. S_prox_hover `20261024_105742_fb0189b` (source `fb0189b`, save `b2ab158`,
trunk `1b513ec`; = `LATEST_<branch>` until S_repl_break, below).
**A replaced sub's line breaks at the same point on every lens (2026-10-24, SESSION ORDER item 5's second pass, the
charts' other elements — the user's request + decision `plans/qa_chart_edits_inputs/user_spec_20261024_replacement_break.md`)
— FIGURE-ONLY:** the replacement break now reads the SUB's pool-wide replacer (`SidRecord.meta
["ended_by_starting_idx"]`), not this lens's records. 27 / 27 CSVs md5-equal to S_prox_hover; H1 81/234 unchanged;
counter 153 → 154 traces (+ `M15 PB sub5 (prior)` at 3760; sub 5's dotted run 3621 → 3760 → 3819), confluence
301 → 302 (+ `M15 PB sub3 (prior)` at 3304; sub 3's dotted run 3047 → 3304 → 3490); shapes and layouts unchanged; no
other trace changed. **Positional `fig_scope.py` breaks across it** (one inserted trace per M15 chart shifts every
later trace) — diff figures with `plans/qa_chart_edits_inputs/fig_named_diff.py BASE CUR` (name-keyed); positional
`cmp_save.py` compares CSVs and is fine. `chart_census.py` vs S_prox_hover: `dots_PB` +1 on (0,1,5) counter and
(0,1,3) confluence, nothing else (redirecting its output needs `PYTHONIOENCODING=utf-8`). Skip grep 23. S_repl_break
`20261024_171443_56517e7` (source `56517e7`, save `1cfe56d`, trunk `7f6fed3`; = `LATEST_<branch>` until S_uc1_prox,
below).
**WVMI back on the chart, step 1: `first_counter` reads the H1 cycles' first sd proximity trigger (2026-10-26,
SESSION ORDER item 5's second pass — the user's design `plans/qa_chart_edits_inputs/user_spec_20261024_wvmi_on_chart.md`)
— ZERO CHANGE:** `multitf/uc1_trigger.detect_uc1_triggers` reads the proximity lists
(`zone_proximity.first_sd_proximity_triggers`, shared with the main WVMI gate), not the gated H1 WVMI records. 27 /
27 CSVs byte-equal to S_repl_break; the three charts' traces, shapes and layouts JSON-equal (81 / 154 / 302 traces,
hovers included); the run log's `[uc1_trigger]` lines the same 3 ((0,1), (1,1), (1,2)). Not a save-format boundary.
S_uc1_prox `20261026_175906_257f8ba` (source `257f8ba`, save `be92848`, trunk `7d4fa57`; = `LATEST_<branch>` until
S_h1_ungated, below).
**WVMI back on the chart, step 2: the H1 WVMI without its gate (2026-10-26, the same design file) — BEHAVIOUR, the H1
`_wvmi.csv` only:** every H1 cycle with a CTS_CONFIRMED gets a record (`_compute_wvmi_records` is ungated for the main
too); the main's `triggered_by_*` became attribution (`_stamp_main_wvmi_attribution`: the cycle's first zone-proximity
trigger when sd, else None — a declared rule-3 meaning change). = the prediction: H1 `_wvmi.csv` 3 → 5 rows, the 3
old rows byte-equal; + (0,0) buy, locked by cycle 1, FB 110 / LB 430 / FP 436 / LP 590, breakout 0.733, pullback
0.521; + (1,0) sell, locked by cycle 1, `cycle_collapsed` True, FB 690 / LB 709 / FP 712 / LP 728, breakout 0.390,
pullback 0.250; both with empty `triggered_by_*`, meta = the five keys with None + `structure_path_id`. Run log
`[wvmi] total=3, locked=1` → `total=5, locked=3`; the same 3 `[uc1_trigger]` lines. 26 / 27 CSVs byte-equal; the
three charts unchanged (`fig_named_diff.py`: no trace added / removed / changed, layouts equal). Not a save-format
boundary (same columns), but the rows follow the CTS_CONFIRMED processing order, so (0,0) (439) lands first and
(1,0) (727) third: diff the H1 `_wvmi.csv` KEYED
(`cmp_wvmi_keyed.py`: 2 rows added, every other row unchanged) — positional `cmp_save.py` shows every row shifted.
Skip grep 23. S_h1_ungated `20261026_183615_76b5a17` (source `76b5a17`, save `e9a58a3`, trunk `4e41902`; =
`LATEST_<branch>` until S_lp_furthest, below).
**WVMI back on the chart, step 3: the new LP rule (2026-10-26, the same design file "Step 3") — BEHAVIOUR (2 `_wvmi.csv`)
+ a SAVE-FORMAT BOUNDARY (the 3 structure-events CSVs):** the WVMI temporary LP = the furthest qualified close in the
pullback direction, its first one by how the CTS confirmed (`zones/wvmi._temporary_lp_sequence`; was the close
nearest the BOS zone's outer); a pullback `CTS_CONFIRMED` gains `meta["pattern_anchor_idx"]`; a `[wvmi] WARNING` when
the lock changes the LP. = the prediction: `cmp_wvmi_keyed.py` — exactly 6 records change `lp_idx`,
`pullback_momentum` and their side's `buy_` / `sell_momentum` (`status` stays `created`): H1 sid 0 c1 896 → 901
(0.759 → 0.478); confluence sub 0 c4 1898 → 1938 (0.881 → 0.585), sub 1 c4 2455 → 2469 (0.356 → 1.019), sub 2 c1
2734 → 2828 (0.657 → 2.425), sub 4 c0 3568 → 3588 (0.817 → 1.299), sub 7 c0 4177 → 4196 (0.279 → 0.554); the
counter `_wvmi.csv` unchanged. `cmp_events_pattern_anchor.py` (new, `plans/qa_chart_edits_inputs/`): the key on H1 4
/ counter 7 / confluence 19 pullback `CTS_CONFIRMED` rows, every other cell equal. 22 / 27 CSVs byte-equal; the three
charts unchanged (`fig_named_diff.py`: 81/234, 154/136, 302/307). Skip grep 23 → 33 (the 10 WARNING lines above —
predicted 2: the prediction deduplicated tracker runs by content). `[wvmi] total=5, locked=3`; the same 3
`[uc1_trigger]` lines. S_lp_furthest `20261026_192655_f63df34` (source `f63df34`, save `e8583de`, trunk `5d055ba`;
= `LATEST_<branch>` until S_lp_rows, below; the save replay = the compared replay, 27 / 27 CSVs byte-equal).
**WVMI back on the chart, step 4: the per-LP rows + `*_wvmi_lps.csv` (2026-10-26, the same design file "Step 4") — 3
NEW CSVs, nothing else:** every LP a WVMI record has had is kept as a row (`WVMIRecord.lp_rows`; statuses `temporary` /
`locked` / `open` / `unlocked` with their why / when) and exported one row per dot — collapsed records none (the user,
round 10), an LP before its sub's `start_idx` with `relative_dir = NA (before start_idx)` (round 11). = the prediction
in every cell (`plans/qa_chart_edits_inputs/wvmi_step4_check.py` vs `wvmi_step4_predict.py`): H1 57 rows,
counter 74, confluence 185 (`temporary` 54 / 68 / 171, `locked` 1 / 3 / 7, `unlocked` 1 / 3 / 7, `open` 1 / 0 / 0);
each record's last row = its `_wvmi.csv` LP. 27 / 27 old CSVs byte-equal + the 3 new (27 → 30: against an older
save `cmp_fig_full.py` lists them `NEW (no baseline)`); the three charts unchanged (`fig_named_diff.py`); skip grep
33 (the same lines); `[wvmi] total=5, locked=3`; the same 3 `[uc1_trigger]` lines. Suite 2012 (2013 with the
session-close pin); mutants 30 / 30 (+ 2 equivalent, run, surviving).
S_lp_rows `20261026_220225_588014a` (source `588014a`, save `04d2f0a`, trunk `fef0135`; = `LATEST_<branch>` until
S_wvmi_marks, below; the save replay = the compared replay).
**WVMI back on the chart, step 5: the LP marks above the volume bars (2026-10-26, the same design file "Step 5" +
rounds 12–14; canonical CHARTING_SPEC §13) — FIGURE-ONLY:** one mark per per-LP row of every non-collapsed record on
the volume panel's own axis `yaxis3` — sell on a row just above the bars, buy one step higher; a sub's a dot (size
5), an H1 LP an open circle (size 8 — the user's final value), on the M15 charts at the hour's last M15 candle; the
hover = the accepted drafts (`TF=1H>>15M` on an H1 circle's M15 copy); the volume axis `[0, max × 1.4]` (static +
zoom JS, ×1.1 before). = the prediction: 30 / 30 CSVs byte-equal to S_lp_rows (`wvmi_step4_check.py` 0 problems);
traces ADDED only — H1 81 → 83 (`WVMI LP buy` 42 / `sell` 15), counter 154 → 158 (`M15 WVMI LP` buy 27 / sell 47,
`H1 WVMI LP` 42 / 15), confluence 302 → 306 (88 / 97, 42 / 15); shapes unchanged; the layouts gain `yaxis2.range` +
`yaxis3` only. **Positional `fig_scope.py` breaks across it** — diff figures with `fig_named_diff.py` (+ the layout
delta); `plans/qa_chart_edits_inputs/wvmi_step5_check.py BASE CUR` re-builds every mark (x / row / hover) from the
per-LP CSVs; `chart_census.py` counts the marks (`wvmi_lp_marks` per trace). A zoomed PNG that keeps the volume panel:
`zoom_png_volume.py` (`zoom_png.py` flattens it). Skip grep 33 (the same lines) + the log line `[chart][wvmi] rendered
57 LP circles`. Suite 2039; mutants 44 / 44 (+ 1 equivalent, run, surviving). S_wvmi_marks
`20261026_230619_2dc9e5d` (source `2dc9e5d`, save `915743d`, trunk `cb56121`; = `LATEST_<branch>` from 2026-10-26;
the save replay = the compared replay). "WVMI back on the chart" is complete.
**PLAN_T T1 (2026-10-27, SESSION ORDER item 5b — the first_confluence probe's Phase 2 MS reads what its sub reads) —
ZERO CHANGE:** the probe's MS gets the caller's timeframe values (M15 proximity 5 / narrow-gap 30, not the helper's H1
default 8 / 50), the pair's pip size and its sub's cross-fib limit (`cross_fib_limit.first_confluence_probe_limit`,
THE formula, called on a probe-cache miss); + the log-only probe/sub consistency tripwire above. = the prediction
(PLAN_T §2.3 / §2.6): 30 / 30 CSVs byte-equal to S_wvmi_marks, the three charts equal by trace name
(`fig_named_diff.py`), the probe's log lines identical, skip grep 33, the tripwire silent. Byte-identical BY
CONSTRUCTION while the cycle-0 carve-out blocks every cycle 0 (true at T1, before T2 narrowed it; the two values are read only by the proximity
confirmation; the probe reads nothing after cycle 0's confirmation); on the window the probe's Phase 2 MS runs once
(FC (0,0), whose limit is none = today's). Not a save-format boundary. Suite 2065; mutants 26 / 26. S_T1
`20261027_131836_453f227` (source `453f227`, save `4d7318c`, trunk `1d7a6a0`; = `LATEST_<branch>` from 2026-10-27; the
save replay = 30 / 30 CSVs byte-equal to S_wvmi_marks again).
**PLAN_T T2 (2026-10-27, the cycle-0 carve-out kept only on the main sid 0 and the sub its cycle-0 first_confluence
creates, + that trigger's probe) — BEHAVIOUR, 6 CSVs, subs 4, 7, 8 only:** = the measured prediction (PLAN_T §3.3;
30 / 30 CSVs equal to the variant `c0T2`). Vs S_T1: the two M15 `structure_events` (sub 8 on both lenses, subs 4 / 7
on confluence: the cycle-0 CTS_CONFIRMED moves to its proximity candle 4098 / 3336 / 4071 `sd_zone_proximity`, the old
pattern candle 4103 / 3489 / 4157 becomes CTS_RECONFIRMED — counter 591 → 592, confluence 1388 → 1391 events), the
two M15 `fib_lifecycle` (`meta.locked_at` on those cycle-0 fibs), the two M15 `wvmi_lps` (counter 74 → 76, confluence
185 → 216: the temporary LPs from the proximity confirmation on); H1, triggers, subs, KL / POI / IC, `_wvmi.csv`
unchanged. Charts (`fig_named_diff.py`): H1 equal; counter `M15 WVMI LP buy` (+2 marks); confluence `M15 WVMI LP buy`
(+16) + `sell` (+15); shapes + layouts equal. Skip grep 33 (both new tripwires silent: the probe/sub one and `the
cycle-0 carve-out's first_confluence … did not create its sub`). Not a save-format boundary (same columns;
positional `cmp_save` fine — diff the events keyed with `qa_cycle0_carveout_inputs/c0_evdiff.py BASE CUR`). Suite
2096; mutants 36 / 36. S_T2 `20261027_140757_017c920` (source `017c920`, save `c0153a9`, trunk `1564ccd`; =
`LATEST_<branch>` from 2026-10-27; the save = the compared replay, reused).

**M15.counter** and **M15.confluence** (9 files each — 7 per-lens debug CSVs
with the same filename suffixes prefixed `*_M15_counter_` /
`*_M15_confluence_`, plus the two per-lens pool tables):

| File pattern | Written by |
|--|--|
| `*_M15_{lens}_structure_events.csv` | chart loop (`run_replay.py`) |
| `*_M15_{lens}_kl_zones.csv` | chart loop |
| `*_M15_{lens}_poi_zones.csv` | chart loop |
| `*_M15_{lens}_ic_candidates.csv` (`_ip_` until PLAN_L landing 1) | chart loop |
| `*_M15_{lens}_fib_lifecycle.csv` | chart loop |
| `*_M15_{lens}_wvmi.csv` | chart loop |
| `*_M15_{lens}_wvmi_lps.csv` — the per-LP WVMI rows (step 4, 2026-10-26; `relative_dir` from this lens's `attrs["sids"]`) | chart loop |
| `*_M15_{lens}_subs.csv` — one row per unique sub on this lens | `debug/export_sub_tables.py`, BEFORE the chart loop |
| `*_M15_{lens}_triggers.csv` — one row per `TriggerRecord` on this lens (incl. zero-length) | `debug/export_sub_tables.py`, BEFORE the chart loop |

**Pool-wide** (1 file): `*_M15_unresolved_triggers.csv` — one row per
`UnresolvedTrigger` (`debug/export_sub_tables.py`, written once).

Total: **11 + 2×9 + 1 = 30 CSVs** per replay (the three `*_wvmi_lps.csv` added by "WVMI back on the chart" step 4,
2026-10-26 — a save before it has 27, so `cmp_fig_full.py` lists the three as `NEW (no baseline)` and exits 1;
POI-zones CSVs added 2026-06-08;
pool tables added by Plan C 2026-09-20; the three `*_ip_candidates.csv` by PLAN_H
landing A 2026-10-06, renamed `*_ic_candidates.csv` by PLAN_L landing 1. A save before PLAN_H landing A has 24, so against
one `cmp_fig_full.py` lists the three as `NEW (no baseline)` and exits 1; a save from PLAN_H landing A up to S_TBU has 27
with the old names (3 MISSING + 3 NEW — the boundary "PLAN_L landing 1" above), and `cmp_save.py`
never lists a current-only file). `*_M15_{lens}_sids.csv` is **GONE**
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
`(direction, starting_idx)` — `sub_id`s are 0–8 on the reference window since PLAN_R step 3 (0–10 at step 2, 0–7
before): start order since PLAN_I S1, so a new sub renumbers every later one) rather than a prior save; from the next `/commit-save` on
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

**The three `_wvmi.csv` files since Plan G (2026-09-30; save-format boundary: saves
before it lack the column and the ungated sub rows).** A `cycle_collapsed` column
right after `lp_locked`; one row per record per lens CSV the sub is on (a dual-lens
sub's records on BOTH, like its zones), with `structure_path_id` = that lens's path
(== `meta["structure_path_id"]`); `triggered_by_event_idx` / `_type` per lens — empty
when the lens has no WVMI-class trigger in the sub's window — and
every int column (`triggered_by_event_idx`, `locked_by_cycle_id`, the idx / id columns) written as an
int (`Int64`, never `710.0` / `1.0`; saves before the Int64 commit show `locked_by_cycle_id` as `1.0`). Diff them
KEYED, not with `cmp_save.py` (positional: a new row or a mid-row column shifts every
later cell): `engine_v2/plans/plan_e_inputs/review_scripts/cmp_wvmi_keyed.py
BASE_DIR CUR_DIR` — rows by `(sub_id, bos_structure_id, bos_cycle_id)`, columns by
name, the meta per key; it also checks every key once per CSV and the path column ==
the meta path (exit 1 otherwise).

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
variant replays must differ exactly in PLAN_E §8's cells. Tool:
`engine_v2/plans/plan_e_inputs/review_scripts/cmp_save.py BASE CUR_DEBUG CUR_CHARTS [--strip k]`
(the CSV cell diff + this figure diff in one; the E4 variant recipe is in that folder's README). It does NOT compare hover
text or `customdata`: for a hover / label change (or to prove none), run
`review_scripts/cmp_hover.py BASE CUR_CHARTS [--match TEXT]` — the changed `hovertemplate` count per chart, how many
changed outside `TEXT`, and the `customdata` count (the wave-candle label fix `a8ed1cc`: 11 / 66 / 33, 0 outside). For a **byte-identical** claim run `review_scripts/cmp_fig_full.py BASE` — every CSV by md5 and each
chart's FULL figure JSON (every trace attribute + the layout), exit 1 on any difference (§13.5.e, 2026-09-30). Its CSV
diff is POSITIONAL: on a header change (an added / renamed column) diff that CSV keyed by column name. Its TRACE diff is
positional too: one inserted trace (a new POI's hover lines) shifts every later index, so every later trace reads as
"differing" (PLAN_L L2: 233 of 282 confluence traces) — for a change that adds or removes traces, diff the traces /
shapes as MULTISETS of their full JSON, summarised by trace name (`plan_h_inputs/cmp_l2.py` §4,
`plan_h_inputs/l2_hover_diff.py` for the changed hover texts). A one-line FINGERPRINT per chart of that multiset
(every trace's full JSON, hovers included, + the layout with its shapes sorted): `plan_m_inputs/fig_hash.py CHARTS_DIR
[NAME_PREFIX]` — equal hashes = equal figures up to trace order; pass the window's prefix (e.g.
`NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_`) on `artifacts/charts`, which holds many runs (a save folder holds
one). PLAN_M persists a trial's hashes to predict a landing's figures exactly (`plan_m_inputs/trial_fig_hash.txt`).
It counts changed CELLS; for a change to meta VALUES (one meta cell can hold several changed keys) use
`review_scripts/cmp_meta_keys.py BASE CUR` — per-key counts, each delta checked to be `+slice_begin`
(Post-E·2: 547 cells = 715 keys).
For a meta-KEY rename (LANDMINES "Event Contract Rules" rule 3) use `review_scripts/cmp_meta_rename.py BASE CUR
old=new …` — every CSV: BASE with only the listed keys renamed in place == CUR, compared as raw text (Post-E·4:
150 cells = 197 keys, exit 0); `old=` (an empty new name) = the key is DELETED (Post-E·5: 8 cells = 22 keys).

## Chart Count Parity (Corroborating check)

Chart trace/shape counts tally everything rendered for an entity, so a
mismatch confirms divergence even when you can't immediately tell which
CSV moved. They are no longer the *only* sub-entity signal (the
per-entity CSVs above are primary), but they're cheap to read from
stdout and catch chart-side regressions (style registry, hover,
lifecycle gating) that the CSVs alone miss.

After the replay run, verify the three trace/shape counts in the stdout
match the baseline run's counts exactly. Per the latest run the standard
counts are (verified 2026-10-26 on S_h1_ungated `20261026_183615_76b5a17`, unchanged from S_uc1_prox
`20261026_175906_257f8ba` and S_repl_break `20261024_171443_56517e7` — the replacement break on every lens
added one PB-dot trace per M15 chart, counter 153 → 154 and confluence 301 → 302, H1 and every shape unchanged — to
S_prox_hover `20261024_105742_fb0189b`'s counts, unchanged from S_wave_hover
`20261023_194643_18711dc`, S_fib_hover `20261023_191547_563f5f8`, S_zone_hover
`20261023_153743_56bb20d`, S_dot_hover
`20261022_200747_ab2fcc1`, S_S3 `20261021_130400_7b62a47`
and S_S2 `20261021_081421_cbaed31`
— PLAN_S step 2 moved the shapes from S_S1
`20261021_020427_69c099b`'s 236 / 138 / 309: the H1 POI sid 1 cycle 2's off / on pair 992 / 998 is gone, on H1 and
both M15 overlays; S_S1's were unchanged from S_R3 `20261020_225749_d424099`
— PLAN_R step 3 moved the two M15 lenses from
S_R2's 164/144, 257/263 (2 subs fewer, sub_7 / sub_8 re-windowed; H1 unchanged); PLAN_R step 2 had moved them from
S_Q's 146/135, 289/293 (3 new subs, re-windowed records); S_Q's were unchanged from S_P
`20261013_231327_23f395d` — PLAN_P moved them from S_O's 83/239,
137/123, 282/276: H1 3 POIs gone, sub 7's cycle 0 now drawn, the Step 1A POIs; those were unchanged since PLAN_L
landing 2 (S_L2, 2026-10-10: its 7 new POIs moved them from 81/236, 135/120, 274/264) through S_M1, S_M2, S_N,
S_N_hover and S_O; config window 2025-12-01→2026-01-20, auto-extended to
2025-11-15. Window-dependent, re-baseline when the config window or chart rendering changes;
older values in **History** below):

- `H1` chart: traces=81, shapes=234
- `M15.counter` chart: traces=154, shapes=136
- `M15.confluence` chart: traces=302, shapes=307

(Before the 2026-10-09 re-baseline the block listed the 2026-09-24 Plan F values — H1
85/245, counter 151/124, confluence 294/233; landings between re-baselines are not
re-attributed here — each `/compare` checks its baseline save's counts, not this block.)

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

A run with all CSVs byte-identical (30; 27 before 2026-10-26, 24 before 2026-10-06) but shifted chart counts means a
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
