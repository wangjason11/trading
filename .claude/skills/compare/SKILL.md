---
name: compare
description: Compare current replay output against previous /commit-save to detect unintended changes.
user-invocable: true
allowed-tools: Bash, Read, Glob, Grep, Write
argument-hint:
---

# Compare Replay Output Against Previous Commit-Save

Compare the current code's replay output against the most recent `/commit-save` output to ensure changes don't unintentionally alter prior logic.

## Instructions

### 1. Find Previous Commit-Save Output

```bash
# Read the LATEST pointer
PREV_FOLDER=$(cat artifacts/commits/LATEST)
PREV_PATH="artifacts/commits/${PREV_FOLDER}"

# Verify it exists
if [ ! -d "${PREV_PATH}" ]; then
    echo "ERROR: No previous commit-save found. Run /commit-save first."
    exit 1
fi
```

### 2. Run Replay on Current Code

```bash
python -m engine_v2.run_replay
```

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

Previous commit-save: 20260202_143000_abc1234
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

RECOMMENDATION: ✅ Safe to commit
```

OR if unexpected changes:

```
UNEXPECTED CHANGES:
- POI zones decreased from 4 to 2 (POI logic not touched in this iteration)

RECOMMENDATION: ⚠️ Investigate before commit
  - Check poi_zones.py for unintended changes
  - Verify Fib state handling
```

### 6. Cascading Change Detection

When a change is detected, trace its downstream effects:

```
CASCADE ANALYSIS:

Root cause: candle_type changed at idx=707 (normal → maru)
  ↓
Effect 1: continuous pattern at 707-709 no longer matches Pattern3
  ↓
Effect 2: Reversal watch at idx=707 fails (no valid pattern)
  ↓
Effect 3: Reversal shifts from idx=710 to idx=748
  ↓
Effect 4: sid=1 starts later, all cycle timing shifts
  ↓
Effect 5: BOS for sid=1 cycle=1 moves from idx=728 to idx=826

All changes are causally linked to the root change.
```

## Key Files to Compare

| Current Location | Previous Location |
|------------------|-------------------|
| `artifacts/debug/*_final.csv` | `artifacts/commits/<folder>/*_final.csv` |
| `artifacts/debug/*_raw.csv` | `artifacts/commits/<folder>/*_raw.csv` |
| `artifacts/debug/*_structure_levels.csv` | `artifacts/commits/<folder>/*_structure_levels.csv` |

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
4. If unexpected → investigate
5. If expected → /commit-save  # Create new checkpoint
```
