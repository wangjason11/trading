---
name: commit-save
description: Commit changes and save replay outputs to a timestamped folder for later comparison.
user-invocable: true
allowed-tools: Bash, Read, Write, Glob
argument-hint: [commit message]
---

# Commit and Save Replay Outputs

Commit the current changes, run replay, and save the outputs to a timestamped folder in `artifacts/commits/` for later comparison with `/compare`.

## Instructions

### 1. Commit Current Changes

If `$ARGUMENTS` is provided, use it as the commit message. Otherwise, follow the standard commit flow (check status, draft message, commit).

```bash
# Stage and commit
git add -A
git commit -m "<commit_message>

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### 2. Get Commit Info

```bash
# Get short hash and timestamp
COMMIT_HASH=$(git rev-parse --short HEAD)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FOLDER_NAME="${TIMESTAMP}_${COMMIT_HASH}"
```

### 3. Create Output Folder

```bash
mkdir -p artifacts/commits/${FOLDER_NAME}
```

### 4. Run Replay

Run replay normally - outputs go to standard locations:

```bash
python -m engine_v2.run_replay
```

### 5. Copy Outputs to Commit Folder

Copy all relevant outputs to the new folder:

```bash
# Copy debug CSVs
cp artifacts/debug/*.csv artifacts/commits/${FOLDER_NAME}/

# Copy charts
cp artifacts/charts/*.html artifacts/commits/${FOLDER_NAME}/
cp artifacts/charts/*.png artifacts/commits/${FOLDER_NAME}/ 2>/dev/null || true

# Save commit metadata
echo "commit_hash=${COMMIT_HASH}" > artifacts/commits/${FOLDER_NAME}/metadata.txt
echo "timestamp=${TIMESTAMP}" >> artifacts/commits/${FOLDER_NAME}/metadata.txt
echo "commit_message=<message>" >> artifacts/commits/${FOLDER_NAME}/metadata.txt
```

### 6. Update LATEST Pointer

Create/update the LATEST file to point to this folder:

```bash
echo "${FOLDER_NAME}" > artifacts/commits/LATEST
```

### 7. Commit the Saved Outputs

```bash
git add artifacts/commits/${FOLDER_NAME}/ artifacts/commits/LATEST
git commit -m "Save replay outputs for commit ${COMMIT_HASH}

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### 8. Report Success

Output a summary:

```
=== COMMIT-SAVE COMPLETE ===

Commit: <hash> - <message>
Outputs saved to: artifacts/commits/<folder_name>/

Contents:
- CSV files: <count>
- Charts: <count>
- Metadata: metadata.txt

LATEST pointer updated.

Next steps:
- Make your changes
- Run /compare before next commit to check for regressions
```

## Folder Structure

```
artifacts/
└── commits/
    ├── LATEST                           # Contains name of most recent folder
    ├── 20260202_143000_abc1234/
    │   ├── metadata.txt                 # Commit hash, timestamp, message
    │   ├── NZD_USD_H1_..._raw.csv
    │   ├── NZD_USD_H1_..._final.csv
    │   ├── swings.csv
    │   ├── structure_levels.csv
    │   └── NZD_USD_H1_..._sd-1_....html
    └── 20260203_091500_def5678/
        └── ...
```

## Why This Matters

This creates a checkpoint of replay outputs that `/compare` can use to detect:
- Regression bugs (prior logic broken)
- Unintended side effects (cascading changes)
- Shifted events (BOS/CTS moved to different indices)

Always run `/commit-save` when you've completed a logical unit of work and are ready to checkpoint.
