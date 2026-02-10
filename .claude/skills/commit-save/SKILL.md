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

### 4. Run Replay (with timestamp marker)

Create a timestamp marker BEFORE running replay, then run replay:

```bash
# Create marker file to track "before replay" time
touch artifacts/commits/${FOLDER_NAME}/.before_replay_marker

# Run replay - outputs go to standard locations
python -m engine_v2.run_replay
```

### 5. Copy ONLY New Outputs to Commit Folder

Copy only files that were created/modified AFTER the marker (i.e., during this replay run):

```bash
# Copy only NEW debug CSVs (modified after marker)
find artifacts/debug -maxdepth 1 -name "*.csv" -newer artifacts/commits/${FOLDER_NAME}/.before_replay_marker -exec cp {} artifacts/commits/${FOLDER_NAME}/ \;

# Copy only NEW charts (modified after marker)
find artifacts/charts -maxdepth 1 -name "*.html" -newer artifacts/commits/${FOLDER_NAME}/.before_replay_marker -exec cp {} artifacts/commits/${FOLDER_NAME}/ \;
find artifacts/charts -maxdepth 1 -name "*.png" -newer artifacts/commits/${FOLDER_NAME}/.before_replay_marker -exec cp {} artifacts/commits/${FOLDER_NAME}/ \; 2>/dev/null || true

# Remove the marker file (no longer needed)
rm artifacts/commits/${FOLDER_NAME}/.before_replay_marker

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
    │   ├── NZD_USD_H1_..._raw.csv       # Only files from THIS run
    │   ├── NZD_USD_H1_..._final.csv
    │   ├── NZD_USD_H1_..._kl_zones.csv
    │   ├── NZD_USD_H1_..._structure_levels.csv
    │   ├── NZD_USD_H1_....html
    │   └── NZD_USD_H1_....png
    └── 20260203_091500_def5678/
        └── ...
```

**Note:** Only files created/modified during the replay run are saved (not historical files from previous runs).

## Why This Matters

This creates a checkpoint of replay outputs that `/compare` can use to detect:
- Regression bugs (prior logic broken)
- Unintended side effects (cascading changes)
- Shifted events (BOS/CTS moved to different indices)

Always run `/commit-save` when you've completed a logical unit of work and are ready to checkpoint.
