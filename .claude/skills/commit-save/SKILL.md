---
name: commit-save
description: Capture session learnings, commit changes, and save replay outputs to a timestamped folder for later comparison.
user-invocable: true
allowed-tools: Bash, Read, Write, Glob, Skill
argument-hint: [commit message]
---

# Capture Learnings, Commit, and Save Replay Outputs

`/commit-save` runs at natural checkpoints in the work — when a logical unit of work is done. Those are also the right moments to persist learnings, so this skill bundles `/remember` into the flow:

1. Capture learnings via `/remember` (project docs, LANDMINES, memory) — runs FIRST so any doc edits land in the same source commit
2. Commit current changes
3. Run the replay and save outputs to a timestamped folder in `artifacts/commits/` for later `/compare`

## Environment note (IMPORTANT)

Each `Bash` tool call is an **isolated shell invocation** — shell variables
do not persist between calls. Do NOT rely on `$COMMIT_HASH` / `$FOLDER_NAME`
being defined in later steps. Either:
- **Re-derive the value in each step** from `git` / filesystem state (preferred — robust)
- **Combine dependent steps into a single Bash call** so variables stay in scope

The instructions below are written to be re-derived per step.

## Instructions

### 1. Capture Session Learnings via `/remember`

Invoke the `remember` skill via the `Skill` tool **before staging anything**:

```
Skill(skill="remember")
```

This lets `/remember` propose candidate learnings (gotchas, landmines,
spec clarifications, memory updates), surface them to the user for
confirmation if non-obvious, and edit the appropriate files. Any files
it touches become part of the working tree and will be picked up by the
source commit in Step 2.

If `/remember` reports nothing worth saving, continue. Don't force a
learning that isn't there.

If the session was purely mechanical (e.g., a one-line rename or a
revert) and there's clearly nothing to remember, you may skip this step
— but err on the side of running it. The cost is one prompt; the
benefit is durable knowledge for future sessions.

### 2. Commit Current Changes

If `$ARGUMENTS` is provided, use it as the commit message. Otherwise, follow
the standard commit flow (check status, draft message, commit).

**Include any files `/remember` edited** in this commit (LANDMINES.md,
GOTCHAS.md, etc.). They are part of the same logical unit of work.

```bash
# Inspect and draft; then stage and commit. Prefer listing changed files
# explicitly over `git add -A` so untracked scratch files aren't included.
git add <explicit files>
git commit -m "<commit_message>

Co-Authored-By: Claude <noreply@anthropic.com>"
```

After committing, the source commit hash is `HEAD` until you make more
commits. Re-derive it when needed as `git rev-parse --short HEAD`.

### 3. Create Output Folder

Build the folder name from the current HEAD and a fresh timestamp, then
create the folder and a marker file in one call so the variables stay
in-scope:

```bash
COMMIT_HASH=$(git rev-parse --short HEAD) \
TIMESTAMP=$(date +%Y%m%d_%H%M%S); \
FOLDER_NAME="${TIMESTAMP}_${COMMIT_HASH}"; \
mkdir -p "artifacts/commits/${FOLDER_NAME}" && \
touch "artifacts/commits/${FOLDER_NAME}/.before_replay_marker" && \
echo "FOLDER_NAME=${FOLDER_NAME}"
```

The folder name is now visible on disk — subsequent steps re-derive it by
finding the newest directory in `artifacts/commits/`:

```bash
FOLDER_NAME=$(ls -t artifacts/commits/ | grep -v '^LATEST$' | head -1)
```

### 4. Run Replay

```bash
# Run replay - outputs go to standard locations (artifacts/debug, artifacts/charts)
python -m engine_v2.run_replay
```

### 5. Copy ONLY New Outputs to Commit Folder

Find the newest commit folder and copy files newer than its marker:

```bash
FOLDER_NAME=$(ls -t artifacts/commits/ | grep -v '^LATEST$' | head -1); \
MARKER="artifacts/commits/${FOLDER_NAME}/.before_replay_marker"; \
find artifacts/debug  -maxdepth 1 -name "*.csv"  -newer "$MARKER" -exec cp {} "artifacts/commits/${FOLDER_NAME}/" \; ; \
find artifacts/charts -maxdepth 1 -name "*.html" -newer "$MARKER" -exec cp {} "artifacts/commits/${FOLDER_NAME}/" \; ; \
find artifacts/charts -maxdepth 1 -name "*.png"  -newer "$MARKER" -exec cp {} "artifacts/commits/${FOLDER_NAME}/" \; 2>/dev/null || true; \
rm "$MARKER"; \
ls "artifacts/commits/${FOLDER_NAME}/"
```

### 6. Write Metadata + Update LATEST

```bash
FOLDER_NAME=$(ls -t artifacts/commits/ | grep -v '^LATEST$' | head -1); \
COMMIT_HASH=$(echo "$FOLDER_NAME" | awk -F_ '{print $NF}'); \
TIMESTAMP=$(echo "$FOLDER_NAME" | awk -F_ '{print $1"_"$2}'); \
printf "commit_hash=%s\ntimestamp=%s\ncommit_message=<message>\n" \
  "$COMMIT_HASH" "$TIMESTAMP" \
  > "artifacts/commits/${FOLDER_NAME}/metadata.txt"; \
echo "${FOLDER_NAME}" > artifacts/commits/LATEST
```

The folder name is `YYYYMMDD_HHMMSS_<hash>` — the commit hash is the last
underscore-delimited segment, the timestamp is the first two.

### 7. Commit the Saved Outputs

Always re-derive `COMMIT_HASH` here — if you used a stale variable from an
earlier step, or `HEAD~N`, you may reference the wrong commit. At this
point HEAD is still the source commit (we haven't committed the outputs
yet), so `git rev-parse --short HEAD` gives the correct hash.

```bash
FOLDER_NAME=$(ls -t artifacts/commits/ | grep -v '^LATEST$' | head -1); \
COMMIT_HASH=$(git rev-parse --short HEAD); \
git add "artifacts/commits/${FOLDER_NAME}/" artifacts/commits/LATEST && \
git commit -m "Save replay outputs for commit ${COMMIT_HASH}

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### 8. Report Success

Output a summary:

```
=== COMMIT-SAVE COMPLETE ===

Learnings captured: <one-line summary from /remember, or "none">
Source commit: <hash> - <message>
Save-outputs commit: <hash>
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
