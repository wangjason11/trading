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
3. Run the replay and save outputs to a branch-namespaced timestamped folder in `artifacts/commits/<branch>/` for later `/compare`
4. Cherry-pick the save commit onto `artifacts-trunk` so artifacts remain visible across branch checkouts

## Folder layout (since 2026-05-23)

```
artifacts/commits/
├── LATEST_<branch>                     # per-branch pointer to most recent save folder
├── <branch>/
│   ├── 20260523_182020_abc1234/
│   │   ├── metadata.txt
│   │   ├── NZD_USD_H1_..._final.csv
│   │   └── ...
│   └── 20260524_091200_def5678/
└── <other-branch>/
    └── ...
```

**Why branch-namespaced:** prevents cross-branch collision and makes the
source branch explicit in the path. Combined with the `artifacts-trunk`
cherry-pick step below, ensures saves remain accessible regardless of
which branch is currently checked out. See `/compare` skill for how
baselines are resolved.

**Legacy flat folders** (`artifacts/commits/<ts>_<hash>/` without a
branch prefix) remain in place — `/compare` falls back to the flat path
and the global `LATEST` file if no per-branch pointer is found.

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

**Guard:** if the current branch is `artifacts-trunk`, abort with an
error. That branch only receives cherry-picked save commits; no work
should happen on it directly.

### 3. Create Output Folder (branch-namespaced)

Build the folder under the current branch's namespace from the source
commit hash + a fresh timestamp, then create the folder and a marker
file in one call so the variables stay in-scope:

```bash
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD) \
COMMIT_HASH=$(git rev-parse --short HEAD) \
TIMESTAMP=$(date +%Y%m%d_%H%M%S); \
FOLDER_NAME="${TIMESTAMP}_${COMMIT_HASH}"; \
FOLDER_PATH="artifacts/commits/${CURRENT_BRANCH}/${FOLDER_NAME}"; \
mkdir -p "${FOLDER_PATH}" && \
touch "${FOLDER_PATH}/.before_replay_marker" && \
echo "FOLDER_PATH=${FOLDER_PATH}"
```

Subsequent steps re-derive the folder path by reading the current branch
and finding the newest subdirectory:

```bash
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
FOLDER_NAME=$(ls -t "artifacts/commits/${CURRENT_BRANCH}/" | head -1)
FOLDER_PATH="artifacts/commits/${CURRENT_BRANCH}/${FOLDER_NAME}"
```

### 4. Run Replay

```bash
# Run replay - outputs go to standard locations (artifacts/debug, artifacts/charts)
python -m engine_v2.run_replay
```

### 5. Copy ONLY New Outputs to Commit Folder

Find the newest commit folder for the current branch and copy files
newer than its marker:

```bash
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD); \
FOLDER_NAME=$(ls -t "artifacts/commits/${CURRENT_BRANCH}/" | head -1); \
FOLDER_PATH="artifacts/commits/${CURRENT_BRANCH}/${FOLDER_NAME}"; \
MARKER="${FOLDER_PATH}/.before_replay_marker"; \
find artifacts/debug  -maxdepth 1 -name "*.csv"  -newer "$MARKER" -exec cp {} "${FOLDER_PATH}/" \; ; \
find artifacts/charts -maxdepth 1 -name "*.html" -newer "$MARKER" -exec cp {} "${FOLDER_PATH}/" \; ; \
find artifacts/charts -maxdepth 1 -name "*.png"  -newer "$MARKER" -exec cp {} "${FOLDER_PATH}/" \; 2>/dev/null || true; \
rm "$MARKER"; \
ls "${FOLDER_PATH}/"
```

### 6. Write Metadata + Update Per-Branch LATEST Pointer

```bash
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD); \
FOLDER_NAME=$(ls -t "artifacts/commits/${CURRENT_BRANCH}/" | head -1); \
FOLDER_PATH="artifacts/commits/${CURRENT_BRANCH}/${FOLDER_NAME}"; \
COMMIT_HASH=$(echo "$FOLDER_NAME" | awk -F_ '{print $NF}'); \
TIMESTAMP=$(echo "$FOLDER_NAME" | awk -F_ '{print $1"_"$2}'); \
printf "commit_hash=%s\ntimestamp=%s\nbranch=%s\ncommit_message=<message>\n" \
  "$COMMIT_HASH" "$TIMESTAMP" "$CURRENT_BRANCH" \
  > "${FOLDER_PATH}/metadata.txt"; \
echo "${FOLDER_NAME}" > "artifacts/commits/LATEST_${CURRENT_BRANCH}"
```

The folder name is `YYYYMMDD_HHMMSS_<hash>` — the commit hash is the last
underscore-delimited segment, the timestamp is the first two. The
per-branch `LATEST_<branch>` pointer holds just the folder name (not the
full path), since the path's branch prefix is implicit in the pointer
filename.

### 7. Commit the Saved Outputs

Always re-derive `COMMIT_HASH` here — if you used a stale variable from an
earlier step, or `HEAD~N`, you may reference the wrong commit. At this
point HEAD is still the source commit (we haven't committed the outputs
yet), so `git rev-parse --short HEAD` gives the correct hash.

```bash
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD); \
FOLDER_NAME=$(ls -t "artifacts/commits/${CURRENT_BRANCH}/" | head -1); \
FOLDER_PATH="artifacts/commits/${CURRENT_BRANCH}/${FOLDER_NAME}"; \
COMMIT_HASH=$(git rev-parse --short HEAD); \
git add "${FOLDER_PATH}/" "artifacts/commits/LATEST_${CURRENT_BRANCH}" && \
git commit -m "Save replay outputs for commit ${COMMIT_HASH}

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### 7b. Cherry-Pick Save Commit Onto `artifacts-trunk`

This step is what guarantees the save survives a checkout to any other
branch. The save commit touches only files under
`artifacts/commits/<branch>/...` and `artifacts/commits/LATEST_<branch>`,
both of which are unique per branch — so the cherry-pick is conflict-free.

```bash
ORIG_BRANCH=$(git rev-parse --abbrev-ref HEAD); \
SAVE_COMMIT=$(git rev-parse HEAD); \
if git show-ref --verify --quiet refs/heads/artifacts-trunk; then \
    git checkout artifacts-trunk && \
    git cherry-pick "${SAVE_COMMIT}" && \
    git checkout "${ORIG_BRANCH}" && \
    echo "Cherry-picked ${SAVE_COMMIT:0:7} onto artifacts-trunk"; \
else \
    echo "WARN: artifacts-trunk branch not found locally; skipping cherry-pick."; \
    echo "      To enable cross-branch retention, run: git branch artifacts-trunk"; \
fi
```

**If the cherry-pick fails** (rare — typically only if you manually
edited LATEST_<branch> on artifacts-trunk to a value that conflicts):
resolve manually, finish the cherry-pick, then `git checkout
${ORIG_BRANCH}`. The save still exists on the source branch regardless;
the cherry-pick is a safety net.

**Do NOT do work on `artifacts-trunk`.** It only receives cherry-picks
from this skill. Its code state is whatever it was when the branch was
created (intentionally stale — it's not meant for development).

### 8. Report Success

Output a summary:

```
=== COMMIT-SAVE COMPLETE ===

Learnings captured: <one-line summary from /remember, or "none">
Source commit: <hash> - <message>
Save-outputs commit: <hash>
Outputs saved to: artifacts/commits/<branch>/<folder_name>/
Cherry-picked onto: artifacts-trunk (or "skipped — branch missing")

Contents:
- CSV files: <count>
- Charts: <count>
- Metadata: metadata.txt

LATEST_<branch> pointer updated.

Next steps:
- Make your changes
- Run /compare before next commit to check for regressions
```

## Why This Matters

This creates a checkpoint of replay outputs that `/compare` can use to detect:
- Regression bugs (prior logic broken)
- Unintended side effects (cascading changes)
- Shifted events (BOS/CTS moved to different indices)

The branch-namespaced layout + `artifacts-trunk` cherry-pick ensures
that:
- Saves from hybrid/debug branches don't get lost when checking back
  out to the main work branch (the cross-branch visibility bug that
  surfaced 2026-05-23 with `sub-debug-c2-baseline`)
- `/compare` baselines are unambiguously scoped to the current branch
- A single branch (`artifacts-trunk`) holds the union of all saves for
  reference / debugging — checkout once to inspect any historical save

Always run `/commit-save` when you've completed a logical unit of work and are ready to checkpoint.
