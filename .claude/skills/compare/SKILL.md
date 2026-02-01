---
name: compare
description: Compare replay output between current code and previous commit to detect unintended changes.
user-invocable: true
allowed-tools: Bash, Read, Glob, Grep, Write
argument-hint:
---

# Compare Replay Output Against Previous Commit

Compare the current code's replay output against the previous commit to ensure changes don't unintentionally alter prior logic.

## Instructions

1. **Save current state:**
   - Note the current commit hash
   - Stash any uncommitted changes if present

2. **Run replay on previous commit:**
   - Checkout the previous commit: `git checkout HEAD~1`
   - Run replay: `python -m engine_v2.run_replay`
   - Save key metrics to a temp file (use scratchpad directory)
   - Capture: structure events, zone counts, candle patterns, Fib states, etc.

3. **Run replay on current code:**
   - Checkout back to original branch/commit: `git checkout -`
   - Pop any stashed changes if applicable
   - Run replay: `python -m engine_v2.run_replay`
   - Save key metrics to a temp file

4. **Compare and summarize:**
   - Load both result sets
   - Identify what stayed the same vs what changed
   - Focus on:
     - **Candle patterns**: pinbar, maru, engulfing, star counts
     - **Structure events**: CTS/BOS counts, state transitions
     - **Zones**: KL zone count, POI zone count, bounds
     - **Fib states**: activation/lock counts
     - **Imbalance**: count of imbalance candles
   - Flag any unexpected changes (changes in areas not touched by recent work)

5. **Report findings:**
   - Summarize what's consistent (good)
   - Highlight what changed and whether it's expected
   - If unexpected changes found, recommend investigation before commit

## Key Metrics to Capture

```python
# From df and events, capture:
metrics = {
    "total_candles": len(df),
    "structure_events": len([e for e in events if e.type in ["CTS_CONFIRMED", "BOS_CONFIRMED"]]),
    "state_changes": len([e for e in events if e.type == "STATE_CHANGED"]),
    "kl_zones": len(df.attrs.get("kl_zones", [])),
    "poi_zones": len(df.attrs.get("poi_zones", [])),
    "imbalance_candles": df["is_imbalance"].sum() if "is_imbalance" in df.columns else 0,
    "pinbar_count": (df["candle_type"] == "pinbar").sum(),
    "maru_count": (df["candle_type"] == "maru").sum(),
    "fib_states": len(fib_tracker.get_fibs_for_charting()) if fib_tracker else 0,
}
```

## Output Format

```
=== COMPARE: Previous Commit vs Current ===

Previous: <commit_hash> (<commit_message>)
Current:  <commit_hash> (<commit_message>)

UNCHANGED (as expected):
- Total candles: 500
- Pinbar count: 45
- Maru count: 32
- ...

CHANGED:
- POI zones: 2 → 3 (expected: new POI zone logic)
- Fib states: 4 → 5 (expected: cross-cycle Fib added)

UNEXPECTED CHANGES:
- [None found] OR
- KL zones: 5 → 4 (INVESTIGATE: KL logic not touched)

Recommendation: [Safe to commit] OR [Investigate before commit]
```

## Why This Matters

This comparison catches:
- Regression bugs (prior logic broken by new changes)
- Unintended side effects (changing one thing affects another)
- Missing test coverage (changes that slip through)

Run `/compare` before every commit and merge to maintain code integrity.
