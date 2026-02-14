---
name: remember
description: Save learnings, gotchas, or insights to appropriate documentation files and update skills so they persist across sessions.
user-invocable: true
allowed-tools: Read, Edit, Glob, Grep
argument-hint: [topic or lesson to remember]
---

# Save Learning to Documentation & Skills

Let's update our documentation, memory, and skills where appropriate so we are more knowledgeable & efficient in the future, and don't make the same errors/bugs. Ensure there are no unnecessary duplications with what we've already documented. Ensure there are no contradictions or inconsistencies. Ensure if information is outdated or contradicts with new information, we remove it or update it. If you ever have any doubts about whether you should add / change / remove something, always clarify with me.

## Instructions

1. **Identify what to remember:**
   - If the user provided `$ARGUMENTS`, use that as the topic
   - If no arguments, summarize the key learnings from the recent conversation

2. **Determine the appropriate documentation file(s):**
   - `engine_v2/GOTCHAS.md` - Debugging lessons, common mistakes, non-obvious behaviors
   - `engine_v2/LANDMINES.md` - Critical constraints, things that will break if violated
   - `engine_v2/GLOSSARY.md` - Domain terminology definitions
   - `engine_v2/structure/MARKET_STRUCTURE_SPEC.md` - CTS/BOS/Range/Reversal semantics
   - `engine_v2/zones/KL_ZONES_SPEC.md` - Zone construction and behavior
   - `engine_v2/zones/POI_ZONES_SPEC.md` - POI/Fib zone specification
   - `engine_v2/zones/WAVE_CANDLES_SPEC.md` - Wave candle identification algorithm
   - `engine_v2/zones/WVMI_SPEC.md` - Wave Volume Momentum Indicator lifecycle + formulas
   - `engine_v2/charting/CHARTING_SPEC.md` - Chart overlay rules, style registry
   - `engine_v2/ARCHITECTURE.md` - System design, event contracts
   - `engine_v2/PROJECT_PRINCIPLES.md` - Non-negotiable guardrails
   - `engine_v2/WORKFLOWS.md` - Development and debugging workflows

3. **Check for duplications, contradictions, and outdated info:**
   - Read the target documentation file(s)
   - Compare new learnings against existing entries
   - Only add if it provides new insight not already covered
   - If related to an existing entry, consider extending that entry instead
   - If new info contradicts existing documentation, UPDATE or REMOVE the outdated content
   - If unsure whether to add/change/remove something, ASK the user first

4. **Format appropriately** for the target file:
   - Match the existing style and structure of the document
   - Keep entries concise but complete enough to be useful
   - Include code snippets if relevant (keep brief)

5. **Add/update/remove the learning** in the appropriate section(s)

6. **Review and update skills if applicable:**
   - Read each skill file in `.claude/skills/*/SKILL.md`
   - Based on the session's learnings and discussions, check if any skill's instructions can be improved, clarified, or extended
   - Examples of skill improvements:
     - A workflow step that was missing or unclear
     - A new edge case the skill should handle
     - Updated file paths or command patterns
     - Better defaults or instructions based on what we learned
   - Only update skills when there's a clear improvement — don't change skills for unrelated learnings
   - If unsure whether a skill change is warranted, ASK the user first

7. **Confirm** what was added/updated/removed and where (or explain why nothing was changed if already covered)

## Common Patterns

| Learning Type | Target File |
|--------------|-------------|
| "X doesn't work because Y" | GOTCHAS.md |
| "Never do X" / "Always do Y first" | LANDMINES.md |
| "Term X means Y" | GLOSSARY.md |
| "Feature X works by doing Y" | Relevant SPEC.md |
| "The pattern for X is Y" | ARCHITECTURE.md or relevant SPEC.md |
| "Skill X should also do Y" | `.claude/skills/X/SKILL.md` |

## Available Skills

| Skill | File | Purpose |
|-------|------|---------|
| `commit-save` | `.claude/skills/commit-save/SKILL.md` | Commit + replay + save outputs for comparison |
| `compare` | `.claude/skills/compare/SKILL.md` | Compare current replay against last commit-save |
| `prepare` | `.claude/skills/prepare/SKILL.md` | Review memory/docs/codebase before new feature specs |
| `remember` | `.claude/skills/remember/SKILL.md` | This skill — save learnings to docs & skills |
