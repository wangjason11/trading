---
name: remember
description: Save learnings, gotchas, or insights to appropriate documentation files and update skills so they persist across sessions.
user-invocable: true
allowed-tools: Read, Edit, Glob, Grep
argument-hint: [topic or lesson to remember]
---

# Save Learning to Documentation & Skills

Let's update our documentation, memory, and skills where appropriate so we are more knowledgeable & efficient in the future, and don't make the same errors/bugs. Ensure there are no unnecessary duplications with what we've already documented. Ensure there are no contradictions or inconsistencies. Ensure if information is outdated or contradicts with new information, we remove it or update it. If you ever have any doubts about whether you should add / change / remove something, always clarify with me.

## Continuous mode (the DEFAULT since 2026-09-22)

Learnings are captured **throughout** a session, not in one sweep at the end — an
end-of-session pass compresses hours of discussion from memory and flattens the
early, most carefully-reasoned exchanges. Standing rule + trigger list + routing
table: `memory/feedback_continuous_documentation.md` (loaded every session via
`MEMORY.md`). This skill is the procedure.

**Two tempos — do not conflate them:**

| | When | What |
|---|---|---|
| **Capture** | the moment it is said / measured | verbatim quote or measured fact + why it matters. Destination obvious and the entry self-contained → write it into the target doc now. Still unresolved → ONE line into `memory/_INBOX.md` |
| **File + reconcile** | every checkpoint: after each landed change, at each `/compare` chart-review pause, at `/commit-save` | drain the inbox, route each entry (table below), then run the reconcile checklist over everything written since the last checkpoint |

Batching the reconcile is *better* than doing it per item: related entries checked
in one pass see each other; separate passes cannot, and that is how duplicates are
born. Capture, however, is never batched.

**Also sweep retrospectively at each checkpoint:** *what was said since the last
checkpoint that a future session would need and cannot re-derive?* In-the-moment
vigilance misses things; this catch-up is what makes the rule hold.

**Reconcile checklist (every checkpoint, not just at the end):**

1. Entry already exists? → **edit in place**; never append a second, contradicting one.
2. New fact contradicts something older? → update/delete the old text in the same edit.
3. Same fact now in two files? → keep the canonical home, replace the other with a pointer.
4. Rule changed? → code comment + spec + style registry / skill values + memory move in the SAME commit.
5. Is `MEMORY.md` true END TO END — status line, "next session priorities", topic sections? Stale spots
   are never only where you just edited (2026-09-22: the ACTIVE THREAD was updated while four other
   sections still described the superseded rule).

**Invoked explicitly (`/remember`) or at `/commit-save` step 1:** do the same
thing, but scoped to everything not yet filed — the inbox plus the retrospective
sweep over the whole session.

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

6. **Update session memory (`MEMORY.md`) if applicable:**
   
   Session memory lives at: `C:\Users\wangj\.claude\projects\C--Users-wangj-OneDrive-Documents-codingproj-Project-Retire-forex-engine-v2\memory\`
   
   `MEMORY.md` is loaded into every new conversation — it's how we resume efficiently across sessions. Update it when the learning involves **session-level context that isn't derivable from code or project docs:**
   
   **When to update MEMORY.md:**
   - Week/part status changes (started, done, blocked)
   - New architectural decisions and their rationale (the "why" behind choices)
   - Completed features (add to the completed list)
   - New key files added to the project
   - Setup or workflow changes
   - Non-obvious constraints discovered during debugging
   - Multi-TF or cross-module integration points
   
   **When NOT to update MEMORY.md (already covered by project docs):**
   - Detailed spec behavior (belongs in SPEC.md files)
   - Debugging recipes or code patterns (belongs in GOTCHAS.md)
   - Critical constraints with code examples (belongs in LANDMINES.md)
   - Terminology definitions (belongs in GLOSSARY.md)
   
   **How to update:**
   - Read the current `MEMORY.md` first
   - Update existing sections in-place (don't append duplicates)
   - If a section grows too long, condense older entries
   - For new standalone memory files, follow the auto-memory two-step process:
     1. Write the memory file with frontmatter (name, description, type)
     2. Add a one-line pointer in `MEMORY.md`
   - Keep `MEMORY.md` under 200 lines (after that, content gets truncated in context)
   
   **Guiding principle:** A new session reading only `MEMORY.md` should know: what week/part we're on, what's done, what's in progress, and any non-obvious context needed to continue.

7. **Review and update skills if applicable:**
   - Read each skill file in `.claude/skills/*/SKILL.md`
   - Based on the session's learnings and discussions, check if any skill's instructions can be improved, clarified, or extended
   - Examples of skill improvements:
     - A workflow step that was missing or unclear
     - A new edge case the skill should handle
     - Updated file paths or command patterns
     - Better defaults or instructions based on what we learned
   - Only update skills when there's a clear improvement — don't change skills for unrelated learnings
   - If unsure whether a skill change is warranted, ASK the user first

8. **Confirm** what was added/updated/removed and where (or explain why nothing was changed if already covered)

## Common Patterns

| Learning Type | Target File |
|--------------|-------------|
| "X doesn't work because Y" | GOTCHAS.md |
| "Never do X" / "Always do Y first" | LANDMINES.md |
| "Term X means Y" | GLOSSARY.md |
| "Feature X works by doing Y" | Relevant SPEC.md |
| "The pattern for X is Y" | ARCHITECTURE.md or relevant SPEC.md |
| "Skill X should also do Y" | `.claude/skills/X/SKILL.md` |
| "We finished X" / "Week N Part M done" | `MEMORY.md` (status update) |
| "We decided to do X because Y" | `MEMORY.md` (architectural decision) |
| "New module X added for Y" | `MEMORY.md` (key files update) |

## Available Skills

| Skill | File | Purpose |
|-------|------|---------|
| `commit-save` | `.claude/skills/commit-save/SKILL.md` | Commit + replay + save outputs for comparison |
| `compare` | `.claude/skills/compare/SKILL.md` | Compare current replay against last commit-save |
| `prepare` | `.claude/skills/prepare/SKILL.md` | Review memory/docs/codebase before new feature specs |
| `remember` | `.claude/skills/remember/SKILL.md` | This skill — save learnings to docs & skills |
