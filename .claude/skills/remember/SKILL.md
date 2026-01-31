---
name: remember
description: Save learnings, gotchas, or insights to appropriate documentation files so they persist across sessions.
user-invocable: true
allowed-tools: Read, Edit, Glob, Grep
argument-hint: [topic or lesson to remember]
---

# Save Learning to Documentation

Let's add the learning to our documentation & memory where appropriate so we don't forget in the future. Ensure there are no unnecessary duplications with what we've already documented.

## Instructions

1. **Identify what to remember:**
   - If the user provided `$ARGUMENTS`, use that as the topic
   - If no arguments, summarize the key learnings from the recent conversation

2. **Determine the appropriate documentation file(s):**
   - `engine_v2/GOTCHAS.md` - Debugging lessons, common mistakes, non-obvious behaviors
   - `engine_v2/LANDMINES.md` - Critical constraints, things that will break if violated
   - `engine_v2/GLOSSARY.md` - Domain terminology definitions
   - `engine_v2/MARKET_STRUCTURE_SPEC.md` - CTS/BOS/Range/Reversal semantics
   - `engine_v2/zones/KL_ZONES_SPEC.md` - Zone construction and behavior
   - `engine_v2/zones/POI_ZONES_SPEC.md` - POI/Fib zone specification
   - `engine_v2/charting/CHARTING_SPEC.md` - Chart overlay rules, style registry
   - `engine_v2/ARCHITECTURE.md` - System design, event contracts
   - `engine_v2/PROJECT_PRINCIPLES.md` - Non-negotiable guardrails
   - `engine_v2/WORKFLOWS.md` - Development and debugging workflows

3. **Check for duplications:**
   - Read the target documentation file(s)
   - Compare new learnings against existing entries
   - Only add if it provides new insight not already covered
   - If related to an existing entry, consider extending that entry instead

4. **Format appropriately** for the target file:
   - Match the existing style and structure of the document
   - Keep entries concise but complete enough to be useful
   - Include code snippets if relevant (keep brief)

5. **Add the learning** to the appropriate section(s)

6. **Confirm** what was added and where (or explain why nothing was added if duplicated)

## Common Patterns

| Learning Type | Target File |
|--------------|-------------|
| "X doesn't work because Y" | GOTCHAS.md |
| "Never do X" / "Always do Y first" | LANDMINES.md |
| "Term X means Y" | GLOSSARY.md |
| "Feature X works by doing Y" | Relevant SPEC.md |
| "The pattern for X is Y" | ARCHITECTURE.md or relevant SPEC.md |
