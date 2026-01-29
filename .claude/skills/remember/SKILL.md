---
name: remember
description: Save learnings, gotchas, or insights to CLAUDE.md so they persist across sessions. Use when the user wants to remember something for the future.
user-invocable: true
allowed-tools: Read, Edit
argument-hint: [topic or lesson to remember]
---

# Save Learning to Memory

The user wants to save a learning, gotcha, or insight to the project's memory file (CLAUDE.md) so it persists across sessions.

## Instructions

1. **Read the current CLAUDE.md file** at `engine_v2/CLAUDE.md`

2. **Identify what to remember:**
   - If the user provided `$ARGUMENTS`, use that as the topic
   - If no arguments, ask the user what they want to remember OR summarize the key learnings from the recent conversation

3. **Format the learning** with:
   - A clear, descriptive heading
   - The problem or context
   - The solution or insight
   - Any code snippets if relevant (keep brief)

4. **Add to the "Lessons Learned (Known Gotchas)" section** in CLAUDE.md
   - If this section doesn't exist, create it before the end of the file
   - Keep entries concise but complete enough to be useful

5. **Confirm** what was added to the user

## Example Entry Format

```markdown
### Brief Descriptive Title

**Problem:** What went wrong or what was confusing

**Solution/Insight:** The fix or key understanding

**Why it matters:** Brief explanation of impact
```
