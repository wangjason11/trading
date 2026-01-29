---
name: remember
description: Save learnings, gotchas, or insights to CLAUDE.md so they persist across sessions. Focus on debugging and logic elements, not chart formatting.
user-invocable: true
allowed-tools: Read, Edit
argument-hint: [topic or lesson to remember]
---

# Save Learning to Memory

Add the learning to our memory so we don't forget in the future. Focus more on debugging and logic elements instead of chart formatting. Ensure there are no unnecessary duplications with what's already in memory.

## Instructions

1. **Read the current CLAUDE.md file** at `engine_v2/CLAUDE.md`
   - Review the existing "Lessons Learned (Known Gotchas)" section
   - Note what's already documented to avoid duplications

2. **Identify what to remember:**
   - If the user provided `$ARGUMENTS`, use that as the topic
   - If no arguments, summarize the key **debugging and logic** learnings from the recent conversation
   - Skip chart formatting or styling topics

3. **Check for duplications:**
   - Compare new learnings against existing entries
   - Only add if it provides new insight not already covered
   - If related to an existing entry, consider extending that entry instead of creating a new one

4. **Format the learning** with:
   - A clear, descriptive heading
   - The problem or context
   - The solution or insight
   - Any code snippets if relevant (keep brief)

5. **Add to the "Lessons Learned (Known Gotchas)" section** in CLAUDE.md
   - If this section doesn't exist, create it before the end of the file
   - Keep entries concise but complete enough to be useful

6. **Confirm** what was added to the user (or explain why nothing was added if duplicated)

## Example Entry Format

```markdown
### Brief Descriptive Title

**Problem:** What went wrong or what was confusing

**Solution/Insight:** The fix or key understanding

**Why it matters:** Brief explanation of impact
```
