# CLAUDE.md — repo root

**Project context lives in [`engine_v2/CLAUDE.md`](engine_v2/CLAUDE.md)** — current status, hot
paths, the LOCKED pipeline order, guardrails, debug checklist. Read it first. Also at this root:
`PROJECT_PRINCIPLES.md` (non-negotiables), `README.md`, `WEEKLY_DEFINITION_OF_DONE.md`, and
`IDEA_PARKING_LOT.md` — the ONE canonical register of every deferred / parked / planned-but-unbuilt item (a new
deferral gets a line there + its detail in the owning doc).

This file exists so the rule below is in context from the first turn: it is the root `CLAUDE.md`
that loads at startup, whereas `engine_v2/CLAUDE.md` is nested and may not be.

## Documentation cadence (standing rule, 2026-09-22)

Capture learnings **as they happen**, not in one sweep at the end of the session — an end-of-session
pass flattens the early, most carefully-reasoned discussions. Write the nuance down the moment it is
said (verbatim + why it matters), then **file and reconcile at every checkpoint**: after each landed
change, at each `/compare` pause, and at `/commit-save`.

Frequent writes make *reconciling* the risk, not omission: **edit existing entries in place** rather
than appending a contradicting one, keep one canonical home per fact (everything else cross-links),
and move code + spec + skill values + memory in the **same commit**. Unresolved items go one line into
the durable `memory/_INBOX.md` — **if it is non-empty at a cold start, drain it before new work.**

Full rule, trigger list and routing table: `memory/feedback_continuous_documentation.md` (loaded every
session via `MEMORY.md`). Procedure: the `/remember` skill's "Continuous mode".
