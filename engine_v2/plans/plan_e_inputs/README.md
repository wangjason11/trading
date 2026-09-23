# Plan E inputs — raw inventories

These files back the digest [`../PLAN_E_inputs.md`](../PLAN_E_inputs.md). **Read the digest first.** It supersedes
these files wherever they disagree, because it applies the user's final naming decisions of 2026-09-23 (digest §0.1).
Examples of superseded proposals in these files:
- `cts_extreme_idx` / `bos_extreme_idx`. The new keys are `cts_anchor_idx` / `bos_anchor_idx`.
- `*_extreme_idx` for structure endpoints. An endpoint is an `*_anchor_idx`.
- `extreme_time`.
- Additive aliases and "old keys never removed". The amendment is an atomic migration with no aliases.
- Retiring the word "anchor". It has two realms (pattern / market structure).

They were produced by read-only multi-agent workflows in the zones-pass session (2026-09-22/23). Each file is rendered
in full from the workflow's consolidated JSON result, one block per inventoried site. The per-reader partial results
were not kept.

| File | Workflow | What it holds |
|---|---|---|
| `naming_inv.txt` | naming inventory | 70 rows: every candle-index name checked against the moment / anchor / extreme standard (verdict, holds, proposed name, export effect); the 15 exported changes; the frozen event-contract bridge; sizing; proposed sequencing |
| `anchor_feas.txt` | anchor meaning + flip feasibility | 50 "anchor" rows (JOB A); 74 reads of `CTS_ESTABLISHED` / `BOS_CONFIRMED` / `CTS_UPDATED` `.idx`, classified LOCATION / TIMING / BOTH (JOB B); the flip's blast radius, migration plan, risks and open questions |
| `extreme_final.txt` | "extreme" classification | 101 rows by category (pattern extreme, raw extreme, MS anchor, moment bug, bare element, misnomer…) with `final_name`; where "extreme" survives; renames to anchor |
| `zones_timing_audit.txt` | zones-timing audit (6 readers, 2 verifiers, 1 critic) | The POI activation-gate audit that preceded Plan D: the POI read sites, the same-class sweep, the consumers-and-deltas prediction, the docs and tests findings, the verifier verdicts and the critic findings. **Historical where it predicts Plan D.** Plan D landed as predicted, see `../PLAN_D_poi_activation_moment.md` §9 |

- **Coordinates:** line references are at HEAD `e2e0f89`. Plan D (`0a4eadc`) shifted four `.py` files; convert with
  the offset table in the digest's §0. Doc (`*.md`) line numbers are stale, so re-grep them.
- **Counts:** measured on save `20260922_195430_aadb887`. The current baseline `20260923_172626_0a4eadc` differs only
  in Plan D's cells.
- **Scrubbing:** absolute local paths were reduced to repo-relative paths or `<session scratchpad>`. The files contain
  no credentials.
