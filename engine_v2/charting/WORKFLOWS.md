# Workflows: Replay, Debugging, and Development Process

This doc explains how we work on this repo so changes remain safe and explainable.

---

## Replay workflow (canonical)

1. Run `run_replay.py` to fetch and replay a dataset and generate:
   - raw CSV export
   - final pipeline CSV export
   - printed summaries (pattern counts, structure levels, zone stats)
2. Attach structure levels + zones to df attrs:
   - `df.attrs["structure_levels"]`
   - `df.attrs["kl_zones"]`【fileciteturn1file10】
3. Export chart artifacts using export_plotly.

---

## Debug checklist (when something looks wrong)

### A) Confirm pipeline ordering
Zones depend on base features; base features must occur before structure.【fileciteturn2file1】

### B) Confirm structure_id filtering
If zones “disappear”, confirm the chart is selecting the most recent structure_id and the zones carry that meta field.【fileciteturn2file0】

### C) Confirm timing indices
When something “happens too late/too early”, check:
- PatternEvent.apply_idx (end_idx vs confirmation_idx)
- StructureEvent.idx vs meta["confirmed_at"]
- Zone meta["confirmed_idx"] rules (BOS vs CTS)【fileciteturn2file2】

### D) Confirm thresholds
If ranges or zones don’t expand:
- check whether the correct threshold-update event is emitted
- ensure that threshold updates are not coming from unrelated sources (e.g., only range sync updates CTS threshold)【fileciteturn2file14】

---

## Branching + PR discipline (hard rules)

- One branch per week from `main` (e.g., `week6-kl-zones`).
- Optional short-lived day/topic branches.
- Merge to `main` only when the week’s Definition of Done is met.
- Keep a replay “golden dataset” output to regression-test chart behavior.

---

## Contribution guidelines

### Naming & docstrings
- Prefer explicit names tied to domain language (CTS/BOS/range/reversal watch).
- Docstrings should carry the canonical semantics (not just “what code does”).

### Testing style
- The chart is the primary integration test.
- Add lightweight invariant checks in core engines to catch silent corruption (MarketStructure already does this).【fileciteturn1file11】

