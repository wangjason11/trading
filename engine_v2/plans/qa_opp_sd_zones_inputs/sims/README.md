# PLAN_R simulators and shadow runs (copied from the 2026-10-19/20 session scratchpad)

These are the measurement scripts behind PLAN_R's predictions (`../../PLAN_R_opp_sd_zones_proximity.md`). They are
read-only with respect to the engine: each one monkeypatches in memory and prints. The scratchpad dies with its
session, so they live here.

**How to run.** Copy this folder into your own session scratchpad and run the scripts there, from the repo root, with
`PYTHONPATH` = the repo root. Several scripts write pickles next to themselves; running them in place would leave
untracked `*.pkl` files in the repo.

| Script | What it is |
|---|---|
| `r4_sim.py`, `r4_parent_only.py`, `r4_zones.py`, `r4_subs.py` | Round 4: the first ladder simulator on S_Q's subs (passes tested only while a zone is a candidate) |
| `r5_shadow.py` | The shadow pipeline: today's code with the (0,1) / (1,2) proximity lists replaced by the new chains. Variants: `base`, `fix0a`, `full`. Logs: `shadow_fix0a.log` (the step-0a pin "(1,2) lo 0 → 3611"), `shadow_full2.log` (the §5.4 pool table; the 1 `sibling-CTS unavailable` line at :8639) |
| `review_measure/m1_frames.py` → `frames.pkl` | The H1 + M15 frames, fetched like the engine (H1 == S_Q `final.csv`); the H1 ↔ M15 mapping (LOH(h) = 4h+3 for h ≤ 1056) |
| `review_measure/m2_h1.py` → `h1_cycles.pkl` | The H1-only pipeline: captures `check_zone_proximity`'s inputs and S_Q's 6 triggers |
| **`review_measure/m4_sim.py`** | **The reference implementation of PLAN_R §1** (sticky source sub, pass from usable, last rung never passed, lock by pass, strict most-recent cycle, signed half-up gap, per-side counters). It reproduces step 1 (6 triggers) and step 2's firing chain. Flags: `pass_from`, `src_qual` (decided: `usable`, `True`). Needs the two pickles above. Log: `m4.log`. Not modelled: the Q2 (b) unlock of the opp_sd lock (decided 2026-10-20 after it was written; no opp_sd-side pass occurs on this window) and the end-freeze (its sub zones come from the real projections, which are already clipped) |
| `review_measure/m5_pytest_fallback.py` | Step 0b: the MS fallback patched to 8; `m5_8.log` (the 9-file set, 242 / 242), `m5_full_8.log` (full suite, 1743 / 1743) |
| `review_measure/m3_pass.py`, `m3b.py`, `m6_ext.py`, `m7_inners.py` | Pass candles, high / low extremes, zone inners (0 of 39 move) |
| `review_rules/capture.py`, `step1_impl.py` | The rules reviewer's independent step-1 implementation on the captured inputs (incl. Rules 2 / 3) |
| `review_rules/passes_from_start.py`, `r4_sim_from_scan_start.py`, `last_rung_pass.py` | The Q1 evidence (passes from the scan start kill the overlay) and the last-rung pass measurement |
| `review_sweep/m_views.py`, `m_trunc.py` | The sub zone view checks: the natural-end view leaks post-end cycles; zones as of the usable moment are identical on a truncated frame (41 / 41) |
