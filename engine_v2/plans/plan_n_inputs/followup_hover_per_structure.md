# PLAN_N follow-up — the H1 range hover lines snapshot EVERY structure (found by PLAN_N's landing review, 2026-10-12)

**Status: LANDED 2026-10-12 as option B (the user widened it at the landing session) — committed `9f9406d`, saved
S_N_hover `20261012_194022_9f9406d` (save `cc3d3ff`, trunk `efd95b9`); see "The landing decision" below. The
per-structure fix of the next sections is the dated record of what was built first (= option A).** Split out of
PLAN_N so that landing stays one cause (byte-identical figures).
Register: `IDEA_PARKING_LOT.md` §A "H1 range hover lines snapshot every structure".

## The defect
`charting/export_plotly.py`, the H1 range hover-line builder's third replay pass (the only one that reaches the figure)
replays the range events in `_range_events_chart_order` — a `RANGE_STARTED` at its ANCHOR — but after each event writes
`range_by_candle[ev.idx] = <every structure's open range>`. A STARTED replayed at its anchor therefore writes, at its
moment, a snapshot that freezes the OTHER structures as of the anchor; a candle with no later write at that idx keeps
the stale state, and later candles forward-fill from it. It also starts a label range's line at its anchor when another
structure happens to have an event on that candle (normally a label line starts at its first back-filled update).
Pre-existing for label ranges (their 2–5 candle anchor→confirm window); PLAN_N N5 extends the exposure to pullback- /
proximity-created ranges (anchored at the CTS anchor).

## The instance on S_M2 (H1)
sid 1's label range: `RANGE_RESET` at 748 (2025-12-31 02:00), `RANGE_STARTED` at 750 (04:00) anchored at 748, its
back-filled `RANGE_UPDATED` 749, 751 …; sid 0 has a `RANGE_UPDATED` at 748 (lo 0.57764) after one at 747 (lo 0.57809).
Replay order at key 748: sid 1 RESET, sid 1 STARTED (snapshot written at 750 — sid 0 still at its 747 state), sid 0
UPDATED (snapshot at 748 — includes sid 1's new range). Result: sid 0's hover at candle 750 shows lo 0.57809 /
last update 747 (stale; the truth is 0.57764 / 748), and sid 1's line has a point at 748 (its anchor).

## The fix (built, measured 2026-10-12)
Each event snapshots ONLY ITS OWN structure; a candle takes each structure's latest snapshot at or before it. The
replacement block for the pass (from `# For simplicity, build hover lines …` up to `# Build hover traces per
structure_id`):

```python
        # For simplicity, build hover lines per structure_id by iterating through
        # visible candles and checking which ranges are active at each candle
        # Reset and rebuild properly.
        # Each event snapshots ONLY ITS OWN structure (PLAN_N landing review, 2026-10-12): a RANGE_STARTED is replayed at
        # its anchor — before other structures' events stamped between its anchor and its moment — so a snapshot of
        # EVERY structure taken at its moment froze the others as of its anchor (a reset between the two then
        # vanished from the hover line). A candle takes each structure's latest snapshot at or before it.
        open_ranges = {}
        snaps_by_sid = {}     # {sid: {idx: state dict, or None = no open range}}

        for ev in all_range_events:
            sid = int(ev.meta.get("structure_id", 0))
            sd = int(ev.meta.get("struct_direction", 1))
            idx = int(ev.idx)

            if ev.type == "RANGE_STARTED":
                start_idx = _range_left_idx(ev)
                open_ranges[sid] = {
                    "start_idx": start_idx,
                    "hi": float(ev.meta.get("hi", 0)),
                    "lo": float(ev.meta.get("lo", 0)),
                    "last_update_idx": start_idx,
                    "struct_direction": sd,
                }
            elif ev.type == "RANGE_UPDATED":
                if sid in open_ranges:
                    open_ranges[sid]["hi"] = float(ev.meta.get("hi", open_ranges[sid]["hi"]))
                    open_ranges[sid]["lo"] = float(ev.meta.get("lo", open_ranges[sid]["lo"]))
                    open_ranges[sid]["last_update_idx"] = idx
            elif ev.type == "RANGE_RESET":
                if sid in open_ranges:
                    del open_ranges[sid]

            # Snapshot this event's structure at this idx
            rng_now = open_ranges.get(sid)
            snaps_by_sid.setdefault(sid, {})[idx] = dict(rng_now) if rng_now is not None else None

        # Per visible candle: each structure's latest snapshot at or before it (none = no open range)
        range_by_candle = {}  # {idx: {sid: {hi, lo, start_idx, last_update_idx, sd}}}
        for sid, snaps in snaps_by_sid.items():
            cur = None
            for idx in sorted(set(idx_set) | set(snaps)):
                if idx in snaps:
                    cur = snaps[idx]
                if cur is not None and idx in idx_set:
                    range_by_candle.setdefault(idx, {})[sid] = cur

```

**Measured on the reference window (a replay with the fix vs S_M2):** every CSV unchanged; M15 figures identical; the
H1 figure differs in exactly 4 traces — `range:top:sid0` / `range:bottom:sid0` at candle 750 (customdata lo 0.57809 →
0.57764, last_update 747 → 748; the bottom line's y), `range:top:sid1` / `range:bottom:sid1` lose their point at 748
(341 → 340 points). Shapes (the boxes) identical.

## The pin (moves into `tests/test_plan_n_naming_audit.py` with the fix)

```python
def test_p4_each_structures_hover_line_ignores_the_others_replay_order(tmp_path, monkeypatch):
    """(The landing review's finding 1.) sid 0: a label range anchored at 1, confirmed at 3, reset at 12; sid 1: a
    pullback range at 14 anchored at 8 — replayed at 8, BEFORE sid 0's reset. A snapshot of every structure taken at
    14 froze sid 0 as still open (its line came back from 14 on) and sid 1's line began at 12; each structure's own
    snapshots end sid 0 at its reset and start sid 1 at its moment."""
    df = _prepare_df(_make_uptrend_data(32))
    df.attrs["structure_events"] = [
        _rng(3, "RANGE_STARTED", 0, range_anchor_idx=1, confirmed_at=3, hi=1.0, lo=0.9),
        _rng(12, "RANGE_RESET", 0),
        _rng(14, "RANGE_STARTED", 1, range_anchor_idx=8, confirmed_at=14, reason="pullback_created_range",
             hi=2.0, lo=1.9),
    ]
    fig = _render_h1(df, monkeypatch, tmp_path, ranges=True)
    t = [_ts(x) for x in df["time"]]
    span = {tr.name: (_ts(min(tr.x)), _ts(max(tr.x)), len(tr.x)) for tr in fig.data
            if (tr.name or "").startswith("range:top:")}
    assert span == {"range:top:sid0": (t[3], t[11], 9), "range:top:sid1": (t[14], t[-1], len(t) - 14)}
```

Mutant (killed by it, full suite, 2026-10-12 — `mutants_n_review_result.txt` H1): restore the all-structures
snapshot — `snaps_by_sid.setdefault(sid, {})[idx] = …` → write every structure in `set(snaps_by_sid) | set(open_ranges)
| {sid}` at `idx`.

## Docs at landing
CHARTING_SPEC "Range rectangles" (drop the known-defect note; the hover line is per structure); the `export_plotly`
comment; PLAN_N §12 finding 1; the register line closed; a `/compare` prediction = the 4 traces above.

## The landing decision (2026-10-12, the landing session) — option B: the hover line = the box, candle for candle

**The second symptom (found before applying the fix above).** The same mechanism — a `RANGE_STARTED` replayed at its
anchor but snapshotted at its moment `ev.idx` — also stales a structure's OWN line: the STARTED's snapshot is written
before the back-filled `RANGE_UPDATED`s stamped between its anchor and its moment apply, so the moment candle, and every
candle forward-filled from it, shows the anchor-time state. The per-structure fix (A) keeps it. On S_N: **16 candles on
7 label ranges**. Example — sid 1's label range, anchor H1 711 (2025-12-29 13:00), confirmed 716 (18:00); its
back-filled updates raised the top to 0.58108 (candle 714's high, 16:00): the hover shows hi 0.58002 / last update 711
on candles 716–723 (8 hours) while the box top is drawn at 0.58108. Measured by an offline replay of the pass on S_N's
events (scratchpad `hover_sim.py`; its replay of the current pass == the S_N figure, 1114 hover points; option A
reproduces the measured 2-point delta above).

**The candidates (all change the same 4 traces only).** A — land as built (2 candle points; the 16 stale candles to a
later `/compare`). A2 — A plus "a candle shows its structure's state after the last-replayed event stamped at or before
it" (18 candle points; the line starts unchanged). B — snapshot at each event's chart-order (replay) candle: the hover
line carries the box's data candle for candle, and a range's line starts where its box starts (28 candle points).
**The user chose B** (2026-10-12), and to delete the builder's dead first two passes in the same change (register §A
"Dead code in the H1 range hover-line builder"; no output effect).

**B's code.** With every snapshot taken at the event's replay candle, snapshots are written in replay order, so the
all-structures snapshot is correct again and the per-structure split of A is unnecessary: the change to pass 3 is the
snapshot's key (`ev.idx` → the replay candle, from a `_range_events_with_replay_idx` helper shared with the box order),
plus a forward-fill that also carries a snapshot from before a sliced chart's first candle (a range open before an
`idx_range` / `max_points` slice shows from the slice's first candle, like its clamped box; the old pass snapshotted only
visible candles — no effect on the unsliced replay chart). The mutant of A ("restore the all-structures snapshot") is
therefore no longer a defect under B; B's mutants are listed with the landing.

**The written prediction (`/compare` vs S_N), before the replay.** All 27 CSVs md5-equal; both M15 figures identical
(`fig_hash` equal); `run.log` equal up to timings (chart code only). The H1 figure: shapes identical, layout identical,
exactly 4 traces differ — `range:top:sid0` / `range:bottom:sid0` (773 → 780 points) and `range:top:sid1` /
`range:bottom:sid1` (341 → 345 points); 28 (sid, candle) changes, each in both of its structure's traces' customdata
(the y only where that line's own bound changed):

| sid | candles | old → new (`range_start_idx`, hi, lo, `last_update_idx`) | why |
|---|---|---|---|
| 0 | 115; 173; 196, 197; 310; 431; 653 | none → the new label range from its anchor (7 points) | the line starts at the anchor, like the box |
| 0 | 175 | (173, 0.56966, 0.56851, 173) → (…, 0.56847, 174) | confirm candle stale |
| 0 | 313 | (310, 0.57766, 0.57732, 310) → (310, 0.57774, 0.57714, 312) | confirm candle stale |
| 0 | 655–657 | (653, 0.58534, 0.584, 653) → (…, 0.58392, 654) | confirm candle stale + forward-filled |
| 0 | 750 | (653, 0.58534, 0.57809, 747) → (…, 0.57764, 748) | the cross-structure instance (sid 1's STARTED) |
| 1 | 711; 762; 903, 904 | none → the new label range from its anchor (4 points) | the line starts at the anchor |
| 1 | 716–723 | (711, 0.58002, 0.57902, 711) → (711, 0.58108, 0.57902, 714) | confirm candle stale + forward-filled |
| 1 | 750 | (748, 0.57835, 0.57764, 748) → (748, 0.57841, 0.57764, 749) | confirm candle stale |
| 1 | 764–765 | (762, 0.57552, 0.574, 762) → (762, 0.57566, 0.574, 763) | confirm candle stale + forward-filled |

sid 1's point at 748 (its anchor) is KEPT under B (option A removed it).

**`/compare` vs S_N (2026-10-12) — = the prediction exactly.** Replay 86.1 s wall (exit 0); fetch gate PASS (4228 /
5 chunks); grep 22 (= S_N: the collapsed-cycle / unresolved-trigger lines + FutureWarnings); counts H1 83 / 239,
counter 137 / 123, confluence 282 / 276 (= S_N). **27 / 27 CSVs md5-equal**, no new / missing file. `fig_hash`: both
M15 hashes EQUAL to S_N, H1 `21f98a10…` → `f89c0cba…`. The keyed H1 trace diff (scratchpad `h1_trace_diff.py`, per
trace name, points keyed by candle): shapes IDENTICAL, layout IDENTICAL, **exactly 4 changed traces** —
`range:top/bottom:sid0` 773 → 780, `range:top/bottom:sid1` 341 → 345 — every added / changed point = the table above.
`run.log` 9105 lines (= PLAN_N §13's S_N log; the S_N log itself is not saved, so no line diff — chart code only).
Tools: `h1_trace_diff.py` (output `followup_h1_diff_vs_SN.txt`), `hover_sim.py` / `hover_sim_a2.py` (the decision).
Chart review: the user, **"looks good go ahead"** (2026-10-12) → committed + saved.

**Pins + mutants (as landed).** `tests/test_plan_n_naming_audit.py` 18 → 21: the note's probe re-derived for B
(`test_p4_each_range_hover_line_runs_where_its_box_runs_across_structures`: sid 0 (t1, t11, 11), sid 1 (t8, end)),
`test_p4_a_label_ranges_hover_keeps_its_back_filled_update_on_the_confirm_candle`,
`test_p4_a_sliced_charts_hover_line_starts_at_its_first_candle_like_the_clamped_box`; P4's first-point pin t[6] →
t[5] (the anchor). All 4 fail on the pre-follow-up builder. Suite **1712** passed (HEAD collects 1709 — PLAN_N's
"1708" was its passed count; +3 here). `mutants_hover_followup.py` (HB1 snapshot at the event's idx, HB2 forward-fill
visible candles only, HB3 no forward-fill, HB4 the helper's replay idx = the event's idx, HB5 a STARTED's last update
at its moment, LR1 re-anchored, LR2, LR3): **8 / 8 KILLED**, full suite each (`mutants_hover_followup_result.txt`).
