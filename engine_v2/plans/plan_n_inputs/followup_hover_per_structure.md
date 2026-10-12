# PLAN_N follow-up — the H1 range hover lines snapshot EVERY structure (found by PLAN_N's landing review, 2026-10-12)

**Status: BUILT + MEASURED, NOT LANDED** — split out of PLAN_N so that landing stays one cause (byte-identical figures).
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
