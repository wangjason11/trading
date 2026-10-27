# Item 5's final part: the parent-overlay toggle + chart zoom (2026-10-27 — DONE: landed, the user's look approved)

SESSION ORDER item 5 (register `IDEA_PARKING_LOT.md` §D), its FINAL part. Every round the user's words verbatim, then
the measurements and the decisions. Canonical once landed: CHARTING_SPEC (+ PART4 §16.4 for the toggle).

## The ask (the user, 2026-10-27, verbatim — register §D SESSION ORDER)

"Afterward that, we will complete the final parts of charting updates, which will consist of 1) adding a toggle to hide
parent (1H) chart overlay elements, and 2) adding an ability to zoom into certain parts of the chart via a) a +/- switch
or tool and b) allowing user to input an idx for the chart to zoom into. We can discuss in details those things when we
get to them." And: "The final part we can do in the next session".

## What exists today (measured on S_T2 `20261027_140757_017c920`, 2026-10-27)

**The toggle — background.** PART4 §16.4 planned ONE legend group `Parent (<parent_path_id>)` holding every parent
overlay element, default ON, plotly's native group toggle; never built. Today no H1-overlay trace or shape carries a
`legendgroup`; the only group on the M15 charts is the sub fibs' `m15_fib` (its first box's rectangle is the legend
entry "M15 fib", every box's 5 shapes + its hover trace in the group — the precedent that a shape toggles with its
group in the pinned plotly 6.5.0).

**The parent (H1) elements on each M15 chart** (census `parent_overlay_census.py`; the same set on both lenses —
counter 158 traces / 136 shapes, confluence 306 / 307):

| Element | Traces | Shapes | Drawn by |
|---|---|---|---|
| H1 "bo" labels / pb-pr-rv state labels | 2 (`H1 bo`, `H1 state labels`) | — | `_render_h1_overlay` |
| H1 swing lines + the PB→BOS line | 3 (`H1 swing sid=0/1`, `H1 PB→BOS sid=0`) | — | same |
| H1 CTS / BOS / PB dots | 3 | — | same |
| H1 KL zones | 36 (6 outlines + 30 invisible hover lines) | 21 (15 active fills, 6 confirm lines) | same |
| H1 wave candle lines | 11 invisible hover hitboxes (unnamed, `TF=1H`) | 11 solid full-height lines (`yref="paper"`) | same |
| H1 POI zones | 4 invisible hover lines | 8 (3 active fills, 2 outline rects, 3 confirm lines) | same |
| H1 Prev BOS line | 1 | — | same |
| H1 zone-proximity triggers (black x) | 1 (`zone_proximity:trigger`, its OWN legend entry today) | — | `_render_proximity_triggers_overlay` |
| H1 WVMI LP circles (volume panel, `yaxis3`) | 2 (`H1 WVMI LP buy/sell`) | — | `_render_wvmi_lp_marks` |
| **Total per M15 chart** | **63** | **40** | |

The legend today (M15 charts): `struct:CONFIRMED:+1/-1`, `struct:SUCCESS:+1/-1`, `volume:spike`,
`zone_proximity:trigger`, `M15 fib` — 7 entries, titled "Overlays".

**Zoom — what exists.** (1) plotly's default modebar (top right, on hover; `write_html` passes no config): box zoom,
pan, **zoom in / zoom out (+ / −)**, autoscale, reset. Its +/− scale every non-fixed axis about the centre — the time
axis AND the price axis (`yaxis3`, the WVMI rows, is `fixedrange`); the price axis is never re-fitted to the visible
candles (plotly's autorange spans the whole series, so an x-only zoom leaves the candles flat). (2) The injected volume
JS (`_inject_volume_autoscale_js`, both charts): on any x-range change, `yaxis2` = `[0, visible bars' max × 1.4]`
(`volume.headroom`). (3) A static zoom EXPORT, unused: `run_replay.py` `ZOOM = None` → a second H1 chart file
`_zoom_<i0>-<i1>` sliced to an H1 idx range (H1 only; not interactive). The charts carry `rangebreaks` (weekend gaps
removed) on a time x axis, so an idx → time map must ship with the page.

## Round 1 (2026-10-27) — the four choices

Options put (all listed in chat first; the recommendation marked): the toggle — (a) one legend entry `Parent
(H1.main)` [rec.], (b) an on-chart button, (c) per-element legend entries, (d) per-element entries + an all button;
the coverage — all with the triggers folded in [rec.], all except the triggers, all except the WVMI circles, all except
both; the +/- tool — custom time-only +/- with the price axis re-fitted to the visible candles [rec.], plotly's toolbar
+/- as is, mouse-wheel zoom, custom +/- and wheel; the idx input — an idx box + a 15M/1H selector on the M15 charts
[rec.], the chart's own TF only, a from/to range, re-centre keeping the width. Defaults stated, not asked: zoom on all
three charts, the toggle on the two M15 charts only (H1 has no parent overlay), CSVs byte-equal.

**The user's answers:** toggle = **"On-chart button"**; coverage = **"All except both"** (the proximity triggers keep
their own legend entry, the H1 WVMI LP circles always show — the button covers 60 traces + 40 shapes); +/- = **"Custom
+/- and wheel"** (time-only +/- buttons with the price re-fit, plus scroll-wheel zoom); idx = **"idx + 15M/1H selector
(Recommended)"** (one box, a 15M/1H selector on the M15 charts — an idx_1H centres on its hour's four M15 candles; a
fixed window, +/-48 M15 / +/-24 H1 candles to be tuned on the example; the candle briefly marked).

Next: a clickable prototype (the S_T2 chart HTMLs post-processed — no engine change) for the user's look.

## The prototype (2026-10-27) — built, checked headless, shown to the user

`proto_toggle_zoom.py SAVE OUT` copies S_T2's three chart HTMLs to the gitignored `artifacts/proto_item5_final/` with
the control bar (`proto_chart_controls.js`) appended and, on the two M15 charts, the parent elements tagged
`legendgroup="parent_h1"` (found by name / colour — asserted 60 traces + 40 shapes on each lens). `proto_headless_check.py
HTML SHOTS STEPS` drives it in headless Chrome (choreographer, kaleido's driver — the Claude-in-Chrome extension was not
connected) and prints the visible idx span, the axes, the hidden parent counts, the mark, the message per step.

The bar (above the title): `Parent (1H): shown|hidden` (M15 only) | `-` `+` | `idx [ ]` + `15M|1H` (M15 only) `Go`.
Parameters (the knobs for the user's look): idx window +/-48 M15 / +/-24 H1 candles; the price re-fit = the visible
candles' low..high + 6 % each side; wheel 1.2x per notch about the cursor (plot area only); the mark a blue band
`rgba(30,120,255,0.18)` over the candle (an idx_1H: its hour's four M15 candles) for 2.5 s; minimum 10 candles.

Checked (headless, both chart kinds): idx 3336 (15M, confluence) -> 3288..3384 = +/-48, marked; idx_1H 651 on the M15
chart -> 2558..2654 (the hour's four candles centred); idx 651 on H1 -> 627..675 = +/-24; `+` / `-` halve / double
(97 -> 49 -> 97 -> 193); one wheel notch 193 -> 161 and back; an x-only zoom re-fits the price (0.55644..0.58698 ->
0.5754..0.58286 over 1000..1400); the button hides 60 / 60 traces + 40 / 40 shapes and shows them again; idx -5 / 99999
-> "no 15M candle at idx -5 — showing idx 0" (the window edge). Two prototype bugs found and fixed on the way (both
build notes): plotly REPLACES `_fullLayout.xaxis` on a full relayout (the mark's `shapes` update) — a cached axis read
the stale range, so `+` / `-` / the wheel / the re-fit worked from the initial view (read the axis fresh each call);
the imbalance candles are their OWN candlestick traces (`imbalance:bullish` / `imbalance:bearish`, same customdata
layout) — reading only `OHLC` missed them (H1 idx 651 is one); merged, the idx run 0..1057 (H1) / 0..4227 (M15)
without a gap.

## Round 2 (2026-10-27) — the user's look at the prototype, verbatim

"I think this looks good. I like the highlight. I think it should stay until the next jump or if idx is null. This allows
me to delete the idx, then press go, and it removes the highlight. Right now, if I delete the idx, then press go, it
gives an error that says "enter an idx". We can make it such that the action of pressing go with a null idx input
equate to removing the idx highlight (if any).

One other comment is that on the 1H chart, when using the idx -> go, the initial zoomed in view feels too zoomed in.

One additional thing I was curious as about, is that currently current when I hold down the mouse and click in the
chart, it creates a zoomed in view of the box I create. Is there a way where say I can switch modes to between that and
another mode where when I hold down the mouse and move around, it instead keeps the exact same window and slides the
window based on how I move the mouse? If this is too complicated, we don't have to implement it now.

Otherwise, I have no other comments. THis generally looks good."

**Taken as:** (1) the mark persists until the next jump; Go / Enter on an EMPTY box clears it (no message); (2) the H1
window +/-24 -> +/-48 (97 candles on every chart, ~4 days on H1 — the author's pick, a one-number re-render); (3) a
`Box zoom | Pan` switch in the bar — plotly's `dragmode` (the toolbar's Pan icon already existed: it moves price AND
time); pan = time only (`yaxis.fixedrange` while panning), the window width kept, the price re-fit on release (the
round-1 re-fit rule); the toolbar's Pan follows the same rule and the switch tracks it. Plotly's own toolbar +/- stays
(the user did not ask to remove it).

**The prototype after round 2 (built + checked headless, 2026-10-27):** the mark stays (still there after 3.5 s); Go on
an empty box clears it with no message; `abc` -> "not an idx: abc"; H1 idx 651 -> 603..699 (+/-48); the `Box zoom | Pan`
switch (`drag: pan/yfixed`). **A third build note — plotly pans by CLOCK time:** across a rangebreak its pan kept the
range's linear width, so the window shrank (H1 idx 300, a quarter-width drag: 97 -> 82 candles) or stuck (the
Christmas window 603..699: an end inside the break, the drag back did nothing). Fix: plotly's live drag kept for the
motion; on release the window is re-set in CANDLE space — the same count, shifted by round(drag px / plot width x
candles) (`mousedown` in the plot + `mouseup` record the drag; the next x-only relayout applies it). Measured: idx 300
252..348 -> 228..324 -> back 252..348; Christmas 603..699 -> 579..675 -> back 603..699; a 70 % drag shifts 68; the
counter chart with the parent hidden pans the same and keeps it hidden.

## The plan + prediction (2026-10-27, written BEFORE the build; the user: "everything looks good thank you. please go ahead")

**One landing, one commit.** The toggle and the zoom share the control bar (one script, one injection), and the two
deltas stay attributable: the figure-JSON delta is the toggle's group tags only; the zoom adds an HTML script only.
Solo (a figure-only landing, like the WVMI marks' step 5).

Build:
1. `export_m15_chart`: `PARENT_OVERLAY_GROUP = "parent_overlay"`; `_tag_parent_overlay(fig, first_trace, first_shape)`
   right after the `_render_h1_overlay` call puts every trace + shape that call drew into the group. The proximity
   triggers and the H1 WVMI circles are drawn after it, so they stay out (the user's "All except both").
2. `charting/chart_controls.js` (new): the prototype's script, reading its candle table from the config (not from the
   plotly traces, so the imbalance-candle traces need no merging), the parent TF + group from the config, ASCII only.
3. `export_plotly`: `_chart_controls_cfg(dfx, *, tf, idx_1h=None, parent_tf=None, parent_group=None)` = the candle
   table `[idx, time (UTC wall clock "YYYY-MM-DD HH:MM:SS"), low, high(, idx_1H)]` from the DRAWN frame + the knobs;
   `_inject_chart_controls_js(html_path, cfg)` appends the script (utf-8). Both exporters call it after the volume
   JS, with or without the volume panel (H1: tf "1H"; M15: "15M", the candle hover's idx_1H, parent "1H", the group).
4. `style_registry` `chart.controls`: `idx_half_window` 48, `price_pad` 0.06, `min_candles` 10, `wheel_step` 1.2,
   `idx_mark_color` rgba(30,120,255,0.18) — the values the user looked at.
5. Tests `tests/test_chart_controls.py`: the group = exactly what the overlay drew (a stubbed overlay + one pin on the
   real one), the config table (order, idx_1H, time format, a sliced frame, the knobs), the injection, both exporters'
   HTML. The JS behaviour: the headless harness on the replay's charts (`proto_headless_check.py`).
6. Docs: CHARTING_SPEC new §14 "Chart controls" (+ §9's zoom line), PART4 §16.4 (built as an on-chart button, not a
   legend group — the user's round 1), the register (item 5 DONE), the code map, CLAUDE.md's status.

**Prediction vs S_T2:** 30 / 30 CSVs byte-equal; the run log unchanged (skip grep 33, `[wvmi] total=5, locked=3`, 3
`[uc1_trigger]`, 57 LP circles; counts H1 83 / 234, counter 158 / 136, confluence 306 / 307). H1 chart: figure JSON
equal. Each M15 chart: traces + shapes in the same order, exactly 60 traces and 40 shapes differ, each ONLY by
`legendgroup = "parent_overlay"` — the same 60 / 40 the prototype's name / colour classifier picked; the layout
otherwise equal. The HTML: one control script per chart; the candle table H1 1058 rows (idx 0..1057, no idx_1H), each
M15 4228 rows (0..4227) with idx_1H 0..1056. Headless on the replay's charts = the prototype's measured steps.

## Build log (2026-10-27) — = the prediction

Built as planned: `PARENT_OVERLAY_GROUP` + `_tag_parent_overlay` (export_m15_chart), `_chart_controls_cfg` +
`_inject_chart_controls_js` (export_plotly, both exporters call it after the volume JS, with or without the volume
panel), `charting/chart_controls.js` (the prototype's script reading its candle table from the config — the imbalance
traces need no merging; ASCII only; never caches `_fullLayout.xaxis`), `style_registry` `chart.controls`. My own
`assert js.count("__CFG__") == 1` caught the script's header comment naming the placeholder (a second replace target)
on the first test run — reworded. Two fixture slips in the new tests, both mine: a POI's legs vs its indices (1-1-3
needs the end at IC + 3), and the sub WVMI dots are drawn AFTER the overlay (with the H1 circles), not before.

Checks: `tests/test_chart_controls.py` 12 tests; suite **2096 -> 2108**. Replay (69.95 s wall): fetch gate PASS; skip
grep 33 (the same families: the 10 `[wvmi] WARNING` lines, the collapsed-cycle skips, 6 FutureWarnings); `[wvmi]
total=5, locked=3`; 3 `[uc1_trigger]`; 57 LP circles; counts 83 / 234, 158 / 136, 306 / 307.
`controls_landing_check.py S_T2 artifacts/debug artifacts/charts`: **ALL OK** — 30 / 30 CSVs byte-equal; H1 figure
equal; each M15 chart 60 traces + 40 shapes differ ONLY by `legendgroup = "parent_overlay"`, exactly the prototype
classifier's set; layout otherwise equal; one script per chart; the candle table H1 1058 rows (0..1057, contiguous), M15
4228 (0..4227) with idx_1H 0..1056 (no unmapped). `fig_named_diff.py`: 120 changed traces, all `['legendgroup']`.
Headless on the replay's charts (`proto_headless_check.py`) = the prototype's measured steps: idx 3336 -> 3288..3384;
+ / - 97 -> 49 -> 97 -> 193; wheel 193 -> 161 -> 193; x-only zoom re-fits the price; Parent 60 / 60 + 40 / 40 hidden and
back, kept hidden across an idx jump; idx_1H 651 -> 2558..2654; empty Go clears; `abc` / -5 notes; pan out and back
exact on both chart kinds (Christmas 603..699 -> 579..675 -> 603..699; idx 300 252..348 -> 228..324 -> back).
Mutants (`mutants_chart_controls.py`, the full suite each): **33 / 33 KILLED**, each by the assertion sketched for it.
A slip on the way: the patch that turned `measure_after_overlay` from a deleted line (a NameError kill, worthless)
into a real move ran chained before the background runner and failed its own assert unseen — the first run used the
weak version (17 failures = every M15 render crashing). Caught by asking why 17, not 2; fixed with the Edit tool and
re-run alone: KILLED by exactly the two group tests. Lesson: a setup step chained before a background job reports
into the job's log — check it before reading the job's verdict.

## The user's look (2026-10-27) — APPROVED

Shown the replay's three charts (`artifacts/charts/NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5{,_M15_counter,
_M15_confluence}.html`) with the landing's results. The user, verbatim: "charts look good, go ahead with the commit
flow". No change asked. Item 5's final part is DONE, and with it SESSION ORDER item 5; next: item 6, zone size
adjustments (register §D).

Committed: source `5dfb5b3`; save S_controls `20261027_164527_5dfb5b3` (save `32cde54`, trunk `6ab4b62`) =
`LATEST_<branch>`. The save replay (105.77 s wall — the chart stages slower than the landing's 69.95 s, machine
variance): fetch gate PASS, skip grep 33 (the same families), `controls_landing_check.py` vs S_T2 ALL OK again, the
three PNGs byte-equal to S_T2. `cmp_save` vs S_T2: 0 real cells, `traces_xy_equal=True` everywhere,
`shapes_equal=False` on both M15 charts (the 40 tagged shapes — expected).
