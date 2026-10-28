# Candle body-pip floor — design notes

> **STATUS: PROVISIONAL (2026-06-16).** The candle-classification body-pip floor
> below is **not finalized**. Both the floor *values* and the *approach itself*
> (reclassify-to-pinbar) are expected to be revisited and possibly re-edited or
> reverted as the strategy is optimized. Treat this as a working iteration, not
> settled logic. The other two changes committed alongside it
> (`one_maru_opposite` disqualification, sibling-CTS direction filter) ARE
> intended to be kept.

## What this controls

A per-timeframe absolute floor on a candle's **real body length** (in pips),
wired by `apply_candle_classification(timeframe=...)` →
`DEFAULT_PINBAR_BODY_PIP_FLOOR_BY_TF` (`candle_classifier.py`) →
`CandleParams.pinbar_body_pip_floor` → `compute_candle_features` builds the
per-candle floor (`candles_v2.fixed_body_floor`: the same value on every row, in
price units) → consumed by `candles_v2.py::classify_candles(…, body_floor=…)`
step 4. Pip size is per-pair (JPY=0.01 else 0.0001).

**Since PLAN_U U1 (2026-10-28) the floor is a per-candle input**, and every
classified frame carries three columns (H1 `*_final.csv`, the M15 lens dfs, every
sub slice): `pinbar_body_floor` (the floor the candle was judged against, price
units, NaN = no floor), `candle_type_before_floor` (the body-% type, after the
special-maru promotion) and `is_floor_pinbar` (a pinbar because of the floor
only). The candle hover shows them (`pinbar_body_floor=… | candle_type_before_floor=…`,
CHARTING_SPEC §15 "The candle hover's body-floor line"). An institutional-pattern stacked leg is judged
against the floor of its LAST candle (`institutional_pattern._stack_types`).
U2 replaces the fixed value with the rolling one ("Direction" below; PLAN_U).

Current floors (calibrated 2026-06-09 on NZD_USD via
`debug/candle_size_distribution.py`):

| TF | floor (pips) |
|----|--------------|
| H1 | 2.20 |
| M15 | 1.0 |
| M5 | 0.5 |

## Current logic (what we changed TO, 2026-06-09)

**A candle whose real body is shorter than the per-TF floor is reclassified
`pinbar`** — applied *last* in `classify_candles` (after the body_pct bands and
the special-maru promotion), so it **trumps** maru/normal/special-maru
regardless of `body_pct`. Rationale: a body that small carries no directional
conviction, so it shouldn't be treated as a maru/normal regardless of how its
body compares to its wicks.

Consequences:
- The `is_big_maru` body-pip floor (see "previous logic") was **removed** — it's
  now redundant: a sub-floor candle can no longer be a maru in the first place,
  so it never reaches the big-maru ratio race nor the prior-maru pool.
- `is_big_maru` / `is_big_normal` are therefore back to **ratio-only**.
- `pinbar_dir` computation was **moved below** the special-maru block so the
  newly reclassified pinbars are directioned by the same wick-geometry rule.
- Strict `<` with an `EPS` guard (1e-12) so a body sitting exactly at the floor
  keeps its original type: a nominal N-pip body can float-round to
  N × pip − ~2e-15. (The GOTCHAS entry "Body-pip floor float precision" this
  line used to cite was never written — the reason lives here and in the
  `classify_candles` comment.)

Type-write order in `classify_candles` (last write wins):
1. default `normal`
2. body_pct bands → `pinbar` / `maru`
3. special-maru promotion (0.50–0.64 band + tiny close-side wick) → `maru`
4. **body-pip floor → `pinbar`** (this rule; trumps all the above)

## Readers — what a floor pinbar reaches (code map, 2026-10-27)

Every reader of `candle_type == "pinbar"` or of `pinbar_dir`, so the floor reaches more than the IC ranking:
the breakout patterns (`patterns/structure_patterns.py`: continuous :238-302 — its pinbar / normal / maru slots,
double_maru :398, one_maru_continuous :505-533, one_maru_opposite :578-617), MS's range-candidate qualification
(`market_structure.py:1083`), the KL base patterns (`kl_zones_v1.py` 2-candle :144-171, 1-candle big tail :202-212,
3-candle star :259-264), IC Step 2's tiebreak and Step 3's ranking rules 1 / 2 (`poi_zones.prefer_candle_type`,
:707), the institutional pattern's leg ranks (the stacked legs too: `institutional_pattern._stack_types`), wave
candles (`wave_candles.py:97-104`), WVMI (`wvmi.py:37-44`) and the candle hover. Line numbers as of 2026-10-27.

**Most floor pinbars have no direction** (`pinbar_dir` 0; measured 2026-10-27 on today's floor): H1 34 of 40 (all 7
marus + 27 of 33 normals), M15 118 of 142 (all 34 marus + 84 of 108 normals). A demoted maru can never get one: its
body is ≥ 50 % of its range, so no wick can exceed `pinbar_distance` 0.5. So a floor pinbar passes the readers that
read the type only (IC ranking rule 1's exclusion, the KL star's middle candle, the breakout patterns' pinbar slots,
MS's range candidates) and fails the ones that read the direction (KL "long tail" / "big tail", wave candles, WVMI).
The rolling floor keeps that shape: 68 of the 74 new H1 pinbars at U2 are directionless (PLAN_U §5.2).

## Previous logic (what we changed FROM — replaced, kept here for reference)

The old mechanism was `big_body_pip_floor` (`DEFAULT_BIG_BODY_PIP_FLOOR_BY_TF`,
values **H1=3 / M15=2 / M5=1**). It did NOT touch the candle *type*; it was an
**additive gate on the `is_big_maru` flag only**:

- `is_big_maru` required **both** the rolling-max ratio (`>= big_maru_threshold`)
  **and** the absolute body-pip floor to pass.
- `is_big_normal` was deliberately left **ratio-only** (no floor) so pattern
  alt-paths that gate on `c1.is_big_normal_as1` (e.g. `one_maru_continuous`)
  kept qualifying even when `c1`'s body was small.
- A small-bodied candle could still be classified `maru` or `normal`; only its
  *big-maru flag* was suppressed.

### Why we replaced it
- The old floor only suppressed the `is_big_maru` *flag*, leaving small marus in
  the candle-type stream (and in the prior-maru pool), so they still polluted
  downstream ratio races and pattern matching. Moving the floor to the
  **type-assignment layer** removes them at the source.
- The first floor *values* tried under the new approach (3/2/1, reused from the
  old table) were **too aggressive at the lower TFs**: M15=2 demoted ~23% of
  marus, M5=1 ~20% of all candles, and it pushed an H1 `sid=0` reversal ~190
  candles late (the saves contradict where that +190 belongs — "Measured from the saves" below). The recalibrated 2.20/1.0/0.5 give a gentler, proportional bite
  — measured with the engine's own rule on 2026-10-27: H1 3.8 % of all candles / 2.4 % of marus / 12.4 % of
  normals, M15 3.4 % / 2.4 % / 10.5 %, M5 3.0 % / 2.6 % / 8.7 % ("Measured distribution" below). The "~4–5 % /
  ~3 % / ~13–15 %" first quoted here came from `debug/candle_size_distribution.py`'s demotion table, which
  compares without the EPS guard, so a body exactly AT a 0.1-pip floor is counted by float noise.

### Measured from the saves (2026-10-27)

The same window and candles in every save compared (`plans/qa_pinbar_floor_inputs/floor_history_from_saves.py` +
`.txt`). The June 2026 session transcripts are not on this machine (the oldest is 2026-09-27), and why exactly 2.20
(not 2.0 or 2.25) was chosen is not written down anywhere.
- **The first floor** (`804d19d`, 2026-05-28, the big-maru gate H1 3 / M15 2 / M5 1) against `b46bec8` (no floor):
  the candle types are unchanged and the H1 event counts too, but 103 H1 `is_big_maru_as0` flags go — 101 on
  pinbars, 2 on normals, 0 on marus (directional big pinbars 81 → 13). `is_big_maru` is a range flag on any type,
  so the "maru" gate mostly cut big pinbars (the KL "no base big tail", the pinbar wave-candle and WVMI rules) — a
  side effect never written down. Today's rule restored them (79 directional big pinbars).
- **The May motivating case**, M15 4047 (2026-01-16 01:45, a 1.8-pip special maru): without a floor the counter sub
  confirms its CTS at 4048 (one_maru_continuous), BOS_CONFIRMED at 4054 (`confirmed_at`), CTS_ESTABLISHED 4054
  (`b46bec8`); with the big-maru floor none of these (`804d19d`). Today's M15 floor 1.0 does NOT demote it (no candle in
  4040-4056 changes type), yet S_controls has no CTS_CONFIRMED there — something else now prevents it (not measured
  which).
- **The switch commit** `2391597` (save `20260616_154122`, the pinbar floor 2.2 / 1.0) against `18a6b32`: the H1
  types change on 40 candles; H1 sid 0 loses the cycle CTS_ESTABLISHED 636 / CTS_CONFIRMED 640 (one_maru_continuous),
  and its REVERSAL_CANDIDATE moves 707 (apply 710) → 897 (apply 902), +190. The bullet above puts the +190 on the
  rejected 3 / 2 / 1 trial; the saves show it on the commit that landed 2.2 / 1.0. That commit also landed the
  `one_maru_opposite` disqualification (an H1 change), so the saves cannot say which change moved the reversal.

## Measured distribution — NZD_USD, the replay window (2026-10-27)

The data: H1 2025-11-16 22:00 → 2026-01-19 23:00 UTC, 1058 candles (the replay's own auto-extended frame, so H1
positions = the replay's candle idx); M15 4228 and M5 12679 candles over the same span (M5 is never run by the
pipeline). Classification is NATURAL — the floor OFF — so the tables show the population the floor acts on.
Prices are quoted to 0.1 pip, so every body is a multiple of 0.1 pip and the floor acts in 0.1-pip steps (H1 2.2
demotes bodies ≤ 2.1, M15 1.0 bodies ≤ 0.9, M5 0.5 bodies ≤ 0.4). Scripts + raw outputs:
`engine_v2/plans/qa_pinbar_floor_inputs/` (`deciles.py` → `deciles.txt`, `rolling_pinbar_percentiles.py` →
`rolling_pinbar_percentiles.txt`; run from the repo root; re-run after a window change).

**Type counts** (natural; and with today's floor):

| TF | candles | maru | normal | pinbar | with today's floor: maru / normal / pinbar |
|---|---|---|---|---|---|
| H1 | 1058 | 296 (28.0 %) | 266 (25.1 %) | 496 (46.9 %) | 289 / 233 / 536 |
| M15 | 4228 | 1392 (32.9 %) | 1028 (24.3 %) | 1808 (42.8 %) | 1358 / 920 / 1950 |
| M5 | 12679 | 4814 (38.0 %) | 2935 (23.1 %) | 4930 (38.9 %) | 4690 / 2679 / 5310 |

**Body and range deciles (pips) by natural type.** ★ = where today's floor falls.

H1 (median body 2.8; floor 2.2):

| | candles | p10 | p20 | p30 | p40 | p50 | p60 | p70 | p80 | p90 |
|---|---|---|---|---|---|---|---|---|---|---|
| body, all | 1058 | 0.5 | 1.0 | 1.6 | 2.1 ★ | 2.8 | 3.6 | 4.6 | 5.6 | 7.7 |
| body, maru | 296 | 3.2 | 4.0 | 4.8 | 5.5 | 6.1 | 6.9 | 7.8 | 9.0 | 11.9 |
| body, normal | 266 | 2.0 ★ | 2.4 | 2.8 | 3.3 | 3.7 | 4.2 | 4.8 | 5.3 | 7.0 |
| body, pinbar | 496 | 0.2 | 0.4 | 0.7 | 1.0 | 1.2 | 1.5 | 1.9 | 2.2 | 3.0 |
| range, all | 1058 | 4.1 | 4.9 | 5.6 | 6.3 | 7.0 | 7.8 | 8.9 | 10.4 | 12.8 |
| range, maru | 296 | 4.6 | 5.7 | 6.5 | 7.3 | 8.0 | 9.0 | 10.4 | 11.5 | 14.8 |
| range, normal | 266 | 4.4 | 5.1 | 5.8 | 6.4 | 7.3 | 8.2 | 9.0 | 10.4 | 13.0 |
| range, pinbar | 496 | 3.8 | 4.5 | 5.1 | 5.7 | 6.3 | 7.0 | 8.0 | 9.4 | 11.4 |

M15 (median body 1.5; floor 1.0):

| | candles | p10 | p20 | p30 | p40 | p50 | p60 | p70 | p80 | p90 |
|---|---|---|---|---|---|---|---|---|---|---|
| body, all | 4228 | 0.3 | 0.6 | 0.8 ★ | 1.1 | 1.5 | 1.9 | 2.4 | 3.0 | 4.1 |
| body, maru | 1392 | 1.5 | 2.0 | 2.3 | 2.6 | 3.0 | 3.4 | 3.9 | 4.6 | 6.0 |
| body, normal | 1028 | 0.9 ★ | 1.2 | 1.4 | 1.6 | 1.8 | 2.0 | 2.3 | 2.7 | 3.4 |
| body, pinbar | 1808 | 0.1 | 0.2 | 0.4 | 0.5 | 0.6 | 0.8 | 0.9 | 1.1 | 1.5 |
| range, all | 4228 | 1.9 | 2.4 | 2.8 | 3.1 | 3.5 | 3.9 | 4.4 | 5.2 | 6.4 |
| range, maru | 1392 | 2.1 | 2.7 | 3.2 | 3.6 | 3.9 | 4.4 | 5.0 | 5.9 | 7.4 |
| range, normal | 1028 | 2.0 | 2.4 | 2.8 | 3.2 | 3.6 | 4.0 | 4.5 | 5.2 | 6.4 |
| range, pinbar | 1808 | 1.8 | 2.2 | 2.5 | 2.8 | 3.1 | 3.5 | 4.0 | 4.5 | 5.6 |

M5 (median body 0.9; floor 0.5):

| | candles | p10 | p20 | p30 | p40 | p50 | p60 | p70 | p80 | p90 |
|---|---|---|---|---|---|---|---|---|---|---|
| body, all | 12679 | 0.2 | 0.3 | 0.5 ★ | 0.6 | 0.9 | 1.1 | 1.4 | 1.8 | 2.5 |
| body, maru | 4814 | 0.8 | 1.0 | 1.3 | 1.5 | 1.7 | 2.0 | 2.3 | 2.7 | 3.4 |
| body, normal | 2935 | 0.5 | 0.6 | 0.8 | 0.9 | 1.0 | 1.1 | 1.3 | 1.5 | 1.9 |
| body, pinbar | 4930 | 0.0 | 0.1 | 0.2 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.8 |
| range, all | 12679 | 1.0 | 1.2 | 1.5 | 1.7 | 2.0 | 2.2 | 2.5 | 2.9 | 3.7 |
| range, maru | 4814 | 1.0 | 1.4 | 1.7 | 2.0 | 2.2 | 2.6 | 2.9 | 3.4 | 4.2 |
| range, normal | 2935 | 1.0 | 1.3 | 1.5 | 1.7 | 2.0 | 2.2 | 2.5 | 2.9 | 3.6 |
| range, pinbar | 4930 | 0.9 | 1.1 | 1.3 | 1.5 | 1.7 | 1.9 | 2.2 | 2.5 | 3.0 |

Body deciles ÷ the TF's median body — the same shape at every TF: H1 0.18 0.36 0.57 0.75 1.00 1.29 1.64 2.00
2.75; M15 0.20 0.40 0.53 0.73 1.00 1.27 1.60 2.00 2.73; M5 0.22 0.33 0.56 0.67 1.00 1.22 1.56 2.00 2.78. Today's
floors sit at H1 0.79× / M15 0.67× / M5 0.56× the median, yet demote the same share of each type at H1 and M15
(2.4 % of marus at both).

**Rolling pinbar body percentiles** (the user's measurement, 2026-10-27). For every candle: the 90 most recent
natural pinbars STRICTLY BEFORE it, and the 80th, 85th and 90th percentile of their bodies (numpy's linear
interpolation; the p85 series added the same day). The series starts at the first candle with 90 prior pinbars; its deciles count one value per
candle (a window weighs as many candles as it stays current for).

| TF | series starts | candles in it | series | min | p10 | p20 | p30 | p40 | p50 | p60 | p70 | p80 | p90 | max | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H1 | idx 212, 2025-11-27 18:00 | 846 | p80 | 1.92 | 2.10 | 2.10 | 2.12 | 2.20 | 2.20 | 2.22 | 2.30 | 2.40 | 2.44 | 2.60 | 2.24 |
| | | | p85 | 2.10 | 2.20 | 2.26 | 2.36 | 2.40 | 2.40 | 2.59 | 2.60 | 2.60 | 2.76 | 2.96 | 2.48 |
| | | | p90 | 2.30 | 2.40 | 2.40 | 2.61 | 2.82 | 3.00 | 3.01 | 3.13 | 3.30 | 3.33 | 3.40 | 2.89 |
| M15 | idx 221, 2025-11-19 05:15 | 4007 | p80 | 0.80 | 1.00 | 1.00 | 1.00 | 1.10 | 1.10 | 1.12 | 1.20 | 1.20 | 1.32 | 1.62 | 1.13 |
| | | | p85 | 1.00 | 1.10 | 1.16 | 1.20 | 1.20 | 1.20 | 1.26 | 1.30 | 1.36 | 1.46 | 1.80 | 1.26 |
| | | | p90 | 1.02 | 1.20 | 1.22 | 1.30 | 1.31 | 1.40 | 1.50 | 1.51 | 1.60 | 1.71 | 2.51 | 1.44 |
| M5 | idx 223, 2025-11-17 16:35 | 12456 | p80 | 0.40 | 0.50 | 0.60 | 0.60 | 0.60 | 0.60 | 0.60 | 0.60 | 0.70 | 0.72 | 1.00 | 0.61 |
| | | | p85 | 0.46 | 0.60 | 0.60 | 0.60 | 0.66 | 0.70 | 0.70 | 0.70 | 0.80 | 0.80 | 1.13 | 0.69 |
| | | | p90 | 0.50 | 0.70 | 0.70 | 0.71 | 0.80 | 0.80 | 0.80 | 0.81 | 0.90 | 1.00 | 1.40 | 0.81 |

Against today's floors: the rolling p80 is below the floor on H1 35.1 % of the series' candles (exactly at it
19.3 %), M15 3.3 % (at it 28.2 %), M5 1.9 % (at it 12.4 %); the rolling p85 on H1 2.7 % (at it 9.3 %), M15
never (at it 6.5 %), M5 0.1 % (at it 2.9 %); the rolling p90 never is.

**What a rolling floor would demote** (a shadow, 2026-10-27, `rolling_floor_demotions.py`): a natural maru /
normal whose body is strictly below its candle's rolling floor, over the candles WITH a full window (H1 846 from
idx 212, M15 4007 from idx 221, M5 12456 from idx 223), next to today's fixed floor over the same candles:

| TF | floor | demoted (% of the candles) | marus (% of marus) | normals (% of normals) | with a pin direction |
|---|---|---|---|---|---|
| H1 | fixed 2.2 | 39 (4.6 %) | 7 (3.1 %) | 32 (15.2 %) | 6 |
| | rolling p80 | 42 (5.0 %) | 8 (3.5 %) | 34 (16.1 %) | 6 |
| | rolling p85 | 66 (7.8 %) | 10 (4.4 %) | 56 (26.5 %) | 9 |
| | rolling p90 | 98 (11.6 %) | 20 (8.7 %) | 78 (37.0 %) | 10 |
| M15 | fixed 1.0 | 132 (3.3 %) | 28 (2.1 %) | 104 (10.6 %) | 22 |
| | rolling p80 | 208 (5.2 %) | 44 (3.4 %) | 164 (16.7 %) | 32 |
| | rolling p85 | 276 (6.9 %) | 64 (4.9 %) | 212 (21.6 %) | 38 |
| | rolling p90 | 419 (10.5 %) | 118 (9.0 %) | 301 (30.6 %) | 50 |
| M5 | fixed 0.5 | 373 (3.0 %) | 120 (2.5 %) | 253 (8.8 %) | 71 |
| | rolling p80 | 679 (5.5 %) | 221 (4.7 %) | 458 (15.9 %) | 104 |
| | rolling p85 | 914 (7.3 %) | 298 (6.3 %) | 616 (21.4 %) | 126 |
| | rolling p90 | 1412 (11.3 %) | 485 (10.2 %) | 927 (32.2 %) | 171 |

**Which pinbars fill the window — natural vs final** (a shadow, 2026-10-28, `window_natural_vs_final.py` +
`.txt`; H1 + M15 only — the replay never builds an M5 frame). The user asked to try both: the 90 most recent
**natural** (body-%) pinbars, or the 90 most recent **final** pinbars (natural + the ones the floor demoted —
computed candle by candle, so each demotion moves the next floors). Their guess: "counting the demoted pinbars as
well may help smooth out the issue ... regarding volatile stretches because it may include small body demoted
candles that can help even out the high volatile periods." Each frame got a seed (the history before idx 0,
classified, then dropped), so all its candles have a full window; the seeded natural floor equals the frame-only one
wherever both exist (H1 846 candles, M15 4007). Over the whole frame (H1 1058 candles, M15 4228):

| TF | floor | demoted (% of the frame) | marus (% of 296 / 1392) | normals (% of 266 / 1028) | with a pin direction |
|---|---|---|---|---|---|
| H1 | fixed 2.2 | 40 (3.8 %) | 7 (2.4 %) | 33 (12.4 %) | 6 |
| | natural p80 / final p80 | 45 / 40 | 8 / 7 | 37 / 33 | 7 / 5 |
| | natural p85 / final p85 | 73 / 75 | 11 / 11 | 62 / 64 | 10 / 11 |
| | natural p90 / final p90 | 114 (10.8 %) / 101 (9.5 %) | 22 / 17 | 92 (34.6 %) / 84 (31.6 %) | 12 / 11 |
| M15 | fixed 1.0 | 142 (3.4 %) | 34 (2.4 %) | 108 (10.5 %) | 24 |
| | natural p80 / final p80 | 221 / 193 | 50 / 45 | 171 / 148 | 35 / 30 |
| | natural p85 / final p85 | 293 / 255 | 71 / 64 | 222 / 191 | 43 / 38 |
| | natural p90 / final p90 | 436 (10.3 %) / 410 (9.7 %) | 125 / 112 | 311 (30.3 %) / 298 (29.0 %) | 55 / 50 |

- **They differ little.** At p90: H1 100 candles demoted by both, 14 by the natural window only, 1 by the final only;
  M15 391 / 45 / 19. Median floor H1 3.03 natural vs 3.00 final, M15 1.40 both. The final p90 lands between the
  natural p85 and p90: the demoted candles enter the window with bodies below the floor, so the window's top tenth
  sits slightly lower — like a slightly lower percentile.
- **Reach back** (the candle to its window's oldest pinbar, median): H1 191 h natural vs 159 h final; M15 53 h vs
  43 h. The final window covers about a sixth less time, so it reacts somewhat faster.
- **The volatility guess, mixed.** After the most volatile week (2025-11-17, median H1 body 3.8 pips) the next week's
  median floor is 3.40 natural vs 3.13 final, yet the demotions barely move (16 vs 15); in the quietest week
  (2025-12-22, median body 2.1) the final floor is HIGHER (3.00 vs 2.81), its shorter reach dominated by the busier
  week before. Correlation of the floor with the previous day's median range (Spearman): H1 0.29 natural vs 0.22
  final, M15 0.56 vs 0.66. The lag stays under both: H1 weeks with the same median body 2.5 got a floor of 2.40
  (2025-12-29, 9 % of marus + normals demoted) and 3.00 (2026-01-12, 32 %). The knob for the lag is the window size,
  not which pinbars fill it.
- **How much history each needs.** The natural window needs exactly 90 natural pinbars before idx 0: a seed with
  93 (H1, 15 days) already gives the 60-day seed's result. The final window depends on where its computation
  started, until that start is far enough back: H1 4 candles still differ at 15 days (93 natural pinbars), 1 at
  20 days (130), 0 at 25 days (161); M15 0 from 6 days (159) for both windows. A 30-day and a 60-day seed give the
  same frame for both windows. So the final window's guard is an empirical margin (about 2 windows of natural
  pinbars here), not an exact count.

## Direction — the rolling floor (the user, 2026-10-27 / 28)

The user, 2026-10-27: "For the floor logic, I want the floor to be dynamic based on 90 most recent pinbar candles,
and the floor should be the most recent 90 pinbar's p90. We would also need to put in guards to retrieve additional
historical data if necessary to fulfill this requirement." 2026-10-28, on the open questions: "I generally agree
with all your recommendations. The one thing I'd like to try though is both using body-% pinbars only and also the
ones the floor itself demoted. I'd like to see how much they would differ." Agreed (the canonical build spec:
`engine_v2/plans/PLAN_U_rolling_pinbar_body_floor.md` — U1 landed 2026-10-28, U2 and U3 next):
- the floor of a candle = the p90 (numpy linear interpolation) of the bodies of the 90 most recent pinbars strictly
  before it; a body strictly below it makes a pinbar (today's `<` rule; a body equal to the floor keeps its type);
  the window size and the percentile are config settings;
- which pinbars fill the window: both are tried (natural, final), each a replay of its own;
- the history: a seed fetched before the frame's first candle, used only to fill the first windows and then dropped
  (the frame and its candle indices unchanged); a replay stops with an error when the seed can't supply the window,
  a frame without enough history (unit tests, short frames) gets no floor on those candles;
- the institutional pattern's stacked legs take the floor at the leg's last candle;
- each candle's floor and a "pinbar by the floor" flag go in the final CSV and the candle hover (a rendered example
  first);
- the fixed per-TF table (`DEFAULT_PINBAR_BODY_PIP_FLOOR_BY_TF`, `CandleParams.pinbar_body_pip_floor`) is removed.

## Open / may-revisit (why this is provisional)
- **Floor values** are a first calibration on NZD_USD only — likely to be tuned,
  and possibly **split per currency pair**, during strategy optimization.
- **The reclassify-to-pinbar approach itself** may be revisited (e.g. a separate
  "no-conviction" type, or a softer demotion) if it interacts badly with pattern
  detection downstream. Input: most floor pinbars are directionless, so they already act like a separate type for
  the readers of `pinbar_dir` ("Readers" above).
- Watch the interaction with `one_maru_opposite` / `one_maru_continuous` pattern
  alt-paths, which historically depended on big-flag semantics.
