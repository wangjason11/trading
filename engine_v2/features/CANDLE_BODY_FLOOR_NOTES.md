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
`CandleParams.pinbar_body_pip_floor` → consumed in
`candles_v2.py::classify_candles`. Pip size is per-pair (JPY=0.01 else 0.0001).

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
- Strict `<` with an `EPS` guard so a body sitting exactly at the floor keeps its
  original type (float-precision boundary — see GOTCHAS "Body-pip floor float
  precision").

Type-write order in `classify_candles` (last write wins):
1. default `normal`
2. body_pct bands → `pinbar` / `maru`
3. special-maru promotion (0.50–0.64 band + tiny close-side wick) → `maru`
4. **body-pip floor → `pinbar`** (this rule; trumps all the above)

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
  candles late. The recalibrated 2.20/1.0/0.5 give a gentler, proportional bite
  (~4–5% of all candles, ~3% of marus, ~13–15% of normals per TF).

## Open / may-revisit (why this is provisional)
- **Floor values** are a first calibration on NZD_USD only — likely to be tuned,
  and possibly **split per currency pair**, during strategy optimization.
- **The reclassify-to-pinbar approach itself** may be revisited (e.g. a separate
  "no-conviction" type, or a softer demotion) if it interacts badly with pattern
  detection downstream.
- Watch the interaction with `one_maru_opposite` / `one_maru_continuous` pattern
  alt-paths, which historically depended on big-flag semantics.
