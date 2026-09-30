## Plan F consolidated review (working tree on 754a642)

**Method:** I re-checked every should_fix against the code myself. For the coverage claims, I copied `engine_v2/` from the working tree into the scratchpad, applied each mutation there and ran the full suite. The repo is untouched (`git status` is the same as at the start). In the copy, the unmutated baseline gives 764 passed, 1 skipped, 1 xfail and 1 failure. The failure is `test_smoke::test_oanda_history_smoke`, which fails only because the copy has no `oanda.cfg`. The skip comes from `artifacts/` also being absent in the copy. A mutant "survives" when its result is identical to that baseline. The candidate tests are in `C:/Users/wangj/AppData/Local/Temp/claude/C--Users-wangj-OneDrive-Documents-codingproj-Project-Retire-forex-engine-v2/bbfc6b47-b6c5-4c36-9a7f-d9c6858c602b/scratchpad/test_zz_consol.py`.

### Must fix
None.

### Should fix

**1. `FibTracker._evaluating` leaves `_in_event` stuck when `event_moment` raises** (lenses 1 and 2) — `engine_v2/zones/fib_tracker.py:335-339`
- **What happens:** `self._in_event = True` (:337) and `self._evaluated_at = event_moment(event)` (:338) both run before the `try:`. `event_moment` raises in three cases (`market_structure.py:182-189`):
  - KeyError when `via` or `confirmed_at` is missing.
  - TypeError when `confirmed_at` is None.
  - ValueError for any other event type.
- **Consequence:** after any of those, every later handler call on the same tracker fails with the misleading `AssertionError: FibTracker event handlers must not nest`.
- **Reproduced:** `test_G_evaluating_stuck_flag` fails on the working-tree copy with that AssertionError at fib_tracker.py:335.
- **Production impact:** latent. `orchestrator.py:158` builds a new tracker for each run. Tests with `pytest.raises`, a notebook, or the Phase-3 live driver can hit it.
- **Fix:** resolve the moment before touching any state.
  ```python
  assert not self._in_event, "FibTracker event handlers must not nest"
  moment = event_moment(event)
  previous = self._evaluated_at
  self._in_event = True
  self._evaluated_at = moment
  try: yield
  finally: ...
  ```
- **Pin test:** `pytest.raises(KeyError)` on a CTS_UPDATED without `via`; then the same tracker must accept a valid raw CTS_UPDATED.

**2. Nothing pins that the uncut paths stay uncut at their call sites** (lenses 1 and 3 merged) — `engine_v2/structure/market_structure.py:2085` and `engine_v2/zones/poi_zones.py:1087`
- **Mutation evidence:** each of these survives the full suite:
  - `market_structure.py:2085` changed to `evaluated_at=int(cts_idx)` (MS cycle-0 snapshot, cond2).
  - `poi_zones.py:1087` changed to `evaluated_at=int(cts_idx)` (in-flight select).
- **Why the existing tests miss it:**
  - `test_guard_uncut_paths_are_byte_identical` (test_imbalance_c3_knowability.py:281-291) calls the routine directly, not the callers.
  - The M1 test (:299-314) uses `structure_id=0`, which returns `'intra'` at fib_tracker.py:166 before any imbalance read.
  - The §5 pin for `select_fib_anchor_for_cycle(..., evaluated_at=None)` was never written. The only test of that function is the TypeError test (test_cross_cycle_fib_routine.py:250-262).
- **Candidate tests, all passing on the working tree:**
  - **F (MS stub):** `MarketStructure.__new__`, one gap with c2 == CTS_0 = 20, then `_update_cycle0_data()`. Asserts `cycle0_data["has_unfilled"] is True`. It fails on the `:2085` mutant.
  - **E (spy):** monkeypatch `poi_zones.select_fib_anchor_for_cycle` with a spy, then call `compute_poi_inners_for_cycle(df,30,1.2,40,1.35,1,structure_id=1,cycle_id=1,c0_data=c0)`. Asserts the recorded `evaluated_at` values are `[None]`. It fails on the `:1087` mutant, where the spy records `[40]`.
  - **D (label):** gaps (15,15) and (40,40, 1.20–1.30), c0 = {bos 10, cts 20, has_unfilled True}. `select_fib_anchor_for_cycle(df,1,1,30,1.2,40,1.35,c0,0.70,struct_direction=1,evaluated_at=X)` returns `'scenario_2_cross'` for X=None, `'scenario_3'` for X=40 and `'scenario_2_cross'` for X=41. This pins the routine, not the call site.
- **Also:** rename `test_guard_uncut_paths_are_byte_identical` to what it actually checks, e.g. `test_guard_routine_evaluated_at_none_is_uncut`.

**3. Three FibTracker cut sites have no test** (lens 3) — `engine_v2/zones/fib_tracker.py:1432`, `:980`, and `:1618/:1623/:1625/:1653`
- **Mutation evidence:** each mutant survives the full suite:

  | Site | Mutation | Candidate test | Caught? |
  |---|---|---|---|
  | :1432 | read `c0.get("has_unfilled", False)` instead of `_c0_has_unfilled_now` | A | yes |
  | :980 | `evaluated_at=None` | B, case `[40-False]` | yes |
  | :1618/:1623/:1625/:1653 | revert to the uncut `has_unfilled_imbalance(..., evaluated_at=None)` | C | yes |

- **Candidate tests** (in the scratch file, all passing on the working tree):
  - **A:** sid 1 h1; CTS_0 established at 20 with rv 15 (Scenario 1 TRUE); one gap with c2 25. The raw update at 25 returns None; the raw update at 26 gives `activated_at == 26`.
  - **B:** parametrized on `confirmed_at` 40 → no cross, 41 → cross. This is lens 3's fixture as written.
  - **C:** needs one correction to lens 3's description. `_get_latest_cross` returns `(key, FibState)` (fib_tracker.py:2091-2101), so the test must index `[1]`:
    ```python
    df = _df(80, [_gap(15), _gap(35, top=1.30, bottom=1.20), _gap(45, top=1.45, bottom=1.40)]); df.at[42, "l"] = 1.22
    # CTS_0 EST 20 (bos 10, rv 100) -> CTS_CONFIRMED 25 -> CTS_1 EST 40 (confirmed_at=40, bos 30)
    cross = tracker._get_latest_cross(1, 1)[1]; assert cross.active
    _quiet(tracker.on_cts_updated, _ev("CTS_UPDATED", 45, 1.46, 1, 1), df, 100)
    cross = tracker._get_latest_cross(1, 1)[1]
    assert cross.active is False and cross.meta["deactivated_at"] == 45 and (1, 1) not in tracker._fibs
    ```

### Nits
- **Plan wording vs code comments.** `plans/PLAN_F_imbalance_c3_knowability.md:159` says the two layers agree on "cond2 only". The code says "cond2 / cond3" (fib_tracker.py:142-144 and :960-962), and the code is right: the cond3 window ends at CTS_0, before the moment, so the cut cannot change it. Carry "cond2 and cond3 agree; cond1 may diverge (M1)" into the plan and the §6 docs.
- **`fib_tracker.py:1565`** — the `_update_fib_cts` DEACTIVATED print (and optionally :1561) should append `(evaluated_at={self._evaluated_at})`. That is where a not-yet-formed gap now shows up. This changes the log only.
- **`imbalance.py:140`** (lenses 1 and 2) — the line "Two as-ofs (...):" introduces nothing. Finish the sentence (`check_to_idx` = fill horizon; `evaluated_at` = moment) or delete it.
- **`poi_zones.py:647-652`** — the debug dump filters on `inst.start_idx <= z_scan_end`, but the sweep skips when `enter_idx = max(formed_at, first_active) > scan_end` (:930-932). An instance whose c2 equals `scan_end` is printed as relevant but never entered. Filter on `inst.formed_at <= z_scan_end`, or soften the "matches" comment.
- **Measurement scripts** — `plans/plan_f_inputs/shadow_measure.py:132-133`, `variant_run.py:91-92` and `inners_shadow.py:52-55` now raise TypeError on the landed tree because `evaluated_at` is required. Add a one-line header to each: "runs against 754a642 (pre-Plan-F) only".
- **`test_imbalance_c3_knowability.py:350`** — `event_moment(e) == e.idx` for raw-via events cannot fail (by definition). Replace it with the pattern half: `patterns = [e for e in updates if e.meta["via"] != CTS_UPDATED_RAW_VIA]; assert patterns and all(event_moment(e) is None for e in patterns)`.
- **`test_imbalance_c3_knowability.py:122-123`** — the docstring should name the mechanism: "(14,14) commit-fills at 19; the run (19,22) forms only at 20, so the POI deactivates at 19; pre-Plan-F it entered at its c2 = 19."
- **`_evaluating` restore is only partly pinned.** Deleting just `self._evaluated_at = previous` (fib_tracker.py:342) survives the full suite. Add a pin: after any handler call, `tracker._evaluated_at is None`; a nested `_evaluating` raises AssertionError.

### Refuted
- **Lens 3, "no_restore … never resets `_in_event` or the value → 766 passed":** refuted in that form. Dropping both lines 342-343 makes 59 tests fail, because the next handler hits the nesting assert. Only the `_evaluated_at` half is unpinned; that residue is kept as the last nit.
- Nothing else was dropped. Every should_fix held on re-check.

### Verified OK
- `event_moment` (market_structure.py:170-190) behaves as the plan says: `confirmed_at` for CTS_ESTABLISHED, the raw `via` rule for CTS_UPDATED, `idx` for CTS_THRESHOLD_UPDATED, otherwise it raises. All reads use direct indexing.
- `has_unfilled_imbalance` (imbalance.py:126-177) takes `evaluated_at` as keyword-only and required; `None` keeps today's uncut path, a value uses the formed-prefix path.
- `select_fib_anchor_for_cycle` (fib_tracker.py:101-205) takes a required keyword-only `evaluated_at` and forwards it to the routine.
- Both in-flight uncut calls pass `None` explicitly, with a stated reason: market_structure.py:2079-2085 and poi_zones.py:1083-1087.
- `_c0_has_unfilled_now` (fib_tracker.py:354-363) returns False for an absent c0 and otherwise uses the same window and horizon as the cache writes. The cache stays uncut at :1420-1423.
- The lenses' other verified_ok items are consistent with the code I read; I found nothing contradicting them.
- Candidate tests A–F: all pass on the working-tree copy, and each fails on its own mutant.