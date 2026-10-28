"""PLAN_W 9a mutants (§1.4): `common/types.require_positional_index` and its two callers.

    python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py engine_v2/plans/plan_w_inputs/mutants_9a.py

Killers (`tests/test_imbalance.py`): `test_the_fill_checks_raise_on_a_frame_not_indexed_0_to_n_minus_1`,
`test_the_fill_checks_accept_an_int64_index_0_to_n_minus_1`, `test_fill_cache_separate_candles_and_agrees_with_
is_filled[...-int64_index]`. Anchors use LF (the runner reads text mode).
"""
TY = "engine_v2/common/types.py"
PZ = "engine_v2/zones/poi_zones.py"
_INT = '    if n == 0 or (index.dtype.kind in "iu" and list(index) == list(range(n))):'

MUT = [
    # the check removed from `is_filled` (it scans by position whatever the labels)
    ("v9a_is_filled_no_check", TY,
     '        n = require_positional_index(df, "ImbalanceInstance.is_filled")',
     "        n = len(df)"),
    # the check removed from the cache
    ("v9a_cache_no_check", PZ,
     '    n = require_positional_index(df, "poi_zones._compute_fill_idx_cache")',
     "    n = len(df)"),
    # the integer-index branch dropped (an int64 0..n-1 frame rejected)
    ("v9a_no_int_branch", TY, _INT, "    if n == 0:"),
    # the old cache's identity test (first 0, last n-1) in place of the full comparison
    ("v9a_first_last_only", TY, _INT,
     '    if n == 0 or (index.dtype.kind in "iu" and index[0] == 0 and index[-1] == n - 1):'),
    # the empty-frame pass dropped (an empty object-dtype frame raises)
    ("v9a_no_empty_pass", TY, _INT,
     '    if (index.dtype.kind in "iu" and list(index) == list(range(n))):'),
]
