# Plan E cold-review prototypes (2026-09-24, sequencing/tests lens)

Throwaway scripts the reviewer wrote at HEAD `2d4d2b4`; kept because E1/E2 reuse them (PLAN_E §4.3, §6.1, §6.4).

| File | What | Plan E use |
|---|---|---|
| `guard.py` | runs the `test_render_sub_projection` `geometry` fixture and lists mirrored sub-event meta `*_idx`/`*_at` keys vs `_EVENT_META_IDX_KEYS` | prototype of the E1 guard test (§4.3); gave the `KNOWN_SLICE_LOCAL` allow-list |
| `e4sim.py` | `_run_downstream_pipeline` on the lagging fixtures, events cloned with EST/BOS `idx := confirmed_at`, outputs compared | prototype of the E2 E4-simulation test (§6.4); fails at HEAD in `h1` and `cross_cycle` modes |
| `census_plugin.py` → `census.json` | pytest plugin counting test-built CTS_ESTABLISHED / BOS_CONFIRMED events and their meta keys | fixture census: 17 files, 23 construction sites, ≈301 events (§6.1) |
| `e4flip_plugin.py` | pytest plugin patching the emitters to pass `apply_idx` with readers untouched | measured: EST-only flip fails 3 tests, BOS-only 15 (§6.1) |

Not maintained; re-check against HEAD before reuse.
