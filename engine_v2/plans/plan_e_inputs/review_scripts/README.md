# Plan E cold-review prototypes (2026-09-24, sequencing/tests lens)

Throwaway scripts the reviewer wrote at HEAD `2d4d2b4`; kept because E1/E2 reuse them (PLAN_E §4.3, §6.1, §6.4).

| File | What | Plan E use |
|---|---|---|
| `guard.py` | runs the `test_render_sub_projection` `geometry` fixture and lists mirrored sub-event meta `*_idx`/`*_at` keys vs `_EVENT_META_IDX_KEYS` | prototype of the E1 guard test (§4.3); gave the `KNOWN_SLICE_LOCAL` allow-list |
| `e4sim.py` | `_run_downstream_pipeline` on the lagging fixtures, events cloned with EST/BOS `idx := confirmed_at`, outputs compared | prototype of the E2 E4-simulation test (§6.4); fails at HEAD in `h1` and `cross_cycle` modes |
| `census_plugin.py` → `census.json` | pytest plugin counting test-built CTS_ESTABLISHED / BOS_CONFIRMED events and their meta keys | fixture census: 17 files, 23 construction sites, ≈301 events (§6.1) |
| `e4flip_plugin.py` | patches the emitters so `ev.idx := confirmed_at` right after emission (`FLIP=est` / `bos` / `both`; readers untouched). **Maintained since E2** — the E4 variant replay: `FLIP=both python -c "import sys; sys.path.insert(0,'engine_v2/plans/plan_e_inputs/review_scripts'); import e4flip_plugin, runpy; runpy.run_module('engine_v2.run_replay', run_name='__main__')" > run_variant.log 2>&1` (overwrites artifacts/ — copy the normal replay's outputs aside first, re-run a normal replay before any save) | E2's completeness proof (PLAN_E §6.7): EST-only == §8 E4a, BOS-only == §8 E4b, both == E4a + E4b; re-run after the last E3 stage |
| `cmp_save.py` | cell diff of the 24 CSVs (+ `--strip` meta keys) and figure-JSON x/y + shapes diff of the 3 charts against a baseline folder. **Maintained since E2** | every E2 `/compare` and variant; see its docstring (short base dir on Windows) |

The other scripts are not maintained; re-check them against HEAD before reuse.
