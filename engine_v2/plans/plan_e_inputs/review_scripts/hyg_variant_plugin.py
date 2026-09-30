"""Post-E·2 measurement variant (2026-09-26): shift EVERY family F1-F8 slice-local ->
entity-absolute at the mirror, readers untouched. A figure diff vs the baseline = a coupled
reader; the CSV key diff (cmp_meta_keys.py) should == the census. A NO-OP since Post-E·2 landed
(the keys are in the shift lists; the lists are de-duplicated below) — kept as the TEMPLATE for a
future "shift at the mirror" variant. Also run the suite with it active:
`PYTHONPATH=<this dir> python -m pytest -p hyg_variant_plugin` (all pass = no test pins the old
values — and none pins the new ones either).

usage (repo root): python -c "import sys; sys.path.insert(0, r'<dir>'); import hyg_variant_plugin, runpy; runpy.run_module('engine_v2.run_replay', run_name='__main__')"
Env HYG=F1,F2,... restricts the families (default all).
"""
import os

from engine_v2.multitf import entity_df_mutation as edm

FAM = {
    "F1": ("ev", ("effective_idx",)),
    "F2": ("ev", ("start_idx", "cts_idx", "confirm_idx", "pullback_apply_idx", "proximity_apply_idx")),
    "F3": ("ev", ("expires_idx",)),
    "F4": ("ev", ("pb_start",)),
    "F5": ("zone", ("expanded_last_idx",)),
    "F6": ("zone", ("bos_idx", "cts_idx")),
    "F7": ("fib", ("activated_at", "locked_at", "reactivated_at", "cycle1_bos_idx")),
    "F8": ("wc", ("anchor_idx",)),
}
ON = [f.strip() for f in os.environ.get("HYG", ",".join(FAM)).split(",") if f.strip()]
ev_keys = tuple(k for f in ON if FAM[f][0] == "ev" for k in FAM[f][1])
zone_keys = tuple(k for f in ON if FAM[f][0] == "zone" for k in FAM[f][1])
fib_keys = tuple(k for f in ON if FAM[f][0] == "fib" for k in FAM[f][1])
wc_on = "F8" in ON

edm._EVENT_META_IDX_KEYS = tuple(dict.fromkeys(tuple(edm._EVENT_META_IDX_KEYS) + ev_keys))
edm._ZONE_META_IDX_KEYS = tuple(dict.fromkeys(tuple(edm._ZONE_META_IDX_KEYS) + zone_keys))

_orig_shift = edm._shift_meta_indices


def _shift(meta, keys, offset):
    if tuple(keys) == ("deactivated_at",):   # the fib call site
        keys = tuple(keys) + fib_keys
    return _orig_shift(meta, keys, offset)


edm._shift_meta_indices = _shift

_orig_mirror = edm.mirror_lower_tf_result_to_entity_df


def _mirror(entity_df, result, *, structure_path_id):
    n0 = len(entity_df.attrs.get("wave_candles", []))
    _orig_mirror(entity_df, result, structure_path_id=structure_path_id)
    if wc_on:
        sb = int(result.meta.get("slice_begin", 0))
        for wc in entity_df.attrs["wave_candles"][n0:]:
            v = wc.meta.get("anchor_idx")
            if isinstance(v, int) and not hasattr(edm, "_WAVE_CANDLE_META_IDX_KEYS"):
                wc.meta["anchor_idx"] = v + sb   # the mirror's own fresh dict


edm.mirror_lower_tf_result_to_entity_df = _mirror
print(f"[hyg_variant] families={ON} ev+={ev_keys} zone+={zone_keys} fib+={fib_keys} wc={wc_on}")
