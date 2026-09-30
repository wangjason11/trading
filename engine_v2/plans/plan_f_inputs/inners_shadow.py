# RUNS AGAINST 754a642 (pre-Plan-F) ONLY: it calls has_unfilled_imbalance without evaluated_at,
# which Plan F made a required keyword. Kept as the measurement record of PLAN_F_inputs.md §2.
"""Shadow the MS in-flight POI-inner resolver: does the c3 rule change the INNER PRICES?

Returns the ORIGINAL inners (outputs unchanged). For every resolver call, recomputes the
inners with has_unfilled_imbalance patched to the R1-exact rule at K = the MS moment
(pattern path apply_idx; raw path i), and logs calls where the inner lists differ, plus the
cycle-0 snapshot has_unfilled differences. Also records every sd-proximity CTS confirmation
check that fires, to see whether a look-ahead-only inner was ever the trigger.
"""
import collections
import json
import os
import runpy
import sys
import atexit

REPO = os.getcwd()
sys.path.insert(0, REPO)
OUT = os.environ.get("INNERS_OUT", "inners_diff.jsonl")

import engine_v2.patterns.imbalance as imb_mod
import engine_v2.zones.poi_zones as poi_mod
import engine_v2.zones.cross_cycle_fib as ccf_mod
import engine_v2.structure.structure_engine as se_mod
import engine_v2.structure.market_structure as ms_mod

_orig_has = imb_mod.has_unfilled_imbalance
_orig_resolver = poi_mod.compute_poi_inners_for_cycle
stats = collections.Counter()
_fh = open(OUT, "w", encoding="utf-8")
STATE = {"K": None}


def _frames(skip, depth=40):
    f = sys._getframe(skip)
    out = []
    while f is not None and len(out) < depth:
        out.append(f)
        f = f.f_back
    return out


def _ms_moment():
    for fr in _frames(2):
        n = fr.f_code.co_name
        if n == "_apply_pattern_at_apply_idx":
            return int(fr.f_locals["apply_idx"]), "apply_idx"
        if n == "_maybe_update_cts_pre_confirm":
            return int(fr.f_locals["i"]), "raw_i"
    return None, "none"


def _has_r1(df, start_idx, end_idx, check_to_idx, fill_threshold=0.70, *, direction=None):
    K = STATE["K"]
    if K is None:
        return _orig_has(df, start_idx, end_idx, check_to_idx, fill_threshold, direction=direction)
    for inst in df.attrs.get("imbalances", []):
        if direction is not None and inst.direction != direction:
            continue
        if inst.start_idx >= K:
            continue
        vis_end = min(inst.end_idx, K - 1)
        if not (inst.start_idx <= end_idx and vis_end >= start_idx):
            continue
        if not inst.is_filled(df, check_to_idx, fill_threshold):
            return True
    return False


def shadow_resolver(df, bos_idx, bos_price, cts_idx, cts_price, struct_direction,
                    structure_id=0, cycle_id=0, fill_threshold=0.70, c0_data=None):
    orig = _orig_resolver(df, bos_idx, bos_price, cts_idx, cts_price, struct_direction,
                          structure_id, cycle_id, fill_threshold, c0_data)
    K, how = _ms_moment()
    stats[f"resolver calls via {how}"] += 1
    STATE["K"] = K
    poi_mod.has_unfilled_imbalance = _has_r1
    ccf_mod.has_unfilled_imbalance = _has_r1
    try:
        new = _orig_resolver(df, bos_idx, bos_price, cts_idx, cts_price, struct_direction,
                             structure_id, cycle_id, fill_threshold, c0_data)
    finally:
        poi_mod.has_unfilled_imbalance = _orig_has
        ccf_mod.has_unfilled_imbalance = _orig_has
        STATE["K"] = None
    if sorted(orig) != sorted(new):
        stats[f"INNERS DIFFER via {how}"] += 1
        _fh.write(json.dumps({"kind": "inners", "how": how, "K": K, "n": len(df),
                              "t0": str(df["time"].iloc[0]) if "time" in df.columns else None,
                              "sid": structure_id, "cycle": cycle_id, "bos_idx": bos_idx,
                              "cts_idx": cts_idx, "sd": struct_direction,
                              "orig": sorted(orig), "new": sorted(new)}) + "\n")
    return orig


se_mod.compute_poi_inners_for_cycle = shadow_resolver

# sd-proximity confirmations that fired (which inner triggered them)
_orig_fire = ms_mod.MarketStructure._fire_cts_confirmation_via_proximity


def shadow_fire(self, i, trigger_inner, zone_kind, *a, **k):
    st = self.state
    _fh.write(json.dumps({"kind": "prox_fire", "i": int(i), "inner": float(trigger_inner),
                          "zone_kind": str(zone_kind), "n": len(self.df),
                          "t0": str(self.df["time"].iloc[0]) if "time" in self.df.columns else None,
                          "sid": int(st.structure_id), "cycle": int(st.cts_cycle_id),
                          "poi_inners": [float(x) for x in (st.poi_inners_for_cycle or [])]}) + "\n")
    stats[f"prox fires zone_kind={zone_kind}"] += 1
    return _orig_fire(self, i, trigger_inner, zone_kind, *a, **k)


ms_mod.MarketStructure._fire_cts_confirmation_via_proximity = shadow_fire


@atexit.register
def _summary():
    _fh.close()
    print("\n=== INNERS SHADOW SUMMARY ===")
    for k in sorted(stats):
        print(f"{stats[k]:7d}  {k}")


sys.argv = ["engine_v2.run_replay"]
runpy.run_module("engine_v2.run_replay", run_name="__main__", alter_sys=True)
