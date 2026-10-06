"""PLAN_H §8.4 — the minimal BEHAVIOUR patches of the variant replays (no meta migration).

    python engine_v2/plans/plan_h_inputs/variant_patch.py TREE V1|V2|V3

TREE = a `git archive` of landing A (`TREE/engine_v2/...`). Each edit replaces an exact text that must occur ONCE
(else AssertionError — the tree is not A).
  V1 — the DRAWN POI only: `derive_poi_zones` builds 0–1 POI per record = `select_ip_ic`'s pick (meta `versions` =
       `[ip_rank]`, the keys unchanged); activation cond5 = `ic_touches_zone` on the AS-OF 61.8–80 zone (one
       helper `fib_zone_bounds`), reason `ic_not_touching_fib_zone`; `first_active` floored at the pick's
       `pattern_confirmed_idx` (R1). The MS resolver unchanged.
  V2 — the MS resolver only: `compute_poi_inners_for_cycle` returns the pick's inner (0–1), `select_ip_ic` OUTSIDE
       the `try` (PR1). The drawn POIs unchanged.
  V3 — both.
`_compute_poi_activation_history` keeps its signature (+ `pattern_confirmed_idx=None`) so A's direct test callers
still run under the census.
"""
import sys
from pathlib import Path

TREE, VARIANT = Path(sys.argv[1]), sys.argv[2]
assert VARIANT in ("V1", "V2", "V3"), VARIANT
P = TREE / "engine_v2" / "zones" / "poi_zones.py"
src = P.read_text(encoding="utf-8")


def sub(old: str, new: str) -> None:
    global src
    n = src.count(old)
    assert n == 1, (n, old[:120])
    src = src.replace(old, new)


V1 = [
    # 1. derive_poi_zones: the pick instead of the variants
    ("""        # Find IC candidates
        candidates = find_ic_candidates(df, fib_state, config)

        if not candidates:
            continue

        # Select IC variants
        ic_variants = select_ic_variants(candidates, df, fib_state, config)

        if not ic_variants:
            continue
""",
     """        # PLAN_H V1: ONE IC = the institutional-pattern pick (no pattern -> no POI)
        _ip_sel = select_ip_ic(df, fib_state, config)
        if _ip_sel.pick is None:
            continue
        ic_variants = {int(_ip_sel.pick.ic_idx): [_ip_sel.pick.ip_rank]}
        _ip_confirmed_idx = int(_ip_sel.pick.pattern_confirmed_idx)
"""),
    # 2. pass the R1 floor
    ("""                lifecycle_floor_idx=struct_start_by_sid.get(int(sid)),
            )
""",
     """                lifecycle_floor_idx=struct_start_by_sid.get(int(sid)),
                pattern_confirmed_idx=_ip_confirmed_idx,
            )
"""),
    # 3. the signature
    ("""    lifecycle_floor_idx: Optional[int] = None,
) -> List[Dict[str, Any]]:
    \"\"\"Produce the POI's activation history via an event-driven sweep.""",
     """    lifecycle_floor_idx: Optional[int] = None,
    pattern_confirmed_idx: Optional[int] = None,
) -> List[Dict[str, Any]]:
    \"\"\"Produce the POI's activation history via an event-driven sweep."""),
    # 4. first_active + the R1 floor
    ("""    if lifecycle_floor_idx is not None:
        first_active = max(first_active, int(lifecycle_floor_idx))
    if first_active > scan_end or ic_idx not in df.index:""",
     """    if lifecycle_floor_idx is not None:
        first_active = max(first_active, int(lifecycle_floor_idx))
    if pattern_confirmed_idx is not None:
        first_active = max(first_active, int(pattern_confirmed_idx))
    if first_active > scan_end or ic_idx not in df.index:"""),
    # 5. cond5 = the touch of the as-of zone
    ("""        if sd == 1:
            fib_level_min = anchor_high - range_size * (fib_min_pct / 100.0)
            fib_level_max = anchor_high - range_size * (fib_max_pct / 100.0)
        else:
            fib_level_min = anchor_low + range_size * (fib_min_pct / 100.0)
            fib_level_max = anchor_low + range_size * (fib_max_pct / 100.0)
        overlap = calculate_candle_overlap_pct(
            ic_high, ic_low, fib_level_min, fib_level_max,
        )
        return sorted(v for v, th in variant_thresholds.items() if overlap >= th)
""",
     """        zone_lo, zone_hi = fib_zone_bounds(
            anchor_high, anchor_low, sd, POIConfig(ic_fib_min=fib_min_pct, ic_fib_max=fib_max_pct))
        return ["touch"] if ic_touches_zone(ic_high, ic_low, zone_lo, zone_hi) else []
"""),
    # 6. the deactivation reason (R3)
    ("""                reasons.append("variant_below_v30")""",
     """                reasons.append("ic_not_touching_fib_zone")"""),
]

V2 = [
    ("""        config = POIConfig(fill_threshold=fill_threshold)
        candidates = find_ic_candidates(df, fib_state, config)
        if not candidates:
            return []
        ic_variants = select_ic_variants(candidates, df, fib_state, config)
        if not ic_variants:
            return []

        inners: List[float] = []
        for ic_idx in ic_variants.keys():
            if ic_idx not in df.index:
                continue
            if sd == 1:
                inners.append(float(df.loc[ic_idx, "h"]))  # buy POI inner = IC top
            else:
                inners.append(float(df.loc[ic_idx, "l"]))  # sell POI inner = IC bottom
        return inners
    except Exception:
        return []
""",
     """    except Exception:
        return []
    # PLAN_H V2: the institutional-pattern pick, OUTSIDE the try (PR1 — fail loudly)
    config = POIConfig(fill_threshold=fill_threshold)
    sel = select_ip_ic(df, fib_state, config)
    if sel.pick is None:
        return []
    ic_idx = int(sel.pick.ic_idx)
    return [float(df.loc[ic_idx, "h"]) if sd == 1 else float(df.loc[ic_idx, "l"])]
"""),
]

edits = (V1 if VARIANT in ("V1", "V3") else []) + (V2 if VARIANT in ("V2", "V3") else [])
for old, new in edits:
    sub(old, new)
P.write_text(src, encoding="utf-8")
print(f"[variant_patch] {VARIANT}: {len(edits)} edits -> {P}")
