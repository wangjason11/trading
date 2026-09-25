"""Static parent tables for the sub-structure sweep (PART4 §17.6, Plan C §3).

Computed ONCE from the H1 event stream before any sub work. One helper
replaces the three derivations the Phase-1 chain carried
(`parent_cycle_floor_h1` in `run_pipeline`, `parent_struct_end_m15` +
`parent_end_lookup` in `_run_multi_tf_dual`, the trigger detectors'
`lifecycle_end_idx` / `_find_m15_lifecycle_end`):

```
rev_by_sid[S]      = STATE_CHANGED→reversal idx of sid S      (compute_reversal_idx_by_sid —
                                                               NOT REVERSAL_CANDIDATE.apply_idx, a prediction)
struct_start[S]    = first CTS_EST moment / reversal handoff   (compute_struct_start_by_sid; Plan E E3f)
cts_moment[(S,C)]  = CTS_ESTABLISHED.meta["confirmed_at"]      # the MOMENT the cycle was established
                                                               #   (== BOS_CONFIRMED.confirmed_at, definitional;
                                                               #   last-seen per (S,C)) — NOT the CTS anchor (the extreme)
parent_sd[S]       = CTS_ESTABLISHED.meta["struct_direction"]  # one direction per sid
floor_h1[(S,C)]    = max(struct_start[S], cts_moment[(S,C)])   # == the cycle's CLAMPED lifecycle-start
end_h1[(S,C)]      = floor_h1[(S,C+1)] if it exists            # next cycle's CLAMPED start (same rule as
                     else rev_by_sid[S], else None             #   compute_cycle_lifecycle)
floor_m15 / end_m15 = LOH(...)                                 # every map must succeed (assert)
degenerate[(S,C)]  = end_m15 is not None and floor_m15 >= end_m15
```

`LOH` = `_map_parent_idx_to_m15_hour_end` (last M15 candle of the H1 hour) —
the mapper for every TIMING value. Never the price-extreme mapper.

Asserts (raise, do not degrade — §17.7): every `(S,C)` that has a
`BOS_CONFIRMED` has a `CTS_ESTABLISHED`; `BOS_CONFIRMED(S,C).meta["confirmed_at"]
== CTS_ESTABLISHED(S,C).meta["confirmed_at"]` (the definitional identity — NEVER
against the CTS anchor `meta["cts_anchor_idx"]`, the extreme, which can precede
the apply candle; `CTS_ESTABLISHED.idx` is the moment since Plan E E4a); every
LOH map returns non-None.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from engine_v2.zones.structure_lifecycle import (
    compute_reversal_idx_by_sid,
    compute_struct_start_by_sid,
)

Cycle = Tuple[int, int]


@dataclass(frozen=True)
class ParentTables:
    rev_by_sid: Dict[int, int]
    struct_start: Dict[int, int]
    cts_moment: Dict[Cycle, int]
    parent_sd: Dict[int, int]
    floor_h1: Dict[Cycle, int]
    end_h1: Dict[Cycle, Optional[int]]
    floor_m15: Dict[Cycle, int]
    end_m15: Dict[Cycle, Optional[int]]
    degenerate: Dict[Cycle, bool]

    def floor(self, parent_sid: int, parent_cycle_id: int) -> int:
        """Entity-absolute M15 floor of the cycle (KeyError → no CTS_ESTABLISHED
        for `(S,C)`; callers assert before reaching here)."""
        return self.floor_m15[(int(parent_sid), int(parent_cycle_id))]

    def end(self, parent_sid: int, parent_cycle_id: int) -> Optional[int]:
        return self.end_m15[(int(parent_sid), int(parent_cycle_id))]

    def is_degenerate(self, parent_sid: int, parent_cycle_id: int) -> bool:
        return self.degenerate[(int(parent_sid), int(parent_cycle_id))]

    def has_cycle(self, parent_sid: int, parent_cycle_id: int) -> bool:
        return (int(parent_sid), int(parent_cycle_id)) in self.cts_moment

    def cycles(self) -> List[Cycle]:
        return sorted(self.cts_moment.keys())


def _default_loh(parent_idx: int, h1_df: Any, m15_df: Any) -> Optional[int]:
    from engine_v2.multitf.entity_df_mutation import _map_parent_idx_to_m15_hour_end
    return _map_parent_idx_to_m15_hour_end(int(parent_idx), h1_df, m15_df)


def build_parent_tables(
    sorted_events: List[Any],
    h1_df: Any,
    m15_df: Any,
    *,
    loh: Optional[Callable[[int, Any, Any], Optional[int]]] = None,
    log: Callable[[str], None] = print,
) -> ParentTables:
    """Build the §17.6 static tables from the H1 event stream.

    `loh(parent_idx, h1_df, m15_df) -> Optional[int]` defaults to
    `_map_parent_idx_to_m15_hour_end` (injectable for unit tests with
    synthetic frames). Logs one `[parent_tables]` line per cycle and one
    `WARNING [parent_tables] degenerate parent cycle` per degenerate cycle.
    """
    loh_fn = loh if loh is not None else _default_loh

    rev_by_sid = compute_reversal_idx_by_sid(sorted_events)
    struct_start = compute_struct_start_by_sid(sorted_events, rev_by_sid, None)

    cts_moment: Dict[Cycle, int] = {}
    parent_sd: Dict[int, int] = {}
    bos_moment: Dict[Cycle, int] = {}
    for ev in sorted_events:
        meta = getattr(ev, "meta", None) or {}
        et = getattr(ev, "type", None)
        if et == "CTS_ESTABLISHED":
            sid = meta.get("structure_id")
            cyc = meta.get("cycle_id")
            if sid is None or cyc is None:
                continue
            key = (int(sid), int(cyc))
            # The MOMENT (apply candle), never the CTS anchor `meta["cts_anchor_idx"]` (the extreme). A
            # CTS_ESTABLISHED always carries confirmed_at (market_structure
            # stamps it at emission); assert rather than fall back to the extreme.
            assert meta.get("confirmed_at") is not None, (
                f"[parent_tables] CTS_ESTABLISHED {key} lacks meta['confirmed_at']"
            )
            cts_moment[key] = int(meta["confirmed_at"])          # last-seen wins
            sd = meta.get("struct_direction")
            if sd is not None and int(sd) != 0:
                parent_sd.setdefault(int(sid), int(sd))
        elif et == "BOS_CONFIRMED":
            sid = meta.get("structure_id")
            cyc = meta.get("cycle_id")
            if sid is None or cyc is None:
                continue
            ca = meta.get("confirmed_at")
            if ca is not None:
                bos_moment[(int(sid), int(cyc))] = int(ca)        # last-seen wins

    # Definitional identity: BOS_CONFIRMED.confirmed_at == CTS_ESTABLISHED.confirmed_at
    # (both are the same apply_idx). Every cycle with a BOS has a CTS_ESTABLISHED.
    for key, ca in bos_moment.items():
        assert key in cts_moment, (
            f"[parent_tables] BOS_CONFIRMED {key} has no CTS_ESTABLISHED"
        )
        assert cts_moment[key] == ca, (
            f"[parent_tables] BOS_CONFIRMED{key}.confirmed_at={ca} != "
            f"CTS_ESTABLISHED{key}.confirmed_at={cts_moment[key]} (definitional identity)"
        )

    floor_h1: Dict[Cycle, int] = {}
    for (s, c), moment in cts_moment.items():
        ss = struct_start.get(s)
        floor_h1[(s, c)] = max(int(moment), int(ss)) if ss is not None else int(moment)

    end_h1: Dict[Cycle, Optional[int]] = {}
    for (s, c) in cts_moment:
        nxt = (s, c + 1)
        if nxt in floor_h1:
            end_h1[(s, c)] = floor_h1[nxt]
        elif s in rev_by_sid:
            end_h1[(s, c)] = int(rev_by_sid[s])
        else:
            end_h1[(s, c)] = None

    floor_m15: Dict[Cycle, int] = {}
    end_m15: Dict[Cycle, Optional[int]] = {}
    degenerate: Dict[Cycle, bool] = {}
    for key in sorted(cts_moment):
        fm = loh_fn(floor_h1[key], h1_df, m15_df)
        assert fm is not None, (
            f"[parent_tables] LOH map failed for floor_h1{key}={floor_h1[key]}"
        )
        floor_m15[key] = int(fm)
        eh = end_h1[key]
        if eh is None:
            end_m15[key] = None
        else:
            em = loh_fn(eh, h1_df, m15_df)
            assert em is not None, (
                f"[parent_tables] LOH map failed for end_h1{key}={eh}"
            )
            end_m15[key] = int(em)
        degenerate[key] = end_m15[key] is not None and floor_m15[key] >= end_m15[key]
        log(
            f"[parent_tables] {key} floor_h1={floor_h1[key]} end_h1={end_h1[key]} "
            f"floor_m15={floor_m15[key]} end_m15={end_m15[key]} "
            f"degenerate={degenerate[key]}"
        )
        if degenerate[key]:
            log(
                f"WARNING [parent_tables] degenerate parent cycle {key}: "
                f"floor={floor_m15[key]} end={end_m15[key]}"
            )

    return ParentTables(
        rev_by_sid=dict(rev_by_sid),
        struct_start=dict(struct_start),
        cts_moment=cts_moment,
        parent_sd=parent_sd,
        floor_h1=floor_h1,
        end_h1=end_h1,
        floor_m15=floor_m15,
        end_m15=end_m15,
        degenerate=degenerate,
    )
