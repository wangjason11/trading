from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd

from engine_v2.common.types import PatternEvent, StructureLevel, REQUIRED_CANDLE_COLS
from engine_v2.features.candle_classifier import apply_candle_classification
from engine_v2.patterns.pattern_engine import detect_patterns
from engine_v2.structure.structure_engine import compute_structure
from engine_v2.multitf.registry import StructureRegistry
from engine_v2.multitf.sid_records import (
    build_sid_records_for_main,
    build_sid_records_for_subordinate,
)
from engine_v2.multitf.first_confluence_trigger import (
    detect_first_confluence_triggers,
)
from engine_v2.multitf.subsequent_confluence_trigger import (
    detect_subsequent_confluence_triggers,
)
from engine_v2.multitf.subsequent_counter_trigger import (
    detect_subsequent_counter_triggers,
)

from engine_v2.zones.kl_zones_v1 import derive_kl_zones_v1
from engine_v2.zones.structure_lifecycle import compute_struct_start_by_sid

# Week 7: POI zones
from engine_v2.zones.poi_zones import derive_poi_zones, POIConfig
from engine_v2.patterns.imbalance import compute_imbalance

# Week 8: Wave candle identification
from engine_v2.zones.wave_candles import identify_wave_candles, WaveCandleResult

# Week 8: WVMI
from engine_v2.zones.wvmi import WVMITracker
from engine_v2.zones.zone_proximity import check_zone_proximity
from engine_v2.structure.structure_engine import _pip_size_from_pair

# Week 7: Fib tracking
from engine_v2.zones.fib_tracker import FibTracker, FibTrackerConfig


@dataclass
class PipelineResult:
    df: pd.DataFrame
    patterns: List[PatternEvent]
    structure: List[StructureLevel]
    meta: Dict[str, Any]


def _run_downstream_pipeline(
    df: pd.DataFrame,
    events: list,
    struct_direction: int,
    *,
    source_kinds: Optional[List[str]] = None,
    fib_mode: str = "h1",
    length_threshold: float = 0.7,
    log_prefix: str = "",
    timeframe: str = "H1",
    structure_path_id: str = "H1.main",
    skip_wvmi: bool = False,
    lifecycle_floor: Optional[int] = None,
    lifecycle_cap: Optional[int] = None,
    cap_reason: str = "lifecycle_end",
) -> Dict[str, Any]:
    """Run downstream pipeline (KL zones -> wave candles -> Fib -> POI -> WVMI).

    Extracted from run_pipeline so both H1 and lower-TF pipelines can reuse.

    Returns dict with keys: kl_zones, wave_candles, fib_states, fib_tracker,
    poi_zones, wvmi, wvmi_records, prev_bos_lines, sorted_events,
    zone_proximity_triggers

    `skip_wvmi=True` disables the entity-local zone-proximity gate and the
    WVMI sweep entirely. Used by sub entities (Part 4 §8.3 / §8.4): sub WVMI
    is parent-event-driven and computed by the orchestrator after the sub's
    LowerTFResult is built — see `multitf/sub_wvmi.py`.
    """
    pfx = f"[{log_prefix}]" if log_prefix else ""

    # 5) KL zones consume structure events (not levels).
    #
    # The wave-candles + WVMI computation needs BOTH BOS and CTS zones
    # (CTS wave candles come from the CTS zone — see WVMITracker.on_cts_confirmed).
    # So derive the full set internally; the caller's `source_kinds` filter only
    # narrows what gets RETURNED (and charted). For the H1 main caller
    # `source_kinds=None` ⇒ filter is a no-op. For sub callers
    # `source_kinds=["BOS"]` ⇒ chart sees BOS-only zones, but wave-candles
    # internally still see both kinds (Part 4 §8.3 / §8.4 requirement).
    all_kl_zones = derive_kl_zones_v1(
        df,
        events,
        struct_direction=struct_direction,
        length_threshold=length_threshold,
        source_kinds=None,
        lifecycle_floor=lifecycle_floor,
        lifecycle_cap=lifecycle_cap,
        cap_reason=cap_reason,
    )
    if source_kinds is None:
        kl_zones = all_kl_zones
    else:
        kl_zones = [z for z in all_kl_zones if z.source_kind in source_kinds]

    print(f"{pfx}[kl_zones] total=", len(kl_zones))
    if kl_zones:
        from collections import Counter
        print(f"{pfx}[kl_zones] base_pattern counts:", Counter([z.meta.get("base_pattern") for z in kl_zones]).most_common(10))
        print(f"{pfx}[kl_zones] active buy:", sum(1 for z in kl_zones if z.side=="buy" and z.meta.get("status")=="active"))
        print(f"{pfx}[kl_zones] active sell:", sum(1 for z in kl_zones if z.side=="sell" and z.meta.get("status")=="active"))

    # 5b) Wave candle identification — use the full zone set so we produce
    # both BOS and CTS wave candles even when the caller filters via
    # `source_kinds`.
    wave_candle_results: List[WaveCandleResult] = []
    for zone in all_kl_zones:
        z_sid = zone.meta.get("structure_id")
        z_cycle = zone.meta.get("cycle_id")
        z_sd = zone.meta.get("struct_direction", struct_direction)
        z_anchor = zone.meta.get("anchor_idx")

        if z_sid is None or z_cycle is None or z_anchor is None:
            continue

        result = identify_wave_candles(
            anchor_idx=z_anchor,
            anchor_type=zone.source_kind,
            zone=zone,
            events=events,
            structure_id=z_sid,
            struct_direction=z_sd,
            df=df,
        )
        if result is not None:
            wave_candle_results.append(result)

    wc_with_last = sum(1 for wc in wave_candle_results if wc.last_wave_candle_idx is not None)
    wc_with_first = sum(1 for wc in wave_candle_results if wc.first_wave_candle_idx is not None)
    print(f"{pfx}[wave_candles] total={len(wave_candle_results)}, with_last={wc_with_last}, with_first={wc_with_first}")

    # Sort events by idx (used by WVMI, Fib tracking, prev BOS lines)
    sorted_events = sorted(events, key=lambda e: (e.idx, e.type))

    # Find reversal_confirmed_idx per structure (from REVERSAL_CANDIDATE apply_idx)
    reversal_confirmed_by_sid = {}  # {sid: apply_idx}
    for ev in events:
        if ev.type == "REVERSAL_CANDIDATE":
            prev_sid = ev.meta.get("structure_id", 0)
            apply_idx = ev.meta.get("apply_idx")
            if apply_idx is not None:
                reversal_confirmed_by_sid[prev_sid + 1] = apply_idx

    # 6) Fib tracking
    fib_tracker = FibTracker(
        FibTrackerConfig(
            fib_levels=[30.0, 50.0, 61.8, 80.0],
            fill_threshold=0.70,
        ),
        fib_mode=fib_mode,
    )

    bos_by_cycle = {}  # {(sid, cycle_id): (bos_idx, bos_price)}

    # Track previous structure's direction (for Scenario 1 revert check)
    prev_sd_by_sid = {}  # {sid: prev_sd}
    for ev in events:
        if ev.type == "REVERSAL_CANDIDATE":
            prev_sid = ev.meta.get("structure_id", 0)
            prev_sd = ev.meta.get("struct_direction", 0)
            prev_sd_by_sid[prev_sid + 1] = prev_sd

    def _get_prev_bos_outer(sid: int) -> tuple:
        """Get max expanded outer threshold of prev structure's last BOS zone."""
        prev_sid = sid - 1
        if prev_sid < 0:
            return None, None

        prev_bos_zones = [
            z for z in kl_zones
            if z.meta.get("structure_id") == prev_sid and z.source_kind == "BOS"
        ]
        if not prev_bos_zones:
            return None, None

        last_bos_zone = max(prev_bos_zones, key=lambda z: z.meta.get("cycle_id", 0))
        prev_sd = 1 if last_bos_zone.side == "buy" else -1

        bounds_steps = last_bos_zone.meta.get("bounds_steps", [])
        if not bounds_steps:
            return last_bos_zone.meta.get("outer"), prev_sd

        if prev_sd == 1:
            max_expanded_outer = min(step["bottom"] for step in bounds_steps)
        else:
            max_expanded_outer = max(step["top"] for step in bounds_steps)

        return max_expanded_outer, prev_sd

    for ev in sorted_events:
        sid = ev.meta.get("structure_id", 0)
        cycle_id = ev.meta.get("cycle_id", 0)
        key = (sid, cycle_id)

        if ev.type == "BOS_CONFIRMED":
            bos_by_cycle[key] = (ev.idx, ev.price)

        elif ev.type == "CTS_ESTABLISHED":
            if key in bos_by_cycle:
                bos_idx, bos_price = bos_by_cycle[key]
                reversal_idx = reversal_confirmed_by_sid.get(sid)

                prev_bos_outer, prev_sd = None, None
                if sid >= 1 and cycle_id == 1:
                    prev_bos_outer, prev_sd = _get_prev_bos_outer(sid)

                fib_tracker.on_cts_established(
                    ev, df, bos_idx, bos_price, reversal_idx,
                    prev_bos_outer, prev_sd
                )

        elif ev.type == "CTS_UPDATED":
            reversal_idx = reversal_confirmed_by_sid.get(sid)
            fib_tracker.on_cts_updated(ev, df, reversal_idx)

        elif ev.type == "CTS_CONFIRMED":
            fib_tracker.on_cts_confirmed(ev)

        elif ev.type == "CTS_THRESHOLD_UPDATED":
            # Used by cross_cycle mode to trigger pre-established cross-fib
            # checks. No-op in h1 mode.
            fib_tracker.on_cts_threshold_updated(ev, df)

    # Wire the reversal terminal onto the ended sid's cycles BEFORE finalize so
    # the derived `status` reflects it (FIB_LIFECYCLE_SPEC §7 / Session 2
    # deferral). Runs for the H1 main tracker and each sub tracker.
    fib_tracker.set_reversal_terminals(reversal_confirmed_by_sid)

    # Project the scalar lifecycle axes (start_idx/end_idx/end_reason/status)
    # onto FibState records. `active` is condition-only; terminals live in
    # end_idx/end_reason + derived `status`, which the POI gate and charts read.
    # Passing events + floor/cap (the same ints KL/POI get) puts fib on the
    # shared structure_lifecycle helper (FIB_LIFECYCLE_SPEC §15.6, part b):
    # start_idx clamps to the structure/parent floor and the cycle pass-through
    # end is fed as an earliest-wins terminal candidate. Runs for the H1 main
    # tracker (floor/cap None) and each sub (slice-local floor/cap).
    fib_tracker._finalize_lifecycle_fields(
        events=events,
        lifecycle_floor=lifecycle_floor,
        lifecycle_cap=lifecycle_cap,
        cap_reason=cap_reason,
    )

    fib_states = fib_tracker.get_fibs_for_charting()
    print(f"{pfx}[fib_tracker] total fibs={len(fib_states)}, active={sum(1 for f in fib_states if f.active)}")

    # 7) Prev BOS lines
    prev_bos_lines = []
    last_bos_by_sid = {}
    for ev in sorted_events:
        if ev.type == "BOS_CONFIRMED":
            sid = ev.meta.get("structure_id", 0)
            last_bos_by_sid[sid] = (ev.idx, ev.price)

    for sid, rv_idx in reversal_confirmed_by_sid.items():
        prev_sid = sid - 1
        if prev_sid not in last_bos_by_sid:
            continue

        start_idx, price = last_bos_by_sid[prev_sid]

        end_idx = None
        for ev in sorted_events:
            ev_sid = ev.meta.get("structure_id", 0)
            if ev_sid != sid:
                continue
            if ev.type not in ("CTS_ESTABLISHED", "CTS_UPDATED"):
                continue
            if ev.idx >= rv_idx:
                end_idx = ev.idx
                break

        if end_idx is not None:
            prev_bos_lines.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "price": price,
                "structure_id": sid,
                "prev_structure_id": prev_sid,
            })
            print(f"{pfx}[prev_bos_line] sid={sid}: start_idx={start_idx} end_idx={end_idx} price={price:.5f}")

    # 8) POI zones
    poi_config = POIConfig(
        ic_fib_min=61.8,
        ic_fib_max=80.0,
        v30_threshold=0.30,
        v60_threshold=0.60,
        v90_threshold=0.90,
        fill_threshold=0.70,
    )
    poi_zones = derive_poi_zones(
        df,
        events,
        fib_tracker=fib_tracker,
        config=poi_config,
        lifecycle_floor=lifecycle_floor,
        lifecycle_cap=lifecycle_cap,
        cap_reason=cap_reason,
    )
    print(f"{pfx}[poi_zones] total=", len(poi_zones))

    # 9) WVMI — entity-local proximity-gated. Skipped for sub entities
    # (Part 4 §8.3 / §8.4: sub WVMI is parent-event-driven, computed by
    # the orchestrator from `multitf/sub_wvmi.py` after the sub is built).
    wvmi_records: list = []
    zone_proximity_triggers: Dict[tuple, list] = {}

    if not skip_wvmi:
        wvmi_tracker = WVMITracker(structure_path_id=structure_path_id)

        pip_size = _pip_size_from_pair(df)

        # Zone proximity triggers — alternating sd/opp_sd per cycle.
        # WVMI gate uses only the first sd trigger per cycle (backward-compat
        # for main entity until §13.5 cleanup; sub gating moved to parent
        # events in 3d.iii).
        zone_proximity_triggers = check_zone_proximity(
            df=df,
            sorted_events=sorted_events,
            kl_zones=kl_zones,
            poi_zones=poi_zones,
            pip_size=pip_size,
            timeframe=timeframe,
        )

        proximity_candles: Dict[tuple, dict] = {}
        for key, trigs in zone_proximity_triggers.items():
            if trigs and trigs[0].direction == "sd":
                first_sd = trigs[0]
                proximity_candles[key] = {
                    # Part 4 §8.7 schema: attribution to the trigger event.
                    "triggered_by_event_idx": first_sd.idx,
                    "triggered_by_event_type": "ZONE_PROXIMITY_TRIGGER",
                    "structure_path_id": structure_path_id,
                    "trigger_inner": first_sd.trigger_inner,
                    "proximity_pips": first_sd.proximity_pips,
                }

        for ev in sorted_events:
            if ev.type == "CTS_CONFIRMED":
                sid = ev.meta.get("structure_id", 0)
                cycle_id = ev.meta.get("cycle_id", 0)
                if (sid, cycle_id) in proximity_candles:
                    rec = wvmi_tracker.on_cts_confirmed(ev, df, wave_candle_results, kl_zones)
                    if rec is not None:
                        rec.meta.update(proximity_candles[(sid, cycle_id)])

        for ev in sorted_events:
            if ev.type == "BOS_CONFIRMED":
                wvmi_tracker.on_bos_confirmed(ev, df, wave_candle_results)

        wvmi_tracker.update_temporary_lp(df, kl_zones)

        wvmi_records = wvmi_tracker.get_records()
        print(f"{pfx}[wvmi] total={len(wvmi_records)}, locked={sum(1 for r in wvmi_records if r.lp_locked)}")
    else:
        print(f"{pfx}[wvmi] skipped (parent-event-driven for sub entities)")

    return {
        "kl_zones": kl_zones,
        "wave_candles": wave_candle_results,
        "fib_states": fib_states,
        "fib_tracker": fib_tracker,
        "poi_zones": poi_zones,
        "wvmi_records": wvmi_records,
        "prev_bos_lines": prev_bos_lines,
        "sorted_events": sorted_events,
        "zone_proximity_triggers": zone_proximity_triggers,
    }


def run_pipeline(
    df: pd.DataFrame,
    *,
    lower_timeframes: tuple = (),
) -> PipelineResult:
    """
    Orchestrator (Week 6 KL Zones ordering):

      input df
        -> candle classification (candles_v2 features)
        -> pattern engine
        -> imbalance patterns
        -> market structure (df + structure_events)
        -> KL zones derived from structure confirmation events (base patterns identified on-demand)
        -> attach df.attrs["kl_zones"] for charting

    Zones remain event-driven and do not add rewinds/waits.
    """
    _validate_input(df)
    timing: Dict[str, float] = {}

    # 1) Candle features
    _t0 = time.perf_counter()
    c_res = apply_candle_classification(df)
    timing["candle_features"] = time.perf_counter() - _t0

    # 2) Pattern engine (structure patterns used by market structure)
    _t0 = time.perf_counter()
    p_res = detect_patterns(c_res.df)
    timing["patterns"] = time.perf_counter() - _t0

    # ✅ Debug: confirm structure-pattern markers exist
    print("[patterns]", p_res.notes)
    print(p_res.df["pat"].value_counts().head())

    # 3) Imbalance patterns (Week 7) - columns + instances in df.attrs
    _t0 = time.perf_counter()
    df_with_imbalance = compute_imbalance(p_res.df)
    timing["imbalance"] = time.perf_counter() - _t0
    imbalance_candles = int(df_with_imbalance["is_imbalance"].sum())
    imbalance_instances = len(df_with_imbalance.attrs.get("imbalances", []))
    print(f"[imbalance] candles={imbalance_candles} instances={imbalance_instances}")

    # 4) Market structure (must return events + struct_direction)
    _t0 = time.perf_counter()
    s_res = compute_structure(df_with_imbalance)
    timing["market_structure"] = time.perf_counter() - _t0

    # Propagate imbalance instances through the structure df (so downstream
    # consumers reading s_res.df.attrs see them)
    s_res.df.attrs["imbalances"] = df_with_imbalance.attrs.get("imbalances", [])

    meta: Dict[str, Any] = {
        "notes": {
            "candles": c_res.notes,
            "patterns": p_res.notes,
            "structure": s_res.notes,
        },
        "imbalances": df_with_imbalance.attrs.get("imbalances", []),
    }

    # 5-9) Downstream pipeline (KL zones -> wave candles -> Fib -> POI -> WVMI)
    _t0 = time.perf_counter()
    downstream = _run_downstream_pipeline(
        s_res.df,
        s_res.events,
        s_res.struct_direction,
    )
    timing["downstream"] = time.perf_counter() - _t0

    kl_zones = downstream["kl_zones"]
    wave_candle_results = downstream["wave_candles"]
    fib_states = downstream["fib_states"]
    poi_zones = downstream["poi_zones"]
    wvmi_records = downstream["wvmi_records"]
    prev_bos_lines = downstream["prev_bos_lines"]
    sorted_events = downstream["sorted_events"]
    zone_proximity_triggers = downstream["zone_proximity_triggers"]

    meta["kl_zones"] = kl_zones
    meta["wave_candles"] = wave_candle_results
    meta["fib_states"] = fib_states
    meta["poi_zones"] = poi_zones
    meta["fib_tracker"] = downstream["fib_tracker"]
    meta["wvmi"] = wvmi_records
    meta["prev_bos_lines"] = prev_bos_lines
    meta["zone_proximity_triggers"] = zone_proximity_triggers

    # DEPRECATED (Part 4 transitional): direct df.attrs writes. Step 2
    # routes charts through StructureRegistry; these writes are kept so
    # code paths still reaching into df.attrs (debug exporters, ad-hoc
    # inspection) keep working. Removed in migration plan Step 5.
    s_res.df.attrs["kl_zones"] = kl_zones
    s_res.df.attrs["wave_candles"] = wave_candle_results
    s_res.df.attrs["wvmi"] = wvmi_records
    s_res.df.attrs["poi_zones"] = poi_zones
    s_res.df.attrs["zone_proximity_triggers"] = zone_proximity_triggers
    s_res.df.attrs["structure_events"] = s_res.events
    s_res.df.attrs["fib_states"] = fib_states
    s_res.df.attrs["prev_bos_lines"] = prev_bos_lines

    # Stand up the StructureRegistry. Same df reference as today's
    # monolithic flow, but charts now resolve their data through this.
    registry = StructureRegistry()
    registry.register(
        "H1.main",
        df=s_res.df,
        timeframe="H1",
        role="main",
    )
    meta["registry"] = registry

    # Part 4 Step 3a: per-sid records + first_confluence (var 1) trigger
    # detection. Dormant in 3a — no consumer reads these yet. 3b–3d will
    # build the confluence sub from these triggers and reuse the records
    # for cross-entity lookups.
    _t0 = time.perf_counter()
    main_sids = build_sid_records_for_main(s_res.events)
    s_res.df.attrs["sids"] = main_sids
    print(f"[sid_records] H1.main sids={len(main_sids)}")

    first_confluence_triggers = detect_first_confluence_triggers(
        sorted_events, parent_tf="H1",
    )
    s_res.df.attrs["first_confluence_triggers"] = first_confluence_triggers
    meta["first_confluence_triggers"] = first_confluence_triggers
    fc_pending = sum(1 for t in first_confluence_triggers if t.status == "pending")
    print(
        f"[first_confluence_trigger] detected={len(first_confluence_triggers)} "
        f"pending={fc_pending}"
    )

    # Part 4 Step 3d: subsequent_confluence (var 3) trigger detection.
    # Walks parent zone_proximity_triggers for opp_sd-after-sd patterns
    # within each parent cycle. Reference-zone resolution (§4.3.4) is
    # descriptive only — the probe derives its own BOS_0 internally.
    subsequent_confluence_triggers = detect_subsequent_confluence_triggers(
        sorted_events,
        zone_proximity_triggers,
        s_res.df,
        parent_tf="H1",
    )
    s_res.df.attrs["subsequent_confluence_triggers"] = (
        subsequent_confluence_triggers
    )
    meta["subsequent_confluence_triggers"] = subsequent_confluence_triggers
    print(
        f"[subsequent_confluence_trigger] "
        f"detected={len(subsequent_confluence_triggers)}"
    )

    # Part 4 Step 3d.iv: subsequent_counter (var 4) trigger detection.
    # Walks parent zone_proximity_triggers for sd-after-CTS-after-sd
    # (Λ/V geometry: sd → CTS → sd within a cycle's alternating list).
    # Reference-zone (active parent CTS zone) is descriptive only — the
    # probe derives its own BOS_0 internally.
    subsequent_counter_triggers = detect_subsequent_counter_triggers(
        sorted_events,
        zone_proximity_triggers,
        s_res.df,
        parent_tf="H1",
    )
    s_res.df.attrs["subsequent_counter_triggers"] = (
        subsequent_counter_triggers
    )
    meta["subsequent_counter_triggers"] = subsequent_counter_triggers
    print(
        f"[subsequent_counter_trigger] "
        f"detected={len(subsequent_counter_triggers)}"
    )
    timing["trigger_detectors"] = time.perf_counter() - _t0

    # 10) Multi-TF analysis (if configured)
    lower_tf_results = []
    confluence_results: List[Any] = []
    if lower_timeframes and "M15" in lower_timeframes:
        # Parent-cycle lifecycle-start floors (H1 coords) for the sub lifecycle
        # clamp (PART4 §5; plan B1, 2026-05-27). For each parent (sid, cycle):
        # max(parent structure lifecycle-start, that cycle's CTS_ESTABLISHED idx)
        # — i.e. the parent cycle's clamped lifecycle-start (which already embeds
        # the parent_sid floor). Subs map these H1 idxs to M15 (last-of-hour) and
        # floor their zone/POI activation so a sub cycle/structure never becomes
        # active before its parent cycle is alive.
        _h1_rev_by_sid: Dict[int, int] = {}
        for ev in sorted_events:
            if ev.type == "STATE_CHANGED" and ev.meta.get("to") == "reversal":
                _rsid = int(ev.meta.get("structure_id", 0))
                if _rsid not in _h1_rev_by_sid or int(ev.idx) > _h1_rev_by_sid[_rsid]:
                    _h1_rev_by_sid[_rsid] = int(ev.idx)
        _h1_struct_start = compute_struct_start_by_sid(sorted_events, _h1_rev_by_sid, None)
        parent_cycle_floor_h1: Dict[tuple, int] = {}
        for ev in sorted_events:
            if ev.type == "CTS_ESTABLISHED":
                _csid = int(ev.meta.get("structure_id", 0))
                _ccyc = int(ev.meta.get("cycle_id", 0))
                _sid_floor = _h1_struct_start.get(_csid, int(ev.idx))
                parent_cycle_floor_h1[(_csid, _ccyc)] = max(_sid_floor, int(ev.idx))

        # Two-entity cadence driver (Session 3, 2026-05-31): confluence +
        # counter M15 entities are built INTERLEAVED in trigger-cadence order
        # so each can read the other as sibling. Replaces the former
        # confluence-first-then-counter serial pair (which couldn't satisfy
        # subsequent_confluence's reference to the counter sibling). See
        # `_run_multi_tf_dual` + `build_two_entity_parent_cycle`.
        _t0 = time.perf_counter()
        confluence_results, lower_tf_results = _run_multi_tf_dual(
            s_res.df,
            sorted_events,
            wvmi_records,
            kl_zones,
            poi_zones,
            meta,
            registry,
            first_confluence_triggers=first_confluence_triggers,
            subsequent_confluence_triggers=subsequent_confluence_triggers,
            subsequent_counter_triggers=subsequent_counter_triggers,
            main_zone_proximity_triggers=zone_proximity_triggers,
            parent_cycle_floor_h1=parent_cycle_floor_h1,
        )
        timing["multi_tf_dual"] = time.perf_counter() - _t0

    meta["lower_tf_results"] = lower_tf_results
    meta["first_confluence_results"] = confluence_results
    # §13.5.c.iii: chart consumers (M15 chart + H1 chart's M15 overlay)
    # now read sub data directly from the registered M15 entities'
    # `df.attrs[...]`. The legacy `s_res.df.attrs["lower_tf_results"]`
    # facade-list write has been removed.
    meta["timing"] = timing

    return PipelineResult(
        df=s_res.df,
        patterns=p_res.events,
        structure=s_res.levels,
        meta=meta,
    )


def _confluence_trigger_stream(
    key: tuple,
    main_first_sd_by_cycle: Dict[tuple, int],
    var4_all_sorted: list,
) -> List[Tuple[int, str]]:
    """Parent trigger stream for a confluence parent cycle (§8.3 / §8.5).

    Confluence sub WVMI is initiated at sd-prox-class parent events: the
    main's first sd-prox after CTS, plus each subsequent_counter (var 4) in
    this parent cycle. Returned as `(parent_idx, event_type)` pairs in
    parent-df coords; the caller (`_assign_trigger_centric_sub_wvmi`) sorts and
    maps them to M15.
    """
    stream: List[Tuple[int, str]] = []
    sd_idx = main_first_sd_by_cycle.get(key)
    if sd_idx is not None:
        stream.append((int(sd_idx), "ZONE_PROXIMITY_TRIGGER"))
    for v4 in var4_all_sorted:
        if (v4.parent_sid, v4.parent_cycle_id) == key:
            stream.append(
                (int(v4.trigger_event_idx), "SUBSEQUENT_COUNTER_TRIGGER")
            )
    return stream


def _counter_trigger_stream(
    key: tuple,
    var3_all_sorted: list,
) -> List[Tuple[int, str]]:
    """Parent trigger stream for a counter parent cycle (§8.4 / §8.5).

    Counter sub WVMI is initiated at CTS-prox-class parent events: each
    subsequent_confluence (var 3) in this parent cycle.
    """
    stream: List[Tuple[int, str]] = []
    for v3 in var3_all_sorted:
        if (v3.parent_sid, v3.parent_cycle_id) == key:
            stream.append(
                (int(v3.trigger_event_idx), "SUBSEQUENT_CONFLUENCE_TRIGGER")
            )
    return stream


def _assign_trigger_centric_sub_wvmi(
    results: list,
    trigger_stream: List[Tuple[int, str]],
    *,
    parent_df: pd.DataFrame,
    m15_df: pd.DataFrame,
    sub_path_id: str,
) -> Dict[str, Any]:
    """Trigger-centric sub WVMI gating for one parent cycle (§8.3 / §8.4 / §8.5).

    For each parent trigger (time-ordered), find whichever sub sid in
    `results` is ACTIVE at the trigger moment — active window =
    `[start_trigger_idx, m15_end_idx]` (lifecycle-start to effective end, both
    entity-absolute M15 idx) — and sweep that sid's WVMI once, stamping the
    trigger. A sid touched by multiple triggers is swept once (continuous
    tracker; the sweep already covers all the sid's cycles), so the earliest
    (initiating) trigger wins attribution.

    This is how reversal-born sids are gated — there is NO `use_case` special
    case. A reversal sid is swept iff a parent trigger of the entity's class
    lands inside its window; otherwise it gets no WVMI (correct under the
    parent-driven model). `triggered_by_event_idx` stays in parent-df coords
    (LANDMINE "WVMI Records Carry Mixed-Coordinate Meta") — never translated.

    Returns counts `{"acted", "records", "by_started_by"}`.
    """
    from engine_v2.multitf.entity_df_mutation import (
        _map_parent_idx_to_m15_hour_end,
        persist_facade_wvmi_to_entity_df,
    )
    from engine_v2.multitf.sub_wvmi import (
        ParentTrigger,
        compute_parent_driven_sub_wvmi,
    )

    counts: Dict[str, Any] = {"acted": 0, "records": 0, "by_started_by": {}}
    swept: set = set()
    for parent_idx, event_type in sorted(trigger_stream, key=lambda t: t[0]):
        m15_idx = _map_parent_idx_to_m15_hour_end(
            int(parent_idx), parent_df, m15_df,
        )
        if m15_idx is None:
            continue
        # Active sid = the most-recently-started sid whose
        # [start_trigger_idx, m15_end_idx] window contains the trigger.
        active = None
        best_start = -1
        for res in results:
            start = res.meta.get("start_trigger_idx")
            end = res.meta.get("m15_end_idx")
            if start is None or end is None:
                continue
            if start <= m15_idx <= end and start > best_start:
                active = res
                best_start = start
        if active is None:
            continue
        # Identity tuple (parent_sid, parent_cycle_id, sub_sid) — sweep each
        # sid at most once.
        identity = (
            active.meta.get("parent_sid"),
            active.meta.get("parent_cycle_id"),
            active.meta.get("sub_sid"),
        )
        if identity in swept:
            continue
        recs = compute_parent_driven_sub_wvmi(
            active,
            sub_path_id=sub_path_id,
            parent_trigger=ParentTrigger(
                idx=int(parent_idx),
                event_type=event_type,
                parent_path_id="H1.main",
            ),
        )
        active.wvmi_records = recs
        persist_facade_wvmi_to_entity_df(
            m15_df, active, structure_path_id=sub_path_id,
        )
        swept.add(identity)
        if recs:
            counts["acted"] += 1
            counts["records"] += len(recs)
            sb = active.meta.get("started_by", "?")
            counts["by_started_by"][sb] = (
                counts["by_started_by"].get(sb, 0) + len(recs)
            )
    return counts


def _merge_wvmi_counts(acc: Dict[str, Any], c: Dict[str, Any]) -> None:
    """Accumulate one `_assign_trigger_centric_sub_wvmi` count dict into `acc`."""
    acc["acted"] += c["acted"]
    acc["records"] += c["records"]
    for _sb, _n in c["by_started_by"].items():
        acc["by_started_by"][_sb] = acc["by_started_by"].get(_sb, 0) + _n


def _run_multi_tf_dual(
    h1_df: pd.DataFrame,
    sorted_events: list,
    wvmi_records: list,
    kl_zones: list,
    poi_zones: list,
    meta: Dict[str, Any],
    registry: StructureRegistry,
    *,
    first_confluence_triggers: list,
    subsequent_confluence_triggers: Optional[list] = None,
    subsequent_counter_triggers: Optional[list] = None,
    main_zone_proximity_triggers: Optional[Dict[tuple, list]] = None,
    parent_cycle_floor_h1: Optional[Dict[tuple, int]] = None,
) -> Tuple[list, list]:
    """Build the M15.confluence + M15.counter entities INTERLEAVED in cadence
    order so each can read the other as sibling (Part 4 §6.1 + unified-probe
    Session 3 two-entity driver).

    Replaces the former `_run_first_confluence_multi_tf` (confluence, var1+var3)
    + `_run_multi_tf` (counter, var2+var4) pair, which built each entity fully
    and serially. Serial order cannot satisfy `subsequent_confluence`'s
    reference to the counter sibling (counter wasn't built yet); interleaving
    by cadence resolves every sibling-CTS reference because the read is always
    strictly earlier than the reading sid's own trigger.

    Per parent cycle, `build_two_entity_parent_cycle` advances both chains in
    lock-step. Trigger detection, grouping, the per-cycle trigger-centric sub
    WVMI passes, sid-records, and registry registration are unchanged and run
    per-entity AFTER the builds — WVMI is per-entity-isolated (confluence WVMI
    reads var4 trigger META, not built counter sids, and vice versa), so the
    deferred timing is byte-identical to the serial version.

    Returns ``(confluence_results, counter_results)``.
    """
    from collections import defaultdict

    from engine_v2.multitf.first_confluence_pipeline import (
        to_multi_tf_trigger as first_confluence_to_mt,
    )
    from engine_v2.multitf.subsequent_confluence_pipeline import (
        to_multi_tf_trigger as subsequent_confluence_to_mt,
    )
    from engine_v2.multitf.subsequent_counter_pipeline import (
        to_multi_tf_trigger as subsequent_counter_to_mt,
    )
    from engine_v2.multitf.uc1_trigger import detect_uc1_triggers
    from engine_v2.multitf.data_bridge import (
        fetch_lower_tf_data, prepare_lower_tf_data,
    )
    from engine_v2.multitf.entity_df_mutation import (
        build_two_entity_parent_cycle,
    )

    conf_path = "H1.main >> M15.confluence"
    ctr_path = "H1.main >> M15.counter"

    # --- Trigger grouping inputs ---
    var1_finalized = [
        t for t in (first_confluence_triggers or []) if t.status == "finalized"
    ]
    var3_all = list(subsequent_confluence_triggers or [])
    var2_triggers = detect_uc1_triggers(
        sorted_events, h1_df, wvmi_records, kl_zones,
    )
    var4_all = list(subsequent_counter_triggers or [])

    print(
        f"[multi_tf:dual] confluence var1 finalized="
        f"{len(var1_finalized)}/{len(first_confluence_triggers or [])} "
        f"var3={len(var3_all)} | counter var2={len(var2_triggers)} "
        f"var4={len(var4_all)}"
    )

    if not var1_finalized and not var2_triggers:
        if var3_all or var4_all:
            print("[multi_tf:dual] no bootstraps (var1/var2) — no subs built")
        return [], []

    # --- Prepare BOTH M15 entity dfs from one raw fetch (frame-aligned) ---
    pair = h1_df.attrs.get("pair", "NZD_USD")
    h1_start = pd.to_datetime(h1_df["time"].iloc[0], utc=True)
    h1_end = pd.to_datetime(h1_df["time"].iloc[-1], utc=True)
    m15_raw = fetch_lower_tf_data(pair, "M15", h1_start, h1_end)
    if m15_raw is None or m15_raw.empty:
        print("[multi_tf:dual] WARNING: No M15 data available")
        return [], []
    conf_m15 = prepare_lower_tf_data(m15_raw.copy())
    conf_m15.attrs["pair"] = pair
    ctr_m15 = prepare_lower_tf_data(m15_raw.copy())
    ctr_m15.attrs["pair"] = pair
    if var2_triggers:
        # Preserve the legacy `meta["m15_df_prepared"]` write (set only when the
        # counter entity builds, matching the old `_run_multi_tf`).
        meta["m15_df_prepared"] = ctr_m15
    print(f"[multi_tf:dual] M15 data prepared: {len(conf_m15)} candles x2")

    # --- Group bootstraps + subs by parent cycle ---
    conf_bootstrap_by_cycle: Dict[tuple, Any] = {}
    for v1 in var1_finalized:
        mt = first_confluence_to_mt(v1, h1_df)
        conf_bootstrap_by_cycle[(mt.parent_sid, mt.parent_cycle_id)] = mt
    conf_subs_by_cycle: Dict[tuple, list] = defaultdict(list)
    for v3 in var3_all:
        mt = subsequent_confluence_to_mt(v3, h1_df)
        conf_subs_by_cycle[(mt.parent_sid, mt.parent_cycle_id)].append(mt)

    ctr_bootstrap_by_cycle: Dict[tuple, Any] = {}
    for v2 in var2_triggers:
        ctr_bootstrap_by_cycle[(v2.parent_sid, v2.parent_cycle_id)] = v2
    ctr_subs_by_cycle: Dict[tuple, list] = defaultdict(list)
    for v4 in var4_all:
        mt = subsequent_counter_to_mt(v4, h1_df)
        ctr_subs_by_cycle[(mt.parent_sid, mt.parent_cycle_id)].append(mt)

    # --- Build phase: interleave both entities per parent cycle in
    #     chronological order. (parent_sid, parent_cycle_id) lexicographic ==
    #     chronological (sids increase over time, cycles within a sid). ---
    all_cycle_keys = sorted(
        set(conf_bootstrap_by_cycle) | set(ctr_bootstrap_by_cycle)
    )
    conf_results_by_cycle: Dict[tuple, list] = {}
    ctr_results_by_cycle: Dict[tuple, list] = {}
    for key in all_cycle_keys:
        conf_bs = conf_bootstrap_by_cycle.get(key)
        ctr_bs = ctr_bootstrap_by_cycle.get(key)
        print(
            f"[multi_tf:dual] cycle parent_sid={key[0]} parent_cycle={key[1]} "
            f"conf={'Y' if conf_bs else '-'}"
            f"(subs={len(conf_subs_by_cycle.get(key, []))}) "
            f"ctr={'Y' if ctr_bs else '-'}"
            f"(subs={len(ctr_subs_by_cycle.get(key, []))})"
        )
        cr, kr = build_two_entity_parent_cycle(
            h1_df,
            conf_df=conf_m15,
            conf_bootstrap=conf_bs,
            conf_subs=conf_subs_by_cycle.get(key, []),
            conf_path=conf_path,
            ctr_df=ctr_m15,
            ctr_bootstrap=ctr_bs,
            ctr_subs=ctr_subs_by_cycle.get(key, []),
            ctr_path=ctr_path,
            parent_cycle_floor_h1=parent_cycle_floor_h1,
        )
        conf_results_by_cycle[key] = cr
        ctr_results_by_cycle[key] = kr

    # --- WVMI gating lookups ---
    main_zpt = main_zone_proximity_triggers or {}
    main_first_sd_by_cycle: Dict[tuple, int] = {}
    for k, trig_list in main_zpt.items():
        if trig_list and trig_list[0].direction == "sd":
            main_first_sd_by_cycle[k] = int(trig_list[0].idx)
    var4_all_sorted = sorted(var4_all, key=lambda t: t.trigger_event_idx)
    var3_all_sorted = sorted(var3_all, key=lambda t: t.trigger_event_idx)

    # --- Assemble confluence (legacy order: var1 trigger_event_idx) + WVMI ---
    confluence_results: list = []
    conf_wvmi: Dict[str, Any] = {"acted": 0, "records": 0, "by_started_by": {}}
    conf_cycle_order = sorted(
        conf_bootstrap_by_cycle.keys(),
        key=lambda k: conf_bootstrap_by_cycle[k].meta.get("trigger_event_idx", 0),
    )
    for key in conf_cycle_order:
        cycle_results = conf_results_by_cycle.get(key, [])
        stream = _confluence_trigger_stream(
            key, main_first_sd_by_cycle, var4_all_sorted,
        )
        c = _assign_trigger_centric_sub_wvmi(
            cycle_results, stream,
            parent_df=h1_df, m15_df=conf_m15, sub_path_id=conf_path,
        )
        _merge_wvmi_counts(conf_wvmi, c)
        confluence_results.extend(cycle_results)

    if confluence_results:
        sub_sids = build_sid_records_for_subordinate(confluence_results)
        conf_m15.attrs["sids"] = sub_sids
        print(f"[sid_records] {conf_path} sids={len(sub_sids)}")
        registry.register(
            conf_path, df=conf_m15, timeframe="M15",
            role="subordinate", starting_alignment="confluence",
        )
    print(
        f"[multi_tf:dual] confluence results: {len(confluence_results)} sids; "
        f"wvmi acted={conf_wvmi['acted']} records={conf_wvmi['records']} "
        f"by_started_by={conf_wvmi['by_started_by']}"
    )

    # --- Assemble counter (legacy order: var2 probe_end_idx) + WVMI ---
    counter_results: list = []
    ctr_wvmi: Dict[str, Any] = {"acted": 0, "records": 0, "by_started_by": {}}
    ctr_cycle_order = sorted(
        ctr_bootstrap_by_cycle.keys(),
        key=lambda k: ctr_bootstrap_by_cycle[k].meta.get("probe_end_idx", 0) or 0,
    )
    for key in ctr_cycle_order:
        cycle_results = ctr_results_by_cycle.get(key, [])
        stream = _counter_trigger_stream(key, var3_all_sorted)
        c = _assign_trigger_centric_sub_wvmi(
            cycle_results, stream,
            parent_df=h1_df, m15_df=ctr_m15, sub_path_id=ctr_path,
        )
        _merge_wvmi_counts(ctr_wvmi, c)
        counter_results.extend(cycle_results)

    if counter_results:
        sub_sids = build_sid_records_for_subordinate(counter_results)
        ctr_m15.attrs["sids"] = sub_sids
        print(f"[sid_records] {ctr_path} sids={len(sub_sids)}")
        registry.register(
            ctr_path, df=ctr_m15, timeframe="M15",
            role="subordinate", starting_alignment="counter",
        )
    print(
        f"[multi_tf:dual] counter results: {len(counter_results)} sids; "
        f"wvmi acted={ctr_wvmi['acted']} records={ctr_wvmi['records']} "
        f"by_started_by={ctr_wvmi['by_started_by']}"
    )

    return confluence_results, counter_results


def _validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_CANDLE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[pipeline] Missing required columns: {missing}")
    if df.empty:
        raise ValueError("[pipeline] Input df is empty")
