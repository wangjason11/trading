from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

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
    )
    if source_kinds is None:
        kl_zones = all_kl_zones
    else:
        kl_zones = [z for z in all_kl_zones if z.source_kind in source_kinds]

    print(f"{pfx}[kl_zones] total=", len(kl_zones))
    if kl_zones:
        from collections import Counter
        print(f"{pfx}[kl_zones] base_pattern counts:", Counter([z.meta.get("base_pattern") for z in kl_zones]).most_common(10))
        print(f"{pfx}[kl_zones] active buy:", sum(1 for z in kl_zones if z.side=="buy" and z.meta.get("active")))
        print(f"{pfx}[kl_zones] active sell:", sum(1 for z in kl_zones if z.side=="sell" and z.meta.get("active")))

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
        _t0 = time.perf_counter()
        lower_tf_results = _run_multi_tf(
            s_res.df,
            sorted_events,
            wvmi_records,
            kl_zones,
            poi_zones,
            meta,
            registry,
            subsequent_counter_triggers=subsequent_counter_triggers,
        )
        timing["multi_tf_counter"] = time.perf_counter() - _t0

        # Part 4 Step 3b/3d: confluence subs (var 1 + var 3).
        # Built independently of first_counter — gets its own M15 entity df.
        # Var 1 sids land first; var 3 sids appended afterward (the spec's
        # in-place overwrite semantics from §6.1 are deferred to a later
        # substep — see subsequent_confluence_pipeline.py docstring).
        _t0 = time.perf_counter()
        confluence_results = _run_first_confluence_multi_tf(
            s_res.df,
            first_confluence_triggers,
            registry,
            subsequent_confluence_triggers=subsequent_confluence_triggers,
            subsequent_counter_triggers=subsequent_counter_triggers,
            main_zone_proximity_triggers=zone_proximity_triggers,
        )
        timing["multi_tf_confluence"] = time.perf_counter() - _t0

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


def _confluence_wvmi_for_facade(
    facade,
    *,
    main_first_sd_by_cycle: Dict[tuple, int],
    var4_all_sorted: list,
    sub_path_id: str,
):
    """Compute parent-driven sub WVMI for one confluence facade per §8.3.

    Var 1 sid: gate on main's first sd-prox in same parent cycle.
    Var 3 sid: gate on first var 4 trigger AFTER this var 3 in same
    parent cycle (skip if none — see LANDMINES "Skip when no later
    gate exists is intentional").

    Returns the records list (already attached to facade.wvmi_records by
    caller) for activation/count reporting.
    """
    from engine_v2.multitf.sub_wvmi import (
        ParentTrigger,
        compute_parent_driven_sub_wvmi,
    )

    key = (facade.trigger.parent_sid, facade.trigger.parent_cycle_id)
    if facade.trigger.use_case == "first_confluence":
        sd_idx = main_first_sd_by_cycle.get(key)
        if sd_idx is None:
            return []
        return compute_parent_driven_sub_wvmi(
            facade,
            sub_path_id=sub_path_id,
            parent_trigger=ParentTrigger(
                idx=sd_idx,
                event_type="ZONE_PROXIMITY_TRIGGER",
                parent_path_id="H1.main",
            ),
        )
    if facade.trigger.use_case == "subsequent_confluence":
        after_idx = int(facade.trigger.meta.get("trigger_event_idx", -1))
        for v4 in var4_all_sorted:
            if (v4.parent_sid == facade.trigger.parent_sid
                    and v4.parent_cycle_id == facade.trigger.parent_cycle_id
                    and int(v4.trigger_event_idx) > after_idx):
                return compute_parent_driven_sub_wvmi(
                    facade,
                    sub_path_id=sub_path_id,
                    parent_trigger=ParentTrigger(
                        idx=int(v4.trigger_event_idx),
                        event_type="SUBSEQUENT_COUNTER_TRIGGER",
                        parent_path_id="H1.main",
                    ),
                )
        return []
    return []


def _counter_wvmi_for_facade(
    facade,
    *,
    v3_first_idx_by_cycle: Dict[tuple, int],
    var3_all_sorted: list,
    sub_path_id: str,
):
    """Compute parent-driven sub WVMI for one counter facade per §8.4.

    Var 2 sid: gate on first var 3 trigger in same parent cycle.
    Var 4 sid: gate on first var 3 trigger AFTER this var 4 in same
    parent cycle (skip if none).
    """
    from engine_v2.multitf.sub_wvmi import (
        ParentTrigger,
        compute_parent_driven_sub_wvmi,
    )

    if facade.trigger.use_case == "first_counter":
        key = (facade.trigger.parent_sid, facade.trigger.parent_cycle_id)
        v3_idx = v3_first_idx_by_cycle.get(key)
        if v3_idx is None:
            return []
        return compute_parent_driven_sub_wvmi(
            facade,
            sub_path_id=sub_path_id,
            parent_trigger=ParentTrigger(
                idx=v3_idx,
                event_type="SUBSEQUENT_CONFLUENCE_TRIGGER",
                parent_path_id="H1.main",
            ),
        )
    if facade.trigger.use_case == "subsequent_counter":
        after_idx = int(facade.trigger.meta.get("trigger_event_idx", -1))
        for v3 in var3_all_sorted:
            if (v3.parent_sid == facade.trigger.parent_sid
                    and v3.parent_cycle_id == facade.trigger.parent_cycle_id
                    and int(v3.trigger_event_idx) > after_idx):
                return compute_parent_driven_sub_wvmi(
                    facade,
                    sub_path_id=sub_path_id,
                    parent_trigger=ParentTrigger(
                        idx=int(v3.trigger_event_idx),
                        event_type="SUBSEQUENT_CONFLUENCE_TRIGGER",
                        parent_path_id="H1.main",
                    ),
                )
        return []
    return []


def _run_first_confluence_multi_tf(
    h1_df: pd.DataFrame,
    triggers: list,
    registry: StructureRegistry,
    subsequent_confluence_triggers: Optional[list] = None,
    subsequent_counter_triggers: Optional[list] = None,
    main_zone_proximity_triggers: Optional[Dict[tuple, list]] = None,
) -> list:
    """Build confluence subs (var 1 bootstrap + var 3 subsequents) and
    register M15.confluence.

    Phase 2 (merge-and-bound, §6.1 REVISED 2026-05-25): triggers are grouped
    by `(parent_sid, parent_cycle_id)`. Each cycle's var 1 (finalized) is the
    sid=0 bootstrap; its var 3 triggers are the subsequents. The
    `build_parent_cycle_chain` driver stitches a sequential, non-overlapping
    sid chain per cycle (bounded single-structure sids; reversal or next
    subsequent starts the next sid). No cascade, no last-per-cycle carve-out —
    every var 3 builds as its own sid, falling out of the merge-and-bound
    build for free (old §13.5.d subsumed). Pending var 1 → that cycle has no
    sid=0, so its var 3 don't build (§4.3.2 / §14).

    `subsequent_counter_triggers` is the FULL var 4 list, used by §8.3 path 2
    to gate var 3 sub WVMI (the `wvmi_hook` below).
    """
    from collections import defaultdict

    from engine_v2.multitf.first_confluence_pipeline import (
        to_multi_tf_trigger as first_confluence_to_mt,
    )
    from engine_v2.multitf.subsequent_confluence_pipeline import (
        to_multi_tf_trigger as subsequent_confluence_to_mt,
    )
    from engine_v2.multitf.data_bridge import (
        fetch_lower_tf_data,
        prepare_lower_tf_data,
    )
    from engine_v2.multitf.entity_df_mutation import (
        build_parent_cycle_chain,
        persist_facade_wvmi_to_entity_df,
    )

    var1_finalized = [t for t in (triggers or []) if t.status == "finalized"]
    var3_all = list(subsequent_confluence_triggers or [])

    if not var1_finalized:
        # No bootstrap anywhere → no confluence subs (var 3 can't be sid 0).
        if triggers or var3_all:
            print(f"[multi_tf:confluence] no finalized var 1 bootstrap "
                  f"({len(triggers or [])} var1, {len(var3_all)} var3) — "
                  f"no subs built")
        return []

    print(f"[multi_tf:confluence] var 1 finalized: "
          f"{len(var1_finalized)}/{len(triggers or [])}, var 3 total: "
          f"{len(var3_all)} (all built, merge-and-bound)")

    pair = h1_df.attrs.get("pair", "NZD_USD")
    h1_start = pd.to_datetime(h1_df["time"].iloc[0], utc=True)
    h1_end = pd.to_datetime(h1_df["time"].iloc[-1], utc=True)

    m15_raw = fetch_lower_tf_data(pair, "M15", h1_start, h1_end)
    if m15_raw is None or m15_raw.empty:
        print("[multi_tf:confluence] WARNING: No M15 data available")
        return []

    m15_df = prepare_lower_tf_data(m15_raw)
    m15_df.attrs["pair"] = pair

    sub_path_id = "H1.main >> M15.confluence"

    # Group by parent cycle: one var 1 bootstrap per cycle, var 3 subsequents.
    bootstrap_by_cycle: Dict[tuple, Any] = {}
    for v1 in var1_finalized:
        mt = first_confluence_to_mt(v1, h1_df)
        bootstrap_by_cycle[(mt.parent_sid, mt.parent_cycle_id)] = mt
    subs_by_cycle: Dict[tuple, list] = defaultdict(list)
    for v3 in var3_all:
        mt = subsequent_confluence_to_mt(v3, h1_df)
        subs_by_cycle[(mt.parent_sid, mt.parent_cycle_id)].append(mt)

    # WVMI gating lookups (precomputed once)
    main_zpt = main_zone_proximity_triggers or {}
    main_first_sd_by_cycle: Dict[tuple, int] = {}
    for key, trig_list in main_zpt.items():
        if trig_list and trig_list[0].direction == "sd":
            main_first_sd_by_cycle[key] = int(trig_list[0].idx)
    var4_all_sorted = sorted(
        list(subsequent_counter_triggers or []),
        key=lambda t: t.trigger_event_idx,
    )

    # Per-built-sid WVMI hook (parent-driven, §8.3). Self-skips
    # use_case="reversal" via _confluence_wvmi_for_facade. Reversal-born sid
    # WVMI gating is a Phase 5 concern.
    counts = {"var1_act": 0, "var1_rec": 0, "var3_act": 0, "var3_rec": 0}

    def wvmi_hook(result, entity_sid: int) -> None:
        recs = _confluence_wvmi_for_facade(
            result,
            main_first_sd_by_cycle=main_first_sd_by_cycle,
            var4_all_sorted=var4_all_sorted,
            sub_path_id=sub_path_id,
        )
        result.wvmi_records = recs
        persist_facade_wvmi_to_entity_df(
            m15_df, result, new_sid_id=entity_sid,
            structure_path_id=sub_path_id,
        )
        if recs and result.trigger.use_case == "first_confluence":
            counts["var1_act"] += 1
            counts["var1_rec"] += len(recs)
        elif recs and result.trigger.use_case == "subsequent_confluence":
            counts["var3_act"] += 1
            counts["var3_rec"] += len(recs)

    # Process cycles in time order of the bootstrap's trigger candle.
    cycle_keys = sorted(
        bootstrap_by_cycle.keys(),
        key=lambda k: bootstrap_by_cycle[k].meta.get("trigger_event_idx", 0),
    )

    results: list = []
    next_entity_sid = 0
    for key in cycle_keys:
        bootstrap = bootstrap_by_cycle[key]
        subs = subs_by_cycle.get(key, [])
        print(f"[multi_tf:confluence] cycle parent_sid={key[0]} "
              f"parent_cycle={key[1]} subsequents={len(subs)}")
        cycle_results, next_entity_sid = build_parent_cycle_chain(
            m15_df, h1_df,
            bootstrap=bootstrap, subsequents=subs,
            sub_path_id=sub_path_id, first_entity_sid=next_entity_sid,
            wvmi_hook=wvmi_hook,
        )
        results.extend(cycle_results)

    print(f"[multi_tf:confluence] results: {len(results)} sids "
          f"(var1+var3+reversal, merge-and-bound)")
    print(f"[multi_tf:confluence_wvmi] activated var1 cycles="
          f"{counts['var1_act']} total records={counts['var1_rec']}; "
          f"var3 cycles={counts['var3_act']} total records={counts['var3_rec']}")

    if results:
        # Chart consumes m15_df.attrs[...] (events, kl_zones, poi_zones,
        # fib_states, wave_candles, wvmi, prev_bos_lines) populated by the
        # chain build per sid.
        sub_sids = build_sid_records_for_subordinate(results)
        m15_df.attrs["sids"] = sub_sids
        print(f"[sid_records] {sub_path_id} sids={len(sub_sids)}")
        registry.register(
            sub_path_id,
            df=m15_df,
            timeframe="M15",
            role="subordinate",
            starting_alignment="confluence",
        )

    return results


def _run_multi_tf(
    h1_df: pd.DataFrame,
    sorted_events: list,
    wvmi_records: list,
    kl_zones: list,
    poi_zones: list,
    meta: Dict[str, Any],
    registry: StructureRegistry,
    subsequent_counter_triggers: Optional[list] = None,
) -> list:
    """Run multi-TF analysis for the M15.counter entity (var 2 + var 4).

    Phase 2 (merge-and-bound, §6.1 REVISED 2026-05-25): triggers are grouped
    by `(parent_sid, parent_cycle_id)`. Each cycle's var 2 (first_counter) is
    the sid=0 bootstrap; its var 4 triggers are the subsequents. The
    `build_parent_cycle_chain` driver stitches a sequential, non-overlapping
    sid chain per cycle. No cascade, no var 4 last-per-cycle carve-out — every
    var 4 builds as its own sid (old §13.5.d subsumed). A cycle with no var 2
    bootstrap doesn't build (var 4 can't be sid 0).
    """
    from collections import defaultdict

    from engine_v2.multitf.uc1_trigger import detect_uc1_triggers
    from engine_v2.multitf.data_bridge import fetch_lower_tf_data, prepare_lower_tf_data
    from engine_v2.multitf.subsequent_counter_pipeline import (
        to_multi_tf_trigger as subsequent_counter_to_mt,
    )
    from engine_v2.multitf.entity_df_mutation import (
        build_parent_cycle_chain,
        persist_facade_wvmi_to_entity_df,
    )

    var2_triggers = detect_uc1_triggers(
        sorted_events, h1_df, wvmi_records, kl_zones,
    )
    print(f"[multi_tf] first_counter triggers detected: {len(var2_triggers)}")

    var4_all = list(subsequent_counter_triggers or [])
    print(f"[multi_tf] subsequent_counter triggers (all built): {len(var4_all)}")

    if not var2_triggers:
        # No bootstrap anywhere → no counter subs (var 4 can't be sid 0).
        if var4_all:
            print(f"[multi_tf] no var 2 bootstrap ({len(var4_all)} var4) — "
                  f"no counter subs built")
        return []

    # Fetch M15 entity data once
    pair = h1_df.attrs.get("pair", "NZD_USD")
    h1_start = pd.to_datetime(h1_df["time"].iloc[0], utc=True)
    h1_end = pd.to_datetime(h1_df["time"].iloc[-1], utc=True)

    m15_df_raw = fetch_lower_tf_data(pair, "M15", h1_start, h1_end)
    if m15_df_raw is None or m15_df_raw.empty:
        print("[multi_tf] WARNING: No M15 data available, skipping multi-TF")
        return []

    m15_df_prepared = prepare_lower_tf_data(m15_df_raw)
    m15_df_prepared.attrs["pair"] = pair
    meta["m15_df_prepared"] = m15_df_prepared
    print(f"[multi_tf] M15 data prepared: {len(m15_df_prepared)} candles")

    sub_path_id = "H1.main >> M15.counter"

    # Group by parent cycle: one var 2 bootstrap per cycle, var 4 subsequents.
    bootstrap_by_cycle: Dict[tuple, Any] = {}
    for v2 in var2_triggers:
        bootstrap_by_cycle[(v2.parent_sid, v2.parent_cycle_id)] = v2
    subs_by_cycle: Dict[tuple, list] = defaultdict(list)
    for v4 in var4_all:
        mt = subsequent_counter_to_mt(v4, h1_df)
        subs_by_cycle[(mt.parent_sid, mt.parent_cycle_id)].append(mt)

    # WVMI gating lookups
    var3_triggers = meta.get("subsequent_confluence_triggers", [])
    v3_first_idx_by_cycle: Dict[tuple, int] = {}
    for t in var3_triggers:
        key = (t.parent_sid, t.parent_cycle_id)
        if key not in v3_first_idx_by_cycle:
            v3_first_idx_by_cycle[key] = int(t.trigger_event_idx)
    var3_all_sorted = sorted(
        list(var3_triggers or []),
        key=lambda t: t.trigger_event_idx,
    )

    counts = {"var2_act": 0, "var2_rec": 0, "var4_act": 0, "var4_rec": 0}

    def wvmi_hook(result, entity_sid: int) -> None:
        recs = _counter_wvmi_for_facade(
            result,
            v3_first_idx_by_cycle=v3_first_idx_by_cycle,
            var3_all_sorted=var3_all_sorted,
            sub_path_id=sub_path_id,
        )
        result.wvmi_records = recs
        persist_facade_wvmi_to_entity_df(
            m15_df_prepared, result, new_sid_id=entity_sid,
            structure_path_id=sub_path_id,
        )
        if recs and result.trigger.use_case == "first_counter":
            counts["var2_act"] += 1
            counts["var2_rec"] += len(recs)
        elif recs and result.trigger.use_case == "subsequent_counter":
            counts["var4_act"] += 1
            counts["var4_rec"] += len(recs)

    # var 2 trigger candle for time-ordering = probe_end_idx (first sd-prox).
    cycle_keys = sorted(
        bootstrap_by_cycle.keys(),
        key=lambda k: bootstrap_by_cycle[k].meta.get("probe_end_idx", 0) or 0,
    )

    lower_tf_results: list = []
    next_entity_sid = 0
    for key in cycle_keys:
        bootstrap = bootstrap_by_cycle[key]
        subs = subs_by_cycle.get(key, [])
        print(f"[multi_tf] cycle parent_sid={key[0]} parent_cycle={key[1]} "
              f"subsequents={len(subs)}")
        cycle_results, next_entity_sid = build_parent_cycle_chain(
            m15_df_prepared, h1_df,
            bootstrap=bootstrap, subsequents=subs,
            sub_path_id=sub_path_id, first_entity_sid=next_entity_sid,
            wvmi_hook=wvmi_hook,
        )
        lower_tf_results.extend(cycle_results)

    print(f"[multi_tf] counter results: {len(lower_tf_results)} sids "
          f"(var2+var4+reversal, merge-and-bound)")
    print(f"[multi_tf:counter_wvmi] activated var2 cycles={counts['var2_act']} "
          f"total records={counts['var2_rec']}; var4 cycles={counts['var4_act']} "
          f"total records={counts['var4_rec']}")

    if lower_tf_results:
        # Chart consumes m15_df_prepared.attrs[...] populated by the chain
        # build per sid.
        sub_sids = build_sid_records_for_subordinate(lower_tf_results)
        m15_df_prepared.attrs["sids"] = sub_sids
        print(f"[sid_records] {sub_path_id} sids={len(sub_sids)}")
        registry.register(
            sub_path_id,
            df=m15_df_prepared,
            timeframe="M15",
            role="subordinate",
            starting_alignment="counter",
        )

    return lower_tf_results


def _validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_CANDLE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[pipeline] Missing required columns: {missing}")
    if df.empty:
        raise ValueError("[pipeline] Input df is empty")
