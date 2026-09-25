from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd

from engine_v2.common.types import PatternEvent, StructureLevel, REQUIRED_CANDLE_COLS
from engine_v2.structure import event_fields as ef
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


def _prev_bos_lines(sorted_events: list, reversal_confirmed_by_sid: dict, pfx: str = "") -> list:
    """The previous structure's last BOS, drawn from its ANCHOR (START) to the
    ANCHOR of the first CTS of the next structure known at/after the reversal
    (END; PLAN_E Q6). Extracted from `_run_downstream_pipeline` (Plan E E2c
    landing review) so the START / END roles are unit-pinned."""
    prev_bos_lines = []
    last_bos_anchor_by_sid = {}
    for ev in sorted_events:
        if ev.type == "BOS_CONFIRMED":
            sid = ev.meta.get("structure_id", 0)
            last_bos_anchor_by_sid[sid] = (ef.bos_anchor_idx(ev), ev.price)   # the line START (location)

    for sid, rv_idx in reversal_confirmed_by_sid.items():
        prev_sid = sid - 1
        if prev_sid not in last_bos_anchor_by_sid:
            continue

        start_idx, price = last_bos_anchor_by_sid[prev_sid]

        # "The first CTS of sid known at/after the reversal": a TIME — the
        # earliest MOMENT among sid's CTS_ESTABLISHED / CTS_UPDATED with moment
        # >= the reversal (Plan E E3d; a pattern-path CTS_UPDATED's since E3·0).
        # Picked by moment, not by processing order: a pattern-path update that
        # regresses the CTS (zones-audit latent bug (a)) is stamped before a
        # raw update it is known after (E3c/E3d landing review). `min` is stable
        # on ties. The line END is the winner's CTS anchor (a location, Q6).
        qualifying = [
            ev for ev in sorted_events
            if ev.meta.get("structure_id", 0) == sid
            and ev.type in ("CTS_ESTABLISHED", "CTS_UPDATED")
            and ef.event_moment(ev) >= rv_idx
        ]
        end_idx = ef.cts_anchor_idx(min(qualifying, key=ef.event_moment)) if qualifying else None

        if end_idx is not None:
            prev_bos_lines.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "price": price,
                "structure_id": sid,
                "prev_structure_id": prev_sid,
            })
            print(f"{pfx}[prev_bos_line] sid={sid}: start_idx={start_idx} end_idx={end_idx} price={price:.5f}")
    return prev_bos_lines


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

    # The event processing order (used by WVMI, Fib tracking, prev BOS lines):
    # today's (idx, type), pinned against the Plan E E4 flip (PLAN_E Q3; LANDMINES
    # "Event Sort Order Is a Dispatch Invariant").
    sorted_events = sorted(events, key=ef.processing_order_key)

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

    bos_anchor_by_cycle = {}  # {(sid, cycle_id): (bos_anchor_idx, bos_price)}

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
            bos_anchor_by_cycle[key] = (ef.bos_anchor_idx(ev), ev.price)   # the fib's BOS point (location)

        elif ev.type == "CTS_ESTABLISHED":
            if key in bos_anchor_by_cycle:
                bos_anchor_idx, bos_price = bos_anchor_by_cycle[key]
                reversal_idx = reversal_confirmed_by_sid.get(sid)

                prev_bos_outer, prev_sd = None, None
                # §11b: pass P_rev (prev-BOS-outer) at EVERY cycle for sid >= 1
                # (was cycle 1 only). FibTracker stashes it set-once for the
                # multi-cycle cross target ceiling M. Cycle 1 still uses it for
                # the Scenario-1 revert check exactly as before.
                if sid >= 1:
                    prev_bos_outer, prev_sd = _get_prev_bos_outer(sid)

                fib_tracker.on_cts_established(
                    ev, df, bos_anchor_idx, bos_price, reversal_idx,
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
    prev_bos_lines = _prev_bos_lines(sorted_events, reversal_confirmed_by_sid, pfx)

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
        # Sub-structure pool driver (PART4 §17, Plan C): parent tables + the
        # lifecycle sweep + one projection per unique sub, mirrored per lens.
        # The parent-cycle floors/ends every record uses are computed inside
        # (`multitf/parent_tables.py`, on the CTS-established MOMENT).
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
    parent-df coords; the caller (`_assign_sub_wvmi_per_sub`) sorts and maps
    them to M15.
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


def _assign_sub_wvmi_per_sub(
    sub_results: list,
    streams_by_lens: Dict[str, List[Tuple[int, str]]],
    *,
    parent_df: pd.DataFrame,
    m15_df: pd.DataFrame,
    lens_dfs: Dict[str, pd.DataFrame],
    lens_paths: Dict[str, str],
) -> Dict[str, Any]:
    """Trigger-centric sub WVMI, ONE sweep per unique sub (§17.10 — the user's
    stated lean, implemented minimally so the code runs; the deferred WVMI
    pass may return to per-(sub, lens) sweeps — do not treat as settled).

    For each sub projection: window = `[start_idx, m15_end_idx]` (the sub's
    real-time lifecycle, edge for an open sub); stream = the union of the
    confluence and the counter parent-trigger streams (over ALL parent cycles
    — a sub spans cycles) restricted to the sub's lenses; the FIRST trigger
    (by parent idx, LOH-mapped) inside the window sweeps the sub once
    (`compute_parent_driven_sub_wvmi`), with the sweeping trigger's lens
    deciding the records' `structure_path_id`; the records are persisted into
    every lens df the sub is on. Dedup key `sub_id`. `triggered_by_event_idx`
    stays in parent-df coords (LANDMINE "WVMI Records Carry Mixed-Coordinate
    Meta") — never translated.

    Returns counts `{"acted", "records", "by_started_by", "by_lens"}`.
    """
    from engine_v2.multitf.entity_df_mutation import (
        _map_parent_idx_to_m15_hour_end,
        persist_facade_wvmi_to_entity_df,
    )
    from engine_v2.multitf.sub_wvmi import (
        ParentTrigger,
        compute_parent_driven_sub_wvmi,
    )

    counts: Dict[str, Any] = {
        "acted": 0, "records": 0, "by_started_by": {}, "by_lens": {},
    }
    # LOH-map each stream once: (m15_idx, parent_idx, event_type, lens).
    mapped: List[Tuple[int, int, str, str]] = []
    for lens, stream in streams_by_lens.items():
        for parent_idx, event_type in stream:
            m15_idx = _map_parent_idx_to_m15_hour_end(int(parent_idx), parent_df, m15_df)
            if m15_idx is None:
                continue
            mapped.append((int(m15_idx), int(parent_idx), event_type, lens))
    mapped.sort(key=lambda t: (t[0], t[1]))

    swept: set = set()
    for res in sub_results:
        sub_id = res.meta.get("sub_id")
        start = res.meta.get("start_idx")
        end = res.meta.get("m15_end_idx")
        lenses = set(res.meta.get("lenses") or ())
        if sub_id is None or start is None or end is None or sub_id in swept:
            continue
        hit = next(
            (t for t in mapped if t[3] in lenses and start <= t[0] <= end), None,
        )
        if hit is None:
            continue
        _m15_idx, parent_idx, event_type, lens = hit
        recs = compute_parent_driven_sub_wvmi(
            res,
            sub_path_id=lens_paths[lens],
            parent_trigger=ParentTrigger(
                idx=int(parent_idx),
                event_type=event_type,
                parent_path_id="H1.main",
            ),
        )
        res.wvmi_records = recs
        for l in sorted(lenses):
            persist_facade_wvmi_to_entity_df(
                lens_dfs[l], res, structure_path_id=lens_paths[l],
            )
        swept.add(sub_id)
        if recs:
            counts["acted"] += 1
            counts["records"] += len(recs)
            sb = res.meta.get("started_by", "?")
            counts["by_started_by"][sb] = counts["by_started_by"].get(sb, 0) + len(recs)
            for l in lenses:
                counts["by_lens"][l] = counts["by_lens"].get(l, 0) + len(recs)
    return counts


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
) -> Tuple[list, list]:
    """Build the M15 sub-structure pool and the two lens dfs (PART4 §17,
    Plan C — replaces the Phase-1 two-entity cadence driver):

      1. detect the H1 triggers (var1 first_confluence, var2 first_counter,
         var3 subsequent_confluence, var4 subsequent_counter) and tag each
         with its lens + `trigger_idx = LOH(trigger_event_idx)`;
      2. prepare ONE shared M15 feature frame (`prepare_lower_tf_data` once)
         and two lens dfs (copies — views for the chart/export readers);
      3. `build_parent_tables` (§17.6) from the H1 events;
      4. `run_lifecycle_sweep` (§17.6) with the real resolvers / geometry
         builder / reversal handoff injected — every unique sub + its
         TriggerRecords + the unresolved-trigger log land in the pool;
      5. ONE projection per sub (`render_sub_projection`) mirrored into every
         lens df it belongs to, in `start_idx` order (later-live wins
         overlapping structure columns);
      6. per-sub WVMI (§17.10 minimal);
      7. `attrs["sids"]` (one SidRecord per sub), `attrs["triggers"]` (this
         lens's records incl. zero-length), `attrs["unresolved_triggers"]`
         (pool-wide) on each lens df + registry registration.

    Returns ``(confluence_results, counter_results)`` — the per-sub
    projections rendered on each lens, in `start_idx` order.
    """
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
        _map_parent_idx_to_m15_hour_end,
        _resolve_reversal_start,
        _resolve_trigger_m15_start,
        build_or_get_geometry,
        render_sub_projection,
    )
    from engine_v2.multitf.lifecycle_sweep import SweepTrigger, run_lifecycle_sweep
    from engine_v2.multitf.parent_tables import build_parent_tables
    from engine_v2.multitf.sub_structure_pool import (
        LENS_CONFLUENCE, LENS_COUNTER, SubStructurePool, resolve_lens,
    )

    parent_path = "H1.main"
    lens_paths = {
        LENS_CONFLUENCE: "H1.main >> M15.confluence",
        LENS_COUNTER: "H1.main >> M15.counter",
    }

    # --- 1. Trigger detection ---
    var1_all = list(first_confluence_triggers or [])
    var1_finalized = [t for t in var1_all if t.status == "finalized"]
    var3_all = list(subsequent_confluence_triggers or [])
    var2_triggers = detect_uc1_triggers(
        sorted_events, h1_df, wvmi_records, kl_zones,
    )
    var4_all = list(subsequent_counter_triggers or [])

    print(
        f"[multi_tf:dual] confluence var1 finalized="
        f"{len(var1_finalized)}/{len(var1_all)} "
        f"var3={len(var3_all)} | counter var2={len(var2_triggers)} "
        f"var4={len(var4_all)}"
    )
    if not var1_all and not var2_triggers and not var3_all and not var4_all:
        print("[multi_tf:dual] no triggers — no subs built")
        return [], []

    # --- 2. ONE shared M15 feature frame + the two lens dfs ---
    pair = h1_df.attrs.get("pair", "NZD_USD")
    h1_start = pd.to_datetime(h1_df["time"].iloc[0], utc=True)
    h1_end = pd.to_datetime(h1_df["time"].iloc[-1], utc=True)
    m15_raw = fetch_lower_tf_data(pair, "M15", h1_start, h1_end)
    if m15_raw is None or m15_raw.empty:
        print("[multi_tf:dual] WARNING: No M15 data available")
        return [], []
    m15 = prepare_lower_tf_data(m15_raw)
    m15.attrs["pair"] = pair
    lens_dfs = {
        LENS_CONFLUENCE: m15.copy(),
        LENS_COUNTER: m15.copy(),
    }
    for _df in lens_dfs.values():
        _df.attrs["pair"] = pair
    print(f"[multi_tf:dual] M15 data prepared: {len(m15)} candles (shared frame + 2 lens dfs)")

    # --- 3. Static parent tables (§17.6) ---
    tables = build_parent_tables(sorted_events, h1_df, m15)

    # --- Sweep triggers: lens-tagged + LOH-mapped ---
    def _loh(parent_idx: int) -> int:
        m = _map_parent_idx_to_m15_hour_end(int(parent_idx), h1_df, m15)
        assert m is not None, f"[multi_tf:dual] LOH map failed for H1 idx {parent_idx}"
        return int(m)

    sweep_triggers: List[SweepTrigger] = []
    for v1 in var1_all:
        mt = first_confluence_to_mt(v1, h1_df)
        sweep_triggers.append(SweepTrigger(
            lens=resolve_lens("first_confluence"),
            parent_sid=int(mt.parent_sid), parent_cycle_id=int(mt.parent_cycle_id),
            trigger_type="first_confluence",
            trigger_idx=_loh(v1.trigger_event_idx), direction=int(mt.lower_sd),
            trigger_event_idx=int(v1.trigger_event_idx), source=mt,
            pending=(v1.status != "finalized"),
            probe_input_idx=int(v1.input_idx),
        ))
    for v2 in var2_triggers:
        tei = v2.meta.get("trigger_event_idx")
        assert tei is not None, (
            f"[multi_tf:dual] first_counter sid={v2.parent_sid} cycle={v2.parent_cycle_id} "
            f"has no trigger_event_idx"
        )
        sweep_triggers.append(SweepTrigger(
            lens=resolve_lens("first_counter"),
            parent_sid=int(v2.parent_sid), parent_cycle_id=int(v2.parent_cycle_id),
            trigger_type="first_counter",
            trigger_idx=_loh(tei), direction=int(v2.lower_sd),
            trigger_event_idx=int(tei), source=v2,
            probe_input_idx=v2.meta.get("probe_input_idx"),
        ))
    for v3 in var3_all:
        mt = subsequent_confluence_to_mt(v3, h1_df)
        sweep_triggers.append(SweepTrigger(
            lens=resolve_lens("subsequent_confluence"),
            parent_sid=int(mt.parent_sid), parent_cycle_id=int(mt.parent_cycle_id),
            trigger_type="subsequent_confluence",
            trigger_idx=_loh(v3.trigger_event_idx), direction=int(mt.lower_sd),
            trigger_event_idx=int(v3.trigger_event_idx), source=mt,
            probe_input_idx=int(v3.input_idx),
        ))
    for v4 in var4_all:
        mt = subsequent_counter_to_mt(v4, h1_df)
        sweep_triggers.append(SweepTrigger(
            lens=resolve_lens("subsequent_counter"),
            parent_sid=int(mt.parent_sid), parent_cycle_id=int(mt.parent_cycle_id),
            trigger_type="subsequent_counter",
            trigger_idx=_loh(v4.trigger_event_idx), direction=int(mt.lower_sd),
            trigger_event_idx=int(v4.trigger_event_idx), source=mt,
            probe_input_idx=int(v4.input_idx),
        ))

    # --- 4. The sweep (§17.6) ---
    sub_pool = SubStructurePool()

    def _resolve_start(t: SweepTrigger, hi: int):
        return _resolve_trigger_m15_start(
            t.source, h1_df, m15, pool=sub_pool, hi=hi, parent_path=parent_path,
        )

    def _build_geometry(key, bos0_inner):
        return build_or_get_geometry(
            sub_pool, m15, parent_path=key.parent_path, sd=key.direction,
            start_abs=key.starting_idx, bos0_inner=bos0_inner, timeframe=key.sub_tf,
        )

    def _resolve_reversal(sub, R, probe_direction):
        return _resolve_reversal_start(
            sub_pool, sub, R, probe_direction, m15, timeframe="M15", parent_path=parent_path,
        )

    sweep = run_lifecycle_sweep(
        sweep_triggers, pool=sub_pool, tables=tables,
        resolve_start=_resolve_start, build_geometry=_build_geometry,
        resolve_reversal=_resolve_reversal, parent_path=parent_path, sub_tf="M15",
    )
    n_records = len(sub_pool.all_records())
    print(
        f"[pool] unique subs={len(sub_pool.all())} records={n_records} "
        f"unresolved={len(sweep.unresolved)} spawned_reversals={len(sweep.spawned)}"
    )
    for _s in sub_pool.all():
        print(
            f"[pool] sub_id={_s.sub_id} key=(M15,{_s.direction},{_s.starting_idx}) "
            f"start={_s.start_idx} end={_s.end_idx} reason={_s.end_reason} "
            f"lenses={sorted(_s.lenses())} nat_rev={_s.natural_reversal_idx} "
            f"relative_dir={_s.relative_dir_segments} "
            f"records={[(r.lens, (r.parent_sid, r.parent_cycle_id), r.trigger_sub_sid, r.trigger_type, r.trigger_idx, r.start_idx, r.end_idx, r.end_reason, 'ZL' if r.is_zero_length else '') for r in _s.records]}"
        )
    for u in sweep.unresolved:
        print(
            f"[pool] unresolved lens={u.lens} parent=({u.parent_sid},{u.parent_cycle_id}) "
            f"type={u.trigger_type} trigger_idx={u.trigger_idx} reason={u.reason} detail={u.detail}"
        )

    # --- 5. One projection per sub, mirrored per lens, in start_idx order ---
    rendered = [s for s in sub_pool.all() if s.start_idx is not None]
    rendered.sort(key=lambda s: (s.start_idx, s.sub_id))
    results_by_lens: Dict[str, list] = {LENS_CONFLUENCE: [], LENS_COUNTER: []}
    all_results: list = []
    for sub in rendered:
        res = render_sub_projection(
            sub, m15, lens_paths=lens_paths, lens_dfs=lens_dfs, timeframe="M15",
        )
        all_results.append(res)
        for lens in sub.lenses():
            results_by_lens[lens].append(res)
    for sub in sub_pool.all():
        if sub.start_idx is None:
            print(
                f"[pool] sub_id={sub.sub_id} key=(M15,{sub.direction},{sub.starting_idx}) "
                f"has no live record — logged, not rendered"
            )

    # --- 6. WVMI (§17.10 minimal: one sweep per unique sub) ---
    main_zpt = main_zone_proximity_triggers or {}
    main_first_sd_by_cycle: Dict[tuple, int] = {}
    for k, trig_list in main_zpt.items():
        if trig_list and trig_list[0].direction == "sd":
            main_first_sd_by_cycle[k] = int(trig_list[0].idx)
    var4_all_sorted = sorted(var4_all, key=lambda t: t.trigger_event_idx)
    var3_all_sorted = sorted(var3_all, key=lambda t: t.trigger_event_idx)
    conf_stream: List[Tuple[int, str]] = []
    ctr_stream: List[Tuple[int, str]] = []
    for key in tables.cycles():
        conf_stream.extend(_confluence_trigger_stream(key, main_first_sd_by_cycle, var4_all_sorted))
        ctr_stream.extend(_counter_trigger_stream(key, var3_all_sorted))
    wvmi_counts = _assign_sub_wvmi_per_sub(
        all_results,
        {LENS_CONFLUENCE: conf_stream, LENS_COUNTER: ctr_stream},
        parent_df=h1_df, m15_df=m15, lens_dfs=lens_dfs, lens_paths=lens_paths,
    )
    print(
        f"[multi_tf:dual] sub wvmi acted={wvmi_counts['acted']} "
        f"records={wvmi_counts['records']} by_started_by={wvmi_counts['by_started_by']} "
        f"by_lens={wvmi_counts['by_lens']}"
    )

    # --- 7. Sid records, record tables, registry ---
    unresolved = list(sweep.unresolved)
    for lens, lens_df in lens_dfs.items():
        lens_results = results_by_lens[lens]
        lens_df.attrs["triggers"] = [
            r for r in sub_pool.all_records() if r.lens == lens
        ]
        lens_df.attrs["unresolved_triggers"] = unresolved
        lens_df.attrs["sids"] = build_sid_records_for_subordinate(lens_results)
        print(
            f"[sid_records] {lens_paths[lens]} subs={len(lens_df.attrs['sids'])} "
            f"records={len(lens_df.attrs['triggers'])}"
        )
        if lens_results:
            registry.register(
                lens_paths[lens], df=lens_df, timeframe="M15",
                role="subordinate", starting_alignment=lens,
            )
    meta["sub_pool"] = sub_pool
    meta["parent_tables"] = tables
    # Every lens df (registered or not) for the pool-table exports — a lens
    # with records but no rendered sub is not registered (no chart) yet its
    # `_triggers.csv` and the pool-wide unresolved table must still be written.
    meta["sub_lens_dfs"] = dict(lens_dfs)
    print(
        f"[multi_tf:dual] confluence results: {len(results_by_lens[LENS_CONFLUENCE])} subs; "
        f"counter results: {len(results_by_lens[LENS_COUNTER])} subs"
    )
    return results_by_lens[LENS_CONFLUENCE], results_by_lens[LENS_COUNTER]


def _validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_CANDLE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[pipeline] Missing required columns: {missing}")
    if df.empty:
        raise ValueError("[pipeline] Input df is empty")
