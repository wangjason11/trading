from __future__ import annotations

import time
from pathlib import Path
from datetime import timedelta
from typing import Optional, Tuple
from engine_v2.config import CONFIG
from engine_v2.data.provider_oanda import get_history
from engine_v2.pipeline.orchestrator import run_pipeline
from engine_v2.charting.export_plotly import export_chart_plotly
from engine_v2.structure.identify_start import identify_start_scenario_1

chart_cfg = {
    # Week 4 “pattern work mode” defaults:
    "candle_types": {"pinbar": False, "maru": False},  
    "patterns": {"engulfing": False, "star": False, "continuous": True, "double_maru": True, "one_maru_continuous": True, "one_maru_opposite": True},

    "struct_state": {"labels": True},
    "range_visual": {"rectangles": True},

    # keep these off for now (declutter)
    "structure": {"levels": True, "swings": False},
    "zones": {
        "KL": True, "OB": False, "POI": True,
        "num_structures": 3,  # how many recent structure_ids to show zones for
        # Per-sub-entity overlay toggles for the H1 chart (default: all off).
        # Add an entry to enable, e.g. {"M15.counter": {"KL": True}}.
        # See CHARTING_SPEC.md "Subordinate overlays on parent charts".
        "subordinate_overlays": {},
    },
}


def export_charts_with_optional_zoom(
    *,
    registry,
    path_id: str,
    structure_levels,
    chart_cfg,
    title_prefix: str,
    basename: str,
    zoom: Optional[Tuple[int, int]] = None,
):
    """
    Always exports a full chart.
    Optionally exports a zoomed chart if `zoom=(i0, i1)` is provided.

    Part 4 Step 2: chart sourced from the StructureRegistry by ``path_id``.
    """

    # --- Full chart ---
    paths_full = export_chart_plotly(
        registry=registry,
        path_id=path_id,
        title=f"{title_prefix} (full)",
        basename=basename,
        structure_levels=structure_levels,
        cfg=chart_cfg,
    )

    print(f"Chart FULL HTML: {paths_full.html_path}")
    print(f"Chart FULL PNG : {paths_full.png_path}")

    # --- Optional zoom chart ---
    if zoom is not None:
        i0, i1 = zoom
        zoom_basename = f"{basename}_zoom_{i0}-{i1}"

        paths_zoom = export_chart_plotly(
            registry=registry,
            path_id=path_id,
            title=f"{title_prefix} (zoom {i0}–{i1})",
            basename=zoom_basename,
            structure_levels=structure_levels,
            cfg=chart_cfg,
            idx_range=(i0, i1),
        )

        print(f"Chart ZOOM HTML: {paths_zoom.html_path}")
        print(f"Chart ZOOM PNG : {paths_zoom.png_path}")

    return paths_full

def _fmt_float(x: float) -> str:
    # filename-safe float: 0.0001 -> "0p0001"
    s = f"{x:.10f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")

def make_basename(*, pair: str, timeframe: str, start, end, struct_direction: int, eps: float, range_min_k: int, range_max_k: int) -> str:
    return (
        f"{pair}_{timeframe}_{start.date()}_{end.date()}"
        f"_sd{struct_direction}_eps{_fmt_float(eps)}_rk{range_min_k}-{range_max_k}"
    )

def export_csv(df, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def _timeframe_to_timedelta(tf: str) -> timedelta:
    tf = str(tf).upper().strip()
    if tf.startswith("S"):
        return timedelta(seconds=int(tf[1:]))
    if tf.startswith("M"):
        return timedelta(minutes=int(tf[1:]))
    if tf.startswith("H"):
        return timedelta(hours=int(tf[1:]))
    if tf in ("D", "DAILY"):
        return timedelta(days=1)
    raise ValueError(f"Unsupported timeframe for auto-extend: {tf}")


def _floor_to_day_start(ts):
    return ts.replace(hour=0, minute=0, second=0, microsecond=0)

def fetch_history_with_auto_extend(
    *,
    pair: str,
    timeframe: str,
    start,
    end,
    lookback_days: int = 183,
    min_history: int = 50,
    max_extend_iters: int = 8,
    extend_days: int = 4,
):
    """
    Week 6 Part 3A requirement:
    If identify_start_scenario_1 indicates the chosen extreme is 'too early',
    extend the dataset earlier and retry until it passes (or guard trips).
    """
    cur_start = _floor_to_day_start(start)
    last_decision = None
    for k in range(int(max_extend_iters)):
        df = get_history(pair=pair, timeframe=timeframe, start=cur_start, end=end)

        input_idx = int(df.index.max())
        decision = identify_start_scenario_1(
            df, input_idx=input_idx, lookback_days=lookback_days, min_history=min_history
        )
        last_decision = decision

        if not decision.meta.get("too_early", False):
            return df, cur_start, decision

        raw_start_idx = int(decision.meta.get("raw_start_idx", decision.start_idx))

        prev_start = cur_start
        cur_start = _floor_to_day_start(cur_start - timedelta(days=int(extend_days)))

        print(
            f"[auto_extend] iter={k+1} too_early raw_start_idx={raw_start_idx} "
            f"min_history={min_history} => extending start by {extend_days} days "
            f"{prev_start} -> {cur_start}"
        )

    print("[auto_extend] WARNING: hit max_extend_iters; proceeding with last fetched dataset.")
    df = get_history(pair=pair, timeframe=timeframe, start=cur_start, end=end)

    input_idx = int(df.index.max())
    last_decision = identify_start_scenario_1(
        df, input_idx=input_idx, lookback_days=lookback_days, min_history=min_history
    )
    return df, cur_start, last_decision


def main() -> None:
    # df = get_history(
    #     pair=CONFIG.pair,
    #     timeframe=CONFIG.timeframe,
    #     start=CONFIG.start,
    #     end=CONFIG.end,
    # )

    _t_wallclock_start = time.perf_counter()

    _t0 = time.perf_counter()
    df, effective_start, start_decision = fetch_history_with_auto_extend(
        pair=CONFIG.pair,
        timeframe=CONFIG.timeframe,
        start=_floor_to_day_start(CONFIG.start),
        end=CONFIG.end,
        lookback_days=183,
        min_history=50,
        extend_days=4,
        max_extend_iters=8,
    )
    _t_data_fetch = time.perf_counter() - _t0
    df.attrs["pair"] = CONFIG.pair

    raw_path = (
        f"artifacts/debug/"
        f"{CONFIG.pair}_{CONFIG.timeframe}_{effective_start.date()}_{CONFIG.end.date()}_raw.csv"
    )
    export_csv(df.copy(), raw_path)

    res = run_pipeline(df, lower_timeframes=CONFIG.lower_timeframes)
    registry = res.meta["registry"]
    timing: dict = dict(res.meta.get("timing", {}))

    print(res.df["is_range"].value_counts())
    print(res.df[res.df["is_range"] == 1][["is_range_confirm_idx", "is_range_lag"]].head())

    from engine_v2.debug.export_structure import export_levels

    res.df.attrs["structure_levels"] = res.structure
    res.df.attrs["kl_zones"] = res.meta.get("kl_zones", [])

    print("=== Replay Summary ===")
    print(f"pair={CONFIG.pair} tf={CONFIG.timeframe}")
    print(f"candles={len(res.df)}")
    print(f"pattern_events={len(res.patterns)}")
    print(f"structure_levels={len(res.structure)}")
    print("notes:", res.meta.get("notes", {}))

    final_path = (
        f"artifacts/debug/"
        f"{CONFIG.pair}_{CONFIG.timeframe}_{effective_start.date()}_{CONFIG.end.date()}_final.csv"
    )
    export_csv(res.df, final_path)

    print(f"Raw data: {raw_path}")
    print(f"Full pipeline data: {final_path}")

    # Export chart artifacts
    # basename = f"{CONFIG.pair}_{CONFIG.timeframe}_{CONFIG.start.date()}_{CONFIG.end.date()}"

    # Keep in sync with structure_engine MarketStructure(...) args for now
    # struct_direction = 1

    sd_series = res.df["struct_direction"].astype(int)
    sd_last = int(sd_series[sd_series != 0].iloc[-1]) if (sd_series != 0).any() else 0
    struct_direction = sd_last

    eps = 0.0001
    range_min_k = 2
    range_max_k = 5

    basename = make_basename(
        pair=CONFIG.pair,
        timeframe=CONFIG.timeframe,
        start=effective_start,
        end=CONFIG.end,
        struct_direction=struct_direction,
        eps=eps,
        range_min_k=range_min_k,
        range_max_k=range_max_k,
    )
    
    print("DEBUG structure_levels:", len(res.structure))

    export_levels(res.structure, f"artifacts/debug/{basename}_structure_levels.csv")

    # ---------------------------
    # Chart export
    # ---------------------------

    TITLE_PREFIX = f"{CONFIG.pair} {CONFIG.timeframe}"
    ZOOM = None              # e.g. None or (300, 520)

    _t0 = time.perf_counter()
    export_charts_with_optional_zoom(
        registry=registry,
        path_id="H1.main",
        structure_levels=res.structure,
        chart_cfg=chart_cfg,
        title_prefix=TITLE_PREFIX,
        basename=basename,
        zoom=ZOOM,
    )
    timing["chart_h1"] = time.perf_counter() - _t0

    # M15 chart exports (one per registered M15 sub entity).
    # Part 4 §16.1 / §16.2: each entity gets its own chart, candles in own
    # TF, immediate parent overlaid. Both M15.counter and M15.confluence
    # share the same H1.main parent overlay; only the per-entity sub data
    # (events, zones, wave candles, sub WVMI) differs.
    m15_sub_path_ids = ["H1.main >> M15.counter", "H1.main >> M15.confluence"]
    for sub_path_id in m15_sub_path_ids:
        if registry.get(sub_path_id) is None:
            continue
        from engine_v2.charting.export_m15_chart import export_m15_chart_plotly
        # Filename suffix: last path segment, dots → underscores.
        # E.g. "H1.main >> M15.counter" → "M15_counter".
        leaf_label = sub_path_id.split(" >> ")[-1].replace(".", "_")
        _t0 = time.perf_counter()
        m15_paths = export_m15_chart_plotly(
            registry=registry,
            path_id=sub_path_id,
            title=f"{CONFIG.pair} {sub_path_id}",
            basename=f"{basename}_{leaf_label}",
            cfg=chart_cfg,
        )
        timing[f"chart_{leaf_label.lower()}"] = time.perf_counter() - _t0
        print(f"Chart {leaf_label} HTML: {m15_paths.html_path}")
        print(f"Chart {leaf_label} PNG : {m15_paths.png_path}")

        # Debug: export per-sub KL zones (entity-absolute indices after
        # mirror) so we can inspect sub-native zones without parsing the
        # chart HTML.
        from engine_v2.debug.export_zones import export_kl_zones
        sub_entry = registry.get(sub_path_id)
        if sub_entry is not None:
            sub_zones = sub_entry.df.attrs.get("kl_zones", []) or []
            export_kl_zones(
                sub_zones,
                f"artifacts/debug/{basename}_{leaf_label}_kl_zones.csv",
            )

            # Debug: export per-sub SidRecords so we can see each sid's
            # boundary (creation_event_idx) and trigger info.
            sub_sids = sub_entry.df.attrs.get("sids", []) or []
            if sub_sids:
                import pandas as _pd
                _pd.DataFrame([{
                    "parent_sid": s.parent_sid,
                    "parent_cycle_id": s.parent_cycle_id,
                    "sub_sid": s.sub_sid,
                    "starting_sd": s.starting_sd,
                    "creation_event_idx": s.creation_event_idx,
                    "end_event_idx": s.end_event_idx,
                    "end_reason": s.end_reason,
                    "meta": s.meta,
                } for s in sub_sids]).to_csv(
                    f"artifacts/debug/{basename}_{leaf_label}_sids.csv",
                    index=False,
                )

            # Debug: export per-sub WVMI records (parent-event-driven,
            # trigger-centric — see WVMI_SPEC "Sub entities"). Not in any other
            # CSV, so this is the only inspectable surface besides chart hover.
            from engine_v2.debug.export_wvmi import export_wvmi
            export_wvmi(
                sub_entry.df.attrs.get("wvmi", []) or [],
                f"artifacts/debug/{basename}_{leaf_label}_wvmi.csv",
            )

            # Debug: per-sub fib lifecycle (start_idx/end_idx/end_reason/status).
            # Fib was LATENT (no fib CSV, M15 fib lines off) — this is the only
            # inspectable surface (FIB_LIFECYCLE_SPEC §15.8).
            from engine_v2.debug.export_fib_lifecycle import export_fib_lifecycle
            export_fib_lifecycle(
                sub_entry.df.attrs.get("fib_states", []) or [],
                f"artifacts/debug/{basename}_{leaf_label}_fib_lifecycle.csv",
                structure_path_id=sub_path_id,
            )

    from engine_v2.debug.export_zones import export_kl_zones
    export_kl_zones(res.meta.get("kl_zones", []), f"artifacts/debug/{basename}_kl_zones.csv")

    from engine_v2.debug.export_wvmi import export_wvmi
    export_wvmi(res.meta.get("wvmi", []) or [], f"artifacts/debug/{basename}_wvmi.csv")

    from engine_v2.debug.export_fib_lifecycle import export_fib_lifecycle
    export_fib_lifecycle(
        res.meta.get("fib_states", []) or [],
        f"artifacts/debug/{basename}_fib_lifecycle.csv",
        structure_path_id="H1.main",
    )

    from engine_v2.debug.export_events import export_structure_events
    export_structure_events(
        res.df.attrs.get("structure_events", []),
        f"artifacts/debug/{basename}_structure_events.csv",
    )

    from engine_v2.debug.export_imbalances import export_imbalance_instances
    export_imbalance_instances(
        res.df.attrs.get("imbalances", []),
        f"artifacts/debug/{basename}_imbalance_instances.csv",
    )

    # ---------------------------
    # Timing summary
    # ---------------------------
    _t_wallclock_total = time.perf_counter() - _t_wallclock_start
    # Prepend data fetch in stage order; preserve insertion order from pipeline
    timing_ordered = {"data_fetch": _t_data_fetch, **timing}
    _tracked_sum = sum(timing_ordered.values())
    _untracked = _t_wallclock_total - _tracked_sum

    print()
    print("=== Replay Timing ===")
    for stage, secs in timing_ordered.items():
        print(f"{stage:<32s} {secs:>8.2f}s")
    print(f"{'-' * 32}")
    print(f"{'tracked subtotal':<32s} {_tracked_sum:>8.2f}s")
    print(f"{'untracked overhead':<32s} {_untracked:>8.2f}s")
    print(f"{'total (wall clock)':<32s} {_t_wallclock_total:>8.2f}s")


if __name__ == "__main__":
    main()
