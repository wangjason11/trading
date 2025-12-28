from __future__ import annotations

from engine_v2.config import CONFIG
from engine_v2.data.provider_oanda import get_history
from engine_v2.pipeline.orchestrator import run_pipeline
from engine_v2.charting.export_plotly import export_chart_plotly


def main() -> None:
    df = get_history(
        pair=CONFIG.pair,
        timeframe=CONFIG.timeframe,
        start=CONFIG.start,
        end=CONFIG.end,
    )

    res = run_pipeline(df)

    from engine_v2.debug.export_structure import export_levels, export_swings
    from engine_v2.structure.structure_v1 import compute_structure_levels

    # Use the same parameters you settled on for the "good range"
    swings, levels = compute_structure_levels(res.df, left=6, right=6)

    export_swings(swings, "artifacts/debug/swings.csv")
    export_levels(levels, "artifacts/debug/structure_levels.csv")

    print("Exported:", "artifacts/debug/swings.csv", "artifacts/debug/structure_levels.csv")


    res.df.attrs["structure_levels"] = res.structure

    print("=== Replay Summary ===")
    print(f"pair={CONFIG.pair} tf={CONFIG.timeframe}")
    print(f"candles={len(res.df)}")
    print(f"pattern_events={len(res.patterns)}")
    print(f"structure_levels={len(res.structure)}")
    print("notes:", res.meta.get("notes", {}))

    # Export chart artifacts
    basename = f"{CONFIG.pair}_{CONFIG.timeframe}_{CONFIG.start.date()}_{CONFIG.end.date()}"
    print("DEBUG structure_levels:", len(res.structure))
    paths = export_chart_plotly(
        res.df,
        title=f"...",
        basename=basename,
        structure_levels=res.structure,
    )
    print(f"Chart HTML: {paths.html_path}")
    print(f"Chart PNG : {paths.png_path}")


if __name__ == "__main__":
    main()
