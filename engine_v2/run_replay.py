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

    print("=== Replay Summary ===")
    print(f"pair={CONFIG.pair} tf={CONFIG.timeframe}")
    print(f"candles={len(res.df)}")
    print(f"pattern_events={len(res.patterns)}")
    print(f"structure_levels={len(res.structure)}")
    print("notes:", res.meta.get("notes", {}))

    # Export chart artifacts
    basename = f"{CONFIG.pair}_{CONFIG.timeframe}_{CONFIG.start.date()}_{CONFIG.end.date()}"
    paths = export_chart_plotly(
        res.df,
        title=f"{CONFIG.pair} {CONFIG.timeframe} ({CONFIG.start.date()} → {CONFIG.end.date()})",
        basename=basename,
    )
    print(f"Chart HTML: {paths.html_path}")
    print(f"Chart PNG : {paths.png_path}")


if __name__ == "__main__":
    main()
