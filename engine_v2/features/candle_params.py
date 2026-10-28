from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class CandleParams:
    maru: float = 0.65
    pinbar: float = 0.4
    pinbar_distance: float = 0.5
    normal_distance: float = 0.4

    big_maru_threshold: float = 0.65
    big_normal_threshold: float = 0.5
    lookback: int = 5

    # The rolling pinbar body floor (PLAN_U U2; the user, 2026-10-27 / 28 —
    # `engine_v2/features/CANDLE_BODY_FLOOR_NOTES.md`). A candle's floor = the
    # `pinbar_floor_percentile` (numpy linear) of the bodies of the
    # `pinbar_floor_window` most recent pinbars STRICTLY before it; a body-%
    # maru / normal whose body is strictly below it becomes a pinbar
    # (`classify_candles` step 4). No full window -> no floor.
    # `pinbar_floor_window_source`: "final" (the default since PLAN_U U3 —
    # the user's pick, 2026-10-28) = the body-% pinbars + the ones the floor
    # demoted fill the window, candle by candle; "natural" (U2; kept as a
    # switch) = the body-% pinbars only.
    # `pinbar_floor_seed_windows`: the history fetched before a replay's frame
    # (`run_replay.fetch_floor_seed`) holds at least this many windows of
    # body-% pinbars. Read by `features/pinbar_body_floor.py` and the seed
    # guard. Replaced the fixed per-TF pip table (H1 2.2 / M15 1.0 / M5 0.5,
    # `pinbar_body_pip_floor`, removed at U2).
    pinbar_floor_window: int = 90
    pinbar_floor_percentile: float = 90.0
    pinbar_floor_window_source: str = "final"
    pinbar_floor_seed_windows: int = 2

    special_maru: float = 0.5
    special_maru_distance: float = 0.1
