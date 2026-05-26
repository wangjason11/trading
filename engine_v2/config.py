from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class ReplayConfig:
    pair: str
    timeframe: str
    start: datetime
    end: datetime
    lower_timeframes: tuple = ()  # e.g. ("M15",) to enable UC1


# ---------------------------
# Week 1 defaults (locked)
# ---------------------------
CONFIG = ReplayConfig(
    # pair="NZD_USD",
    # timeframe="M15",
    # # Use UTC for deterministic replay.
    # start=datetime(2025, 11, 25, 0, 0, 0, tzinfo=timezone.utc),
    # end=datetime(2025, 12, 3, 23, 59, 59, tzinfo=timezone.utc),

    # pair="NZD_USD",
    # timeframe="H1",
    # # Use UTC for deterministic replay.
    # start=datetime(2025, 12, 19, 0, 0, 0, tzinfo=timezone.utc),
    # end=datetime(2025, 12, 29, 23, 59, 59, tzinfo=timezone.utc),

    # Prior window — POI two-stroke first sign-off (commit 6069ade, replay
    # outputs in b81184b). Sub-debug session 2026-05-14 originally
    # shortened from full 2026-01-20 to cut replay runtime.
    # pair="NZD_USD",
    # timeframe="H1",
    # start=datetime(2025, 12, 1, 0, 0, 0, tzinfo=timezone.utc),
    # end=datetime(2025, 12, 27, 0, 0, 0, tzinfo=timezone.utc),
    # lower_timeframes=("M15",),

    # POI two-stroke incremental validation (2026-05-23 onward): stepped end
    # forward in stages from 2025-12-27 (signed off in 6069ade / b81184b)
    # toward the full 2026-01-20 window. Step 2026-01-07 (~11 days past prior
    # sign-off, incl. New Year low-liquidity stretch) was signed off on the
    # shorter window; restored to full below for Phase 5 sub-WVMI.
    # pair="NZD_USD",
    # timeframe="H1",
    # start=datetime(2025, 12, 1, 0, 0, 0, tzinfo=timezone.utc),
    # end=datetime(2026, 1, 7, 0, 0, 0, tzinfo=timezone.utc),
    # lower_timeframes=("M15",),

    # Full original window restored 2026-05-26 (Phase 5 sub-WVMI). The build
    # was validated through 2026-01-07 on the shorter window; expanded back to
    # the full 2026-01-20 end to surface var4 (subsequent_counter) triggers for
    # trigger-centric vs sid-centric validation.
    pair="NZD_USD",
    timeframe="H1",
    # Use UTC for deterministic replay.
    start=datetime(2025, 12, 1, 0, 0, 0, tzinfo=timezone.utc),
    end=datetime(2026, 1, 20, 0, 0, 0, tzinfo=timezone.utc),
    lower_timeframes=("M15",),

    # pair="NZD_USD",
    # timeframe="H1",
    # # Use UTC for deterministic replay.
    # start=datetime(2025, 12, 19, 0, 0, 0, tzinfo=timezone.utc),
    # end=datetime(2025, 12, 29, 23, 59, 59, tzinfo=timezone.utc),

    # pair="NZD_USD",
    # timeframe="H1",
    # # Use UTC for deterministic replay.
    # start=datetime(2025, 12, 19, 0, 0, 0, tzinfo=timezone.utc),
    # end=datetime(2025, 12, 29, 23, 59, 59, tzinfo=timezone.utc),

    # pair="NZD_USD",
    # timeframe="H1",
    # # Use UTC for deterministic replay.
    # start=datetime(2025, 12, 19, 0, 0, 0, tzinfo=timezone.utc),
    # end=datetime(2025, 12, 29, 23, 59, 59, tzinfo=timezone.utc),
)
