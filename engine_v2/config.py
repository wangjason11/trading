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

    pair="NZD_USD",
    timeframe="H1",
    # Use UTC for deterministic replay.
    start=datetime(2025, 12, 1, 0, 0, 0, tzinfo=timezone.utc),
    # Sub-debug session 2026-05-14: temporary shortened window
    # (end 2025-12-27 vs. normal 2026-01-20) to cut replay runtime
    # while debugging subordinate-structure issues. Restore to
    # 2026-01-20 once sub debug for this window is signed off.
    end=datetime(2025, 12, 27, 0, 0, 0, tzinfo=timezone.utc),
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
