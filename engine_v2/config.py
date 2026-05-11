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
    end=datetime(2026, 1, 20, 0, 0, 0, tzinfo=timezone.utc),
    # Main-structure debugging mode (2026-05-11): lower TFs disabled to
    # cut replay runtime while we verify the H1 main pipeline (extra BOS
    # at 157 / 705 + zone-proximity-trigger marker work). Restore to
    # ("M15",) once main structure is verified correct and we resume
    # subordinate-trigger debugging / Part 4 §13.5.d.
    lower_timeframes=(),

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
