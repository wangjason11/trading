from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class ReplayConfig:
    pair: str
    timeframe: str
    start: datetime
    end: datetime


# ---------------------------
# Week 1 defaults (locked)
# ---------------------------
CONFIG = ReplayConfig(
    pair="NZD_USD",
    timeframe="M15",
    # Use UTC for deterministic replay.
    start=datetime(2025, 11, 25, 12, 0, 0, tzinfo=timezone.utc),
    end=datetime(2025, 12, 3, 23, 59, 59, tzinfo=timezone.utc),
)
