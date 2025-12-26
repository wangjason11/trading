from __future__ import annotations

from engine_v2.config import CONFIG
from engine_v2.common.types import REQUIRED_CANDLE_COLS
from engine_v2.data.provider_oanda import get_history


def test_oanda_history_smoke():
    df = get_history(
        pair=CONFIG.pair,
        timeframe=CONFIG.timeframe,
        start=CONFIG.start,
        end=CONFIG.end,
    )

    assert not df.empty, "Expected non-empty dataframe from OANDA"
    for col in REQUIRED_CANDLE_COLS:
        assert col in df.columns, f"Missing column: {col}"

    # time should be increasing
    assert df["time"].is_monotonic_increasing

    # basic sanity: OHLC are numeric
    assert (df["h"] >= df["l"]).all()
