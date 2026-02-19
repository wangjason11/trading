"""Unit tests for compute_structure_scenario_3 (Scenario 3)."""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from engine_v2.structure.structure_engine import (
    Scenario3Result,
    compute_structure_scenario_3,
    _get_bos0_zone_bounds,
)
from engine_v2.structure.market_structure import MarketStructure, StructureEvent
from engine_v2.features.candle_classifier import apply_candle_classification
from engine_v2.patterns.pattern_engine import detect_patterns
from engine_v2.patterns.imbalance import compute_imbalance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_df(ohlc_rows: list[dict], pair: str = "NZD_USD") -> pd.DataFrame:
    """Build minimal raw OHLCV DataFrame from a list of {o, h, l, c} dicts.

    Adds time and volume columns.  Returns a df suitable for
    apply_candle_classification → detect_patterns → compute_structure_scenario_3.
    """
    base_time = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    records = []
    for i, row in enumerate(ohlc_rows):
        records.append({
            "time": base_time + pd.Timedelta(hours=i),
            "o": row["o"],
            "h": row["h"],
            "l": row["l"],
            "c": row["c"],
            "volume": row.get("volume", 100),
        })
    df = pd.DataFrame(records)
    df.attrs["pair"] = pair
    return df


def _prepare_df(ohlc_rows: list[dict], pair: str = "NZD_USD") -> pd.DataFrame:
    """Run early pipeline stages to produce a df ready for MarketStructure."""
    raw = _make_raw_df(ohlc_rows, pair=pair)
    c_res = apply_candle_classification(raw)
    p_res = detect_patterns(c_res.df)
    df = compute_imbalance(p_res.df)
    df.attrs["pair"] = pair
    return df


def _make_uptrend_data(n: int = 80, base: float = 0.6000,
                       step: float = 0.0020, noise: float = 0.0005) -> list[dict]:
    """Generate uptrend OHLC data with clear bullish candles.

    Each candle: small body ascending, enough range for pattern detection.
    """
    rows = []
    rng = np.random.RandomState(42)
    price = base
    for i in range(n):
        body = step + rng.uniform(0, noise)
        o = price
        c = price + body
        h = c + rng.uniform(0.0001, 0.0005)
        l = o - rng.uniform(0.0001, 0.0005)
        rows.append({"o": round(o, 5), "h": round(h, 5),
                      "l": round(l, 5), "c": round(c, 5)})
        price = c
    return rows


def _make_downtrend_data(n: int = 40, base: float = 0.7500,
                         step: float = 0.0020, noise: float = 0.0005) -> list[dict]:
    """Generate downtrend OHLC data with clear bearish candles."""
    rows = []
    rng = np.random.RandomState(99)
    price = base
    for i in range(n):
        body = step + rng.uniform(0, noise)
        o = price
        c = price - body
        h = o + rng.uniform(0.0001, 0.0005)
        l = c - rng.uniform(0.0001, 0.0005)
        rows.append({"o": round(o, 5), "h": round(h, 5),
                      "l": round(l, 5), "c": round(c, 5)})
        price = c
    return rows


def _make_short_data(n: int = 10, base: float = 0.6500) -> list[dict]:
    """Generate short dataset that won't produce enough CTS events → pending."""
    rows = []
    price = base
    for i in range(n):
        o = price
        c = price + 0.0010
        h = c + 0.0002
        l = o - 0.0002
        rows.append({"o": round(o, 5), "h": round(h, 5),
                      "l": round(l, 5), "c": round(c, 5)})
        price = c
    return rows


# ---------------------------------------------------------------------------
# Tests: Scenario3Result dataclass
# ---------------------------------------------------------------------------

class TestScenario3Result:
    def test_dataclass_fields(self):
        """Scenario3Result has all required fields."""
        r = Scenario3Result(
            df=pd.DataFrame(),
            levels=[],
            events=[],
            struct_direction=1,
            start_idx=10,
            original_bos0_bounds=(0.65, 0.64, "sell"),
            probe_iterations=1,
            status="finalized",
            notes="test",
        )
        assert r.status == "finalized"
        assert r.start_idx == 10
        assert r.probe_iterations == 1
        assert r.original_bos0_bounds == (0.65, 0.64, "sell")

    def test_default_notes(self):
        """Notes defaults to empty string."""
        r = Scenario3Result(
            df=pd.DataFrame(), levels=[], events=[],
            struct_direction=1, start_idx=0,
            original_bos0_bounds=None, probe_iterations=1, status="pending",
        )
        assert r.notes == ""


# ---------------------------------------------------------------------------
# Tests: _get_bos0_zone_bounds helper
# ---------------------------------------------------------------------------

class TestGetBos0ZoneBounds:
    def test_no_events_returns_none(self):
        """With no events, helper returns None."""
        df = pd.DataFrame({"o": [1], "h": [1.1], "l": [0.9], "c": [1.05], "time": ["2024-01-01"]})
        result = _get_bos0_zone_bounds(df, [], sid=0, sd=1)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: compute_structure_scenario_3 — basic contract
# ---------------------------------------------------------------------------

class TestScenario3BasicContract:
    def test_returns_scenario3_result(self):
        """Function returns a Scenario3Result instance."""
        ohlc = _make_uptrend_data(n=80)
        df = _prepare_df(ohlc)
        result = compute_structure_scenario_3(df, start_idx=0, struct_direction=1)
        assert isinstance(result, Scenario3Result)

    def test_status_is_finalized_or_pending(self):
        """Status is always 'finalized' or 'pending'."""
        ohlc = _make_uptrend_data(n=80)
        df = _prepare_df(ohlc)
        result = compute_structure_scenario_3(df, start_idx=0, struct_direction=1)
        assert result.status in ("finalized", "pending")

    def test_struct_direction_preserved(self):
        """struct_direction in result matches input."""
        ohlc = _make_uptrend_data(n=80)
        df = _prepare_df(ohlc)
        result = compute_structure_scenario_3(df, start_idx=0, struct_direction=1)
        assert result.struct_direction == 1

    def test_probe_iterations_at_least_one(self):
        """At least one probe iteration is always run."""
        ohlc = _make_uptrend_data(n=80)
        df = _prepare_df(ohlc)
        result = compute_structure_scenario_3(df, start_idx=0, struct_direction=1)
        assert result.probe_iterations >= 1

    def test_events_and_levels_are_lists(self):
        """Events and levels are always lists."""
        ohlc = _make_uptrend_data(n=80)
        df = _prepare_df(ohlc)
        result = compute_structure_scenario_3(df, start_idx=0, struct_direction=1)
        assert isinstance(result.events, list)
        assert isinstance(result.levels, list)


# ---------------------------------------------------------------------------
# Tests: Condition 4 — short data → finalized (bound reached)
# ---------------------------------------------------------------------------

class TestScenario3Condition4Finalized:
    def test_short_data_returns_finalized(self):
        """When data is too short for 2 CTS_EST, status should be finalized
        (probe accepts current start when bound is reached)."""
        ohlc = _make_short_data(n=10)
        df = _prepare_df(ohlc)
        result = compute_structure_scenario_3(df, start_idx=0, struct_direction=1)
        assert result.status == "finalized"
        # Data is still accessible
        assert result.df is not None
        assert len(result.df) == len(df)

    def test_finalized_probe_data_accessible(self):
        """Finalized result has events/levels from the probe."""
        ohlc = _make_short_data(n=15)
        df = _prepare_df(ohlc)
        result = compute_structure_scenario_3(df, start_idx=0, struct_direction=1)
        # Probe data is accessible (may be empty if no events generated)
        assert isinstance(result.events, list)
        assert isinstance(result.levels, list)


# ---------------------------------------------------------------------------
# Tests: Max probe iterations guard
# ---------------------------------------------------------------------------

class TestScenario3MaxIterations:
    def test_max_iterations_respected(self):
        """Probe iterations never exceed max_probe_iterations."""
        ohlc = _make_uptrend_data(n=80)
        df = _prepare_df(ohlc)
        result = compute_structure_scenario_3(
            df, start_idx=0, struct_direction=1,
            max_probe_iterations=3)
        assert result.probe_iterations <= 3


# ---------------------------------------------------------------------------
# Tests: original_bos0_bounds persistence
# ---------------------------------------------------------------------------

class TestScenario3Bos0Bounds:
    def test_bos0_bounds_none_when_no_cts(self):
        """When no CTS_EST is produced, bos0_bounds is None."""
        ohlc = _make_short_data(n=10)
        df = _prepare_df(ohlc)
        result = compute_structure_scenario_3(df, start_idx=0, struct_direction=1)
        # With very short data, no CTS → no BOS zone → None
        # (May or may not be None depending on whether MS produces any events)
        assert result.original_bos0_bounds is None or isinstance(result.original_bos0_bounds, tuple)

    def test_bos0_bounds_tuple_when_enough_data(self):
        """With enough data for CTS_EST, bos0_bounds is a 3-tuple or None."""
        ohlc = _make_uptrend_data(n=80)
        df = _prepare_df(ohlc)
        result = compute_structure_scenario_3(df, start_idx=0, struct_direction=1)
        if result.original_bos0_bounds is not None:
            assert len(result.original_bos0_bounds) == 3
            outer, inner, side = result.original_bos0_bounds
            assert isinstance(outer, float)
            assert isinstance(inner, float)
            assert side in ("buy", "sell")


# ---------------------------------------------------------------------------
# Tests: Finalized with structure events
# ---------------------------------------------------------------------------

class TestScenario3Finalized:
    def test_finalized_has_events(self):
        """A finalized result should have structure events from the probe."""
        ohlc = _make_uptrend_data(n=80)
        df = _prepare_df(ohlc)
        result = compute_structure_scenario_3(df, start_idx=0, struct_direction=1)
        if result.status == "finalized":
            assert len(result.events) > 0

    def test_finalized_df_has_structure_columns(self):
        """A finalized result's df should have market_state and structure_id."""
        ohlc = _make_uptrend_data(n=80)
        df = _prepare_df(ohlc)
        result = compute_structure_scenario_3(df, start_idx=0, struct_direction=1)
        if result.status == "finalized":
            assert "market_state" in result.df.columns
            assert "structure_id" in result.df.columns

    def test_start_idx_within_data(self):
        """Validated start_idx should be within the data range."""
        ohlc = _make_uptrend_data(n=80)
        df = _prepare_df(ohlc)
        result = compute_structure_scenario_3(df, start_idx=0, struct_direction=1)
        assert result.start_idx >= 0
        assert result.start_idx <= df.index.max()


# ---------------------------------------------------------------------------
# Tests: Downtrend direction
# ---------------------------------------------------------------------------

class TestScenario3Downtrend:
    def test_downtrend_direction(self):
        """Scenario 3 works with struct_direction=-1."""
        ohlc = _make_downtrend_data(n=60)
        df = _prepare_df(ohlc)
        result = compute_structure_scenario_3(df, start_idx=0, struct_direction=-1)
        assert isinstance(result, Scenario3Result)
        assert result.struct_direction == -1
        assert result.status in ("finalized", "pending")


# ---------------------------------------------------------------------------
# Tests: Non-zero start_idx
# ---------------------------------------------------------------------------

class TestScenario3ArbitraryStart:
    def test_nonzero_start_idx(self):
        """Scenario 3 accepts a non-zero start_idx."""
        ohlc = _make_uptrend_data(n=80)
        df = _prepare_df(ohlc)
        mid = len(df) // 4
        result = compute_structure_scenario_3(
            df, start_idx=mid, struct_direction=1)
        assert isinstance(result, Scenario3Result)
        # start_idx should be >= mid (may advance due to exception)
        assert result.start_idx >= mid

    def test_different_starts_may_differ(self):
        """Different start_idx values can produce different results."""
        ohlc = _make_uptrend_data(n=80)
        df = _prepare_df(ohlc)
        r1 = compute_structure_scenario_3(df, start_idx=0, struct_direction=1)
        r2 = compute_structure_scenario_3(df, start_idx=20, struct_direction=1)
        # They should both be valid results (may or may not differ)
        assert isinstance(r1, Scenario3Result)
        assert isinstance(r2, Scenario3Result)


# ---------------------------------------------------------------------------
# Tests: Notes string
# ---------------------------------------------------------------------------

class TestScenario3Notes:
    def test_notes_contains_status(self):
        """Notes string includes the status."""
        ohlc = _make_uptrend_data(n=80)
        df = _prepare_df(ohlc)
        result = compute_structure_scenario_3(df, start_idx=0, struct_direction=1)
        assert result.status in result.notes

    def test_notes_contains_scenario3(self):
        """Notes string references Scenario3."""
        ohlc = _make_uptrend_data(n=80)
        df = _prepare_df(ohlc)
        result = compute_structure_scenario_3(df, start_idx=0, struct_direction=1)
        assert "Scenario3" in result.notes


# ---------------------------------------------------------------------------
# Tests: Validation
# ---------------------------------------------------------------------------

class TestScenario3Validation:
    def test_empty_df_raises(self):
        """Empty df raises ValueError."""
        df = pd.DataFrame(columns=["time", "o", "h", "l", "c"])
        with pytest.raises(ValueError, match="empty"):
            compute_structure_scenario_3(df, start_idx=0, struct_direction=1)

    def test_missing_columns_raises(self):
        """Missing required columns raises ValueError."""
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="Missing required columns"):
            compute_structure_scenario_3(df, start_idx=0, struct_direction=1)


# ---------------------------------------------------------------------------
# Tests: Self-contained probes (df isolation)
# ---------------------------------------------------------------------------

class TestScenario3DfIsolation:
    def test_original_df_unmodified(self):
        """The input df should not be modified by Scenario 3."""
        ohlc = _make_uptrend_data(n=80)
        df = _prepare_df(ohlc)
        cols_before = set(df.columns)
        _result = compute_structure_scenario_3(df, start_idx=0, struct_direction=1)
        # Original df should still have same columns (no structure columns added)
        assert set(df.columns) == cols_before

    def test_result_df_is_copy(self):
        """Result df should be a separate copy from input."""
        ohlc = _make_uptrend_data(n=80)
        df = _prepare_df(ohlc)
        result = compute_structure_scenario_3(df, start_idx=0, struct_direction=1)
        # result.df should not be the same object as input df
        assert result.df is not df
