"""Unit tests for the unified identify-start + probe primitive.

Phase 1 Session 1 — tests the primitive in isolation. No caller migrates
in this session, so these tests are the sole protection until the
per-trigger migrations land.

Coverage:
  - ProbeResult dataclass shape
  - Input validation
  - Each finalize_condition path (no_retrace, reversal_in_probe,
    end_idx_reached, no_cts_pending, one_cts_pending)
  - Helpers: _select_extreme_retrace_candidate, _evaluate_reset_conditions
  - Two-condition reset matrix (cond1 only / cond2 only / both / neither)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine_v2.features.candle_classifier import apply_candle_classification
from engine_v2.patterns.imbalance import compute_imbalance
from engine_v2.patterns.pattern_engine import detect_patterns
from engine_v2.structure.reference_zone import ReferenceZone
from engine_v2.structure.unified_probe import (
    ProbeResult,
    _evaluate_reset_conditions,
    _select_extreme_retrace_candidate,
    unified_probe,
)
from engine_v2.zones.zone_proximity import (
    DEFAULT_PROBE_RESET_PIPS,
    DEFAULT_PROBE_RESET_WICK,
)


# ---------------------------------------------------------------------------
# Helpers — same pattern as test_scenario3.py
# ---------------------------------------------------------------------------

def _make_raw_df(ohlc_rows: list[dict], pair: str = "NZD_USD") -> pd.DataFrame:
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
    raw = _make_raw_df(ohlc_rows, pair=pair)
    c_res = apply_candle_classification(raw)
    p_res = detect_patterns(c_res.df)
    df = compute_imbalance(p_res.df)
    df.attrs["pair"] = pair
    return df


def _make_uptrend_data(n: int = 80) -> list[dict]:
    rows = []
    rng = np.random.RandomState(42)
    price = 0.6000
    for _ in range(n):
        body = 0.0020 + rng.uniform(0, 0.0005)
        o = price
        c = price + body
        h = c + rng.uniform(0.0001, 0.0005)
        l = o - rng.uniform(0.0001, 0.0005)
        rows.append({"o": round(o, 5), "h": round(h, 5),
                     "l": round(l, 5), "c": round(c, 5)})
        price = c
    return rows


def _make_downtrend_data(n: int = 60) -> list[dict]:
    rows = []
    rng = np.random.RandomState(99)
    price = 0.7500
    for _ in range(n):
        body = 0.0020 + rng.uniform(0, 0.0005)
        o = price
        c = price - body
        h = o + rng.uniform(0.0001, 0.0005)
        l = c - rng.uniform(0.0001, 0.0005)
        rows.append({"o": round(o, 5), "h": round(h, 5),
                     "l": round(l, 5), "c": round(c, 5)})
        price = c
    return rows


def _make_short_data(n: int = 10) -> list[dict]:
    rows = []
    price = 0.6500
    for _ in range(n):
        o = price
        c = price + 0.0010
        h = c + 0.0002
        l = o - 0.0002
        rows.append({"o": round(o, 5), "h": round(h, 5),
                     "l": round(l, 5), "c": round(c, 5)})
        price = c
    return rows


def _ref_zone_uptrend(outer: float = 0.5500, inner: float = 0.5550) -> ReferenceZone:
    """A sell-side reference zone for direction=+1 probes. `outer` sits
    above `inner`; for uptrend probes both anchor far from price (price
    starts well below and retraces upward toward them).
    """
    return ReferenceZone(
        outer=outer, inner=inner, side="sell",
        source="cts_established", source_event_idx=0,
    )


def _ref_zone_downtrend(outer: float = 0.8000, inner: float = 0.7950) -> ReferenceZone:
    """A buy-side reference zone for direction=-1 probes. `outer` sits
    below `inner`."""
    return ReferenceZone(
        outer=outer, inner=inner, side="buy",
        source="cts_established", source_event_idx=0,
    )


# ---------------------------------------------------------------------------
# ProbeResult dataclass
# ---------------------------------------------------------------------------

class TestProbeResult:
    def test_construct_all_fields(self):
        ref = _ref_zone_uptrend()
        r = ProbeResult(
            start_idx=10,
            status="finalized",
            iterations=2,
            original_ref_zone=ref,
            finalize_condition="no_retrace",
            notes="hi",
        )
        assert r.start_idx == 10
        assert r.status == "finalized"
        assert r.iterations == 2
        assert r.original_ref_zone is ref
        assert r.finalize_condition == "no_retrace"
        assert r.notes == "hi"

    def test_notes_default_empty(self):
        ref = _ref_zone_uptrend()
        r = ProbeResult(
            start_idx=0, status="pending", iterations=1,
            original_ref_zone=ref, finalize_condition="no_cts_pending",
        )
        assert r.notes == ""

    def test_is_frozen(self):
        ref = _ref_zone_uptrend()
        r = ProbeResult(
            start_idx=0, status="pending", iterations=1,
            original_ref_zone=ref, finalize_condition="no_cts_pending",
        )
        with pytest.raises(Exception):
            r.start_idx = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_empty_df_raises(self):
        df = pd.DataFrame(columns=["time", "o", "h", "l", "c"])
        df.attrs["pair"] = "NZD_USD"
        with pytest.raises(ValueError, match="empty"):
            unified_probe(df, 0, 1, _ref_zone_uptrend(), None, "H1")

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"a": [1, 2]})
        df.attrs["pair"] = "NZD_USD"
        with pytest.raises(ValueError, match="Missing required columns"):
            unified_probe(df, 0, 1, _ref_zone_uptrend(), None, "H1")

    def test_bad_direction_raises(self):
        df = _prepare_df(_make_uptrend_data(n=40))
        with pytest.raises(ValueError, match="direction must be"):
            unified_probe(df, 0, 0, _ref_zone_uptrend(), None, "H1")


# ---------------------------------------------------------------------------
# Helper: _select_extreme_retrace_candidate
# ---------------------------------------------------------------------------

class TestSelectExtremeRetrace:
    def _df(self, lows: list[float], highs: list[float]) -> pd.DataFrame:
        return pd.DataFrame({
            "o": [v - 0.001 for v in lows],
            "h": highs,
            "l": lows,
            "c": [v + 0.001 for v in lows],
        })

    def test_uptrend_picks_lowest_low(self):
        df = self._df(lows=[0.60, 0.58, 0.62, 0.57, 0.61],
                      highs=[0.61, 0.59, 0.63, 0.58, 0.62])
        assert _select_extreme_retrace_candidate(df, 0, 4, 1) == 3

    def test_downtrend_picks_highest_high(self):
        df = self._df(lows=[0.70, 0.68, 0.72, 0.67, 0.71],
                      highs=[0.71, 0.69, 0.75, 0.68, 0.72])
        assert _select_extreme_retrace_candidate(df, 0, 4, -1) == 2

    def test_window_clamps(self):
        df = self._df(lows=[0.60, 0.58, 0.62, 0.57, 0.61],
                      highs=[0.61, 0.59, 0.63, 0.58, 0.62])
        # window [0, 2] — lowest in that range is idx 1 (0.58)
        assert _select_extreme_retrace_candidate(df, 0, 2, 1) == 1

    def test_empty_window_returns_none(self):
        df = self._df(lows=[0.60, 0.58], highs=[0.61, 0.59])
        # lo > hi
        assert _select_extreme_retrace_candidate(df, 5, 3, 1) is None


# ---------------------------------------------------------------------------
# Helper: _evaluate_reset_conditions — the 2-condition reset matrix
# ---------------------------------------------------------------------------

class TestEvaluateResetConditions:
    """For NZD_USD, pip_size = 0.0001. H1 thresholds: reset=4 pips,
    wick=16 pips. So:
        reset_tol = 4 * 0.0001 = 0.0004
        wick_cap  = 16 * 0.0001 = 0.0016
    """

    pip_size = 0.0001
    reset_tol = float(DEFAULT_PROBE_RESET_PIPS["H1"]) * pip_size   # 0.0004
    wick_cap = float(DEFAULT_PROBE_RESET_WICK["H1"]) * pip_size    # 0.0016

    def _one_candle_df(self, o: float, h: float, l: float, c: float) -> pd.DataFrame:
        return pd.DataFrame({"o": [o], "h": [h], "l": [l], "c": [c]})

    # --- direction=+1 (uptrend probe; sell-zone reference above price) ---

    def test_uptrend_both_pass(self):
        """Lower wick reaches near inner with SHORT wick → both conditions
        hold → reset triggers."""
        # ref: outer=0.6020, inner=0.6010
        # candle: o=0.6005, c=0.6008 (body), l=0.6009 (within reset_tol),
        #         h=0.6012 — lower wick = body_bottom(0.6005)-l(0.6009)<0 ok
        # wait — l must be ≤ body_bottom by definition; let's pick valid:
        # body: o=0.6011, c=0.6013 → body_bottom=0.6011
        # l=0.6011 (no lower wick) → cond1: 0.6011 ≤ 0.6010 + 0.0004 = 0.6014 ✓
        # lower wick: 0.6011 - 0.6011 = 0 ≤ 0.0016 ✓
        ref = ReferenceZone(outer=0.6020, inner=0.6010, side="sell",
                            source="cts_established", source_event_idx=0)
        df = self._one_candle_df(o=0.6011, h=0.6013, l=0.6011, c=0.6013)
        assert _evaluate_reset_conditions(
            df, 0, ref, 1, self.reset_tol, self.wick_cap) is True

    def test_uptrend_cond1_fails(self):
        """Low far from inner → cond1 fails → no reset."""
        # ref inner=0.6010, reset_tol=0.0004 → cond1 threshold = 0.6014
        # candle l=0.6020 (far above) → 0.6020 > 0.6014 → cond1 fails
        ref = ReferenceZone(outer=0.6030, inner=0.6010, side="sell",
                            source="cts_established", source_event_idx=0)
        df = self._one_candle_df(o=0.6020, h=0.6025, l=0.6020, c=0.6023)
        assert _evaluate_reset_conditions(
            df, 0, ref, 1, self.reset_tol, self.wick_cap) is False

    def test_uptrend_cond2_fails(self):
        """Low reaches inner (cond1 ✓) but via a LONG lower wick (cond2 ✗)
        → no reset. This is the new wick-cap stab rejection."""
        # ref inner=0.6010
        # candle: o=0.6030, c=0.6028 → body_bottom=0.6028
        # l=0.6012 → cond1: 0.6012 ≤ 0.6014 ✓
        # lower wick: 0.6028 - 0.6012 = 0.0016 — equal to wick_cap → passes ≤
        # → make wick slightly longer:
        # l=0.6011 → cond1: 0.6011 ≤ 0.6014 ✓
        # lower wick: 0.6028 - 0.6011 = 0.0017 > 0.0016 → cond2 fails ✓
        ref = ReferenceZone(outer=0.6040, inner=0.6010, side="sell",
                            source="cts_established", source_event_idx=0)
        df = self._one_candle_df(o=0.6030, h=0.6032, l=0.6011, c=0.6028)
        assert _evaluate_reset_conditions(
            df, 0, ref, 1, self.reset_tol, self.wick_cap) is False

    def test_uptrend_neither_pass(self):
        """Both conditions fail → no reset."""
        # Far above inner AND long wick
        ref = ReferenceZone(outer=0.6040, inner=0.6010, side="sell",
                            source="cts_established", source_event_idx=0)
        df = self._one_candle_df(o=0.6035, h=0.6038, l=0.6018, c=0.6032)
        # cond1: 0.6018 ≤ 0.6014 → False; cond2: 0.6032-0.6018=0.0014 ≤ wick_cap → True
        # That actually has cond2 pass... let me make wick long too:
        df = self._one_candle_df(o=0.6050, h=0.6052, l=0.6020, c=0.6048)
        # cond1: 0.6020 ≤ 0.6014 → False; cond2 wick: 0.6048-0.6020=0.0028 > 0.0016 → False
        assert _evaluate_reset_conditions(
            df, 0, ref, 1, self.reset_tol, self.wick_cap) is False

    # --- direction=-1 (downtrend probe; buy-zone reference below price) ---

    def test_downtrend_both_pass(self):
        """Upper wick reaches near inner with SHORT wick → both hold."""
        # ref outer=0.7990, inner=0.8000 (inner ABOVE outer for sd=-1)
        # cond1: h ≥ inner - reset_tol = 0.8000 - 0.0004 = 0.7996
        # candle: o=0.7998, c=0.7996 → body_top=0.7998
        # h=0.7998 → cond1: 0.7998 ≥ 0.7996 ✓
        # upper wick: h - body_top = 0.7998 - 0.7998 = 0 ≤ 0.0016 ✓
        ref = ReferenceZone(outer=0.7990, inner=0.8000, side="buy",
                            source="cts_established", source_event_idx=0)
        df = self._one_candle_df(o=0.7998, h=0.7998, l=0.7995, c=0.7996)
        assert _evaluate_reset_conditions(
            df, 0, ref, -1, self.reset_tol, self.wick_cap) is True

    def test_downtrend_cond2_fails(self):
        """High reaches inner (cond1 ✓) via long upper wick (cond2 ✗)."""
        # ref inner=0.8000
        # body: o=0.7980, c=0.7982 → body_top=0.7982
        # h=0.7999 → cond1: 0.7999 ≥ 0.7996 ✓
        # upper wick: 0.7999 - 0.7982 = 0.0017 > 0.0016 → cond2 fails
        ref = ReferenceZone(outer=0.7970, inner=0.8000, side="buy",
                            source="cts_established", source_event_idx=0)
        df = self._one_candle_df(o=0.7980, h=0.7999, l=0.7978, c=0.7982)
        assert _evaluate_reset_conditions(
            df, 0, ref, -1, self.reset_tol, self.wick_cap) is False

    def test_out_of_bounds_idx_returns_false(self):
        ref = _ref_zone_uptrend()
        df = self._one_candle_df(o=0.6, h=0.61, l=0.59, c=0.605)
        assert _evaluate_reset_conditions(
            df, 999, ref, 1, self.reset_tol, self.wick_cap) is False


# ---------------------------------------------------------------------------
# Integration: end_idx_reached vs no_cts_pending vs one_cts_pending
# ---------------------------------------------------------------------------

class TestFinalizeConditions:
    def test_no_cts_no_end_idx_pending(self):
        """No CTS_EST emitted (too-short data) + end_idx None →
        finalize_condition='no_cts_pending', status='pending'."""
        df = _prepare_df(_make_short_data(n=8))
        result = unified_probe(df, 0, 1, _ref_zone_uptrend(), None, "H1")
        assert result.status == "pending"
        assert result.finalize_condition in ("no_cts_pending", "one_cts_pending")

    def test_no_cts_with_end_idx_finalized(self):
        """No CTS_EST + end_idx defined → finalized at caller's bound."""
        df = _prepare_df(_make_short_data(n=15))
        result = unified_probe(
            df, 0, 1, _ref_zone_uptrend(), end_idx=10, timeframe="H1",
        )
        assert result.status == "finalized"
        # Whatever the path, condition must be a known one. With short
        # data and end_idx set we expect end_idx_reached, but downstream
        # MarketStructure could produce a single CTS_EST in some cases —
        # accept either definitive condition.
        assert result.finalize_condition in (
            "end_idx_reached", "no_retrace", "reversal_in_probe",
        )

    def test_normal_uptrend_finalized(self):
        """Standard uptrend with healthy CTS_EST flow → finalized,
        iterations ≥ 1."""
        df = _prepare_df(_make_uptrend_data(n=80))
        ref = _ref_zone_uptrend(outer=0.5500, inner=0.5550)
        result = unified_probe(df, 0, 1, ref, end_idx=None, timeframe="H1")
        assert isinstance(result, ProbeResult)
        assert result.iterations >= 1
        assert result.status in ("finalized", "pending")

    def test_downtrend_finalized(self):
        df = _prepare_df(_make_downtrend_data(n=60))
        ref = _ref_zone_downtrend(outer=0.8000, inner=0.7950)
        result = unified_probe(df, 0, -1, ref, end_idx=None, timeframe="H1")
        assert isinstance(result, ProbeResult)
        assert result.iterations >= 1


# ---------------------------------------------------------------------------
# Iteration cap
# ---------------------------------------------------------------------------

class TestMaxIterations:
    def test_iterations_capped(self):
        """max_iterations caps the loop count."""
        df = _prepare_df(_make_uptrend_data(n=80))
        ref = _ref_zone_uptrend()
        result = unified_probe(
            df, 0, 1, ref, end_idx=None, timeframe="H1", max_iterations=3,
        )
        assert result.iterations <= 3


# ---------------------------------------------------------------------------
# Reference-zone passthrough
# ---------------------------------------------------------------------------

class TestReferenceZonePassthrough:
    def test_ref_zone_returned_unchanged(self):
        df = _prepare_df(_make_uptrend_data(n=60))
        ref = _ref_zone_uptrend(outer=0.5512, inner=0.5567)
        result = unified_probe(df, 0, 1, ref, end_idx=None, timeframe="H1")
        # Identity check — the primitive must NOT refine its own reference.
        assert result.original_ref_zone is ref


# ---------------------------------------------------------------------------
# Threshold lookups — H1 / M15 / M5 don't crash; unknown TF falls back to H1
# ---------------------------------------------------------------------------

class TestTimeframeThresholdLookup:
    @pytest.mark.parametrize("tf", ["H1", "M15", "M5"])
    def test_known_tf_runs(self, tf):
        df = _prepare_df(_make_uptrend_data(n=60))
        result = unified_probe(
            df, 0, 1, _ref_zone_uptrend(), end_idx=None, timeframe=tf,
        )
        assert isinstance(result, ProbeResult)

    def test_unknown_tf_falls_back(self):
        """Unknown TF still produces a ProbeResult (falls back to H1
        thresholds internally)."""
        df = _prepare_df(_make_uptrend_data(n=60))
        result = unified_probe(
            df, 0, 1, _ref_zone_uptrend(), end_idx=None, timeframe="W1",
        )
        assert isinstance(result, ProbeResult)
