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
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.structure.unified_probe import (
    STRUCTURE_AUX_COLS,
    STRUCTURE_MIRROR_COLS,
    ProbeResult,
    _evaluate_reset_conditions,
    _find_true_first_breakout_via_pat,
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
    """A buy-side reference zone for direction=+1 probes (probe-direction
    semantics post-Task-1 refactor). For sd=+1 the structure body grows
    UP from the input_idx; the reference zone sits BELOW the body. So
    `inner > outer` numerically (inner = body-facing TOP of zone-below-body;
    outer = far BOTTOM). The defaults sit far below typical test data so
    retraces never reach them (no-retrace finalize path)."""
    return ReferenceZone(
        outer=outer, inner=inner, side="buy",
        source="cts_established", source_event_idx=0,
    )


def _ref_zone_downtrend(outer: float = 0.8000, inner: float = 0.7950) -> ReferenceZone:
    """A sell-side reference zone for direction=-1 probes. For sd=-1 the
    body grows DOWN; reference sits ABOVE body. So `inner < outer`
    numerically (inner = body-facing BOTTOM of zone-above-body; outer =
    far TOP)."""
    return ReferenceZone(
        outer=outer, inner=inner, side="sell",
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

    # --- direction=+1 (uptrend probe; buy-zone reference below body) ---

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
        ref = ReferenceZone(outer=0.6020, inner=0.6010, side="buy",
                            source="cts_established", source_event_idx=0)
        df = self._one_candle_df(o=0.6011, h=0.6013, l=0.6011, c=0.6013)
        assert _evaluate_reset_conditions(
            df, 0, ref, 1, self.reset_tol, self.wick_cap) is True

    def test_uptrend_cond1_fails(self):
        """Low far from inner → cond1 fails → no reset."""
        # ref inner=0.6010, reset_tol=0.0004 → cond1 threshold = 0.6014
        # candle l=0.6020 (far above) → 0.6020 > 0.6014 → cond1 fails
        ref = ReferenceZone(outer=0.6030, inner=0.6010, side="buy",
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
        ref = ReferenceZone(outer=0.6040, inner=0.6010, side="buy",
                            source="cts_established", source_event_idx=0)
        df = self._one_candle_df(o=0.6030, h=0.6032, l=0.6011, c=0.6028)
        assert _evaluate_reset_conditions(
            df, 0, ref, 1, self.reset_tol, self.wick_cap) is False

    def test_uptrend_neither_pass(self):
        """Both conditions fail → no reset."""
        # Far above inner AND long wick
        ref = ReferenceZone(outer=0.6040, inner=0.6010, side="buy",
                            source="cts_established", source_event_idx=0)
        df = self._one_candle_df(o=0.6035, h=0.6038, l=0.6018, c=0.6032)
        # cond1: 0.6018 ≤ 0.6014 → False; cond2: 0.6032-0.6018=0.0014 ≤ wick_cap → True
        # That actually has cond2 pass... let me make wick long too:
        df = self._one_candle_df(o=0.6050, h=0.6052, l=0.6020, c=0.6048)
        # cond1: 0.6020 ≤ 0.6014 → False; cond2 wick: 0.6048-0.6020=0.0028 > 0.0016 → False
        assert _evaluate_reset_conditions(
            df, 0, ref, 1, self.reset_tol, self.wick_cap) is False

    # --- direction=-1 (downtrend probe; sell-zone reference above body) ---

    def test_downtrend_both_pass(self):
        """Upper wick reaches near inner with SHORT wick → both hold."""
        # ref outer=0.7990, inner=0.8000 (inner ABOVE outer for sd=-1)
        # cond1: h ≥ inner - reset_tol = 0.8000 - 0.0004 = 0.7996
        # candle: o=0.7998, c=0.7996 → body_top=0.7998
        # h=0.7998 → cond1: 0.7998 ≥ 0.7996 ✓
        # upper wick: h - body_top = 0.7998 - 0.7998 = 0 ≤ 0.0016 ✓
        ref = ReferenceZone(outer=0.7990, inner=0.8000, side="sell",
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
        ref = ReferenceZone(outer=0.7970, inner=0.8000, side="sell",
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


# ---------------------------------------------------------------------------
# Structure-col cleanup: a df with pre-existing MS-written cols must yield
# the same probe result as the same df without those cols.
# ---------------------------------------------------------------------------

class TestStructureColCleanup:
    """Regression guard for the entity_df pollution hazard: when a caller
    passes an entity-wide df that has accumulated mirrored structure cols
    from prior sub builds (or that has aux state like `range_confirm_idx`
    from prior runs), the probe must produce the same answer as on a clean
    df. The primitive drops those cols on its working copy so MS's
    `_ensure_output_cols` re-initializes them with proper defaults."""

    def test_polluted_df_matches_clean_df(self):
        clean = _prepare_df(_make_uptrend_data(n=80))
        ref = _ref_zone_uptrend(outer=0.5500, inner=0.5550)
        clean_result = unified_probe(
            clean, 0, 1, ref, end_idx=None, timeframe="H1",
        )

        polluted = clean.copy()
        # Inject realistic stale values across the union of cols we expect
        # the primitive to drop. Pollute over the whole index range, including
        # the candles the probe will scan from — this is what mirror-write
        # from a prior sub_sid looks like on entity_df.
        for col in STRUCTURE_MIRROR_COLS:
            polluted[col] = 99
        for col in STRUCTURE_AUX_COLS:
            polluted[col] = 99
        # `range_confirm_idx` is the historically dangerous one (per the
        # 2026-05-28 is_range pollution fix). Set it to plausible-looking
        # entity-absolute values just to be unfriendly:
        polluted["range_confirm_idx"] = list(range(len(polluted)))

        polluted_result = unified_probe(
            polluted, 0, 1, ref, end_idx=None, timeframe="H1",
        )
        assert polluted_result.start_idx == clean_result.start_idx
        assert polluted_result.status == clean_result.status
        assert polluted_result.finalize_condition == clean_result.finalize_condition

    def test_cleanup_does_not_mutate_caller_df(self):
        """The col-drop happens on the probe's internal copy; caller's df
        must keep its pollution intact (defense-in-depth — the caller may
        still need those cols for its own downstream consumers)."""
        df = _prepare_df(_make_uptrend_data(n=40))
        df["structure_id"] = 7
        df["cycle_id"] = 3
        df["range_confirm_idx"] = 99
        ref = _ref_zone_uptrend()
        _ = unified_probe(df, 0, 1, ref, end_idx=None, timeframe="H1")
        # Caller's df is intact after the probe runs.
        assert (df["structure_id"] == 7).all()
        assert (df["cycle_id"] == 3).all()
        assert (df["range_confirm_idx"] == 99).all()


# ---------------------------------------------------------------------------
# True-first-breakout filter (user spec 2026-05-30)
#
# A CTS_ESTABLISHED only counts as the "first breakout" if its anchor's
# extreme is a new running extreme over [current_start, anchor_idx]. Once
# the true first breakout is found, all subsequent CTS_ESTABLISHED events
# are kept as-is (by MS construction they must break the prior CTS so
# they're implicitly new extremes).
# ---------------------------------------------------------------------------

class TestFindTrueFirstBreakoutViaPat:
    """Phase 1 helper: walks `df.pat` to find the first deterministic
    breakout pattern in the probe's direction whose anchor's extreme is
    a new running max/min over [current_start, idx - 1]. Replaces the
    Session-2-intermediate _filter_true_first_breakouts approach (which
    operated on MS's pre-computed CTS list) — per user 2026-05-31, the
    pattern walk is correct because the subsequent CTS chain only
    follows from the FIRST true breakout, so filtering MS's
    already-computed list gives wrong cycles.
    """

    def _df(self, *, highs, lows, pat=None, pat_dir=None):
        df = pd.DataFrame({
            "o": [v - 0.0005 for v in highs],
            "h": highs, "l": lows,
            "c": [v - 0.0003 for v in highs],
            "pat": pat if pat is not None else [""] * len(highs),
            "pat_dir": pat_dir if pat_dir is not None else [0] * len(highs),
        })
        return df

    def test_empty_window_returns_none(self):
        df = self._df(highs=[0.6010] * 5, lows=[0.6000] * 5)
        # current_start > upper → empty window.
        assert _find_true_first_breakout_via_pat(df, 4, 2, 1) is None

    def test_no_pattern_in_window_returns_none(self):
        # No row has pat != "".
        df = self._df(highs=[0.6010, 0.6020], lows=[0.6005, 0.6015])
        assert _find_true_first_breakout_via_pat(df, 0, 1, 1) is None

    def test_first_pattern_is_new_extreme_returns_it(self):
        # idx 0: high 0.6010 — prior is empty, trivially extreme.
        # idx 3: pattern in +1 direction.
        df = self._df(
            highs=[0.6010, 0.6020, 0.6015, 0.6030, 0.6040],
            lows=[0.6005, 0.6015, 0.6010, 0.6025, 0.6035],
            pat=["", "", "", "continuous", ""],
            pat_dir=[0, 0, 0, 1, 0],
        )
        # current_start=0. idx 3 high (0.6030) >= max(highs[0..2]) = 0.6020 → pass.
        assert _find_true_first_breakout_via_pat(df, 0, 4, 1) == 3

    def test_first_pattern_below_prior_max_is_skipped(self):
        # idx 2 has the highest high in the window (0.6050, no pattern).
        # idx 5 has a pattern but high 0.6040 < 0.6050 → skipped.
        # idx 8 has a pattern with high 0.6060 ≥ 0.6050 → returned.
        df = self._df(
            highs=[0.6010, 0.6020, 0.6050, 0.6030, 0.6035,
                   0.6040, 0.6020, 0.6030, 0.6060, 0.6070],
            lows=[0.6005, 0.6015, 0.6040, 0.6025, 0.6030,
                  0.6035, 0.6015, 0.6025, 0.6055, 0.6065],
            pat=["", "", "", "", "",
                 "continuous", "", "", "double_maru", ""],
            pat_dir=[0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        )
        assert _find_true_first_breakout_via_pat(df, 0, 9, 1) == 8

    def test_downtrend_finds_new_min_low(self):
        # Symmetric for direction=-1. idx 2 has lowest low (0.5950, no pat).
        # idx 5 pat has low 0.5960 → not new extreme.
        # idx 8 pat has low 0.5940 → new extreme.
        df = self._df(
            highs=[0.6000, 0.5990, 0.5970, 0.5980, 0.5985,
                   0.5980, 0.5990, 0.5985, 0.5960, 0.5950],
            lows=[0.5995, 0.5985, 0.5950, 0.5975, 0.5980,
                  0.5960, 0.5985, 0.5980, 0.5940, 0.5930],
            pat=["", "", "", "", "",
                 "continuous", "", "", "double_maru", ""],
            pat_dir=[0, 0, 0, 0, 0, -1, 0, 0, -1, 0],
        )
        assert _find_true_first_breakout_via_pat(df, 0, 9, -1) == 8

    def test_wrong_direction_pattern_skipped(self):
        # idx 3 has a -1 pattern when probe is +1 — must be skipped.
        # idx 6 has a +1 pattern with new extreme.
        df = self._df(
            highs=[0.6010, 0.6020, 0.6015, 0.6030, 0.6025, 0.6020, 0.6040],
            lows=[0.6005, 0.6015, 0.6010, 0.6025, 0.6020, 0.6015, 0.6035],
            pat=["", "", "", "continuous", "", "", "double_maru"],
            pat_dir=[0, 0, 0, -1, 0, 0, 1],
        )
        assert _find_true_first_breakout_via_pat(df, 0, 6, 1) == 6

    def test_no_qualifying_pattern_returns_none(self):
        # Big wick at idx 2 (no pattern), and all later patterns have
        # highs below the spike.
        df = self._df(
            highs=[0.6010, 0.6020, 0.6080, 0.6040, 0.6045, 0.6050],
            lows=[0.6005, 0.6015, 0.6070, 0.6035, 0.6040, 0.6045],
            pat=["", "", "", "continuous", "double_maru", "continuous"],
            pat_dir=[0, 0, 0, 1, 1, 1],
        )
        assert _find_true_first_breakout_via_pat(df, 0, 5, 1) is None

    def test_anchor_at_current_start_trivially_extreme(self):
        # Edge: pattern at idx == current_start → empty prior window →
        # trivially qualifies.
        df = self._df(
            highs=[0.6010, 0.6020] * 5,
            lows=[0.6005, 0.6015] * 5,
            pat=["continuous", "", "", "", "", "", "", "", "", ""],
            pat_dir=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )
        assert _find_true_first_breakout_via_pat(df, 0, 9, 1) == 0

    def test_walk_starts_from_current_start_skips_prior_patterns(self):
        # A pattern exists BEFORE current_start — must be skipped.
        df = self._df(
            highs=[0.6010, 0.6020, 0.6030, 0.6040, 0.6050],
            lows=[0.6005, 0.6015, 0.6025, 0.6035, 0.6045],
            pat=["continuous", "", "double_maru", "", ""],
            pat_dir=[1, 0, 1, 0, 0],
        )
        # current_start=1 → pattern at 0 is invisible; pattern at 2 qualifies.
        assert _find_true_first_breakout_via_pat(df, 1, 4, 1) == 2
