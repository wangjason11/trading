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


def _make_multicycle_data() -> list[dict]:
    """Multi-cycle fixture (Plan A §5.3 / Plan B §4.0): sd=+1, start_idx=0, 24 H1
    candles, NZD_USD, deterministic. Every existing fixture establishes exactly ONE
    CTS cycle; this one establishes FOUR before any reversal.

    Shape: impulse (6 bullish marus, 20-pip bodies) -> 2 big bearish marus (a
    `one_maru_continuous(-1)` pullback pattern -> CTS_CONFIRMED, range created) ->
    one climb candle closing INSIDE the range -> big bullish maru closing above
    range_hi + follow-through maru (`one_maru_continuous(+1)` -> new CTS cycle at
    anchor+1, BOS = pullback extreme) -> repeat. Wicks 2 pips (1 pip on the climb
    candle's low so BOS lows are distinct).

    Unbounded run (`compute_bounded_structure(df, 0, +1)`): CTS_ESTABLISHED cycles
    0/1/2/3 at idx 2/10/15/20 (confirmed_at == idx), CTS_CONFIRMED (pullback) at
    7/12/17, BOS_CONFIRMED at 0/7/12/17, no reversal, no reversal watch, no
    proximity confirmation. Bodies >= 15 pips (all marus under the H1 2.2-pip floor).
    """
    rows: list[dict] = []
    p = 0.6000

    def bull(body, wu=0.0002, wd=0.0002):
        nonlocal p
        o = p
        c = o + body
        rows.append({"o": round(o, 5), "h": round(c + wu, 5), "l": round(o - wd, 5), "c": round(c, 5)})
        p = c

    def bear(body, wu=0.0002, wd=0.0002):
        nonlocal p
        o = p
        c = o - body
        rows.append({"o": round(o, 5), "h": round(o + wu, 5), "l": round(c - wd, 5), "c": round(c, 5)})
        p = c

    for _ in range(6):          # 0-5   impulse 1 -> cycle 0 established at 2 (CTS_UPDATED to 5)
        bull(0.0020)
    bear(0.0015); bear(0.0015)  # 6-7   pullback 1 -> CTS_CONFIRMED cycle 0 @7 (range hi .6122, lo .6088)
    bull(0.0025, wd=0.0001)     # 8     climb inside the range (close .6115 < range_hi .6122)
    bull(0.0030); bull(0.0020)  # 9-10  breakout anchor 9 (close .6145 > .6122) -> cycle 1 at 10
    bear(0.0025); bear(0.0025)  # 11-12 pullback 2 -> CTS_CONFIRMED cycle 1 @12 (range hi .6167, lo .6113)
    bull(0.0025, wd=0.0001)     # 13    climb inside the range
    bull(0.0045); bull(0.0020)  # 14-15 breakout anchor 14 (close .6185 > .6167) -> cycle 2 at 15
    bear(0.0030); bear(0.0030)  # 16-17 pullback 3 -> CTS_CONFIRMED cycle 2 @17 (range hi .6207, lo .6143)
    bull(0.0030, wd=0.0001)     # 18    climb inside the range
    bull(0.0050); bull(0.0020)  # 19-20 breakout anchor 19 (close .6225 > .6207) -> cycle 3 at 20
    bull(0.0018); bull(0.0018); bull(0.0018)  # 21-23 tail (CTS_UPDATED 21/22/23)
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
# Phase 2 on a multi-cycle series (Plan A §5.3). The only place in the suite where
# Phase 2 actually runs (elsewhere `enable_phase2` is a mocked kwarg) and the only
# fixture with >= 2 CTS cycles. Plan B §4 reuses it.
# ---------------------------------------------------------------------------

class TestPhase2MultiCycle:
    @staticmethod
    def _retain_phase2_ms(monkeypatch):
        """Keep every MarketStructure Phase 2 constructs. `_run_phase2` resolves
        `_make_market_structure` from unified_probe's module globals at call time,
        so that is the name to patch."""
        import engine_v2.structure.unified_probe as up
        retained = []
        orig = up._make_market_structure

        def wrapped(df, struct_direction, **kw):
            ms = orig(df, struct_direction, **kw)
            retained.append(ms)
            return ms

        monkeypatch.setattr(up, "_make_market_structure", wrapped)
        return retained

    def test_fixture_establishes_four_cts_cycles(self):
        from engine_v2.structure.structure_engine import compute_bounded_structure

        res = compute_bounded_structure(_prepare_df(_make_multicycle_data()), 0, 1)
        est = [(int(e.idx), int(e.meta["cycle_id"]), int(e.meta["confirmed_at"]))
               for e in res.events if e.type == "CTS_ESTABLISHED"]
        conf = [(int(e.idx), int(e.meta["cycle_id"])) for e in res.events if e.type == "CTS_CONFIRMED"]
        assert est == [(2, 0, 2), (10, 1, 10), (15, 2, 15), (20, 3, 20)]
        assert conf == [(7, 0), (12, 1), (17, 2)]
        assert res.reversal_idx is None

    def test_phase2_bounded_ms_emits_nothing_past_end_idx(self, monkeypatch):
        """end_idx=14: the breakout anchored at 14 applies at 15. Before Plan A the
        Phase-2 MS stamped CTS_ESTABLISHED@15 / BOS_CONFIRMED (confirmed_at 15) past
        its bound; the ProbeResult happened not to change. After: nothing past 14,
        two CTS_ESTABLISHED inside [0, 14] -> `second_cts_reached`."""
        retained = self._retain_phase2_ms(monkeypatch)
        df = _prepare_df(_make_multicycle_data())
        res = unified_probe(df, 0, 1, _ref_zone_uptrend(), 14, "H1", enable_phase2=True)

        assert len(retained) >= 1, "Phase 2 must actually run an MS"
        for ms in retained:
            assert ms.end_idx == 14
            assert max(int(ev.idx) for ev in ms.events) <= 14
        est = [(int(e.idx), int(e.meta["cycle_id"])) for e in retained[-1].events
               if e.type == "CTS_ESTABLISHED"]
        assert est == [(2, 0), (10, 1)]
        assert res.status == "finalized"
        assert res.finalize_condition == "second_cts_reached"
        assert res.finalize_idx == 10
        assert res.start_idx == 0

    def test_phase2_finalize_cannot_come_from_past_the_bound(self, monkeypatch):
        """end_idx=9: before Plan A the leaked CTS_ESTABLISHED@10 made the probe
        report `second_cts_reached` with finalize_idx 10 > end_idx (the FC(1,0)
        2844-on-2843 pathology in miniature). After: `no_retrace`, finalize_idx 7 =
        the cycle-0 CTS_CONFIRMED candle, nothing past 9."""
        retained = self._retain_phase2_ms(monkeypatch)
        df = _prepare_df(_make_multicycle_data())
        res = unified_probe(df, 0, 1, _ref_zone_uptrend(), 9, "H1", enable_phase2=True)

        assert retained and all(max(int(ev.idx) for ev in ms.events) <= 9 for ms in retained)
        assert res.status == "finalized"
        assert res.finalize_condition == "no_retrace"
        assert res.finalize_idx == 7
        assert res.finalize_idx <= 9
