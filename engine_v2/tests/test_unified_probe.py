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
  - Plan C §7 renames (the probe's search bound is `probe_end_idx`, its output
    anchor is `ProbeResult.starting_idx` / `_DetResult.starting_idx`; the old
    spellings are gone) — `TestPlanCRenames`
  - Plan C §9.3: the `finalize_idx` per-`finalize_condition` table of the
    `ProbeResult` docstring, one test per row — `TestFinalizeIdxTable`

Vocabulary (Plan C §1 / §7): `starting_idx` is HISTORICAL (the structural anchor,
the pool key); `start_idx` is a REAL-TIME lifecycle value and does not exist on a
probe result. `probe_end_idx` is a COMPUTE bound (inclusive upper edge of the search
window, like `run_cap`), unrelated to a lifecycle `end_idx`. `MarketStructure`'s
own `end_idx` (the MS bound) is NOT renamed — `ms.end_idx` reads below are correct.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from engine_v2.structure import event_fields as ef
from engine_v2.features.candle_classifier import apply_candle_classification
from engine_v2.patterns.imbalance import compute_imbalance
from engine_v2.patterns.pattern_engine import detect_patterns
from engine_v2.patterns.structure_patterns import BreakoutPatterns
from engine_v2.structure.reference_zone import ReferenceZone
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.tests._event_factory import make_cts_established
from engine_v2.structure.unified_probe import (
    STRUCTURE_AUX_COLS,
    STRUCTURE_MIRROR_COLS,
    ProbeResult,
    _DetResult,
    _evaluate_reset_conditions,
    _run_phase1,
    _run_phase2,
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
        source="cts_established", anchor_idx=0,
    )


def _ref_zone_downtrend(outer: float = 0.8000, inner: float = 0.7950) -> ReferenceZone:
    """A sell-side reference zone for direction=-1 probes. For sd=-1 the
    body grows DOWN; reference sits ABOVE body. So `inner < outer`
    numerically (inner = body-facing BOTTOM of zone-above-body; outer =
    far TOP)."""
    return ReferenceZone(
        outer=outer, inner=inner, side="sell",
        source="cts_established", anchor_idx=0,
    )


def _R(o: float, h: float, l: float, c: float) -> dict:
    return {"o": round(o, 5), "h": round(h, 5), "l": round(l, 5), "c": round(c, 5)}


def _make_second_cts_moment_after_anchor_data() -> list[dict]:
    """The 2nd CTS's EXTREME (its anchor `meta["cts_anchor_idx"]` 9) precedes its MOMENT
    (`meta["confirmed_at"]` 10, its `idx` since Plan E E4a) — the natural anchor != moment
    case the Plan B / Plan C
    `second_cts_reached` rule is about. Byte-for-byte the candles of
    `test_ms_stop_after_cts._make_watch_over_second_cts_data` (crafted + independently
    verified 2026-09-20; copied here so this file stays self-contained and import-acyclic —
    that module imports THIS file's fixtures). sd=+1, start 0, 18 H1 candles, NZD_USD.

    Candles 0-8 = `_make_multicycle_data()[:9]` (cycle 0 established at 2, CTS_CONFIRMED at 7
    with `cts_anchor_idx` 5, range hi .6122 / lo .6088, climb 8 inside it).
    9   big bull maru closing .6145 ABOVE range_hi = c0 of `one_maru_opposite(+1)`; its high
        .6147 is the cycle-1 CTS extreme (the CTS anchor 9).
    10  SMALL bearish normal -> OMO SUCCESS, apply 10 -> 2nd CTS_ESTABLISHED (anchor 9,
        confirmed_at = idx 10), BOS_1 = l7 .6088.
    11  big bear maru closing .6060 < BOS_1 -> REVERSAL_WATCH_START (expires 16) +
        REVERSAL_CANDIDATE anchor 11 apply 14.
    12-13 bull pinbar / small bull normal (the -1 confirmation scan skips them).
    14  bear maru closing .6035 -> reversal applied at 14 (terminal); quiescent from here, so
        Plan B's early stop lands at 15.
    15-17 bull fillers.

    Under `unified_probe(..., probe_end_idx=17, enable_phase2=True)`: n_cts = 2 at the exit
    (so the reversal at 14 is NOT `reversal_in_probe`), no retrace resets ->
    `second_cts_reached`, finalize = the 2nd CTS's moment 10 (NOT its anchor 9).
    """
    rows = list(_make_multicycle_data()[:9])
    rows += [
        _R(0.61150, 0.61470, 0.61130, 0.61450),   # 9
        _R(0.61440, 0.61460, 0.61360, 0.61380),   # 10
        _R(0.61380, 0.61400, 0.60580, 0.60600),   # 11
        _R(0.61000, 0.61460, 0.60400, 0.61400),   # 12
        _R(0.61400, 0.61460, 0.61380, 0.61440),   # 13
        _R(0.61440, 0.61460, 0.60330, 0.60350),   # 14
        _R(0.60350, 0.60430, 0.60330, 0.60410),   # 15
        _R(0.60410, 0.60490, 0.60390, 0.60470),   # 16
        _R(0.60470, 0.60550, 0.60450, 0.60530),   # 17
    ]
    return rows


def _make_cycle0_reversal_data() -> list[dict]:
    """A structure that REVERSES inside cycle 0 (the `reversal_in_probe` row: reversal
    with < 2 CTS_ESTABLISHED). sd=+1, start 0, 13 H1 candles, NZD_USD; verified by running
    it (2026-09-20).

    Candles 0-8 = `_make_multicycle_data()[:9]` (cycle 0 established at 2; BOS_0 = the
    structure's start low l0 = .5998 (candle 0: o .6000, 2-pip lower wick); CTS_CONFIRMED at
    7; range hi .6122 / lo .6088; climb 8 closes .6115 inside it).
    9   big bear maru o .6115 -> c .5990: close < BOS_0 .5998 -> BOS close-break ->
        REVERSAL_WATCH_START@9 (expires min(9+5, edge 12) = 12) and the -1 pattern anchored
        at 9 (`detect_best_for_anchor(9, -1, bos_frozen=.5998)`) applies at 10 ->
        REVERSAL_CANDIDATE anchor 9 apply 10.
    10  follow-through bear maru (the pattern's apply candle) -> the reversal APPLIES at 10
        (terminal): `market_state == "reversal"` from 10, `reversal_idx` 10.
    11-12 small bull fillers (nothing is read after the terminal apply).

    Under the probe: Phase 1 finds cycle 0 at 2 (any bound >= 2); Phase 2's MS run ends in
    reversal with n_cts = 1 < 2 -> `reversal_in_probe`, finalize = the reversal apply idx 10.
    """
    rows = list(_make_multicycle_data()[:9])
    rows += [
        _R(0.61150, 0.61170, 0.59880, 0.59900),   # 9   close-breaks BOS_0 .5998
        _R(0.59900, 0.59920, 0.59680, 0.59700),   # 10  reversal apply candle
        _R(0.59700, 0.59780, 0.59680, 0.59760),   # 11
        _R(0.59760, 0.59840, 0.59740, 0.59820),   # 12
    ]
    return rows


def _retain_phase2_ms(monkeypatch) -> list:
    """Keep every MarketStructure Phase 2 constructs (so a test can read the probe's OWN
    events / df). `_run_phase2` resolves `_make_market_structure` from unified_probe's
    module globals at call time, so that is the name to patch. Module-level twin of
    `TestPhase2MultiCycle._retain_phase2_ms`."""
    import engine_v2.structure.unified_probe as up
    retained = []
    orig = up._make_market_structure

    def wrapped(df, **kw):
        ms = orig(df, **kw)
        retained.append(ms)
        return ms

    monkeypatch.setattr(up, "_make_market_structure", wrapped)
    return retained


# ---------------------------------------------------------------------------
# ProbeResult dataclass
# ---------------------------------------------------------------------------

class TestProbeResult:
    def test_construct_all_fields(self):
        ref = _ref_zone_uptrend()
        r = ProbeResult(
            starting_idx=10,
            status="finalized",
            iterations=2,
            original_ref_zone=ref,
            finalize_condition="no_retrace",
            notes="hi",
        )
        assert r.starting_idx == 10
        assert r.status == "finalized"
        assert r.iterations == 2
        assert r.original_ref_zone is ref
        assert r.finalize_condition == "no_retrace"
        assert r.notes == "hi"

    def test_notes_default_empty(self):
        ref = _ref_zone_uptrend()
        r = ProbeResult(
            starting_idx=0, status="pending", iterations=1,
            original_ref_zone=ref, finalize_condition="no_cts_pending",
        )
        assert r.notes == ""

    def test_is_frozen(self):
        ref = _ref_zone_uptrend()
        r = ProbeResult(
            starting_idx=0, status="pending", iterations=1,
            original_ref_zone=ref, finalize_condition="no_cts_pending",
        )
        with pytest.raises(Exception):
            r.starting_idx = 5  # type: ignore[misc]


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
                            source="cts_established", anchor_idx=0)
        df = self._one_candle_df(o=0.6011, h=0.6013, l=0.6011, c=0.6013)
        assert _evaluate_reset_conditions(
            df, 0, ref, 1, self.reset_tol, self.wick_cap) is True

    def test_uptrend_cond1_fails(self):
        """Low far from inner → cond1 fails → no reset."""
        # ref inner=0.6010, reset_tol=0.0004 → cond1 threshold = 0.6014
        # candle l=0.6020 (far above) → 0.6020 > 0.6014 → cond1 fails
        ref = ReferenceZone(outer=0.6030, inner=0.6010, side="buy",
                            source="cts_established", anchor_idx=0)
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
                            source="cts_established", anchor_idx=0)
        df = self._one_candle_df(o=0.6030, h=0.6032, l=0.6011, c=0.6028)
        assert _evaluate_reset_conditions(
            df, 0, ref, 1, self.reset_tol, self.wick_cap) is False

    def test_uptrend_neither_pass(self):
        """Both conditions fail → no reset."""
        # Far above inner AND long wick
        ref = ReferenceZone(outer=0.6040, inner=0.6010, side="buy",
                            source="cts_established", anchor_idx=0)
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
                            source="cts_established", anchor_idx=0)
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
                            source="cts_established", anchor_idx=0)
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
        """No CTS_EST emitted (too-short data) + probe_end_idx None →
        finalize_condition='no_cts_pending', status='pending'."""
        df = _prepare_df(_make_short_data(n=8))
        result = unified_probe(df, 0, 1, _ref_zone_uptrend(), None, "H1")
        assert result.status == "pending"
        assert result.finalize_condition in ("no_cts_pending", "one_cts_pending")

    def test_no_cts_with_end_idx_finalized(self):
        """No CTS_EST + probe_end_idx defined → finalized at caller's bound."""
        df = _prepare_df(_make_short_data(n=15))
        result = unified_probe(
            df, 0, 1, _ref_zone_uptrend(), probe_end_idx=10, timeframe="H1",
        )
        assert result.status == "finalized"
        # Whatever the path, condition must be a known one. With short
        # data and probe_end_idx set we expect end_idx_reached, but downstream
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
        result = unified_probe(df, 0, 1, ref, probe_end_idx=None, timeframe="H1")
        assert isinstance(result, ProbeResult)
        assert result.iterations >= 1
        assert result.status in ("finalized", "pending")

    def test_downtrend_finalized(self):
        df = _prepare_df(_make_downtrend_data(n=60))
        ref = _ref_zone_downtrend(outer=0.8000, inner=0.7950)
        result = unified_probe(df, 0, -1, ref, probe_end_idx=None, timeframe="H1")
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
            df, 0, 1, ref, probe_end_idx=None, timeframe="H1", max_iterations=3,
        )
        assert result.iterations <= 3


# ---------------------------------------------------------------------------
# Reference-zone passthrough
# ---------------------------------------------------------------------------

class TestReferenceZonePassthrough:
    def test_ref_zone_returned_unchanged(self):
        df = _prepare_df(_make_uptrend_data(n=60))
        ref = _ref_zone_uptrend(outer=0.5512, inner=0.5567)
        result = unified_probe(df, 0, 1, ref, probe_end_idx=None, timeframe="H1")
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
            df, 0, 1, _ref_zone_uptrend(), probe_end_idx=None, timeframe=tf,
        )
        assert isinstance(result, ProbeResult)

    def test_unknown_tf_falls_back(self):
        """Unknown TF still produces a ProbeResult (falls back to H1
        thresholds internally)."""
        df = _prepare_df(_make_uptrend_data(n=60))
        result = unified_probe(
            df, 0, 1, _ref_zone_uptrend(), probe_end_idx=None, timeframe="W1",
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
            clean, 0, 1, ref, probe_end_idx=None, timeframe="H1",
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
            polluted, 0, 1, ref, probe_end_idx=None, timeframe="H1",
        )
        assert polluted_result.starting_idx == clean_result.starting_idx
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
        _ = unified_probe(df, 0, 1, ref, probe_end_idx=None, timeframe="H1")
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
        """probe_end_idx=14: the breakout anchored at 14 applies at 15. Before Plan A the
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
        assert res.starting_idx == 0

    def test_phase2_finalize_cannot_come_from_past_the_bound(self, monkeypatch):
        """probe_end_idx=9: before Plan A the leaked CTS_ESTABLISHED@10 made the probe
        report `second_cts_reached` with finalize_idx 10 > probe_end_idx (the FC(1,0)
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


# ---------------------------------------------------------------------------
# Plan B — the double-CTS rule is a true early stop in Phase 2
# ---------------------------------------------------------------------------

class TestPhase2EarlyStop:
    """Plan B §4.2 / §4.3. Phase 2 passes `stop_after_cts_established=2` to its MS; the
    classification then runs on the truncated event list and must equal classify-at-exit."""

    @staticmethod
    def _retain(monkeypatch, *, disable_stop: bool):
        import engine_v2.structure.unified_probe as up
        retained = []
        orig = up._make_market_structure

        def wrapped(df, **kw):
            if disable_stop:
                kw.pop("stop_after_cts_established", None)
            ms = orig(df, **kw)
            retained.append(ms)
            return ms

        monkeypatch.setattr(up, "_make_market_structure", wrapped)
        return retained

    def test_phase2_passes_the_stop_option_and_stops_early(self, monkeypatch, capsys):
        """probe_end_idx=22 (past the 3rd cycle at 15 and the 4th at 20): the shipped probe stops
        at the first quiescent point after the 2nd CTS (11) and reads nothing later."""
        retained = self._retain(monkeypatch, disable_stop=False)
        df = _prepare_df(_make_multicycle_data())
        res = unified_probe(df, 0, 1, _ref_zone_uptrend(), 22, "H1", enable_phase2=True)
        assert len(retained) == 1
        ms = retained[0]
        assert ms.stop_after_cts_established == 2
        assert ms.early_stop_idx == 11
        est = [(int(e.idx), int(e.meta["confirmed_at"])) for e in ms.events
               if e.type == "CTS_ESTABLISHED"]
        assert est == [(2, 2), (10, 10)]
        assert max(int(e.idx) for e in ms.events) <= 10
        assert res.finalize_condition == "second_cts_reached"
        assert res.finalize_idx == 10
        assert res.starting_idx == 0
        out = capsys.readouterr().out
        stop_lines = [l for l in out.splitlines() if l.startswith("[unified_probe phase2] early stop")]
        assert len(stop_lines) == 1
        # PLAN-AMBIGUITY: Plan C §7 lists the three signatures + the ProbeResult docstring +
        # the §5.3 cache log strings as rename sites, but its acceptance grep ("only the
        # probe_end_idx / starting_idx spellings remain in unified_probe.py") also covers this
        # early-stop print's `end_idx=` label. Either label spelling is accepted; the VALUES
        # (stop 11, bound 22, extreme 10 == moment 10) are pinned exactly.
        assert re.fullmatch(
            r"\[unified_probe phase2\] early stop: p2_iter=1 stop_idx=11 "
            r"(probe_)?end_idx=22 cts1_anchor=10 cts1_moment=10",
            stop_lines[0],
        ), stop_lines[0]

    @pytest.mark.parametrize("probe_end_idx", [10, 11, 14, 22, 23])
    def test_early_stop_equals_classify_at_exit(self, monkeypatch, probe_end_idx):
        """§2 equivalence claim in unit form: the shipped probe (early stop) and the same
        probe with the stop popped (MS runs to probe_end_idx, classify at exit) return the same
        `_DetResult` field by field. `probe_end_idx=10` is the edge where the 2nd CTS lands on the
        last in-bound step: two CTS_ESTABLISHED, `second_cts_reached`, but NO early stop
        (`early_stop_idx` None, no print) — "stopped early" is never derived from the count."""
        from engine_v2.structure.unified_probe import _run_phase2
        df = _prepare_df(_make_multicycle_data())
        ref = _ref_zone_uptrend()
        pip = 0.0001
        kw = dict(reset_tol=DEFAULT_PROBE_RESET_PIPS["H1"] * pip,
                  wick_cap=DEFAULT_PROBE_RESET_WICK["H1"] * pip, max_iterations=10)

        retained_on = self._retain(monkeypatch, disable_stop=False)
        on = _run_phase2(df, 0, ref.inner, ref.outer, 1, ref, probe_end_idx, **kw)
        monkeypatch.undo()
        retained_off = self._retain(monkeypatch, disable_stop=True)
        off = _run_phase2(df, 0, ref.inner, ref.outer, 1, ref, probe_end_idx, **kw)

        assert on == off
        assert on.finalize_condition == "second_cts_reached"
        assert on.finalize_idx == 10
        assert on.starting_idx == 0          # Plan C §7: `_DetResult.starting_idx`
        assert retained_off[-1].early_stop_idx is None
        sig = lambda evs: [(e.type, int(e.idx), e.price, repr(sorted(e.meta.items()))) for e in evs]
        if probe_end_idx == 10:
            # Nothing to pre-empt: the run ends at its bound, no early stop is claimed.
            assert retained_on[-1].early_stop_idx is None
            assert sig(retained_on[-1].events) == sig(retained_off[-1].events)
        else:
            # The stopped run really did read less than the full one (a prefix of it).
            assert retained_on[-1].early_stop_idx == 11
            assert len(retained_on[-1].events) <= len(retained_off[-1].events)
            assert sig(retained_off[-1].events)[: len(retained_on[-1].events)] == sig(retained_on[-1].events)

    def test_no_second_cts_runs_are_untouched(self, monkeypatch):
        """`n_cts <= 1` at `probe_end_idx` (=9): the option is inert, the run reaches the bound,
        and the classification is the Plan A `no_retrace` / finalize 7 result."""
        retained = self._retain(monkeypatch, disable_stop=False)
        df = _prepare_df(_make_multicycle_data())
        res = unified_probe(df, 0, 1, _ref_zone_uptrend(), 9, "H1", enable_phase2=True)
        assert retained[-1].early_stop_idx is None
        assert res.finalize_condition == "no_retrace"
        assert res.finalize_idx == 7


class TestSecondCtsMoment:
    """Plan B §3.3 / §4.3 — `second_cts_reached` finalizes at the 2nd CTS's MOMENT
    (`meta["confirmed_at"]`), not its `.idx` (the extreme inside the pattern span)."""

    def test_returns_the_moment_not_the_anchor(self):
        from engine_v2.structure.unified_probe import _second_cts_moment
        cts_est = [
            make_cts_established(cts_anchor_idx=458, confirmed_at=458, price=0.5,
                                 cycle_id=0, struct_direction=None),
            make_cts_established(cts_anchor_idx=1223, confirmed_at=1224, price=0.5,
                                 cycle_id=1, struct_direction=None),
        ]
        assert _second_cts_moment(cts_est) == 1224

    @pytest.mark.illegal_event_contract  # a CTS_ESTABLISHED without its moment (contract-illegal)
    def test_raises_without_the_meta_key(self):
        """Plan E E2d: no `.get("confirmed_at", ev.idx)` fallback to the anchor —
        a missing moment raises (LANDMINES "Event Contract Rules")."""
        from engine_v2.structure.unified_probe import _second_cts_moment
        cts_est = [
            StructureEvent(idx=1, category="STRUCTURE", type="CTS_ESTABLISHED", price=0.5, meta={}),
            StructureEvent(idx=9, category="STRUCTURE", type="CTS_ESTABLISHED", price=0.5, meta={}),
        ]
        with pytest.raises(KeyError):
            _second_cts_moment(cts_est)


# ---------------------------------------------------------------------------
# Plan C §7 — the renames: `probe_end_idx` (the search bound) and `starting_idx`
# (the output anchor) are the ONLY spellings on the probe's API.
# ---------------------------------------------------------------------------

class TestPlanCRenames:
    """Plan C §7: `unified_probe(..., end_idx=...)` -> `probe_end_idx=` on all three
    signatures (`unified_probe`, `_run_phase1`, `_run_phase2`); `ProbeResult.start_idx` /
    `_DetResult.start_idx` -> `starting_idx`. Hard renames, no alias: the old keyword is a
    `TypeError`, the old attribute does not exist. `MarketStructure(end_idx=...)` and
    `BreakoutPatterns(end_idx=...)` are MS/detector bounds and are NOT renamed."""

    _pip = 0.0001   # NZD_USD
    _kw = dict(
        reset_tol=float(DEFAULT_PROBE_RESET_PIPS["H1"]) * _pip,
        wick_cap=float(DEFAULT_PROBE_RESET_WICK["H1"]) * _pip,
        max_iterations=10,
    )

    def test_probe_result_starting_idx_is_the_only_spelling(self):
        ref = _ref_zone_uptrend()
        r = ProbeResult(
            starting_idx=10, status="finalized", iterations=1,
            original_ref_zone=ref, finalize_condition="no_retrace",
        )
        assert r.starting_idx == 10
        assert not hasattr(r, "start_idx")
        with pytest.raises(TypeError):
            ProbeResult(
                start_idx=10, status="finalized", iterations=1,   # type: ignore[call-arg]
                original_ref_zone=ref, finalize_condition="no_retrace",
            )

    def test_det_result_starting_idx_is_the_only_spelling(self):
        d = _DetResult(
            starting_idx=7, status="pending", finalize_condition="max_iterations",
            iterations=1, bos0_inner=0.6090, bos0_outer=0.6050, cts0_established_idx=5,
            finalize_idx=None,
        )
        assert d.starting_idx == 7
        assert not hasattr(d, "start_idx")
        with pytest.raises(TypeError):
            _DetResult(
                start_idx=7, status="pending", finalize_condition="max_iterations",   # type: ignore[call-arg]
                iterations=1, bos0_inner=0.6090, bos0_outer=0.6050, cts0_established_idx=5,
                finalize_idx=None,
            )

    def test_unified_probe_rejects_end_idx_keyword(self):
        df = _prepare_df(_make_multicycle_data())
        with pytest.raises(TypeError):
            unified_probe(df, 0, 1, _ref_zone_uptrend(), end_idx=14, timeframe="H1")   # type: ignore[call-arg]

    def test_unified_probe_probe_end_idx_keyword_is_the_positional_bound(self):
        """`probe_end_idx=` by keyword == the 5th positional argument (the bound)."""
        df = _prepare_df(_make_multicycle_data())
        ref = _ref_zone_uptrend()
        by_kw = unified_probe(df, 0, 1, ref, probe_end_idx=14, timeframe="H1")
        by_pos = unified_probe(df, 0, 1, ref, 14, "H1")
        assert by_kw == by_pos
        # Phase 1 `no_retrace` on this fixture -> finalize = the bound (see TestFinalizeIdxTable)
        assert by_kw.finalize_condition == "no_retrace"
        assert by_kw.finalize_idx == 14
        assert by_kw.starting_idx == 0

    def test_run_phase1_rejects_end_idx_keyword(self):
        df = _prepare_df(_make_multicycle_data())
        bp = BreakoutPatterns(df, end_idx=14)   # detector bound — not renamed
        with pytest.raises(TypeError):
            _run_phase1(df, bp, 0, 1, _ref_zone_uptrend(), end_idx=14, **self._kw)   # type: ignore[call-arg]

    def test_run_phase1_accepts_probe_end_idx_keyword(self):
        df = _prepare_df(_make_multicycle_data())
        bp = BreakoutPatterns(df, end_idx=14)
        det = _run_phase1(df, bp, 0, 1, _ref_zone_uptrend(), probe_end_idx=14, **self._kw)
        assert isinstance(det, _DetResult)
        assert det.starting_idx == 0
        assert det.finalize_condition == "no_retrace"
        assert det.finalize_idx == 14          # Phase 1 finalized -> the bound

    def test_run_phase2_rejects_end_idx_keyword(self):
        df = _prepare_df(_make_multicycle_data())
        ref = _ref_zone_uptrend()
        with pytest.raises(TypeError):
            _run_phase2(df, 0, ref.inner, ref.outer, 1, ref, end_idx=9, **self._kw)   # type: ignore[call-arg]

    def test_run_phase2_accepts_probe_end_idx_keyword(self):
        df = _prepare_df(_make_multicycle_data())
        ref = _ref_zone_uptrend()
        p2 = _run_phase2(df, 0, ref.inner, ref.outer, 1, ref, probe_end_idx=9, **self._kw)
        assert isinstance(p2, _DetResult)
        assert p2.starting_idx == 0
        # cycle 0 confirmed in-window (CTS_CONFIRMED@7 <= 9) -> `no_retrace`, finalize = 7
        assert p2.finalize_condition == "no_retrace"
        assert p2.finalize_idx == 7


# ---------------------------------------------------------------------------
# Plan C §9.3 — `finalize_idx` per finalize_condition (the ProbeResult docstring
# table). Before this class the file had ZERO assertions on most rows.
# ---------------------------------------------------------------------------

class TestFinalizeIdxTable:
    """One test per row of the `ProbeResult.finalize_idx` table:

        Phase 1 `no_retrace`                          -> `probe_end_idx`
        Phase 1 `end_idx_reached`                     -> `probe_end_idx`
        Phase 2 `second_cts_reached`                  -> 2nd CTS_ESTABLISHED.meta["confirmed_at"]
                                                         (the MOMENT — not `.idx`, the extreme)
        Phase 2 `reversal_in_probe`                   -> the reversal apply idx
        Phase 2 `no_retrace`, cycle 0 confirmed       -> CTS_0_CONFIRMED.idx
        Phase 2 `no_retrace`, cycle 0 NOT confirmed   -> `probe_end_idx`
        Phase 2 `end_idx_reached`                     -> `probe_end_idx`
        pending (`no_cts_pending` / `one_cts_pending` / `max_iterations`) -> None

    Every expected value is hand-derived in the test body and cross-checked against the
    probe's OWN Phase-2 MS run (retained via `_retain_phase2_ms`) where one exists.

    Fixture facts (`_make_multicycle_data`, sd=+1, ref inner .5550 far below the data so no
    retrace ever resets): cycle 0 CTS_ESTABLISHED idx 2 / confirmed_at 2; CTS_CONFIRMED idx 7
    with `cts_anchor_idx` 5; cycle 1 CTS_ESTABLISHED idx 10 / confirmed_at 10; no reversal.
    Lows: l3 .6058, l7 .6088 (the deepest retrace after cycle 0).

    Not separately constructed: Phase 2 `max_iterations` (needs a candidate that fails cond2
    in Phase 1's wider window `[est+1, bound]` yet passes in Phase 2's narrower
    `[est+1, cts_anchor-1]`); the row's `None` rule is pinned on the Phase-1 path, and Phase 2
    initialises `finalize_idx = None` and writes it only on a finalized exit.
    """

    _pip = 0.0001   # NZD_USD

    # --- Phase 1 (enable_phase2=False: no MS at all) --------------------------------

    @pytest.mark.parametrize("probe_end_idx", [9, 14, 22])
    def test_phase1_no_retrace_is_probe_end_idx(self, probe_end_idx):
        # Breakout search over [0, bound] finds cycle 0 at est 2. Deepest retrace in
        # [3, bound] = candle 7 (l .6088); cond1 needs l <= inner .5550 + 4 pips = .5554 ->
        # fails -> `no_retrace`, finalized. Phase 1's decision read candles through the
        # bound -> finalize = probe_end_idx.
        df = _prepare_df(_make_multicycle_data())
        res = unified_probe(df, 0, 1, _ref_zone_uptrend(), probe_end_idx=probe_end_idx, timeframe="H1")
        assert res.status == "finalized"
        assert res.finalize_condition == "no_retrace"
        assert res.cts0_established_idx == 2
        assert res.iterations == 1
        assert res.finalize_idx == probe_end_idx
        assert res.starting_idx == 0

    @pytest.mark.parametrize("probe_end_idx", [0, 1])
    def test_phase1_end_idx_reached_is_probe_end_idx(self, probe_end_idx):
        # Cycle 0 is established at 2: a bound of 0 or 1 leaves NO true first breakout in the
        # window (`find_true_first_breakout` -> None) and the bound is set -> `end_idx_reached`,
        # finalize = probe_end_idx, no CTS_0.
        df = _prepare_df(_make_multicycle_data())
        res = unified_probe(df, 0, 1, _ref_zone_uptrend(), probe_end_idx=probe_end_idx, timeframe="H1")
        assert res.status == "finalized"
        assert res.finalize_condition == "end_idx_reached"
        assert res.cts0_established_idx is None
        assert res.finalize_idx == probe_end_idx
        assert res.starting_idx == 0

    def test_phase1_end_idx_reached_when_no_breakout_exists_at_all(self):
        # A +1 probe on an all-bearish series: no bullish breakout pattern exists anywhere,
        # so the bound (20) is reached without a CTS_0 -> finalize = probe_end_idx = 20.
        df = _prepare_df(_make_downtrend_data(n=60))
        res = unified_probe(df, 0, 1, _ref_zone_uptrend(), probe_end_idx=20, timeframe="H1")
        assert res.status == "finalized"
        assert res.finalize_condition == "end_idx_reached"
        assert res.cts0_established_idx is None
        assert res.finalize_idx == 20

    # --- Phase 2 (enable_phase2=True: the MS-based path) ----------------------------

    @pytest.mark.parametrize("probe_end_idx", [10, 22])
    def test_phase2_second_cts_reached_is_the_second_cts_moment(self, monkeypatch, probe_end_idx):
        # Bound >= 10 puts cycle 1's CTS_ESTABLISHED (idx 10, confirmed_at 10) inside the run
        # (10 = the 2nd CTS on the last in-bound step, no early stop; 22 = Plan B early stop);
        # cycle 0 confirmed at 7 with anchor 5 -> retrace window [3, 4], deepest = candle 3
        # (l .6058) fails cond1 -> no reset; n_cts >= 2 -> `second_cts_reached`,
        # finalize = the 2nd CTS's moment = 10. NOTE: on this fixture idx == confirmed_at
        # (10 == 10); the idx != moment case is the next test.
        retained = _retain_phase2_ms(monkeypatch)
        df = _prepare_df(_make_multicycle_data())
        res = unified_probe(df, 0, 1, _ref_zone_uptrend(), probe_end_idx, "H1", enable_phase2=True)
        assert res.status == "finalized"
        assert res.finalize_condition == "second_cts_reached"
        cts_est = [e for e in retained[-1].events if e.type == "CTS_ESTABLISHED"]
        assert [(int(e.idx), int(e.meta["confirmed_at"])) for e in cts_est[:2]] == [(2, 2), (10, 10)]
        assert res.finalize_idx == int(cts_est[1].meta["confirmed_at"]) == 10
        assert res.starting_idx == 0

    def test_phase2_second_cts_reached_is_the_moment_not_the_anchor(self, monkeypatch):
        # `_make_second_cts_moment_after_anchor_data`: the 2nd CTS has anchor 9 (its extreme,
        # h9 .6147) but confirmed_at 10 (the OMO apply candle). finalize = 10, NOT 9.
        # The reversal at 14 does not make this `reversal_in_probe`: n_cts = 2 at the exit.
        retained = _retain_phase2_ms(monkeypatch)
        df = _prepare_df(_make_second_cts_moment_after_anchor_data())
        res = unified_probe(df, 0, 1, _ref_zone_uptrend(), 17, "H1", enable_phase2=True)
        assert res.status == "finalized"
        assert res.finalize_condition == "second_cts_reached"
        cts_est = [e for e in retained[-1].events if e.type == "CTS_ESTABLISHED"]
        assert [(ef.cts_anchor_idx(e), int(e.meta["confirmed_at"])) for e in cts_est] == [(2, 2), (9, 10)]
        assert ef.cts_anchor_idx(cts_est[1]) != int(cts_est[1].meta["confirmed_at"])   # the fixture's point
        # Plan E E1 value pin: the pattern-realm anchor is the OMO's FIRST candle 9 (c0) —
        # not its end / apply candle 10 — and the reversal close-break candle is 11.
        assert cts_est[1].meta["pattern_anchor_idx"] == 9
        rv = [(e.type, e.meta["pattern_anchor_idx"]) for e in retained[-1].events
              if e.type in ("REVERSAL_WATCH_START", "REVERSAL_CANDIDATE")]
        assert rv == [("REVERSAL_WATCH_START", 11), ("REVERSAL_CANDIDATE", 11)]
        assert res.finalize_idx == int(cts_est[1].meta["confirmed_at"]) == 10
        assert res.finalize_idx != ef.cts_anchor_idx(cts_est[1])
        assert res.starting_idx == 0

    @pytest.mark.parametrize("probe_end_idx", [None, 12])
    def test_phase2_reversal_in_probe_is_the_reversal_apply_idx(self, monkeypatch, probe_end_idx):
        # `_make_cycle0_reversal_data`: candle 9 closes .5990 < BOS_0 .5998 -> watch + the -1
        # pattern anchored at 9 applies at 10 -> reversal applied at 10 with only ONE
        # CTS_ESTABLISHED (cycle 0 at 2) -> `reversal_in_probe`, finalize = the reversal
        # apply idx 10 (with `probe_end_idx=12` = the data edge, 10 != the bound; with None the
        # live-mode path gives the same answer).
        retained = _retain_phase2_ms(monkeypatch)
        df = _prepare_df(_make_cycle0_reversal_data())
        res = unified_probe(df, 0, 1, _ref_zone_uptrend(), probe_end_idx, "H1", enable_phase2=True)
        assert res.status == "finalized"
        assert res.finalize_condition == "reversal_in_probe"
        ms = retained[-1]
        assert sum(1 for e in ms.events if e.type == "CTS_ESTABLISHED") == 1
        cands = [(int(e.idx), int(e.meta["apply_idx"])) for e in ms.events if e.type == "REVERSAL_CANDIDATE"]
        assert cands == [(9, 10)]
        rev_rows = ms.df.index[ms.df["market_state"].astype(str).str.lower() == "reversal"]
        assert int(rev_rows.min()) == 10
        assert res.finalize_idx == int(rev_rows.min()) == 10
        assert res.cts0_established_idx == 2
        assert res.starting_idx == 0

    @pytest.mark.parametrize("probe_end_idx", [8, 9])
    def test_phase2_no_retrace_with_cycle0_confirmed_is_cts0_confirmed_idx(self, monkeypatch, probe_end_idx):
        # Bound 8/9: cycle 0 CONFIRMED at 7 (anchor 5) is in-window, cycle 1 (est 10) is not.
        # Retrace window [3, anchor-1 = 4] -> candle 3 (l .6058) fails cond1 -> `no_retrace`;
        # the decision needed cycle 0's confirmation -> finalize = CTS_0_CONFIRMED.idx = 7
        # (NOT the bound: 7 != 8 / 9).
        retained = _retain_phase2_ms(monkeypatch)
        df = _prepare_df(_make_multicycle_data())
        res = unified_probe(df, 0, 1, _ref_zone_uptrend(), probe_end_idx, "H1", enable_phase2=True)
        assert res.status == "finalized"
        assert res.finalize_condition == "no_retrace"
        conf = [e for e in retained[-1].events if e.type == "CTS_CONFIRMED" and e.meta.get("cycle_id") == 0]
        assert [(int(e.idx), int(e.meta["cts_anchor_idx"])) for e in conf] == [(7, 5)]
        assert sum(1 for e in retained[-1].events if e.type == "CTS_ESTABLISHED") == 1
        assert res.finalize_idx == int(conf[0].idx) == 7
        assert res.finalize_idx != probe_end_idx
        assert res.starting_idx == 0

    @pytest.mark.parametrize("probe_end_idx", [3, 6])
    def test_phase2_no_retrace_without_cycle0_confirmed_is_probe_end_idx(self, monkeypatch, probe_end_idx):
        # Bound < 7: cycle 0 is established (2) but NOT confirmed inside the run (CTS_CONFIRMED
        # would be at 7). Retrace window falls back to [3, bound]; its deepest candle is a
        # bull candle whose low (>= .6058) fails cond1 -> `no_retrace` (else branch),
        # finalize = probe_end_idx.
        retained = _retain_phase2_ms(monkeypatch)
        df = _prepare_df(_make_multicycle_data())
        res = unified_probe(df, 0, 1, _ref_zone_uptrend(), probe_end_idx, "H1", enable_phase2=True)
        assert res.status == "finalized"
        assert res.finalize_condition == "no_retrace"
        types = [e.type for e in retained[-1].events]
        assert types.count("CTS_ESTABLISHED") == 1
        assert "CTS_CONFIRMED" not in types
        assert res.cts0_established_idx == 2
        assert res.finalize_idx == probe_end_idx
        assert res.starting_idx == 0

    @pytest.mark.parametrize("probe_end_idx", [0, 1])
    def test_phase2_end_idx_reached_is_probe_end_idx(self, monkeypatch, probe_end_idx):
        # Bound 0/1 (< the cycle-0 establishment at 2): Phase 1 -> `end_idx_reached`; Phase 2's
        # MS run emits no CTS_ESTABLISHED and the bound is set -> `end_idx_reached`,
        # finalize = probe_end_idx.
        retained = _retain_phase2_ms(monkeypatch)
        df = _prepare_df(_make_multicycle_data())
        res = unified_probe(df, 0, 1, _ref_zone_uptrend(), probe_end_idx, "H1", enable_phase2=True)
        assert len(retained) == 1, "Phase 2 must actually run an MS"
        assert not any(e.type == "CTS_ESTABLISHED" for e in retained[-1].events)
        assert res.status == "finalized"
        assert res.finalize_condition == "end_idx_reached"
        assert res.cts0_established_idx is None
        assert res.finalize_idx == probe_end_idx
        assert res.starting_idx == 0

    # --- pending rows -> None ---------------------------------------------------------

    def test_phase1_no_cts_pending_is_none(self):
        # +1 probe on an all-bearish series with NO bound: no bullish breakout anywhere ->
        # `no_cts_pending`, pending, finalize None (a sid never builds from a pending probe).
        df = _prepare_df(_make_downtrend_data(n=60))
        res = unified_probe(df, 0, 1, _ref_zone_uptrend(), probe_end_idx=None, timeframe="H1")
        assert res.status == "pending"
        assert res.finalize_condition == "no_cts_pending"
        assert res.cts0_established_idx is None
        assert res.finalize_idx is None

    def test_phase1_one_cts_pending_is_none(self):
        # Cycle 0 found at 2 but no bound -> the retrace window cannot be bounded ->
        # `one_cts_pending`, pending, finalize None.
        df = _prepare_df(_make_multicycle_data())
        res = unified_probe(df, 0, 1, _ref_zone_uptrend(), probe_end_idx=None, timeframe="H1")
        assert res.status == "pending"
        assert res.finalize_condition == "one_cts_pending"
        assert res.cts0_established_idx == 2
        assert res.finalize_idx is None

    def test_phase1_max_iterations_is_none(self):
        # Reference inner .6090 (a buy zone just under candle 7's low): iter 1's BOS_0 gate is
        # the inner, so the first close past .6090 is candle 4 (.6100) -> breakout anchored at
        # 4, est 5. Deepest retrace in [6, 9] = candle 7 (l .6088): cond1 .6088 <= .6094 ok,
        # cond2 lower wick = body_bottom .6090 - .6088 = 2 pips <= 16 ok -> RESET to 7.
        # `max_iterations=1` exhausts the loop on that reset -> `max_iterations`, pending,
        # finalize None; `starting_idx` = the reset start 7.
        df = _prepare_df(_make_multicycle_data())
        ref = ReferenceZone(outer=0.6050, inner=0.6090, side="buy",
                            source="cts_established", anchor_idx=0)
        res = unified_probe(df, 0, 1, ref, probe_end_idx=9, timeframe="H1", max_iterations=1)
        assert res.status == "pending"
        assert res.finalize_condition == "max_iterations"
        assert res.iterations == 1
        assert res.cts0_established_idx == 5
        assert res.starting_idx == 7
        assert res.finalize_idx is None

    def test_phase2_no_cts_pending_is_none(self, monkeypatch):
        # As the Phase-1 row, with Phase 2 on: its MS run finds no CTS either and the bound is
        # None -> `no_cts_pending`, finalize None.
        retained = _retain_phase2_ms(monkeypatch)
        df = _prepare_df(_make_downtrend_data(n=60))
        res = unified_probe(df, 0, 1, _ref_zone_uptrend(), None, "H1", enable_phase2=True)
        assert len(retained) == 1
        assert not any(e.type == "CTS_ESTABLISHED" for e in retained[-1].events)
        assert res.status == "pending"
        assert res.finalize_condition == "no_cts_pending"
        assert res.finalize_idx is None

    def test_phase2_one_cts_pending_is_none(self, monkeypatch):
        # `_make_uptrend_data(n=80)`: an unbroken climb — cycle 0 established at 2, never
        # CONFIRMED (no pullback), no reversal, no 2nd CTS, bound None -> the retrace window
        # cannot be bounded -> `one_cts_pending`, finalize None.
        retained = _retain_phase2_ms(monkeypatch)
        df = _prepare_df(_make_uptrend_data(n=80))
        res = unified_probe(df, 0, 1, _ref_zone_uptrend(), None, "H1", enable_phase2=True)
        types = [e.type for e in retained[-1].events]
        assert types.count("CTS_ESTABLISHED") == 1
        assert "CTS_CONFIRMED" not in types
        assert res.status == "pending"
        assert res.finalize_condition == "one_cts_pending"
        assert res.cts0_established_idx == 2
        assert res.finalize_idx is None


def test_phase2_retrace_window_opens_after_the_first_cts_moment(monkeypatch):
    """Plan E E3c: Phase 2's retrace window starts at the first CTS_ESTABLISHED's
    MOMENT + 1 (like Phase 1's `tfb.est_idx + 1`), not its anchor + 1. Stubbed MS
    run: CTS_0 anchor 9, moment 12, no confirmation, probe_end 20 → the candidate
    search is asked over [13, 20]."""
    import engine_v2.structure.unified_probe as up
    from types import SimpleNamespace
    n = 25
    df = pd.DataFrame({"o": [0.6] * n, "h": [0.601] * n, "l": [0.599] * n, "c": [0.6] * n,
                       "market_state": ["breakout"] * n, "structure_id": [0] * n})
    est = make_cts_established(cts_anchor_idx=9, confirmed_at=12, price=0.61,
                               structure_id=0, cycle_id=0, struct_direction=1)
    stub = SimpleNamespace(debug=False, early_stop_idx=None,
                           run=lambda: (df.copy(), [est], None))
    monkeypatch.setattr(up, "_make_market_structure", lambda *a, **k: stub)
    seen = []
    monkeypatch.setattr(up, "_select_extreme_retrace_candidate",
                        lambda d, lo, hi, direction: seen.append((lo, hi)) or None)
    ref = ReferenceZone(outer=0.5980, inner=0.5990, side="buy",
                        source="cts_confirmed", anchor_idx=0)
    res = up._run_phase2(df, 0, 0.5990, 0.5980, 1, ref, 20, 0.0, 0.0, max_iterations=2)
    assert seen == [(13, 20)]
    assert res.finalize_condition == "no_retrace"
