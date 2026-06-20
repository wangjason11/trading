"""Unit tests for KL-zone inner/outer geometry — the zone-inversion invariant.

A zone "inverts" when its `inner` lands beyond its `outer` (on the far side of
the base candle's extreme). `outer` is always a window extreme (`base_low` /
`base_high`); a valid zone has `inner` on the interior side, i.e.
`base_low <= inner <= base_high`.

These tests pin two facts established in the (c) review (2026-06-20):

1. **2-candle and star families are inversion-safe BY CONSTRUCTION.** Their inner
   is always an `o`/`c`/`mid_price` of a candle WITHIN the multi-candle window,
   and `compute_base_window_features` sets `base_low/base_high` = min/max over
   that whole window — so the inner is necessarily within `[base_low, base_high]`
   = within the outer. No bound is needed; this test guards that property so a
   future refactor can't silently break it.

2. **pinbar keeps its tight closest-neighbour rule + an inner-side bound ((c),
   2026-06-20).** The pinbar inner is still the i±1 neighbour O/C closest to the
   reference extreme (tight zone), but candidates BEYOND the outer are now
   dropped first, so a neighbour body past the base extreme can no longer become
   the inner (no inversion). It is intentionally NOT the base/inside-bar
   inner-edge rule, which over-widens pinbar zones. The bound is inert in the
   normal case (closest neighbour already within the outer) -> byte-identical.
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine_v2.zones.kl_zones_v1 import (
    compute_base_window_features,
    find_pinbar_threshold,
    zone_thresholds,
)


def _df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal OHLC df with mid_price = (h+l)/2 (engine convention)."""
    df = pd.DataFrame(rows)
    df["mid_price"] = (df["h"] + df["l"]) / 2.0
    df.attrs["pair"] = "NZD_USD"
    return df


# A varied 8-candle window — mixed bullish/bearish bodies, no two identical, so
# the within-range assertions are non-trivial. base_idx=3 leaves room for the
# 3-candle star window (3,4,5) and ±5 neighbours.
_ROWS = [
    {"o": 1.0000, "h": 1.0030, "l": 0.9990, "c": 1.0020},  # 0 bull
    {"o": 1.0020, "h": 1.0025, "l": 0.9980, "c": 0.9995},  # 1 bear
    {"o": 0.9995, "h": 1.0040, "l": 0.9990, "c": 1.0035},  # 2 bull
    {"o": 1.0035, "h": 1.0060, "l": 1.0010, "c": 1.0015},  # 3 bear (base_idx)
    {"o": 1.0015, "h": 1.0050, "l": 1.0005, "c": 1.0045},  # 4 bull
    {"o": 1.0045, "h": 1.0048, "l": 1.0000, "c": 1.0008},  # 5 bear
    {"o": 1.0008, "h": 1.0033, "l": 1.0002, "c": 1.0030},  # 6 bull
    {"o": 1.0030, "h": 1.0032, "l": 0.9998, "c": 1.0004},  # 7 bear
]

_TWO_CANDLE = [
    "no base",
    "no base 1st big",
    "no base 2nd big",
    "no base long tails up",
    "no base long tails down",
]
_STAR = [
    "no base star",
    "no base star 1st big",
    "no base star 2nd big",
]


class TestInnerWithinOuterByConstruction:
    """2-candle + star: inner ∈ [base_low, base_high] for every sd × bos."""

    @pytest.mark.parametrize("pattern", _TWO_CANDLE + _STAR)
    @pytest.mark.parametrize("sd", [1, -1])
    @pytest.mark.parametrize("bos", [True, False])
    def test_inner_within_window_extremes(self, pattern, sd, bos):
        df = _df(_ROWS)
        base_idx = 3
        feats = compute_base_window_features(df, base_idx, pattern)
        base_low, base_high = feats["base_low"], feats["base_high"]

        outer, inner = zone_thresholds(df, base_idx, sd, pattern, bos=bos)

        # Outer is one of the window extremes.
        assert outer in (pytest.approx(base_low), pytest.approx(base_high))
        # Inner never sits beyond the window range => never beyond the outer.
        assert base_low - 1e-12 <= inner <= base_high + 1e-12, (
            f"{pattern} sd={sd} bos={bos}: inner {inner} outside "
            f"[{base_low}, {base_high}]"
        )
        # Explicit non-inversion: inner on the interior side of the outer.
        if outer == pytest.approx(base_high):
            assert inner <= outer + 1e-12  # outer is the top -> inner below it
        else:
            assert inner >= outer - 1e-12  # outer is the bottom -> inner above it


class TestPinbarInnerSideBound:
    """pinbar keeps the tight closest-neighbour rule (bound inert in the normal
    case) but drops candidates beyond the outer so the zone can't invert."""

    def test_normal_case_picks_closest_inbound_neighbour(self):
        # All i±1 neighbour O/C are within the outer -> the bound is inert and
        # the inner is exactly the closest neighbour O/C (original behaviour).
        df = _df(_ROWS)
        base_idx = 3
        for sd in (1, -1):
            for bos in (True, False):
                use_low_ref = (bos and sd == 1) or ((not bos) and sd == -1)
                ref = float(df.loc[base_idx, "l" if use_low_ref else "h"])
                cands = [
                    float(df.loc[base_idx - 1, "o"]), float(df.loc[base_idx - 1, "c"]),
                    float(df.loc[base_idx + 1, "o"]), float(df.loc[base_idx + 1, "c"]),
                ]
                inbound = [x for x in cands if (x >= ref if use_low_ref else x <= ref)]
                expected = min(inbound, key=lambda x: abs(x - ref))
                got = find_pinbar_threshold(df, base_idx, bos=bos, struct_direction=sd)
                assert got == pytest.approx(expected), f"sd={sd} bos={bos}"
                # And it's within the outer (no inversion).
                assert (got >= ref - 1e-12) if use_low_ref else (got <= ref + 1e-12)

    def test_pinbar_inner_bounded_despite_neighbour_beyond_extreme(self):
        # Pinbar at idx 5 with a long LOWER tail; BOS sd=+1 => outer = its low.
        # Neighbour idx 6 has c=0.99900, BELOW the pinbar low (0.99950): the old
        # algorithm picked it (closest O/C to the low ref) -> inner < outer
        # (INVERTED). The inner-edge rule uses idx6's body TOP (0.99980) instead.
        rows = [
            {"o": 1.00200, "h": 1.00250, "l": 1.00100, "c": 1.00150},  # 0
            {"o": 1.00150, "h": 1.00180, "l": 1.00050, "c": 1.00080},  # 1
            {"o": 1.00080, "h": 1.00120, "l": 1.00020, "c": 1.00100},  # 2
            {"o": 1.00100, "h": 1.00130, "l": 1.00030, "c": 1.00060},  # 3
            {"o": 1.00050, "h": 1.00070, "l": 0.99960, "c": 1.00010},  # 4 body-top 1.00050
            {"o": 1.00060, "h": 1.00090, "l": 0.99950, "c": 1.00065},  # 5 PINBAR (low 0.99950)
            {"o": 0.99980, "h": 1.00000, "l": 0.99900, "c": 0.99900},  # 6 c below pinbar low
            {"o": 1.00040, "h": 1.00055, "l": 0.99970, "c": 1.00030},  # 7 body-top 1.00040
        ]
        df = _df(rows)
        outer = float(df.loc[5, "l"])  # 0.99950, BOS sd=+1
        inner = find_pinbar_threshold(df, 5, bos=True, struct_direction=1)
        assert inner >= outer - 1e-12, (
            f"pinbar inner {inner} inverted below outer {outer}"
        )
        # And specifically not the old buggy far-side neighbour close (0.99900).
        assert inner != pytest.approx(0.99900)
