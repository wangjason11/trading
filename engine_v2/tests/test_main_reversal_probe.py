"""Unit tests for the main (H1) reversal handoff via unified_probe (Step 4 of
the true-first-breakout cycle-0 redesign).

`compute_structure` previously selected a post-reversal structure's start with
Scenario 2 + Exception 1 + Exception 2. Step 4 replaces that with the SAME
`unified_probe` + scan-from-start path the subordinate reversals already use:
the prior sid's most recent CTS is the reference, the probe runs in the flipped
direction over [prior-CTS-extreme, reversal apply idx] and hands back a decision
(start + BOS_0 inner), and the reversed structure's cycle-0 CTS_0 is then
established by a fresh scan-from-start MS run gated on that BOS_0 inner.

These tests pin:
  1. The reversal handoff fires (>=2 structures; sid=1 establishes cycle 0).
  2. The reversal apply idx is a single candle (.min() == .max()) — the
     unification assumption behind retiring the reversal_start/confirmed pair.
  3. Agreement: sid=1's cycle-0 CTS_0 establishment matches the start +
     bos0_inner decision an independently-reconstructed probe produces, fed
     through the shared `find_true_first_breakout` routine — i.e. compute_structure
     genuinely routes the reversal through the probe + scan path.

Fixtures mirror test_bounded_structure.py / test_scenario3.py (self-contained).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine_v2.features.candle_classifier import apply_candle_classification
from engine_v2.patterns.pattern_engine import detect_patterns
from engine_v2.patterns.imbalance import compute_imbalance
from engine_v2.patterns.structure_patterns import BreakoutPatterns
from engine_v2.structure.identify_start import identify_start_scenario_1
from engine_v2.structure.reference_zone import (
    build_ad_hoc_bos0_reference_zone,
    build_reference_zone_from_cts_event,
)
from engine_v2.structure.structure_engine import (
    compute_bounded_structure,
    compute_structure,
)
from engine_v2.structure.true_first_breakout import find_true_first_breakout
from engine_v2.structure.unified_probe import unified_probe
from engine_v2.zones.kl_zones_v1 import derive_kl_zones_v1


# ---------------------------------------------------------------------------
# Helpers (mirror test_bounded_structure.py)
# ---------------------------------------------------------------------------

def _make_raw_df(ohlc_rows: list[dict], pair: str = "NZD_USD") -> pd.DataFrame:
    base_time = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    records = []
    for i, row in enumerate(ohlc_rows):
        records.append({
            "time": base_time + pd.Timedelta(hours=i),
            "o": row["o"], "h": row["h"], "l": row["l"], "c": row["c"],
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


def _seg(rows, price, n, direction, step, noise, rng):
    for _ in range(n):
        body = step + rng.uniform(0, noise)
        o = price
        c = price + direction * body
        if direction > 0:
            h = c + rng.uniform(0.0001, 0.0004)
            l = o - rng.uniform(0.0001, 0.0004)
        else:
            h = o + rng.uniform(0.0001, 0.0004)
            l = c - rng.uniform(0.0001, 0.0004)
        rows.append({"o": round(o, 5), "h": round(h, 5),
                     "l": round(l, 5), "c": round(c, 5)})
        price = c
    return price


def _make_reversing_data() -> list[dict]:
    """Up impulses with pullbacks (CTS confirms -> BOS forms), a strong down leg
    that close-breaks the BOS (sid=0, a downtrend Scenario 1 anchors on), then a
    trailing up leg that close-breaks the down structure's BOS and drives the
    reversal into sid=1. Empirically: compute_structure yields sid=0 (cts0@53)
    reversing into sid=1 (cts0@62) on this series."""
    rng = np.random.RandomState(7)
    rows: list[dict] = []
    price = 0.6000
    for _ in range(3):
        price = _seg(rows, price, 6, +1, 0.0020, 0.0004, rng)   # impulse up
        price = _seg(rows, price, 3, -1, 0.0008, 0.0003, rng)   # shallow pullback
    price = _seg(rows, price, 6, +1, 0.0020, 0.0004, rng)       # impulse up -> BOS
    price = _seg(rows, price, 20, -1, 0.0022, 0.0004, rng)      # down leg (sid=0)
    price = _seg(rows, price, 8, -1, 0.0006, 0.0003, rng)       # tail
    price = _seg(rows, price, 15, +1, 0.0024, 0.0004, rng)      # trailing up -> reversal into sid=1
    return rows


def _cts0_established_idx(events, sid: int):
    """Cycle-0 CTS_ESTABLISHED idx for a structure_id, or None."""
    for ev in events:
        if (
            ev.type == "CTS_ESTABLISHED"
            and ev.meta.get("structure_id") == sid
            and ev.meta.get("cycle_id") == 0
        ):
            return int(ev.idx)
    return None


def _max_structure_id(events) -> int:
    ids = [int(ev.meta.get("structure_id", 0)) for ev in events
           if ev.meta.get("structure_id") is not None]
    return max(ids) if ids else 0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMainReversalProbe:
    def test_reversal_handoff_produces_second_structure(self):
        """A reversing series yields >=2 main structures and sid=1 establishes
        its cycle-0 CTS via the new reversal handoff."""
        df = _prepare_df(_make_reversing_data())
        res = compute_structure(df)
        assert _max_structure_id(res.events) >= 1, "reversal handoff did not roll into sid>=1"
        assert _cts0_established_idx(res.events, 1) is not None, \
            "sid=1 did not establish cycle-0 CTS"

    def test_reversal_apply_idx_is_single_candle(self):
        """The sid=0 reversal mask is exactly one candle (.min() == .max()) —
        the invariant behind retiring the reversal_start/confirmed pair. If this
        ever broke, compute_structure WARNs rather than crashing, but on real
        structure it should hold."""
        df = _prepare_df(_make_reversing_data())
        d0 = identify_start_scenario_1(df, input_idx=int(df.index.max()),
                                       lookback_days=183, min_history=50)
        ref0 = build_ad_hoc_bos0_reference_zone(df, int(d0.start_idx),
                                                int(d0.struct_direction))
        bounded0 = compute_bounded_structure(
            df, int(d0.start_idx), int(d0.struct_direction),
            enforce_cts0_new_extreme=ref0 is not None,
            bos0_inner=ref0.inner if ref0 is not None else None,
        )
        rev_mask = (
            (bounded0.df["market_state"].astype(str).str.lower() == "reversal")
            & (bounded0.df["structure_id"].astype(int) == 0)
        )
        assert rev_mask.any(), "fixture did not reverse sid=0"
        idxs = bounded0.df.loc[rev_mask].index
        assert int(idxs.min()) == int(idxs.max()), \
            "reversal mask spans >1 candle — unification assumption violated"

    def test_sid1_cts0_agrees_with_reconstructed_probe(self):
        """sid=1's cycle-0 CTS_0 establishment matches the start + bos0_inner an
        independently-reconstructed reversal probe produces, fed through the
        shared find_true_first_breakout routine. This proves compute_structure
        routes the reversal through unified_probe + scan-from-start (not the old
        Scenario2/Exc1/Exc2 path)."""
        df = _prepare_df(_make_reversing_data())

        # Reconstruct exactly what compute_structure's loop does after sid=0.
        d0 = identify_start_scenario_1(df, input_idx=int(df.index.max()),
                                       lookback_days=183, min_history=50)
        sd = int(d0.struct_direction)
        s0 = int(d0.start_idx)
        ref0 = build_ad_hoc_bos0_reference_zone(df, s0, sd)
        bounded0 = compute_bounded_structure(
            df, s0, sd,
            enforce_cts0_new_extreme=ref0 is not None,
            bos0_inner=ref0.inner if ref0 is not None else None,
        )
        assert bounded0.reversal_idx is not None, "fixture did not reverse sid=0"
        rev_apply = int(bounded0.reversal_idx)

        probe_sd = -sd
        kl0 = derive_kl_zones_v1(bounded0.df, bounded0.events, struct_direction=sd)
        ref_zone = build_reference_zone_from_cts_event(
            bounded0.events, kl0, bounded0.df, sid=0,
            probe_direction=probe_sd, idx_window=None,
        )
        assert ref_zone is not None, "no prior CTS reference for the reversal probe"
        probe = unified_probe(
            bounded0.df.copy(),
            input_idx=int(ref_zone.anchor_idx),
            direction=probe_sd,
            reference_zone=ref_zone,
            probe_end_idx=rev_apply,      # Plan C §7: the probe's search bound
            timeframe="H1",
            enable_phase2=False,
        )
        assert probe.status == "finalized"

        # MS scan-from-start re-finds CTS_0 over [start, end-of-df] gated on the
        # probe's bos0_inner (mirrors MarketStructure._get_cts0_tfb).
        bp = BreakoutPatterns(df)
        tfb = find_true_first_breakout(
            bp, int(probe.starting_idx), int(len(df) - 1), probe_sd, probe.bos0_inner,
        )
        assert tfb is not None, "reconstructed probe found no true first breakout"
        expected_est = int(tfb.est_idx)

        # Actual: compute_structure end-to-end.
        res = compute_structure(df)
        actual_est = _cts0_established_idx(res.events, 1)
        assert actual_est == expected_est, (
            f"sid=1 CTS_0 est idx {actual_est} != reconstructed probe est "
            f"{expected_est}"
        )
