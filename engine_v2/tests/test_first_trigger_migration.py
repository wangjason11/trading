"""Unit + light-integration tests for the Session 2 first_* trigger migration
(post-pivot 2026-05-29 — first_confluence uses ad-hoc BOS_0 on M15;
first_counter uses sibling first_confluence's most recent CTS).

Covers:
  - Refactored `map_candle_to_lower_tf` (dropped `mapping_sd` → `parent_extreme_dir`).
  - `_resolve_first_via_unified_probe` wiring (happy path + each defensive
    None-return branch) under the new ref-zone design.
  - `_resolve_trigger_m15_start` dispatcher routes by use_case and threads
    `sibling_entity_df` through.

The unified-probe primitive itself is exercised in test_unified_probe.py;
the legacy probe path (subsequent_*) is covered by test_sub_chain.py.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pandas as pd
import pytest

from engine_v2.common.types import KLZone
from engine_v2.multitf.data_bridge import map_candle_to_lower_tf
from engine_v2.multitf.entity_df_mutation import (
    _resolve_first_via_unified_probe,
    _resolve_trigger_m15_start,
)
from engine_v2.multitf.types import MultiTFTrigger
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.structure.reference_zone import ReferenceZone
from engine_v2.structure.unified_probe import ProbeResult


# ---------------------------------------------------------------------------
# Fixtures — synthetic H1 + derived M15 (4 M15 per H1).
# ---------------------------------------------------------------------------

def _h1_df_uptrend(n_hours: int = 20, base_price: float = 0.6000) -> pd.DataFrame:
    base_time = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    rows = []
    price = base_price
    for i in range(n_hours):
        o = price
        c = price + 0.0030
        h = c + 0.0005
        l = o - 0.0005
        rows.append({
            "time": base_time + pd.Timedelta(hours=i),
            "o": round(o, 5), "h": round(h, 5),
            "l": round(l, 5), "c": round(c, 5),
            "volume": 100,
        })
        price = c
    df = pd.DataFrame(rows)
    df.attrs["pair"] = "NZD_USD"
    return df


def _m15_from_h1(h1: pd.DataFrame) -> pd.DataFrame:
    """4 M15 per H1 hour. M15 #0 carries the H1 high, M15 #1 carries the H1 low."""
    rows = []
    for _, hr in h1.iterrows():
        ht = hr["time"]
        ho, hh, hl, hc = float(hr["o"]), float(hr["h"]), float(hr["l"]), float(hr["c"])
        rows.append({
            "time": ht + pd.Timedelta(minutes=0),
            "o": ho, "h": hh, "l": round(ho - 0.0002, 5),
            "c": round(ho + 0.0010, 5), "volume": 25,
        })
        rows.append({
            "time": ht + pd.Timedelta(minutes=15),
            "o": round(ho + 0.0010, 5), "h": round(ho + 0.0015, 5),
            "l": hl, "c": round(ho + 0.0008, 5), "volume": 25,
        })
        rows.append({
            "time": ht + pd.Timedelta(minutes=30),
            "o": round(ho + 0.0008, 5), "h": round(ho + 0.0018, 5),
            "l": round(ho + 0.0005, 5), "c": round(ho + 0.0020, 5),
            "volume": 25,
        })
        rows.append({
            "time": ht + pd.Timedelta(minutes=45),
            "o": round(ho + 0.0020, 5), "h": round(ho + 0.0028, 5),
            "l": round(ho + 0.0018, 5), "c": hc, "volume": 25,
        })
    df = pd.DataFrame(rows)
    df.attrs["pair"] = "NZD_USD"
    return df


# ---------------------------------------------------------------------------
# map_candle_to_lower_tf — post-refactor signature
# ---------------------------------------------------------------------------

class TestMapCandleToLowerTf:
    def test_extreme_dir_plus_one_picks_max_high(self):
        h1 = _h1_df_uptrend(n_hours=3)
        m15 = _m15_from_h1(h1)
        result = map_candle_to_lower_tf(
            h1.loc[1, "time"], parent_extreme_dir=1, m15_df=m15,
        )
        assert result == 4  # H1 hour 1 → M15 #0

    def test_extreme_dir_minus_one_picks_min_low(self):
        h1 = _h1_df_uptrend(n_hours=3)
        m15 = _m15_from_h1(h1)
        result = map_candle_to_lower_tf(
            h1.loc[1, "time"], parent_extreme_dir=-1, m15_df=m15,
        )
        assert result == 5  # H1 hour 1 → M15 #1

    def test_empty_hour_returns_none(self):
        m15 = _m15_from_h1(_h1_df_uptrend(n_hours=3))
        far_future = pd.Timestamp("2030-01-01 00:00", tz="UTC")
        assert map_candle_to_lower_tf(far_future, 1, m15) is None

    def test_signature_no_h1_extreme_price_arg(self):
        m15 = _m15_from_h1(_h1_df_uptrend(n_hours=2))
        with pytest.raises(TypeError):
            map_candle_to_lower_tf(  # type: ignore[call-arg]
                m15.loc[0, "time"], 0.6005, 1, m15,
            )


# ---------------------------------------------------------------------------
# Helpers for trigger + sibling-events fixtures
# ---------------------------------------------------------------------------

def _make_trigger(
    use_case: str, parent_sid: int = 0, parent_cycle_id: int = 0,
    parent_sd: int = 1, lower_sd: int = -1,
    probe_input_idx: int = 1, probe_end_idx: int = 5,
) -> MultiTFTrigger:
    return MultiTFTrigger(
        parent_tf="H1",
        parent_sid=parent_sid,
        parent_cycle_id=parent_cycle_id,
        parent_sd=parent_sd,
        use_case=use_case,
        lower_tf="M15",
        lower_sd=lower_sd,
        start_time=pd.Timestamp("2024-01-01 01:00", tz="UTC"),
        start_price=0.6005,
        lifecycle_end_idx=None,
        meta={
            "probe_input_idx": probe_input_idx,
            "probe_end_idx": probe_end_idx,
            "trigger_event_idx": probe_end_idx,
        },
    )


def _fake_probe_result(start_idx: int = 4) -> ProbeResult:
    return ProbeResult(
        start_idx=start_idx, status="finalized", iterations=1,
        original_ref_zone=ReferenceZone(
            outer=0.6010, inner=0.6020, side="buy",
            source="ad_hoc_bos_0", source_event_idx=0,
        ),
        finalize_condition="no_retrace",
    )


def _make_sibling_cts_event(
    parent_sid: int, parent_cycle_id: int, sub_sid: int,
    idx: int, cts_anchor_idx: int,
    cycle_id: int = 0, struct_direction: int = 1,
) -> StructureEvent:
    """Build a CTS_CONFIRMED StructureEvent that looks like one the
    sibling first_confluence sub would emit on its M15 entity_df."""
    return StructureEvent(
        idx=idx,
        category="STRUCTURE",
        type="CTS_CONFIRMED",
        price=0.6020,
        meta={
            "structure_id": 0,  # local to the sibling's MS run
            "cycle_id": cycle_id,
            "cts_anchor_idx": cts_anchor_idx,
            "struct_direction": struct_direction,
            "parent_sid": parent_sid,
            "parent_cycle_id": parent_cycle_id,
            "sub_sid": sub_sid,
            "confirmed_at": idx,
        },
    )


def _make_kl_zone(
    source_kind: str,
    parent_sid: int, parent_cycle_id: int, sub_sid: int,
    sid: int = 0, cycle_id: int = 0,
    top: float = 0.6020, bottom: float = 0.6010,
) -> KLZone:
    t = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    return KLZone(
        start_time=t, end_time=None, side="sell",
        top=top, bottom=bottom, source_kind=source_kind,
        source_time=t, source_price=top,
        meta={
            "structure_id": sid, "cycle_id": cycle_id,
            "parent_sid": parent_sid,
            "parent_cycle_id": parent_cycle_id,
            "sub_sid": sub_sid,
        },
    )


# ---------------------------------------------------------------------------
# _resolve_first_via_unified_probe — wiring tests under the NEW design
# ---------------------------------------------------------------------------

class TestResolveFirstViaUnifiedProbe:

    def _fixtures(self):
        h1 = _h1_df_uptrend(n_hours=10)
        m15 = _m15_from_h1(h1)
        return h1, m15

    def test_first_confluence_uses_ad_hoc_bos_zone(self):
        h1, m15 = self._fixtures()
        trig = _make_trigger(
            "first_confluence", parent_cycle_id=1,
            parent_sd=1, lower_sd=1,
            probe_input_idx=3, probe_end_idx=7,
        )
        fake = _fake_probe_result(start_idx=15)
        # Patch BOTH the ref-zone derivation (so the synthetic data
        # doesn't need a real base pattern) and the unified_probe call.
        with patch(
            "engine_v2.multitf.entity_df_mutation."
            "_build_first_confluence_ref_zone",
            return_value=ReferenceZone(
                outer=0.6000, inner=0.6020, side="buy",
                source="ad_hoc_bos_0", source_event_idx=13,
            ),
        ) as mock_ref, patch(
            "engine_v2.structure.unified_probe.unified_probe",
            return_value=fake,
        ) as mock_probe:
            m15_start, parent_idx = _resolve_first_via_unified_probe(
                trig, h1, m15, sibling_entity_df=None,
            )
        # Ad-hoc helper called with the price-mapped m15_input_idx
        # (parent_extreme_dir = -lower_sd = -1 → min-low of H1 hour 3 → M15 #1 = idx 13).
        ref_args, _ = mock_ref.call_args
        assert ref_args[1] == 13              # m15_input_idx
        assert ref_args[2] == 1               # probe_direction = lower_sd
        # unified_probe wired with the ad_hoc_bos_0 ref
        _, kwargs = mock_probe.call_args
        assert kwargs["reference_zone"].source == "ad_hoc_bos_0"
        assert kwargs["direction"] == 1
        assert m15_start == 15
        assert parent_idx == 3

    def test_first_counter_uses_sibling_confluence_cts(self):
        h1, m15 = self._fixtures()
        # Build a sibling confluence entity_df with one CTS_CONFIRMED event.
        sib = m15.copy()
        sib.attrs["events"] = [
            _make_sibling_cts_event(
                parent_sid=0, parent_cycle_id=2, sub_sid=0,
                idx=20, cts_anchor_idx=18, cycle_id=0,
                struct_direction=1,
            ),
        ]
        sib.attrs["kl_zones"] = [
            _make_kl_zone(
                "CTS", parent_sid=0, parent_cycle_id=2, sub_sid=0,
                sid=0, cycle_id=0,
            ),
        ]
        trig = _make_trigger(
            "first_counter", parent_cycle_id=2,
            parent_sd=1, lower_sd=-1,
            probe_input_idx=1, probe_end_idx=5,
        )
        fake = _fake_probe_result(start_idx=22)
        with patch(
            "engine_v2.structure.unified_probe.unified_probe",
            return_value=fake,
        ) as mock_probe:
            m15_start, parent_idx = _resolve_first_via_unified_probe(
                trig, h1, m15, sibling_entity_df=sib,
            )
        _, kwargs = mock_probe.call_args
        assert kwargs["direction"] == -1
        # Reference came from the sibling CTS event (CONFIRMED → existing
        # kl_zone path or ad-hoc fallback; both produce a non-None ref).
        ref = kwargs["reference_zone"]
        assert ref.source in (
            "cts_confirmed", "cts_updated", "cts_established",
        )
        assert m15_start == 22
        assert parent_idx == 1

    def test_first_counter_without_sibling_events_returns_none(self):
        h1, m15 = self._fixtures()
        sib = m15.copy()
        sib.attrs["events"] = []
        sib.attrs["kl_zones"] = []
        trig = _make_trigger(
            "first_counter", parent_cycle_id=2,
            parent_sd=1, lower_sd=-1,
            probe_input_idx=1, probe_end_idx=5,
        )
        m15_start, parent_idx = _resolve_first_via_unified_probe(
            trig, h1, m15, sibling_entity_df=sib,
        )
        assert m15_start is None
        assert parent_idx == 1  # still reported for traceability

    def test_first_counter_without_sibling_df_returns_none(self):
        h1, m15 = self._fixtures()
        trig = _make_trigger(
            "first_counter", probe_input_idx=1, probe_end_idx=5,
        )
        m15_start, parent_idx = _resolve_first_via_unified_probe(
            trig, h1, m15, sibling_entity_df=None,
        )
        assert m15_start is None
        assert parent_idx == 1

    def test_missing_probe_meta_returns_none(self):
        h1, m15 = self._fixtures()
        trig = _make_trigger("first_confluence")
        trig.meta.pop("probe_input_idx")
        m15_start, parent_idx = _resolve_first_via_unified_probe(
            trig, h1, m15, sibling_entity_df=None,
        )
        assert m15_start is None
        assert parent_idx is None

    def test_unsupported_use_case_raises(self):
        h1, m15 = self._fixtures()
        # Need to make ref derivation not fail before the use_case check —
        # patch the ad-hoc builder so the function gets past mapping.
        trig = _make_trigger("subsequent_counter")
        with pytest.raises(ValueError, match="unsupported"):
            _resolve_first_via_unified_probe(trig, h1, m15, sibling_entity_df=None)

    def test_pending_probe_returns_none(self):
        h1, m15 = self._fixtures()
        trig = _make_trigger(
            "first_confluence", parent_sd=1, lower_sd=1,
            probe_input_idx=3, probe_end_idx=7,
        )
        pending = ProbeResult(
            start_idx=0, status="pending", iterations=1,
            original_ref_zone=ReferenceZone(
                outer=0.6000, inner=0.6020, side="buy",
                source="ad_hoc_bos_0", source_event_idx=0,
            ),
            finalize_condition="no_cts_pending",
        )
        with patch(
            "engine_v2.multitf.entity_df_mutation."
            "_build_first_confluence_ref_zone",
            return_value=ReferenceZone(
                outer=0.6000, inner=0.6020, side="buy",
                source="ad_hoc_bos_0", source_event_idx=0,
            ),
        ), patch(
            "engine_v2.structure.unified_probe.unified_probe",
            return_value=pending,
        ):
            m15_start, parent_idx = _resolve_first_via_unified_probe(
                trig, h1, m15, sibling_entity_df=None,
            )
        assert m15_start is None
        assert parent_idx == 3


# ---------------------------------------------------------------------------
# _resolve_trigger_m15_start dispatcher — routes + threads sibling_entity_df
# ---------------------------------------------------------------------------

class TestDispatcher:

    def test_first_counter_routes_to_unified_with_sibling(self):
        h1 = _h1_df_uptrend(n_hours=5)
        m15 = _m15_from_h1(h1)
        sib = m15.copy()
        sib.attrs["events"] = []
        trig = _make_trigger("first_counter", probe_input_idx=1, probe_end_idx=3)
        with patch(
            "engine_v2.multitf.entity_df_mutation."
            "_resolve_first_via_unified_probe",
            return_value=(99, 1),
        ) as mock_new, patch(
            "engine_v2.multitf.entity_df_mutation._resolve_via_legacy_probe"
        ) as mock_legacy:
            m15_start, parent_idx = _resolve_trigger_m15_start(
                trig, h1, m15, sibling_entity_df=sib,
            )
        mock_new.assert_called_once()
        _, kwargs = mock_new.call_args
        assert kwargs["sibling_entity_df"] is sib
        mock_legacy.assert_not_called()
        assert (m15_start, parent_idx) == (99, 1)

    def test_first_confluence_routes_to_unified_no_sibling_needed(self):
        h1 = _h1_df_uptrend(n_hours=5)
        m15 = _m15_from_h1(h1)
        trig = _make_trigger(
            "first_confluence", parent_sd=1, lower_sd=1,
            probe_input_idx=2, probe_end_idx=4,
        )
        with patch(
            "engine_v2.multitf.entity_df_mutation."
            "_resolve_first_via_unified_probe",
            return_value=(77, 2),
        ) as mock_new, patch(
            "engine_v2.multitf.entity_df_mutation._resolve_via_legacy_probe"
        ) as mock_legacy:
            _resolve_trigger_m15_start(trig, h1, m15, sibling_entity_df=None)
        mock_new.assert_called_once()
        _, kwargs = mock_new.call_args
        assert kwargs["sibling_entity_df"] is None
        mock_legacy.assert_not_called()

    def test_subsequent_confluence_routes_to_legacy(self):
        h1 = _h1_df_uptrend(n_hours=5)
        m15 = _m15_from_h1(h1)
        trig = _make_trigger("subsequent_confluence", lower_sd=1)
        with patch(
            "engine_v2.multitf.entity_df_mutation."
            "_resolve_first_via_unified_probe"
        ) as mock_new, patch(
            "engine_v2.multitf.entity_df_mutation._resolve_via_legacy_probe",
            return_value=(50, 7),
        ) as mock_legacy:
            m15_start, parent_idx = _resolve_trigger_m15_start(
                trig, h1, m15, sibling_entity_df=None,
            )
        mock_new.assert_not_called()
        mock_legacy.assert_called_once()
        assert (m15_start, parent_idx) == (50, 7)

    def test_subsequent_counter_routes_to_legacy(self):
        h1 = _h1_df_uptrend(n_hours=5)
        m15 = _m15_from_h1(h1)
        trig = _make_trigger("subsequent_counter")
        with patch(
            "engine_v2.multitf.entity_df_mutation."
            "_resolve_first_via_unified_probe"
        ) as mock_new, patch(
            "engine_v2.multitf.entity_df_mutation._resolve_via_legacy_probe",
            return_value=(60, 8),
        ):
            _resolve_trigger_m15_start(trig, h1, m15)
        mock_new.assert_not_called()
