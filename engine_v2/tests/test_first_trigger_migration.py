"""Unit + light-integration tests for the unified-probe start resolvers
(Session 3 uniform rule, 2026-05-31).

Three resolver paths, dispatched by use_case in `_resolve_trigger_m15_start`:
  - `first_confluence` → `_resolve_first_confluence_via_unified_probe`: own
    ad-hoc BOS_0 ref, parent-BOS-extreme price-mapped input, Phase 1+2.
  - `first_counter` / `subsequent_confluence` / `subsequent_counter` →
    `_resolve_sibling_cts_via_unified_probe`: input AND reference co-sourced
    from the SIBLING entity's most recent CTS in the trigger's M15 window,
    Phase 1 only.
  - `_LEGACY_PROBE_USE_CASES` (empty by default) → legacy parent-TF probe
    (bisect escape hatch).

Covers:
  - Refactored `map_candle_to_lower_tf` (dropped `mapping_sd` → `parent_extreme_dir`).
  - first_confluence resolver wiring (happy path + defensive None branches).
  - sibling-CTS resolver wiring (happy path + each None branch + per-variation
    idx window + sub_sid disambiguation).
  - dispatcher routing + sibling_entity_df threading + legacy escape hatch.

The unified-probe primitive itself is exercised in test_unified_probe.py.
"""
from __future__ import annotations

from typing import Optional
from unittest.mock import patch

import pandas as pd
import pytest

from engine_v2.common.types import KLZone
from engine_v2.multitf.data_bridge import map_candle_to_lower_tf
from engine_v2.multitf import entity_df_mutation as edm
from engine_v2.multitf.entity_df_mutation import (
    _resolve_first_confluence_via_unified_probe,
    _resolve_sibling_cts_via_unified_probe,
    _resolve_trigger_m15_start,
    _sibling_cts_idx_window,
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
    prior_sd_trigger_idx: Optional[int] = None,
    prior_cts_prox_idx: Optional[int] = None,
) -> MultiTFTrigger:
    meta = {
        "probe_input_idx": probe_input_idx,
        "probe_end_idx": probe_end_idx,
        "trigger_event_idx": probe_end_idx,
    }
    if prior_sd_trigger_idx is not None:
        meta["prior_sd_trigger_idx"] = prior_sd_trigger_idx
    if prior_cts_prox_idx is not None:
        meta["prior_cts_prox_idx"] = prior_cts_prox_idx
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
        meta=meta,
    )


def _fake_probe_result(start_idx: int = 4, source: str = "ad_hoc_bos_0") -> ProbeResult:
    return ProbeResult(
        start_idx=start_idx, status="finalized", iterations=1,
        original_ref_zone=ReferenceZone(
            outer=0.6010, inner=0.6020, side="buy",
            source=source, source_event_idx=0,
        ),
        finalize_condition="no_retrace",
    )


def _make_sibling_cts_event(
    parent_sid: int, parent_cycle_id: int, sub_sid: int,
    idx: int, cts_anchor_idx: int,
    cycle_id: int = 0, struct_direction: int = 1,
) -> StructureEvent:
    """A CTS_CONFIRMED StructureEvent like a sibling sub would emit on its
    own M15 entity_df (structure_id=0 LOCAL to that sub's MS run)."""
    return StructureEvent(
        idx=idx,
        category="STRUCTURE",
        type="CTS_CONFIRMED",
        price=0.6020,
        meta={
            "structure_id": 0,
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
# _resolve_first_confluence_via_unified_probe — own ad-hoc BOS_0
# ---------------------------------------------------------------------------

class TestResolveFirstConfluence:

    def _fixtures(self):
        h1 = _h1_df_uptrend(n_hours=10)
        m15 = _m15_from_h1(h1)
        return h1, m15

    def test_happy_path_uses_ad_hoc_bos_zone_and_phase2(self):
        h1, m15 = self._fixtures()
        trig = _make_trigger(
            "first_confluence", parent_cycle_id=1,
            parent_sd=1, lower_sd=1,
            probe_input_idx=3, probe_end_idx=7,
        )
        fake = _fake_probe_result(start_idx=15)
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
            m15_start, parent_idx, _bos0 = _resolve_first_confluence_via_unified_probe(
                trig, h1, m15,
            )
        # input price-mapped with parent_extreme_dir = -lower_sd = -1
        # → min-low of H1 hour 3 → M15 #1 = idx 13.
        ref_args, _ = mock_ref.call_args
        assert ref_args[1] == 13
        assert ref_args[2] == 1               # probe_direction = lower_sd
        _, kwargs = mock_probe.call_args
        assert kwargs["reference_zone"].source == "ad_hoc_bos_0"
        assert kwargs["direction"] == 1
        assert kwargs["enable_phase2"] is True  # first_confluence ONLY
        assert m15_start == 15
        assert parent_idx == 3                 # parent BOS extreme metadata

    def test_missing_probe_meta_returns_none(self):
        h1, m15 = self._fixtures()
        trig = _make_trigger("first_confluence")
        trig.meta.pop("probe_input_idx")
        m15_start, parent_idx, _bos0 = _resolve_first_confluence_via_unified_probe(
            trig, h1, m15,
        )
        assert m15_start is None
        assert parent_idx is None

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
            m15_start, parent_idx, _bos0 = _resolve_first_confluence_via_unified_probe(
                trig, h1, m15,
            )
        assert m15_start is None
        assert parent_idx == 3

    def test_ref_zone_none_returns_none(self):
        h1, m15 = self._fixtures()
        trig = _make_trigger(
            "first_confluence", parent_sd=1, lower_sd=1,
            probe_input_idx=3, probe_end_idx=7,
        )
        with patch(
            "engine_v2.multitf.entity_df_mutation."
            "_build_first_confluence_ref_zone",
            return_value=None,
        ):
            m15_start, parent_idx, _bos0 = _resolve_first_confluence_via_unified_probe(
                trig, h1, m15,
            )
        assert m15_start is None
        assert parent_idx == 3


# ---------------------------------------------------------------------------
# _resolve_sibling_cts_via_unified_probe — input + ref co-sourced from sibling
# ---------------------------------------------------------------------------

class TestResolveSiblingCts:

    def _fixtures(self):
        h1 = _h1_df_uptrend(n_hours=10)
        m15 = _m15_from_h1(h1)
        return h1, m15

    def _sibling_with_cts(self, parent_cycle_id, sub_sid, idx, anchor):
        m15 = _m15_from_h1(_h1_df_uptrend(n_hours=10))
        sib = m15.copy()
        sib.attrs["events"] = [
            _make_sibling_cts_event(
                parent_sid=0, parent_cycle_id=parent_cycle_id, sub_sid=sub_sid,
                idx=idx, cts_anchor_idx=anchor, cycle_id=0, struct_direction=1,
            ),
        ]
        sib.attrs["kl_zones"] = [
            _make_kl_zone(
                "CTS", parent_sid=0, parent_cycle_id=parent_cycle_id,
                sub_sid=sub_sid, sid=0, cycle_id=0,
            ),
        ]
        return sib

    def test_first_counter_input_equals_sibling_cts_extreme(self):
        h1, m15 = self._fixtures()
        sib = self._sibling_with_cts(parent_cycle_id=2, sub_sid=0, idx=8, anchor=6)
        trig = _make_trigger(
            "first_counter", parent_cycle_id=2,
            parent_sd=1, lower_sd=-1,
            probe_end_idx=5,   # H1 idx 5 → M15 last-of-hour = 23
        )
        fake = _fake_probe_result(start_idx=12, source="cts_confirmed")
        with patch(
            "engine_v2.structure.unified_probe.unified_probe",
            return_value=fake,
        ) as mock_probe:
            m15_start, meta_idx, _bos0 = _resolve_sibling_cts_via_unified_probe(
                trig, h1, m15, sibling_entity_df=sib,
            )
        _, kwargs = mock_probe.call_args
        assert kwargs["direction"] == -1
        assert kwargs["enable_phase2"] is False      # sibling path = Phase 1 only
        # input_idx == ref.source_event_idx == sibling CTS extreme (anchor 6)
        assert kwargs["input_idx"] == 6
        assert kwargs["reference_zone"].source == "cts_confirmed"
        assert kwargs["reference_zone"].source_event_idx == 6
        assert m15_start == 12
        assert meta_idx == 6

    def test_subsequent_confluence_routes_and_reads_sibling(self):
        h1, m15 = self._fixtures()
        sib = self._sibling_with_cts(parent_cycle_id=0, sub_sid=0, idx=8, anchor=6)
        trig = _make_trigger(
            "subsequent_confluence", parent_cycle_id=0,
            parent_sd=1, lower_sd=1, probe_end_idx=5,
            prior_sd_trigger_idx=1,
        )
        fake = _fake_probe_result(start_idx=12, source="cts_confirmed")
        with patch(
            "engine_v2.structure.unified_probe.unified_probe",
            return_value=fake,
        ) as mock_probe:
            m15_start, meta_idx, _bos0 = _resolve_sibling_cts_via_unified_probe(
                trig, h1, m15, sibling_entity_df=sib,
            )
        _, kwargs = mock_probe.call_args
        assert kwargs["direction"] == 1
        assert kwargs["input_idx"] == 6
        assert m15_start == 12

    def test_no_sibling_df_uses_ad_hoc_fallback(self):
        # No sibling df → PART4 §4.3.4 step-5 fallback: own-entity window
        # extreme + ad-hoc BOS_0. Patch the ad-hoc builder + probe so the test
        # doesn't depend on real base-pattern derivation on synthetic data.
        h1, m15 = self._fixtures()
        trig = _make_trigger("first_counter", probe_end_idx=5)
        fb_ref = ReferenceZone(
            outer=0.6010, inner=0.6020, side="sell",
            source="ad_hoc_bos_0", source_event_idx=6,
        )
        with patch(
            "engine_v2.multitf.entity_df_mutation."
            "_build_first_confluence_ref_zone",
            return_value=fb_ref,
        ) as mock_fb, patch(
            "engine_v2.structure.unified_probe.unified_probe",
            return_value=_fake_probe_result(start_idx=9, source="ad_hoc_bos_0"),
        ) as mock_probe:
            m15_start, meta_idx, _bos0 = _resolve_sibling_cts_via_unified_probe(
                trig, h1, m15, sibling_entity_df=None,
            )
        mock_fb.assert_called_once()           # fallback path taken
        _, kwargs = mock_probe.call_args
        assert kwargs["reference_zone"].source == "ad_hoc_bos_0"
        assert kwargs["input_idx"] == 6        # fallback ref's source_event_idx
        assert m15_start == 9
        assert meta_idx == 6

    def test_no_sibling_events_uses_ad_hoc_fallback(self):
        h1, m15 = self._fixtures()
        sib = m15.copy()
        sib.attrs["events"] = []
        sib.attrs["kl_zones"] = []
        trig = _make_trigger("first_counter", probe_end_idx=5)
        fb_ref = ReferenceZone(
            outer=0.6010, inner=0.6020, side="sell",
            source="ad_hoc_bos_0", source_event_idx=6,
        )
        with patch(
            "engine_v2.multitf.entity_df_mutation."
            "_build_first_confluence_ref_zone",
            return_value=fb_ref,
        ) as mock_fb, patch(
            "engine_v2.structure.unified_probe.unified_probe",
            return_value=_fake_probe_result(start_idx=9, source="ad_hoc_bos_0"),
        ):
            m15_start, meta_idx, _bos0 = _resolve_sibling_cts_via_unified_probe(
                trig, h1, m15, sibling_entity_df=sib,
            )
        mock_fb.assert_called_once()
        assert m15_start == 9
        assert meta_idx == 6

    def test_both_sibling_and_fallback_unavailable_returns_none(self):
        # Sibling has no CTS AND the ad-hoc fallback can't derive a zone
        # (base pattern fails) → genuinely return None.
        h1, m15 = self._fixtures()
        trig = _make_trigger("first_counter", probe_end_idx=5)
        with patch(
            "engine_v2.multitf.entity_df_mutation."
            "_build_first_confluence_ref_zone",
            return_value=None,
        ):
            m15_start, meta_idx, _bos0 = _resolve_sibling_cts_via_unified_probe(
                trig, h1, m15, sibling_entity_df=None,
            )
        assert m15_start is None
        assert meta_idx is None

    def test_missing_probe_end_returns_none(self):
        h1, m15 = self._fixtures()
        sib = self._sibling_with_cts(parent_cycle_id=2, sub_sid=0, idx=8, anchor=6)
        trig = _make_trigger("first_counter", parent_cycle_id=2, probe_end_idx=5)
        trig.meta.pop("probe_end_idx")
        m15_start, meta_idx, _bos0 = _resolve_sibling_cts_via_unified_probe(
            trig, h1, m15, sibling_entity_df=sib,
        )
        assert m15_start is None
        assert meta_idx is None

    def test_degenerate_window_input_at_or_after_end_returns_none(self):
        h1, m15 = self._fixtures()
        # CTS event INSIDE the window (idx 23 == window hi) with its anchor
        # AT the end boundary (23) → ref builds, but input_idx (= anchor 23)
        # >= m15_end_idx (23) → degenerate-window branch.
        sib = self._sibling_with_cts(parent_cycle_id=2, sub_sid=0, idx=23, anchor=23)
        trig = _make_trigger("first_counter", parent_cycle_id=2, probe_end_idx=5)
        m15_start, meta_idx, _bos0 = _resolve_sibling_cts_via_unified_probe(
            trig, h1, m15, sibling_entity_df=sib,
        )
        assert m15_start is None
        assert meta_idx == 23      # sibling CTS extreme reported for traceability

    def test_pending_probe_returns_none(self):
        h1, m15 = self._fixtures()
        sib = self._sibling_with_cts(parent_cycle_id=2, sub_sid=0, idx=8, anchor=6)
        trig = _make_trigger("first_counter", parent_cycle_id=2, probe_end_idx=5)
        pending = ProbeResult(
            start_idx=0, status="pending", iterations=1,
            original_ref_zone=ReferenceZone(
                outer=0.6010, inner=0.6020, side="sell",
                source="cts_confirmed", source_event_idx=6,
            ),
            finalize_condition="no_cts_pending",
        )
        with patch(
            "engine_v2.structure.unified_probe.unified_probe",
            return_value=pending,
        ):
            m15_start, meta_idx, _bos0 = _resolve_sibling_cts_via_unified_probe(
                trig, h1, m15, sibling_entity_df=sib,
            )
        assert m15_start is None
        assert meta_idx == 6

    def test_sub_sid_disambiguation_picks_winning_subs_zone(self):
        """Two sibling subs share LOCAL cycle_id=0. The most-recent CTS wins
        (sub_sid=1); its zone — not the earlier sub_sid=0 zone — must drive
        the reference, even though both match (structure_id=0, cycle_id=0)."""
        h1, m15 = self._fixtures()
        sib = m15.copy()
        sib.attrs["events"] = [
            _make_sibling_cts_event(
                parent_sid=0, parent_cycle_id=2, sub_sid=0,
                idx=6, cts_anchor_idx=4, cycle_id=0,
            ),
            _make_sibling_cts_event(
                parent_sid=0, parent_cycle_id=2, sub_sid=1,
                idx=10, cts_anchor_idx=8, cycle_id=0,
            ),
        ]
        # sub_sid=0 zone listed FIRST (would win a naive first-match lookup);
        # sub_sid=1 zone has a distinct bottom so we can tell them apart.
        sib.attrs["kl_zones"] = [
            _make_kl_zone("CTS", 0, 2, sub_sid=0, cycle_id=0,
                          top=0.6020, bottom=0.6010),
            _make_kl_zone("CTS", 0, 2, sub_sid=1, cycle_id=0,
                          top=0.6060, bottom=0.6050),
        ]
        trig = _make_trigger(
            "first_counter", parent_cycle_id=2,
            parent_sd=1, lower_sd=-1, probe_end_idx=5,
        )
        captured = {}

        def _capture(*a, **k):
            captured["ref"] = k["reference_zone"]
            captured["input"] = k["input_idx"]
            return _fake_probe_result(start_idx=12, source="cts_confirmed")

        with patch(
            "engine_v2.structure.unified_probe.unified_probe", _capture,
        ):
            _resolve_sibling_cts_via_unified_probe(
                trig, h1, m15, sibling_entity_df=sib,
            )
        # Winner = idx 10 (sub_sid=1), anchor 8. lower_sd=-1 → inner = z.bottom.
        assert captured["input"] == 8
        assert captured["ref"].source_event_idx == 8
        assert captured["ref"].inner == pytest.approx(0.6050)  # sub_sid=1 zone


# ---------------------------------------------------------------------------
# _sibling_cts_idx_window — per-variation lower bound
# ---------------------------------------------------------------------------

class TestSiblingCtsIdxWindow:

    def test_first_counter_lo_is_zero(self):
        h1 = _h1_df_uptrend(n_hours=5)
        m15 = _m15_from_h1(h1)
        trig = _make_trigger("first_counter", probe_end_idx=4)
        lo, hi = _sibling_cts_idx_window(trig, h1, m15, m15_end_idx=19)
        assert lo == 0
        assert hi == 19

    def test_subsequent_confluence_lo_from_prior_sd(self):
        h1 = _h1_df_uptrend(n_hours=5)
        m15 = _m15_from_h1(h1)
        trig = _make_trigger(
            "subsequent_confluence", lower_sd=1,
            probe_end_idx=4, prior_sd_trigger_idx=1,
        )
        lo, hi = _sibling_cts_idx_window(trig, h1, m15, m15_end_idx=19)
        # prior_sd H1 idx 1 → last-M15-of-hour = 7
        assert lo == 7
        assert hi == 19

    def test_subsequent_counter_lo_from_prior_cts(self):
        h1 = _h1_df_uptrend(n_hours=5)
        m15 = _m15_from_h1(h1)
        trig = _make_trigger(
            "subsequent_counter", lower_sd=-1,
            probe_end_idx=4, prior_cts_prox_idx=2,
        )
        lo, hi = _sibling_cts_idx_window(trig, h1, m15, m15_end_idx=19)
        # prior_cts H1 idx 2 → last-M15-of-hour = 11
        assert lo == 11
        assert hi == 19

    def test_missing_prior_meta_degrades_to_zero(self):
        h1 = _h1_df_uptrend(n_hours=5)
        m15 = _m15_from_h1(h1)
        trig = _make_trigger("subsequent_confluence", lower_sd=1, probe_end_idx=4)
        lo, hi = _sibling_cts_idx_window(trig, h1, m15, m15_end_idx=19)
        assert lo == 0


# ---------------------------------------------------------------------------
# _resolve_trigger_m15_start dispatcher — routing + sibling threading
# ---------------------------------------------------------------------------

class TestDispatcher:

    def test_first_confluence_routes_to_first_confluence_resolver(self):
        h1 = _h1_df_uptrend(n_hours=5)
        m15 = _m15_from_h1(h1)
        trig = _make_trigger(
            "first_confluence", parent_sd=1, lower_sd=1,
            probe_input_idx=2, probe_end_idx=4,
        )
        with patch(
            "engine_v2.multitf.entity_df_mutation."
            "_resolve_first_confluence_via_unified_probe",
            return_value=(77, 2),
        ) as mock_conf, patch(
            "engine_v2.multitf.entity_df_mutation."
            "_resolve_sibling_cts_via_unified_probe",
        ) as mock_sib:
            out = _resolve_trigger_m15_start(trig, h1, m15, sibling_entity_df=None)
        mock_conf.assert_called_once()
        mock_sib.assert_not_called()
        assert out == (77, 2)

    @pytest.mark.parametrize(
        "use_case,lower_sd",
        [
            ("first_counter", -1),
            ("subsequent_confluence", 1),
            ("subsequent_counter", -1),
        ],
    )
    def test_sibling_variations_route_to_sibling_resolver(self, use_case, lower_sd):
        h1 = _h1_df_uptrend(n_hours=5)
        m15 = _m15_from_h1(h1)
        sib = m15.copy()
        trig = _make_trigger(use_case, lower_sd=lower_sd, probe_end_idx=3)
        with patch(
            "engine_v2.multitf.entity_df_mutation."
            "_resolve_sibling_cts_via_unified_probe",
            return_value=(99, 6),
        ) as mock_sib, patch(
            "engine_v2.multitf.entity_df_mutation."
            "_resolve_first_confluence_via_unified_probe",
        ) as mock_conf:
            out = _resolve_trigger_m15_start(
                trig, h1, m15, sibling_entity_df=sib,
            )
        mock_conf.assert_not_called()
        mock_sib.assert_called_once()
        _, kwargs = mock_sib.call_args
        assert kwargs["sibling_entity_df"] is sib
        assert out == (99, 6)

    def test_legacy_escape_hatch(self, monkeypatch):
        h1 = _h1_df_uptrend(n_hours=5)
        m15 = _m15_from_h1(h1)
        # Force subsequent_counter back onto the legacy probe via the escape hatch.
        monkeypatch.setattr(
            edm, "_LEGACY_PROBE_USE_CASES", frozenset({"subsequent_counter"}),
        )
        trig = _make_trigger("subsequent_counter", probe_end_idx=3)
        with patch(
            "engine_v2.multitf.entity_df_mutation._resolve_via_legacy_probe",
            return_value=(60, 8),
        ) as mock_legacy, patch(
            "engine_v2.multitf.entity_df_mutation."
            "_resolve_sibling_cts_via_unified_probe",
        ) as mock_sib:
            out = _resolve_trigger_m15_start(trig, h1, m15)
        mock_legacy.assert_called_once()
        mock_sib.assert_not_called()
        assert out == (60, 8)
