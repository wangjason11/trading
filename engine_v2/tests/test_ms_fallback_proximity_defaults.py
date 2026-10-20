"""PLAN_R step 0b: MarketStructure's fallback proximity defaults equal the H1 table values.

`market_structure` cannot import `zones/zone_proximity.py` (zone_proximity imports `StructureEvent` from it), so its
fallbacks for direct / test callers that pass no `proximity_pips` / `min_gap_pips` are literals. They used to be "kept
in sync manually": `_FALLBACK_PROXIMITY_PIPS` stayed 9 after the H1 default became 8 (`804d19d`, 2026-05-28) until
PLAN_R step 0b. This test replaces the manual sync with a check that fails loudly. Every production build passes the
TF value (`structure_engine` / `unified_probe`), so only direct constructions read these.
"""
from engine_v2.structure import market_structure as ms
from engine_v2.structure.market_structure import MarketStructure
from engine_v2.zones.zone_proximity import (
    DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS,
    DEFAULT_PROXIMITY_PIPS,
)


def test_fallback_proximity_pips_equals_the_h1_default():
    assert ms._FALLBACK_PROXIMITY_PIPS == DEFAULT_PROXIMITY_PIPS["H1"] == 8


def test_fallback_min_gap_pips_equals_the_h1_default():
    assert ms._FALLBACK_MIN_GAP_PIPS == DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS["H1"] == 50


def test_a_direct_construction_reads_the_fallbacks():
    import pandas as pd
    df = pd.DataFrame({"o": [1.0], "h": [1.0], "l": [1.0], "c": [1.0]})
    m = MarketStructure(df, struct_direction=1)
    assert m.proximity_pips == DEFAULT_PROXIMITY_PIPS["H1"]
    assert m.min_gap_pips == DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS["H1"]
