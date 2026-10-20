MS = "engine_v2/structure/market_structure.py"
MUT = [
    ("0b-1 the constant back to 9", MS,
     "_FALLBACK_PROXIMITY_PIPS = 8  #", "_FALLBACK_PROXIMITY_PIPS = 9  #"),
    ("0b-2 min gap fallback off by one", MS,
     "_FALLBACK_MIN_GAP_PIPS = 50  #", "_FALLBACK_MIN_GAP_PIPS = 49  #"),
]
