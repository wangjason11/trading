"""On top of make_variant_c0.py drop: the FC probe's Phase 2 MS gets the CALLER's timeframe (M15 proximity 5 / gap 30)
instead of the H1 default (8 / 50) - the 'proximity_pips for probes' deferred cleanup, to see sub 0's lifecycle start."""
import sys
from pathlib import Path
p = Path(sys.argv[1]) / "engine_v2/structure/unified_probe.py"
b = p.read_bytes().decode("utf-8"); nl = "\r\n" if "\r\n" in b else "\n"
pairs = [
    ("    *,\n    max_iterations: int,\n) -> _DetResult:\n    \"\"\"Phase 2 ",
     "    *,\n    max_iterations: int,\n    timeframe: str = \"H1\",\n) -> _DetResult:\n    \"\"\"Phase 2 "),
    ("            stop_after_cts_established=2,\n", "            stop_after_cts_established=2,\n            timeframe=timeframe,\n"),
    ("        reset_tol, wick_cap,\n        max_iterations=max_iterations,\n    )\n\n    notes = (\n        f\"unified_probe deterministic+phase2",
     "        reset_tol, wick_cap,\n        max_iterations=max_iterations, timeframe=timeframe,\n    )\n\n    notes = (\n        f\"unified_probe deterministic+phase2"),
]
for o, n in pairs:
    o, n = o.replace("\n", nl), n.replace("\n", nl)
    assert b.count(o) == 1, (o, b.count(o))
    b = b.replace(o, n)
p.write_bytes(b.encode("utf-8")); print("patched unified_probe Phase 2 timeframe")
