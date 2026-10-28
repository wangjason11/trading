"""PLAN_U U3 (§6): patch a `git archive` copy of the tree (argv[1]) so the pinbar body floor's window is filled by
the FINAL pinbars (body-% + the ones the floor demoted) — `CandleParams.pinbar_floor_window_source` "natural" ->
"final", nothing else. The setting's one read site is `features/pinbar_body_floor.rolling_pinbar_body_floor`; every
engine `CandleParams()` takes the default (`candle_classifier.apply_candle_classification`, `poi_zones.select_ic`,
the seed guard in `run_replay` — which counts body-% pinbars, so the `[floor_seed]` lines do not change).

    git archive HEAD | tar -x -C <short %TEMP% dir>; cp oanda.cfg <dir>; python <this> <dir>
    cd <dir>; PYTHONIOENCODING=utf-8 python -m engine_v2.run_replay > run.log 2>&1
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
rel = "engine_v2/features/candle_params.py"
p = ROOT / rel
b = p.read_bytes().decode("utf-8")
old = '    pinbar_floor_window_source: str = "natural"'
new = '    pinbar_floor_window_source: str = "final"'
assert b.count(old) == 1, b.count(old)
p.write_bytes(b.replace(old, new).encode("utf-8"))
print(f"patched {rel}: pinbar_floor_window_source = \"final\"")
