"""Every earlier mutants file's anchors in the files 8c touched: matched once at BASE (pre-8c), and now?

    PYTHONIOENCODING=utf-8 python engine_v2/plans/plan_v_inputs/anchor_scan_8c.py e5617a0 > engine_v2/plans/plan_v_inputs/anchor_scan_8c.txt

Same check as `anchor_scan.py` (8a / 8b); here 8a's + 8b's mutants files are scanned too, only 8c's own is skipped.
"""
import glob, importlib.util, subprocess, sys
BASE = sys.argv[1]
TOUCHED = {"engine_v2/zones/wave_candles.py", "engine_v2/charting/export_m15_chart.py",
           "engine_v2/charting/export_plotly.py"}
old_src = {f: subprocess.run(["git", "show", f"{BASE}:{f}"], capture_output=True).stdout.decode("utf-8").replace("\r\n", "\n") for f in TOUCHED}
new_src = {f: open(f, encoding="utf-8").read() for f in TOUCHED}
stale = 0
checked = 0
files = 0
for path in sorted(glob.glob("engine_v2/plans/**/mutants*.py", recursive=True)):
    if path.replace("\\", "/").endswith("plan_v_inputs/mutants_8c.py"):
        continue
    try:
        spec = importlib.util.spec_from_file_location("m", path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        muts = list(m.MUT)
    except Exception as e:
        print(f"SKIP {path}: {e!r}"); continue
    files += 1
    for mid, rel, old, new in muts:
        if rel not in TOUCHED:
            continue
        a, b = old_src[rel].count(old), new_src[rel].count(old)
        checked += a == 1
        if a == 1 and b != 1:
            stale += 1
            print(f"STALE {path} {mid} ({rel}): {a} -> {b}")
print(f"{stale} stale of {checked} anchors that matched once at the base ({files} mutants files read)")
